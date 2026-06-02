"""Compare GAPIT3 vs TorchGenomics on the MDP fixture across:

  1. GLM (-log10p corr; TG GLM Wald)
  2. MLM (-log10p corr; TG SingleTraitLMM Wald + VanRaden K)
  3. FarmCPU (-log10p corr + significant-SNP intersection; TG FarmCPU)
  4. BLINK (-log10p corr + significant-SNP intersection; TG BLINK)

Reference outputs come from benchmark/gapit_results/ (committed pre-Pillar-A
canonical fixture) OR a fresh GAPIT3 re-run, whichever run_gapit.sh produced.

Tolerance gates anchor in docs/validation.md §16. GLM/MLM math is
well-defined and asserts the §16 floors. FarmCPU/BLINK are stochastic
multi-locus methods; we assert a weaker per-SNP corr (>= 0.5) plus a
significant-SNP set intersection, matching the existing
tests/test_golden_gapit.py contract (per-SNP corr ≥ 0.99 was the
aspirational target; the model uses pseudo-QTN selection ordering, so we
floor at 0.5 for stochastic stability and surface the actual value as an
extra).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# ── Tolerance gates ──────────────────────────────────────────────────────────
# GLM and MLM are deterministic; assert §16 floors directly.
TOL_GLM_LOGP_CORR = 0.99       # GLM is the simplest model; tight
TOL_MLM_LOGP_CORR = 0.99       # MLM with VanRaden K + 3 PCs
# FarmCPU / BLINK are stochastic (pseudo-QTN order, sub-sampling, LD pruning).
# Per-SNP correlation between GAPIT and any other implementation is generally
# in the 0.5-0.9 range; we floor at 0.5 to detect breakage but record the
# actual value as an extra. Set-intersection on top-k SNPs catches "we found
# the same hits even though pseudo-QTN trajectories differed".
TOL_FARMCPU_LOGP_CORR = 0.5
TOL_BLINK_LOGP_CORR = 0.5
TOL_TOPK_OVERLAP = 0.3   # of GAPIT's top-10 SNPs, ≥ 30% in TG's top-50


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.config import STAT_DTYPE, NumericalConfig  # noqa: E402
from torchgenomics.linalg.kinship import grm_zhang  # noqa: E402
from torchgenomics.models.base import VariantMeta  # noqa: E402
from torchgenomics.models.blink import BLINK  # noqa: E402
from torchgenomics.models.farmcpu import FarmCPU  # noqa: E402
from torchgenomics.models.glm import GLM  # noqa: E402
from torchgenomics.models.single_trait_lmm import SingleTraitLMM  # noqa: E402


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    direction: str
    note: str = ""

    def __str__(self) -> str:
        cmp = ">=" if self.direction == "min" else "<="
        flag = "PASS" if self.passed else "FAIL"
        return f"  [{flag}] {self.name:50s} observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"


@dataclass
class ComparisonReport:
    name: str
    n_compared: int
    checks: list[CheckResult] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print(self) -> None:
        bar = "=" * 80
        print(bar)
        print(f"  {self.name}  (n_compared = {self.n_compared})")
        print(bar)
        for c in self.checks:
            print(c)
        if self.extras:
            print("  -- extras --")
            for k, v in self.extras.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.6e}")
                else:
                    print(f"    {k}: {v}")
        print()


def _check_min(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs >= thr, observed=obs, threshold=thr, direction="min", note=note)


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr, direction="max", note=note)


def _logp_corr(p_ref: np.ndarray, p_tg: np.ndarray) -> float:
    p_ref = np.clip(p_ref, 1e-300, 1.0)
    p_tg = np.clip(p_tg, 1e-300, 1.0)
    a = -np.log10(p_ref)
    b = -np.log10(p_tg)
    if a.std() < 1e-15 or b.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _topk_overlap(
    df_ref: pd.DataFrame, df_tg: pd.DataFrame,
    snp_col_ref: str = "SNP", snp_col_tg: str = "SNP",
    p_col_ref: str = "P.value", p_col_tg: str = "P.value",
    k_ref: int = 10, k_tg: int = 50,
) -> float:
    top_ref = set(
        df_ref.dropna(subset=[p_col_ref])
              .sort_values(p_col_ref).head(k_ref)[snp_col_ref].astype(str).tolist()
    )
    top_tg = set(
        df_tg.dropna(subset=[p_col_tg])
             .sort_values(p_col_tg).head(k_tg)[snp_col_tg].astype(str).tolist()
    )
    if len(top_ref) == 0:
        return 0.0
    return len(top_ref & top_tg) / len(top_ref)


# ── Data loader (mirrors benchmark/run_torchgenomics.py + golden test) ───────────


def _load_mdp(data_dir: Path) -> dict:
    pheno = pd.read_csv(data_dir / "mdp_traits.txt", sep="\t")
    geno_full = pd.read_csv(data_dir / "mdp_numeric.txt", sep="\t")
    snpmap = pd.read_csv(data_dir / "mdp_SNP_information.txt", sep="\t")

    pheno["Taxa"] = pheno["Taxa"].astype(str)
    geno_full["taxa"] = geno_full["taxa"].astype(str)

    # GAPIT computes kinship from the FULL geno set, then aligns by Taxa.
    geno_all = geno_full.set_index("taxa")
    G_all = torch.tensor(geno_all.values, dtype=STAT_DTYPE)
    for j in range(G_all.shape[1]):
        col = G_all[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G_all[mask, j] = col[~mask].mean()
    K_full, _ = grm_zhang(G_all)
    all_taxa = list(geno_all.index)

    # Phenotype subset on EarHT (matches GAPIT)
    pheno_sub = pheno[["Taxa", "EarHT"]].dropna(subset=["EarHT"]).copy()
    common = sorted(set(pheno_sub["Taxa"]) & set(geno_full["taxa"]))
    pheno_sub = pheno_sub[pheno_sub["Taxa"].isin(common)].set_index("Taxa").loc[common]
    geno_sub = geno_full[geno_full["taxa"].isin(common)].set_index("taxa").loc[common]

    n, m = geno_sub.shape[0], geno_sub.shape[1]
    snp_names = list(geno_sub.columns)

    G = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()
    Y = torch.tensor(pheno_sub["EarHT"].values, dtype=STAT_DTYPE)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    # Kinship subset (same per-Taxa order as G, Y)
    idx_in_all = [all_taxa.index(t) for t in common]
    idx_t = torch.tensor(idx_in_all, dtype=torch.long)
    K_sub = K_full[idx_t][:, idx_t]

    snpmap_dict = {}
    for _, row in snpmap.iterrows():
        snpmap_dict[row["SNP"]] = (str(int(row["Chromosome"])), int(row["Position"]))
    chrs, pos = [], []
    for s in snp_names:
        c, p = snpmap_dict.get(s, ("0", 0))
        chrs.append(c)
        pos.append(p)

    vmeta = VariantMeta(
        snp=snp_names, chr=chrs, pos=pos,
        a1=["A"] * m, a2=["G"] * m,
    )
    return {
        "n": n, "m": m, "G": G, "Y": Y, "X0": X0, "K": K_sub,
        "vmeta": vmeta, "snp_names": snp_names,
    }


# ── Comparisons ──────────────────────────────────────────────────────────────


def _normalise_gapit_df(df: pd.DataFrame) -> pd.DataFrame:
    """GAPIT writes 'Position ' (trailing space) on some columns; fix it."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def compare_glm(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GAPIT GLM (no kinship) vs TorchGenomics GLM Wald."""
    ref = _normalise_gapit_df(pd.read_csv(out_dir / "GLM_GWAS.csv"))
    d = _load_mdp(data_dir)

    model = GLM()
    nf = model.fit_null(d["Y"], d["X0"])
    res = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({
        "SNP": res.snp,
        "P.tg": res.p.cpu().numpy(),
        "beta_tg": res.beta.cpu().numpy(),
    })
    merged = pd.merge(ref, tg_df, on="SNP").dropna(subset=["P.value", "P.tg"])

    corr = _logp_corr(merged["P.value"].values, merged["P.tg"].values)
    overlap = _topk_overlap(ref, tg_df, p_col_tg="P.tg")

    rep = ComparisonReport(
        name=f"GAPIT GLM vs TG GLM (n_snps_aligned={len(merged)})",
        n_compared=len(merged),
    )
    rep.checks.append(_check_min("−log10(p) correlation", corr, TOL_GLM_LOGP_CORR))
    rep.checks.append(_check_min("Top-10 GAPIT vs Top-50 TG overlap", overlap, TOL_TOPK_OVERLAP))
    rep.extras["GAPIT min P"] = float(merged["P.value"].min())
    rep.extras["TG min P"] = float(merged["P.tg"].min())
    return rep


def compare_mlm(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GAPIT MLM (VanRaden K + 3 PCs by default) vs TG SingleTraitLMM Wald.

    Note: GAPIT's MLM defaults add 3 principal components as fixed effects
    (`PCA.total = 3`). TG fits with intercept-only (no PCs). This induces a
    well-known but bounded discrepancy in the per-SNP β/SE; the −log10(p)
    correlation typically lands at 0.95–0.99 on the MDP fixture.
    """
    ref = _normalise_gapit_df(pd.read_csv(out_dir / "MLM_GWAS.csv"))
    d = _load_mdp(data_dir)

    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(d["Y"], d["X0"], K=d["K"])
    res = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({
        "SNP": res.snp,
        "P.tg": res.p.cpu().numpy(),
    })
    merged = pd.merge(ref, tg_df, on="SNP").dropna(subset=["P.value", "P.tg"])

    corr = _logp_corr(merged["P.value"].values, merged["P.tg"].values)
    overlap = _topk_overlap(ref, tg_df, p_col_tg="P.tg")

    rep = ComparisonReport(
        name=f"GAPIT MLM vs TG SingleTraitLMM Wald (n_snps_aligned={len(merged)})",
        n_compared=len(merged),
    )
    rep.checks.append(_check_min("−log10(p) correlation", corr, TOL_MLM_LOGP_CORR))
    rep.checks.append(_check_min("Top-10 GAPIT vs Top-50 TG overlap", overlap, TOL_TOPK_OVERLAP))
    rep.extras["GAPIT min P"] = float(merged["P.value"].min())
    rep.extras["TG min P"] = float(merged["P.tg"].min())
    rep.extras["TG sig2_g"] = float(nf.sig2_g)
    rep.extras["TG sig2_e"] = float(nf.sig2_e)
    return rep


def compare_farmcpu(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GAPIT FarmCPU vs TG FarmCPU (per-SNP corr + top-k intersection)."""
    ref = _normalise_gapit_df(pd.read_csv(out_dir / "FarmCPU_GWAS.csv"))
    d = _load_mdp(data_dir)

    model = FarmCPU(max_iter=10, p_threshold=0.01, method_sub="reward")
    nf = model.fit_null(d["Y"], d["X0"])
    res = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({"SNP": res.snp, "P.tg": res.p.cpu().numpy()})
    merged = pd.merge(ref, tg_df, on="SNP").dropna(subset=["P.value", "P.tg"])

    corr = _logp_corr(merged["P.value"].values, merged["P.tg"].values)
    overlap = _topk_overlap(ref, tg_df, p_col_tg="P.tg")

    rep = ComparisonReport(
        name=f"GAPIT FarmCPU vs TG FarmCPU (n_snps_aligned={len(merged)})",
        n_compared=len(merged),
    )
    rep.checks.append(_check_min("−log10(p) correlation (stochastic)", corr, TOL_FARMCPU_LOGP_CORR))
    rep.checks.append(_check_min("Top-10 GAPIT vs Top-50 TG overlap", overlap, TOL_TOPK_OVERLAP))
    rep.extras["GAPIT min P"] = float(merged["P.value"].min())
    rep.extras["TG min P"] = float(merged["P.tg"].min())
    return rep


def compare_blink(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GAPIT BLINK vs TG BLINK."""
    ref = _normalise_gapit_df(pd.read_csv(out_dir / "BLINK_GWAS.csv"))
    d = _load_mdp(data_dir)

    model = BLINK(max_iter=10, cutoff=0.05, ld_threshold=0.7, method_sub="reward")
    nf = model.fit_null(d["Y"], d["X0"])
    res = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({"SNP": res.snp, "P.tg": res.p.cpu().numpy()})
    merged = pd.merge(ref, tg_df, on="SNP").dropna(subset=["P.value", "P.tg"])

    corr = _logp_corr(merged["P.value"].values, merged["P.tg"].values)
    overlap = _topk_overlap(ref, tg_df, p_col_tg="P.tg")

    rep = ComparisonReport(
        name=f"GAPIT BLINK vs TG BLINK (n_snps_aligned={len(merged)})",
        n_compared=len(merged),
    )
    rep.checks.append(_check_min("−log10(p) correlation (stochastic)", corr, TOL_BLINK_LOGP_CORR))
    rep.checks.append(_check_min("Top-10 GAPIT vs Top-50 TG overlap", overlap, TOL_TOPK_OVERLAP))
    rep.extras["GAPIT min P"] = float(merged["P.value"].min())
    rep.extras["TG min P"] = float(merged["P.tg"].min())
    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def run_all(data_dir: Path, out_dir: Path, json_out: Path | None = None) -> int:
    reports = [
        compare_glm(data_dir, out_dir),
        compare_mlm(data_dir, out_dir),
        compare_farmcpu(data_dir, out_dir),
        compare_blink(data_dir, out_dir),
    ]
    print()
    for r in reports:
        r.print()

    n_pass = sum(r.passed for r in reports)
    n_fail = len(reports) - n_pass
    print(f"=== {n_pass}/{len(reports)} comparisons passed ===")

    if json_out is not None:
        json_out.write_text(json.dumps(
            [{
                "name": r.name,
                "n_compared": r.n_compared,
                "passed": r.passed,
                "checks": [asdict(c) for c in r.checks],
                "extras": r.extras,
            } for r in reports],
            indent=2, default=float,
        ))
        print(f"[compare] wrote {json_out}")

    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "outputs"))
    ap.add_argument("--json", default=None, help="optional JSON report path")
    args = ap.parse_args()
    return run_all(
        Path(args.data_dir),
        Path(args.out_dir),
        Path(args.json) if args.json else None,
    )


if __name__ == "__main__":
    sys.exit(main())
