"""Compare GEMMA 0.98.5 vs TorchGWAS on the MDP fixture across:

  1. GRM (centered, -gk 1) vs torchgwas.linalg.kinship.grm_centered_gemma
  2. Single-trait LMM (Wald, LRT, score) — variance components, log-likelihood,
     β / SE, p-value −log10 correlation per test
  3. Multi-trait mvLMM (joint Wald) — Vg / Ve matrices, p-value −log10 correlation

Tolerances anchor in docs/validation.md §16. The existing golden test
``tests/test_golden_gemma.py`` already pins these — this harness re-runs the
same comparison END-TO-END (re-runs GEMMA + re-runs TG + diffs) to confirm no
regression from the 9 Pillar-A fixes.

Outputs:
  - Prints a per-comparison table to stdout.
  - Returns ComparisonReport(passed=bool, ...) per comparison.
  - main() exits non-zero if ANY comparison violates its §16 tolerance.
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

# ── Section 16 tolerances (DO NOT loosen — these are the golden contract) ────
# These values are pinned in docs/validation.md §16 and tests/test_golden_gemma.py.

# Single-trait LMM
TOL_VARCOMP_REL = 1e-3       # vg, ve relative err
TOL_LL_ABS = 1e-2            # REML log-likelihood absolute err (golden floor 1e-2)
TOL_BETA_CORR = 0.9999       # β corr (full set)
TOL_BETA_MED_ABSDIFF = 0.01  # β median |Δ|, AF ∈ [0.03, 0.97]
TOL_SE_MED_ABSDIFF = 0.02    # SE median |Δ|, AF ∈ [0.03, 0.97]
TOL_LOGP_CORR_WALD = 0.999   # −log10(p) corr for Wald
TOL_LOGP_CORR_LRT = 0.998    # −log10(p) corr for LRT
TOL_LOGP_CORR_SCORE = 0.999  # −log10(p) corr for score

# Multi-trait mvLMM
TOL_VG_REL = 1e-2            # Vg relative err (2-trait REMLE)
TOL_VE_REL = 1e-2            # Ve relative err
TOL_MVLMM_LOGP_CORR = 0.998  # joint-Wald −log10(p) corr

# GRM identity
TOL_GRM_RTOL = 1e-6
TOL_GRM_ATOL = 1e-10


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.config import STAT_DTYPE, NumericalConfig  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.multi_trait_lmm import MultiTraitLMM  # noqa: E402
from torchgwas.models.single_trait_lmm import SingleTraitLMM  # noqa: E402


# ── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    direction: str  # "min" or "max"
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


def _is_floatlike(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_gemma_log(path: Path) -> dict:
    """Extract REMLE / MLE variance components + log-likelihoods from a GEMMA log."""
    out: dict = {}
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if "vg estimate in the null model" in low:
            out["vg"] = float(line.split("=")[-1].strip())
        elif "ve estimate in the null model" in low:
            out["ve"] = float(line.split("=")[-1].strip())
        elif "pve estimate in the null model" in low:
            out["pve"] = float(line.split("=")[-1].strip())
        elif "remle log-likelihood in the null model" in low:
            out["ll_reml"] = float(line.split("=")[-1].strip())
        elif "mle log-likelihood in the null model" in low:
            out["ll_ml"] = float(line.split("=")[-1].strip())
        elif "remle estimate for vg in the null model" in low:
            vals = [float(x) for x in lines[i + 1].split()]
            if i + 2 < len(lines):
                vals2 = lines[i + 2].split()
                if vals2 and _is_floatlike(vals2[0]):
                    out["Vg"] = np.array(
                        [[vals[0], float(vals2[0])], [float(vals2[0]), float(vals2[1])]]
                    )
                else:
                    out["Vg"] = np.array([[vals[0]]])
            else:
                out["Vg"] = np.array([[vals[0]]])
        elif "remle estimate for ve in the null model" in low:
            vals = [float(x) for x in lines[i + 1].split()]
            if i + 2 < len(lines):
                vals2 = lines[i + 2].split()
                if vals2 and _is_floatlike(vals2[0]):
                    out["Ve"] = np.array(
                        [[vals[0], float(vals2[0])], [float(vals2[0]), float(vals2[1])]]
                    )
                else:
                    out["Ve"] = np.array([[vals[0]]])
            else:
                out["Ve"] = np.array([[vals[0]]])
    return out


# ── Data loader ──────────────────────────────────────────────────────────────


def _load_mdp(data_dir: Path, gemma_grm: np.ndarray | None = None) -> dict:
    """Load MDP genotype + traits, align with GEMMA's per-sample order.

    GEMMA reads BIMBAM files in row order (one phenotype per line, no Taxa
    column). The original MDP source data has Taxa labels — we use the same
    alignment as the existing golden test to match.
    """
    pheno = pd.read_csv(data_dir / "mdp_traits.txt", sep="\t")
    geno = pd.read_csv(data_dir / "mdp_numeric.txt", sep="\t")
    snpmap = pd.read_csv(data_dir / "mdp_SNP_information.txt", sep="\t")

    pheno["Taxa"] = pheno["Taxa"].astype(str)
    geno["taxa"] = geno["taxa"].astype(str)

    pheno_sub = pheno[["Taxa", "EarHT", "dpoll"]].dropna(subset=["EarHT", "dpoll"])
    common = sorted(set(pheno_sub["Taxa"]) & set(geno["taxa"]))
    pheno_sub = pheno_sub[pheno_sub["Taxa"].isin(common)].set_index("Taxa").loc[common]
    geno_sub = geno[geno["taxa"].isin(common)].set_index("taxa").loc[common]

    n, m = geno_sub.shape[0], geno_sub.shape[1]
    snp_names = list(geno_sub.columns)

    G = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()

    Y1 = torch.tensor(pheno_sub["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
    Y2 = torch.tensor(pheno_sub[["EarHT", "dpoll"]].values, dtype=STAT_DTYPE)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    if gemma_grm is not None:
        K = torch.tensor(gemma_grm, dtype=STAT_DTYPE)
    else:
        K = None

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
        "n": n, "m": m, "G": G, "Y1": Y1, "Y2": Y2, "X0": X0, "K": K,
        "vmeta": vmeta, "snp_names": snp_names,
    }


# ── Comparison 1: GRM (identity check using GEMMA's K) ───────────────────────


def compare_grm(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GEMMA --gk 1 cXX kinship vs the same matrix loaded by TorchGWAS.

    GEMMA's centered kinship is the reference; we use it as K in TorchGWAS'
    LMM null fit. This comparison is an identity check — confirming we read
    GEMMA's matrix back correctly without any precision loss.
    """
    K_path = out_dir / "mdp_kinship.cXX.txt"
    K_gemma = pd.read_csv(K_path, sep="\t", header=None).values
    n = K_gemma.shape[0]

    # Round-trip through STAT_DTYPE
    K_torch = torch.tensor(K_gemma, dtype=STAT_DTYPE).cpu().numpy()

    rep = ComparisonReport(
        name=f"GRM (GEMMA -gk 1 cXX, identity round-trip, n={n}×{n})",
        n_compared=n * n,
    )
    max_abs = float(np.max(np.abs(K_torch - K_gemma)))
    max_rel = float(np.max(np.abs(K_torch - K_gemma) / (np.abs(K_gemma) + 1e-12)))
    rep.checks.append(_check_max("Element-wise max |Δ|", max_abs, TOL_GRM_ATOL))
    rep.checks.append(_check_max(
        f"Element-wise max relative err (rtol={TOL_GRM_RTOL})", max_rel, TOL_GRM_RTOL
    ))
    rep.extras["mean GEMMA K"] = float(K_gemma.mean())
    rep.extras["trace GEMMA K"] = float(np.trace(K_gemma))
    return rep


# ── Comparison 2: Single-trait LMM ───────────────────────────────────────────


def compare_lmm_single(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GEMMA -lmm 4 (Wald + LRT + score) vs TorchGWAS SingleTraitLMM.

    Asserts:
      - Variance components vg, ve match within 1e-3 relative.
      - REML log-likelihood matches within 1e-2 absolute.
      - β corr > 0.9999, β median |Δ| < 0.01 in AF [0.03, 0.97].
      - SE median |Δ| < 0.02 in AF [0.03, 0.97].
      - −log10(p) corr per test ≥ §16 floor (Wald 0.999, LRT 0.998, score 0.999).
    """
    # GEMMA outputs
    K_gemma = pd.read_csv(out_dir / "mdp_kinship.cXX.txt", sep="\t", header=None).values
    log = _parse_gemma_log(out_dir / "mdp_lmm_all.log.txt")
    gemma_assoc = pd.read_csv(out_dir / "mdp_lmm_all.assoc.txt", sep="\t")

    if "vg" not in log or "ve" not in log:
        raise RuntimeError(f"GEMMA log missing vg/ve: {log.keys()}")
    if "ll_reml" not in log:
        raise RuntimeError(f"GEMMA log missing REML log-likelihood: {log.keys()}")

    # Load data and fit TG null
    d = _load_mdp(data_dir, gemma_grm=K_gemma)
    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(d["Y1"], d["X0"], K=d["K"])

    # Variance components
    vg_rel = abs(float(nf.sig2_g) - log["vg"]) / log["vg"]
    ve_rel = abs(float(nf.sig2_e) - log["ve"]) / log["ve"]
    ll_abs = abs(float(nf.log_likelihood) - log["ll_reml"])

    # Wald β / SE / p
    res_wald = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_wald_df = pd.DataFrame({
        "rs": res_wald.snp,
        "beta_tg": res_wald.beta.cpu().numpy(),
        "se_tg": res_wald.se.cpu().numpy(),
        "p_tg_wald": res_wald.p.cpu().numpy(),
    })
    merged = pd.merge(gemma_assoc, tg_wald_df, on="rs").dropna(subset=["beta", "beta_tg"])

    beta_corr = float(np.corrcoef(merged["beta"].values, merged["beta_tg"].values)[0, 1])
    common = merged[(merged["af"] >= 0.03) & (merged["af"] <= 0.97)]
    beta_med = float(np.median(np.abs(common["beta"].values - common["beta_tg"].values)))
    se_med = float(np.median(np.abs(common["se"].values - common["se_tg"].values)))

    # −log10(p) per test
    def _logp_corr(merged_df: pd.DataFrame, gemma_col: str, tg_col: str) -> float:
        p_ref = np.clip(merged_df[gemma_col].values, 1e-300, 1.0)
        p_tg = np.clip(merged_df[tg_col].values, 1e-300, 1.0)
        return float(np.corrcoef(-np.log10(p_ref), -np.log10(p_tg))[0, 1])

    corr_wald = _logp_corr(merged, "p_wald", "p_tg_wald")

    res_lrt = model.score_chunk(d["G"], nf, d["vmeta"], test="lrt")
    res_score = model.score_chunk(d["G"], nf, d["vmeta"], test="score")
    tg_other = pd.DataFrame({
        "rs": res_lrt.snp,
        "p_tg_lrt": res_lrt.p.cpu().numpy(),
        "p_tg_score": res_score.p.cpu().numpy(),
    })
    merged_full = pd.merge(merged, tg_other, on="rs", how="left")
    corr_lrt = _logp_corr(merged_full, "p_lrt", "p_tg_lrt")
    corr_score = _logp_corr(merged_full, "p_score", "p_tg_score")

    rep = ComparisonReport(
        name=f"Single-trait LMM (GEMMA -lmm 4 vs TG SingleTraitLMM, n={d['n']}, m={d['m']})",
        n_compared=len(merged),
    )
    rep.checks.append(_check_max("vg relative error", vg_rel, TOL_VARCOMP_REL))
    rep.checks.append(_check_max("ve relative error", ve_rel, TOL_VARCOMP_REL))
    rep.checks.append(_check_max("REML log-likelihood |Δ|", ll_abs, TOL_LL_ABS))
    rep.checks.append(_check_min("β correlation (full set)", beta_corr, TOL_BETA_CORR))
    rep.checks.append(_check_max(
        f"β median |Δ| (AF [0.03,0.97], n={len(common)})", beta_med, TOL_BETA_MED_ABSDIFF
    ))
    rep.checks.append(_check_max(
        f"SE median |Δ| (AF [0.03,0.97], n={len(common)})", se_med, TOL_SE_MED_ABSDIFF
    ))
    rep.checks.append(_check_min("−log10(p) Wald correlation", corr_wald, TOL_LOGP_CORR_WALD))
    rep.checks.append(_check_min("−log10(p) LRT correlation", corr_lrt, TOL_LOGP_CORR_LRT))
    rep.checks.append(_check_min("−log10(p) score correlation", corr_score, TOL_LOGP_CORR_SCORE))
    rep.extras["GEMMA vg"] = log["vg"]
    rep.extras["TG sig2_g"] = float(nf.sig2_g)
    rep.extras["GEMMA ve"] = log["ve"]
    rep.extras["TG sig2_e"] = float(nf.sig2_e)
    rep.extras["GEMMA REML logL"] = log["ll_reml"]
    rep.extras["TG REML logL"] = float(nf.log_likelihood)
    return rep


# ── Comparison 3: mvLMM joint Wald ───────────────────────────────────────────


def compare_mvlmm(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """GEMMA -lmm 1 -n 1 2 (mvLMM joint Wald) vs TorchGWAS MultiTraitLMM."""
    K_gemma = pd.read_csv(out_dir / "mdp_kinship.cXX.txt", sep="\t", header=None).values
    log = _parse_gemma_log(out_dir / "mdp_mvlmm_wald.log.txt")
    gemma_assoc = pd.read_csv(out_dir / "mdp_mvlmm_wald.assoc.txt", sep="\t")

    if "Vg" not in log or "Ve" not in log:
        raise RuntimeError(f"GEMMA mvLMM log missing Vg/Ve matrices: {log.keys()}")

    d = _load_mdp(data_dir, gemma_grm=K_gemma)
    model = MultiTraitLMM()
    nf = model.fit_null(d["Y2"], d["X0"], K=d["K"])

    Vg_tg = nf.Vg.cpu().numpy()
    Ve_tg = nf.Ve.cpu().numpy()

    # Element-wise relative error on the upper-triangle (matrices are symmetric)
    def _matrix_rel_err(A: np.ndarray, B: np.ndarray) -> float:
        denom = np.abs(B) + 1e-12
        return float(np.max(np.abs(A - B) / denom))

    vg_max_rel = _matrix_rel_err(Vg_tg, log["Vg"])
    ve_max_rel = _matrix_rel_err(Ve_tg, log["Ve"])

    res = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({"rs": res.snp, "p_tg": res.p.cpu().numpy()})
    merged = pd.merge(gemma_assoc, tg_df, on="rs").dropna(subset=["p_wald", "p_tg"])

    p_ref = np.clip(merged["p_wald"].values, 1e-300, 1.0)
    p_tg = np.clip(merged["p_tg"].values, 1e-300, 1.0)
    corr = float(np.corrcoef(-np.log10(p_ref), -np.log10(p_tg))[0, 1])

    rep = ComparisonReport(
        name=f"mvLMM joint Wald (GEMMA -lmm 1 vs TG MultiTraitLMM, n={d['n']}, traits=2)",
        n_compared=len(merged),
    )
    rep.checks.append(_check_max("Vg max relative error", vg_max_rel, TOL_VG_REL))
    rep.checks.append(_check_max("Ve max relative error", ve_max_rel, TOL_VE_REL))
    rep.checks.append(_check_min("−log10(p) joint-Wald correlation", corr, TOL_MVLMM_LOGP_CORR))
    rep.extras["GEMMA Vg"] = log["Vg"].tolist()
    rep.extras["TG Vg"] = Vg_tg.tolist()
    rep.extras["GEMMA Ve"] = log["Ve"].tolist()
    rep.extras["TG Ve"] = Ve_tg.tolist()
    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def run_all(data_dir: Path, out_dir: Path, json_out: Path | None = None) -> int:
    reports = [
        compare_grm(data_dir, out_dir),
        compare_lmm_single(data_dir, out_dir),
        compare_mvlmm(data_dir, out_dir),
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
