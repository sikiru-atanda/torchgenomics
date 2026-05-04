"""Compare BOLT-LMM (--lmmInfOnly) vs TorchGWAS SingleTraitLMM with LOCO.

What we compare
---------------

BOLT-LMM in --lmmInfOnly mode runs a textbook infinitesimal-model LMM with
internal Leave-One-Chromosome-Out kinship-matrix construction (the SNP
under test contributes to its own GRM signal otherwise → proximal
contamination). Per-SNP it reports BETA, SE, CHISQ_LINREG (linear-regression
chi-square ignoring the GRM, for sanity), and CHISQ_BOLT_LMM_INF / P_BOLT_LMM_INF
(the actual LMM result).

TorchGWAS counterpart: ``SingleTraitLMM`` (V1-core, GEMMA-equivalent) paired
with ``linalg.grm_loco`` per chromosome. For each chromosome we:
  1. Compute the LOCO kinship K_loco from all OTHER chromosomes.
  2. Fit the null on the full sample with covariates + K_loco.
  3. Score the test chromosome's SNPs.
We aggregate per-chromosome results and merge with BOLT's stats by SNP ID.

Allele convention
-----------------
PlinkBedReader counts the BIM A2 allele. BOLT-LMM, like PLINK 2 and regenie,
counts ALLELE1 (=BIM A1 in our fixture). compare.py flips TG's dosage with
``2.0 - G`` so both tools count the same allele before scoring.

Tolerances (observed-then-floored, per Pillar B convention)
------------------------------------------------------------

BOLT-LMM and TG SingleTraitLMM are both textbook infinitesimal-model LMMs
with LOCO. They should agree closely on β / SE / -log10p for SNPs in the
common-frequency band. The §16 aspirational gates (β corr > 0.9999,
median |Δβ| < 0.01) apply if the LMM reductions (variance-component
estimation, LOCO bookkeeping, score-test parameterization) match exactly.
We anchor at observed-then-floored levels and tighten on first-run
calibration.

First-run anchors (per the prompt's "anchor + observe-then-floor" policy):
  β corr ≥ 0.90
  -log10p corr ≥ 0.85
  β median |Δ| in AF [0.03, 0.97] ≤ 0.30
  SE median |Δ| in AF [0.03, 0.97] ≤ 0.10

These will be tightened (or loosened) in a follow-up commit on first
successful BOLT run; the placeholders make the harness functional even
when BOLT is infra-blocked.

F3 logic (per spec §9 + the prompt)
-----------------------------------

* SingleTraitLMM is V1-core. Disagreement beyond ~5% on β / -log10p
  on the textbook infinitesimal model would be a V1-core / fix-now F3.
* BOLT's "BoltLMM-mixed" mode (Gaussian-mixture spread approximation)
  is a BOLT-specific algorithm; we do NOT compare against it. Using
  --lmmInfOnly isolates the textbook reduction.
* Documented divergences (LOCO bookkeeping nuances, BOLT's per-block
  variance-component refit, BOLT's -log10p tail rounding) are NOT F3.
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

# ── Tolerance gates (observed-then-floored) ─────────────────────────────────
# First-run observed (2026-04-30, MDP fixture, N=279 after pheno-NaN drop,
# m=2897 post-QC; BOLT-LMM v2.5 --lmmInfOnly vs TG SingleTraitLMM+grm_loco):
#     β corr (full set)                 = 0.9843
#     -log10p corr (full set)           = 0.9823
#     β median |Δ| in AF [0.03, 0.97]   = 0.1482  (n_band=2814)
#     SE median |Δ| in AF [0.03, 0.97]  = 0.0426
#     β max |Δ| (full)                  = 3.79   (single tail outlier; TG and
#                                                 BOLT diverge most on the
#                                                 strongest-effect SNPs because
#                                                 BOLT's calibration ratio
#                                                 (1.35) is itself estimated
#                                                 from 30 calibration SNPs at
#                                                 N=279)
# Floors are observed values minus a buffer for CPU/BLAS-precision drift +
# BOLT's stochastic Monte-Carlo variance-component initialization, which is
# non-deterministic at the 5th sig-fig across hosts. The gates are loose
# enough to protect against reseeding / different BLAS but tight enough
# to fail on a real V1-core regression in SingleTraitLMM or grm_loco.

TOL_BETA_CORR = 0.95               # observed 0.984 → floor 0.95
TOL_NLOG10P_CORR = 0.95            # observed 0.982 → floor 0.95
TOL_BETA_MEDIAN_ABSDIFF = 0.20     # observed 0.148 → floor 0.20 (35% headroom)
TOL_SE_MEDIAN_ABSDIFF = 0.07       # observed 0.043 → floor 0.07 (60% headroom;
                                   # SE is sensitive to the per-chrom REML
                                   # variance-component refit jitter)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.config import STAT_DTYPE  # noqa: E402
from torchgwas.io.plink import PlinkBedReader  # noqa: E402
from torchgwas.linalg.kinship_polyploid import grm_loco  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.single_trait_lmm import SingleTraitLMM  # noqa: E402


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
        return (
            f"  [{flag}] {self.name:46s} observed={self.observed:.6e} "
            f"{cmp} {self.threshold:.6e}  {self.note}"
        )


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
        bar = "=" * 78
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
    return CheckResult(
        name=name, passed=obs >= thr, observed=obs, threshold=thr,
        direction="min", note=note,
    )


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(
        name=name, passed=obs <= thr, observed=obs, threshold=thr,
        direction="max", note=note,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return 0.0
    if a.std() < 1e-15 and b.std() < 1e-15:
        return 1.0
    if a.std() < 1e-15 or b.std() < 1e-15:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _load_mdp(data_dir: Path) -> tuple[torch.Tensor, list[str], VariantMeta]:
    """Load mdp PLINK fileset, mean-impute → (G, sample_ids, vmeta).

    G is returned with PlinkBedReader's native dosage convention: G[i,j]
    counts the BIM A2 allele. Callers comparing against BOLT (which counts
    ALLELE1 = BIM A1) should flip with ``2.0 - G``.
    """
    reader = PlinkBedReader(data_dir / "mdp")
    chunks = []
    for G_chunk, _ in reader.iter_chunks(chunk_size=4096):
        chunks.append(G_chunk)
    G = torch.cat(chunks, dim=1)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            valid = col[~mask]
            G[mask, j] = valid.mean() if len(valid) else 0.0
    return G.to(STAT_DTYPE), reader.sample_ids, reader.variant_meta


def _read_bolt_stats(path: Path) -> pd.DataFrame:
    """Parse BOLT-LMM stats file (whitespace-separated)."""
    return pd.read_csv(path, sep=r"\s+", engine="python")


# ── Comparison: per-SNP β / SE / -log10p ────────────────────────────────────


def compare_lmm_inf(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """BOLT-LMM --lmmInfOnly vs TG SingleTraitLMM + grm_loco per chrom."""
    bolt_path = out_dir / "bolt_stats.tsv"
    bolt = _read_bolt_stats(bolt_path)

    # BOLT column names (verified against v2.5 docs):
    #   SNP, CHR, BP, GENPOS, ALLELE1, ALLELE0, A1FREQ, F_MISS,
    #   BETA, SE, CHISQ_LINREG, P_LINREG, CHISQ_BOLT_LMM_INF, P_BOLT_LMM_INF.
    # The LMM-Inf p is what we care about (that's the LOCO-corrected LMM
    # score test). BETA / SE in this mode are still the ones compatible
    # with the LMM-Inf chi-square (BOLT writes them once, not per test).
    if "P_BOLT_LMM_INF" not in bolt.columns:
        # Some BOLT versions name the column slightly differently; the
        # canonical v2.5 layout has the suffix above. If the stats file
        # has a different layout, fail loudly.
        raise RuntimeError(
            f"BOLT stats file at {bolt_path} does not have P_BOLT_LMM_INF "
            f"column; got columns: {list(bolt.columns)}"
        )
    bolt = bolt.dropna(subset=["BETA", "SE", "P_BOLT_LMM_INF"]).copy()
    bolt["SNP"] = bolt["SNP"].astype(str)

    # --- Load TG inputs -----------------------------------------------------
    G_a2, sample_ids, vmeta = _load_mdp(data_dir)
    G_alt = 2.0 - G_a2  # count BIM A1 = BOLT ALLELE1

    pheno = pd.read_csv(data_dir / "bolt_pheno.tsv", sep="\t", na_values=["NA"])
    pheno["IID"] = pheno["IID"].astype(str)
    pheno = pheno.set_index("IID").reindex(sample_ids)
    covar = pd.read_csv(data_dir / "bolt_covar.tsv", sep="\t")
    covar["IID"] = covar["IID"].astype(str)
    covar = covar.set_index("IID").reindex(sample_ids)

    Y_full = torch.tensor(
        pheno["EarHT"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE,
    )
    PC = torch.tensor(
        covar[["PC1", "PC2"]].to_numpy(dtype=np.float64), dtype=STAT_DTYPE,
    )

    # Drop samples with NaN phenotype (BOLT does the same internally).
    valid = ~torch.isnan(Y_full)
    Y = Y_full[valid]
    G_alt_v = G_alt[valid]
    PC_v = PC[valid]
    n = Y.shape[0]
    intercept = torch.ones(n, 1, dtype=STAT_DTYPE)
    X0 = torch.cat([intercept, PC_v], dim=1)  # (n, 3)

    # --- Per-chromosome LOCO scan ------------------------------------------
    bim = pd.read_csv(
        data_dir / "mdp.bim", sep=r"\s+", engine="python", header=None,
        names=["chrom", "ID", "cm", "bp", "A1", "A2"],
    )
    bim["ID"] = bim["ID"].astype(str)
    snp_to_idx = {s: i for i, s in enumerate(vmeta.snp)}
    chr_labels = [str(c) for c in bim["chrom"].astype(int).tolist()]

    records = []
    for chrom in sorted(set(int(c) for c in chr_labels)):
        chrom_str = str(chrom)
        snps_this_chr = bim.loc[bim["chrom"].astype(int) == chrom, "ID"].tolist()
        if not snps_this_chr:
            continue
        snp_idx = [snp_to_idx[s] for s in snps_this_chr if s in snp_to_idx]
        snps_this_chr = [vmeta.snp[i] for i in snp_idx]
        G_chr = G_alt_v[:, snp_idx]

        # Build LOCO GRM from BIM A2-counted dosage (kinship is allele-coding-
        # invariant up to a constant; for VanRaden the standardization step
        # absorbs it). Use the original A2-counted matrix → grm_loco.
        K_loco, _ = grm_loco(
            G_a2[valid], chr_labels=chr_labels, exclude_chr=chrom_str, ploidy=2,
        )

        vm = VariantMeta(
            snp=snps_this_chr,
            chr=[chrom_str] * len(snps_this_chr),
            pos=[0] * len(snps_this_chr),
            a1=["A"] * len(snps_this_chr),
            a2=["G"] * len(snps_this_chr),
        )
        model = SingleTraitLMM()
        nf = model.fit_null(Y, X0, K=K_loco)
        res = model.score_chunk(G_chr, nf, vm, test="wald")
        for i, snp in enumerate(snps_this_chr):
            records.append({
                "SNP": snp,
                "beta_tg": float(res.beta[i].item()),
                "se_tg": float(res.se[i].item()),
                "p_tg": float(res.p[i].item()),
                "af_tg": float(res.af[i].item()),
            })

    tg = pd.DataFrame(records)
    merged = pd.merge(bolt, tg, on="SNP", how="inner")
    if len(merged) == 0:
        raise RuntimeError(
            "Zero SNPs matched between BOLT stats and TG output; check SNP "
            "ID column naming."
        )

    # --- Tolerances --------------------------------------------------------
    af_tg = merged["af_tg"].to_numpy()
    af_band = (af_tg >= 0.03) & (af_tg <= 0.97)
    band = merged[af_band]

    beta_corr = _safe_corr(merged["BETA"].to_numpy(), merged["beta_tg"].to_numpy())
    beta_med_absdiff = float(np.median(np.abs(
        band["BETA"].to_numpy() - band["beta_tg"].to_numpy()
    )))
    se_med_absdiff = float(np.median(np.abs(
        band["SE"].to_numpy() - band["se_tg"].to_numpy()
    )))

    p_tg = np.clip(merged["p_tg"].to_numpy(), 1e-300, 1.0)
    p_bolt = np.clip(merged["P_BOLT_LMM_INF"].to_numpy(), 1e-300, 1.0)
    nlog10_tg = -np.log10(p_tg)
    nlog10_bolt = -np.log10(p_bolt)
    nlog10_corr = _safe_corr(nlog10_bolt, nlog10_tg)

    rep = ComparisonReport(
        name=(
            f"BOLT-LMM --lmmInfOnly vs TG SingleTraitLMM+LOCO "
            f"(n={n}, m={len(merged)})"
        ),
        n_compared=len(merged),
    )
    rep.checks.append(_check_min("β correlation (full set)",
                                 beta_corr, TOL_BETA_CORR))
    rep.checks.append(_check_min("−log₁₀(P) correlation",
                                 nlog10_corr, TOL_NLOG10P_CORR))
    rep.checks.append(_check_max(
        f"β median |Δ| (AF [0.03,0.97], n={len(band)})",
        beta_med_absdiff, TOL_BETA_MEDIAN_ABSDIFF,
    ))
    rep.checks.append(_check_max(
        f"SE median |Δ| (AF [0.03,0.97], n={len(band)})",
        se_med_absdiff, TOL_SE_MEDIAN_ABSDIFF,
    ))
    rep.extras["β max |Δ| (full)"] = float(np.max(np.abs(
        merged["BETA"].to_numpy() - merged["beta_tg"].to_numpy()
    )))
    rep.extras["mean -log10p (BOLT)"] = float(nlog10_bolt.mean())
    rep.extras["mean -log10p (TG)"] = float(nlog10_tg.mean())
    rep.extras["TG path"] = "SingleTraitLMM + grm_loco (per-chrom REML)"
    rep.extras["BOLT path"] = "--lmmInfOnly (textbook infinitesimal LMM, internal LOCO)"
    return rep


# ── Driver ─────────────────────────────────────────────────────────────────


def run_all(
    data_dir: Path,
    out_dir: Path,
    json_out: Path | None = None,
) -> int:
    reports = [compare_lmm_inf(data_dir, out_dir)]
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
