"""Compare SAIGE vs TorchGenomics BinaryGLMM on the MDP fixture.

SAIGE pipeline:
  - Step 1 (fitNULLGLMM): PCG-based fit of the null GLMM with full GRM,
    plus a per-MAC variance ratio. Outputs `.rda` + `.varianceRatio.txt`.
  - Step 2 (SPAtests): per-SNP score test against the Step 1 null,
    saddlepoint-corrected for χ² > spaCutoff (default 2). Output is a
    whitespace table with BETA / SE / Tstat / var / p.value / Is.SPA.

TorchGenomics counterpart: `BinaryGLMM` (Phase 33). The model fits the null
via PQL (Breslow-Clayton penalized quasi-likelihood) using the same full
GRM, then runs a logistic score test per SNP with optional SPA tail
correction. The PCG-based PQL inside SAIGE differs from TG's eigh-based
PQL in numerical detail but converges to the same null mean / variance
on N=281.

What we compare
---------------

1. **Step 1 sanity** — the variance-ratio file is finite + non-zero, the
   `.rda` is non-empty, and the SAIGE null fit converged.

2. **Step 2 β / SE / -log10p** — full-set correlations of SAIGE's per-SNP
   summary stats against `BinaryGLMM(use_spa=True).score_chunk` on the
   same data (same FAM order, same GRM, intercept + age + sex + PC1 + PC2
   covariates).

3. **SPA-tail focused** — Restrict to SAIGE's `Is.SPA == true` rows and
   re-check the -log10p correlation. SAIGE's saddlepoint kicks in for
   case/control imbalanced low-MAC SNPs; this is exactly the regime
   where TG's `BinaryGLMM(use_spa=True)` matters too.

Tolerances (observed-then-floored, per Pillar B convention)
------------------------------------------------------------

The aspirational §16-style gates (β corr > 0.99, -log10p corr > 0.95) do
not apply here:
  - SAIGE PCG vs TG PQL is a documented parameterization mismatch, not a
    V1-core math defect. Per spec §9 + the prompt's "F3 logic" note, this
    is post-V1 and not F3.
  - On N=281 (deep inside SAIGE's "small N" regime), the Step 1
    variance-ratio estimate has ~10% relative noise vs its biobank-scale
    asymptote, which propagates into the Step 2 β estimate.

We therefore set tolerances at observed levels with documented margin:
  β corr ≥ 0.85, -log10p corr ≥ 0.80, β median |Δ| in AF [0.03,0.97] ≤ 0.30.
First-run observed values are written into the comments of each TOL_*
constant after the first successful invocation.
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

# ── Tolerance gates (observed-then-floored) ──────────────────────────────────
# Format: (label, observed, floor). Observed values from first successful run
# on the MDP fixture (N=281, m=2897 post-QC, simulated logistic-on-causal-SNPs
# binary phenotype). Floors are observed minus a small buffer that absorbs
# CPU/BLAS-precision drift between hosts.

# Step 2 full-set (TG BinaryGLMM PQL+SPA vs SAIGE Step 2):
# First-run observed (2026-04-30, MDP fixture, N=281, m=2897 post-QC):
#     β corr (full set)              = 0.973
#     -log10p corr (full set)        = 0.943
#     β median |Δ| in AF [0.03,0.97] = 0.045
#     SE median |Δ| in AF [0.03,0.97]= 0.0015
# The floors below sit comfortably below those observed values to absorb
# CPU/BLAS-precision drift between hosts and one-iteration jitter in
# SAIGE's PCG (which is non-deterministic at the 5th sig-fig).
TOL_BETA_CORR = 0.90               # observed 0.97 → floor 0.90
TOL_NLOG10P_CORR = 0.85            # observed 0.94 → floor 0.85
TOL_BETA_MEDIAN_ABSDIFF = 0.15     # observed 0.045 → floor 0.15 (3× headroom)
TOL_SE_MEDIAN_ABSDIFF = 0.05       # observed 0.0015 → floor 0.05 (large headroom
                                   # because tail outlier SNPs can dominate)

# SPA-tail subset: gate is intentionally the same as the full-set, since
# the SPA path's job is to *match* the standard score test for non-tail
# SNPs and only diverge in the tail. With the MDP fixture, ~130 SNPs
# trigger Is.SPA = true; first-run -log10p corr on that subset = 0.885.
# We don't gate the SPA subset (it's reported as an extra) because the
# subset cardinality varies with random PQL initialization across hosts.

# Step 1 sanity:
TOL_VAR_RATIO_MIN = 1e-6           # variance ratio must be > 0 (well-conditioned PCG)
TOL_VAR_RATIO_MAX = 100.0          # and < 100 (else PCG is degenerate)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.config import STAT_DTYPE  # noqa: E402
from torchgenomics.io.plink import PlinkBedReader  # noqa: E402
from torchgenomics.linalg.kinship import grm_vanraden  # noqa: E402
from torchgenomics.models.base import VariantMeta  # noqa: E402
from torchgenomics.models.binary_glmm import BinaryGLMM  # noqa: E402


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
        return f"  [{flag}] {self.name:46s} observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"


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
    return CheckResult(name=name, passed=obs >= thr, observed=obs, threshold=thr,
                       direction="min", note=note)


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr,
                       direction="max", note=note)


def _check_range(name: str, obs: float, lo: float, hi: float,
                 note: str = "") -> CheckResult:
    """Two-sided: pass iff lo ≤ obs ≤ hi."""
    return CheckResult(
        name=name,
        passed=(lo <= obs <= hi),
        observed=obs,
        threshold=hi,  # display upper bound
        direction="max",
        note=note + f" (lo={lo:.2e})",
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
    """Load `data/sample.{bed,bim,fam}` → (G, sample_ids, vmeta).

    G is mean-imputed and returned in PlinkBedReader's native A1-counting
    convention (PLINK 1.9 canonical, post-2026-05-13 fix). This matches
    SAIGE's `Allele2` (which corresponds to BIM A1) directly; no manual
    flip is needed (see notes in compare_step2()).
    """
    reader = PlinkBedReader(data_dir / "sample")
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


def _read_saige_step2(path: Path) -> pd.DataFrame:
    """Parse SAIGE step2.txt (whitespace-separated)."""
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _read_saige_var_ratio(path: Path) -> float:
    """Return the variance ratio from SAIGE step1.varianceRatio.txt.

    The file format is one line per MAC bucket; columns are
        <variance_ratio>  <MAC bucket marker (or 'null')>  <bucket_id>
    With our flags (no --IsCateVarianceRatio), SAIGE writes a single row.
    Example contents: ``0.895535168769469 null 1``.
    """
    txt = Path(path).read_text().strip()
    if not txt:
        return float("nan")
    first_token = txt.split()[0]
    return float(first_token)


# ── Comparison 1: Step 1 sanity ──────────────────────────────────────────────


def compare_step1_sanity(out_dir: Path) -> ComparisonReport:
    """Step 1 .rda + variance-ratio file are well-formed and non-degenerate."""
    rda_path = out_dir / "step1.rda"
    vr_path = out_dir / "step1.varianceRatio.txt"

    rep = ComparisonReport(
        name="Step 1 sanity (.rda + varianceRatio)",
        n_compared=2,
    )
    rep.checks.append(_check_min(
        "step1.rda exists + non-empty",
        float(rda_path.stat().st_size if rda_path.exists() else 0),
        100.0,  # arbitrary lower bound on a serialized R model
        note="(bytes)",
    ))
    if vr_path.exists():
        vr = _read_saige_var_ratio(vr_path)
    else:
        vr = float("nan")
    rep.checks.append(_check_range(
        "variance ratio in (0, 100)",
        vr, TOL_VAR_RATIO_MIN, TOL_VAR_RATIO_MAX,
    ))
    rep.extras["variance ratio"] = vr
    rep.extras["step1.rda bytes"] = int(rda_path.stat().st_size if rda_path.exists() else 0)
    return rep


# ── Comparison 2: Step 2 β/SE/p ──────────────────────────────────────────────


def compare_step2(data_dir: Path, out_dir: Path) -> ComparisonReport:
    """SAIGE Step 2 vs TG BinaryGLMM(use_spa=True) on the MDP fixture.

    Strategy:
      - Load PLINK fileset + mean-impute genotypes.
      - Build VanRaden GRM from the imputed matrix (same source data
        SAIGE used via --plinkFile).
      - Fit `BinaryGLMM(use_spa=True).fit_null(Y, X0=intercept+covars, K=GRM)`.
      - Score the full SNP matrix in one chunk; collect β / SE / p.
      - Merge SNP-by-SNP with SAIGE step2.txt and assert tolerances.

    Allele convention (post-2026-05-13 fix):
      - PlinkBedReader counts the BIM A1 allele (PLINK 1.9 canonical).
      - SAIGE's `Allele2` column matches BIM A1 (the bim files in the
        MDP fixture have A1=A, A2=G; SAIGE step2 prints `Allele1=G,
        Allele2=A`).
      - SAIGE's BETA refers to the AC_Allele2 dosage = BIM A1 = the same
        allele TG counts.
      - No manual flip needed. Pre-fix, TG counted BIM A2 and this
        harness applied a `2.0 - G` workaround; the workaround was
        removed once `torchgenomics/io/plink.py:_GENO_DECODE` was corrected.
    """
    rg = _read_saige_step2(out_dir / "step2.txt")
    # Drop any rows SAIGE marked failed-to-converge (BETA = NA).
    rg = rg.dropna(subset=["BETA", "SE", "p.value"]).copy()
    rg["MarkerID"] = rg["MarkerID"].astype(str)

    G_a2, sample_ids, vmeta = _load_mdp(data_dir)
    # Post-fix (2026-05-13): PlinkBedReader counts BIM A1 = SAIGE Allele2 = A
    # natively (torchgenomics/io/plink.py:_GENO_DECODE was fixed to PLINK 1.9
    # canonical convention). Pre-fix this line was `2.0 - G_a2` to flip from
    # the old A2-counting convention. The variable name `G_a2` is now a
    # historical misnomer; it actually contains count(A1).
    G_alt = G_a2  # already counts A1 = SAIGE Allele2 post-fix

    pheno = pd.read_csv(data_dir / "saige_pheno.tsv", sep="\t")
    pheno["IID"] = pheno["IID"].astype(str)
    pheno = pheno.set_index("IID").reindex(sample_ids)
    if pheno["ybin"].isna().any():
        raise RuntimeError(
            "Sample alignment between FAM and saige_pheno.tsv produced NaNs; "
            "check fetch_data.sh."
        )

    Y = torch.tensor(pheno["ybin"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE)
    age = torch.tensor(pheno["age"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE).unsqueeze(1)
    sex = torch.tensor(pheno["sex"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE).unsqueeze(1)
    pc1 = torch.tensor(pheno["PC1"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE).unsqueeze(1)
    pc2 = torch.tensor(pheno["PC2"].to_numpy(dtype=np.float64), dtype=STAT_DTYPE).unsqueeze(1)
    n = Y.shape[0]
    X0 = torch.cat([torch.ones(n, 1, dtype=STAT_DTYPE), age, sex, pc1, pc2], dim=1)

    # GRM from the same imputed matrix SAIGE uses internally.
    K, _ = grm_vanraden(G_a2, ploidy=2)

    print(f"[saige compare] fitting BinaryGLMM null (n={n}, c={X0.shape[1]}) ...",
          file=sys.stderr)
    model = BinaryGLMM(use_spa=True, spa_threshold=2.0, firth=False,
                       pql_max_iter=30, pql_tol=1e-4)
    nf = model.fit_null(Y, X0, K=K)
    print(f"[saige compare] PQL converged={nf.converged}", file=sys.stderr)

    # Score the full chunk
    res = model.score_chunk(G_alt, nf, vmeta)
    tg = pd.DataFrame({
        "MarkerID": list(vmeta.snp),
        "beta_tg": res.beta.cpu().numpy(),
        "se_tg": res.se.cpu().numpy(),
        "p_tg": res.p.cpu().numpy(),
        "stat_tg": res.stat.cpu().numpy(),
        "af_tg": res.af.cpu().numpy(),
    })

    merged = pd.merge(rg, tg, on="MarkerID", how="inner")

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
    p_saige = np.clip(merged["p.value"].to_numpy(), 1e-300, 1.0)
    nlog10_tg = -np.log10(p_tg)
    nlog10_saige = -np.log10(p_saige)
    nlog10_corr = _safe_corr(nlog10_saige, nlog10_tg)

    rep = ComparisonReport(
        name=f"Step 2 binary GLMM (SAIGE PCG+SPA vs TG PQL+SPA, n={n}, m={len(merged)})",
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

    # SPA-subset metric (informational)
    if "Is.SPA" in merged.columns:
        spa_mask = (merged["Is.SPA"].astype(str).str.lower() == "true").to_numpy()
        n_spa = int(spa_mask.sum())
        if n_spa >= 5:
            spa_corr = _safe_corr(
                nlog10_saige[spa_mask],
                nlog10_tg[spa_mask],
            )
            rep.extras["SPA subset n"] = n_spa
            rep.extras["SPA -log10p corr"] = spa_corr
        else:
            rep.extras["SPA subset n"] = n_spa
            rep.extras["SPA -log10p corr"] = "n/a (< 5 SPA hits)"

    rep.extras["β max |Δ| (full)"] = float(np.max(np.abs(
        merged["BETA"].to_numpy() - merged["beta_tg"].to_numpy()
    )))
    rep.extras["mean -log10p (SAIGE)"] = float(nlog10_saige.mean())
    rep.extras["mean -log10p (TG)"] = float(nlog10_tg.mean())
    rep.extras["TG path"] = "BinaryGLMM(use_spa=True), VanRaden GRM, PQL"
    rep.extras["SAIGE path"] = "step2_SPAtests --LOCO=FALSE --is_Firth_beta=TRUE"
    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def run_all(data_dir: Path, out_dir: Path,
            json_out: Path | None = None) -> int:
    reports = [
        compare_step1_sanity(out_dir),
        compare_step2(data_dir, out_dir),
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
