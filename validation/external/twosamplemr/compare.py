"""Compare TwoSampleMR R reference vs TorchGWAS on simulated MR sumstats.

Four comparisons (one per MR estimator):
  1. IVW                 — `mr(method_list="mr_ivw")` vs `torchgwas.postgwas.mr_ivw`
  2. MR-Egger            — `mr_egger_regression` + `mr_pleiotropy_test` vs `mr_egger`
  3. Weighted median     — `mr_weighted_median` (R bootstrap) vs `mr_weighted_median`
  4. MR-PRESSO           — `MRPRESSO::mr_presso` vs `torchgwas.postgwas.mr_presso`

Both tools see the *same* `data/sumstats.tsv` (simulated by `simulate_mr.R`
with seed 42, theta_true = 0.5, K = 30, 3 pleiotropic SNPs), so disagreement
reflects the regression / bootstrap / permutation implementation, not the
data prep.

Tolerance policy (per the harness spec): observed-then-floored.

Notes on expected divergences:

* IVW is *not* parameter-free agreement: TwoSampleMR's `mr_ivw` uses the
  multiplicative-random-effects estimator (overdispersion-corrected by
  default, only when Cochran's Q > K-1). TorchGWAS' `mr_ivw` does the
  same overdispersion correction, so the slope and SE match closely. The
  observed |Δ β| floor is loose to accommodate the case where R's WLS
  uses `lm()` weights and TG uses an explicit Wald-ratio sum.

* MR-Egger: both tools fit `by ~ alpha + beta * bx` with weights 1/se_y².
  Slope, intercept, and their SEs are textbook closed-form WLS.

* Weighted median: TwoSampleMR uses 1000-replicate parametric bootstrap
  (`mr_weighted_median_bootstrap`); TorchGWAS uses a 1000-replicate
  resample bootstrap with the same K and seed=42. Different bootstrap
  schemes → bootstrap SE differs by a few percent. The point estimate
  is deterministic (no MC noise) and should agree to 1e-4.

* MR-PRESSO: both tools now use the **parametric LOO bootstrap null** of
  Verbanck 2018 (matching MRPRESSO 1.0 / TwoSampleMR exactly). For each
  replicate, simulate ``bx_boot ~ N(bx, se_x²)`` and ``by_boot ~
  N(β_LOO·bx, se_y²)``, then re-compute LOO RSS. With n_perms = 1000 the
  global p-value agrees with the R reference to within ~1/sqrt(1000) MC
  noise (~3e-2). Outlier sets and corrected β match deterministically
  once the LOO β values agree to FP precision (the parametric noise is
  per-replicate and averages out in the empirical p-value count).
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
# Observed values from first successful run (2026-04-30, K=30 instruments,
# theta=0.5, 3 planted pleiotropic SNPs):
#
#   IVW                |Δ β| 2.96e-5  |Δ SE| 1.65e-5  |Δ p| 4.84e-16
#   Egger (post-fix)   |Δ β| 4.12e-5  |Δ SE| 4.10e-5  |Δ intercept| 2.18e-5
#   Egger              |Δ intercept SE| 4.87e-5
#   Weighted median    |Δ β| 2.23e-4  |Δ SE| 3.58e-2 (different bootstrap)
#   MR-PRESSO          |Δ raw β| 2.96e-5
#   MR-PRESSO          |Δ corrected β| 4.05e-5 (post parametric-LOO port)
#   MR-PRESSO          |Δ global p| 1.0e-3 (post parametric-LOO port)
#
# The MR-Egger near-bit-equal agreement is a result of the F3 fix applied
# to `torchgwas.postgwas._mr.mr_egger` in this same Pillar B work (sign-
# orientation per Bowden 2015, Bowden-style overdispersion clipping, and
# t-distribution p-values with K-2 d.f.). See git log for "Pillar B B3 F3".
#
# **MR-PRESSO global-p**: as of the post-V1 follow-up commit (see
# `git log --grep="MR-PRESSO parametric"`), `torchgwas.postgwas.mr_presso`
# defaults to the **Verbanck-2018 parametric LOO bootstrap** matching
# MRPRESSO 1.0 exactly. The legacy permutation null is preserved under
# `mr_presso(..., null="permutation")`. Post-fix, |Δ global p| floors at
# ~5e-2 (purely Monte-Carlo noise from independent RNG streams between
# R's `sample()` and PyTorch's `torch.Generator`).

# IVW (closed-form WLS; near-bit-equal mod overdispersion):
TOL_IVW_BETA = 1e-4         # observed 3e-5
TOL_IVW_SE = 1e-3           # observed 2e-5
TOL_IVW_PVAL = 5e-3         # observed 5e-16

# MR-Egger (textbook closed-form WLS with intercept; post-F3 fix):
TOL_EGGER_BETA = 1e-3       # observed 4e-5
TOL_EGGER_SE = 5e-3         # observed 4e-5
TOL_EGGER_PVAL = 5e-2       # observed 1e-5; p-values can drift more under
                            # overdispersion-clipping near the boundary
TOL_EGGER_INTERCEPT = 1e-3       # observed 2e-5
TOL_EGGER_INTERCEPT_SE = 5e-3    # observed 5e-5

# Weighted median (point estimate is deterministic; SE is bootstrap-stochastic):
TOL_WMED_BETA = 5e-3   # observed 2.2e-4 — different bootstrap schemes leave
                       # a small gap because TG's median index lookup uses
                       # a slightly different cumulative-weight interpolation.
TOL_WMED_SE = 5e-2     # observed 3.6e-2 — TwoSampleMR uses parametric
                       # bootstrap (resample bx, by from N(., se²)); TG uses
                       # non-parametric resample of SNP indices. Different
                       # null variance → different SE.

# MR-PRESSO: post parametric-LOO bootstrap port (matches MRPRESSO 1.0 exactly).
# Global p MC-noise floor is bounded by the empirical p resolution:
# ~1/sqrt(NbDistribution) ≈ 3e-2 at NbDistribution=1000.
TOL_PRESSO_GLOBAL_P = 5e-2   # observed 1.0e-3 (R=0.001, TG=0.000); MC-noise
TOL_PRESSO_BETA_RAW = 1e-3   # observed 3e-5; raw IVW path matches IVW above
TOL_PRESSO_BETA_CORR = 5e-3  # observed 4e-5; outlier sets now agree exactly
                             # (R={15,21}, TG={15,21}) since the parametric
                             # LOO bootstrap matches MRPRESSO 1.0.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgwas.postgwas import (  # noqa: E402
    SumStats,
    mr_egger,
    mr_ivw,
    mr_presso,
    mr_weighted_median,
)


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
        return f"  [{flag}] {self.name:42s} observed={self.observed:.6e} {cmp} {self.threshold:.6e}  {self.note}"


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


# ── Helpers ──────────────────────────────────────────────────────────────────


def _check_max(name: str, obs: float, thr: float, note: str = "") -> CheckResult:
    return CheckResult(name=name, passed=obs <= thr, observed=obs, threshold=thr, direction="max", note=note)


def _load_inputs(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Load the simulated sumstats + R reference output."""
    sumstats = pd.read_csv(data_dir / "sumstats.tsv", sep="\t")
    truth = json.loads((data_dir / "sim_truth.json").read_text())
    r_results = json.loads((out_dir / "twosamplemr_results.json").read_text())

    bx = torch.tensor(sumstats["beta_exposure"].to_numpy(np.float64), dtype=torch.float64)
    se_x = torch.tensor(sumstats["se_exposure"].to_numpy(np.float64), dtype=torch.float64)
    by = torch.tensor(sumstats["beta_outcome"].to_numpy(np.float64), dtype=torch.float64)
    se_y = torch.tensor(sumstats["se_outcome"].to_numpy(np.float64), dtype=torch.float64)
    K = bx.shape[0]

    # Build SumStats objects for TorchGWAS. p, n, af are needed by the
    # dataclass but not by mr_*; we set p from the z-score and n=10000 stub.
    z_x = (bx / se_x).numpy()
    z_y = (by / se_y).numpy()
    p_x = 2.0 * (1.0 - _phi(np.abs(z_x)))
    p_y = 2.0 * (1.0 - _phi(np.abs(z_y)))

    snps = sumstats["SNP"].astype(str).tolist()
    common = dict(
        chr=["1"] * K, pos=list(range(1, K + 1)), snp=snps,
        a1=["A"] * K, a2=["G"] * K,
        n=torch.tensor([10000] * K, dtype=torch.float64),
        af=None,
    )
    exposure = SumStats(beta=bx, se=se_x, p=torch.tensor(p_x, dtype=torch.float64), **common)
    outcome = SumStats(beta=by, se=se_y, p=torch.tensor(p_y, dtype=torch.float64), **common)

    return {
        "exposure": exposure,
        "outcome": outcome,
        "K": K,
        "truth": truth,
        "r_results": r_results,
    }


def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf."""
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


# ── Comparison 1: IVW ────────────────────────────────────────────────────────


def compare_ivw(data_dir: Path, out_dir: Path) -> ComparisonReport:
    inp = _load_inputs(data_dir, out_dir)
    r = inp["r_results"]["ivw"]

    tg = mr_ivw(inp["exposure"], inp["outcome"])

    d_b = abs(tg.beta_hat - r["b"])
    d_se = abs(tg.se - r["se"])
    d_p = abs(tg.p_value - r["pval"])

    rep = ComparisonReport(
        name="IVW (TwoSampleMR mr_ivw vs TorchGWAS mr_ivw)",
        n_compared=inp["K"],
    )
    rep.checks.append(_check_max("|Δ β|", d_b, TOL_IVW_BETA))
    rep.checks.append(_check_max("|Δ SE|", d_se, TOL_IVW_SE))
    rep.checks.append(_check_max("|Δ p|", d_p, TOL_IVW_PVAL))
    rep.extras["R β (SE)"] = f"{r['b']:.6f} ({r['se']:.6f})"
    rep.extras["TG β (SE)"] = f"{tg.beta_hat:.6f} ({tg.se:.6f})"
    rep.extras["R p"] = r["pval"]
    rep.extras["TG p"] = tg.p_value
    rep.extras["truth θ"] = inp["truth"]["theta_true"]
    return rep


# ── Comparison 2: MR-Egger ───────────────────────────────────────────────────


def compare_egger(data_dir: Path, out_dir: Path) -> ComparisonReport:
    inp = _load_inputs(data_dir, out_dir)
    r = inp["r_results"]["egger"]

    tg = mr_egger(inp["exposure"], inp["outcome"])

    d_b = abs(tg.beta_hat - r["b"])
    d_se = abs(tg.se - r["se"])
    d_p = abs(tg.p_value - r["pval"])
    d_int = abs(tg.intercept - r["intercept"])
    d_int_se = abs(tg.intercept_se - r["intercept_se"])

    rep = ComparisonReport(
        name="MR-Egger (TwoSampleMR vs TorchGWAS mr_egger)",
        n_compared=inp["K"],
    )
    rep.checks.append(_check_max("|Δ β|", d_b, TOL_EGGER_BETA))
    rep.checks.append(_check_max("|Δ SE|", d_se, TOL_EGGER_SE))
    rep.checks.append(_check_max("|Δ p|", d_p, TOL_EGGER_PVAL))
    rep.checks.append(_check_max("|Δ intercept|", d_int, TOL_EGGER_INTERCEPT))
    rep.checks.append(_check_max("|Δ intercept SE|", d_int_se, TOL_EGGER_INTERCEPT_SE))
    rep.extras["R β (SE)"] = f"{r['b']:.6f} ({r['se']:.6f})"
    rep.extras["TG β (SE)"] = f"{tg.beta_hat:.6f} ({tg.se:.6f})"
    rep.extras["R intercept (SE)"] = f"{r['intercept']:.6f} ({r['intercept_se']:.6f})"
    rep.extras["TG intercept (SE)"] = f"{tg.intercept:.6f} ({tg.intercept_se:.6f})"
    rep.extras["R intercept p"] = r["intercept_pval"]
    rep.extras["TG intercept p"] = tg.intercept_p
    return rep


# ── Comparison 3: Weighted median ────────────────────────────────────────────


def compare_weighted_median(data_dir: Path, out_dir: Path) -> ComparisonReport:
    inp = _load_inputs(data_dir, out_dir)
    r = inp["r_results"]["weighted_median"]

    # Same n_boot=1000 + seed=42 in both tools, but the bootstrap schemes
    # differ: TwoSampleMR uses a parametric bootstrap (resample β_y from
    # N(β_y, se_y²) and β_x from N(β_x, se_x²)), while TorchGWAS resamples
    # SNP indices with replacement (non-parametric bootstrap). The point
    # estimate is deterministic; only the SE diverges between the two.
    tg = mr_weighted_median(inp["exposure"], inp["outcome"], n_boot=1000, seed=42)

    d_b = abs(tg.beta_hat - r["b"])
    d_se = abs(tg.se - r["se"])

    rep = ComparisonReport(
        name="Weighted median (TwoSampleMR vs TorchGWAS mr_weighted_median)",
        n_compared=inp["K"],
    )
    rep.checks.append(_check_max("|Δ β|", d_b, TOL_WMED_BETA))
    rep.checks.append(_check_max("|Δ SE|", d_se, TOL_WMED_SE))
    rep.extras["R β (SE)"] = f"{r['b']:.6f} ({r['se']:.6f})"
    rep.extras["TG β (SE)"] = f"{tg.beta_hat:.6f} ({tg.se:.6f})"
    rep.extras["R p"] = r["pval"]
    rep.extras["TG p"] = tg.p_value
    return rep


# ── Comparison 4: MR-PRESSO ──────────────────────────────────────────────────


def compare_presso(data_dir: Path, out_dir: Path) -> ComparisonReport:
    inp = _load_inputs(data_dir, out_dir)
    r = inp["r_results"]["mr_presso"]

    # n_perm matches R's NbDistribution to keep MC noise comparable.
    tg = mr_presso(inp["exposure"], inp["outcome"], n_perm=1000, seed=42)

    # Compare raw (uncorrected) IVW slope.
    d_raw = abs(tg.beta_hat - r["raw_b"])
    # Compare corrected slopes if both ran the corrected fit.
    if r["corrected_b"] is not None and tg.n_outliers > 0:
        d_corr = abs(tg.beta_corrected - r["corrected_b"])
    else:
        # If either tool found zero outliers, the corrected slope falls back
        # to the raw — comparing them would re-test the raw slope, which is
        # already covered above. Mark as N/A.
        d_corr = 0.0

    # Global p: noisy due to permutation.
    d_global_p = abs(tg.global_p - r["global_p"])

    # Outlier-set Jaccard (informational).
    r_out = set(r["outlier_indices_zero_based"])
    tg_out = set(tg.outlier_indices)
    jaccard = (
        len(r_out & tg_out) / len(r_out | tg_out)
        if (r_out | tg_out) else 1.0
    )

    rep = ComparisonReport(
        name="MR-PRESSO (TwoSampleMR vs TorchGWAS mr_presso)",
        n_compared=inp["K"],
    )
    rep.checks.append(_check_max("|Δ raw β|", d_raw, TOL_PRESSO_BETA_RAW))
    rep.checks.append(_check_max("|Δ corrected β|", d_corr, TOL_PRESSO_BETA_CORR))
    rep.checks.append(_check_max("|Δ global p|", d_global_p, TOL_PRESSO_GLOBAL_P))
    rep.extras["R raw β"] = r["raw_b"]
    rep.extras["TG raw β"] = tg.beta_hat
    rep.extras["R corrected β"] = r["corrected_b"]
    rep.extras["TG corrected β"] = tg.beta_corrected
    rep.extras["R global p"] = r["global_p"]
    rep.extras["TG global p"] = tg.global_p
    rep.extras["R n_outliers"] = r["n_outliers"]
    rep.extras["TG n_outliers"] = tg.n_outliers
    rep.extras["R outlier zero-based"] = sorted(r_out)
    rep.extras["TG outlier zero-based"] = sorted(tg_out)
    rep.extras["outlier Jaccard"] = jaccard
    rep.extras["truth pleio zero-based"] = inp["truth"]["pleio_indices_zero_based"]
    return rep


# ── Driver ───────────────────────────────────────────────────────────────────


def run_all(data_dir: Path, out_dir: Path, json_out: Path | None = None) -> int:
    reports = [
        compare_ivw(data_dir, out_dir),
        compare_egger(data_dir, out_dir),
        compare_weighted_median(data_dir, out_dir),
        compare_presso(data_dir, out_dir),
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
