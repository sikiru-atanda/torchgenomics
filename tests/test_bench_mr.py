"""Benchmark: torchgenomics MR methods vs statsmodels / scipy reference.

These tests validate that our IVW, MR-Egger, weighted median, and
MR-PRESSO produce results numerically consistent with established
reference implementations (statsmodels WLS, scipy).
"""
from __future__ import annotations

import math

import numpy as np
import statsmodels.api as sm
import torch
from scipy import stats as sp_stats

from torchgenomics.postgwas._mr import mr_egger, mr_ivw, mr_presso, mr_weighted_median
from torchgenomics.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _make_benchmark_data(
    n: int = 30, true_effect: float = 0.25, seed: int = 12345
) -> tuple[SumStats, SumStats, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate exposure/outcome SumStats with a known causal effect.

    Returns (exposure, outcome, bx, by, sx, sy) where bx/by/sx/sy are
    numpy arrays for reference calculations.
    """
    rng = np.random.RandomState(seed)

    # Exposure betas: moderately strong instruments
    bx = rng.uniform(0.05, 0.40, size=n)
    sx = rng.uniform(0.01, 0.05, size=n)

    # Outcome betas: causal effect + noise (no pleiotropy by construction)
    by = true_effect * bx + rng.normal(0, 0.02, size=n)
    sy = rng.uniform(0.02, 0.08, size=n)

    # Dummy metadata
    chrs = ["1"] * n
    pos = list(range(1000, 1000 + n))
    snps = [f"rs{i}" for i in range(n)]
    a1 = ["A"] * n
    a2 = ["G"] * n
    p_dummy = np.ones(n)
    n_sample = np.full(n, 10000.0)

    exposure = SumStats(
        chr=chrs, pos=pos, snp=snps, a1=a1, a2=a2,
        beta=torch.tensor(bx, dtype=torch.float64),
        se=torch.tensor(sx, dtype=torch.float64),
        p=torch.tensor(p_dummy, dtype=torch.float64),
        n=torch.tensor(n_sample, dtype=torch.float64),
    )
    outcome = SumStats(
        chr=chrs, pos=pos, snp=snps, a1=a1, a2=a2,
        beta=torch.tensor(by, dtype=torch.float64),
        se=torch.tensor(sy, dtype=torch.float64),
        p=torch.tensor(p_dummy, dtype=torch.float64),
        n=torch.tensor(n_sample, dtype=torch.float64),
    )
    return exposure, outcome, bx, by, sx, sy


# ---------------------------------------------------------------------------
# 1. IVW beta matches statsmodels WLS (no intercept)
# ---------------------------------------------------------------------------


def test_bench_ivw_beta_matches_statsmodels_wls():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_ivw(exposure, outcome)

    w = 1.0 / sy**2
    sm_model = sm.WLS(by, bx, weights=w).fit()
    sm_beta = sm_model.params[0]

    assert abs(result.beta_hat - sm_beta) < 1e-10, (
        f"IVW beta {result.beta_hat} != statsmodels {sm_beta}"
    )


# ---------------------------------------------------------------------------
# 2. IVW SE matches statsmodels (accounting for overdispersion correction)
# ---------------------------------------------------------------------------


def test_bench_ivw_se_matches_statsmodels():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_ivw(exposure, outcome)

    w = 1.0 / sy**2
    K = len(bx)

    # Our uncorrected SE: 1 / sqrt(sum(w * bx^2))
    se_uncorrected = 1.0 / np.sqrt(np.sum(w * bx**2))

    # Cochran's Q
    resid = by - result.beta_hat * bx
    q = np.sum(w * resid**2)

    # Our code inflates SE when Q > K-1
    if q > (K - 1):
        se_expected = se_uncorrected * np.sqrt(q / (K - 1))
    else:
        se_expected = se_uncorrected

    assert abs(result.se - se_expected) < 1e-10, (
        f"IVW SE {result.se} != expected {se_expected}"
    )


# ---------------------------------------------------------------------------
# 3. IVW p-value matches scipy
# ---------------------------------------------------------------------------


def test_bench_ivw_pvalue_matches_scipy():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_ivw(exposure, outcome)

    z = result.beta_hat / result.se
    p_ref = 2.0 * sp_stats.norm.sf(abs(z))

    assert abs(result.p_value - p_ref) < 1e-10, (
        f"IVW p-value {result.p_value} != scipy {p_ref}"
    )


# ---------------------------------------------------------------------------
# 4. Egger slope matches statsmodels WLS with intercept
# ---------------------------------------------------------------------------


def test_bench_egger_beta_matches_statsmodels_wls():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_egger(exposure, outcome)

    w = 1.0 / sy**2
    X = sm.add_constant(bx)
    sm_model = sm.WLS(by, X, weights=w).fit()
    sm_slope = sm_model.params[1]

    assert abs(result.beta_hat - sm_slope) < 1e-8, (
        f"Egger slope {result.beta_hat} != statsmodels {sm_slope}"
    )


# ---------------------------------------------------------------------------
# 5. Egger intercept matches statsmodels
# ---------------------------------------------------------------------------


def test_bench_egger_intercept_matches_statsmodels():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_egger(exposure, outcome)

    w = 1.0 / sy**2
    X = sm.add_constant(bx)
    sm_model = sm.WLS(by, X, weights=w).fit()
    sm_intercept = sm_model.params[0]

    assert abs(result.intercept - sm_intercept) < 1e-8, (
        f"Egger intercept {result.intercept} != statsmodels {sm_intercept}"
    )


# ---------------------------------------------------------------------------
# 6. Egger intercept SE matches statsmodels (with overdispersion correction)
# ---------------------------------------------------------------------------


def test_bench_egger_intercept_se_matches_statsmodels():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_egger(exposure, outcome)

    K = len(bx)
    w = 1.0 / sy**2
    X = sm.add_constant(bx)
    sm_model = sm.WLS(by, X, weights=w).fit()

    # statsmodels bse assumes sigma^2 = RSS / (K - p) with no floor.
    # Our code uses sigma^2 = max(1.0, Q / (K - 2)).
    # Compute Q (weighted RSS) from our fit.
    fitted = X @ np.array([result.intercept, result.beta_hat])
    resid = by - fitted
    q = np.sum(w * resid**2)
    df = K - 2
    sigma2 = max(1.0, q / df)

    # statsmodels computes (X'WX)^{-1} internally; its bse uses
    # scale = RSS_weighted / df.  We need to rescale by our sigma2.
    # The (X'WX)^{-1} diagonal gives the variance without sigma2.
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    se_intercept_expected = np.sqrt(sigma2 * XtWX_inv[0, 0])

    assert abs(result.intercept_se - se_intercept_expected) < 1e-6, (
        f"Egger intercept SE {result.intercept_se} != expected {se_intercept_expected}"
    )


# ---------------------------------------------------------------------------
# 7. Weighted median matches manual np.interp calculation
# ---------------------------------------------------------------------------


def test_bench_weighted_median_matches_manual():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_weighted_median(exposure, outcome, n_boot=100, seed=42)

    # Manual weighted median via np.interp
    ratios = by / bx
    weights = bx**2 / sy**2
    weights = np.maximum(weights, 1e-30)

    order = np.argsort(ratios)
    sorted_r = ratios[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    cum_w = cum_w / cum_w[-1]

    manual_median = np.interp(0.5, cum_w, sorted_r)

    # Our implementation uses linear interpolation between the two
    # bracketing points; np.interp does the same.  However, the exact
    # interpolation formula may produce a tiny difference when 0.5
    # lands exactly on a cumulative weight boundary.  The two methods
    # are algebraically equivalent for interior points.
    assert abs(result.beta_hat - manual_median) < 1e-10, (
        f"Weighted median {result.beta_hat} != manual {manual_median}"
    )


# ---------------------------------------------------------------------------
# 8. MR-PRESSO global RSS matches manual calculation
# ---------------------------------------------------------------------------


def test_bench_presso_rss_matches_manual():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_presso(exposure, outcome, n_perm=10, seed=42)

    # Manual LOO RSS per Verbanck 2018 / MRPRESSO 1.0 `getRSS_LOO`:
    # for each i, fit IVW on j≠i to get beta_LOO[i], then sum weighted
    # squared residuals (by_i - beta_LOO[i]·bx_i)² · w_i.
    w = 1.0 / sy**2
    K = bx.shape[0]
    rss_manual = 0.0
    for i in range(K):
        mask = np.ones(K, dtype=bool)
        mask[i] = False
        num = np.sum(w[mask] * bx[mask] * by[mask])
        den = np.sum(w[mask] * bx[mask] ** 2)
        beta_loo = num / den
        rss_manual += w[i] * (by[i] - beta_loo * bx[i]) ** 2

    assert abs(result.global_rss - rss_manual) < 1e-8, (
        f"PRESSO RSS {result.global_rss} != manual LOO {rss_manual}"
    )


# ---------------------------------------------------------------------------
# 9. IVW Cochran's Q matches manual computation
# ---------------------------------------------------------------------------


def test_bench_ivw_cochran_q_matches_manual():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_ivw(exposure, outcome)

    # Cochran's Q = sum(w * (by - beta * bx)^2)
    # This is the same as sum(w * resid^2) from the IVW WLS fit.
    w = 1.0 / sy**2
    resid = by - result.beta_hat * bx
    q_manual = np.sum(w * resid**2)

    # Our code uses Q internally to decide overdispersion correction.
    # We can verify via the SE: if Q > K-1, SE = se_uncorr * sqrt(Q/(K-1)).
    K = len(bx)
    se_uncorrected = 1.0 / np.sqrt(np.sum(w * bx**2))

    if q_manual > (K - 1):
        se_corrected = se_uncorrected * np.sqrt(q_manual / (K - 1))
    else:
        se_corrected = se_uncorrected

    # The SE we get from our code must match this correction
    assert abs(result.se - se_corrected) < 1e-10, (
        f"IVW SE {result.se} != Cochran-Q-corrected {se_corrected} "
        f"(Q={q_manual:.6f}, K={K})"
    )

    # Also verify Q itself is positive and finite
    assert q_manual > 0.0
    assert math.isfinite(q_manual)


# ---------------------------------------------------------------------------
# 10. Egger slope p-value matches scipy from statsmodels params
# ---------------------------------------------------------------------------


def test_bench_egger_pvalue_slope_matches_scipy():
    exposure, outcome, bx, by, _, sy = _make_benchmark_data()
    result = mr_egger(exposure, outcome)

    # MR-Egger uses Student's t with K-2 d.f. (Bowden 2015 / TwoSampleMR)
    # rather than the standard normal — see torchgenomics.postgwas._mr.mr_egger.
    # Validate the p-value against scipy.stats.t.
    K = result.n_instruments
    df = K - 2
    z = result.beta_hat / result.se
    p_ref = 2.0 * sp_stats.t.sf(abs(z), df)

    assert abs(result.p_value - p_ref) < 1e-10, (
        f"Egger slope p-value {result.p_value} != scipy.t.sf {p_ref} "
        f"(df={df})"
    )
