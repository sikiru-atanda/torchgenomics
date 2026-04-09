"""Mendelian Randomization (MR) methods.

Four standard estimators for two-sample MR causal inference:

* **IVW** (inverse-variance weighted) --- default, assumes all instruments
  are valid (no horizontal pleiotropy). The Wald ratio per SNP is
  beta_outcome / beta_exposure; the IVW estimate is the precision-weighted
  mean of these ratios (equivalently, the slope from WLS of beta_Y on
  beta_X with weights 1/se_Y^2, no intercept).

* **MR-Egger** --- allows directional pleiotropy by fitting an intercept
  (Bowden et al. 2015). The intercept tests the InSIDE assumption; a
  non-zero intercept signals average pleiotropy. Uses the same WLS
  formulation but *with* intercept. Egger I-squared statistic quantifies
  instrument strength after accounting for measurement error.

* **Weighted median** --- consistent when >= 50% of the weight comes from
  valid instruments (Bowden et al. 2016). Computes per-SNP Wald ratios,
  orders them, and returns the value at the 50th percentile of the
  cumulative weight distribution.

* **MR-PRESSO** (Mendelian Randomization Pleiotropy RESidual Sum and
  Outlier, Verbanck et al. 2018) --- detects and corrects for horizontal
  pleiotropy outliers via a leave-one-out residual approach. Global test
  for pleiotropy + per-SNP outlier test + corrected IVW after outlier
  removal.

All methods accept aligned exposure and outcome ``SumStats`` objects. The
caller is responsible for instrument selection (e.g. genome-wide
significant SNPs in the exposure GWAS) and allele harmonization (the
``align_sumstats`` helper in this package handles that).

**Polyploid compatibility.** All four estimators operate purely on effect
sizes and standard errors from GWAS summary statistics. The Wald ratio
``beta_Y / beta_X`` and the WLS regression are ploidy-agnostic — the
upstream GWAS scan already produces valid z-statistics for any ploidy,
and the causal estimate is invariant to the per-allele scaling imposed
by higher ploidy levels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from ._sumstats import SumStats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MRResult:
    """Result from a Mendelian Randomization analysis."""

    method: str  # "ivw", "egger", "weighted_median", "mr_presso"
    beta_hat: float  # causal estimate
    se: float  # standard error
    p_value: float  # two-sided p-value
    n_instruments: int  # number of IVs used

    # Egger-specific
    intercept: float = 0.0
    intercept_se: float = 0.0
    intercept_p: float = 1.0
    egger_i2: float = 0.0  # instrument strength

    # MR-PRESSO-specific
    global_rss: float = 0.0
    global_p: float = 1.0
    outlier_indices: list[int] = field(default_factory=list)
    n_outliers: int = 0
    beta_corrected: float = 0.0
    se_corrected: float = 0.0
    p_corrected: float = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_sided_p(z: Tensor) -> Tensor:
    """Two-sided p-value from z-score using torch.erfc (no scipy needed)."""
    # p = 2 * Phi(-|z|) = erfc(|z| / sqrt(2))
    return torch.erfc(z.abs() / (2.0**0.5))


def _two_sided_p_scalar(z: float) -> float:
    """Two-sided p-value from a scalar z-score."""
    return _two_sided_p(torch.tensor(z, dtype=torch.float64)).item()


def _validate_inputs(exposure: SumStats, outcome: SumStats) -> None:
    """Check that exposure and outcome are compatible for MR."""
    if exposure.m != outcome.m:
        raise ValueError(
            f"exposure and outcome must have the same number of SNPs, "
            f"got {exposure.m} vs {outcome.m}. "
            f"Use align_sumstats() to harmonize first."
        )
    if exposure.m < 3:
        raise ValueError(
            f"At least 3 instruments are required for MR, got {exposure.m}."
        )
    # Warn about weak instruments (near-zero exposure betas)
    bx = exposure.beta.to(torch.float64)
    weak = (bx.abs() < 1e-10).sum().item()
    if weak > 0:
        warnings.warn(
            f"{int(weak)} instrument(s) have |beta_exposure| < 1e-10. "
            f"These weak instruments can produce unstable Wald ratios.",
            stacklevel=3,
        )


def _ivw_fit(
    bx: Tensor, by: Tensor, se_y: Tensor
) -> tuple[float, float, float, float]:
    """Core IVW WLS fit (no intercept).

    Returns (beta_hat, se, cochran_q, p_value).
    """
    w = 1.0 / (se_y**2)  # precision weights

    # beta_hat = sum(w * bx * by) / sum(w * bx^2)
    numerator = (w * bx * by).sum()
    denominator = (w * bx**2).sum()
    beta_hat = (numerator / denominator).item()

    # Second-order SE (Burgess et al. 2013)
    se_hat = (1.0 / denominator).sqrt().item()

    # Cochran's Q for heterogeneity
    resid = by - beta_hat * bx
    q = (w * resid**2).sum().item()

    # Overdispersion correction: inflate SE if Q > K-1
    K = bx.shape[0]
    if K > 1 and q > (K - 1):
        se_hat *= (q / (K - 1)) ** 0.5

    p_value = _two_sided_p_scalar(beta_hat / se_hat) if se_hat > 0 else 1.0
    return beta_hat, se_hat, q, p_value


# ---------------------------------------------------------------------------
# IVW
# ---------------------------------------------------------------------------


def mr_ivw(exposure: SumStats, outcome: SumStats) -> MRResult:
    """Inverse-variance weighted MR (fixed-effect, no intercept).

    The IVW estimator is the precision-weighted mean of per-SNP Wald
    ratios (beta_Y / beta_X), equivalent to WLS regression of beta_Y on
    beta_X with weights 1/se_Y^2 and no intercept.

    Overdispersion correction (multiplicative random effects) is applied
    when Cochran's Q exceeds the degrees of freedom.

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics (instruments).
    outcome : SumStats
        Outcome GWAS summary statistics at the same SNPs.

    Returns
    -------
    MRResult
        Causal estimate, SE, and two-sided p-value.
    """
    _validate_inputs(exposure, outcome)

    bx = exposure.beta.to(torch.float64)
    by = outcome.beta.to(torch.float64)
    se_y = outcome.se.to(torch.float64)

    beta_hat, se_hat, q, p_value = _ivw_fit(bx, by, se_y)

    return MRResult(
        method="ivw",
        beta_hat=beta_hat,
        se=se_hat,
        p_value=p_value,
        n_instruments=exposure.m,
    )


# ---------------------------------------------------------------------------
# MR-Egger
# ---------------------------------------------------------------------------


def mr_egger(exposure: SumStats, outcome: SumStats) -> MRResult:
    """MR-Egger regression (allows directional pleiotropy).

    Fits the WLS model ``by = alpha + beta * bx + eps`` with weights
    ``1/se_y^2``.  A non-zero intercept ``alpha`` indicates average
    directional pleiotropy, violating the standard IVW assumption.

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics (instruments).
    outcome : SumStats
        Outcome GWAS summary statistics at the same SNPs.

    Returns
    -------
    MRResult
        Includes slope, intercept, intercept SE/p-value, and Egger I-squared.
    """
    _validate_inputs(exposure, outcome)

    bx = exposure.beta.to(torch.float64)
    by = outcome.beta.to(torch.float64)
    se_x = exposure.se.to(torch.float64)
    se_y = outcome.se.to(torch.float64)
    K = bx.shape[0]

    w = 1.0 / (se_y**2)  # (K,)

    # Design matrix X = [1, bx]  (K x 2)
    ones = torch.ones_like(bx)
    X = torch.stack([ones, bx], dim=1)  # (K, 2)

    # WLS: (X'WX)^{-1} X'Wy
    W = torch.diag(w)  # (K, K)
    XtWX = X.T @ W @ X  # (2, 2)
    XtWy = X.T @ (w * by)  # (2,)
    coef = torch.linalg.solve(XtWX, XtWy)  # (2,)
    alpha_hat = coef[0].item()  # intercept
    beta_hat = coef[1].item()  # slope (causal estimate)

    # Residuals and sigma^2 estimate
    fitted = X @ coef  # (K,)
    resid = by - fitted
    q = (w * resid**2).sum().item()  # Cochran's Q

    # Degrees of freedom for Egger = K - 2
    df = K - 2
    sigma2 = max(1.0, q / df)  # overdispersion factor

    # Covariance of coefficients: sigma^2 * (X'WX)^{-1}
    XtWX_inv = torch.linalg.inv(XtWX)
    cov = sigma2 * XtWX_inv

    se_alpha = cov[0, 0].sqrt().item()
    se_beta = cov[1, 1].sqrt().item()

    p_beta = _two_sided_p_scalar(beta_hat / se_beta) if se_beta > 0 else 1.0
    p_alpha = _two_sided_p_scalar(alpha_hat / se_alpha) if se_alpha > 0 else 1.0

    # Egger I-squared: instrument strength after measurement error.
    # I^2_GX = 1 - sum(se_x^2) / sum((bx - mean(bx))^2)
    # A low I^2 (< 0.9) warns of weak instrument bias in Egger.
    bx_mean = bx.mean()
    bx_var_sum = ((bx - bx_mean) ** 2).sum()
    se_x_sq_sum = (se_x**2).sum()
    if bx_var_sum.item() > 0:
        egger_i2 = max(0.0, 1.0 - (se_x_sq_sum / bx_var_sum).item())
    else:
        egger_i2 = 0.0

    return MRResult(
        method="egger",
        beta_hat=beta_hat,
        se=se_beta,
        p_value=p_beta,
        n_instruments=K,
        intercept=alpha_hat,
        intercept_se=se_alpha,
        intercept_p=p_alpha,
        egger_i2=egger_i2,
    )


# ---------------------------------------------------------------------------
# Weighted median
# ---------------------------------------------------------------------------


def _weighted_median(ratios: Tensor, weights: Tensor) -> float:
    """Weighted median: value at 50th percentile of cumulative weight.

    Parameters
    ----------
    ratios : (K,) Wald ratios.
    weights : (K,) positive weights (need not sum to 1).

    Returns
    -------
    float
        The weighted median.
    """
    # Sort by ratio
    order = ratios.argsort()
    sorted_r = ratios[order]
    sorted_w = weights[order]

    # Normalise cumulative weights to [0, 1]
    cum_w = sorted_w.cumsum(dim=0)
    cum_w = cum_w / cum_w[-1]

    # First index where cumulative weight >= 0.5
    idx = (cum_w >= 0.5).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return sorted_r[-1].item()

    j = idx[0, 0].item()
    if j == 0:
        return sorted_r[0].item()

    # Linear interpolation between j-1 and j
    w_lo = cum_w[j - 1].item()
    w_hi = cum_w[j].item()
    r_lo = sorted_r[j - 1].item()
    r_hi = sorted_r[j].item()

    if w_hi - w_lo < 1e-30:
        return r_hi

    frac = (0.5 - w_lo) / (w_hi - w_lo)
    return r_lo + frac * (r_hi - r_lo)


def mr_weighted_median(
    exposure: SumStats,
    outcome: SumStats,
    n_boot: int = 1000,
    seed: int = 42,
) -> MRResult:
    """Weighted median MR estimator.

    Consistent when at least 50% of the weight comes from valid
    instruments (Bowden et al. 2016).

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics.
    outcome : SumStats
        Outcome GWAS summary statistics.
    n_boot : int
        Number of bootstrap resamples for SE estimation.
    seed : int
        Random seed for bootstrap reproducibility.

    Returns
    -------
    MRResult
        Causal estimate, bootstrap SE, and p-value.
    """
    _validate_inputs(exposure, outcome)

    bx = exposure.beta.to(torch.float64)
    by = outcome.beta.to(torch.float64)
    se_y = outcome.se.to(torch.float64)
    K = bx.shape[0]

    # Wald ratios and inverse-variance weights of the Wald ratio
    ratios = by / bx  # (K,)
    # Var(Wald) ~ se_y^2 / bx^2  =>  w = bx^2 / se_y^2
    weights = (bx**2) / (se_y**2)
    weights = weights.clamp(min=1e-30)

    beta_hat = _weighted_median(ratios, weights)

    # Bootstrap SE
    gen = torch.Generator(device=bx.device)
    gen.manual_seed(seed)

    boot_estimates = torch.empty(n_boot, dtype=torch.float64, device=bx.device)
    for b in range(n_boot):
        idx = torch.randint(0, K, (K,), generator=gen, device=bx.device)
        boot_estimates[b] = _weighted_median(ratios[idx], weights[idx])

    se_hat = boot_estimates.std().item()
    p_value = _two_sided_p_scalar(beta_hat / se_hat) if se_hat > 0 else 1.0

    return MRResult(
        method="weighted_median",
        beta_hat=beta_hat,
        se=se_hat,
        p_value=p_value,
        n_instruments=K,
    )


# ---------------------------------------------------------------------------
# MR-PRESSO
# ---------------------------------------------------------------------------


def mr_presso(
    exposure: SumStats,
    outcome: SumStats,
    n_perm: int = 1000,
    outlier_threshold: float = 0.05,
    seed: int = 42,
) -> MRResult:
    """MR-PRESSO: pleiotropy residual sum and outlier detection.

    Three steps (Verbanck et al. 2018):

    1. **Global test** -- compute observed RSS from IVW residuals, then
       permute outcome betas ``n_perm`` times to build a null
       distribution; ``global_p`` is the fraction of permuted RSS >=
       observed.

    2. **Outlier detection** -- leave-one-out IVW residuals compared
       against the same LOO permutation null; SNPs with p < ``outlier_threshold``
       are flagged.

    3. **Corrected IVW** -- re-fit IVW excluding detected outliers.

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics.
    outcome : SumStats
        Outcome GWAS summary statistics.
    n_perm : int
        Number of permutations for the global and outlier tests.
    outlier_threshold : float
        Significance threshold for per-SNP outlier detection.
    seed : int
        Random seed for permutation reproducibility.

    Returns
    -------
    MRResult
        Includes global test p-value, outlier indices, and corrected
        causal estimate after outlier removal.
    """
    _validate_inputs(exposure, outcome)

    bx = exposure.beta.to(torch.float64)
    by = outcome.beta.to(torch.float64)
    se_y = outcome.se.to(torch.float64)
    K = bx.shape[0]

    w = 1.0 / (se_y**2)

    # -- Observed IVW fit and RSS --
    beta_obs, _, _, _ = _ivw_fit(bx, by, se_y)
    resid_obs = by - beta_obs * bx
    rss_obs = (w * resid_obs**2).sum().item()

    # -- Step 1: Global test via permutation --
    gen = torch.Generator(device=bx.device)
    gen.manual_seed(seed)

    rss_perm = torch.empty(n_perm, dtype=torch.float64, device=bx.device)
    for t in range(n_perm):
        perm_idx = torch.randperm(K, generator=gen, device=bx.device)
        by_perm = by[perm_idx]
        beta_p, _, _, _ = _ivw_fit(bx, by_perm, se_y)
        resid_p = by_perm - beta_p * bx
        rss_perm[t] = (w * resid_p**2).sum()

    global_p = ((rss_perm >= rss_obs).sum().item() + 1) / (n_perm + 1)

    # -- Step 2: Leave-one-out outlier detection --
    # For each SNP j, fit IVW without j and compute its squared residual.
    loo_resid_obs = torch.empty(K, dtype=torch.float64, device=bx.device)
    for j in range(K):
        mask = torch.ones(K, dtype=torch.bool, device=bx.device)
        mask[j] = False
        bx_loo = bx[mask]
        by_loo = by[mask]
        se_y_loo = se_y[mask]
        w_loo = 1.0 / (se_y_loo**2)

        num = (w_loo * bx_loo * by_loo).sum()
        den = (w_loo * bx_loo**2).sum()
        beta_loo = num / den

        # Predicted value for SNP j and its weighted squared residual
        pred_j = beta_loo * bx[j]
        loo_resid_obs[j] = w[j] * (by[j] - pred_j) ** 2

    # Permutation null for LOO residuals
    loo_resid_perm = torch.empty(
        n_perm, K, dtype=torch.float64, device=bx.device
    )

    gen2 = torch.Generator(device=bx.device)
    gen2.manual_seed(seed + 1)  # different seed from global test

    for t in range(n_perm):
        perm_idx = torch.randperm(K, generator=gen2, device=bx.device)
        by_perm = by[perm_idx]
        for j in range(K):
            mask = torch.ones(K, dtype=torch.bool, device=bx.device)
            mask[j] = False
            bx_loo = bx[mask]
            by_perm_loo = by_perm[mask]
            se_y_loo = se_y[mask]
            w_loo = 1.0 / (se_y_loo**2)

            num = (w_loo * bx_loo * by_perm_loo).sum()
            den = (w_loo * bx_loo**2).sum()
            beta_loo = num / den

            pred_j = beta_loo * bx[j]
            loo_resid_perm[t, j] = w[j] * (by_perm[j] - pred_j) ** 2

    # Per-SNP outlier p-values
    outlier_indices: list[int] = []
    for j in range(K):
        perm_vals = loo_resid_perm[:, j]
        p_j = ((perm_vals >= loo_resid_obs[j]).sum().item() + 1) / (n_perm + 1)
        if p_j < outlier_threshold:
            outlier_indices.append(j)

    n_outliers = len(outlier_indices)

    # -- Step 3: Corrected IVW after removing outliers --
    if n_outliers > 0 and (K - n_outliers) >= 3:
        keep = torch.ones(K, dtype=torch.bool, device=bx.device)
        for j in outlier_indices:
            keep[j] = False
        beta_corr, se_corr, _, p_corr = _ivw_fit(
            bx[keep], by[keep], se_y[keep]
        )
    else:
        beta_corr = beta_obs
        se_corr = 0.0
        p_corr = 1.0
        # Recompute SE for the uncorrected case
        _, se_corr, _, p_corr = _ivw_fit(bx, by, se_y)

    # Original IVW for the top-level fields (uncorrected)
    beta_hat, se_hat, _, p_value = _ivw_fit(bx, by, se_y)

    return MRResult(
        method="mr_presso",
        beta_hat=beta_hat,
        se=se_hat,
        p_value=p_value,
        n_instruments=K,
        global_rss=rss_obs,
        global_p=global_p,
        outlier_indices=outlier_indices,
        n_outliers=n_outliers,
        beta_corrected=beta_corr,
        se_corrected=se_corr,
        p_corrected=p_corr,
    )


# ---------------------------------------------------------------------------
# Convenience: run all methods
# ---------------------------------------------------------------------------


def mr_all(
    exposure: SumStats,
    outcome: SumStats,
    n_boot: int = 1000,
    n_perm: int = 1000,
    seed: int = 42,
) -> list[MRResult]:
    """Run all four MR methods and return results as a list.

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics (instruments).
    outcome : SumStats
        Outcome GWAS summary statistics at the same SNPs.
    n_boot : int
        Bootstrap resamples for weighted median SE.
    n_perm : int
        Permutations for MR-PRESSO global and outlier tests.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[MRResult]
        Results from IVW, Egger, weighted median, and MR-PRESSO
        (in that order).
    """
    return [
        mr_ivw(exposure, outcome),
        mr_egger(exposure, outcome),
        mr_weighted_median(exposure, outcome, n_boot=n_boot, seed=seed),
        mr_presso(
            exposure, outcome, n_perm=n_perm, outlier_threshold=0.05, seed=seed
        ),
    ]
