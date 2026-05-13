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

    # ── Bowden-2015 orientation step ─────────────────────────────────────────
    # MR-Egger requires that all exposure betas are positive so the intercept
    # `alpha` measures average directional pleiotropy with a single, well-
    # defined sign convention. Reference: Bowden, Davey Smith & Burgess
    # (2015) "Mendelian randomization with invalid instruments..." §3.
    # TwoSampleMR's `mr_egger_regression` (MRC-IEU) does the same flip.
    # We replicate the formulation exactly:
    #     by_j  ← by_j * sign(bx_j)
    #     bx_j  ← |bx_j|
    # (zero exposure betas are treated as positive — same as TwoSampleMR's
    # `sign0` helper).
    sgn = torch.where(bx == 0, torch.ones_like(bx), torch.sign(bx))
    bx_oriented = bx * sgn  # = abs(bx)
    by_oriented = by * sgn

    w = 1.0 / (se_y**2)  # (K,)

    # Design matrix X = [1, bx_oriented]  (K x 2)
    ones = torch.ones_like(bx_oriented)
    X = torch.stack([ones, bx_oriented], dim=1)  # (K, 2)

    # WLS: (X'WX)^{-1} X'Wy
    W = torch.diag(w)  # (K, K)
    XtWX = X.T @ W @ X  # (2, 2)
    XtWy = X.T @ (w * by_oriented)  # (2,)
    coef = torch.linalg.solve(XtWX, XtWy)  # (2,)
    alpha_hat = coef[0].item()  # intercept (directional pleiotropy)
    beta_hat = coef[1].item()  # slope (causal estimate)

    # Residuals and sigma^2 estimate
    fitted = X @ coef  # (K,)
    resid = by_oriented - fitted

    # ── Bowden-2015 SE convention (matches TwoSampleMR exactly) ──────────────
    # `lm()` in R reports SEs scaled by sigma_hat = sqrt(RSS / (K - 2)). The
    # MR-Egger convention only deflates the SE when sigma_hat < 1 (under-
    # dispersion); when sigma_hat > 1 (overdispersion), SE is left at the
    # `lm()` default. TwoSampleMR codes this as `coef[, 2] / min(1, sigma)`,
    # which is equivalent to `coef[, 2] * max(1, 1 / sigma)`.
    # We compute sigma_hat = sqrt(RSS / (K - 2)) where RSS is the *unweighted*
    # weighted residual sum of squares — i.e. the residuals at the WLS fit
    # divided by their se_y so that under no overdispersion they are i.i.d.
    # standard normal in expectation. This is what `lm(weights=1/se_y^2)`
    # produces internally.
    df = max(K - 2, 1)
    rss_w = (w * resid**2).sum().item()  # weighted SSR (Cochran's Q)
    sigma_hat = (rss_w / df) ** 0.5  # = `summary(lm(...))$sigma`

    # Default SE from WLS is sigma^2 * (X'WX)^{-1}; lm() reports the diagonal
    # square-roots. We compute the same and then apply the divide-by-min(1,σ)
    # rule used by Bowden-2015 / TwoSampleMR so we agree across both paths.
    XtWX_inv = torch.linalg.inv(XtWX)
    cov_default = (sigma_hat**2) * XtWX_inv

    se_alpha_default = cov_default[0, 0].sqrt().item()
    se_beta_default = cov_default[1, 1].sqrt().item()

    sigma_clip = min(1.0, sigma_hat) if sigma_hat > 0 else 1.0
    se_alpha = se_alpha_default / sigma_clip
    se_beta = se_beta_default / sigma_clip

    # ── p-values via t-distribution with df = K - 2 ──────────────────────────
    # Bowden-2015 / TwoSampleMR use Student's t with K-2 d.f. (textbook
    # small-sample WLS inference). The previous TorchGWAS implementation
    # used the standard normal, which is the K → ∞ limit and inflates the
    # p-value for typical MR scans (K = 20–100). We switch to t for
    # alignment.
    from scipy.stats import t as _student_t  # lazy import; scipy is a
                                              # required runtime dep already.
    p_beta = (
        2.0 * _student_t.sf(abs(beta_hat / se_beta), df) if se_beta > 0 else 1.0
    )
    p_alpha = (
        2.0 * _student_t.sf(abs(alpha_hat / se_alpha), df)
        if se_alpha > 0 else 1.0
    )

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


def _ivw_loo_betas(bx: Tensor, by: Tensor, w: Tensor) -> Tensor:
    """Vectorized leave-one-out WLS slope (no intercept).

    For each i, returns ``beta_LOO[i] = sum_{j≠i} w_j bx_j by_j /
    sum_{j≠i} w_j bx_j^2``. Used by both the observed RSS_LOO and
    every parametric-bootstrap replicate.
    """
    num_total = (w * bx * by).sum()
    den_total = (w * bx**2).sum()
    num_loo = num_total - w * bx * by
    den_loo = den_total - w * bx**2
    return num_loo / den_loo.clamp(min=1e-30)


def _rss_loo(bx: Tensor, by: Tensor, w: Tensor) -> tuple[float, Tensor]:
    """LOO weighted RSS (Verbanck 2018 / MRPRESSO `getRSS_LOO`).

    Returns ``(RSS, beta_loo)`` where ``RSS = sum_i w_i * (by_i -
    beta_LOO[i] * bx_i)^2`` and ``beta_loo[i]`` is the i-th leave-one-
    out IVW slope.
    """
    beta_loo = _ivw_loo_betas(bx, by, w)
    resid = by - beta_loo * bx
    rss = (w * resid**2).sum().item()
    return rss, beta_loo


def mr_presso(
    exposure: SumStats,
    outcome: SumStats,
    n_perm: int = 1000,
    outlier_threshold: float = 0.05,
    seed: int = 42,
    null: str = "parametric",
) -> MRResult:
    """MR-PRESSO: pleiotropy residual sum and outlier detection.

    Implements the Verbanck-2018 / MRPRESSO-1.0 algorithm with two
    selectable null distributions:

    * ``null="parametric"`` (default; matches TwoSampleMR / MRPRESSO).
      For each SNP i, draw ``bx_boot[i] ~ N(bx[i], se_x[i]²)`` and
      ``by_boot[i] ~ N(β_LOO_obs[i] · bx[i], se_y[i]²)`` per replicate;
      recompute LOO RSS on the simulated dataset; compare against the
      observed LOO RSS.

    * ``null="permutation"`` (legacy TorchGWAS behavior). Permute
      outcome betas against fixed exposure betas and recompute IVW RSS.
      Faster but yields a different reference null distribution and is
      not directly comparable to TwoSampleMR / MRPRESSO p-values.

    Three steps (Verbanck et al. 2018, Eq. 2):

    1. **Global test** — compute observed LOO RSS, then resample
       per the chosen null and re-compute LOO RSS for each replicate;
       ``global_p`` is the fraction of resampled RSS strictly greater
       than the observed RSS (matches MRPRESSO 1.0 ``getRSS_LOO``).

    2. **Outlier detection** — per-SNP unweighted residual ``Dif_i =
       by_i - bx_i·β_LOO_obs[i]`` compared against simulated residuals
       ``Exp_t,i = by_boot_t,i - bx_boot_t,i·β_LOO_obs[i]``. Bonferroni-
       corrected p-value ``min(p_i·K, 1)``. SNPs with corrected
       ``p < outlier_threshold`` are flagged.

    3. **Corrected IVW** — re-fit IVW excluding detected outliers.

    Parameters
    ----------
    exposure : SumStats
        Exposure GWAS summary statistics.
    outcome : SumStats
        Outcome GWAS summary statistics.
    n_perm : int
        Number of bootstrap / permutation replicates for the global and
        outlier tests. MRPRESSO requires ``n_perm > K``.
    outlier_threshold : float
        Significance threshold for per-SNP outlier detection
        (Bonferroni-corrected).
    seed : int
        RNG seed for replicate reproducibility.
    null : {"parametric", "permutation"}
        Resampling scheme for the null. ``"parametric"`` matches
        TwoSampleMR / MRPRESSO 1.0; ``"permutation"`` preserves the
        original TorchGWAS behavior. Default is ``"parametric"``.

    Returns
    -------
    MRResult
        Includes global test p-value (Verbanck-style empirical), outlier
        indices (zero-based, Bonferroni-corrected), and corrected causal
        estimate after outlier removal.
    """
    _validate_inputs(exposure, outcome)
    if null not in ("parametric", "permutation"):
        raise ValueError(
            f"`null` must be 'parametric' or 'permutation', got {null!r}."
        )

    bx_raw = exposure.beta.to(torch.float64)
    by_raw = outcome.beta.to(torch.float64)
    se_x = exposure.se.to(torch.float64)
    se_y = outcome.se.to(torch.float64)
    K = bx_raw.shape[0]

    # ── MRPRESSO orientation: flip by sign(bx[0]) so bx[0] >= 0 ─────────
    # This matches the R source line:
    #     data[, c(BetaY, BetaX)] <- data[, c(BetaY, BetaX)] *
    #                                sign(data[, BetaExposure[1]])
    # For a single exposure this is just multiplication of (bx, by) by a
    # global sign, which preserves the IVW slope and RSS. Kept verbatim
    # for behavioral parity with TwoSampleMR.
    sgn = 1.0 if bx_raw[0].item() >= 0 else -1.0
    bx = bx_raw * sgn
    by = by_raw * sgn

    w = 1.0 / (se_y**2)

    # ── Observed LOO RSS + per-SNP β_LOO ────────────────────────────────
    rss_obs, beta_loo_obs = _rss_loo(bx, by, w)

    # Choose dispatch for replicate sampling.
    if null == "permutation":
        rss_perm, by_boot_all, bx_boot_all = _presso_null_permutation(
            bx, by, w, n_perm=n_perm, seed=seed,
        )
    else:
        rss_perm, by_boot_all, bx_boot_all = _presso_null_parametric(
            bx, by, se_x, se_y, w, beta_loo_obs,
            n_perm=n_perm, seed=seed,
        )

    # ── Step 1: Global test ─────────────────────────────────────────────
    # MRPRESSO 1.0: `Pvalue = sum(RSSexp > RSSobs) / NbDistribution`.
    # Strict ">" (not ">=") and no +1 / +1 add-on — we replicate exactly.
    global_p = (rss_perm > rss_obs).sum().item() / float(n_perm)

    # ── Step 2: Outlier test (only if global p < SignifThreshold) ───────
    # When the global test is non-significant the R code simply skips the
    # outlier loop and reports OUTLIERtest = FALSE (i.e. no outliers).
    outlier_indices: list[int] = []
    p_outlier = torch.ones(K, dtype=torch.float64, device=bx.device)

    run_outlier = global_p < outlier_threshold
    if run_outlier:
        # Observed unweighted residual at each SNP using its β_LOO.
        dif = by - bx * beta_loo_obs  # (K,)

        # Simulated residuals at each SNP using the **observed** β_LOO.
        # Per R source:
        #     Exp = randomSNP[, BetaY] - randomSNP[, BetaX] * RSSobs[[2]][SNV]
        # i.e. uses the observed LOO beta, not a re-computed one.
        # Shape: (n_perm, K) for each.
        exp_resid = by_boot_all - bx_boot_all * beta_loo_obs.unsqueeze(0)

        # Per-SNP p = mean(Exp^2 > Dif^2) over replicates; then
        # Bonferroni correction: min(p * K, 1).
        per_snp = (exp_resid**2 > (dif**2).unsqueeze(0)).to(torch.float64)
        p_raw = per_snp.mean(dim=0)
        p_outlier = (p_raw * K).clamp(max=1.0)

        outlier_indices = [
            int(j) for j in (p_outlier < outlier_threshold).nonzero(
                as_tuple=False
            ).squeeze(-1).tolist()
        ]

    n_outliers = len(outlier_indices)

    # ── Step 3: Corrected IVW after removing outliers ───────────────────
    if n_outliers > 0 and (K - n_outliers) >= 3:
        keep = torch.ones(K, dtype=torch.bool, device=bx.device)
        for j in outlier_indices:
            keep[j] = False
        beta_corr, se_corr, _, p_corr = _ivw_fit(
            bx[keep], by[keep], se_y[keep]
        )
    else:
        # No outliers identified → corrected estimate falls back to raw.
        beta_corr, se_corr, _, p_corr = _ivw_fit(bx, by, se_y)

    # Top-level fields are the uncorrected (raw) IVW estimate. Note: we
    # use the orientation-adjusted (bx, by) here so the slope is always
    # reported in the original (raw) sign convention since multiplying
    # both by the same global sign cancels.
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


def _presso_null_parametric(
    bx: Tensor,
    by: Tensor,
    se_x: Tensor,
    se_y: Tensor,
    w: Tensor,
    beta_loo_obs: Tensor,
    *,
    n_perm: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Verbanck-2018 / MRPRESSO 1.0 parametric LOO bootstrap null.

    For each replicate t and SNP i:
        bx_boot[t, i] ~ N(bx[i], se_x[i]^2)
        by_boot[t, i] ~ N(beta_loo_obs[i] * bx[i], se_y[i]^2)
    then re-compute the LOO RSS on the simulated dataset using the
    **same** weights w = 1/se_y^2 (taken from the original outcome SE,
    not a re-computed one — matches R source exactly).

    Returns (rss_perm (n_perm,), by_boot (n_perm, K), bx_boot (n_perm, K)).
    """
    K = bx.shape[0]
    device = bx.device

    # Sample on CPU then move; CPU Generator is deterministic across
    # platforms, while CUDA Generators yield slightly different streams.
    # K * n_perm is small (≤ a few million) so the H2D transfer cost is
    # negligible compared with the R reference's MRPRESSO replicate loop.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bx_eps = torch.randn(
        n_perm, K, dtype=torch.float64, generator=gen,
    ).to(device)
    by_eps = torch.randn(
        n_perm, K, dtype=torch.float64, generator=gen,
    ).to(device)

    # Predicted by_i under the LOO causal effect: β_LOO[i] * bx[i]
    pred_by = (beta_loo_obs * bx).unsqueeze(0)  # (1, K)

    bx_boot = bx.unsqueeze(0) + bx_eps * se_x.unsqueeze(0)  # (n_perm, K)
    by_boot = pred_by + by_eps * se_y.unsqueeze(0)  # (n_perm, K)

    # Vectorized LOO RSS: for each replicate t, compute
    #     beta_LOO[t, i] = (S_xy[t] - w_i bx_boot[t,i] by_boot[t,i]) /
    #                      (S_xx[t] - w_i bx_boot[t,i]^2)
    # then RSS[t] = sum_i w_i * (by_boot[t,i] - beta_LOO[t,i] bx_boot[t,i])^2
    # Note: weights are the original w (1/se_y^2), not se_y_boot.
    w_b = w.unsqueeze(0)  # (1, K)
    wxy = w_b * bx_boot * by_boot  # (n_perm, K)
    wxx = w_b * bx_boot**2  # (n_perm, K)

    Sxy = wxy.sum(dim=1, keepdim=True)  # (n_perm, 1)
    Sxx = wxx.sum(dim=1, keepdim=True)  # (n_perm, 1)

    num_loo = Sxy - wxy  # (n_perm, K)
    den_loo = (Sxx - wxx).clamp(min=1e-30)  # (n_perm, K)
    beta_loo_b = num_loo / den_loo  # (n_perm, K)

    resid = by_boot - beta_loo_b * bx_boot  # (n_perm, K)
    rss_perm = (w_b * resid**2).sum(dim=1)  # (n_perm,)

    return rss_perm, by_boot, bx_boot


def _presso_null_permutation(
    bx: Tensor,
    by: Tensor,
    w: Tensor,
    *,
    n_perm: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Legacy TorchGWAS permutation null (pre-2026-04-30 behavior).

    Permutes outcome betas against fixed exposure betas, refitting LOO
    RSS each replicate. Different reference null than MRPRESSO; kept for
    backward-compatibility under ``null="permutation"``.

    Returns (rss_perm, by_perm_all, bx_unchanged_repeated).
    """
    K = bx.shape[0]
    device = bx.device

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    by_perm_all = torch.empty(n_perm, K, dtype=torch.float64, device=device)
    rss_perm = torch.empty(n_perm, dtype=torch.float64, device=device)

    for t in range(n_perm):
        perm_idx = torch.randperm(K, generator=gen, device=device)
        by_p = by[perm_idx]
        rss_t, _ = _rss_loo(bx, by_p, w)
        by_perm_all[t] = by_p
        rss_perm[t] = rss_t

    bx_perm_all = bx.unsqueeze(0).expand(n_perm, K).contiguous()
    return rss_perm, by_perm_all, bx_perm_all


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
