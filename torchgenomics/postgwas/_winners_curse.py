"""Winner's curse correction for GWAS effect sizes.

Three methods to correct the upward bias in effect-size estimates for
variants selected at genome-wide significance:

* **Conditional likelihood** (Ghosh et al., 2008) — closed-form shrinkage
  based on the truncated normal distribution:

      beta_adj = beta - se * [phi(c - z) - phi(-c - z)]
                            / [Phi(z - c) + Phi(-z - c)]

  where ``z = beta / se`` and ``c = z_{alpha/2}``.

* **FIQT** (FDR Inverse Quantile Transformation, Bigdeli et al., 2022) —
  estimates a local false discovery rate for each z-score and shrinks
  proportionally:

      lfdr(z) = pi0 * phi(z) / f(z)
      z_adj   = z * (1 - lfdr)

  where ``f(z)`` is the marginal density estimated by Gaussian KDE.

* **Bootstrap** — parametric bootstrap that resamples from
  ``N(beta_obs, se^2)``, applies the selection threshold, and estimates
  the selection bias as ``mean(selected) - beta_obs``.

**Polyploid compatibility.** All three methods operate on GWAS-reported
beta and SE, which are ploidy-agnostic. The z-score ``z = beta / se``
is invariant under ploidy scaling, so all corrections are valid for
arbitrary ploidy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------

_SQRT2 = 2.0 ** 0.5
_SQRT2PI = (2.0 * 3.141592653589793) ** 0.5


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * torch.erfc(-x / _SQRT2)


def _normal_sf(x: Tensor) -> Tensor:
    return 0.5 * torch.erfc(x / _SQRT2)


def _normal_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x ** 2) / _SQRT2PI


def _normal_quantile(p: Tensor) -> Tensor:
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(p)
    from scipy.stats import norm  # type: ignore[import-untyped]
    return torch.tensor(
        norm.ppf(p.detach().cpu().numpy()),
        dtype=p.dtype, device=p.device,
    )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WinnersCurseResult:
    """Result from winner's curse correction.

    Attributes
    ----------
    method : str
        Name of the correction method used.
    beta_adjusted : (m,) Tensor
        Corrected effect sizes.
    se_adjusted : (m,) Tensor or None
        Adjusted standard errors (available for conditional likelihood).
    shrinkage_factor : (m,) Tensor
        Ratio ``beta_adjusted / beta_original`` (clamped to avoid division
        by zero when ``beta_original ~ 0``).
    n_corrected : int
        Number of variants that received non-trivial correction.
    """

    method: str
    beta_adjusted: Tensor
    se_adjusted: Tensor | None
    shrinkage_factor: Tensor
    n_corrected: int


# ---------------------------------------------------------------------------
# Method A: Conditional likelihood (Ghosh 2008)
# ---------------------------------------------------------------------------

def conditional_likelihood(
    beta: Tensor,
    se: Tensor,
    alpha: float = 5e-8,
) -> WinnersCurseResult:
    """Conditional-likelihood winner's curse correction.

    Shrinks effect sizes toward zero using the truncated normal
    conditional MLE (Ghosh et al. 2008).

    Parameters
    ----------
    beta : (m,) Tensor
        Observed effect sizes.
    se : (m,) Tensor
        Standard errors.
    alpha : float
        Significance threshold (variants with ``|z| < z_alpha`` are
        left uncorrected).

    Returns
    -------
    WinnersCurseResult
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)

    z = beta / se
    c = _normal_quantile(
        torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64, device=beta.device)
    )

    # Numerator: phi(c - z) - phi(-c - z)
    num = _normal_pdf(c - z) - _normal_pdf(-c - z)
    # Denominator: Phi(z - c) + Phi(-z - c) = P(|Z| > c under shift z)
    den = _normal_cdf(z - c) + _normal_cdf(-z - c)
    den = torch.clamp(den, min=1e-300)

    correction = se * (num / den)

    # Only correct significant variants
    sig_mask = torch.abs(z) >= c
    beta_adj = beta.clone()
    beta_adj[sig_mask] = beta[sig_mask] - correction[sig_mask]

    # Shrinkage factor
    safe_beta = beta.clone()
    safe_beta[safe_beta.abs() < 1e-300] = 1e-300
    shrinkage = beta_adj / safe_beta

    return WinnersCurseResult(
        method="conditional_likelihood",
        beta_adjusted=beta_adj,
        se_adjusted=None,
        shrinkage_factor=shrinkage,
        n_corrected=int(sig_mask.sum().item()),
    )


# ---------------------------------------------------------------------------
# Method B: FIQT (Bigdeli 2022)
# ---------------------------------------------------------------------------

def fiqt(
    z: Tensor,
    se: Tensor,
    pi0: float | None = None,
    bandwidth: float | None = None,
) -> WinnersCurseResult:
    """FIQT (FDR Inverse Quantile Transformation) correction.

    Estimates the local FDR for each z-score and shrinks proportionally.

    Parameters
    ----------
    z : (m,) Tensor
        Z-scores from the *full* GWAS (not just significant hits).
    se : (m,) Tensor
        Standard errors.
    pi0 : float or None
        Null proportion. If None, estimated from the data using the
        median p-value method.
    bandwidth : float or None
        Bandwidth for Gaussian KDE. If None, Silverman's rule of thumb.

    Returns
    -------
    WinnersCurseResult
    """
    z = z.to(torch.float64)
    se = se.to(torch.float64)
    m = z.shape[0]
    if m == 0:
        raise ValueError("fiqt requires at least one z-score")

    # FDR Inverse Quantile Transformation (Bigdeli et al. 2016, Bioinformatics):
    #   1. two-sided p-values from the z-scores,
    #   2. Benjamini-Hochberg adjust them,
    #   3. back-transform each adjusted p to a shrunken z via
    #      mu_z = sign(z) * qnorm(1 - p_adj/2) = sign(z) * -Phi^{-1}(p_adj/2).
    # The previous body did Gaussian-KDE local-fdr shrinkage instead — a
    # different (valid) estimator, but NOT FIQT, so it did not match the
    # reference FIQT implementation. pi0/bandwidth are unused by FIQT and kept
    # only for signature compatibility.
    del pi0, bandwidth

    p_vals = (2.0 * _normal_sf(torch.abs(z))).clamp(min=1e-300, max=1.0)

    # Benjamini-Hochberg adjusted p-values (step-up, reverse cumulative min).
    sorted_p, sort_idx = p_vals.sort()
    ranks = torch.arange(1, m + 1, dtype=torch.float64, device=z.device)
    adj_sorted = (sorted_p * m / ranks).flip(0).cummin(0).values.flip(0)
    adj_sorted = adj_sorted.clamp(max=1.0)
    p_adj = torch.empty_like(p_vals)
    p_adj[sort_idx] = adj_sorted

    # Back-transform. For p_adj == 1 the shrunken z is 0; ndtri(0.5)=0 handles it.
    z_adj = torch.sign(z) * (-torch.special.ndtri((p_adj / 2.0).clamp(min=1e-300, max=0.5)))

    beta_adj = z_adj * se
    beta_orig = z * se
    safe_orig = beta_orig.clone()
    safe_orig[safe_orig.abs() < 1e-300] = 1e-300
    shrinkage = beta_adj / safe_orig

    n_corrected = int((p_adj > 0.01).sum().item())

    return WinnersCurseResult(
        method="fiqt",
        beta_adjusted=beta_adj,
        se_adjusted=None,
        shrinkage_factor=shrinkage,
        n_corrected=n_corrected,
    )


# ---------------------------------------------------------------------------
# Method C: Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_correction(
    beta: Tensor,
    se: Tensor,
    alpha: float = 5e-8,
    n_boot: int = 10000,
    seed: int | None = None,
) -> WinnersCurseResult:
    """Parametric bootstrap winner's curse correction.

    For each significant variant, resamples from ``N(beta, se^2)``,
    applies the significance threshold, and estimates the conditional
    bias.

    Parameters
    ----------
    beta : (m,) Tensor
        Observed effect sizes.
    se : (m,) Tensor
        Standard errors.
    alpha : float
        Significance threshold.
    n_boot : int
        Number of bootstrap replicates.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    WinnersCurseResult
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)

    z_alpha = _normal_quantile(
        torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64, device=beta.device)
    )

    sig_mask = (torch.abs(beta / se)) >= z_alpha
    beta_adj = beta.clone()

    if seed is not None:
        rng = torch.Generator(device=beta.device)
        rng.manual_seed(seed)
    else:
        rng = None

    sig_indices = torch.where(sig_mask)[0]
    for idx in sig_indices:
        b = beta[idx]
        s = se[idx]
        # Resample: beta_star ~ N(b, s^2)
        if rng is not None:
            noise = torch.randn(n_boot, dtype=torch.float64,
                                device=beta.device, generator=rng)
        else:
            noise = torch.randn(n_boot, dtype=torch.float64,
                                device=beta.device)
        beta_star = b + s * noise
        z_star = beta_star / s
        # Apply selection: keep only those that would be "significant"
        selected = beta_star[torch.abs(z_star) >= z_alpha]
        if selected.numel() > 0:
            bias = selected.mean() - b
            beta_adj[idx] = b - bias

    safe_beta = beta.clone()
    safe_beta[safe_beta.abs() < 1e-300] = 1e-300
    shrinkage = beta_adj / safe_beta

    return WinnersCurseResult(
        method="bootstrap",
        beta_adjusted=beta_adj,
        se_adjusted=None,
        shrinkage_factor=shrinkage,
        n_corrected=int(sig_mask.sum().item()),
    )


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------

def correct_winners_curse(
    beta: Tensor,
    se: Tensor,
    method: str = "conditional_likelihood",
    alpha: float = 5e-8,
    **kwargs,
) -> WinnersCurseResult:
    """Correct winner's curse using the specified method.

    Parameters
    ----------
    beta : (m,) Tensor
        Observed effect sizes.
    se : (m,) Tensor
        Standard errors.
    method : str
        One of ``"conditional_likelihood"``, ``"fiqt"``, ``"bootstrap"``.
    alpha : float
        Significance threshold (used by CL and bootstrap; ignored by FIQT).
    **kwargs
        Extra keyword arguments passed to the chosen method.

    Returns
    -------
    WinnersCurseResult
    """
    if method == "conditional_likelihood":
        return conditional_likelihood(beta, se, alpha=alpha, **kwargs)
    elif method == "fiqt":
        z = beta / se
        return fiqt(z, se, **kwargs)
    elif method == "bootstrap":
        return bootstrap_correction(beta, se, alpha=alpha, **kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'conditional_likelihood', "
            f"'fiqt', or 'bootstrap'."
        )
