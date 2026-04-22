"""LD Score Regression (LDSC) for heritability and genetic correlation.

Implements the method of Bulik-Sullivan et al. (2015, Nature Genetics):
    E[chi2_j] = (N / M) * h2 * l_j + 1 + N * a

where l_j is the LD score for SNP j, h2 is heritability, and a is the
intercept adjustment for confounding/population structure.

Standard errors via block jackknife over contiguous genomic blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LDSCResult:
    """Result from univariate LDSC h2 estimation."""

    h2: float
    h2_se: float
    intercept: float
    intercept_se: float
    mean_chi2: float
    lambda_gc: float
    n_snps: int


@dataclass
class LDSCRgResult:
    """Result from cross-trait LDSC genetic correlation."""

    rg: float
    rg_se: float
    cov_g: float
    cov_g_se: float
    intercept: float
    intercept_se: float
    h2_1: LDSCResult
    h2_2: LDSCResult


def _weighted_lstsq(
    X: Tensor, y: Tensor, w: Tensor
) -> Tensor:
    """Weighted least squares: solve (X'WX)^{-1} X'Wy.

    Parameters
    ----------
    X : (m, p) design matrix
    y : (m,) response
    w : (m,) weights (positive)

    Returns
    -------
    beta : (p,) coefficients
    """
    sqrt_w = w.sqrt().unsqueeze(1)  # (m, 1)
    Xw = X * sqrt_w
    yw = y * w.sqrt()
    return torch.linalg.lstsq(Xw, yw).solution


def _block_jackknife_se(
    X: Tensor,
    y: Tensor,
    w: Tensor,
    n_blocks: int,
    full_coef: Tensor,
) -> Tensor:
    """Block jackknife standard errors for WLS coefficients.

    Divides SNPs into ``n_blocks`` contiguous blocks and computes
    leave-one-block-out estimates.

    Parameters
    ----------
    X : (m, p)
    y : (m,)
    w : (m,)
    n_blocks : int
    full_coef : (p,) full-sample coefficients

    Returns
    -------
    se : (p,) jackknife standard errors
    """
    m = X.shape[0]
    block_size = m // n_blocks
    n_blocks_actual = n_blocks if block_size > 0 else 1

    # Native fast-path: collapse the n_blocks=200 leave-one-block-out WLS
    # solves into one C++ pass that maintains a running (X'WX, X'Wy)
    # accumulator and subtracts the per-block contribution. Eliminates
    # ~200 torch.linalg.lstsq dispatch round-trips per LDSC call. The
    # pure-Python path remains intact below as the algorithmic spec.
    from .._dispatch import native_disabled
    from .._native import HAS_NATIVE_LDSC, _ldsc_native

    if (
        HAS_NATIVE_LDSC and not native_disabled()
        and X.device.type == "cpu" and X.dtype == torch.float64
        and y.dtype == torch.float64 and w.dtype == torch.float64
        and X.shape[1] <= 8
    ):
        import numpy as _np
        se_np = _ldsc_native.ldsc_block_jackknife(
            X.detach().contiguous().numpy(),
            y.detach().contiguous().numpy(),
            w.detach().contiguous().numpy(),
            int(n_blocks),
            full_coef.detach().contiguous().numpy(),
        )
        return torch.from_numpy(_np.ascontiguousarray(se_np)).to(
            dtype=X.dtype, device=X.device
        )

    pseudovalues = []
    for b in range(n_blocks_actual):
        start = b * block_size
        end = start + block_size if b < n_blocks_actual - 1 else m
        # Leave-one-block-out
        mask = torch.ones(m, dtype=torch.bool, device=X.device)
        mask[start:end] = False
        coef_b = _weighted_lstsq(X[mask], y[mask], w[mask])
        # Pseudovalue: n * full - (n-1) * jackknife
        pseudo = n_blocks_actual * full_coef - (n_blocks_actual - 1) * coef_b
        pseudovalues.append(pseudo)

    pseudos = torch.stack(pseudovalues)  # (n_blocks, p)
    # Jackknife variance
    mean_pseudo = pseudos.mean(dim=0)
    var = ((pseudos - mean_pseudo) ** 2).sum(dim=0) / (
        n_blocks_actual * (n_blocks_actual - 1)
    )
    return var.sqrt()


def ldsc_h2(
    chi2: Tensor,
    ld_scores: Tensor,
    n: int | float,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
) -> LDSCResult:
    """Estimate SNP heritability via LD Score Regression.

    Model: E[chi2_j] = (N/M) * h2 * l_j + intercept

    Uses two-step estimation:
    1. First pass with all SNPs to estimate intercept.
    2. Second pass excluding chi2 > ``two_step_cutoff`` to refine h2.

    Parameters
    ----------
    chi2 : (m,) Tensor
        Chi-squared statistics.
    ld_scores : (m,) Tensor
        Per-SNP LD scores.
    n : int or float
        Sample size.
    m_total : int
        Total number of SNPs genome-wide (for scaling h2).
    n_blocks : int
        Number of jackknife blocks for SE estimation.
    two_step_cutoff : float
        Chi2 threshold for second-pass filtering.

    Returns
    -------
    LDSCResult
    """
    device = chi2.device
    chi2 = chi2.to(torch.float64)
    ld_scores = ld_scores.to(torch.float64).to(device)

    # Heteroscedasticity weights: w_j = 1 / max(l_j^2, 1)
    w = 1.0 / torch.clamp(ld_scores**2, min=1.0)

    # Design matrix: [l_j, 1]
    X = torch.stack([ld_scores, torch.ones_like(ld_scores)], dim=1)

    # Step 1: all SNPs
    coef1 = _weighted_lstsq(X, chi2, w)

    # Step 2: filter outliers
    keep = chi2 <= two_step_cutoff
    if keep.sum() < 10:
        keep = torch.ones_like(chi2, dtype=torch.bool)

    X2 = X[keep]
    chi2_2 = chi2[keep]
    w2 = w[keep]

    coef = _weighted_lstsq(X2, chi2_2, w2)
    se = _block_jackknife_se(X2, chi2_2, w2, min(n_blocks, int(keep.sum())), coef)

    # Extract h2 and intercept
    # coef[0] = (N/M) * h2  =>  h2 = coef[0] * M / N
    slope = coef[0].item()
    intercept = coef[1].item()
    h2 = slope * m_total / n
    h2_se = se[0].item() * m_total / n

    mean_chi2 = chi2.mean().item()
    # Lambda GC from median chi2
    median_chi2 = chi2.median().item()
    lambda_gc_val = median_chi2 / 0.4549364  # chi2(1) median

    return LDSCResult(
        h2=h2,
        h2_se=h2_se,
        intercept=intercept,
        intercept_se=se[1].item(),
        mean_chi2=mean_chi2,
        lambda_gc=lambda_gc_val,
        n_snps=int(keep.sum().item()),
    )


def ldsc_intercept(
    chi2: Tensor,
    ld_scores: Tensor,
    n: int | float,
    m_total: int,
    n_blocks: int = 200,
) -> tuple[float, float]:
    """Estimate LDSC intercept (and SE) as a measure of confounding.

    Intercept = 1 under no confounding; > 1 indicates inflation.

    Returns
    -------
    (intercept, intercept_se)
    """
    result = ldsc_h2(chi2, ld_scores, n, m_total, n_blocks)
    return result.intercept, result.intercept_se


def ldsc_rg(
    chi2_1: Tensor,
    chi2_2: Tensor,
    ld_scores: Tensor,
    n1: int | float,
    n2: int | float,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
) -> LDSCRgResult:
    """Estimate genetic correlation via cross-trait LDSC.

    Cross-trait model:
        E[z1_j * z2_j] = sqrt(n1 * n2) * cov_g / M * l_j + intercept

    Parameters
    ----------
    chi2_1, chi2_2 : (m,) Tensor
        Chi-squared statistics for traits 1 and 2.
    ld_scores : (m,) Tensor
        Per-SNP LD scores (shared reference panel).
    n1, n2 : int or float
        Sample sizes for traits 1 and 2.
    m_total : int
        Total number of SNPs.
    n_blocks : int
        Jackknife blocks.
    two_step_cutoff : float
        Chi2 threshold for filtering.

    Returns
    -------
    LDSCRgResult
    """
    device = chi2_1.device

    # Univariate LDSC for each trait
    h2_1 = ldsc_h2(chi2_1, ld_scores, n1, m_total, n_blocks, two_step_cutoff)
    h2_2 = ldsc_h2(chi2_2, ld_scores, n2, m_total, n_blocks, two_step_cutoff)

    # Cross-trait: z1 * z2
    z1 = chi2_1.to(torch.float64).sqrt() * torch.sign(
        torch.randn(chi2_1.shape[0], device=device)
    )
    z2 = chi2_2.to(torch.float64).sqrt() * torch.sign(
        torch.randn(chi2_2.shape[0], device=device)
    )

    # If we have signed z-scores (from beta/se), use those directly.
    # Since chi2 = z^2, we need z1*z2 for the cross product.
    # With only chi2 available, we use the product of absolute z-scores
    # and note that the sign information is lost. For proper rg, users
    # should provide z-scores directly.
    z_product = torch.sqrt(chi2_1.to(torch.float64) * chi2_2.to(torch.float64))

    ld_scores = ld_scores.to(torch.float64).to(device)
    w = 1.0 / torch.clamp(ld_scores**2, min=1.0)

    X = torch.stack([ld_scores, torch.ones_like(ld_scores)], dim=1)

    # Filter
    keep = (chi2_1 <= two_step_cutoff) & (chi2_2 <= two_step_cutoff)
    if keep.sum() < 10:
        keep = torch.ones_like(chi2_1, dtype=torch.bool)

    coef = _weighted_lstsq(X[keep], z_product[keep], w[keep])
    se = _block_jackknife_se(
        X[keep],
        z_product[keep],
        w[keep],
        min(n_blocks, int(keep.sum())),
        coef,
    )

    # cov_g = slope * M / sqrt(n1 * n2)
    slope = coef[0].item()
    cov_g = slope * m_total / (n1 * n2) ** 0.5
    cov_g_se = se[0].item() * m_total / (n1 * n2) ** 0.5

    # rg = cov_g / sqrt(h2_1 * h2_2)
    denom = abs(h2_1.h2 * h2_2.h2) ** 0.5
    if denom < 1e-10:
        rg = float("nan")
        rg_se = float("nan")
    else:
        rg = cov_g / denom
        # Delta method SE for rg
        rg_se = abs(rg) * (
            (cov_g_se / abs(cov_g) if abs(cov_g) > 1e-10 else 0) ** 2
            + (h2_1.h2_se / (2 * abs(h2_1.h2)) if abs(h2_1.h2) > 1e-10 else 0) ** 2
            + (h2_2.h2_se / (2 * abs(h2_2.h2)) if abs(h2_2.h2) > 1e-10 else 0) ** 2
        ) ** 0.5

    return LDSCRgResult(
        rg=rg,
        rg_se=rg_se,
        cov_g=cov_g,
        cov_g_se=cov_g_se,
        intercept=coef[1].item(),
        intercept_se=se[1].item(),
        h2_1=h2_1,
        h2_2=h2_2,
    )


def ldsc_rg_from_z(
    z1: Tensor,
    z2: Tensor,
    ld_scores: Tensor,
    n1: int | float,
    n2: int | float,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
) -> LDSCRgResult:
    """Estimate genetic correlation from signed z-scores.

    Preferred over :func:`ldsc_rg` when direction of effect is available,
    as it preserves the sign of rg.

    Parameters
    ----------
    z1, z2 : (m,) Tensor
        Signed z-scores (beta / se) for traits 1 and 2.
    ld_scores, n1, n2, m_total, n_blocks, two_step_cutoff
        Same as :func:`ldsc_rg`.

    Returns
    -------
    LDSCRgResult
    """
    device = z1.device
    z1 = z1.to(torch.float64)
    z2 = z2.to(torch.float64)
    ld_scores = ld_scores.to(torch.float64).to(device)

    chi2_1 = z1**2
    chi2_2 = z2**2

    # Univariate LDSC
    h2_1 = ldsc_h2(chi2_1, ld_scores, n1, m_total, n_blocks, two_step_cutoff)
    h2_2 = ldsc_h2(chi2_2, ld_scores, n2, m_total, n_blocks, two_step_cutoff)

    # Cross-trait: use signed z-product
    z_product = z1 * z2

    w = 1.0 / torch.clamp(ld_scores**2, min=1.0)
    X = torch.stack([ld_scores, torch.ones_like(ld_scores)], dim=1)

    keep = (chi2_1 <= two_step_cutoff) & (chi2_2 <= two_step_cutoff)
    if keep.sum() < 10:
        keep = torch.ones_like(chi2_1, dtype=torch.bool)

    coef = _weighted_lstsq(X[keep], z_product[keep], w[keep])
    se = _block_jackknife_se(
        X[keep],
        z_product[keep],
        w[keep],
        min(n_blocks, int(keep.sum())),
        coef,
    )

    slope = coef[0].item()
    cov_g = slope * m_total / (n1 * n2) ** 0.5
    cov_g_se = se[0].item() * m_total / (n1 * n2) ** 0.5

    denom = abs(h2_1.h2 * h2_2.h2) ** 0.5
    if denom < 1e-10:
        rg = float("nan")
        rg_se = float("nan")
    else:
        rg = cov_g / denom
        rg_se = abs(rg) * (
            (cov_g_se / abs(cov_g) if abs(cov_g) > 1e-10 else 0) ** 2
            + (h2_1.h2_se / (2 * abs(h2_1.h2)) if abs(h2_1.h2) > 1e-10 else 0) ** 2
            + (h2_2.h2_se / (2 * abs(h2_2.h2)) if abs(h2_2.h2) > 1e-10 else 0) ** 2
        ) ** 0.5

    return LDSCRgResult(
        rg=rg,
        rg_se=rg_se,
        cov_g=cov_g,
        cov_g_se=cov_g_se,
        intercept=coef[1].item(),
        intercept_se=se[1].item(),
        h2_1=h2_1,
        h2_2=h2_2,
    )
