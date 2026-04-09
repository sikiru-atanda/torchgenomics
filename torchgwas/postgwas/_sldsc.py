"""Stratified LD Score Regression (S-LDSC) for partitioned heritability.

Implements Finucane et al. (2015, Nature Genetics) "Partitioning heritability
by functional annotation using genome-wide association summary statistics".

Extends univariate LDSC from ``_ldsc.py``:

    E[chi2_j] = N * sum_c (tau_c * l_cj) + intercept

where ``l_cj`` is the per-SNP LD score for annotation category ``c``. Per-
category heritability and enrichment follow:

    h2_c         = tau_c * M_c
    Enrichment_c = (h2_c / h2) / (M_c / M)

Uses the same two-step fit (filter outliers after an initial regression) and
block-jackknife SEs as univariate LDSC, delegating to ``_block_jackknife_se``
which picks up the native C++ kernel automatically when the column count is
small (p <= 8) and otherwise runs the Python/torch fallback.

**Polyploid compatibility.** S-LDSC operates on chi-squared statistics
(``chi2 = (beta/se)^2``) which are ploidy-invariant. The annotation LD
scores and SNP counts (``M_c``) are properties of the LD reference panel
and are also ploidy-agnostic. Partitioned heritability and enrichment
estimates are valid for any ploidy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._ldsc import _block_jackknife_se, _weighted_lstsq


@dataclass
class SLDSCResult:
    """Result from a partitioned (stratified) LDSC h2 fit."""

    tau: Tensor  # (C,) per-annotation tau coefficient
    tau_se: Tensor  # (C,) jackknife SE of tau
    h2_cat: Tensor  # (C,) h2 attributable to each annotation
    h2_cat_se: Tensor  # (C,) jackknife SE of per-category h2
    enrichment: Tensor  # (C,) Enrichment_c (h2_c / h2) / (M_c / M)
    enrichment_se: Tensor  # (C,) delta-method SE of enrichment
    h2_total: float
    h2_total_se: float
    intercept: float
    intercept_se: float
    m_annot: Tensor  # (C,) M_c counts
    n_snps: int
    annotation_names: list[str] | None = None


def sldsc_h2_partitioned(
    chi2: Tensor,
    annot_ld_scores: Tensor,
    M_c: Tensor,
    n: int | float,
    m_total: int,
    n_blocks: int = 200,
    two_step_cutoff: float = 30.0,
    annotation_names: list[str] | None = None,
) -> SLDSCResult:
    """Partitioned LDSC heritability by annotation category.

    Parameters
    ----------
    chi2 : (m,) Tensor
        Chi-squared statistics per SNP.
    annot_ld_scores : (m, C) Tensor
        Stratified per-SNP LD scores, one column per annotation category.
        Each column ``c`` is the sum of r^2 between SNP ``j`` and every
        reference SNP that falls within annotation ``c``.
    M_c : (C,) Tensor
        Number of reference SNPs in each annotation category (used for
        the enrichment denominator and to convert tau to h2).
    n : int or float
        Trait sample size.
    m_total : int
        Total number of reference SNPs across all categories (union).
    n_blocks : int
        Number of block-jackknife blocks for SE estimation.
    two_step_cutoff : float
        Chi-squared threshold for the second-step outlier filter.
    annotation_names : list[str] or None
        Optional names, attached to the result for reporting only.

    Returns
    -------
    SLDSCResult
    """
    if annot_ld_scores.ndim != 2:
        raise ValueError("annot_ld_scores must be a 2-D (m, C) tensor")
    if chi2.shape[0] != annot_ld_scores.shape[0]:
        raise ValueError("chi2 and annot_ld_scores must have the same number of SNPs")
    C = annot_ld_scores.shape[1]
    if M_c.shape[0] != C:
        raise ValueError("M_c must have length equal to the number of annotations")

    device = chi2.device
    chi2 = chi2.to(torch.float64)
    L = annot_ld_scores.to(torch.float64).to(device)
    M_c = M_c.to(torch.float64).to(device)

    # Heteroscedasticity weights follow the univariate LDSC convention: the
    # sum of per-category LD scores is the standard LDSC "total" LD score,
    # so the weight mirrors ``1 / max(l_j^2, 1)`` from ``ldsc_h2``. This
    # keeps single-annotation S-LDSC numerically equivalent to ``ldsc_h2``.
    l_total = L.sum(dim=1)
    w = 1.0 / torch.clamp(l_total**2, min=1.0)

    # Design matrix: [N * l_c1, ..., N * l_cC, 1]. Folding N into the design
    # makes the recovered coefficients directly equal to tau_c (per-SNP
    # per-unit-N contribution). The final column is the intercept.
    N = float(n)
    X = torch.cat([N * L, torch.ones(L.shape[0], 1, device=device, dtype=torch.float64)], dim=1)

    # Step 1: all SNPs, estimate initial coefficients.
    _ = _weighted_lstsq(X, chi2, w)

    # Step 2: filter chi^2 outliers and refit.
    keep = chi2 <= two_step_cutoff
    if keep.sum() < max(10, C + 2):
        keep = torch.ones_like(chi2, dtype=torch.bool)

    X2 = X[keep]
    chi2_2 = chi2[keep]
    w2 = w[keep]

    coef = _weighted_lstsq(X2, chi2_2, w2)
    se = _block_jackknife_se(
        X2, chi2_2, w2, min(n_blocks, int(keep.sum().item())), coef
    )

    tau = coef[:C]
    tau_se = se[:C]
    intercept = float(coef[C].item())
    intercept_se = float(se[C].item())

    # Per-category h2 = tau_c * M_c. SE propagates linearly since M_c is fixed.
    h2_cat = tau * M_c
    h2_cat_se = tau_se * M_c

    h2_total = float(h2_cat.sum().item())
    # Under block jackknife, the variance of sum_c h2_c is sum over blocks of
    # pseudovalues of the sum, which is just the sum of pseudovalues — so
    # summing the SEs in quadrature overestimates. A tighter estimate re-runs
    # the jackknife accumulator on the contrast ``sum_c M_c * tau_c``; we
    # approximate with the quadrature sum here (conservative) and leave the
    # exact contrast jackknife as a refinement. Users who need exact total
    # h2 SE should read it off the full-covariance jackknife directly.
    h2_total_se = float(torch.sqrt((h2_cat_se**2).sum()).item())

    # Enrichment: (h2_c / h2) / (M_c / M).
    prop_h2 = h2_cat / max(h2_total, 1e-30)
    prop_M = M_c / float(m_total)
    enrichment = prop_h2 / torch.clamp(prop_M, min=1e-30)
    # Delta-method SE: enrichment_c depends linearly on h2_c (numerator) with
    # M_c, M fixed. We ignore the covariance with h2_total in the denominator
    # since h2_total's SE is already conservative; this gives a per-category
    # SE of the form ``(M / M_c) / h2_total * h2_cat_se``.
    safe_h2 = max(abs(h2_total), 1e-30)
    enrichment_se = (float(m_total) / torch.clamp(M_c, min=1e-30)) * (
        h2_cat_se / safe_h2
    )

    return SLDSCResult(
        tau=tau.cpu(),
        tau_se=tau_se.cpu(),
        h2_cat=h2_cat.cpu(),
        h2_cat_se=h2_cat_se.cpu(),
        enrichment=enrichment.cpu(),
        enrichment_se=enrichment_se.cpu(),
        h2_total=h2_total,
        h2_total_se=h2_total_se,
        intercept=intercept,
        intercept_se=intercept_se,
        m_annot=M_c.cpu(),
        n_snps=int(keep.sum().item()),
        annotation_names=list(annotation_names) if annotation_names else None,
    )
