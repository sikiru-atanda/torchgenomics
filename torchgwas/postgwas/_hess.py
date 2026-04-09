"""HESS: Heritability Estimation from Summary Statistics.

Implements Shi et al. (2016, AJHG), "Local genetic correlation gives
insights into the shared genetic architecture of complex traits".

Partitions genome-wide heritability into local regions using z-scores
and LD (correlation) matrices:

For region r with m_r SNPs:

    h2_r = (z_r^T R_r^{-1} z_r  -  m_r) / N

Using eigendecomposition for numerical stability:

    R_r = U D U^T
    h2_r = (sum_k (u_k^T z_r)^2 / d_k  -  m_r) / N

where only eigenvalues ``d_k >= threshold`` are kept. The variance is:

    var(h2_r) = 2 * trace(R_r^{-2}) / N^2

Genome-wide heritability: ``h2 = sum_r h2_r``.

Local genetic correlation between traits 1 and 2:

    rho_r = z1_r^T R_r^{-1} z2_r / N

**Polyploid compatibility.** HESS operates on z-scores from GWAS
summary statistics and LD matrices from reference panels. Both are
ploidy-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HESSRegionResult:
    """Per-region heritability estimate.

    Attributes
    ----------
    region_id : str
        Region label (e.g. "chr1:1000000-2000000").
    chrom : str
        Chromosome.
    start : int
        Start index (SNP index within the z-score vector).
    end : int
        End index (exclusive).
    h2_local : float
        Local heritability estimate.
    h2_local_se : float
        Standard error of h2_local.
    n_snps : int
        Number of SNPs in the region.
    n_eigenvalues_kept : int
        Number of eigenvalues above threshold (effective rank).
    """

    region_id: str
    chrom: str
    start: int
    end: int
    h2_local: float
    h2_local_se: float
    n_snps: int
    n_eigenvalues_kept: int


@dataclass
class HESSResult:
    """Aggregate HESS result across regions.

    Attributes
    ----------
    regions : list[HESSRegionResult]
        Per-region results.
    h2_total : float
        Sum of local heritabilities.
    h2_total_se : float
        Standard error of total h2 (quadrature sum).
    n_regions : int
        Number of regions.
    n_snps_total : int
        Total SNPs across all regions.
    """

    regions: list[HESSRegionResult]
    h2_total: float
    h2_total_se: float
    n_regions: int
    n_snps_total: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def hess_local_h2(
    z: Tensor,
    ld_matrix: Tensor,
    n: int | float,
    region_bounds: list[tuple[int, int]],
    region_labels: list[str] | None = None,
    eigenvalue_threshold: float = 1.0,
) -> HESSResult:
    """Partition heritability into local regions (HESS).

    Parameters
    ----------
    z : (m,) Tensor
        Z-scores from GWAS.
    ld_matrix : (m, m) Tensor
        LD correlation matrix (reference panel).
    n : int or float
        GWAS sample size.
    region_bounds : list of (start, end) tuples
        SNP index ranges for each region (end is exclusive).
    region_labels : list[str] or None
        Optional labels for each region.
    eigenvalue_threshold : float
        Minimum eigenvalue to retain (default 1.0).

    Returns
    -------
    HESSResult

    Raises
    ------
    ValueError
        If region_bounds is empty or indices are out of range.
    """
    if len(region_bounds) == 0:
        raise ValueError("At least one region is required.")

    z = z.to(torch.float64)
    R = ld_matrix.to(torch.float64)
    N = float(n)

    regions: list[HESSRegionResult] = []
    h2_sum = 0.0
    se_sq_sum = 0.0
    total_snps = 0

    for r_idx, (start, end) in enumerate(region_bounds):
        m_r = end - start
        if m_r <= 0:
            continue

        z_r = z[start:end]
        R_r = R[start:end, start:end]

        # Eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(R_r)

        # Keep eigenvalues above threshold
        keep = eigvals >= eigenvalue_threshold
        n_kept = int(keep.sum().item())

        if n_kept == 0:
            # All eigenvalues below threshold — skip
            label = (region_labels[r_idx] if region_labels
                     else f"region_{r_idx}")
            regions.append(HESSRegionResult(
                region_id=label, chrom="", start=start, end=end,
                h2_local=0.0, h2_local_se=0.0, n_snps=m_r,
                n_eigenvalues_kept=0,
            ))
            total_snps += m_r
            continue

        d = eigvals[keep]
        U = eigvecs[:, keep]

        # h2_r = (sum_k (u_k^T z_r)^2 / d_k - m_r) / N
        proj = U.t() @ z_r  # (n_kept,)
        quad_form = (proj ** 2 / d).sum()
        h2_r = float((quad_form - m_r) / N)

        # var(h2_r) = 2 * trace(R_r^{-2}) / N^2
        # With eigendecomposition: trace(R^{-2}) = sum(1/d_k^2)
        tr_R_inv2 = (1.0 / d ** 2).sum()
        se_r = float(torch.sqrt(2.0 * tr_R_inv2).item()) / N

        label = region_labels[r_idx] if region_labels else f"region_{r_idx}"
        regions.append(HESSRegionResult(
            region_id=label, chrom="", start=start, end=end,
            h2_local=h2_r, h2_local_se=se_r, n_snps=m_r,
            n_eigenvalues_kept=n_kept,
        ))

        h2_sum += h2_r
        se_sq_sum += se_r ** 2
        total_snps += m_r

    return HESSResult(
        regions=regions,
        h2_total=h2_sum,
        h2_total_se=se_sq_sum ** 0.5,
        n_regions=len(regions),
        n_snps_total=total_snps,
    )


def hess_local_rg(
    z1: Tensor,
    z2: Tensor,
    ld_matrix: Tensor,
    n1: int | float,
    n2: int | float,
    region_bounds: list[tuple[int, int]],
    region_labels: list[str] | None = None,
    eigenvalue_threshold: float = 1.0,
) -> HESSResult:
    """Local genetic correlation between two traits (HESS).

    Uses the cross-term analog of the local h2 estimator:

        rho_r = z1_r^T R_r^{-1} z2_r / sqrt(N1 * N2)

    Parameters
    ----------
    z1 : (m,) Tensor
        Z-scores from trait 1.
    z2 : (m,) Tensor
        Z-scores from trait 2.
    ld_matrix : (m, m) Tensor
        LD correlation matrix.
    n1, n2 : int or float
        Sample sizes for traits 1 and 2.
    region_bounds : list of (start, end) tuples
        SNP index ranges for each region.
    region_labels : list[str] or None
        Optional labels.
    eigenvalue_threshold : float
        Minimum eigenvalue to retain.

    Returns
    -------
    HESSResult
        ``h2_local`` fields contain the local genetic covariance
        (not heritability) for each region.
    """
    if len(region_bounds) == 0:
        raise ValueError("At least one region is required.")

    z1 = z1.to(torch.float64)
    z2 = z2.to(torch.float64)
    R = ld_matrix.to(torch.float64)
    N_eff = (float(n1) * float(n2)) ** 0.5

    regions: list[HESSRegionResult] = []
    rg_sum = 0.0
    se_sq_sum = 0.0
    total_snps = 0

    for r_idx, (start, end) in enumerate(region_bounds):
        m_r = end - start
        if m_r <= 0:
            continue

        z1_r = z1[start:end]
        z2_r = z2[start:end]
        R_r = R[start:end, start:end]

        eigvals, eigvecs = torch.linalg.eigh(R_r)
        keep = eigvals >= eigenvalue_threshold
        n_kept = int(keep.sum().item())

        if n_kept == 0:
            label = (region_labels[r_idx] if region_labels
                     else f"region_{r_idx}")
            regions.append(HESSRegionResult(
                region_id=label, chrom="", start=start, end=end,
                h2_local=0.0, h2_local_se=0.0, n_snps=m_r,
                n_eigenvalues_kept=0,
            ))
            total_snps += m_r
            continue

        d = eigvals[keep]
        U = eigvecs[:, keep]

        proj1 = U.t() @ z1_r
        proj2 = U.t() @ z2_r
        cross = (proj1 * proj2 / d).sum()
        rg_r = float(cross.item()) / N_eff

        tr_R_inv2 = (1.0 / d ** 2).sum()
        se_r = float(torch.sqrt(2.0 * tr_R_inv2).item()) / N_eff

        label = region_labels[r_idx] if region_labels else f"region_{r_idx}"
        regions.append(HESSRegionResult(
            region_id=label, chrom="", start=start, end=end,
            h2_local=rg_r, h2_local_se=se_r, n_snps=m_r,
            n_eigenvalues_kept=n_kept,
        ))

        rg_sum += rg_r
        se_sq_sum += se_r ** 2
        total_snps += m_r

    return HESSResult(
        regions=regions,
        h2_total=rg_sum,
        h2_total_se=se_sq_sum ** 0.5,
        n_regions=len(regions),
        n_snps_total=total_snps,
    )
