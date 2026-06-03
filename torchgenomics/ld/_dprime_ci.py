"""Likelihood-based confidence intervals for D-prime.

Evaluates the composite log-likelihood of D' on a dense grid and
extracts the 95% CI as the region where the log-likelihood is within
1.92 of the maximum (chi-squared 1-df half-quantile).

Fully vectorized over B SNP pairs for GPU acceleration.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-10
_CHI2_HALF = 1.9207  # 0.5 * chi2.ppf(0.95, 1) = 0.5 * 3.841 = 1.92


def dprime_confidence_interval(
    counts: Tensor,
    *,
    n_grid: int = 201,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute D' MLE and 95% CI via grid likelihood evaluation.

    Parameters
    ----------
    counts : (B, 3, 3) float64
        Genotype count tables.
    n_grid : int
        Number of grid points for D' in [-1, 1].

    Returns
    -------
    dprime_hat : (B,) MLE of D'
    ci_low : (B,) lower bound of 95% CI
    ci_high : (B,) upper bound of 95% CI
    """
    B = counts.shape[0]
    device = counts.device

    # Marginal allele frequencies
    N = counts.sum(dim=(1, 2)).clamp(min=1.0)  # (B,)
    n_loc1 = counts.sum(dim=2)  # (B, 3)
    n_loc2 = counts.sum(dim=1)  # (B, 3)

    p_A = ((2 * n_loc1[:, 2] + n_loc1[:, 1]) / (2 * N)).clamp(_EPS, 1 - _EPS)
    p_B = ((2 * n_loc2[:, 2] + n_loc2[:, 1]) / (2 * N)).clamp(_EPS, 1 - _EPS)

    p_a = 1.0 - p_A
    p_b = 1.0 - p_B

    # D' grid
    d_grid = torch.linspace(-1.0, 1.0, n_grid, dtype=torch.float64, device=device)

    # Convert D' to D for each grid point and pair
    # D_max depends on sign of D'
    D_max_pos = torch.min(p_A * p_b, p_a * p_B).clamp(min=_EPS)  # (B,)
    D_max_neg = torch.min(p_A * p_B, p_a * p_b).clamp(min=_EPS)  # (B,)

    # D' grid: (G,), expand to (G, B)
    d_grid_2d = d_grid.unsqueeze(1).expand(n_grid, B)
    D_max = torch.where(d_grid_2d >= 0, D_max_pos.unsqueeze(0), D_max_neg.unsqueeze(0))
    D = d_grid_2d * D_max  # (G, B)

    # Haplotype frequencies from (p_A, p_B, D)
    p_AB = (p_A.unsqueeze(0) * p_B.unsqueeze(0) + D).clamp(min=_EPS)  # (G, B)
    p_Ab = (p_A.unsqueeze(0) * p_b.unsqueeze(0) - D).clamp(min=_EPS)
    p_aB = (p_a.unsqueeze(0) * p_B.unsqueeze(0) - D).clamp(min=_EPS)
    p_ab = (p_a.unsqueeze(0) * p_b.unsqueeze(0) + D).clamp(min=_EPS)

    # Genotype probabilities under HWE: P(gi, gj) = sum over haplotype configs
    # For biallelic loci, 9 genotype classes (gi in {0,1,2}, gj in {0,1,2})
    # P(gi=a, gj=c) depends on haplotype frequencies.
    #
    # gi=0,gj=0: (p_ab)^2
    # gi=0,gj=1: 2*p_ab*p_aB
    # gi=0,gj=2: (p_aB)^2
    # gi=1,gj=0: 2*p_ab*p_Ab
    # gi=1,gj=1: 2*(p_AB*p_ab + p_Ab*p_aB)
    # gi=1,gj=2: 2*p_AB*p_aB      (corrected: 2*p_aB*p_AB... no)
    # Actually the proper derivation:
    # P(gi, gj) = sum_{h1,h2} P(haplotype pair h1,h2) where
    # h1 = (allele_locus1, allele_locus2), diploid = two haplotypes
    #
    # Simpler: use the standard result for two-locus HWE genotype probabilities.

    geno_probs = _genotype_probs_hwe(p_AB, p_Ab, p_aB, p_ab)  # (G, B, 3, 3)

    # Log-likelihood
    geno_probs = geno_probs.clamp(min=_EPS)
    log_probs = geno_probs.log()  # (G, B, 3, 3)
    counts_expanded = counts.unsqueeze(0)  # (1, B, 3, 3)
    loglik = (counts_expanded * log_probs).sum(dim=(2, 3))  # (G, B)

    # MLE
    max_loglik, max_idx = loglik.max(dim=0)  # (B,)
    dprime_hat = d_grid[max_idx]

    # 95% CI: region where loglik >= max_loglik - chi2_half
    in_ci = loglik >= (max_loglik.unsqueeze(0) - _CHI2_HALF)  # (G, B)

    # Find lower and upper bounds
    # For each pair, find first and last grid index in CI
    ci_low_idx = _first_true_index(in_ci, dim=0)  # (B,)
    ci_high_idx = _last_true_index(in_ci, dim=0)  # (B,)

    ci_low = d_grid[ci_low_idx]
    ci_high = d_grid[ci_high_idx]

    return dprime_hat, ci_low, ci_high


def _genotype_probs_hwe(
    p_AB: Tensor, p_Ab: Tensor, p_aB: Tensor, p_ab: Tensor,
) -> Tensor:
    """Compute 3x3 genotype probabilities under two-locus HWE.

    Under HWE, a diploid individual draws two haplotypes independently.
    P(gi=a, gj=c) = sum over all ways to get a alleles at locus 1
    and c alleles at locus 2 from two independent haplotype draws.

    Parameters
    ----------
    p_AB, p_Ab, p_aB, p_ab : (G, B) or broadcastable

    Returns
    -------
    probs : (G, B, 3, 3)
    """
    # Haplotype pair probabilities (ordered pairs, then symmetrized)
    # A single haplotype has 4 types: AB, Ab, aB, ab
    # Two haplotypes (diploid): 4x4 = 16 ordered pairs
    # Ordered pair (h1, h2) probability = p_h1 * p_h2
    #
    # Genotype at locus 1: count of A alleles = (A in h1) + (A in h2)
    # Genotype at locus 2: count of B alleles = (B in h1) + (B in h2)
    #
    # Build the mapping:
    #   AB: locus1=A(1), locus2=B(1)
    #   Ab: locus1=A(1), locus2=b(0)
    #   aB: locus1=a(0), locus2=B(1)
    #   ab: locus1=a(0), locus2=b(0)

    shape = p_AB.shape  # (G, B) or similar
    probs = torch.zeros(*shape, 3, 3, dtype=p_AB.dtype, device=p_AB.device)

    # Map: haplotype alleles (a1, a2) where a1=allele at locus1, a2=allele at locus2
    # AB=(1,1), Ab=(1,0), aB=(0,1), ab=(0,0)
    hap = [p_AB, p_Ab, p_aB, p_ab]
    a1 = [1, 1, 0, 0]  # A allele at locus 1
    a2 = [1, 0, 1, 0]  # B allele at locus 2

    for i in range(4):
        for j in range(4):
            gi = a1[i] + a1[j]  # genotype at locus 1 (0, 1, or 2)
            gj = a2[i] + a2[j]  # genotype at locus 2 (0, 1, or 2)
            probs[..., gi, gj] = probs[..., gi, gj] + hap[i] * hap[j]

    return probs


def _first_true_index(mask: Tensor, dim: int) -> Tensor:
    """Find the first True index along a dimension."""
    # Create index tensor
    indices = torch.arange(mask.shape[dim], device=mask.device)
    shape = [1] * mask.ndim
    shape[dim] = -1
    indices = indices.view(shape).expand_as(mask)

    # Set False positions to a large value
    large = mask.shape[dim]
    result = torch.where(mask, indices, torch.full_like(indices, large))
    return result.min(dim=dim).values.clamp(max=mask.shape[dim] - 1)


def _last_true_index(mask: Tensor, dim: int) -> Tensor:
    """Find the last True index along a dimension."""
    indices = torch.arange(mask.shape[dim], device=mask.device)
    shape = [1] * mask.ndim
    shape[dim] = -1
    indices = indices.view(shape).expand_as(mask)

    result = torch.where(mask, indices, torch.full_like(indices, -1))
    return result.max(dim=dim).values.clamp(min=0)
