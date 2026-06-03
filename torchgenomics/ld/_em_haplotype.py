"""EM algorithm for haplotype frequency estimation from unphased genotypes.

Given a 3x3 genotype count table for two biallelic loci, estimates the
four haplotype frequencies (p_AB, p_Ab, p_aB, p_ab) using the Hill (1974)
EM algorithm.  Batched over B SNP pairs for GPU efficiency.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-10


def build_genotype_counts(G: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
    """Build 3x3 genotype count tables for batched SNP pairs.

    Parameters
    ----------
    G : (n, m) float
        Genotype dosage matrix.  Values are rounded to {0, 1, 2}.
    idx_i, idx_j : (B,) long
        Column indices for each pair.

    Returns
    -------
    counts : (B, 3, 3) float64
        counts[b, a, c] = number of individuals with genotype a at locus i
        and genotype c at locus j  (a, c in {0, 1, 2}).
    """
    B = idx_i.shape[0]
    n = G.shape[0]
    device = G.device

    gi = G[:, idx_i].round().long().clamp(0, 2)  # (n, B)
    gj = G[:, idx_j].round().long().clamp(0, 2)  # (n, B)

    # Encode each (gi, gj) pair as a single index in [0, 8]
    flat = gi * 3 + gj  # (n, B)

    counts = torch.zeros(B, 9, dtype=torch.float64, device=device)
    for k in range(9):
        counts[:, k] = (flat == k).sum(dim=0).to(torch.float64)

    return counts.view(B, 3, 3)


def em_haplotype_freq(
    counts: Tensor,
    *,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> Tensor:
    """Estimate haplotype frequencies via EM (Hill 1974).

    Parameters
    ----------
    counts : (B, 3, 3) float64
        Genotype count tables.  counts[b, a, c] is the count of individuals
        with genotype a at locus 1 and c at locus 2.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on haplotype frequency change.

    Returns
    -------
    hapfreq : (B, 4) float64
        Haplotype frequencies [p_AB, p_Ab, p_aB, p_ab].
    """
    B = counts.shape[0]
    device = counts.device

    # Total samples per pair
    N = counts.sum(dim=(1, 2))  # (B,)
    N = N.clamp(min=1.0)

    # Marginal allele frequencies from genotype counts
    # Locus 1: freq(A) = (2*n_2. + n_1.) / (2*N)
    n_locus1 = counts.sum(dim=2)  # (B, 3)  sum over locus 2
    p_A = (2 * n_locus1[:, 2] + n_locus1[:, 1]) / (2 * N)

    n_locus2 = counts.sum(dim=1)  # (B, 3)  sum over locus 1
    p_B = (2 * n_locus2[:, 2] + n_locus2[:, 1]) / (2 * N)

    p_A = p_A.clamp(_EPS, 1 - _EPS)
    p_B = p_B.clamp(_EPS, 1 - _EPS)

    # Initialize under linkage equilibrium
    p_AB = p_A * p_B  # (B,)

    # Double heterozygote count: genotype (1, 1)
    n_11 = counts[:, 1, 1]  # (B,)

    for _ in range(max_iter):
        p_ab = 1.0 - p_A - p_B + p_AB
        p_Ab = p_A - p_AB
        p_aB = p_B - p_AB

        # Clamp to valid range
        p_ab = p_ab.clamp(min=_EPS)
        p_Ab = p_Ab.clamp(min=_EPS)
        p_aB = p_aB.clamp(min=_EPS)

        # E-step: split double heterozygotes
        coupling = p_AB * p_ab
        repulsion = p_Ab * p_aB
        denom = coupling + repulsion
        denom = denom.clamp(min=_EPS)
        frac_coupling = coupling / denom  # (B,)

        # Number of AB haplotypes contributed by double-hets
        n_AB_from_het = n_11 * frac_coupling

        # M-step: count AB haplotypes from all genotype classes
        # Genotype (2,2) contributes 2 AB, (2,1) contributes 1 AB + 1 Ab, etc.
        n_AB = (
            2 * counts[:, 2, 2]
            + counts[:, 2, 1]
            + counts[:, 1, 2]
            + n_AB_from_het
        )
        p_AB_new = n_AB / (2 * N)
        upper = torch.min(p_A, p_B).clamp(min=_EPS)
        p_AB_new = torch.clamp(p_AB_new, min=_EPS)
        p_AB_new = torch.min(p_AB_new, upper)

        # Check convergence
        change = (p_AB_new - p_AB).abs().max()
        p_AB = p_AB_new
        if change < tol:
            break

    # Final haplotype frequencies
    p_ab = (1.0 - p_A - p_B + p_AB).clamp(min=_EPS)
    p_Ab = (p_A - p_AB).clamp(min=_EPS)
    p_aB = (p_B - p_AB).clamp(min=_EPS)

    # Normalize to sum to 1
    total = p_AB + p_Ab + p_aB + p_ab
    hapfreq = torch.stack([p_AB, p_Ab, p_aB, p_ab], dim=1) / total.unsqueeze(1)

    return hapfreq
