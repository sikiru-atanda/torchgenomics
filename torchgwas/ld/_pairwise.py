"""Pairwise LD statistics: r-squared and D-prime.

Provides both phased (direct counting) and unphased (EM-based) paths
for computing D'.  r-squared is always computed from dosage correlation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._em_haplotype import build_genotype_counts, em_haplotype_freq

_EPS = 1e-10


# ── r-squared ───────────────────────────────────────────────────────

def compute_r2_matrix(G: Tensor) -> Tensor:
    """Compute pairwise r-squared matrix from genotype dosages.

    Uses Pearson correlation with N-1 denominator (matches R's cor()).
    Extracted from the pattern in blink.py and farmcpu.py.

    Parameters
    ----------
    G : (n, m) Tensor
        Genotype dosage matrix.

    Returns
    -------
    r2 : (m, m) float64
        Symmetric r-squared matrix.
    """
    G = G.to(torch.float64)
    G_c = G - G.mean(dim=0, keepdim=True)
    stds = G_c.std(dim=0, keepdim=True)
    stds = torch.clamp(stds, min=_EPS)
    G_norm = G_c / stds
    r = (G_norm.T @ G_norm) / (G_norm.shape[0] - 1)
    r = r.clamp(-1.0, 1.0)
    return r ** 2


def compute_r2_pairs(G: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
    """Compute r-squared for specific SNP pairs.

    Parameters
    ----------
    G : (n, m)
    idx_i, idx_j : (B,)

    Returns
    -------
    r2 : (B,) float64
    """
    G = G.to(torch.float64)
    gi = G[:, idx_i]  # (n, B)
    gj = G[:, idx_j]  # (n, B)

    gi_c = gi - gi.mean(dim=0, keepdim=True)
    gj_c = gj - gj.mean(dim=0, keepdim=True)

    cov = (gi_c * gj_c).sum(dim=0) / (G.shape[0] - 1)
    si = gi_c.std(dim=0).clamp(min=_EPS)
    sj = gj_c.std(dim=0).clamp(min=_EPS)

    r = cov / (si * sj)
    return r.clamp(-1.0, 1.0) ** 2


# ── D-prime ─────────────────────────────────────────────────────────

def compute_dprime_phased(
    haplotypes: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
) -> Tensor:
    """Compute D' from phased haplotype data (direct counting).

    Parameters
    ----------
    haplotypes : (n, ploidy, m) Tensor
        Binary haplotype matrix (0/1).
    idx_i, idx_j : (B,) long
        SNP pair indices.

    Returns
    -------
    dprime : (B,) float64
    """
    # Flatten to (n*ploidy, m)
    n, ploidy, m = haplotypes.shape
    H = haplotypes.reshape(n * ploidy, m).to(torch.float64)
    N = H.shape[0]

    hi = H[:, idx_i]  # (N, B)
    hj = H[:, idx_j]  # (N, B)

    p_A = hi.mean(dim=0)  # (B,)
    p_B = hj.mean(dim=0)  # (B,)
    p_AB = (hi * hj).mean(dim=0)  # (B,)

    D = p_AB - p_A * p_B  # (B,)

    return _normalize_d(D, p_A, p_B)


def compute_dprime_unphased(
    G: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    *,
    em_max_iter: int = 20,
) -> Tensor:
    """Compute D' from unphased genotype dosages via EM.

    Parameters
    ----------
    G : (n, m)
    idx_i, idx_j : (B,)
    em_max_iter : int

    Returns
    -------
    dprime : (B,) float64
    """
    counts = build_genotype_counts(G, idx_i, idx_j)  # (B, 3, 3)
    hapfreq = em_haplotype_freq(counts, max_iter=em_max_iter)  # (B, 4)

    p_AB = hapfreq[:, 0]
    p_Ab = hapfreq[:, 1]
    p_aB = hapfreq[:, 2]

    p_A = p_AB + p_Ab
    p_B = p_AB + p_aB
    D = p_AB - p_A * p_B

    return _normalize_d(D, p_A, p_B)


def _normalize_d(D: Tensor, p_A: Tensor, p_B: Tensor) -> Tensor:
    """Normalize D to D' = D / D_max."""
    p_a = 1.0 - p_A
    p_b = 1.0 - p_B

    D_max_pos = torch.min(p_A * p_b, p_a * p_B).clamp(min=_EPS)
    D_max_neg = torch.min(p_A * p_B, p_a * p_b).clamp(min=_EPS)

    dprime = torch.where(
        D >= 0,
        D / D_max_pos,
        D / D_max_neg,
    )
    return dprime.clamp(-1.0, 1.0)
