"""Genotype centering, variance-scaling, and allele frequency (ploidy-aware)."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_allele_frequencies(G: Tensor, ploidy: int = 2) -> Tensor:
    """Compute per-SNP allele frequencies: mean(dosage) / ploidy.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix. NaN values are ignored.
    ploidy : int
        Organism ploidy level (2 for diploid, 4 for tetraploid, etc.).

    Returns
    -------
    Tensor, shape (m,)
        Per-SNP allele frequencies in [0, 1].
    """
    # Replace NaN with 0 for sum, count non-NaN
    mask = ~torch.isnan(G)
    G_zero = torch.where(mask, G, torch.zeros_like(G))
    n_obs = mask.sum(dim=0).to(G.dtype)  # (m,)

    # All-missing columns → NaN (undefined AF); otherwise compute normally
    safe_n = torch.clamp(n_obs, min=1.0)
    af = G_zero.sum(dim=0) / (safe_n * ploidy)
    af = torch.where(n_obs > 0, af, torch.full_like(af, float("nan")))

    return af


def compute_maf(af: Tensor) -> Tensor:
    """Minor allele frequency: min(af, 1 - af).

    Parameters
    ----------
    af : Tensor, shape (m,)
        Allele frequencies.

    Returns
    -------
    Tensor, shape (m,)
        Minor allele frequencies in [0, 0.5].
    """
    return torch.min(af, 1.0 - af)


def center_genotypes(G: Tensor, ploidy: int = 2) -> Tensor:
    """Subtract per-SNP mean from dosage matrix.

    NaN values remain NaN after centering. This matches GEMMA's
    preprocessing: center by 2*af (for diploid) or k*af (for polyploid).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (may contain NaN).
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    Tensor, shape (n, m)
        Centered dosage matrix.
    """
    af = compute_allele_frequencies(G, ploidy=ploidy)
    means = af * ploidy  # (m,) per-SNP mean dosage

    G_centered = G - means.unsqueeze(0)
    return G_centered


def scale_genotypes(G: Tensor, ploidy: int = 2) -> Tensor:
    """Center and divide by per-SNP std (ploidy-aware: expected variance under HWE).

    Under Hardy-Weinberg, the variance of dosage for ploidy k is:
        Var = k * af * (1 - af)

    This gives unit-variance columns after centering. Matches GEMMA's
    standardization convention for diploid (k=2).

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (may contain NaN).
    ploidy : int
        Organism ploidy level.

    Returns
    -------
    Tensor, shape (n, m)
        Centered and variance-scaled dosage matrix.
    """
    af = compute_allele_frequencies(G, ploidy=ploidy)

    # Expected variance under HWE
    var = ploidy * af * (1.0 - af)
    std = torch.sqrt(torch.clamp(var, min=1e-10))  # avoid div by zero for monomorphic

    means = af * ploidy
    G_scaled = (G - means.unsqueeze(0)) / std.unsqueeze(0)

    return G_scaled
