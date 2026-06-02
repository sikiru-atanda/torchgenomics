"""Sparse GRM: fastGWA-style threshold sparsification.

Builds a VanRaden GRM via streaming, then zeroes out entries below a
relatedness threshold. The result is stored as a sparse tensor, enabling
efficient matrix-vector products for PCG-REML without O(n^2) memory for
the full dense matrix.

References
----------
Jiang et al. (2019). "A resource-efficient tool for mixed model association
analysis of large-scale data." Nature Genetics 51:1749-1755. (fastGWA)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor

from ..preprocess.standardize import compute_allele_frequencies

logger = logging.getLogger(__name__)


def sparse_grm_streaming(
    chunk_iter: Iterator[tuple[Tensor, object]],
    n_samples: int,
    threshold: float = 0.05,
    ploidy: int = 2,
    device: torch.device | None = None,
) -> tuple[Tensor, float, int]:
    """Build a sparse GRM by streaming chunks and thresholding.

    Algorithm
    ---------
    1. Accumulate full VanRaden GRM chunk-by-chunk (same as
       ``grm_vanraden_streaming``).
    2. Zero out off-diagonal entries where |K[i,j]| < threshold.
    3. Convert to sparse COO tensor.

    Note: Step 1 still requires O(n^2) memory temporarily. For truly
    massive n where even the dense accumulation is infeasible, use
    ``nystrom_approximate`` instead. This function targets the regime
    where the dense GRM fits in memory but we want sparse storage for
    efficient PCG solves.

    Parameters
    ----------
    chunk_iter : Iterator
        Yields (G_chunk, variant_meta) tuples.
    n_samples : int
        Total number of samples.
    threshold : float
        Relatedness threshold; entries with |K[i,j]| < threshold are zeroed.
    ploidy : int
        Organism ploidy level.
    device : torch.device, optional
        Computation device.

    Returns
    -------
    (Tensor, float, int)
        Sparse COO GRM, normalizer value, and number of SNPs used.
    """
    if device is None:
        device = torch.device("cpu")

    dtype = torch.float64
    K_accum = torch.zeros(n_samples, n_samples, dtype=dtype, device=device)
    normalizer = torch.tensor(0.0, dtype=dtype, device=device)
    total_snps = 0

    for G_chunk, _ in chunk_iter:
        G_f64 = G_chunk.to(dtype=dtype, device=device)
        af = compute_allele_frequencies(G_f64, ploidy=ploidy)
        means = af * ploidy
        X_c = G_f64 - means.unsqueeze(0)

        K_accum += X_c @ X_c.T
        normalizer += (ploidy * af * (1.0 - af)).sum()
        total_snps += G_chunk.shape[1]

    if normalizer < 1e-10:
        logger.warning("Sparse GRM: normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=dtype, device=device)

    K_accum /= normalizer

    # Sparsify: keep diagonal + off-diagonal entries above threshold
    mask = torch.abs(K_accum) >= threshold
    # Always keep diagonal
    diag_idx = torch.arange(n_samples, device=device)
    mask[diag_idx, diag_idx] = True

    n_total = n_samples * n_samples
    n_kept = mask.sum().item()
    sparsity = 1.0 - n_kept / n_total

    # Zero out below-threshold entries and convert to sparse
    K_sparse = K_accum * mask.to(dtype)
    K_sparse_coo = K_sparse.to_sparse_coo()

    logger.info(
        "Sparse GRM: n=%d, threshold=%.3f, kept=%d/%d entries (%.1f%% sparse), "
        "snps=%d",
        n_samples, threshold, n_kept, n_total, sparsity * 100, total_snps,
    )

    return K_sparse_coo, float(normalizer), total_snps


def sparse_grm_matvec(K_sparse: Tensor, x: Tensor) -> Tensor:
    """Efficient sparse matrix-vector product: K_sparse @ x.

    Parameters
    ----------
    K_sparse : Tensor (sparse COO or CSR)
        Sparse GRM.
    x : Tensor, shape (n,) or (n, k)
        Vector(s) to multiply.

    Returns
    -------
    Tensor, same shape as x.
    """
    return torch.sparse.mm(K_sparse, x.unsqueeze(-1) if x.ndim == 1 else x).squeeze(-1) if x.ndim == 1 else torch.sparse.mm(K_sparse, x)


def make_sparse_matvec(
    K_sparse: Tensor,
    sig2_g: float,
    sig2_e: float,
) -> Callable[[Tensor], Tensor]:
    """Create a matvec callable for V = sig2_g * K + sig2_e * I.

    This is the covariance matrix matvec needed by PCG-REML:
        V @ x = sig2_g * (K @ x) + sig2_e * x

    Parameters
    ----------
    K_sparse : Tensor (sparse)
        Sparse GRM.
    sig2_g : float
        Genetic variance component.
    sig2_e : float
        Residual variance component.

    Returns
    -------
    Callable
        Function mapping x -> V @ x.
    """
    def _matvec(x: Tensor) -> Tensor:
        if x.ndim == 1:
            Kx = torch.sparse.mm(K_sparse, x.unsqueeze(1)).squeeze(1)
        else:
            Kx = torch.sparse.mm(K_sparse, x)
        return sig2_g * Kx + sig2_e * x

    return _matvec
