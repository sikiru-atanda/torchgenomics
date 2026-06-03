"""Nystrom GRM approximation: low-rank kinship without forming full n x n matrix.

Selects a random subset of landmark samples, builds the small landmark-landmark
GRM (W) and the cross-kernel matrix (C), then extracts approximate eigenpairs
from the Nystrom extension: K ≈ C @ W^{-1} @ C^T.

Memory usage: O(n * n_landmarks) instead of O(n^2).
Compatible with streaming chunk iterators from Phase 12.

References
----------
Williams & Seeger (2001). "Using the Nystrom Method to Speed Up Kernel Machines."
NIPS 2001.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from torch import Tensor

from ..preprocess.standardize import compute_allele_frequencies
from .eigh import EigenDecomp

logger = logging.getLogger(__name__)


def nystrom_approximate(
    chunk_iter: Iterator[tuple[Tensor, object]],
    n_samples: int,
    n_landmarks: int = 500,
    ploidy: int = 2,
    seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[EigenDecomp, float]:
    """Streaming Nystrom GRM approximation.

    Builds a low-rank approximation of the VanRaden GRM from streaming
    genotype chunks, never forming the full n x n matrix.

    Algorithm
    ---------
    1. Select ``n_landmarks`` random sample indices.
    2. Stream chunks: for each centered chunk X_c (n, m_chunk):
       - Accumulate W += X_c[landmarks] @ X_c[landmarks]^T  (l x l)
       - Accumulate C += X_c @ X_c[landmarks]^T              (n x l)
       - Accumulate normalizer += sum(k * p * (1-p))
    3. Normalize: W /= normalizer, C /= normalizer
    4. Eigendecompose small W: W = U_w diag(S_w) U_w^T
    5. Compute Nystrom eigenvectors: U = C @ U_w @ diag(1/sqrt(S_w))
    6. Return EigenDecomp(eigenvalues=S_w, eigenvectors=U)

    Parameters
    ----------
    chunk_iter : Iterator
        Yields (G_chunk, variant_meta) tuples. G_chunk is (n, m_chunk).
    n_samples : int
        Total number of samples.
    n_landmarks : int
        Number of landmark samples (default 500).
    ploidy : int
        Organism ploidy level.
    seed : int, optional
        Random seed for landmark selection.
    device : torch.device, optional
        Computation device.

    Returns
    -------
    (EigenDecomp, float)
        Low-rank eigenpairs and the GRM normalizer value.
    """
    if device is None:
        device = torch.device("cpu")

    dtype = torch.float64
    n_landmarks = min(n_landmarks, n_samples)

    # Step 1: Select random landmark indices
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    landmark_idx = torch.randperm(n_samples, generator=gen)[:n_landmarks].sort().values
    landmark_list = landmark_idx.tolist()

    # Step 2: Stream and accumulate W (l x l) and C (n x l)
    W = torch.zeros(n_landmarks, n_landmarks, dtype=dtype, device=device)
    C = torch.zeros(n_samples, n_landmarks, dtype=dtype, device=device)
    normalizer = torch.tensor(0.0, dtype=dtype, device=device)
    total_snps = 0

    for G_chunk, _ in chunk_iter:
        G_f64 = G_chunk.to(dtype=dtype, device=device)

        # Allele frequencies and centering
        af = compute_allele_frequencies(G_f64, ploidy=ploidy)
        means = af * ploidy
        X_c = G_f64 - means.unsqueeze(0)  # (n, m_chunk)

        # Landmark rows
        X_c_land = X_c[landmark_list]  # (l, m_chunk)

        # Accumulate: W += X_c_land @ X_c_land^T, C += X_c @ X_c_land^T
        W += X_c_land @ X_c_land.T
        C += X_c @ X_c_land.T

        # Normalizer
        normalizer += (ploidy * af * (1.0 - af)).sum()
        total_snps += G_chunk.shape[1]

    if normalizer < 1e-10:
        logger.warning("Nystrom: GRM normalizer near zero.")
        normalizer = torch.tensor(1.0, dtype=dtype, device=device)

    # Step 3: Normalize
    W /= normalizer
    C /= normalizer

    # Step 4: Eigendecompose the small landmark GRM
    W = (W + W.T) / 2.0  # symmetrize
    evals_w, evecs_w = torch.linalg.eigh(W)

    # Descending order
    evals_w = evals_w.flip(0)
    evecs_w = evecs_w.flip(1)

    # Filter out near-zero/negative eigenvalues
    keep_mask = evals_w > 1e-10
    n_keep = keep_mask.sum().item()
    if n_keep == 0:
        logger.warning("Nystrom: all landmark eigenvalues near zero.")
        return EigenDecomp(
            eigenvalues=torch.zeros(1, dtype=dtype, device=device),
            eigenvectors=torch.zeros(n_samples, 1, dtype=dtype, device=device),
        ), float(normalizer)

    evals_w = evals_w[:n_keep]
    evecs_w = evecs_w[:, :n_keep]

    # Step 5: Nystrom eigenvectors
    # U = C @ U_w @ diag(1/sqrt(S_w))  and then normalize columns
    inv_sqrt_evals = 1.0 / torch.sqrt(evals_w)
    U = C @ evecs_w @ torch.diag(inv_sqrt_evals)  # (n, n_keep)

    # Normalize eigenvectors to unit length
    norms = U.norm(dim=0, keepdim=True)
    norms = torch.clamp(norms, min=1e-10)
    U = U / norms

    logger.info(
        "Nystrom: n=%d, landmarks=%d, kept=%d, total_snps=%d, "
        "top eigenvalue=%.4e, smallest kept=%.4e",
        n_samples, n_landmarks, n_keep, total_snps,
        evals_w[0].item(), evals_w[-1].item(),
    )

    return EigenDecomp(eigenvalues=evals_w, eigenvectors=U), float(normalizer)
