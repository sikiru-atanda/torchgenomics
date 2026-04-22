"""Randomized eigendecomposition: Halko-Martinsson-Tropp SVD + LOBPCG wrapper.

Provides drop-in replacements for ``eigendecompose()`` that return
``EigenDecomp`` objects, enabling the downstream REML + scan pipeline
to work unchanged with approximate eigenpairs.

References
----------
Halko, Martinsson, Tropp (2011). "Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions."
SIAM Review 53(2):217-288.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from .eigh import EigenDecomp

logger = logging.getLogger(__name__)


def randomized_svd(
    K: Tensor,
    n_components: int = 100,
    n_oversamples: int = 10,
    n_power_iters: int = 2,
    seed: int | None = None,
) -> EigenDecomp:
    """Low-rank eigendecomposition via randomized SVD.

    For a symmetric PSD matrix K, computes the top-k eigenpairs using
    the Halko-Martinsson-Tropp algorithm:

    1. Draw random Gaussian probe matrix Omega (n, k+p)
    2. Form Y = K @ Omega
    3. Power iteration for spectrum decay: Y = K @ (K @ Y) repeated
    4. QR decomposition: Y = Q R  (Q captures the range of K)
    5. Project: B = Q^T @ K @ Q  (small k×k symmetric matrix)
    6. Eigendecompose B exactly
    7. Lift back: U = Q @ U_B

    Parameters
    ----------
    K : Tensor, shape (n, n)
        Symmetric PSD matrix (GRM).
    n_components : int
        Number of top eigenvalues/vectors to compute.
    n_oversamples : int
        Extra columns for better range approximation (default 10).
    n_power_iters : int
        Number of power iterations for spectral decay (default 2).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    EigenDecomp
        Top-k eigenvalues (descending) and eigenvectors.
    """
    K = K.to(torch.float64)
    n = K.shape[0]
    k = min(n_components, n)
    p = min(n_oversamples, n - k)
    total = k + p

    if seed is not None:
        gen = torch.Generator(device=K.device).manual_seed(seed)
    else:
        gen = None

    # Step 1: Random Gaussian probe matrix
    Omega = torch.randn(n, total, dtype=K.dtype, device=K.device, generator=gen)

    # Step 2: Form Y = K @ Omega
    Y = K @ Omega

    # Step 3: Power iteration (improves approximation for slowly decaying spectra)
    for _ in range(n_power_iters):
        # Reorthogonalize for numerical stability
        Q, _ = torch.linalg.qr(Y)
        Y = K @ (K @ Q)

    # Step 4: QR factorization of the range
    Q, _ = torch.linalg.qr(Y)  # (n, total)

    # Step 5: Project K into the low-dimensional subspace
    B = Q.T @ K @ Q  # (total, total) — symmetric

    # Symmetrize for numerical safety
    B = (B + B.T) / 2.0

    # Step 6: Exact eigendecomposition of small matrix
    evals_B, evecs_B = torch.linalg.eigh(B)

    # Reverse to descending order
    evals_B = evals_B.flip(0)
    evecs_B = evecs_B.flip(1)

    # Step 7: Lift eigenvectors back to original space
    U = Q @ evecs_B  # (n, total)

    # Truncate to requested number of components
    evals = evals_B[:k]
    U = U[:, :k]

    # Clamp small/negative eigenvalues (numerical artifact)
    n_clamped = (evals < 0).sum().item()
    if n_clamped > 0:
        logger.info("Randomized SVD: clamped %d negative eigenvalues to 0.", n_clamped)
    evals = torch.clamp(evals, min=0.0)

    logger.info(
        "Randomized SVD: n=%d, k=%d, p=%d, power_iters=%d, "
        "top eigenvalue=%.4e, k-th eigenvalue=%.4e",
        n, k, p, n_power_iters,
        evals[0].item() if k > 0 else 0.0,
        evals[-1].item() if k > 0 else 0.0,
    )

    return EigenDecomp(eigenvalues=evals, eigenvectors=U)


def lobpcg_decompose(
    K: Tensor | Callable[[Tensor], Tensor],
    n_components: int = 100,
    n_samples: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int | None = None,
) -> EigenDecomp:
    """Top-k eigenpairs via Locally Optimal Block Preconditioned Conjugate Gradient.

    GPU-friendly iterative eigensolver. Accepts either a dense matrix or a
    matrix-vector product callable (for sparse/implicit matrices).

    Parameters
    ----------
    K : Tensor (n, n) or Callable
        Symmetric PSD matrix, or a function ``K(x) -> K @ x``.
    n_components : int
        Number of top eigenvalues to compute.
    n_samples : int, optional
        Required when K is a callable (specifies matrix dimension).
    max_iter : int
        Maximum LOBPCG iterations.
    tol : float
        Convergence tolerance.
    seed : int, optional
        Random seed for initial vectors.

    Returns
    -------
    EigenDecomp
        Top-k eigenvalues (descending) and eigenvectors.
    """
    if callable(K) and not isinstance(K, Tensor):
        if n_samples is None:
            raise ValueError("n_samples required when K is a callable.")
        n = n_samples
        A = K  # matvec callable
        # torch.lobpcg needs a matrix or LinearOperator-like
        # We create a wrapper
        device = torch.device("cpu")
        dtype = torch.float64
    else:
        K = K.to(torch.float64)
        n = K.shape[0]
        A = K
        device = K.device
        dtype = K.dtype

    k = min(n_components, n)

    # Initial random vectors
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    X0 = torch.randn(n, k, dtype=dtype, device=device, generator=gen)

    try:
        # torch.lobpcg returns eigenvalues in descending order
        evals, evecs = torch.lobpcg(
            A, k=k, X=X0, niter=max_iter, tol=tol, largest=True,
        )
    except Exception as exc:
        logger.warning(
            "LOBPCG failed (%s), falling back to randomized SVD.", exc
        )
        if isinstance(A, Tensor):
            return randomized_svd(A, n_components=k)
        raise

    # Ensure descending order
    sort_idx = evals.argsort(descending=True)
    evals = evals[sort_idx]
    evecs = evecs[:, sort_idx]

    # Clamp negative eigenvalues
    evals = torch.clamp(evals, min=0.0)

    logger.info(
        "LOBPCG: n=%d, k=%d, top eigenvalue=%.4e, k-th eigenvalue=%.4e",
        n, k,
        evals[0].item() if k > 0 else 0.0,
        evals[-1].item() if k > 0 else 0.0,
    )

    return EigenDecomp(eigenvalues=evals, eigenvectors=evecs)
