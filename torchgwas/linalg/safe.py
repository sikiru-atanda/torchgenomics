"""Safe Cholesky with adaptive jitter, logdet, symmetrize."""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_JITTER_GROWTH = 10.0


def safe_cholesky(K: Tensor, jitter_factor: float = 1e-6) -> Tensor:
    """Cholesky with adaptive diagonal jitter.

    Starts with eps = jitter_factor * trace(K) / n. If Cholesky fails,
    multiplies eps by 10 and retries up to 5 times.

    Parameters
    ----------
    K : Tensor, shape (n, n)
        Symmetric positive semi-definite matrix.
    jitter_factor : float
        Initial jitter as fraction of mean diagonal.

    Returns
    -------
    Tensor, shape (n, n)
        Lower-triangular Cholesky factor L such that L @ L.T ≈ K.

    Raises
    ------
    RuntimeError
        If Cholesky fails after all retry attempts.
    """
    K = K.to(torch.float64)
    n = K.shape[0]

    # Symmetrize
    K = (K + K.T) / 2.0

    # Try without jitter first
    try:
        return torch.linalg.cholesky(K)
    except RuntimeError:
        pass

    # Adaptive jitter
    trace_val = torch.diagonal(K).sum()
    eps = jitter_factor * trace_val / n

    for attempt in range(_MAX_RETRIES):
        try:
            K_jittered = K + eps * torch.eye(n, dtype=K.dtype, device=K.device)
            L = torch.linalg.cholesky(K_jittered)
            logger.info(
                "Cholesky succeeded with jitter=%.2e (attempt %d).",
                eps, attempt + 1,
            )
            return L
        except RuntimeError:
            eps *= _JITTER_GROWTH

    raise RuntimeError(
        f"Cholesky decomposition failed after {_MAX_RETRIES} attempts "
        f"(final jitter={eps:.2e}). Matrix may not be positive semi-definite."
    )


def safe_logdet(K: Tensor, jitter_factor: float = 1e-6) -> Tensor:
    """Compute log-determinant via safe Cholesky: log|K| = 2 * sum(log(diag(L))).

    Parameters
    ----------
    K : Tensor, shape (n, n)
        SPD matrix.

    Returns
    -------
    Tensor (scalar)
        log|K|.
    """
    L = safe_cholesky(K, jitter_factor=jitter_factor)
    return 2.0 * torch.log(torch.diagonal(L)).sum()


def symmetrize(K: Tensor) -> Tensor:
    """Symmetrize a matrix: K -> (K + K^T) / 2."""
    return (K + K.T) / 2.0
