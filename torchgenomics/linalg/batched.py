"""Batched Cholesky and solve for (*, d, d) blocks (mvLMM)."""

from __future__ import annotations

import torch
from torch import Tensor


def batched_cholesky(A: Tensor) -> Tensor:
    """Cholesky factor batched (*, d, d) SPD matrices.

    Parameters
    ----------
    A : Tensor, shape (*, d, d)
        Batch of symmetric positive definite matrices.

    Returns
    -------
    Tensor, shape (*, d, d)
        Lower-triangular Cholesky factors.
    """
    return torch.linalg.cholesky(A)


def batched_cholesky_solve(L: Tensor, B: Tensor) -> Tensor:
    """Solve L L^T X = B for batched (*, d, d) systems.

    Parameters
    ----------
    L : Tensor, shape (*, d, d)
        Lower-triangular Cholesky factors.
    B : Tensor, shape (*, d, k)
        Right-hand side matrices.

    Returns
    -------
    Tensor, shape (*, d, k)
        Solution matrices X.
    """
    return torch.cholesky_solve(B, L)
