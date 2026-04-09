"""Low-rank update helpers for the Woodbury identity."""

from __future__ import annotations

import torch
from torch import Tensor


def woodbury_inverse(A_inv: Tensor, U: Tensor, C: Tensor, V: Tensor) -> Tensor:
    """(A + UCV)^{-1} via the Woodbury identity.

    (A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    Used in LMM for efficient rank-1 updates when adding a SNP to the model.

    Parameters
    ----------
    A_inv : Tensor, shape (n, n)
        Inverse of the base matrix A.
    U : Tensor, shape (n, k)
        Left factor of the low-rank update.
    C : Tensor, shape (k, k)
        Core matrix of the update.
    V : Tensor, shape (k, n)
        Right factor of the low-rank update.

    Returns
    -------
    Tensor, shape (n, n)
        (A + UCV)^{-1}
    """
    # A^{-1} U
    AiU = A_inv @ U  # (n, k)

    # C^{-1} + V A^{-1} U
    C_inv = torch.linalg.inv(C)  # (k, k)
    inner = C_inv + V @ AiU  # (k, k)

    # (C^{-1} + V A^{-1} U)^{-1}
    inner_inv = torch.linalg.inv(inner)  # (k, k)

    # V A^{-1}
    VAi = V @ A_inv  # (k, n)

    # Result: A^{-1} - A^{-1} U inner_inv V A^{-1}
    return A_inv - AiU @ inner_inv @ VAi


def woodbury_logdet(A_logdet: Tensor, A_inv: Tensor, U: Tensor, C: Tensor, V: Tensor) -> Tensor:
    """log|A + UCV| via the matrix determinant lemma.

    log|A + UCV| = log|A| + log|C| + log|C^{-1} + V A^{-1} U|

    Parameters
    ----------
    A_logdet : Tensor (scalar)
        log|A|.
    A_inv : Tensor, shape (n, n)
        A^{-1}.
    U, C, V : Tensor
        Low-rank factors.

    Returns
    -------
    Tensor (scalar)
        log|A + UCV|.
    """
    AiU = A_inv @ U
    C_inv = torch.linalg.inv(C)
    inner = C_inv + V @ AiU

    return A_logdet + torch.linalg.slogdet(C)[1] + torch.linalg.slogdet(inner)[1]
