"""Eigendecomposition: exact eigh, truncation, rotation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class EigenDecomp:
    """Eigendecomposition result: K = U diag(evals) U^T."""

    eigenvalues: Tensor  # (n,) or (k,) sorted descending
    eigenvectors: Tensor  # (n, n) or (n, k) truncated


def eigendecompose(
    K: Tensor,
    *,
    n_components: int | None = None,
    eigenvalue_floor: float = 0.0,
) -> EigenDecomp:
    """Eigendecompose K = U diag(evals) U^T with negative-eigenvalue clamping.

    Uses ``torch.linalg.eigh`` (ascending order) then reverses to descending.

    Parameters
    ----------
    K : Tensor, shape (n, n)
        Symmetric positive semi-definite matrix (GRM).
    n_components : int, optional
        If set, retain only the top *n_components* eigenvalues/vectors.
        Use for truncated approximations.
    eigenvalue_floor : float
        Eigenvalues below this threshold are clamped to this value.
        Prevents numerical issues from near-zero or slightly negative
        eigenvalues (which arise from finite-precision GRM computation).

    Returns
    -------
    EigenDecomp
        Eigenvalues (descending) and corresponding eigenvectors.
    """
    K = K.to(torch.float64)

    # Symmetrize (numerical safety)
    K = (K + K.T) / 2.0

    # Eigendecompose (eigh returns ascending order)
    evals, evecs = torch.linalg.eigh(K)

    # Reverse to descending order
    evals = evals.flip(0)
    evecs = evecs.flip(1)

    # Clamp small/negative eigenvalues
    n_clamped = (evals < eigenvalue_floor).sum().item()
    if n_clamped > 0:
        logger.info(
            "Clamped %d eigenvalues below %.1e to floor.",
            n_clamped, eigenvalue_floor,
        )
    evals = torch.clamp(evals, min=eigenvalue_floor)

    # Optional truncation
    if n_components is not None and n_components < len(evals):
        evals = evals[:n_components]
        evecs = evecs[:, :n_components]

    return EigenDecomp(eigenvalues=evals, eigenvectors=evecs)


def auto_n_components(
    eigenvalues: Tensor,
    variance_explained: float = 0.999,
) -> int:
    """Select number of eigencomponents to capture target variance.

    Parameters
    ----------
    eigenvalues : (n,) sorted descending
    variance_explained : fraction of total variance to retain (default 0.999)

    Returns
    -------
    k : number of components needed
    """
    total = eigenvalues.sum().item()
    if total <= 0:
        return len(eigenvalues)

    cumsum = eigenvalues.cumsum(dim=0)
    target = variance_explained * total
    k = int((cumsum >= target).float().argmax().item()) + 1
    return min(k, len(eigenvalues))


def rotate(M: Tensor, U: Tensor) -> Tensor:
    """Rotate matrix M into eigenspace: M_rot = U^T @ M.

    Parameters
    ----------
    M : Tensor, shape (n, ...)
        Matrix to rotate (Y, X0, or G_chunk).
    U : Tensor, shape (n, n) or (n, k)
        Eigenvectors from eigendecompose.

    Returns
    -------
    Tensor, shape (n, ...) or (k, ...)
        Rotated matrix.
    """
    return U.T @ M


def compute_weights(
    eigenvalues: Tensor,
    sig2_g: float,
    sig2_e: float,
) -> Tensor:
    """Compute per-individual weights in rotated space.

    In the rotated eigenspace, the covariance diagonal is:
        w_i = 1 / (evals_i * sig2_g + sig2_e)

    These weights are used for weighted least squares in the per-SNP scan.

    Parameters
    ----------
    eigenvalues : Tensor, shape (n,)
        Eigenvalues from GRM decomposition.
    sig2_g : float
        Genetic variance component.
    sig2_e : float
        Residual variance component.

    Returns
    -------
    Tensor, shape (n,)
        Per-individual weights (inverse variance diagonal).
    """
    V_diag = eigenvalues * sig2_g + sig2_e
    return 1.0 / V_diag
