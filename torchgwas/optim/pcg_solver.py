"""Mode E: Preconditioned Conjugate Gradient solver for large-scale REML.

When direct Cholesky solves become too expensive (n > 100K with multiple
random effects), the linear system inside each REML iteration can be solved
iteratively via PCG.  This follows the BLUPF90 scaling strategy (Tsuruta
et al., 2001) adapted for PyTorch tensors.

The solver is matrix-free: the coefficient matrix A is provided as a
callable ``A_matvec(x) -> Ax`` so it can be applied implicitly via
matrix-vector products without forming the full matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PCGResult:
    """Result of a PCG solve."""

    x: Tensor          # Solution vector
    n_iters: int       # Iterations used
    converged: bool    # Whether residual norm < tolerance
    residual_norm: float  # Final ||r|| / ||b||


def pcg_solve(
    A_matvec: Callable[[Tensor], Tensor],
    b: Tensor,
    *,
    x0: Optional[Tensor] = None,
    precond: Optional[Callable[[Tensor], Tensor]] = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> PCGResult:
    """Preconditioned Conjugate Gradient solver for Ax = b.

    Parameters
    ----------
    A_matvec : callable
        Matrix-vector product function: A_matvec(x) returns A @ x.
        A must be symmetric positive definite.
    b : Tensor
        Right-hand side vector.
    x0 : Tensor, optional
        Initial guess. Defaults to zeros.
    precond : callable, optional
        Preconditioner function: precond(r) returns M^{-1} @ r where M
        approximates A.  If None, uses identity (unpreconditioned CG).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance: stop when ||r|| / ||b|| < tol.

    Returns
    -------
    PCGResult
        Solution, iteration count, convergence flag, and final residual norm.
    """
    b_norm = b.norm()
    if b_norm < 1e-30:
        return PCGResult(
            x=torch.zeros_like(b),
            n_iters=0,
            converged=True,
            residual_norm=0.0,
        )

    # Initial guess
    x = x0.clone() if x0 is not None else torch.zeros_like(b)

    # Initial residual
    r = b - A_matvec(x) if x0 is not None else b.clone()

    # Apply preconditioner
    z = precond(r) if precond is not None else r.clone()
    p = z.clone()
    rz = (r * z).sum()

    converged = False
    for i in range(max_iter):
        Ap = A_matvec(p)
        pAp = (p * Ap).sum()

        # Guard against near-zero denominator (shouldn't happen for SPD A)
        if pAp.abs() < 1e-30:
            logger.warning("PCG: pAp near zero at iteration %d", i)
            break

        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        r_norm = r.norm() / b_norm
        if r_norm < tol:
            converged = True
            break

        z_new = precond(r) if precond is not None else r.clone()
        rz_new = (r * z_new).sum()

        beta = rz_new / rz
        p = z_new + beta * p
        rz = rz_new

    n_iters = i + 1 if not converged or i > 0 else 0
    final_norm = float(r.norm() / b_norm)

    if not converged:
        logger.warning(
            "PCG did not converge in %d iterations (residual=%.2e, tol=%.2e)",
            max_iter, final_norm, tol,
        )

    return PCGResult(x=x, n_iters=n_iters, converged=converged, residual_norm=final_norm)


def diagonal_preconditioner(A_diag: Tensor) -> Callable[[Tensor], Tensor]:
    """Create a diagonal (Jacobi) preconditioner from the diagonal of A.

    Parameters
    ----------
    A_diag : Tensor
        Diagonal elements of the coefficient matrix A.

    Returns
    -------
    callable
        Function that applies M^{-1} @ r where M = diag(A).
    """
    inv_diag = 1.0 / torch.clamp(A_diag.abs(), min=1e-15)

    def _apply(r: Tensor) -> Tensor:
        return inv_diag * r

    return _apply
