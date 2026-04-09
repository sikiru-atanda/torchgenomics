"""SQUAREM: Squared iterative method for accelerating fixed-point iterations.

Varadhan, R. & Roland, C. (2008). *Scandinavian Journal of Statistics* 35, 335-353.

Accelerates any monotone EM-type iteration by extrapolating along the
secant direction with a step length chosen to preserve monotonicity.
Typical speed-up: converges in ~1/3 of the unaccelerated iterations.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def squarem(
    fixed_point_fn: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    step_min: float = 1.0,
    step_max: float = 1.0,
    obj_fn: Optional[Callable[[Tensor], float]] = None,
) -> tuple[Tensor, int, bool]:
    """SQUAREM acceleration of a fixed-point mapping.

    Parameters
    ----------
    fixed_point_fn : callable
        One step of the base iteration: ``x_{k+1} = F(x_k)``.
        Must accept and return a 1-D ``Tensor`` of the same shape.
    x0 : Tensor
        Initial parameter vector (1-D).
    max_iter : int
        Maximum number of SQUAREM iterations (each calls ``fixed_point_fn``
        twice, so total base-iteration cost is ≤ 2 * max_iter).
    tol : float
        Convergence criterion on ``||x_{k+1} - x_k|| / (||x_k|| + 1e-15)``.
    step_min, step_max : float
        Initial bounds on the step length α.  Adapted automatically.
    obj_fn : callable, optional
        Objective to maximise (e.g. log-likelihood).  When supplied, the
        algorithm backtracks the step length if the objective decreases,
        guaranteeing monotone ascent.

    Returns
    -------
    x : Tensor
        Converged parameter vector.
    n_iter : int
        Number of SQUAREM iterations performed.
    converged : bool
        Whether the tolerance was met within ``max_iter``.
    """
    x = x0.clone()
    s_min = step_min
    s_max = step_max

    if obj_fn is not None:
        obj_prev = obj_fn(x)

    converged = False
    n_iter = 0

    for k in range(max_iter):
        # Two base-iteration steps
        x1 = fixed_point_fn(x)
        x2 = fixed_point_fn(x1)

        # Secant quantities
        r = x1 - x       # first-order increment
        v = x2 - x1 - r  # second-order difference

        r_norm2 = (r * r).sum()
        v_norm2 = (v * v).sum()

        if v_norm2 < 1e-30:
            # Second-order difference vanishes — accept x2 as-is
            x = x2
            n_iter = k + 1
            rel = r_norm2.sqrt() / (x.norm() + 1e-15)
            if rel < tol:
                converged = True
                break
            continue

        # Step length (negative ratio ≈ eigenvalue estimate)
        alpha = -(r_norm2 / v_norm2).clamp(min=s_min, max=s_max)

        # Extrapolated candidate
        x_new = x - 2.0 * alpha * r + alpha * alpha * v

        # Stabilise with one base-iteration step
        x_new = fixed_point_fn(x_new)

        # Monotonicity check (if objective provided)
        if obj_fn is not None:
            obj_new = obj_fn(x_new)
            # Backtrack if objective decreased
            if obj_new < obj_prev:
                x_new = x2
                obj_new = obj_fn(x_new)
            obj_prev = obj_new

        # Convergence check
        rel = (x_new - x).norm() / (x.norm() + 1e-15)
        x = x_new
        n_iter = k + 1

        # Adaptive step bounds
        if alpha.abs() < s_min.abs() if isinstance(s_min, Tensor) else alpha.abs().item() < abs(s_min):
            s_min = alpha.item() if isinstance(alpha, Tensor) else alpha
        if alpha.abs() > s_max.abs() if isinstance(s_max, Tensor) else alpha.abs().item() > abs(s_max):
            s_max = alpha.item() if isinstance(alpha, Tensor) else alpha

        if rel < tol:
            converged = True
            break

    return x, n_iter, converged
