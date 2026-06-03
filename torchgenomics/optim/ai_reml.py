"""Mode B: Average Information REML (single-trait 2-param Newton-Raphson on lambda).

Matches GEMMA's approach: parameterize by lambda = sig_g^2 / sig_e^2,
profile out sig_e^2 analytically, and use Newton-Raphson with the Average
Information matrix as the Hessian approximation.

Quadratic convergence near optimum.  Damped updates with backtracking
line search to ensure monotonic likelihood increase.
"""

from __future__ import annotations

import logging

from torch import Tensor

from .reml_math import reml_derivatives, reml_loglikelihood

logger = logging.getLogger(__name__)


def ai_reml_single(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    lam_init: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    lam_min: float = 1e-10,
    lam_max: float = 1e10,
) -> tuple[float, float, float, list[dict]]:
    """AI-REML for single-trait LMM via Newton-Raphson on lambda.

    Parameters
    ----------
    Y_rot : (n,) or (n, 1) — rotated phenotype
    X0_rot : (n, c) — rotated covariates
    eigenvalues : (n,) — GRM eigenvalues
    lam_init : initial lambda (default: 1.0)
    max_iter : maximum iterations
    tol : relative convergence tolerance on log-likelihood
    lam_min, lam_max : bounds for lambda

    Returns
    -------
    sig2_g : float — genetic variance
    sig2_e : float — residual variance
    ll : float — final REML log-likelihood
    trace : list[dict] — optimizer trace for diagnostics
    """
    lam = lam_init if lam_init is not None else 1.0
    lam = max(lam_min, min(lam_max, lam))

    ll_prev = float("-inf")
    trace: list[dict] = []

    # Track best solution seen (guards against oscillation)
    best_lam = lam
    best_ll = float("-inf")

    for it in range(max_iter):
        ll, dl, d2l = reml_derivatives(lam, Y_rot, X0_rot, eigenvalues)

        # Track best
        if ll > best_ll:
            best_ll = ll
            best_lam = lam

        trace.append({
            "iter": it,
            "mode": "AI-REML",
            "lambda": lam,
            "ll": ll,
            "dl": dl,
            "d2l": d2l,
        })

        # Check convergence
        if it > 0 and abs(ll - ll_prev) < tol * max(abs(ll_prev), 1.0):
            logger.info(
                "AI-REML converged in %d iterations: lambda=%.6e, ll=%.6f",
                it, lam, ll,
            )
            break

        # Newton step: delta = -dl / d2l
        # AI convention: d2l ≈ -AI, so delta = dl / AI = -dl / d2l
        if abs(d2l) < 1e-20:
            # Hessian too flat — take a small gradient step
            delta = 0.1 * dl if dl > 0 else -0.1
        else:
            delta = -dl / d2l

        # Damped update with backtracking line search
        step_size = 1.0

        # Strict Armijo: require non-decrease (maximizing ll)
        for bt in range(15):
            lam_new = lam + step_size * delta
            lam_new = max(lam_min, min(lam_max, lam_new))

            ll_new, _, _ = reml_loglikelihood(lam_new, Y_rot, X0_rot, eigenvalues)

            if ll_new >= ll:  # strict non-decrease
                break
            step_size *= 0.5
        else:
            # Backtracking exhausted — try a small gradient ascent step
            grad_step = 0.01 * dl if abs(dl) > 1e-20 else 0.0
            lam_new = max(lam_min, min(lam_max, lam + grad_step))

        ll_prev = ll
        lam = lam_new
    else:
        logger.warning(
            "AI-REML did not converge in %d iterations (lambda=%.6e, ll=%.6f). "
            "Best: lambda=%.6e, ll=%.6f",
            max_iter, lam, ll, best_lam, best_ll,
        )
        # Use the best lambda seen during iterations
        lam = best_lam

    # Final evaluation at converged (or best) lambda
    ll_final, sig2_e, sig2_g = reml_loglikelihood(lam, Y_rot, X0_rot, eigenvalues)

    return sig2_g, sig2_e, ll_final, trace
