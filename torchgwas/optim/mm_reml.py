"""Mode D: EM-REML (Expectation-Maximization) algorithm for REML.

Monotone and globally convergent.  Updates (sig2_g, sig2_e) jointly
using the standard EM-REML updates in rotated eigenspace:

    sig2_g_new = sig2_g + sig2_g^2 * (y^T P K P y - tr(PK)) / df
    sig2_e_new = sig2_e + sig2_e^2 * (y^T P P y - tr(P)) / df

These updates are guaranteed to increase the REML log-likelihood
at every iteration (Dempster et al. 1977; McLachlan & Krishnan 2008).

Used as fallback when Newton-type methods (AI-REML) oscillate or diverge.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from .reml_math import reml_loglikelihood

logger = logging.getLogger(__name__)


def mm_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    lam_init: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    lam_min: float = 1e-10,
    lam_max: float = 1e10,
) -> tuple[float, float, float, list[dict]]:
    """EM-REML for single-trait LMM with joint (sig2_g, sig2_e) updates.

    Unlike the ratio estimator (lam = yPKPy / trPK), this uses proper
    joint EM updates that are guaranteed monotone in the REML log-likelihood.

    Parameters
    ----------
    lam_init : initial lambda = sig2_g / sig2_e (converted to initial variances)

    Returns
    -------
    sig2_g, sig2_e, ll, trace
    """
    y = Y_rot.squeeze()
    X = X0_rot
    n = y.shape[0]
    c = X.shape[1]
    df = n - c
    evals = eigenvalues

    # Initialize from lambda
    lam = lam_init if lam_init is not None else 1.0
    lam = max(lam_min, min(lam_max, lam))

    # Convert lambda to (sig2_g, sig2_e) via one evaluation
    _, sig2_e, sig2_g = reml_loglikelihood(lam, Y_rot, X0_rot, eigenvalues)

    ll_prev = float("-inf")
    trace: list[dict] = []

    for it in range(max_iter):
        # H diagonal in rotated space: evals * sig2_g + sig2_e
        H_diag = evals * sig2_g + sig2_e
        H_inv = 1.0 / H_diag

        # Weighted LS for fixed effects
        wX = H_inv.unsqueeze(1) * X
        XtHiX = X.T @ wX
        XtHiX_inv = torch.linalg.inv(XtHiX)
        XtHiy = X.T @ (H_inv * y)
        beta = XtHiX_inv @ XtHiy

        # P operator: P = H^{-1} - H^{-1} X (X^T H^{-1} X)^{-1} X^T H^{-1}
        wX_XtHiXi = wX @ XtHiX_inv
        P_diag = H_inv - (wX_XtHiXi * wX).sum(dim=1)

        # Py
        resid = y - X @ beta
        Py = H_inv * resid - wX @ (XtHiX_inv @ (X.T @ (H_inv * resid)))

        # EM-REML sufficient statistics
        yPKPy = (Py * evals * Py).sum().item()   # y^T P K P y
        trPK = (P_diag * evals).sum().item()      # tr(P K)
        yPPy = (Py * Py).sum().item()              # y^T P P y
        trP = P_diag.sum().item()                   # tr(P)

        # Joint EM-REML updates (guaranteed monotone increase of REML ll)
        sig2_g_new = sig2_g + sig2_g ** 2 * (yPKPy - trPK) / df
        sig2_e_new = sig2_e + sig2_e ** 2 * (yPPy - trP) / df

        # Floor to prevent collapse to zero
        sig2_g_new = max(sig2_g_new, 1e-10)
        sig2_e_new = max(sig2_e_new, 1e-10)

        # Evaluate log-likelihood at new parameters
        lam_new = sig2_g_new / max(sig2_e_new, 1e-20)
        ll, _, _ = reml_loglikelihood(lam_new, Y_rot, X0_rot, eigenvalues)

        trace.append({
            "iter": it,
            "mode": "MM",
            "lambda": lam_new,
            "sig2_g": sig2_g_new,
            "sig2_e": sig2_e_new,
            "ll": ll,
        })

        # Convergence check
        if it > 0 and abs(ll - ll_prev) < tol * max(abs(ll_prev), 1.0):
            logger.info(
                "MM-REML converged in %d iterations: sig2_g=%.6e, sig2_e=%.6e, ll=%.6f",
                it, sig2_g_new, sig2_e_new, ll,
            )
            break

        ll_prev = ll
        sig2_g = sig2_g_new
        sig2_e = sig2_e_new
    else:
        logger.warning("MM-REML did not converge in %d iterations.", max_iter)

    # Final evaluation
    lam_final = sig2_g / max(sig2_e, 1e-20)
    ll_final, sig2_e_final, sig2_g_final = reml_loglikelihood(
        lam_final, Y_rot, X0_rot, eigenvalues,
    )
    return sig2_g_final, sig2_e_final, ll_final, trace
