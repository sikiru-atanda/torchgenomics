"""Mode D alt: Fisher scoring with expected information.

Single-trait REML optimizer using the *expected* (Fisher) information
matrix for the variance ratio lambda = sig2_g / sig2_e.

Algorithm
---------
At each iteration in rotated eigenspace, parameterize by lambda:

    H_ii = evals_i * lambda + 1
    P_diag = H_inv - (wX (X^T H_inv X)^{-1} wX^T)_ii

The score (gradient) and expected information are:

    score(lambda) = -0.5 * (tr(P dV) - y^T P dV P y / sig2_e)
    I_E(lambda)   =  0.5 * tr(P dV P dV)
                  =  0.5 * sum_i (P_diag_i * evals_i)^2

Fisher-scoring step: lambda <- lambda + score / I_E.

This differs from AI-REML (``optim.ai_reml``) which uses the average of
observed and expected information; pure Fisher scoring uses the expected
information only and is the "Mode D alt" optimizer in the layered stack.

The function returns ``(sig2_g, sig2_e, log_likelihood)`` after profiling
sig2_e analytically per :func:`reml_loglikelihood`.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .reml_math import reml_loglikelihood

logger = logging.getLogger(__name__)


def fisher_scoring_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    lam_init: float = 1.0,
    lam_min: float = 1e-10,
    lam_max: float = 1e10,
) -> tuple[float, float, float]:
    """Fisher-scoring REML for a single-trait LMM.

    Parameters
    ----------
    Y_rot : (n,) or (n, 1) — rotated phenotype.
    X0_rot : (n, c) — rotated covariates (intercept first).
    eigenvalues : (n,) — GRM eigenvalues.
    max_iter : int — maximum scoring iterations.
    tol : float — relative-change convergence tolerance on lambda.
    lam_init : float — starting lambda = sig2_g / sig2_e.
    lam_min, lam_max : float — bounds enforced after each step.

    Returns
    -------
    (sig2_g, sig2_e, log_likelihood) : final variance components and
    REML log-likelihood at the converged lambda.
    """
    eigenvalues = eigenvalues.to(Y_rot.dtype)
    y = Y_rot.squeeze()
    n = eigenvalues.shape[0]
    c = X0_rot.shape[1]

    lam = float(max(min(lam_init, lam_max), lam_min))

    for it in range(max_iter):
        # H diagonal in rotated eigenspace
        H_diag = eigenvalues * lam + 1.0
        H_inv_diag = 1.0 / H_diag

        # GLS via P-operator
        wX = H_inv_diag.unsqueeze(1) * X0_rot
        XtWX = X0_rot.T @ wX
        XtWX_inv = torch.linalg.inv(XtWX)

        # P diagonal: w_i - (wX_i (XtWX)^{-1} wX_i^T)
        wX_XtWXi = wX @ XtWX_inv
        P_diag = H_inv_diag - (wX_XtWXi * wX).sum(dim=1)

        # P @ y in rotated space
        beta0 = XtWX_inv @ (X0_rot.T @ (H_inv_diag * y))
        residual = y - X0_rot @ beta0
        Py = H_inv_diag * residual

        # Profile sig2_e
        df = max(n - c, 1)
        yPy = float((y * Py).sum())
        sig2_e = max(yPy / df, 1e-20)

        # Score wrt lambda: -0.5 * (tr(P dV) - yPy_lam / sig2_e)
        # where dV = sig2_e * diag(evals) → tr(P dV)/sig2_e = sum P_diag * evals
        tr_PdV_over_se = float((P_diag * eigenvalues).sum())
        yP_dV_Py_over_se = float((Py * eigenvalues * Py).sum()) / sig2_e
        score = -0.5 * (tr_PdV_over_se - yP_dV_Py_over_se)

        # Expected information: 0.5 * sum (P_diag_i * evals_i)^2
        info = 0.5 * float(((P_diag * eigenvalues) ** 2).sum())

        if info < 1e-30:
            logger.warning(
                "fisher_scoring_reml: information near zero at iter %d, stopping.",
                it,
            )
            break

        step = score / info
        lam_new = lam + step

        # Trust-region-style bounding to keep lambda in valid range
        if lam_new < lam_min:
            lam_new = lam_min
        if lam_new > lam_max:
            lam_new = lam_max

        rel_change = abs(lam_new - lam) / max(lam, 1e-12)
        lam = lam_new

        if rel_change < tol:
            break

    ll, sig2_e, sig2_g = reml_loglikelihood(lam, Y_rot, X0_rot, eigenvalues)
    logger.info(
        "fisher_scoring_reml: lam=%.6e, sig2_g=%.6e, sig2_e=%.6e, ll=%.6f",
        lam, sig2_g, sig2_e, ll,
    )
    return float(sig2_g), float(sig2_e), float(ll)
