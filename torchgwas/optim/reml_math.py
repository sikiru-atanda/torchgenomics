"""Core REML mathematics for single-trait LMM in rotated eigenspace.

All computations are O(n) sums over eigenvalues — the eigendecomposition
trick makes per-evaluation cost linear.  GEMMA parameterizes by
lambda = sig_g^2 / sig_e^2 and profiles out sig_e^2 analytically.

References
----------
Zhou & Stephens, Nat Genet 2012 — Equations (7)-(9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE


@dataclass
class REMLResult:
    """Output of a single REML evaluation."""

    log_likelihood: float
    sig2_g: float
    sig2_e: float
    lam: float  # lambda = sig2_g / sig2_e
    n_iter: int = 0
    converged: bool = False


# ---------------------------------------------------------------------------
# P-operator (projection away from fixed effects)
# ---------------------------------------------------------------------------

def _compute_P_quantities(
    Y_rot: Tensor,
    X0_rot: Tensor,
    H_inv_diag: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute Py and PX0 using the P-operator in rotated space.

    P = H^{-1} - H^{-1} X (X^T H^{-1} X)^{-1} X^T H^{-1}

    In rotated space H is diagonal, so all these are cheap.

    Parameters
    ----------
    Y_rot : (n,) or (n, 1)
    X0_rot : (n, c)
    H_inv_diag : (n,)  — 1 / (evals * lambda + 1)

    Returns
    -------
    Py : (n,) — P @ y in rotated space
    beta0 : (c,) — GLS estimate of fixed effects under null
    """
    y = Y_rot.squeeze()  # (n,)
    X = X0_rot  # (n, c)
    w = H_inv_diag  # (n,)

    # Weighted X and y
    wX = w.unsqueeze(1) * X  # (n, c)
    wy = w * y  # (n,)

    # Normal equations: (X^T W X) beta = X^T W y
    XtWX = X.T @ wX  # (c, c)
    XtWy = X.T @ wy  # (c,)

    # Solve for beta0
    beta0 = torch.linalg.solve(XtWX, XtWy)  # (c,)

    # Py = W(y - X beta0)
    residual = y - X @ beta0  # (n,)
    Py = w * residual  # (n,)

    return Py, beta0


# ---------------------------------------------------------------------------
# REML log-likelihood (profiled over sig_e^2)
# ---------------------------------------------------------------------------

def reml_loglikelihood(
    lam: float,
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
) -> tuple[float, float, float]:
    """Evaluate the REML log-likelihood at a given lambda.

    GEMMA convention: parameterize by lambda = sig_g^2 / sig_e^2.
    In rotated space, H_ii = evals_i * lambda + 1, so:

        -2 * ll_REML = (n - c) * log(2*pi*sig_e^2)
                      + sum_i log(H_ii)
                      + log|X^T H^{-1} X|
                      + (y^T P y) / sig_e^2

    where sig_e^2 is profiled out analytically:

        sig_e^2 = y^T P y / (n - c)

    Parameters
    ----------
    lam : float
        lambda = sig_g^2 / sig_e^2. Must be >= 0.
    Y_rot, X0_rot : rotated phenotype and covariates
    eigenvalues : (n,) from GRM eigendecomposition

    Returns
    -------
    ll : float — REML log-likelihood
    sig2_e : float — profiled residual variance
    sig2_g : float — genetic variance (lam * sig2_e)
    """
    n = eigenvalues.shape[0]
    c = X0_rot.shape[1]
    df = n - c  # residual degrees of freedom

    # H diagonal: evals * lambda + 1
    H_diag = eigenvalues * lam + 1.0  # (n,)
    H_inv_diag = 1.0 / H_diag  # (n,)

    # P-operator quantities
    Py, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv_diag)
    y = Y_rot.squeeze()

    # Quadratic form: y^T P y
    yPy = (y * Py).sum()

    # Profile sig_e^2
    sig2_e = (yPy / df).item()
    sig2_e = max(sig2_e, 1e-20)  # floor for numerical safety
    sig2_g = lam * sig2_e

    # Log-likelihood components
    log_H = torch.log(H_diag).sum()  # sum_i log(H_ii)

    # log|X^T H^{-1} X|
    wX = H_inv_diag.unsqueeze(1) * X0_rot
    XtWX = X0_rot.T @ wX
    sign, log_det_XtWX = torch.linalg.slogdet(XtWX)
    log_det_XtWX = log_det_XtWX if sign > 0 else torch.tensor(0.0, dtype=STAT_DTYPE)

    # -2 * ll_REML = df * log(2*pi*sig2_e) + log_H + log_det_XtWX + df
    # (the last term comes from yPy/sig2_e = df after profiling)
    import math
    neg2ll = df * math.log(2.0 * math.pi * sig2_e) + log_H.item() + log_det_XtWX.item() + df

    ll = -0.5 * neg2ll

    return ll, sig2_e, sig2_g


# ---------------------------------------------------------------------------
# REML first and second derivatives w.r.t. lambda
# ---------------------------------------------------------------------------

def reml_derivatives(
    lam: float,
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
) -> tuple[float, float, float]:
    """First and second derivatives of REML log-likelihood w.r.t. lambda.

    Used by AI-REML (Newton-Raphson).

    Returns
    -------
    ll : float — log-likelihood
    dl : float — d(ll)/d(lambda)
    d2l : float — d^2(ll)/d(lambda)^2 (AI approximation: -0.5 * y^T P dV P dV P y)
    """
    n = eigenvalues.shape[0]
    c = X0_rot.shape[1]
    df = n - c

    H_diag = eigenvalues * lam + 1.0
    H_inv_diag = 1.0 / H_diag

    Py, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv_diag)
    y = Y_rot.squeeze()

    yPy = (y * Py).sum()
    sig2_e = max((yPy / df).item(), 1e-20)

    # dV/dlambda in rotated space = diag(eigenvalues) (since V = sig_e^2 * H)
    # dH/dlambda = diag(eigenvalues)
    # dl/dlambda = -0.5 * [tr(P * dH) - y^T P (dH) P y / sig_e^2]
    #            = -0.5 * [sum(P_ii * evals_i) - (Py .* evals .* Py) / sig_e^2 ]

    # trace(P * dH/dlambda) where dH/dlambda = diag(evals)
    # P_ii = H_inv_ii - (H_inv X (X^T H_inv X)^{-1} X^T H_inv)_ii
    # Rather than computing P_ii, use the identity:
    # tr(P dH) = sum_i evals_i * H_inv_ii - sum_ij evals_i * (wX)_ia * (XtWX_inv)_ab * (wX)_ib

    wX = H_inv_diag.unsqueeze(1) * X0_rot  # (n, c)
    XtWX = X0_rot.T @ wX
    XtWX_inv = torch.linalg.inv(XtWX)

    # P diagonal elements (rotated space):
    # P_ii = w_i - sum_ab (wX_ia * XtWX_inv_ab * wX_ib)
    wX_XtWXi = wX @ XtWX_inv  # (n, c)
    P_diag = H_inv_diag - (wX_XtWXi * wX).sum(dim=1)  # (n,)

    # tr(P * dH/dlambda)
    tr_PdH = (P_diag * eigenvalues).sum()

    # y^T P dH P y = (Py * evals * Py) (since dH = diag(evals) in rotated space)
    yPdHPy = (Py * eigenvalues * Py).sum()

    # Score (first derivative):
    # dl/dlam = -0.5 * (tr_PdH - yPdHPy / sig2_e)
    dl = -0.5 * (tr_PdH - yPdHPy / sig2_e).item()

    # AI (Average Information) = 0.5 * y^T P dV P dV P y / sig_e^4
    # This approximates -d2l/dlam^2 for Newton step.
    # In rotated space with profiled sig2_e:
    # AI = 0.5 * yP(dH)P(dH)Py / sig2_e^2
    PdHPy = eigenvalues * Py  # dH @ Py (diagonal)
    # Now we need P @ PdHPy, which requires re-applying P
    # P @ v = H_inv * v - H_inv X (X^T H_inv X)^{-1} X^T H_inv v
    v = PdHPy
    wv = H_inv_diag * v
    Xtv = X0_rot.T @ wv
    Pv = wv - wX @ (XtWX_inv @ Xtv)

    yPdHPdHPy = (Py * eigenvalues * Pv).sum()
    ai = 0.5 * yPdHPdHPy / (sig2_e ** 2)

    # For Newton: d2l ≈ -AI
    d2l = -ai.item()

    # Also compute ll for convergence check
    log_H = torch.log(H_diag).sum()
    sign, log_det_XtWX = torch.linalg.slogdet(XtWX)
    log_det_v = log_det_XtWX.item() if sign.item() > 0 else 0.0

    import math
    neg2ll = df * math.log(2 * math.pi * sig2_e) + log_H.item() + log_det_v + df
    ll = -0.5 * neg2ll

    return ll, dl, d2l
