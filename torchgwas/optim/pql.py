"""Penalized Quasi-Likelihood (PQL) iteration for GLMM null fitting.

Implements the SAIGE-style PQL loop: iteratively construct working response
and weights from the current GLM fit, then solve a weighted LMM in the
inner loop.  The eigendecomposition of W^{1/2} K W^{1/2} is cached from the
first iteration and reused (SAIGE approximation).

References
----------
- Breslow & Clayton (1993). Approximate inference in GLMM.
- Zhou et al. (2018). SAIGE: efficiently controls for case-control
  imbalance and sample relatedness in GLMM.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def pql_fit(
    Y: Tensor,
    X0: Tensor,
    K: Tensor,
    mu_init: Tensor,
    link_variance_fn,
    link_derivative_fn,
    link_fn,
    *,
    max_outer: int = 30,
    tol: float = 1e-4,
) -> dict:
    """Run PQL iteration to fit a GLMM null model.

    Parameters
    ----------
    Y : (n,) — phenotype
    X0 : (n, c) — covariate matrix
    K : (n, n) — kinship / GRM
    mu_init : (n,) — initial fitted values from GLM fit (e.g. from BinaryGLM)
    link_variance_fn : callable
        V(mu) returning per-individual variance (n,).
        For binary: mu*(1-mu). For ordinal: custom.
    link_derivative_fn : callable
        g'(mu) = d eta / d mu.  For logit: 1 / (mu*(1-mu)).
    link_fn : callable
        g(mu) returning the linear predictor.  For logit: log(mu/(1-mu)).
    max_outer : int
        Maximum PQL outer iterations.
    tol : float
        Convergence tolerance on max|beta_new - beta_old|.

    Returns
    -------
    dict with keys:
        mu : (n,) fitted probabilities
        beta : (c,) fixed effects
        sig2_g : float — genetic variance
        sig2_e : float — residual variance (quasi)
        tau : (n,) — diagonal of V^{-1}
        converged : bool
        eigenvalues : (n,) — cached eigenvalues
        eigenvectors : (n, n) — cached eigenvectors
        log_likelihood : float — penalized quasi-log-likelihood
    """
    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    K = K.to(STAT_DTYPE)

    n, c = X0.shape
    mu = mu_init.to(STAT_DTYPE).clone()

    # Cache eigendecomposition of K (done once)
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    # Reverse to descending
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
    eigenvectors = eigenvectors.flip(1)

    beta = torch.zeros(c, dtype=STAT_DTYPE, device=Y.device)
    sig2_g = 0.5
    sig2_e = 1.0

    converged = False

    for outer in range(max_outer):
        # Step 1: Working weights and response
        W_var = link_variance_fn(mu)  # (n,) = mu*(1-mu) for binary
        W_var = torch.clamp(W_var, min=1e-10)

        eta = link_fn(mu)  # (n,)
        z = eta + (Y - mu) / W_var  # (n,) working response

        # Step 2: Weighted LMM in rotated space
        # V = sig2_g * K + sig2_e * diag(1/W_var)
        # This is equivalent to a weighted LMM with weights W_var
        # Rotate: U' z, U' X0
        # In rotated space: cov = sig2_g * diag(evals) + sig2_e * U' diag(1/W) U

        # SAIGE simplification: treat W as diagonal, form
        # working model: z ~ X0*beta + u + e, where
        # u ~ N(0, sig2_g * K), e_i ~ N(0, sig2_e / W_i)

        # For computational tractability, use the approximation:
        # Scale by sqrt(W): z_w = sqrt(W)*z, X_w = sqrt(W)*X0
        # Then: z_w ~ X_w*beta + u_w + e_w with cov(u_w) = sig2_g * sqrt(W)*K*sqrt(W)
        # and var(e_w) = sig2_e

        sqrtW = torch.sqrt(W_var)
        z_w = sqrtW * z
        X_w = sqrtW.unsqueeze(1) * X0

        # Form weighted kernel: K_w = diag(sqrtW) @ K @ diag(sqrtW)
        # Then eigendecompose K_w.  For SAIGE, we cache this from iteration 1.
        if outer == 0:
            K_w = sqrtW.unsqueeze(1) * K * sqrtW.unsqueeze(0)
            K_w = 0.5 * (K_w + K_w.T)  # symmetrise
            evals_w, evecs_w = torch.linalg.eigh(K_w)
            evals_w = evals_w.flip(0).clamp(min=0.0)
            evecs_w = evecs_w.flip(1)

        # Rotate
        z_rot = evecs_w.T @ z_w
        X_rot = evecs_w.T @ X_w

        # In rotated space: z_rot ~ X_rot*beta + u_rot + e_rot
        # where var(u_rot_i) = sig2_g * evals_w_i, var(e_rot_i) = sig2_e
        # So total var_i = sig2_g * evals_w_i + sig2_e

        # Optimize sig2_g, sig2_e via profile REML
        sig2_g, sig2_e = _profile_reml_vc(z_rot, X_rot, evals_w,
                                            sig2_g_init=sig2_g,
                                            sig2_e_init=sig2_e)

        # Compute beta
        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (evals_w * lam + 1.0)
        wX = H_inv.unsqueeze(1) * X_rot
        XtHiX = X_rot.T @ wX
        XtHiz = X_rot.T @ (H_inv * z_rot)

        try:
            beta_new = torch.linalg.solve(XtHiX, XtHiz)
        except Exception:
            logger.warning("PQL: linalg.solve failed at iteration %d", outer)
            break

        # Back to original space: compute eta and mu
        eta_new = X0 @ beta_new

        # Add BLUP: u = sig2_g * K * V^{-1} * (z - X0*beta)
        # V = sig2_g * K_w + sig2_e * I (in weighted space)
        # P y_w = H^{-1} y_w - H^{-1} X_w (X_w' H^{-1} X_w)^{-1} X_w' H^{-1} y_w
        z_w_curr = sqrtW * (z - X0 @ beta_new)
        z_rot_resid = evecs_w.T @ z_w_curr
        # BLUP in rotated space: u_rot = sig2_g * evals_w / (sig2_g * evals_w + sig2_e) * resid_rot
        shrink = (sig2_g * evals_w) / (sig2_g * evals_w + sig2_e)
        u_rot = shrink * z_rot_resid
        u_w = evecs_w @ u_rot
        # Back to original: u = u_w / sqrt(W)
        u = u_w / sqrtW.clamp(min=1e-10)

        eta_new = eta_new + u

        # Convert to mu (inverse link)
        mu_new = torch.sigmoid(eta_new)  # logit link
        mu_new = torch.clamp(mu_new, min=1e-10, max=1.0 - 1e-10)

        # Convergence check
        param_change = (beta_new - beta).abs().max().item()

        beta = beta_new
        mu = mu_new

        if param_change < tol and outer > 0:
            converged = True
            break

    # Compute quasi-log-likelihood
    ll = _quasi_loglik(Y, mu, sig2_g, sig2_e, K)

    return {
        "mu": mu,
        "beta": beta,
        "sig2_g": sig2_g,
        "sig2_e": sig2_e,
        "converged": converged,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "evals_w": evals_w,
        "evecs_w": evecs_w,
        "log_likelihood": ll,
        "n_outer": outer + 1,
    }


def _profile_reml_vc(
    z_rot: Tensor,
    X_rot: Tensor,
    evals: Tensor,
    sig2_g_init: float = 0.5,
    sig2_e_init: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Estimate variance components via profile REML in rotated space.

    Uses the standard REML grid / Newton approach from the LMM.
    For speed, we do a simple grid search on lambda = sig2_g/sig2_e
    then extract both components.
    """
    n = z_rot.shape[0]
    c = X_rot.shape[1]

    best_ll = float("-inf")
    best_lam = sig2_g_init / max(sig2_e_init, 1e-20)

    # Coarse grid then refine
    for lam in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, best_lam]:
        if lam <= 0:
            continue
        H_inv = 1.0 / (evals * lam + 1.0)
        wX = H_inv.unsqueeze(1) * X_rot
        XtHiX = X_rot.T @ wX
        XtHiz = X_rot.T @ (H_inv * z_rot)

        try:
            beta = torch.linalg.solve(XtHiX, XtHiz)
        except Exception:
            continue

        resid = z_rot - X_rot @ beta
        yPy = (resid * H_inv * resid).sum().item()
        log_det_H = torch.log(evals * lam + 1.0).sum().item()
        sign, log_det_XtHiX = torch.linalg.slogdet(XtHiX)
        if sign.item() <= 0:
            continue
        log_det_XtHiX = log_det_XtHiX.item()

        # REML log-likelihood (up to constant)
        sig2_e_hat = yPy / max(n - c, 1)
        ll = -0.5 * ((n - c) * math.log(max(sig2_e_hat, 1e-20))
                      + log_det_H + log_det_XtHiX)

        if ll > best_ll:
            best_ll = ll
            best_lam = lam

    # Extract components
    # sig2_e from residuals at best lambda
    H_inv = 1.0 / (evals * best_lam + 1.0)
    wX = H_inv.unsqueeze(1) * X_rot
    XtHiX = X_rot.T @ wX
    XtHiz = X_rot.T @ (H_inv * z_rot)
    beta = torch.linalg.solve(XtHiX, XtHiz)
    resid = z_rot - X_rot @ beta
    yPy = (resid * H_inv * resid).sum().item()
    sig2_e = max(yPy / max(n - c, 1), 1e-10)
    sig2_g = max(best_lam * sig2_e, 1e-10)

    return sig2_g, sig2_e


def _quasi_loglik(Y: Tensor, mu: Tensor, sig2_g: float, sig2_e: float,
                   K: Tensor) -> float:
    """Penalized quasi-log-likelihood for GLMM (binary case)."""
    # Binomial log-likelihood component
    mu_c = mu.clamp(min=1e-300, max=1.0 - 1e-300)
    ll_binom = (Y * torch.log(mu_c) + (1.0 - Y) * torch.log(1.0 - mu_c)).sum().item()

    # Penalty from random effect (simplified)
    # -0.5 * n * log(sig2_g) (omit K-dependent terms for speed)
    n = Y.shape[0]
    ll_pen = -0.5 * n * math.log(max(sig2_g, 1e-20))

    return ll_binom + ll_pen
