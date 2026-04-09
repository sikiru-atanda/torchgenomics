"""Multi-Environment PQL iteration for categorical GLMM across environments.

Extends the single-environment PQL (pql.py) to handle E environments
simultaneously.  At each outer iteration: compute per-env working response
and weights, then solve a multi-trait weighted LMM (E "traits") in the
inner loop using the EED trick from mvlmm_reml.py.

The eigendecomposition of W_avg^{1/2} K W_avg^{1/2} is cached from the first
iteration (SAIGE approximation).  Σ_g (E×E) is estimated via LBFGS on
log-Cholesky factors, warm-started between PQL iterations.

References
----------
- Breslow & Clayton (1993). Approximate inference in GLMM.
- Zhou et al. (2018). SAIGE.
- Zhou & Stephens (2014). mvLMM / GEMMA.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def multi_env_pql_fit(
    Y: Tensor,
    X0: Tensor,
    K: Tensor,
    mu_init: Tensor,
    family: str,
    *,
    n_categories: int | None = None,
    max_outer: int = 30,
    tol: float = 1e-4,
    max_inner_reml: int = 20,
) -> dict:
    """Run multi-environment PQL to fit a ME-GLMM null model.

    Parameters
    ----------
    Y : (n, E) — phenotype matrix (one column per environment)
    X0 : (n, c) — covariate matrix
    K : (n, n) — kinship / GRM
    mu_init : (n, E) — initial fitted values from per-env GLM
    family : "binary" or "ordinal"
    n_categories : int — for ordinal family, number of categories J
    max_outer : int — maximum PQL outer iterations
    tol : float — convergence tolerance on max|β_new − β_old|
    max_inner_reml : int — max LBFGS iterations for inner REML

    Returns
    -------
    dict with keys:
        mu : (n, E) fitted values
        beta : (c, E) fixed effects
        Sigma_g : (E, E) genetic covariance across environments
        Sigma_e : (E, E) residual covariance
        converged : bool
        eigenvalues : (n,) raw K eigenvalues
        eigenvectors : (n, n) raw K eigenvectors
        evals_w : (n,) weighted kernel eigenvalues
        evecs_w : (n, n) weighted kernel eigenvectors
        W : (n, E) working weights at convergence
        log_likelihood : float
        n_outer : int
        gamma : (n, E, J-1) cumulative probs (ordinal only)
        thresholds : (E, J-1) per-env thresholds (ordinal only)
    """
    from .mvlmm_reml import compute_sigma_inv, mvlmm_reml_loglikelihood

    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    K = K.to(STAT_DTYPE)

    n, E = Y.shape
    c = X0.shape[1]
    mu = mu_init.to(STAT_DTYPE).clone()
    mu = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)

    # Eigendecompose raw K (used for NullFit)
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
    eigenvectors = eigenvectors.flip(1)

    # Initialize beta, Sigma_g, Sigma_e
    beta = torch.zeros(c, E, dtype=STAT_DTYPE, device=Y.device)
    Sigma_g = 0.5 * torch.eye(E, dtype=STAT_DTYPE, device=Y.device)
    Sigma_e = torch.eye(E, dtype=STAT_DTYPE, device=Y.device)

    # For ordinal: track per-env thresholds and gamma
    gamma = None
    thresholds = None
    if family == "ordinal" and n_categories is not None:
        J = n_categories
        thresholds = torch.zeros(E, J - 1, dtype=STAT_DTYPE, device=Y.device)
        gamma = torch.zeros(n, E, J - 1, dtype=STAT_DTYPE, device=Y.device)

    converged = False
    evals_w = None
    evecs_w = None

    for outer in range(max_outer):
        # Step 1: Per-env working weights and response
        W, z, eta = _compute_working_quantities(
            Y, mu, X0, beta, family, n_categories, gamma, thresholds,
        )

        # Step 2: Average weights for eigendecomposition (SAIGE approx)
        W_avg = W.mean(dim=1)  # (n,)
        W_avg = torch.clamp(W_avg, min=1e-10)
        sqrtW_avg = torch.sqrt(W_avg)

        # Cache weighted kernel eigendecomposition at iteration 0
        if outer == 0:
            K_w = sqrtW_avg.unsqueeze(1) * K * sqrtW_avg.unsqueeze(0)
            K_w = 0.5 * (K_w + K_w.T)
            evals_w, evecs_w = torch.linalg.eigh(K_w)
            evals_w = evals_w.flip(0).clamp(min=0.0)
            evecs_w = evecs_w.flip(1)

        # Step 3: Rotate per-env data
        # Scale by sqrt(W_avg) then rotate
        z_scaled = sqrtW_avg.unsqueeze(1) * z  # (n, E)
        X_scaled = sqrtW_avg.unsqueeze(1) * X0  # (n, c)

        Z_rot = evecs_w.T @ z_scaled  # (n, E)
        X_rot = evecs_w.T @ X_scaled  # (n, c)

        # Step 4: Multi-trait REML inner solve
        Sigma_g, Sigma_e, reml_ll = _inner_reml(
            Z_rot, X_rot, evals_w, Sigma_g, Sigma_e,
            max_iter=max_inner_reml,
        )

        # Step 5: GLS fixed effects via mvlmm_null_quantities pattern
        W_inv, _ = compute_sigma_inv(evals_w, Sigma_g, Sigma_e)  # (n, E, E)

        # Normal equations: XtWX (cE, cE), XtWZ (cE,)
        XtWX = torch.einsum('ia,ist,ib->asbt', X_rot, W_inv, X_rot)
        XtWX = XtWX.reshape(c * E, c * E)
        WZ = torch.einsum('ist,is->it', W_inv, Z_rot)  # (n, E)
        XtWZ = torch.einsum('ia,it->at', X_rot, WZ).reshape(c * E)

        try:
            B_vec = torch.linalg.solve(
                XtWX + 1e-8 * torch.eye(c * E, dtype=STAT_DTYPE, device=Y.device),
                XtWZ,
            )
        except Exception:
            logger.warning("ME-PQL: GLS solve failed at iteration %d", outer)
            break

        beta_new = B_vec.reshape(c, E)

        # Step 6: BLUP per environment
        # Residuals in rotated space
        R_rot = Z_rot - X_rot @ beta_new  # (n, E)
        # BLUP: u_rot_i = Σ_g · λ_i · Σ_i⁻¹ · r_rot_i
        # Σ_i = λ_i·Σ_g + Σ_e, so Σ_g·λ_i·Σ_i⁻¹ = Σ_g·λ_i·W_inv_i (already computed)
        # Actually: u_rot_i = λ_i · Σ_g @ Σ_i⁻¹ @ r_rot_i
        u_rot = torch.zeros(n, E, dtype=STAT_DTYPE, device=Y.device)
        for i in range(n):
            SgLi = evals_w[i] * Sigma_g  # (E, E)
            u_rot[i] = SgLi @ (W_inv[i] @ R_rot[i])  # (E,)

        # Back-transform to original space
        u_scaled = evecs_w @ u_rot  # (n, E)
        u = u_scaled / sqrtW_avg.unsqueeze(1).clamp(min=1e-10)

        # Step 7: Update eta and mu
        eta_new = X0 @ beta_new + u  # (n, E)

        if family == "binary":
            mu_new = torch.sigmoid(eta_new)
            mu_new = torch.clamp(mu_new, min=1e-10, max=1.0 - 1e-10)
        elif family == "ordinal":
            mu_new, gamma, thresholds = _update_ordinal_mu(
                eta_new, Y, n_categories,
            )
        else:
            raise ValueError(f"Unsupported family: {family}")

        # Convergence check
        param_change = (beta_new - beta).abs().max().item()

        beta = beta_new
        mu = mu_new

        if param_change < tol and outer > 0:
            converged = True
            break

    # Recompute final W
    W_final, _, _ = _compute_working_quantities(
        Y, mu, X0, beta, family, n_categories, gamma, thresholds,
    )

    # Quasi-log-likelihood
    ll = _multi_env_quasi_loglik(Y, mu, family)

    logger.info(
        "ME-PQL: n=%d, E=%d, family=%s, converged=%s, iters=%d",
        n, E, family, converged, outer + 1,
    )

    result = {
        "mu": mu,
        "beta": beta,
        "Sigma_g": Sigma_g,
        "Sigma_e": Sigma_e,
        "converged": converged,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "evals_w": evals_w,
        "evecs_w": evecs_w,
        "W": W_final,
        "log_likelihood": ll,
        "n_outer": outer + 1,
    }

    if family == "ordinal":
        result["gamma"] = gamma
        result["thresholds"] = thresholds

    return result


def _compute_working_quantities(
    Y: Tensor,
    mu: Tensor,
    X0: Tensor,
    beta: Tensor,
    family: str,
    n_categories: int | None,
    gamma: Tensor | None,
    thresholds: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-env working weights W and working response z.

    Returns
    -------
    W : (n, E) working weights
    z : (n, E) working response
    eta : (n, E) linear predictor
    """
    n, E = Y.shape

    if family == "binary":
        W = mu * (1.0 - mu)  # (n, E)
        W = torch.clamp(W, min=1e-10)
        eta = torch.log(mu / (1.0 - mu))  # logit
        z = eta + (Y - mu) / W
    elif family == "ordinal":
        J = n_categories
        W = torch.zeros(n, E, dtype=STAT_DTYPE, device=Y.device)
        z = torch.zeros(n, E, dtype=STAT_DTYPE, device=Y.device)
        eta = X0 @ beta  # (n, E) — linear predictor (without thresholds)

        for e in range(E):
            # Compute cumulative probs per env
            # gamma_ij = sigmoid(alpha_j - eta_i)
            for j in range(J - 1):
                if gamma is not None and thresholds is not None:
                    gamma[:, e, j] = torch.sigmoid(thresholds[e, j] - eta[:, e])
                w_j = gamma[:, e, j] * (1.0 - gamma[:, e, j])
                W[:, e] += w_j

            W[:, e] = torch.clamp(W[:, e], min=1e-10)

            # Expected value: E[Y] = sum_j j * P(Y=j)
            # P(Y=j) = gamma_j - gamma_{j-1} with gamma_0=0, gamma_J=1
            E_Y = torch.zeros(n, dtype=STAT_DTYPE, device=Y.device)
            for j in range(J):
                if j == 0:
                    p_j = gamma[:, e, 0] if J > 1 else torch.ones(n, dtype=STAT_DTYPE, device=Y.device)
                elif j == J - 1:
                    p_j = 1.0 - gamma[:, e, j - 1]
                else:
                    p_j = gamma[:, e, j] - gamma[:, e, j - 1]
                p_j = torch.clamp(p_j, min=1e-10)
                E_Y += j * p_j

            z[:, e] = eta[:, e] + (Y[:, e] - E_Y) / W[:, e]
    else:
        raise ValueError(f"Unsupported family: {family}")

    return W, z, eta


def _update_ordinal_mu(
    eta: Tensor,
    Y: Tensor,
    n_categories: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Update ordinal mu (expected value) and gamma/thresholds from eta.

    For ordinal, mu_ie = E[Y_ie] under the proportional odds model.
    Thresholds are re-estimated from observed frequencies per env.
    """
    n, E = Y.shape
    J = n_categories
    device = Y.device

    # Re-estimate thresholds from cumulative frequencies per env
    thresholds = torch.zeros(E, J - 1, dtype=STAT_DTYPE, device=device)
    for e in range(E):
        for j in range(J - 1):
            cum_freq = (Y[:, e] <= j).float().mean().clamp(min=0.01, max=0.99)
            thresholds[e, j] = torch.log(cum_freq / (1.0 - cum_freq))
        # Ensure ordering
        thresholds[e], _ = thresholds[e].sort()

    # Compute gamma and mu
    gamma = torch.zeros(n, E, J - 1, dtype=STAT_DTYPE, device=device)
    mu = torch.zeros(n, E, dtype=STAT_DTYPE, device=device)

    for e in range(E):
        for j in range(J - 1):
            gamma[:, e, j] = torch.sigmoid(thresholds[e, j] - eta[:, e])

        # E[Y_ie] = sum_j j * P(Y=j)
        for j in range(J):
            if j == 0:
                p_j = gamma[:, e, 0]
            elif j == J - 1:
                p_j = 1.0 - gamma[:, e, j - 1]
            else:
                p_j = gamma[:, e, j] - gamma[:, e, j - 1]
            p_j = torch.clamp(p_j, min=1e-10)
            mu[:, e] += j * p_j

    return mu, gamma, thresholds


def _inner_reml(
    Z_rot: Tensor,
    X_rot: Tensor,
    evals_w: Tensor,
    Sg_init: Tensor,
    Se_init: Tensor,
    max_iter: int = 20,
) -> tuple[Tensor, Tensor, float]:
    """Estimate Σ_g, Σ_e via LBFGS on log-Cholesky factors.

    Uses a differentiable REML closure (autograd-friendly) warm-started
    from previous PQL iteration's estimates.
    """
    n, E = Z_rot.shape
    c = X_rot.shape[1]
    df = n - c
    device = Z_rot.device

    # Detach data to avoid leaking grad through PQL outer loop
    Z_rot_d = Z_rot.detach()
    X_rot_d = X_rot.detach()
    evals_d = evals_w.detach()

    # Parameterize via Cholesky factors
    jitter = 1e-6 * torch.eye(E, dtype=STAT_DTYPE, device=device)
    try:
        Lg = torch.linalg.cholesky(Sg_init + jitter)
    except Exception:
        Lg = torch.eye(E, dtype=STAT_DTYPE, device=device) * 0.5

    try:
        Le = torch.linalg.cholesky(Se_init + jitter)
    except Exception:
        Le = torch.eye(E, dtype=STAT_DTYPE, device=device)

    # Flatten lower-triangular elements as parameters
    tril_idx = torch.tril_indices(E, E)
    params_g = Lg[tril_idx[0], tril_idx[1]].clone().requires_grad_(True)
    params_e = Le[tril_idx[0], tril_idx[1]].clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [params_g, params_e],
        max_iter=1,
        line_search_fn="strong_wolfe",
    )

    best_ll = float("-inf")
    best_Sg = Sg_init.clone()
    best_Se = Se_init.clone()

    def _build_spd(params, tril_idx, E, device):
        """Reconstruct SPD matrix from lower-triangular params."""
        L = torch.zeros(E, E, dtype=STAT_DTYPE, device=device)
        L[tril_idx[0], tril_idx[1]] = params
        # Softplus on diagonal to ensure positivity
        diag_idx = torch.arange(E, device=device)
        L[diag_idx, diag_idx] = torch.nn.functional.softplus(L[diag_idx, diag_idx]) + 1e-4
        return L @ L.T

    def _reml_loglik_differentiable(Sg, Se):
        """Differentiable REML log-likelihood (returns Tensor, not float)."""
        # Sigma_i = evals[i] * Sg + Se → (n, E, E)
        Sigma = evals_d.view(n, 1, 1) * Sg.unsqueeze(0) + Se.unsqueeze(0)
        # Batched Cholesky
        L_sig = torch.linalg.cholesky(Sigma)
        W_inv = torch.cholesky_inverse(L_sig)  # (n, E, E)
        logdet = 2.0 * torch.log(torch.diagonal(L_sig, dim1=-2, dim2=-1)).sum(dim=-1)

        # GLS: B = (X'WX)^{-1} X'WZ
        XtWX = torch.einsum('ia,ist,ib->asbt', X_rot_d, W_inv, X_rot_d)
        XtWX = XtWX.reshape(c * E, c * E)
        WZ = torch.einsum('ist,is->it', W_inv, Z_rot_d)
        XtWZ = torch.einsum('ia,it->at', X_rot_d, WZ).reshape(c * E)
        B_vec = torch.linalg.solve(XtWX, XtWZ)
        B = B_vec.reshape(c, E)

        # Residuals and quadratic form
        R = Z_rot_d - X_rot_d @ B
        WR = torch.einsum('ist,is->it', W_inv, R)
        quad = (R * WR).sum()

        sum_logdet = logdet.sum()
        sign, logdet_XtWX = torch.linalg.slogdet(XtWX)
        logdet_XtWX_val = logdet_XtWX if sign > 0 else torch.tensor(0.0, dtype=STAT_DTYPE, device=device)

        neg2ll = sum_logdet + quad + df * E * math.log(2 * math.pi) + logdet_XtWX_val
        return -0.5 * neg2ll

    for it in range(max_iter):
        def closure():
            optimizer.zero_grad()
            Sg = _build_spd(params_g, tril_idx, E, device)
            Se = _build_spd(params_e, tril_idx, E, device)
            ll = _reml_loglik_differentiable(Sg, Se)
            loss = -ll
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            break

        # Extract current estimates
        with torch.no_grad():
            Sg = _build_spd(params_g, tril_idx, E, device)
            Se = _build_spd(params_e, tril_idx, E, device)
            ll_val = _reml_loglik_differentiable(Sg, Se).item()

        if ll_val > best_ll:
            best_ll = ll_val
            best_Sg = Sg.detach().clone()
            best_Se = Se.detach().clone()

    return best_Sg, best_Se, best_ll


def _multi_env_quasi_loglik(Y: Tensor, mu: Tensor, family: str) -> float:
    """Penalized quasi-log-likelihood for multi-environment GLMM."""
    if family == "binary":
        mu_c = mu.clamp(min=1e-300, max=1.0 - 1e-300)
        ll = (Y * torch.log(mu_c) + (1.0 - Y) * torch.log(1.0 - mu_c)).sum().item()
    elif family == "ordinal":
        # Approximate: treat mu as E[Y] and use Gaussian quasi-likelihood
        ll = -0.5 * ((Y - mu) ** 2).sum().item()
    else:
        ll = 0.0
    return ll
