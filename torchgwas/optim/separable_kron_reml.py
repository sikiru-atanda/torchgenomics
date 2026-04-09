"""Separable Kronecker REML: Vg = Vg_trait kron Vg_env via LBFGS-autograd.

For multi-trait multi-environment (MT-MET) models, parameterizes the genetic
covariance as a Kronecker product of a trait covariance (d x d) and an
environment covariance (E x E).  This reduces the number of Vg parameters
from dE(dE+1)/2 to d(d+1)/2 + E(E+1)/2.

The residual covariance Ve is always unstructured (dE x dE).

References:
    - Smith et al. (2001): FA and separable models for MET in plant breeding
    - Malosetti et al. (2013): Mixed models for MET GWAS
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def _init_cholesky_factor(d: int, scale: float = 1.0, device: torch.device = None) -> Tensor:
    """Initialize a lower-triangular Cholesky factor."""
    return torch.eye(d, dtype=STAT_DTYPE, device=device) * math.sqrt(max(scale, 1e-6))


def _enforce_lower_pos(L_param: Tensor) -> Tensor:
    """Enforce lower-triangular with positive diagonal."""
    L = torch.tril(L_param).clone()
    diag = torch.diag(L)
    L = L - torch.diag(diag) + torch.diag(diag.abs().clamp(min=1e-6))
    return L


def separable_kron_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    n_traits: int,
    n_envs: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    Vg_trait_init: Optional[Tensor] = None,
    Vg_env_init: Optional[Tensor] = None,
    Ve_init: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, float, list[dict]]:
    """LBFGS-autograd REML with separable Kronecker Vg constraint.

    Parameterizes Vg = Vg_trait(d,d) kron Vg_env(E,E) using Cholesky factors,
    with autograd computing exact gradients through ``torch.kron()``.

    Parameters
    ----------
    Y_rot : (n, dE) rotated phenotype matrix (trait-major column order).
    X0_rot : (n, c) rotated covariates.
    eigenvalues : (n,) GRM eigenvalues.
    n_traits : d — number of traits.
    n_envs : E — number of environments.
    max_iter : max outer LBFGS iterations.
    tol : convergence tolerance on relative LL change.
    Vg_trait_init : (d, d) initial trait genetic covariance.
    Vg_env_init : (E, E) initial environment genetic covariance.
    Ve_init : (dE, dE) initial residual covariance.

    Returns
    -------
    Vg_trait : (d, d)
    Vg_env : (E, E)
    Ve : (dE, dE)
    ll : float — best REML log-likelihood
    trace : list[dict]
    """
    n = Y_rot.shape[0]
    d = n_traits
    E = n_envs
    dE = d * E
    c = X0_rot.shape[1]
    df = n - c
    device = Y_rot.device

    if Y_rot.ndim == 1:
        Y_rot = Y_rot.unsqueeze(1)
    assert Y_rot.shape[1] == dE, f"Expected {dE} columns, got {Y_rot.shape[1]}"

    # --- Initialize Cholesky factors ---
    var_y = Y_rot.var(dim=0).mean().item()

    if Vg_trait_init is not None:
        Lt = torch.linalg.cholesky(Vg_trait_init)
    else:
        Lt = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    if Vg_env_init is not None:
        Le = torch.linalg.cholesky(Vg_env_init)
    else:
        Le = _init_cholesky_factor(E, scale=1.0, device=device)

    if Ve_init is not None:
        Lve = torch.linalg.cholesky(Ve_init)
    else:
        Lve = _init_cholesky_factor(dE, scale=var_y * 0.5, device=device)

    # Make autograd parameters
    Lt_param = Lt.clone().detach().contiguous().requires_grad_(True)
    Le_param = Le.clone().detach().contiguous().requires_grad_(True)
    Lve_param = Lve.clone().detach().contiguous().requires_grad_(True)

    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg_t = None
    best_Vg_e = None
    best_Ve = None

    optimizer = torch.optim.LBFGS(
        [Lt_param, Le_param, Lve_param],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            # Enforce lower-triangular + positive diagonal
            Lt_lower = _enforce_lower_pos(Lt_param)
            Le_lower = _enforce_lower_pos(Le_param)
            Lve_lower = _enforce_lower_pos(Lve_param)

            Vg_t = Lt_lower @ Lt_lower.T  # (d, d)
            Vg_e = Le_lower @ Le_lower.T  # (E, E)
            Vg = torch.kron(Vg_t, Vg_e)   # (dE, dE)
            Ve = Lve_lower @ Lve_lower.T   # (dE, dE)

            # Per-individual covariance: Sigma_i = lambda_i * Vg + Ve
            Sigma = eigenvalues.view(n, 1, 1) * Vg.unsqueeze(0) + Ve.unsqueeze(0)
            # Small jitter for numerical stability near singular boundaries
            Sigma = Sigma + 1e-6 * torch.eye(dE, dtype=STAT_DTYPE, device=device).unsqueeze(0)
            L_sigma = torch.linalg.cholesky(Sigma)
            W = torch.cholesky_inverse(L_sigma)
            logdet_sigma = 2.0 * torch.log(
                torch.diagonal(L_sigma, dim1=-2, dim2=-1)
            ).sum(dim=-1)

            # GLS fixed effects
            XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)
            XtWX = XtWX.reshape(c * dE, c * dE)
            WY = torch.einsum('ist,is->it', W, Y_rot)
            XtWY = torch.einsum('ia,it->at', X0_rot, WY).reshape(c * dE)
            B_vec = torch.linalg.solve(XtWX, XtWY)
            B = B_vec.reshape(c, dE)

            R = Y_rot - X0_rot @ B
            WR = torch.einsum('ist,is->it', W, R)
            quad = (R * WR).sum()

            sign, logdet_XtWX = torch.linalg.slogdet(XtWX)

            neg2ll = logdet_sigma.sum() + quad + df * dE * math.log(2 * math.pi)
            if sign > 0:
                neg2ll = neg2ll + logdet_XtWX

            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            Lt_lower = _enforce_lower_pos(Lt_param)
            Le_lower = _enforce_lower_pos(Le_param)
            Lve_lower = _enforce_lower_pos(Lve_param)

            Vg_t = Lt_lower @ Lt_lower.T
            Vg_e = Le_lower @ Le_lower.T
            Ve = Lve_lower @ Lve_lower.T
            ll = -loss.item()

        trace.append({
            "iter": outer_it,
            "mode": "separable-kron-LBFGS",
            "ll": ll,
        })

        if ll > best_ll:
            best_ll = ll
            best_Vg_t = Vg_t.clone()
            best_Vg_e = Vg_e.clone()
            best_Ve = Ve.clone()

        # Convergence check
        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "Separable-Kron LBFGS converged in %d iterations, ll=%.6f",
                outer_it, ll,
            )
            break
    else:
        logger.warning(
            "Separable-Kron LBFGS did not converge in %d iterations.", max_iter,
        )

    # Scale normalization for identifiability (applied once at output):
    # Fix trace(Vg_env) = E so that Vg_t kron Vg_e is unique.
    # Must NOT be done inside the loop — modifying param.data corrupts
    # the LBFGS (s,y) Hessian approximation.
    if best_Vg_e is not None:
        scale = best_Vg_e.trace() / E
        if scale > 1e-10:
            best_Vg_e = best_Vg_e / scale
            best_Vg_t = best_Vg_t * scale

    return (
        best_Vg_t.detach(),
        best_Vg_e.detach(),
        best_Ve.detach(),
        best_ll,
        trace,
    )


def separable_kron_reml_ked(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    n_traits: int,
    n_envs: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    Vg_trait_init: Optional[Tensor] = None,
    Vg_env_init: Optional[Tensor] = None,
    Ve_trait_init: Optional[Tensor] = None,
    Ve_env_init: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float, list[dict]]:
    """Separable REML with KED diagonal likelihood — scales to dE > 1000.

    Both Vg and Ve are separable Kronecker products:
        Vg = Vg_t kron Vg_e,  Ve = Ve_t kron Ve_e

    Uses joint diagonalization (Cholesky-whiten Ve, eigendecompose Vg) to
    reduce per-individual covariance to a diagonal. Likelihood evaluation
    is O(n*dE) instead of O(n*(dE)^3).

    Parameters
    ----------
    Y_rot : (n, dE) rotated phenotype matrix (trait-major column order)
    X0_rot : (n, c) rotated covariates
    eigenvalues : (n,) GRM eigenvalues
    n_traits, n_envs : d, E dimensions
    max_iter, tol : optimization parameters
    Vg_trait_init, Vg_env_init : (d,d), (E,E) initial genetic covariance factors
    Ve_trait_init, Ve_env_init : (d,d), (E,E) initial residual covariance factors

    Returns
    -------
    Vg_trait, Vg_env, Ve_trait, Ve_env, ll, trace
    """
    from ..linalg.kronecker_eed import _joint_diag_factor

    n = Y_rot.shape[0]
    d = n_traits
    E = n_envs
    dE = d * E
    c = X0_rot.shape[1]
    df = n - c
    device = Y_rot.device

    if Y_rot.ndim == 1:
        Y_rot = Y_rot.unsqueeze(1)
    assert Y_rot.shape[1] == dE, f"Expected {dE} columns, got {Y_rot.shape[1]}"

    var_y = Y_rot.var(dim=0).mean().item()

    # Initialize Cholesky factors for Vg_t, Vg_e, Ve_t, Ve_e
    if Vg_trait_init is not None:
        Lt = torch.linalg.cholesky(
            Vg_trait_init + 1e-6 * torch.eye(d, dtype=STAT_DTYPE, device=device))
    else:
        Lt = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    if Vg_env_init is not None:
        Le = torch.linalg.cholesky(
            Vg_env_init + 1e-6 * torch.eye(E, dtype=STAT_DTYPE, device=device))
    else:
        Le = _init_cholesky_factor(E, scale=1.0, device=device)

    if Ve_trait_init is not None:
        Lvet = torch.linalg.cholesky(
            Ve_trait_init + 1e-6 * torch.eye(d, dtype=STAT_DTYPE, device=device))
    else:
        Lvet = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    if Ve_env_init is not None:
        Lvee = torch.linalg.cholesky(
            Ve_env_init + 1e-6 * torch.eye(E, dtype=STAT_DTYPE, device=device))
    else:
        Lvee = _init_cholesky_factor(E, scale=1.0, device=device)

    Lt_param = Lt.clone().detach().contiguous().requires_grad_(True)
    Le_param = Le.clone().detach().contiguous().requires_grad_(True)
    Lvet_param = Lvet.clone().detach().contiguous().requires_grad_(True)
    Lvee_param = Lvee.clone().detach().contiguous().requires_grad_(True)

    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg_t = best_Vg_e = best_Ve_t = best_Ve_e = None

    optimizer = torch.optim.LBFGS(
        [Lt_param, Le_param, Lvet_param, Lvee_param],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            Lt_lower = _enforce_lower_pos(Lt_param)
            Le_lower = _enforce_lower_pos(Le_param)
            Lvet_lower = _enforce_lower_pos(Lvet_param)
            Lvee_lower = _enforce_lower_pos(Lvee_param)

            Vg_t = Lt_lower @ Lt_lower.T   # (d, d)
            Vg_e = Le_lower @ Le_lower.T   # (E, E)
            Ve_t = Lvet_lower @ Lvet_lower.T  # (d, d)
            Ve_e = Lvee_lower @ Lvee_lower.T  # (E, E)

            # Joint diagonalization via Cholesky whitening
            # For trait factor: Le_vt^{-1} Vg_t Le_vt^{-T} = Qt Λt Qt'
            # Use eigendecomposition-based whitening with penalty fallback
            # for LBFGS line search robustness
            def _safe_whiten(Ve_mat, Vg_mat, dim):
                """Whiten Vg by Ve using eigen-based sqrt inverse."""
                Ve_sym = (Ve_mat + Ve_mat.T) / 2.0
                # Add jitter proportional to trace for numerical safety
                jit = Ve_sym.detach().abs().max() * 1e-6 + 1e-8
                Ve_sym = Ve_sym + jit * torch.eye(dim, dtype=STAT_DTYPE, device=device)
                evals_ve, evecs_ve = torch.linalg.eigh(Ve_sym)
                evals_ve = evals_ve.clamp(min=1e-6)
                # Ve^{-1/2} = Q diag(1/sqrt(evals)) Q'
                inv_sqrt = evecs_ve * (1.0 / evals_ve.sqrt()).unsqueeze(0)
                Ve_inv_sqrt = inv_sqrt @ evecs_ve.T  # (dim, dim)
                # Whitened Vg
                Vg_w = Ve_inv_sqrt @ Vg_mat @ Ve_inv_sqrt.T
                Vg_w = (Vg_w + Vg_w.T) / 2.0
                lam, Q = torch.linalg.eigh(Vg_w)
                lam = lam.flip(0).clamp(min=0.0)
                Q = Q.flip(1)
                # Transform: T = Ve^{-1/2} @ Q (same role as Le^{-T} @ Q)
                T = Ve_inv_sqrt @ Q
                return lam, T

            try:
                lam_t, Tt_loc = _safe_whiten(Ve_t, Vg_t, d)
                lam_e, Te_loc = _safe_whiten(Ve_e, Vg_e, E)
            except (torch._C._LinAlgError, RuntimeError):
                # Return large penalty to steer optimizer away
                loss = torch.tensor(1e10, dtype=STAT_DTYPE, device=device,
                                    requires_grad=True)
                loss.backward()
                return loss

            # Transforms already computed by _safe_whiten
            Tt = Tt_loc  # (d, d)
            Te = Te_loc  # (E, E)

            # Rotate Y into KED basis: Y_ked = (Tt kron Te)' Y_rot
            Y3 = Y_rot.reshape(n, d, E)
            Y3 = torch.einsum('td,nde->nte', Tt.T, Y3)
            Y3 = torch.einsum('ef,ndf->nde', Te.T, Y3)
            Y_ked = Y3.reshape(n, dE)

            # Diagonal covariance: Sigma_diag[i, a*E+b] = lam_K[i]*lam_t[a]*lam_e[b] + 1
            Sigma_diag = (
                eigenvalues.view(n, 1, 1)
                * lam_t.view(1, d, 1)
                * lam_e.view(1, 1, E)
                + 1.0
            ).reshape(n, dE).clamp(min=1e-20)

            W_diag = 1.0 / Sigma_diag  # (n, dE)
            logdet_sigma = torch.log(Sigma_diag).sum(dim=1)  # (n,)

            # GLS with diagonal W — decoupled per trait-env dim
            WY = W_diag * Y_ked  # (n, dE)
            XtWY = X0_rot.T @ WY  # (c, dE)

            XtWX_diag = torch.einsum('na,nj,nb->jab', X0_rot, W_diag, X0_rot)  # (dE, c, c)
            B_blocks = torch.linalg.solve(
                XtWX_diag, XtWY.T.unsqueeze(2))  # (dE, c, 1)
            B = B_blocks.squeeze(2)  # (dE, c)

            R_ked = Y_ked - X0_rot @ B.T  # (n, dE)
            WR = W_diag * R_ked  # (n, dE)
            quad = (R_ked * WR).sum()

            # logdet(XtWX) — sum of logdets of (c,c) blocks
            sign_all, logdet_blocks = torch.linalg.slogdet(XtWX_diag)
            logdet_XtWX = logdet_blocks[sign_all > 0].sum()

            neg2ll = (logdet_sigma.sum() + quad
                      + df * dE * math.log(2 * math.pi)
                      + logdet_XtWX)

            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            Lt_lower = _enforce_lower_pos(Lt_param)
            Le_lower = _enforce_lower_pos(Le_param)
            Lvet_lower = _enforce_lower_pos(Lvet_param)
            Lvee_lower = _enforce_lower_pos(Lvee_param)

            Vg_t = Lt_lower @ Lt_lower.T
            Vg_e = Le_lower @ Le_lower.T
            Ve_t = Lvet_lower @ Lvet_lower.T
            Ve_e = Lvee_lower @ Lvee_lower.T
            ll = -loss.item()

        trace.append({
            "iter": outer_it,
            "mode": "separable-kron-KED-LBFGS",
            "ll": ll,
        })

        if ll > best_ll:
            best_ll = ll
            best_Vg_t = Vg_t.clone()
            best_Vg_e = Vg_e.clone()
            best_Ve_t = Ve_t.clone()
            best_Ve_e = Ve_e.clone()

        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "Separable-Kron-KED LBFGS converged in %d iters, ll=%.6f",
                outer_it, ll,
            )
            break
    else:
        logger.warning(
            "Separable-Kron-KED LBFGS did not converge in %d iterations.",
            max_iter,
        )

    # Scale normalization
    if best_Vg_e is not None:
        scale = best_Vg_e.trace() / E
        if scale > 1e-10:
            best_Vg_e = best_Vg_e / scale
            best_Vg_t = best_Vg_t * scale

    return (
        best_Vg_t.detach(),
        best_Vg_e.detach(),
        best_Ve_t.detach(),
        best_Ve_e.detach(),
        best_ll,
        trace,
    )
