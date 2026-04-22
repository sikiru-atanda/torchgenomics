"""GxE LBFGS-autograd REML with 3 variance components: Vg, Vge, Ve.

Extends the standard LBFGS REML (2 components) with a third GxE interaction
variance component. Per-individual covariance in the rotated eigenspace:

    Sigma_i = evals[i] * Vg + env_cov[i] * evals[i] * Vge + Ve

where env_cov[i] = z_i^2 for continuous environment variables.

Uses log-Cholesky parameterization (3 lower-triangular factors) to guarantee
positive-definiteness of all variance components.

References:
    - StructLMM (Moore et al., Nat Genet 2019)
    - MAGEE (Wang et al.)
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def _init_cholesky_factor(d: int, scale: float = 1.0, device=None) -> Tensor:
    """Initialize a lower-triangular Cholesky factor."""
    L = torch.eye(d, dtype=STAT_DTYPE, device=device) * math.sqrt(max(scale, 1e-6))
    return L


def gxe_lbfgs_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    env_cov: Tensor,
    *,
    n_traits: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
    Vg_init: Tensor | None = None,
    Vge_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, float, list[dict]]:
    """LBFGS-autograd REML with 3 variance components: Vg, Vge, Ve.

    Parameters
    ----------
    Y_rot : (n, d) rotated phenotype matrix.
    X0_rot : (n, c) rotated covariates.
    eigenvalues : (n,) GRM eigenvalues.
    env_cov : (n,) per-individual environment covariance weight (e.g. z_i^2).
    n_traits : int — number of traits d.
    max_iter : int — max outer LBFGS iterations.
    tol : float — convergence tolerance on relative LL change.
    Vg_init, Vge_init, Ve_init : (d, d) initial covariance matrices.

    Returns
    -------
    Vg : (d, d) estimated genetic covariance.
    Vge : (d, d) estimated GxE interaction covariance.
    Ve : (d, d) estimated residual covariance.
    ll : float — final REML log-likelihood.
    trace : list[dict] — optimizer trace.
    """
    n = Y_rot.shape[0]
    d = n_traits if Y_rot.ndim == 1 else Y_rot.shape[1]
    c = X0_rot.shape[1]
    df = n - c
    device = Y_rot.device

    if Y_rot.ndim == 1:
        Y_rot = Y_rot.unsqueeze(1)

    # Initialize Cholesky factors
    if Vg_init is not None:
        Lg = torch.linalg.cholesky(
            Vg_init + torch.eye(d, dtype=STAT_DTYPE, device=device) * 1e-6
        )
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Lg = _init_cholesky_factor(d, scale=var_y * 0.4, device=device)

    if Vge_init is not None:
        Vge_safe = Vge_init + torch.eye(d, dtype=STAT_DTYPE, device=device) * 1e-6
        Lge = torch.linalg.cholesky(Vge_safe)
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Lge = _init_cholesky_factor(d, scale=var_y * 0.1, device=device)

    if Ve_init is not None:
        Le = torch.linalg.cholesky(
            Ve_init + torch.eye(d, dtype=STAT_DTYPE, device=device) * 1e-6
        )
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Le = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    # Parameters for autograd
    Lg_param = Lg.clone().detach().contiguous().requires_grad_(True)
    Lge_param = Lge.clone().detach().contiguous().requires_grad_(True)
    Le_param = Le.clone().detach().contiguous().requires_grad_(True)

    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg = None
    best_Vge = None
    best_Ve = None

    optimizer = torch.optim.LBFGS(
        [Lg_param, Lge_param, Le_param],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    evals = eigenvalues  # (n,)
    ec = env_cov  # (n,)

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            # Enforce lower-triangular with positive diagonal
            Lg_lower = torch.tril(Lg_param)
            Lge_lower = torch.tril(Lge_param)
            Le_lower = torch.tril(Le_param)

            Lg_lower = Lg_lower.clone()
            Lge_lower = Lge_lower.clone()
            Le_lower = Le_lower.clone()

            diag_g = torch.diag(Lg_lower)
            diag_ge = torch.diag(Lge_lower)
            diag_e = torch.diag(Le_lower)

            Lg_lower = Lg_lower - torch.diag(diag_g) + torch.diag(diag_g.abs().clamp(min=1e-6))
            Lge_lower = Lge_lower - torch.diag(diag_ge) + torch.diag(diag_ge.abs().clamp(min=1e-6))
            Le_lower = Le_lower - torch.diag(diag_e) + torch.diag(diag_e.abs().clamp(min=1e-6))

            Vg = Lg_lower @ Lg_lower.T
            Vge = Lge_lower @ Lge_lower.T
            Ve = Le_lower @ Le_lower.T

            # Per-individual covariance:
            # Sigma_i = evals[i]*Vg + env_cov[i]*evals[i]*Vge + Ve
            Sigma = (evals.view(n, 1, 1) * Vg.unsqueeze(0) +
                     (ec * evals).view(n, 1, 1) * Vge.unsqueeze(0) +
                     Ve.unsqueeze(0))  # (n, d, d)

            L_sigma = torch.linalg.cholesky(Sigma)
            W = torch.cholesky_inverse(L_sigma)  # (n, d, d)

            logdet_sigma = 2.0 * torch.log(
                torch.diagonal(L_sigma, dim1=-2, dim2=-1)
            ).sum(dim=-1)  # (n,)

            # GLS fixed effects
            XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)
            XtWX = XtWX.reshape(c * d, c * d)
            WY = torch.einsum('ist,is->it', W, Y_rot)
            XtWY = torch.einsum('ia,it->at', X0_rot, WY).reshape(c * d)
            B_vec = torch.linalg.solve(XtWX, XtWY)
            B = B_vec.reshape(c, d)

            R = Y_rot - X0_rot @ B
            WR = torch.einsum('ist,is->it', W, R)
            quad = (R * WR).sum()

            sign, logdet_XtWX = torch.linalg.slogdet(XtWX)

            neg2ll = logdet_sigma.sum() + quad + df * d * math.log(2 * math.pi)
            if sign > 0:
                neg2ll = neg2ll + logdet_XtWX

            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            Lg_lower = torch.tril(Lg_param)
            Lge_lower = torch.tril(Lge_param)
            Le_lower = torch.tril(Le_param)
            Vg = Lg_lower @ Lg_lower.T
            Vge = Lge_lower @ Lge_lower.T
            Ve = Le_lower @ Le_lower.T
            ll = -loss.item()

        trace.append({
            "iter": outer_it,
            "mode": "GxE-LBFGS-autograd",
            "ll": ll,
        })

        if ll > best_ll:
            best_ll = ll
            best_Vg = Vg.clone()
            best_Vge = Vge.clone()
            best_Ve = Ve.clone()

        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "GxE LBFGS-autograd converged in %d outer iterations, ll=%.6f",
                outer_it, ll,
            )
            break
    else:
        logger.warning(
            "GxE LBFGS-autograd did not converge in %d outer iterations.", max_iter,
        )

    return (
        best_Vg.detach(),
        best_Vge.detach(),
        best_Ve.detach(),
        best_ll,
        trace,
    )
