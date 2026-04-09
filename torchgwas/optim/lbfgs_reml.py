"""Mode C: LBFGS on autograd REML log-likelihood (Cholesky parameterization).

Novel to TorchGWAS: torch.autograd computes exact gradients of the REML
objective w.r.t. Cholesky factors — no hand-derived score equations needed.
Cholesky parameterization ensures Vg, Ve stay positive definite by construction.

Scales to large d (10+ traits) without implementing the AI matrix.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .mvlmm_reml import compute_sigma_inv

logger = logging.getLogger(__name__)


def _cholesky_to_spd(L: Tensor) -> Tensor:
    """Convert lower-triangular L to SPD matrix L @ L^T."""
    return L @ L.T


def _init_cholesky_factor(d: int, scale: float = 1.0, device: torch.device = None) -> Tensor:
    """Initialize a lower-triangular Cholesky factor."""
    L = torch.eye(d, dtype=STAT_DTYPE, device=device) * math.sqrt(scale)
    return L


def lbfgs_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    n_traits: int = 1,
    max_iter: int = 100,
    tol: float = 1e-6,
    Vg_init: Optional[Tensor] = None,
    Ve_init: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, float, list[dict]]:
    """LBFGS optimization of multi-trait REML on Cholesky factors.

    Parameters
    ----------
    Y_rot : (n, d) — rotated phenotype matrix
    X0_rot : (n, c) — rotated covariates
    eigenvalues : (n,) — GRM eigenvalues
    n_traits : int — number of traits d
    max_iter : int — max LBFGS iterations
    tol : float — convergence tolerance
    Vg_init, Ve_init : (d, d) initial covariance matrices

    Returns
    -------
    Vg : (d, d) — estimated genetic covariance
    Ve : (d, d) — estimated residual covariance
    ll : float — final REML log-likelihood
    trace : list[dict] — optimizer trace
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
        Lg = torch.linalg.cholesky(Vg_init)
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Lg = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    if Ve_init is not None:
        Le = torch.linalg.cholesky(Ve_init)
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Le = _init_cholesky_factor(d, scale=var_y * 0.5, device=device)

    # Make parameters for autograd
    # We optimize the lower-triangular entries of Lg and Le
    # .contiguous() needed because cholesky output may not be contiguous
    Lg_param = Lg.clone().detach().contiguous().requires_grad_(True)
    Le_param = Le.clone().detach().contiguous().requires_grad_(True)

    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg = None
    best_Ve = None

    optimizer = torch.optim.LBFGS(
        [Lg_param, Le_param],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            # Enforce lower-triangular structure
            Lg_lower = torch.tril(Lg_param)
            Le_lower = torch.tril(Le_param)

            # Ensure positive diagonal
            Lg_lower = Lg_lower.clone()
            Le_lower = Le_lower.clone()
            diag_g = torch.diag(Lg_lower)
            diag_e = torch.diag(Le_lower)
            Lg_lower = Lg_lower - torch.diag(diag_g) + torch.diag(diag_g.abs().clamp(min=1e-6))
            Le_lower = Le_lower - torch.diag(diag_e) + torch.diag(diag_e.abs().clamp(min=1e-6))

            Vg = Lg_lower @ Lg_lower.T
            Ve = Le_lower @ Le_lower.T

            # Compute negative REML log-likelihood
            Sigma = eigenvalues.view(n, 1, 1) * Vg.unsqueeze(0) + Ve.unsqueeze(0)
            L_sigma = torch.linalg.cholesky(Sigma)
            W = torch.cholesky_inverse(L_sigma)
            logdet_sigma = 2.0 * torch.log(
                torch.diagonal(L_sigma, dim1=-2, dim2=-1)
            ).sum(dim=-1)

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

            loss = 0.5 * neg2ll  # minimize -ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            Lg_lower = torch.tril(Lg_param)
            Le_lower = torch.tril(Le_param)
            Vg = Lg_lower @ Lg_lower.T
            Ve = Le_lower @ Le_lower.T
            ll = -loss.item()

        trace.append({
            "iter": outer_it,
            "mode": "LBFGS-autograd",
            "ll": ll,
        })

        if ll > best_ll:
            best_ll = ll
            best_Vg = Vg.clone()
            best_Ve = Ve.clone()

        # Check convergence
        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "LBFGS-autograd converged in %d outer iterations, ll=%.6f", outer_it, ll,
            )
            break
    else:
        logger.warning("LBFGS-autograd did not converge in %d outer iterations.", max_iter)

    return best_Vg.detach(), best_Ve.detach(), best_ll, trace
