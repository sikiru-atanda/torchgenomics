"""LBFGS-autograd REML for multiple kernel variance components.

Optimizes V = Σ_k σ²_k K_k + σ²_e I with log-parameterization so all
variance components remain positive by construction.  The full V matrix
is assembled each iteration — O(n³) per likelihood evaluation (no
eigendecomposition trick, since multiple kernels are not simultaneously
diagonalizable).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from ..linalg.safe import safe_cholesky

logger = logging.getLogger(__name__)


def multikernel_reml(
    Y: Tensor,
    X0: Tensor,
    kernels: list[Tensor],
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    init_vars: Optional[list[float]] = None,
) -> tuple[list[float], float, list[dict]]:
    """LBFGS-autograd REML for K scalar variance components.

    Maximises the REML log-likelihood:
        ℓ = -½ [log|V| + log|X₀ᵀV⁻¹X₀| + (Y-X₀β̂)ᵀ V⁻¹ (Y-X₀β̂) + df·log(2π)]

    where V = Σ_k exp(θ_k) K_k + exp(θ_e) I  and θ are unconstrained.

    Parameters
    ----------
    Y : (n,) phenotype vector.
    X0 : (n, c) covariate matrix (intercept as first column).
    kernels : list of (n, n) kernel matrices.
    max_iter : maximum outer LBFGS iterations.
    tol : convergence tolerance on relative LL change.
    init_vars : initial variance component values [σ²_1, ..., σ²_K, σ²_e].
        Length must be len(kernels) + 1.  If None, initialised from Y variance.

    Returns
    -------
    variances : list[float]
        Estimated [σ²_1, ..., σ²_K, σ²_e].
    log_likelihood : float
        Final REML log-likelihood.
    trace : list[dict]
        Optimizer trace with keys {"iter", "mode", "ll"}.
    """
    Y = Y.to(STAT_DTYPE).squeeze()
    X0 = X0.to(STAT_DTYPE)
    kernels = [K.to(STAT_DTYPE) for K in kernels]

    n = Y.shape[0]
    c = X0.shape[1]
    df = n - c
    n_kernels = len(kernels)
    device = Y.device

    # --- Initialise log-variance parameters ---
    if init_vars is not None:
        assert len(init_vars) == n_kernels + 1, (
            f"init_vars length {len(init_vars)} != n_kernels+1={n_kernels + 1}"
        )
        log_vars = [math.log(max(v, 1e-8)) for v in init_vars]
    else:
        var_y = Y.var().item()
        share = var_y / (n_kernels + 1)
        log_vars = [math.log(max(share, 1e-8))] * (n_kernels + 1)

    # Unconstrained parameters: [log(σ²_1), ..., log(σ²_K), log(σ²_e)]
    theta = torch.tensor(log_vars, dtype=STAT_DTYPE, device=device, requires_grad=True)

    trace: list[dict] = []
    best_ll = float("-inf")
    best_vars: list[float] = []

    optimizer = torch.optim.LBFGS(
        [theta],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()
            vars_ = torch.exp(theta)  # (n_kernels + 1,)

            # Assemble V = Σ_k σ²_k K_k + σ²_e I
            V = vars_[-1] * torch.eye(n, dtype=STAT_DTYPE, device=device)
            for k in range(n_kernels):
                V = V + vars_[k] * kernels[k]

            # Cholesky for log-determinant and solve
            L_V = torch.linalg.cholesky(V)
            logdet_V = 2.0 * torch.log(torch.diagonal(L_V)).sum()

            # V⁻¹ X₀ and V⁻¹ Y via Cholesky solve
            V_inv_X0 = torch.cholesky_solve(X0, L_V)
            V_inv_Y = torch.cholesky_solve(Y.unsqueeze(1), L_V).squeeze(1)

            # GLS fixed effects: β̂ = (X₀ᵀ V⁻¹ X₀)⁻¹ X₀ᵀ V⁻¹ Y
            XtVinvX = X0.T @ V_inv_X0
            XtVinvY = X0.T @ V_inv_Y
            beta = torch.linalg.solve(XtVinvX, XtVinvY)

            # Residual quadratic form
            r = Y - X0 @ beta
            V_inv_r = torch.cholesky_solve(r.unsqueeze(1), L_V).squeeze(1)
            quad = (r * V_inv_r).sum()

            # log|X₀ᵀ V⁻¹ X₀|
            sign, logdet_XtVinvX = torch.linalg.slogdet(XtVinvX)
            logdet_term = logdet_XtVinvX if sign > 0 else torch.tensor(0.0, dtype=STAT_DTYPE, device=device)

            # Negative REML log-likelihood (× 2)
            neg2ll = logdet_V + logdet_term + quad + df * math.log(2.0 * math.pi)
            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            ll = -loss.item()
            current_vars = torch.exp(theta).detach().tolist()

        trace.append({
            "iter": outer_it,
            "mode": "LBFGS-multikernel",
            "ll": ll,
        })

        if ll > best_ll:
            best_ll = ll
            best_vars = current_vars

        # Convergence check
        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "Multi-kernel REML converged in %d iterations, ll=%.6f",
                outer_it, ll,
            )
            break
    else:
        logger.warning(
            "Multi-kernel REML did not converge in %d iterations.", max_iter,
        )

    logger.info(
        "Multi-kernel REML: %d kernels, variances=%s, ll=%.6f",
        n_kernels,
        [f"{v:.4f}" for v in best_vars],
        best_ll,
    )

    return best_vars, best_ll, trace
