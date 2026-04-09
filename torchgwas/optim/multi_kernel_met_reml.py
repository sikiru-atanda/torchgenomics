"""Multi-kernel MET REML optimizer: V = Σ_j K_j ⊗ Vg_j + I ⊗ Ve.

When multiple kernel matrices are provided (e.g., additive + dominance +
epistasis), the EED trick cannot be used because multiple kernels are not
simultaneously diagonalizable.  Instead, this module builds the full V
matrix and optimizes via L-BFGS with autograd on the direct REML objective.

Complexity: O(n³E³) per REML iteration — feasible for moderate n (≤ 5000)
and E (≤ 10).  The expensive V^{-1} is computed once at convergence and
reused for per-SNP GLS scanning across millions of markers.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def multi_kernel_met_reml(
    Y: Tensor,
    X0: Tensor,
    kernels: dict[str, Tensor],
    *,
    n_envs: int,
    vg_structure: str = "unstructured",
    fa_rank: int = 0,
    max_iter: int = 150,
    tol: float = 1e-6,
) -> dict:
    """Multi-kernel MET REML via direct V construction and L-BFGS.

    Parameters
    ----------
    Y : (n, E) — wide phenotype matrix.
    X0 : (n, c) — covariates.
    kernels : dict mapping kernel name → (n, n) kernel matrix.
    n_envs : number of environments E.
    vg_structure : "unstructured" or "fa" (if fa, fa_rank must be set).
    fa_rank : rank k for FA structure (0 = unstructured).
    max_iter : max L-BFGS outer iterations.
    tol : convergence tolerance.

    Returns
    -------
    dict with keys:
        "Vg_dict" : {name: (E, E) Tensor} genetic covariance per kernel
        "Ve" : (E, E) residual covariance
        "loglik" : float
        "trace" : list[dict]
        "V_inv" : (nE, nE) precomputed inverse for scanning
        "converged" : bool
    """
    n, E = Y.shape
    c = X0.shape[1]
    df = n - c
    device = Y.device
    n_kernels = len(kernels)
    kernel_names = list(kernels.keys())

    # Initialize Vg per kernel and Ve
    var_y = Y.var(dim=0).mean().item()
    share_per_kernel = 0.5 * var_y / max(n_kernels, 1)

    # Cholesky parameters for each Vg_j and Ve
    Lg_params = {}
    for name in kernel_names:
        L_init = torch.eye(E, dtype=STAT_DTYPE, device=device) * math.sqrt(share_per_kernel)
        Lg_params[name] = L_init.clone().detach().contiguous().requires_grad_(True)

    Le_init = torch.eye(E, dtype=STAT_DTYPE, device=device) * math.sqrt(0.5 * var_y)
    Le_param = Le_init.clone().detach().contiguous().requires_grad_(True)

    all_params = list(Lg_params.values()) + [Le_param]

    # Precompute Kronecker structure: each kernel contributes kron(K_j, Vg_j)
    # We vectorize Y as vec(Y^T) with column-major (environment-first) layout
    # so V = Σ_j K_j ⊗ Vg_j + I_n ⊗ Ve, operating on (nE,) vectors

    # Pre-store kernels as list for closure
    K_list = [kernels[name] for name in kernel_names]

    Y_vec = Y.T.reshape(-1)  # (nE,) — environment-major vectorization
    # Build X_kron = I_E ⊗ X0, shape (nE, cE)
    X_kron = torch.zeros(n * E, c * E, dtype=STAT_DTYPE, device=device)
    for e in range(E):
        X_kron[e * n:(e + 1) * n, e * c:(e + 1) * c] = X0

    trace: list[dict] = []
    best_ll = float("-inf")
    best_result: Optional[dict] = None

    optimizer = torch.optim.LBFGS(
        all_params,
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            # Reconstruct Vg per kernel
            V = torch.zeros(n * E, n * E, dtype=STAT_DTYPE, device=device)
            for idx, name in enumerate(kernel_names):
                Lg_lower = torch.tril(Lg_params[name])
                Lg_lower = Lg_lower.clone()
                diag_g = torch.diag(Lg_lower)
                Lg_lower = Lg_lower - torch.diag(diag_g) + torch.diag(
                    diag_g.abs().clamp(min=1e-6)
                )
                Vg_j = Lg_lower @ Lg_lower.T
                # kron(Vg_j, K_j) in environment-major layout
                V = V + torch.kron(Vg_j, K_list[idx])

            # Add Ve
            Le_lower = torch.tril(Le_param)
            Le_lower = Le_lower.clone()
            diag_e = torch.diag(Le_lower)
            Le_lower = Le_lower - torch.diag(diag_e) + torch.diag(
                diag_e.abs().clamp(min=1e-6)
            )
            Ve = Le_lower @ Le_lower.T
            V = V + torch.kron(Ve, torch.eye(n, dtype=STAT_DTYPE, device=device))

            # Cholesky factorize V
            L_V = torch.linalg.cholesky(V)
            logdet_V = 2.0 * torch.log(torch.diag(L_V)).sum()

            # GLS: solve for fixed effects
            V_inv_X = torch.linalg.solve(V, X_kron)  # (nE, cE)
            XtVinvX = X_kron.T @ V_inv_X  # (cE, cE)
            V_inv_Y = torch.linalg.solve(V, Y_vec)  # (nE,)
            XtVinvY = X_kron.T @ V_inv_Y  # (cE,)
            B_vec = torch.linalg.solve(XtVinvX, XtVinvY)

            R = Y_vec - X_kron @ B_vec
            V_inv_R = torch.linalg.solve(V, R)
            quad = (R * V_inv_R).sum()

            sign, logdet_XtVinvX = torch.linalg.slogdet(XtVinvX)

            neg2ll = logdet_V + quad + df * E * math.log(2.0 * math.pi)
            if sign > 0:
                neg2ll = neg2ll + logdet_XtVinvX

            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        with torch.no_grad():
            ll = -loss.item()

            # Extract current estimates
            Vg_dict = {}
            for name in kernel_names:
                Lg_lower = torch.tril(Lg_params[name])
                Vg_dict[name] = (Lg_lower @ Lg_lower.T).clone()

            Le_lower = torch.tril(Le_param)
            Ve = (Le_lower @ Le_lower.T).clone()

        trace.append({"iter": outer_it, "mode": "MultiKernel-LBFGS", "ll": ll})

        if ll > best_ll:
            best_ll = ll
            best_result = {
                "Vg_dict": {n: v.clone() for n, v in Vg_dict.items()},
                "Ve": Ve.clone(),
                "loglik": ll,
            }

        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "Multi-kernel MET REML converged in %d iterations, ll=%.6f",
                outer_it, ll,
            )
            break
    else:
        logger.warning(
            "Multi-kernel MET REML did not converge in %d iterations.", max_iter,
        )

    # Compute V_inv at convergence for scanning
    with torch.no_grad():
        V_final = torch.zeros(n * E, n * E, dtype=STAT_DTYPE, device=device)
        for idx, name in enumerate(kernel_names):
            V_final = V_final + torch.kron(best_result["Vg_dict"][name], K_list[idx])
        V_final = V_final + torch.kron(
            best_result["Ve"], torch.eye(n, dtype=STAT_DTYPE, device=device)
        )
        V_inv = torch.linalg.inv(V_final)

    converged = len(trace) > 0 and (
        len(trace) < max_iter
        or (
            len(trace) >= 2
            and abs(trace[-1]["ll"] - trace[-2]["ll"]) < tol * max(abs(trace[-1]["ll"]), 1.0)
        )
    )

    best_result["trace"] = trace
    best_result["V_inv"] = V_inv
    best_result["converged"] = converged
    return best_result
