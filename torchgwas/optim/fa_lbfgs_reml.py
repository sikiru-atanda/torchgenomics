"""FA(k) REML optimizer: Factor Analytic structure on Vg via L-BFGS with autograd.

Constrains the genetic covariance matrix to Factor Analytic form:

    Vg = Lambda @ Lambda^T + diag(psi)

where Lambda is an E x k lower-triangular loading matrix (for identifiability)
and psi is an E-vector of environment-specific variances.  This reduces the
number of genetic-covariance parameters from E(E+1)/2 to E*k - k(k-1)/2 + E,
which is critical for multi-environment trials where E >> k.

The parameterization follows ASReml-R conventions:
- Lambda diagonal entries are stored as log(L_jj) -> exp() for positivity
- Lambda sub-diagonal entries are unconstrained
- psi entries are stored as log(psi_i) -> exp() for positivity
- Ve is parameterized via full Cholesky factor (same as lbfgs_reml.py)

References:
    - Smith et al. (2001) — FA models for multi-environment trials
    - Thompson et al. (2003) — ASReml FA implementation
    - Cullis et al. (2010) — FA selection in plant breeding
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


def _build_loading_map(n_envs: int, fa_rank: int) -> list[tuple[int, int, bool]]:
    """Build column-major lower-triangular index map for Lambda.

    Returns a list of (row, col, is_diagonal) tuples enumerating the free
    parameters of an E x k lower-triangular matrix in column-major order.

    The number of free parameters is E*k - k(k-1)/2.
    """
    loading_map: list[tuple[int, int, bool]] = []
    for j in range(fa_rank):  # column
        for i in range(j, n_envs):  # row from diagonal downward
            loading_map.append((i, j, i == j))
    return loading_map


def _init_fa_from_vg(
    Vg_init: Tensor,
    n_envs: int,
    fa_rank: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Project Vg_init onto the FA(k) manifold via truncated eigendecomposition.

    Returns (Lambda, psi) where Lambda is E x k lower-triangular with positive
    diagonal (via QR rotation) and psi = max(diag(Vg) - diag(LL^T), 1e-6).
    """
    eigvals, eigvecs = torch.linalg.eigh(Vg_init)
    # Take top-k (eigh returns ascending order)
    top_vals = eigvals[-fa_rank:].flip(0)  # descending
    top_vecs = eigvecs[:, -fa_rank:].flip(1)  # corresponding vectors

    # Scale eigenvectors by sqrt(eigenvalues) to form initial loadings
    top_vals_clamped = top_vals.clamp(min=1e-10)
    Lambda_raw = top_vecs * top_vals_clamped.sqrt().unsqueeze(0)  # (E, k)

    # QR rotation to get lower-triangular with positive diagonal
    # torch.linalg.qr on Lambda_raw^T gives Q, R where Lambda_raw^T = Q @ R
    # so Lambda_raw = R^T @ Q^T, and R^T is lower-triangular
    Q, R = torch.linalg.qr(Lambda_raw.T)  # Q: (k, k), R: (k, E)
    # Make diagonal of R positive
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1.0
    R = signs.unsqueeze(1) * R
    Q = Q * signs.unsqueeze(0)
    Lambda = R.T  # (E, k), lower-triangular structure from the QR factorization

    # Zero out upper triangle explicitly
    Lambda = torch.tril(Lambda)

    # Ensure positive diagonal
    for j in range(fa_rank):
        if Lambda[j, j] < 0:
            Lambda[:, j] = -Lambda[:, j]

    psi_vals = torch.diag(Vg_init) - torch.diag(Lambda @ Lambda.T)
    psi_vals = psi_vals.clamp(min=1e-6)

    return Lambda.to(dtype=STAT_DTYPE, device=device), psi_vals.to(dtype=STAT_DTYPE, device=device)


def _init_fa_default(
    var_y: float,
    n_envs: int,
    fa_rank: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Default 70/30 variance split initialization for FA(k).

    Factor share (70% of variance) is split equally across k factors.
    Specific variance (30%) is uniform across environments.
    """
    factor_share = 0.7 * var_y / fa_rank

    Lambda = torch.zeros(n_envs, fa_rank, dtype=STAT_DTYPE, device=device)
    for j in range(fa_rank):
        # Diagonal: sqrt of per-factor variance share
        Lambda[j, j] = math.sqrt(factor_share)
        # Off-diagonal (below diagonal): small perturbation
        for i in range(j + 1, n_envs):
            Lambda[i, j] = 0.01 * math.sqrt(factor_share)

    psi = torch.full((n_envs,), 0.3 * var_y, dtype=STAT_DTYPE, device=device)

    return Lambda, psi


def _lambda_to_params(
    Lambda: Tensor,
    psi: Tensor,
    loading_map: list[tuple[int, int, bool]],
) -> tuple[Tensor, Tensor]:
    """Convert Lambda matrix and psi vector to unconstrained parameter vectors.

    Diagonal entries of Lambda are stored as log(value).
    Off-diagonal entries are stored as-is.
    psi entries are stored as log(value).
    """
    n_loading_params = len(loading_map)
    loading_params = torch.zeros(n_loading_params, dtype=STAT_DTYPE, device=Lambda.device)

    for idx, (row, col, is_diag) in enumerate(loading_map):
        if is_diag:
            loading_params[idx] = torch.log(Lambda[row, col].clamp(min=1e-6))
        else:
            loading_params[idx] = Lambda[row, col]

    log_psi = torch.log(psi.clamp(min=1e-10))

    return loading_params, log_psi


def fa_lbfgs_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    n_envs: int,
    fa_rank: int = 1,
    max_iter: int = 150,
    tol: float = 1e-6,
    Vg_init: Optional[Tensor] = None,
    Ve_init: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, float, list[dict], Tensor, Tensor]:
    """FA(k) REML optimizer using L-BFGS with autograd.

    Fits a multi-environment linear mixed model where the genetic covariance
    Vg is constrained to Factor Analytic structure: Vg = Lambda @ Lambda^T + diag(psi).
    The residual covariance Ve is unstructured (full Cholesky parameterization).

    Parameters
    ----------
    Y_rot : (n, E) — rotated phenotype matrix (n individuals, E environments).
    X0_rot : (n, c) — rotated covariate matrix.
    eigenvalues : (n,) — GRM eigenvalues from spectral decomposition.
    n_envs : int — number of environments E (must match Y_rot.shape[1]).
    fa_rank : int — rank k of the factor analytic structure (default 1).
    max_iter : int — maximum outer L-BFGS iterations (default 150).
    tol : float — convergence tolerance on relative log-likelihood change.
    Vg_init : (E, E) optional — initial genetic covariance (projected onto FA manifold).
    Ve_init : (E, E) optional — initial residual covariance.

    Returns
    -------
    Vg : (E, E) — estimated genetic covariance (Lambda @ Lambda^T + diag(psi)).
    Ve : (E, E) — estimated residual covariance.
    loglik : float — final REML log-likelihood.
    trace : list[dict] — per-iteration optimizer trace.
    Lambda : (E, k) — estimated factor loading matrix (lower-triangular).
    psi : (E,) — estimated specific variance vector.

    Raises
    ------
    ValueError
        If fa_rank >= n_envs (FA rank must be strictly less than the number
        of environments for the model to be identifiable and reduced-rank).
    """
    # --- Validation ---
    if fa_rank >= n_envs:
        raise ValueError(
            f"FA rank ({fa_rank}) must be strictly less than the number of "
            f"environments ({n_envs}). Use full-rank lbfgs_reml() instead."
        )

    E = n_envs
    k = fa_rank
    n = Y_rot.shape[0]
    c = X0_rot.shape[1]
    d = E  # environments are the "traits" dimension
    df = n - c
    device = Y_rot.device

    if Y_rot.ndim == 1:
        Y_rot = Y_rot.unsqueeze(1)

    # --- Build loading map ---
    loading_map = _build_loading_map(E, k)

    # --- Initialize FA parameters ---
    if Vg_init is not None:
        Lambda_init, psi_init = _init_fa_from_vg(Vg_init, E, k, device)
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Lambda_init, psi_init = _init_fa_default(var_y, E, k, device)

    # Convert to unconstrained parameters
    loading_params_init, log_psi_init = _lambda_to_params(
        Lambda_init, psi_init, loading_map
    )

    loading_params = loading_params_init.clone().detach().contiguous().requires_grad_(True)
    log_psi_param = log_psi_init.clone().detach().contiguous().requires_grad_(True)

    # --- Initialize Ve via Cholesky ---
    if Ve_init is not None:
        Le = torch.linalg.cholesky(
            Ve_init + torch.eye(d, dtype=STAT_DTYPE, device=device) * 1e-6
        )
    else:
        var_y = Y_rot.var(dim=0).mean().item()
        Le = torch.eye(d, dtype=STAT_DTYPE, device=device) * math.sqrt(max(var_y * 0.5, 1e-6))

    Le_param = Le.clone().detach().contiguous().requires_grad_(True)

    # --- Optimizer setup ---
    trace: list[dict] = []
    best_ll = float("-inf")
    best_Vg: Optional[Tensor] = None
    best_Ve: Optional[Tensor] = None
    best_Lambda: Optional[Tensor] = None
    best_psi: Optional[Tensor] = None

    optimizer = torch.optim.LBFGS(
        [loading_params, log_psi_param, Le_param],
        max_iter=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=1e-10,
    )

    # --- Outer iteration loop ---
    for outer_it in range(max_iter):

        def closure():
            optimizer.zero_grad()

            # Reconstruct Lambda from parameters
            Lambda = torch.zeros(E, k, dtype=STAT_DTYPE, device=device)
            idx = 0
            for row, col, is_diag in loading_map:
                if is_diag:
                    Lambda[row, col] = torch.exp(
                        loading_params[idx].clamp(-14.0, 4.6)
                    )  # bounds: [~1e-6, ~100]
                else:
                    Lambda[row, col] = loading_params[idx].clamp(-100.0, 100.0)
                idx += 1

            psi = torch.exp(
                log_psi_param.clamp(-23.0, 14.0)
            )  # bounds: [~1e-10, ~1e6]

            Vg = Lambda @ Lambda.T + torch.diag(psi)

            # Ve from Cholesky (same as lbfgs_reml)
            Le_lower = torch.tril(Le_param)
            Le_lower = Le_lower.clone()
            diag_e = torch.diag(Le_lower)
            Le_lower = Le_lower - torch.diag(diag_e) + torch.diag(
                diag_e.abs().clamp(min=1e-6)
            )
            Ve = Le_lower @ Le_lower.T

            # --- REML objective (identical structure to lbfgs_reml) ---
            # Per-individual covariance: Sigma_i = evals[i] * Vg + Ve
            Sigma = eigenvalues.view(n, 1, 1) * Vg.unsqueeze(0) + Ve.unsqueeze(0)

            L_sigma = torch.linalg.cholesky(Sigma)
            W = torch.cholesky_inverse(L_sigma)  # (n, d, d)

            logdet_sigma = 2.0 * torch.log(
                torch.diagonal(L_sigma, dim1=-2, dim2=-1)
            ).sum(dim=-1)  # (n,)

            # GLS fixed effects
            XtWX = torch.einsum("ia,ist,ib->asbt", X0_rot, W, X0_rot)
            XtWX = XtWX.reshape(c * d, c * d)

            WY = torch.einsum("ist,is->it", W, Y_rot)
            XtWY = torch.einsum("ia,it->at", X0_rot, WY).reshape(c * d)

            B_vec = torch.linalg.solve(XtWX, XtWY)
            B = B_vec.reshape(c, d)

            R = Y_rot - X0_rot @ B
            WR = torch.einsum("ist,is->it", W, R)
            quad = (R * WR).sum()

            sign, logdet_XtWX = torch.linalg.slogdet(XtWX)

            neg2ll = logdet_sigma.sum() + quad + df * d * math.log(2.0 * math.pi)
            if sign > 0:
                neg2ll = neg2ll + logdet_XtWX

            loss = 0.5 * neg2ll
            loss.backward()
            return loss

        loss = optimizer.step(closure)

        # --- Extract current estimates (no grad) ---
        with torch.no_grad():
            Lambda = torch.zeros(E, k, dtype=STAT_DTYPE, device=device)
            idx = 0
            for row, col, is_diag in loading_map:
                if is_diag:
                    Lambda[row, col] = torch.exp(
                        loading_params[idx].clamp(-14.0, 4.6)
                    )
                else:
                    Lambda[row, col] = loading_params[idx].clamp(-100.0, 100.0)
                idx += 1

            psi = torch.exp(log_psi_param.clamp(-23.0, 14.0))
            Vg = Lambda @ Lambda.T + torch.diag(psi)

            Le_lower = torch.tril(Le_param)
            Ve = Le_lower @ Le_lower.T

            ll = -loss.item()

        trace.append({
            "iter": outer_it,
            "mode": "FA-LBFGS-autograd",
            "ll": ll,
            "fa_rank": k,
        })

        if ll > best_ll:
            best_ll = ll
            best_Vg = Vg.clone()
            best_Ve = Ve.clone()
            best_Lambda = Lambda.clone()
            best_psi = psi.clone()

        # Check convergence
        if outer_it > 0 and abs(ll - trace[-2]["ll"]) < tol * max(abs(ll), 1.0):
            logger.info(
                "FA(%d) LBFGS-autograd converged in %d outer iterations, ll=%.6f",
                k, outer_it, ll,
            )
            break
    else:
        logger.warning(
            "FA(%d) LBFGS-autograd did not converge in %d outer iterations.",
            k, max_iter,
        )

    return (
        best_Vg.detach(),
        best_Ve.detach(),
        best_ll,
        trace,
        best_Lambda.detach(),
        best_psi.detach(),
    )
