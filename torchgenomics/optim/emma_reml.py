"""EMMA-style grid-search REML for GAPIT compatibility.

Implements the variance-component estimation used by GAPIT's EMMA
(Kang et al., 2008): grid search over log10(delta) from -10 to +10
with 100 points, followed by Brent's method refinement on the
REML score equation.

delta = sig2_e / sig2_g = 1 / lambda (EMMA convention)
lambda = sig2_g / sig2_e           (GEMMA convention)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.optimize import brentq
from torch import Tensor

from .reml_math import reml_derivatives, reml_loglikelihood

logger = logging.getLogger(__name__)


def gapit_emma_remle(
    Y: Tensor,
    X0: Tensor,
    K: Tensor,
    *,
    n_grid: int = 100,
    log_delta_min: float = -10.0,
    log_delta_max: float = 10.0,
) -> tuple[float, float, float, float, list[dict]]:
    """GAPIT-exact EMMA REML using restricted eigendecomposition.

    Matches GAPIT's ``GAPIT.emma.REMLE`` + ``emma.eigen.R.wo.Z`` exactly:
    1. Compute S = I - X(X'X)^{-1}X' (projection away from fixed effects)
    2. Eigendecompose S(K+I)S → restricted eigenvalues (n-q values)
    3. Grid search over log(delta) with natural exponential
    4. Refine via uniroot (Brent's method)

    Parameters
    ----------
    Y : (n,) — phenotype (NOT rotated)
    X0 : (n, c) — covariates (NOT rotated)
    K : (n, n) — kinship matrix

    Returns
    -------
    sig2_g, sig2_e, ll, delta, trace
    """
    Y = Y.to(torch.float64).squeeze()
    X0 = X0.to(torch.float64)
    K = K.to(torch.float64)
    n = Y.shape[0]
    q = X0.shape[1]

    trace: list[dict] = []

    # --- Step 1: Restricted eigendecomposition (emma.eigen.R.wo.Z) ---
    # S = I - X(X'X)^{-1}X'
    XtX = X0.T @ X0
    XtX_inv = torch.linalg.inv(XtX)
    S = torch.eye(n, dtype=torch.float64, device=K.device) - X0 @ XtX_inv @ X0.T

    # Eigendecompose S(K+I)S
    K_plus_I = K + torch.eye(n, dtype=torch.float64, device=K.device)
    SKS = S @ K_plus_I @ S
    SKS = (SKS + SKS.T) / 2.0  # Symmetrize

    eig_vals, eig_vecs = torch.linalg.eigh(SKS)
    # Reverse to descending order
    eig_vals = eig_vals.flip(0)
    eig_vecs = eig_vecs.flip(1)

    # Take top n-q eigenvalues, subtract 1 (GAPIT convention)
    nq = n - q
    restricted_evals = eig_vals[:nq] - 1.0
    restricted_evecs = eig_vecs[:, :nq]

    # Clamp negative restricted eigenvalues to 0 (GAPIT behavior)
    restricted_evals = torch.clamp(restricted_evals, min=0.0)

    # Compute etas = eigvecs_R' @ y
    etas = restricted_evecs.T @ Y  # (nq,)
    etas_np = etas.detach().cpu().numpy().astype(np.float64)
    lambda_np = restricted_evals.detach().cpu().numpy().astype(np.float64)

    # --- Step 2: Grid search over log(delta) ---
    # GAPIT: logdelta = (0:ngrids)/ngrids * (ulim - llim) + llim
    n_grid_pts = n_grid + 1
    log_deltas = np.linspace(log_delta_min, log_delta_max, n_grid_pts)
    deltas = np.exp(log_deltas)

    # Vectorized REML LL and derivative (matching GAPIT's formulas exactly)
    # LL = 0.5 * (nq*(log(nq/(2π)) - 1 - log(Σ η²/(λ+δ))) - Σ log(λ+δ))
    # dLL = 0.5 * δ * (nq*Σ(η²/(λ+δ)²)/Σ(η²/(λ+δ)) - Σ(1/(λ+δ)))

    etasq = etas_np * etas_np  # (nq,)

    lls = np.zeros(n_grid_pts)
    dls = np.zeros(n_grid_pts)

    for i, delta in enumerate(deltas):
        ldelta = lambda_np + delta  # (nq,)
        etasq_over_ldelta = etasq / ldelta
        sum_etasq_ldelta = etasq_over_ldelta.sum()

        ll_i = 0.5 * (nq * (np.log(nq / (2 * np.pi)) - 1
                             - np.log(sum_etasq_ldelta))
                       - np.log(ldelta).sum())
        lls[i] = ll_i

        dl_i = 0.5 * (nq * (etasq / (ldelta * ldelta)).sum()
                       / sum_etasq_ldelta
                       - (1.0 / ldelta).sum())
        dls[i] = dl_i

        trace.append({
            "mode": "gapit_emma_grid", "iter": i,
            "delta": float(delta), "ll": float(ll_i), "dl": float(dl_i),
        })

    best_grid = int(np.argmax(lls))
    best_ll = lls[best_grid]
    best_delta = deltas[best_grid]

    logger.info(
        "GAPIT EMMA grid: best at grid[%d], delta=%.6e, ll=%.6f",
        best_grid, best_delta, best_ll,
    )

    # --- Step 3: Refine with uniroot/Brent (matching GAPIT exactly) ---
    # GAPIT searches for sign changes in dLL (derivative w.r.t. delta)
    # and refines in log(delta) space using uniroot.

    def _reml_dll_logdelta(logdelta_val):
        """GAPIT's emma.delta.REML.dLL.wo.Z in log(delta) space."""
        d = np.exp(logdelta_val)
        ld = lambda_np + d
        return float(0.5 * (nq * (etasq / (ld * ld)).sum()
                            / (etasq / ld).sum()
                            - (1.0 / ld).sum()))

    def _reml_ll_logdelta(logdelta_val):
        """GAPIT's emma.delta.REML.LL.wo.Z."""
        d = np.exp(logdelta_val)
        ld = lambda_np + d
        return float(0.5 * (nq * (np.log(nq / (2 * np.pi)) - 1
                                   - np.log((etasq / ld).sum()))
                            - np.log(ld).sum()))

    opt_logdeltas = []
    opt_lls = []

    # Check boundary conditions (matching GAPIT)
    esp = 1e-10  # GAPIT default
    if dls[0] < esp:
        opt_logdeltas.append(log_delta_min)
        opt_lls.append(_reml_ll_logdelta(log_delta_min))

    if dls[-1] > -esp:
        opt_logdeltas.append(log_delta_max)
        opt_lls.append(_reml_ll_logdelta(log_delta_max))

    # Find sign changes (GAPIT: positive-to-negative only)
    for i in range(n_grid_pts - 1):
        if dls[i] * dls[i + 1] < 0 and dls[i] > 0 and dls[i + 1] < 0:
            try:
                from scipy.optimize import brentq
                root = brentq(
                    _reml_dll_logdelta,
                    log_deltas[i], log_deltas[i + 1],
                    xtol=1e-12, maxiter=100,
                )
                opt_logdeltas.append(root)
                opt_lls.append(_reml_ll_logdelta(root))
            except (ValueError, RuntimeError) as e:
                logger.debug("Brent refinement failed: %s", e)

    if opt_lls:
        best_idx = int(np.argmax(opt_lls))
        best_delta = np.exp(opt_logdeltas[best_idx])
        best_ll = opt_lls[best_idx]

    # --- Step 4: Extract variance components ---
    # sig2_g = Σ η²/(λ_R + δ) / (n-q)  (profiled genetic variance)
    # sig2_e = sig2_g * δ
    ld_final = lambda_np + best_delta
    sig2_g = float((etasq / ld_final).sum() / nq)
    sig2_e = sig2_g * best_delta

    logger.info(
        "GAPIT EMMA REMLE: sig2_g=%.6e, sig2_e=%.6e, delta=%.6e, ll=%.6f",
        sig2_g, sig2_e, best_delta, best_ll,
    )

    trace.append({
        "mode": "gapit_emma_final",
        "delta": float(best_delta),
        "sig2_g": sig2_g, "sig2_e": sig2_e,
        "ll": float(best_ll),
    })

    return sig2_g, sig2_e, float(best_ll), float(best_delta), trace


def emma_reml_single(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    n_grid: int = 100,
    log_delta_min: float = -10.0,
    log_delta_max: float = 10.0,
    tol: float = 1e-8,
    max_refine_iter: int = 100,
) -> tuple[float, float, float, list[dict]]:
    """EMMA-style REML estimation via grid search + Brent refinement.

    Parameters
    ----------
    Y_rot, X0_rot, eigenvalues : rotated data from eigendecomposition
    n_grid : number of grid points (EMMA default: 100)
    log_delta_min, log_delta_max : search range for log10(delta)
    tol : tolerance for Brent's method
    max_refine_iter : max iterations for Brent's method

    Returns
    -------
    sig2_g, sig2_e, ll, trace
    """
    trace: list[dict] = []

    # --- Step 1: Grid search over log(delta) (natural log, matching GAPIT) ---
    # GAPIT: logdelta = (0:ngrids)/ngrids * (ulim - llim) + llim → 101 points
    # delta = exp(logdelta)  (natural exponential, NOT base-10)
    n_grid_pts = n_grid + 1  # GAPIT uses ngrids+1 points (0:ngrids)
    log_deltas = np.linspace(log_delta_min, log_delta_max, n_grid_pts)
    deltas = np.exp(log_deltas)  # Natural exponential (matches GAPIT)
    # lambda = 1/delta (convert EMMA convention to GEMMA convention)
    lambdas = 1.0 / deltas

    lls = np.zeros(n_grid_pts)
    dls = np.zeros(n_grid_pts)

    for i, lam in enumerate(lambdas):
        ll_i, dl_i, _ = reml_derivatives(float(lam), Y_rot, X0_rot, eigenvalues)
        lls[i] = ll_i
        dls[i] = dl_i
        trace.append({
            "mode": "emma_grid", "iter": i,
            "lambda": float(lam), "delta": float(deltas[i]),
            "ll": ll_i, "dl": dl_i,
        })

    best_grid = int(np.argmax(lls))
    best_ll = lls[best_grid]
    best_lam = lambdas[best_grid]

    logger.info(
        "EMMA grid search: best at grid[%d], delta=%.4e, lambda=%.4e, ll=%.6f",
        best_grid, deltas[best_grid], best_lam, best_ll,
    )

    # --- Step 2: Refine with Brent's method on the score equation ---
    # Find intervals where the derivative changes sign (EMMA approach)
    # The REML score dl/dlambda should cross zero at the optimum.
    # Since delta = 1/lambda, we search in delta space.
    # dl/d(log_delta) = -dl/d(log_lambda) (chain rule with sign flip)
    # EMMA searches for sign changes in dl w.r.t. log(delta).

    # The derivative w.r.t. lambda is dl. When searching in delta space,
    # the sign flips: a positive dl in lambda-space means we should
    # increase lambda (decrease delta). So we look for sign changes
    # in dl (lambda-space) and refine in lambda-space directly.

    refined = False
    refined_lam = best_lam
    refined_ll = best_ll

    # Look for sign changes in dl across the grid
    for i in range(n_grid_pts - 1):
        if dls[i] * dls[i + 1] < 0:
            # Sign change: root between lambdas[i] and lambdas[i+1]
            # Note: lambdas decrease as i increases (since deltas increase)
            lam_lo = min(lambdas[i], lambdas[i + 1])
            lam_hi = max(lambdas[i], lambdas[i + 1])

            def score_fn(lam_val):
                _, dl_val, _ = reml_derivatives(lam_val, Y_rot, X0_rot, eigenvalues)
                return dl_val

            try:
                lam_root = brentq(score_fn, lam_lo, lam_hi, xtol=tol, maxiter=max_refine_iter)
                ll_root, _, _ = reml_loglikelihood(lam_root, Y_rot, X0_rot, eigenvalues)

                if ll_root > refined_ll:
                    refined_lam = lam_root
                    refined_ll = ll_root
                    refined = True

                trace.append({
                    "mode": "emma_brent", "iter": i,
                    "lambda": lam_root, "ll": ll_root,
                    "interval": [float(lam_lo), float(lam_hi)],
                })
            except (ValueError, RuntimeError) as e:
                logger.debug("Brent refinement failed for interval [%.4e, %.4e]: %s", lam_lo, lam_hi, e)

    # --- Step 3: Extract final variance components ---
    final_ll, sig2_e, sig2_g = reml_loglikelihood(refined_lam, Y_rot, X0_rot, eigenvalues)

    logger.info(
        "EMMA REML: sig2_g=%.6e, sig2_e=%.6e, lambda=%.6e, ll=%.6f, refined=%s",
        sig2_g, sig2_e, refined_lam, final_ll, refined,
    )

    trace.append({
        "mode": "emma_final",
        "lambda": float(refined_lam),
        "sig2_g": sig2_g, "sig2_e": sig2_e,
        "ll": final_ll, "refined": refined,
    })

    return sig2_g, sig2_e, final_ll, trace
