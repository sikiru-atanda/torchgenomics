"""PCG-based REML for sparse GRM path.

Estimates variance components (sig2_g, sig2_e) without eigendecomposing the
covariance matrix. Uses PCG solves for V^{-1}y and V^{-1}X, and stochastic
trace/log-determinant estimators for the REML log-likelihood.

This enables LMM analysis with sparse GRMs (fastGWA-style) at biobank scale,
where eigendecomposition of the full n x n matrix is infeasible.

The REML log-likelihood (up to constants) is:
    -2 LL_REML = log|V| + log|X^T V^{-1} X| + y^T P y
where V = sig2_g * K + sig2_e * I, P = V^{-1} - V^{-1}X(X^T V^{-1}X)^{-1}X^T V^{-1}

We optimize over lambda = sig2_g / sig2_e using Brent's method.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..models.base import NullFit
from .pcg_solver import diagonal_preconditioner, pcg_solve
from .stochastic_trace import stochastic_logdet

logger = logging.getLogger(__name__)


def sparse_reml_fit(
    Y: Tensor,
    X0: Tensor,
    K_matvec: Callable[[Tensor], Tensor],
    K_diag: Tensor,
    n: int,
    config: NumericalConfig | None = None,
    n_probes: int = 30,
    lanczos_iters: int = 50,
    pcg_tol: float = 1e-8,
    pcg_max_iter: int = 500,
    seed: int | None = None,
) -> NullFit:
    """REML estimation via PCG + stochastic trace/logdet.

    Optimizes lambda = sig2_g / sig2_e via grid search + Brent refinement,
    where each lambda evaluation uses PCG solves and stochastic estimators.

    Parameters
    ----------
    Y : Tensor, shape (n,)
        Phenotype vector.
    X0 : Tensor, shape (n, c)
        Covariate matrix (intercept as first column).
    K_matvec : Callable
        Function computing K @ x for sparse GRM K.
    K_diag : Tensor, shape (n,)
        Diagonal of K (for Jacobi preconditioner).
    n : int
        Number of samples.
    config : NumericalConfig, optional
        Numerical configuration.
    n_probes : int
        Number of stochastic probes for trace/logdet estimation.
    lanczos_iters : int
        Lanczos iterations for log-determinant estimation.
    pcg_tol : float
        PCG convergence tolerance.
    pcg_max_iter : int
        Maximum PCG iterations.
    seed : int, optional
        Random seed for stochastic estimators.

    Returns
    -------
    NullFit
        Fitted null model with sig2_g, sig2_e, log-likelihood.
    """
    if config is None:
        config = NumericalConfig()

    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    K_diag = K_diag.to(STAT_DTYPE)
    device = Y.device
    c = X0.shape[1]

    if Y.ndim == 2:
        Y = Y.squeeze(1)

    def _make_V_matvec(lam: float) -> Callable[[Tensor], Tensor]:
        """V(lambda) @ x = lambda * K @ x + x."""
        def _mv(x: Tensor) -> Tensor:
            return lam * K_matvec(x) + x
        return _mv

    def _make_precond(lam: float) -> Callable[[Tensor], Tensor]:
        """Jacobi preconditioner for V(lambda) = lambda * K + I."""
        V_diag = lam * K_diag + 1.0
        return diagonal_preconditioner(V_diag)

    def _eval_reml_neg2ll(lam: float) -> float:
        """Evaluate -2 * REML log-likelihood at given lambda.

        -2 LL = n*log(sig2_e) + log|V/sig2_e| + log|X^T V^{-1} X / sig2_e^c|
                + y^T P y / sig2_e

        where V/sig2_e = lambda*K + I, and we work in the sig2_e=1 parameterization
        then recover sig2_e = y^T P y / (n - c).
        """
        V_matvec = _make_V_matvec(lam)
        precond = _make_precond(lam)

        # PCG solve: V^{-1} y
        result_y = pcg_solve(V_matvec, Y, precond=precond, tol=pcg_tol, max_iter=pcg_max_iter)
        Vinv_y = result_y.x

        # PCG solve: V^{-1} X (column by column)
        Vinv_X = torch.zeros_like(X0)
        for j in range(c):
            result_xj = pcg_solve(V_matvec, X0[:, j], precond=precond, tol=pcg_tol, max_iter=pcg_max_iter)
            Vinv_X[:, j] = result_xj.x

        # P y = V^{-1} y - V^{-1} X (X^T V^{-1} X)^{-1} X^T V^{-1} y
        XtVinvX = X0.T @ Vinv_X  # (c, c)
        XtVinvy = X0.T @ Vinv_y  # (c,)
        try:
            beta0 = torch.linalg.solve(XtVinvX, XtVinvy)
        except torch.linalg.LinAlgError:
            return float("inf")

        Py = Vinv_y - Vinv_X @ beta0

        # y^T P y (in sig2_e=1 parameterization)
        yPy = float((Y * Py).sum())

        # REML estimate of sig2_e
        sig2_e_hat = yPy / (n - c)
        if sig2_e_hat <= 0:
            return float("inf")

        # log|V/sig2_e| via stochastic Lanczos quadrature
        logdet_V = stochastic_logdet(
            V_matvec, n, n_probes=n_probes, lanczos_iters=lanczos_iters,
            seed=seed, device=device,
        )

        # log|X^T V^{-1} X| (small c x c matrix — exact)
        sign, logdet_XVX = torch.linalg.slogdet(XtVinvX)
        if sign.item() <= 0:
            return float("inf")
        logdet_XVX_val = float(logdet_XVX)

        # -2 LL_REML (up to constant)
        neg2ll = (
            (n - c) * math.log(sig2_e_hat)
            + logdet_V
            + logdet_XVX_val
            + (n - c)  # from y^T P y / sig2_e = (n-c)
        )

        return neg2ll

    # Grid search over lambda values
    log_lambdas = torch.linspace(-5, 5, 11).tolist()
    best_ll = float("inf")
    best_lam = 1.0

    for log_lam in log_lambdas:
        lam = 10.0 ** log_lam
        try:
            ll = _eval_reml_neg2ll(lam)
            if ll < best_ll:
                best_ll = ll
                best_lam = lam
        except Exception:
            continue

    # Brent refinement around the best grid point
    from scipy.optimize import minimize_scalar

    log_best = math.log10(best_lam)
    bracket_lo = max(log_best - 1.0, -10.0)
    bracket_hi = min(log_best + 1.0, 10.0)

    def _obj(log_lam: float) -> float:
        try:
            return _eval_reml_neg2ll(10.0 ** log_lam)
        except Exception:
            return float("inf")

    result = minimize_scalar(_obj, bounds=(bracket_lo, bracket_hi), method="bounded",
                             options={"xatol": 1e-4, "maxiter": 20})
    opt_lam = 10.0 ** result.x
    opt_ll = result.fun

    if opt_ll < best_ll:
        best_lam = opt_lam
        best_ll = opt_ll

    # Final solve at optimal lambda to get sig2_e, sig2_g, Py
    V_matvec = _make_V_matvec(best_lam)
    precond = _make_precond(best_lam)

    result_y = pcg_solve(V_matvec, Y, precond=precond, tol=pcg_tol, max_iter=pcg_max_iter)
    Vinv_y = result_y.x

    Vinv_X = torch.zeros_like(X0)
    for j in range(c):
        result_xj = pcg_solve(V_matvec, X0[:, j], precond=precond, tol=pcg_tol, max_iter=pcg_max_iter)
        Vinv_X[:, j] = result_xj.x

    XtVinvX = X0.T @ Vinv_X
    XtVinvy = X0.T @ Vinv_y
    beta0 = torch.linalg.solve(XtVinvX, XtVinvy)
    Py = Vinv_y - Vinv_X @ beta0

    yPy = float((Y * Py).sum())
    sig2_e = yPy / (n - c)
    sig2_g = sig2_e * best_lam

    logger.info(
        "Sparse REML: lambda=%.4e, sig2_g=%.6e, sig2_e=%.6e, h2=%.4f, -2ll=%.4f",
        best_lam, sig2_g, sig2_e,
        sig2_g / (sig2_g + sig2_e) if (sig2_g + sig2_e) > 0 else 0.0,
        best_ll,
    )

    return NullFit(
        sig2_g=sig2_g,
        sig2_e=sig2_e,
        log_likelihood=-best_ll / 2.0,
        converged=result_y.converged,
        device=device,
        # Store PCG-solved quantities for score test
        Y_rot=Py,  # Reuse Y_rot field for P @ y
        X0_rot=Vinv_X,  # Reuse X0_rot field for V^{-1} @ X
        b0=beta0,
        approximate=True,
        approx_method="sparse",
        approx_config={
            "lambda": best_lam,
            "n_probes": n_probes,
            "lanczos_iters": lanczos_iters,
            "pcg_tol": pcg_tol,
        },
    )
