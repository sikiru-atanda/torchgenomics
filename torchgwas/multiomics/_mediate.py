"""Single-triple GRM-corrected mediation: SNP → M → Y."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..linalg.eigh import rotate
from ..models.base import NullFit
from ..models.single_trait_lmm import SingleTraitLMM
from ._se import bootstrap_se, monte_carlo_se, sobel_se
from ._sensitivity import imai_rho_sensitivity
from ._types import MediationResult

logger = logging.getLogger("torchgwas.multiomics")


def _as_tensor(x, dtype=torch.float64) -> Tensor:
    t = torch.as_tensor(x, dtype=dtype)
    return t.detach().cpu()


def _wls(X: Tensor, y: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Weighted least squares in the rotated eigenspace.

    Returns ``(beta, var_beta_diag, residuals)`` where ``var_beta_diag`` is the
    diagonal of (X^T W X)^{-1} — already calibrated because ``w`` is the inverse
    covariance of the rotated residuals. ``residuals`` are in rotated space.
    """
    wX = w.unsqueeze(1) * X
    XtWX = X.T @ wX
    XtWy = wX.T @ y
    beta = torch.linalg.solve(XtWX, XtWy)
    cov_beta = torch.linalg.inv(XtWX)
    var_beta = torch.diagonal(cov_beta)
    resid = y - X @ beta
    return beta, var_beta, resid


def _fit_two_stage(
    Y_r: Tensor,
    SNP_r: Tensor,
    M_r: Tensor,
    X0_r: Tensor,
    w: Tensor,
) -> dict[str, float]:
    """Run the three rotated WLS fits sharing a single weight vector."""
    n = Y_r.shape[0]

    # Stage M: M ~ SNP + X0
    Xm = torch.cat([SNP_r.unsqueeze(1), X0_r], dim=1)
    beta_m, var_m, resid_m = _wls(Xm, M_r, w)
    a = float(beta_m[0])
    var_a = float(var_m[0])
    sigma_v = float(torch.sqrt((w * resid_m * resid_m).sum() / max(n - Xm.shape[1], 1)).clamp(min=1e-12))

    # Stage Y total: Y ~ SNP + X0
    Xc = torch.cat([SNP_r.unsqueeze(1), X0_r], dim=1)
    beta_c, var_c, _ = _wls(Xc, Y_r, w)
    c = float(beta_c[0])
    var_c_ = float(var_c[0])

    # Stage Y direct: Y ~ SNP + M + X0
    Xy = torch.cat([SNP_r.unsqueeze(1), M_r.unsqueeze(1), X0_r], dim=1)
    beta_y, var_y, resid_y = _wls(Xy, Y_r, w)
    c_prime = float(beta_y[0])
    var_cp = float(var_y[0])
    b = float(beta_y[1])
    var_b = float(var_y[1])
    sigma_u = float(torch.sqrt((w * resid_y * resid_y).sum() / max(n - Xy.shape[1], 1)).clamp(min=1e-12))

    return {
        "a": a, "var_a": var_a,
        "b": b, "var_b": var_b,
        "c": c, "var_c": var_c_,
        "c_prime": c_prime, "var_c_prime": var_cp,
        "sigma_v": sigma_v, "sigma_u": sigma_u,
    }


def fit_mediation_null(Y, K, covariates=None) -> NullFit:
    """Fit the shared null model on Y once; reuse across many (SNP, M) triples.

    The returned NullFit carries the eigendecomposition of K, the rotated Y / X0,
    and the variance components needed to build WLS weights for both stages.
    """
    Y_t = _as_tensor(Y).reshape(-1)
    K_t = _as_tensor(K)
    n = Y_t.shape[0]
    intercept = torch.ones(n, 1, dtype=torch.float64)
    if covariates is None:
        X0 = intercept
    else:
        X0 = torch.cat([intercept, _as_tensor(covariates).reshape(n, -1)], dim=1)
    return SingleTraitLMM().fit_null(Y_t, X0, K_t)


def _mediate_from_nullfit(
    nf: NullFit,
    SNP: Tensor,
    M: Tensor,
    *,
    se: str,
    n_mc_draws: int,
    n_boot: int,
    sensitivity: bool,
    seed: int | None,
) -> MediationResult:
    U = nf.eigenvectors
    SNP_r = rotate(SNP, U).squeeze()
    M_r = rotate(M, U).squeeze()
    Y_r = nf.Y_rot.squeeze()
    X0_r = nf.X0_rot
    sig2_g = float(nf.sig2_g)
    sig2_e = float(nf.sig2_e)
    w = 1.0 / (nf.eigenvalues * sig2_g + sig2_e).clamp(min=1e-20)
    n = Y_r.shape[0]

    fit_out = _fit_two_stage(Y_r, SNP_r, M_r, X0_r, w)
    a, b = fit_out["a"], fit_out["b"]
    var_a, var_b = fit_out["var_a"], fit_out["var_b"]
    c, c_prime = fit_out["c"], fit_out["c_prime"]
    indirect = a * b
    total = c

    if se == "sobel":
        s, ci_l, ci_u, p = sobel_se(a, b, var_a, var_b)
    elif se == "monte-carlo":
        s, ci_l, ci_u, p = monte_carlo_se(a, b, var_a, var_b, cov_ab=0.0,
                                          n_draws=n_mc_draws, seed=seed)
    elif se == "bootstrap":
        Y_np = Y_r.numpy()  # already rotated; bootstrap on rotated rows
        SNP_np = SNP_r.numpy()
        M_np = M_r.numpy()
        X_np = X0_r.numpy()
        w_np = w.numpy()

        def refit(idx: np.ndarray) -> tuple[float, float]:
            Xs = X_np[idx]
            snp = SNP_np[idx]
            m = M_np[idx]
            y = Y_np[idx]
            ww = np.sqrt(w_np[idx])
            Xm = np.column_stack([snp, Xs]) * ww[:, None]
            ym = m * ww
            beta_m, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            a_b = float(beta_m[0])
            Xy = np.column_stack([snp, m, Xs]) * ww[:, None]
            yy = y * ww
            beta_y, *_ = np.linalg.lstsq(Xy, yy, rcond=None)
            b_b = float(beta_y[1])
            return a_b, b_b

        s, ci_l, ci_u, p = bootstrap_se(refit, n=n, n_boot=n_boot, seed=seed)
    else:
        raise ValueError(f"se must be 'sobel', 'monte-carlo', or 'bootstrap'; got {se!r}")

    inconsistent = (c_prime != 0.0) and (indirect != 0.0) and ((c_prime > 0) != (indirect > 0))
    if total == 0.0 or inconsistent:
        proportion = float("nan")
    else:
        proportion = indirect / total

    rho = imai_rho_sensitivity(b, fit_out["sigma_v"], fit_out["sigma_u"]) if sensitivity else None

    return MediationResult(
        a=a, a_se=float(np.sqrt(max(var_a, 0.0))),
        b=b, b_se=float(np.sqrt(max(var_b, 0.0))),
        c=c, c_se=float(np.sqrt(max(fit_out["var_c"], 0.0))),
        c_prime=c_prime, c_prime_se=float(np.sqrt(max(fit_out["var_c_prime"], 0.0))),
        indirect=indirect, indirect_se=s,
        indirect_ci_lower=ci_l, indirect_ci_upper=ci_u,
        indirect_pvalue=p,
        total=total,
        proportion_mediated=proportion,
        inconsistent=inconsistent,
        sensitivity_rho=rho,
        n=n,
        se_method=se,
        fit_method="two-stage",
    )


def mediate_lmm(
    Y,
    SNP,
    M,
    K,
    *,
    covariates=None,
    fit: str = "two-stage",
    se: str = "monte-carlo",
    n_mc_draws: int = 10_000,
    n_boot: int = 1_000,
    sensitivity: bool = True,
    seed: Optional[int] = None,
) -> MediationResult:
    """GRM-corrected causal mediation for a single (SNP, mediator, outcome) triple.

    Two-stage fit (default) uses one eigendecomposition of K shared across both
    stages and a single set of variance-component estimates (from the null
    model on Y) that calibrates the WLS weights for both equations.

    Parameters
    ----------
    Y, SNP, M : array-like (n,)
    K : array-like (n, n) GRM
    covariates : array-like (n, c) or None
    fit : {"two-stage", "joint"} — joint not yet supported in Phase 49.
    se : {"sobel", "monte-carlo", "bootstrap"}
    """
    if fit not in ("two-stage", "joint"):
        raise ValueError(f"fit must be 'two-stage' or 'joint'; got {fit!r}")
    if fit == "joint":
        raise NotImplementedError(
            "fit='joint' is deferred to Phase 49b; use 'two-stage' for now."
        )

    Y_t = _as_tensor(Y).reshape(-1)
    SNP_t = _as_tensor(SNP).reshape(-1)
    M_t = _as_tensor(M).reshape(-1)
    K_t = _as_tensor(K)
    n = Y_t.shape[0]
    if SNP_t.shape[0] != n or M_t.shape[0] != n or K_t.shape != (n, n):
        raise ValueError(
            f"Shape mismatch: Y={Y_t.shape}, SNP={SNP_t.shape}, M={M_t.shape}, K={K_t.shape}."
        )

    nf = fit_mediation_null(Y_t, K_t, covariates=covariates)
    return _mediate_from_nullfit(
        nf, SNP_t, M_t,
        se=se, n_mc_draws=n_mc_draws, n_boot=n_boot,
        sensitivity=sensitivity, seed=seed,
    )
