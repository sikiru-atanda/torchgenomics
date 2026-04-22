"""Multi-mediator gene-set joint Wald mediation (Phase 49b).

Given a mediator set M of size k (e.g. all genes in a pathway), we fit:

    Stage M (multivariate): M = SNP * a + X0 * A0 + V,   a in R^k
    Stage Y:                Y = SNP * c' + M * b + X0 * g + u,  b in R^k

in the K-eigenspace with rotated WLS weights ``w = 1 / (lambda * sig2_g + sig2_e)``
sourced from a single null fit of Y on X0 alone (matching ``mediate_lmm``). The
per-mediator indirect effect is ``ind_j = a_j * b_j`` and its delta-method
covariance (block-diagonal joint covariance of (a, b)) is

    Cov(ind) = diag(b) Cov(a) diag(b) + diag(a) Cov(b) diag(a).

The joint Wald statistic ``ind.T @ Cov(ind)^-1 @ ind`` is chi^2_k under H0 that
every element of the indirect vector is zero. We also return the scalar
set-level summary ``sum_indirect = sum(a_j * b_j)`` with its delta-method SE.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..linalg.eigh import rotate
from ._mediate import _as_tensor, fit_mediation_null
from ._types import GeneSetMediationResult


def _wls_multi(X: Tensor, Y: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """Weighted least squares with (possibly) multi-column RHS.

    Returns (beta, cov_beta) where beta has shape (p,) if Y is 1-D or (p, d)
    if Y is (n, d), and cov_beta is the (p, p) parameter covariance (shared
    across RHS columns when residual variance has already been folded into w).
    """
    wX = w.unsqueeze(1) * X
    XtWX = X.T @ wX
    XtWy = wX.T @ Y
    beta = torch.linalg.solve(XtWX, XtWy)
    cov_beta = torch.linalg.inv(XtWX)
    return beta, cov_beta


def _chi2_sf(x: float, df: int) -> float:
    """Chi-squared survival function via the regularised upper incomplete gamma."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:  # pragma: no cover - scipy is a hard dep elsewhere
        # Fallback: normal approximation (rough but never used in practice).
        import math
        z = (x - df) / math.sqrt(max(2 * df, 1))
        return 0.5 * math.erfc(z / math.sqrt(2))


def mediate_gene_set(
    Y,
    SNP,
    M_set,
    K,
    *,
    covariates=None,
    seed: int | None = None,  # retained for API symmetry with mediate_lmm
) -> GeneSetMediationResult:
    """Multi-mediator joint-Wald mediation for a mediator set.

    Parameters
    ----------
    Y : array-like (n,)
    SNP : array-like (n,)
    M_set : array-like (n, k) — the mediator set (one column per feature).
    K : array-like (n, n) GRM.
    covariates : array-like (n, c) or None.
    """
    del seed  # unused; present to keep the API shape aligned with mediate_lmm
    Y_t = _as_tensor(Y).reshape(-1)
    SNP_t = _as_tensor(SNP).reshape(-1)
    M_t = _as_tensor(M_set)
    K_t = _as_tensor(K)
    n = Y_t.shape[0]
    if M_t.ndim == 1:
        M_t = M_t.unsqueeze(1)
    if M_t.shape[0] != n:
        raise ValueError(f"M_set must have {n} rows; got {tuple(M_t.shape)}.")
    if SNP_t.shape[0] != n or K_t.shape != (n, n):
        raise ValueError("Shape mismatch among Y, SNP, K.")
    k = M_t.shape[1]

    nf = fit_mediation_null(Y_t, K_t, covariates=covariates)
    U = nf.eigenvectors
    sig2_g = float(nf.sig2_g)
    sig2_e = float(nf.sig2_e)
    w = 1.0 / (nf.eigenvalues * sig2_g + sig2_e).clamp(min=1e-20)

    SNP_r = rotate(SNP_t.unsqueeze(1), U).squeeze(1)
    M_r = rotate(M_t, U)
    Y_r = nf.Y_rot.reshape(-1)
    X0_r = nf.X0_rot

    # --- Stage M (multivariate): M = SNP * a + X0 * A0 + V ---
    Xm = torch.cat([SNP_r.unsqueeze(1), X0_r], dim=1)
    beta_m, cov_m = _wls_multi(Xm, M_r, w)  # beta_m: (p_m, k), cov_m: (p_m, p_m)
    a = beta_m[0, :].clone()
    # Cov(a) across mediators: Cov(a_j, a_l) = cov_m[0, 0] * Cov_across_features(residuals).
    # Under the usual two-stage assumption that Stage-M noise is independent of
    # SNP-adjusted Y residuals, and treating each mediator's Stage-M residual
    # variance as absorbed into the shared weights, the cross-mediator Cov(a)
    # equals cov_m[0, 0] * R where R is the residual correlation across
    # mediators. We estimate R from rotated residuals.
    resid_m = M_r - Xm @ beta_m  # (n, k)
    # Weighted covariance of residuals across mediators.
    wM = w.unsqueeze(1) * resid_m
    denom = float(max(n - Xm.shape[1], 1))
    R_resid = (resid_m.T @ wM) / denom  # (k, k), in-weight residual covariance
    # Scale residual covariance so its diagonal matches each mediator's own Stage-M
    # variance estimator: weighted SSR / (n - p_m). Equivalent to R_resid directly.
    a_cov = float(cov_m[0, 0]) * R_resid

    # --- Stage Y: Y = SNP * c' + M * b + X0 * g + u ---
    Xy = torch.cat([SNP_r.unsqueeze(1), M_r, X0_r], dim=1)
    beta_y, cov_y = _wls_multi(Xy, Y_r, w)
    # Rescale cov_y by weighted residual variance (unlike Stage M, Y is univariate
    # so we can pull the scale factor out directly).
    resid_y = Y_r - Xy @ beta_y
    sse = float((w * resid_y * resid_y).sum().item())
    dof = max(n - Xy.shape[1], 1)
    sigma2_y = sse / dof
    cov_y_scaled = cov_y * sigma2_y
    b = beta_y[1:1 + k].clone()
    b_cov = cov_y_scaled[1:1 + k, 1:1 + k].clone()
    # Similarly rescale a_cov by Stage-M per-mediator variance already baked
    # into R_resid; no double scaling needed.

    # --- Indirect vector and delta-method covariance ---
    indirect = a * b
    diag_a = torch.diag(a)
    diag_b = torch.diag(b)
    indirect_cov = diag_b @ a_cov @ diag_b + diag_a @ b_cov @ diag_a

    # Symmetrise for numerical stability.
    indirect_cov = 0.5 * (indirect_cov + indirect_cov.T)

    # --- Joint Wald chi^2 ---
    rank_deficient = False
    try:
        solved = torch.linalg.solve(indirect_cov, indirect.unsqueeze(1)).squeeze(1)
        chi2_stat = float((indirect @ solved).item())
    except Exception:
        rank_deficient = True
        # Moore-Penrose pseudoinverse fallback.
        pinv = torch.linalg.pinv(indirect_cov)
        chi2_stat = float((indirect @ (pinv @ indirect)).item())
    chi2_stat = max(chi2_stat, 0.0)
    df = k
    pvalue = _chi2_sf(chi2_stat, df)

    # --- Scalar set summary: sum_indirect ---
    ones = torch.ones(k, dtype=indirect.dtype)
    sum_indirect = float(indirect.sum().item())
    var_sum = float((ones @ (indirect_cov @ ones)).item())
    sum_indirect_se = float(torch.sqrt(torch.tensor(max(var_sum, 0.0))).item())
    if sum_indirect_se > 0.0:
        z = sum_indirect / sum_indirect_se
        import math
        sum_p = math.erfc(abs(z) / math.sqrt(2.0))
    else:
        sum_p = float("nan")

    return GeneSetMediationResult(
        a=a,
        a_cov=a_cov,
        b=b,
        b_cov=b_cov,
        indirect=indirect,
        indirect_cov=indirect_cov,
        chi2=chi2_stat,
        df=df,
        pvalue=pvalue,
        sum_indirect=sum_indirect,
        sum_indirect_se=sum_indirect_se,
        sum_indirect_pvalue=sum_p,
        n=n,
        k=k,
        rank_deficient=rank_deficient,
    )
