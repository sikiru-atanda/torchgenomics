"""Test-statistic dispatch: chi2_sf, Wald, LRT, Score.

Centralized test-statistic functions used by the models module.
Uses scipy.stats for p-value computation (reliable tail behavior).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def chi2_sf(stat: Tensor, df: int) -> Tensor:
    """Survival function (1 - CDF) of chi-square distribution.

    Parameters
    ----------
    stat : (*,) — chi-squared test statistics
    df : int — degrees of freedom

    Returns
    -------
    p : (*,) — p-values, clamped to [1e-300, 1.0]
    """
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=stat.dtype, device=stat.device)


def wald_test(beta: Tensor, se: Tensor, df: int = 1) -> tuple[Tensor, Tensor]:
    """Wald test: stat = (beta/se)^2 ~ chi2(df).

    Parameters
    ----------
    beta : (*,) — effect estimates
    se : (*,) — standard errors
    df : int — degrees of freedom (default 1)

    Returns
    -------
    (stat, p) — test statistics and p-values
    """
    se_safe = torch.clamp(se.abs(), min=1e-20)
    stat = (beta / se_safe) ** 2
    p = chi2_sf(stat, df=df)
    return stat, p


def lrt_test(ll_null: float, ll_alt: Tensor, df: int = 1) -> tuple[Tensor, Tensor]:
    """Likelihood ratio test: stat = 2*(ll_alt - ll_null) ~ chi2(df).

    Parameters
    ----------
    ll_null : float — null log-likelihood
    ll_alt : (*,) — alternative log-likelihoods (per SNP)
    df : int — degrees of freedom (default 1)

    Returns
    -------
    (stat, p) — test statistics and p-values
    """
    stat = 2.0 * (ll_alt - ll_null)
    stat = torch.clamp(stat, min=0.0)  # LRT stat is non-negative
    p = chi2_sf(stat, df=df)
    return stat, p


def score_test(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """Score test: stat = U^2 / V ~ chi2(1).

    Parameters
    ----------
    U : (*,) — score statistics (gradient of log-likelihood at null)
    V : (*,) — variance of U under the null

    Returns
    -------
    (stat, p) — test statistics and p-values
    """
    V_safe = torch.clamp(V.abs(), min=1e-20)
    stat = U ** 2 / V_safe
    p = chi2_sf(stat, df=1)
    return stat, p


def apply_contrast(
    beta: Tensor, Var_beta: Tensor, C: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Linear contrast Wald test: H0: C @ beta = 0 with stat ~ chi2(rank(C)).

    Batched over the leading dimension of ``beta`` / ``Var_beta`` (one SNP per
    row), so a single call evaluates the contrast for an entire chunk of SNPs
    in parallel.

    Parameters
    ----------
    beta : Tensor
        ``(m, b)`` coefficient estimates per SNP.
    Var_beta : Tensor
        ``(m, b, b)`` coefficient covariance per SNP.
    C : Tensor
        ``(q, b)`` contrast matrix testing ``H0: C @ beta = 0``. Must have
        full row rank ``q ≤ b``.

    Returns
    -------
    Cbeta : Tensor
        ``(m, q)`` contrast point estimates.
    CVCt : Tensor
        ``(m, q, q)`` contrast covariance matrices.
    stat : Tensor
        ``(m,)`` Wald χ² statistics, clamped to ≥ 0.
    p : Tensor
        ``(m,)`` p-values from chi-square with ``df = q``.

    Notes
    -----
    Used by random-regression GWAS for intercept / slope / time-varying
    contrasts on basis-coefficient effects, and by reaction-norm GWAS for
    stable / contingent splits on environment effects.
    """
    Cbeta = beta @ C.T  # (m, q)
    CVCt = C @ Var_beta @ C.T  # (m, q, q)
    rhs = Cbeta.unsqueeze(-1)  # (m, q, 1)
    sol = torch.linalg.solve(CVCt, rhs).squeeze(-1)  # (m, q)
    stat = torch.clamp((Cbeta * sol).sum(dim=-1), min=0.0)
    p = chi2_sf(stat, df=int(C.shape[0]))
    return Cbeta, CVCt, stat, p


def score_test_multi_df(U: Tensor, V: Tensor, df: int) -> tuple[Tensor, Tensor]:
    """Multi-degree-of-freedom score test: stat = U' V^{-1} U ~ chi2(df).

    Parameters
    ----------
    U : (m, d) — score vectors for m SNPs, each d-dimensional
    V : (m, d, d) — variance matrices for each SNP

    Returns
    -------
    (stat, p) — (m,) test statistics and p-values
    """
    m = U.shape[0]
    stat = torch.zeros(m, dtype=U.dtype, device=U.device)
    for i in range(m):
        try:
            V_inv_U = torch.linalg.solve(V[i], U[i])
            stat[i] = torch.clamp((U[i] * V_inv_U).sum(), min=0.0)
        except Exception:
            stat[i] = 0.0
    p = chi2_sf(stat, df=df)
    return stat, p
