"""GLM: batched ordinary least-squares scan via residualization.

Standard linear regression GWAS (no random effects): y = Xb + e.
Matches PLINK --linear for reference equivalence.

The residualization trick: project y and G onto the orthogonal complement
of X0, then the per-SNP regression reduces to a simple ratio.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import BaseModel, NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class GLM:
    """Standard linear regression GWAS (no random effects).

    Implements :class:`BaseModel`.  Fits null via OLS, scans SNPs
    with batched residualization for speed.
    """

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null model via OLS: y = X0 @ b0 + e.

        Parameters
        ----------
        Y : (n,) or (n, 1) — phenotype
        X0 : (n, c) — covariates (intercept as first column)
        K : ignored (GLM has no random effects)
        """
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)

        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n = Y.shape[0]
        c = X0.shape[1]

        # OLS: b0 = (X0^T X0)^{-1} X0^T y
        XtX = X0.T @ X0  # (c, c)
        Xty = X0.T @ Y  # (c,)
        b0 = torch.linalg.solve(XtX, Xty)  # (c,)

        # Residuals and residual variance
        resid = Y - X0 @ b0
        df = n - c
        sig2_e = (resid @ resid).item() / df

        # Projection matrix: P = I - X0 (X0^T X0)^{-1} X0^T
        # We store the pieces needed for efficient per-SNP scan
        XtX_inv = torch.linalg.inv(XtX)  # (c, c)

        # Project Y: Py = resid (already computed)
        # For SNPs: Pg = g - X0 @ (XtX_inv @ X0^T @ g) — done in score_chunk

        logger.info(
            "GLM null fit: n=%d, c=%d, sig2_e=%.6e", n, c, sig2_e,
        )

        return NullFit(
            sig2_e=sig2_e,
            Y_rot=resid,  # reuse Y_rot field for residualized Y
            X0_rot=X0,  # store X0 for projection in scan
            M00=XtX_inv,  # store (X^T X)^{-1} for projection
            b0=b0,
            log_likelihood=_ols_loglikelihood(resid, sig2_e, n, c),
            converged=True,
            device=Y.device,
        )

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score SNP chunk via residualized OLS.

        For each SNP g:
          1. Residualize: g_resid = g - X0 @ (X^T X)^{-1} @ X0^T g
          2. beta = (g_resid^T y_resid) / (g_resid^T g_resid)
          3. Var(beta) = sig2_e / (g_resid^T g_resid)
          4. Wald: T = beta^2 / Var(beta) ~ chi2(1)
        """
        if test not in ("wald",):
            raise ValueError(f"GLM supports 'wald' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        y_resid = null_fit.Y_rot  # (n,) residualized phenotype
        X0 = null_fit.X0_rot  # (n, c)
        XtX_inv = null_fit.M00  # (c, c)
        sig2_e = null_fit.sig2_e

        # Residualize genotypes: g_resid = g - X0 @ XtX_inv @ X0^T @ g
        X0tG = X0.T @ G_chunk  # (c, m)
        G_resid = G_chunk - X0 @ (XtX_inv @ X0tG)  # (n, m)

        # Per-SNP regression on residuals
        gtg = (G_resid * G_resid).sum(dim=0)  # (m,)
        gty = (G_resid * y_resid.unsqueeze(1)).sum(dim=0)  # (m,)

        gtg_safe = torch.clamp(gtg, min=1e-20)
        beta = gty / gtg_safe  # (m,)
        var_beta = sig2_e / gtg_safe  # (m,)
        se = torch.sqrt(var_beta)  # (m,)

        # F-statistic: F = beta^2 / var_beta ~ F(1, n-c-1)
        stat = beta ** 2 / var_beta  # (m,)

        # P-values from F(1, n-c-1) distribution (matches GAPIT/PLINK)
        c = X0.shape[1]
        df2 = n - c - 1
        p = _f_sf(stat, df1=1, df2=df2)

        # Allele frequencies
        af = G_chunk.mean(dim=0) / 2.0

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,
            se=se,
            stat=stat,
            p=p,
            test="wald",
        )


def _ols_loglikelihood(resid: Tensor, sig2_e: float, n: int, c: int) -> float:
    """OLS log-likelihood for diagnostics."""
    import math
    df = n - c
    ll = -0.5 * (df * math.log(2 * math.pi * sig2_e) + df)
    return ll


def _f_sf(stat: Tensor, df1: int = 1, df2: int = 1) -> Tensor:
    """P-values from F-distribution (matches GAPIT/PLINK --linear)."""
    import scipy.stats as sp_stats
    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.f.sf(stat_np, dfn=df1, dfd=df2)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
