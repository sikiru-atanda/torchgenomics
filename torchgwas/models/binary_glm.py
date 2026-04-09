"""Binary GLM: logistic regression GWAS with score test and optional SPA.

Implements logistic regression for case-control (binary 0/1) phenotypes.
Null model fitted via IRLS (iteratively reweighted least squares).
SNP scanning uses the efficient score test with optional saddlepoint
approximation (SPA) for calibrated tail p-values under case-control imbalance.

Optionally applies Firth's penalized likelihood (Jeffreys prior) to
stabilise the null fit when separation is detected.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import BaseModel, NullFit, ScanResult, VariantMeta
from .glm_link import LogitLink

logger = logging.getLogger(__name__)


class BinaryGLM:
    """Logistic regression GWAS for binary phenotypes.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    firth : bool
        Use Firth's penalized likelihood for the null fit (default False).
    max_iter : int
        Maximum IRLS iterations (default 25).
    tol : float
        Convergence tolerance on deviance change (default 1e-8).
    use_spa : bool
        Apply saddlepoint approximation for tail p-values (default True).
    spa_threshold : float
        Chi-square threshold above which SPA is applied (default 2.0).
    """

    def __init__(
        self,
        firth: bool = False,
        max_iter: int = 25,
        tol: float = 1e-8,
        use_spa: bool = True,
        spa_threshold: float = 2.0,
        ploidy: int = 2,
    ) -> None:
        self.firth = firth
        self.max_iter = max_iter
        self.tol = tol
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold
        self.ploidy = ploidy
        self.link = LogitLink()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null logistic regression model via IRLS.

        Parameters
        ----------
        Y : (n,) — binary phenotype (0 or 1)
        X0 : (n, c) — covariates (intercept as first column)
        K : ignored (no random effects in GLM)

        Returns
        -------
        NullFit with GLM-specific attributes attached.
        """
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = X0.shape

        # Initialize coefficients
        # Start from intercept-only: beta_0 = logit(mean(Y))
        p_bar = Y.mean().clamp(min=0.01, max=0.99)
        beta = torch.zeros(c, dtype=STAT_DTYPE, device=Y.device)
        beta[0] = math.log(p_bar / (1.0 - p_bar))

        converged = False
        dev_old = float("inf")

        for it in range(self.max_iter):
            eta = X0 @ beta  # (n,)
            mu = self.link.inverse(eta)  # (n,)
            mu = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)

            W = mu * (1.0 - mu)  # (n,) working weights

            # Working response
            z = eta + (Y - mu) / W  # (n,)

            # Weighted least squares: (X0' W X0) beta = X0' W z
            WX = W.unsqueeze(1) * X0  # (n, c)
            XtWX = X0.T @ WX  # (c, c)
            XtWz = X0.T @ (W * z)  # (c,)

            if self.firth:
                # Firth correction: add 0.5 * X0' diag(h) (1 - 2*mu)
                # h_i = diag(H) where H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}
                XtWX_inv = torch.linalg.inv(XtWX)
                H = WX @ XtWX_inv @ WX.T  # (n, n) — hat matrix
                h = torch.diag(H)  # (n,)
                firth_score = 0.5 * X0.T @ (h * (1.0 - 2.0 * mu))  # (c,)
                XtWz = XtWz + firth_score

            beta_new = torch.linalg.solve(XtWX, XtWz)

            # Deviance: -2 * sum(Y*log(mu) + (1-Y)*log(1-mu))
            log_mu = torch.log(mu.clamp(min=1e-300))
            log_1mu = torch.log((1.0 - mu).clamp(min=1e-300))
            dev = -2.0 * (Y * log_mu + (1.0 - Y) * log_1mu).sum().item()

            if abs(dev - dev_old) < self.tol:
                beta = beta_new
                converged = True
                break

            dev_old = dev
            beta = beta_new

        # Final fitted values
        eta = X0 @ beta
        mu = self.link.inverse(eta)
        mu = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        W = mu * (1.0 - mu)

        # Log-likelihood
        log_mu = torch.log(mu.clamp(min=1e-300))
        log_1mu = torch.log((1.0 - mu).clamp(min=1e-300))
        ll = (Y * log_mu + (1.0 - Y) * log_1mu).sum().item()

        # Store (X0'WX0)^{-1} for score test
        WX = W.unsqueeze(1) * X0
        XtWX = X0.T @ WX
        XtWX_inv = torch.linalg.inv(XtWX)

        logger.info(
            "BinaryGLM null fit: n=%d, c=%d, converged=%s, deviance=%.4f",
            n, c, converged, dev_old,
        )

        nf = NullFit(
            sig2_e=None,
            Y_rot=Y,  # store original Y
            X0_rot=X0,
            M00=XtWX_inv,  # (X0'WX0)^{-1}
            b0=beta,
            weights=W,  # (n,) working weights
            log_likelihood=ll,
            converged=converged,
            device=Y.device,
        )

        # Attach GLM-specific attributes
        nf._glm_mu = mu
        nf._glm_W = W
        nf._glm_family = "binary"

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via logistic score test.

        For each SNP g:
          U = g'(Y - mu_0)
          V = g'Wg - g'WX0 (X0'WX0)^{-1} X0'Wg  (Schur complement)
          T = U^2 / V ~ chi2(1)

        Optionally applies SPA for tail p-values.
        """
        if test not in ("score",):
            raise ValueError(f"BinaryGLM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot  # (n,)
        mu = null_fit._glm_mu  # (n,)
        W = null_fit._glm_W  # (n,)
        X0 = null_fit.X0_rot  # (n, c)
        XtWX_inv = null_fit.M00  # (c, c)

        # Pearson residuals
        resid = Y - mu  # (n,)

        # Score statistic: U = g' (Y - mu)
        U = G_chunk.T @ resid  # (m,)

        # Variance: V = g'Wg - g'WX0 (X0'WX0)^{-1} X0'Wg
        WG = W.unsqueeze(1) * G_chunk  # (n, m)
        gWg = (G_chunk * WG).sum(dim=0)  # (m,)
        X0tWG = X0.T @ WG  # (c, m)
        correction = (X0tWG * (XtWX_inv @ X0tWG)).sum(dim=0)  # (m,)
        V = gWg - correction  # (m,)
        V = torch.clamp(V, min=1e-20)

        # Chi-square statistic
        stat = U ** 2 / V  # (m,)

        # P-values
        if self.use_spa:
            from ..stats.spa import saddlepoint_pvalue
            p = saddlepoint_pvalue(U, mu, G_chunk,
                                    threshold=self.spa_threshold,
                                    variance=V)
        else:
            from ..stats.tests import chi2_sf
            p = chi2_sf(stat, df=1)

        # Approximate beta and SE from score
        beta_approx = U / V
        se_approx = 1.0 / torch.sqrt(V)

        # Allele frequencies
        af = G_chunk.mean(dim=0) / float(self.ploidy)

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta_approx,
            se=se_approx,
            stat=stat,
            p=p,
            test="score",
        )
