"""Binary GLMM: logistic mixed model GWAS with PQL + score test + SPA.

Implements SAIGE-style binary GLMM for case-control phenotypes with
population structure (kinship / GRM).  The null model is fitted via
PQL (penalized quasi-likelihood): iteratively constructs working
response and weights, then solves a weighted LMM in each inner step.

SNP scanning uses the score test with the PQL-fitted mu_0 as the null
expectation.  Optional SPA for calibrated tail p-values under
case-control imbalance.

References
----------
- Zhou et al. (2018). SAIGE: efficiently controls for case-control
  imbalance and sample relatedness in GLMM.
- Breslow & Clayton (1993). Approximate inference in GLMM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import BaseModel, NullFit, ScanResult, VariantMeta
from .glm_link import LogitLink

logger = logging.getLogger(__name__)


class BinaryGLMM:
    """Binary GLMM with PQL null fitting and score test scan.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    use_spa : bool
        Apply saddlepoint approximation for tail p-values (default True).
    spa_threshold : float
        Chi-square threshold above which SPA is applied (default 2.0).
    firth : bool
        Use Firth correction for the initial GLM fit (default False).
    pql_max_iter : int
        Maximum PQL outer iterations (default 30).
    pql_tol : float
        PQL convergence tolerance (default 1e-4).
    """

    def __init__(
        self,
        use_spa: bool = True,
        spa_threshold: float = 2.0,
        firth: bool = False,
        pql_max_iter: int = 30,
        pql_tol: float = 1e-4,
        ploidy: int = 2,
    ) -> None:
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold
        self.firth = firth
        self.pql_max_iter = pql_max_iter
        self.pql_tol = pql_tol
        self.ploidy = ploidy
        self.link = LogitLink()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null binary GLMM via PQL.

        Parameters
        ----------
        Y : (n,) — binary phenotype (0/1)
        X0 : (n, c) — covariate matrix
        K : (n, n) — kinship / GRM (required for GLMM)

        Returns
        -------
        NullFit with GLMM-specific attributes.
        """
        if K is None:
            raise ValueError(
                "BinaryGLMM requires a kinship matrix K. "
                "Use BinaryGLM for no random effects."
            )

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = X0.shape

        # Step 1: Initialize with BinaryGLM fit
        from .binary_glm import BinaryGLM
        glm_init = BinaryGLM(firth=self.firth)
        glm_nf = glm_init.fit_null(Y, X0)
        mu_init = glm_nf._glm_mu

        # Step 2: Run PQL iteration
        from ..optim.pql import pql_fit

        def var_fn(mu):
            return mu * (1.0 - mu)

        def deriv_fn(mu):
            return 1.0 / (mu * (1.0 - mu)).clamp(min=1e-10)

        def link_fn(mu):
            return torch.log(mu / (1.0 - mu))

        pql_result = pql_fit(
            Y, X0, K,
            mu_init=mu_init,
            link_variance_fn=var_fn,
            link_derivative_fn=deriv_fn,
            link_fn=link_fn,
            max_outer=self.pql_max_iter,
            tol=self.pql_tol,
        )

        mu = pql_result["mu"]
        mu = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        W = mu * (1.0 - mu)

        # Compute (X0'WX0)^{-1} for score test covariate adjustment
        WX = W.unsqueeze(1) * X0
        XtWX = X0.T @ WX
        XtWX_inv = torch.linalg.inv(XtWX)

        logger.info(
            "BinaryGLMM null fit: n=%d, c=%d, converged=%s, sig2_g=%.4e, "
            "sig2_e=%.4e, PQL iters=%d",
            n, c, pql_result["converged"],
            pql_result["sig2_g"], pql_result["sig2_e"],
            pql_result["n_outer"],
        )

        nf = NullFit(
            sig2_g=pql_result["sig2_g"],
            sig2_e=pql_result["sig2_e"],
            eigenvalues=pql_result["eigenvalues"],
            eigenvectors=pql_result["eigenvectors"],
            Y_rot=Y,
            X0_rot=X0,
            M00=XtWX_inv,
            b0=pql_result["beta"],
            weights=W,
            log_likelihood=pql_result["log_likelihood"],
            converged=pql_result["converged"],
            device=Y.device,
        )

        # Attach GLMM-specific attributes
        nf._glm_mu = mu
        nf._glm_W = W
        nf._glm_family = "binary_glmm"
        nf._pql_evals_w = pql_result["evals_w"]
        nf._pql_evecs_w = pql_result["evecs_w"]

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via logistic score test with PQL-fitted mu.

        Uses the same score test as BinaryGLM but with mu from PQL.
        Optionally applies SPA for tail p-values.
        """
        if test not in ("score",):
            raise ValueError(f"BinaryGLMM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot
        mu = null_fit._glm_mu
        W = null_fit._glm_W
        X0 = null_fit.X0_rot
        XtWX_inv = null_fit.M00

        # Score: U = g'(Y - mu)
        resid = Y - mu
        U = G_chunk.T @ resid

        # Variance: V = g'Wg - g'WX0 (X0'WX0)^{-1} X0'Wg
        WG = W.unsqueeze(1) * G_chunk
        gWg = (G_chunk * WG).sum(dim=0)
        X0tWG = X0.T @ WG
        correction = (X0tWG * (XtWX_inv @ X0tWG)).sum(dim=0)
        V = gWg - correction
        V = torch.clamp(V, min=1e-20)

        stat = U ** 2 / V

        # P-values
        if self.use_spa:
            from ..stats.spa import saddlepoint_pvalue
            p = saddlepoint_pvalue(U, mu, G_chunk,
                                    threshold=self.spa_threshold,
                                    variance=V)
        else:
            from ..stats.tests import chi2_sf
            p = chi2_sf(stat, df=1)

        beta_approx = U / V
        se_approx = 1.0 / torch.sqrt(V)
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
