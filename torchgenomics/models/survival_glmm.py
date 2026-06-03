"""Survival GWAS: Cox PH mixed model with martingale residual score test.

Implements a Cox proportional hazards frailty model for time-to-event GWAS:

    h(t | X_i, u_i) = h0(t) * exp(X_i' beta + u_i)
    u ~ N(0, sig2_g * K)

The null model is fitted via PQL on the Poisson working model (Breslow-
Clayton 1993).  SNP scanning uses the martingale residual score test with
optional SPA for calibrated tail p-values under censoring imbalance.

References
----------
- Bi et al. (2020). SPACox: fast genome-wide time-to-event analysis.
- Dey et al. (2022). GATE: efficient frailty model approach.
- He & Kulminski (2020). COXMEG: fast Cox mixed-effects GWAS.
- Breslow & Clayton (1993). Approximate inference in GLMM.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class SurvivalGLMM:
    """Cox PH mixed model with PQL null fitting and score test scan.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    use_spa : bool
        Apply SPA for calibrated tail p-values (default True).
    spa_threshold : float
        Chi-square threshold for SPA activation (default 2.0).
    pql_max_iter : int
        Maximum PQL outer iterations (default 30).
    pql_tol : float
        PQL convergence tolerance (default 1e-4).
    ties : str
        Tie-handling method: "breslow" (default). Efron is future.
    ploidy : int
        Organism ploidy for allele frequency computation (default 2).
    """

    def __init__(
        self,
        use_spa: bool = True,
        spa_threshold: float = 2.0,
        pql_max_iter: int = 30,
        pql_tol: float = 1e-4,
        ties: str = "breslow",
        ploidy: int = 2,
    ) -> None:
        if ties not in ("breslow",):
            raise ValueError(f"ties must be 'breslow', got '{ties}'")
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold
        self.pql_max_iter = pql_max_iter
        self.pql_tol = pql_tol
        self.ties = ties
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null Cox PH frailty model via PQL.

        Parameters
        ----------
        Y : (n, 2) — column 0 = follow-up time, column 1 = event indicator (0/1).
        X0 : (n, c) — covariate matrix (include intercept if desired).
        K : (n, n) — kinship / GRM (required).

        Returns
        -------
        NullFit with survival-specific attributes.
        """
        if K is None:
            raise ValueError("SurvivalGLMM requires a kinship matrix K.")
        if Y.ndim != 2 or Y.shape[1] != 2:
            raise ValueError(
                f"Y must be (n, 2) with columns [time, event], got shape {Y.shape}"
            )

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        time = Y[:, 0]
        event = Y[:, 1]
        n = time.shape[0]
        c = X0.shape[1]

        # Validate
        if (time <= 0).any():
            raise ValueError("Follow-up times must be positive.")
        if not ((event == 0) | (event == 1)).all():
            raise ValueError("Event indicators must be 0 or 1.")
        if event.sum() < 1:
            raise ValueError(
                "No events observed (all censored). Cannot fit Cox model."
            )

        # PQL fit
        from ..optim.cox_pql import cox_pql_fit

        pql_result = cox_pql_fit(
            time, event, X0, K,
            max_outer=self.pql_max_iter,
            tol=self.pql_tol,
        )

        E = pql_result["mu"]          # expected events
        M = pql_result["martingale"]   # delta - E
        W = E.clone()                  # working weight = E_i

        # Precompute (X0' W X0)^{-1} for score test Schur complement
        WX = W.unsqueeze(1) * X0  # (n, c)
        XtWX = X0.T @ WX  # (c, c)
        XtWX_inv = torch.linalg.inv(
            XtWX + 1e-8 * torch.eye(c, dtype=STAT_DTYPE, device=Y.device)
        )

        logger.info(
            "SurvivalGLMM null fit: n=%d, events=%d, censor_rate=%.2f, "
            "converged=%s, PQL iters=%d, sig2_g=%.4f",
            n, int(event.sum().item()),
            1.0 - event.mean().item(),
            pql_result["converged"],
            pql_result["n_outer"],
            pql_result["sig2_g"],
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

        # Attach survival-specific attributes
        nf._surv_martingale = M       # (n,) martingale residuals
        nf._surv_expected = E         # (n,) expected events
        nf._surv_W = W                # (n,) weights = E_i
        nf._surv_time = time          # (n,) follow-up times
        nf._surv_event = event        # (n,) event indicators
        nf._glm_mu = E               # for SPA compatibility
        nf._glm_W = W
        nf._glm_family = "survival_glmm"

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNPs via martingale residual score test.

        The score statistic for SNP j is:
            U_j = g_j' M  where M_i = delta_i - E_i (martingale residuals)
            V_j = g_j' W g_j - g_j' W X0 (X0'WX0)^{-1} X0' W g_j
            stat_j = U_j^2 / V_j ~ chi2(1) under H0

        Parameters
        ----------
        G_chunk : (n, m) genotype matrix
        null_fit : NullFit from fit_null
        variant_meta : VariantMeta with SNP identifiers
        test : str, only "score" supported

        Returns
        -------
        ScanResult
        """
        if test != "score":
            raise ValueError(
                f"SurvivalGLMM supports 'score' test only, got '{test}'."
            )

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        M = null_fit._surv_martingale  # (n,)
        W = null_fit._surv_W           # (n,)
        X0 = null_fit.X0_rot           # (n, c)
        XtWX_inv = null_fit.M00        # (c, c)

        # Score: U = G' M
        U = G_chunk.T @ M  # (m,)

        # Variance: Schur complement
        WG = W.unsqueeze(1) * G_chunk  # (n, m)
        gWg = (G_chunk * WG).sum(dim=0)  # (m,)
        X0tWG = X0.T @ WG  # (c, m)
        correction = (X0tWG * (XtWX_inv @ X0tWG)).sum(dim=0)  # (m,)
        V = torch.clamp(gWg - correction, min=1e-20)

        # Test statistic
        stat = U ** 2 / V

        # P-values
        from ..stats.tests import chi2_sf
        p = chi2_sf(stat, df=1)

        # Optional SPA
        if self.use_spa:
            from ..stats.spa import saddlepoint_pvalue
            # Clamp E_i to (0, 0.999) for CGF computation (SPACox convention)
            mu_spa = null_fit._surv_expected.clamp(min=1e-10, max=0.999)
            p = saddlepoint_pvalue(
                U, mu_spa, G_chunk,
                threshold=self.spa_threshold,
                variance=V,
            )

        # Effect estimates
        beta_approx = U / V
        se_approx = 1.0 / torch.sqrt(V)

        # Allele frequency
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
