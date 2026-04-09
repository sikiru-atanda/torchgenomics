"""Multi-Environment GLMM: categorical phenotypes across environments with PQL.

Extends the single-environment GLMMs (BinaryGLMM, OrdinalGLMM) to handle
binary or ordinal phenotypes measured across E environments with structured
genetic covariance Σ_g across environments and kinship correction.

The null model is fitted via multi-environment PQL: per-env working response
and weights → multi-trait weighted LMM inner solve (E "traits") → BLUP.
The eigendecomposition is cached (SAIGE approximation).

SNP scanning uses the conditional score test with PQL-fitted μ_ie as null.
Provides: joint χ²(E), per-env marginals χ²(1), homogeneity χ²(E-1),
and optional reaction-norm decomposition (stable + GxE).

SPA is applied to per-env binary marginals only (not to multi-df tests).

References
----------
- Zhou et al. (2018). SAIGE.
- Zhou & Stephens (2014). Efficient mvLMM (GEMMA).
- Breslow & Clayton (1993). Approximate inference in GLMM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import BaseModel, NullFit, VariantMeta

logger = logging.getLogger(__name__)


class MultiEnvGLMM:
    """Multi-environment GLMM with PQL null fitting and score test scan.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    family : str
        "binary" or "ordinal".
    n_categories : int or None
        For ordinal family, number of categories J (>= 2).
    parameterization : str
        "per_env" or "reaction_norm" — controls additional test statistics.
    use_spa : bool
        Apply SPA for per-env binary marginals (default True).
    spa_threshold : float
        Chi-square threshold for SPA activation (default 2.0).
    firth : bool
        Use Firth correction for per-env GLM initialization (default False).
    pql_max_iter : int
        Maximum PQL outer iterations (default 30).
    pql_tol : float
        PQL convergence tolerance (default 1e-4).
    ploidy : int
        Organism ploidy for allele frequency computation (default 2).
    """

    def __init__(
        self,
        family: str = "binary",
        n_categories: int | None = None,
        parameterization: str = "per_env",
        use_spa: bool = True,
        spa_threshold: float = 2.0,
        firth: bool = False,
        pql_max_iter: int = 30,
        pql_tol: float = 1e-4,
        ploidy: int = 2,
    ) -> None:
        if family not in ("binary", "ordinal"):
            raise ValueError(f"family must be 'binary' or 'ordinal', got '{family}'")
        if family == "ordinal" and (n_categories is None or n_categories < 2):
            raise ValueError("ordinal family requires n_categories >= 2")
        if parameterization not in ("per_env", "reaction_norm"):
            raise ValueError(f"parameterization must be 'per_env' or 'reaction_norm'")
        self.family = family
        self.n_categories = n_categories
        self.parameterization = parameterization
        self.use_spa = use_spa
        self.spa_threshold = spa_threshold
        self.firth = firth
        self.pql_max_iter = pql_max_iter
        self.pql_tol = pql_tol
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        *,
        env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null multi-environment GLMM via PQL.

        Parameters
        ----------
        Y : (n, E) — phenotype matrix, one column per environment.
        X0 : (n, c) — covariate matrix.
        K : (n, n) — kinship / GRM (required).
        env_names : optional list of environment names.

        Returns
        -------
        NullFit with multi-env GLMM attributes.
        """
        if K is None:
            raise ValueError("MultiEnvGLMM requires a kinship matrix K.")
        if Y.ndim != 2:
            raise ValueError(f"Y must be (n, E), got shape {Y.shape}")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        n, E = Y.shape
        c = X0.shape[1]

        # Step 1: Per-env GLM initialization
        mu_init = self._init_per_env_glm(Y, X0)

        # Step 2: Multi-env PQL
        from ..optim.multi_env_pql import multi_env_pql_fit

        pql_result = multi_env_pql_fit(
            Y, X0, K, mu_init,
            family=self.family,
            n_categories=self.n_categories,
            max_outer=self.pql_max_iter,
            tol=self.pql_tol,
        )

        mu = pql_result["mu"]
        W = pql_result["W"]  # (n, E)

        # Step 3: Precompute XtWX_inv per environment for score test
        XtWX_inv_per_env = []
        for e in range(E):
            W_e = W[:, e]  # (n,)
            WX = W_e.unsqueeze(1) * X0  # (n, c)
            XtWX_e = X0.T @ WX  # (c, c)
            XtWX_inv_e = torch.linalg.inv(
                XtWX_e + 1e-8 * torch.eye(c, dtype=STAT_DTYPE, device=Y.device)
            )
            XtWX_inv_per_env.append(XtWX_inv_e)
        XtWX_inv_stack = torch.stack(XtWX_inv_per_env)  # (E, c, c)

        logger.info(
            "MultiEnvGLMM null fit: n=%d, E=%d, family=%s, converged=%s, "
            "PQL iters=%d",
            n, E, self.family, pql_result["converged"],
            pql_result["n_outer"],
        )

        nf = NullFit(
            sig2_g=None,
            sig2_e=None,
            eigenvalues=pql_result["eigenvalues"],
            eigenvectors=pql_result["eigenvectors"],
            Y_rot=Y,
            X0_rot=X0,
            b0=pql_result["beta"],
            weights=W,
            log_likelihood=pql_result["log_likelihood"],
            converged=pql_result["converged"],
            device=Y.device,
        )

        # Attach ME-GLMM specific attributes
        nf._me_glmm_mu = mu  # (n, E)
        nf._me_glmm_W = W  # (n, E)
        nf._me_glmm_family = self.family
        nf._me_glmm_E = E
        nf._me_glmm_Sigma_g = pql_result["Sigma_g"]  # (E, E)
        nf._me_glmm_Sigma_e = pql_result["Sigma_e"]  # (E, E)
        nf._me_glmm_XtWX_inv = XtWX_inv_stack  # (E, c, c)
        nf._me_glmm_parameterization = self.parameterization
        nf._me_glmm_env_names = env_names or [f"env_{e}" for e in range(E)]

        if self.family == "ordinal":
            nf._me_glmm_gamma = pql_result.get("gamma")
            nf._me_glmm_thresholds = pql_result.get("thresholds")

        return nf

    def _init_per_env_glm(self, Y: Tensor, X0: Tensor) -> Tensor:
        """Initialize mu via per-env GLM fits."""
        n, E = Y.shape
        mu_init = torch.zeros(n, E, dtype=STAT_DTYPE, device=Y.device)

        if self.family == "binary":
            from .binary_glm import BinaryGLM
            for e in range(E):
                glm = BinaryGLM(firth=self.firth, use_spa=False)
                nf_e = glm.fit_null(Y[:, e], X0)
                mu_init[:, e] = nf_e._glm_mu
        elif self.family == "ordinal":
            from .ordinal_glm import OrdinalGLM
            for e in range(E):
                glm = OrdinalGLM(n_categories=self.n_categories)
                nf_e = glm.fit_null(Y[:, e], X0)
                # For ordinal, use expected value E[Y] as mu
                pi = nf_e._glm_pi  # (n, J)
                J = pi.shape[1]
                E_Y = sum(j * pi[:, j] for j in range(J))
                mu_init[:, e] = E_Y.clamp(min=1e-10, max=J - 1 - 1e-10)

        return mu_init

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ):
        """Score SNP chunk via multi-environment conditional score test.

        Returns EnvScanResult with joint, per-env, and homogeneity tests.
        """
        from .multi_env_lmm import EnvScanResult

        if test not in ("score",):
            raise ValueError(f"MultiEnvGLMM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot  # (n, E)
        mu = null_fit._me_glmm_mu  # (n, E)
        W = null_fit._me_glmm_W  # (n, E)
        X0 = null_fit.X0_rot  # (n, c)
        XtWX_inv = null_fit._me_glmm_XtWX_inv  # (E, c, c)
        E = null_fit._me_glmm_E
        device = G_chunk.device

        # Per-env score and variance (Schur complement)
        U = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
        V_diag = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)

        if self.family == "binary":
            for e in range(E):
                resid_e = Y[:, e] - mu[:, e]  # (n,)
                U[:, e] = G_chunk.T @ resid_e  # (m,)

                # V_ee = g'W_e g - g'W_e X0 (X0'W_e X0)^{-1} X0'W_e g
                W_e = W[:, e]  # (n,)
                WG = W_e.unsqueeze(1) * G_chunk  # (n, m)
                gWg = (G_chunk * WG).sum(dim=0)  # (m,)
                X0tWG = X0.T @ WG  # (c, m)
                correction = (X0tWG * (XtWX_inv[e] @ X0tWG)).sum(dim=0)  # (m,)
                V_diag[:, e] = torch.clamp(gWg - correction, min=1e-20)

        elif self.family == "ordinal":
            gamma = null_fit._me_glmm_gamma  # (n, E, J-1)
            J = self.n_categories

            for e in range(E):
                # Score: U_e = sum_j g'(D_je - gamma_je)
                score_e = torch.zeros(m, dtype=STAT_DTYPE, device=device)
                var_e = torch.zeros(m, dtype=STAT_DTYPE, device=device)
                for j in range(J - 1):
                    D_j = (Y[:, e] <= j).float()  # (n,)
                    resid_j = D_j - gamma[:, e, j]  # (n,)
                    score_e += G_chunk.T @ resid_j
                    w_j = gamma[:, e, j] * (1.0 - gamma[:, e, j])
                    var_e += (G_chunk ** 2).T @ w_j

                U[:, e] = score_e
                V_diag[:, e] = torch.clamp(var_e, min=1e-20)

        # --- Per-env marginal tests: χ²(1) ---
        stat_marginal = U ** 2 / V_diag  # (m, E)
        from ..stats.tests import chi2_sf
        p_marginal = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            p_marginal[:, e] = chi2_sf(stat_marginal[:, e], df=1)

        # --- SPA for per-env binary marginals ---
        if self.family == "binary" and self.use_spa:
            from ..stats.spa import saddlepoint_pvalue
            for e in range(E):
                p_marginal[:, e] = saddlepoint_pvalue(
                    U[:, e], mu[:, e], G_chunk,
                    threshold=self.spa_threshold,
                    variance=V_diag[:, e],
                )

        # --- Joint test: χ²(E) ---
        # V is block-diagonal: V = diag(V_11, ..., V_EE)
        # stat_joint = sum_e U_e^2 / V_ee
        stat_joint = (U ** 2 / V_diag).sum(dim=1)  # (m,)
        p_joint = chi2_sf(stat_joint, df=E)

        # --- Approximate beta and SE ---
        beta_approx = U / V_diag  # (m, E)
        se_approx = 1.0 / torch.sqrt(V_diag)  # (m, E)

        # --- Var(beta) for homogeneity and reaction-norm tests ---
        # Under block-diagonal V: Var_beta[e,e] = 1/V_ee, Var_beta[e,f] = 0
        Var_beta = torch.zeros(m, E, E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            Var_beta[:, e, e] = 1.0 / V_diag[:, e]

        # --- Homogeneity test: χ²(E-1) ---
        C = _build_contrast_matrix(E, device)  # (E-1, E)
        Cb = torch.einsum('de,je->jd', C, beta_approx)  # (m, E-1)
        CVCt = torch.einsum('de,jef,gf->jdg', C, Var_beta, C)  # (m, E-1, E-1)
        CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
        stat_hom = torch.clamp((Cb * CVCt_inv_Cb).sum(dim=-1), min=0.0)
        p_hom = chi2_sf(stat_hom, df=E - 1)

        # --- Reaction-norm tests ---
        rn_fields = {}
        if self.parameterization == "reaction_norm":
            rn_fields = _compute_reaction_norm(beta_approx, Var_beta, E, device)

        af = G_chunk.mean(dim=0) / float(self.ploidy)

        return EnvScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta_approx,
            se=se_approx,
            stat=stat_joint,
            p=p_joint,
            stat_homogeneity=stat_hom,
            p_homogeneity=p_hom,
            stat_marginal=stat_marginal,
            p_marginal=p_marginal,
            test="score",
            env_names=null_fit._me_glmm_env_names,
            **rn_fields,
        )


# ---------------------------------------------------------------------------
# Utility functions (local copies to avoid circular imports)
# ---------------------------------------------------------------------------

def _build_contrast_matrix(E: int, device: torch.device) -> Tensor:
    """Successive-differences contrast matrix (E-1, E)."""
    C = torch.zeros(E - 1, E, dtype=STAT_DTYPE, device=device)
    for i in range(E - 1):
        C[i, i] = 1.0
        C[i, i + 1] = -1.0
    return C


def _compute_reaction_norm(
    beta: Tensor,
    Var_beta: Tensor,
    E: int,
    device: torch.device,
) -> dict:
    """Reaction-norm decomposition: stable effect + GxE."""
    from ..stats.tests import chi2_sf

    # Stable effect: α = mean(β)
    alpha = beta.mean(dim=-1)  # (m,)
    ones_E = torch.ones(E, dtype=STAT_DTYPE, device=device)
    var_alpha = torch.einsum("mij,i,j->m", Var_beta, ones_E, ones_E) / (E * E)
    var_alpha = var_alpha.clamp(min=1e-30)
    se_alpha = torch.sqrt(var_alpha)

    stat_stable = torch.clamp(alpha ** 2 / var_alpha, min=0.0)
    p_stable = chi2_sf(stat_stable, df=1)

    # GxE deviations
    delta = beta - alpha.unsqueeze(-1)  # (m, E)
    C = _build_contrast_matrix(E, device)
    Cd = torch.einsum("de,me->md", C, delta)
    CVCt = torch.einsum("de,mef,gf->mdg", C, Var_beta, C)
    CVCt_inv_Cd = torch.linalg.solve(CVCt, Cd.unsqueeze(-1)).squeeze(-1)
    stat_gxe = torch.clamp((Cd * CVCt_inv_Cd).sum(dim=-1), min=0.0)
    p_gxe = chi2_sf(stat_gxe, df=E - 1)

    return dict(
        alpha=alpha,
        se_alpha=se_alpha,
        stat_stable=stat_stable,
        p_stable=p_stable,
        delta=delta,
        stat_gxe=stat_gxe,
        p_gxe=p_gxe,
    )
