"""Ordinal GLM: cumulative logit (proportional odds) regression GWAS.

Implements the proportional odds model for ordinal phenotypes (e.g.,
disease severity 0/1/2/3). Null model fitted via IRLS on the cumulative
logit parameterisation. SNP scanning uses the score test.

Model: logit(P(Y <= j)) = alpha_j - X'beta, for j = 1, ..., J-1.

The score test for ordinal regression sums contributions across
category boundaries, giving a chi2(1) test statistic per SNP.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta
from .glm_link import CumulativeLogitLink

logger = logging.getLogger(__name__)


class OrdinalGLM:
    """Ordinal regression GWAS (proportional odds / cumulative logit).

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories J (required, >= 2).
    max_iter : int
        Maximum IRLS iterations (default 50).
    tol : float
        Convergence tolerance on parameter change (default 1e-8).
    """

    def __init__(
        self,
        n_categories: int,
        max_iter: int = 50,
        tol: float = 1e-8,
        ploidy: int = 2,
    ) -> None:
        if n_categories < 2:
            raise ValueError(f"n_categories must be >= 2, got {n_categories}")
        self.n_categories = n_categories
        self.max_iter = max_iter
        self.tol = tol
        self.ploidy = ploidy
        self.link = CumulativeLogitLink(n_categories)

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null proportional odds model via IRLS.

        Parameters
        ----------
        Y : (n,) — ordinal phenotype with values in {0, 1, ..., J-1}
        X0 : (n, c) — covariates (intercept should NOT be included;
             thresholds absorb the intercept).
        K : ignored (no random effects)

        Returns
        -------
        NullFit with ordinal-GLM-specific attributes.
        """
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n = Y.shape[0]
        c = X0.shape[1]
        J = self.n_categories

        # Initialise thresholds from observed frequencies
        self.link.init_thresholds(Y, dtype=STAT_DTYPE, device=Y.device)

        # For proportional odds null (no SNP): only thresholds, no X0 effect
        # Parameter vector: theta = [alpha_1, ..., alpha_{J-1}, beta_1, ..., beta_{c-1}]
        # Note: if X0 has an intercept column, we only use non-intercept columns
        # because thresholds absorb the intercept.
        # Detect intercept: if first column is all 1s, skip it
        has_intercept = (X0[:, 0].std() < 1e-10)
        if has_intercept and c > 1:
            X_cov = X0[:, 1:]  # (n, c-1) non-intercept covariates
        else:
            X_cov = torch.zeros(n, 0, dtype=STAT_DTYPE, device=Y.device)

        n_cov = X_cov.shape[1]
        n_thresh = J - 1

        # Parameter vector: [alpha_1, ..., alpha_{J-1}, beta_covariates]
        alpha = self.link.thresholds.clone()  # (J-1,)
        beta_cov = torch.zeros(n_cov, dtype=STAT_DTYPE, device=Y.device)

        converged = False
        ll_old = float("-inf")

        for it in range(self.max_iter):
            # Compute eta = X_cov @ beta_cov (no thresholds in eta)
            eta = X_cov @ beta_cov if n_cov > 0 else torch.zeros(
                n, dtype=STAT_DTYPE, device=Y.device)

            # Update link thresholds
            self.link.thresholds = alpha

            # Cumulative probs: gamma_ij = expit(alpha_j - eta_i)
            gamma = self.link.cumulative_probs(eta)  # (n, J-1)
            pi = self.link.category_probs(eta)  # (n, J)
            pi = torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

            # Log-likelihood (vectorized)
            Y_int = Y.long()
            ll = torch.log(pi[torch.arange(n, device=Y.device), Y_int].clamp(min=1e-300)).sum().item()

            # Construct score vector and Hessian for Newton-Raphson
            # Score for alpha_j: sum_i [I(Y_i <= j) - gamma_ij] * gamma_ij * (1-gamma_ij) / pi_yi
            # This is complex; use a simplified IRLS approach.

            # Build the score and Hessian via indicator matrices
            # D_j = I(Y <= j) — (n,) for each j
            # Score for alpha_j: sum_i (D_ij - gamma_ij) where the working
            # model uses gamma_ij * (1-gamma_ij) as weights.

            score = torch.zeros(n_thresh + n_cov, dtype=STAT_DTYPE,
                                device=Y.device)
            H = torch.zeros(n_thresh + n_cov, n_thresh + n_cov,
                            dtype=STAT_DTYPE, device=Y.device)

            for j in range(n_thresh):
                D_j = (j >= Y).float()  # (n,)
                gamma_j = gamma[:, j]  # (n,)
                w_j = gamma_j * (1.0 - gamma_j)  # (n,) weight for boundary j
                w_j = torch.clamp(w_j, min=1e-10)
                resid_j = D_j - gamma_j  # (n,)

                # Score for alpha_j
                score[j] = resid_j.sum()

                # Hessian diagonal for alpha_j
                H[j, j] = -w_j.sum()

                # Cross-terms between alpha_j and alpha_k
                for k in range(j + 1, n_thresh):
                    gamma_k = gamma[:, k]
                    # Covariance term
                    cross = -(gamma_j * (1.0 - gamma_k)).sum()
                    H[j, k] = cross
                    H[k, j] = cross

                # Cross-terms between alpha_j and beta_cov
                if n_cov > 0:
                    # Score for beta from boundary j
                    score[n_thresh:] += X_cov.T @ resid_j

                    # Hessian cross-term
                    H[j, n_thresh:] = -(w_j.unsqueeze(1) * X_cov).sum(dim=0)
                    H[n_thresh:, j] = H[j, n_thresh:]

            # Hessian for beta_cov block
            if n_cov > 0:
                # Total weight: sum across boundaries
                w_total = torch.zeros(n, dtype=STAT_DTYPE, device=Y.device)
                for j in range(n_thresh):
                    w_total += gamma[:, j] * (1.0 - gamma[:, j])
                WX = w_total.unsqueeze(1) * X_cov  # (n, n_cov)
                H[n_thresh:, n_thresh:] = -(X_cov.T @ WX)

            # Newton step: theta_new = theta - H^{-1} score
            # H is negative definite, so -H is positive definite
            try:
                delta = torch.linalg.solve(-H, score)
            except Exception:
                logger.warning("Hessian solve failed at iteration %d", it)
                break

            theta = torch.cat([alpha, beta_cov])
            theta_new = theta + delta

            alpha = theta_new[:n_thresh]
            if n_cov > 0:
                beta_cov = theta_new[n_thresh:]

            # Ensure thresholds are ordered
            alpha, _ = alpha.sort()

            # Convergence check
            if delta.abs().max().item() < self.tol:
                converged = True
                break

            ll_old = ll

        # Final fitted values
        self.link.thresholds = alpha
        eta_final = X_cov @ beta_cov if n_cov > 0 else torch.zeros(
            n, dtype=STAT_DTYPE, device=Y.device)
        gamma_final = self.link.cumulative_probs(eta_final)
        pi_final = self.link.category_probs(eta_final)

        logger.info(
            "OrdinalGLM null fit: n=%d, J=%d, converged=%s, ll=%.4f",
            n, J, converged, ll,
        )

        nf = NullFit(
            sig2_e=None,
            Y_rot=Y,
            X0_rot=X0,
            b0=torch.cat([alpha, beta_cov]),
            log_likelihood=ll,
            converged=converged,
            device=Y.device,
        )

        # Attach ordinal-specific attributes
        nf._glm_family = "ordinal"
        nf._glm_gamma = gamma_final  # (n, J-1) cumulative probs
        nf._glm_pi = pi_final  # (n, J) category probs
        nf._glm_thresholds = alpha
        nf._glm_n_categories = J
        nf._glm_X_cov = X_cov
        nf._glm_H = H  # Hessian for score test adjustment

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via ordinal score test.

        The score for SNP g under proportional odds is:
          U = sum_{j=1}^{J-1} g'(D_j - gamma_j)
        where D_j = I(Y <= j) and gamma_j = P(Y <= j) under null.

        Variance computed via Schur complement adjusting for thresholds + covariates.
        """
        if test not in ("score",):
            raise ValueError(f"OrdinalGLM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot  # (n,)
        gamma = null_fit._glm_gamma  # (n, J-1)
        J = null_fit._glm_n_categories

        # Score: U_g = sum_j g'(D_j - gamma_j) for each boundary j
        U = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)
        V_raw = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)

        H = null_fit._glm_H  # (n_thresh+n_cov, n_thresh+n_cov) negative Hessian
        X_cov = null_fit._glm_X_cov  # (n, n_cov)
        n_thresh = J - 1
        n_cov = X_cov.shape[1]
        n_params = n_thresh + n_cov

        # Cross-information between SNP score and null params: (m, n_params)
        C = torch.zeros(m, n_params, dtype=STAT_DTYPE, device=G_chunk.device)

        for j in range(J - 1):
            D_j = (j >= Y).float()  # (n,)
            resid_j = D_j - gamma[:, j]  # (n,)
            U += G_chunk.T @ resid_j  # (m,)

            # Variance contribution from boundary j
            w_j = gamma[:, j] * (1.0 - gamma[:, j])
            V_raw += (G_chunk ** 2).T @ w_j  # (m,)

            # Cross-info with alpha_j: sum_i g_si * w_j_i
            C[:, j] = G_chunk.T @ w_j  # (m,)

        # Cross-info with beta_cov: sum_i g_si * W_total_i * X_cov_il
        if n_cov > 0:
            W_total = torch.zeros(n, dtype=STAT_DTYPE, device=G_chunk.device)
            for j in range(J - 1):
                W_total += gamma[:, j] * (1.0 - gamma[:, j])
            WX = W_total.unsqueeze(1) * X_cov  # (n, n_cov)
            C[:, n_thresh:] = G_chunk.T @ WX  # (m, n_cov)

        # Schur complement: V_adj = V_raw - C' (-H)^{-1} C
        I_null = -H  # positive definite information matrix
        try:
            I_inv_C = torch.linalg.solve(I_null, C.T)  # (n_params, m)
            correction = (C.T * I_inv_C).sum(dim=0)  # (m,)
        except Exception:
            correction = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)

        V_adj = V_raw - correction
        V_adj = torch.clamp(V_adj, min=1e-20)

        # Chi-square statistic
        stat = U ** 2 / V_adj

        # P-values
        from ..stats.tests import chi2_sf
        p = chi2_sf(stat, df=1)

        # Approximate beta and SE
        beta_approx = U / V_adj
        se_approx = 1.0 / torch.sqrt(V_adj)

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
