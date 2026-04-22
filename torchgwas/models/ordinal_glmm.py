"""Ordinal GLMM: cumulative logit mixed model GWAS with PQL.

Implements ordinal GLMM for ordinal phenotypes (e.g. disease severity
0/1/2/3) with population structure correction via kinship.  Null model
fitted via PQL with cumulative logit link.  SNP scanning uses the
ordinal score test.
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


class OrdinalGLMM:
    """Ordinal GLMM with PQL null fitting and score test scan.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories J (>= 2).
    pql_max_iter : int
        Maximum PQL outer iterations (default 30).
    pql_tol : float
        PQL convergence tolerance (default 1e-4).
    """

    def __init__(
        self,
        n_categories: int,
        pql_max_iter: int = 30,
        pql_tol: float = 1e-4,
        ploidy: int = 2,
    ) -> None:
        if n_categories < 2:
            raise ValueError(f"n_categories must be >= 2, got {n_categories}")
        self.n_categories = n_categories
        self.pql_max_iter = pql_max_iter
        self.pql_tol = pql_tol
        self.ploidy = ploidy
        self.link = CumulativeLogitLink(n_categories)

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null ordinal GLMM via PQL.

        Parameters
        ----------
        Y : (n,) — ordinal phenotype in {0, ..., J-1}
        X0 : (n, c) — covariate matrix
        K : (n, n) — kinship / GRM (required)

        Returns
        -------
        NullFit with ordinal GLMM-specific attributes.
        """
        if K is None:
            raise ValueError(
                "OrdinalGLMM requires a kinship matrix K. "
                "Use OrdinalGLM for no random effects."
            )

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = X0.shape
        J = self.n_categories

        # Step 1: Initialize with OrdinalGLM fit
        from .ordinal_glm import OrdinalGLM
        glm_init = OrdinalGLM(n_categories=J)
        glm_nf = glm_init.fit_null(Y, X0)

        # For ordinal PQL, we use a latent variable approach:
        # Map ordinal Y to latent continuous z using the GLM thresholds
        # Then run PQL on the latent variable
        thresholds = glm_nf._glm_thresholds  # (J-1,)
        gamma = glm_nf._glm_gamma  # (n, J-1) cumulative probs
        pi = glm_nf._glm_pi  # (n, J) category probs

        # Working response: for each observation, use the expected value
        # under the current ordinal model on the latent logistic scale.
        # Use the midpoint of the interval for the observed category.
        Y_int = Y.long()

        # Construct latent working response
        # For category j, latent z ~ logistic, truncated to (alpha_{j-1}, alpha_j)
        # We approximate: z_i ≈ E[z | Y_i = y_i]
        # For simplicity, use threshold midpoints
        alpha_ext = torch.cat([
            torch.tensor([-10.0], dtype=STAT_DTYPE, device=Y.device),
            thresholds,
            torch.tensor([10.0], dtype=STAT_DTYPE, device=Y.device),
        ])  # (J+1,)

        # Vectorized latent initialization: midpoint of observed category interval
        z_latent = 0.5 * (alpha_ext[Y_int] + alpha_ext[Y_int + 1])

        # Step 2: PQL on latent response with identity-like link
        # Eigendecompose K once
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
        eigenvectors = eigenvectors.flip(1)

        # Working weights from ordinal model (vectorized sum over J-1 boundaries)
        W_total = (gamma * (1.0 - gamma)).sum(dim=1).clamp(min=1e-10)

        # Weighted LMM iteration
        sig2_g = 0.5
        sig2_e = 1.0
        beta = torch.zeros(c, dtype=STAT_DTYPE, device=Y.device)
        converged = False

        for outer in range(self.pql_max_iter):
            sqrtW = torch.sqrt(W_total)
            z_w = sqrtW * z_latent
            X_w = sqrtW.unsqueeze(1) * X0

            if outer == 0:
                K_w = sqrtW.unsqueeze(1) * K * sqrtW.unsqueeze(0)
                K_w = 0.5 * (K_w + K_w.T)
                evals_w, evecs_w = torch.linalg.eigh(K_w)
                evals_w = evals_w.flip(0).clamp(min=0.0)
                evecs_w = evecs_w.flip(1)

            z_rot = evecs_w.T @ z_w
            X_rot = evecs_w.T @ X_w

            # Profile REML for variance components
            from ..optim.pql import _profile_reml_vc
            sig2_g, sig2_e = _profile_reml_vc(
                z_rot, X_rot, evals_w,
                sig2_g_init=sig2_g, sig2_e_init=sig2_e)

            lam = sig2_g / max(sig2_e, 1e-20)
            H_inv = 1.0 / (evals_w * lam + 1.0)
            wX = H_inv.unsqueeze(1) * X_rot
            XtHiX = X_rot.T @ wX
            XtHiz = X_rot.T @ (H_inv * z_rot)

            try:
                beta_new = torch.linalg.solve(XtHiX, XtHiz)
            except Exception:
                logger.warning("OrdinalGLMM PQL: solve failed iter %d", outer)
                break

            param_change = (beta_new - beta).abs().max().item()
            beta = beta_new

            # Update latent working response with BLUP
            z_w_resid = sqrtW * (z_latent - X0 @ beta)
            z_rot_resid = evecs_w.T @ z_w_resid
            shrink = (sig2_g * evals_w) / (sig2_g * evals_w + sig2_e)
            u_rot = shrink * z_rot_resid
            u_w = evecs_w @ u_rot
            u = u_w / sqrtW.clamp(min=1e-10)
            eta_new = X0 @ beta + u

            # Re-estimate thresholds given current eta via Newton-Raphson
            n_thresh = J - 1
            for _nr in range(5):
                self.link.thresholds = thresholds
                gamma_nr = self.link.cumulative_probs(eta_new)  # (n, J-1)
                score_a = torch.zeros(n_thresh, dtype=STAT_DTYPE,
                                      device=Y.device)
                H_a = torch.zeros(n_thresh, n_thresh, dtype=STAT_DTYPE,
                                  device=Y.device)
                for jj in range(n_thresh):
                    D_jj = (jj >= Y).float()
                    g_jj = gamma_nr[:, jj]
                    w_jj = g_jj * (1.0 - g_jj)
                    score_a[jj] = (D_jj - g_jj).sum()
                    H_a[jj, jj] = -w_jj.sum()
                    for kk in range(jj + 1, n_thresh):
                        g_kk = gamma_nr[:, kk]
                        cross = -(g_jj * (1.0 - g_kk)).sum()
                        H_a[jj, kk] = cross
                        H_a[kk, jj] = cross
                try:
                    delta_a = torch.linalg.solve(-H_a, score_a)
                    thresholds = thresholds + delta_a
                    thresholds, _ = thresholds.sort()
                except Exception:
                    break
                if delta_a.abs().max().item() < 1e-6:
                    break

            # Update ordinal model: re-derive gamma, pi from updated thresholds
            self.link.thresholds = thresholds
            gamma = self.link.cumulative_probs(eta_new)
            pi = self.link.category_probs(eta_new)
            pi = torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

            # Update weights (vectorized)
            W_total = (gamma * (1.0 - gamma)).sum(dim=1).clamp(min=1e-10)

            # Update latent working response (vectorized)
            # E[Y] = sum_j j * pi_j for each individual
            j_vals = torch.arange(J, dtype=STAT_DTYPE, device=Y.device)
            E_Y = (pi * j_vals.unsqueeze(0)).sum(dim=1)  # (n,)
            z_latent = eta_new + (Y - E_Y) / W_total

            if param_change < self.pql_tol and outer > 0:
                converged = True
                break

        # Compute log-likelihood (vectorized)
        ll = torch.log(pi[torch.arange(n, device=Y.device), Y_int].clamp(min=1e-300)).sum().item()

        logger.info(
            "OrdinalGLMM null fit: n=%d, J=%d, converged=%s, sig2_g=%.4e, "
            "sig2_e=%.4e",
            n, J, converged, sig2_g, sig2_e,
        )

        nf = NullFit(
            sig2_g=sig2_g,
            sig2_e=sig2_e,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            Y_rot=Y,
            X0_rot=X0,
            b0=beta,
            log_likelihood=ll,
            converged=converged,
            device=Y.device,
        )

        nf._glm_family = "ordinal_glmm"
        nf._glm_gamma = gamma
        nf._glm_pi = pi
        nf._glm_thresholds = thresholds
        nf._glm_n_categories = J

        # Fit per-boundary BinaryGLMMs for calibrated score tests (POLMM)
        from .binary_glmm import BinaryGLMM
        boundary_fits = []
        for j in range(J - 1):
            Y_bin_j = (j < Y).float()
            bglmm = BinaryGLMM(use_spa=False)
            nf_bin = bglmm.fit_null(Y_bin_j, X0, K=K)
            boundary_fits.append(nf_bin)
        nf._boundary_fits = boundary_fits

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via per-boundary BinaryGLMM tests + Cauchy (POLMM).

        Each boundary j uses a separate BinaryGLMM null fit for
        properly calibrated score and variance.  Per-boundary p-values
        are combined via the Cauchy combination test (ACAT).
        """
        if test not in ("score",):
            raise ValueError(
                f"OrdinalGLMM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        J = null_fit._glm_n_categories
        boundary_fits = null_fit._boundary_fits

        from ..stats.cauchy import cauchy_combination
        from .binary_glmm import BinaryGLMM

        # Per-boundary BinaryGLMM score tests
        p_boundaries = []
        U_best = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)
        V_best = torch.ones(m, dtype=STAT_DTYPE, device=G_chunk.device)
        best_stat = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)

        bglmm = BinaryGLMM(use_spa=False)
        for j in range(J - 1):
            nf_bin = boundary_fits[j]
            sr_bin = bglmm.score_chunk(G_chunk, nf_bin, variant_meta)
            p_boundaries.append(sr_bin.p)

            # Track most significant boundary for beta/SE
            better = sr_bin.stat > best_stat
            U_best = torch.where(better, sr_bin.beta * sr_bin.se ** 2,
                                 U_best)
            V_best = torch.where(better, sr_bin.se ** 2, V_best)
            best_stat = torch.where(better, sr_bin.stat, best_stat)

        # Combine per-boundary p-values via Cauchy combination
        p_mat = torch.stack(p_boundaries, dim=1)  # (m, J-1)
        p_combined = cauchy_combination(p_mat)  # (m,)

        beta_approx = sr_bin.beta if J == 2 else U_best / V_best
        se_approx = sr_bin.se if J == 2 else 1.0 / torch.sqrt(
            V_best.clamp(min=1e-20))
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
            stat=best_stat,
            p=p_combined,
            test="score",
        )


def _expected_y(pi_i: Tensor, J: int) -> float:
    """Expected value of Y given category probabilities."""
    return sum(j * pi_i[j].item() for j in range(J))
