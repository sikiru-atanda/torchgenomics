"""Multinomial GLMM: multinomial logit mixed model GWAS with PQL.

Implements multinomial GLMM for unordered categorical phenotypes with
population structure correction via kinship.  The null model uses PQL
with a latent variable approach: map multinomial Y to J-1 working
traits, then use multi-trait weighted LMM for the inner solve.

SNP scanning uses the multi-df score test: chi2(J-1) per SNP,
with PQL-fitted pi_0 as the null expectations.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class MultinomialGLMM:
    """Multinomial GLMM with PQL null fitting and multi-df score test.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    n_classes : int or None
        Number of classes J. Auto-detected from Y if None.
    pql_max_iter : int
        Maximum PQL outer iterations (default 30).
    pql_tol : float
        PQL convergence tolerance (default 1e-4).
    """

    def __init__(
        self,
        n_classes: int | None = None,
        pql_max_iter: int = 30,
        pql_tol: float = 1e-4,
        ploidy: int = 2,
    ) -> None:
        self.n_classes = n_classes
        self.pql_max_iter = pql_max_iter
        self.pql_tol = pql_tol
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null multinomial GLMM via PQL.

        Parameters
        ----------
        Y : (n,) — categorical phenotype in {0, ..., J-1}
        X0 : (n, c) — covariate matrix
        K : (n, n) — kinship / GRM (required)

        Returns
        -------
        NullFit with multinomial GLMM-specific attributes.
        """
        if K is None:
            raise ValueError(
                "MultinomialGLMM requires a kinship matrix K. "
                "Use MultinomialGLM for no random effects."
            )

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = X0.shape
        Y_int = Y.long()

        J = self.n_classes
        if J is None:
            J = int(Y_int.max().item()) + 1
            self.n_classes = J

        # Step 1: Initialize with MultinomialGLM fit
        from .multinomial_glm import MultinomialGLM
        glm_init = MultinomialGLM(n_classes=J)
        glm_nf = glm_init.fit_null(Y, X0)
        pi = glm_nf._glm_pi  # (n, J)

        # Step 2: PQL with latent working traits
        # For multinomial, we work with J-1 working responses
        # z_ij = eta_ij + (Y_ij - pi_ij) / pi_ij(1-pi_ij)
        # where Y_ij = I(Y_i = j) and eta_ij = log(pi_ij / pi_{i,J-1})

        # Eigendecompose K once
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.flip(0).clamp(min=0.0)
        eigenvectors = eigenvectors.flip(1)

        B = glm_nf._glm_B.clone()  # (J-1, c)
        sig2_g = 0.5
        sig2_e = 1.0
        converged = False

        Y_onehot = torch.zeros(n, J, dtype=STAT_DTYPE, device=Y.device)
        Y_onehot.scatter_(1, Y_int.unsqueeze(1), 1.0)

        for outer in range(self.pql_max_iter):
            # Working weights: W_j = pi_j(1 - pi_j) for each non-reference class
            W = pi[:, :J - 1] * (1.0 - pi[:, :J - 1])  # (n, J-1)
            W = torch.clamp(W, min=1e-10)

            # Working response for each trait j
            R = Y_onehot[:, :J - 1] - pi[:, :J - 1]  # (n, J-1)
            eta = X0 @ B.T  # (n, J-1)
            z = eta + R / W  # (n, J-1)

            # For simplicity, fit PQL on each working trait independently
            # (diagonal approximation — ignores cross-class correlations)
            beta_new = torch.zeros_like(B)

            for j in range(J - 1):
                w_j = W[:, j]
                z_j = z[:, j]
                sqrtW_j = torch.sqrt(w_j)

                if outer == 0 and j == 0:
                    # Cache weighted kernel (use average weights)
                    w_avg = W.mean(dim=1)
                    sqrtW_avg = torch.sqrt(w_avg)
                    K_w = sqrtW_avg.unsqueeze(1) * K * sqrtW_avg.unsqueeze(0)
                    K_w = 0.5 * (K_w + K_w.T)
                    evals_w, evecs_w = torch.linalg.eigh(K_w)
                    evals_w = evals_w.flip(0).clamp(min=0.0)
                    evecs_w = evecs_w.flip(1)

                z_w_j = sqrtW_j * z_j
                X_w_j = sqrtW_j.unsqueeze(1) * X0

                z_rot_j = evecs_w.T @ z_w_j
                X_rot_j = evecs_w.T @ X_w_j

                # Fit variance components for this trait
                from ..optim.pql import _profile_reml_vc
                sig2_g_j, sig2_e_j = _profile_reml_vc(
                    z_rot_j, X_rot_j, evals_w,
                    sig2_g_init=sig2_g, sig2_e_init=sig2_e)

                lam_j = sig2_g_j / max(sig2_e_j, 1e-20)
                H_inv_j = 1.0 / (evals_w * lam_j + 1.0)
                wX_j = H_inv_j.unsqueeze(1) * X_rot_j
                XtHiX_j = X_rot_j.T @ wX_j
                XtHiz_j = X_rot_j.T @ (H_inv_j * z_rot_j)

                try:
                    beta_j = torch.linalg.solve(XtHiX_j, XtHiz_j)
                except Exception:
                    beta_j = B[j]

                beta_new[j] = beta_j

            # Average variance components across traits
            sig2_g = sig2_g_j  # last trait's estimate
            sig2_e = sig2_e_j

            param_change = (beta_new - B).abs().max().item()
            B = beta_new

            # Update pi
            eta_new = X0 @ B.T  # (n, J-1)
            eta_full = torch.cat([eta_new, torch.zeros(n, 1, dtype=STAT_DTYPE,
                                                        device=Y.device)], dim=1)
            pi = torch.softmax(eta_full, dim=1)
            pi = torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

            if param_change < self.pql_tol and outer > 0:
                converged = True
                break

        # Log-likelihood (vectorized)
        ll = torch.log(pi[torch.arange(n, device=Y.device), Y_int].clamp(min=1e-300)).sum().item()

        logger.info(
            "MultinomialGLMM null fit: n=%d, J=%d, converged=%s, "
            "sig2_g=%.4e, sig2_e=%.4e",
            n, J, converged, sig2_g, sig2_e,
        )

        nf = NullFit(
            sig2_g=sig2_g,
            sig2_e=sig2_e,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            Y_rot=Y,
            X0_rot=X0,
            b0=B.reshape(-1),
            log_likelihood=ll,
            converged=converged,
            device=Y.device,
        )

        nf._glm_family = "multinomial_glmm"
        nf._glm_pi = pi
        nf._glm_B = B
        nf._glm_n_classes = J

        # Store null information matrix for Schur complement in score test
        I_null = torch.zeros((J - 1) * c, (J - 1) * c,
                             dtype=STAT_DTYPE, device=Y.device)
        for jj in range(J - 1):
            for kk in range(J - 1):
                if jj == kk:
                    w_jk = pi[:, jj] * (1.0 - pi[:, jj])
                else:
                    w_jk = -pi[:, jj] * pi[:, kk]
                WX = w_jk.unsqueeze(1) * X0
                I_null[jj * c:(jj + 1) * c, kk * c:(kk + 1) * c] = X0.T @ WX
        nf._glm_I_null = I_null

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via multi-df multinomial score test.

        Same formula as MultinomialGLM but using PQL-fitted pi.
        """
        if test not in ("score",):
            raise ValueError(
                f"MultinomialGLMM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot
        pi = null_fit._glm_pi
        J = null_fit._glm_n_classes
        Y_int = Y.long()

        Y_onehot = torch.zeros(n, J, dtype=STAT_DTYPE, device=G_chunk.device)
        Y_onehot.scatter_(1, Y_int.unsqueeze(1), 1.0)

        # Vectorized multi-df score test across all SNPs simultaneously
        X0 = null_fit.X0_rot
        c = X0.shape[1]
        R = Y_onehot[:, :J - 1] - pi[:, :J - 1]  # (n, J-1)
        U_all = G_chunk.T @ R  # (m, J-1)

        # Build V_raw and cross-info C for Schur complement
        G2 = G_chunk ** 2  # (n, m)
        V_raw = torch.zeros(m, J - 1, J - 1, dtype=STAT_DTYPE,
                            device=G_chunk.device)
        C_all = torch.zeros(m, J - 1, (J - 1) * c, dtype=STAT_DTYPE,
                            device=G_chunk.device)

        for j in range(J - 1):
            for k in range(J - 1):
                if j == k:
                    w = pi[:, j] * (1.0 - pi[:, j])
                else:
                    w = -pi[:, j] * pi[:, k]
                V_raw[:, j, k] = G2.T @ w
                wX = w.unsqueeze(1) * X0  # (n, c)
                C_all[:, j, k * c:(k + 1) * c] = G_chunk.T @ wX

        # Schur complement: V_adj = V_raw - C @ I_null^{-1} @ C'
        I_null = null_fit._glm_I_null
        try:
            I_inv_Ct = torch.linalg.solve(
                I_null.unsqueeze(0),
                C_all.transpose(1, 2))
            correction = torch.bmm(C_all, I_inv_Ct)
            V_all = V_raw - correction
        except Exception:
            V_all = V_raw

        # Batched solve
        try:
            V_inv_U = torch.linalg.solve(V_all, U_all.unsqueeze(2))
            stat_all = (U_all.unsqueeze(2) * V_inv_U).sum(dim=1).squeeze(1)
            stat_all = torch.clamp(stat_all, min=0.0)
        except Exception:
            stat_all = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)
            for s in range(m):
                try:
                    V_inv_U_s = torch.linalg.solve(V_all[s], U_all[s])
                    stat_all[s] = max((U_all[s] * V_inv_U_s).sum().item(), 0.0)
                except Exception:
                    stat_all[s] = 0.0

        from ..stats.tests import chi2_sf
        p_all = chi2_sf(stat_all, df=J - 1)

        beta_nan = torch.full((m,), float("nan"), dtype=STAT_DTYPE,
                              device=G_chunk.device)
        se_nan = torch.full((m,), float("nan"), dtype=STAT_DTYPE,
                            device=G_chunk.device)
        af = G_chunk.mean(dim=0) / float(self.ploidy)

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta_nan,
            se=se_nan,
            stat=stat_all,
            p=p_all,
            test="score",
        )
