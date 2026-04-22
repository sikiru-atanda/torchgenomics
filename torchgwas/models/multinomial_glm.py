"""Multinomial GLM: multinomial logit regression GWAS.

Implements multinomial logistic regression for unordered categorical
phenotypes (e.g., flower colour, disease subtype). Null model fitted
via IRLS. SNP scanning uses the multi-df score test: chi2(J-1) per SNP.

Model: P(Y_i = j) = exp(X_i' beta_j) / sum_k exp(X_i' beta_k)
with reference category J (last category, beta_J = 0).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class MultinomialGLM:
    """Multinomial logistic regression GWAS.

    Conforms to :class:`BaseModel` protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    n_classes : int or None
        Number of classes J. If None, auto-detected from Y in fit_null.
    max_iter : int
        Maximum IRLS iterations (default 50).
    tol : float
        Convergence tolerance (default 1e-8).
    """

    def __init__(
        self,
        n_classes: int | None = None,
        max_iter: int = 50,
        tol: float = 1e-8,
        ploidy: int = 2,
    ) -> None:
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.ploidy = ploidy

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null multinomial logit model via IRLS.

        Parameters
        ----------
        Y : (n,) — categorical phenotype with values in {0, 1, ..., J-1}
        X0 : (n, c) — covariates (intercept as first column)
        K : ignored

        Returns
        -------
        NullFit with multinomial-GLM-specific attributes.
        """
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = X0.shape
        Y_int = Y.long()

        J = self.n_classes
        if J is None:
            J = int(Y_int.max().item()) + 1
            self.n_classes = J

        if J < 2:
            raise ValueError(f"Need at least 2 classes, got {J}")

        # Reference category: J-1 (last)
        # Parameters: (J-1) x c matrix B where B[j, :] = beta_j
        B = torch.zeros(J - 1, c, dtype=STAT_DTYPE, device=Y.device)

        converged = False

        for it in range(self.max_iter):
            # Compute class probabilities via softmax
            # eta_j = X0 @ B[j, :] for j=0..J-2, eta_{J-1} = 0 (reference)
            eta = X0 @ B.T  # (n, J-1)
            eta_full = torch.cat([eta, torch.zeros(n, 1, dtype=STAT_DTYPE,
                                                    device=Y.device)], dim=1)
            pi = torch.softmax(eta_full, dim=1)  # (n, J)
            pi = torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

            # One-hot encoding of Y
            Y_onehot = torch.zeros(n, J, dtype=STAT_DTYPE, device=Y.device)
            Y_onehot.scatter_(1, Y_int.unsqueeze(1), 1.0)

            # Residuals for non-reference classes: r_j = Y_j - pi_j
            R = Y_onehot[:, :J - 1] - pi[:, :J - 1]  # (n, J-1)

            # Score vector: (c*(J-1),) = vec([X0'R])
            score = (X0.T @ R).T.reshape(-1)  # ((J-1)*c,)

            # Hessian: block structure
            # H[j,k] = -X0' diag(pi_j (delta_jk - pi_k)) X0
            H = torch.zeros((J - 1) * c, (J - 1) * c,
                            dtype=STAT_DTYPE, device=Y.device)

            for j in range(J - 1):
                for k in range(J - 1):
                    if j == k:
                        w_jk = pi[:, j] * (1.0 - pi[:, j])
                    else:
                        w_jk = -pi[:, j] * pi[:, k]
                    WX = w_jk.unsqueeze(1) * X0
                    block = X0.T @ WX  # (c, c)
                    H[j * c:(j + 1) * c, k * c:(k + 1) * c] = -block

            # Newton step
            try:
                delta = torch.linalg.solve(-H, score)
            except Exception:
                logger.warning("Hessian solve failed at iteration %d", it)
                break

            B_flat = B.reshape(-1)
            B_new = B_flat + delta

            if delta.abs().max().item() < self.tol:
                B = B_new.reshape(J - 1, c)
                converged = True
                break

            B = B_new.reshape(J - 1, c)

        # Final probabilities
        eta = X0 @ B.T
        eta_full = torch.cat([eta, torch.zeros(n, 1, dtype=STAT_DTYPE,
                                                device=Y.device)], dim=1)
        pi = torch.softmax(eta_full, dim=1)
        pi = torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

        # Log-likelihood (vectorized)
        ll = torch.log(pi[torch.arange(n, device=Y.device), Y_int].clamp(min=1e-300)).sum().item()

        logger.info(
            "MultinomialGLM null fit: n=%d, J=%d, c=%d, converged=%s, ll=%.4f",
            n, J, c, converged, ll,
        )

        nf = NullFit(
            sig2_e=None,
            Y_rot=Y,
            X0_rot=X0,
            b0=B.reshape(-1),
            log_likelihood=ll,
            converged=converged,
            device=Y.device,
        )

        nf._glm_family = "multinomial"
        nf._glm_pi = pi  # (n, J)
        nf._glm_B = B  # (J-1, c)
        nf._glm_n_classes = J

        # Store null information matrix for Schur complement in score test
        # I_null[j*c:(j+1)*c, k*c:(k+1)*c] = X0' diag(w_jk) X0
        I_null = torch.zeros((J - 1) * c, (J - 1) * c,
                             dtype=STAT_DTYPE, device=Y.device)
        for j in range(J - 1):
            for k in range(J - 1):
                if j == k:
                    w_jk = pi[:, j] * (1.0 - pi[:, j])
                else:
                    w_jk = -pi[:, j] * pi[:, k]
                WX = w_jk.unsqueeze(1) * X0
                I_null[j * c:(j + 1) * c, k * c:(k + 1) * c] = X0.T @ WX
        nf._glm_I_null = I_null  # ((J-1)*c, (J-1)*c) information matrix

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "score",
    ) -> ScanResult:
        """Score SNP chunk via multi-df multinomial score test.

        Per SNP, the score is (J-1)-dimensional:
          U_j = g'(Y_j - pi_j) for j = 0, ..., J-2
        Variance:
          V_jk = g' diag(pi_j(delta_jk - pi_k)) g - adjustment for X0
        Test statistic: T = U' V^{-1} U ~ chi2(J-1)
        """
        if test not in ("score",):
            raise ValueError(f"MultinomialGLM supports 'score' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot
        pi = null_fit._glm_pi  # (n, J)
        J = null_fit._glm_n_classes
        X0 = null_fit.X0_rot
        Y_int = Y.long()

        # One-hot
        Y_onehot = torch.zeros(n, J, dtype=STAT_DTYPE, device=G_chunk.device)
        Y_onehot.scatter_(1, Y_int.unsqueeze(1), 1.0)

        # Vectorized multi-df score test across all SNPs simultaneously
        # Score: U_j(s) = G[:, s]' (Y_j - pi_j) for j=0..J-2
        R = Y_onehot[:, :J - 1] - pi[:, :J - 1]  # (n, J-1)
        U_all = G_chunk.T @ R  # (m, J-1)

        # Variance matrix V(s) for each SNP: V_jk(s) = G_s' diag(w_jk) G_s
        # with Schur complement adjustment for X0 covariates
        c = X0.shape[1]
        G2 = G_chunk ** 2  # (n, m)
        V_raw = torch.zeros(m, J - 1, J - 1, dtype=STAT_DTYPE,
                            device=G_chunk.device)

        # Cross-information C(s) shape (m, J-1, (J-1)*c):
        # C[s, j, k*c+l] = G[:,s]' (w_jk * X0[:,l])
        C_all = torch.zeros(m, J - 1, (J - 1) * c, dtype=STAT_DTYPE,
                            device=G_chunk.device)

        for j in range(J - 1):
            for k in range(J - 1):
                if j == k:
                    w = pi[:, j] * (1.0 - pi[:, j])  # (n,)
                else:
                    w = -pi[:, j] * pi[:, k]  # (n,)
                V_raw[:, j, k] = G2.T @ w  # (m,)
                # Cross-info: G' diag(w) X0 for each column of X0
                wX = w.unsqueeze(1) * X0  # (n, c)
                C_all[:, j, k * c:(k + 1) * c] = G_chunk.T @ wX  # (m, c)

        # Schur complement: V_adj = V_raw - C @ I_null^{-1} @ C'
        I_null = null_fit._glm_I_null  # ((J-1)*c, (J-1)*c)
        try:
            # Solve I_null @ X = C' for X, then correction = C @ X
            # C_all: (m, J-1, (J-1)*c), need C_all reshaped for batch solve
            # correction(s) = C(s) @ I_null^{-1} @ C(s)' — (J-1, J-1)
            I_inv_Ct = torch.linalg.solve(
                I_null.unsqueeze(0),
                C_all.transpose(1, 2))  # (m, (J-1)*c, J-1)
            correction = torch.bmm(C_all, I_inv_Ct)  # (m, J-1, J-1)
            V_all = V_raw - correction
        except Exception:
            V_all = V_raw

        # Batched solve: T(s) = U(s)' V(s)^{-1} U(s) for all m SNPs
        try:
            V_inv_U = torch.linalg.solve(V_all, U_all.unsqueeze(2))  # (m, J-1, 1)
            stat_all = (U_all.unsqueeze(2) * V_inv_U).sum(dim=1).squeeze(1)  # (m,)
            stat_all = torch.clamp(stat_all, min=0.0)
        except Exception:
            # Fallback: per-SNP solve for singular V matrices
            stat_all = torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device)
            for s in range(m):
                try:
                    V_inv_U_s = torch.linalg.solve(V_all[s], U_all[s])
                    stat_all[s] = max((U_all[s] * V_inv_U_s).sum().item(), 0.0)
                except Exception:
                    stat_all[s] = 0.0

        # P-values from chi2(J-1)
        from ..stats.tests import chi2_sf
        p_all = chi2_sf(stat_all, df=J - 1)

        # No single beta/SE for multinomial — store NaN
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
