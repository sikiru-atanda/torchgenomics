"""Multi-trait Linear Mixed Model (mvLMM): Y = XB + G + E.

Implements the EED (Efficient Eigen-Decomposition) trick from GEMMA
(Zhou & Stephens, Nat Methods 2014):

1. Eigendecompose K once
2. Rotate all data → per-individual Sigma_i = evals[i]*Vg + Ve (d×d)
3. Fit Vg, Ve via LBFGS-autograd on Cholesky factors
4. Per-SNP scan via Schur complement: O(n*d^3) per SNP

Supports joint Wald test chi2(d) and per-trait Wald test chi2(1).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import eigendecompose, rotate
from ..optim.lbfgs_reml import lbfgs_reml
from ..optim.mvlmm_reml import compute_sigma_inv, mvlmm_null_quantities
from .base import BaseModel, NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


class MultiTraitLMM:
    """Multi-trait LMM with EED trick.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(self, config: Optional[NumericalConfig] = None) -> None:
        self.config = config or NumericalConfig()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the multi-trait null model (no SNP effect).

        Parameters
        ----------
        Y : (n, d) — phenotype matrix (d >= 2 traits)
        X0 : (n, c) — covariates (intercept as first column)
        K : (n, n) — GRM / kinship matrix

        Returns
        -------
        NullFit with cached rotated quantities for per-SNP scan.
        """
        if K is None:
            raise ValueError("MultiTraitLMM requires a kinship matrix K.")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        n, d = Y.shape
        if d < 2:
            raise ValueError(
                f"MultiTraitLMM requires >= 2 traits, got {d}. "
                "Use SingleTraitLMM for 1 trait."
            )

        # --- Step 1: Eigendecompose K ---
        ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)

        # --- Step 2: Rotate data ---
        Y_rot = rotate(Y, ed.eigenvectors)  # (n, d)
        X0_rot = rotate(X0, ed.eigenvectors)  # (n, c)

        # --- Step 3: Fit Vg, Ve ---
        # Initialize from sample covariance split: this ensures the
        # off-diagonal signs start near the true values.
        R = Y_rot - X0_rot @ torch.linalg.lstsq(X0_rot, Y_rot).solution
        S_cov = (R.T @ R) / max(n - X0.shape[1], 1)
        S_cov = S_cov + torch.eye(d, dtype=STAT_DTYPE, device=Y.device) * 1e-6
        Vg_init = S_cov * 0.5
        Ve_init = S_cov * 0.5

        method = self.config.multi_trait_reml_method
        if method == "pxem_nr":
            from ..optim.pxem_nr_mvreml import pxem_nr_mvreml
            try:
                Vg, Ve, ll, trace = pxem_nr_mvreml(
                    Y_rot, X0_rot, ed.eigenvalues,
                    n_traits=d,
                    max_iter=self.config.reml_max_iter,
                    tol=self.config.reml_convergence_tol,
                    em_iters=self.config.multi_trait_em_iters,
                    Vg_init=Vg_init,
                    Ve_init=Ve_init,
                )
            except Exception as e:
                logger.warning(
                    "PX-EM+NR failed (%s), falling back to LBFGS.", e,
                )
                method = "lbfgs"  # fall through to LBFGS below

        if method == "triad":
            from ..optim.triad_reml import triad_reml
            try:
                Vg, Ve, ll, trace = triad_reml(
                    Y_rot, X0_rot, ed.eigenvalues,
                    n_traits=d,
                    max_iter=self.config.reml_max_iter,
                    tol=self.config.reml_convergence_tol,
                    Vg_init=Vg_init,
                    Ve_init=Ve_init,
                )
            except Exception as e:
                logger.warning(
                    "TRIAD failed (%s), falling back to LBFGS.", e,
                )
                method = "lbfgs"  # fall through to LBFGS below

        if method == "lbfgs":
            Vg, Ve, ll, trace = lbfgs_reml(
                Y_rot, X0_rot, ed.eigenvalues,
                n_traits=d,
                max_iter=self.config.reml_max_iter,
                tol=self.config.reml_convergence_tol,
                Vg_init=Vg_init,
                Ve_init=Ve_init,
            )

        # --- Step 4: Precompute null quantities for scan ---
        null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, ed.eigenvalues)

        converged = len(trace) > 0 and (
            len(trace) < self.config.reml_max_iter or
            (len(trace) >= 2 and
             abs(trace[-1]["ll"] - trace[-2]["ll"]) <
             self.config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0))
        )

        logger.info(
            "mvLMM null fit: d=%d traits, ll=%.4f, converged=%s", d, ll, converged,
        )

        return NullFit(
            Vg=Vg,
            Ve=Ve,
            eigenvalues=ed.eigenvalues,
            eigenvectors=ed.eigenvectors,
            Y_rot=Y_rot,
            X0_rot=X0_rot,
            M00=null_q["M00"],
            b0=null_q["b0"],
            weights=null_q["W"],  # (n, d, d) per-individual inverse covariance
            log_likelihood=ll,
            optimizer_trace=trace,
            converged=converged,
            device=ed.eigenvalues.device,
        )

    def update_null(self, null_fit: NullFit, *, max_iter: int = 100) -> NullFit:
        """Resume optimization from a previous NullFit (ASReml-R update style)."""
        from .base import update_null
        return update_null(null_fit, max_iter=max_iter, config=self.config)

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score a genotype chunk via full-system GLS solve per SNP.

        Uses GEMMA's double eigendecomposition (EED) trick to diagonalize
        the per-individual trait covariance, then solves per-trait 2×2
        GLS systems for each SNP.  This avoids the catastrophic cancellation
        that the Schur complement approach suffers for d >= 3.

        Parameters
        ----------
        G_chunk : (n, m) — genotype dosage
        null_fit : output of fit_null
        variant_meta : marker metadata
        test : "wald" (joint Wald chi2(d))

        Returns
        -------
        ScanResult with per-SNP association statistics.
        """
        if test not in ("wald",):
            raise ValueError(f"mvLMM currently supports 'wald' test, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)

        af = G_chunk.mean(dim=0) / 2.0  # diploid convention

        W = null_fit.weights  # (n, d, d)
        Y_rot = null_fit.Y_rot  # (n, d)
        X0_rot = null_fit.X0_rot  # (n, c)
        M00 = null_fit.M00  # (cd, cd)
        b0 = null_fit.b0  # (cd,)

        d = Y_rot.shape[1]
        c = X0_rot.shape[1]

        # --- Full-system GLS: build (m, (c+1)*d, (c+1)*d) and solve ---
        # This avoids the catastrophic cancellation inherent in the
        # Schur complement subtraction M11 - M01^T M00^{-1} M01.

        # Precompute W @ Y for RHS
        WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, d)

        # M11[j, s, t] = sum_i g_rot[i,j]^2 * W[i,s,t]
        g2 = G_rot ** 2  # (n, m)
        M11 = torch.einsum('ni,nst->ist', g2, W)  # (m, d, d)

        # M01_raw[j, a, t, s] = sum_i X0_rot[i,a] * W[i,t,s] * G_rot[i,j]
        M01_raw = torch.einsum('ia,its,ij->jats', X0_rot, W, G_rot)  # (m, c, d, d)
        M01 = M01_raw.reshape(m, c * d, d)  # (m, cd, d)

        # b1[j, s] = sum_i g_rot[i,j] * WY[i,s]
        b1 = torch.einsum('ij,is->js', G_rot, WY)  # (m, d)

        # Build full block system for each SNP:
        #   M_full[j] = [[M00,    M01[j] ],    rhs[j] = [b0  ]
        #                [M01[j]^T, M11[j]]]              [b1[j]]
        dim_full = c * d + d
        M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE,
                             device=G_chunk.device)
        M_full[:, :c*d, :c*d] = M00.unsqueeze(0)
        M_full[:, :c*d, c*d:] = M01
        M_full[:, c*d:, :c*d] = M01.transpose(1, 2)
        M_full[:, c*d:, c*d:] = M11

        rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE,
                               device=G_chunk.device)
        rhs_full[:, :c*d] = b0.unsqueeze(0)
        rhs_full[:, c*d:] = b1

        # Guard monomorphic SNPs: add jitter to diagonal where needed
        diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
        singular_mask = diag_min < 1e-10
        if singular_mask.any():
            jitter = torch.eye(dim_full, dtype=STAT_DTYPE,
                               device=M_full.device) * 1e-10
            M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

        # Solve for [B_hat; beta_hat] per SNP
        coef = torch.linalg.solve(M_full, rhs_full)  # (m, dim_full)
        beta = coef[:, c*d:]  # (m, d)

        # Variance of beta: Var(beta) = M_full_inv[cd:, cd:] per SNP
        # Solve M_full @ X = [0; I_d] to get the relevant columns of the inverse
        rhs_var = torch.zeros(m, dim_full, d, dtype=STAT_DTYPE,
                              device=G_chunk.device)
        rhs_var[:, c*d:, :] = torch.eye(d, dtype=STAT_DTYPE,
                                         device=G_chunk.device).unsqueeze(0)
        Minv_cols = torch.linalg.solve(M_full, rhs_var)  # (m, dim_full, d)
        Var_beta = Minv_cols[:, c*d:, :]  # (m, d, d)

        # SE: sqrt(diag(Var_beta))
        se = torch.sqrt(
            torch.clamp(torch.diagonal(Var_beta, dim1=-2, dim2=-1), min=1e-30)
        )  # (m, d)

        # Joint Wald test: W = beta^T @ Var(beta)^{-1} @ beta ~ chi2(d)
        # Use solve to avoid inverting Var_beta
        Var_inv_beta = torch.linalg.solve(Var_beta, beta.unsqueeze(-1)).squeeze(-1)
        stat_joint = (beta * Var_inv_beta).sum(dim=-1)  # (m,)

        # Clamp negative stats (numerical artifact for monomorphic SNPs)
        stat_joint = torch.clamp(stat_joint, min=0.0)

        # P-values
        p = _chi2_sf(stat_joint, df=d)

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,  # (m, d)
            se=se,  # (m, d)
            stat=stat_joint,  # (m,)
            p=p,  # (m,)
            test="wald",
        )


def _chi2_sf(stat: Tensor, df: int = 1) -> Tensor:
    """Compute p-values from chi2 statistics."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
