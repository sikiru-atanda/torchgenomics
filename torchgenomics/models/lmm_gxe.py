"""Gene-Environment Interaction LMM models (Phase 16).

Two complementary models:

1. HetLMM (single-trait): y = Xβ + gγ + (g⊙z)η + u + ε
   Lightweight, uses eigendecomposition trick, 2-column Schur complement.
   Three tests per SNP: main (1df), interaction (1df), joint (2df).

2. GxELMM (multi-trait Kronecker): V = K⊗Vg + (K#env_cov)⊗Vge + I⊗Ve
   Extends mvLMM with a third variance component for GxE.
   LBFGS-autograd on 3 Cholesky factors.

References:
    - StructLMM (Moore et al., Nat Genet 2019)
    - MAGEE (Wang et al.)
    - SPAGxECCT (Bi et al., Nat Commun 2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from .base import NullFit, VariantMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GxE-specific result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GxEScanResult:
    """Extended scan result with main + interaction + joint tests."""

    # Variant identifiers
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor  # (m,)

    # Main genetic effect
    beta_main: Tensor  # (m,) or (m, d)
    se_main: Tensor
    stat_main: Tensor  # chi²(1) or chi²(d)
    p_main: Tensor

    # GxE interaction effect
    beta_interact: Tensor  # (m,) or (m, d)
    se_interact: Tensor
    stat_interact: Tensor
    p_interact: Tensor

    # Joint test (main + interaction)
    stat_joint: Tensor  # chi²(2) or chi²(2d)
    p_joint: Tensor

    test: str = "wald"
    n_obs: Tensor | None = None

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# HetLMM — single-trait GxE interaction LMM
# ---------------------------------------------------------------------------

class HetLMM:
    """Single-trait GxE interaction LMM.

    y = Xβ + gγ + (g⊙z)η + u + ε
    u ~ N(0, σ²_g K), ε ~ N(0, σ²_e I)

    Uses eigendecomposition trick (single K). The per-SNP scan
    is a 2-column Schur complement — same cost as a 2-covariate LMM.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(self, config: NumericalConfig | None = None) -> None:
        self.config = config or NumericalConfig()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        *,
        env: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model (no SNP main or interaction effect).

        Parameters
        ----------
        Y : (n,) phenotype vector.
        X0 : (n, c) covariate matrix (intercept as first column).
        K : (n, n) kinship matrix.
        env : (n,) environment variable (continuous or binary).

        Returns
        -------
        NullFit with cached rotated quantities + environment.
        """
        from ..linalg.eigh import eigendecompose, rotate
        from ..optim.controller import OptimizerController
        from ..optim.reml_math import _compute_P_quantities

        if K is None:
            raise ValueError("HetLMM requires a kinship matrix K.")
        if env is None:
            raise ValueError("HetLMM requires an environment variable 'env'.")

        Y = Y.to(STAT_DTYPE).squeeze()
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        env = env.to(STAT_DTYPE).squeeze()

        n = Y.shape[0]

        # Standardize environment variable (mean=0, std=1)
        env_mean = env.mean()
        env_std = env.std()
        if env_std > 1e-10:
            env_std_val = env_std
        else:
            env_std_val = torch.tensor(1.0, dtype=STAT_DTYPE)
        env_z = (env - env_mean) / env_std_val

        # Eigendecompose K
        ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)

        # Rotate data
        Y_rot = rotate(Y, ed.eigenvectors)
        X0_rot = rotate(X0, ed.eigenvectors)
        env_rot = rotate(env_z, ed.eigenvectors)

        # Fit variance components (null: no SNP effect)
        controller = OptimizerController(config=self.config)
        null_fit = controller.fit(
            Y_rot, X0_rot, ed.eigenvalues,
            n_traits=1,
            max_iter=self.config.reml_max_iter,
        )

        # Store eigenvectors and environment for scan
        null_fit.eigenvectors = ed.eigenvectors
        null_fit.env_rot = env_rot
        null_fit.env_z = env_z

        # Precompute M00 for Schur complement
        lam = null_fit.sig2_g / max(null_fit.sig2_e, 1e-20)
        H_inv = 1.0 / (ed.eigenvalues * lam + 1.0)
        wX = H_inv.unsqueeze(1) * X0_rot
        null_fit.M00 = X0_rot.T @ wX  # (c, c)

        # Precompute Py (P-projected Y)
        Py, beta0 = _compute_P_quantities(Y_rot, X0_rot, H_inv)
        null_fit.Py = Py
        null_fit.b0 = beta0

        logger.info(
            "HetLMM null fit: sig2_g=%.4f, sig2_e=%.4f, h2=%.4f",
            null_fit.sig2_g, null_fit.sig2_e,
            null_fit.sig2_g / (null_fit.sig2_g + null_fit.sig2_e),
        )

        return null_fit

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> GxEScanResult:
        """Score genotype chunk with 2-column design: [g, g⊙z].

        Per-SNP Wald test in rotated eigenspace using weighted LS.
        Returns main, interaction, and joint test p-values.

        Parameters
        ----------
        G_chunk : (n, m) genotype dosage.
        null_fit : output of fit_null.
        variant_meta : marker metadata.
        test : "wald" only.

        Returns
        -------
        GxEScanResult with main, interaction, and joint test results.
        """
        from ..linalg.eigh import rotate

        if test != "wald":
            raise ValueError(f"HetLMM supports 'wald' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)
        env_rot = null_fit.env_rot  # (n,)
        af = G_chunk.mean(dim=0) / 2.0

        # GxE columns: g⊙z in rotated space
        # Note: rotate(g⊙z, U) = U^T(g⊙z) ≠ (U^Tg)⊙(U^Tz)
        # Must compute g⊙z BEFORE rotation
        env_z = null_fit.env_z  # (n,) in original space
        GZ = G_chunk * env_z.unsqueeze(1)  # (n, m) element-wise
        GZ_rot = rotate(GZ, U)  # (n, m)

        # Weights
        sig2_g = null_fit.sig2_g
        sig2_e = null_fit.sig2_e
        evals = null_fit.eigenvalues
        lam = sig2_g / max(sig2_e, 1e-20)
        H_inv = 1.0 / (evals * lam + 1.0)  # (n,)

        Py = null_fit.Py  # (n,)
        X0_rot = null_fit.X0_rot  # (n, c)
        M00 = null_fit.M00  # (c, c)
        c = X0_rot.shape[1]

        # --- Build 2×2 Schur complement per SNP ---
        # Design: x1 = g_rot, x2 = gz_rot
        # M11 block: [[g^TWg, g^TWgz], [gz^TWg, gz^TWgz]] for each SNP
        wG = H_inv.unsqueeze(1) * G_rot  # (n, m)
        wGZ = H_inv.unsqueeze(1) * GZ_rot  # (n, m)

        # 2×2 blocks: (m, 2, 2)
        M11_00 = (G_rot * wG).sum(dim=0)  # (m,) g^T W g
        M11_01 = (G_rot * wGZ).sum(dim=0)  # (m,) g^T W gz
        M11_11 = (GZ_rot * wGZ).sum(dim=0)  # (m,) gz^T W gz

        # M01 block: (m, c, 2) — X0^T W [g, gz]
        M01_g = X0_rot.T @ wG  # (c, m)
        M01_gz = X0_rot.T @ wGZ  # (c, m)

        # Schur complement: S = M11 - M01^T M00^{-1} M01
        # For each SNP: S[j] (2×2) = M11[j] - M01[:,j]^T M00^{-1} M01[:,j]
        M00_inv_M01_g = torch.linalg.solve(M00, M01_g)  # (c, m)
        M00_inv_M01_gz = torch.linalg.solve(M00, M01_gz)  # (c, m)

        # Correction terms: M01^T M00^{-1} M01
        corr_00 = (M01_g * M00_inv_M01_g).sum(dim=0)  # (m,)
        corr_01 = (M01_g * M00_inv_M01_gz).sum(dim=0)  # (m,)
        corr_11 = (M01_gz * M00_inv_M01_gz).sum(dim=0)  # (m,)

        S_00 = M11_00 - corr_00  # (m,)
        S_01 = M11_01 - corr_01  # (m,)
        S_11 = M11_11 - corr_11  # (m,)

        # RHS: [g^T Py, gz^T Py]
        gPy = (G_rot * Py.unsqueeze(1)).sum(dim=0)  # (m,)
        gzPy = (GZ_rot * Py.unsqueeze(1)).sum(dim=0)  # (m,)

        # Solve 2×2 system: S @ [γ, η]^T = [gPy, gzPy]
        # Using Cramer's rule for efficiency
        det_S = S_00 * S_11 - S_01 ** 2  # (m,)
        det_S_safe = torch.clamp(det_S.abs(), min=1e-20) * det_S.sign()
        det_S_safe = torch.where(det_S_safe.abs() < 1e-20,
                                 torch.tensor(1e-20, dtype=STAT_DTYPE),
                                 det_S_safe)

        # [γ̂, η̂] via 2×2 inverse
        gamma = (S_11 * gPy - S_01 * gzPy) / det_S_safe  # (m,)
        eta = (S_00 * gzPy - S_01 * gPy) / det_S_safe  # (m,)

        # Variance: Var = sig2_e * S^{-1}
        # S^{-1} = [[S_11, -S_01], [-S_01, S_00]] / det_S
        var_gamma = sig2_e * S_11 / det_S_safe  # (m,)
        var_eta = sig2_e * S_00 / det_S_safe  # (m,)
        cov_gamma_eta = -sig2_e * S_01 / det_S_safe  # (m,)

        se_gamma = torch.sqrt(torch.clamp(var_gamma, min=1e-20))
        se_eta = torch.sqrt(torch.clamp(var_eta, min=1e-20))

        # --- Main test: γ²/Var(γ) ~ F(1, n-c-2) ---
        stat_main = gamma ** 2 / torch.clamp(var_gamma, min=1e-20)
        stat_main = torch.clamp(stat_main, min=0.0)

        # --- Interaction test: η²/Var(η) ~ F(1, n-c-2) ---
        stat_interact = eta ** 2 / torch.clamp(var_eta, min=1e-20)
        stat_interact = torch.clamp(stat_interact, min=0.0)

        # --- Joint test: [γ,η]^T S/sig2_e [γ,η] ~ F(2, n-c-2) ---
        # = (1/sig2_e) * (γ*gPy + η*gzPy)
        stat_joint = (gamma * gPy + eta * gzPy) / max(sig2_e, 1e-20)
        stat_joint = torch.clamp(stat_joint, min=0.0)

        # P-values from F-distribution
        df2 = n - c - 2  # 2 extra params: γ, η
        p_main = _f_sf(stat_main, df1=1, df2=df2)
        p_interact = _f_sf(stat_interact, df1=1, df2=df2)
        p_joint = _f_sf(stat_joint / 2.0, df1=2, df2=df2)  # F(2, df2) = chi2(2)/2

        return GxEScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta_main=gamma,
            se_main=se_gamma,
            stat_main=stat_main,
            p_main=p_main,
            beta_interact=eta,
            se_interact=se_eta,
            stat_interact=stat_interact,
            p_interact=p_interact,
            stat_joint=stat_joint,
            p_joint=p_joint,
            test="wald",
        )


# ---------------------------------------------------------------------------
# GxELMM — multi-trait Kronecker GxE interaction LMM
# ---------------------------------------------------------------------------

class GxELMM:
    """Multi-trait GxE interaction LMM with Kronecker structure.

    V = K⊗Vg + (K · diag(env_cov))⊗Vge + I⊗Ve

    Uses eigendecomposition trick. Per-individual covariance:
        Sigma_i = evals[i]*Vg + env_cov[i]*evals[i]*Vge + Ve

    Three Cholesky factors optimized via LBFGS-autograd.

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(self, config: NumericalConfig | None = None) -> None:
        self.config = config or NumericalConfig()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        *,
        env: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model with 3 variance components: Vg, Vge, Ve.

        Parameters
        ----------
        Y : (n, d) phenotype matrix (d >= 2 traits).
        X0 : (n, c) covariate matrix.
        K : (n, n) kinship matrix.
        env : (n,) environment variable.

        Returns
        -------
        NullFit with cached quantities for GxE scan.
        """
        from ..linalg.eigh import eigendecompose, rotate
        from ..optim.gxe_lbfgs_reml import gxe_lbfgs_reml

        if K is None:
            raise ValueError("GxELMM requires a kinship matrix K.")
        if env is None:
            raise ValueError("GxELMM requires an environment variable 'env'.")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)
        env = env.to(STAT_DTYPE).squeeze()

        n, d = Y.shape
        if d < 2:
            raise ValueError("GxELMM requires >= 2 traits. Use HetLMM for single-trait.")

        # Environment covariance weight per individual
        env_mean = env.mean()
        env_std = env.std()
        env_z = (env - env_mean) / (env_std if env_std > 1e-10 else 1.0)
        env_cov = env_z ** 2  # (n,)

        # Eigendecompose K
        ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)

        # Rotate data
        Y_rot = rotate(Y, ed.eigenvectors)
        X0_rot = rotate(X0, ed.eigenvectors)

        # Initialize variance components from sample covariance
        R = Y_rot - X0_rot @ torch.linalg.lstsq(X0_rot, Y_rot).solution
        S_cov = (R.T @ R) / max(n - X0.shape[1], 1)
        S_cov = S_cov + torch.eye(d, dtype=STAT_DTYPE, device=Y.device) * 1e-6

        # Fit 3-component REML
        Vg, Vge, Ve, ll, trace = gxe_lbfgs_reml(
            Y_rot, X0_rot, ed.eigenvalues, env_cov,
            n_traits=d,
            max_iter=self.config.reml_max_iter,
            tol=self.config.reml_convergence_tol,
            Vg_init=S_cov * 0.4,
            Vge_init=S_cov * 0.1,
            Ve_init=S_cov * 0.5,
        )

        # Precompute null quantities with 3-component Sigma
        # Sigma_i = evals[i]*Vg + env_cov[i]*evals[i]*Vge + Ve
        Sigma = (ed.eigenvalues.view(n, 1, 1) * Vg.unsqueeze(0) +
                 (env_cov * ed.eigenvalues).view(n, 1, 1) * Vge.unsqueeze(0) +
                 Ve.unsqueeze(0))
        L_sigma = torch.linalg.cholesky(Sigma)
        W = torch.cholesky_inverse(L_sigma)  # (n, d, d)

        # Normal equations
        XtWX = torch.einsum('ia,ist,ib->asbt', X0_rot, W, X0_rot)
        c = X0_rot.shape[1]
        XtWX = XtWX.reshape(c * d, c * d)
        WY = torch.einsum('ist,is->it', W, Y_rot)
        XtWY = torch.einsum('ia,it->at', X0_rot, WY).reshape(c * d)

        converged = len(trace) > 0 and (
            len(trace) < self.config.reml_max_iter or
            (len(trace) >= 2 and
             abs(trace[-1]["ll"] - trace[-2]["ll"]) <
             self.config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0))
        )

        logger.info(
            "GxELMM null fit: d=%d, ll=%.4f, converged=%s", d, ll, converged,
        )

        nf = NullFit(
            Vg=Vg,
            Ve=Ve,
            eigenvalues=ed.eigenvalues,
            eigenvectors=ed.eigenvectors,
            Y_rot=Y_rot,
            X0_rot=X0_rot,
            M00=XtWX,
            b0=XtWY,
            weights=W,
            log_likelihood=ll,
            optimizer_trace=trace,
            converged=converged,
            device=Y.device,
        )
        nf.Vge = Vge
        nf.env_z = env_z
        nf.env_cov = env_cov

        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> GxEScanResult:
        """Per-SNP scan: main (d df) + interaction (d df) + joint (2d df).

        Extends mvLMM full-system GLS with interaction columns [g_rot, g_rot⊙z_rot].
        """
        from ..linalg.eigh import rotate

        if test != "wald":
            raise ValueError(f"GxELMM supports 'wald' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)

        # Interaction columns: g⊙z before rotation
        env_z = null_fit.env_z  # (n,)
        GZ = G_chunk * env_z.unsqueeze(1)  # (n, m)
        GZ_rot = rotate(GZ, U)  # (n, m)

        af = G_chunk.mean(dim=0) / 2.0
        W = null_fit.weights  # (n, d, d)
        Y_rot = null_fit.Y_rot  # (n, d)
        X0_rot = null_fit.X0_rot  # (n, c)
        M00 = null_fit.M00  # (cd, cd)
        b0 = null_fit.b0  # (cd,)
        d = Y_rot.shape[1]
        c = X0_rot.shape[1]

        # Precompute W @ Y
        WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, d)

        # M11 block for [g, gz]: (m, 2d, 2d)
        g2 = G_rot ** 2  # (n, m)
        gz2 = GZ_rot ** 2  # (n, m)
        g_gz = G_rot * GZ_rot  # (n, m)

        # Main-main block: (m, d, d)
        Mgg = torch.einsum('ni,nst->ist', g2, W)
        # Interact-interact block: (m, d, d)
        Mee = torch.einsum('ni,nst->ist', gz2, W)
        # Main-interact cross block: (m, d, d)
        Mge = torch.einsum('ni,nst->ist', g_gz, W)

        # M01 blocks: X0^T W [g, gz]
        M01_g = torch.einsum('ia,its,ij->jats', X0_rot, W, G_rot).reshape(m, c * d, d)
        M01_gz = torch.einsum('ia,its,ij->jats', X0_rot, W, GZ_rot).reshape(m, c * d, d)

        # RHS: [b1_g, b1_gz]
        b1_g = torch.einsum('ij,is->js', G_rot, WY)  # (m, d)
        b1_gz = torch.einsum('ij,is->js', GZ_rot, WY)  # (m, d)

        # Build full block system per SNP: dim = cd + 2d
        dim_full = c * d + 2 * d
        M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE, device=G_chunk.device)

        # Fill blocks
        M_full[:, :c*d, :c*d] = M00.unsqueeze(0)
        M_full[:, :c*d, c*d:c*d+d] = M01_g
        M_full[:, :c*d, c*d+d:] = M01_gz
        M_full[:, c*d:c*d+d, :c*d] = M01_g.transpose(1, 2)
        M_full[:, c*d+d:, :c*d] = M01_gz.transpose(1, 2)
        M_full[:, c*d:c*d+d, c*d:c*d+d] = Mgg
        M_full[:, c*d:c*d+d, c*d+d:] = Mge
        M_full[:, c*d+d:, c*d:c*d+d] = Mge.transpose(1, 2)
        M_full[:, c*d+d:, c*d+d:] = Mee

        rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE, device=G_chunk.device)
        rhs_full[:, :c*d] = b0.unsqueeze(0)
        rhs_full[:, c*d:c*d+d] = b1_g
        rhs_full[:, c*d+d:] = b1_gz

        # Jitter for singular SNPs
        diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
        singular_mask = diag_min < 1e-10
        if singular_mask.any():
            jitter = torch.eye(dim_full, dtype=STAT_DTYPE, device=M_full.device) * 1e-10
            M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

        # Solve full system
        coef = torch.linalg.solve(M_full, rhs_full)  # (m, dim_full)
        beta_main = coef[:, c*d:c*d+d]  # (m, d)
        beta_interact = coef[:, c*d+d:]  # (m, d)

        # Variance from inverse
        rhs_var = torch.zeros(m, dim_full, 2*d, dtype=STAT_DTYPE, device=G_chunk.device)
        rhs_var[:, c*d:c*d+d, :d] = torch.eye(d, dtype=STAT_DTYPE, device=G_chunk.device)
        rhs_var[:, c*d+d:, d:] = torch.eye(d, dtype=STAT_DTYPE, device=G_chunk.device)
        Minv_cols = torch.linalg.solve(M_full, rhs_var)  # (m, dim_full, 2d)

        Var_main = Minv_cols[:, c*d:c*d+d, :d]  # (m, d, d)
        Var_interact = Minv_cols[:, c*d+d:, d:]  # (m, d, d)
        Var_joint = Minv_cols[:, c*d:, :]  # (m, 2d, 2d)

        se_main = torch.sqrt(torch.clamp(
            torch.diagonal(Var_main, dim1=-2, dim2=-1), min=1e-30))
        se_interact = torch.sqrt(torch.clamp(
            torch.diagonal(Var_interact, dim1=-2, dim2=-1), min=1e-30))

        # Main Wald: beta_main^T Var_main^{-1} beta_main ~ chi²(d)
        Vm_inv_bm = torch.linalg.solve(Var_main, beta_main.unsqueeze(-1)).squeeze(-1)
        stat_main = torch.clamp((beta_main * Vm_inv_bm).sum(dim=-1), min=0.0)

        # Interaction Wald: beta_interact^T Var_interact^{-1} beta_interact ~ chi²(d)
        Vi_inv_bi = torch.linalg.solve(Var_interact, beta_interact.unsqueeze(-1)).squeeze(-1)
        stat_interact = torch.clamp((beta_interact * Vi_inv_bi).sum(dim=-1), min=0.0)

        # Joint Wald: [main; interact]^T Var_joint^{-1} [main; interact] ~ chi²(2d)
        beta_joint = torch.cat([beta_main, beta_interact], dim=-1)  # (m, 2d)
        Vj_inv_bj = torch.linalg.solve(Var_joint, beta_joint.unsqueeze(-1)).squeeze(-1)
        stat_joint = torch.clamp((beta_joint * Vj_inv_bj).sum(dim=-1), min=0.0)

        # P-values
        p_main = _chi2_sf(stat_main, df=d)
        p_interact = _chi2_sf(stat_interact, df=d)
        p_joint = _chi2_sf(stat_joint, df=2*d)

        return GxEScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta_main=beta_main,
            se_main=se_main,
            stat_main=stat_main,
            p_main=p_main,
            beta_interact=beta_interact,
            se_interact=se_interact,
            stat_interact=stat_interact,
            p_interact=p_interact,
            stat_joint=stat_joint,
            p_joint=p_joint,
            test="wald",
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _f_sf(stat: Tensor, df1: int = 1, df2: int = 1) -> Tensor:
    """P-values from F-distribution."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.f.sf(stat_np, dfn=df1, dfd=df2)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)


def _chi2_sf(stat: Tensor, df: int = 1) -> Tensor:
    """P-values from chi² distribution."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
