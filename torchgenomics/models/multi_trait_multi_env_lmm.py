"""Multi-Trait Multi-Environment LMM (MT-MET GWAS).

Combines multi-trait (d traits) and multi-environment (E environments) into
a single model with dE pseudo-traits.  Supports three Vg structures:

1. **Unstructured**: Vg is (dE, dE) — full flexibility, delegates to MultiTraitLMM.
2. **Separable Kronecker**: Vg = Vg_trait(d,d) kron Vg_env(E,E) — parsimonious.
3. **FA(k)**: Factor Analytic on the full dE space.

Column ordering is trait-major: column index = t*E + e.

References:
    - Smith et al. (2001): FA and separable covariance for MET
    - Malosetti et al. (2013): Mixed models for MET GWAS in plant breeding
    - Zhou & Stephens (2014): Efficient mvLMM (GEMMA)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.eigh import eigendecompose, rotate
from ..optim.mvlmm_reml import mvlmm_null_quantities
from .base import NullFit, VariantMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTMETScanResult
# ---------------------------------------------------------------------------

@dataclass
class MTMETScanResult:
    """Per-variant association statistics from a multi-trait multi-env scan."""

    # Variant identifiers
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor  # (m,)

    # Full effects: (m, d, E)
    beta: Tensor
    se: Tensor

    # Joint test: all dE betas = 0, chi2(dE)
    stat: Tensor  # (m,)
    p: Tensor  # (m,)

    # Per-trait across envs: chi2(E)
    stat_per_trait: Tensor  # (m, d)
    p_per_trait: Tensor  # (m, d)

    # Per-env across traits: chi2(d)
    stat_per_env: Tensor  # (m, E)
    p_per_env: Tensor  # (m, E)

    # Per-trait GxE: chi2(E-1)
    stat_gxe_per_trait: Tensor  # (m, d)
    p_gxe_per_trait: Tensor  # (m, d)

    # Metadata
    test: str = "wald"
    trait_names: list[str] | None = None
    env_names: list[str] | None = None
    inference_type: str = "marginal"

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FA_PATTERN = re.compile(r"^fa\((\d+)\)$", re.IGNORECASE)


def _parse_vg_structure(vg_structure: str, dE: int) -> tuple[str, int]:
    """Parse vg_structure string.

    Returns (kind, fa_rank).  kind is "unstructured", "separable", or "fa".
    """
    low = vg_structure.lower().strip()
    if low == "unstructured":
        return "unstructured", 0
    if low == "separable":
        return "separable", 0
    m = _FA_PATTERN.match(vg_structure)
    if m is not None:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"FA rank must be >= 1, got {k}.")
        if k >= dE:
            raise ValueError(
                f"FA rank k={k} must be < dE={dE}. Use 'unstructured'."
            )
        return "fa", k
    raise ValueError(
        f"Invalid vg_structure '{vg_structure}'. "
        f"Use 'unstructured', 'separable', or 'fa(k)'."
    )


def _build_contrast_matrix(size: int, device: torch.device) -> Tensor:
    """Successive-differences contrast matrix (size-1, size)."""
    C = torch.zeros(size - 1, size, dtype=STAT_DTYPE, device=device)
    for i in range(size - 1):
        C[i, i] = 1.0
        C[i, i + 1] = -1.0
    return C


def _chi2_sf(stat: Tensor, df: int) -> Tensor:
    """P-values from chi2 statistics."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)


def _wald_subblock(
    beta_flat: Tensor,
    Var_beta: Tensor,
    indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute Wald stat and p for a sub-block of beta.

    Parameters
    ----------
    beta_flat : (m, dE)
    Var_beta : (m, dE, dE)
    indices : (k,) integer indices into the dE dimension.

    Returns
    -------
    stat : (m,) Wald chi2(k)
    p : (m,) p-values
    """
    beta_sub = beta_flat[:, indices]  # (m, k)
    Var_sub = Var_beta[:, indices][:, :, indices]  # (m, k, k)

    Var_inv_b = torch.linalg.solve(Var_sub, beta_sub.unsqueeze(-1)).squeeze(-1)
    stat = torch.clamp((beta_sub * Var_inv_b).sum(dim=-1), min=0.0)
    p = _chi2_sf(stat, df=len(indices))
    return stat, p


def _wald_contrast(
    beta_sub: Tensor,
    Var_sub: Tensor,
    C: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute Wald stat for contrast C on a sub-block.

    Parameters
    ----------
    beta_sub : (m, k)
    Var_sub : (m, k, k)
    C : (k-1, k) contrast matrix

    Returns
    -------
    stat : (m,) chi2(k-1)
    p : (m,)
    """
    Cb = torch.einsum('de,me->md', C, beta_sub)  # (m, k-1)
    CVCt = torch.einsum('de,mef,gf->mdg', C, Var_sub, C)  # (m, k-1, k-1)
    CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
    stat = torch.clamp((Cb * CVCt_inv_Cb).sum(dim=-1), min=0.0)
    p = _chi2_sf(stat, df=C.shape[0])
    return stat, p


# ---------------------------------------------------------------------------
# GLS Wald scan (reused from MultiTraitLMM pattern)
# ---------------------------------------------------------------------------

def _gls_wald_scan(
    G_rot: Tensor,
    null_fit: NullFit,
    dE: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Full-system GLS Wald scan for dE pseudo-traits.

    Returns
    -------
    beta_flat : (m, dE)
    se_flat : (m, dE)
    Var_beta : (m, dE, dE)
    stat_joint : (m,)
    """
    n, m = G_rot.shape
    device = G_rot.device

    W = null_fit.weights       # (n, dE, dE)
    Y_rot = null_fit.Y_rot     # (n, dE)
    X0_rot = null_fit.X0_rot   # (n, c)
    M00 = null_fit.M00         # (c*dE, c*dE)
    b0 = null_fit.b0           # (c*dE,)
    c = X0_rot.shape[1]

    # Weighted phenotype
    WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, dE)

    # M11: (m, dE, dE) — g^T W g per SNP
    g2 = G_rot ** 2
    M11 = torch.einsum('ni,nst->ist', g2, W)

    # M01: (m, c*dE, dE) — X0^T W g per SNP
    M01_raw = torch.einsum('ia,its,ij->jats', X0_rot, W, G_rot)  # (m, c, dE, dE)
    # Reshape: the (c, dE) covariates are blocked as cd rows
    # M01_raw[j, a, t, s] but we need (m, c*dE, dE)
    # Reorder: for each SNP j, the (a,t) pairs form cd rows
    M01 = M01_raw.reshape(m, c * dE, dE)

    # b1: (m, dE) — g^T W Y per SNP
    b1 = torch.einsum('ij,is->js', G_rot, WY)

    # Build full (c*dE + dE) x (c*dE + dE) system per SNP
    dim_full = c * dE + dE
    M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE, device=device)
    M_full[:, :c*dE, :c*dE] = M00.unsqueeze(0)
    M_full[:, :c*dE, c*dE:] = M01
    M_full[:, c*dE:, :c*dE] = M01.transpose(1, 2)
    M_full[:, c*dE:, c*dE:] = M11

    rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE, device=device)
    rhs_full[:, :c*dE] = b0.unsqueeze(0)
    rhs_full[:, c*dE:] = b1

    # Guard monomorphic SNPs
    diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
    singular_mask = diag_min < 1e-10
    if singular_mask.any():
        jitter = torch.eye(dim_full, dtype=STAT_DTYPE, device=device) * 1e-10
        M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

    # Solve for beta
    coef = torch.linalg.solve(M_full, rhs_full)  # (m, dim_full)
    beta_flat = coef[:, c*dE:]  # (m, dE)

    # Var(beta): solve M_full @ X = [0; I_dE]
    rhs_var = torch.zeros(m, dim_full, dE, dtype=STAT_DTYPE, device=device)
    rhs_var[:, c*dE:, :] = torch.eye(dE, dtype=STAT_DTYPE, device=device).unsqueeze(0)
    Minv_cols = torch.linalg.solve(M_full, rhs_var)  # (m, dim_full, dE)
    Var_beta = Minv_cols[:, c*dE:, :]  # (m, dE, dE)

    # SE
    se_flat = torch.sqrt(
        torch.clamp(torch.diagonal(Var_beta, dim1=-2, dim2=-1), min=1e-30)
    )  # (m, dE)

    # Joint Wald: beta^T Var^{-1} beta ~ chi2(dE)
    Var_inv_beta = torch.linalg.solve(Var_beta, beta_flat.unsqueeze(-1)).squeeze(-1)
    stat_joint = torch.clamp((beta_flat * Var_inv_beta).sum(dim=-1), min=0.0)

    return beta_flat, se_flat, Var_beta, stat_joint


# ---------------------------------------------------------------------------
# KED Score test scan (scalable path for large dE)
# ---------------------------------------------------------------------------

# Threshold for switching from Wald to score test
_KED_THRESHOLD = 64


def _score_test_scan_ked(
    G_rot: Tensor,
    null_fit: NullFit,
    dE: int,
    d: int,
    E: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Score test scan using KED diagonal precision — O(m*n*dE).

    Uses precomputed diagonal weights and KED-basis residuals from fit_null.
    No per-SNP matrix solve needed.

    Returns
    -------
    stat_joint, p_joint : (m,)  — chi2(dE) joint test
    stat_per_trait, p_per_trait : (m, d) — chi2(E) per-trait
    stat_per_env, p_per_env : (m, E) — chi2(d) per-env
    """
    n, m = G_rot.shape
    device = G_rot.device

    W_diag = null_fit._ked_W_diag        # (n, dE)
    R_ked = null_fit._ked_residuals       # (n, dE)
    XtWX_inv = null_fit._ked_XtWX_inv    # (dE, c, c)
    X0_rot = null_fit.X0_rot             # (n, c)
    c = X0_rot.shape[1]

    # Rotate genotypes into KED basis
    Tt = null_fit._ked_Tt  # (d, d)
    Te = null_fit._ked_Te  # (E, E)

    # Score vector: U[s, j] = G_rot[:, s]' @ (W_diag[:, j] * R_ked[:, j])
    WR = W_diag * R_ked  # (n, dE)
    U = G_rot.T @ WR  # (m, dE)

    # Raw variance: V_raw[s, j] = sum_i G_rot[i,s]^2 * W_diag[i, j]
    G2 = G_rot ** 2  # (n, m)
    V_raw = G2.T @ W_diag  # (m, dE)

    # Schur complement for covariates:
    # correction[s, j] = (G' W[:,j] X0) @ XtWX_inv[j] @ (X0' W[:,j] G)
    # For each j: GtWX[s, j, a] = sum_i G[i,s] * W[i,j] * X0[i,a]
    # This is (m, dE, c) — can be large, but manageable
    # correction[s, j] = GtWX[s,j,:] @ XtWX_inv[j] @ GtWX[s,j,:]'

    # Compute per-j cross-information efficiently
    # GtWX_j = G_rot.T @ diag(W[:,j]) @ X0 for each j — vectorized:
    WX = W_diag.unsqueeze(2) * X0_rot.unsqueeze(1)  # (n, dE, c)
    GtWX = torch.einsum('ni,njc->ijc', G_rot, WX)  # (m, dE, c)

    # correction[s, j] = GtWX[s,j,:] @ XtWX_inv[j] @ GtWX[s,j,:]'
    # = sum_ab GtWX[s,j,a] * XtWX_inv[j,a,b] * GtWX[s,j,b]
    corr = torch.einsum('mjc,jcd,mjd->mj', GtWX, XtWX_inv, GtWX)  # (m, dE)

    V_adj = (V_raw - corr).clamp(min=1e-20)  # (m, dE)

    # Joint test statistic: sum_j U_j^2 / V_j ~ chi2(dE) (diagonal approx)
    stat_joint = (U ** 2 / V_adj).sum(dim=1)  # (m,)
    p_joint = _chi2_sf(stat_joint, df=dE)

    # Per-trait tests: chi2(E) — sum over environments within each trait
    stat_per_trait = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
    p_per_trait = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
    for t in range(d):
        idx = slice(t * E, (t + 1) * E)
        stat_t = (U[:, idx] ** 2 / V_adj[:, idx]).sum(dim=1)
        stat_per_trait[:, t] = stat_t
        p_per_trait[:, t] = _chi2_sf(stat_t, df=E)

    # Per-env tests: chi2(d) — sum over traits within each env
    stat_per_env = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
    p_per_env = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
    for e in range(E):
        idx = torch.arange(d, device=device) * E + e
        stat_e = (U[:, idx] ** 2 / V_adj[:, idx]).sum(dim=1)
        stat_per_env[:, e] = stat_e
        p_per_env[:, e] = _chi2_sf(stat_e, df=d)

    return (stat_joint, p_joint, stat_per_trait, p_per_trait,
            stat_per_env, p_per_env, U, V_adj)


# ---------------------------------------------------------------------------
# MultiTraitMultiEnvLMM
# ---------------------------------------------------------------------------

class MultiTraitMultiEnvLMM:
    """Multi-trait multi-environment LMM (MT-MET).

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.

    Parameters
    ----------
    config : NumericalConfig, optional
    vg_structure : str
        ``"separable"`` (default) — Vg = Vg_trait kron Vg_env.
        ``"unstructured"`` — full (dE, dE) Vg.
        ``"fa(k)"`` — factor analytic on the dE space.
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        vg_structure: str = "separable",
    ) -> None:
        self.config = config or NumericalConfig()
        self.vg_structure = vg_structure

    # ------------------------------------------------------------------
    # fit_null
    # ------------------------------------------------------------------

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        *,
        n_traits: int | None = None,
        n_envs: int | None = None,
        trait_names: list[str] | None = None,
        env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the MT-MET null model.

        Parameters
        ----------
        Y : (n, d, E) or (n, dE) phenotype matrix.
            If 3D, reshaped to (n, dE) in trait-major order.
            If 2D, n_traits and n_envs must be provided.
        X0 : (n, c) covariates (intercept as first column).
        K : (n, n) GRM / kinship matrix.
        n_traits, n_envs : required if Y is 2D.
        trait_names : list of trait labels.
        env_names : list of environment labels.
        """
        if K is None:
            raise ValueError("MultiTraitMultiEnvLMM requires a kinship matrix K.")

        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        # --- Determine d and E ---
        if Y.ndim == 3:
            n, d, E = Y.shape
            Y_wide = Y.reshape(n, d * E)  # trait-major
        elif Y.ndim == 2:
            n, dE = Y.shape
            if n_traits is None or n_envs is None:
                raise ValueError(
                    "For 2D Y, must provide n_traits and n_envs."
                )
            d, E = n_traits, n_envs
            if d * E != dE:
                raise ValueError(
                    f"n_traits({d}) * n_envs({E}) = {d*E} != Y.shape[1]={dE}"
                )
            Y_wide = Y
        else:
            raise ValueError(f"Y must be 2D or 3D, got {Y.ndim}D.")

        dE = d * E
        if d < 2:
            raise ValueError(
                f"Need d >= 2 traits (got {d}). Use MultiEnvLMM for single-trait."
            )
        if E < 2:
            raise ValueError(
                f"Need E >= 2 environments (got {E}). Use MultiTraitLMM for single-env."
            )

        # --- Complete-case filter ---
        has_nan = torch.isnan(Y_wide).any(dim=1)
        if has_nan.any():
            keep = ~has_nan
            idx = keep.nonzero(as_tuple=True)[0]
            Y_wide = Y_wide[idx]
            X0 = X0[idx]
            K = K[idx][:, idx]
            n = Y_wide.shape[0]
            logger.info(
                "Complete-case filter: dropped %d rows with NaN, %d remain.",
                has_nan.sum().item(), n,
            )

        # --- Parse Vg structure ---
        vg_kind, fa_rank = _parse_vg_structure(self.vg_structure, dE)

        # --- Eigendecompose K ---
        ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)
        Y_rot = rotate(Y_wide, ed.eigenvectors)
        X0_rot = rotate(X0, ed.eigenvectors)

        # --- Initialize from sample covariance ---
        c = X0_rot.shape[1]
        R = Y_rot - X0_rot @ torch.linalg.lstsq(X0_rot, Y_rot).solution
        S_cov = (R.T @ R) / max(n - c, 1)
        S_cov = S_cov + torch.eye(dE, dtype=STAT_DTYPE, device=Y.device) * 1e-6

        # --- Fit by structure ---
        if vg_kind == "unstructured":
            from ..optim.lbfgs_reml import lbfgs_reml
            Vg_init = S_cov * 0.5
            Ve_init = S_cov * 0.5
            Vg, Ve, ll, opt_trace = lbfgs_reml(
                Y_rot, X0_rot, ed.eigenvalues,
                n_traits=dE,
                max_iter=self.config.reml_max_iter,
                tol=self.config.reml_convergence_tol,
                Vg_init=Vg_init, Ve_init=Ve_init,
            )

        elif vg_kind == "separable":
            from ..optim.separable_kron_reml import separable_kron_reml

            # Extract marginal initial estimates for trait and env components
            # Trait cov: average over env pairs
            Vg_trait_init = torch.zeros(d, d, dtype=STAT_DTYPE, device=Y.device)
            for t1 in range(d):
                for t2 in range(d):
                    # Average S_cov[t1*E+e, t2*E+e] across e
                    val = 0.0
                    for e in range(E):
                        val += S_cov[t1 * E + e, t2 * E + e].item()
                    Vg_trait_init[t1, t2] = val / E * 0.5

            Vg_env_init = torch.zeros(E, E, dtype=STAT_DTYPE, device=Y.device)
            for e1 in range(E):
                for e2 in range(E):
                    val = 0.0
                    for t in range(d):
                        val += S_cov[t * E + e1, t * E + e2].item()
                    Vg_env_init[e1, e2] = val / d * 0.5

            # Ensure SPD
            Vg_trait_init = Vg_trait_init + torch.eye(d, dtype=STAT_DTYPE, device=Y.device) * 1e-4
            Vg_env_init = Vg_env_init + torch.eye(E, dtype=STAT_DTYPE, device=Y.device) * 1e-4

            Ve_init = S_cov * 0.5

            Vg_trait, Vg_env, Ve, ll, opt_trace = separable_kron_reml(
                Y_rot, X0_rot, ed.eigenvalues,
                n_traits=d, n_envs=E,
                max_iter=self.config.reml_max_iter,
                tol=self.config.reml_convergence_tol,
                Vg_trait_init=Vg_trait_init,
                Vg_env_init=Vg_env_init,
                Ve_init=Ve_init,
            )
            Vg = torch.kron(Vg_trait, Vg_env)

        elif vg_kind == "fa":
            from ..optim.fa_lbfgs_reml import fa_lbfgs_reml
            Vg_init = S_cov * 0.5
            Ve_init = S_cov * 0.5
            Vg, Ve, ll, opt_trace, _Lambda, _psi = fa_lbfgs_reml(
                Y_rot, X0_rot, ed.eigenvalues,
                n_envs=dE,
                fa_rank=fa_rank,
                max_iter=self.config.reml_max_iter,
                tol=self.config.reml_convergence_tol,
                Vg_init=Vg_init, Ve_init=Ve_init,
            )
        else:
            raise ValueError(f"Unknown vg_kind: {vg_kind}")

        # --- Precompute null quantities ---
        null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, ed.eigenvalues)

        converged = len(opt_trace) > 1 and (
            opt_trace[-1].get("ll", float("-inf"))
            >= opt_trace[-2].get("ll", float("-inf")) - 1e-3
        )

        nf = NullFit(
            Vg=Vg,
            Ve=Ve,
            eigenvalues=ed.eigenvalues,
            eigenvectors=ed.eigenvectors,
            Y_rot=Y_rot,
            X0_rot=X0_rot,
            M00=null_q["M00"],
            b0=null_q["b0"],
            weights=null_q["W"],
            log_likelihood=ll,
            optimizer_trace=opt_trace,
            converged=converged,
            device=ed.eigenvalues.device,
        )

        # Store MT-MET metadata
        nf.n_traits = d
        nf.n_envs = E
        nf.trait_names = trait_names or [f"trait_{i}" for i in range(d)]
        nf.env_names = env_names or [f"env_{i}" for i in range(E)]
        nf.vg_structure = self.vg_structure

        if vg_kind == "separable":
            nf.Vg_trait = Vg_trait
            nf.Vg_env = Vg_env

            # For large dE, precompute KED quantities for score test scan
            if dE > _KED_THRESHOLD:
                from ..linalg.kronecker_eed import ked_reml_quantities, kronecker_eed_from_full
                ked = kronecker_eed_from_full(Vg, Ve, d, E)
                ked_q = ked_reml_quantities(ked, ed.eigenvalues, Y_rot, X0_rot)
                nf._ked_W_diag = ked_q["W_diag"]
                nf._ked_residuals = ked_q["residuals_ked"]
                nf._ked_XtWX_inv = ked_q["XtWX_inv_blocks"]
                nf._ked_Tt = ked_q["Tt"]
                nf._ked_Te = ked_q["Te"]
                nf._ked_available = True
                logger.info(
                    "KED precomputed for score test: dE=%d > threshold=%d",
                    dE, _KED_THRESHOLD,
                )
            else:
                nf._ked_available = False

        if vg_kind == "fa":
            nf.fa_Lambda = _Lambda
            nf.fa_psi = _psi
            nf.fa_rank = fa_rank

        logger.info(
            "MT-MET null fit: d=%d traits, E=%d envs, vg=%s, ll=%.4f, converged=%s",
            d, E, self.vg_structure, ll, converged,
        )

        return nf

    # ------------------------------------------------------------------
    # score_chunk
    # ------------------------------------------------------------------

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> MTMETScanResult:
        """Score a genotype chunk with MT-MET Wald tests.

        Returns MTMETScanResult with joint, per-trait, per-env, and GxE tests.
        """
        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        device = G_chunk.device

        d = null_fit.n_traits
        E = null_fit.n_envs
        dE = d * E

        # Handle empty chunk
        if m == 0:
            empty = torch.zeros(0, dtype=STAT_DTYPE, device=device)
            return MTMETScanResult(
                chr=[], pos=[], snp=[], a1=[], a2=[],
                af=empty,
                beta=torch.zeros(0, d, E, dtype=STAT_DTYPE, device=device),
                se=torch.zeros(0, d, E, dtype=STAT_DTYPE, device=device),
                stat=empty, p=empty,
                stat_per_trait=torch.zeros(0, d, dtype=STAT_DTYPE, device=device),
                p_per_trait=torch.zeros(0, d, dtype=STAT_DTYPE, device=device),
                stat_per_env=torch.zeros(0, E, dtype=STAT_DTYPE, device=device),
                p_per_env=torch.zeros(0, E, dtype=STAT_DTYPE, device=device),
                stat_gxe_per_trait=torch.zeros(0, d, dtype=STAT_DTYPE, device=device),
                p_gxe_per_trait=torch.zeros(0, d, dtype=STAT_DTYPE, device=device),
                test=test,
                trait_names=null_fit.trait_names,
                env_names=null_fit.env_names,
            )

        # Allele frequency
        af = G_chunk.mean(dim=0) / 2.0

        # Rotate genotypes
        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)

        # --- Tiered dispatch: KED score test for large dE, Wald for small ---
        use_ked = (
            dE > _KED_THRESHOLD
            and getattr(null_fit, '_ked_available', False)
        )

        if use_ked:
            # Score test path — O(m*n*dE), no per-SNP matrix solve
            (stat_joint, p_joint, stat_per_trait, p_per_trait,
             stat_per_env, p_per_env, U_score, V_adj) = _score_test_scan_ked(
                G_rot, null_fit, dE, d, E,
            )
            # Approximate beta/SE from score: beta ≈ U/V, SE ≈ 1/sqrt(V)
            beta_flat = U_score / V_adj
            se_flat = 1.0 / torch.sqrt(V_adj)
            beta = beta_flat.reshape(m, d, E)
            se = se_flat.reshape(m, d, E)

            # GxE tests from score test (difference across envs)
            stat_gxe = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            p_gxe = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            if E > 1:
                for t in range(d):
                    idx = slice(t * E, (t + 1) * E)
                    U_t = U_score[:, idx]  # (m, E)
                    V_t = V_adj[:, idx]    # (m, E)
                    # GxE: test whether effects are equal across envs
                    # Use difference contrast: E-1 contrasts
                    # Score-based: diff = U[:,e] / sqrt(V[:,e]) - U[:,0] / sqrt(V[:,0])
                    # Simplified: sum of squared standardized differences
                    z_t = U_t / torch.sqrt(V_t.clamp(min=1e-20))  # (m, E)
                    z_mean = z_t.mean(dim=1, keepdim=True)
                    gxe_stat = ((z_t - z_mean) ** 2).sum(dim=1)
                    stat_gxe[:, t] = gxe_stat
                    p_gxe[:, t] = _chi2_sf(gxe_stat, df=E - 1)
            test = "score"
        else:
            # Wald test path — exact, for small dE
            beta_flat, se_flat, Var_beta, stat_joint = _gls_wald_scan(
                G_rot, null_fit, dE,
            )
            p_joint = _chi2_sf(stat_joint, df=dE)
            beta = beta_flat.reshape(m, d, E)
            se = se_flat.reshape(m, d, E)

            stat_per_trait = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            p_per_trait = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            for t in range(d):
                idx = torch.arange(t * E, (t + 1) * E, device=device)
                s, p_ = _wald_subblock(beta_flat, Var_beta, idx)
                stat_per_trait[:, t] = s
                p_per_trait[:, t] = p_

            stat_per_env = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
            p_per_env = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
            for e in range(E):
                idx = torch.arange(d, device=device) * E + e
                s, p_ = _wald_subblock(beta_flat, Var_beta, idx)
                stat_per_env[:, e] = s
                p_per_env[:, e] = p_

            stat_gxe = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            p_gxe = torch.zeros(m, d, dtype=STAT_DTYPE, device=device)
            if E > 1:
                C_env = _build_contrast_matrix(E, device)
                for t in range(d):
                    idx = slice(t * E, (t + 1) * E)
                    beta_t = beta_flat[:, idx]
                    Var_t = Var_beta[:, idx, idx]
                    s, p_ = _wald_contrast(beta_t, Var_t, C_env)
                    stat_gxe[:, t] = s
                    p_gxe[:, t] = p_

        return MTMETScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,
            se=se,
            stat=stat_joint,
            p=p_joint,
            stat_per_trait=stat_per_trait,
            p_per_trait=p_per_trait,
            stat_per_env=stat_per_env,
            p_per_env=p_per_env,
            stat_gxe_per_trait=stat_gxe,
            p_gxe_per_trait=p_gxe,
            test=test,
            trait_names=null_fit.trait_names,
            env_names=null_fit.env_names,
        )

    # ------------------------------------------------------------------
    # Interpretive methods
    # ------------------------------------------------------------------

    @staticmethod
    def genetic_correlation_traits(null_fit: NullFit) -> Tensor:
        """Genetic correlation between traits (d, d).

        For separable structure, uses Vg_trait directly.
        Otherwise, extracts marginal trait blocks from full Vg.
        """
        d = null_fit.n_traits
        E = null_fit.n_envs

        if hasattr(null_fit, "Vg_trait") and null_fit.Vg_trait is not None:
            Vg_t = null_fit.Vg_trait
        else:
            # Extract: Vg_t[t1, t2] = average of Vg[t1*E+e, t2*E+e] over e
            Vg = null_fit.Vg
            Vg_t = torch.zeros(d, d, dtype=Vg.dtype, device=Vg.device)
            for t1 in range(d):
                for t2 in range(d):
                    val = 0.0
                    for e in range(E):
                        val += Vg[t1 * E + e, t2 * E + e].item()
                    Vg_t[t1, t2] = val / E

        diag = Vg_t.diag().clamp(min=1e-20)
        D_inv = 1.0 / diag.sqrt()
        return Vg_t * D_inv.unsqueeze(0) * D_inv.unsqueeze(1)

    @staticmethod
    def genetic_correlation_envs(null_fit: NullFit) -> Tensor:
        """Genetic correlation between environments (E, E).

        For separable structure, uses Vg_env directly.
        Otherwise, extracts marginal env blocks from full Vg.
        """
        d = null_fit.n_traits
        E = null_fit.n_envs

        if hasattr(null_fit, "Vg_env") and null_fit.Vg_env is not None:
            Vg_e = null_fit.Vg_env
        else:
            Vg = null_fit.Vg
            Vg_e = torch.zeros(E, E, dtype=Vg.dtype, device=Vg.device)
            for e1 in range(E):
                for e2 in range(E):
                    val = 0.0
                    for t in range(d):
                        val += Vg[t * E + e1, t * E + e2].item()
                    Vg_e[e1, e2] = val / d

        diag = Vg_e.diag().clamp(min=1e-20)
        D_inv = 1.0 / diag.sqrt()
        return Vg_e * D_inv.unsqueeze(0) * D_inv.unsqueeze(1)

    @staticmethod
    def per_trait_per_env_heritability(null_fit: NullFit) -> Tensor:
        """Per-trait per-env heritability (d, E)."""
        d = null_fit.n_traits
        E = null_fit.n_envs
        Vg = null_fit.Vg
        Ve = null_fit.Ve

        h2 = torch.zeros(d, E, dtype=Vg.dtype, device=Vg.device)
        for t in range(d):
            for e in range(E):
                idx = t * E + e
                vg = Vg[idx, idx].item()
                ve = Ve[idx, idx].item()
                h2[t, e] = vg / max(vg + ve, 1e-20)
        return h2
