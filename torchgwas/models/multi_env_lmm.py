"""Multi-Environment LMM (MET-GWAS): single trait across E discrete environments.

Supports three model families that share the same GLS scan infrastructure:

**Family 1 — Reaction-Norm** (``parameterization="reaction_norm"``):
    Decomposes per-SNP effects into stable (α) and GxE deviations (δ_e).
    α = mean(β_e), δ_e = β_e − α, with Σ δ_e = 0.
    Tests: stable H₀:α=0 (1 df), GxE H₀:δ=0 (E−1 df), any H₀:all=0 (E df).

**Family 2 — Per-Environment mvLMM** (``parameterization="per_env"``):
    Per-env betas β₁,…,β_E estimated via GLS.
    Tests: joint χ²(E), homogeneity χ²(E−1), per-env marginals χ²(1).

**Family 3 — FA(k) Vg structure** (``vg_structure="fa(k)"``):
    Vg = ΛΛ^T + diag(ψ), Λ is E×k lower-triangular.
    Fewer params: E·k − k(k−1)/2 + E vs E(E+1)/2 for unstructured.
    Compatible with both parameterizations above.

Single-kernel uses EED trick (O(n·E³)).  Multi-kernel uses direct V^{-1}
(O(n³E³)) — see Phase B2.

References:
    - Zhou & Stephens (2014): Efficient mvLMM (GEMMA)
    - Malosetti et al. (2013): Mixed models for MET in plant breeding
    - Piepho (1997): Stability analysis using mixed models
    - Smith et al. (2001): FA models for multi-environment trials
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
from .base import NullFit, VariantMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnvScanResult — per-SNP results with environment-specific fields
# ---------------------------------------------------------------------------

@dataclass
class EnvScanResult:
    """Per-variant association statistics from a multi-environment scan.

    Extends ScanResult semantics with homogeneity and per-environment tests.
    ``stat``/``p`` are the joint test (compatible with downstream pipelines).
    """

    # Variant identifiers
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor  # (m,)

    # Per-environment effect estimates
    beta: Tensor  # (m, E)
    se: Tensor  # (m, E)

    # Joint test: H₀: β_1=…=β_E=0, χ²(E)
    stat: Tensor  # (m,) — alias for stat_joint
    p: Tensor  # (m,) — alias for p_joint

    # Homogeneity test: H₀: β_1=β_2=…=β_E, χ²(E−1)
    stat_homogeneity: Tensor  # (m,)
    p_homogeneity: Tensor  # (m,)

    # Per-environment marginal tests: χ²(1) each
    stat_marginal: Tensor  # (m, E)
    p_marginal: Tensor  # (m, E)

    # Reaction-norm fields (populated when parameterization="reaction_norm")
    alpha: Tensor | None = None  # (m,) stable effect across environments
    se_alpha: Tensor | None = None  # (m,) SE of stable effect
    stat_stable: Tensor | None = None  # (m,) χ²(1) for stable effect
    p_stable: Tensor | None = None  # (m,)
    delta: Tensor | None = None  # (m, E) GxE deviations (sum to zero)
    stat_gxe: Tensor | None = None  # (m,) χ²(E-1) for GxE
    p_gxe: Tensor | None = None  # (m,)

    test: str = "wald"
    n_obs: Tensor | None = None
    env_names: list[str] | None = None
    inference_type: str = "marginal"

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def reshape_long_to_wide(
    Y_long: Tensor,
    sample_ids: Tensor,
    env_ids: Tensor,
) -> tuple[Tensor, Tensor, list[int]]:
    """Reshape long-format phenotype to wide (n, E) matrix.

    Parameters
    ----------
    Y_long : (N,) phenotype values.
    sample_ids : (N,) integer sample codes (0-based).
    env_ids : (N,) integer environment codes (0-based).

    Returns
    -------
    Y_wide : (n, E) with NaN for unobserved sample-environment pairs.
    unique_samples : (n,) unique sample indices (sorted).
    unique_envs : list of unique environment indices (sorted).
    """
    unique_samples = torch.unique(sample_ids)
    unique_envs = torch.unique(env_ids).tolist()
    n = len(unique_samples)
    E = len(unique_envs)

    # Map original sample IDs to consecutive 0..n-1
    sample_map = {int(s): i for i, s in enumerate(unique_samples.tolist())}
    env_map = {int(e): j for j, e in enumerate(unique_envs)}

    Y_wide = torch.full((n, E), float("nan"), dtype=Y_long.dtype)
    for idx in range(len(Y_long)):
        i = sample_map[int(sample_ids[idx])]
        j = env_map[int(env_ids[idx])]
        Y_wide[i, j] = Y_long[idx]

    return Y_wide, unique_samples, unique_envs


def complete_case_filter(
    Y_wide: Tensor,
    X0: Tensor,
    K: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Drop rows with any NaN across environments.

    Returns
    -------
    Y_filtered, X0_filtered, K_filtered, keep_mask (n,) boolean.
    """
    keep = ~torch.isnan(Y_wide).any(dim=1)  # (n,)
    idx = keep.nonzero(as_tuple=True)[0]
    return Y_wide[idx], X0[idx], K[idx][:, idx], keep


# ---------------------------------------------------------------------------
# Contrast matrix for homogeneity test
# ---------------------------------------------------------------------------

def _build_contrast_matrix(E: int, device: torch.device) -> Tensor:
    """Successive-differences contrast matrix (E-1, E).

    Tests H₀: β_1 = β_2 = … = β_E by testing all successive differences.
    """
    C = torch.zeros(E - 1, E, dtype=STAT_DTYPE, device=device)
    for i in range(E - 1):
        C[i, i] = 1.0
        C[i, i + 1] = -1.0
    return C


# ---------------------------------------------------------------------------
# Reaction-norm tests: stable α and GxE deviations δ_e
# ---------------------------------------------------------------------------

def _compute_reaction_norm_tests(
    beta: Tensor,
    Var_beta: Tensor,
    E: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Extract reaction-norm tests from per-env GLS estimates.

    Parameters
    ----------
    beta : (m, E) per-environment effect estimates.
    Var_beta : (m, E, E) variance-covariance of beta.
    E : number of environments.
    device : torch device.

    Returns
    -------
    alpha, se_alpha, stat_stable, p_stable, delta, stat_gxe, p_gxe
    """
    # Stable effect: α = mean(β) across environments
    alpha = beta.mean(dim=-1)  # (m,)

    # Var(α) = (1/E²) Σ_{i,j} Var_beta[:, i, j]
    ones_E = torch.ones(E, dtype=STAT_DTYPE, device=device)
    var_alpha = torch.einsum("mij,i,j->m", Var_beta, ones_E, ones_E) / (E * E)
    var_alpha = var_alpha.clamp(min=1e-30)
    se_alpha = torch.sqrt(var_alpha)  # (m,)

    # Stable test: χ²(1) = α² / Var(α)
    stat_stable = torch.clamp(alpha ** 2 / var_alpha, min=0.0)
    p_stable = _chi2_sf(stat_stable, df=1)

    # GxE deviations: δ_e = β_e − α
    delta = beta - alpha.unsqueeze(-1)  # (m, E)

    # GxE test: χ²(E-1) using contrast matrix
    C = _build_contrast_matrix(E, device)  # (E-1, E)
    Cd = torch.einsum("de,me->md", C, delta)  # (m, E-1)
    CVCt = torch.einsum("de,mef,gf->mdg", C, Var_beta, C)  # (m, E-1, E-1)
    CVCt_inv_Cd = torch.linalg.solve(CVCt, Cd.unsqueeze(-1)).squeeze(-1)
    stat_gxe = torch.clamp((Cd * CVCt_inv_Cd).sum(dim=-1), min=0.0)
    p_gxe = _chi2_sf(stat_gxe, df=E - 1)

    return alpha, se_alpha, stat_stable, p_stable, delta, stat_gxe, p_gxe


# ---------------------------------------------------------------------------
# FA structure parser
# ---------------------------------------------------------------------------

_FA_PATTERN = re.compile(r"^fa\((\d+)\)$", re.IGNORECASE)


def _parse_vg_structure(vg_structure: str, E: int) -> tuple[str, int]:
    """Parse vg_structure string and validate.

    Returns
    -------
    (kind, fa_rank) where kind is "unstructured" or "fa".
    fa_rank is 0 for unstructured.
    """
    if vg_structure.lower() == "unstructured":
        return "unstructured", 0

    m = _FA_PATTERN.match(vg_structure)
    if m is None:
        raise ValueError(
            f"Invalid vg_structure '{vg_structure}'. "
            f"Use 'unstructured' or 'fa(k)' where k is a positive integer."
        )
    k = int(m.group(1))
    if k < 1:
        raise ValueError(f"FA rank must be >= 1, got {k}.")
    if k >= E:
        raise ValueError(
            f"FA rank k={k} must be strictly less than the number of "
            f"environments E={E}. Use 'unstructured' for full-rank."
        )
    return "fa", k


# ---------------------------------------------------------------------------
# MultiEnvLMM
# ---------------------------------------------------------------------------

class MultiEnvLMM:
    """Single-trait multi-environment LMM (MET-GWAS).

    Treats E discrete environments as pseudo-traits and delegates to
    MultiTraitLMM for variance component estimation and the EED trick.

    Parameters
    ----------
    config : optional numerical configuration.
    parameterization : "per_env" (default) or "reaction_norm".
        Controls how per-SNP effects are decomposed:
        - "per_env": per-environment betas with joint/homogeneity/marginal tests
        - "reaction_norm": stable α + GxE deviations δ_e (sum-to-zero)
    vg_structure : "unstructured" (default) or "fa(k)" for factor analytic.
        Controls the structure of the genetic covariance Vg:
        - "unstructured": full E(E+1)/2 parameters (delegates to MultiTraitLMM)
        - "fa(k)": Factor Analytic with rank k, Vg = ΛΛ^T + diag(ψ)

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(
        self,
        config: NumericalConfig | None = None,
        parameterization: str = "per_env",
        vg_structure: str = "unstructured",
    ) -> None:
        if parameterization not in ("per_env", "reaction_norm"):
            raise ValueError(
                f"parameterization must be 'per_env' or 'reaction_norm', "
                f"got '{parameterization}'"
            )
        self.config = config or NumericalConfig()
        self.parameterization = parameterization
        self.vg_structure = vg_structure

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | dict[str, Tensor] | None = None,
        *,
        env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model treating environments as pseudo-traits.

        Parameters
        ----------
        Y : (n, E) — phenotype matrix, one column per environment.
            NaN entries indicate unobserved sample-environment pairs;
            rows with any NaN are dropped (complete-case analysis).
        X0 : (n, c) — covariates (intercept as first column).
        K : (n, n) GRM, or dict mapping kernel name → (n, n) matrix
            for multi-kernel models (e.g., additive + dominance).
        env_names : optional names for each environment column.

        Returns
        -------
        NullFit with Vg (E×E) genetic covariance across environments,
        Ve (E×E) residual covariance, and cached scan quantities.
        """
        if K is None:
            raise ValueError("MultiEnvLMM requires a kinship matrix K.")

        Y = Y.to(STAT_DTYPE)
        if Y.dim() == 1:
            raise ValueError(
                "Y must be (n, E) with E >= 2 environments. "
                "Got 1-d vector; use SingleTraitLMM for single environment."
            )

        n, E = Y.shape
        if E < 2:
            raise ValueError(
                f"MultiEnvLMM requires >= 2 environments, got {E}."
            )

        # Parse and validate vg_structure (raises ValueError if k >= E)
        vg_kind, fa_rank = _parse_vg_structure(self.vg_structure, E)

        # Normalize K to dict form
        is_multi_kernel = isinstance(K, dict)
        if is_multi_kernel:
            K_dict = {name: k.to(STAT_DTYPE) for name, k in K.items()}
            K_single = list(K_dict.values())[0]  # for complete_case_filter
        else:
            K_single = K.to(STAT_DTYPE)
            K_dict = None

        # Complete-case filtering
        has_nan = torch.isnan(Y).any()
        if has_nan:
            Y_filt, X0_filt, K_filt_single, keep_mask = complete_case_filter(
                Y, X0, K_single
            )
            n_dropped = n - int(keep_mask.sum())
            logger.info(
                "Complete-case filter: dropped %d of %d samples (%.1f%% missing)",
                n_dropped, n, 100.0 * n_dropped / n,
            )
            if Y_filt.shape[0] < E + X0.shape[1] + 1:
                raise ValueError(
                    f"Too few complete cases ({Y_filt.shape[0]}) for "
                    f"{E} environments + {X0.shape[1]} covariates."
                )
            # Filter all kernels with same mask
            if is_multi_kernel:
                idx = keep_mask.nonzero(as_tuple=True)[0]
                K_dict_filt = {
                    name: k[idx][:, idx] for name, k in K_dict.items()
                }
            else:
                K_dict_filt = None
        else:
            Y_filt, X0_filt = Y, X0
            K_filt_single = K_single
            keep_mask = torch.ones(n, dtype=torch.bool)
            K_dict_filt = K_dict

        # Route to appropriate fitting strategy
        if is_multi_kernel and len(K_dict) > 1:
            nf = self._fit_null_multi_kernel(
                Y_filt, X0_filt, K_dict_filt, E, vg_kind, fa_rank, **kwargs,
            )
        elif vg_kind == "fa":
            K_fit = K_filt_single if not is_multi_kernel else list(K_dict_filt.values())[0]
            nf = self._fit_null_fa(Y_filt, X0_filt, K_fit, E, fa_rank, **kwargs)
        else:
            K_fit = K_filt_single if not is_multi_kernel else list(K_dict_filt.values())[0]
            from .multi_trait_lmm import MultiTraitLMM
            mt_model = MultiTraitLMM(config=self.config)
            nf = mt_model.fit_null(Y_filt, X0_filt, K_fit, **kwargs)

        # Store environment metadata
        nf.env_names = env_names or [f"env_{i}" for i in range(E)]
        nf.n_env = E
        nf.keep_mask = keep_mask
        nf.parameterization = self.parameterization
        nf.vg_structure = self.vg_structure

        # Log interpretive quantities
        h2 = self.per_env_heritability(nf)
        logger.info(
            "MET null fit: %d environments, %d complete samples, "
            "vg_structure=%s, converged=%s",
            E, Y_filt.shape[0], self.vg_structure, nf.converged,
        )
        for e in range(E):
            logger.info(
                "  env %s: h²=%.3f", nf.env_names[e], h2[e].item(),
            )

        return nf

    def _fit_null_fa(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor,
        E: int,
        fa_rank: int,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model with FA(k) structure on Vg using EED trick."""
        from ..optim.fa_lbfgs_reml import fa_lbfgs_reml
        from ..optim.mvlmm_reml import mvlmm_null_quantities

        n = Y.shape[0]
        c = X0.shape[1]

        # Step 1: Eigendecompose K
        ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)

        # Step 2: Rotate data
        Y_rot = rotate(Y, ed.eigenvectors)  # (n, E)
        X0_rot = rotate(X0, ed.eigenvectors)  # (n, c)

        # Step 3: Initialize from sample covariance
        R = Y_rot - X0_rot @ torch.linalg.lstsq(X0_rot, Y_rot).solution
        S_cov = (R.T @ R) / max(n - c, 1)
        S_cov = S_cov + torch.eye(E, dtype=STAT_DTYPE, device=Y.device) * 1e-6
        Vg_init = S_cov * 0.5
        Ve_init = S_cov * 0.5

        # Step 4: FA(k) REML
        Vg, Ve, ll, trace, Lambda, psi = fa_lbfgs_reml(
            Y_rot, X0_rot, ed.eigenvalues,
            n_envs=E,
            fa_rank=fa_rank,
            max_iter=self.config.reml_max_iter,
            tol=self.config.reml_convergence_tol,
            Vg_init=Vg_init,
            Ve_init=Ve_init,
        )

        # Step 5: Precompute null quantities for scan
        null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, ed.eigenvalues)

        converged = len(trace) > 0 and (
            len(trace) < self.config.reml_max_iter
            or (
                len(trace) >= 2
                and abs(trace[-1]["ll"] - trace[-2]["ll"])
                < self.config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0)
            )
        )

        logger.info(
            "FA(%d) null fit: E=%d, ll=%.4f, converged=%s", fa_rank, E, ll, converged,
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
            optimizer_trace=trace,
            converged=converged,
            device=ed.eigenvalues.device,
        )
        # Store FA-specific quantities
        nf.fa_Lambda = Lambda
        nf.fa_psi = psi
        nf.fa_rank = fa_rank
        return nf

    def _fit_null_multi_kernel(
        self,
        Y: Tensor,
        X0: Tensor,
        K_dict: dict[str, Tensor],
        E: int,
        vg_kind: str,
        fa_rank: int,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model with multiple kernels using direct V^{-1}."""
        from ..optim.multi_kernel_met_reml import multi_kernel_met_reml

        result = multi_kernel_met_reml(
            Y, X0, K_dict,
            n_envs=E,
            vg_structure=vg_kind,
            fa_rank=fa_rank,
            max_iter=self.config.reml_max_iter,
            tol=self.config.reml_convergence_tol,
        )

        # Sum Vg across kernels for summary / interpretive use
        Vg_sum = sum(result["Vg_dict"].values())

        nf = NullFit(
            Vg=Vg_sum,
            Ve=result["Ve"],
            log_likelihood=result["loglik"],
            optimizer_trace=result["trace"],
            converged=result["converged"],
            device=Y.device,
        )
        nf.V_inv = result["V_inv"]
        nf.Vg_dict = result["Vg_dict"]
        nf.is_multi_kernel = True
        nf.Y_wide = Y  # needed for multi-kernel scan
        nf.X0 = X0
        return nf

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> EnvScanResult:
        """Score a genotype chunk with joint, homogeneity, and marginal tests.

        Parameters
        ----------
        G_chunk : (n_original, m) or (n_filtered, m) genotype dosages.
            If n_original, the keep_mask from fit_null is applied.
        null_fit : output of fit_null.
        variant_meta : per-variant annotation.
        test : "wald" only.

        Returns
        -------
        EnvScanResult with joint, homogeneity, and per-environment tests.
        """
        if test not in ("wald",):
            raise ValueError(
                f"MultiEnvLMM supports 'wald' test, got '{test}'."
            )

        # Apply complete-case mask if needed
        keep_mask = getattr(null_fit, "keep_mask", None)
        if keep_mask is not None and G_chunk.shape[0] == len(keep_mask):
            G_chunk = G_chunk[keep_mask]

        G_chunk = G_chunk.to(STAT_DTYPE)
        af = G_chunk.mean(dim=0) / 2.0

        # Dispatch: multi-kernel uses precomputed V_inv
        if getattr(null_fit, "is_multi_kernel", False):
            return self._score_chunk_multi_kernel(G_chunk, null_fit, variant_meta, af)

        n, m = G_chunk.shape

        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)

        W = null_fit.weights  # (n, E, E)
        Y_rot = null_fit.Y_rot  # (n, E)
        X0_rot = null_fit.X0_rot  # (n, c)
        M00 = null_fit.M00  # (cE, cE)
        b0 = null_fit.b0  # (cE,)

        E = Y_rot.shape[1]
        c = X0_rot.shape[1]
        device = G_chunk.device

        # --- Full-system GLS (same as MultiTraitLMM.score_chunk) ---
        WY = torch.einsum('ist,is->it', W, Y_rot)  # (n, E)

        g2 = G_rot ** 2  # (n, m)
        M11 = torch.einsum('ni,nst->ist', g2, W)  # (m, E, E)

        M01_raw = torch.einsum('ia,its,ij->jats', X0_rot, W, G_rot)  # (m, c, E, E)
        M01 = M01_raw.reshape(m, c * E, E)  # (m, cE, E)

        b1 = torch.einsum('ij,is->js', G_rot, WY)  # (m, E)

        dim_full = c * E + E
        M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE, device=device)
        M_full[:, :c*E, :c*E] = M00.unsqueeze(0)
        M_full[:, :c*E, c*E:] = M01
        M_full[:, c*E:, :c*E] = M01.transpose(1, 2)
        M_full[:, c*E:, c*E:] = M11

        rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE, device=device)
        rhs_full[:, :c*E] = b0.unsqueeze(0)
        rhs_full[:, c*E:] = b1

        # Guard monomorphic SNPs
        diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
        singular_mask = diag_min < 1e-10
        if singular_mask.any():
            jitter = torch.eye(dim_full, dtype=STAT_DTYPE, device=device) * 1e-10
            M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

        # Solve for beta per SNP
        coef = torch.linalg.solve(M_full, rhs_full)  # (m, dim_full)
        beta = coef[:, c*E:]  # (m, E)

        # Var(beta) via M_full^{-1}[cE:, cE:]
        rhs_var = torch.zeros(m, dim_full, E, dtype=STAT_DTYPE, device=device)
        rhs_var[:, c*E:, :] = torch.eye(E, dtype=STAT_DTYPE, device=device).unsqueeze(0)
        Minv_cols = torch.linalg.solve(M_full, rhs_var)  # (m, dim_full, E)
        Var_beta = Minv_cols[:, c*E:, :]  # (m, E, E)

        se = torch.sqrt(
            torch.clamp(torch.diagonal(Var_beta, dim1=-2, dim2=-1), min=1e-30)
        )  # (m, E)

        # --- TEST 1: Joint Wald test χ²(E) ---
        Var_inv_beta = torch.linalg.solve(Var_beta, beta.unsqueeze(-1)).squeeze(-1)
        stat_joint = torch.clamp((beta * Var_inv_beta).sum(dim=-1), min=0.0)
        p_joint = _chi2_sf(stat_joint, df=E)

        # --- TEST 2: Homogeneity test χ²(E-1) ---
        C = _build_contrast_matrix(E, device)  # (E-1, E)
        Cb = torch.einsum('de,je->jd', C, beta)  # (m, E-1)
        CVCt = torch.einsum('de,jef,gf->jdg', C, Var_beta, C)  # (m, E-1, E-1)

        # Wald: (Cβ)^T (C Var C^T)^{-1} (Cβ)
        CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
        stat_hom = torch.clamp((Cb * CVCt_inv_Cb).sum(dim=-1), min=0.0)
        p_hom = _chi2_sf(stat_hom, df=E - 1)

        # --- TEST 3: Per-environment marginal χ²(1) ---
        stat_marginal = torch.clamp(beta ** 2 / torch.clamp(se ** 2, min=1e-30), min=0.0)
        p_marginal = _chi2_sf_2d(stat_marginal, df=1)

        # --- Reaction-norm tests (if requested) ---
        rn_fields: dict[str, Any] = {}
        parameterization = getattr(null_fit, "parameterization", self.parameterization)
        if parameterization == "reaction_norm":
            (
                alpha, se_alpha, stat_stable, p_stable,
                delta, stat_gxe, p_gxe,
            ) = _compute_reaction_norm_tests(beta, Var_beta, E, device)
            rn_fields = dict(
                alpha=alpha,
                se_alpha=se_alpha,
                stat_stable=stat_stable,
                p_stable=p_stable,
                delta=delta,
                stat_gxe=stat_gxe,
                p_gxe=p_gxe,
            )

        return EnvScanResult(
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
            stat_homogeneity=stat_hom,
            p_homogeneity=p_hom,
            stat_marginal=stat_marginal,
            p_marginal=p_marginal,
            test="wald",
            env_names=getattr(null_fit, "env_names", None),
            **rn_fields,
        )

    def _score_chunk_multi_kernel(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        af: Tensor,
    ) -> EnvScanResult:
        """Score chunk using precomputed V_inv from multi-kernel fit."""
        n, m = G_chunk.shape
        E = null_fit.n_env
        device = G_chunk.device
        V_inv = null_fit.V_inv  # (nE, nE)
        Y_wide = null_fit.Y_wide  # (n, E)
        X0 = null_fit.X0  # (n, c)
        c = X0.shape[1]

        Y_vec = Y_wide.T.reshape(-1)  # (nE,)

        # Build X_kron = I_E ⊗ X0, shape (nE, cE)
        X_kron = torch.zeros(n * E, c * E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            X_kron[e * n:(e + 1) * n, e * c:(e + 1) * c] = X0

        # Per-SNP GLS via V_inv
        # For each SNP j, expand genotype into the Kronecker system:
        # G_kron_j = [g_j, 0, ..., 0; 0, g_j, 0, ...; ...; 0, ..., g_j] (nE, E)
        # Then M11 = G_kron_j^T V_inv G_kron_j, etc.
        # Efficient batch computation:

        # V_inv as (E, E, n, n) block matrix
        V_inv_blocks = V_inv.reshape(E, n, E, n).permute(0, 2, 1, 3)  # (E, E, n, n)

        # M11[j] = Σ_{e,f} g_j^T V_inv_{e,f} g_j → (m, E, E)
        M11 = torch.zeros(m, E, E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            for f in range(E):
                Vinv_ef = V_inv_blocks[e, f]  # (n, n)
                Vinv_g = Vinv_ef @ G_chunk  # (n, m)
                M11[:, e, f] = (G_chunk * Vinv_g).sum(dim=0)  # (m,)

        # b1[j] = Σ_f g_j^T V_inv_{e,f} y_f → (m, E)
        b1 = torch.zeros(m, E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            for f in range(E):
                Vinv_ef = V_inv_blocks[e, f]  # (n, n)
                y_f = Y_wide[:, f]  # (n,)
                Vinv_y = Vinv_ef @ y_f  # (n,)
                b1[:, e] += G_chunk.T @ Vinv_y  # (m,)

        # M01: X0^T V_inv_{e,f} G → (m, cE, E)
        M01 = torch.zeros(m, c * E, E, dtype=STAT_DTYPE, device=device)
        for e in range(E):
            for f in range(E):
                Vinv_ef = V_inv_blocks[e, f]  # (n, n)
                Vinv_g = Vinv_ef @ G_chunk  # (n, m)
                contrib = X0.T @ Vinv_g  # (c, m)
                M01[:, e * c:(e + 1) * c, f] = contrib.T  # (m, c)

        # M00: X_kron^T V_inv X_kron → (cE, cE), constant across SNPs
        M00 = X_kron.T @ (V_inv @ X_kron)  # (cE, cE)

        # b0: X_kron^T V_inv y → (cE,)
        b0 = X_kron.T @ (V_inv @ Y_vec)  # (cE,)

        # Assemble full system and solve (same as EED path)
        dim_full = c * E + E
        M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE, device=device)
        M_full[:, :c * E, :c * E] = M00.unsqueeze(0)
        M_full[:, :c * E, c * E:] = M01
        M_full[:, c * E:, :c * E] = M01.transpose(1, 2)
        M_full[:, c * E:, c * E:] = M11

        rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE, device=device)
        rhs_full[:, :c * E] = b0.unsqueeze(0)
        rhs_full[:, c * E:] = b1

        # Guard monomorphic SNPs
        diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
        singular_mask = diag_min < 1e-10
        if singular_mask.any():
            jitter = torch.eye(dim_full, dtype=STAT_DTYPE, device=device) * 1e-10
            M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

        coef = torch.linalg.solve(M_full, rhs_full)
        beta = coef[:, c * E:]  # (m, E)

        rhs_var = torch.zeros(m, dim_full, E, dtype=STAT_DTYPE, device=device)
        rhs_var[:, c * E:, :] = torch.eye(E, dtype=STAT_DTYPE, device=device).unsqueeze(0)
        Minv_cols = torch.linalg.solve(M_full, rhs_var)
        Var_beta = Minv_cols[:, c * E:, :]  # (m, E, E)

        se = torch.sqrt(
            torch.clamp(torch.diagonal(Var_beta, dim1=-2, dim2=-1), min=1e-30)
        )

        # Tests (same as EED path)
        Var_inv_beta = torch.linalg.solve(Var_beta, beta.unsqueeze(-1)).squeeze(-1)
        stat_joint = torch.clamp((beta * Var_inv_beta).sum(dim=-1), min=0.0)
        p_joint = _chi2_sf(stat_joint, df=E)

        C = _build_contrast_matrix(E, device)
        Cb = torch.einsum("de,je->jd", C, beta)
        CVCt = torch.einsum("de,jef,gf->jdg", C, Var_beta, C)
        CVCt_inv_Cb = torch.linalg.solve(CVCt, Cb.unsqueeze(-1)).squeeze(-1)
        stat_hom = torch.clamp((Cb * CVCt_inv_Cb).sum(dim=-1), min=0.0)
        p_hom = _chi2_sf(stat_hom, df=E - 1)

        stat_marginal = torch.clamp(beta ** 2 / torch.clamp(se ** 2, min=1e-30), min=0.0)
        p_marginal = _chi2_sf_2d(stat_marginal, df=1)

        # Reaction-norm tests
        rn_fields: dict[str, Any] = {}
        parameterization = getattr(null_fit, "parameterization", self.parameterization)
        if parameterization == "reaction_norm":
            (
                alpha, se_alpha, stat_stable, p_stable,
                delta, stat_gxe, p_gxe,
            ) = _compute_reaction_norm_tests(beta, Var_beta, E, device)
            rn_fields = dict(
                alpha=alpha, se_alpha=se_alpha,
                stat_stable=stat_stable, p_stable=p_stable,
                delta=delta, stat_gxe=stat_gxe, p_gxe=p_gxe,
            )

        return EnvScanResult(
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
            stat_homogeneity=stat_hom,
            p_homogeneity=p_hom,
            stat_marginal=stat_marginal,
            p_marginal=p_marginal,
            test="wald",
            env_names=getattr(null_fit, "env_names", None),
            **rn_fields,
        )

    def update_null(self, null_fit: NullFit, *, max_iter: int = 100) -> NullFit:
        """Resume optimization from a previous NullFit."""
        from .base import update_null
        return update_null(null_fit, max_iter=max_iter, config=self.config)

    # --- Interpretive methods ---

    @staticmethod
    def genetic_correlation(null_fit: NullFit) -> Tensor:
        """Genetic correlation matrix across environments from Vg.

        rg[i,j] = Vg[i,j] / sqrt(Vg[i,i] * Vg[j,j])
        """
        Vg = null_fit.Vg  # (E, E)
        d = torch.sqrt(torch.clamp(torch.diag(Vg), min=1e-30))
        return Vg / (d.unsqueeze(0) * d.unsqueeze(1))

    @staticmethod
    def per_env_heritability(null_fit: NullFit) -> Tensor:
        """Per-environment heritability: h²_e = Vg[e,e] / (Vg[e,e] + Ve[e,e])."""
        vg_diag = torch.diag(null_fit.Vg)
        ve_diag = torch.diag(null_fit.Ve)
        return vg_diag / torch.clamp(vg_diag + ve_diag, min=1e-30)

    @staticmethod
    def fa_loadings(null_fit: NullFit) -> Tensor:
        """Return the FA loading matrix Lambda (E, k).

        Only available when ``vg_structure="fa(k)"`` was used.
        """
        Lambda = getattr(null_fit, "fa_Lambda", None)
        if Lambda is None:
            raise AttributeError(
                "FA loadings not available — was fit_null called with vg_structure='fa(k)'?"
            )
        return Lambda

    @staticmethod
    def proportion_gxe(null_fit: NullFit) -> float:
        """Proportion of trace(Vg) NOT explained by the rank-1 stable component.

        Interpretation: 0 = perfectly stable genetics, 1 = fully environment-specific.
        Uses the eigendecomposition of Vg regardless of vg_structure.
        """
        eigvals = torch.linalg.eigvalsh(null_fit.Vg)
        total = eigvals.sum().item()
        if total < 1e-30:
            return 0.0
        top1 = eigvals[-1].item()
        return 1.0 - top1 / total

    @staticmethod
    def stability_index(result: EnvScanResult) -> Tensor:
        """Per-SNP stability index: stat_stable / (stat_stable + stat_gxe).

        Returns (m,) tensor in [0, 1]. Near 1 = stable effect, near 0 = GxE.
        Requires ``parameterization="reaction_norm"`` in the scan.
        """
        if result.stat_stable is None or result.stat_gxe is None:
            raise AttributeError(
                "stability_index requires reaction_norm parameterization."
            )
        denom = result.stat_stable + result.stat_gxe
        return result.stat_stable / denom.clamp(min=1e-30)

    @staticmethod
    def env_specific_variance(null_fit: NullFit) -> dict[str, dict[str, float]]:
        """Return per-environment genetic and residual variance components."""
        env_names = getattr(null_fit, "env_names", None) or [
            f"env_{i}" for i in range(null_fit.Vg.shape[0])
        ]
        result = {}
        for e, name in enumerate(env_names):
            result[name] = {
                "Vg": null_fit.Vg[e, e].item(),
                "Ve": null_fit.Ve[e, e].item(),
                "h2": (null_fit.Vg[e, e] / (null_fit.Vg[e, e] + null_fit.Ve[e, e])).item(),
            }
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chi2_sf(stat: Tensor, df: int = 1) -> Tensor:
    """P-values from chi² statistics (1-d)."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)


def _chi2_sf_2d(stat: Tensor, df: int = 1) -> Tensor:
    """P-values from chi² statistics (2-d: m × E)."""
    import numpy as np
    import scipy.stats as sp_stats

    shape = stat.shape
    stat_np = stat.detach().cpu().numpy().astype(np.float64).ravel()
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0).reshape(shape)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
