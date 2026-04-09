"""Multi-Kernel Linear Mixed Model (Phase 15).

Partitions genetic variance into additive, dominance, and epistatic components:
    V = K_a σ²_a + K_d σ²_d + K_aa σ²_aa + I σ²_e

Cannot use the eigendecomposition trick (multiple kernels are not simultaneously
diagonalizable), so works directly with the full covariance matrix V.

References:
    - Muñoz et al. (2014): Multi-kernel models for polyploid species
    - Su et al. (2012): Epistatic GRM via Hadamard products
    - Vitezica et al. (2013): Orthogonal dominance relationship matrix
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from .base import BaseModel, NullFit, ScanResult, VariantMeta

logger = logging.getLogger(__name__)


def build_multi_kernels(
    G: Tensor,
    ploidy: int = 2,
    include_dominance: bool = True,
    include_epistatic: bool = True,
) -> tuple[list[Tensor], list[str]]:
    """Build additive + dominance + epistatic kernels from genotype matrix.

    Parameters
    ----------
    G : (n, m) genotype dosage matrix.
    ploidy : ploidy level.
    include_dominance : include Vitezica dominance GRM.
    include_epistatic : include additive×additive epistatic GRM.

    Returns
    -------
    kernels : list of (n, n) kernel matrices.
    names : list of kernel names.
    """
    from ..linalg.kinship import grm_vanraden
    from ..linalg.kinship_advanced import grm_vitezica_dominance
    from ..linalg.kinship_polyploid import grm_epistatic_hadamard

    K_add, _ = grm_vanraden(G, ploidy=ploidy)
    kernels = [K_add]
    names = ["additive"]

    if include_dominance:
        K_dom, _ = grm_vitezica_dominance(G, ploidy=ploidy)
        kernels.append(K_dom)
        names.append("dominance")

        if include_epistatic:
            epi = grm_epistatic_hadamard(K_add, K_dom)
            kernels.append(epi["K_aa"])
            names.append("epistatic_aa")
    elif include_epistatic:
        epi = grm_epistatic_hadamard(K_add)
        kernels.append(epi["K_aa"])
        names.append("epistatic_aa")

    return kernels, names


class MultiKernelLMM:
    """Multi-kernel LMM: y = X₀β + Σ_k g_k + e, g_k ~ N(0, K_k σ²_k).

    Conforms to the BaseModel protocol: ``fit_null`` + ``score_chunk``.
    """

    def __init__(self, config: Optional[NumericalConfig] = None) -> None:
        self.config = config or NumericalConfig()

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        *,
        kernels: Optional[list[Tensor]] = None,
        kernel_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit null model with multiple kernels.

        Parameters
        ----------
        Y : (n,) phenotype vector.
        X0 : (n, c) covariate matrix (intercept as first column).
        K : (n, n) single kinship matrix (used if kernels is None).
        kernels : list of (n, n) kernel matrices.
        kernel_names : labels for each kernel (e.g. ["additive", "dominance"]).

        Returns
        -------
        NullFit with cached V⁻¹ and variance components.
        """
        from ..linalg.safe import safe_cholesky
        from ..optim.multikernel_reml import multikernel_reml

        if kernels is None:
            if K is None:
                raise ValueError(
                    "MultiKernelLMM requires either 'kernels' or 'K'."
                )
            kernels = [K]
            kernel_names = kernel_names or ["genetic"]

        if kernel_names is None:
            kernel_names = [f"kernel_{i}" for i in range(len(kernels))]

        Y = Y.to(STAT_DTYPE).squeeze()
        X0 = X0.to(STAT_DTYPE)
        kernels = [K_k.to(STAT_DTYPE) for K_k in kernels]
        n = Y.shape[0]
        device = Y.device

        # --- Estimate variance components via LBFGS ---
        variances, ll, trace = multikernel_reml(
            Y, X0, kernels,
            max_iter=self.config.reml_max_iter,
            tol=self.config.reml_convergence_tol,
        )

        # Unpack: variances = [σ²_1, ..., σ²_K, σ²_e]
        sig2_components = variances[:-1]
        sig2_e = variances[-1]

        # --- Assemble V and compute V⁻¹ ---
        V = sig2_e * torch.eye(n, dtype=STAT_DTYPE, device=device)
        for k, K_k in enumerate(kernels):
            V = V + sig2_components[k] * K_k

        L_V = safe_cholesky(V, jitter_factor=self.config.cholesky_jitter_factor)
        V_inv = torch.cholesky_inverse(L_V)

        # --- GLS null fixed effects ---
        V_inv_X0 = V_inv @ X0
        XtVinvX = X0.T @ V_inv_X0
        XtVinvY = V_inv_X0.T @ Y
        beta0 = torch.linalg.solve(XtVinvX, XtVinvY)

        # Null residuals
        r0 = Y - X0 @ beta0

        # Partitioned heritability
        total_var = sum(variances)
        partitioned_h2 = {}
        for k, name in enumerate(kernel_names):
            partitioned_h2[name] = sig2_components[k] / total_var
        partitioned_h2["residual"] = sig2_e / total_var

        # Variance component dict for storage
        variance_dict = {}
        for k, name in enumerate(kernel_names):
            variance_dict[name] = sig2_components[k]
        variance_dict["residual"] = sig2_e

        converged = len(trace) > 0 and (
            len(trace) < self.config.reml_max_iter or
            (len(trace) >= 2 and
             abs(trace[-1]["ll"] - trace[-2]["ll"]) <
             self.config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0))
        )

        logger.info(
            "Multi-kernel null fit: %d kernels, ll=%.4f, converged=%s, h²=%s",
            len(kernels), ll, converged,
            {k: f"{v:.3f}" for k, v in partitioned_h2.items()},
        )

        nf = NullFit(
            sig2_g=sum(sig2_components),
            sig2_e=sig2_e,
            log_likelihood=ll,
            optimizer_trace=trace,
            converged=converged,
            device=device,
        )
        # Store multi-kernel-specific quantities as attributes
        nf.V_inv = V_inv
        nf.X0 = X0
        nf.beta0 = beta0
        nf.r0 = r0
        nf.kernels = kernels
        nf.kernel_names = kernel_names
        nf.variances = variances
        nf.variance_dict = variance_dict
        nf.partitioned_h2 = partitioned_h2

        return nf

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
        """Score a genotype chunk using precomputed V⁻¹.

        Per-SNP Wald test:
            β̂_j = (g_j^T V⁻¹ g_j)⁻¹ · g_j^T V⁻¹ r₀
            SE²_j = (g_j^T V⁻¹ g_j)⁻¹
            T_j = (β̂_j / SE_j)²  ~ chi²(1)

        Parameters
        ----------
        G_chunk : (n, m) genotype dosage matrix.
        null_fit : output of fit_null.
        variant_meta : marker metadata.
        test : "wald" only.

        Returns
        -------
        ScanResult with per-SNP association statistics.
        """
        if test not in ("wald",):
            raise ValueError(
                f"MultiKernelLMM currently supports 'wald' test, got '{test}'."
            )

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        V_inv = null_fit.V_inv
        r0 = null_fit.r0
        X0 = null_fit.X0  # stored in fit_null

        af = G_chunk.mean(dim=0) / 2.0  # diploid convention

        # Vectorised: V_inv @ G_chunk = (n, m)
        V_inv_G = V_inv @ G_chunk

        # P-matrix projection: P = V⁻¹ - V⁻¹X(X^TV⁻¹X)⁻¹X^TV⁻¹
        # g^T P g = g^T V⁻¹ g - (g^T V⁻¹ X)(X^T V⁻¹ X)⁻¹(X^T V⁻¹ g)
        V_inv_X0 = V_inv @ X0  # (n, c)
        XtVinvX = X0.T @ V_inv_X0  # (c, c)
        XtVinvX_inv = torch.linalg.inv(XtVinvX)  # (c, c)

        # g^T V⁻¹ g for each SNP: (m,)
        GtVinvG = (G_chunk * V_inv_G).sum(dim=0)

        # g^T V⁻¹ X for each SNP: (m, c)
        GtVinvX = G_chunk.T @ V_inv_X0

        # Correction: (g^T V⁻¹ X)(X^T V⁻¹ X)⁻¹(X^T V⁻¹ g) per SNP: (m,)
        correction = (GtVinvX @ XtVinvX_inv * GtVinvX).sum(dim=1)

        # g^T P g (proper Wald denominator accounting for fixed effects)
        GtPG = GtVinvG - correction
        GtPG_safe = torch.clamp(GtPG, min=1e-20)

        # g^T P Y = g^T V⁻¹ r₀ (since r₀ = Y - X₀β̂₀ and P projects out X₀)
        V_inv_r0 = V_inv @ r0
        GtPY = G_chunk.T @ V_inv_r0  # (m,)

        # Effect estimates using P-matrix
        beta = GtPY / GtPG_safe  # (m,)
        se2 = 1.0 / GtPG_safe  # (m,)
        se = torch.sqrt(se2)  # (m,)

        # Wald statistic: chi²(1) = (β/SE)²
        stat = (beta / se) ** 2  # (m,)
        stat = torch.clamp(stat, min=0.0)

        # P-values from chi²(1)
        p = _chi2_sf(stat, df=1)

        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,
            se=se,
            stat=stat,
            p=p,
            test="wald",
        )

    def per_kernel_wald_tests(
        self, null_fit: NullFit,
    ) -> dict[str, tuple[float, float]]:
        """Wald Z-test for each variance component: H₀: σ²_k = 0.

        Uses the observed Fisher information (Hessian of neg-REML at optimum)
        to compute standard errors.

        Parameters
        ----------
        null_fit : output of fit_null (must have multi-kernel attributes).

        Returns
        -------
        dict mapping kernel_name → (z_statistic, p_value).
        """
        kernels = null_fit.kernels
        kernel_names = null_fit.kernel_names
        variances = null_fit.variances
        n_kernels = len(kernels)
        device = null_fit.device

        n = kernels[0].shape[0]
        V_inv = null_fit.V_inv
        r0 = null_fit.r0

        # Fisher information from variance of the score:
        # I_kl = ½ tr(V⁻¹ K_k V⁻¹ K_l) for variance components
        # For log-parameterisation: I_kl^log = σ²_k σ²_l · I_kl
        kernel_mats = list(kernels) + [torch.eye(n, dtype=STAT_DTYPE, device=device)]
        n_params = len(kernel_mats)

        # Precompute V⁻¹ K_k
        VinvK = [V_inv @ K_k for K_k in kernel_mats]

        # Fisher information matrix
        fisher = torch.zeros(n_params, n_params, dtype=STAT_DTYPE, device=device)
        for k in range(n_params):
            for l in range(k, n_params):
                # tr(V⁻¹ K_k V⁻¹ K_l) = tr(VinvK[k] @ VinvK[l])
                # = sum(VinvK[k] * VinvK[l]^T)
                val = 0.5 * (VinvK[k] * VinvK[l].T).sum()
                fisher[k, l] = val
                fisher[l, k] = val

        # Standard errors: SE(σ̂²_k) = sqrt(diag(Fisher⁻¹))
        try:
            fisher_inv = torch.linalg.inv(fisher)
            se_vars = torch.sqrt(torch.clamp(torch.diagonal(fisher_inv), min=1e-20))
        except Exception:
            # Fisher matrix singular — return NaN
            result = {}
            for k, name in enumerate(kernel_names):
                result[name] = (float("nan"), float("nan"))
            result["residual"] = (float("nan"), float("nan"))
            return result

        result = {}
        all_names = list(kernel_names) + ["residual"]
        for k, name in enumerate(all_names):
            z = variances[k] / se_vars[k].item()
            # One-sided test (variance ≥ 0): p = 1 - Φ(z)
            p = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
            result[name] = (z, p)

        return result


def _chi2_sf(stat: Tensor, df: int = 1) -> Tensor:
    """Compute p-values from chi² statistics via scipy."""
    import numpy as np
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
