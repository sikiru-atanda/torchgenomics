"""BaseModel protocol and core dataclasses: NullFit, ScanResult, VariantMeta.

Also provides ``update_null()`` for ASReml-R-style warm-start resumption of
a previous NullFit without redoing eigendecomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant metadata (carried through from reader to output)
# ---------------------------------------------------------------------------

@dataclass
class VariantMeta:
    """Per-variant annotation attached to genotype chunks."""

    snp: list[str]  # variant IDs
    chr: list[str]  # chromosome labels
    pos: list[int]  # base-pair positions
    a1: list[str]  # effect allele
    a2: list[str]  # other allele

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# NullFit — output of fit_null()
# ---------------------------------------------------------------------------

@dataclass
class NullFit:
    """Cached null-model quantities reused across all SNP chunks.

    Concrete models populate the fields relevant to their method. Fields
    that do not apply to a particular model are left as ``None``.
    """

    # Variance components
    sig2_g: Optional[float] = None  # genetic variance (single-trait)
    sig2_e: Optional[float] = None  # residual variance (single-trait)
    Vg: Optional[Tensor] = None  # (d, d) genetic covariance (multi-trait)
    Ve: Optional[Tensor] = None  # (d, d) residual covariance (multi-trait)

    # Eigendecomposition artefacts
    eigenvalues: Optional[Tensor] = None  # (n,)
    eigenvectors: Optional[Tensor] = None  # (n, n) or (n, k) truncated

    # Rotated quantities (U^T Y, U^T X)
    Y_rot: Optional[Tensor] = None
    X0_rot: Optional[Tensor] = None

    # Null-model normal equations (for Schur-complement scan)
    M00: Optional[Tensor] = None
    b0: Optional[Tensor] = None

    # Per-individual weights (inverse covariance diagonal in rotated space)
    weights: Optional[Tensor] = None  # (n,) single-trait or (n, d, d) multi-trait

    # Log-likelihood at null
    log_likelihood: Optional[float] = None

    # Optimizer trace for diagnostics
    optimizer_trace: list[dict[str, Any]] = field(default_factory=list)

    # Convergence flag
    converged: bool = False

    # Device the tensors live on
    device: Optional[torch.device] = None

    # Approximate backend metadata (Phase 13)
    approximate: bool = False
    approx_method: Optional[str] = None  # "randomized_svd", "nystrom", "sparse", "lobpcg"
    approx_config: Optional[dict] = None  # Parameters used (n_components, threshold, etc.)


# ---------------------------------------------------------------------------
# ScanResult — output of score_chunk()
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Per-variant association statistics from a single genotype chunk.

    All tensor fields have first dimension m (number of variants in chunk).
    The schema is consistent across all models (GLM, LMM, mvLMM, etc.).
    """

    # Variant identifiers
    chr: list[str]  # (m,) chromosome
    pos: list[int]  # (m,) base-pair position
    snp: list[str]  # (m,) variant ID
    a1: list[str]  # (m,) effect allele
    a2: list[str]  # (m,) other allele

    # Allele frequency
    af: Tensor  # (m,) allele frequency of a1

    # Effect estimates
    beta: Tensor  # (m,) single-trait or (m, d) multi-trait; NaN for score test
    se: Tensor  # (m,) or (m, d); NaN for score test

    # Test statistics and p-values
    stat: Tensor  # (m,) chi-square test statistic
    p: Tensor  # (m,) p-value

    # Which test was used
    test: str  # "score", "wald", or "lrt"

    # Number of samples used (may vary if per-SNP missingness filtering)
    n_obs: Optional[Tensor] = None  # (m,)

    # Inference type: "marginal" (GLM/LMM — calibrated), or
    # "post_selection" (FarmCPU/BLINK — conditional on selected pseudo-QTNs).
    # Charter Section 5: FarmCPU/BLINK p-values are NOT marginal genome-wide
    # p-values; they are conditional on the iteratively selected covariate set.
    inference_type: str = "marginal"

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# BaseModel protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseModel(Protocol):
    """Contract that every GWAS model must satisfy.

    Implementing classes provide ``fit_null`` (estimate variance components and
    cache rotated quantities under the null) and ``score_chunk`` (test a batch
    of m SNPs against the cached null).

    The UnifiedScanner calls these two methods — nothing else.
    """

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the null model (no SNP effect).

        Parameters
        ----------
        Y : Tensor, shape (n,) or (n, d)
            Phenotype matrix.  Single column for GLM/LMM, d columns for mvLMM.
        X0 : Tensor, shape (n, c)
            Covariate matrix (intercept must be the first column).
        K : Tensor, shape (n, n), optional
            Kinship / GRM.  Required for LMM-family, ignored by GLM.

        Returns
        -------
        NullFit
            Cached null-model quantities for use in ``score_chunk``.
        """
        ...

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Score a genotype chunk against the cached null.

        Parameters
        ----------
        G_chunk : Tensor, shape (n, m)
            Genotype dosage matrix for m SNPs.
        null_fit : NullFit
            Output of ``fit_null``.
        variant_meta : VariantMeta
            Marker IDs, chromosomes, positions, alleles for the m SNPs.
        test : str
            One of ``"score"``, ``"wald"``, ``"lrt"``.

        Returns
        -------
        ScanResult
            Per-variant association statistics.
        """
        ...


# ---------------------------------------------------------------------------
# update_null — ASReml-R-style warm-start resumption
# ---------------------------------------------------------------------------

def update_null(
    null_fit: NullFit,
    *,
    max_iter: int = 100,
    config: Optional[Any] = None,
) -> NullFit:
    """Resume optimization from a previous NullFit (ASReml-R update style).

    Reuses cached eigendecomposition and rotated data from the previous fit,
    warm-starts the optimizer from previous variance component estimates, and
    runs additional iterations.  The optimizer trace is accumulated (old +
    new entries).

    This is the recommended way to handle non-convergence: instead of
    increasing ``max_iter`` and starting over, call ``update_null()`` to
    continue from where the previous fit stopped.

    Parameters
    ----------
    null_fit : NullFit
        Previous (possibly non-converged) null model fit.
    max_iter : int
        Maximum iterations for the resumed optimizer run.
    config : NumericalConfig, optional
        Override numerical configuration.  If None, uses default config.

    Returns
    -------
    NullFit
        Updated null model with refined variance components, recomputed
        downstream quantities (weights, M00, b0), and accumulated trace.

    Examples
    --------
    >>> model = SingleTraitLMM()
    >>> nf = model.fit_null(Y, X0, K)  # may not converge
    >>> if not nf.converged:
    ...     nf = model.update_null(nf, max_iter=200)
    """
    from ..config import NumericalConfig

    cfg = config or NumericalConfig()
    old_trace = list(null_fit.optimizer_trace)

    # Detect model type from NullFit fields
    is_multi_kernel = (
        (hasattr(null_fit, "kernels") and null_fit.kernels is not None)
        or getattr(null_fit, "is_multi_kernel", False)
    )
    is_fa = hasattr(null_fit, "fa_rank") and null_fit.fa_rank is not None
    is_rr_met = getattr(null_fit, "is_rr_met", False)
    is_random_regression = (
        not is_rr_met
        and hasattr(null_fit, "K_coef") and null_fit.K_coef is not None
        and hasattr(null_fit, "basis_kind") and null_fit.basis_kind is not None
    )
    is_multi_trait = null_fit.Vg is not None
    is_single_trait = null_fit.sig2_g is not None and not is_multi_trait

    if is_fa and not is_multi_kernel and not is_rr_met:
        updated = _update_fa(null_fit, max_iter=max_iter, config=cfg)
    elif is_single_trait and not is_multi_kernel:
        updated = _update_single_trait(null_fit, max_iter=max_iter, config=cfg)
    elif is_rr_met:
        updated = _update_random_regression_met(
            null_fit, max_iter=max_iter, config=cfg
        )
    elif is_random_regression:
        updated = _update_random_regression(null_fit, max_iter=max_iter, config=cfg)
    elif is_multi_trait:
        updated = _update_multi_trait(null_fit, max_iter=max_iter, config=cfg)
    elif is_multi_kernel:
        updated = _update_multi_kernel(null_fit, max_iter=max_iter, config=cfg)
    else:
        raise ValueError(
            "Cannot determine model type from NullFit. "
            "Requires sig2_g (single-trait), Vg (multi-trait), or kernels (multi-kernel)."
        )

    # Tag new trace entries and prepend old trace
    for entry in updated.optimizer_trace:
        entry["update"] = True
    updated.optimizer_trace = old_trace + updated.optimizer_trace

    # Preserve metadata from original fit
    if updated.eigenvectors is None:
        updated.eigenvectors = null_fit.eigenvectors
    updated.approximate = null_fit.approximate
    updated.approx_method = null_fit.approx_method
    updated.approx_config = null_fit.approx_config

    logger.info(
        "update_null: converged=%s, ll=%.4f (was %.4f), total trace=%d entries",
        updated.converged,
        updated.log_likelihood or 0.0,
        null_fit.log_likelihood or 0.0,
        len(updated.optimizer_trace),
    )

    return updated


def _update_single_trait(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """Single-trait warm-start via OptimizerController."""
    from ..optim.controller import OptimizerController

    if null_fit.Y_rot is None or null_fit.X0_rot is None or null_fit.eigenvalues is None:
        raise ValueError("NullFit missing cached rotation data (Y_rot, X0_rot, eigenvalues).")

    controller = OptimizerController(config=config)
    updated = controller.fit(
        null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues,
        n_traits=1,
        max_iter=max_iter,
        sig2_g_init=null_fit.sig2_g,
        sig2_e_init=null_fit.sig2_e,
    )
    return updated


def _update_multi_trait(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """Multi-trait warm-start via configured optimizer."""
    from ..optim.mvlmm_reml import mvlmm_null_quantities

    if null_fit.Y_rot is None or null_fit.X0_rot is None or null_fit.eigenvalues is None:
        raise ValueError("NullFit missing cached rotation data.")
    if null_fit.Vg is None or null_fit.Ve is None:
        raise ValueError("NullFit missing Vg/Ve for multi-trait update.")

    d = null_fit.Vg.shape[0]
    tol = config.reml_convergence_tol

    method = config.multi_trait_reml_method
    Vg_init = null_fit.Vg
    Ve_init = null_fit.Ve

    # Try configured method, fall back to lbfgs
    success = False
    if method == "pxem_nr":
        from ..optim.pxem_nr_mvreml import pxem_nr_mvreml
        try:
            Vg, Ve, ll, trace = pxem_nr_mvreml(
                null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues,
                n_traits=d, max_iter=max_iter, tol=tol,
                em_iters=config.multi_trait_em_iters,
                Vg_init=Vg_init, Ve_init=Ve_init,
            )
            success = True
        except Exception:
            logger.info("pxem_nr failed in update, falling back to lbfgs")

    if not success and method == "triad":
        from ..optim.triad_reml import triad_reml
        try:
            Vg, Ve, ll, trace = triad_reml(
                null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues,
                n_traits=d, max_iter=max_iter, tol=tol,
                Vg_init=Vg_init, Ve_init=Ve_init,
            )
            success = True
        except Exception:
            logger.info("triad failed in update, falling back to lbfgs")

    if not success:
        from ..optim.lbfgs_reml import lbfgs_reml
        Vg, Ve, ll, trace = lbfgs_reml(
            null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues,
            n_traits=d, max_iter=max_iter, tol=tol,
            Vg_init=Vg_init, Ve_init=Ve_init,
        )

    null_q = mvlmm_null_quantities(Vg, Ve, null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues)

    converged = len(trace) > 0 and (
        len(trace) < max_iter or
        (len(trace) >= 2 and
         abs(trace[-1]["ll"] - trace[-2]["ll"]) < tol * max(abs(trace[-1]["ll"]), 1.0))
    )

    updated = NullFit(
        Vg=Vg, Ve=Ve,
        eigenvalues=null_fit.eigenvalues,
        eigenvectors=null_fit.eigenvectors,
        Y_rot=null_fit.Y_rot, X0_rot=null_fit.X0_rot,
        M00=null_q["M00"], b0=null_q["b0"], weights=null_q["W"],
        log_likelihood=ll, optimizer_trace=trace,
        converged=converged, device=null_fit.device,
    )
    # Preserve MET metadata if present
    for attr in ("env_names", "n_env", "keep_mask", "parameterization", "vg_structure"):
        if hasattr(null_fit, attr):
            setattr(updated, attr, getattr(null_fit, attr))
    return updated


def _update_random_regression(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """Random-regression warm-start: delegate to multi-trait update,
    then re-attach the RR-specific metadata (basis, projection caches,
    permanent-environment block, etc.) so the resumed fit remains a
    valid RR NullFit."""
    updated = _update_multi_trait(null_fit, max_iter=max_iter, config=config)
    # Re-attach all RR metadata.
    rr_attrs = (
        "basis_kind", "basis_params", "b",
        "Phi_list", "proj_weights", "n_obs_per_indiv", "sample_index",
        "k_coef_structure", "include_pe", "K_pe", "sigma2_e_residual",
        "spatial_coef", "spatial_lambda_row", "spatial_lambda_col",
        "Y_long_cleaned", "Vg_T_stacked", "Ve_T_stacked",
    )
    for attr in rr_attrs:
        if hasattr(null_fit, attr):
            setattr(updated, attr, getattr(null_fit, attr))
    # K_coef alias must follow the new Vg.
    updated.K_coef = updated.Vg
    return updated


def _update_random_regression_met(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """RR-MET warm-start: refit via MultiTraitMultiEnvLMM with the
    cached structure (separable / unstructured / fa(k)) and re-attach the
    RR-MET metadata so the resumed fit remains a valid RR-MET NullFit."""
    from .multi_trait_multi_env_lmm import MultiTraitMultiEnvLMM
    from ..optim.mvlmm_reml import mvlmm_null_quantities

    if null_fit.Y_rot is None or null_fit.X0_rot is None or null_fit.eigenvalues is None:
        raise ValueError("RR-MET update_null: NullFit missing cached rotation data.")

    b = null_fit.b
    E = null_fit.E_envs
    bE = b * E
    structure = getattr(null_fit, "vg_structure_rr", "separable")

    # Refit the (n, bE) rotated data through MT-MET's REML kernels.
    # We can't easily re-enter MT-MET.fit_null directly because it expects
    # un-rotated Y/X0/K; instead we call the structure-specific REML kernel
    # used inside MT-MET.fit_null with warm-start initialization.
    Y_rot = null_fit.Y_rot
    X0_rot = null_fit.X0_rot
    eigenvalues = null_fit.eigenvalues
    Vg_init = null_fit.Vg
    Ve_init = null_fit.Ve
    tol = config.reml_convergence_tol

    if structure.lower() == "separable":
        from ..optim.separable_kron_reml import separable_kron_reml
        Vg_trait_init = null_fit.K_coef
        Vg_env_init = null_fit.Vg_env
        Vg_trait, Vg_env, Ve, ll, trace = separable_kron_reml(
            Y_rot, X0_rot, eigenvalues,
            n_traits=b, n_envs=E,
            max_iter=max_iter, tol=tol,
            Vg_trait_init=Vg_trait_init,
            Vg_env_init=Vg_env_init,
            Ve_init=Ve_init,
        )
        Vg = torch.kron(Vg_trait, Vg_env)
    elif structure.lower() == "unstructured":
        from ..optim.lbfgs_reml import lbfgs_reml
        Vg, Ve, ll, trace = lbfgs_reml(
            Y_rot, X0_rot, eigenvalues,
            n_traits=bE, max_iter=max_iter, tol=tol,
            Vg_init=Vg_init, Ve_init=Ve_init,
        )
        Vg_trait = None
        Vg_env = None
    elif structure.lower().startswith("fa("):
        import re
        from ..optim.fa_lbfgs_reml import fa_lbfgs_reml
        m = re.match(r"^fa\((\d+)\)$", structure, re.IGNORECASE)
        fa_rank = int(m.group(1))
        Vg, Ve, ll, trace, _Lambda, _psi = fa_lbfgs_reml(
            Y_rot, X0_rot, eigenvalues,
            n_envs=bE, fa_rank=fa_rank,
            max_iter=max_iter, tol=tol,
            Vg_init=Vg_init, Ve_init=Ve_init,
        )
        Vg_trait = None
        Vg_env = None
    else:
        raise ValueError(f"RR-MET update_null: unknown structure '{structure}'.")

    null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, eigenvalues)
    converged = len(trace) > 0 and len(trace) < max_iter

    updated = NullFit(
        Vg=Vg, Ve=Ve,
        eigenvalues=eigenvalues,
        eigenvectors=null_fit.eigenvectors,
        Y_rot=Y_rot, X0_rot=X0_rot,
        M00=null_q["M00"], b0=null_q["b0"], weights=null_q["W"],
        log_likelihood=ll, optimizer_trace=trace,
        converged=converged, device=null_fit.device,
    )
    # Re-attach all RR-MET metadata
    updated.is_rr_met = True
    updated.basis_kind = null_fit.basis_kind
    updated.basis_params = null_fit.basis_params
    updated.b = b
    updated.E_envs = E
    updated.env_index = null_fit.env_index
    updated.Phi_list_per_env = null_fit.Phi_list_per_env
    updated.proj_sample_index = null_fit.proj_sample_index
    updated.n_obs_per_indiv_per_env = getattr(null_fit, "n_obs_per_indiv_per_env", None)
    updated.vg_structure_rr = structure
    updated.n_traits = b
    updated.n_envs = E

    if Vg_trait is not None:
        updated.Vg_trait = Vg_trait
        updated.Vg_env = Vg_env
        updated.K_coef = Vg_trait
    else:
        # Recompute marginal K_coef and Vg_env from the refit unstructured Vg
        K_coef = torch.zeros(b, b, dtype=Vg.dtype, device=Vg.device)
        for k1 in range(b):
            for k2 in range(b):
                val = 0.0
                for e in range(E):
                    val += Vg[k1 * E + e, k2 * E + e].item()
                K_coef[k1, k2] = val / E
        Vg_env_marg = torch.zeros(E, E, dtype=Vg.dtype, device=Vg.device)
        for e1 in range(E):
            for e2 in range(E):
                val = 0.0
                for k in range(b):
                    val += Vg[k * E + e1, k * E + e2].item()
                Vg_env_marg[e1, e2] = val / b
        updated.K_coef = K_coef
        updated.Vg_env = Vg_env_marg

    return updated


def _update_fa(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """FA(k) warm-start: resume using fa_lbfgs_reml with current Vg as init."""
    from ..optim.fa_lbfgs_reml import fa_lbfgs_reml
    from ..optim.mvlmm_reml import mvlmm_null_quantities

    if null_fit.Y_rot is None or null_fit.X0_rot is None or null_fit.eigenvalues is None:
        raise ValueError("NullFit missing cached rotation data.")

    E = null_fit.Vg.shape[0]
    fa_rank = null_fit.fa_rank

    Vg, Ve, ll, trace, Lambda, psi = fa_lbfgs_reml(
        null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues,
        n_envs=E, fa_rank=fa_rank,
        max_iter=max_iter, tol=config.reml_convergence_tol,
        Vg_init=null_fit.Vg, Ve_init=null_fit.Ve,
    )

    null_q = mvlmm_null_quantities(Vg, Ve, null_fit.Y_rot, null_fit.X0_rot, null_fit.eigenvalues)

    converged = len(trace) > 0 and (
        len(trace) < max_iter
        or (len(trace) >= 2 and
            abs(trace[-1]["ll"] - trace[-2]["ll"])
            < config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0))
    )

    combined_trace = list(null_fit.optimizer_trace) + trace

    updated = NullFit(
        Vg=Vg, Ve=Ve,
        eigenvalues=null_fit.eigenvalues,
        eigenvectors=null_fit.eigenvectors,
        Y_rot=null_fit.Y_rot, X0_rot=null_fit.X0_rot,
        M00=null_q["M00"], b0=null_q["b0"], weights=null_q["W"],
        log_likelihood=ll, optimizer_trace=combined_trace,
        converged=converged, device=null_fit.device,
    )
    # Preserve FA metadata
    updated.fa_Lambda = Lambda
    updated.fa_psi = psi
    updated.fa_rank = fa_rank
    # Preserve MET metadata if present
    for attr in ("env_names", "n_env", "keep_mask", "parameterization", "vg_structure"):
        if hasattr(null_fit, attr):
            setattr(updated, attr, getattr(null_fit, attr))
    return updated


def _update_multi_kernel(
    null_fit: NullFit, *, max_iter: int, config: Any,
) -> NullFit:
    """Multi-kernel warm-start via multikernel_reml."""
    from ..linalg.safe import safe_cholesky
    from ..optim.multikernel_reml import multikernel_reml

    kernels = null_fit.kernels
    X0 = null_fit.X0

    # Multi-kernel doesn't store Y directly; reconstruct from X0 @ beta0 + r0
    if hasattr(null_fit, "beta0") and hasattr(null_fit, "r0"):
        Y = X0 @ null_fit.beta0 + null_fit.r0
    elif null_fit.Y_rot is not None:
        Y = null_fit.Y_rot
    else:
        raise ValueError("NullFit missing Y data for multi-kernel update.")

    # Use previous variance estimates as init
    init_vars = null_fit.variances

    variances, ll, trace = multikernel_reml(
        Y, X0, kernels,
        max_iter=max_iter, tol=config.reml_convergence_tol,
        init_vars=init_vars,
    )

    sig2_components = variances[:-1]
    sig2_e = variances[-1]
    n = Y.shape[0]
    device = Y.device

    from ..config import STAT_DTYPE
    V = sig2_e * torch.eye(n, dtype=STAT_DTYPE, device=device)
    for k, K_k in enumerate(kernels):
        V = V + sig2_components[k] * K_k

    L_V = safe_cholesky(V, jitter_factor=config.cholesky_jitter_factor)
    V_inv = torch.cholesky_inverse(L_V)

    V_inv_X0 = V_inv @ X0
    XtVinvX = X0.T @ V_inv_X0
    XtVinvY = V_inv_X0.T @ Y
    beta0 = torch.linalg.solve(XtVinvX, XtVinvY)
    r0 = Y - X0 @ beta0

    total_var = sum(variances)
    kernel_names = null_fit.kernel_names
    partitioned_h2 = {}
    variance_dict = {}
    for k, name in enumerate(kernel_names):
        partitioned_h2[name] = sig2_components[k] / total_var
        variance_dict[name] = sig2_components[k]
    partitioned_h2["residual"] = sig2_e / total_var
    variance_dict["residual"] = sig2_e

    converged = len(trace) > 0 and (
        len(trace) < max_iter or
        (len(trace) >= 2 and
         abs(trace[-1]["ll"] - trace[-2]["ll"]) <
         config.reml_convergence_tol * max(abs(trace[-1]["ll"]), 1.0))
    )

    nf = NullFit(
        sig2_g=sum(sig2_components), sig2_e=sig2_e,
        log_likelihood=ll, optimizer_trace=trace,
        converged=converged, device=device,
    )
    nf.V_inv = V_inv
    nf.X0 = X0
    nf.Y_rot = Y
    nf.beta0 = beta0
    nf.r0 = r0
    nf.kernels = kernels
    nf.kernel_names = kernel_names
    nf.variances = variances
    nf.variance_dict = variance_dict
    nf.partitioned_h2 = partitioned_h2
    return nf
