"""Random Regression LMM for longitudinal / spatio-temporal GWAS.

Implements:
- ``longitudinal_to_wide``: per-individual OLS projection of long-format
  observations onto a basis-coefficient space, producing an (n, b) "pseudo-trait"
  matrix and per-individual coefficient precision tensor (n, b, b). This is the
  bridge that reduces a random regression model to a multi-trait LMM.
- ``RandomRegressionLMM`` (skeleton — null-fit and scan added in subsequent
  steps).

Mathematical reduction
----------------------
For individual i with T_i observations y_i = (y_{i,t_{i1}}, ..., y_{i,t_{iT_i}}):

    y_i = Phi_i a_i + e_i,    Phi_i = [phi(t_{i1}); ...; phi(t_{iT_i})]  shape (T_i, b)

The unweighted OLS projection a_hat_i = (Phi_i' Phi_i)^{-1} Phi_i' y_i is a
``b``-vector of "pseudo-trait" values. Stacking across individuals gives an
``(n, b)`` matrix that can be fed to a multi-trait LMM where the basis-coefficient
covariance ``K_coef`` plays the role of ``Vg``.

The per-individual sampling precision of a_hat_i (assuming residual variance
sigma^2_e absorbed by the outer LMM) is

    Prec_i = (Phi_i' Phi_i) / sigma^2_e ~  proportional to Phi_i' Phi_i.

We return the un-scaled gram matrices (n, b, b) so the outer LMM can absorb
sigma^2_e during REML.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.basis import (
    evaluate_basis_at,
    place_knots,
    standardize_time,
)
from ..linalg.eigh import rotate
from ..stats.tests import apply_contrast, chi2_sf
from .base import NullFit, VariantMeta

logger = logging.getLogger(__name__)


@dataclass
class LongitudinalProjection:
    """Per-individual projection of long-format phenotypes onto basis coefficients.

    Attributes
    ----------
    Y_wide : Tensor
        ``(n, b)`` projected coefficients per individual.
    Phi_list : list[Tensor]
        Per-individual design matrices, each ``(T_i, b)``.
    gram_list : list[Tensor]
        Per-individual ``Phi_i' Phi_i`` matrices, each ``(b, b)``.
    weights : Tensor
        ``(n, b, b)`` per-individual coefficient precision (= ``Phi_i' Phi_i``).
        Up to a residual scale absorbed by REML, this is the inverse covariance
        of the projected coefficients.
    sample_index : Tensor
        ``(n,)`` ordered unique sample identifiers (integer codes).
    n_obs_per_indiv : Tensor
        ``(n,)`` number of observations per individual.
    basis_kind : str
    basis_params : dict
    b : int
    """

    Y_wide: Tensor
    Phi_list: list[Tensor]
    gram_list: list[Tensor]
    weights: Tensor
    sample_index: Tensor
    n_obs_per_indiv: Tensor
    basis_kind: str
    basis_params: dict
    b: int


def longitudinal_to_wide(
    Y_long: Tensor,
    sample_ids: Tensor,
    time_values: Tensor,
    basis_kind: str = "legendre",
    *,
    order: int = 3,
    n_interior_knots: int = 4,
    degree: int = 3,
    knot_kind: str = "quantile",
    t_min: float | None = None,
    t_max: float | None = None,
) -> LongitudinalProjection:
    """Project long-format longitudinal phenotypes onto basis coefficients.

    Builds a per-individual design matrix Phi_i from the basis evaluated at the
    individual's observation times, then OLS-projects y_i to a b-vector of
    pseudo-trait coefficients. Stacks across individuals to produce an (n, b)
    matrix consumable by a multi-trait LMM.

    Parameters
    ----------
    Y_long : Tensor
        ``(N,)`` phenotype values (one per observation row).
    sample_ids : Tensor
        ``(N,)`` integer sample codes.
    time_values : Tensor
        ``(N,)`` raw time values.
    basis_kind : {"legendre", "bspline"}
    order : int
        Legendre polynomial order (b = order + 1). Used when
        ``basis_kind="legendre"``.
    n_interior_knots : int
        Number of interior B-spline knots. Used when ``basis_kind="bspline"``.
    degree : int
        B-spline degree.
    knot_kind : {"quantile", "uniform", "extended"}
        Knot placement strategy for B-splines.
    t_min, t_max : float, optional
        Reference range for time standardization. If omitted, taken from
        ``time_values``. Useful when re-projecting at a held-out individual
        whose times must align with the training reference frame.

    Returns
    -------
    LongitudinalProjection

    Raises
    ------
    ValueError
        If any individual has fewer observations than the basis dimension
        ``b`` (rank-deficient projection).
    """
    Y_long = Y_long.to(STAT_DTYPE).reshape(-1)
    time_values = time_values.to(STAT_DTYPE).reshape(-1)
    sample_ids = sample_ids.reshape(-1)

    if Y_long.shape[0] != sample_ids.shape[0] or Y_long.shape[0] != time_values.shape[0]:
        raise ValueError(
            "longitudinal_to_wide: Y_long, sample_ids, and time_values must "
            f"share length. Got {Y_long.shape[0]}, {sample_ids.shape[0]}, "
            f"{time_values.shape[0]}."
        )

    # --- Resolve basis configuration & b ---
    if basis_kind == "legendre":
        if order < 0:
            raise ValueError("longitudinal_to_wide: order must be >= 0.")
        b = order + 1
        if t_min is None or t_max is None:
            _, t_min_res, t_max_res = standardize_time(time_values)
            t_min = float(t_min_res) if t_min is None else float(t_min)
            t_max = float(t_max_res) if t_max is None else float(t_max)
        basis_params: dict = {"order": order, "t_min": t_min, "t_max": t_max}
    elif basis_kind == "bspline":
        if n_interior_knots < 0:
            raise ValueError("longitudinal_to_wide: n_interior_knots must be >= 0.")
        knots = place_knots(
            time_values, n_interior=n_interior_knots, kind=knot_kind, degree=degree
        )
        b = len(knots) + degree - 1
        basis_params = {"knots": knots, "degree": degree}
    else:
        raise ValueError(f"longitudinal_to_wide: unknown basis_kind '{basis_kind}'.")

    # --- Group observations per individual (preserve first-seen order) ---
    sample_index, inverse = torch.unique(sample_ids, return_inverse=True, sorted=True)
    n = int(sample_index.shape[0])

    Y_wide = torch.zeros(n, b, dtype=STAT_DTYPE, device=Y_long.device)
    weights = torch.zeros(n, b, b, dtype=STAT_DTYPE, device=Y_long.device)
    n_obs = torch.zeros(n, dtype=torch.int64, device=Y_long.device)
    Phi_list: list[Tensor] = []
    gram_list: list[Tensor] = []

    for i in range(n):
        mask = inverse == i
        t_i = time_values[mask]
        y_i = Y_long[mask]
        T_i = int(t_i.shape[0])
        n_obs[i] = T_i

        if T_i < b:
            raise ValueError(
                f"longitudinal_to_wide: individual index {i} (id="
                f"{sample_index[i].item()}) has T_i={T_i} observations, "
                f"fewer than basis dimension b={b}. Rank-deficient projection."
            )

        Phi_i = evaluate_basis_at(t_i, basis_kind, basis_params)  # (T_i, b)
        gram_i = Phi_i.T @ Phi_i  # (b, b)

        # OLS projection: a_hat_i = (Phi_i' Phi_i)^{-1} Phi_i' y_i
        # Use lstsq for numerical stability over explicit inverse.
        rhs = Phi_i.T @ y_i  # (b,)
        a_hat = torch.linalg.solve(gram_i, rhs)  # (b,)

        Y_wide[i] = a_hat
        weights[i] = gram_i
        Phi_list.append(Phi_i)
        gram_list.append(gram_i)

    return LongitudinalProjection(
        Y_wide=Y_wide,
        Phi_list=Phi_list,
        gram_list=gram_list,
        weights=weights,
        sample_index=sample_index,
        n_obs_per_indiv=n_obs,
        basis_kind=basis_kind,
        basis_params=basis_params,
        b=b,
    )


# ── RRScanResult ────────────────────────────────────────────────────


@dataclass
class RRScanResult:
    """Per-variant random regression GWAS results.

    Mirrors :class:`ScanResult` but exposes the four test types specific to
    random regression: joint, intercept, slope, time-varying. The ``stat``
    and ``p`` attributes alias the joint test so the result remains
    interchangeable with the standard merge/FDR pipeline.
    """

    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor

    beta: Tensor       # (m, b) coefficient effects
    se: Tensor         # (m, b)
    Var_beta: Tensor   # (m, b, b) full coefficient covariance per SNP

    # Joint test χ²(b)
    stat_joint: Tensor
    p_joint: Tensor

    # Intercept test χ²(1) — first basis coefficient (time-stable for Legendre)
    stat_intercept: Tensor
    p_intercept: Tensor

    # Slope test χ²(1) — second basis coefficient (linear trend for Legendre)
    stat_slope: Tensor
    p_slope: Tensor

    # Time-varying test χ²(b−1) — all coefficients except the intercept
    stat_time_varying: Tensor
    p_time_varying: Tensor

    test: str = "wald"
    inference_type: str = "marginal"
    n_obs: Tensor | None = None

    # Optional per-time-point reconstruction (populated when eval_times is given)
    # eval_times shape: (n_t,) raw time values
    # beta_at_t shape:  (m, n_t)   — point estimate β(t) = φ(t)' β_j
    # se_at_t shape:    (m, n_t)
    # stat_at_t shape:  (m, n_t)   — χ²(1) per (SNP, time)
    # p_at_t shape:     (m, n_t)
    eval_times: Tensor | None = None
    beta_at_t: Tensor | None = None
    se_at_t: Tensor | None = None
    stat_at_t: Tensor | None = None
    p_at_t: Tensor | None = None

    # Metadata
    basis_kind: str = "legendre"
    basis_params: dict | None = None
    b: int = 0

    def __len__(self) -> int:
        return len(self.snp)

    @property
    def stat(self) -> Tensor:
        """Alias to joint test (interchangeable with ScanResult.stat)."""
        return self.stat_joint

    @property
    def p(self) -> Tensor:
        """Alias to joint test (interchangeable with ScanResult.p)."""
        return self.p_joint


# NOTE: chi2_sf and apply_contrast are imported from torchgwas.stats.tests.
# They were extracted from this module in Phase 38 step 5 so multi_env_lmm,
# rr_lmm, and any future contrast-test consumers can share the same kernel.


# ── RandomRegressionLMM ──────────────────────────────────────────────


class RandomRegressionLMM:
    """Random Regression LMM for longitudinal GWAS.

    .. note::
        Skeleton class. ``fit_null`` and ``score_chunk`` are implemented in
        the next steps of Phase 38. Currently exposes the projection helper
        as a static method.
    """

    def __init__(
        self,
        basis: str = "legendre",
        order: int = 3,
        n_interior_knots: int = 4,
        degree: int = 3,
        knot_kind: str = "quantile",
        k_coef_structure: str = "unstructured",
        include_pe: bool = False,
        mode: str = "projection",
        config: NumericalConfig | None = None,
    ) -> None:
        if basis not in ("legendre", "bspline"):
            raise ValueError(f"RandomRegressionLMM: unknown basis '{basis}'.")
        if k_coef_structure not in ("unstructured", "diagonal") and not k_coef_structure.startswith("fa("):
            raise ValueError(
                f"RandomRegressionLMM: unknown k_coef_structure '{k_coef_structure}'."
            )
        if mode not in ("projection", "stacked"):
            raise ValueError(f"RandomRegressionLMM: unknown mode '{mode}'.")

        self.basis = basis
        self.order = order
        self.n_interior_knots = n_interior_knots
        self.degree = degree
        self.knot_kind = knot_kind
        self.k_coef_structure = k_coef_structure
        self.include_pe = include_pe
        self.mode = mode
        self.config = config or NumericalConfig()

    def project(
        self,
        Y_long: Tensor,
        sample_ids: Tensor,
        time_values: Tensor,
        *,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> LongitudinalProjection:
        """Run :func:`longitudinal_to_wide` with the model's basis configuration."""
        return longitudinal_to_wide(
            Y_long,
            sample_ids,
            time_values,
            basis_kind=self.basis,
            order=self.order,
            n_interior_knots=self.n_interior_knots,
            degree=self.degree,
            knot_kind=self.knot_kind,
            t_min=t_min,
            t_max=t_max,
        )

    def fit_null(
        self,
        Y_long: Tensor,
        X0: Tensor,
        K: Tensor,
        *,
        sample_ids: Tensor,
        time_values: Tensor,
        t_min: float | None = None,
        t_max: float | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the random regression null model.

        Long-format observations are projected to a per-individual ``(n, b)``
        coefficient matrix and then dispatched through :class:`MultiTraitLMM`,
        which estimates the basis-coefficient covariance ``K_coef`` (playing
        the role of ``Vg``) and residual coefficient covariance ``Ve``.

        Parameters
        ----------
        Y_long : Tensor
            ``(N,)`` long-format phenotype values.
        X0 : Tensor
            ``(n, c)`` per-individual covariate matrix (intercept first).
            Must be ordered to match the projected ``sample_index``.
        K : Tensor
            ``(n, n)`` GRM / kinship matrix, ordered to match
            ``sample_index``.
        sample_ids : Tensor
            ``(N,)`` integer sample codes (one per long-format row).
        time_values : Tensor
            ``(N,)`` raw time values (one per long-format row).
        t_min, t_max : float, optional
            Reference range for time standardization.

        Returns
        -------
        NullFit
            Cached null-model quantities. RR-specific metadata is attached:
            ``K_coef``, ``basis_kind``, ``basis_params``, ``b``, ``Phi_list``,
            ``proj_weights``, ``n_obs_per_indiv``, ``sample_index``.
        """
        if K is None:
            raise ValueError("RandomRegressionLMM.fit_null requires a kinship matrix K.")
        if self.mode not in ("projection", "stacked"):
            raise NotImplementedError(
                f"RandomRegressionLMM mode='{self.mode}' not yet implemented."
            )

        # 1. Project long-format observations to coefficient space
        proj = self.project(
            Y_long, sample_ids, time_values, t_min=t_min, t_max=t_max
        )

        n = int(proj.Y_wide.shape[0])
        b = proj.b

        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        if X0.dim() != 2 or X0.shape[0] != n:
            raise ValueError(
                f"RandomRegressionLMM.fit_null: X0 must have shape (n={n}, c); "
                f"got {tuple(X0.shape)}."
            )
        if K.shape != (n, n):
            raise ValueError(
                f"RandomRegressionLMM.fit_null: K must have shape ({n}, {n}); "
                f"got {tuple(K.shape)}."
            )
        if b < 2:
            raise ValueError(
                f"RandomRegressionLMM.fit_null: basis dimension b={b} < 2. "
                "Use SingleTraitLMM directly for a constant-only basis."
            )

        # 2. Dispatch projected coefficients through the appropriate REML path.
        #    "unstructured" reuses the well-tested MultiTraitLMM driver;
        #    "diagonal" / "fa(k)" go through the rr_reml wrappers and then
        #    rebuild the same NullFit structure that MultiTraitLMM produces.
        from ..optim.rr_reml import (
            parse_k_coef_structure,
            rr_reml_diagonal,
            rr_reml_fa,
        )
        from .multi_trait_lmm import MultiTraitLMM

        kind, fa_rank = parse_k_coef_structure(self.k_coef_structure)

        # ── Stacked-mode verification path ──────────────────────────
        # Treats the T common time points as T pseudo-traits via
        # MultiTraitLMM, then projects the estimated Vg_T / Ve_T into
        # basis-coefficient space. Requires balanced data (same time
        # vector across all individuals). Used to cross-check the
        # projection-mode K_coef estimate.
        if self.mode == "stacked":
            from ..linalg.eigh import eigendecompose, rotate
            from ..optim.mvlmm_reml import mvlmm_null_quantities
            from .multi_trait_lmm import MultiTraitLMM

            # Verify balanced layout: every individual has same n_obs and
            # same time vector (in the same order). Use the first
            # individual's times as the canonical reference.
            n_obs_unique = torch.unique(proj.n_obs_per_indiv)
            if n_obs_unique.numel() != 1:
                raise ValueError(
                    "RandomRegressionLMM mode='stacked' requires balanced data "
                    "(every individual must have the same number of observations). "
                    f"Got T_i values: {sorted(set(proj.n_obs_per_indiv.tolist()))}."
                )
            T_common = int(n_obs_unique.item())
            if T_common < b:
                raise ValueError(
                    f"stacked mode: T_common={T_common} < b={b}; cannot project."
                )
            # Reference time vector from first individual.
            ref_Phi = proj.Phi_list[0]  # (T_common, b)
            for i in range(1, len(proj.Phi_list)):
                if not torch.allclose(proj.Phi_list[i], ref_Phi, atol=1e-10):
                    raise ValueError(
                        "stacked mode: individuals have different time grids; "
                        "data is not balanced."
                    )

            # Build wide (n, T_common) phenotype matrix in the same row
            # order as proj.sample_index. We re-bucket Y_long by the
            # original sample id ordering used in the projection.
            sample_index = proj.sample_index
            n_indiv = int(sample_index.shape[0])
            Y_long_t = Y_long.to(STAT_DTYPE).reshape(-1)
            sample_ids_t = sample_ids.reshape(-1)
            time_values_t = time_values.to(STAT_DTYPE).reshape(-1)
            Y_wide_T = torch.zeros(
                n_indiv, T_common, dtype=STAT_DTYPE, device=Y_long_t.device
            )
            # Common time vector (sorted) for stable column ordering.
            t_first_mask = sample_ids_t == sample_index[0]
            t_common_vals, t_perm = torch.sort(time_values_t[t_first_mask])
            for i in range(n_indiv):
                mask = sample_ids_t == sample_index[i]
                t_i = time_values_t[mask]
                y_i = Y_long_t[mask]
                # Sort by time so columns match canonical t_common_vals.
                order_i = torch.argsort(t_i)
                Y_wide_T[i] = y_i[order_i]

            # Fit T-trait MultiTraitLMM in stacked space.
            inner = MultiTraitLMM(config=self.config)
            stacked_fit = inner.fit_null(Y_wide_T, X0, K)
            Vg_T = stacked_fit.Vg
            Ve_T = stacked_fit.Ve

            # Project Vg_T, Ve_T into basis-coefficient space via the
            # canonical Phi: K_coef = Φ⁺ Vg_T Φ⁺ᵀ where Φ⁺ = (ΦᵀΦ)⁻¹ Φᵀ
            # is the b×T pseudo-inverse for the shared (sorted) grid.
            t_common_phi = evaluate_basis_at(
                t_common_vals, proj.basis_kind, proj.basis_params
            ).to(STAT_DTYPE)  # (T_common, b)
            gram_phi = t_common_phi.T @ t_common_phi  # (b, b)
            phi_pinv = torch.linalg.solve(gram_phi, t_common_phi.T)  # (b, T_common)
            K_coef_stacked = phi_pinv @ Vg_T @ phi_pinv.T
            Ve_coef_stacked = phi_pinv @ Ve_T @ phi_pinv.T
            # Symmetrize for numerical safety.
            K_coef_stacked = 0.5 * (K_coef_stacked + K_coef_stacked.T)
            Ve_coef_stacked = 0.5 * (Ve_coef_stacked + Ve_coef_stacked.T)

            # Build a coefficient-space NullFit so the standard score_chunk
            # path works on the stacked estimate. We rotate the projected
            # (n, b) Y_wide and run mvlmm_null_quantities with the
            # stacked-derived Vg/Ve.
            ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)
            Y_rot = rotate(proj.Y_wide, ed.eigenvectors)
            X0_rot = rotate(X0, ed.eigenvectors)
            null_q = mvlmm_null_quantities(
                K_coef_stacked, Ve_coef_stacked, Y_rot, X0_rot, ed.eigenvalues
            )
            null_fit = NullFit(
                Vg=K_coef_stacked,
                Ve=Ve_coef_stacked,
                eigenvalues=ed.eigenvalues,
                eigenvectors=ed.eigenvectors,
                Y_rot=Y_rot,
                X0_rot=X0_rot,
                M00=null_q["M00"],
                b0=null_q["b0"],
                weights=null_q["W"],
                log_likelihood=stacked_fit.log_likelihood,
                optimizer_trace=stacked_fit.optimizer_trace,
                converged=stacked_fit.converged,
                device=ed.eigenvalues.device,
            )
            # Stash stacked diagnostics so tests can inspect them.
            null_fit.Vg_T_stacked = Vg_T
            null_fit.Ve_T_stacked = Ve_T

        elif kind == "unstructured":
            inner = MultiTraitLMM(config=self.config)
            null_fit = inner.fit_null(proj.Y_wide, X0, K)
        else:
            # Build the same rotated quantities MultiTraitLMM caches, then
            # call the structure-specific REML wrapper, then assemble NullFit.
            from ..linalg.eigh import eigendecompose, rotate
            from ..optim.mvlmm_reml import mvlmm_null_quantities

            ed = eigendecompose(K, eigenvalue_floor=self.config.eigenvalue_floor)
            Y_rot = rotate(proj.Y_wide, ed.eigenvectors)
            X0_rot = rotate(X0, ed.eigenvectors)

            if kind == "diagonal":
                Vg, Ve, ll, trace = rr_reml_diagonal(
                    Y_rot, X0_rot, ed.eigenvalues, b=b,
                )
            elif kind == "fa":
                if fa_rank is None or fa_rank <= 0:
                    raise ValueError(
                        f"RandomRegressionLMM.fit_null: fa rank must be a positive int, "
                        f"got '{self.k_coef_structure}'."
                    )
                if fa_rank >= b:
                    raise ValueError(
                        f"RandomRegressionLMM.fit_null: fa({fa_rank}) requires "
                        f"fa_rank < b={b}. Use 'unstructured' for full-rank K_coef."
                    )
                Vg, Ve, ll, trace = rr_reml_fa(
                    Y_rot, X0_rot, ed.eigenvalues, b=b, fa_rank=fa_rank,
                )
            else:  # pragma: no cover
                raise AssertionError(f"unhandled k_coef kind: {kind}")

            null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, ed.eigenvalues)
            converged = (
                len(trace) > 0 and len(trace) < self.config.reml_max_iter
            )
            null_fit = NullFit(
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

        # 3. Attach RR-specific metadata so update_null and downstream
        #    consumers can recognize this as a random regression fit.
        null_fit.K_coef = null_fit.Vg
        null_fit.basis_kind = proj.basis_kind
        null_fit.basis_params = proj.basis_params
        null_fit.b = b
        null_fit.Phi_list = proj.Phi_list
        null_fit.proj_weights = proj.weights
        null_fit.n_obs_per_indiv = proj.n_obs_per_indiv
        null_fit.sample_index = proj.sample_index
        null_fit.k_coef_structure = self.k_coef_structure
        null_fit.include_pe = self.include_pe
        null_fit.K_pe = None
        null_fit.sigma2_e_residual = None

        # 4. Optional permanent-environment decomposition of Ve.
        #    Decomposes the fitted (b,b) residual covariance into a
        #    permanent-environment block K_pe (cross-coefficient correlations
        #    that persist within an individual) plus a residual sampling-noise
        #    floor σ²_e_residual × mean((Φ_i' Φ_i)^{-1}).
        #    Identifiability requires every individual to have T_i ≥ 2 so
        #    that the per-individual gram matrix is invertible.
        if self.include_pe:
            min_T = int(proj.n_obs_per_indiv.min().item())
            if min_T < 2:
                raise ValueError(
                    f"RandomRegressionLMM.fit_null: include_pe=True requires "
                    f"every individual to have T_i ≥ 2 observations; "
                    f"smallest individual has T_i={min_T}."
                )
            # Average per-individual sampling-noise scale: mean of
            # (Φ_i' Φ_i)^{-1} across individuals. The OLS sampling noise of
            # the per-individual coefficient estimate is σ²_e (Φ_i' Φ_i)^{-1}.
            mean_sampling = torch.zeros(b, b, dtype=STAT_DTYPE, device=null_fit.Vg.device)
            for gram in proj.gram_list:
                mean_sampling = mean_sampling + torch.linalg.inv(gram)
            mean_sampling = mean_sampling / float(len(proj.gram_list))

            # Estimate σ²_e_residual as the largest scalar that keeps
            # K_pe = Ve - σ²_e_residual * mean_sampling positive semidefinite.
            # Use the smallest generalized eigenvalue of (Ve, mean_sampling).
            try:
                eigvals = torch.linalg.eigvalsh(
                    torch.linalg.solve(mean_sampling, null_fit.Ve)
                )
                sigma2_e_residual = float(torch.clamp(eigvals.min(), min=0.0).item())
            except Exception:
                sigma2_e_residual = 0.0
            K_pe = null_fit.Ve - sigma2_e_residual * mean_sampling
            # Symmetrize and project tiny negatives to zero
            K_pe = 0.5 * (K_pe + K_pe.T)
            evals_pe, evecs_pe = torch.linalg.eigh(K_pe)
            evals_pe = torch.clamp(evals_pe, min=0.0)
            K_pe = (evecs_pe * evals_pe.unsqueeze(0)) @ evecs_pe.T

            null_fit.K_pe = K_pe
            null_fit.sigma2_e_residual = sigma2_e_residual

        logger.info(
            "RandomRegressionLMM null fit: n=%d, b=%d, basis=%s, ll=%.4f, converged=%s",
            n, b, proj.basis_kind, null_fit.log_likelihood or 0.0, null_fit.converged,
        )

        return null_fit

    # ── Interpretive helpers (Phase 38, Step 9) ─────────────────────
    #
    # These post-fit summaries turn the (b, b) basis-coefficient
    # covariances K_coef = Vg and Ve into time-domain quantities a
    # quantitative geneticist can read directly: variance curves,
    # heritability curves, genetic correlation surfaces, and the
    # functional principal components (FPCs) of the genetic variance
    # operator.

    @staticmethod
    def _phi_at(null_fit: NullFit, t_query: Tensor) -> Tensor:
        """Re-evaluate the basis stored on ``null_fit`` at given times."""
        if not hasattr(null_fit, "basis_kind") or null_fit.basis_kind is None:
            raise ValueError(
                "RandomRegressionLMM helper: null_fit has no basis metadata "
                "(not a random regression NullFit)."
            )
        t_query = torch.as_tensor(t_query, dtype=STAT_DTYPE).reshape(-1)
        return evaluate_basis_at(t_query, null_fit.basis_kind, null_fit.basis_params)

    @classmethod
    def genetic_variance_curve(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Genetic variance σ²_g(t) = φ(t)' K_coef φ(t) at each query time.

        Returns a ``(T_query,)`` tensor.
        """
        Phi = cls._phi_at(null_fit, t_query)
        K_coef = null_fit.K_coef
        # diag(Phi K_coef Phi') = sum_{ij} Phi_ti K_ij Phi_tj
        return torch.einsum('ti,ij,tj->t', Phi, K_coef, Phi)

    @classmethod
    def residual_variance_curve(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Residual coefficient variance σ²_e(t) = φ(t)' Ve φ(t)."""
        Phi = cls._phi_at(null_fit, t_query)
        return torch.einsum('ti,ij,tj->t', Phi, null_fit.Ve, Phi)

    @classmethod
    def heritability_curve(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Pointwise narrow-sense h²(t) = σ²_g(t) / (σ²_g(t) + σ²_e(t))."""
        var_g = cls.genetic_variance_curve(null_fit, t_query)
        var_e = cls.residual_variance_curve(null_fit, t_query)
        denom = torch.clamp(var_g + var_e, min=1e-30)
        return var_g / denom

    @classmethod
    def genetic_correlation_surface(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Genetic correlation surface r_g(t, s).

        Returns a symmetric ``(T_query, T_query)`` tensor with diagonal
        equal to 1.0 (up to floating-point) and off-diagonals equal to
        Cov_g(t, s) / sqrt(Var_g(t) Var_g(s)).
        """
        Phi = cls._phi_at(null_fit, t_query)
        K_coef = null_fit.K_coef
        Cov = Phi @ K_coef @ Phi.T  # (T, T)
        diag = torch.clamp(torch.diagonal(Cov), min=1e-30)
        denom = torch.sqrt(diag.unsqueeze(0) * diag.unsqueeze(1))
        return Cov / denom

    @classmethod
    def eigenfunctions(
        cls,
        null_fit: NullFit,
        t_query: Tensor,
        n_components: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Functional principal components of the genetic covariance.

        Returns ``(eigenvalues, eigenfunctions)`` where ``eigenvalues`` is
        a ``(b,)`` (or ``(n_components,)``) descending tensor of K_coef
        eigenvalues and ``eigenfunctions`` is a ``(T_query, b)`` matrix of
        FPCs evaluated on the query grid: column k is φ(t)' v_k where v_k
        is the k-th eigenvector of K_coef.

        These are the discrete analogue of the Karhunen-Loève expansion of
        the genetic random function.
        """
        Phi = cls._phi_at(null_fit, t_query)
        K_coef = null_fit.K_coef
        evals, evecs = torch.linalg.eigh(K_coef)
        # Reorder to descending eigenvalues for FPC convention
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        if n_components is not None:
            evals = evals[:n_components]
            evecs = evecs[:, :n_components]
        eigenfns = Phi @ evecs  # (T_query, k)
        return evals, eigenfns

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
        *,
        eval_times: Tensor | None = None,
        **kwargs: Any,
    ) -> RRScanResult:
        """Score a genotype chunk against the random regression null model.

        Solves a per-SNP full-block GLS system for the b-vector of basis-
        coefficient effects ``β_j``, then evaluates four base contrast tests
        plus an optional per-time-point reconstruction:

        - **Joint** χ²(b): tests ``H0: β = 0``  (any time-varying effect)
        - **Intercept** χ²(1): tests ``H0: β_0 = 0``  (time-stable effect)
        - **Slope** χ²(1): tests ``H0: β_1 = 0``  (linear trend; b ≥ 2)
        - **Time-varying** χ²(b−1): tests ``H0: β_{1..b-1} = 0``
          (anything beyond a constant)
        - **Effect at time t** χ²(1) (when ``eval_times`` is supplied):
          tests ``H0: β(t) = 0`` where ``β(t) = φ(t)' β_j``. Uses the basis
          stored on ``null_fit`` to re-evaluate at user-supplied raw time
          values.

        Parameters
        ----------
        G_chunk : Tensor
            ``(n, m)`` genotype dosage matrix (one row per individual).
        null_fit : NullFit
            Output of :meth:`fit_null`.
        variant_meta : VariantMeta
            Per-variant identifiers.
        test : str
            Currently only ``"wald"`` is supported.
        eval_times : Tensor, optional
            ``(n_t,)`` raw time values at which to reconstruct per-time
            effects. When supplied, ``RRScanResult.beta_at_t / se_at_t /
            stat_at_t / p_at_t`` are populated.

        Returns
        -------
        RRScanResult
            Per-variant statistics for all four base test types and (if
            ``eval_times`` was supplied) the per-time-point χ²(1) tests.
        """
        if test != "wald":
            raise ValueError(
                f"RandomRegressionLMM.score_chunk currently supports 'wald', got '{test}'."
            )

        if not hasattr(null_fit, "K_coef") or null_fit.K_coef is None:
            raise ValueError(
                "score_chunk: null_fit is not a random regression NullFit "
                "(missing K_coef metadata)."
            )

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        b = null_fit.b

        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)  # (n, m)

        af = G_chunk.mean(dim=0) / 2.0  # diploid convention

        W = null_fit.weights      # (n, b, b)
        Y_rot = null_fit.Y_rot    # (n, b)
        X0_rot = null_fit.X0_rot  # (n, c)
        M00 = null_fit.M00        # (cb, cb)
        b0 = null_fit.b0          # (cb,)
        c = X0_rot.shape[1]

        # --- Full-system GLS: replicate the MultiTraitLMM template (d → b) ---
        WY = torch.einsum('ist,is->it', W, Y_rot)              # (n, b)
        g2 = G_rot ** 2                                         # (n, m)
        M11 = torch.einsum('ni,nst->ist', g2, W)                # (m, b, b)
        M01_raw = torch.einsum('ia,its,ij->jats', X0_rot, W, G_rot)  # (m, c, b, b)
        M01 = M01_raw.reshape(m, c * b, b)                      # (m, cb, b)
        b1 = torch.einsum('ij,is->js', G_rot, WY)               # (m, b)

        dim_full = c * b + b
        M_full = torch.zeros(m, dim_full, dim_full, dtype=STAT_DTYPE, device=G_chunk.device)
        M_full[:, :c*b, :c*b] = M00.unsqueeze(0)
        M_full[:, :c*b, c*b:] = M01
        M_full[:, c*b:, :c*b] = M01.transpose(1, 2)
        M_full[:, c*b:, c*b:] = M11

        rhs_full = torch.zeros(m, dim_full, dtype=STAT_DTYPE, device=G_chunk.device)
        rhs_full[:, :c*b] = b0.unsqueeze(0)
        rhs_full[:, c*b:] = b1

        # Jitter monomorphic SNPs
        diag_min = torch.diagonal(M_full, dim1=-2, dim2=-1).abs().min(dim=-1).values
        singular_mask = diag_min < 1e-10
        if singular_mask.any():
            jitter = torch.eye(dim_full, dtype=STAT_DTYPE, device=M_full.device) * 1e-10
            M_full = M_full + singular_mask.view(-1, 1, 1) * jitter.unsqueeze(0)

        coef = torch.linalg.solve(M_full, rhs_full)  # (m, dim_full)
        beta = coef[:, c*b:]  # (m, b)

        # Var(beta) = lower-right block of M_full^{-1}: solve M_full @ X = [0; I_b]
        rhs_var = torch.zeros(m, dim_full, b, dtype=STAT_DTYPE, device=G_chunk.device)
        rhs_var[:, c*b:, :] = torch.eye(b, dtype=STAT_DTYPE, device=G_chunk.device).unsqueeze(0)
        Minv_cols = torch.linalg.solve(M_full, rhs_var)
        Var_beta = Minv_cols[:, c*b:, :]  # (m, b, b)

        se = torch.sqrt(
            torch.clamp(torch.diagonal(Var_beta, dim1=-2, dim2=-1), min=1e-30)
        )  # (m, b)

        # --- Joint test χ²(b) ---
        Var_inv_beta = torch.linalg.solve(Var_beta, beta.unsqueeze(-1)).squeeze(-1)
        stat_joint = torch.clamp((beta * Var_inv_beta).sum(dim=-1), min=0.0)
        p_joint = chi2_sf(stat_joint, df=b)

        # --- Contrast tests via shared apply_contrast helper ---
        device = beta.device

        # Intercept: test β_0 = 0
        C_intercept = torch.zeros(1, b, dtype=STAT_DTYPE, device=device)
        C_intercept[0, 0] = 1.0
        _, _, stat_intercept, p_intercept = apply_contrast(beta, Var_beta, C_intercept)

        # Slope: test β_1 = 0 (only meaningful when b ≥ 2)
        if b >= 2:
            C_slope = torch.zeros(1, b, dtype=STAT_DTYPE, device=device)
            C_slope[0, 1] = 1.0
            _, _, stat_slope, p_slope = apply_contrast(beta, Var_beta, C_slope)
        else:
            stat_slope = torch.full((m,), float("nan"), dtype=STAT_DTYPE, device=device)
            p_slope = torch.full((m,), float("nan"), dtype=STAT_DTYPE, device=device)

        # Time-varying: test β_{1..b-1} = 0 (everything beyond the intercept)
        if b >= 2:
            C_tv = torch.zeros(b - 1, b, dtype=STAT_DTYPE, device=device)
            for i in range(b - 1):
                C_tv[i, i + 1] = 1.0
            _, _, stat_tv, p_tv = apply_contrast(beta, Var_beta, C_tv)
        else:
            stat_tv = torch.full((m,), float("nan"), dtype=STAT_DTYPE, device=device)
            p_tv = torch.full((m,), float("nan"), dtype=STAT_DTYPE, device=device)

        # --- Optional per-time-point reconstruction χ²(1) at each eval_times[k] ---
        eval_times_out: Tensor | None = None
        beta_at_t_out: Tensor | None = None
        se_at_t_out: Tensor | None = None
        stat_at_t_out: Tensor | None = None
        p_at_t_out: Tensor | None = None
        if eval_times is not None:
            eval_times_t = torch.as_tensor(eval_times, dtype=STAT_DTYPE, device=device).reshape(-1)
            n_t = int(eval_times_t.shape[0])

            if not hasattr(null_fit, "basis_kind") or null_fit.basis_kind is None:
                raise ValueError(
                    "score_chunk: eval_times requires null_fit.basis_kind / basis_params; "
                    "got a non-RR null fit."
                )
            # Phi_eval: (n_t, b) — basis re-evaluated at user time points
            Phi_eval = evaluate_basis_at(
                eval_times_t, null_fit.basis_kind, null_fit.basis_params
            ).to(device=device, dtype=STAT_DTYPE)

            if Phi_eval.shape[1] != b:
                raise ValueError(
                    f"score_chunk: basis re-evaluation produced b={Phi_eval.shape[1]}, "
                    f"expected b={b}. Check basis_params consistency."
                )

            # β(t)  = Phi_eval @ β_j   ->  (m, n_t)
            beta_at_t_out = beta @ Phi_eval.T

            # Var(β(t)) = φ(t)' Var_beta φ(t) per time, per SNP
            # Compute  Var_beta_phi : (m, b, n_t) = Var_beta @ Phi_eval^T
            Var_beta_phi = torch.einsum('mij,tj->mit', Var_beta, Phi_eval)
            # var_at_t : (m, n_t) — diagonal of φ' V φ via einsum (φ_t · (V φ_t))
            var_at_t = torch.einsum('tj,mjt->mt', Phi_eval, Var_beta_phi)
            var_at_t = torch.clamp(var_at_t, min=1e-30)
            se_at_t_out = torch.sqrt(var_at_t)

            # χ²(1) Wald per (SNP, time)
            stat_at_t_out = (beta_at_t_out ** 2) / var_at_t
            stat_at_t_out = torch.clamp(stat_at_t_out, min=0.0)
            # chi2_sf supports any tensor shape; df=1 broadcasts.
            p_at_t_out = chi2_sf(stat_at_t_out, df=1)
            eval_times_out = eval_times_t

        return RRScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=af,
            beta=beta,
            se=se,
            Var_beta=Var_beta,
            stat_joint=stat_joint,
            p_joint=p_joint,
            stat_intercept=stat_intercept,
            p_intercept=p_intercept,
            stat_slope=stat_slope,
            p_slope=p_slope,
            stat_time_varying=stat_tv,
            p_time_varying=p_tv,
            eval_times=eval_times_out,
            beta_at_t=beta_at_t_out,
            se_at_t=se_at_t_out,
            stat_at_t=stat_at_t_out,
            p_at_t=p_at_t_out,
            test="wald",
            basis_kind=null_fit.basis_kind,
            basis_params=null_fit.basis_params,
            b=b,
        )
