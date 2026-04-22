"""Random Regression × Multi-Environment LMM (Phase 39).

Extends Phase 38's :class:`RandomRegressionLMM` to longitudinal phenotypes
measured in **multiple environments / trials / years** on the same set of
individuals.  The natural data shape is

    (n samples, b basis coefficients, E environments)

which we flatten to ``b * E`` pseudo-traits and feed into the existing
:class:`MultiTraitMultiEnvLMM` (Phase 25) machinery, with the basis-coef axis
playing the role of the "trait" dimension.  This makes the implementation a
thin composition over an already-validated REML stack:

- ``Vg = K_coef ⊗ Vg_env``  (separable Kronecker, the default)
- per-SNP β has shape ``(m, b, E)`` and is tested via 9 RR-aware contrasts
  (joint, per-env, intercept-per-env, time-varying-per-env, stable-per-coef,
  joint-stable-vs-GxE, mean-curve, intercept-only-GxE, per-time × per-env)

Critical correctness invariant
------------------------------
**All E environments share a single global ``(t_min, t_max)`` and (for
B-spline) a single global knot vector placed on the pooled times across all
environments.**  If each env were standardized to its own time range, the
basis coefficient ``β_{k,e}`` would have a different physical meaning per
environment and the across-env contrasts (stable / GxE / per-time × per-env)
would become uninterpretable.  ``project_multi_env`` owns the pooling — the
caller does **not** pass per-env time references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..config import STAT_DTYPE, NumericalConfig
from ..linalg.basis import (
    evaluate_basis_at,
    place_knots,
    standardize_time,
)
from ..stats.tests import apply_contrast, chi2_sf
from .base import NullFit, VariantMeta
from .multi_trait_multi_env_lmm import (
    MultiTraitMultiEnvLMM,
    _gls_wald_scan,
)

logger = logging.getLogger(__name__)


# ── Per-environment projection helper ───────────────────────────────


def _project_one_env_with_params(
    Y_env: Tensor,
    sample_ids_env: Tensor,
    times_env: Tensor,
    basis_kind: str,
    basis_params: dict,
    b: int,
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    """Per-individual OLS projection within one environment, using a
    pre-built basis specification (so all environments share parameters).

    Returns
    -------
    Y_wide_e : (n_e, b) projected coefficients
    sample_index_e : (n_e,) ordered unique sample IDs in this env
    n_obs_e : (n_e,) observations per individual
    Phi_list_e : list of (T_i, b) per-individual design matrices
    """
    Y_env = Y_env.to(STAT_DTYPE).reshape(-1)
    times_env = times_env.to(STAT_DTYPE).reshape(-1)
    sample_ids_env = sample_ids_env.reshape(-1)

    sample_index_e, inverse = torch.unique(
        sample_ids_env, return_inverse=True, sorted=True
    )
    n_e = int(sample_index_e.shape[0])

    Y_wide_e = torch.zeros(n_e, b, dtype=STAT_DTYPE, device=Y_env.device)
    n_obs_e = torch.zeros(n_e, dtype=torch.int64, device=Y_env.device)
    Phi_list_e: list[Tensor] = []

    for i in range(n_e):
        mask = inverse == i
        t_i = times_env[mask]
        y_i = Y_env[mask]
        T_i = int(t_i.shape[0])
        n_obs_e[i] = T_i

        if T_i < b:
            raise ValueError(
                f"project_multi_env: individual id={int(sample_index_e[i].item())} "
                f"has T_i={T_i} observations in this environment, fewer than "
                f"basis dimension b={b}. Rank-deficient projection."
            )

        Phi_i = evaluate_basis_at(t_i, basis_kind, basis_params).to(STAT_DTYPE)
        gram_i = Phi_i.T @ Phi_i  # (b, b)
        rhs = Phi_i.T @ y_i  # (b,)
        a_hat = torch.linalg.solve(gram_i, rhs)

        Y_wide_e[i] = a_hat
        Phi_list_e.append(Phi_i)

    return Y_wide_e, sample_index_e, n_obs_e, Phi_list_e


# ── MultiEnvLongitudinalProjection dataclass ────────────────────────


@dataclass
class MultiEnvLongitudinalProjection:
    """Multi-environment per-individual projection of long-format phenotypes.

    Attributes
    ----------
    Y_proj : Tensor
        ``(n, b, E)`` projected basis coefficients per (individual, env).
        ``NaN`` where ``(i, e)`` was not observed.
    sample_index : Tensor
        ``(n,)`` global sorted unique sample IDs (across all envs).
    env_index : Tensor
        ``(E,)`` sorted unique environment codes.
    Phi_list_per_env : list[list[Tensor]]
        ``Phi_list_per_env[e][i]`` is the ``(T_{i,e}, b)`` design matrix for
        individual ``i`` in environment ``e``.  Indexing follows
        ``sample_index_per_env[e]`` (the env-local sample order), **not** the
        global ``sample_index``.
    sample_index_per_env : list[Tensor]
        For each env ``e``, the env-local sorted unique sample IDs that were
        actually observed in that env.
    n_obs_per_indiv_per_env : Tensor
        ``(n, E)`` int counts; 0 where ``(i, e)`` is missing.
    basis_kind : str
    basis_params : dict
        For ``"legendre"``: ``{"order", "t_min", "t_max"}`` (pooled across
        envs).  For ``"bspline"``: ``{"knots", "degree"}`` with shared knots.
    b : int
    E : int
    """

    Y_proj: Tensor
    sample_index: Tensor
    env_index: Tensor
    Phi_list_per_env: list[list[Tensor]]
    sample_index_per_env: list[Tensor]
    n_obs_per_indiv_per_env: Tensor
    basis_kind: str
    basis_params: dict
    b: int
    E: int


def project_multi_env(
    Y_long: Tensor,
    sample_ids: Tensor,
    env_ids: Tensor,
    time_values: Tensor,
    basis_kind: str = "legendre",
    *,
    order: int = 3,
    n_interior_knots: int = 4,
    degree: int = 3,
    knot_kind: str = "quantile",
) -> MultiEnvLongitudinalProjection:
    """Per-(individual, environment) basis projection with shared time frame.

    For each environment ``e`` and individual ``i`` observed in that env,
    fits an OLS projection of ``y_{i,e}(t)`` onto a basis evaluated at the
    individual's observation times.  All environments use the **same**
    standardized time range (Legendre) or **same** knot vector (B-spline)
    derived from the **pooled** time vector across all environments — this
    is the load-bearing correctness invariant for cross-environment
    contrasts.

    Parameters
    ----------
    Y_long : Tensor
        ``(N,)`` long-format phenotype values.
    sample_ids : Tensor
        ``(N,)`` integer sample codes.
    env_ids : Tensor
        ``(N,)`` integer environment codes.
    time_values : Tensor
        ``(N,)`` raw time values (same units across environments).
    basis_kind : {"legendre", "bspline"}
    order, n_interior_knots, degree, knot_kind : basis hyper-parameters
        See :func:`torchgwas.linalg.basis.legendre_basis` and
        :func:`place_knots`.

    Returns
    -------
    MultiEnvLongitudinalProjection

    Notes
    -----
    The user **does not** pass ``t_min`` / ``t_max`` / ``knots`` because they
    are derived globally from the pooled time vector and shared across all
    environments.  Per-env time standardization would invalidate every
    cross-environment contrast (tests 5, 6, 7, 8 in
    :class:`RandomRegressionMultiEnvLMM`).
    """
    Y_long = Y_long.to(STAT_DTYPE).reshape(-1)
    time_values = time_values.to(STAT_DTYPE).reshape(-1)
    sample_ids = sample_ids.reshape(-1)
    env_ids = env_ids.reshape(-1)

    if not (Y_long.shape[0] == sample_ids.shape[0] == env_ids.shape[0] == time_values.shape[0]):
        raise ValueError(
            "project_multi_env: Y_long, sample_ids, env_ids, time_values must "
            f"share length. Got {Y_long.shape[0]}, {sample_ids.shape[0]}, "
            f"{env_ids.shape[0]}, {time_values.shape[0]}."
        )

    # --- Build SHARED basis params from POOLED times ---
    if basis_kind == "legendre":
        if order < 0:
            raise ValueError("project_multi_env: order must be >= 0.")
        b = order + 1
        _, t_min, t_max = standardize_time(time_values)
        basis_params: dict = {"order": int(order), "t_min": float(t_min), "t_max": float(t_max)}
    elif basis_kind == "bspline":
        if n_interior_knots < 0:
            raise ValueError("project_multi_env: n_interior_knots must be >= 0.")
        # Pool times across envs to place knots
        knots = place_knots(
            time_values, n_interior=n_interior_knots, kind=knot_kind, degree=degree
        )
        b = int(knots.shape[0]) + degree - 1
        basis_params = {"knots": knots, "degree": int(degree)}
    else:
        raise ValueError(f"project_multi_env: unknown basis_kind '{basis_kind}'.")

    # --- Global sample index across ALL envs ---
    global_sample_index = torch.unique(sample_ids, sorted=True)
    n = int(global_sample_index.shape[0])

    # --- Sorted unique env codes ---
    env_index = torch.unique(env_ids, sorted=True)
    E = int(env_index.shape[0])
    if E < 1:
        raise ValueError("project_multi_env: no environments found.")

    # --- Per-env projection ---
    Y_proj = torch.full(
        (n, b, E), float("nan"), dtype=STAT_DTYPE, device=Y_long.device
    )
    n_obs_pe = torch.zeros(n, E, dtype=torch.int64, device=Y_long.device)
    Phi_list_per_env: list[list[Tensor]] = []
    sample_index_per_env: list[Tensor] = []

    # Map global sample id -> row index for efficient scatter
    global_id_to_row = {int(s.item()): i for i, s in enumerate(global_sample_index)}

    for ie, e_code in enumerate(env_index.tolist()):
        mask_e = env_ids == int(e_code)
        Y_e = Y_long[mask_e]
        ids_e = sample_ids[mask_e]
        t_e = time_values[mask_e]

        Y_wide_e, sample_index_e, n_obs_e, Phi_list_e = _project_one_env_with_params(
            Y_e, ids_e, t_e, basis_kind, basis_params, b
        )
        Phi_list_per_env.append(Phi_list_e)
        sample_index_per_env.append(sample_index_e)

        # Scatter env-local rows into the global (n, b, E) tensor
        for i_local, sid in enumerate(sample_index_e.tolist()):
            row = global_id_to_row[int(sid)]
            Y_proj[row, :, ie] = Y_wide_e[i_local]
            n_obs_pe[row, ie] = int(n_obs_e[i_local].item())

    return MultiEnvLongitudinalProjection(
        Y_proj=Y_proj,
        sample_index=global_sample_index,
        env_index=env_index,
        Phi_list_per_env=Phi_list_per_env,
        sample_index_per_env=sample_index_per_env,
        n_obs_per_indiv_per_env=n_obs_pe,
        basis_kind=basis_kind,
        basis_params=basis_params,
        b=b,
        E=E,
    )


# ── Contrast catalog ────────────────────────────────────────────────


def _diff_matrix(E: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Successive-differences contrast ``D_E`` of shape ``(E-1, E)``."""
    D = torch.zeros(E - 1, E, dtype=dtype, device=device)
    for i in range(E - 1):
        D[i, i] = 1.0
        D[i, i + 1] = -1.0
    return D


def _build_rr_met_contrasts(
    b: int, E: int, device: torch.device, dtype: torch.dtype = STAT_DTYPE
) -> dict[str, Tensor]:
    """Build the 9-test RR-MET contrast catalog.

    Returns a dict mapping test name -> contrast matrix ``C`` shape
    ``(q, b*E)`` in **basis-major** column ordering ``column = k*E + e``,
    matching the ``Y.reshape(n, b*E)`` row-major flatten of
    :class:`MultiEnvLongitudinalProjection`.

    Tests
    -----
    - ``"joint"``        : df = b*E
    - ``"per_env_e"``    : per-env Wald, df = b   (one per env, ``e=0..E-1``)
    - ``"intercept_e"``  : intercept (β_{0,e}=0), df = 1
    - ``"tv_e"``         : time-varying β_{1:,e}=0, df = b-1 (only if b ≥ 2)
    - ``"stable_k"``     : stable across envs at coef k, df = E-1 (one per k)
    - ``"gxe_joint"``    : full curve constant in env, df = b*(E-1)
    - ``"mean_curve"``   : mean curve nonzero, df = b
    - ``"gxe_intercept"``: intercept-only GxE, df = E-1

    Notes on the Kronecker pattern
    ------------------------------
    With column index ``k*E + e``, the row-major flatten of a ``(b, E)`` β
    matrix produces ``vec(β)[k*E+e] = β[k, e]``.  Identifying the ``b``-axis
    as the "outer" Kronecker factor and the ``E``-axis as the "inner",
    contrasts of the form ``A ⊗ B`` (with ``A`` acting on the basis axis and
    ``B`` on the environment axis) act on this row-major vec exactly as
    ``(A ⊗ B) @ vec(β)``.  This is the same convention MT-MET uses for its
    trait-major dE flatten.

    The genuinely-new pieces of math are ``"gxe_joint" = I_b ⊗ D_E`` and
    ``"gxe_intercept" = e_0^T ⊗ D_E``: these test whether the full curve
    (or just the intercept) is constant in env, parameterizing the
    "constant-in-e" complement via the rank ``E-1`` successive-differences
    matrix ``D_E``.
    """
    bE = b * E
    I_b = torch.eye(b, dtype=dtype, device=device)
    I_E = torch.eye(E, dtype=dtype, device=device)

    contrasts: dict[str, Tensor] = {}

    # 1. Joint
    contrasts["joint"] = torch.eye(bE, dtype=dtype, device=device)

    # 2. Per-env e: I_b ⊗ e_e^T   (b, bE)
    for e in range(E):
        e_e = torch.zeros(1, E, dtype=dtype, device=device)
        e_e[0, e] = 1.0
        contrasts[f"per_env_{e}"] = torch.kron(I_b, e_e)

    # 3. Intercept per env e: e_0^T ⊗ e_e^T   (1, bE)
    e_0_b = torch.zeros(1, b, dtype=dtype, device=device)
    e_0_b[0, 0] = 1.0
    for e in range(E):
        e_e = torch.zeros(1, E, dtype=dtype, device=device)
        e_e[0, e] = 1.0
        contrasts[f"intercept_{e}"] = torch.kron(e_0_b, e_e)

    # 4. Time-varying per env e: [0|I_{b-1}] ⊗ e_e^T  (only if b ≥ 2)
    if b >= 2:
        TV = torch.zeros(b - 1, b, dtype=dtype, device=device)
        for i in range(b - 1):
            TV[i, i + 1] = 1.0
        for e in range(E):
            e_e = torch.zeros(1, E, dtype=dtype, device=device)
            e_e[0, e] = 1.0
            contrasts[f"tv_{e}"] = torch.kron(TV, e_e)

    if E >= 2:
        D_E = _diff_matrix(E, device, dtype)

        # 5. Stable per coef k: e_k^T ⊗ D_E    (E-1, bE)
        for k in range(b):
            e_k = torch.zeros(1, b, dtype=dtype, device=device)
            e_k[0, k] = 1.0
            contrasts[f"stable_{k}"] = torch.kron(e_k, D_E)

        # 6. Joint stable-vs-GxE: I_b ⊗ D_E   (b(E-1), bE)
        contrasts["gxe_joint"] = torch.kron(I_b, D_E)

        # 8. Intercept-only GxE: e_0^T ⊗ D_E    (E-1, bE)
        contrasts["gxe_intercept"] = torch.kron(e_0_b, D_E)

    # 7. Mean curve: I_b ⊗ (1_E^T / E)   (b, bE)
    one_over_E = torch.full((1, E), 1.0 / E, dtype=dtype, device=device)
    contrasts["mean_curve"] = torch.kron(I_b, one_over_E)

    return contrasts


# ── RRMetScanResult ─────────────────────────────────────────────────


@dataclass
class RRMetScanResult:
    """Per-variant random-regression × multi-env GWAS results.

    The 9 RR-MET tests live in dictionaries keyed by test name (see
    :func:`_build_rr_met_contrasts`).  ``stat`` and ``p`` alias the
    ``"joint"`` test so the result remains drop-in compatible with the
    standard scan/merge pipeline.
    """

    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    af: Tensor

    beta: Tensor       # (m, b, E) coefficient effects per (basis, env)
    se: Tensor         # (m, b, E)
    Var_beta: Tensor   # (m, bE, bE) full coefficient covariance per SNP

    # Joint test χ²(bE)
    stat_joint: Tensor
    p_joint: Tensor

    # All other contrast tests, keyed by name
    stat_by_test: dict[str, Tensor] = field(default_factory=dict)
    p_by_test: dict[str, Tensor] = field(default_factory=dict)
    df_by_test: dict[str, int] = field(default_factory=dict)

    # Optional per-time-point × per-env reconstruction
    eval_times: Tensor | None = None
    beta_at_t: Tensor | None = None        # (m, n_t, E)
    se_at_t: Tensor | None = None          # (m, n_t, E)
    stat_at_t: Tensor | None = None        # (m, n_t, E)
    p_at_t: Tensor | None = None           # (m, n_t, E)

    test: str = "wald"
    inference_type: str = "marginal"

    # Metadata
    basis_kind: str = "legendre"
    basis_params: dict | None = None
    b: int = 0
    E: int = 0
    env_index: Tensor | None = None

    def __len__(self) -> int:
        return len(self.snp)

    @property
    def stat(self) -> Tensor:
        return self.stat_joint

    @property
    def p(self) -> Tensor:
        return self.p_joint


# ── RandomRegressionMultiEnvLMM ─────────────────────────────────────


class RandomRegressionMultiEnvLMM:
    """Random Regression LMM × Multi-Environment GWAS.

    Composes :class:`MultiTraitMultiEnvLMM` (Phase 25) with the per-(i, e)
    basis projection of :func:`project_multi_env`, giving longitudinal /
    spectral GWAS in multi-environment trials with separable Kronecker
    genetic covariance ``Vg = K_coef ⊗ Vg_env`` by default.

    Parameters
    ----------
    basis : {"legendre", "bspline"}
    order, n_interior_knots, degree, knot_kind : basis hyper-parameters
    vg_structure : {"separable", "unstructured", "fa(k)"}
        Structure of the ``b*E``-dimensional genetic covariance.
        ``"separable"`` (default) parameterizes ``Vg = K_coef ⊗ Vg_env`` and
        scales to large ``E``.  ``"unstructured"`` is full ``(bE, bE)`` and
        only allowed for ``b*E ≤ 12``.  ``"fa(k)"`` is rank-``k`` factor
        analytic on the full ``bE`` space.
    config : NumericalConfig, optional
    """

    _UNSTRUCTURED_GUARD = 12

    def __init__(
        self,
        basis: str = "legendre",
        order: int = 3,
        n_interior_knots: int = 4,
        degree: int = 3,
        knot_kind: str = "quantile",
        vg_structure: str = "separable",
        config: NumericalConfig | None = None,
    ) -> None:
        if basis not in ("legendre", "bspline"):
            raise ValueError(f"RandomRegressionMultiEnvLMM: unknown basis '{basis}'.")
        self.basis = basis
        self.order = order
        self.n_interior_knots = n_interior_knots
        self.degree = degree
        self.knot_kind = knot_kind
        self.vg_structure = vg_structure
        self.config = config or NumericalConfig()

    # ------------------------------------------------------------------
    # fit_null
    # ------------------------------------------------------------------

    def project(
        self,
        Y_long: Tensor,
        sample_ids: Tensor,
        env_ids: Tensor,
        time_values: Tensor,
    ) -> MultiEnvLongitudinalProjection:
        """Run :func:`project_multi_env` with this model's basis configuration."""
        return project_multi_env(
            Y_long, sample_ids, env_ids, time_values,
            basis_kind=self.basis,
            order=self.order,
            n_interior_knots=self.n_interior_knots,
            degree=self.degree,
            knot_kind=self.knot_kind,
        )

    def fit_null(
        self,
        Y_long: Tensor,
        X0: Tensor,
        K: Tensor,
        *,
        sample_ids: Tensor,
        env_ids: Tensor,
        time_values: Tensor,
        env_names: list[str] | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the RR-MET null model.

        Parameters
        ----------
        Y_long : (N,) long-format phenotype values
        X0 : (n, c) per-individual covariates, ordered to match the global
            sample index returned by :func:`project_multi_env` (sorted unique
            sample IDs).
        K : (n, n) GRM ordered to match the same sample index.
        sample_ids : (N,) integer sample codes
        env_ids : (N,) integer environment codes
        time_values : (N,) raw time values
        env_names : optional list of env labels (length E)

        Returns
        -------
        NullFit
            Standard MT-MET null fit with RR-MET metadata attached as
            attributes: ``K_coef``, ``Vg_env``, ``basis_kind``,
            ``basis_params``, ``b``, ``E_envs``, ``env_index``,
            ``Phi_list_per_env``, ``proj_sample_index``, ``is_rr_met``.
        """
        if K is None:
            raise ValueError("RandomRegressionMultiEnvLMM.fit_null requires a kinship matrix K.")

        proj = self.project(Y_long, sample_ids, env_ids, time_values)
        n = int(proj.sample_index.shape[0])
        b, E = proj.b, proj.E

        X0 = X0.to(STAT_DTYPE)
        K = K.to(STAT_DTYPE)

        if X0.dim() != 2 or X0.shape[0] != n:
            raise ValueError(
                f"fit_null: X0 must have shape (n={n}, c); got {tuple(X0.shape)}."
            )
        if K.shape != (n, n):
            raise ValueError(
                f"fit_null: K must have shape ({n}, {n}); got {tuple(K.shape)}."
            )
        if b < 2:
            raise ValueError(
                f"fit_null: basis dimension b={b} < 2. Use MultiEnvLMM directly."
            )
        if E < 2:
            raise ValueError(
                f"fit_null: number of envs E={E} < 2. Use RandomRegressionLMM directly."
            )

        bE = b * E
        if self.vg_structure.lower() == "unstructured" and bE > self._UNSTRUCTURED_GUARD:
            raise ValueError(
                f"fit_null: vg_structure='unstructured' is only allowed for "
                f"b*E ≤ {self._UNSTRUCTURED_GUARD}; got b={b}, E={E}, bE={bE}. "
                "Use 'separable' (default) or 'fa(k)'."
            )

        # Reshape (n, b, E) -> (n, b*E) row-major: column = k*E + e.
        # MT-MET uses the same trait-major convention with d := b traits.
        Y_wide = proj.Y_proj.reshape(n, bE)

        inner = MultiTraitMultiEnvLMM(
            config=self.config, vg_structure=self.vg_structure
        )
        nf = inner.fit_null(
            Y_wide, X0, K,
            n_traits=b, n_envs=E,
            trait_names=[f"coef_{k}" for k in range(b)],
            env_names=env_names,
        )

        # --- Attach RR-MET metadata ---
        nf.is_rr_met = True
        nf.basis_kind = proj.basis_kind
        nf.basis_params = proj.basis_params
        nf.b = b
        nf.E_envs = E
        nf.env_index = proj.env_index
        nf.Phi_list_per_env = proj.Phi_list_per_env
        nf.proj_sample_index = proj.sample_index
        nf.n_obs_per_indiv_per_env = proj.n_obs_per_indiv_per_env
        nf.vg_structure_rr = self.vg_structure

        # K_coef alias: for separable mode this is Vg_trait (the basis-coef
        # block of the Kronecker factorization). For unstructured / fa modes
        # we expose the marginal basis-coef covariance as the average over
        # envs of the diagonal-env blocks of Vg.
        if hasattr(nf, "Vg_trait") and nf.Vg_trait is not None:
            nf.K_coef = nf.Vg_trait
        else:
            # Marginal basis-coef covariance from full (bE, bE) Vg
            Vg = nf.Vg
            K_coef = torch.zeros(b, b, dtype=Vg.dtype, device=Vg.device)
            for k1 in range(b):
                for k2 in range(b):
                    val = 0.0
                    for e in range(E):
                        val += Vg[k1 * E + e, k2 * E + e].item()
                    K_coef[k1, k2] = val / E
            nf.K_coef = K_coef

        if not hasattr(nf, "Vg_env") or nf.Vg_env is None:
            # Marginal env covariance from full Vg
            Vg = nf.Vg
            Vg_env = torch.zeros(E, E, dtype=Vg.dtype, device=Vg.device)
            for e1 in range(E):
                for e2 in range(E):
                    val = 0.0
                    for k in range(b):
                        val += Vg[k * E + e1, k * E + e2].item()
                    Vg_env[e1, e2] = val / b
            nf.Vg_env = Vg_env

        logger.info(
            "RR-MET null fit: n=%d, b=%d, E=%d, basis=%s, vg=%s, ll=%.4f",
            n, b, E, proj.basis_kind, self.vg_structure, nf.log_likelihood or 0.0,
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
        *,
        eval_times: Tensor | None = None,
        **kwargs: Any,
    ) -> RRMetScanResult:
        """Score a genotype chunk with RR-MET Wald tests.

        Reuses MT-MET's :func:`_gls_wald_scan` as the per-SNP backend (so
        ``β`` and ``Var(β)`` are exact GLS estimates), then evaluates the
        9-test contrast catalog from :func:`_build_rr_met_contrasts` plus an
        optional per-time × per-env reconstruction.
        """
        if test != "wald":
            raise ValueError(
                f"RandomRegressionMultiEnvLMM.score_chunk supports 'wald', got '{test}'."
            )
        if not getattr(null_fit, "is_rr_met", False):
            raise ValueError(
                "score_chunk: null_fit is not a RR-MET NullFit (missing is_rr_met)."
            )

        from ..linalg.eigh import rotate

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape
        b = null_fit.b
        E = null_fit.E_envs
        bE = b * E

        device = G_chunk.device
        af = G_chunk.mean(dim=0) / 2.0

        # Rotate G into the kinship eigenbasis
        U = null_fit.eigenvectors
        G_rot = rotate(G_chunk, U)

        # Per-SNP β (m, bE) and Var(β) (m, bE, bE) via MT-MET's GLS Wald
        beta_flat, se_flat, Var_beta, stat_joint = _gls_wald_scan(
            G_rot, null_fit, bE
        )
        p_joint = chi2_sf(stat_joint, df=bE)
        beta = beta_flat.reshape(m, b, E)
        se = se_flat.reshape(m, b, E)

        # 9-test contrast catalog
        contrasts = _build_rr_met_contrasts(b, E, device, dtype=STAT_DTYPE)
        stat_by_test: dict[str, Tensor] = {}
        p_by_test: dict[str, Tensor] = {}
        df_by_test: dict[str, int] = {}
        for name, C in contrasts.items():
            if name == "joint":
                stat_by_test[name] = stat_joint
                p_by_test[name] = p_joint
                df_by_test[name] = bE
                continue
            _, _, stat_c, p_c = apply_contrast(beta_flat, Var_beta, C)
            stat_by_test[name] = stat_c
            p_by_test[name] = p_c
            df_by_test[name] = int(C.shape[0])

        # --- Optional per-time × per-env reconstruction ---
        eval_times_out: Tensor | None = None
        beta_at_t_out: Tensor | None = None
        se_at_t_out: Tensor | None = None
        stat_at_t_out: Tensor | None = None
        p_at_t_out: Tensor | None = None
        if eval_times is not None:
            eval_times_t = torch.as_tensor(
                eval_times, dtype=STAT_DTYPE, device=device
            ).reshape(-1)
            n_t = int(eval_times_t.shape[0])

            Phi_eval = evaluate_basis_at(
                eval_times_t, null_fit.basis_kind, null_fit.basis_params
            ).to(device=device, dtype=STAT_DTYPE)  # (n_t, b)
            if Phi_eval.shape[1] != b:
                raise ValueError(
                    f"score_chunk: basis re-evaluation produced b={Phi_eval.shape[1]}, "
                    f"expected b={b}."
                )

            # β(t, e) = φ(t)^T β[:, :, e]   ->   (m, n_t, E)
            beta_at_t_out = torch.einsum("ti,mie->mte", Phi_eval, beta)

            # Per-(t, e) variance via per-env contrast on the full bE block.
            # For each (t, e): C_te = φ(t)^T ⊗ e_e^T   shape (1, bE)
            # Build all C_te in one tensor: (n_t, E, 1, bE)
            # var_te = C_te @ Var_beta @ C_te^T  →  (m, n_t, E)
            #
            # To stay efficient: reshape Var_beta as (m, b, E, b, E) and
            # contract on basis dims with Phi_eval twice and pick the
            # diagonal env block.  For each SNP m and (t, e):
            #   var = sum_{k1,k2} φ_t[k1] * Var_beta_reshaped[m, k1, e, k2, e] * φ_t[k2]
            # Extract per-env diagonal blocks of Var_beta in basis-coef space.
            # Reshape (m, bE, bE) -> (m, b, E, b, E), then pick e1 == e2 = e.
            # Result indexed as Var_diag_env[m, k1, k2, e] = Var_beta[m, k1*E+e, k2*E+e].
            Var_beta_resh = Var_beta.reshape(m, b, E, b, E)
            e_idx = torch.arange(E, device=device)
            # Fancy indexing on the two env axes (dims 2 and 4) collapses them
            # into a single leading axis of size E.  Result: (E, m, b, b).
            Var_diag_env = Var_beta_resh[:, :, e_idx, :, e_idx]
            Var_diag_env = Var_diag_env.permute(1, 2, 3, 0).contiguous()  # (m, b, b, E)

            # var_at_t[m, t, e] = sum_{k1, k2} Phi[t,k1] * V[m,k1,k2,e] * Phi[t,k2]
            var_at_t = torch.einsum(
                "tk,mkle,tl->mte", Phi_eval, Var_diag_env, Phi_eval
            )
            var_at_t = torch.clamp(var_at_t, min=1e-30)
            se_at_t_out = torch.sqrt(var_at_t)
            stat_at_t_out = torch.clamp(beta_at_t_out ** 2 / var_at_t, min=0.0)
            p_at_t_out = chi2_sf(stat_at_t_out, df=1)
            eval_times_out = eval_times_t

        return RRMetScanResult(
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
            stat_by_test=stat_by_test,
            p_by_test=p_by_test,
            df_by_test=df_by_test,
            eval_times=eval_times_out,
            beta_at_t=beta_at_t_out,
            se_at_t=se_at_t_out,
            stat_at_t=stat_at_t_out,
            p_at_t=p_at_t_out,
            test="wald",
            basis_kind=null_fit.basis_kind,
            basis_params=null_fit.basis_params,
            b=b,
            E=E,
            env_index=null_fit.env_index,
        )

    # ------------------------------------------------------------------
    # Interpretive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _phi_at(null_fit: NullFit, t_query: Tensor) -> Tensor:
        if not getattr(null_fit, "basis_kind", None):
            raise ValueError("null_fit has no basis metadata.")
        t_query = torch.as_tensor(t_query, dtype=STAT_DTYPE).reshape(-1)
        return evaluate_basis_at(
            t_query, null_fit.basis_kind, null_fit.basis_params
        ).to(STAT_DTYPE)

    @classmethod
    def genetic_variance_surface(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Per-environment genetic variance σ²_g(t, e) on the time grid.

        Returns ``(T_query, E)`` tensor.  Computed under the separable
        Kronecker model as ``σ²_g(t, e) = (φ(t)^T K_coef φ(t)) · Vg_env[e, e]``.
        For unstructured / fa modes, expanded directly from the full
        ``(bE, bE)`` Vg.
        """
        Phi = cls._phi_at(null_fit, t_query)  # (T, b)
        b = null_fit.b
        E = null_fit.E_envs

        if hasattr(null_fit, "Vg_trait") and null_fit.Vg_trait is not None:
            K_coef = null_fit.K_coef
            var_g_t = torch.einsum("ti,ij,tj->t", Phi, K_coef, Phi)  # (T,)
            return var_g_t.unsqueeze(1) * null_fit.Vg_env.diag().unsqueeze(0)
        # Unstructured fallback
        Vg_resh = null_fit.Vg.reshape(b, E, b, E)
        # var(t, e) = sum_{k1,k2} Phi[t,k1] * Vg[k1, e, k2, e] * Phi[t,k2]
        e_idx = torch.arange(E, device=null_fit.Vg.device)
        Vg_diag_env = Vg_resh[:, e_idx, :, e_idx]  # (E, b, b)
        Vg_diag_env = Vg_diag_env.permute(1, 2, 0).contiguous()  # (b, b, E)
        return torch.einsum("tk,kle,tl->te", Phi, Vg_diag_env, Phi)

    @classmethod
    def heritability_surface(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Per-(t, e) narrow-sense heritability ``h²(t, e)``.

        ``var_g(t, e) / (var_g(t, e) + var_e(t, e))``, where the residual
        variance surface uses the same per-env diagonal block of ``Ve``.
        """
        var_g = cls.genetic_variance_surface(null_fit, t_query)  # (T, E)
        Phi = cls._phi_at(null_fit, t_query)
        b = null_fit.b
        E = null_fit.E_envs
        Ve = null_fit.Ve  # (bE, bE)
        Ve_resh = Ve.reshape(b, E, b, E)
        e_idx = torch.arange(E, device=Ve.device)
        Ve_diag_env = Ve_resh[:, e_idx, :, e_idx].permute(1, 2, 0).contiguous()  # (b, b, E)
        var_e = torch.einsum("tk,kle,tl->te", Phi, Ve_diag_env, Phi)
        denom = torch.clamp(var_g + var_e, min=1e-30)
        return var_g / denom

    @classmethod
    def genetic_correlation_between_envs_at_time(
        cls, null_fit: NullFit, t_query: Tensor
    ) -> Tensor:
        """Genetic correlation surface ``r_g((t, e1), (t, e2))`` per query t.

        Returns ``(T_query, E, E)``: at each query time the ``E×E`` matrix of
        genetic correlations between environments at that time point.
        """
        Phi = cls._phi_at(null_fit, t_query)  # (T, b)
        b = null_fit.b
        E = null_fit.E_envs
        Vg_resh = null_fit.Vg.reshape(b, E, b, E)
        # Cov(t, e1, e2) = sum_{k1,k2} Phi[t,k1] * Vg[k1,e1,k2,e2] * Phi[t,k2]
        Cov = torch.einsum("tk,kelf,tl->tef", Phi, Vg_resh, Phi)  # (T, E, E)
        diag = torch.diagonal(Cov, dim1=-2, dim2=-1).clamp(min=1e-30)  # (T, E)
        denom = torch.sqrt(diag.unsqueeze(2) * diag.unsqueeze(1))  # (T, E, E)
        return Cov / denom

    @classmethod
    def env_specific_eigenfunctions(
        cls,
        null_fit: NullFit,
        env: int,
        t_query: Tensor,
        n_components: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Functional principal components of the env-``e`` genetic operator.

        Builds the per-env basis-coef block ``K_coef^{(e)} = Vg[k1, e, k2, e]``
        and returns its eigenvalues plus the FPCs evaluated on the query
        time grid.
        """
        Phi = cls._phi_at(null_fit, t_query)  # (T, b)
        b = null_fit.b
        E = null_fit.E_envs
        if env < 0 or env >= E:
            raise ValueError(f"env index {env} out of range [0, {E}).")
        Vg_resh = null_fit.Vg.reshape(b, E, b, E)
        K_e = Vg_resh[:, env, :, env]  # (b, b)
        K_e = 0.5 * (K_e + K_e.T)
        evals, evecs = torch.linalg.eigh(K_e)
        order = torch.argsort(evals, descending=True)
        evals = evals[order]
        evecs = evecs[:, order]
        if n_components is not None:
            evals = evals[:n_components]
            evecs = evecs[:, :n_components]
        return evals, Phi @ evecs
