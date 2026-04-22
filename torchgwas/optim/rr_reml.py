"""Random Regression REML wrappers (Phase 38, Step 6).

Thin shims that adapt TorchGWAS's existing REML routines to the random
regression use case where the "traits" are basis-coefficient channels and
``K_coef = Vg`` plays the role of genetic covariance on those channels.

Three structures are supported, mirroring the
``RandomRegressionLMM(k_coef_structure=...)`` argument:

- ``"unstructured"`` — full ``(b, b)`` Vg via :func:`lbfgs_reml`. Default
  for ``b ≤ 5``. This is what ``MultiTraitLMM.fit_null`` already uses
  internally; the wrapper exists so RR-LMM can dispatch identically.
- ``"diagonal"`` — assumes both ``Vg`` and ``Ve`` are diagonal in the basis
  basis (no cross-coefficient pleiotropy). Under this constraint each
  coefficient channel decouples, so REML factorizes into ``b`` independent
  univariate fits via :func:`emma_reml_single`. Fast option for large ``b``.
- ``"fa(k)"`` — reduced-rank Factor Analytic structure ``Vg = Λ Λ' + diag(ψ)``
  via :func:`fa_lbfgs_reml`. Used when ``b > 5``.

All three return the same tuple shape ``(Vg, Ve, ll, trace)`` so the caller
can plug them into :func:`mvlmm_null_quantities` uniformly.
"""

from __future__ import annotations

import logging
import re

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .emma_reml import emma_reml_single
from .fa_lbfgs_reml import fa_lbfgs_reml
from .lbfgs_reml import lbfgs_reml

logger = logging.getLogger(__name__)


def parse_k_coef_structure(spec: str) -> tuple[str, int | None]:
    """Parse a ``k_coef_structure`` string into ``(kind, fa_rank)``.

    Examples
    --------
    >>> parse_k_coef_structure("unstructured")
    ('unstructured', None)
    >>> parse_k_coef_structure("diagonal")
    ('diagonal', None)
    >>> parse_k_coef_structure("fa(2)")
    ('fa', 2)
    """
    if spec in ("unstructured", "diagonal"):
        return spec, None
    m = re.fullmatch(r"fa\((\d+)\)", spec)
    if m is not None:
        return "fa", int(m.group(1))
    raise ValueError(f"parse_k_coef_structure: unknown spec '{spec}'.")


def rr_reml_unstructured(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    b: int,
    Vg_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
    max_iter: int = 150,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor, float, list[dict]]:
    """Unstructured K_coef REML — direct passthrough to LBFGS multi-trait REML.

    Parameters mirror :func:`lbfgs_reml`. ``b`` is the number of basis
    coefficients (= the number of pseudo-traits).
    """
    return lbfgs_reml(
        Y_rot, X0_rot, eigenvalues,
        n_traits=b,
        Vg_init=Vg_init,
        Ve_init=Ve_init,
        max_iter=max_iter,
        tol=tol,
    )


def rr_reml_diagonal(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    b: int,
    n_grid: int = 100,
    log_delta_min: float = -10.0,
    log_delta_max: float = 10.0,
) -> tuple[Tensor, Tensor, float, list[dict]]:
    """Diagonal K_coef REML — ``b`` independent univariate fits.

    Under a diagonal Vg / Ve assumption each basis-coefficient channel is
    statistically independent of the others, so REML factorizes:

        ll(Vg, Ve) = Σ_k ll_univariate(σ²_g_k, σ²_e_k)

    The univariate solver is :func:`emma_reml_single` (GAPIT-style EMMA grid
    + Brent refine), called once per channel. The returned ``Vg`` and ``Ve``
    are diagonal ``(b, b)`` matrices.

    Parameters
    ----------
    Y_rot : (n, b) — rotated coefficient matrix.
    X0_rot : (n, c) — rotated covariates.
    eigenvalues : (n,) — GRM eigenvalues from spectral decomposition.
    b : int — number of basis coefficients.
    n_grid, log_delta_min, log_delta_max : grid search params for EMMA.

    Returns
    -------
    Vg : (b, b) diagonal genetic covariance.
    Ve : (b, b) diagonal residual covariance.
    loglik : sum of per-channel REML log-likelihoods.
    trace : per-channel optimizer trace dicts.
    """
    if Y_rot.ndim != 2 or Y_rot.shape[1] != b:
        raise ValueError(
            f"rr_reml_diagonal: expected Y_rot shape (n, {b}); got {tuple(Y_rot.shape)}."
        )
    device = Y_rot.device
    dtype = Y_rot.dtype if Y_rot.dtype.is_floating_point else STAT_DTYPE

    sig2_g = torch.zeros(b, dtype=dtype, device=device)
    sig2_e = torch.zeros(b, dtype=dtype, device=device)
    total_ll = 0.0
    trace: list[dict] = []

    for k in range(b):
        sg_k, se_k, ll_k, tr_k = emma_reml_single(
            Y_rot[:, k], X0_rot, eigenvalues,
            n_grid=n_grid,
            log_delta_min=log_delta_min,
            log_delta_max=log_delta_max,
        )
        sig2_g[k] = float(sg_k)
        sig2_e[k] = float(se_k)
        total_ll += float(ll_k)
        # Annotate trace entries with the channel index for downstream logging
        for entry in tr_k:
            entry = dict(entry)
            entry["channel"] = k
            trace.append(entry)

    Vg = torch.diag(sig2_g)
    Ve = torch.diag(sig2_e)
    logger.info("rr_reml_diagonal: b=%d, total_ll=%.4f", b, total_ll)
    return Vg, Ve, total_ll, trace


def rr_reml_fa(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    b: int,
    fa_rank: int,
    Vg_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
    max_iter: int = 150,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor, float, list[dict]]:
    """Factor analytic K_coef REML — ``Vg = Λ Λ' + diag(ψ)``.

    Wraps :func:`fa_lbfgs_reml` and discards the extra ``Lambda``, ``psi``
    return values to match the ``(Vg, Ve, ll, trace)`` quad expected by the
    RR-LMM call site.

    Raises
    ------
    ValueError
        If ``fa_rank >= b``. The factor-analytic structure must be strictly
        reduced-rank to be identifiable; for full-rank use ``"unstructured"``.
    """
    if fa_rank >= b:
        raise ValueError(
            f"rr_reml_fa: fa_rank ({fa_rank}) must be strictly less than b ({b}). "
            "Use rr_reml_unstructured for full-rank K_coef."
        )
    Vg, Ve, ll, trace, _Lambda, _psi = fa_lbfgs_reml(
        Y_rot, X0_rot, eigenvalues,
        n_envs=b,
        fa_rank=fa_rank,
        Vg_init=Vg_init,
        Ve_init=Ve_init,
        max_iter=max_iter,
        tol=tol,
    )
    return Vg, Ve, ll, trace
