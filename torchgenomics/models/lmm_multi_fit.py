"""mvLMM null fitting via AI-REML / LBFGS-autograd.

Originally landed as Phase 5 stubs; the canonical AI-REML and LBFGS
multi-trait REML routines now live in :mod:`torchgenomics.optim`. These
wrappers expose the canonical-path API ``fit_mvlmm_null_ai_reml`` /
``fit_mvlmm_null_lbfgs`` returning a populated :class:`NullFit` so the
auditor and external callers continue to work.

The wrappers delegate to:

- ``pxem_nr_mvreml`` — Mode A/B PX-EM warm-start + AI-REML Newton-Raphson
- ``lbfgs_reml`` — Mode C LBFGS on autograd REML log-likelihood

Both return ``(Vg, Ve, log_likelihood, trace)``; this module wraps
that 4-tuple plus the precomputed null quantities into a ``NullFit``.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit


def _ensure_2d(Y_rot: Tensor) -> Tensor:
    """Promote 1-D Y_rot to (n, 1) for the multi-trait optimizers."""
    if Y_rot.ndim == 1:
        return Y_rot.unsqueeze(1)
    return Y_rot


def _build_null_fit(
    Vg: Tensor,
    Ve: Tensor,
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    log_likelihood: float,
    trace: list[dict[str, Any]],
    *,
    max_iter: int,
    tol: float,
) -> NullFit:
    """Wrap optimizer outputs into a :class:`NullFit`.

    Computes M00, b0, and per-individual W = (Vg ⊗ d_i + Ve)^{-1} via
    the standard mvLMM null-quantities helper.

    The ``converged`` flag is taken **directly from the optimizer's
    self-reported flag** (stashed by ``pxem_nr_mvreml`` / ``lbfgs_reml``
    in ``trace[-1]["converged"]``) when present, falling back to the
    legacy trace-tail relative-ll heuristic for backward compatibility
    with optimizer outputs that pre-date the post-V1 flag-plumbing
    follow-up. The plumbed flag is more reliable: the trace-tail
    heuristic returns False when the optimizer stops at exactly
    ``max_iter`` even if the optimizer-side convergence test passed on
    the final step.
    """
    from .multi_trait_lmm import mvlmm_null_quantities

    Y_rot = _ensure_2d(Y_rot)
    null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, eigenvalues)

    if trace and "converged" in trace[-1]:
        # Optimizer-plumbed flag (preferred path).
        converged = bool(trace[-1]["converged"])
    else:
        # Legacy trace-tail heuristic (kept for backward compat with any
        # caller that constructs a hand-rolled trace).
        converged = (
            len(trace) > 0
            and len(trace) < max_iter
            and (
                len(trace) < 2
                or abs(trace[-1].get("ll", 0.0) - trace[-2].get("ll", 0.0))
                < tol * max(abs(trace[-1].get("ll", 1.0)), 1.0)
            )
        )

    return NullFit(
        Vg=Vg,
        Ve=Ve,
        eigenvalues=eigenvalues,
        Y_rot=Y_rot,
        X0_rot=X0_rot,
        M00=null_q["M00"],
        b0=null_q["b0"],
        weights=null_q["W"],
        log_likelihood=log_likelihood,
        optimizer_trace=trace,
        converged=converged,
        device=eigenvalues.device,
    )


def fit_mvlmm_null_lbfgs(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    Vg_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
) -> NullFit:
    """Fit Vg, Ve via LBFGS on autograd REML log-likelihood (Cholesky param).

    Parameters
    ----------
    Y_rot : (n, d) or (n,)
        Rotated phenotype matrix in eigenspace U^T y.
    X0_rot : (n, c)
        Rotated covariates.
    eigenvalues : (n,)
        Kinship eigenvalues from ``eigendecompose(K)``.
    max_iter, tol : LBFGS hyper-parameters.
    Vg_init, Ve_init : (d, d) initial covariance matrices, optional.

    Returns
    -------
    NullFit
        With Vg / Ve / Y_rot / X0_rot / eigenvalues / M00 / b0 / weights /
        log_likelihood / optimizer_trace / converged populated.
    """
    from ..optim.lbfgs_reml import lbfgs_reml

    Y_rot_2d = _ensure_2d(Y_rot.to(STAT_DTYPE))
    n_traits = Y_rot_2d.shape[1]
    if n_traits < 2:
        raise ValueError(
            "fit_mvlmm_null_lbfgs requires d >= 2 traits; got "
            f"d = {n_traits}. Use torchgenomics.models.SingleTraitLMM "
            "for single-trait LMM fits."
        )

    Vg, Ve, ll, trace = lbfgs_reml(
        Y_rot_2d,
        X0_rot.to(STAT_DTYPE),
        eigenvalues.to(STAT_DTYPE),
        n_traits=n_traits,
        max_iter=max_iter,
        tol=tol,
        Vg_init=Vg_init,
        Ve_init=Ve_init,
    )

    return _build_null_fit(
        Vg, Ve, Y_rot_2d, X0_rot, eigenvalues, ll, trace,
        max_iter=max_iter, tol=tol,
    )


def fit_mvlmm_null_ai_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
    em_iters: int = 20,
    Vg_init: Tensor | None = None,
    Ve_init: Tensor | None = None,
) -> NullFit:
    """Fit Vg, Ve via PX-EM warm-start → AI-REML Newton-Raphson.

    Parameters
    ----------
    Y_rot : (n, d) or (n,)
        Rotated phenotype matrix.
    X0_rot : (n, c)
        Rotated covariates.
    eigenvalues : (n,)
        Kinship eigenvalues.
    max_iter : int
        Total iteration budget (EM + NR combined).
    tol : float
        Relative convergence tolerance.
    em_iters : int
        Number of PX-EM warm-start iterations before switching to AI-REML.
    Vg_init, Ve_init : (d, d) initial covariance matrices, optional.

    Returns
    -------
    NullFit
    """
    from ..optim.pxem_nr_mvreml import pxem_nr_mvreml

    Y_rot_2d = _ensure_2d(Y_rot.to(STAT_DTYPE))
    n_traits = Y_rot_2d.shape[1]

    Vg, Ve, ll, trace = pxem_nr_mvreml(
        Y_rot_2d,
        X0_rot.to(STAT_DTYPE),
        eigenvalues.to(STAT_DTYPE),
        n_traits=n_traits,
        max_iter=max_iter,
        tol=tol,
        em_iters=em_iters,
        Vg_init=Vg_init,
        Ve_init=Ve_init,
    )

    return _build_null_fit(
        Vg, Ve, Y_rot_2d, X0_rot, eigenvalues, ll, trace,
        max_iter=max_iter, tol=tol,
    )


__all__ = [
    "fit_mvlmm_null_ai_reml",
    "fit_mvlmm_null_lbfgs",
]
