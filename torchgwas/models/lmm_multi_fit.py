"""mvLMM null fitting: LBFGS-autograd on Cholesky-parameterized REML + AI-REML."""

from __future__ import annotations

import torch
from torch import Tensor

from ..optim.lbfgs_reml import lbfgs_reml
from ..optim.mvlmm_reml import mvlmm_null_quantities
from ..optim.pxem_nr_mvreml import pxem_nr_mvreml
from .base import NullFit


def _nullfit_from_rotated(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    Vg: Tensor,
    Ve: Tensor,
    ll: float,
    trace: list[dict],
) -> NullFit:
    null_q = mvlmm_null_quantities(Vg, Ve, Y_rot, X0_rot, eigenvalues)
    converged = len(trace) > 0 and (
        len(trace) < 100 or
        (
            len(trace) >= 2
            and abs(trace[-1].get("ll", ll) - trace[-2].get("ll", ll))
            < 1e-6 * max(abs(trace[-1].get("ll", ll)), 1.0)
        )
    )
    return NullFit(
        Vg=Vg,
        Ve=Ve,
        eigenvalues=eigenvalues,
        eigenvectors=None,
        Y_rot=Y_rot,
        X0_rot=X0_rot,
        M00=null_q["M00"],
        b0=null_q["b0"],
        weights=null_q["W"],
        log_likelihood=ll,
        optimizer_trace=trace,
        converged=converged,
        device=eigenvalues.device,
    )


def fit_mvlmm_null_lbfgs(Y_rot: Tensor, X0_rot: Tensor, eigenvalues: Tensor) -> NullFit:
    """Fit Vg, Ve via LBFGS on autograd REML log-likelihood (Cholesky param)."""
    if Y_rot.ndim != 2 or Y_rot.shape[1] < 2:
        raise ValueError("fit_mvlmm_null_lbfgs expects Y_rot with shape (n, d), d >= 2.")
    n_traits = int(Y_rot.shape[1])
    Vg, Ve, ll, trace = lbfgs_reml(Y_rot, X0_rot, eigenvalues, n_traits=n_traits)
    return _nullfit_from_rotated(Y_rot, X0_rot, eigenvalues, Vg, Ve, ll, trace)


def fit_mvlmm_null_ai_reml(Y_rot: Tensor, X0_rot: Tensor, eigenvalues: Tensor) -> NullFit:
    """Fit Vg, Ve via the implemented PX-EM/Newton multi-trait REML path."""
    if Y_rot.ndim != 2 or Y_rot.shape[1] < 2:
        raise ValueError("fit_mvlmm_null_ai_reml expects Y_rot with shape (n, d), d >= 2.")
    n_traits = int(Y_rot.shape[1])
    R = Y_rot - X0_rot @ torch.linalg.lstsq(X0_rot, Y_rot).solution
    S_cov = (R.T @ R) / max(Y_rot.shape[0] - X0_rot.shape[1], 1)
    S_cov = S_cov + torch.eye(n_traits, dtype=Y_rot.dtype, device=Y_rot.device) * 1e-6
    Vg, Ve, ll, trace = pxem_nr_mvreml(
        Y_rot,
        X0_rot,
        eigenvalues,
        n_traits=n_traits,
        Vg_init=S_cov * 0.5,
        Ve_init=S_cov * 0.5,
    )
    return _nullfit_from_rotated(Y_rot, X0_rot, eigenvalues, Vg, Ve, ll, trace)
