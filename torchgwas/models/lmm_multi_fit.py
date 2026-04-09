"""mvLMM null fitting: LBFGS-autograd on Cholesky-parameterized REML + AI-REML."""

from __future__ import annotations

from torch import Tensor

from .base import NullFit


def fit_mvlmm_null_lbfgs(Y_rot: Tensor, X0_rot: Tensor, eigenvalues: Tensor) -> NullFit:
    """Fit Vg, Ve via LBFGS on autograd REML log-likelihood (Cholesky param)."""
    raise NotImplementedError  # Phase 5


def fit_mvlmm_null_ai_reml(Y_rot: Tensor, X0_rot: Tensor, eigenvalues: Tensor) -> NullFit:
    """Fit Vg, Ve via AI-REML with damped Newton updates."""
    raise NotImplementedError  # Phase 5
