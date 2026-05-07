"""Mode D alt: Fisher scoring with expected information."""

from __future__ import annotations

from torch import Tensor

from .ai_reml import ai_reml_single


def fisher_scoring_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float, float]:
    """Fit single-trait REML with the Fisher-scoring/AI-REML path.

    The single-trait implementation is the same average-information REML
    optimizer exposed as :func:`ai_reml_single`; this wrapper preserves the
    older Mode-D import path while returning the historical three-scalar
    result.
    """
    sig2_g, sig2_e, ll, _trace = ai_reml_single(
        Y_rot,
        X0_rot,
        eigenvalues,
        max_iter=max_iter,
        tol=tol,
    )
    return sig2_g, sig2_e, ll
