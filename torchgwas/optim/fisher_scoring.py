"""Mode D alt: Fisher scoring with expected information."""

from __future__ import annotations

from torch import Tensor


def fisher_scoring_reml(
    Y_rot: Tensor,
    X0_rot: Tensor,
    eigenvalues: Tensor,
    *,
    max_iter: int = 100,
) -> tuple[float, float, float]:
    """Fisher scoring using expected information matrix."""
    raise NotImplementedError  # Phase 4
