"""P-value calibration checks versus reference tools (GEMMA, GAPIT)."""

from __future__ import annotations

from torch import Tensor


def compare_pvalues(ours: Tensor, reference: Tensor, tolerance: float = 1e-4) -> dict:
    """Compare p-values against a reference tool. Returns agreement metrics."""
    raise NotImplementedError  # Phase 4
