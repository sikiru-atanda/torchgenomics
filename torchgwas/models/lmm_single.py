"""Single-trait LMM: AI-REML variance components, score/wald/lrt scan."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base import NullFit, ScanResult, VariantMeta


class SingleTraitLMM:
    """Single-trait linear mixed model: y = Xb + Zg + e, g ~ N(0, K*sig2_g).

    Implements :class:`BaseModel`.  GEMMA-equivalent eigendecomposition
    trick: fit null in O(n) per REML iteration after one-time O(n^3) eigh.
    """

    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor | None = None, **kwargs: Any) -> NullFit:
        raise NotImplementedError  # Phase 4

    def score_chunk(self, G_chunk: Tensor, null_fit: NullFit, variant_meta: VariantMeta, test: str = "wald") -> ScanResult:
        raise NotImplementedError  # Phase 4
