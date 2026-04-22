"""Multi-trait mvLMM: EED rotation, Kronecker covariance, Schur-complement scan."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from .base import NullFit, ScanResult, VariantMeta


class MultiTraitLMM:
    """Multi-trait linear mixed model with Kronecker covariance.

    V = Vg ⊗ K + Ve ⊗ I.  Uses the Efficient Eigen-Decomposition (EED)
    trick to reduce per-SNP cost from O(n^3 d^3) to O(n d^3).

    Implements :class:`BaseModel`.
    """

    def fit_null(self, Y: Tensor, X0: Tensor, K: Tensor | None = None, **kwargs: Any) -> NullFit:
        raise NotImplementedError  # Phase 5

    def score_chunk(self, G_chunk: Tensor, null_fit: NullFit, variant_meta: VariantMeta, test: str = "wald") -> ScanResult:
        raise NotImplementedError  # Phase 5
