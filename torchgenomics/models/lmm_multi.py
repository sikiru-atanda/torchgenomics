"""Canonical-path re-export for MultiTraitLMM.

Phase 5 originally landed the implementation under
:mod:`torchgenomics.models.multi_trait_lmm`; this module preserves the
``torchgenomics.models.lmm_multi`` import path used by the validation
auditor and external callers. It re-exports the real class so the
defining module of ``MultiTraitLMM`` continues to be
``multi_trait_lmm``.
"""

from __future__ import annotations

from .base import NullFit, ScanResult, VariantMeta  # noqa: F401
from .multi_trait_lmm import MultiTraitLMM  # noqa: F401

__all__ = [
    "MultiTraitLMM",
    "NullFit",
    "ScanResult",
    "VariantMeta",
]
