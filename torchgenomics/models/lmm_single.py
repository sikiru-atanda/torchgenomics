"""Canonical-path re-export for SingleTraitLMM.

Phase 4 originally landed the implementation under
:mod:`torchgenomics.models.single_trait_lmm`; this module preserves the
``torchgenomics.models.lmm_single`` import path used by the validation
auditor and external callers. It re-exports the real class so the
defining module of ``SingleTraitLMM`` continues to be
``single_trait_lmm`` (preserving ``__module__`` for downstream
introspection and golden tests).
"""

from __future__ import annotations

from .base import NullFit, ScanResult, VariantMeta  # noqa: F401
from .single_trait_lmm import SingleTraitLMM  # noqa: F401

__all__ = [
    "SingleTraitLMM",
    "NullFit",
    "ScanResult",
    "VariantMeta",
]
