"""Legacy import tests for multi-trait LMM compatibility."""

from __future__ import annotations

from torchgwas.models.lmm_multi import MultiTraitLMM as LegacyMultiTraitLMM
from torchgwas.models.multi_trait_lmm import MultiTraitLMM


def test_legacy_multi_trait_lmm_import_is_shim():
    assert LegacyMultiTraitLMM is MultiTraitLMM


def test_legacy_multi_trait_lmm_instantiates():
    model = LegacyMultiTraitLMM()
    assert hasattr(model, "fit_null")
    assert hasattr(model, "score_chunk")
