"""Legacy import tests for single-trait LMM compatibility."""

from __future__ import annotations

from torchgenomics.models.lmm_single import SingleTraitLMM as LegacySingleTraitLMM
from torchgenomics.models.single_trait_lmm import SingleTraitLMM


def test_legacy_single_trait_lmm_import_is_shim():
    assert LegacySingleTraitLMM is SingleTraitLMM


def test_legacy_single_trait_lmm_instantiates():
    model = LegacySingleTraitLMM()
    assert hasattr(model, "fit_null")
    assert hasattr(model, "score_chunk")
