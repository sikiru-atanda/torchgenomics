"""Tier 1 unit tests for LD reference loader and metadata."""
from __future__ import annotations

import pytest

from torchgwas.postgwas._ld_ref_metadata import (
    LDReferenceMetadata,
    check_metadata_compatibility,
    MetadataMismatchError,
)


def test_metadata_dataclass_required_fields():
    """LDReferenceMetadata requires cohort_id, n, build, panel_provenance."""
    meta = LDReferenceMetadata(
        cohort_id="UKB_unrelated_British",
        n=337491,
        build="GRCh38",
        panel_provenance="UKB v3 imputed",
    )
    assert meta.cohort_id == "UKB_unrelated_British"
    assert meta.n == 337491
    assert meta.build == "GRCh38"
    assert meta.panel_provenance == "UKB v3 imputed"


def test_metadata_serialization_round_trip():
    """LDReferenceMetadata serializes to dict and back."""
    original = LDReferenceMetadata(
        cohort_id="MDP",
        n=281,
        build="B73_v4",
        panel_provenance="MDP genotype 281x3093",
    )
    d = original.to_dict()
    assert d == {
        "cohort_id": "MDP",
        "n": 281,
        "build": "B73_v4",
        "panel_provenance": "MDP genotype 281x3093",
        "schema_version": 1,
    }
    restored = LDReferenceMetadata.from_dict(d)
    assert restored == original


def test_metadata_compatible_when_identical():
    """check_metadata_compatibility passes when ref and sumstats match."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "X", "build": "GRCh38"}
    # Should NOT raise and should NOT warn
    check_metadata_compatibility(ref, ss_meta)


def test_metadata_soft_mismatch_warns_on_cohort_id():
    """Soft mismatch (different cohort_id) emits UserWarning."""
    ref = LDReferenceMetadata(cohort_id="UKB", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "FinnGen", "build": "GRCh38"}
    with pytest.warns(UserWarning, match="cohort_id mismatch"):
        check_metadata_compatibility(ref, ss_meta)


def test_metadata_hard_mismatch_raises_on_build():
    """Hard mismatch (different build) raises MetadataMismatchError."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "X", "build": "GRCh37"}
    with pytest.raises(MetadataMismatchError, match="build mismatch"):
        check_metadata_compatibility(ref, ss_meta)


def test_metadata_missing_sumstats_metadata_warns():
    """If sumstats don't carry metadata, emit a UserWarning advising to add it."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    with pytest.warns(UserWarning, match="sumstats has no metadata"):
        check_metadata_compatibility(ref, ss_meta={})
