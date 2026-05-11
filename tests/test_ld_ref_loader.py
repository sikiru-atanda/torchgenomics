"""Tier 1 unit tests for LD reference loader and metadata."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgwas.postgwas._ld_ref_loader import (
    load_ld_reference,
    save_ld_reference,
    LDReferenceUnsupportedFormatError,
)
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


def test_save_and_load_pt_format(tmp_path):
    """Round-trip a small LD ref through .pt format with metadata."""
    R = torch.eye(10, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(10)]
    meta = LDReferenceMetadata(
        cohort_id="test", n=100, build="GRCh38", panel_provenance="synthetic"
    )
    path = tmp_path / "ld.pt"
    save_ld_reference(path, R, snp_ids, meta)

    R_loaded, snp_ids_loaded, meta_loaded = load_ld_reference(path)
    assert torch.allclose(R, R_loaded)
    assert snp_ids == snp_ids_loaded
    assert meta == meta_loaded


def test_save_and_load_npz_format(tmp_path):
    """Round-trip a small LD ref through .npz format with metadata."""
    R = torch.eye(10, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(10)]
    meta = LDReferenceMetadata(
        cohort_id="test", n=100, build="GRCh38", panel_provenance="synthetic"
    )
    path = tmp_path / "ld.npz"
    save_ld_reference(path, R, snp_ids, meta)

    R_loaded, snp_ids_loaded, meta_loaded = load_ld_reference(path)
    assert torch.allclose(R, R_loaded)
    assert snp_ids == snp_ids_loaded
    assert meta == meta_loaded


def test_load_unsupported_format_raises(tmp_path):
    """Unsupported file extension raises LDReferenceUnsupportedFormatError."""
    path = tmp_path / "ld.bcor"
    path.write_bytes(b"dummy")
    with pytest.raises(LDReferenceUnsupportedFormatError, match="bcor"):
        load_ld_reference(path)


def test_load_format_inferred_from_extension(tmp_path):
    """Format is inferred from file extension; .pt vs .npz handled differently."""
    R = torch.eye(5, dtype=torch.float64)
    snp_ids = ["rs1", "rs2", "rs3", "rs4", "rs5"]
    meta = LDReferenceMetadata(
        cohort_id="t", n=10, build="GRCh38", panel_provenance="p"
    )
    pt_path = tmp_path / "ld.pt"
    npz_path = tmp_path / "ld.npz"
    save_ld_reference(pt_path, R, snp_ids, meta)
    save_ld_reference(npz_path, R, snp_ids, meta)

    R_pt, _, _ = load_ld_reference(pt_path)
    R_npz, _, _ = load_ld_reference(npz_path)
    assert torch.allclose(R_pt, R_npz)
