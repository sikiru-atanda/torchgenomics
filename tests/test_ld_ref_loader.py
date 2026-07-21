"""Tier 1 unit tests for LD reference loader and metadata."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgenomics.postgwas._ld_ref_loader import (
    LDReferenceUnsupportedFormatError,
    load_ld_reference,
    save_ld_reference,
)
from torchgenomics.postgwas._ld_ref_metadata import (
    LDReferenceMetadata,
    MetadataMismatchError,
    check_metadata_compatibility,
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


from torchgenomics.postgwas._ld_ref_loader import compute_in_sample_ld


def test_in_sample_ld_matches_corrcoef():
    """compute_in_sample_ld matches np.corrcoef(G.T) for a synthetic panel."""
    rng = np.random.default_rng(42)
    n, p = 100, 20
    G = torch.from_numpy(rng.standard_normal((n, p)).astype(np.float64))
    R = compute_in_sample_ld(G)
    R_ref = torch.from_numpy(np.corrcoef(G.numpy(), rowvar=False))
    assert torch.allclose(R, R_ref, atol=1e-10)


def test_in_sample_ld_handles_constant_columns():
    """Constant columns produce NaN correlations; assert documented behavior."""
    G = torch.zeros(50, 5, dtype=torch.float64)
    G[:, 0] = 1.0  # constant column
    G[:, 1] = torch.arange(50, dtype=torch.float64)  # variable
    R = compute_in_sample_ld(G)
    # diagonal is 1 except for constant columns (which become NaN)
    assert torch.isnan(R[0, 0]) or R[0, 0] == 1.0
    # variable column should have R[1,1] = 1.0
    assert torch.isclose(R[1, 1], torch.tensor(1.0, dtype=torch.float64))


def test_in_sample_ld_diagonal_is_one():
    """For a non-constant panel, diagonal of R is exactly 1.0."""
    rng = np.random.default_rng(7)
    G = torch.from_numpy(rng.standard_normal((30, 8)).astype(np.float64))
    R = compute_in_sample_ld(G)
    assert torch.allclose(R.diag(), torch.ones(8, dtype=torch.float64), atol=1e-12)


def test_in_sample_ld_symmetric():
    """R is symmetric to numerical precision."""
    rng = np.random.default_rng(11)
    G = torch.from_numpy(rng.standard_normal((50, 12)).astype(np.float64))
    R = compute_in_sample_ld(G)
    assert torch.allclose(R, R.T, atol=1e-12)


from torchgenomics.postgwas._ld_ref_loader import (
    BlockSpec,
    decompose_into_blocks,
)


def test_blocks_explicit_regions_preserved():
    """When regions are explicitly passed, decompose returns them unchanged."""
    R = torch.eye(100, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(100)]
    explicit = [BlockSpec(start=0, stop=40), BlockSpec(start=40, stop=100)]
    blocks = decompose_into_blocks(R, snp_ids, regions=explicit, max_block_size=5000)
    assert blocks == explicit


def test_blocks_below_threshold_returns_single_block():
    """If p <= max_block_size, return a single block covering all SNPs."""
    R = torch.eye(100, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(100)]
    blocks = decompose_into_blocks(R, snp_ids, regions=None, max_block_size=5000)
    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].stop == 100


def test_blocks_above_threshold_splits():
    """If p > max_block_size, blocks are produced (may auto-detect or fall back)."""
    R = torch.eye(50, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(50)]
    blocks = decompose_into_blocks(R, snp_ids, regions=None, max_block_size=10)
    # Expect at least 5 blocks of ~10 SNPs each
    assert len(blocks) >= 5
    # Blocks tile the entire range without overlap
    assert blocks[0].start == 0
    assert blocks[-1].stop == 50
    for i in range(len(blocks) - 1):
        assert blocks[i].stop == blocks[i + 1].start
