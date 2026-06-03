"""Phase 1: PLINK BED/PGEN reader tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from torchgenomics.io.plink import PlinkBedReader

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TINY_PREFIX = FIXTURE_DIR / "tiny"


@pytest.fixture
def bed_reader():
    return PlinkBedReader(TINY_PREFIX)


@pytest.fixture
def truth():
    return np.load(FIXTURE_DIR / "tiny_dosage_truth.npy")


def _write_known_bed(tmp_path: Path, codes_per_variant: list[list[int]],
                     a1: str = "A", a2: str = "G") -> Path:
    """Write a tiny BED with explicit 2-bit codes for direct convention checking.

    codes_per_variant[v][i] is the 2-bit code (0,1,2,3) for sample i, variant v.
    Returns the prefix (no extension).
    """
    n_variants = len(codes_per_variant)
    n_samples = len(codes_per_variant[0])
    bytes_per_snp = (n_samples + 3) // 4
    prefix = tmp_path / "bed_check"
    with open(prefix.with_suffix(".bed"), "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        for v in range(n_variants):
            row = np.zeros(bytes_per_snp, dtype=np.uint8)
            for i, code in enumerate(codes_per_variant[v]):
                row[i // 4] |= (code & 0b11) << (2 * (i % 4))
            fh.write(row.tobytes())
    with open(prefix.with_suffix(".bim"), "w") as fh:
        for v in range(n_variants):
            fh.write(f"22\trs{v:04d}\t0\t{(v+1)*100}\t{a1}\t{a2}\n")
    with open(prefix.with_suffix(".fam"), "w") as fh:
        for i in range(n_samples):
            fh.write(f"FAM{i:03d} IND{i:03d} 0 0 0 -9\n")
    return prefix


def test_bed_dosage_counts_a1_per_plink_convention(tmp_path):
    """Dosage decoded from PLINK BED must equal count of A1 (BIM column 5).

    Per PLINK 1.9 spec, the 2-bit codes are:
      00 → homozygous A1   (count_of_A1 = 2)
      01 → missing
      10 → heterozygous    (count_of_A1 = 1)
      11 → homozygous A2   (count_of_A1 = 0)

    PLINK 1.9's --glm and regenie both report BETA on the A1 effect-allele
    convention. TG must match this convention so cross-tool BETA columns
    are sign-aligned. Regression for the sign-flip bug surfaced by NA3
    Track B (raw β Pearson = -1.000 vs regenie before this fix).
    """
    # variant 0: codes 00, 10, 11, 01 → dosage 2, 1, 0, NaN per PLINK
    codes = [[0b00, 0b10, 0b11, 0b01]]
    prefix = _write_known_bed(tmp_path, codes, a1="A", a2="G")
    reader = PlinkBedReader(prefix)
    for G, vmeta in reader.iter_chunks(chunk_size=1):
        observed = G[:, 0]
        # Expected per PLINK convention (count of A1):
        #   sample 0 (00, hom A1) → 2 copies of A1
        #   sample 1 (10, het)    → 1 copy
        #   sample 2 (11, hom A2) → 0 copies of A1
        #   sample 3 (01, miss)   → NaN
        assert observed[0].item() == 2.0, (
            f"PLINK 0b00 (homozygous A1) must decode to dosage 2 "
            f"(2 copies of A1), got {observed[0].item()}"
        )
        assert observed[1].item() == 1.0, (
            f"PLINK 0b10 (heterozygous) must decode to dosage 1, "
            f"got {observed[1].item()}"
        )
        assert observed[2].item() == 0.0, (
            f"PLINK 0b11 (homozygous A2) must decode to dosage 0 "
            f"(0 copies of A1), got {observed[2].item()}"
        )
        assert torch.isnan(observed[3]), (
            f"PLINK 0b01 (missing) must decode to NaN, got {observed[3].item()}"
        )
        # VariantMeta should still report a1='A' (BIM col 5), a2='G' (col 6)
        assert vmeta.a1[0] == "A"
        assert vmeta.a2[0] == "G"
        break


class TestPlinkBedReader:
    """Tests for torchgenomics.io.plink.PlinkBedReader."""

    def test_read_bed_returns_float_tensor(self, bed_reader):
        """Reader produces float64 dosage tensor from .bed file."""
        for G, _ in bed_reader.iter_chunks(chunk_size=bed_reader.n_variants):
            assert G.dtype == torch.float64
            break

    def test_bed_shape_matches_fam_bim(self, bed_reader):
        """Output shape is (n_samples, n_variants) matching .fam and .bim counts."""
        assert bed_reader.n_samples == 10
        assert bed_reader.n_variants == 20
        for G, vmeta in bed_reader.iter_chunks(chunk_size=bed_reader.n_variants):
            assert G.shape == (10, 20)
            assert len(vmeta) == 20

    def test_bed_values_in_dosage_range(self, bed_reader):
        """All dosage values are in {0, 1, 2, NaN}."""
        for G, _ in bed_reader.iter_chunks(chunk_size=bed_reader.n_variants):
            valid = G[~torch.isnan(G)]
            assert torch.all((valid == 0) | (valid == 1) | (valid == 2))

    def test_iter_chunks_yields_correct_sizes(self, bed_reader):
        """iter_chunks yields chunks of the specified size."""
        chunk_size = 7
        sizes = []
        for G, vmeta in bed_reader.iter_chunks(chunk_size=chunk_size):
            sizes.append(G.shape[1])
            assert G.shape[0] == 10
            assert len(vmeta) == G.shape[1]
        # 20 variants / 7 = [7, 7, 6]
        assert sizes == [7, 7, 6]

    def test_missing_genotypes_encoded_as_nan(self, bed_reader, truth):
        """Missing genotypes (0b01 in BED) become NaN in output tensor."""
        for G, _ in bed_reader.iter_chunks(chunk_size=bed_reader.n_variants):
            # truth is (m, n), G is (n, m)
            G_np = G.numpy().T  # -> (m, n)
            # Check specific known missing position
            assert np.isnan(truth[2, 5])
            assert np.isnan(G_np[2, 5])

    def test_dosage_matches_truth(self, bed_reader, truth):
        """Decoded dosage matches ground truth from fixture generation."""
        for G, _ in bed_reader.iter_chunks(chunk_size=bed_reader.n_variants):
            G_np = G.numpy().T  # (m, n)
            # Compare non-NaN values
            mask = ~np.isnan(truth)
            np.testing.assert_array_equal(G_np[mask], truth[mask])

    def test_sample_ids(self, bed_reader):
        """Sample IDs are parsed from .fam IID column."""
        ids = bed_reader.sample_ids
        assert len(ids) == 10
        assert ids[0] == "IND000"
        assert ids[9] == "IND009"

    def test_variant_meta(self, bed_reader):
        """Variant metadata is parsed from .bim file."""
        vmeta = bed_reader.variant_meta
        assert len(vmeta) == 20
        assert vmeta.snp[0] == "rs0000"
        assert vmeta.chr[0] == "1"
        assert vmeta.pos[0] == 1000
        assert vmeta.a1[0] == "A"
        assert vmeta.a2[0] == "G"

    def test_strip_extension(self):
        """Reader works when given path with .bed extension."""
        reader = PlinkBedReader(TINY_PREFIX.with_suffix(".bed"))
        assert reader.n_samples == 10

    def test_missing_file_raises(self):
        """Raises FileNotFoundError for non-existent fileset."""
        with pytest.raises(FileNotFoundError):
            PlinkBedReader(FIXTURE_DIR / "nonexistent")

    def test_duplicate_iids_detected(self, tmp_path):
        """Raises ValueError for duplicate IIDs in .fam."""
        # Create a .fam with duplicate IIDs
        (tmp_path / "dup.fam").write_text("FAM1 IND001 0 0 0 -9\nFAM2 IND001 0 0 0 -9\n")
        (tmp_path / "dup.bim").write_text("1\trs1\t0\t100\tA\tG\n")
        # Create minimal .bed
        (tmp_path / "dup.bed").write_bytes(bytes([0x6C, 0x1B, 0x01, 0x00]))
        with pytest.raises(ValueError, match="Duplicate"):
            PlinkBedReader(tmp_path / "dup")


class TestPlink2PgenReader:
    """Tests for torchgenomics.io.plink2.Plink2PgenReader."""

    def test_read_pgen_returns_float_tensor(self):
        """Reader produces float64 dosage tensor from .pgen file."""
        pytest.skip("Requires pgenlib package")

    def test_pgen_bed_equivalence(self):
        """Same dataset in .bed and .pgen yields identical dosage tensors."""
        pytest.skip("Requires pgenlib package")
