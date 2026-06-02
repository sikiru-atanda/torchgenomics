"""Phase 1: Format detection, HapMap, Numeric CSV reader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchgenomics.io.detect import detect_format
from torchgenomics.io.hapmap import HapMapReader
from torchgenomics.io.numeric import NumericDosageReader

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestFormatDetection:
    """Tests for torchgenomics.io.detect.detect_format."""

    def test_detect_bed(self):
        """Detects PLINK BED format from .bed file."""
        assert detect_format(FIXTURE_DIR / "tiny.bed") == "bed"

    def test_detect_hapmap(self):
        """Detects HapMap format from header sentinel columns."""
        assert detect_format(FIXTURE_DIR / "tiny.hmp.txt") == "hapmap"

    def test_detect_csv(self):
        """Detects CSV format from .csv extension + numeric data."""
        assert detect_format(FIXTURE_DIR / "tiny_dosage.csv") == "csv"

    def test_detect_unknown_raises(self):
        """Raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            detect_format(FIXTURE_DIR / "nonexistent.xyz")

    def test_detect_unknown_format_raises(self, tmp_path):
        """Raises ValueError for unrecognizable binary file."""
        mystery = tmp_path / "mystery.dat"
        mystery.write_bytes(b"\x00\x00\x00\x00\x00")
        with pytest.raises(ValueError, match="Cannot determine"):
            detect_format(mystery)


class TestHapMapReader:
    """Tests for torchgenomics.io.hapmap.HapMapReader."""

    def test_read_hapmap(self):
        """HapMap reader loads correctly."""
        reader = HapMapReader(FIXTURE_DIR / "tiny.hmp.txt")
        assert reader.n_samples == 10
        assert reader.n_variants == 20

    def test_hapmap_dosage_values(self):
        """HapMap dosage values are in {0, 1, 2, NaN}."""
        reader = HapMapReader(FIXTURE_DIR / "tiny.hmp.txt")
        for G, vmeta in reader.iter_chunks(chunk_size=reader.n_variants):
            assert G.shape == (10, 20)
            assert G.dtype == torch.float64
            valid = G[~torch.isnan(G)]
            assert torch.all((valid == 0) | (valid == 1) | (valid == 2))

    def test_hapmap_sample_ids(self):
        """Sample IDs are extracted from header columns."""
        reader = HapMapReader(FIXTURE_DIR / "tiny.hmp.txt")
        assert reader.sample_ids[0] == "IND000"

    def test_hapmap_variant_meta(self):
        """Variant metadata parsed from rs#/alleles/chrom/pos columns."""
        reader = HapMapReader(FIXTURE_DIR / "tiny.hmp.txt")
        for _, vmeta in reader.iter_chunks(chunk_size=5):
            assert vmeta.snp[0] == "rs0000"
            assert vmeta.chr[0] == "1"
            break


class TestNumericDosageReader:
    """Tests for torchgenomics.io.numeric.NumericDosageReader."""

    def test_read_csv(self):
        """CSV reader loads correctly."""
        reader = NumericDosageReader(
            FIXTURE_DIR / "tiny_dosage.csv",
            map_path=str(FIXTURE_DIR / "tiny_dosage.map"),
        )
        assert reader.n_samples == 10
        assert reader.n_variants == 20

    def test_csv_dosage_shape(self):
        """Output tensor has correct shape."""
        reader = NumericDosageReader(FIXTURE_DIR / "tiny_dosage.csv")
        for G, vmeta in reader.iter_chunks(chunk_size=reader.n_variants):
            assert G.shape == (10, 20)
            assert G.dtype == torch.float64

    def test_csv_with_map_file(self):
        """Map file provides chr/pos metadata."""
        reader = NumericDosageReader(
            FIXTURE_DIR / "tiny_dosage.csv",
            map_path=str(FIXTURE_DIR / "tiny_dosage.map"),
        )
        for _, vmeta in reader.iter_chunks(chunk_size=5):
            assert vmeta.chr[0] == "1"
            assert vmeta.pos[0] == 1000
            break

    def test_csv_auto_discovers_map(self):
        """Map file is auto-discovered from adjacent .map file."""
        reader = NumericDosageReader(FIXTURE_DIR / "tiny_dosage.csv")
        # discover_map_file should find tiny_dosage.map
        for _, vmeta in reader.iter_chunks(chunk_size=5):
            assert vmeta.chr[0] == "1"  # not "0" placeholder
            break


class TestFormatRoundTrip:
    """Format round-trip equivalence (Section 16)."""

    def test_vcf_bed_equivalence(self):
        """Same dataset in VCF and BED yields equivalent dosage tensors."""
        pytest.skip("VCF fixtures not yet created; requires cyvcf2")

    def test_hapmap_bed_dosage_range(self):
        """Both HapMap and BED produce valid {0,1,2,NaN} dosages."""
        from torchgenomics.io.plink import PlinkBedReader
        bed = PlinkBedReader(FIXTURE_DIR / "tiny")
        hmp = HapMapReader(FIXTURE_DIR / "tiny.hmp.txt")

        for G_bed, _ in bed.iter_chunks(chunk_size=bed.n_variants):
            valid_bed = G_bed[~torch.isnan(G_bed)]
            assert torch.all((valid_bed >= 0) & (valid_bed <= 2))

        for G_hmp, _ in hmp.iter_chunks(chunk_size=hmp.n_variants):
            valid_hmp = G_hmp[~torch.isnan(G_hmp)]
            assert torch.all((valid_hmp >= 0) & (valid_hmp <= 2))
