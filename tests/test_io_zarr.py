"""Tests for ZarrReader (Phase 12)."""

from pathlib import Path

import numpy as np
import pytest
import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ZARR_PATH = FIXTURE_DIR / "tiny.zarr"
N_SAMPLES = 10
N_VARIANTS = 20

zarr = pytest.importorskip("zarr")


@pytest.fixture
def reader():
    from torchgwas.io.zarr import ZarrReader
    return ZarrReader(str(ZARR_PATH))


class TestZarrReaderProperties:
    def test_n_samples(self, reader):
        assert reader.n_samples == N_SAMPLES

    def test_n_variants(self, reader):
        assert reader.n_variants == N_VARIANTS

    def test_sample_ids(self, reader):
        ids = reader.sample_ids
        assert len(ids) == N_SAMPLES
        assert ids[0] == "IND000"
        assert ids[-1] == f"IND{N_SAMPLES - 1:03d}"

    def test_variant_meta(self, reader):
        vmeta = reader.variant_meta
        assert len(vmeta.snp) == N_VARIANTS
        assert vmeta.snp[0] == "rs0000"
        assert vmeta.chr[0] in ("1", "2", "3")
        assert vmeta.pos[0] == 1000
        assert vmeta.a1[0] == "A"
        assert vmeta.a2[0] == "G"


class TestZarrReaderChunks:
    def test_full_single_chunk(self, reader):
        """Reading all variants in one chunk."""
        chunks = list(reader.iter_chunks(chunk_size=N_VARIANTS))
        assert len(chunks) == 1
        G, vmeta = chunks[0]
        assert G.shape == (N_SAMPLES, N_VARIANTS)
        assert G.dtype == torch.float64
        assert len(vmeta.snp) == N_VARIANTS

    def test_multiple_chunks(self, reader):
        """Reading in small chunks."""
        chunk_size = 7
        chunks = list(reader.iter_chunks(chunk_size=chunk_size))
        expected_n_chunks = (N_VARIANTS + chunk_size - 1) // chunk_size
        assert len(chunks) == expected_n_chunks

        # First chunk has full size
        assert chunks[0][0].shape == (N_SAMPLES, chunk_size)
        # Last chunk may be smaller
        last_expected = N_VARIANTS - chunk_size * (expected_n_chunks - 1)
        assert chunks[-1][0].shape == (N_SAMPLES, last_expected)

    def test_chunk_dosage_range(self, reader):
        """Dosage values should be in [0, 2]."""
        for G, _ in reader.iter_chunks(chunk_size=10):
            assert G.min() >= 0.0
            assert G.max() <= 2.0

    def test_total_variants_across_chunks(self, reader):
        """Total variants from all chunks equals n_variants."""
        total = sum(G.shape[1] for G, _ in reader.iter_chunks(chunk_size=8))
        assert total == N_VARIANTS

    def test_repeated_iteration(self, reader):
        """iter_chunks can be called multiple times (fresh iterator each time)."""
        chunks1 = list(reader.iter_chunks(chunk_size=10))
        chunks2 = list(reader.iter_chunks(chunk_size=10))
        assert len(chunks1) == len(chunks2)
        for (G1, _), (G2, _) in zip(chunks1, chunks2):
            assert torch.equal(G1, G2)


class TestZarrReaderMatchesPlink:
    def test_dosage_matches_truth(self, reader):
        """Zarr dosage should match the PLINK ground truth (NaN → 0)."""
        truth = np.load(FIXTURE_DIR / "tiny_dosage_truth.npy")  # (n_var, n_samp)
        truth_samp_major = np.nan_to_num(truth.T, nan=0.0)  # (n_samp, n_var)

        G, _ = next(iter(reader.iter_chunks(chunk_size=N_VARIANTS)))
        np.testing.assert_allclose(G.numpy(), truth_samp_major, atol=1e-6)
