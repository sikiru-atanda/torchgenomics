"""Tests for HDF5Reader (Phase 12)."""

import pytest
import numpy as np
import torch
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
HDF5_PATH = FIXTURE_DIR / "tiny.h5"
N_SAMPLES = 10
N_VARIANTS = 20

h5py = pytest.importorskip("h5py")


@pytest.fixture
def reader():
    from torchgwas.io.hdf5 import HDF5Reader
    return HDF5Reader(str(HDF5_PATH))


class TestHDF5ReaderProperties:
    def test_n_samples(self, reader):
        assert reader.n_samples == N_SAMPLES

    def test_n_variants(self, reader):
        assert reader.n_variants == N_VARIANTS

    def test_sample_ids(self, reader):
        ids = reader.sample_ids
        assert len(ids) == N_SAMPLES
        assert ids[0] == "IND000"

    def test_variant_meta(self, reader):
        vmeta = reader.variant_meta
        assert len(vmeta.snp) == N_VARIANTS
        assert vmeta.snp[0] == "rs0000"
        assert vmeta.pos[0] == 1000


class TestHDF5ReaderChunks:
    def test_full_single_chunk(self, reader):
        chunks = list(reader.iter_chunks(chunk_size=N_VARIANTS))
        assert len(chunks) == 1
        G, vmeta = chunks[0]
        assert G.shape == (N_SAMPLES, N_VARIANTS)
        assert G.dtype == torch.float64

    def test_multiple_chunks(self, reader):
        chunk_size = 7
        chunks = list(reader.iter_chunks(chunk_size=chunk_size))
        expected_n_chunks = (N_VARIANTS + chunk_size - 1) // chunk_size
        assert len(chunks) == expected_n_chunks

    def test_chunk_dosage_range(self, reader):
        for G, _ in reader.iter_chunks(chunk_size=10):
            assert G.min() >= 0.0
            assert G.max() <= 2.0

    def test_repeated_iteration(self, reader):
        chunks1 = list(reader.iter_chunks(chunk_size=10))
        chunks2 = list(reader.iter_chunks(chunk_size=10))
        for (G1, _), (G2, _) in zip(chunks1, chunks2):
            assert torch.equal(G1, G2)


class TestHDF5ReaderMatchesPlink:
    def test_dosage_matches_truth(self, reader):
        truth = np.load(FIXTURE_DIR / "tiny_dosage_truth.npy")
        truth_samp_major = np.nan_to_num(truth.T, nan=0.0)

        G, _ = next(iter(reader.iter_chunks(chunk_size=N_VARIANTS)))
        np.testing.assert_allclose(G.numpy(), truth_samp_major, atol=1e-6)


class TestHDF5ReaderCleanup:
    def test_close(self, reader):
        reader.close()
        # Should not raise on second close
        reader.close()
