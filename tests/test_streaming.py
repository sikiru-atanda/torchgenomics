"""Integration tests for Phase 12 streaming pipeline."""

import pytest
import numpy as np
import torch
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures"
N_SAMPLES = 10
N_VARIANTS = 20

zarr = pytest.importorskip("zarr")


class TestSampleAlignedReader:
    def test_reindexing(self):
        """SampleAlignedReader correctly reindexes rows."""
        from torchgwas.io.zarr import ZarrReader
        from torchgwas.io.aligned import SampleAlignedReader

        inner = ZarrReader(str(FIXTURE_DIR / "tiny.zarr"))

        # Select a subset and reverse order
        keep = [5, 3, 1, 0]
        aligned_ids = [inner.sample_ids[i] for i in keep]
        aligned = SampleAlignedReader(inner, keep, aligned_ids)

        assert aligned.n_samples == 4
        assert aligned.n_variants == N_VARIANTS
        assert aligned.sample_ids == aligned_ids

        # Compare chunk data
        G_full, _ = next(iter(inner.iter_chunks(chunk_size=N_VARIANTS)))
        G_aligned, _ = next(iter(aligned.iter_chunks(chunk_size=N_VARIANTS)))

        assert G_aligned.shape == (4, N_VARIANTS)
        for out_idx, in_idx in enumerate(keep):
            torch.testing.assert_close(G_aligned[out_idx], G_full[in_idx])

    def test_multiple_iterations(self):
        """SampleAlignedReader supports repeated iter_chunks calls."""
        from torchgwas.io.zarr import ZarrReader
        from torchgwas.io.aligned import SampleAlignedReader

        inner = ZarrReader(str(FIXTURE_DIR / "tiny.zarr"))
        aligned = SampleAlignedReader(inner, list(range(5)), inner.sample_ids[:5])

        chunks1 = list(aligned.iter_chunks(chunk_size=10))
        chunks2 = list(aligned.iter_chunks(chunk_size=10))
        assert len(chunks1) == len(chunks2)
        for (G1, _), (G2, _) in zip(chunks1, chunks2):
            torch.testing.assert_close(G1, G2)


class TestStreamingGRM:
    def test_streaming_matches_full(self):
        """Streaming VanRaden GRM matches full-materialized VanRaden GRM."""
        from torchgwas.io.zarr import ZarrReader
        from torchgwas.linalg.kinship import grm_vanraden, grm_vanraden_streaming

        reader = ZarrReader(str(FIXTURE_DIR / "tiny.zarr"))

        # Full GRM (materialize)
        G_full, _ = next(iter(reader.iter_chunks(chunk_size=N_VARIANTS)))
        K_full, meta_full = grm_vanraden(G_full)

        # Streaming GRM (chunk by chunk, small chunks to test accumulation)
        K_stream, meta_stream = grm_vanraden_streaming(
            reader.iter_chunks(chunk_size=5),
            n_samples=reader.n_samples,
        )

        # Streaming uses FP32 GEMM + FP64 accumulator, so ~1e-7 tolerance
        torch.testing.assert_close(K_stream, K_full, atol=1e-6, rtol=1e-5)
        assert meta_stream.n_snps_used == meta_full.n_snps_used
        assert abs(meta_stream.normalizer - meta_full.normalizer) < 1e-10

    def test_streaming_grm_from_hdf5(self):
        """Streaming GRM from HDF5 reader produces valid output."""
        h5py = pytest.importorskip("h5py")
        from torchgwas.io.hdf5 import HDF5Reader
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        reader = HDF5Reader(str(FIXTURE_DIR / "tiny.h5"))
        K, meta = grm_vanraden_streaming(
            reader.iter_chunks(chunk_size=7),
            n_samples=reader.n_samples,
        )

        assert K.shape == (N_SAMPLES, N_SAMPLES)
        assert K.dtype == torch.float64
        # GRM should be symmetric
        torch.testing.assert_close(K, K.T)
        # Diagonal should be positive
        assert K.diag().min() > 0


class TestPCGSolver:
    def test_solves_spd_system(self):
        """PCG solves Ax=b for a known SPD system."""
        from torchgwas.optim.pcg_solver import pcg_solve

        n = 50
        torch.manual_seed(42)
        L = torch.randn(n, n, dtype=torch.float64)
        A = L @ L.T + torch.eye(n, dtype=torch.float64)  # SPD
        b = torch.randn(n, dtype=torch.float64)
        x_true = torch.linalg.solve(A, b)

        result = pcg_solve(lambda x: A @ x, b, tol=1e-10)
        assert result.converged
        torch.testing.assert_close(result.x, x_true, atol=1e-6, rtol=1e-6)

    def test_with_preconditioner(self):
        """PCG with diagonal preconditioner converges faster."""
        from torchgwas.optim.pcg_solver import pcg_solve, diagonal_preconditioner

        n = 100
        torch.manual_seed(42)
        # Ill-conditioned system
        diag = torch.logspace(-2, 2, n, dtype=torch.float64)
        A = torch.diag(diag)
        b = torch.randn(n, dtype=torch.float64)

        # Without preconditioner
        r1 = pcg_solve(lambda x: A @ x, b, tol=1e-10)

        # With diagonal preconditioner
        precond = diagonal_preconditioner(diag)
        r2 = pcg_solve(lambda x: A @ x, b, precond=precond, tol=1e-10)

        assert r2.converged
        # Preconditioner should converge in 1 iteration for diagonal system
        assert r2.n_iters <= 2

    def test_zero_rhs(self):
        """PCG with zero RHS returns zero."""
        from torchgwas.optim.pcg_solver import pcg_solve

        b = torch.zeros(10, dtype=torch.float64)
        result = pcg_solve(lambda x: x, b)
        assert result.converged
        assert result.x.norm() == 0.0


class TestPrefetchIterator:
    def test_sync_passthrough(self):
        """PrefetchIterator passes through unchanged on CPU."""
        from torchgwas.io.zarr import ZarrReader
        from torchgwas.scan.prefetch import PrefetchIterator

        reader = ZarrReader(str(FIXTURE_DIR / "tiny.zarr"))
        raw_chunks = list(reader.iter_chunks(chunk_size=10))

        prefetched = list(PrefetchIterator(
            reader.iter_chunks(chunk_size=10),
            device=torch.device("cpu"),
        ))

        assert len(prefetched) == len(raw_chunks)
        for (G1, v1), (G2, v2) in zip(raw_chunks, prefetched):
            torch.testing.assert_close(G1, G2)
            assert v1.snp == v2.snp
