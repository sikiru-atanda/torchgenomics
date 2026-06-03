"""Tests for the dedicated GPU kernels in
``torchgenomics.preprocess._impute_gpu``.

The kernels are written entirely in vectorized torch, so they run on any
device — including CPU. This file deliberately exercises them on CPU
tensors so the GPU code paths are validated even on machines without a
CUDA GPU. Whenever a CUDA device *is* present we additionally run the
same fixtures on CUDA and assert that CUDA and CPU produce identical
results, which proves the kernels are device-independent.
"""

from __future__ import annotations

import os

import pytest
import torch

from torchgenomics.preprocess._impute_gpu import (
    impute_knn_gpu,
    impute_ld_gpu,
    impute_mode_gpu,
)
from torchgenomics.preprocess.impute import (
    impute_knn,
    impute_ld,
    impute_mode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_reference_mode(G):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return impute_mode(G)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _python_reference_knn(G, K, k):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return impute_knn(G, K, k=k)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _python_reference_ld(G, w):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return impute_ld(G, window_size=w)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# impute_mode_gpu — semantic correctness on CPU tensors
# ---------------------------------------------------------------------------


def test_mode_gpu_matches_python_diploid():
    torch.manual_seed(7)
    G = torch.randint(0, 3, (200, 50), dtype=torch.float64)
    G[torch.rand(G.shape) < 0.1] = float("nan")

    mask = torch.isnan(G)
    G_int = torch.where(
        mask,
        torch.tensor(-1, dtype=torch.long),
        torch.round(G).long(),
    )
    max_dosage = int(G_int[G_int >= 0].max().item())

    modes_gpu = impute_mode_gpu(G_int, max_dosage, G.shape[1])

    out_p = _python_reference_mode(G.clone())
    out_g = G.clone()
    out_g[mask] = modes_gpu.unsqueeze(0).expand_as(G).to(G.dtype)[mask]
    assert torch.allclose(out_g, out_p)


def test_mode_gpu_matches_python_polyploid():
    torch.manual_seed(13)
    G = torch.randint(0, 5, (150, 30), dtype=torch.float64)
    G[torch.rand(G.shape) < 0.05] = float("nan")

    mask = torch.isnan(G)
    G_int = torch.where(
        mask, torch.tensor(-1, dtype=torch.long), torch.round(G).long()
    )
    max_dosage = int(G_int[G_int >= 0].max().item())

    modes_gpu = impute_mode_gpu(G_int, max_dosage, G.shape[1])

    out_p = _python_reference_mode(G.clone())
    out_g = G.clone()
    out_g[mask] = modes_gpu.unsqueeze(0).expand_as(G).to(G.dtype)[mask]
    assert torch.allclose(out_g, out_p)


def test_mode_gpu_all_missing_column_returns_zero():
    G_int = torch.full((4, 3), -1, dtype=torch.long)
    G_int[:, 1] = torch.tensor([0, 1, 1, 2])  # second col has data
    modes = impute_mode_gpu(G_int, max_dosage=2, n_cols=3)
    assert modes.tolist() == [0.0, 1.0, 0.0]


def test_mode_gpu_first_wins_tie_break():
    # Counts tied between bins 0 and 1; argmax must pick 0.
    G_int = torch.tensor([[0, 0, 1, 1]], dtype=torch.long).T  # (4, 1)
    modes = impute_mode_gpu(G_int, max_dosage=1, n_cols=1)
    assert modes.tolist() == [0.0]


# ---------------------------------------------------------------------------
# impute_knn_gpu — semantic correctness on CPU tensors
# ---------------------------------------------------------------------------


def _make_K(G_complete: torch.Tensor) -> torch.Tensor:
    Gc = G_complete - G_complete.mean(dim=0, keepdim=True)
    K = (Gc @ Gc.T) / Gc.shape[1]
    return K


def test_knn_gpu_matches_python_basic():
    torch.manual_seed(11)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.1] = float("nan")

    K_mod = K.clone()
    K_mod.fill_diagonal_(-float("inf"))
    _, knn_idx = torch.topk(K_mod, k=5, dim=1)
    mask = torch.isnan(G)

    out_g = impute_knn_gpu(G, K, knn_idx, mask)
    out_p = _python_reference_knn(G.clone(), K, k=5)
    assert torch.allclose(out_g, out_p, atol=1e-10)


def test_knn_gpu_matches_python_high_missing():
    torch.manual_seed(23)
    G_full = torch.randint(0, 3, (60, 50), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.4] = float("nan")

    K_mod = K.clone()
    K_mod.fill_diagonal_(-float("inf"))
    _, knn_idx = torch.topk(K_mod, k=7, dim=1)
    mask = torch.isnan(G)

    out_g = impute_knn_gpu(G, K, knn_idx, mask)
    out_p = _python_reference_knn(G.clone(), K, k=7)
    assert torch.allclose(out_g, out_p, atol=1e-10)


def test_knn_gpu_chunking_invariant():
    """Output must not depend on the column-chunk size."""
    torch.manual_seed(99)
    G_full = torch.randint(0, 3, (30, 64), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.2] = float("nan")
    K_mod = K.clone()
    K_mod.fill_diagonal_(-float("inf"))
    _, knn_idx = torch.topk(K_mod, k=4, dim=1)
    mask = torch.isnan(G)

    big = impute_knn_gpu(G, K, knn_idx, mask, chunk_cols=4096)
    small = impute_knn_gpu(G, K, knn_idx, mask, chunk_cols=7)
    assert torch.allclose(big, small)


# ---------------------------------------------------------------------------
# impute_ld_gpu — semantic correctness on CPU tensors
# ---------------------------------------------------------------------------


def test_ld_gpu_matches_python_basic():
    torch.manual_seed(101)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.1] = float("nan")

    from torchgenomics.preprocess.impute import impute_mean
    G_complete = impute_mean(G)
    mask = torch.isnan(G)

    out_g = impute_ld_gpu(G, G_complete, mask, window_size=5)
    out_p = _python_reference_ld(G.clone(), w=5)
    assert torch.allclose(out_g, out_p, atol=1e-9)


def test_ld_gpu_matches_python_wide_window():
    torch.manual_seed(202)
    G_full = torch.randint(0, 3, (50, 60), dtype=torch.float64)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.15] = float("nan")

    from torchgenomics.preprocess.impute import impute_mean
    G_complete = impute_mean(G)
    mask = torch.isnan(G)

    out_g = impute_ld_gpu(G, G_complete, mask, window_size=20)
    out_p = _python_reference_ld(G.clone(), w=20)
    assert torch.allclose(out_g, out_p, atol=1e-9)


def test_ld_gpu_polyploid_window():
    torch.manual_seed(303)
    G_full = torch.randint(0, 5, (45, 35), dtype=torch.float64)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.2] = float("nan")

    from torchgenomics.preprocess.impute import impute_mean
    G_complete = impute_mean(G)
    mask = torch.isnan(G)

    out_g = impute_ld_gpu(G, G_complete, mask, window_size=10)
    out_p = _python_reference_ld(G.clone(), w=10)
    assert torch.allclose(out_g, out_p, atol=1e-9)


# ---------------------------------------------------------------------------
# CUDA cross-device parity (skipped without a GPU)
# ---------------------------------------------------------------------------


@cuda_required
def test_mode_gpu_cuda_matches_cpu():
    torch.manual_seed(7)
    G_int = torch.randint(-1, 3, (200, 50), dtype=torch.long)
    out_cpu = impute_mode_gpu(G_int, 2, 50)
    out_cuda = impute_mode_gpu(G_int.cuda(), 2, 50).cpu()
    assert torch.equal(out_cpu, out_cuda)


@cuda_required
def test_knn_gpu_cuda_matches_cpu():
    torch.manual_seed(11)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.1] = float("nan")
    K_mod = K.clone()
    K_mod.fill_diagonal_(-float("inf"))
    _, knn_idx = torch.topk(K_mod, k=5, dim=1)
    mask = torch.isnan(G)

    out_cpu = impute_knn_gpu(G, K, knn_idx, mask)
    out_cuda = impute_knn_gpu(
        G.cuda(), K.cuda(), knn_idx.cuda(), mask.cuda()
    ).cpu()
    assert torch.allclose(out_cpu, out_cuda, atol=1e-10)


@cuda_required
def test_ld_gpu_cuda_matches_cpu():
    torch.manual_seed(101)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    G = G_full.clone()
    G[torch.rand(G.shape) < 0.1] = float("nan")
    from torchgenomics.preprocess.impute import impute_mean
    G_complete = impute_mean(G)
    mask = torch.isnan(G)

    out_cpu = impute_ld_gpu(G, G_complete, mask, 5)
    out_cuda = impute_ld_gpu(
        G.cuda(), G_complete.cuda(), mask.cuda(), 5
    ).cpu()
    assert torch.allclose(out_cpu, out_cuda, atol=1e-9)


@cuda_required
def test_dispatcher_routes_cuda_to_gpu_kernel():
    """End-to-end: a CUDA tensor must hit the GPU path, not the C++ path."""
    torch.manual_seed(7)
    G = torch.randint(0, 3, (200, 50), dtype=torch.float64).cuda()
    G[torch.rand(G.shape, device="cuda") < 0.1] = float("nan")
    out = impute_mode(G)
    assert out.device.type == "cuda"
    assert not torch.isnan(out).any()
