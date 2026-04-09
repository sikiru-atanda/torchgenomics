"""Tests for the native C++ solid-spine LD block accelerator
(``torchgwas._native._spine_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_SPINE, _spine_native
from torchgwas.ld import detect_blocks
from torchgwas.ld._blocks import _spine_native_enabled

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_SPINE,
    reason="Native spine extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_SPINE is True
    assert hasattr(_spine_native, "spine_partition")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _spine_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_all_strong_single_block():
    n = 5
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = 0.95
    out = _spine_native.spine_partition(M, n, 0.8, 2)
    assert out == [(0, 4)]


def test_native_all_weak_no_blocks():
    n = 5
    M = np.zeros((n, n), dtype=np.float64)
    out = _spine_native.spine_partition(M, n, 0.5, 2)
    assert out == []


def test_native_two_disjoint_blocks():
    n = 6
    M = np.zeros((n, n), dtype=np.float64)
    # Strong within {0,1,2}
    for i in range(3):
        for j in range(i + 1, 3):
            M[i, j] = 0.9
    # Strong within {3,4,5}
    for i in range(3, 6):
        for j in range(i + 1, 6):
            M[i, j] = 0.9
    out = _spine_native.spine_partition(M, n, 0.8, 2)
    assert (0, 2) in out
    assert (3, 5) in out


def test_native_min_block_filter_drops_short():
    n = 4
    M = np.zeros((n, n), dtype=np.float64)
    M[0, 1] = 0.9  # only a 2-SNP candidate
    out = _spine_native.spine_partition(M, n, 0.8, 3)
    assert out == []


def test_native_invalid_shape_raises():
    M = np.zeros((4, 5), dtype=np.float64)
    with pytest.raises(Exception):
        _spine_native.spine_partition(M, 4, 0.5, 2)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _make_block_genotypes(rng, n_indiv, blocks, noise=0.05):
    snps = []
    for size in blocks:
        h = (rng.random(n_indiv) < 0.5).astype(np.float64) * 2.0
        for _ in range(size):
            mask = rng.random(n_indiv) < noise
            x = h.copy()
            x[mask] = 2.0 - x[mask]
            snps.append(x)
    return torch.from_numpy(np.stack(snps, axis=1))


def _equivalent(a, b):
    A = sorted(tuple(sorted(blk.variant_indices)) for blk in a)
    B = sorted(tuple(sorted(blk.variant_indices)) for blk in b)
    return A == B


def test_dispatch_native_matches_python_blocks(monkeypatch):
    rng = np.random.default_rng(11)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[5, 4, 6])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m
    ids = [f"snp{i}" for i in range(m)]

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.5, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.5, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(77)
    G = torch.from_numpy(
        rng.integers(0, 3, size=(60, 18)).astype(np.float64)
    )
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m
    ids = [f"snp{i}" for i in range(m)]

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.7, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.7, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_relaxed(monkeypatch):
    rng = np.random.default_rng(33)
    G = _make_block_genotypes(rng, n_indiv=70, blocks=[4, 3, 5, 4])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m
    ids = [f"snp{i}" for i in range(m)]

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.3, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks(
        G, pos, chrs, ids, method="spine",
        d_prime_threshold=0.3, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)
