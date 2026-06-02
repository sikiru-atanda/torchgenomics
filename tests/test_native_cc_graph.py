"""Tests for the native C++ cc-graph adjacency-build accelerator
(``torchgenomics._native._cc_graph_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_CC_GRAPH, _cc_graph_native
from torchgenomics.ld._blocks_literature import (
    _cc_graph_native_enabled,
    detect_blocks_cc_graph,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_CC_GRAPH,
    reason="Native cc-graph extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_CC_GRAPH is True
    assert hasattr(_cc_graph_native, "cc_graph_windowed_adj")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _cc_graph_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _center_and_var(G):
    Gc = G - G.mean(axis=0, keepdims=True)
    var = (Gc * Gc).mean(axis=0)
    return Gc, var


def test_native_perfectly_correlated_pair():
    n = 30
    g = np.array([0.0, 1.0, 2.0] * (n // 3), dtype=np.float64)[:n]
    G = np.stack([g, g, g], axis=1)  # 3 identical SNPs
    Gc, var = _center_and_var(G)
    pos = np.arange(3, dtype=np.int64) * 1000
    adj = _cc_graph_native.cc_graph_windowed_adj(
        Gc, var, pos, 5, 1e9, 0.5,
    )
    # All off-diagonal entries should be 1.0 (within float tolerance)
    for i in range(3):
        for j in range(3):
            if i == j:
                assert adj[i, j] == 0.0  # diagonal stays zero
            else:
                assert adj[i, j] == pytest.approx(1.0, abs=1e-9)


def test_native_uncorrelated_pair():
    rng = np.random.default_rng(0)
    n = 200
    g0 = rng.standard_normal(n)
    g1 = rng.standard_normal(n)
    G = np.stack([g0, g1], axis=1)
    Gc, var = _center_and_var(G)
    pos = np.array([0, 1000], dtype=np.int64)
    adj = _cc_graph_native.cc_graph_windowed_adj(
        Gc, var, pos, 5, 1e9, 0.5,
    )
    # Random independent SNPs should have r² well below 0.5
    assert adj[0, 1] == 0.0
    assert adj[1, 0] == 0.0


def test_native_window_clip():
    n, m = 20, 5
    rng = np.random.default_rng(1)
    G = rng.standard_normal((n, m))
    # Force two strongly correlated SNPs at indices 0 and 4
    G[:, 4] = G[:, 0]
    Gc, var = _center_and_var(G)
    pos = np.arange(m, dtype=np.int64) * 100
    # Window of 2 means we never inspect (0, 4)
    adj = _cc_graph_native.cc_graph_windowed_adj(
        Gc, var, pos, 2, 1e9, 0.5,
    )
    assert adj[0, 4] == 0.0


def test_native_distance_break():
    n, m = 20, 5
    g = np.array([0.0, 1.0, 2.0] * (n // 3 + 1), dtype=np.float64)[:n]
    G = np.stack([g] * m, axis=1)
    Gc, var = _center_and_var(G)
    # Big jump between SNP 2 and SNP 3
    pos = np.array([0, 100, 200, 10_000_000, 10_000_100], dtype=np.int64)
    adj = _cc_graph_native.cc_graph_windowed_adj(
        Gc, var, pos, 100, 500.0, 0.5,
    )
    # No edge crossing the gap
    for i in range(3):
        for j in range(3, 5):
            assert adj[i, j] == 0.0


def test_native_invalid_shape_raises():
    G = np.zeros((10, 4), dtype=np.float64)
    var = np.zeros(3, dtype=np.float64)  # mismatch
    pos = np.zeros(4, dtype=np.int64)
    with pytest.raises(Exception):
        _cc_graph_native.cc_graph_windowed_adj(G, var, pos, 5, 1e9, 0.5)


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
    rng = np.random.default_rng(2)
    G = _make_block_genotypes(rng, n_indiv=100, blocks=[5, 4, 6])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.4, window=10, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.4, window=10, max_kb=1000.0
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(13)
    G = torch.from_numpy(
        rng.integers(0, 3, size=(80, 25)).astype(np.float64)
    )
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.2, window=10, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.2, window=10, max_kb=1000.0
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_multi_chrom(monkeypatch):
    rng = np.random.default_rng(21)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[4, 5, 4])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.3, window=10, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_cc_graph(
        G, pos, chrs, r2_threshold=0.3, window=10, max_kb=1000.0
    )
    assert _equivalent(blocks_native, blocks_python)
