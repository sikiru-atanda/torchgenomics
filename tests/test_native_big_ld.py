"""Tests for the native C++ Big-LD candidate-interval accelerator
(``torchgenomics._native._big_ld_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_BIG_LD, _big_ld_native
from torchgenomics.ld._blocks_literature import (
    _big_ld_native_enabled,
    detect_blocks_big_ld,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_BIG_LD,
    reason="Native Big-LD extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_BIG_LD is True
    assert hasattr(_big_ld_native, "big_ld_intervals")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _big_ld_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _block_diag_r2(block_sizes, off_block_r2=0.05):
    """Build a synthetic R² matrix with high within-block correlation
    and low between-block correlation."""
    m = sum(block_sizes)
    R = np.full((m, m), off_block_r2, dtype=np.float64)
    start = 0
    for sz in block_sizes:
        R[start:start + sz, start:start + sz] = 0.9
        start += sz
    np.fill_diagonal(R, 1.0)
    return R


def test_native_one_strong_block_yields_intervals():
    R = _block_diag_r2([5])
    pos = np.arange(5, dtype=np.int64) * 1000
    out = _big_ld_native.big_ld_intervals(
        R, pos, 0.5, 2, 10, 1e9
    )
    # All (i, j) pairs with j >= i+1 within window should be accepted
    # because the entire block has mean off-diagonal r² = 0.9 >= 0.5
    assert len(out) > 0
    for i, j, w in out:
        assert 0 <= i < j < 5
        size = j - i + 1
        assert w == pytest.approx(size * 0.9, rel=1e-9)


def test_native_two_strong_blocks_no_cross_intervals():
    R = _block_diag_r2([4, 4], off_block_r2=0.0)
    pos = np.arange(8, dtype=np.int64) * 1000
    # Strict threshold: only fully-within-block intervals (mean 0.9) pass.
    out = _big_ld_native.big_ld_intervals(
        R, pos, 0.7, 2, 10, 1e9
    )
    assert len(out) > 0
    for i, j, _ in out:
        assert (j < 4) or (i >= 4), f"cross-block ({i},{j}) accepted"


def test_native_distance_break():
    """Distance constraint should stop the j loop early."""
    R = _block_diag_r2([5])
    pos = np.array([0, 100, 200, 300, 1_000_000], dtype=np.int64)
    out = _big_ld_native.big_ld_intervals(
        R, pos, 0.5, 2, 10, 500.0  # 500 bp max
    )
    # No interval should include SNP 4 (it is far away)
    for i, j, _ in out:
        assert j < 4


def test_native_min_block_snps_filter():
    R = _block_diag_r2([3])
    pos = np.arange(3, dtype=np.int64) * 1000
    out = _big_ld_native.big_ld_intervals(
        R, pos, 0.5, 4, 10, 1e9  # min size 4 — nothing can pass
    )
    assert out == []


def test_native_invalid_shape_raises():
    R = np.zeros((3, 4), dtype=np.float64)
    pos = np.zeros(3, dtype=np.int64)
    with pytest.raises(Exception):
        _big_ld_native.big_ld_intervals(R, pos, 0.5, 2, 10, 1e9)


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


def _equivalent_blocks(a, b):
    A = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in a)
    B = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in b)
    return A == B


def test_dispatch_native_matches_python_blocks(monkeypatch):
    rng = np.random.default_rng(0)
    G = _make_block_genotypes(rng, n_indiv=120, blocks=[5, 4, 6, 3])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.4, window_size=20, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.4, window_size=20, max_kb=1000.0
    )
    assert _equivalent_blocks(blocks_native, blocks_python)
    for a, b in zip(
        sorted(blocks_native, key=lambda x: x.variant_indices[0]),
        sorted(blocks_python, key=lambda x: x.variant_indices[0]),
    ):
        assert a.mean_r2 == pytest.approx(b.mean_r2, rel=1e-9, abs=1e-9)


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(13)
    G = torch.from_numpy(
        rng.integers(0, 3, size=(80, 30)).astype(np.float64)
    )
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.2, window_size=10, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.2, window_size=10, max_kb=1000.0
    )
    assert _equivalent_blocks(blocks_native, blocks_python)


def test_dispatch_native_matches_python_multi_chrom(monkeypatch):
    rng = np.random.default_rng(21)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[4, 5, 4])
    m = G.shape[1]
    pos = list(range(m))
    # Two chromosomes
    chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.3, window_size=15, max_kb=1000.0
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_big_ld(
        G, pos, chrs, r2_threshold=0.3, window_size=15, max_kb=1000.0
    )
    assert _equivalent_blocks(blocks_native, blocks_python)
