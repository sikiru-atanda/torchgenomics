"""Tests for the native C++ GWAS-aligned block-DP accelerator
(``torchgenomics._native._gwas_aligned_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_GWAS_ALIGNED, _gwas_aligned_native
from torchgenomics.ld._blocks_novel import (
    _gwas_aligned_native_enabled,
    detect_blocks_gwas_aligned,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_GWAS_ALIGNED,
    reason="Native gwas_aligned extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_GWAS_ALIGNED is True
    assert hasattr(_gwas_aligned_native, "gwas_aligned_dp")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _gwas_aligned_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_identity_partition():
    # Identity correlation matrix: every block is perfectly concentrated
    # (leading eigenvalue = trace, so concentration = 1/size). The DP
    # should return non-overlapping coverage of all SNPs.
    m = 8
    R = np.eye(m, dtype=np.float64)
    pos = (np.arange(m, dtype=np.int64) * 100)
    blocks = _gwas_aligned_native.gwas_aligned_dp(
        R, pos, 4, 0.1, 2, 1e9,
    )
    assert all(0 <= s <= e < m for s, e in blocks)
    # Non-overlapping ascending order
    last_end = -1
    for s, e in blocks:
        assert s > last_end
        last_end = e


def test_native_correlated_blocks_recovered():
    # Two correlated blocks of 4 SNPs separated by an independent SNP.
    # Use moderate (not perfect) correlation so the rank-1 condition-number
    # penalty doesn't dominate the concentration term.
    m = 9
    R = np.eye(m, dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i != j:
                R[i, j] = 0.7
                R[i + 5, j + 5] = 0.7
    pos = np.arange(m, dtype=np.int64) * 100
    blocks = _gwas_aligned_native.gwas_aligned_dp(
        R, pos, 6, 0.05, 2, 1e9,
    )
    # Non-overlapping ascending coverage
    last_end = -1
    for s, e in blocks:
        assert s > last_end
        assert e >= s
        last_end = e
    # No block should be empty / single-SNP (min_block_snps=2)
    for s, e in blocks:
        assert e - s + 1 >= 2


def test_native_distance_break_excludes_far_pairs():
    m = 6
    R = np.ones((m, m), dtype=np.float64)
    # SNPs 0..2 close, SNPs 3..5 close, but a huge gap between
    pos = np.array([0, 100, 200, 10_000_000, 10_000_100, 10_000_200], dtype=np.int64)
    blocks = _gwas_aligned_native.gwas_aligned_dp(
        R, pos, 6, 0.1, 2, 500.0,
    )
    # No block should span the gap
    for s, e in blocks:
        if s <= 2 and e >= 3:
            pytest.fail(f"block ({s},{e}) crosses the physical gap")


def test_native_invalid_shape_raises():
    R = np.zeros((4, 5), dtype=np.float64)
    pos = np.zeros(4, dtype=np.int64)
    with pytest.raises(Exception):
        _gwas_aligned_native.gwas_aligned_dp(R, pos, 4, 0.1, 2, 1e9)


def test_native_min_block_too_large_returns_empty():
    R = np.eye(3, dtype=np.float64)
    pos = np.array([0, 100, 200], dtype=np.int64)
    blocks = _gwas_aligned_native.gwas_aligned_dp(R, pos, 5, 0.1, 5, 1e9)
    assert blocks == []


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
    rng = np.random.default_rng(7)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[5, 4, 6])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=8, min_block_snps=2,
        condition_penalty=0.1, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=8, min_block_snps=2,
        condition_penalty=0.1, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(99)
    G = torch.from_numpy(
        rng.integers(0, 3, size=(60, 20)).astype(np.float64)
    )
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=6, min_block_snps=2,
        condition_penalty=0.05, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=6, min_block_snps=2,
        condition_penalty=0.05, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_multi_chrom(monkeypatch):
    rng = np.random.default_rng(123)
    G = _make_block_genotypes(rng, n_indiv=70, blocks=[4, 4, 5])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=6, min_block_snps=2,
        condition_penalty=0.1, max_kb=1e6,
    )
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gwas_aligned(
        G, pos, chrs, max_block_snps=6, min_block_snps=2,
        condition_penalty=0.1, max_kb=1e6,
    )
    assert _equivalent(blocks_native, blocks_python)
