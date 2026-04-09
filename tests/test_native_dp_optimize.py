"""Tests for the native C++ DP-optimize block-detection accelerator
(``torchgwas._native._dp_optimize_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_DP_OPTIMIZE, _dp_optimize_native
from torchgwas.ld._blocks_literature import (
    _dp_optimize_native_enabled,
    detect_blocks_dp_optimize,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_DP_OPTIMIZE,
    reason="Native DP-optimize extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_DP_OPTIMIZE is True
    assert hasattr(_dp_optimize_native, "dp_optimize_segment")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _dp_optimize_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _identity_R(m):
    R = np.eye(m, dtype=np.float64)
    return R


def test_native_tag_snp_identity_returns_segments():
    """With R=I and threshold=0.5, every SNP must be its own tag, but
    block size is constrained — at least one segment is returned."""
    n, m = 20, 10
    G = np.random.default_rng(0).integers(0, 3, size=(n, m)).astype(np.float64)
    R = _identity_R(m)
    pos = np.arange(m, dtype=np.int64) * 1000
    out = _dp_optimize_native.dp_optimize_segment(
        G, R, pos,
        1,           # tag_snp
        1.0,         # penalty
        2, 5,        # min/max block snps
        1e9,         # max_bp
        0.01, 0.5,   # hap thresh, tag thresh
    )
    # Some segmentation must be returned
    assert isinstance(out, list)
    for s, e in out:
        assert 0 <= s <= e < m
        assert e - s + 1 >= 2


def test_native_haplotype_diversity_homogeneous_block():
    """All-zero haplotype block → 1 distinct pattern → cost = 1."""
    n, m = 30, 6
    G = np.zeros((n, m), dtype=np.float64)
    R = np.zeros((m, m), dtype=np.float64)
    pos = np.arange(m, dtype=np.int64) * 1000
    out = _dp_optimize_native.dp_optimize_segment(
        G, R, pos,
        0,           # haplotype_diversity
        0.0, 2, 6, 1e9,
        0.01, 0.8,
    )
    assert len(out) >= 1
    # The segmentation must cover all m SNPs
    covered = set()
    for s, e in out:
        for k in range(s, e + 1):
            covered.add(k)
    assert covered == set(range(m))


def test_native_distance_constraint_blocks():
    """A huge gap between SNP 4 and SNP 5 should force a split there."""
    n, m = 20, 8
    G = np.zeros((n, m), dtype=np.float64)
    R = np.eye(m, dtype=np.float64)
    pos = np.array(
        [0, 100, 200, 300, 400, 10_000_000, 10_000_100, 10_000_200],
        dtype=np.int64,
    )
    out = _dp_optimize_native.dp_optimize_segment(
        G, R, pos, 0, 0.0, 2, 8, 1000.0, 0.01, 0.8
    )
    # No segment may straddle SNPs 4 and 5
    for s, e in out:
        assert not (s <= 4 and e >= 5)


def test_native_invalid_objective_raises():
    n, m = 10, 4
    G = np.zeros((n, m), dtype=np.float64)
    R = np.eye(m, dtype=np.float64)
    pos = np.arange(m, dtype=np.int64)
    with pytest.raises(Exception):
        _dp_optimize_native.dp_optimize_segment(
            G, R, pos, 7, 0.0, 2, 4, 1e9, 0.01, 0.8
        )


def test_native_invalid_shape_raises():
    G = np.zeros((10, 4), dtype=np.float64)
    R = np.zeros((3, 4), dtype=np.float64)  # not square
    pos = np.arange(4, dtype=np.int64)
    with pytest.raises(Exception):
        _dp_optimize_native.dp_optimize_segment(
            G, R, pos, 1, 0.0, 2, 4, 1e9, 0.01, 0.8
        )


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
    A = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in a)
    B = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in b)
    return A == B


def test_dispatch_native_matches_python_hap_div(monkeypatch):
    rng = np.random.default_rng(3)
    G = _make_block_genotypes(rng, n_indiv=100, blocks=[4, 5, 4])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="haplotype_diversity",
        max_block_snps=8, max_kb=1000.0,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="haplotype_diversity",
        max_block_snps=8, max_kb=1000.0,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_tag_snp(monkeypatch):
    rng = np.random.default_rng(7)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[4, 4, 4])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="tag_snp",
        tag_r2_threshold=0.5,
        max_block_snps=6, max_kb=1000.0,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="tag_snp",
        tag_r2_threshold=0.5,
        max_block_snps=6, max_kb=1000.0,
    )
    assert _equivalent(blocks_native, blocks_python)


def test_dispatch_native_matches_python_multi_chrom(monkeypatch):
    rng = np.random.default_rng(11)
    G = _make_block_genotypes(rng, n_indiv=80, blocks=[4, 5, 4])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="haplotype_diversity",
        max_block_snps=8, max_kb=1000.0,
    )
    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_dp_optimize(
        G, pos, chrs,
        objective="haplotype_diversity",
        max_block_snps=8, max_kb=1000.0,
    )
    assert _equivalent(blocks_native, blocks_python)
