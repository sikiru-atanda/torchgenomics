"""Tests for the native C++ uncertainty-blocks accelerator
(``torchgwas._native._uncertainty_blocks_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import (
    HAS_NATIVE_UNCERTAINTY_BLOCKS,
    _uncertainty_blocks_native,
)
from torchgwas.ld._blocks_novel import (
    _uncertainty_blocks_native_enabled,
    detect_blocks_uncertainty,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_UNCERTAINTY_BLOCKS,
    reason="Native uncertainty_blocks extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_UNCERTAINTY_BLOCKS is True
    assert hasattr(_uncertainty_blocks_native, "uncertainty_per_chrom_blocks")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _uncertainty_blocks_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _make_synthetic(M, blocks, off_block_r2=0.0, in_block_r2=0.9):
    """Build (weights, quality, quality_ok) for M markers, with given block ranges."""
    W = np.full((M, M), off_block_r2, dtype=np.float64)
    np.fill_diagonal(W, 1.0)
    for s, e in blocks:
        for i in range(s, e + 1):
            for j in range(s, e + 1):
                if i != j:
                    W[i, j] = in_block_r2
    q = np.ones(M, dtype=np.float64)
    qok = np.ones(M, dtype=bool)
    return W, q, qok


def test_native_single_block_recovered():
    M = 6
    W, q, qok = _make_synthetic(M, [(0, 2), (3, 5)], off_block_r2=0.0, in_block_r2=0.8)
    chr_mask = np.arange(M, dtype=np.int64)
    chr_pos = np.arange(M, dtype=np.int64) * 1000
    out = _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
        W, q, qok, chr_mask, chr_pos, 1e9, 0.5, 2,
    )
    # Two blocks expected: [0..2] and [3..5]
    assert len(out) == 2
    starts = sorted(b[0] for b in out)
    ends = sorted(b[1] for b in out)
    assert starts == [0, 3]
    assert ends == [2, 5]
    for s, e, mean_r2, q_block in out:
        assert mean_r2 == pytest.approx(0.8)
        assert q_block == pytest.approx(1.0)


def test_native_min_block_filter():
    # Single isolated marker should not produce a block.
    M = 4
    W, q, qok = _make_synthetic(M, [(0, 1)], off_block_r2=0.0, in_block_r2=0.9)
    chr_mask = np.arange(M, dtype=np.int64)
    chr_pos = np.arange(M, dtype=np.int64) * 1000
    out = _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
        W, q, qok, chr_mask, chr_pos, 1e9, 0.5, 3,
    )
    assert len(out) == 0  # block size 2 < min 3


def test_native_quality_filter_zero_pair():
    # All quality_ok=False -> empty pair list -> no blocks.
    M = 4
    W, q, _ = _make_synthetic(M, [(0, 3)], in_block_r2=0.9)
    qok = np.zeros(M, dtype=bool)
    chr_mask = np.arange(M, dtype=np.int64)
    chr_pos = np.arange(M, dtype=np.int64) * 1000
    out = _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
        W, q, qok, chr_mask, chr_pos, 1e9, 0.5, 2,
    )
    assert len(out) == 0


def test_native_max_bp_no_pair():
    # max_bp too small to admit any pair within window -> empty.
    M = 4
    W, q, qok = _make_synthetic(M, [(0, 3)], in_block_r2=0.9)
    chr_mask = np.arange(M, dtype=np.int64)
    chr_pos = (np.arange(M, dtype=np.int64) * 10_000)
    out = _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
        W, q, qok, chr_mask, chr_pos, 1000.0, 0.5, 2,
    )
    assert len(out) == 0


def test_native_invalid_shape_raises():
    W = np.zeros((4, 5), dtype=np.float64)  # not square
    q = np.zeros(4, dtype=np.float64)
    qok = np.ones(4, dtype=bool)
    cm = np.arange(4, dtype=np.int64)
    cp = np.arange(4, dtype=np.int64)
    with pytest.raises(Exception):
        _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
            W, q, qok, cm, cp, 1e9, 0.5, 2,
        )


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _make_gp_fixture(seed: int, n: int = 80, m: int = 8, ploidy: int = 2):
    torch.manual_seed(seed)
    probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
    for i in range(n):
        for j in range(m):
            true_dose = torch.randint(0, ploidy + 1, (1,)).item()
            probs[i, j, true_dose] = 0.9
            for k in range(ploidy + 1):
                if k != true_dose:
                    probs[i, j, k] = 0.05
    d_vals = torch.arange(ploidy + 1, dtype=torch.float64)
    G = (probs * d_vals).sum(dim=-1)
    pos = [1000 * (j + 1) for j in range(m)]
    chrs = ["1"] * m
    ids = [f"rs{j}" for j in range(m)]
    return G, probs, pos, chrs, ids, ploidy


def _python_reference(*args, **kwargs):
    os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    try:
        return detect_blocks_uncertainty(*args, **kwargs)
    finally:
        del os.environ["TORCHGWAS_DISABLE_NATIVE"]


def _native_dispatch(*args, **kwargs):
    os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    return detect_blocks_uncertainty(*args, **kwargs)


def _block_signature(blocks):
    return [
        (
            b.region.chr,
            b.region.start,
            b.region.end,
            b.n_variants,
            tuple(b.variant_indices),
            round(float(b.mean_r2), 9),
            round(float(b.uncertainty_metric or 0.0), 9),
        )
        for b in blocks
    ]


def test_dispatch_native_matches_python_synthetic_high_confidence():
    G, probs, pos, chrs, ids, ploidy = _make_gp_fixture(seed=11)
    out_n = _native_dispatch(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.05, max_kb=100.0, min_block_snps=2,
    )
    out_p = _python_reference(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.05, max_kb=100.0, min_block_snps=2,
    )
    assert _block_signature(out_n) == _block_signature(out_p)


def test_dispatch_native_matches_python_random_seed():
    G, probs, pos, chrs, ids, ploidy = _make_gp_fixture(seed=42, n=60, m=10)
    out_n = _native_dispatch(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.1, max_kb=200.0, min_block_snps=2,
    )
    out_p = _python_reference(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.1, max_kb=200.0, min_block_snps=2,
    )
    assert _block_signature(out_n) == _block_signature(out_p)


def test_dispatch_native_matches_python_multi_chr():
    G, probs, pos, chrs, ids, ploidy = _make_gp_fixture(seed=7, n=50, m=8)
    # Split across two chromosomes
    chrs = ["1"] * 4 + ["2"] * 4
    out_n = _native_dispatch(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.05, max_kb=200.0, min_block_snps=2,
    )
    out_p = _python_reference(
        G, probs, ploidy, pos, chrs, ids,
        r2_threshold=0.05, max_kb=200.0, min_block_snps=2,
    )
    assert _block_signature(out_n) == _block_signature(out_p)
