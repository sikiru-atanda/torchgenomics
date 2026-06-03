"""Tests for the native C++ Gabriel block-detection accelerator
(``torchgenomics._native._gabriel_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_GABRIEL, _gabriel_native
from torchgenomics.ld import compute_pairwise_ld
from torchgenomics.ld._blocks import (
    _gabriel_native_enabled,
    detect_blocks_gabriel,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_GABRIEL,
    reason="Native Gabriel extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_GABRIEL is True
    assert hasattr(_gabriel_native, "gabriel_blocks")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _gabriel_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _all_pairs_strong(n: int):
    """Build a fully strong-LD pair list for n SNPs (all pairs i<j)."""
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)
    P = len(idx_i)
    return (
        np.asarray(idx_i, dtype=np.int64),
        np.asarray(idx_j, dtype=np.int64),
        np.ones(P, dtype=np.uint8),       # strong_ld
        np.zeros(P, dtype=np.uint8),      # strong_rec
        np.ones(P, dtype=np.uint8),       # informative
        np.full(P, 0.9, dtype=np.float64),  # r2
        np.full(P, 0.95, dtype=np.float64), # dprime
    )


def test_native_all_strong_pairs_one_block():
    """All pairs strong → one block spanning [0, n-1]."""
    n = 6
    idx_i, idx_j, sld, srec, inf, r2, dp = _all_pairs_strong(n)
    out = _gabriel_native.gabriel_blocks(
        n, idx_i, idx_j, sld, srec, inf, r2, dp,
        0.95, 0.04, 2,
    )
    assert len(out) == 1
    s, e, mr2, mdp = out[0]
    assert (s, e) == (0, n - 1)
    assert mr2 == pytest.approx(0.9)
    assert mdp == pytest.approx(0.95)


def test_native_no_informative_no_blocks():
    """No informative pairs → empty result."""
    n = 5
    idx_i, idx_j, _sld, _srec, _inf, r2, dp = _all_pairs_strong(n)
    sld = np.zeros_like(_sld)
    srec = np.zeros_like(_srec)
    inf = np.zeros_like(_inf)
    out = _gabriel_native.gabriel_blocks(
        n, idx_i, idx_j, sld, srec, inf, r2, dp,
        0.95, 0.04, 2,
    )
    assert out == []


def test_native_two_disjoint_strong_blocks():
    """Two strong sub-blocks separated by a recombination region."""
    n = 8
    idx_i, idx_j = [], []
    sld, srec, inf, r2, dp = [], [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)
            same_block = (i < 4 and j < 4) or (i >= 4 and j >= 4)
            sld.append(1 if same_block else 0)
            srec.append(0 if same_block else 1)
            inf.append(1)
            r2.append(0.9 if same_block else 0.05)
            dp.append(0.95 if same_block else 0.1)
    out = _gabriel_native.gabriel_blocks(
        n,
        np.asarray(idx_i, dtype=np.int64),
        np.asarray(idx_j, dtype=np.int64),
        np.asarray(sld, dtype=np.uint8),
        np.asarray(srec, dtype=np.uint8),
        np.asarray(inf, dtype=np.uint8),
        np.asarray(r2, dtype=np.float64),
        np.asarray(dp, dtype=np.float64),
        0.95, 0.04, 2,
    )
    starts_ends = sorted((s, e) for (s, e, _, _) in out)
    assert starts_ends == [(0, 3), (4, 7)]


def test_native_min_block_snps_filter():
    """min_block_snps=4 rejects a 3-SNP candidate."""
    n = 3
    idx_i, idx_j, sld, srec, inf, r2, dp = _all_pairs_strong(n)
    out = _gabriel_native.gabriel_blocks(
        n, idx_i, idx_j, sld, srec, inf, r2, dp,
        0.95, 0.04, 4,
    )
    assert out == []


def test_native_invalid_pair_array_lengths_raise():
    n = 4
    idx_i = np.zeros(3, dtype=np.int64)
    idx_j = np.zeros(2, dtype=np.int64)  # mismatched
    flags = np.zeros(3, dtype=np.uint8)
    vals = np.zeros(3, dtype=np.float64)
    with pytest.raises(Exception):
        _gabriel_native.gabriel_blocks(
            n, idx_i, idx_j, flags, flags, flags, vals, vals,
            0.95, 0.04, 2,
        )


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _make_block_genotypes(rng, n_indiv, blocks):
    """Build a genotype matrix where SNPs in the same block are highly
    correlated and SNPs across blocks are independent."""
    snps = []
    for size in blocks:
        # latent block haplotype
        h = (rng.random(n_indiv) < 0.5).astype(np.float64) * 2.0
        for _ in range(size):
            noise = rng.random(n_indiv) < 0.05
            x = h.copy()
            x[noise] = 2.0 - x[noise]
            snps.append(x)
    G = np.stack(snps, axis=1)  # (n, m)
    return torch.from_numpy(G)


def _equivalent_blocks(a, b):
    """Compare two LDBlock lists by (start, end) intervals."""
    A = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in a)
    B = sorted((blk.variant_indices[0], blk.variant_indices[-1]) for blk in b)
    return A == B


def test_dispatch_native_matches_python_synthetic_blocks(monkeypatch):
    rng = np.random.default_rng(0)
    G = _make_block_genotypes(rng, n_indiv=120, blocks=[5, 4, 6, 3])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gabriel(pld, pos, chrs)

    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gabriel(pld, pos, chrs)

    assert _equivalent_blocks(blocks_native, blocks_python)


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(7)
    n_indiv = 80
    m = 25
    G = torch.from_numpy(
        rng.integers(0, 3, size=(n_indiv, m)).astype(np.float64)
    )
    pos = list(range(m))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gabriel(pld, pos, chrs)

    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gabriel(pld, pos, chrs)

    assert _equivalent_blocks(blocks_native, blocks_python)
    # Means should agree to floating-point precision
    for a, b in zip(
        sorted(blocks_native, key=lambda x: x.variant_indices[0]),
        sorted(blocks_python, key=lambda x: x.variant_indices[0]),
    ):
        assert a.mean_r2 == pytest.approx(b.mean_r2, rel=1e-9, abs=1e-9)
        assert a.mean_dprime == pytest.approx(b.mean_dprime, rel=1e-9, abs=1e-9)


def test_dispatch_native_matches_python_relaxed_thresholds(monkeypatch):
    """Relaxed thresholds → more accepted blocks; equivalence still holds."""
    rng = np.random.default_rng(11)
    G = _make_block_genotypes(rng, n_indiv=100, blocks=[3, 3, 3, 3])
    m = G.shape[1]
    pos = list(range(m))
    chrs = ["1"] * m
    pld = compute_pairwise_ld(G, pos, chrs, max_kb=1000.0, maf_min=0.0)

    kwargs = dict(strong_pct=0.80, rec_max_pct=0.10, ci_low=0.5, ci_high=0.9)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    blocks_native = detect_blocks_gabriel(pld, pos, chrs, **kwargs)

    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    blocks_python = detect_blocks_gabriel(pld, pos, chrs, **kwargs)

    assert _equivalent_blocks(blocks_native, blocks_python)
