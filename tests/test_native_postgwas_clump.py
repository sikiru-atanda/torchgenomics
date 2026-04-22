"""Tests for the native LD-clumping fast path in ``torchgwas.postgwas._clump``.

Phase 41y reuses the existing ``_ct_native.clump_block`` kernel (built in
Phase 41c for the C+T PGS method) by mapping each chromosome to one block of
a synthetic block-diagonal LD reference. The Python loop in ``ld_clump``
remains in-tree as the algorithmic spec; the dispatcher routes through C++
when the build is present, the tensors are CPU, and ``TORCHGWAS_DISABLE_NATIVE``
is unset.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_CT
from torchgwas.postgwas._clump import ClumpResult, ld_clump

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_CT, reason="native C+T extension not built"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clump_problem(n=200, m=60, n_chroms=3, seed=0):
    """Synthetic genotype + p-value problem with a few strongly correlated
    blocks per chromosome."""
    rng = np.random.default_rng(seed)
    G = rng.binomial(2, 0.3, size=(n, m)).astype(np.float64)

    # Inject correlation structure: every 5 SNPs share a latent
    base_per_chrom = m // n_chroms
    for c in range(n_chroms):
        for blk_start in range(c * base_per_chrom, (c + 1) * base_per_chrom, 5):
            base = G[:, blk_start].copy()
            for j in range(blk_start + 1, min(blk_start + 5, m)):
                noise = rng.binomial(1, 0.15, size=n).astype(np.float64)
                G[:, j] = np.where(noise == 0, base, G[:, j])

    chr_labels = []
    pos = []
    for c in range(n_chroms):
        n_in_chrom = base_per_chrom if c < n_chroms - 1 else m - c * base_per_chrom
        for j in range(n_in_chrom):
            chr_labels.append(str(c + 1))
            pos.append((j + 1) * 50_000)  # 50 kb spacing

    p = rng.uniform(0.0, 1.0, size=m)
    # Force a handful of strongly significant SNPs (no ties)
    sig_idx = rng.choice(m, size=8, replace=False)
    p[sig_idx] = np.linspace(1e-12, 1e-9, 8)
    return (
        torch.from_numpy(p),
        torch.from_numpy(G),
        pos,
        chr_labels,
    )


def _run_ld_clump_with(disable_native: bool, p, G, pos, chr_labels, **kwargs):
    if disable_native:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    try:
        return ld_clump(p, G, pos, chr_labels, **kwargs)
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_smoke_returns_clump_result():
    p, G, pos, chr_labels = _make_clump_problem()
    res = _run_ld_clump_with(False, p, G, pos, chr_labels, p_threshold=1e-6)
    assert isinstance(res, ClumpResult)
    assert res.n_clumps == len(res.index_snps)
    assert res.n_clumps >= 1
    assert res.n_clumps == len(res.clump_members)


def test_no_significant_snps_returns_empty():
    p, G, pos, chr_labels = _make_clump_problem(seed=11)
    p_high = torch.full_like(p, 0.5)
    res = _run_ld_clump_with(False, p_high, G, pos, chr_labels, p_threshold=5e-8)
    assert res.n_clumps == 0
    assert res.index_snps == []
    assert res.clump_members == []


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence
# ---------------------------------------------------------------------------


def _assert_clump_equal(a: ClumpResult, b: ClumpResult):
    assert a.n_clumps == b.n_clumps
    assert a.index_snps == b.index_snps
    assert len(a.clump_members) == len(b.clump_members)
    for ma, mb in zip(a.clump_members, b.clump_members):
        assert sorted(ma) == sorted(mb)


def test_native_matches_python_default():
    p, G, pos, chr_labels = _make_clump_problem(seed=1)
    res_native = _run_ld_clump_with(
        False, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.1
    )
    res_python = _run_ld_clump_with(
        True, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.1
    )
    _assert_clump_equal(res_native, res_python)


def test_native_matches_python_strict_r2():
    p, G, pos, chr_labels = _make_clump_problem(seed=2)
    res_native = _run_ld_clump_with(
        False, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.5
    )
    res_python = _run_ld_clump_with(
        True, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.5
    )
    _assert_clump_equal(res_native, res_python)


def test_native_matches_python_narrow_window():
    p, G, pos, chr_labels = _make_clump_problem(seed=3)
    res_native = _run_ld_clump_with(
        False, p, G, pos, chr_labels,
        p_threshold=1e-6, r2_threshold=0.1, window_kb=80.0,
    )
    res_python = _run_ld_clump_with(
        True, p, G, pos, chr_labels,
        p_threshold=1e-6, r2_threshold=0.1, window_kb=80.0,
    )
    _assert_clump_equal(res_native, res_python)


def test_native_matches_python_single_chromosome():
    p, G, pos, chr_labels = _make_clump_problem(seed=4, n_chroms=1)
    res_native = _run_ld_clump_with(
        False, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.2
    )
    res_python = _run_ld_clump_with(
        True, p, G, pos, chr_labels, p_threshold=1e-6, r2_threshold=0.2
    )
    _assert_clump_equal(res_native, res_python)


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


def test_window_zero_keeps_only_self():
    """A zero-bp window means no SNP can ever absorb another — every
    significant SNP becomes its own index SNP with an empty members list."""
    p, G, pos, chr_labels = _make_clump_problem(seed=5)
    res_native = _run_ld_clump_with(
        False, p, G, pos, chr_labels,
        p_threshold=1e-6, r2_threshold=0.1, window_kb=0.0,
    )
    res_python = _run_ld_clump_with(
        True, p, G, pos, chr_labels,
        p_threshold=1e-6, r2_threshold=0.1, window_kb=0.0,
    )
    _assert_clump_equal(res_native, res_python)
    for members in res_native.clump_members:
        assert members == []


def test_relaxed_p_threshold_keeps_more_leads():
    """A looser p-threshold should never produce fewer clumps than a strict one."""
    p, G, pos, chr_labels = _make_clump_problem(seed=6)
    strict = _run_ld_clump_with(
        False, p, G, pos, chr_labels, p_threshold=1e-9, r2_threshold=0.1
    )
    loose = _run_ld_clump_with(
        False, p, G, pos, chr_labels, p_threshold=1e-3, r2_threshold=0.1
    )
    assert loose.n_clumps >= strict.n_clumps
