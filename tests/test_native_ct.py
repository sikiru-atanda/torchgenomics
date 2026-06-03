"""Tests for the native C++ C+T clumping accelerator (torchgenomics._native._ct_native).

These tests are skipped when the compiled extension is unavailable so that CI
on machines without a C++ toolchain still runs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_CT, _ct_native
from torchgenomics.pgs.ct import (
    ClumpingThresholding,
    _clump_with_ld_reference,
    _clump_with_ld_reference_dispatch,
    _native_enabled,
)
from torchgenomics.pgs.ld_ref import build_ld_reference
from torchgenomics.postgwas._sumstats import SumStats

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_CT,
    reason="Native C+T extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _make_ss_and_ld(m=30, n=2000, n_causal=4, seed=0, mode="full"):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)
    beta_true = torch.zeros(m, dtype=torch.float64)
    causal = torch.randperm(m, generator=g)[:n_causal]
    beta_true[causal] = 0.4 * torch.randn(n_causal, generator=g, dtype=torch.float64)
    y = G @ beta_true + torch.randn(n, generator=g, dtype=torch.float64) * 0.5
    y = (y - y.mean()) / y.std().clamp(min=1e-12)
    beta_hat = (G * y.unsqueeze(1)).sum(0) / n
    se = torch.full((m,), 1.0 / math.sqrt(n), dtype=torch.float64)
    z = beta_hat / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(z.abs() / math.sqrt(2.0))))
    ss = SumStats(
        chr=["1"] * m,
        pos=[100 + 10 * i for i in range(m)],
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=beta_hat,
        se=se,
        p=p,
        n=torch.full((m,), float(n)),
        af=None,
    )
    meta = {"snp": ss.snp, "chr": ss.chr, "pos": ss.pos, "a1": ss.a1, "a2": ss.a2}
    if mode == "full":
        ld = build_ld_reference(G, meta, mode="full")
    else:
        # Three roughly-equal contiguous blocks.
        block_assignments = [i // (max(m // 3, 1)) for i in range(m)]
        block_assignments = [min(b, 2) for b in block_assignments]
        ld = build_ld_reference(G, meta, mode="block", block_assignments=block_assignments)
    return ss, ld


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_CT is True
    assert hasattr(_ct_native, "clump_full")
    assert hasattr(_ct_native, "clump_block")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry points
# ---------------------------------------------------------------------------


def test_clump_full_returns_consistent_shapes():
    rng = np.random.default_rng(0)
    m = 20
    G = rng.standard_normal((500, m))
    G = (G - G.mean(0)) / G.std(0)
    R = (G.T @ G / 499).astype(np.float64)
    p = rng.uniform(0, 1, m)
    p[3] = 1e-12
    p[10] = 1e-9
    chr_codes = np.zeros(m, dtype=np.int64)
    pos = np.arange(m, dtype=np.int64) * 100
    idx, off, mem = _ct_native.clump_full(
        p, R, chr_codes, pos, p_threshold=1e-3, r2_threshold=0.05, window_bp=10000.0,
    )
    assert idx.dtype == np.int64
    assert off.shape == (idx.shape[0] + 1,)
    assert off[0] == 0
    assert off[-1] == mem.shape[0]
    # All retained index SNPs must be below the p threshold
    assert all(p[i] < 1e-3 for i in idx.tolist())


def test_clump_full_chromosome_isolation():
    """SNPs on different chromosomes should never co-clump."""
    rng = np.random.default_rng(1)
    m = 10
    R = np.eye(m, dtype=np.float64)  # zero r^2 across all pairs
    R[0, 1] = R[1, 0] = 0.99  # would clump if not for chromosome split
    p = np.full(m, 0.5)
    p[0] = 1e-10
    p[1] = 1e-10
    chr_codes = np.zeros(m, dtype=np.int64)
    chr_codes[1] = 1  # different chromosome from SNP 0
    pos = np.arange(m, dtype=np.int64)
    idx, off, mem = _ct_native.clump_full(
        p, R, chr_codes, pos, p_threshold=1e-3, r2_threshold=0.5, window_bp=10000.0,
    )
    # Both significant SNPs survive as independent index SNPs since they
    # live on different chromosomes.
    assert sorted(idx.tolist()) == [0, 1]


def test_clump_full_window_filter():
    """Pairs outside the window must not be clumped together."""
    m = 5
    R = np.full((m, m), 0.99, dtype=np.float64)
    np.fill_diagonal(R, 1.0)
    p = np.full(m, 1e-10)
    chr_codes = np.zeros(m, dtype=np.int64)
    pos = np.array([0, 100, 200, 5000, 5100], dtype=np.int64)
    idx, off, mem = _ct_native.clump_full(
        p, R, chr_codes, pos, p_threshold=1e-3, r2_threshold=0.5, window_bp=500.0,
    )
    # Two clusters: positions 0-200 and 5000-5100. Each yields one index SNP.
    assert idx.shape[0] == 2


# ---------------------------------------------------------------------------
# Statistical equivalence: native path vs Python reference
# ---------------------------------------------------------------------------


def test_clump_full_native_matches_python(monkeypatch):
    """Greedy clumping is deterministic — native and Python must match exactly."""
    ss, ld = _make_ss_and_ld(m=40, n=1500, seed=11, mode="full")

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    idx_n, mem_n = _clump_with_ld_reference_dispatch(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.05, window_kb=1.0,
    )

    idx_p, mem_p = _clump_with_ld_reference(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.05, window_kb=1.0,
    )
    assert idx_n == idx_p
    assert [sorted(x) for x in mem_n] == [sorted(x) for x in mem_p]


def test_clump_block_native_matches_python(monkeypatch):
    """Same equivalence on a block-diagonal LD reference."""
    ss, ld = _make_ss_and_ld(m=45, n=1500, seed=13, mode="block")

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    idx_n, mem_n = _clump_with_ld_reference_dispatch(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.05, window_kb=1.0,
    )
    idx_p, mem_p = _clump_with_ld_reference(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.05, window_kb=1.0,
    )
    assert idx_n == idx_p
    assert [sorted(x) for x in mem_n] == [sorted(x) for x in mem_p]


def test_ct_fit_native_vs_python_identical(monkeypatch):
    """End-to-end ClumpingThresholding.fit must produce identical weights."""
    ss, ld = _make_ss_and_ld(m=40, n=1500, seed=21, mode="full")

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    res_native = ClumpingThresholding(seed=0).fit(
        ss, ld, p_threshold=0.05, r2_threshold=0.1, window_kb=1.0,
    )

    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    res_python = ClumpingThresholding(seed=0).fit(
        ss, ld, p_threshold=0.05, r2_threshold=0.1, window_kb=1.0,
    )

    torch.testing.assert_close(res_native.weight, res_python.weight)


def test_ct_dispatch_routes_to_native(monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    ss, ld = _make_ss_and_ld(m=20, n=800, seed=3, mode="full")
    idx, mem = _clump_with_ld_reference_dispatch(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.5, window_kb=1.0,
    )
    assert isinstance(idx, list)
    assert isinstance(mem, list)
    assert all(isinstance(m, list) for m in mem)


def test_ct_dispatch_python_when_disabled(monkeypatch):
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    ss, ld = _make_ss_and_ld(m=20, n=800, seed=4, mode="full")
    idx, mem = _clump_with_ld_reference_dispatch(
        ss.p, ld, chr_labels=ss.chr, pos=ss.pos,
        r2_threshold=0.1, p_threshold=0.5, window_kb=1.0,
    )
    assert isinstance(idx, list)
