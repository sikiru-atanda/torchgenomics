"""Tests for the native C++ LDpred2 accelerator (torchgwas._native._ldpred2_native).

These tests are skipped when the compiled extension is unavailable so that CI
on machines without a C++ toolchain still runs.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_LDPRED2, _ldpred2_native
from torchgwas.pgs.ld_ref import build_ld_reference
from torchgwas.pgs.ldpred2 import (
    LDpred2Auto,
    LDpred2Grid,
    _ldpred2_gibbs_block_dispatch,
    _native_enabled,
)
from torchgwas.postgwas._sumstats import SumStats

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_LDPRED2,
    reason="Native LDpred2 extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _make_ss_and_ld(m=20, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)
    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[0] = 0.3
    beta_true[5] = -0.25
    y = G @ beta_true + torch.randn(n, generator=g, dtype=torch.float64) * 0.5
    y = (y - y.mean()) / y.std().clamp(min=1e-12)
    beta_hat = (G * y.unsqueeze(1)).sum(0) / n
    se = torch.full((m,), 1.0 / math.sqrt(n), dtype=torch.float64)
    z = beta_hat / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(z.abs() / math.sqrt(2.0))))
    ss = SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
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
    ld = build_ld_reference(G, meta, mode="full")
    return ss, ld, beta_true


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_LDPRED2 is True
    assert hasattr(_ldpred2_native, "ldpred2_gibbs_block")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# C++ block sampler direct
# ---------------------------------------------------------------------------


def test_native_ldpred2_block_runs():
    rng = np.random.default_rng(0)
    m = 30
    G = rng.standard_normal((200, m))
    G = (G - G.mean(0)) / G.std(0)
    R = (G.T @ G / 199).astype(np.float64)
    bs = (rng.standard_normal(m) * 0.05).astype(np.float64)
    beta_mean, causal_mean, p_trace, h_trace = _ldpred2_native.ldpred2_gibbs_block(
        bs, R,
        p_causal=0.1, h2=0.3, n_eff=200.0,
        n_iter=200, n_burnin=100,
        sparse=False, update_hyperparams=False,
        seed=42,
    )
    assert beta_mean.shape == (m,)
    assert causal_mean.shape == (m,)
    assert np.all(np.isfinite(beta_mean))
    assert np.all((causal_mean >= 0) & (causal_mean <= 1))
    assert len(p_trace) == 0
    assert len(h_trace) == 0


def test_native_ldpred2_block_hyperparam_update_traces():
    rng = np.random.default_rng(1)
    m = 25
    G = rng.standard_normal((200, m))
    R = (G.T @ G / 199).astype(np.float64)
    bs = (rng.standard_normal(m) * 0.05).astype(np.float64)
    _, _, p_trace, h_trace = _ldpred2_native.ldpred2_gibbs_block(
        bs, R,
        p_causal=0.05, h2=0.2, n_eff=200.0,
        n_iter=120, n_burnin=60,
        sparse=False, update_hyperparams=True,
        seed=7,
    )
    assert len(p_trace) == 120
    assert len(h_trace) == 120
    assert all(0.0 < pp < 1.0 for pp in p_trace)
    assert all(0.0 < hh <= 1.0 for hh in h_trace)


def test_native_ldpred2_block_intra_seed_determinism():
    rng = np.random.default_rng(0)
    m = 20
    G = rng.standard_normal((100, m))
    R = (G.T @ G / 99).astype(np.float64)
    bs = (rng.standard_normal(m) * 0.05).astype(np.float64)
    a = _ldpred2_native.ldpred2_gibbs_block(
        bs, R, p_causal=0.1, h2=0.3, n_eff=100.0,
        n_iter=80, n_burnin=40, sparse=False,
        update_hyperparams=False, seed=11,
    )
    b = _ldpred2_native.ldpred2_gibbs_block(
        bs, R, p_causal=0.1, h2=0.3, n_eff=100.0,
        n_iter=80, n_burnin=40, sparse=False,
        update_hyperparams=False, seed=11,
    )
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
# Statistical equivalence: native path vs Python reference
# ---------------------------------------------------------------------------


def test_ldpred2grid_native_vs_python_correlation(monkeypatch):
    ss, ld, _ = _make_ss_and_ld(m=20, n=1500, seed=11)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    res_native = LDpred2Grid(seed=0).fit(
        ss, ld, grid_p=[0.01, 0.1], grid_h2=[0.2], n_iter=200, n_burnin=100,
    )

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    res_python = LDpred2Grid(seed=0).fit(
        ss, ld, grid_p=[0.01, 0.1], grid_h2=[0.2], n_iter=200, n_burnin=100,
    )

    w_n = res_native.weight.numpy()
    w_p = res_python.weight.numpy()
    if w_n.std() < 1e-12 or w_p.std() < 1e-12:
        pytest.skip("degenerate fixture: zero weight variance")
    corr = float(np.corrcoef(w_n, w_p)[0, 1])
    assert corr > 0.85, f"weight correlation too low: {corr}"


def test_ldpred2auto_native_vs_python_correlation(monkeypatch):
    ss, ld, _ = _make_ss_and_ld(m=20, n=1500, seed=13)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    res_native = LDpred2Auto(seed=0).fit(
        ss, ld, n_iter=300, n_burnin=150, n_chains=2,
    )

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    res_python = LDpred2Auto(seed=0).fit(
        ss, ld, n_iter=300, n_burnin=150, n_chains=2,
    )

    w_n = res_native.weight.numpy()
    w_p = res_python.weight.numpy()
    if w_n.std() < 1e-12 or w_p.std() < 1e-12:
        pytest.skip("degenerate fixture: zero weight variance")
    corr = float(np.corrcoef(w_n, w_p)[0, 1])
    assert corr > 0.85, f"weight correlation too low: {corr}"


def test_ldpred2grid_native_recovers_signs(monkeypatch):
    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    ss, ld, beta_true = _make_ss_and_ld(m=20, n=2000, seed=3)
    res = LDpred2Grid(seed=0).fit(
        ss, ld, grid_p=[0.05, 0.1], grid_h2=[0.3], n_iter=200, n_burnin=100,
    )
    assert torch.sign(res.weight[0]) == torch.sign(beta_true[0])
    assert torch.sign(res.weight[5]) == torch.sign(beta_true[5])


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


def test_ldpred2_dispatch_routes_to_native(monkeypatch):
    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    rng = torch.Generator().manual_seed(0)
    m = 10
    bs = torch.randn(m, dtype=torch.float64, generator=rng) * 0.05
    R = torch.eye(m, dtype=torch.float64)
    bm, cm, traces = _ldpred2_gibbs_block_dispatch(
        bs, R, p_causal=0.1, h2=0.3, n_eff=100.0,
        n_iter=50, n_burnin=20, sparse=False, rng=rng,
    )
    assert bm.shape == (m,)
    assert cm.shape == (m,)
    assert bm.dtype == torch.float64
    assert "p" in traces and "h2" in traces


def test_ldpred2_dispatch_falls_back_for_float32(monkeypatch):
    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    rng = torch.Generator().manual_seed(0)
    m = 8
    bs = torch.randn(m, dtype=torch.float32, generator=rng) * 0.05
    R = torch.eye(m, dtype=torch.float32)
    bm, cm, traces = _ldpred2_gibbs_block_dispatch(
        bs, R, p_causal=0.1, h2=0.3, n_eff=100.0,
        n_iter=30, n_burnin=10, sparse=False, rng=rng,
    )
    assert bm.dtype == torch.float32
    assert cm.dtype == torch.float32
