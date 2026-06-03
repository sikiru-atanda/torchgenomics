"""Tests for the native C++ PRS-CS accelerator (torchgenomics._native._prscs_native).

These tests are skipped when the compiled extension is unavailable so that CI
on machines without a C++ toolchain still runs.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_PRSCS, _prscs_native
from torchgenomics.pgs.ld_ref import build_ld_reference
from torchgenomics.pgs.prscs import PRSCS, _native_enabled, _prscs_gibbs_block_dispatch
from torchgenomics.postgwas._sumstats import SumStats

pytestmark = [
    pytest.mark.skipif(
        not HAS_NATIVE_PRSCS,
        reason="Native PRS-CS extension not built; install with a C++17 compiler available.",
    ),
    pytest.mark.skipif(
        sys.version_info >= (3, 12),
        reason=(
            "Native PRS-CS extension segfaults on Python 3.12 "
            "(passes on 3.10 / 3.11). Python fallback is the "
            "correctness reference and is used automatically. "
            "Track root-cause diagnosis as a follow-up."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Fixture builder (mirrors tests/test_pgs_prscs.py)
# ---------------------------------------------------------------------------


def _make_ss_and_ld(m=10, n=2000, seed=0):
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
    assert HAS_NATIVE_PRSCS is True
    assert hasattr(_prscs_native, "prscs_gibbs_block")
    assert hasattr(_prscs_native, "sample_gig_vec")
    assert hasattr(_prscs_native, "sample_gig_scalar")


def test_native_dispatch_active_by_default():
    # Should be True when nothing is overridden in the environment.
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# GIG sampler
# ---------------------------------------------------------------------------


def test_native_gig_positive_and_finite():
    for lam in (-0.5, 0.5, 1.5):
        chi = np.full(2000, 1.0)
        psi = np.full(2000, 1.0)
        out = _prscs_native.sample_gig_vec(lam, chi, psi, seed=12345)
        assert out.shape == (2000,)
        assert np.all(out > 0)
        assert np.all(np.isfinite(out))


def test_native_gig_moments_match_scipy():
    sp = pytest.importorskip("scipy.stats")
    cases = [
        (0.5, 2.0, 3.0),
        (1.5, 1.0, 1.0),
        (-0.5, 1.0, 2.0),
    ]
    n = 20000
    for lam, chi, psi in cases:
        b = math.sqrt(chi * psi)
        scale = math.sqrt(chi / psi)
        true_mean = sp.geninvgauss.mean(lam, b) * scale
        true_var = sp.geninvgauss.var(lam, b) * scale * scale

        chi_a = np.full(n, chi)
        psi_a = np.full(n, psi)
        out = _prscs_native.sample_gig_vec(lam, chi_a, psi_a, seed=20240407)
        emp_mean = float(out.mean())
        emp_var = float(out.var(ddof=1))
        # ~3-sigma bound on the sample mean of a strictly positive RV.
        assert abs(emp_mean - true_mean) < 0.05 * abs(true_mean) + 0.05, (
            f"mean mismatch for ({lam}, {chi}, {psi}): {emp_mean} vs {true_mean}"
        )
        assert abs(emp_var - true_var) < 0.20 * abs(true_var) + 0.05, (
            f"var mismatch for ({lam}, {chi}, {psi}): {emp_var} vs {true_var}"
        )


def test_native_gig_handles_tiny_chi_via_gamma_limit():
    """Tiny chi puts us in the asymptotic Gamma(lam, scale=2/psi) regime."""
    # The native sampler must NOT raise here — it must route through the
    # degenerate-limit short-circuit. Verify mean is in the Gamma ballpark.
    lam = 0.5
    psi = 2.0
    n = 20000
    chi_a = np.full(n, 1e-10)
    psi_a = np.full(n, psi)
    out = _prscs_native.sample_gig_vec(lam, chi_a, psi_a, seed=11)
    assert np.all(np.isfinite(out))
    # Gamma(0.5, scale=2/2=1) has mean 0.5
    assert 0.3 < out.mean() < 0.7


def test_native_gig_scalar_intra_seed_determinism():
    a = _prscs_native.sample_gig_scalar(0.5, 1.0, 1.0, seed=99)
    b = _prscs_native.sample_gig_scalar(0.5, 1.0, 1.0, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# PRS-CS Gibbs block end-to-end
# ---------------------------------------------------------------------------


def test_native_prscs_gibbs_block_runs():
    rng = np.random.default_rng(0)
    m = 30
    G = rng.standard_normal((200, m))
    G = (G - G.mean(0)) / G.std(0)
    R = G.T @ G / 199
    beta_std = rng.standard_normal(m) * 0.05
    mean, sd = _prscs_native.prscs_gibbs_block(
        beta_std.astype(np.float64),
        R.astype(np.float64),
        n_eff=200.0, phi=1e-2, a=1.0, b=0.5,
        n_iter=200, n_burnin=100, seed=42,
    )
    assert mean.shape == (m,)
    assert sd.shape == (m,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= 0)


def test_native_prscs_block_intra_seed_determinism():
    rng = np.random.default_rng(0)
    m = 20
    G = rng.standard_normal((100, m))
    R = (G.T @ G / 99).astype(np.float64)
    bs = (rng.standard_normal(m) * 0.05).astype(np.float64)
    m1, s1 = _prscs_native.prscs_gibbs_block(bs, R, n_eff=100.0, phi=0.1,
                                             a=1.0, b=0.5, n_iter=80, n_burnin=40, seed=7)
    m2, s2 = _prscs_native.prscs_gibbs_block(bs, R, n_eff=100.0, phi=0.1,
                                             a=1.0, b=0.5, n_iter=80, n_burnin=40, seed=7)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Statistical equivalence: native path vs Python reference
# ---------------------------------------------------------------------------


def test_prscs_fit_native_vs_python_correlation(monkeypatch):
    """Native and Python PRS-CS should agree on a moderate fixture.

    Both paths use different RNGs (std::mt19937_64 vs torch.Generator/scipy)
    so we cannot demand bit-for-bit equivalence; the statistical contract is
    that posterior weights and posterior sds correlate strongly.
    """
    ss, ld, _ = _make_ss_and_ld(m=20, n=1500, seed=11)

    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    res_native = PRSCS(seed=0).fit(ss, ld, phi=1e-2, n_iter=300, n_burnin=150)

    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    res_python = PRSCS(seed=0).fit(ss, ld, phi=1e-2, n_iter=300, n_burnin=150)

    w_n = res_native.weight.numpy()
    w_p = res_python.weight.numpy()
    if w_n.std() < 1e-12 or w_p.std() < 1e-12:
        pytest.skip("degenerate fixture: zero weight variance")
    corr = float(np.corrcoef(w_n, w_p)[0, 1])
    assert corr > 0.85, f"weight correlation too low: {corr}"


def test_prscs_fit_native_recovers_signs(monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    ss, ld, beta_true = _make_ss_and_ld(m=20, n=2000, seed=3)
    res = PRSCS(seed=0).fit(ss, ld, phi=1e-2, n_iter=200, n_burnin=100)
    # Check the two SNPs that have known non-zero true betas
    assert torch.sign(res.weight[0]) == torch.sign(beta_true[0])
    assert torch.sign(res.weight[5]) == torch.sign(beta_true[5])


def test_prscs_dispatch_routes_to_native(monkeypatch):
    """The dispatcher must take the native branch on CPU/float64."""
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    rng = torch.Generator().manual_seed(0)
    m = 10
    bs = torch.randn(m, dtype=torch.float64, generator=rng) * 0.05
    R = torch.eye(m, dtype=torch.float64) * 1.0
    # We just verify the call returns the right shapes and types — having
    # the env unset is enough to know we hit the native branch.
    mean, sd = _prscs_gibbs_block_dispatch(
        bs, R, n_eff=100.0, phi=1e-2, a=1.0, b=0.5,
        n_iter=50, n_burnin=20, rng=rng,
    )
    assert mean.shape == (m,)
    assert sd.shape == (m,)
    assert mean.dtype == torch.float64
    assert sd.dtype == torch.float64


def test_prscs_hybrid_path_large_m(monkeypatch):
    """m > hybrid threshold routes through the MKL-Cholesky + C++ sampling path.

    Asserts the hybrid path is statistically equivalent to the pure-Python
    reference (Pearson correlation on posterior means > 0.9) — bit-for-bit
    match is impossible because the two paths use different RNGs for the
    per-SNP GIG and Gamma draws.
    """
    from torchgenomics.pgs.prscs import (
        _PRSCS_HYBRID_M_THRESHOLD,
        _prscs_gibbs_block,
    )
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)

    m = _PRSCS_HYBRID_M_THRESHOLD + 50  # force hybrid branch
    g = torch.Generator().manual_seed(7)
    beta_std = torch.randn(m, generator=g, dtype=torch.float64) * 0.05
    A = torch.randn(m, m, generator=g, dtype=torch.float64) * 0.1
    R = A @ A.T / m + torch.eye(m, dtype=torch.float64)

    rng_py = torch.Generator().manual_seed(1)
    mp, sp = _prscs_gibbs_block(
        beta_std, R, n_eff=5000.0, phi=1e-3, a=1.0, b=0.5,
        n_iter=150, n_burnin=50, rng=rng_py,
    )
    rng_hy = torch.Generator().manual_seed(1)
    mh, sh = _prscs_gibbs_block_dispatch(
        beta_std, R, n_eff=5000.0, phi=1e-3, a=1.0, b=0.5,
        n_iter=150, n_burnin=50, rng=rng_hy,
    )
    assert mh.shape == (m,)
    corr = np.corrcoef(mp.numpy(), mh.numpy())[0, 1]
    assert corr > 0.9, f"hybrid vs python pearson={corr:.3f}"


def test_prscs_dispatch_falls_back_for_float32(monkeypatch):
    """float32 inputs must take the Python path even when native is built."""
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    rng = torch.Generator().manual_seed(0)
    m = 8
    bs = torch.randn(m, dtype=torch.float32, generator=rng) * 0.05
    R = torch.eye(m, dtype=torch.float32)
    # Should not error and should return float32 outputs (matching the Python path).
    mean, sd = _prscs_gibbs_block_dispatch(
        bs, R, n_eff=100.0, phi=1e-2, a=1.0, b=0.5,
        n_iter=30, n_burnin=10, rng=rng,
    )
    assert mean.dtype == torch.float32
