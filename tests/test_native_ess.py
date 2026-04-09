"""Tests for the native C++ Geyer-ESS accelerator
(``torchgwas._native._ess_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_ESS, _ess_native
from torchgwas.pgs.diagnostics import _native_enabled, ess

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_ESS,
    reason="Native ESS extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_ESS is True
    assert hasattr(_ess_native, "geyer_initial_positive_ess")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _make_ar1_acorr(n_lags: int, m: int, rho: float) -> np.ndarray:
    """Theoretical AR(1) autocorrelation, broadcast to (n_lags, m)."""
    lags = np.arange(n_lags, dtype=np.float64)
    vec = rho ** lags
    return np.broadcast_to(vec[:, None], (n_lags, m)).copy()


def test_native_ess_uncorrelated_chain():
    """Independent samples (acorr = delta) should give ESS == total_samples."""
    n = 50
    m = 4
    acorr = np.zeros((n, m), dtype=np.float64)
    acorr[0, :] = 1.0  # lag-0 only
    out = _ess_native.geyer_initial_positive_ess(acorr, total_samples=200.0)
    assert out.shape == (m,)
    np.testing.assert_allclose(out, np.full(m, 200.0))


def test_native_ess_ar1_decreases_with_rho():
    """Stronger AR(1) correlation should yield smaller ESS."""
    acorr_low = _make_ar1_acorr(80, 3, rho=0.2)
    acorr_high = _make_ar1_acorr(80, 3, rho=0.8)
    out_low = _ess_native.geyer_initial_positive_ess(acorr_low, total_samples=400.0)
    out_high = _ess_native.geyer_initial_positive_ess(acorr_high, total_samples=400.0)
    assert np.all(out_high < out_low)
    # ESS bounded above by total_samples
    assert np.all(out_low <= 400.0 + 1e-9)
    assert np.all(out_high > 0.0)


def test_native_ess_clamped_to_one_when_tau_below_one():
    """If pair sum is non-positive at lag 1, tau stays at 1 → ESS = total."""
    n = 20
    m = 2
    acorr = np.zeros((n, m), dtype=np.float64)
    acorr[0, :] = 1.0
    acorr[1, :] = -0.1  # already negative pair sum at the first probe
    out = _ess_native.geyer_initial_positive_ess(acorr, total_samples=100.0)
    np.testing.assert_allclose(out, np.full(m, 100.0))


def test_native_ess_invalid_dim_raises():
    with pytest.raises(Exception):
        _ess_native.geyer_initial_positive_ess(
            np.zeros(10, dtype=np.float64), total_samples=10.0
        )


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _make_chains(M: int, N: int, m: int, rho: float, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    chains = np.empty((M, N, m), dtype=np.float64)
    for c in range(M):
        x = np.zeros((N, m))
        x[0] = rng.standard_normal(m)
        for t in range(1, N):
            x[t] = rho * x[t - 1] + rng.standard_normal(m)
        chains[c] = x
    return torch.from_numpy(chains)


def test_dispatch_native_matches_python(monkeypatch):
    chains = _make_chains(M=3, N=200, m=5, rho=0.6, seed=1)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    out_native = ess(chains)

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    out_python = ess(chains)

    torch.testing.assert_close(out_native, out_python)


def test_dispatch_native_matches_python_2d(monkeypatch):
    """Single-parameter (2-D) input must match too."""
    chains = _make_chains(M=2, N=150, m=1, rho=0.4, seed=2).squeeze(-1)

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    out_native = ess(chains)

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    out_python = ess(chains)

    torch.testing.assert_close(out_native, out_python)


def test_dispatch_native_matches_python_uncorrelated(monkeypatch):
    rng = np.random.default_rng(3)
    chains = torch.from_numpy(rng.standard_normal((4, 300, 6)))

    monkeypatch.delenv("TORCHGWAS_DISABLE_NATIVE", raising=False)
    out_native = ess(chains)

    monkeypatch.setenv("TORCHGWAS_DISABLE_NATIVE", "1")
    out_python = ess(chains)

    torch.testing.assert_close(out_native, out_python)
