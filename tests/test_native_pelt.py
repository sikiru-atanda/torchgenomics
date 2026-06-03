"""Tests for the native C++ PELT change-point accelerator
(``torchgenomics._native._pelt_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_PELT, _pelt_native
from torchgenomics.ld._changepoint import _native_enabled, dp_changepoint

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_PELT,
    reason="Native PELT extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_PELT is True
    assert hasattr(_pelt_native, "pelt_dp_changepoint")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def _step_signal(n_per: int = 50, levels=(0.1, 0.8, 0.2), noise: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    parts = [rng.normal(level, noise, n_per) for level in levels]
    return np.concatenate(parts)


def test_native_pelt_recovers_step_changepoints():
    sig = _step_signal(n_per=40, levels=(0.1, 0.9, 0.2), noise=0.02, seed=0)
    cps = _pelt_native.pelt_dp_changepoint(sig, penalty=2.0, min_seg=3, cost_fn="gaussian")
    # Two true change-points around index 40 and 80; allow ±5 slack.
    assert len(cps) >= 2
    cps_sorted = sorted(cps)
    assert any(abs(c - 40) <= 5 for c in cps_sorted)
    assert any(abs(c - 80) <= 5 for c in cps_sorted)


def test_native_pelt_no_changepoints_on_constant_signal():
    sig = np.full(60, 0.3, dtype=np.float64)
    cps = _pelt_native.pelt_dp_changepoint(sig, penalty=1.0, min_seg=3, cost_fn="gaussian")
    assert cps == []


def test_native_pelt_short_signal_returns_empty():
    sig = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    cps = _pelt_native.pelt_dp_changepoint(sig, penalty=1.0, min_seg=2, cost_fn="gaussian")
    assert cps == []


def test_native_pelt_poisson_cost():
    rng = np.random.default_rng(1)
    seg1 = rng.poisson(2, 40).astype(np.float64)
    seg2 = rng.poisson(8, 40).astype(np.float64)
    sig = np.concatenate([seg1, seg2])
    cps = _pelt_native.pelt_dp_changepoint(sig, penalty=4.0, min_seg=3, cost_fn="poisson")
    assert any(abs(c - 40) <= 6 for c in cps)


def test_native_pelt_invalid_cost_raises():
    with pytest.raises(Exception):
        _pelt_native.pelt_dp_changepoint(
            np.zeros(20, dtype=np.float64), penalty=1.0, min_seg=2, cost_fn="bogus"
        )


# ---------------------------------------------------------------------------
# Equivalence: native vs Python reference via dispatcher
# ---------------------------------------------------------------------------


def _run_python(sig_t, penalty, min_seg, cost_fn, monkeypatch):
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return list(dp_changepoint(sig_t, penalty=penalty, min_seg=min_seg, cost_fn=cost_fn))


def _run_native(sig_t, penalty, min_seg, cost_fn, monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    return list(dp_changepoint(sig_t, penalty=penalty, min_seg=min_seg, cost_fn=cost_fn))


def test_dispatch_native_matches_python_step(monkeypatch):
    sig_np = _step_signal(n_per=30, levels=(0.0, 1.0, 0.0, 1.0), noise=0.03, seed=2)
    sig = torch.from_numpy(sig_np)
    cps_n = _run_native(sig, 2.0, 3, "gaussian", monkeypatch)
    cps_p = _run_python(sig, 2.0, 3, "gaussian", monkeypatch)
    assert cps_n == cps_p


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(7)
    sig = torch.from_numpy(rng.normal(size=120))
    cps_n = _run_native(sig, 1.5, 2, "gaussian", monkeypatch)
    cps_p = _run_python(sig, 1.5, 2, "gaussian", monkeypatch)
    assert cps_n == cps_p


def test_dispatch_native_matches_python_poisson(monkeypatch):
    rng = np.random.default_rng(11)
    sig_np = np.concatenate([
        rng.poisson(3, 50).astype(np.float64),
        rng.poisson(10, 50).astype(np.float64),
        rng.poisson(2, 50).astype(np.float64),
    ])
    sig = torch.from_numpy(sig_np)
    cps_n = _run_native(sig, 5.0, 3, "poisson", monkeypatch)
    cps_p = _run_python(sig, 5.0, 3, "poisson", monkeypatch)
    assert cps_n == cps_p
