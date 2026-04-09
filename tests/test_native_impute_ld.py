"""Tests for the native C++ impute_ld accelerator
(``torchgwas._native._impute_ld_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import (
    HAS_NATIVE_IMPUTE_LD,
    _impute_ld_native,
)
from torchgwas.preprocess.impute import (
    _impute_ld_native_enabled,
    impute_ld,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_IMPUTE_LD,
    reason="Native impute_ld extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_IMPUTE_LD is True
    assert hasattr(_impute_ld_native, "impute_ld_fill")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGWAS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _impute_ld_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point — small invariants
# ---------------------------------------------------------------------------


def _make_inputs(G_complete: np.ndarray, miss_mask: np.ndarray):
    cm = G_complete.mean(axis=0).astype(np.float64)
    cs = G_complete.std(axis=0, ddof=1).astype(np.float64)
    G_out = G_complete.copy()
    return cm, cs, G_out


def test_native_no_missing_no_change():
    G = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 2.0, 0.0]],
                 dtype=np.float64)
    mask = np.zeros_like(G, dtype=np.uint8)
    cm, cs, G_out = _make_inputs(G, mask)
    G_out_copy = G_out.copy()
    _impute_ld_native.impute_ld_fill(G, mask, cm, cs, 1, G_out)
    assert np.array_equal(G_out, G_out_copy)


def test_native_constant_target_uses_complete_value():
    # Target column j=0 is constant -> std == 0 -> fall back to G_complete[i,j].
    G = np.array(
        [[1.0, 0.0, 2.0],
         [1.0, 1.0, 1.0],
         [1.0, 2.0, 0.0],
         [1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    mask = np.zeros_like(G, dtype=np.uint8)
    mask[0, 0] = 1
    cm, cs, G_out = _make_inputs(G, mask)
    _impute_ld_native.impute_ld_fill(G, mask, cm, cs, 2, G_out)
    assert np.isclose(G_out[0, 0], 1.0)


def test_native_perfectly_correlated_recovers_neighbour():
    # Two perfectly correlated SNPs; missing entry's prediction equals the
    # other column's value at that row (weight on the third uncorrelated
    # column drops out by symmetry).
    rng = np.random.default_rng(0)
    base = rng.normal(size=12)
    G = np.column_stack([base, base, rng.normal(size=12)]).astype(np.float64)
    mask = np.zeros_like(G, dtype=np.uint8)
    mask[3, 0] = 1
    cm, cs, G_out = _make_inputs(G, mask)
    _impute_ld_native.impute_ld_fill(G, mask, cm, cs, 2, G_out)
    # Expected weighted prediction is heavily dominated by the perfectly
    # correlated column 1, value G[3, 1] == base[3].
    assert abs(G_out[3, 0] - base[3]) < 0.4 * abs(base[3]) + 0.5


def test_native_window_zero_falls_back_to_complete():
    G = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]], dtype=np.float64)
    mask = np.array([[1, 0], [0, 0], [0, 0]], dtype=np.uint8)
    cm, cs, G_out = _make_inputs(G, mask)
    # window=0 -> after excluding j the window is empty
    _impute_ld_native.impute_ld_fill(G, mask, cm, cs, 0, G_out)
    assert np.isclose(G_out[0, 0], G[0, 0])


def test_native_invalid_shape_raises():
    G = np.array([0.0, 1.0], dtype=np.float64)
    mask = np.array([0, 0], dtype=np.uint8)
    cm = np.array([0.5], dtype=np.float64)
    cs = np.array([0.5], dtype=np.float64)
    G_out = np.array([0.0, 1.0], dtype=np.float64)
    with pytest.raises(Exception):
        _impute_ld_native.impute_ld_fill(G, mask, cm, cs, 1, G_out)


def test_native_invalid_window_raises():
    G = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    mask = np.zeros_like(G, dtype=np.uint8)
    cm, cs, G_out = _make_inputs(G, mask)
    with pytest.raises(Exception):
        _impute_ld_native.impute_ld_fill(G, mask, cm, cs, -1, G_out)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _python_reference(G, window_size):
    os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    try:
        return impute_ld(G, window_size=window_size)
    finally:
        del os.environ["TORCHGWAS_DISABLE_NATIVE"]


def _native_dispatch(G, window_size):
    os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    return impute_ld(G, window_size=window_size)


def test_dispatch_native_matches_python_basic():
    torch.manual_seed(101)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.1
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), window_size=5)
    out_p = _python_reference(G.clone(), window_size=5)
    assert torch.allclose(out_n, out_p, atol=1e-9)


def test_dispatch_native_matches_python_wide_window():
    torch.manual_seed(202)
    G_full = torch.randint(0, 3, (50, 60), dtype=torch.float64)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.15
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), window_size=20)
    out_p = _python_reference(G.clone(), window_size=20)
    assert torch.allclose(out_n, out_p, atol=1e-9)


def test_dispatch_native_matches_python_polyploid():
    torch.manual_seed(303)
    G_full = torch.randint(0, 5, (45, 35), dtype=torch.float64)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.2
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), window_size=10)
    out_p = _python_reference(G.clone(), window_size=10)
    assert torch.allclose(out_n, out_p, atol=1e-9)
