"""Tests for the native C++ impute_knn accelerator
(``torchgenomics._native._impute_knn_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import (
    HAS_NATIVE_IMPUTE_KNN,
    _impute_knn_native,
)
from torchgenomics.preprocess.impute import (
    _impute_knn_native_enabled,
    impute_knn,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_IMPUTE_KNN,
    reason="Native impute_knn extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_IMPUTE_KNN is True
    assert hasattr(_impute_knn_native, "impute_knn_fill")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _impute_knn_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_simple_weighted_fill():
    # 3 samples, 1 SNP. Sample 0 missing; neighbours 1 (val 0, K=0.8) and
    # 2 (val 2, K=0.2). Weighted mean = (0.8*0 + 0.2*2)/(1.0) = 0.4.
    G_in = np.array([[np.nan], [0.0], [2.0]], dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    K = np.array(
        [[1.0, 0.8, 0.2],
         [0.8, 1.0, 0.5],
         [0.2, 0.5, 1.0]], dtype=np.float64,
    )
    cm = np.array([1.0], dtype=np.float64)
    _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)
    assert np.isclose(G_out[0, 0], 0.4)


def test_native_negative_K_clamped_to_zero():
    G_in = np.array([[np.nan], [0.0], [2.0]], dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    # Both K weights negative -> clamped to 0 -> w_sum == 0 -> unweighted mean.
    K = np.array(
        [[1.0, -0.5, -0.1],
         [-0.5, 1.0, 0.0],
         [-0.1, 0.0, 1.0]], dtype=np.float64,
    )
    cm = np.array([7.0], dtype=np.float64)  # not used here
    _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)
    assert np.isclose(G_out[0, 0], 1.0)  # mean of [0, 2]


def test_native_all_neighbours_nan_uses_col_mean():
    G_in = np.array([[np.nan], [np.nan], [np.nan]], dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    K = np.eye(3, dtype=np.float64)
    cm = np.array([3.14], dtype=np.float64)
    _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)
    assert np.allclose(G_out, 3.14)


def test_native_no_missing_no_change():
    G_in = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    G_out = G_in.copy()
    G_copy = G_in.copy()
    knn = np.array([[1], [0]], dtype=np.int64)
    K = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    cm = np.array([1.0, 0.5], dtype=np.float64)
    _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)
    assert np.array_equal(G_out, G_copy)


def test_native_neighbour_order_independent():
    # Confirms the C++ port reads from G_in (not G_out) so imputation order
    # does not contaminate later neighbour lookups. Two samples both missing
    # at the same SNP; they reference each other in their kNN lists.
    G_in = np.array([[np.nan, 5.0], [np.nan, 5.0], [1.0, 5.0]],
                    dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    K = np.eye(3, dtype=np.float64)  # all neighbours weighted equally
    cm = np.array([1.0, 5.0], dtype=np.float64)
    _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)
    # Neither sample 0 nor sample 1 should see each other's *imputed* value;
    # they fall back to the only valid neighbour, sample 2 (value 1).
    assert np.isclose(G_out[0, 0], 1.0)
    assert np.isclose(G_out[1, 0], 1.0)


def test_native_invalid_shape_raises():
    G_in = np.array([0.0, 1.0], dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[0]], dtype=np.int64)
    K = np.array([[1.0]], dtype=np.float64)
    cm = np.array([0.0], dtype=np.float64)
    with pytest.raises(Exception):
        _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)


def test_native_K_mismatch_raises():
    G_in = np.array([[np.nan], [0.0]], dtype=np.float64)
    G_out = G_in.copy()
    knn = np.array([[1], [0]], dtype=np.int64)
    K = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
                 dtype=np.float64)
    cm = np.array([0.0], dtype=np.float64)
    with pytest.raises(Exception):
        _impute_knn_native.impute_knn_fill(G_in, G_out, knn, K, cm)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _make_K(G_complete: torch.Tensor) -> torch.Tensor:
    """Cheap GRM-like similarity from a complete dosage matrix."""
    Gc = G_complete - G_complete.mean(dim=0, keepdim=True)
    K = (Gc @ Gc.T) / Gc.shape[1]
    return K


def _python_reference(G, K, k):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return impute_knn(G, K, k=k)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _native_dispatch(G, K, k):
    os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    return impute_knn(G, K, k=k)


def test_dispatch_native_matches_python_basic():
    torch.manual_seed(11)
    G_full = torch.randint(0, 3, (40, 30), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.1
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), K, k=5)
    out_p = _python_reference(G.clone(), K, k=5)
    assert torch.allclose(out_n, out_p, atol=1e-10)


def test_dispatch_native_matches_python_high_missing():
    torch.manual_seed(23)
    G_full = torch.randint(0, 3, (60, 50), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.4
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), K, k=7)
    out_p = _python_reference(G.clone(), K, k=7)
    assert torch.allclose(out_n, out_p, atol=1e-10)


def test_dispatch_native_matches_python_polyploid():
    torch.manual_seed(31)
    # Tetraploid dosages 0..4
    G_full = torch.randint(0, 5, (50, 40), dtype=torch.float64)
    K = _make_K(G_full)
    G = G_full.clone()
    nan_mask = torch.rand(G.shape) < 0.15
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone(), K, k=4)
    out_p = _python_reference(G.clone(), K, k=4)
    assert torch.allclose(out_n, out_p, atol=1e-10)
