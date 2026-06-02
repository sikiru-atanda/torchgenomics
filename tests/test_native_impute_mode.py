"""Tests for the native C++ impute_mode accelerator
(``torchgenomics._native._impute_mode_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import (
    HAS_NATIVE_IMPUTE_MODE,
    _impute_mode_native,
)
from torchgenomics.preprocess.impute import (
    _impute_mode_native_enabled,
    impute_mode,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_IMPUTE_MODE,
    reason="Native impute_mode extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_IMPUTE_MODE is True
    assert hasattr(_impute_mode_native, "impute_mode_columns")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _impute_mode_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_simple_mode():
    G = np.array(
        [[0, 2],
         [0, 1],
         [0, 1],
         [1, 2]], dtype=np.int64,
    )
    out = _impute_mode_native.impute_mode_columns(G, 2)
    assert out.tolist() == [0.0, 1.0]


def test_native_skip_negative():
    G = np.array(
        [[0, -1],
         [-1, 1],
         [0, 1],
         [-1, 2]], dtype=np.int64,
    )
    out = _impute_mode_native.impute_mode_columns(G, 2)
    # col 0 valid = [0, 0] -> mode 0
    # col 1 valid = [1, 1, 2] -> mode 1
    assert out.tolist() == [0.0, 1.0]


def test_native_all_missing_column_returns_zero():
    G = np.array([[-1, 0], [-1, 1], [-1, 1]], dtype=np.int64)
    out = _impute_mode_native.impute_mode_columns(G, 2)
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_native_tie_break_lowest():
    # Counts tied -> torch.argmax picks the first (lowest) bin.
    G = np.array(
        [[0, 0, 1, 1, 2, 2]], dtype=np.int64,
    ).T  # (6, 1)
    out = _impute_mode_native.impute_mode_columns(G, 2)
    assert out[0] == 0.0


def test_native_invalid_shape_raises():
    G = np.array([0, 1, 2], dtype=np.int64)  # 1-D
    with pytest.raises(Exception):
        _impute_mode_native.impute_mode_columns(G, 2)


def test_native_invalid_max_dosage_raises():
    G = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(Exception):
        _impute_mode_native.impute_mode_columns(G, -1)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _python_reference(G):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return impute_mode(G)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _native_dispatch(G):
    os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    return impute_mode(G)


def test_dispatch_native_matches_python_diploid():
    torch.manual_seed(7)
    G = torch.randint(0, 3, (200, 50), dtype=torch.float64)
    nan_mask = torch.rand(G.shape) < 0.1
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone())
    out_p = _python_reference(G.clone())
    assert torch.allclose(out_n, out_p)


def test_dispatch_native_matches_python_polyploid():
    torch.manual_seed(13)
    # Tetraploid: dosages 0..4
    G = torch.randint(0, 5, (150, 30), dtype=torch.float64)
    nan_mask = torch.rand(G.shape) < 0.05
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone())
    out_p = _python_reference(G.clone())
    assert torch.allclose(out_n, out_p)


def test_dispatch_native_matches_python_high_missing():
    torch.manual_seed(21)
    G = torch.randint(0, 3, (80, 40), dtype=torch.float64)
    # 60% missing
    nan_mask = torch.rand(G.shape) < 0.6
    G[nan_mask] = float("nan")
    out_n = _native_dispatch(G.clone())
    out_p = _python_reference(G.clone())
    assert torch.allclose(out_n, out_p)
