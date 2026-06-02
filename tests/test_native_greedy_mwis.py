"""Tests for the native C++ greedy_mwis accelerator
(``torchgenomics._native._greedy_mwis_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from torchgenomics._native import HAS_NATIVE_GREEDY_MWIS, _greedy_mwis_native
from torchgenomics.ld._graph_utils import (
    _greedy_mwis_native_enabled,
    greedy_mwis,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_GREEDY_MWIS,
    reason="Native greedy_mwis extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_GREEDY_MWIS is True
    assert hasattr(_greedy_mwis_native, "greedy_mwis")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _greedy_mwis_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_empty_input():
    s = np.array([], dtype=np.int64)
    e = np.array([], dtype=np.int64)
    w = np.array([], dtype=np.float64)
    sel_s, sel_e, sel_w = _greedy_mwis_native.greedy_mwis(s, e, w)
    assert sel_s.shape == (0,)


def test_native_disjoint_intervals_all_kept():
    s = np.array([0, 5, 10], dtype=np.int64)
    e = np.array([3, 8, 15], dtype=np.int64)
    w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    sel_s, sel_e, sel_w = _greedy_mwis_native.greedy_mwis(s, e, w)
    assert sel_s.tolist() == [0, 5, 10]


def test_native_overlapping_keeps_heaviest():
    # Two overlapping intervals — heaviest wins.
    s = np.array([0, 1], dtype=np.int64)
    e = np.array([5, 6], dtype=np.int64)
    w = np.array([1.0, 5.0], dtype=np.float64)
    sel_s, sel_e, sel_w = _greedy_mwis_native.greedy_mwis(s, e, w)
    assert sel_s.tolist() == [1]
    assert sel_e.tolist() == [6]
    assert sel_w.tolist() == [5.0]


def test_native_touching_intervals_overlap():
    # Closed intervals: end of one == start of next counts as overlap.
    s = np.array([0, 5], dtype=np.int64)
    e = np.array([5, 10], dtype=np.int64)
    w = np.array([1.0, 2.0], dtype=np.float64)
    sel_s, sel_e, sel_w = _greedy_mwis_native.greedy_mwis(s, e, w)
    assert sel_s.tolist() == [5]


def test_native_invalid_shape_raises():
    s = np.array([0, 1], dtype=np.int64)
    e = np.array([2], dtype=np.int64)
    w = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(Exception):
        _greedy_mwis_native.greedy_mwis(s, e, w)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _python_reference(intervals):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return greedy_mwis(intervals)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _native_dispatch(intervals):
    os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    return greedy_mwis(intervals)


def test_dispatch_native_matches_python_random_intervals():
    rng = random.Random(0)
    intervals = []
    for _ in range(50):
        a = rng.randint(0, 100)
        b = a + rng.randint(0, 20)
        w = rng.uniform(0.1, 5.0)
        intervals.append((a, b, w))
    out_native = _native_dispatch(intervals)
    out_python = _python_reference(intervals)
    assert out_native == out_python


def test_dispatch_native_matches_python_dense_overlap():
    # Many heavily overlapping intervals
    intervals = [(i, i + 5, float(i % 7 + 1)) for i in range(40)]
    out_native = _native_dispatch(intervals)
    out_python = _python_reference(intervals)
    assert out_native == out_python


def test_dispatch_native_matches_python_tied_weights():
    # Ties: stable sort should pick the input-order interval first.
    intervals = [
        (0, 4, 1.0),
        (2, 6, 1.0),
        (4, 8, 1.0),
        (6, 10, 1.0),
        (8, 12, 1.0),
    ]
    out_native = _native_dispatch(intervals)
    out_python = _python_reference(intervals)
    assert out_native == out_python
