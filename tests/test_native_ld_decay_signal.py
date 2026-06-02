"""Tests for the native C++ ld_decay_signal accelerator
(``torchgenomics._native._ld_decay_signal_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import (
    HAS_NATIVE_LD_DECAY_SIGNAL,
    _ld_decay_signal_native,
)
from torchgenomics.ld._changepoint import (
    _ld_decay_signal_native_enabled,
    ld_decay_signal,
)

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_LD_DECAY_SIGNAL,
    reason="Native ld_decay_signal extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_LD_DECAY_SIGNAL is True
    assert hasattr(_ld_decay_signal_native, "ld_decay_signal")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _ld_decay_signal_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_adjacent_pairs_uniform():
    # All adjacent pairs with r2 = 0.6 -> every interior variant has
    # exactly two distance-1 neighbors and the edges have one each.
    m = 5
    ii = np.arange(m - 1, dtype=np.int64)
    jj = np.arange(1, m, dtype=np.int64)
    r2 = np.full(m - 1, 0.6, dtype=np.float64)
    out = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, m, 5)
    assert out.shape == (m,)
    # Each variant participates in at least one pair, mean is 0.6 everywhere.
    assert np.allclose(out, 0.6)


def test_native_isolated_variant_returns_zero():
    # Variant 3 touches no pair.
    ii = np.array([0, 1], dtype=np.int64)
    jj = np.array([1, 2], dtype=np.int64)
    r2 = np.array([0.5, 0.5], dtype=np.float64)
    out = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, 4, 5)
    assert out[3] == 0.0


def test_native_topk_picks_nearest_distances():
    # 4 variants with two pairs touching variant 1: distance 1 (r2=0.9) and
    # distance 2 (r2=0.1). With k=1 we should keep only the distance-1 pair.
    ii = np.array([0, 1], dtype=np.int64)
    jj = np.array([1, 3], dtype=np.int64)
    r2 = np.array([0.9, 0.1], dtype=np.float64)
    out = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, 4, 1)
    # Variant 1 sees both pairs but k=1 should keep the closer (dist 1, r2=0.9).
    assert out[1] == pytest.approx(0.9)


def test_native_invalid_shape_raises():
    ii = np.array([0, 1], dtype=np.int64)
    jj = np.array([1], dtype=np.int64)
    r2 = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(Exception):
        _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, 4, 5)


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _python_reference(ii, jj, r2, m, k):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        out = ld_decay_signal(
            torch.from_numpy(r2),
            torch.from_numpy(ii),
            torch.from_numpy(jj),
            m,
            k_neighbors=k,
        )
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]
    return out.cpu().numpy()


def test_dispatch_native_matches_python_adjacent():
    rng = np.random.default_rng(0)
    m = 30
    ii = np.arange(m - 1, dtype=np.int64)
    jj = np.arange(1, m, dtype=np.int64)
    r2 = rng.uniform(0.0, 1.0, size=m - 1).astype(np.float64)

    out_native = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, m, 5)
    out_python = _python_reference(ii, jj, r2, m, 5)
    assert np.allclose(out_native, out_python, atol=1e-12)


def test_dispatch_native_matches_python_random_pairs():
    rng = np.random.default_rng(7)
    m = 25
    pairs = []
    for i in range(m):
        for j in range(i + 1, min(i + 4, m)):
            pairs.append((i, j))
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    r2 = rng.uniform(0.0, 1.0, size=len(pairs)).astype(np.float64)

    out_native = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, m, 3)
    out_python = _python_reference(ii, jj, r2, m, 3)
    assert np.allclose(out_native, out_python, atol=1e-12)


def test_dispatch_native_matches_python_dense():
    rng = np.random.default_rng(99)
    m = 12
    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            pairs.append((i, j))
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    r2 = rng.uniform(0.0, 1.0, size=len(pairs)).astype(np.float64)

    out_native = _ld_decay_signal_native.ld_decay_signal(ii, jj, r2, m, 4)
    out_python = _python_reference(ii, jj, r2, m, 4)
    assert np.allclose(out_native, out_python, atol=1e-12)
