"""Tests for the native C++ graph-utils accelerator
(``torchgenomics._native._graph_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_GRAPH, _graph_native
from torchgenomics.ld._graph_utils import _native_enabled, connected_components

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_GRAPH,
    reason="Native graph-utils extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_GRAPH is True
    assert hasattr(_graph_native, "connected_components")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_cc_single_component():
    """Fully connected graph → one component."""
    m = 6
    A = (np.ones((m, m)) - np.eye(m)).astype(np.float64)
    comps = _graph_native.connected_components(A, threshold=0.0)
    assert len(comps) == 1
    assert sorted(comps[0]) == list(range(m))


def test_native_cc_disjoint_clusters():
    """Two disconnected blocks → two components."""
    m = 8
    A = np.zeros((m, m), dtype=np.float64)
    # Block 1: nodes 0..3
    for i in range(4):
        for j in range(4):
            if i != j:
                A[i, j] = 1.0
    # Block 2: nodes 4..7
    for i in range(4, 8):
        for j in range(4, 8):
            if i != j:
                A[i, j] = 1.0
    comps = _graph_native.connected_components(A, threshold=0.0)
    assert len(comps) == 2
    assert sorted(comps[0]) == [0, 1, 2, 3]
    assert sorted(comps[1]) == [4, 5, 6, 7]


def test_native_cc_threshold_filters_weak_edges():
    """Edges below the threshold should not connect components."""
    A = np.array([
        [0.0, 0.9, 0.05, 0.05],
        [0.9, 0.0, 0.05, 0.05],
        [0.05, 0.05, 0.0, 0.9],
        [0.05, 0.05, 0.9, 0.0],
    ], dtype=np.float64)
    comps = _graph_native.connected_components(A, threshold=0.5)
    assert len(comps) == 2
    assert sorted(comps[0]) == [0, 1]
    assert sorted(comps[1]) == [2, 3]


def test_native_cc_isolated_nodes():
    """All-zero adjacency → m singleton components."""
    m = 5
    A = np.zeros((m, m), dtype=np.float64)
    comps = _graph_native.connected_components(A, threshold=0.0)
    assert len(comps) == m
    assert all(len(c) == 1 for c in comps)


def test_native_cc_invalid_shape_raises():
    with pytest.raises(Exception):
        _graph_native.connected_components(
            np.zeros((3, 4), dtype=np.float64), threshold=0.0
        )


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference
# ---------------------------------------------------------------------------


def _run_python(A, threshold, monkeypatch):
    monkeypatch.setenv("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return [list(c) for c in connected_components(A, threshold=threshold)]


def _run_native(A, threshold, monkeypatch):
    monkeypatch.delenv("TORCHGENOMICS_DISABLE_NATIVE", raising=False)
    return [list(c) for c in connected_components(A, threshold=threshold)]


def test_dispatch_native_matches_python_random(monkeypatch):
    rng = np.random.default_rng(0)
    m = 30
    M = rng.uniform(0, 1, size=(m, m))
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    A = torch.from_numpy(M)
    cn = _run_native(A, 0.6, monkeypatch)
    cp = _run_python(A, 0.6, monkeypatch)
    # Sort both representations canonically
    cn_canon = sorted([sorted(c) for c in cn])
    cp_canon = sorted([sorted(c) for c in cp])
    assert cn_canon == cp_canon


def test_dispatch_native_matches_python_chain(monkeypatch):
    """Chain graph: nodes i—(i+1) with random extra edges."""
    rng = np.random.default_rng(1)
    m = 20
    A_np = np.zeros((m, m), dtype=np.float64)
    for i in range(m - 1):
        if rng.uniform() < 0.5:
            A_np[i, i + 1] = 1.0
            A_np[i + 1, i] = 1.0
    A = torch.from_numpy(A_np)
    cn = _run_native(A, 0.0, monkeypatch)
    cp = _run_python(A, 0.0, monkeypatch)
    cn_canon = sorted([sorted(c) for c in cn])
    cp_canon = sorted([sorted(c) for c in cp])
    assert cn_canon == cp_canon


def test_dispatch_native_matches_python_isolated(monkeypatch):
    A = torch.zeros((10, 10), dtype=torch.float64)
    cn = _run_native(A, 0.0, monkeypatch)
    cp = _run_python(A, 0.0, monkeypatch)
    assert cn == cp
