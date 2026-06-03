"""Tests for the native C++ Wall-Pritchard blockiness permutation accelerator
(``torchgenomics._native._wall_pritchard_native``).

Skipped when the compiled extension is unavailable so CI on machines
without a C++ toolchain still runs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgenomics._native import (
    HAS_NATIVE_WALL_PRITCHARD,
    _wall_pritchard_native,
)
from torchgenomics.ld import detect_blocks
from torchgenomics.ld._blocks_diagnostics import _wall_pritchard_native_enabled

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_WALL_PRITCHARD,
    reason="Native wall_pritchard extension not built; install with a C++17 compiler available.",
)


# ---------------------------------------------------------------------------
# Build / capability sanity
# ---------------------------------------------------------------------------


def test_native_module_loads():
    assert HAS_NATIVE_WALL_PRITCHARD is True
    assert hasattr(_wall_pritchard_native, "wall_pritchard_perm")


def test_native_dispatch_active_by_default():
    if os.environ.get("TORCHGENOMICS_DISABLE_NATIVE"):
        pytest.skip("env disables native path")
    assert _wall_pritchard_native_enabled() is True


# ---------------------------------------------------------------------------
# Direct C++ entry point
# ---------------------------------------------------------------------------


def test_native_zero_permutations_returns_zero():
    P = 10
    dp = np.random.default_rng(0).random(P)
    r2 = np.random.default_rng(1).random(P)
    is_adj = np.zeros(P, dtype=bool)
    is_adj[0] = True
    out = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 0.5, 0, 42,
    )
    assert out == 0


def test_native_empty_input():
    dp = np.array([], dtype=np.float64)
    r2 = np.array([], dtype=np.float64)
    is_adj = np.array([], dtype=bool)
    out = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 0.5, 100, 42,
    )
    assert out == 0


def test_native_blockiness_unreachable_returns_zero():
    # If blockiness_obs is set above 1.0 nothing can match it.
    rng = np.random.default_rng(7)
    P = 50
    dp = rng.random(P)
    r2 = rng.random(P)
    is_adj = np.zeros(P, dtype=bool)
    is_adj[:10] = True
    out = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 1.5, 200, 42,
    )
    assert out == 0


def test_native_blockiness_always_reached_returns_n():
    # blockiness_obs = 0 -> Q*Q_adj >= 0 always true
    rng = np.random.default_rng(8)
    P = 50
    dp = rng.random(P)
    r2 = rng.random(P)
    is_adj = np.zeros(P, dtype=bool)
    is_adj[:10] = True
    out = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 0.0, 100, 42,
    )
    assert out == 100


def test_native_invalid_shape_raises():
    dp = np.zeros(5, dtype=np.float64)
    r2 = np.zeros(4, dtype=np.float64)
    is_adj = np.zeros(5, dtype=bool)
    with pytest.raises(Exception):
        _wall_pritchard_native.wall_pritchard_perm(
            dp, r2, is_adj, 0.5, 0.5, 10, 42,
        )


def test_native_seed_determinism():
    rng = np.random.default_rng(11)
    P = 60
    dp = rng.random(P)
    r2 = rng.random(P)
    is_adj = np.zeros(P, dtype=bool)
    is_adj[::3] = True
    a = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 0.4, 500, 12345,
    )
    b = _wall_pritchard_native.wall_pritchard_perm(
        dp, r2, is_adj, 0.5, 0.4, 500, 12345,
    )
    assert a == b


# ---------------------------------------------------------------------------
# Equivalence: native dispatch vs Python reference (statistical, not exact —
# torch.randperm and std::mt19937_64 differ, so we expect close perm-pvalues
# rather than identical counts).
# ---------------------------------------------------------------------------


def _make_genotypes(seed: int, m: int = 20, n: int = 80):
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    pos = [1000 * (j + 1) for j in range(m)]
    chrs = ["1"] * m
    ids = [f"rs{j}" for j in range(m)]
    return G, pos, chrs, ids


def _python_reference(G, pos, chrs, ids, **kw):
    os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    try:
        return detect_blocks(G, pos, chrs, ids, method="wall_pritchard", **kw)
    finally:
        del os.environ["TORCHGENOMICS_DISABLE_NATIVE"]


def _native_dispatch(G, pos, chrs, ids, **kw):
    os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    return detect_blocks(G, pos, chrs, ids, method="wall_pritchard", **kw)


def test_dispatch_native_matches_python_zero_perms():
    # Without permutations both paths must be deterministic and identical.
    G, pos, chrs, ids = _make_genotypes(seed=21, m=20)
    out_n = _native_dispatch(G, pos, chrs, ids, max_kb=100.0)
    out_p = _python_reference(G, pos, chrs, ids, max_kb=100.0)
    assert len(out_n) == 1 and len(out_p) == 1
    assert out_n[0].mean_r2 == pytest.approx(out_p[0].mean_r2)
    assert out_n[0].blockiness_score == pytest.approx(out_p[0].blockiness_score)
    assert "perm_pvalue" not in out_n[0].metadata
    assert "perm_pvalue" not in out_p[0].metadata


def test_dispatch_native_pvalue_close_to_python():
    # With permutations the two paths use different RNGs (mt19937_64 vs
    # torch's). They should agree on observed Q/Q_adj and produce
    # statistically similar perm_pvalues.
    G, pos, chrs, ids = _make_genotypes(seed=33, m=25)
    torch.manual_seed(0)
    out_n = _native_dispatch(
        G, pos, chrs, ids, max_kb=100.0, n_permutations=400,
    )
    torch.manual_seed(0)
    out_p = _python_reference(
        G, pos, chrs, ids, max_kb=100.0, n_permutations=400,
    )

    # Observed statistics are identical (no RNG involved).
    assert out_n[0].metadata["Q"] == pytest.approx(out_p[0].metadata["Q"])
    assert out_n[0].metadata["Q_adj"] == pytest.approx(out_p[0].metadata["Q_adj"])
    assert out_n[0].blockiness_score == pytest.approx(out_p[0].blockiness_score)

    # Permutation p-values should be close.
    p_n = out_n[0].metadata["perm_pvalue"]
    p_p = out_p[0].metadata["perm_pvalue"]
    assert abs(p_n - p_p) < 0.15


def test_dispatch_native_pvalue_in_unit_interval():
    G, pos, chrs, ids = _make_genotypes(seed=55, m=15)
    out_n = _native_dispatch(
        G, pos, chrs, ids, max_kb=100.0, n_permutations=200,
    )
    p = out_n[0].metadata["perm_pvalue"]
    assert 0.0 < p <= 1.0
