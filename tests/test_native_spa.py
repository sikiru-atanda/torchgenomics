"""Tests for the native C++ SPA Lugannani-Rice accelerator.

The native module replaces the per-extreme-SNP saddlepoint Newton solve and
Lugannani-Rice tail computation in
``torchgenomics.stats.spa.saddlepoint_pvalue``. The Python loop remains in-tree
as the algorithmic spec; the dispatcher routes to C++ when the build is
present, the tensors are CPU+float64, and ``TORCHGENOMICS_DISABLE_NATIVE`` is
unset.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from torchgenomics._native import HAS_NATIVE_SPA, _spa_native
from torchgenomics.stats.spa import saddlepoint_pvalue

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_SPA, reason="native SPA extension not built"
)


def _make_imbalanced_problem(n=400, m=20, prevalence=0.05, seed=0):
    """Generate a case-control SPA fixture with extreme prevalence."""
    rng = np.random.default_rng(seed)
    G = torch.tensor(
        rng.binomial(2, 0.3, size=(n, m)).astype(np.float64),
        dtype=torch.float64,
    )
    mu = torch.full((n,), prevalence, dtype=torch.float64)
    Y = torch.tensor(
        rng.binomial(1, prevalence, size=n).astype(np.float64),
        dtype=torch.float64,
    )
    # Score statistic U = g'(Y - mu) per SNP
    resid = Y - mu
    U = (G * resid.unsqueeze(1)).sum(dim=0)
    return mu, G, Y, U


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_build_sanity():
    assert HAS_NATIVE_SPA is True
    assert _spa_native is not None
    assert hasattr(_spa_native, "spa_lugannani_rice")


def test_smoke_returns_unit_interval_or_nan():
    mu, G, Y, U = _make_imbalanced_problem()
    g0 = G[:, 0].numpy()
    p = _spa_native.spa_lugannani_rice(mu.numpy(), g0, float(U[0].item()))
    assert math.isnan(p) or (0.0 < p <= 1.0)


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence (statistical, via the public dispatcher)
# ---------------------------------------------------------------------------


def _run_with(disable_native: bool, mu, G, U):
    if disable_native:
        os.environ["TORCHGENOMICS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)
    try:
        return saddlepoint_pvalue(U.clone(), mu.clone(), G.clone(), threshold=2.0)
    finally:
        os.environ.pop("TORCHGENOMICS_DISABLE_NATIVE", None)


def test_native_matches_python_imbalanced():
    mu, G, _, U = _make_imbalanced_problem(n=400, m=30, prevalence=0.03, seed=1)
    p_native = _run_with(False, mu, G, U)
    p_python = _run_with(True, mu, G, U)
    assert torch.allclose(p_native, p_python, atol=1e-9, rtol=1e-9)


def test_native_matches_python_moderate():
    mu, G, _, U = _make_imbalanced_problem(n=300, m=25, prevalence=0.10, seed=2)
    p_native = _run_with(False, mu, G, U)
    p_python = _run_with(True, mu, G, U)
    assert torch.allclose(p_native, p_python, atol=1e-9, rtol=1e-9)


def test_native_matches_python_balanced():
    """Even at balanced prevalence the dispatcher should agree (most SNPs
    will fall back to chi2; SPA-active ones must match)."""
    mu, G, _, U = _make_imbalanced_problem(n=200, m=15, prevalence=0.5, seed=3)
    p_native = _run_with(False, mu, G, U)
    p_python = _run_with(True, mu, G, U)
    assert torch.allclose(p_native, p_python, atol=1e-9, rtol=1e-9)


def test_native_matches_python_polyploid_dosage():
    """Tetraploid-style dosages in [0, 4]."""
    rng = np.random.default_rng(4)
    n, m = 250, 20
    G = torch.tensor(
        rng.binomial(4, 0.25, size=(n, m)).astype(np.float64),
        dtype=torch.float64,
    )
    mu = torch.full((n,), 0.05, dtype=torch.float64)
    Y = torch.tensor(
        rng.binomial(1, 0.05, size=n).astype(np.float64),
        dtype=torch.float64,
    )
    U = (G * (Y - mu).unsqueeze(1)).sum(dim=0)
    p_native = _run_with(False, mu, G, U)
    p_python = _run_with(True, mu, G, U)
    assert torch.allclose(p_native, p_python, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


def test_zero_observed_returns_unit_or_nan():
    mu, G, _, _ = _make_imbalanced_problem()
    p = _spa_native.spa_lugannani_rice(
        mu.numpy(), G[:, 0].numpy(), 0.0,
    )
    # Saddlepoint at t=0; the formula degenerates → fallback NaN expected.
    assert math.isnan(p) or (0.0 < p <= 1.0)


def test_returns_in_unit_interval_when_finite():
    mu, G, _, U = _make_imbalanced_problem(n=500, m=50, prevalence=0.02, seed=5)
    mu_np = mu.numpy()
    n_finite = 0
    for j in range(G.shape[1]):
        p = _spa_native.spa_lugannani_rice(
            mu_np, G[:, j].numpy(), float(U[j].item()),
        )
        if not math.isnan(p):
            assert 0.0 < p <= 1.0
            n_finite += 1
    assert n_finite > 0  # at least some SNPs produce a valid SPA p-value


def test_shape_mismatch_raises():
    mu, G, _, _ = _make_imbalanced_problem()
    with pytest.raises(Exception):
        _spa_native.spa_lugannani_rice(
            mu.numpy(), G[:-1, 0].numpy(), 1.5,
        )


def test_empty_input_raises():
    with pytest.raises(Exception):
        _spa_native.spa_lugannani_rice(
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            0.5,
        )


def test_2d_input_raises():
    mu, G, _, _ = _make_imbalanced_problem()
    with pytest.raises(Exception):
        _spa_native.spa_lugannani_rice(
            mu.numpy().reshape(-1, 1), G[:, 0].numpy(), 1.0,
        )


def test_dispatcher_disabled_via_env_uses_python():
    """With TORCHGENOMICS_DISABLE_NATIVE=1 the public function must reach the
    Python loop and still produce sensible output."""
    mu, G, _, U = _make_imbalanced_problem(n=200, m=10, prevalence=0.05, seed=6)
    p = _run_with(True, mu, G, U)
    assert torch.isfinite(p).all()
    assert (p > 0).all() and (p <= 1).all()
