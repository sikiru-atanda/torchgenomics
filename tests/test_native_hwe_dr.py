"""Tests for the native HWE-with-double-reduction accelerator.

The native module replaces the per-SNP scipy.stats.chi2.sf + .item() loop
in ``torchgwas.preprocess.qc._compute_hwe_double_reduction``. The pure-Python
loop remains in-tree as the algorithmic spec; the dispatcher routes to C++
when the build is present and ``TORCHGWAS_DISABLE_NATIVE`` is unset.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_HWE, _hwe_native
from torchgwas.preprocess.qc import _compute_hwe_double_reduction

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_HWE, reason="native HWE extension not built"
)


def _make_tetraploid(n=400, m=60, seed=0, dr_alpha=0.0):
    """Sample tetraploid dosages under Haldane DR with given alpha."""
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.15, 0.5, size=m)
    G = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        p = af[j]
        q = 1 - p
        gam = np.array([
            (1 - dr_alpha) * q * q + dr_alpha * q,
            (1 - dr_alpha) * 2.0 * p * q,
            (1 - dr_alpha) * p * p + dr_alpha * p,
        ])
        gam = gam / gam.sum()
        # Convolve gametes -> 5-class genotype distribution
        probs = np.zeros(5)
        for d1 in range(3):
            for d2 in range(3):
                probs[d1 + d2] += gam[d1] * gam[d2]
        probs = probs / probs.sum()
        G[:, j] = rng.choice(5, size=n, p=probs).astype(np.float64)
    return torch.from_numpy(G), torch.from_numpy(af)


def _run_with(disable_native, G, af, ploidy):
    if disable_native:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    try:
        return _compute_hwe_double_reduction(G, af, ploidy=ploidy)
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_build_sanity():
    assert HAS_NATIVE_HWE is True
    assert hasattr(_hwe_native, "hwe_pvalues_double_reduction")


def test_smoke_returns_unit_interval_and_clamped_alpha():
    G, af = _make_tetraploid(dr_alpha=0.0)
    alpha, p = _run_with(False, G, af, ploidy=4)
    assert alpha.shape == (G.shape[1],)
    assert p.shape == (G.shape[1],)
    assert torch.all((p >= 0.0) & (p <= 1.0))
    assert torch.all((alpha >= 0.0) & (alpha <= 1.0 / 6.0 + 1e-12))


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence
# ---------------------------------------------------------------------------


def test_native_matches_python_no_dr():
    G, af = _make_tetraploid(seed=1, dr_alpha=0.0)
    a_n, p_n = _run_with(False, G, af, ploidy=4)
    a_p, p_p = _run_with(True, G, af, ploidy=4)
    assert torch.allclose(a_n, a_p, atol=1e-10, rtol=1e-8)
    assert torch.allclose(p_n, p_p, atol=1e-10, rtol=1e-8)


def test_native_matches_python_with_dr():
    G, af = _make_tetraploid(seed=2, dr_alpha=0.1)
    a_n, p_n = _run_with(False, G, af, ploidy=4)
    a_p, p_p = _run_with(True, G, af, ploidy=4)
    assert torch.allclose(a_n, a_p, atol=1e-10, rtol=1e-8)
    assert torch.allclose(p_n, p_p, atol=1e-10, rtol=1e-8)


def test_native_matches_python_with_missing():
    G, af = _make_tetraploid(seed=3, dr_alpha=0.05)
    rng = np.random.default_rng(3)
    G_np = G.numpy().copy()
    miss = rng.random(G_np.shape) < 0.1
    G_np[miss] = np.nan
    G_miss = torch.from_numpy(G_np)
    a_n, p_n = _run_with(False, G_miss, af, ploidy=4)
    a_p, p_p = _run_with(True, G_miss, af, ploidy=4)
    assert torch.allclose(a_n, a_p, atol=1e-10, rtol=1e-8)
    assert torch.allclose(p_n, p_p, atol=1e-10, rtol=1e-8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_non_tetraploid_returns_defaults():
    G = torch.zeros(50, 4, dtype=torch.float64)
    af = torch.full((4,), 0.3, dtype=torch.float64)
    alpha, p = _run_with(False, G, af, ploidy=6)
    assert torch.all(alpha == 0.0)
    assert torch.all(p == 1.0)


def test_low_n_valid_keeps_defaults():
    G = torch.zeros(8, 5, dtype=torch.float64)
    af = torch.full((5,), 0.3, dtype=torch.float64)
    alpha, p = _run_with(False, G, af, ploidy=4)
    assert torch.all(alpha == 0.0)
    assert torch.all(p == 1.0)


def test_extreme_af_keeps_defaults():
    n = 100
    G = torch.zeros(n, 2, dtype=torch.float64)
    af = torch.tensor([0.0, 1.0], dtype=torch.float64)
    alpha, p = _run_with(False, G, af, ploidy=4)
    assert torch.all(alpha == 0.0)
    assert torch.all(p == 1.0)


def test_shape_mismatch_raises():
    G_int = np.zeros((20, 3), dtype=np.int64)
    af_bad = np.zeros(2, dtype=np.float64)
    with pytest.raises(Exception):
        _hwe_native.hwe_pvalues_double_reduction(G_int, af_bad)
