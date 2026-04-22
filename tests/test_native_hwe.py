"""Tests for the native HWE chi-squared p-value accelerator.

The native module replaces the per-SNP scipy.stats.chi2.sf + .item() loop
in ``torchgwas.preprocess.qc._compute_hwe_pvalue``. The pure-Python loop
remains in-tree as the algorithmic spec; the dispatcher routes to C++ when
the build is present and ``TORCHGWAS_DISABLE_NATIVE`` is unset.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_HWE, _hwe_native
from torchgwas.preprocess.qc import _compute_hwe_pvalue

pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_HWE, reason="native HWE extension not built"
)


def _make_diploid_problem(n=400, m=80, seed=0):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.5, size=m)
    G = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        p = af[j]
        # Sample under HWE: dosages 0/1/2 with probs q^2, 2pq, p^2
        probs = np.array([(1 - p) ** 2, 2 * p * (1 - p), p ** 2])
        G[:, j] = rng.choice(3, size=n, p=probs).astype(np.float64)
    return torch.from_numpy(G), torch.from_numpy(af)


def _make_polyploid_problem(n=400, m=60, ploidy=4, seed=1):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.1, 0.5, size=m)
    G = np.empty((n, m), dtype=np.float64)
    from math import comb
    for j in range(m):
        p = af[j]
        q = 1 - p
        probs = np.array([comb(ploidy, v) * (p ** v) * (q ** (ploidy - v))
                          for v in range(ploidy + 1)])
        probs /= probs.sum()
        G[:, j] = rng.choice(ploidy + 1, size=n, p=probs).astype(np.float64)
    return torch.from_numpy(G), torch.from_numpy(af)


def _run_with(disable_native, G, af, ploidy):
    if disable_native:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    try:
        return _compute_hwe_pvalue(G, af, ploidy=ploidy)
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_build_sanity():
    assert HAS_NATIVE_HWE is True
    assert _hwe_native is not None
    assert hasattr(_hwe_native, "hwe_pvalues")


def test_smoke_returns_unit_interval():
    G, af = _make_diploid_problem()
    p = _run_with(False, G, af, ploidy=2)
    assert p.shape == (G.shape[1],)
    assert torch.all((p >= 0.0) & (p <= 1.0))


def test_polyploid_smoke():
    G, af = _make_polyploid_problem(ploidy=4)
    p = _run_with(False, G, af, ploidy=4)
    assert p.shape == (G.shape[1],)
    assert torch.all((p >= 0.0) & (p <= 1.0))


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence
# ---------------------------------------------------------------------------


def test_native_matches_python_diploid():
    G, af = _make_diploid_problem(seed=2)
    p_native = _run_with(False, G, af, ploidy=2)
    p_python = _run_with(True, G, af, ploidy=2)
    assert torch.allclose(p_native, p_python, atol=1e-10, rtol=1e-8)


def test_native_matches_python_diploid_with_missing():
    G, af = _make_diploid_problem(seed=3)
    rng = np.random.default_rng(3)
    G_np = G.numpy().copy()
    miss_mask = rng.random(G_np.shape) < 0.1
    G_np[miss_mask] = np.nan
    G_miss = torch.from_numpy(G_np)
    p_native = _run_with(False, G_miss, af, ploidy=2)
    p_python = _run_with(True, G_miss, af, ploidy=2)
    assert torch.allclose(p_native, p_python, atol=1e-10, rtol=1e-8)


def test_native_matches_python_tetraploid():
    G, af = _make_polyploid_problem(seed=4, ploidy=4)
    p_native = _run_with(False, G, af, ploidy=4)
    p_python = _run_with(True, G, af, ploidy=4)
    assert torch.allclose(p_native, p_python, atol=1e-10, rtol=1e-8)


def test_native_matches_python_hexaploid():
    G, af = _make_polyploid_problem(seed=5, ploidy=6, m=40)
    p_native = _run_with(False, G, af, ploidy=6)
    p_python = _run_with(True, G, af, ploidy=6)
    assert torch.allclose(p_native, p_python, atol=1e-10, rtol=1e-8)


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


def test_low_n_valid_returns_one():
    """SNPs with fewer than 10 valid samples get HWE p = 1.0."""
    n, m = 8, 5
    G = torch.zeros(n, m, dtype=torch.float64)
    af = torch.full((m,), 0.3, dtype=torch.float64)
    p = _run_with(False, G, af, ploidy=2)
    assert torch.allclose(p, torch.ones(m, dtype=torch.float64))


def test_strong_hwe_violation_low_p():
    """A SNP with all heterozygotes (no homozygotes) at af=0.5 violates HWE
    strongly under the binomial expectation and should produce a small p."""
    n, m = 200, 1
    G = torch.full((n, m), 1.0, dtype=torch.float64)  # all dosage = 1
    af = torch.tensor([0.5], dtype=torch.float64)
    p_native = _run_with(False, G, af, ploidy=2)
    p_python = _run_with(True, G, af, ploidy=2)
    assert p_native.item() < 1e-20
    assert torch.allclose(p_native, p_python, atol=1e-12)


def test_invalid_ploidy_raises():
    G = torch.zeros(20, 3, dtype=torch.float64)
    af = torch.full((3,), 0.3, dtype=torch.float64)
    G_int = G.to(torch.int64).numpy()
    af_np = af.numpy()
    with pytest.raises(Exception):
        _hwe_native.hwe_pvalues(G_int, af_np, 1)


def test_shape_mismatch_raises():
    G_int = np.zeros((20, 3), dtype=np.int64)
    af_bad = np.zeros(2, dtype=np.float64)
    with pytest.raises(Exception):
        _hwe_native.hwe_pvalues(G_int, af_bad, 2)
