"""Tests for the native C++ LDSC block-jackknife accelerator.

The native module replaces the n_blocks=200 leave-one-block-out WLS loop in
``torchgwas.postgwas._ldsc._block_jackknife_se``. The Python loop remains
in-tree as the algorithmic spec; the dispatcher routes to C++ when the
build is present, the tensors are CPU+float64, and ``TORCHGWAS_DISABLE_NATIVE``
is unset.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_LDSC, _ldsc_native
from torchgwas.postgwas._ldsc import (
    _block_jackknife_se,
    _weighted_lstsq,
    ldsc_h2,
    ldsc_rg,
)


pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_LDSC, reason="native LDSC extension not built"
)


def _make_ldsc_problem(m=2000, h2=0.5, n=10000, m_total=10000, seed=0):
    """Generate a synthetic LDSC fixture."""
    rng = np.random.default_rng(seed)
    ld_scores = torch.tensor(
        rng.gamma(2.0, 1.5, size=m).astype(np.float64),
        dtype=torch.float64,
    )
    expected = (n / m_total) * h2 * ld_scores + 1.0
    chi2 = torch.tensor(
        rng.gamma(expected.numpy(), 1.0).astype(np.float64),
        dtype=torch.float64,
    ).clamp(min=1e-6)
    return chi2, ld_scores


def _build_design(chi2, ld_scores):
    w = 1.0 / torch.clamp(ld_scores ** 2, min=1.0)
    X = torch.stack([ld_scores, torch.ones_like(ld_scores)], dim=1)
    return X, chi2, w


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_build_sanity():
    assert HAS_NATIVE_LDSC is True
    assert _ldsc_native is not None
    assert hasattr(_ldsc_native, "ldsc_block_jackknife")


def test_smoke_returns_positive_se():
    chi2, ld = _make_ldsc_problem()
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    se = _ldsc_native.ldsc_block_jackknife(
        X.numpy(), y.numpy(), w.numpy(), 200, full.numpy(),
    )
    assert se.shape == (2,)
    assert np.all(np.isfinite(se))
    assert np.all(se >= 0)


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence
# ---------------------------------------------------------------------------


def _run_jackknife_with(disable_native: bool, X, y, w, n_blocks, full):
    if disable_native:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    try:
        return _block_jackknife_se(X, y, w, n_blocks, full)
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)


def test_native_matches_python_default_blocks():
    chi2, ld = _make_ldsc_problem(seed=1)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    se_native = _run_jackknife_with(False, X, y, w, 200, full)
    se_python = _run_jackknife_with(True, X, y, w, 200, full)
    assert torch.allclose(se_native, se_python, atol=1e-10, rtol=1e-10)


def test_native_matches_python_50_blocks():
    chi2, ld = _make_ldsc_problem(m=1000, seed=2)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    se_native = _run_jackknife_with(False, X, y, w, 50, full)
    se_python = _run_jackknife_with(True, X, y, w, 50, full)
    assert torch.allclose(se_native, se_python, atol=1e-10, rtol=1e-10)


def test_native_matches_python_uneven_block_remainder():
    """m not divisible by n_blocks — last block absorbs the remainder."""
    chi2, ld = _make_ldsc_problem(m=1023, seed=3)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    se_native = _run_jackknife_with(False, X, y, w, 200, full)
    se_python = _run_jackknife_with(True, X, y, w, 200, full)
    assert torch.allclose(se_native, se_python, atol=1e-10, rtol=1e-10)


def test_native_matches_python_p3_design():
    """Generic over p — exercise a 3-column design (LD score + intercept + extra)."""
    chi2, ld = _make_ldsc_problem(m=800, seed=4)
    rng = np.random.default_rng(4)
    extra = torch.tensor(
        rng.normal(0.0, 1.0, size=ld.shape[0]).astype(np.float64),
        dtype=torch.float64,
    )
    X = torch.stack([ld, torch.ones_like(ld), extra], dim=1)
    w = 1.0 / torch.clamp(ld ** 2, min=1.0)
    full = _weighted_lstsq(X, chi2, w)
    se_native = _run_jackknife_with(False, X, chi2, w, 100, full)
    se_python = _run_jackknife_with(True, X, chi2, w, 100, full)
    assert torch.allclose(se_native, se_python, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end ldsc_h2 dispatcher equivalence
# ---------------------------------------------------------------------------


def test_ldsc_h2_dispatcher_native_matches_python():
    chi2, ld = _make_ldsc_problem(m=3000, h2=0.4, n=20000, m_total=20000, seed=5)

    os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    try:
        res_python = ldsc_h2(chi2, ld, n=20000, m_total=20000, n_blocks=200)
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)

    res_native = ldsc_h2(chi2, ld, n=20000, m_total=20000, n_blocks=200)

    # Identical point estimates and SEs (the only stochastic step is the
    # block-jackknife loop and we expect bit-equivalent output modulo
    # float-summation order, which is deterministic for a fixed input).
    assert abs(res_native.h2 - res_python.h2) < 1e-12
    assert abs(res_native.h2_se - res_python.h2_se) < 1e-10
    assert abs(res_native.intercept - res_python.intercept) < 1e-12
    assert abs(res_native.intercept_se - res_python.intercept_se) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


def test_min_blocks_degenerate_does_not_crash():
    """When ``n_blocks`` > ``m`` the jackknife collapses to a single block,
    which is mathematically degenerate (the per-block leave-one-out fit has
    zero observations). The Python reference returns NaN from the
    division-by-zero in the variance formula and the native path returns
    zero — both paths are equally meaningless here, but neither should
    raise. This case never arises in real LDSC calls."""
    chi2, ld = _make_ldsc_problem(m=10, seed=6)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    se_native = _run_jackknife_with(False, X, y, w, 200, full)
    se_python = _run_jackknife_with(True, X, y, w, 200, full)
    assert se_native.shape == se_python.shape == (2,)


def test_invalid_p_too_large_raises():
    rng = np.random.default_rng(7)
    m, p = 100, 9
    X = rng.normal(size=(m, p))
    y = rng.normal(size=m)
    w = np.ones(m)
    full = np.zeros(p)
    with pytest.raises(Exception):
        _ldsc_native.ldsc_block_jackknife(X, y, w, 10, full)


def test_invalid_n_blocks_zero_raises():
    chi2, ld = _make_ldsc_problem(seed=8)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    with pytest.raises(Exception):
        _ldsc_native.ldsc_block_jackknife(
            X.numpy(), y.numpy(), w.numpy(), 0, full.numpy(),
        )


def test_shape_mismatch_raises():
    chi2, ld = _make_ldsc_problem(seed=9)
    X, y, w = _build_design(chi2, ld)
    full = _weighted_lstsq(X, y, w)
    bad_y = y[:-1].clone()
    with pytest.raises(Exception):
        _ldsc_native.ldsc_block_jackknife(
            X.numpy(), bad_y.numpy(), w.numpy(), 50, full.numpy(),
        )
