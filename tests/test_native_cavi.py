"""Tests for the native C++ CAVI inner-sweep accelerator.

The native module replaces the per-SNP coordinate-update loop in
``torchgwas.models.bayesian_vs.BayesianVS._cavi_loop``. The Python loop
remains in-tree as the algorithmic spec; the dispatcher routes to C++
when the build is present, the tensors are CPU+float64, and
``TORCHGWAS_DISABLE_NATIVE`` is unset.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from torchgwas._native import HAS_NATIVE_CAVI, _cavi_native


pytestmark = pytest.mark.skipif(
    not HAS_NATIVE_CAVI, reason="native CAVI extension not built"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_cavi_sweep(G_rot, wG, gWg, r, gamma, mu, sigma2, pi, sig2_beta):
    """Reference one-sweep implementation matching the in-tree Python loop
    in ``BayesianVS._cavi_loop`` line-for-line."""
    p = G_rot.shape[1]
    log_odds_prior = math.log(pi / (1.0 - pi))
    for j in range(p):
        g_j = G_rot[:, j]
        wg_j = wG[:, j]
        old_effect = gamma[j] * mu[j]
        r = r + g_j * old_effect
        sigma2_j = 1.0 / (gWg[j] + 1.0 / sig2_beta)
        mu_j = sigma2_j * torch.dot(wg_j, r)
        log_bf = 0.5 * math.log(sigma2_j.item() / sig2_beta) + \
            0.5 * mu_j ** 2 / sigma2_j
        log_bf = torch.clamp(log_bf, -30.0, 30.0)
        gamma_j = torch.sigmoid(log_odds_prior + log_bf)
        gamma_j = torch.clamp(gamma_j, 1e-10, 1.0 - 1e-10)
        gamma[j] = gamma_j
        mu[j] = mu_j
        sigma2[j] = sigma2_j
        new_effect = gamma_j * mu_j
        r = r - g_j * new_effect
    return r, gamma, mu, sigma2


def _make_problem(n=80, p=40, seed=0):
    torch.manual_seed(seed)
    G = torch.randn(n, p, dtype=torch.float64)
    G = G - G.mean(dim=0, keepdim=True)
    w = torch.rand(n, dtype=torch.float64) + 0.5
    wG = w.unsqueeze(1) * G
    gWg = (G * wG).sum(dim=0)
    r = torch.randn(n, dtype=torch.float64)
    gamma = torch.full((p,), 0.1, dtype=torch.float64)
    mu = torch.zeros(p, dtype=torch.float64)
    sigma2 = 1.0 / (gWg + 1.0 / 0.5)
    return G, wG, gWg, r, gamma, mu, sigma2


# ---------------------------------------------------------------------------
# Build sanity / smoke
# ---------------------------------------------------------------------------


def test_build_sanity():
    assert HAS_NATIVE_CAVI is True
    assert _cavi_native is not None
    assert hasattr(_cavi_native, "cavi_sweep")


def test_smoke_runs_in_place():
    G, wG, gWg, r, gamma, mu, sigma2 = _make_problem()
    r0 = r.clone()
    _cavi_native.cavi_sweep(
        G.numpy(), wG.numpy(), gWg.numpy(),
        r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
        0.1, 0.5,
    )
    # In-place mutation actually happened.
    assert not torch.equal(r, r0)
    assert torch.isfinite(r).all()
    assert torch.isfinite(gamma).all()
    assert (gamma >= 1e-10).all() and (gamma <= 1.0 - 1e-10).all()
    assert (sigma2 > 0).all()


# ---------------------------------------------------------------------------
# Native ↔ Python equivalence
# ---------------------------------------------------------------------------


def test_native_matches_python_one_sweep():
    G, wG, gWg, r_n, gamma_n, mu_n, sigma2_n = _make_problem(seed=1)
    r_p, gamma_p, mu_p, sigma2_p = (
        r_n.clone(), gamma_n.clone(), mu_n.clone(), sigma2_n.clone()
    )

    _cavi_native.cavi_sweep(
        G.numpy(), wG.numpy(), gWg.numpy(),
        r_n.numpy(), gamma_n.numpy(), mu_n.numpy(), sigma2_n.numpy(),
        0.1, 0.5,
    )

    r_p, gamma_p, mu_p, sigma2_p = _python_cavi_sweep(
        G, wG, gWg, r_p, gamma_p, mu_p, sigma2_p, 0.1, 0.5,
    )

    assert torch.allclose(r_n, r_p, atol=1e-12, rtol=1e-12)
    assert torch.allclose(gamma_n, gamma_p, atol=1e-12, rtol=1e-12)
    assert torch.allclose(mu_n, mu_p, atol=1e-12, rtol=1e-12)
    assert torch.allclose(sigma2_n, sigma2_p, atol=1e-12, rtol=1e-12)


def test_native_matches_python_many_sweeps():
    """Drift over 5 sweeps must stay within float64 tolerance."""
    G, wG, gWg, r_n, gamma_n, mu_n, sigma2_n = _make_problem(n=50, p=30, seed=2)
    r_p, gamma_p, mu_p, sigma2_p = (
        r_n.clone(), gamma_n.clone(), mu_n.clone(), sigma2_n.clone()
    )

    pi, sig2_beta = 0.05, 0.3
    for _ in range(5):
        _cavi_native.cavi_sweep(
            G.numpy(), wG.numpy(), gWg.numpy(),
            r_n.numpy(), gamma_n.numpy(), mu_n.numpy(), sigma2_n.numpy(),
            pi, sig2_beta,
        )
        r_p, gamma_p, mu_p, sigma2_p = _python_cavi_sweep(
            G, wG, gWg, r_p, gamma_p, mu_p, sigma2_p, pi, sig2_beta,
        )

    assert torch.allclose(r_n, r_p, atol=1e-10, rtol=1e-10)
    assert torch.allclose(gamma_n, gamma_p, atol=1e-10, rtol=1e-10)
    assert torch.allclose(mu_n, mu_p, atol=1e-10, rtol=1e-10)
    assert torch.allclose(sigma2_n, sigma2_p, atol=1e-12, rtol=1e-12)


def test_native_matches_python_large_p():
    G, wG, gWg, r_n, gamma_n, mu_n, sigma2_n = _make_problem(n=200, p=300, seed=3)
    r_p, gamma_p, mu_p, sigma2_p = (
        r_n.clone(), gamma_n.clone(), mu_n.clone(), sigma2_n.clone()
    )

    _cavi_native.cavi_sweep(
        G.numpy(), wG.numpy(), gWg.numpy(),
        r_n.numpy(), gamma_n.numpy(), mu_n.numpy(), sigma2_n.numpy(),
        0.02, 0.1,
    )
    r_p, gamma_p, mu_p, sigma2_p = _python_cavi_sweep(
        G, wG, gWg, r_p, gamma_p, mu_p, sigma2_p, 0.02, 0.1,
    )

    assert torch.allclose(r_n, r_p, atol=1e-11, rtol=1e-11)
    assert torch.allclose(gamma_n, gamma_p, atol=1e-11, rtol=1e-11)
    assert torch.allclose(mu_n, mu_p, atol=1e-11, rtol=1e-11)


# ---------------------------------------------------------------------------
# Edge cases & input validation
# ---------------------------------------------------------------------------


def test_zero_initial_effects_does_not_modify_residual_first_pass():
    """When gamma=mu=0 the first add-back is a no-op; subsequent SNPs
    still update the residual through new_effect."""
    G, wG, gWg, r, gamma, mu, sigma2 = _make_problem(seed=4)
    gamma.zero_()
    mu.zero_()
    r0 = r.clone()
    _cavi_native.cavi_sweep(
        G.numpy(), wG.numpy(), gWg.numpy(),
        r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
        0.1, 0.5,
    )
    # gamma and mu must have moved off zero for at least some SNPs.
    assert (gamma > 1e-10).any() or (gamma < 1.0 - 1e-10).any()


def test_invalid_pi_raises():
    G, wG, gWg, r, gamma, mu, sigma2 = _make_problem()
    with pytest.raises(Exception):
        _cavi_native.cavi_sweep(
            G.numpy(), wG.numpy(), gWg.numpy(),
            r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
            0.0, 0.5,
        )
    with pytest.raises(Exception):
        _cavi_native.cavi_sweep(
            G.numpy(), wG.numpy(), gWg.numpy(),
            r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
            1.0, 0.5,
        )


def test_invalid_sig2_beta_raises():
    G, wG, gWg, r, gamma, mu, sigma2 = _make_problem()
    with pytest.raises(Exception):
        _cavi_native.cavi_sweep(
            G.numpy(), wG.numpy(), gWg.numpy(),
            r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
            0.1, 0.0,
        )


def test_shape_mismatch_raises():
    G, wG, gWg, r, gamma, mu, sigma2 = _make_problem()
    bad_gWg = gWg[:-1].clone()  # wrong length
    with pytest.raises(Exception):
        _cavi_native.cavi_sweep(
            G.numpy(), wG.numpy(), bad_gWg.numpy(),
            r.numpy(), gamma.numpy(), mu.numpy(), sigma2.numpy(),
            0.1, 0.5,
        )


# ---------------------------------------------------------------------------
# End-to-end dispatcher: BayesianVS.cavi must produce identical fits
# whether the native sweep is enabled or not
# ---------------------------------------------------------------------------


def _fit_cavi_with(disable_native: bool):
    if disable_native:
        os.environ["TORCHGWAS_DISABLE_NATIVE"] = "1"
    else:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)
    try:
        from torchgwas.models.base import VariantMeta
        from torchgwas.models.bayesian_vs import BayesianVS
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        torch.manual_seed(0)
        n, m = 60, 25
        G = torch.randn(n, m, dtype=torch.float64)
        beta_true = torch.zeros(m, dtype=torch.float64)
        beta_true[3] = 0.8
        beta_true[10] = -0.6
        Y = G @ beta_true + 0.3 * torch.randn(n, dtype=torch.float64)
        K = (G @ G.T) / m + 1e-3 * torch.eye(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        lmm = SingleTraitLMM()
        null = lmm.fit_null(Y, X0, K=K)
        vmeta = VariantMeta(
            snp=[f"snp{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["T"] * m,
        )
        bvs = BayesianVS(null_fit=null)
        result = bvs.fit(G, vmeta, method="cavi", max_iter=20, n_signals=2)
        return result
    finally:
        os.environ.pop("TORCHGWAS_DISABLE_NATIVE", None)


def test_dispatcher_native_matches_python_end_to_end():
    res_native = _fit_cavi_with(disable_native=False)
    res_python = _fit_cavi_with(disable_native=True)

    # PIPs should be very close (deterministic, only float-summation order
    # difference between the two paths).
    assert torch.allclose(res_native.pip, res_python.pip, atol=1e-9, rtol=1e-9)
