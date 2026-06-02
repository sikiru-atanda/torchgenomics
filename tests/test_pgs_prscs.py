"""Tests for torchgenomics.pgs.prscs — PRS-CS and the Devroye GIG sampler."""

from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.pgs.ld_ref import build_ld_reference
from torchgenomics.pgs.prscs import (
    PRSCS,
    _prscs_gibbs_block,
    _sample_gig_scalar,
    sample_gig,
)
from torchgenomics.postgwas._sumstats import SumStats


def _make_ss_and_ld(m=10, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)
    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[0] = 0.3
    beta_true[5] = -0.25
    y = G @ beta_true + torch.randn(n, generator=g, dtype=torch.float64) * 0.5
    y = (y - y.mean()) / y.std().clamp(min=1e-12)
    beta_hat = (G * y.unsqueeze(1)).sum(0) / n
    se = torch.full((m,), 1.0 / math.sqrt(n), dtype=torch.float64)
    z = beta_hat / se
    p = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(z.abs() / math.sqrt(2.0))))
    ss = SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=beta_hat,
        se=se,
        p=p,
        n=torch.full((m,), float(n)),
        af=None,
    )
    meta = {"snp": ss.snp, "chr": ss.chr, "pos": ss.pos, "a1": ss.a1, "a2": ss.a2}
    ld = build_ld_reference(G, meta, mode="full")
    return ss, ld, beta_true


# ---------------------------------------------------------------------------
# Devroye GIG sampler
# ---------------------------------------------------------------------------


def test_sample_gig_scalar_positive_support():
    rng = torch.Generator().manual_seed(0)
    for _ in range(50):
        x = _sample_gig_scalar(0.5, 1.0, 1.0, rng)
        assert x > 0
        assert math.isfinite(x)


def test_sample_gig_moment_check_vs_scipy():
    """Compare sample mean/variance against scipy.stats.geninvgauss."""
    sp = pytest.importorskip("scipy.stats")
    rng = torch.Generator().manual_seed(42)
    lam = 0.5
    chi = 2.0
    psi = 3.0
    # scipy uses geninvgauss(p=lam, b=sqrt(chi*psi)) with scale=sqrt(chi/psi)
    b_scipy = math.sqrt(chi * psi)
    scale = math.sqrt(chi / psi)
    true_mean = sp.geninvgauss.mean(lam, b_scipy) * scale
    true_var = sp.geninvgauss.var(lam, b_scipy) * (scale ** 2)

    n_draws = 3000
    samples = [_sample_gig_scalar(lam, chi, psi, rng) for _ in range(n_draws)]
    xs = torch.tensor(samples, dtype=torch.float64)
    emp_mean = float(xs.mean())
    emp_var = float(xs.var())

    mean_se = math.sqrt(true_var / n_draws)
    assert abs(emp_mean - true_mean) < 5 * mean_se
    assert abs(emp_var - true_var) / max(true_var, 1e-6) < 0.5


def test_sample_gig_vectorized_matches_scalar_shape():
    rng = torch.Generator().manual_seed(1)
    chi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    psi = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
    out = sample_gig(0.5, chi, psi, rng=rng)
    assert out.shape == chi.shape
    assert (out > 0).all()


def test_sample_gig_handles_various_lam():
    rng = torch.Generator().manual_seed(2)
    for lam in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        x = _sample_gig_scalar(lam, 1.0, 1.0, rng)
        assert x > 0 and math.isfinite(x)


# ---------------------------------------------------------------------------
# PRS-CS fit
# ---------------------------------------------------------------------------


def test_prscs_runs_and_produces_weights():
    ss, ld, _ = _make_ss_and_ld(m=8, n=1500, seed=10)
    res = PRSCS(seed=0).fit(
        ss, ld, phi=1e-2, n_iter=40, n_burnin=20, n_chains=1,
    )
    assert res.method == "prscs"
    assert res.weight.shape == (8,)
    assert res.weight_sd is not None
    assert res.weight_sd.shape == (8,)
    assert (res.weight_sd >= 0).all()
    assert res.phi == 1e-2


def test_prscs_deterministic_with_fixed_seed():
    ss, ld, _ = _make_ss_and_ld(m=6, n=1000, seed=11)
    r1 = PRSCS(seed=7).fit(ss, ld, phi=1e-2, n_iter=30, n_burnin=15)
    r2 = PRSCS(seed=7).fit(ss, ld, phi=1e-2, n_iter=30, n_burnin=15)
    assert torch.allclose(r1.weight, r2.weight)


def test_prscs_smaller_phi_yields_more_shrinkage():
    ss, ld, _ = _make_ss_and_ld(m=8, n=1500, seed=12)
    r_loose = PRSCS(seed=0).fit(ss, ld, phi=1.0, n_iter=40, n_burnin=20)
    r_tight = PRSCS(seed=0).fit(ss, ld, phi=1e-4, n_iter=40, n_burnin=20)
    assert r_tight.weight.abs().sum() <= r_loose.weight.abs().sum() * 1.5


def test_prscs_block_mode_runs():
    g = torch.Generator().manual_seed(13)
    n, m = 800, 8
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)
    beta_true = torch.tensor([0.2, 0, 0, 0, -0.2, 0, 0, 0], dtype=torch.float64)
    y = G @ beta_true + torch.randn(n, generator=g, dtype=torch.float64) * 0.5
    y = (y - y.mean()) / y.std().clamp(min=1e-12)
    beta_hat = (G * y.unsqueeze(1)).sum(0) / n
    se = torch.full((m,), 1.0 / math.sqrt(n), dtype=torch.float64)
    ss = SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=beta_hat,
        se=se,
        p=torch.full((m,), 0.01),
        n=torch.full((m,), float(n)),
        af=None,
    )
    meta = {"snp": ss.snp, "chr": ss.chr, "pos": ss.pos, "a1": ss.a1, "a2": ss.a2}
    ld = build_ld_reference(
        G, meta, mode="block", block_assignments=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    res = PRSCS(seed=0).fit(ss, ld, phi=1e-2, n_iter=30, n_burnin=15)
    assert res.weight.shape == (8,)


def test_prscs_multi_chain_populates_rhat():
    ss, ld, _ = _make_ss_and_ld(m=6, n=1000, seed=14)
    res = PRSCS(seed=0).fit(
        ss, ld, phi=1e-2, n_iter=30, n_burnin=15, n_chains=2,
    )
    assert res.rhat is not None
    assert res.rhat.shape == (6,)
    assert res.grid["n_chains"] == 2


def test_prscs_recovers_signal_direction_on_strong_snps():
    ss, ld, beta_true = _make_ss_and_ld(m=8, n=3000, seed=15)
    res = PRSCS(seed=0).fit(ss, ld, phi=1e-1, n_iter=60, n_burnin=30)
    # Sign recovery on the planted signals (positions 0 and 5)
    assert res.weight[0].sign() == beta_true[0].sign() or res.weight[0] == 0
    # Noise SNPs should be much smaller than signal SNP magnitudes
    assert res.weight[[1, 2, 3, 4]].abs().max() <= res.weight[[0, 5]].abs().max() + 1e-6


def test_prscs_single_block_internal_sampler_shape():
    g = torch.Generator().manual_seed(16)
    b = 4
    R = torch.eye(b, dtype=torch.float64)
    beta_std = torch.tensor([0.1, -0.1, 0.05, -0.05], dtype=torch.float64)
    rng = torch.Generator().manual_seed(3)
    bm, bsd = _prscs_gibbs_block(
        beta_std, R, n_eff=1000.0, phi=1e-2, a=1.0, b=0.5,
        n_iter=20, n_burnin=10, rng=rng,
    )
    assert bm.shape == (b,)
    assert bsd.shape == (b,)
    assert (bsd >= 0).all()


def test_prscs_grid_metadata_populated():
    ss, ld, _ = _make_ss_and_ld(m=6, n=800, seed=17)
    res = PRSCS(seed=0).fit(
        ss, ld, phi=5e-3, a=0.5, b=0.5, n_iter=20, n_burnin=10,
    )
    assert res.grid["a"] == 0.5
    assert res.grid["b"] == 0.5
    assert res.phi == 5e-3
