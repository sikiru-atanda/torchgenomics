"""Tests for torchgwas.pgs.ldpred2 — LDpred2-inf, LDpred2-grid, LDpred2-auto."""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.pgs.base import LDReference
from torchgwas.pgs.ld_ref import build_ld_reference
from torchgwas.pgs.ldpred2 import (
    LDpred2Auto,
    LDpred2Grid,
    LDpred2Inf,
    _marginal_beta_std,
)
from torchgwas.postgwas._sumstats import SumStats


def _sim_sumstats_with_truth(
    m=30, n=2000, h2=0.3, p_causal=0.2, seed=0,
):
    """Simulate marginal sumstats with known causal structure.

    Produces standardized genotypes G (n, m) (independent), plants causal
    effects, computes y = G @ beta_true + noise, regresses per SNP to get
    marginal (beta_hat, se) consistent with a standard GWAS.
    """
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    # standardize per column
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)

    beta_true = torch.zeros(m, dtype=torch.float64)
    n_causal = max(int(round(m * p_causal)), 1)
    causal_idx = torch.randperm(m, generator=g)[:n_causal]
    prior_var = h2 / n_causal
    beta_true[causal_idx] = torch.randn(n_causal, generator=g, dtype=torch.float64) * math.sqrt(prior_var)

    y = G @ beta_true + torch.randn(n, generator=g, dtype=torch.float64) * math.sqrt(1.0 - h2)
    y = (y - y.mean()) / y.std().clamp(min=1e-12)

    # Per-SNP OLS (univariate)
    gty = (G * y.unsqueeze(1)).sum(0)
    beta_hat = gty / n
    # residual variance assuming single-SNP model
    res_var = ((y.unsqueeze(1) - G * beta_hat.unsqueeze(0)) ** 2).mean(0)
    se = torch.sqrt(res_var / n)
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
    meta = {
        "snp": ss.snp,
        "chr": ss.chr,
        "pos": ss.pos,
        "a1": ss.a1,
        "a2": ss.a2,
    }
    return ss, G, beta_true, meta


# ---------------------------------------------------------------------------


def test_marginal_beta_std_helper():
    ss = SumStats(
        chr=["1", "1"],
        pos=[100, 200],
        snp=["rs1", "rs2"],
        a1=["A", "A"],
        a2=["G", "G"],
        beta=torch.tensor([0.2, 0.4], dtype=torch.float64),
        se=torch.tensor([0.1, 0.1], dtype=torch.float64),
        p=torch.tensor([1e-3, 1e-5], dtype=torch.float64),
        n=torch.tensor([1000.0, 1000.0]),
        af=None,
    )
    beta_std, n_eff = _marginal_beta_std(ss)
    z = ss.beta / ss.se
    assert torch.allclose(beta_std, z / torch.sqrt(n_eff))


def test_ldpred2_inf_matches_manual_closed_form():
    ss, G, beta_true, meta = _sim_sumstats_with_truth(m=10, n=3000, seed=1)
    ld = build_ld_reference(G, meta, mode="full")
    h2 = 0.3
    res = LDpred2Inf().fit(ss, ld, h2=h2)
    # Manual closed-form
    beta_std, n_eff = _marginal_beta_std(ss)
    m = ss.m
    n_avg = float(n_eff.median())
    ridge = m / (n_avg * h2)
    R = ld.R_full
    manual = torch.linalg.solve(
        R + ridge * torch.eye(m, dtype=R.dtype), beta_std
    )
    assert torch.allclose(res.weight, manual, atol=1e-10)
    assert res.h2 == h2
    assert res.converged is True


def test_ldpred2_inf_block_mode_equals_full_when_single_block():
    ss, G, _, meta = _sim_sumstats_with_truth(m=8, n=2000, seed=2)
    ld_full = build_ld_reference(G, meta, mode="full")
    ld_block = build_ld_reference(
        G, meta, mode="block", block_assignments=[0] * 8,
    )
    r_full = LDpred2Inf().fit(ss, ld_full, h2=0.2)
    r_block = LDpred2Inf().fit(ss, ld_block, h2=0.2)
    assert torch.allclose(r_full.weight, r_block.weight, atol=1e-10)


def test_ldpred2_inf_shrinks_toward_zero_for_small_h2():
    ss, G, _, meta = _sim_sumstats_with_truth(m=12, n=2000, seed=3)
    ld = build_ld_reference(G, meta, mode="full")
    r_big = LDpred2Inf().fit(ss, ld, h2=0.9)
    r_small = LDpred2Inf().fit(ss, ld, h2=0.01)
    # Smaller h2 -> stronger ridge -> smaller weights
    assert r_small.weight.abs().sum() < r_big.weight.abs().sum()


def test_ldpred2_inf_zero_sumstats_gives_zero_weights():
    m = 5
    ld = LDReference(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.3, dtype=torch.float64),
        mode="full",
        R_full=torch.eye(m, dtype=torch.float64),
        n_ref=100,
    )
    ss = SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.zeros(m, dtype=torch.float64),
        se=torch.full((m,), 0.01, dtype=torch.float64),
        p=torch.full((m,), 1.0),
        n=torch.full((m,), 1000.0),
        af=None,
    )
    res = LDpred2Inf().fit(ss, ld, h2=0.2)
    assert torch.allclose(res.weight, torch.zeros(m, dtype=torch.float64))


def test_ldpred2_grid_runs_and_produces_pip():
    ss, G, _, meta = _sim_sumstats_with_truth(m=12, n=1500, p_causal=0.25, seed=4)
    ld = build_ld_reference(G, meta, mode="full")
    res = LDpred2Grid(seed=0).fit(
        ss, ld,
        grid_p=[1e-2, 1e-1],
        grid_h2=[0.2],
        n_iter=60, n_burnin=30,
    )
    assert res.method == "ldpred2-grid"
    assert res.weight.shape == (12,)
    assert res.pip is not None and res.pip.shape == (12,)
    assert (res.pip >= 0).all() and (res.pip <= 1).all()
    assert res.grid["n_cells"] == 2


def test_ldpred2_grid_deterministic_with_fixed_seed():
    ss, G, _, meta = _sim_sumstats_with_truth(m=10, n=1000, seed=5)
    ld = build_ld_reference(G, meta, mode="full")
    r1 = LDpred2Grid(seed=42).fit(
        ss, ld, grid_p=[1e-2], grid_h2=[0.2], n_iter=50, n_burnin=25,
    )
    r2 = LDpred2Grid(seed=42).fit(
        ss, ld, grid_p=[1e-2], grid_h2=[0.2], n_iter=50, n_burnin=25,
    )
    assert torch.allclose(r1.weight, r2.weight)


def test_ldpred2_auto_runs_multi_chain_and_populates_rhat():
    ss, G, _, meta = _sim_sumstats_with_truth(m=12, n=1500, seed=6)
    ld = build_ld_reference(G, meta, mode="full")
    res = LDpred2Auto(seed=0).fit(
        ss, ld, n_iter=80, n_burnin=40, n_chains=3, h2_init=0.2, p_init=0.05,
    )
    assert res.method == "ldpred2-auto"
    assert res.weight.shape == (12,)
    assert res.rhat is not None
    assert res.rhat.shape == (12,)
    assert res.h2 is not None
    assert res.p_causal is not None
    assert res.grid["n_chains"] == 3
    assert len(res.grid["chain_p"]) == 3
    assert len(res.grid["chain_h2"]) == 3


def test_ldpred2_auto_recovers_nonzero_signal_direction():
    ss, G, beta_true, meta = _sim_sumstats_with_truth(
        m=10, n=5000, h2=0.5, p_causal=0.3, seed=7,
    )
    ld = build_ld_reference(G, meta, mode="full")
    res = LDpred2Auto(seed=0).fit(
        ss, ld, n_iter=120, n_burnin=60, n_chains=2,
    )
    # Sign recovery on the large-effect SNPs
    big_mask = beta_true.abs() > beta_true.abs().median()
    if big_mask.sum() > 0:
        sign_agree = ((res.weight[big_mask].sign() == beta_true[big_mask].sign()).float().mean())
        assert float(sign_agree) >= 0.5


def test_ldpred2_grid_block_mode_runs():
    ss, G, _, meta = _sim_sumstats_with_truth(m=8, n=1200, seed=8)
    ld = build_ld_reference(
        G, meta, mode="block", block_assignments=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    res = LDpred2Grid(seed=0).fit(
        ss, ld, grid_p=[1e-2], grid_h2=[0.2], n_iter=40, n_burnin=20,
    )
    assert res.weight.shape == (8,)


def test_ldpred2_auto_block_mode_runs():
    ss, G, _, meta = _sim_sumstats_with_truth(m=8, n=1200, seed=9)
    ld = build_ld_reference(
        G, meta, mode="block", block_assignments=[0, 0, 0, 0, 1, 1, 1, 1],
    )
    res = LDpred2Auto(seed=0).fit(
        ss, ld, n_iter=50, n_burnin=25, n_chains=2,
    )
    assert res.weight.shape == (8,)


def test_ldpred2_inf_warns_when_n_is_missing_everywhere():
    ss, G, _, meta = _sim_sumstats_with_truth(m=6, n=500, seed=10)
    ld = build_ld_reference(G, meta, mode="full")
    ss.n = torch.full((ss.m,), float("nan"))
    with pytest.raises(ValueError, match="sample size"):
        LDpred2Inf().fit(ss, ld, h2=0.2)
