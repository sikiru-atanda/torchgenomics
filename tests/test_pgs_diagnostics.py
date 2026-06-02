"""Tests for torchgenomics.pgs.diagnostics — rhat, ess, check_convergence."""

from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.pgs.base import PGSResult
from torchgenomics.pgs.diagnostics import check_convergence, ess, rhat


def test_rhat_approx_one_for_identical_chains():
    torch.manual_seed(0)
    N = 1000
    x = torch.randn(N, dtype=torch.float64)
    chains = torch.stack([x, x.clone()], dim=0).unsqueeze(-1)  # (2, N, 1)
    r = rhat(chains)
    assert r.shape == (1,)
    # Identical chains -> B=0, R-hat = sqrt((N-1)/N) -> 1 as N -> infty
    assert float(r.item()) == pytest.approx(math.sqrt((N - 1) / N), abs=1e-10)


def test_rhat_large_for_disjoint_chains():
    N = 500
    torch.manual_seed(1)
    c1 = 10.0 + 0.1 * torch.randn(N, dtype=torch.float64)
    c2 = -10.0 + 0.1 * torch.randn(N, dtype=torch.float64)
    chains = torch.stack([c1, c2], dim=0)  # (2, N)
    r = rhat(chains)
    assert float(r.item()) > 5.0  # chains disagree strongly


def test_rhat_multi_parameter_shape():
    torch.manual_seed(2)
    chains = torch.randn(3, 200, 4, dtype=torch.float64)
    r = rhat(chains)
    assert r.shape == (4,)
    # independent Gaussian chains: R-hat near 1
    assert (r < 1.2).all()


def test_rhat_rejects_too_few_chains_or_iters():
    with pytest.raises(ValueError):
        rhat(torch.zeros(1, 10, 2))
    with pytest.raises(ValueError):
        rhat(torch.zeros(2, 1, 2))


def test_ess_close_to_N_for_iid_samples():
    torch.manual_seed(3)
    chains = torch.randn(2, 500, 2, dtype=torch.float64)
    e = ess(chains)
    assert e.shape == (2,)
    # For iid, ESS ~ M*N = 1000; allow wide tolerance
    assert (e > 300).all()


def test_ess_smaller_for_strongly_correlated_chain():
    torch.manual_seed(4)
    N = 1000
    # Build an AR(1) chain with rho=0.95
    rho = 0.95
    eps = torch.randn(N, dtype=torch.float64) * math.sqrt(1 - rho * rho)
    x = torch.zeros(N, dtype=torch.float64)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + eps[t]
    chains_ar = torch.stack([x, x.clone() + 0.01 * torch.randn(N, dtype=torch.float64)], dim=0)
    chains_iid = torch.randn(2, N, dtype=torch.float64)
    e_ar = float(ess(chains_ar).item())
    e_iid = float(ess(chains_iid).item())
    assert e_ar < e_iid


def test_check_convergence_uses_rhat_threshold():
    m = 10
    base = PGSResult(
        method="t",
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
        weight=torch.zeros(m),
    )
    base.rhat = torch.full((m,), 1.05)
    assert check_convergence(base, rhat_threshold=1.1) is True

    base.rhat = torch.cat([torch.full((5,), 1.05), torch.full((5,), 1.5)])
    assert check_convergence(base, rhat_threshold=1.1, min_fraction=0.95) is False
    assert check_convergence(base, rhat_threshold=1.1, min_fraction=0.4) is True


def test_check_convergence_fallback_when_rhat_missing():
    m = 3
    r = PGSResult(
        method="t",
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
        weight=torch.zeros(m),
        converged=True,
    )
    assert check_convergence(r) is True
    r.converged = False
    assert check_convergence(r) is False
