"""Tests for saddlepoint approximation (SPA) module.

Covers: CGF computation, SPA correctness, tail calibration, edge cases.
"""

from __future__ import annotations

import math
import pytest
import torch

from torchgwas.stats.spa import (
    _cgf, _cgf_deriv1, _cgf_deriv2, _solve_saddlepoint,
    saddlepoint_pvalue,
)


class TestCGF:
    def test_cgf_at_zero(self):
        """K(0) should be 0 (log(q*1 + mu*1) = log(1) = 0)."""
        mu = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        g = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        t = torch.tensor(0.0, dtype=torch.float64)
        K = _cgf(t, mu, g)
        assert abs(K.item()) < 1e-10

    def test_cgf_deriv1_at_zero(self):
        """K'(0) should equal E[sum g_i (Y_i - mu_i)] = 0 under null."""
        mu = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        g = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        K1 = _cgf_deriv1(0.0, mu, g)
        assert abs(K1) < 1e-10

    def test_cgf_deriv2_positive(self):
        """K''(t) should be positive (variance)."""
        mu = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        g = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        K2 = _cgf_deriv2(0.0, mu, g)
        assert K2 > 0


class TestSaddlepoint:
    def test_saddlepoint_near_zero(self):
        """For small observed score, saddlepoint should be near 0."""
        mu = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
        g = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        t_hat = _solve_saddlepoint(0.01, mu, g)
        assert abs(t_hat) < 1.0


class TestSPAPvalue:
    def test_pvalues_valid(self):
        """SPA p-values should be in [0, 1]."""
        torch.manual_seed(10)
        n, m = 100, 20
        mu = torch.full((n,), 0.3, dtype=torch.float64)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(mu).to(torch.float64)
        score = G.T @ (Y - mu)

        p = saddlepoint_pvalue(score, mu, G, threshold=0.5)
        assert (p >= 0).all()
        assert (p <= 1).all()

    def test_spa_vs_chi2_body(self):
        """In the body (small stats), SPA ≈ chi2."""
        torch.manual_seed(11)
        n, m = 200, 50
        mu = torch.full((n,), 0.5, dtype=torch.float64)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(mu).to(torch.float64)
        score = G.T @ (Y - mu)

        from torchgwas.stats.tests import chi2_sf
        W = mu * (1.0 - mu)
        V = ((G ** 2).T @ W)
        chi2_stat = score ** 2 / torch.clamp(V, min=1e-20)
        p_chi2 = chi2_sf(chi2_stat, df=1)

        # With very high threshold, SPA should match chi2
        p_spa = saddlepoint_pvalue(score, mu, G, threshold=1000.0)
        assert torch.allclose(p_spa, p_chi2, atol=1e-6)

    def test_imbalanced_no_crash(self):
        """SPA should handle extreme imbalance without crashing."""
        torch.manual_seed(12)
        n = 500
        mu = torch.full((n,), 0.02, dtype=torch.float64)
        G = torch.randint(0, 3, (n, 10), dtype=torch.float64)
        Y = torch.bernoulli(mu).to(torch.float64)
        score = G.T @ (Y - mu)

        p = saddlepoint_pvalue(score, mu, G, threshold=0.5)
        assert (p >= 0).all()
        assert (p <= 1).all()
