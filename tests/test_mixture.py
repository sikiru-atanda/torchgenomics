"""Phase 17: Chi-squared mixture distribution tests (Davies + Liu)."""

from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from torchgenomics.stats.mixture import davies_pvalue, liu_pvalue, mixture_chi2_pvalue


class TestDavies:
    """Tests for Davies method chi-squared mixture p-values."""

    def test_equal_eigenvalues_match_chi2(self):
        """When all lambdas = 1, Q ~ chi2(k). Davies should match."""
        k = 5
        lambdas = torch.ones(k, dtype=torch.float64)

        # Test at several quantiles
        for q in [3.0, 5.0, 11.0, 15.0]:
            p_davies = davies_pvalue(q, lambdas)
            p_exact = chi2.sf(q, df=k)
            assert abs(p_davies - p_exact) < 0.01, (
                f"q={q}: Davies={p_davies:.6f}, chi2({k})={p_exact:.6f}"
            )

    def test_single_eigenvalue(self):
        """Q = lambda * chi2(1). P(Q > q) = P(chi2(1) > q/lambda)."""
        lam_val = 3.5
        lambdas = torch.tensor([lam_val], dtype=torch.float64)

        for q in [1.0, 5.0, 10.0]:
            p_davies = davies_pvalue(q, lambdas)
            p_exact = chi2.sf(q / lam_val, df=1)
            assert abs(p_davies - p_exact) < 0.01, (
                f"q={q}: Davies={p_davies:.6f}, exact={p_exact:.6f}"
            )

    def test_two_eigenvalues(self):
        """Two eigenvalues: verify against simulation."""
        torch.manual_seed(42)
        lambdas = torch.tensor([2.0, 0.5], dtype=torch.float64)

        # Simulate Q = 2*chi2_1 + 0.5*chi2_1 via sampling
        n_sim = 100000
        chi2_samples = torch.distributions.Chi2(df=1).sample((n_sim, 2))
        Q_samples = (chi2_samples * lambdas.unsqueeze(0)).sum(dim=1).numpy()

        q_test = 5.0
        p_sim = (Q_samples > q_test).mean()
        p_davies = davies_pvalue(q_test, lambdas)

        # Allow generous tolerance due to simulation variance
        assert abs(p_davies - p_sim) < 0.02, (
            f"Davies={p_davies:.4f}, simulated={p_sim:.4f}"
        )

    def test_zero_stat_returns_one(self):
        """Q = 0 should give p-value = 1."""
        lambdas = torch.tensor([1.0, 2.0], dtype=torch.float64)
        assert davies_pvalue(0.0, lambdas) == 1.0

    def test_extreme_eigenvalue_ratio(self):
        """Large eigenvalue ratio should not crash."""
        lambdas = torch.tensor([100.0, 0.001], dtype=torch.float64)
        p = davies_pvalue(50.0, lambdas)
        assert math.isfinite(p)
        assert 0 < p <= 1.0


class TestLiu:
    """Tests for Liu moment-matching approximation."""

    def test_equal_eigenvalues_approximate(self):
        """Liu should be close to chi2(k) when all lambdas = 1."""
        k = 5
        lambdas = torch.ones(k, dtype=torch.float64)

        q = 9.0
        p_liu = liu_pvalue(q, lambdas)
        p_exact = chi2.sf(q, df=k)
        # Liu is approximate, allow wider tolerance
        assert abs(p_liu - p_exact) < 0.05

    def test_returns_valid_range(self):
        """Liu should always return p in (0, 1]."""
        lambdas = torch.tensor([3.0, 1.0, 0.5], dtype=torch.float64)
        for q in [0.1, 1.0, 5.0, 20.0, 100.0]:
            p = liu_pvalue(q, lambdas)
            assert 0 < p <= 1.0, f"q={q}: p={p}"


class TestDispatcher:
    """Tests for mixture_chi2_pvalue dispatcher."""

    def test_davies_method(self):
        """Dispatcher with method='davies' uses Davies."""
        lambdas = torch.ones(3, dtype=torch.float64)
        p = mixture_chi2_pvalue(5.0, lambdas, method="davies")
        p_exact = chi2.sf(5.0, df=3)
        assert abs(p - p_exact) < 0.01

    def test_liu_method(self):
        """Dispatcher with method='liu' uses Liu."""
        lambdas = torch.ones(3, dtype=torch.float64)
        p = mixture_chi2_pvalue(5.0, lambdas, method="liu")
        assert 0 < p <= 1.0
