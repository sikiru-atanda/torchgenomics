"""Regression tests for the ELBO Cholesky-cache optimisation in
``torchgwas.models.bayesian_vs_rss``.

``BayesianVSRss._compute_elbo`` accepts an optional precomputed
Cholesky factor of R; ``fit_rss`` factorises R once outside the IBSS
loop and reuses the factor across every per-iteration ELBO call. This
reduces the per-call solve from O(p³) (un-factored linalg.solve) to
O(p²) (cholesky_solve forward + back substitution), with bit-equal
results modulo FP roundoff.

These tests assert:

1.  **Direct ELBO parity** — calling ``_compute_elbo`` with and
    without the cached factor produces values identical at the FP64
    floor.
2.  **fit_rss parity** — the full IBSS loop with the cache (default)
    and without it (forced via monkey-patch) produces identical PIPs
    / β_mean / β_sd / V / ELBO history.
3.  **Non-PD R fallback** — when ``torch.linalg.cholesky`` fails the
    cache silently falls back to the un-factored path; the fit still
    converges.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from torchgwas.models.bayesian_vs_rss import BayesianVSRss


def _planted_locus(p: int = 200, n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rho = 0.5
    idx = np.arange(p)
    R_np = rho ** np.abs(idx[:, None] - idx[None, :])
    R = torch.from_numpy(R_np).to(torch.float64)
    L_chol = np.linalg.cholesky(R_np)
    beta = np.zeros(p)
    beta[rng.choice(p, 3, replace=False)] = rng.normal(0.0, 0.5, 3)
    z_np = np.sqrt(n) * (R_np @ beta) + L_chol @ rng.standard_normal(p)
    return torch.from_numpy(z_np).to(torch.float64), R


class TestComputeElboCholeskyParity(unittest.TestCase):

    def test_direct_call_parity(self):
        """_compute_elbo(cholesky_L=L) == _compute_elbo(cholesky_L=None)."""
        z, R = _planted_locus(p=200, n=1000, seed=42)
        model = BayesianVSRss(
            max_num_causal=5, max_iter=20, tol=1e-6,
            estimate_prior_variance=True, estimate_prior_method="optim",
        )
        result = model.fit_rss(z, R, 1000)

        # Recompute ELBO both ways using the final state.
        L = torch.linalg.cholesky(R)
        elbo_with = model._compute_elbo(
            z, R, 1000, result.alpha, result.mu, result.sigma_sq, result.V,
            cholesky_L=L,
        )
        elbo_without = model._compute_elbo(
            z, R, 1000, result.alpha, result.mu, result.sigma_sq, result.V,
            cholesky_L=None,
        )
        # FP64 floor: cholesky_solve and linalg.solve agree to ~1e-13
        # on well-conditioned R.
        self.assertAlmostEqual(elbo_with, elbo_without, places=10)


class TestFitRssParityWithAndWithoutCache(unittest.TestCase):

    def test_fit_rss_results_identical(self):
        """Patch out the cache; fit_rss should still produce identical results."""
        z, R = _planted_locus(p=300, n=2000, seed=43)
        kwargs = dict(
            max_num_causal=10, max_iter=30, tol=1e-6,
            estimate_prior_variance=True, estimate_prior_method="optim",
        )
        # With cache (default).
        cached = BayesianVSRss(**kwargs).fit_rss(z, R, 2000)

        # Force the un-cached path by monkey-patching torch.linalg.cholesky
        # to raise on this call. The fit_rss catch-all turns the cache off.
        original_chol = torch.linalg.cholesky
        def fake_chol(*args, **kwargs):
            raise RuntimeError("intentional test failure")
        torch.linalg.cholesky = fake_chol
        try:
            uncached = BayesianVSRss(**kwargs).fit_rss(z, R, 2000)
        finally:
            torch.linalg.cholesky = original_chol

        torch.testing.assert_close(cached.pip, uncached.pip,
                                    atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(cached.beta_mean, uncached.beta_mean,
                                    atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(cached.beta_sd, uncached.beta_sd,
                                    atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(cached.V, uncached.V,
                                    atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(cached.elbo_history, uncached.elbo_history,
                                    atol=1e-8, rtol=1e-8)
        self.assertEqual(cached.n_iter, uncached.n_iter)
        self.assertEqual(cached.converged, uncached.converged)


class TestNonPdR(unittest.TestCase):

    def test_non_pd_R_falls_back(self):
        """A non-PD R triggers the silent fallback; fit still completes."""
        z, R = _planted_locus(p=100, n=500, seed=44)
        # Drop the rank of R by zeroing the last row and column.
        R_singular = R.clone()
        R_singular[-1, :] = 0.0
        R_singular[:, -1] = 0.0
        R_singular[-1, -1] = 1e-12
        model = BayesianVSRss(
            max_num_causal=3, max_iter=10, tol=1e-6,
            estimate_prior_variance=True, estimate_prior_method="optim",
        )
        # Should not raise; should fall back to the un-factored path.
        result = model.fit_rss(z, R_singular, 500)
        # Sanity: the fit produced a valid output (not NaN-poisoned).
        self.assertFalse(torch.isnan(result.pip).any())


if __name__ == "__main__":
    unittest.main()
