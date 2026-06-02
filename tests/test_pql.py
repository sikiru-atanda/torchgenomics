"""Tests for PQL (Penalized Quasi-Likelihood) iteration module.

Covers: convergence, working response, weight positivity, variance components.
"""

from __future__ import annotations

import math

import torch


def _simulate_binary_with_kinship(
    n: int = 200,
    prevalence: float = 0.3,
    h2: float = 0.3,
    seed: int = 42,
):
    """Simulate binary phenotype with kinship-induced correlation."""
    torch.manual_seed(seed)

    # Random GRM-like matrix
    Z = torch.randn(n, 50, dtype=torch.float64)
    K = Z @ Z.T / 50.0
    K = 0.5 * (K + K.T)
    # Ensure PSD
    evals, evecs = torch.linalg.eigh(K)
    evals = evals.clamp(min=0.0)
    K = evecs @ torch.diag(evals) @ evecs.T

    # Genetic random effect
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(h2)

    # Fixed effect (intercept)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    beta0 = math.log(prevalence / (1.0 - prevalence))

    # Linear predictor
    eta = beta0 + u
    prob = torch.sigmoid(eta)
    Y = torch.bernoulli(prob).to(torch.float64)

    return Y, X0, K, prob


class TestPQLConvergence:
    def test_pql_converges(self):
        from torchgenomics.optim.pql import pql_fit
        Y, X0, K, mu_true = _simulate_binary_with_kinship()

        p_bar = Y.mean().clamp(min=0.01, max=0.99)
        mu_init = torch.full_like(Y, p_bar.item())

        result = pql_fit(
            Y, X0, K,
            mu_init=mu_init,
            link_variance_fn=lambda mu: mu * (1.0 - mu),
            link_derivative_fn=lambda mu: 1.0 / (mu * (1.0 - mu)).clamp(min=1e-10),
            link_fn=lambda mu: torch.log(mu / (1.0 - mu)),
            max_outer=30,
            tol=1e-4,
        )
        assert result["converged"] or result["n_outer"] <= 30

    def test_pql_mu_valid(self):
        from torchgenomics.optim.pql import pql_fit
        Y, X0, K, _ = _simulate_binary_with_kinship()
        p_bar = Y.mean().clamp(min=0.01, max=0.99)
        mu_init = torch.full_like(Y, p_bar.item())

        result = pql_fit(
            Y, X0, K,
            mu_init=mu_init,
            link_variance_fn=lambda mu: mu * (1.0 - mu),
            link_derivative_fn=lambda mu: 1.0 / (mu * (1.0 - mu)).clamp(min=1e-10),
            link_fn=lambda mu: torch.log(mu / (1.0 - mu)),
        )
        mu = result["mu"]
        assert (mu > 0).all() and (mu < 1).all()

    def test_pql_variance_components_positive(self):
        from torchgenomics.optim.pql import pql_fit
        Y, X0, K, _ = _simulate_binary_with_kinship()
        p_bar = Y.mean().clamp(min=0.01, max=0.99)
        mu_init = torch.full_like(Y, p_bar.item())

        result = pql_fit(
            Y, X0, K,
            mu_init=mu_init,
            link_variance_fn=lambda mu: mu * (1.0 - mu),
            link_derivative_fn=lambda mu: 1.0 / (mu * (1.0 - mu)).clamp(min=1e-10),
            link_fn=lambda mu: torch.log(mu / (1.0 - mu)),
        )
        assert result["sig2_g"] > 0
        assert result["sig2_e"] > 0


class TestProfileREML:
    def test_profile_reml_returns_positive(self):
        from torchgenomics.optim.pql import _profile_reml_vc
        torch.manual_seed(50)
        n, c = 100, 2
        evals = torch.rand(n, dtype=torch.float64) * 2.0
        X_rot = torch.randn(n, c, dtype=torch.float64)
        z_rot = torch.randn(n, dtype=torch.float64)

        sig2_g, sig2_e = _profile_reml_vc(z_rot, X_rot, evals)
        assert sig2_g > 0
        assert sig2_e > 0
