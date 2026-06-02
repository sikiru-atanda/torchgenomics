"""Tests for Phase 31: Ordinal GLM (cumulative logit / proportional odds GWAS).

Covers: IRLS convergence, threshold ordering, score test, null calibration,
power, edge cases, protocol.
"""

from __future__ import annotations

import torch

from torchgenomics.models.base import VariantMeta
from torchgenomics.models.ordinal_glm import OrdinalGLM


def _simulate_ordinal_data(
    n: int = 300,
    m: int = 50,
    n_categories: int = 3,
    causal_beta: float = 0.0,
    seed: int = 42,
):
    """Simulate ordinal phenotype via latent variable model."""
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Latent variable: z = causal_beta * g0 + epsilon
    g0 = G[:, 0] - G[:, 0].mean()
    z = causal_beta * g0 + torch.randn(n, dtype=torch.float64)

    # Thresholds: equally spaced
    thresholds = torch.linspace(-1, 1, n_categories - 1)
    Y = torch.zeros(n, dtype=torch.float64)
    for j in range(n_categories - 1):
        Y += (z > thresholds[j]).float()

    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, X0, vmeta


class TestOrdinalGLMFit:
    def test_converges(self):
        G, Y, X0, vmeta = _simulate_ordinal_data()
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(Y, X0)
        assert nf.converged

    def test_thresholds_ordered(self):
        G, Y, X0, vmeta = _simulate_ordinal_data(n_categories=4)
        model = OrdinalGLM(n_categories=4)
        nf = model.fit_null(Y, X0)
        thresholds = nf._glm_thresholds
        for j in range(len(thresholds) - 1):
            assert thresholds[j] <= thresholds[j + 1]

    def test_category_probs_valid(self):
        G, Y, X0, vmeta = _simulate_ordinal_data()
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(Y, X0)
        pi = nf._glm_pi
        assert (pi > 0).all()
        sums = pi.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


class TestOrdinalGLMScan:
    def test_pvalues_valid(self):
        G, Y, X0, vmeta = _simulate_ordinal_data()
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()
        assert result.p.shape[0] == G.shape[1]

    def test_null_calibration(self):
        """Under null (no causal), p-values should not be inflated."""
        G, Y, X0, vmeta = _simulate_ordinal_data(
            n=500, m=200, causal_beta=0.0, seed=100)
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert median_p > 0.15

    def test_power_detects_signal(self):
        """Strong causal effect should be detected."""
        G, Y, X0, vmeta = _simulate_ordinal_data(
            n=500, m=50, causal_beta=0.8, seed=200)
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p[0].item() < 0.05


class TestOrdinalGLMEdge:
    def test_binary_case(self):
        """J=2 should work as binary-like ordinal."""
        G, Y, X0, vmeta = _simulate_ordinal_data(n_categories=2, seed=300)
        model = OrdinalGLM(n_categories=2)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p.shape[0] == G.shape[1]

    def test_many_categories(self):
        """Should handle J=5 categories."""
        G, Y, X0, vmeta = _simulate_ordinal_data(
            n_categories=5, n=400, seed=301)
        model = OrdinalGLM(n_categories=5)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()


class TestOrdinalProtocol:
    def test_basemodel_conformance(self):
        model = OrdinalGLM(n_categories=3)
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
