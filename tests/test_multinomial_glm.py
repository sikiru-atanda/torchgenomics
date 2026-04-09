"""Tests for Phase 31: Multinomial GLM (multinomial logit GWAS).

Covers: IRLS convergence, multi-df score test, null calibration,
power, edge cases, protocol.
"""

from __future__ import annotations

import pytest
import torch

from torchgwas.models.base import VariantMeta
from torchgwas.models.multinomial_glm import MultinomialGLM


def _simulate_multinomial_data(
    n: int = 300,
    m: int = 30,
    n_classes: int = 3,
    causal_beta: float = 0.0,
    seed: int = 42,
):
    """Simulate multinomial phenotype."""
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Class probabilities modulated by causal SNP
    g0 = G[:, 0] - G[:, 0].mean()
    logits = torch.zeros(n, n_classes, dtype=torch.float64)
    if causal_beta != 0:
        logits[:, 0] = causal_beta * g0  # SNP affects class 0 vs others

    probs = torch.softmax(logits, dim=1)
    Y = torch.multinomial(probs, 1).squeeze().to(torch.float64)

    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, X0, vmeta


class TestMultinomialGLMFit:
    def test_converges(self):
        G, Y, X0, vmeta = _simulate_multinomial_data()
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        assert nf.converged

    def test_probs_sum_to_one(self):
        G, Y, X0, vmeta = _simulate_multinomial_data()
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        pi = nf._glm_pi
        sums = pi.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_auto_detect_classes(self):
        G, Y, X0, vmeta = _simulate_multinomial_data(n_classes=4)
        model = MultinomialGLM()  # n_classes=None → auto-detect
        nf = model.fit_null(Y, X0)
        assert nf._glm_n_classes == 4


class TestMultinomialGLMScan:
    def test_pvalues_valid(self):
        G, Y, X0, vmeta = _simulate_multinomial_data()
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_stat_nonneg(self):
        G, Y, X0, vmeta = _simulate_multinomial_data()
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.stat >= 0).all()

    def test_null_calibration(self):
        """Under null, p-values should not be inflated."""
        G, Y, X0, vmeta = _simulate_multinomial_data(
            n=500, m=100, causal_beta=0.0, seed=100)
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert median_p > 0.15

    def test_power_detects_signal(self):
        """Strong causal effect should be detected."""
        G, Y, X0, vmeta = _simulate_multinomial_data(
            n=500, m=30, n_classes=3, causal_beta=1.5, seed=200)
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        # Causal SNP should have small p-value
        assert result.p[0].item() < 0.05


class TestMultinomialGLMEdge:
    def test_two_classes(self):
        """J=2 should work (reduces to binary-like)."""
        G, Y, X0, vmeta = _simulate_multinomial_data(n_classes=2, seed=300)
        model = MultinomialGLM(n_classes=2)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()

    def test_beta_is_nan(self):
        """Multinomial score test returns NaN for beta/SE."""
        G, Y, X0, vmeta = _simulate_multinomial_data()
        model = MultinomialGLM(n_classes=3)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert torch.isnan(result.beta).all()
        assert torch.isnan(result.se).all()


class TestMultinomialProtocol:
    def test_basemodel_conformance(self):
        model = MultinomialGLM(n_classes=3)
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
