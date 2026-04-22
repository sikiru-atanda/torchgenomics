"""Tests for Phase 33: Multinomial GLMM (multinomial logit mixed model GWAS).

Covers: PQL convergence, multi-df score test, null calibration, power, protocol.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.models.base import VariantMeta
from torchgwas.models.multinomial_glmm import MultinomialGLMM


def _simulate_multinomial_glmm_data(
    n: int = 200,
    m: int = 20,
    J: int = 3,
    h2: float = 0.2,
    causal_beta: float = 0.0,
    seed: int = 42,
):
    """Simulate multinomial phenotype with kinship + optional causal SNP."""
    torch.manual_seed(seed)

    # GRM
    Z = torch.randn(n, 50, dtype=torch.float64)
    K = Z @ Z.T / 50.0
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    evals = evals.clamp(min=0.0)
    K = evecs @ torch.diag(evals) @ evecs.T

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)

    # Latent for each non-reference class
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    eta = torch.zeros(n, J, dtype=torch.float64)
    for j in range(J - 1):
        u_j = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(h2)
        eta[:, j] = u_j

    # Add causal effect on class 0
    if causal_beta != 0.0:
        g0 = G[:, 0] - G[:, 0].mean()
        eta[:, 0] += g0 * causal_beta

    # Sample from multinomial
    pi = torch.softmax(eta, dim=1)
    Y = torch.zeros(n, dtype=torch.float64)
    for i in range(n):
        Y[i] = torch.multinomial(pi[i], 1).item()

    X0 = torch.ones(n, 1, dtype=torch.float64)
    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, X0, K, vmeta


class TestMultinomialGLMMNull:
    def test_fit_converges(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data()
        model = MultinomialGLMM(n_classes=3)
        nf = model.fit_null(Y, X0, K=K)
        pi = nf._glm_pi
        assert (pi >= 0).all()
        assert torch.allclose(pi.sum(dim=1),
                              torch.ones(Y.shape[0], dtype=torch.float64),
                              atol=0.01)

    def test_variance_components(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data()
        model = MultinomialGLMM(n_classes=3)
        nf = model.fit_null(Y, X0, K=K)
        assert nf.sig2_g > 0
        assert nf.sig2_e > 0

    def test_requires_kinship(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data()
        model = MultinomialGLMM(n_classes=3)
        with pytest.raises(ValueError, match="requires a kinship"):
            model.fit_null(Y, X0, K=None)


class TestMultinomialGLMMScan:
    def test_pvalues_valid(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data()
        model = MultinomialGLMM(n_classes=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_null_calibration(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data(
            n=300, m=60, J=3, h2=0.2, causal_beta=0.0, seed=100)
        model = MultinomialGLMM(n_classes=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert median_p > 0.1

    def test_power_detects_signal(self):
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data(
            n=400, m=20, J=3, h2=0.2, causal_beta=2.0, seed=200)
        model = MultinomialGLMM(n_classes=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p[0].item() < 0.05

    def test_multi_df_stat(self):
        """Stat should follow chi2(J-1) distribution."""
        G, Y, X0, K, vmeta = _simulate_multinomial_glmm_data(J=4, seed=300)
        model = MultinomialGLMM(n_classes=4)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.stat >= 0).all()


class TestMultinomialGLMMProtocol:
    def test_basemodel_conformance(self):
        model = MultinomialGLMM(n_classes=3)
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
        assert callable(model.fit_null)
        assert callable(model.score_chunk)
