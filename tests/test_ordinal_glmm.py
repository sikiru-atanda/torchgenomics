"""Tests for Phase 32: Ordinal GLMM (cumulative logit mixed model GWAS).

Covers: PQL convergence, score test, null calibration, power, protocol.
"""

from __future__ import annotations

import math
import pytest
import torch

from torchgwas.models.base import VariantMeta
from torchgwas.models.ordinal_glmm import OrdinalGLMM


def _simulate_ordinal_glmm_data(
    n: int = 200,
    m: int = 30,
    J: int = 3,
    h2: float = 0.2,
    causal_beta: float = 0.0,
    seed: int = 42,
):
    """Simulate ordinal phenotype with kinship + optional causal SNP."""
    torch.manual_seed(seed)

    # GRM
    Z = torch.randn(n, 50, dtype=torch.float64)
    K = Z @ Z.T / 50.0
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    evals = evals.clamp(min=0.0)
    K = evecs @ torch.diag(evals) @ evecs.T

    # Genotypes
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)

    # Latent variable
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(h2)
    eta = u

    if causal_beta != 0.0:
        g0 = G[:, 0] - G[:, 0].mean()
        eta = eta + g0 * causal_beta

    # Add logistic noise
    noise = torch.distributions.Gumbel(0, 1).sample((n,)).to(torch.float64) - \
            torch.distributions.Gumbel(0, 1).sample((n,)).to(torch.float64)
    latent = eta + noise

    # Threshold to categories
    thresholds = torch.linspace(-1.0, 1.0, J - 1, dtype=torch.float64)
    Y = torch.zeros(n, dtype=torch.float64)
    for j in range(J - 1):
        Y += (latent > thresholds[j]).float()

    X0 = torch.ones(n, 1, dtype=torch.float64)
    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, X0, K, vmeta


# ===================================================================
# Null model fitting
# ===================================================================

class TestOrdinalGLMMNull:
    def test_fit_null_produces_valid_probs(self):
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data()
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(Y, X0, K=K)
        pi = nf._glm_pi
        assert (pi >= 0).all()
        # Each row should sum close to 1
        assert torch.allclose(pi.sum(dim=1), torch.ones(Y.shape[0], dtype=torch.float64),
                              atol=0.01)

    def test_variance_components_estimated(self):
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data()
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(Y, X0, K=K)
        assert nf.sig2_g > 0
        assert nf.sig2_e > 0

    def test_requires_kinship(self):
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data()
        model = OrdinalGLMM(n_categories=3)
        with pytest.raises(ValueError, match="requires a kinship"):
            model.fit_null(Y, X0, K=None)


# ===================================================================
# Score test
# ===================================================================

class TestOrdinalGLMMScan:
    def test_pvalues_valid(self):
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data()
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_null_calibration(self):
        """Under null, median p-value should not be badly inflated."""
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data(
            n=300, m=80, J=3, h2=0.2, causal_beta=0.0, seed=100)
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert median_p > 0.1

    def test_power_detects_signal(self):
        """With strong causal effect, should detect signal."""
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data(
            n=400, m=30, J=3, h2=0.2, causal_beta=1.5, seed=200)
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p[0].item() < 0.05

    def test_four_categories(self):
        """Should work with J=4."""
        G, Y, X0, K, vmeta = _simulate_ordinal_glmm_data(
            n=200, m=20, J=4, seed=300)
        model = OrdinalGLMM(n_categories=4)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p.shape == (20,)
        assert (result.p >= 0).all()


# ===================================================================
# Protocol
# ===================================================================

class TestOrdinalGLMMProtocol:
    def test_basemodel_conformance(self):
        model = OrdinalGLMM(n_categories=3)
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
        assert callable(model.fit_null)
        assert callable(model.score_chunk)
