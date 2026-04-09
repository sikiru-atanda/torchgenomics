"""Tests for Phase 31: Binary GLM (logistic regression GWAS).

Covers: IRLS convergence, score test, SPA calibration, Firth correction,
null calibration, power, edge cases, protocol conformance.
"""

from __future__ import annotations

import math
import pytest
import torch

from torchgwas.models.base import VariantMeta
from torchgwas.models.binary_glm import BinaryGLM


def _simulate_binary_data(
    n: int = 300,
    m: int = 50,
    prevalence: float = 0.3,
    causal_or: float = 1.0,
    seed: int = 42,
):
    """Simulate binary phenotype with optional causal SNP (index 0)."""
    torch.manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Linear predictor: intercept + causal effect
    logit_p = math.log(prevalence / (1.0 - prevalence))
    if causal_or != 1.0:
        g0 = G[:, 0] - G[:, 0].mean()
        logit_p = logit_p + g0 * math.log(causal_or)

    prob = torch.sigmoid(torch.tensor(logit_p, dtype=torch.float64) if isinstance(logit_p, float)
                         else logit_p)
    Y = torch.bernoulli(prob).to(torch.float64)

    vmeta = VariantMeta(
        snp=[f"SNP{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, X0, vmeta


# ===================================================================
# IRLS convergence
# ===================================================================

class TestBinaryGLMIRLS:
    def test_irls_converges(self):
        G, Y, X0, vmeta = _simulate_binary_data()
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        assert nf.converged

    def test_fitted_probs_valid(self):
        G, Y, X0, vmeta = _simulate_binary_data()
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        mu = nf._glm_mu
        assert (mu > 0).all() and (mu < 1).all()

    def test_firth_converges(self):
        G, Y, X0, vmeta = _simulate_binary_data(n=100, prevalence=0.05, seed=10)
        model = BinaryGLM(firth=True, use_spa=False)
        nf = model.fit_null(Y, X0)
        assert nf.converged

    def test_weights_positive(self):
        G, Y, X0, vmeta = _simulate_binary_data()
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        assert (nf._glm_W > 0).all()


# ===================================================================
# Score test
# ===================================================================

class TestBinaryGLMScan:
    def test_pvalues_valid(self):
        G, Y, X0, vmeta = _simulate_binary_data()
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_stat_nonneg(self):
        G, Y, X0, vmeta = _simulate_binary_data()
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.stat >= 0).all()

    def test_null_calibration(self):
        """Under null, p-values should not be severely inflated."""
        G, Y, X0, vmeta = _simulate_binary_data(
            n=500, m=200, prevalence=0.3, causal_or=1.0, seed=100)
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        assert median_p > 0.2  # should be near 0.5 under null

    def test_power_detects_signal(self):
        """With strong causal OR, should detect the signal."""
        G, Y, X0, vmeta = _simulate_binary_data(
            n=500, m=50, prevalence=0.3, causal_or=3.0, seed=200)
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        # Causal SNP (index 0) should have small p-value
        assert result.p[0].item() < 0.05


# ===================================================================
# SPA
# ===================================================================

class TestBinaryGLMSPA:
    def test_spa_produces_valid_pvalues(self):
        G, Y, X0, vmeta = _simulate_binary_data(n=300, seed=300)
        model = BinaryGLM(use_spa=True, spa_threshold=0.5)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_spa_agrees_with_chi2_in_body(self):
        """SPA and chi2 p-values should be close for non-extreme statistics."""
        G, Y, X0, vmeta = _simulate_binary_data(n=500, seed=301)
        model_spa = BinaryGLM(use_spa=True, spa_threshold=100.0)  # high threshold = no SPA
        model_chi2 = BinaryGLM(use_spa=False)
        nf = model_spa.fit_null(Y, X0)
        res_spa = model_spa.score_chunk(G, nf, vmeta)
        res_chi2 = model_chi2.score_chunk(G, nf, vmeta)
        # P-values should be very close (small numerical differences from
        # covariate adjustment in score_chunk vs SPA's simpler variance)
        assert torch.allclose(res_spa.p, res_chi2.p, atol=1e-3)

    def test_spa_imbalanced_valid(self):
        """SPA should produce valid p-values with imbalanced case-control."""
        G, Y, X0, vmeta = _simulate_binary_data(
            n=500, m=20, prevalence=0.05, seed=302)
        model = BinaryGLM(use_spa=True)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()


# ===================================================================
# Edge cases
# ===================================================================

class TestBinaryGLMEdge:
    def test_monomorphic_snp(self):
        """Monomorphic SNP should get p=1 or similar non-significant."""
        n, m = 100, 5
        torch.manual_seed(400)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        G[:, 0] = 1.0  # monomorphic
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)], chr=["1"]*m,
            pos=list(range(m)), a1=["A"]*m, a2=["G"]*m,
        )
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p[0].item() > 0.5

    def test_polyploid_dosage(self):
        """Should handle dosages in [0, 4] (tetraploid)."""
        n, m = 100, 10
        torch.manual_seed(401)
        G = torch.randint(0, 5, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)], chr=["1"]*m,
            pos=list(range(m)), a1=["A"]*m, a2=["G"]*m,
        )
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y, X0)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p.shape == (m,)
        assert (result.p >= 0).all()


# ===================================================================
# Protocol
# ===================================================================

class TestProtocol:
    def test_basemodel_conformance(self):
        model = BinaryGLM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
        assert callable(model.fit_null)
        assert callable(model.score_chunk)
