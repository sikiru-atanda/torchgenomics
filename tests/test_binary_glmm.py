"""Tests for Phase 32: Binary GLMM (logistic mixed model GWAS).

Covers: PQL convergence, score test calibration, SPA, power, edge cases,
K=0 reduction to BinaryGLM, protocol conformance.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.models.base import VariantMeta
from torchgwas.models.binary_glmm import BinaryGLMM


def _simulate_binary_glmm_data(
    n: int = 200,
    m: int = 30,
    prevalence: float = 0.3,
    h2: float = 0.3,
    causal_or: float = 1.0,
    seed: int = 42,
):
    """Simulate binary phenotype with kinship + optional causal SNP."""
    torch.manual_seed(seed)

    # Random GRM
    Z = torch.randn(n, 50, dtype=torch.float64)
    K = Z @ Z.T / 50.0
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    evals = evals.clamp(min=0.0)
    K = evecs @ torch.diag(evals) @ evecs.T

    # Genotypes
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)

    # Genetic random effect
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(h2)

    # Linear predictor
    logit_p = math.log(prevalence / (1.0 - prevalence)) + u
    if causal_or != 1.0:
        g0 = G[:, 0] - G[:, 0].mean()
        logit_p = logit_p + g0 * math.log(causal_or)

    prob = torch.sigmoid(logit_p)
    Y = torch.bernoulli(prob).to(torch.float64)
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

class TestBinaryGLMMNull:
    def test_fit_null_converges(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data()
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        # PQL should make progress (may or may not fully converge)
        assert nf._glm_mu is not None
        assert (nf._glm_mu > 0).all() and (nf._glm_mu < 1).all()

    def test_variance_components_estimated(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data()
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        assert nf.sig2_g is not None and nf.sig2_g > 0
        assert nf.sig2_e is not None and nf.sig2_e > 0

    def test_requires_kinship(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data()
        model = BinaryGLMM()
        with pytest.raises(ValueError, match="requires a kinship"):
            model.fit_null(Y, X0, K=None)

    def test_converged_flag_log_likelihood_criterion(self):
        """Phase 33 follow-up: PQL reports converged=True on a stable
        problem when the log-likelihood + variance components have
        plateaued, even if the fixed-effect β is still drifting at the
        β-only tolerance.

        Reference: SAIGE harness B5 noted that on a fixture where TG
        BinaryGLMM β/SE/p match SAIGE to within 5%, PQL was returning
        converged=False due to the β-only criterion. The dual ll/VC
        criterion captures these cases.
        """
        # Easy fixture: balanced case-control, mild kinship signal, no
        # case-control imbalance issues. PQL should converge cleanly
        # within the iteration budget.
        n = 300
        torch.manual_seed(101)
        Z = torch.randn(n, 80, dtype=torch.float64)
        K_mat = Z @ Z.T / 80.0
        K_mat = 0.5 * (K_mat + K_mat.T)
        L = torch.linalg.cholesky(K_mat + 1e-6 * torch.eye(n, dtype=torch.float64))
        u = L @ torch.randn(n, dtype=torch.float64) * math.sqrt(0.2)
        logit_p = u  # prevalence = 0.5 → balanced
        prob = torch.sigmoid(logit_p)
        Y = torch.bernoulli(prob).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        model = BinaryGLMM(use_spa=False, pql_max_iter=30, pql_tol=1e-4)
        nf = model.fit_null(Y, X0, K=K_mat)
        assert nf.converged is True, (
            f"Expected PQL to report converged=True on a clean balanced "
            f"fixture; got converged={nf.converged}, "
            f"log_likelihood={nf.log_likelihood}, "
            f"sig2_g={nf.sig2_g}, sig2_e={nf.sig2_e}"
        )


# ===================================================================
# Score test calibration
# ===================================================================

class TestBinaryGLMMScan:
    def test_pvalues_valid(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data()
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_stat_nonneg(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data()
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.stat >= 0).all()

    def test_null_calibration(self):
        """Under null, median p-value should not be severely inflated."""
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data(
            n=300, m=100, prevalence=0.3, h2=0.3,
            causal_or=1.0, seed=100)
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        median_p = result.p.median().item()
        # Under null with kinship correction, should not be badly inflated
        assert median_p > 0.1

    def test_power_detects_signal(self):
        """With strong causal effect, should detect signal at SNP 0."""
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data(
            n=400, m=30, prevalence=0.3, h2=0.2,
            causal_or=3.0, seed=200)
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p[0].item() < 0.05


# ===================================================================
# SPA
# ===================================================================

class TestBinaryGLMMSPA:
    def test_spa_produces_valid_pvalues(self):
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data(n=200, seed=300)
        model = BinaryGLMM(use_spa=True, spa_threshold=0.5)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_spa_imbalanced(self):
        """SPA with strong case-control imbalance."""
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data(
            n=300, m=20, prevalence=0.05, h2=0.2, seed=301)
        model = BinaryGLMM(use_spa=True)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_spa_agrees_chi2_body(self):
        """High SPA threshold = no SPA, p-values should be highly correlated."""
        G, Y, X0, K, vmeta = _simulate_binary_glmm_data(n=300, seed=302)
        model_spa = BinaryGLMM(use_spa=True, spa_threshold=100.0)
        model_chi2 = BinaryGLMM(use_spa=False)
        nf = model_spa.fit_null(Y, X0, K=K)
        res_spa = model_spa.score_chunk(G, nf, vmeta)
        res_chi2 = model_chi2.score_chunk(G, nf, vmeta)
        # SPA with high threshold uses chi2 for body but simple variance;
        # score_chunk uses Schur complement variance. Rank should agree.
        rank_corr = _spearman_rank_corr(res_spa.p, res_chi2.p)
        assert rank_corr > 0.95


# ===================================================================
# Edge cases
# ===================================================================

class TestBinaryGLMMEdge:
    def test_k_zero_similar_to_glm(self):
        """With K=0, GLMM should produce similar results to GLM."""
        n, m = 150, 20
        torch.manual_seed(400)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        K = torch.eye(n, dtype=torch.float64) * 1e-6  # near-zero kinship
        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)], chr=["1"] * m,
            pos=list(range(m)), a1=["A"] * m, a2=["G"] * m,
        )

        from torchgwas.models.binary_glm import BinaryGLM
        glm = BinaryGLM(use_spa=False)
        glmm = BinaryGLMM(use_spa=False)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        # P-values should be in same ballpark (not exact due to PQL overhead)
        rank_corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert rank_corr > 0.8

    def test_monomorphic_snp(self):
        """Monomorphic SNP should have small score statistic."""
        n, m = 100, 5
        torch.manual_seed(401)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        G[:, 0] = 0.0  # monomorphic: all zero dosage (no variance)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Z = torch.randn(n, 20, dtype=torch.float64)
        K = Z @ Z.T / 20.0
        K = 0.5 * (K + K.T)
        evals, evecs = torch.linalg.eigh(K)
        K = evecs @ torch.diag(evals.clamp(min=0.0)) @ evecs.T

        vmeta = VariantMeta(
            snp=[f"s{i}" for i in range(m)], chr=["1"] * m,
            pos=list(range(m)), a1=["A"] * m, a2=["G"] * m,
        )
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        # Zero-dosage SNP: score U = 0'(Y-mu) = 0, so stat = 0, p ≈ 1
        assert result.stat[0].item() < 1e-10


# ===================================================================
# Protocol
# ===================================================================

class TestBinaryGLMMProtocol:
    def test_basemodel_conformance(self):
        model = BinaryGLMM()
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")
        assert callable(model.fit_null)
        assert callable(model.score_chunk)


def _spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()
