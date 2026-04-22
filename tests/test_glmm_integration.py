"""Cross-model integration tests for Phases 31-33.

Covers:
- GLMM(K=0) ≈ GLM reduction
- OrdinalGLM(J=2) ≈ BinaryGLM reduction
- SPA ≈ chi2 in the body
- All models on same data: rank correlation of p-values
"""

from __future__ import annotations

import torch

from torchgwas.models.base import VariantMeta


def _make_vmeta(m):
    return VariantMeta(
        snp=[f"s{i}" for i in range(m)], chr=["1"] * m,
        pos=list(range(m)), a1=["A"] * m, a2=["G"] * m,
    )


def _spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()


class TestReductions:
    def test_ordinal_j2_matches_binary(self):
        """OrdinalGLM with J=2 should produce similar results to BinaryGLM."""
        from torchgwas.models.binary_glm import BinaryGLM
        from torchgwas.models.ordinal_glm import OrdinalGLM

        n, m = 200, 30
        torch.manual_seed(500)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)

        binary = BinaryGLM(use_spa=False)
        ordinal = OrdinalGLM(n_categories=2)

        nf_bin = binary.fit_null(Y, X0)
        nf_ord = ordinal.fit_null(Y, X0)

        res_bin = binary.score_chunk(G, nf_bin, vmeta)
        res_ord = ordinal.score_chunk(G, nf_ord, vmeta)

        # Rank correlation should be very high
        corr = _spearman_rank_corr(res_bin.p, res_ord.p)
        assert corr > 0.9

    def test_spa_approx_chi2_body(self):
        """SPA and chi2 p-values should be highly rank-correlated."""
        from torchgwas.models.binary_glm import BinaryGLM

        n, m = 300, 50
        torch.manual_seed(501)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.5)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)

        spa = BinaryGLM(use_spa=True, spa_threshold=100.0)
        chi2 = BinaryGLM(use_spa=False)

        nf = spa.fit_null(Y, X0)
        res_spa = spa.score_chunk(G, nf, vmeta)
        res_chi2 = chi2.score_chunk(G, nf, vmeta)

        # SPA with high threshold vs chi2 with Schur complement:
        # different variance formulas produce different p-values but
        # rankings should be highly correlated
        corr = _spearman_rank_corr(res_spa.p, res_chi2.p)
        assert corr > 0.95

    def test_binary_glmm_k_zero_matches_glm(self):
        """BinaryGLMM with near-zero K should behave like BinaryGLM."""
        from torchgwas.models.binary_glm import BinaryGLM
        from torchgwas.models.binary_glmm import BinaryGLMM

        n, m = 150, 20
        torch.manual_seed(502)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        K = torch.eye(n, dtype=torch.float64) * 1e-6
        vmeta = _make_vmeta(m)

        glm = BinaryGLM(use_spa=False)
        glmm = BinaryGLMM(use_spa=False)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert corr > 0.8

    def test_multinomial_glmm_k_zero_matches_glm(self):
        """MultinomialGLMM with near-zero K should behave like MultinomialGLM."""
        from torchgwas.models.multinomial_glm import MultinomialGLM
        from torchgwas.models.multinomial_glmm import MultinomialGLMM

        n, m, J = 200, 15, 3
        torch.manual_seed(503)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.randint(0, J, (n,), dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        K = torch.eye(n, dtype=torch.float64) * 1e-6
        vmeta = _make_vmeta(m)

        glm = MultinomialGLM(n_classes=J)
        glmm = MultinomialGLMM(n_classes=J)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert corr > 0.8


class TestCrossModel:
    def test_all_binary_models_on_same_data(self):
        """BinaryGLM, BinaryGLMM on same data: p-values should correlate."""
        from torchgwas.models.binary_glm import BinaryGLM
        from torchgwas.models.binary_glmm import BinaryGLMM

        n, m = 200, 20
        torch.manual_seed(600)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.bernoulli(torch.full((n,), 0.3)).to(torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # Make a real kinship
        Z = torch.randn(n, 30, dtype=torch.float64)
        K = Z @ Z.T / 30.0
        K = 0.5 * (K + K.T)
        evals, evecs = torch.linalg.eigh(K)
        K = evecs @ torch.diag(evals.clamp(min=0.0)) @ evecs.T

        vmeta = _make_vmeta(m)

        glm = BinaryGLM(use_spa=False)
        glmm = BinaryGLMM(use_spa=False)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        # Both should produce valid p-values
        assert (res_glm.p >= 0).all() and (res_glm.p <= 1).all()
        assert (res_glmm.p >= 0).all() and (res_glmm.p <= 1).all()

        # Should be positively correlated
        corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert corr > 0.3

    def test_all_ordinal_models_on_same_data(self):
        """OrdinalGLM, OrdinalGLMM on same data: p-values should correlate."""
        from torchgwas.models.ordinal_glm import OrdinalGLM
        from torchgwas.models.ordinal_glmm import OrdinalGLMM

        n, m, J = 200, 20, 3
        torch.manual_seed(601)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        # Ordinal phenotype
        Y = torch.randint(0, J, (n,), dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        Z = torch.randn(n, 30, dtype=torch.float64)
        K = Z @ Z.T / 30.0
        K = 0.5 * (K + K.T)
        evals, evecs = torch.linalg.eigh(K)
        K = evecs @ torch.diag(evals.clamp(min=0.0)) @ evecs.T

        vmeta = _make_vmeta(m)

        glm = OrdinalGLM(n_categories=J)
        glmm = OrdinalGLMM(n_categories=J)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        assert (res_glm.p >= 0).all() and (res_glm.p <= 1).all()
        assert (res_glmm.p >= 0).all() and (res_glmm.p <= 1).all()

        corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert corr > 0.3

    def test_all_multinomial_models_on_same_data(self):
        """MultinomialGLM, MultinomialGLMM on same data: p-values should correlate."""
        from torchgwas.models.multinomial_glm import MultinomialGLM
        from torchgwas.models.multinomial_glmm import MultinomialGLMM

        n, m, J = 200, 15, 3
        torch.manual_seed(602)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = torch.randint(0, J, (n,), dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        Z = torch.randn(n, 30, dtype=torch.float64)
        K = Z @ Z.T / 30.0
        K = 0.5 * (K + K.T)
        evals, evecs = torch.linalg.eigh(K)
        K = evecs @ torch.diag(evals.clamp(min=0.0)) @ evecs.T

        vmeta = _make_vmeta(m)

        glm = MultinomialGLM(n_classes=J)
        glmm = MultinomialGLMM(n_classes=J)

        nf_glm = glm.fit_null(Y, X0)
        nf_glmm = glmm.fit_null(Y, X0, K=K)

        res_glm = glm.score_chunk(G, nf_glm, vmeta)
        res_glmm = glmm.score_chunk(G, nf_glmm, vmeta)

        assert (res_glm.p >= 0).all() and (res_glm.p <= 1).all()
        assert (res_glmm.p >= 0).all() and (res_glmm.p <= 1).all()

        corr = _spearman_rank_corr(res_glm.p, res_glmm.p)
        assert corr > 0.3
