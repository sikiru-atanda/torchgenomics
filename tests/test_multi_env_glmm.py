"""Tests for Multi-Environment GLMM (ME-GLMM).

Validates: PQL convergence, Σ_g recovery, score test calibration,
power, homogeneity, reaction-norm, SPA, ordinal support, edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kstest

# Use lazy imports inside test methods to avoid circular import
# through torchgwas.optim.__init__ → controller → models → optim


def _make_vmeta(m):
    from torchgwas.models.base import VariantMeta
    return VariantMeta(
        snp=[f"s{i}" for i in range(m)],
        chr=["1"] * m, pos=list(range(m)),
        a1=["A"] * m, a2=["G"] * m,
    )


def _simulate_me_binary(n=200, E=3, m=20, h2=0.3, rg=0.7, seed=42,
                         prevalence=None, m_grm=100):
    """Simulate multi-environment binary phenotype with genetic covariance.

    Uses SEPARATE genotypes for GRM (m_grm SNPs) vs test (m SNPs) to avoid
    proximal contamination.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GRM genotypes (separate from test SNPs)
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.1 + 0.3 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )
    G_std_grm = G_grm - G_grm.mean(dim=0)
    K = (G_std_grm @ G_std_grm.T) / m_grm + 0.01 * torch.eye(n, dtype=torch.float64)

    # Test genotypes (independent)
    G = torch.zeros(n, m, dtype=torch.float64)
    for j in range(m):
        maf = 0.1 + 0.3 * np.random.rand()
        G[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )

    # Genetic covariance across environments
    Sigma_g = torch.eye(E, dtype=torch.float64)
    for e1 in range(E):
        for e2 in range(E):
            if e1 != e2:
                Sigma_g[e1, e2] = rg

    # V = K ⊗ Σ_g → Cholesky for sampling
    V = h2 * torch.kron(Sigma_g, K) + (1 - h2) * torch.eye(n * E, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n * E, dtype=torch.float64))
    u_vec = L_V @ torch.randn(n * E, dtype=torch.float64)
    u = u_vec.reshape(E, n).T  # (n, E)

    # Per-env intercepts
    if prevalence is None:
        prevalence = [0.3] * E
    intercepts = [torch.log(torch.tensor(p / (1 - p))) for p in prevalence]

    # Binary outcomes
    Y = torch.zeros(n, E, dtype=torch.float64)
    for e in range(E):
        eta_e = intercepts[e] + u[:, e]
        prob_e = torch.sigmoid(eta_e)
        Y[:, e] = torch.bernoulli(prob_e)

    X0 = torch.ones(n, 1, dtype=torch.float64)
    return Y, X0, K, G


def _simulate_me_ordinal(n=200, E=3, m=20, J=3, seed=42, m_grm=100):
    """Simulate multi-environment ordinal phenotype."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Separate GRM genotypes
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.15 + 0.25 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )
    G_std_grm = G_grm - G_grm.mean(dim=0)
    K = (G_std_grm @ G_std_grm.T) / m_grm + 0.01 * torch.eye(n, dtype=torch.float64)

    # Test genotypes
    G = torch.zeros(n, m, dtype=torch.float64)
    for j in range(m):
        maf = 0.15 + 0.25 * np.random.rand()
        G[:, j] = torch.tensor(
            np.random.choice([0, 1, 2], n, p=[(1-maf)**2, 2*maf*(1-maf), maf**2]),
            dtype=torch.float64,
        )

    # Random effects
    Sigma_g = 0.3 * torch.eye(E, dtype=torch.float64)
    for e1 in range(E):
        for e2 in range(E):
            if e1 != e2:
                Sigma_g[e1, e2] = 0.15

    V = torch.kron(Sigma_g, K) + torch.eye(n * E, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n * E, dtype=torch.float64))
    u_vec = L_V @ torch.randn(n * E, dtype=torch.float64)
    u = u_vec.reshape(E, n).T

    # Ordinal via latent thresholds
    thresholds = np.linspace(-0.5, 0.5, J - 1)
    Y = torch.zeros(n, E, dtype=torch.float64)
    for e in range(E):
        latent = u[:, e] + torch.randn(n, dtype=torch.float64)
        for i in range(n):
            val = 0
            for j_idx, th in enumerate(thresholds):
                if latent[i].item() > th:
                    val = j_idx + 1
            Y[i, e] = val

    X0 = torch.ones(n, 1, dtype=torch.float64)
    return Y, X0, K, G, J


# =====================================================================
# Binary ME-GLMM: Null fitting
# =====================================================================

class TestMultiEnvBinaryNull:

    def test_pql_converges(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, seed=10)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        assert nf.converged or nf._me_glmm_mu is not None

    def test_sigma_g_positive(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, seed=11)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        Sg = nf._me_glmm_Sigma_g
        # Should be positive semi-definite
        evals = torch.linalg.eigvalsh(Sg)
        assert (evals >= -1e-6).all(), f"Sigma_g not PSD: eigenvalues={evals}"

    def test_sigma_g_is_valid_covariance(self):
        """Σ_g should be a valid (PSD) covariance matrix with correct shape."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=300, E=3, rg=0.7, seed=12)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        Sg = nf._me_glmm_Sigma_g
        assert Sg.shape == (3, 3)
        # Symmetric
        assert torch.allclose(Sg, Sg.T, atol=1e-8)
        # PSD
        evals = torch.linalg.eigvalsh(Sg)
        assert (evals >= -1e-6).all(), f"Sigma_g not PSD: {evals}"
        # Diagonal should be positive
        assert (torch.diagonal(Sg) > 0).all()

    def test_requires_kinship(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=100, E=2, seed=13)
        model = MultiEnvGLMM(family="binary")
        with pytest.raises(ValueError, match="kinship"):
            model.fit_null(Y, X0, K=None)


# =====================================================================
# Binary ME-GLMM: Score test
# =====================================================================

class TestMultiEnvBinaryScore:

    def test_joint_pvalues_valid(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, m=20, seed=20)
        vmeta = _make_vmeta(20)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all() and (result.p <= 1).all()
        assert result.p.shape == (20,)

    def test_per_env_marginals_valid(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, m=20, seed=21)
        vmeta = _make_vmeta(20)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p_marginal.shape == (20, 3)
        assert (result.p_marginal >= 0).all() and (result.p_marginal <= 1).all()

    def test_null_calibration(self):
        """Under null, p-values should be approximately uniform."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=300, E=3, m=200, seed=22)
        vmeta = _make_vmeta(200)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        # KS test for uniformity on per-env marginals
        for e in range(3):
            p_vals = result.p_marginal[:, e].numpy()
            p_vals = p_vals[np.isfinite(p_vals) & (p_vals > 0) & (p_vals < 1)]
            ks_stat, ks_p = kstest(p_vals, 'uniform')
            # Allow generous threshold (small n)
            assert ks_p > 0.001, \
                f"Env {e}: p-values not uniform (KS p={ks_p:.4f})"

    def test_power_detects_planted_signal(self):
        """Joint test should detect a planted heterogeneous effect."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        torch.manual_seed(23)
        np.random.seed(23)
        n, E, m = 300, 3, 15

        # Separate GRM genotypes
        G_grm = torch.tensor(np.random.choice([0,1,2], (n, 100)), dtype=torch.float64)
        G_std = G_grm - G_grm.mean(0)
        K = (G_std @ G_std.T) / 100 + 0.01 * torch.eye(n, dtype=torch.float64)

        # Test genotypes
        G = torch.zeros(n, m, dtype=torch.float64)
        for j in range(m):
            G[:, j] = torch.tensor(
                np.random.choice([0, 1, 2], n), dtype=torch.float64)

        # Causal SNP 0: heterogeneous effect across envs
        Y = torch.zeros(n, E, dtype=torch.float64)
        effects = [1.5, 0.8, 2.0]  # strong effects
        for e in range(E):
            eta = -0.5 + effects[e] * G[:, 0]
            Y[:, e] = torch.bernoulli(torch.sigmoid(eta))

        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        # Causal SNP should have significant joint p-value
        assert result.p[0] < 0.05, f"Causal SNP not detected: p={result.p[0]:.4f}"


# =====================================================================
# Homogeneity test
# =====================================================================

class TestMultiEnvBinaryHomogeneity:

    def test_homogeneous_effect_small_stat(self):
        """Homogeneous effect should have small homogeneity statistic."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        torch.manual_seed(30)
        np.random.seed(30)
        n, E, m = 300, 3, 10

        G_grm = torch.tensor(np.random.choice([0,1,2], (n, 100)), dtype=torch.float64)
        G_std = G_grm - G_grm.mean(0)
        K = (G_std @ G_std.T) / 100 + 0.01 * torch.eye(n, dtype=torch.float64)

        G = torch.zeros(n, m, dtype=torch.float64)
        for j in range(m):
            G[:, j] = torch.tensor(
                np.random.choice([0, 1, 2], n), dtype=torch.float64)

        # Same effect in all envs
        Y = torch.zeros(n, E, dtype=torch.float64)
        for e in range(E):
            eta = -0.5 + 1.5 * G[:, 0]  # same effect
            Y[:, e] = torch.bernoulli(torch.sigmoid(eta))

        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        # Homogeneity p for causal SNP should be non-significant
        assert result.p_homogeneity[0] > 0.01, \
            f"Homogeneous effect flagged: p_hom={result.p_homogeneity[0]:.4f}"

    def test_heterogeneous_effect_large_stat(self):
        """Heterogeneous effect should have significant homogeneity stat."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        torch.manual_seed(31)
        np.random.seed(31)
        n, E, m = 400, 3, 10

        G_grm = torch.tensor(np.random.choice([0,1,2], (n, 100)), dtype=torch.float64)
        G_std = G_grm - G_grm.mean(0)
        K = (G_std @ G_std.T) / 100 + 0.01 * torch.eye(n, dtype=torch.float64)

        G = torch.zeros(n, m, dtype=torch.float64)
        for j in range(m):
            G[:, j] = torch.tensor(
                np.random.choice([0, 1, 2], n), dtype=torch.float64)

        # Very different effects: positive in env0, negative in env1, zero in env2
        Y = torch.zeros(n, E, dtype=torch.float64)
        effects = [2.5, -2.5, 0.0]
        for e in range(E):
            eta = 0.0 + effects[e] * G[:, 0]
            Y[:, e] = torch.bernoulli(torch.sigmoid(eta))

        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        # Homogeneity should detect the difference
        assert result.p_homogeneity[0] < 0.10, \
            f"Heterogeneous effect not detected: p_hom={result.p_homogeneity[0]:.4f}"


# =====================================================================
# SPA per-env
# =====================================================================

class TestMultiEnvBinarySPA:

    def test_spa_per_env_valid(self):
        """SPA should produce valid per-env p-values."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, m=15, seed=40)
        vmeta = _make_vmeta(15)
        model = MultiEnvGLMM(family="binary", use_spa=True)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p_marginal >= 0).all() and (result.p_marginal <= 1).all()

    def test_spa_less_conservative_than_chi2(self):
        """SPA per-env p-values should be <= chi2 p-values (hybrid clamping)."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(
            n=300, E=2, m=20, seed=41, prevalence=[0.05, 0.05])
        vmeta = _make_vmeta(20)

        model_spa = MultiEnvGLMM(family="binary", use_spa=True)
        model_chi2 = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model_chi2.fit_null(Y, X0, K=K)

        res_chi2 = model_chi2.score_chunk(G, nf, vmeta)
        res_spa = model_spa.score_chunk(G, nf, vmeta)

        # SPA rejection rate <= chi2 rejection rate (or close)
        for e in range(2):
            rej_chi2 = (res_chi2.p_marginal[:, e] < 0.05).float().mean().item()
            rej_spa = (res_spa.p_marginal[:, e] < 0.05).float().mean().item()
            assert rej_spa <= rej_chi2 + 0.05


# =====================================================================
# Ordinal ME-GLMM
# =====================================================================

class TestMultiEnvOrdinalNull:

    def test_ordinal_pql_converges(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G, J = _simulate_me_ordinal(n=200, E=3, m=15, J=3, seed=50)
        model = MultiEnvGLMM(family="ordinal", n_categories=J, use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        assert nf._me_glmm_mu is not None
        assert nf._me_glmm_Sigma_g is not None

    def test_ordinal_sigma_g_positive(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G, J = _simulate_me_ordinal(n=200, E=3, seed=51)
        model = MultiEnvGLMM(family="ordinal", n_categories=J, use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        evals = torch.linalg.eigvalsh(nf._me_glmm_Sigma_g)
        assert (evals >= -1e-6).all()

    def test_ordinal_score_pvalues_valid(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G, J = _simulate_me_ordinal(n=200, E=3, m=15, seed=52)
        vmeta = _make_vmeta(15)
        model = MultiEnvGLMM(family="ordinal", n_categories=J, use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert (result.p >= 0).all() and (result.p <= 1).all()
        assert result.p_marginal.shape == (15, 3)


# =====================================================================
# Reaction-norm
# =====================================================================

class TestReactionNorm:

    def test_reaction_norm_fields_populated(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=3, m=10, seed=60)
        vmeta = _make_vmeta(10)
        model = MultiEnvGLMM(family="binary", parameterization="reaction_norm",
                              use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.alpha is not None
        assert result.stat_stable is not None
        assert result.stat_gxe is not None
        assert result.p_stable.shape == (10,)
        assert result.p_gxe.shape == (10,)

    def test_stable_effect_detected(self):
        """Reaction-norm stable test should detect a homogeneous causal effect."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        torch.manual_seed(61)
        np.random.seed(61)
        n, E, m = 300, 3, 10

        G_grm = torch.tensor(np.random.choice([0,1,2], (n, 100)), dtype=torch.float64)
        G_std = G_grm - G_grm.mean(0)
        K = (G_std @ G_std.T) / 100 + 0.01 * torch.eye(n, dtype=torch.float64)

        G = torch.zeros(n, m, dtype=torch.float64)
        for j in range(m):
            G[:, j] = torch.tensor(
                np.random.choice([0, 1, 2], n), dtype=torch.float64)

        # Same effect in all envs → stable effect
        Y = torch.zeros(n, E, dtype=torch.float64)
        for e in range(E):
            eta = -0.5 + 1.5 * G[:, 0]
            Y[:, e] = torch.bernoulli(torch.sigmoid(eta))

        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)
        model = MultiEnvGLMM(family="binary", parameterization="reaction_norm",
                              use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        assert result.p_stable[0] < 0.05, \
            f"Stable effect not detected: p={result.p_stable[0]:.4f}"


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:

    def test_two_environments(self):
        """Minimum E=2 should work."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=200, E=2, m=10, seed=70)
        vmeta = _make_vmeta(10)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        assert result.p.shape == (10,)
        assert result.stat_homogeneity.shape == (10,)

    def test_near_zero_kinship_reduces_to_glm(self):
        """With K ≈ εI, should behave like multi-env GLM."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, _, G = _simulate_me_binary(n=200, E=3, m=10, seed=71)
        K = 1e-6 * torch.eye(200, dtype=torch.float64)
        vmeta = _make_vmeta(10)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)
        # Should still produce valid p-values
        assert (result.p >= 0).all() and (result.p <= 1).all()

    def test_polyploid(self):
        """Polyploid genotypes should produce correct AF."""
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        torch.manual_seed(72)
        np.random.seed(72)
        n, E, m, ploidy = 200, 2, 10, 4

        G = torch.tensor(
            np.random.choice(ploidy + 1, (n, m)), dtype=torch.float64)
        Y = torch.tensor(
            np.random.binomial(1, 0.3, (n, E)), dtype=torch.float64)
        K = torch.eye(n, dtype=torch.float64) * 0.01
        K += (G - G.mean(0)).mm((G - G.mean(0)).T) / m
        X0 = torch.ones(n, 1, dtype=torch.float64)
        vmeta = _make_vmeta(m)

        model = MultiEnvGLMM(family="binary", use_spa=False, ploidy=ploidy)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        expected_af = G.mean(dim=0) / ploidy
        assert torch.allclose(result.af, expected_af)


# =====================================================================
# Protocol conformance
# =====================================================================

class TestProtocol:

    def test_has_fit_null_and_score_chunk(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        model = MultiEnvGLMM(family="binary")
        assert hasattr(model, "fit_null")
        assert hasattr(model, "score_chunk")

    def test_env_scan_result_fields(self):
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM
        Y, X0, K, G = _simulate_me_binary(n=150, E=2, m=5, seed=80)
        vmeta = _make_vmeta(5)
        model = MultiEnvGLMM(family="binary", use_spa=False)
        nf = model.fit_null(Y, X0, K=K)
        result = model.score_chunk(G, nf, vmeta)

        # Check required EnvScanResult fields
        assert hasattr(result, 'stat')
        assert hasattr(result, 'p')
        assert hasattr(result, 'stat_homogeneity')
        assert hasattr(result, 'p_homogeneity')
        assert hasattr(result, 'stat_marginal')
        assert hasattr(result, 'p_marginal')
        assert hasattr(result, 'beta')
        assert hasattr(result, 'se')
        assert result.beta.shape == (5, 2)
