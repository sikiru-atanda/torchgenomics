"""Tests for Phase 20: Multi-trait Threshold-Linear GWAS (Bermann et al. 2026).

Covers:
- Truncated MVN moments (univariate, bivariate, trivariate)
- SQUAREM acceleration
- ThresholdLinearModel EM and NR solvers
- ssGWAS score test
- Polyploid compatibility
- BaseModel protocol conformance
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgwas.linalg.truncated_mvn import (
    bivariate_truncated_moments,
    mvn_truncated_moments,
    truncated_normal_moments,
)
from torchgwas.optim.squarem import squarem


# ===================================================================
# Truncated MVN moments
# ===================================================================

class TestTruncatedNormalMomentsUnivariate:
    """Univariate truncated normal moments."""

    def test_half_normal(self):
        """TN(0, 1, 0, inf) should give half-normal moments."""
        mu = torch.tensor([0.0], dtype=torch.float64)
        sigma = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([float("inf")], dtype=torch.float64)

        m, v = truncated_normal_moments(mu, sigma, a, b)

        expected_m = math.sqrt(2 / math.pi)
        expected_v = 1 - 2 / math.pi
        assert abs(m.item() - expected_m) < 1e-5
        assert abs(v.item() - expected_v) < 1e-5

    def test_symmetric_bounds(self):
        """TN(0, 1, -a, a) should have zero mean."""
        mu = torch.tensor([0.0], dtype=torch.float64)
        sigma = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([-2.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        m, v = truncated_normal_moments(mu, sigma, a, b)

        assert abs(m.item()) < 1e-10

    def test_narrow_bounds(self):
        """TN(0, 1, 0.99, 1.01) mean should be close to 1.0."""
        mu = torch.tensor([0.0], dtype=torch.float64)
        sigma = torch.tensor([1.0], dtype=torch.float64)
        a = torch.tensor([0.99], dtype=torch.float64)
        b = torch.tensor([1.01], dtype=torch.float64)

        m, v = truncated_normal_moments(mu, sigma, a, b)

        assert abs(m.item() - 1.0) < 0.02
        assert v.item() < 0.01  # very small variance

    def test_batched(self):
        """Batched computation matches individual."""
        n = 50
        mu = torch.randn(n, dtype=torch.float64)
        sigma = torch.ones(n, dtype=torch.float64) * 1.5
        a = mu - 1.0
        b = mu + 1.0

        m_batch, v_batch = truncated_normal_moments(mu, sigma, a, b)

        for i in range(5):
            m_i, v_i = truncated_normal_moments(
                mu[i:i+1], sigma[i:i+1], a[i:i+1], b[i:i+1]
            )
            assert abs(m_batch[i].item() - m_i.item()) < 1e-10
            assert abs(v_batch[i].item() - v_i.item()) < 1e-10

    def test_variance_positive(self):
        """Variance should always be positive."""
        mu = torch.randn(100, dtype=torch.float64)
        sigma = torch.ones(100, dtype=torch.float64)
        a = mu - 3.0
        b = mu + 0.5

        _, v = truncated_normal_moments(mu, sigma, a, b)
        assert (v >= 0).all()

    def test_untruncated_recovery(self):
        """Wide bounds should recover un-truncated moments."""
        mu = torch.tensor([2.0], dtype=torch.float64)
        sigma = torch.tensor([3.0], dtype=torch.float64)
        a = torch.tensor([-100.0], dtype=torch.float64)
        b = torch.tensor([100.0], dtype=torch.float64)

        m, v = truncated_normal_moments(mu, sigma, a, b)

        assert abs(m.item() - 2.0) < 0.01
        assert abs(v.item() - 9.0) < 0.1


class TestBivariateTruncatedMoments:
    """Bivariate truncated normal moments."""

    def test_symmetric_zero_mean(self):
        """Symmetric bounds on zero-mean bivariate should give zero mean."""
        n = 10
        mu = torch.zeros(n, 2, dtype=torch.float64)
        Sigma = torch.eye(2, dtype=torch.float64)
        a = torch.full((n, 2), -2.0, dtype=torch.float64)
        b = torch.full((n, 2), 2.0, dtype=torch.float64)

        m, v = bivariate_truncated_moments(mu, Sigma, a, b)

        assert m.shape == (n, 2)
        assert v.shape == (n, 2, 2)
        assert (m.abs() < 0.01).all()

    def test_positive_definite_variance(self):
        """Variance matrix should be PD."""
        n = 5
        mu = torch.randn(n, 2, dtype=torch.float64)
        Sigma = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        a = mu - 1.5
        b = mu + 1.5

        _, v = bivariate_truncated_moments(mu, Sigma, a, b)

        for i in range(n):
            evals = torch.linalg.eigvalsh(v[i])
            assert (evals > -1e-6).all(), f"Non-PD variance at i={i}: {evals}"

    def test_independent_marginals(self):
        """With zero correlation, bivariate marginals match univariate."""
        n = 20
        mu = torch.randn(n, 2, dtype=torch.float64)
        Sigma = torch.eye(2, dtype=torch.float64)
        a = mu - 1.0
        b = mu + 2.0

        m_biv, v_biv = bivariate_truncated_moments(mu, Sigma, a, b)

        for j in range(2):
            m_uni, v_uni = truncated_normal_moments(
                mu[:, j], torch.ones(n, dtype=torch.float64),
                a[:, j], b[:, j],
            )
            assert torch.allclose(m_biv[:, j], m_uni, atol=0.05)


class TestMultivariateTruncatedMoments:
    """General multivariate (QMC-based) moments."""

    def test_delegates_to_univariate(self):
        """c=1 should delegate to univariate."""
        mu = torch.zeros(5, 1, dtype=torch.float64)
        Sigma = torch.eye(1, dtype=torch.float64)
        a = torch.full((5, 1), -1.0, dtype=torch.float64)
        b = torch.full((5, 1), 1.0, dtype=torch.float64)

        m, v = mvn_truncated_moments(mu, Sigma, a, b)
        assert m.shape == (5, 1)
        assert v.shape == (5, 1, 1)

    def test_delegates_to_bivariate(self):
        """c=2 should delegate to bivariate."""
        mu = torch.zeros(3, 2, dtype=torch.float64)
        Sigma = torch.eye(2, dtype=torch.float64)
        a = torch.full((3, 2), -2.0, dtype=torch.float64)
        b = torch.full((3, 2), 2.0, dtype=torch.float64)

        m, v = mvn_truncated_moments(mu, Sigma, a, b)
        assert m.shape == (3, 2)
        assert v.shape == (3, 2, 2)

    def test_trivariate_symmetric(self):
        """Symmetric trivariate truncation should give near-zero mean."""
        mu = torch.zeros(5, 3, dtype=torch.float64)
        Sigma = torch.eye(3, dtype=torch.float64)
        a = torch.full((5, 3), -1.5, dtype=torch.float64)
        b = torch.full((5, 3), 1.5, dtype=torch.float64)

        m, v = mvn_truncated_moments(mu, Sigma, a, b, n_qmc=10000, seed=42)

        assert m.shape == (5, 3)
        assert (m.abs() < 0.05).all()
        # Diagonal variances should be near the univariate TN(0,1,-1.5,1.5) var
        for i in range(3):
            assert abs(v[0, i, i].item() - 0.5735) < 0.1

    def test_trivariate_correlated(self):
        """Correlated trivariate gives reasonable variances."""
        mu = torch.zeros(3, 3, dtype=torch.float64)
        Sigma = torch.eye(3, dtype=torch.float64)
        Sigma[0, 1] = Sigma[1, 0] = 0.3
        Sigma[0, 2] = Sigma[2, 0] = 0.2
        a = torch.full((3, 3), -2.0, dtype=torch.float64)
        b = torch.full((3, 3), 2.0, dtype=torch.float64)

        m, v = mvn_truncated_moments(mu, Sigma, a, b, n_qmc=10000, seed=123)

        assert (m.abs() < 0.05).all()
        # Variance should be less than 1.0 (truncation reduces it)
        for i in range(3):
            assert 0.1 < v[0, i, i].item() < 1.0


# ===================================================================
# SQUAREM
# ===================================================================

class TestSQUAREM:
    """SQUAREM fixed-point accelerator."""

    def test_sqrt_convergence(self):
        """Find sqrt(2) via Newton iteration."""
        def fp(x):
            return 0.5 * (x + 2.0 / x)

        x0 = torch.tensor([5.0], dtype=torch.float64)
        x, n_iter, conv = squarem(fp, x0, tol=1e-12)

        assert conv
        assert abs(x.item() - math.sqrt(2)) < 1e-10

    def test_identity_fixed_point(self):
        """Identity mapping should converge immediately."""
        def fp(x):
            return x.clone()

        x0 = torch.tensor([3.14], dtype=torch.float64)
        x, n_iter, conv = squarem(fp, x0, tol=1e-10)

        assert conv
        assert abs(x.item() - 3.14) < 1e-10

    def test_contraction(self):
        """Simple contraction x -> 0.5*x + 1 converges to x=2."""
        def fp(x):
            return 0.5 * x + 1.0

        x0 = torch.tensor([10.0], dtype=torch.float64)
        x, n_iter, conv = squarem(fp, x0, tol=1e-10)

        assert conv
        assert abs(x.item() - 2.0) < 1e-8

    def test_multidimensional(self):
        """Multi-dimensional contraction."""
        A = torch.tensor([[0.3, 0.1], [0.1, 0.4]], dtype=torch.float64)
        b_vec = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def fp(x):
            return A @ x + b_vec

        # Fixed point: x = (I - A)^{-1} b
        x_star = torch.linalg.solve(torch.eye(2, dtype=torch.float64) - A, b_vec)

        x0 = torch.zeros(2, dtype=torch.float64)
        x, n_iter, conv = squarem(fp, x0, tol=1e-10)

        assert conv
        assert torch.allclose(x, x_star, atol=1e-8)


# ===================================================================
# ThresholdLinearModel
# ===================================================================

class TestThresholdLinearModelEM:
    """EM solver path."""

    @pytest.fixture
    def binary_continuous_data(self):
        """Simulated data: 1 binary + 1 continuous trait."""
        torch.manual_seed(42)
        n, p0 = 200, 2
        R = torch.eye(2, dtype=torch.float64)
        R[0, 1] = R[1, 0] = 0.3
        L = torch.linalg.cholesky(R)
        eps = torch.randn(n, 2, dtype=torch.float64) @ L.T

        X0 = torch.ones(n, p0, dtype=torch.float64)
        X0[:, 1] = torch.randn(n, dtype=torch.float64)
        beta = torch.tensor([[0.0, 1.0], [0.5, -0.3]], dtype=torch.float64)
        liab_true = X0 @ beta + eps

        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = (liab_true[:, 0] > 0).float()
        Y[:, 1] = liab_true[:, 1]

        return Y, X0, R, liab_true

    def test_em_converges(self, binary_continuous_data):
        Y, X0, R, _ = binary_continuous_data
        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=R, G_cov=torch.eye(2, dtype=torch.float64) * 0.5,
            solver="em", max_iter=30,
        )
        nf = model.fit_null(Y, X0)
        assert nf.log_likelihood is not None
        assert nf.log_likelihood < 0  # should be negative

    def test_liability_correlation(self, binary_continuous_data):
        Y, X0, R, liab_true = binary_continuous_data
        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=R, G_cov=torch.eye(2, dtype=torch.float64) * 0.5,
            solver="em", max_iter=30,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        corr = torch.corrcoef(
            torch.stack([ext.liabilities[:, 0], liab_true[:, 0]])
        )[0, 1].item()
        assert corr > 0.60, f"Liability correlation too low: {corr}"


class TestThresholdLinearModelNR:
    """NR solver path."""

    @pytest.fixture
    def binary_data(self):
        """Binary trait only (single ordinal)."""
        torch.manual_seed(123)
        n = 150
        X0 = torch.ones(n, 1, dtype=torch.float64)
        liab = 0.5 + torch.randn(n, dtype=torch.float64)
        Y = torch.zeros(n, 1, dtype=torch.float64)
        Y[:, 0] = (liab > 0).float()
        R = torch.eye(1, dtype=torch.float64)
        return Y, X0, R, liab

    def test_nr_converges(self, binary_data):
        Y, X0, R, _ = binary_data
        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=R, G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=20, em_warmup=3,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        assert ext.converged

    def test_nr_liability_correlation(self, binary_data):
        Y, X0, R, liab = binary_data
        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=R, G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=20, em_warmup=3,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        corr = torch.corrcoef(
            torch.stack([ext.liabilities[:, 0], liab])
        )[0, 1].item()
        assert corr > 0.50, f"NR liability correlation too low: {corr}"

    def test_nr_em_agreement(self):
        """NR and EM should produce similar liabilities."""
        torch.manual_seed(99)
        n = 200
        X0 = torch.ones(n, 2, dtype=torch.float64)
        X0[:, 1] = torch.randn(n, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)
        R[0, 1] = R[1, 0] = 0.2
        L = torch.linalg.cholesky(R)
        eps = torch.randn(n, 2, dtype=torch.float64) @ L.T
        beta = torch.tensor([[0.0, 0.5], [0.3, -0.2]], dtype=torch.float64)
        liab_true = X0 @ beta + eps
        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = (liab_true[:, 0] > 0).float()
        Y[:, 1] = liab_true[:, 1]

        G_cov = torch.eye(2, dtype=torch.float64) * 0.5

        model_em = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=R, G_cov=G_cov, solver="em", max_iter=30,
        )
        model_nr = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=R, G_cov=G_cov, solver="nr", max_iter=20, em_warmup=5,
        )

        nf_em = model_em.fit_null(Y, X0)
        nf_nr = model_nr.fit_null(Y, X0)

        corr = torch.corrcoef(
            torch.stack([
                nf_em._threshold_ext.liabilities[:, 0],
                nf_nr._threshold_ext.liabilities[:, 0],
            ])
        )[0, 1].item()
        assert corr > 0.90, f"NR-EM liability correlation too low: {corr}"


class TestThresholdScanNullPvalues:
    """ssGWAS null p-value calibration."""

    def test_null_pvalues_uniform(self):
        """Under null (random SNPs), p-values should be approximately uniform."""
        torch.manual_seed(77)
        n, m = 300, 100
        X0 = torch.ones(n, 1, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)
        liab = torch.randn(n, dtype=torch.float64)
        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = (liab > 0).float()
        Y[:, 1] = torch.randn(n, dtype=torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=R, G_cov=torch.eye(2, dtype=torch.float64) * 0.5,
            solver="em", max_iter=20,
        )
        nf = model.fit_null(Y, X0)

        G = torch.randn(n, m, dtype=torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        # KS test: p-values should be uniform
        from scipy import stats
        ks_stat, ks_p = stats.kstest(result.p.numpy(), "uniform")
        assert ks_p > 0.001, f"KS test failed: stat={ks_stat:.4f}, p={ks_p:.6f}"

    def test_scan_returns_valid_results(self):
        """Score test should return valid ScanResult."""
        torch.manual_seed(42)
        n, m = 100, 20
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.zeros(n, 1, dtype=torch.float64)
        Y[:, 0] = (torch.randn(n, dtype=torch.float64) > 0).float()

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=15,
        )
        nf = model.fit_null(Y, X0)

        G = torch.randn(n, m, dtype=torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        assert len(result) == m
        assert result.p.shape == (m,)
        assert result.beta.shape == (m,)
        assert result.se.shape == (m,)
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()
        assert result.test == "score"


class TestThresholdPowerDetection:
    """Power to detect planted causal SNP."""

    def test_causal_snp_detected(self):
        """A large-effect causal SNP should have a small p-value."""
        torch.manual_seed(55)
        n = 500
        X0 = torch.ones(n, 1, dtype=torch.float64)

        # Plant a strong causal SNP
        g_causal = torch.randn(n, dtype=torch.float64)
        liab = 0.8 * g_causal + torch.randn(n, dtype=torch.float64)
        Y = torch.zeros(n, 1, dtype=torch.float64)
        Y[:, 0] = (liab > 0).float()

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=15, em_warmup=5,
        )
        nf = model.fit_null(Y, X0)

        m = 20
        G = torch.randn(n, m, dtype=torch.float64)
        G[:, 0] = g_causal  # plant causal at index 0

        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)

        # Causal SNP should be top hit
        assert result.p[0] < 0.05, f"Causal SNP p={result.p[0]:.4f}"


class TestThresholdOrdinalMultiCategory:
    """Ordinal traits with >2 categories."""

    def test_three_category_ordinal(self):
        """3-category ordinal trait should work."""
        torch.manual_seed(33)
        n = 200
        X0 = torch.ones(n, 1, dtype=torch.float64)
        liab = torch.randn(n, dtype=torch.float64)
        # 3 categories: <-0.5 → 0, -0.5 to 0.5 → 1, >0.5 → 2
        cats = torch.zeros(n, dtype=torch.float64)
        cats[liab > -0.5] = 1
        cats[liab > 0.5] = 2
        Y = cats.unsqueeze(1)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[3],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=20,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        assert len(ext.thresholds_list) == 1
        assert len(ext.thresholds_list[0]) == 2  # 3 categories → 2 thresholds

    def test_two_ordinal_one_continuous(self):
        """Two ordinal + one continuous trait jointly."""
        torch.manual_seed(44)
        n = 200
        X0 = torch.ones(n, 1, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64) * 1.0
        L = torch.linalg.cholesky(R)
        eps = torch.randn(n, 3, dtype=torch.float64) @ L.T
        mu = X0 @ torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
        vals = mu + eps

        Y = torch.zeros(n, 3, dtype=torch.float64)
        Y[:, 0] = (vals[:, 0] > 0).float()  # binary
        cats = torch.zeros(n, dtype=torch.float64)
        cats[vals[:, 1] > -0.5] = 1
        cats[vals[:, 1] > 0.5] = 2
        Y[:, 1] = cats  # 3-category
        Y[:, 2] = vals[:, 2]  # continuous

        model = ThresholdLinearModel(
            trait_types=["ordinal", "ordinal", "continuous"],
            n_categories=[2, 3, 0],
            R=R, G_cov=torch.eye(3, dtype=torch.float64) * 0.3,
            solver="em", max_iter=20,
        )
        nf = model.fit_null(Y, X0)
        assert nf.log_likelihood is not None


class TestThresholdPolyploid:
    """Polyploid compatibility (dosage in [0, k])."""

    def test_tetraploid(self):
        """Tetraploid genotypes (dosage 0-4) should work."""
        torch.manual_seed(88)
        n, m = 100, 10
        ploidy = 4
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.zeros(n, 1, dtype=torch.float64)
        Y[:, 0] = (torch.randn(n) > 0).float().to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=15,
        )
        nf = model.fit_null(Y, X0)

        # Polyploid dosages in [0, 4]
        G = torch.randint(0, ploidy + 1, (n, m)).to(torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)
        assert len(result) == m
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()


class TestThresholdProtocol:
    """BaseModel protocol conformance."""

    def test_has_fit_null(self):
        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64),
        )
        assert hasattr(model, "fit_null")

    def test_has_score_chunk(self):
        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64),
        )
        assert hasattr(model, "score_chunk")

    def test_isinstance_check(self):
        from torchgwas.models.base import BaseModel
        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64),
        )
        assert isinstance(model, BaseModel)


# ===================================================================
# Edge-case tests (audit gap coverage)
# ===================================================================

class TestThresholdEdgeCases:
    """Edge cases identified in deep sanity audit."""

    def test_single_ordinal_no_continuous(self):
        """Single ordinal trait with no continuous traits (c1=1, c2=0)."""
        torch.manual_seed(200)
        n = 150
        X0 = torch.ones(n, 1, dtype=torch.float64)
        liab = torch.randn(n, dtype=torch.float64)
        Y = (liab > 0).float().unsqueeze(1).to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=20, em_warmup=3,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        assert ext.converged
        # Liability should correlate with truth
        corr = torch.corrcoef(
            torch.stack([ext.liabilities[:, 0], liab])
        )[0, 1].item()
        assert corr > 0.40, f"Single-ordinal liability corr too low: {corr}"

    def test_single_ordinal_em_solver(self):
        """Single ordinal with EM solver (no NR)."""
        torch.manual_seed(201)
        n = 120
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = (torch.randn(n, dtype=torch.float64) > 0.3).float().unsqueeze(1).to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=25,
        )
        nf = model.fit_null(Y, X0)
        assert nf.log_likelihood is not None

    def test_empty_snp_chunk(self):
        """Empty genotype chunk (m=0) should return empty ScanResult."""
        torch.manual_seed(202)
        n = 80
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = (torch.randn(n, dtype=torch.float64) > 0).float().unsqueeze(1).to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=15,
        )
        nf = model.fit_null(Y, X0)

        G_empty = torch.zeros(n, 0, dtype=torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(snp=[], chr=[], pos=[], a1=[], a2=[])
        result = model.score_chunk(G_empty, nf, vmeta)
        assert len(result) == 0

    def test_more_snps_than_samples(self):
        """m > n scenario should still produce valid p-values."""
        torch.manual_seed(203)
        n, m = 50, 100
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = (torch.randn(n, dtype=torch.float64) > 0).float().unsqueeze(1).to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="em", max_iter=15,
        )
        nf = model.fit_null(Y, X0)

        G = torch.randn(n, m, dtype=torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)
        assert len(result) == m
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_seed_reproducibility(self):
        """Same seed should give identical results."""
        def run_once():
            torch.manual_seed(204)
            n = 100
            X0 = torch.ones(n, 1, dtype=torch.float64)
            Y = (torch.randn(n, dtype=torch.float64) > 0).float().unsqueeze(1).to(torch.float64)
            model = ThresholdLinearModel(
                trait_types=["ordinal"],
                n_categories=[2],
                R=torch.eye(1, dtype=torch.float64),
                G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
                solver="em", max_iter=15,
            )
            nf = model.fit_null(Y, X0)
            return nf._threshold_ext.liabilities.clone()

        liab1 = run_once()
        liab2 = run_once()
        assert torch.allclose(liab1, liab2, atol=1e-12)

    def test_hexaploid(self):
        """Hexaploid genotypes (dosage 0-6) should work."""
        torch.manual_seed(205)
        n, m = 80, 15
        ploidy = 6
        X0 = torch.ones(n, 1, dtype=torch.float64)
        Y = torch.zeros(n, 2, dtype=torch.float64)
        Y[:, 0] = (torch.randn(n) > 0).float().to(torch.float64)
        Y[:, 1] = torch.randn(n, dtype=torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[2, 0],
            R=torch.eye(2, dtype=torch.float64),
            G_cov=torch.eye(2, dtype=torch.float64) * 0.5,
            solver="em", max_iter=15,
        )
        nf = model.fit_null(Y, X0)

        G = torch.randint(0, ploidy + 1, (n, m)).to(torch.float64)
        from torchgwas.models.base import VariantMeta
        vmeta = VariantMeta(
            snp=[f"SNP{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )
        result = model.score_chunk(G, nf, vmeta)
        assert len(result) == m
        assert (result.p >= 0).all()
        assert (result.p <= 1).all()

    def test_three_category_nr_solver(self):
        """3-category ordinal with NR solver."""
        torch.manual_seed(206)
        n = 200
        X0 = torch.ones(n, 1, dtype=torch.float64)
        liab = torch.randn(n, dtype=torch.float64)
        cats = torch.zeros(n, dtype=torch.float64)
        cats[liab > -0.5] = 1
        cats[liab > 0.5] = 2
        Y = cats.unsqueeze(1)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[3],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=20, em_warmup=5,
        )
        nf = model.fit_null(Y, X0)
        ext = nf._threshold_ext
        assert len(ext.thresholds_list[0]) == 2

    def test_multiple_covariates(self):
        """Multiple fixed-effect covariates (p0 > 2)."""
        torch.manual_seed(207)
        n = 200
        X0 = torch.ones(n, 4, dtype=torch.float64)
        X0[:, 1] = torch.randn(n, dtype=torch.float64)
        X0[:, 2] = torch.randn(n, dtype=torch.float64)
        X0[:, 3] = (torch.randn(n, dtype=torch.float64) > 0).float()

        beta = torch.tensor([[0.0], [0.3], [-0.2], [0.5]], dtype=torch.float64)
        liab = X0 @ beta + torch.randn(n, 1, dtype=torch.float64)
        Y = (liab > 0).float().to(torch.float64)

        model = ThresholdLinearModel(
            trait_types=["ordinal"],
            n_categories=[2],
            R=torch.eye(1, dtype=torch.float64),
            G_cov=torch.eye(1, dtype=torch.float64) * 0.5,
            solver="nr", max_iter=20, em_warmup=5,
        )
        nf = model.fit_null(Y, X0)
        assert nf._threshold_ext.converged


# Import here to avoid issues with existing circular imports
from torchgwas.models.threshold_linear import ThresholdLinearModel
