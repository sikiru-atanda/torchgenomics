"""Tests for MultiEnvLMM — single-trait multi-environment GWAS."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.config import NumericalConfig
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.base import VariantMeta
from torchgenomics.models.multi_env_lmm import (
    EnvScanResult,
    MultiEnvLMM,
    complete_case_filter,
    reshape_long_to_wide,
)

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def met_data():
    """Simulated MET data: 150 samples, 3 environments, h²~0.5."""
    torch.manual_seed(42)
    n, m, E = 150, 300, 3
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))

    # Correlated genetic effects across environments (rg ~ 0.7)
    Vg_true = torch.tensor([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.6],
        [0.5, 0.6, 1.0],
    ], dtype=torch.float64)
    L_vg = torch.linalg.cholesky(Vg_true)

    # Draw genetic values: u ~ N(0, K ⊗ Vg)
    Z = torch.randn(n, E, dtype=torch.float64)
    U = L @ Z @ L_vg.T  # (n, E)

    # Add residual noise
    Y = U + torch.randn(n, E, dtype=torch.float64)

    return Y, X0, K, G


@pytest.fixture
def met_data_with_signal(met_data):
    """MET data + 1 planted SNP with heterogeneous effects across envs."""
    Y, X0, K, G = met_data
    n = Y.shape[0]
    # Planted causal SNP with environment-specific effects
    causal = G[:, 0:1].to(torch.float64)
    causal = (causal - causal.mean()) / causal.std().clamp(min=1e-6)
    # Strong effect in env 0, weaker/absent in envs 1,2
    betas = torch.tensor([2.0, 0.3, 0.3], dtype=torch.float64)
    Y = Y + causal * betas.unsqueeze(0)
    return Y, X0, K, G, betas


@pytest.fixture
def met_model():
    return MultiEnvLMM()


# ── Data utility tests ───────────────────────────────────────────────

class TestDataUtilities:
    def test_reshape_long_to_wide(self):
        """Long format → wide format with NaN for missing."""
        Y_long = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        sample_ids = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        env_ids = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)

        Y_wide, samples, envs = reshape_long_to_wide(Y_long, sample_ids, env_ids)

        assert Y_wide.shape == (3, 2)
        assert Y_wide[0, 0] == 1.0
        assert Y_wide[0, 1] == 2.0
        assert Y_wide[1, 0] == 3.0
        assert Y_wide[1, 1] == 4.0
        assert Y_wide[2, 0] == 5.0
        assert torch.isnan(Y_wide[2, 1])  # sample 2 missing env 1

    def test_complete_case_filter(self):
        """Complete-case drops rows with NaN."""
        Y = torch.tensor([
            [1.0, 2.0],
            [float("nan"), 4.0],
            [5.0, 6.0],
        ], dtype=torch.float64)
        X0 = torch.ones(3, 1, dtype=torch.float64)
        K = torch.eye(3, dtype=torch.float64)

        Y_f, X0_f, K_f, mask = complete_case_filter(Y, X0, K)

        assert Y_f.shape == (2, 2)
        assert X0_f.shape == (2, 1)
        assert K_f.shape == (2, 2)
        assert mask.sum() == 2
        assert not mask[1]  # row 1 had NaN


# ── Null model tests ─────────────────────────────────────────────────

class TestFitNull:
    def test_fit_null_converges(self, met_data, met_model):
        """Null model should converge with well-conditioned MET data."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K, env_names=["loc1", "loc2", "loc3"])
        assert nf.converged
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.Vg.shape == (3, 3)
        assert nf.Ve.shape == (3, 3)

    def test_vg_ve_positive_definite(self, met_data, met_model):
        """Vg and Ve should be positive definite."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K)
        eig_g = torch.linalg.eigvalsh(nf.Vg)
        eig_e = torch.linalg.eigvalsh(nf.Ve)
        assert (eig_g > -1e-8).all(), f"Vg not PD: {eig_g}"
        assert (eig_e > -1e-8).all(), f"Ve not PD: {eig_e}"

    def test_env_names_stored(self, met_data, met_model):
        """Environment names should be stored on NullFit."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K, env_names=["field_A", "field_B", "greenhouse"])
        assert nf.env_names == ["field_A", "field_B", "greenhouse"]
        assert nf.n_env == 3

    def test_complete_case_with_nan(self, met_data, met_model):
        """fit_null should handle NaN values by dropping incomplete rows."""
        Y, X0, K, _ = met_data
        Y_nan = Y.clone()
        Y_nan[0, 1] = float("nan")
        Y_nan[5, 0] = float("nan")
        nf = met_model.fit_null(Y_nan, X0, K)
        assert nf.converged
        # keep_mask should exclude rows 0 and 5
        assert not nf.keep_mask[0]
        assert not nf.keep_mask[5]

    def test_rejects_1d_input(self, met_model):
        """Should reject 1-d phenotype."""
        with pytest.raises(ValueError, match="E >= 2"):
            met_model.fit_null(
                torch.randn(10, dtype=torch.float64),
                torch.ones(10, 1, dtype=torch.float64),
                torch.eye(10, dtype=torch.float64),
            )


# ── Interpretive methods ─────────────────────────────────────────────

class TestInterpretive:
    def test_genetic_correlation_range(self, met_data, met_model):
        """Genetic correlations should be in [-1, 1] with diagonal = 1."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K)
        rg = met_model.genetic_correlation(nf)
        assert rg.shape == (3, 3)
        # Diagonal should be 1.0
        assert torch.allclose(torch.diag(rg), torch.ones(3, dtype=rg.dtype), atol=1e-6)
        # Off-diagonal in [-1, 1]
        assert (rg >= -1.0 - 1e-6).all()
        assert (rg <= 1.0 + 1e-6).all()

    def test_per_env_heritability_range(self, met_data, met_model):
        """Per-environment h² should be in [0, 1]."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K)
        h2 = met_model.per_env_heritability(nf)
        assert h2.shape == (3,)
        assert (h2 >= 0.0).all()
        assert (h2 <= 1.0).all()

    def test_env_specific_variance(self, met_data, met_model):
        """env_specific_variance returns dict with correct structure."""
        Y, X0, K, _ = met_data
        nf = met_model.fit_null(Y, X0, K, env_names=["A", "B", "C"])
        vs = met_model.env_specific_variance(nf)
        assert set(vs.keys()) == {"A", "B", "C"}
        for name in ["A", "B", "C"]:
            assert "Vg" in vs[name]
            assert "Ve" in vs[name]
            assert "h2" in vs[name]
            assert vs[name]["Vg"] > 0
            assert vs[name]["Ve"] > 0


# ── Scan tests ───────────────────────────────────────────────────────

class TestScoreChunk:
    def test_output_shapes(self, met_data, met_model):
        """All output tensors should have correct shapes."""
        Y, X0, K, G = met_data
        nf = met_model.fit_null(Y, X0, K)
        m_test = 20
        G_chunk = G[:, :m_test].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m_test)],
            chr=["1"] * m_test,
            pos=list(range(m_test)),
            a1=["A"] * m_test,
            a2=["G"] * m_test,
        )
        result = met_model.score_chunk(G_chunk, nf, vmeta)

        assert isinstance(result, EnvScanResult)
        E = 3
        assert result.beta.shape == (m_test, E)
        assert result.se.shape == (m_test, E)
        assert result.stat.shape == (m_test,)
        assert result.p.shape == (m_test,)
        assert result.stat_homogeneity.shape == (m_test,)
        assert result.p_homogeneity.shape == (m_test,)
        assert result.stat_marginal.shape == (m_test, E)
        assert result.p_marginal.shape == (m_test, E)

    def test_joint_pvalues_valid(self, met_data, met_model):
        """Joint p-values should be in (0, 1]."""
        Y, X0, K, G = met_data
        nf = met_model.fit_null(Y, X0, K)
        G_chunk = G[:, :50].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(50)],
            chr=["1"] * 50,
            pos=list(range(50)),
            a1=["A"] * 50,
            a2=["G"] * 50,
        )
        result = met_model.score_chunk(G_chunk, nf, vmeta)

        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert (result.p_homogeneity > 0).all()
        assert (result.p_homogeneity <= 1.0).all()

    def test_homogeneity_detects_heterogeneity(self, met_data_with_signal):
        """Homogeneity test should detect heterogeneous SNP effects."""
        Y, X0, K, G, betas = met_data_with_signal
        model = MultiEnvLMM()
        nf = model.fit_null(Y, X0, K)

        # Score the causal SNP (index 0) along with some null SNPs
        G_chunk = G[:, :10].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(10)],
            chr=["1"] * 10,
            pos=list(range(10)),
            a1=["A"] * 10,
            a2=["G"] * 10,
        )
        result = model.score_chunk(G_chunk, nf, vmeta)

        # The planted SNP (index 0) should have significant joint test
        assert result.p[0] < 0.01, f"Joint test not significant: p={result.p[0]}"
        # The planted SNP has heterogeneous effects (2.0 vs 0.3, 0.3)
        # so homogeneity should be rejected
        assert result.p_homogeneity[0] < 0.05, (
            f"Homogeneity not rejected for heterogeneous SNP: p={result.p_homogeneity[0]}"
        )

    def test_null_calibration_joint(self, met_data, met_model):
        """Under null, joint test p-values should not be systematically inflated."""
        Y, X0, K, G = met_data
        nf = met_model.fit_null(Y, X0, K)

        # Use SNPs 50-250 (not involved in simulation) as null
        m_null = 200
        G_chunk = G[:, 50:50 + m_null].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m_null)],
            chr=["1"] * m_null,
            pos=list(range(m_null)),
            a1=["A"] * m_null,
            a2=["G"] * m_null,
        )
        result = met_model.score_chunk(G_chunk, nf, vmeta)

        # Median p-value should be > 0.1 (not systematically inflated)
        median_p = result.p.median().item()
        assert median_p > 0.1, f"Joint test inflated: median p = {median_p}"

    def test_null_calibration_homogeneity(self, met_data, met_model):
        """Under null, homogeneity p-values should not be inflated."""
        Y, X0, K, G = met_data
        nf = met_model.fit_null(Y, X0, K)

        m_null = 200
        G_chunk = G[:, 50:50 + m_null].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m_null)],
            chr=["1"] * m_null,
            pos=list(range(m_null)),
            a1=["A"] * m_null,
            a2=["G"] * m_null,
        )
        result = met_model.score_chunk(G_chunk, nf, vmeta)

        median_p = result.p_homogeneity.median().item()
        assert median_p > 0.1, f"Homogeneity test inflated: median p = {median_p}"

    def test_per_env_marginal_consistent(self, met_data, met_model):
        """Per-env marginal stats should equal beta²/se²."""
        Y, X0, K, G = met_data
        nf = met_model.fit_null(Y, X0, K)
        G_chunk = G[:, :10].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(10)],
            chr=["1"] * 10,
            pos=list(range(10)),
            a1=["A"] * 10,
            a2=["G"] * 10,
        )
        result = met_model.score_chunk(G_chunk, nf, vmeta)

        expected = result.beta ** 2 / result.se ** 2
        assert torch.allclose(result.stat_marginal, expected, atol=1e-6)

    def test_scan_with_nan_phenotype(self, met_data, met_model):
        """Scan should work when fit_null was called with NaN phenotypes."""
        Y, X0, K, G = met_data
        Y_nan = Y.clone()
        Y_nan[0, 1] = float("nan")
        nf = met_model.fit_null(Y_nan, X0, K)

        G_chunk = G[:, :10].to(torch.float64)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(10)],
            chr=["1"] * 10,
            pos=list(range(10)),
            a1=["A"] * 10,
            a2=["G"] * 10,
        )
        # G_chunk is (n_original, m), keep_mask should filter
        result = met_model.score_chunk(G_chunk, nf, vmeta)
        assert result.beta.shape[0] == 10
        assert (result.p > 0).all()

    def test_update_null_works(self, met_data):
        """update_null should resume optimization."""
        Y, X0, K, _ = met_data
        config = NumericalConfig()
        config.reml_max_iter = 3
        model = MultiEnvLMM(config=config)
        nf = model.fit_null(Y, X0, K)
        orig_trace_len = len(nf.optimizer_trace)

        updated = model.update_null(nf, max_iter=100)
        assert updated.Vg is not None
        assert len(updated.optimizer_trace) > orig_trace_len
