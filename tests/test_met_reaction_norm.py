"""Tests for reaction-norm parameterization in MultiEnvLMM."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.base import VariantMeta
from torchgenomics.models.multi_env_lmm import MultiEnvLMM

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
    Vg_true = torch.tensor([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.6],
        [0.5, 0.6, 1.0],
    ], dtype=torch.float64)
    L_vg = torch.linalg.cholesky(Vg_true)
    Z = torch.randn(n, E, dtype=torch.float64)
    U = L @ Z @ L_vg.T
    Y = U + torch.randn(n, E, dtype=torch.float64)
    return Y, X0, K, G


@pytest.fixture
def vmeta_10():
    """VariantMeta for 10 test SNPs."""
    return VariantMeta(
        snp=[f"rs{i}" for i in range(10)],
        chr=["1"] * 10,
        pos=list(range(10)),
        a1=["A"] * 10,
        a2=["G"] * 10,
    )


@pytest.fixture
def vmeta_200():
    """VariantMeta for 200 null SNPs."""
    return VariantMeta(
        snp=[f"rs{i}" for i in range(200)],
        chr=["1"] * 200,
        pos=list(range(200)),
        a1=["A"] * 200,
        a2=["G"] * 200,
    )


# ── Tests ─────────────────────────────────────────────────────────────

class TestReactionNorm:
    def test_alpha_equals_mean_beta(self, met_data, vmeta_10):
        """α should equal mean(β_e) algebraically."""
        Y, X0, K, G = met_data
        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        expected_alpha = result.beta.mean(dim=-1)
        assert torch.allclose(result.alpha, expected_alpha, atol=1e-10)

    def test_delta_sum_to_zero(self, met_data, vmeta_10):
        """Σ δ_e should be zero for each SNP."""
        Y, X0, K, G = met_data
        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        delta_sum = result.delta.sum(dim=-1)
        assert torch.allclose(delta_sum, torch.zeros_like(delta_sum), atol=1e-10)

    def test_stat_any_equals_joint(self, met_data, vmeta_10):
        """Reaction-norm joint test (stat/p) should equal the per-env joint test."""
        Y, X0, K, G = met_data
        G_chunk = G[:, :10].to(torch.float64)

        # per_env model
        model_pe = MultiEnvLMM(parameterization="per_env")
        nf_pe = model_pe.fit_null(Y, X0, K)
        result_pe = model_pe.score_chunk(G_chunk, nf_pe, vmeta_10)

        # reaction_norm model (same null fit data)
        model_rn = MultiEnvLMM(parameterization="reaction_norm")
        nf_rn = model_rn.fit_null(Y, X0, K)
        result_rn = model_rn.score_chunk(G_chunk, nf_rn, vmeta_10)

        # Joint stat/p should be identical (same GLS system)
        assert torch.allclose(result_rn.stat, result_pe.stat, atol=1e-6)
        assert torch.allclose(result_rn.p, result_pe.p, atol=1e-6)

    def test_stable_test_calibrated(self, met_data, vmeta_200):
        """Under null, p_stable median > 0.1."""
        Y, X0, K, G = met_data
        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, 50:250].to(torch.float64), nf, vmeta_200)

        median_p = result.p_stable.median().item()
        assert median_p > 0.1, f"Stable test inflated: median p = {median_p}"

    def test_gxe_test_calibrated(self, met_data, vmeta_200):
        """Under null, p_gxe median > 0.1."""
        Y, X0, K, G = met_data
        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, 50:250].to(torch.float64), nf, vmeta_200)

        median_p = result.p_gxe.median().item()
        assert median_p > 0.1, f"GxE test inflated: median p = {median_p}"

    def test_detects_stable_snp(self, met_data, vmeta_10):
        """Plant β=[2,2,2] → p_stable significant, p_gxe not."""
        Y, X0, K, G = met_data
        causal = G[:, 0:1].to(torch.float64)
        causal = (causal - causal.mean()) / causal.std().clamp(min=1e-6)
        betas = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        Y_signal = Y + causal * betas.unsqueeze(0)

        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y_signal, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        assert result.p_stable[0] < 0.01, f"Stable not detected: p={result.p_stable[0]}"
        # GxE should NOT be significant (all betas equal)
        assert result.p_gxe[0] > 0.01, f"GxE spuriously significant: p={result.p_gxe[0]}"

    def test_detects_gxe_snp(self, met_data, vmeta_10):
        """Plant β=[3,0,-3] → p_gxe significant."""
        Y, X0, K, G = met_data
        causal = G[:, 0:1].to(torch.float64)
        causal = (causal - causal.mean()) / causal.std().clamp(min=1e-6)
        betas = torch.tensor([3.0, 0.0, -3.0], dtype=torch.float64)
        Y_signal = Y + causal * betas.unsqueeze(0)

        model = MultiEnvLMM(parameterization="reaction_norm")
        nf = model.fit_null(Y_signal, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        assert result.p_gxe[0] < 0.01, f"GxE not detected: p={result.p_gxe[0]}"

    def test_default_parameterization_unchanged(self, met_data, vmeta_10):
        """Default MultiEnvLMM() should NOT have reaction-norm fields."""
        Y, X0, K, G = met_data
        model = MultiEnvLMM()  # default: per_env
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        assert result.alpha is None
        assert result.delta is None
        assert result.stat_stable is None
        assert result.p_gxe is None
