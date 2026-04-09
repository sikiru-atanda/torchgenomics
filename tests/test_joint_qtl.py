"""Tests for multi-QTL joint model fitting (charter Section 9g)."""

from __future__ import annotations

import pytest
import torch

from torchgwas.models.joint_qtl import fit_joint_qtl
from torchgwas.config import NumericalConfig


@pytest.fixture
def synthetic_data():
    """Synthetic data with two known causal QTL."""
    torch.manual_seed(42)
    n = 200
    m = 50

    G = torch.randn(n, m, dtype=torch.float64)
    K = G @ G.T / m  # simple GRM

    # Two causal SNPs with known effects
    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[5] = 2.0  # QTL 1
    beta_true[25] = 1.5  # QTL 2

    Y = G @ beta_true + torch.randn(n, dtype=torch.float64) * 0.5
    X0 = torch.ones(n, 1, dtype=torch.float64)

    return Y, X0, G, K


class TestFitJointQTL:
    def test_basic_fit(self, synthetic_data):
        """Basic joint fit with two known QTL."""
        Y, X0, G, K = synthetic_data
        result = fit_joint_qtl(
            Y, X0, G, K,
            qtl_indices=[5, 25],
            qtl_snps=["snp_5", "snp_25"],
            backward_elimination=False,
        )
        assert len(result.qtl_snps) == 2
        assert result.beta.shape == (2,)
        assert result.se.shape == (2,)
        assert result.r_squared.shape == (2,)
        assert result.lrt_p.shape == (2,)
        assert result.total_r_squared > 0.0  # QTL explain some variance

    def test_betas_recover_true_effects(self, synthetic_data):
        """Joint betas should approximately recover the true effect sizes."""
        Y, X0, G, K = synthetic_data
        result = fit_joint_qtl(
            Y, X0, G, K,
            qtl_indices=[5, 25],
            backward_elimination=False,
        )
        # True betas: 2.0 and 1.5
        assert abs(result.beta[0].item() - 2.0) < 0.5
        assert abs(result.beta[1].item() - 1.5) < 0.5

    def test_lrt_significant_for_true_qtl(self, synthetic_data):
        """True QTL should have significant LRT p-values."""
        Y, X0, G, K = synthetic_data
        result = fit_joint_qtl(
            Y, X0, G, K,
            qtl_indices=[5, 25],
            backward_elimination=False,
        )
        assert result.lrt_p[0].item() < 0.01
        assert result.lrt_p[1].item() < 0.01

    def test_backward_elimination_drops_null(self, synthetic_data):
        """Backward elimination should drop null QTL."""
        Y, X0, G, K = synthetic_data
        # Include a null SNP (index 40, no effect)
        result = fit_joint_qtl(
            Y, X0, G, K,
            qtl_indices=[5, 25, 40],
            qtl_snps=["snp_5", "snp_25", "snp_40"],
            backward_elimination=True,
            elimination_threshold=0.05,
        )
        # SNP 40 should be eliminated
        assert result.n_eliminated >= 1
        assert "snp_40" not in result.qtl_snps

    def test_empty_qtl_list(self, synthetic_data):
        """Empty QTL list returns empty result."""
        Y, X0, G, K = synthetic_data
        result = fit_joint_qtl(Y, X0, G, K, qtl_indices=[])
        assert len(result.qtl_snps) == 0
        assert result.total_r_squared == 0.0

    def test_single_qtl(self, synthetic_data):
        """Single QTL fit should work."""
        Y, X0, G, K = synthetic_data
        result = fit_joint_qtl(
            Y, X0, G, K,
            qtl_indices=[5],
            backward_elimination=False,
        )
        assert len(result.qtl_snps) == 1
        assert result.beta.shape == (1,)
