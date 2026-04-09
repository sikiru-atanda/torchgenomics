"""Tests for meta-analysis methods."""

import pytest
import torch

from torchgwas.postgwas._meta import (
    meta_fixed_effect,
    meta_han_eskin,
    meta_random_effect,
    meta_sample_size,
)


def _make_studies(m=200, K=5, beta_true=None, se_scale=0.5, seed=42):
    """Generate K studies with known true effects."""
    torch.manual_seed(seed)
    if beta_true is None:
        beta_true = torch.zeros(m, dtype=torch.float64)
    se = torch.rand(m, K, dtype=torch.float64) * se_scale + 0.1
    beta = beta_true.unsqueeze(1) + se * torch.randn(m, K, dtype=torch.float64)
    return beta, se


class TestFixedEffect:
    """Tests for inverse-variance weighted meta-analysis."""

    def test_single_study(self):
        """K=1: meta should equal single study."""
        beta = torch.tensor([[0.5], [1.0], [-0.3]], dtype=torch.float64)
        se = torch.tensor([[0.1], [0.2], [0.15]], dtype=torch.float64)
        result = meta_fixed_effect(beta, se)
        assert result.beta_meta[0].item() == pytest.approx(0.5, abs=1e-10)
        assert result.se_meta[0].item() == pytest.approx(0.1, abs=1e-10)

    def test_identical_studies_reduce_se(self):
        """K identical studies: se_meta = se / sqrt(K)."""
        K = 4
        beta = torch.tensor([[1.0]] * 1, dtype=torch.float64).expand(1, K)
        se = torch.tensor([[0.5]] * 1, dtype=torch.float64).expand(1, K)
        result = meta_fixed_effect(beta, se)
        expected_se = 0.5 / K**0.5
        assert result.se_meta[0].item() == pytest.approx(expected_se, rel=1e-6)

    def test_beta_meta_is_ivw(self):
        """Combined beta should be the inverse-variance weighted average."""
        beta = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        se = torch.tensor([[0.5, 1.0, 0.25]], dtype=torch.float64)
        result = meta_fixed_effect(beta, se)
        # Manual IVW
        w = 1.0 / se**2
        expected = (w * beta).sum(1) / w.sum(1)
        assert result.beta_meta[0].item() == pytest.approx(expected[0].item(), rel=1e-6)

    def test_null_type_i_error(self):
        """Under the null, p_meta should be approximately uniform."""
        torch.manual_seed(10)
        m = 5000
        beta, se = _make_studies(m=m, K=5, beta_true=torch.zeros(m), seed=10)
        result = meta_fixed_effect(beta, se)
        # Check that ~5% of p-values < 0.05
        fpr = (result.p_meta < 0.05).float().mean().item()
        assert 0.02 < fpr < 0.10, f"FPR = {fpr}, expected ~0.05"

    def test_cochran_q_under_heterogeneity(self):
        """Opposite effects across studies should yield large Q."""
        beta = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        se = torch.tensor([[0.1, 0.1]], dtype=torch.float64)
        result = meta_fixed_effect(beta, se)
        assert result.q_stat[0].item() > 10, "Q should be large for opposite effects"
        assert result.i2[0].item() > 0.5, "I2 should be high"


class TestRandomEffect:
    """Tests for DerSimonian-Laird random-effects meta-analysis."""

    def test_no_heterogeneity_matches_fixed(self):
        """Without heterogeneity, random effects ~ fixed effects."""
        torch.manual_seed(20)
        beta = torch.tensor([[1.0, 1.05, 0.95, 1.02]], dtype=torch.float64)
        se = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float64)
        fe = meta_fixed_effect(beta, se)
        re = meta_random_effect(beta, se)
        assert re.beta_meta[0].item() == pytest.approx(fe.beta_meta[0].item(), abs=0.05)

    def test_tau2_positive_under_heterogeneity(self):
        """Tau-squared should be positive when effects vary."""
        beta = torch.tensor([[2.0, -2.0, 1.0, -1.0]], dtype=torch.float64)
        se = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float64)
        result = meta_random_effect(beta, se)
        assert result.tau2[0].item() > 0

    def test_re_wider_ci_than_fe(self):
        """Random-effects SE should be >= fixed-effects SE."""
        beta = torch.tensor([[2.0, -1.0, 0.5, 1.5]], dtype=torch.float64)
        se = torch.tensor([[0.2, 0.2, 0.2, 0.2]], dtype=torch.float64)
        fe = meta_fixed_effect(beta, se)
        re = meta_random_effect(beta, se)
        assert re.se_meta[0].item() >= fe.se_meta[0].item() - 1e-10


class TestStouffer:
    """Tests for sample-size weighted meta-analysis."""

    def test_opposing_directions_cancel(self):
        """Opposite directions should yield p ~ 1 (non-significant)."""
        p = torch.tensor([[0.001, 0.001]], dtype=torch.float64)
        n = torch.tensor([1000.0, 1000.0], dtype=torch.float64)
        direction = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        result = meta_sample_size(p, n, direction)
        assert result.p_meta[0].item() > 0.5

    def test_concordant_directions_significant(self):
        """Same direction small p-values should combine to smaller p."""
        p = torch.tensor([[0.01, 0.01, 0.01]], dtype=torch.float64)
        n = torch.tensor([1000.0, 1000.0, 1000.0], dtype=torch.float64)
        direction = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
        result = meta_sample_size(p, n, direction)
        assert result.p_meta[0].item() < 0.02


class TestHanEskin:
    """Tests for RE2 meta-analysis."""

    def test_re2_returns_valid(self):
        """RE2 should return valid p-values."""
        beta, se = _make_studies(m=100, K=4, seed=30)
        result = meta_han_eskin(beta, se)
        assert (result.p_meta >= 0).all()
        assert (result.p_meta <= 1).all()
        assert result.method == "han_eskin"

    def test_re2_more_powerful_under_heterogeneity(self):
        """RE2 should find more significant SNPs than DL under heterogeneity."""
        torch.manual_seed(40)
        m = 500
        K = 5
        # Some SNPs with heterogeneous but non-zero effects
        beta_true = torch.zeros(m, dtype=torch.float64)
        beta_true[:20] = 0.5  # true signals

        se = torch.ones(m, K, dtype=torch.float64) * 0.3
        beta = beta_true.unsqueeze(1) + se * torch.randn(m, K, dtype=torch.float64)
        # Add heterogeneity to signals
        beta[:20] *= torch.tensor([1.0, 0.5, 1.5, 0.3, 2.0], dtype=torch.float64)

        re_dl = meta_random_effect(beta, se)
        re2 = meta_han_eskin(beta, se)

        # Count significant at p < 0.05
        n_sig_dl = (re_dl.p_meta[:20] < 0.05).sum().item()
        n_sig_re2 = (re2.p_meta[:20] < 0.05).sum().item()
        # RE2 should be at least as powerful (or close)
        assert n_sig_re2 >= n_sig_dl - 2, f"RE2={n_sig_re2} vs DL={n_sig_dl}"
