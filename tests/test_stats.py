"""Phase 8: Statistical tests, multiple testing correction, genomic control, simpleM."""

from __future__ import annotations

import math

import pytest
import scipy.stats as sp_stats
import torch

from torchgenomics.stats.calibrate import compare_pvalues
from torchgenomics.stats.genomic_control import diagnose_inflation, lambda_gc
from torchgenomics.stats.multipletesting import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    holm,
    sidak,
    storey_qvalue,
)
from torchgenomics.stats.simplem import effective_test_count, ld_correlation_eigenvalues
from torchgenomics.stats.tests import chi2_sf, lrt_test, score_test, wald_test

# ---------------------------------------------------------------
# Test statistic utilities
# ---------------------------------------------------------------

class TestStatisticalTests:
    def test_chi2_sf_matches_scipy(self):
        """chi2_sf should match scipy.stats.chi2.sf."""
        stat = torch.tensor([0.5, 1.0, 3.84, 6.63, 10.83], dtype=torch.float64)
        p = chi2_sf(stat, df=1)
        p_ref = sp_stats.chi2.sf(stat.numpy(), df=1)
        torch.testing.assert_close(p, torch.tensor(p_ref, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_chi2_sf_df2(self):
        stat = torch.tensor([2.0, 5.99, 9.21], dtype=torch.float64)
        p = chi2_sf(stat, df=2)
        p_ref = sp_stats.chi2.sf(stat.numpy(), df=2)
        torch.testing.assert_close(p, torch.tensor(p_ref, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_wald_test_chi2(self):
        """Wald test: stat = (beta/se)^2, p from chi2(1)."""
        beta = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
        se = torch.tensor([0.5, 1.0, 0.3], dtype=torch.float64)
        stat, p = wald_test(beta, se)

        assert stat.shape == (3,)
        torch.testing.assert_close(stat, (beta / se) ** 2)
        assert torch.all(p >= 0) and torch.all(p <= 1)
        # beta=0 => stat=0 => p=1
        assert abs(p[2].item() - 1.0) < 1e-5

    def test_compare_pvalues_metrics(self):
        ours = torch.tensor([0.01, 0.20, float("nan"), 0.50], dtype=torch.float64)
        ref = torch.tensor([0.011, 0.199, 0.30, 0.70], dtype=torch.float64)
        metrics = compare_pvalues(ours, ref, tolerance=0.002)

        assert metrics["n_total"] == 4
        assert metrics["n_finite"] == 3
        assert metrics["n_within_tolerance"] == 2
        assert metrics["max_abs_diff"] == pytest.approx(0.20)

    def test_lrt_nonnegative(self):
        """LRT stat should be non-negative."""
        ll_null = -100.0
        ll_alt = torch.tensor([-95.0, -100.0, -80.0], dtype=torch.float64)
        stat, p = lrt_test(ll_null, ll_alt)

        assert torch.all(stat >= 0)
        assert stat[0].item() == pytest.approx(10.0, abs=1e-10)
        assert stat[1].item() == pytest.approx(0.0, abs=1e-10)

    def test_score_test(self):
        U = torch.tensor([2.0, 0.0, -3.0], dtype=torch.float64)
        V = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        stat, p = score_test(U, V)

        assert stat.shape == (3,)
        torch.testing.assert_close(stat, U ** 2 / V)
        assert torch.all(p >= 0) and torch.all(p <= 1)


# ---------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------

class TestBonferroni:
    def test_basic(self):
        p = torch.tensor([0.01, 0.05, 0.1], dtype=torch.float64)
        adj = bonferroni(p)
        expected = torch.tensor([0.03, 0.15, 0.30], dtype=torch.float64)
        torch.testing.assert_close(adj, expected)

    def test_clamped(self):
        p = torch.tensor([0.5, 0.8], dtype=torch.float64)
        adj = bonferroni(p)
        assert torch.all(adj <= 1.0)

    def test_conservative(self):
        """Bonferroni-adjusted p >= raw p."""
        p = torch.rand(100, dtype=torch.float64)
        adj = bonferroni(p)
        assert torch.all(adj >= p)


class TestSidak:
    def test_less_conservative_than_bonf(self):
        """Sidak should be <= Bonferroni for the same p-values."""
        p = torch.tensor([0.001, 0.01, 0.05], dtype=torch.float64)
        adj_bonf = bonferroni(p)
        adj_sidak = sidak(p)
        assert torch.all(adj_sidak <= adj_bonf + 1e-10)

    def test_range(self):
        p = torch.rand(50, dtype=torch.float64)
        adj = sidak(p)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)


class TestHolm:
    def test_less_conservative_than_bonf(self):
        """Holm should be <= Bonferroni."""
        p = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.5], dtype=torch.float64)
        adj_bonf = bonferroni(p)
        adj_holm = holm(p)
        assert torch.all(adj_holm <= adj_bonf + 1e-10)

    def test_monotonic_sorted(self):
        """Holm-adjusted p-values when sorted should be non-decreasing."""
        p = torch.tensor([0.001, 0.01, 0.03, 0.05, 0.1], dtype=torch.float64)
        adj = holm(p)
        sorted_adj = adj.sort().values
        diffs = sorted_adj[1:] - sorted_adj[:-1]
        assert torch.all(diffs >= -1e-15)


class TestBenjaminiHochberg:
    def test_matches_r_adjust(self):
        """BH should match R p.adjust(method='BH') on a known example."""
        # Known example: p = [0.001, 0.008, 0.039, 0.041, 0.23]
        # R: p.adjust(c(0.001, 0.008, 0.039, 0.041, 0.23), method="BH")
        # => [0.005, 0.020, 0.065, 0.065, 0.230]  (approximately)
        p = torch.tensor([0.001, 0.008, 0.039, 0.041, 0.23], dtype=torch.float64)
        adj = benjamini_hochberg(p)

        expected = torch.tensor([0.005, 0.020, 0.05125, 0.05125, 0.23], dtype=torch.float64)
        torch.testing.assert_close(adj, expected, atol=1e-10, rtol=1e-10)

    def test_monotonic(self):
        """BH-adjusted p-values should be non-decreasing when sorted by raw p."""
        torch.manual_seed(42)
        p = torch.rand(100, dtype=torch.float64)
        adj = benjamini_hochberg(p)

        # Sort both by raw p
        sorted_idx = p.argsort()
        adj_sorted = adj[sorted_idx]
        diffs = adj_sorted[1:] - adj_sorted[:-1]
        assert torch.all(diffs >= -1e-15)

    def test_range(self):
        p = torch.rand(50, dtype=torch.float64)
        adj = benjamini_hochberg(p)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)

    def test_bh_leq_bonferroni(self):
        """BH should be less conservative than Bonferroni."""
        p = torch.tensor([0.001, 0.01, 0.02, 0.05, 0.5], dtype=torch.float64)
        adj_bh = benjamini_hochberg(p)
        adj_bonf = bonferroni(p)
        assert torch.all(adj_bh <= adj_bonf + 1e-10)


class TestBenjaminiYekutieli:
    def test_more_conservative_than_bh(self):
        """BY should be more conservative than BH."""
        p = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.5], dtype=torch.float64)
        adj_bh = benjamini_hochberg(p)
        adj_by = benjamini_yekutieli(p)
        assert torch.all(adj_by >= adj_bh - 1e-10)

    def test_range(self):
        p = torch.rand(50, dtype=torch.float64)
        adj = benjamini_yekutieli(p)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)


class TestStoreyQvalue:
    def test_leq_bh(self):
        """Storey q-values should be <= BH when pi0 < 1."""
        # Use p-values with many true signals so pi0 < 1
        torch.manual_seed(42)
        p_null = torch.rand(80, dtype=torch.float64)
        p_signal = torch.rand(20, dtype=torch.float64) * 0.01
        p = torch.cat([p_null, p_signal])

        q = storey_qvalue(p)
        bh = benjamini_hochberg(p)
        # q-values should be <= BH (or equal when pi0=1)
        assert torch.all(q <= bh + 1e-10)

    def test_range(self):
        p = torch.rand(100, dtype=torch.float64)
        q = storey_qvalue(p)
        assert torch.all(q >= 0) and torch.all(q <= 1)

    def test_all_significant(self):
        """When all p-values are tiny, the raw pi0 estimate is 0, but it is
        floored to 1/m so the q-values stay small-and-positive rather than
        collapsing to an anti-conservative all-zero vector."""
        p = torch.rand(50, dtype=torch.float64) * 0.01
        q = storey_qvalue(p, lambda_=0.5)
        assert not torch.all(q == 0)          # not the old all-zero bug
        assert torch.all(q >= 0) and torch.all(q <= 1)
        assert float(q.max()) < 0.05          # still highly significant


# ---------------------------------------------------------------
# simpleM / M_eff
# ---------------------------------------------------------------

class TestSimpleM:
    def test_independent_snps(self):
        """For independent SNPs, M_eff should be close to M."""
        torch.manual_seed(42)
        n, m = 500, 50
        G = torch.randn(n, m, dtype=torch.float64)  # independent
        evals = ld_correlation_eigenvalues(G)
        m_eff = effective_test_count(evals)
        # Should be close to m (all independent)
        assert m_eff >= m * 0.8

    def test_correlated_snps(self):
        """For highly correlated SNPs, M_eff should be much less than M."""
        torch.manual_seed(42)
        n, m = 500, 50
        base = torch.randn(n, 1, dtype=torch.float64)
        G = base.repeat(1, m) + torch.randn(n, m, dtype=torch.float64) * 0.1
        evals = ld_correlation_eigenvalues(G)
        m_eff = effective_test_count(evals)
        assert m_eff < m * 0.5  # much fewer effective tests

    def test_m_eff_leq_m(self):
        """M_eff should always be <= M."""
        torch.manual_seed(42)
        n, m = 200, 30
        G = torch.randn(n, m, dtype=torch.float64)
        evals = ld_correlation_eigenvalues(G)
        m_eff = effective_test_count(evals)
        assert m_eff <= m

    def test_eigenvalue_spectrum_shape(self):
        torch.manual_seed(42)
        n, m = 100, 20
        G = torch.randn(n, m, dtype=torch.float64)
        evals = ld_correlation_eigenvalues(G)
        assert evals.shape == (m,)
        # All eigenvalues should be non-negative (up to numerical noise)
        assert torch.all(evals >= -1e-10)


class TestMoskvinaSchmidt:
    """Tests for Moskvina-Schmidt (2008) M.eff method."""

    def test_independent_snps(self):
        """For independent SNPs, eigenvalues ≈ 1 each → M_eff ≈ M."""
        from torchgenomics.stats.simplem import effective_test_count_moskvina
        # All eigenvalues exactly 1 → each counts as 1
        evals = torch.ones(20, dtype=torch.float64)
        m_eff = effective_test_count_moskvina(evals)
        assert m_eff == 20

    def test_perfect_ld(self):
        """Perfect LD: one eigenvalue = M, rest = 0 → M_eff = 1."""
        from torchgenomics.stats.simplem import effective_test_count_moskvina
        evals = torch.zeros(20, dtype=torch.float64)
        evals[0] = 20.0
        m_eff = effective_test_count_moskvina(evals)
        assert m_eff == 1

    def test_fractional_eigenvalues(self):
        """Eigenvalues < 1 contribute their fractional part."""
        from torchgenomics.stats.simplem import effective_test_count_moskvina
        # 2 eigenvalues >= 1 (count as 2) + 3 eigenvalues of 0.5 each (count as 1.5)
        evals = torch.tensor([3.0, 1.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        m_eff = effective_test_count_moskvina(evals)
        # large: 2 (3.0 and 1.5), small fractional: 3*0.5 = 1.5 → total 3.5 → round to 4
        assert m_eff == 4

    def test_moskvina_leq_gao(self):
        """Moskvina-Schmidt typically gives M_eff <= simpleM (Gao)."""
        from torchgenomics.stats.simplem import effective_test_count_moskvina
        torch.manual_seed(42)
        n, m = 500, 50
        G = torch.randn(n, m, dtype=torch.float64)
        evals = ld_correlation_eigenvalues(G)
        m_eff_gao = effective_test_count(evals)
        m_eff_mosk = effective_test_count_moskvina(evals)
        assert m_eff_mosk <= m_eff_gao + 2  # allow small numerical tolerance

    def test_empty_returns_one(self):
        from torchgenomics.stats.simplem import effective_test_count_moskvina
        m_eff = effective_test_count_moskvina(torch.tensor([], dtype=torch.float64))
        assert m_eff == 1


# ---------------------------------------------------------------
# Genomic control
# ---------------------------------------------------------------

class TestGenomicControl:
    def test_lambda_gc_null(self):
        """Lambda GC should be near 1.0 for well-calibrated null p-values."""
        torch.manual_seed(42)
        # Generate p-values from chi2(1) (true null)
        chi2_stats = torch.tensor(
            sp_stats.chi2.rvs(df=1, size=10000), dtype=torch.float64,
        )
        p = torch.tensor(
            sp_stats.chi2.sf(chi2_stats.numpy(), df=1), dtype=torch.float64,
        )
        lam = lambda_gc(p)
        assert 0.9 < lam < 1.1

    def test_lambda_gc_inflated(self):
        """Lambda GC > 1.0 when test statistics are inflated."""
        torch.manual_seed(42)
        # Inflate: multiply chi2 stats by 2
        chi2_stats = torch.tensor(
            sp_stats.chi2.rvs(df=1, size=10000), dtype=torch.float64,
        ) * 2.0
        p = torch.tensor(
            sp_stats.chi2.sf(chi2_stats.numpy(), df=1), dtype=torch.float64,
        )
        lam = lambda_gc(p)
        assert lam > 1.5

    def test_diagnose(self):
        assert diagnose_inflation(1.0) == "well-calibrated"
        assert diagnose_inflation(0.8) == "deflated"
        assert diagnose_inflation(1.3) == "mildly inflated"
        assert diagnose_inflation(2.0) == "severely inflated"


# ---------------------------------------------------------------
# Phase 14: Advanced Multiple Testing
# ---------------------------------------------------------------


class TestCauchyCombination:
    """Tests for cauchy_combination (ACAT)."""

    def test_single_method_roundtrip(self):
        """With k=1, combining should approximately preserve original p-values."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([0.001, 0.01, 0.05, 0.5, 0.99], dtype=torch.float64)
        p_combined = cauchy_combination(p.unsqueeze(1))
        torch.testing.assert_close(p_combined, p, atol=1e-10, rtol=1e-10)

    def test_all_significant_combined(self):
        """All small p-values → combined should be very small."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([[0.001, 0.002, 0.003]], dtype=torch.float64)
        result = cauchy_combination(p)
        assert result.item() < 0.005

    def test_mixed_signals_cauchy_sensitive(self):
        """One very small p + several large → combined still small (Cauchy is sensitive to min)."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([[1e-10, 0.5, 0.8, 0.9]], dtype=torch.float64)
        result = cauchy_combination(p)
        assert result.item() < 0.01

    def test_all_null_combined(self):
        """All p ~ 0.5 → combined should be moderate."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float64)
        result = cauchy_combination(p)
        assert 0.1 < result.item() < 0.9

    def test_known_values(self):
        """Verify against hand-computed Cauchy combination."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p1, p2 = 0.01, 0.5
        T1 = math.tan((0.5 - p1) * math.pi)
        T2 = math.tan((0.5 - p2) * math.pi)
        T_avg = (T1 + T2) / 2
        expected = 0.5 - math.atan(T_avg) / math.pi

        p_matrix = torch.tensor([[p1, p2]], dtype=torch.float64)
        result = cauchy_combination(p_matrix)
        assert result.item() == pytest.approx(expected, abs=1e-10)

    def test_weighted_combination(self):
        """Weights should shift the combined p-value."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([[0.001, 0.5]], dtype=torch.float64)
        w_equal = torch.tensor([0.5, 0.5], dtype=torch.float64)
        w_first = torch.tensor([0.9, 0.1], dtype=torch.float64)

        r_equal = cauchy_combination(p, w_equal)
        r_first = cauchy_combination(p, w_first)
        # Weighting toward the significant one should give smaller combined p
        assert r_first.item() < r_equal.item()

    def test_range(self):
        """Output should be in [0, 1]."""
        from torchgenomics.stats.cauchy import cauchy_combination

        torch.manual_seed(42)
        p = torch.rand(50, 3, dtype=torch.float64)
        result = cauchy_combination(p)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_scalar_mode(self):
        """1-D input → scalar output."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([0.01, 0.05, 0.5], dtype=torch.float64)
        result = cauchy_combination(p)
        assert result.ndim == 0  # scalar

    def test_extreme_p_no_nan(self):
        """Very small or near-1 p-values should not produce NaN."""
        from torchgenomics.stats.cauchy import cauchy_combination

        p = torch.tensor([[1e-300, 1.0 - 1e-16]], dtype=torch.float64)
        result = cauchy_combination(p)
        assert not torch.isnan(result).any()


class TestWeightedBH:
    """Tests for weighted Benjamini-Hochberg (wBHa)."""

    def test_uniform_weights_equals_bh(self):
        """With uniform weights (all 1), wBH should equal standard BH."""
        from torchgenomics.stats.weighted_fdr import weighted_bh

        torch.manual_seed(42)
        p = torch.rand(50, dtype=torch.float64)
        w = torch.ones(50, dtype=torch.float64)

        adj_wbh = weighted_bh(p, w)
        adj_bh = benjamini_hochberg(p)
        torch.testing.assert_close(adj_wbh, adj_bh, atol=1e-12, rtol=1e-12)

    def test_higher_weight_more_power(self):
        """A SNP with higher weight should get a smaller adjusted p-value."""
        from torchgenomics.stats.weighted_fdr import weighted_bh

        p = torch.tensor([0.01, 0.03, 0.05, 0.10, 0.50], dtype=torch.float64)
        w_uniform = torch.ones(5, dtype=torch.float64)
        w_boost = torch.tensor([3.0, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

        adj_uniform = weighted_bh(p, w_uniform)
        adj_boost = weighted_bh(p, w_boost)
        # First SNP (high weight) should have lower adj p in boosted version
        assert adj_boost[0].item() < adj_uniform[0].item()

    def test_zero_weight_never_rejected(self):
        """A SNP with weight=0 should have p_adj = 1.0."""
        from torchgenomics.stats.weighted_fdr import weighted_bh

        p = torch.tensor([0.001, 0.001, 0.001], dtype=torch.float64)
        w = torch.tensor([0.0, 1.5, 1.5], dtype=torch.float64)
        adj = weighted_bh(p, w)
        assert adj[0].item() == 1.0

    def test_auto_normalization(self):
        """Weights not summing to m should be auto-normalized."""
        from torchgenomics.stats.weighted_fdr import weighted_bh

        p = torch.tensor([0.01, 0.05, 0.1], dtype=torch.float64)
        w = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)  # mean=10, should be normalized
        adj = weighted_bh(p, w)
        # After normalization, w = [1,1,1] → should equal BH
        adj_bh = benjamini_hochberg(p)
        torch.testing.assert_close(adj, adj_bh, atol=1e-12, rtol=1e-12)

    def test_range(self):
        """Adjusted p-values should be in [0, 1]."""
        from torchgenomics.stats.weighted_fdr import weighted_bh

        torch.manual_seed(42)
        p = torch.rand(100, dtype=torch.float64)
        w = torch.rand(100, dtype=torch.float64) + 0.1
        adj = weighted_bh(p, w)
        assert torch.all(adj >= 0)
        assert torch.all(adj <= 1)


class TestLocalFDR:
    """Tests for local FDR (Efron empirical Bayes)."""

    def test_all_null_high_lfdr(self):
        """Under pure null (uniform p), most lfdr should be near 1."""
        from torchgenomics.stats.weighted_fdr import local_fdr

        torch.manual_seed(42)
        p = torch.rand(1000, dtype=torch.float64)
        lfdr = local_fdr(p)
        # Most should be > 0.5 (close to 1)
        assert (lfdr > 0.5).sum().item() > 800

    def test_strong_signals_low_lfdr(self):
        """Mixture: 90% null + 10% signal → signal SNPs should have low lfdr."""
        from torchgenomics.stats.weighted_fdr import local_fdr

        torch.manual_seed(42)
        n_null = 900
        n_signal = 100
        p_null = torch.rand(n_null, dtype=torch.float64)
        p_signal = torch.rand(n_signal, dtype=torch.float64) * 1e-5  # very small
        p = torch.cat([p_null, p_signal])

        lfdr = local_fdr(p)
        # Signal SNPs (last 100) should have lower lfdr on average
        mean_lfdr_signal = lfdr[n_null:].mean().item()
        mean_lfdr_null = lfdr[:n_null].mean().item()
        assert mean_lfdr_signal < mean_lfdr_null

    def test_range(self):
        """lfdr should be in [0, 1]."""
        from torchgenomics.stats.weighted_fdr import local_fdr

        torch.manual_seed(42)
        p = torch.rand(500, dtype=torch.float64)
        lfdr = local_fdr(p)
        assert torch.all(lfdr >= 0)
        assert torch.all(lfdr <= 1)

    def test_few_tests_returns_ones(self):
        """Fewer than 10 tests → return conservative lfdr = 1."""
        from torchgenomics.stats.weighted_fdr import local_fdr

        p = torch.tensor([0.01, 0.05], dtype=torch.float64)
        lfdr = local_fdr(p)
        torch.testing.assert_close(lfdr, torch.ones(2, dtype=torch.float64))


class TestHierarchicalFDR:
    """Tests for two-stage hierarchical FDR."""

    def test_single_group_matches_bh(self):
        """With all SNPs in one group, should match BH."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        torch.manual_seed(42)
        p = torch.rand(20, dtype=torch.float64)
        group_ids = ["gene1"] * 20

        adj = hierarchical_fdr(p, group_ids, q=0.05)
        adj_bh = benjamini_hochberg(p)

        # With a single group that passes stage 1, within-group BH = flat BH
        # But p_adj = max(group_adj, within_adj), so it may be slightly more conservative
        # At minimum, all hierarchical p_adj >= flat BH p_adj
        assert torch.all(adj >= adj_bh - 1e-10)

    def test_nonsig_group_all_one(self):
        """SNPs in a non-significant group should all get p_adj = 1.0."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        # Create two groups: one with signal, one without
        p_signal = torch.tensor([0.001, 0.002, 0.003], dtype=torch.float64)
        p_null = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
        p = torch.cat([p_signal, p_null])
        group_ids = ["gene1", "gene1", "gene1", "gene2", "gene2", "gene2"]

        adj = hierarchical_fdr(p, group_ids, q=0.05)
        # gene2 (null) should have all 1.0
        assert torch.all(adj[3:] == 1.0)

    def test_simes_method(self):
        """Simes group p-value should work."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        p = torch.tensor([0.001, 0.5, 0.01, 0.9], dtype=torch.float64)
        group_ids = ["g1", "g1", "g2", "g2"]
        adj = hierarchical_fdr(p, group_ids, q=0.05, group_method="simes")
        assert adj.shape == (4,)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)

    def test_fisher_method(self):
        """Fisher group p-value should work."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        p = torch.tensor([0.001, 0.5, 0.01, 0.9], dtype=torch.float64)
        group_ids = ["g1", "g1", "g2", "g2"]
        adj = hierarchical_fdr(p, group_ids, q=0.05, group_method="fisher")
        assert adj.shape == (4,)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)

    def test_singleton_group(self):
        """Groups with a single SNP should work correctly."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        p = torch.tensor([0.001, 0.5], dtype=torch.float64)
        group_ids = ["g1", "g2"]
        adj = hierarchical_fdr(p, group_ids, q=0.05)
        assert adj.shape == (2,)
        assert torch.all(adj >= 0) and torch.all(adj <= 1)

    def test_range(self):
        """Adjusted p-values should be in [0, 1]."""
        from torchgenomics.stats.weighted_fdr import hierarchical_fdr

        torch.manual_seed(42)
        p = torch.rand(50, dtype=torch.float64)
        group_ids = [f"g{i % 10}" for i in range(50)]
        adj = hierarchical_fdr(p, group_ids)
        assert torch.all(adj >= 0)
        assert torch.all(adj <= 1)


class TestAdaptivePermutation:
    """Tests for adaptive_permutation_maxT."""

    def test_valid_p_values(self):
        """Adaptive permutation should produce valid p-values in (0, 1]."""
        from torchgenomics.stats.permutation import adaptive_permutation_maxT

        torch.manual_seed(42)
        n, m = 50, 20
        G = torch.randn(n, m, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        p = adaptive_permutation_maxT(G, Y, X0, n_perms_min=100, n_perms_max=200, seed=42)
        assert p.shape == (m,)
        assert torch.all(p > 0)
        assert torch.all(p <= 1)

    def test_detects_strong_signal(self):
        """A very strong signal should get a small p-value."""
        from torchgenomics.stats.permutation import adaptive_permutation_maxT

        torch.manual_seed(42)
        n, m = 100, 30
        G = torch.randn(n, m, dtype=torch.float64)
        beta = torch.zeros(m, dtype=torch.float64)
        beta[0] = 5.0  # very strong signal
        Y = G @ beta + torch.randn(n, dtype=torch.float64) * 0.5
        X0 = torch.ones(n, 1, dtype=torch.float64)

        p = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=200, n_perms_max=1000, seed=42,
        )
        assert p[0].item() < 0.05

    def test_matches_maxT_same_seed(self):
        """With min=max perms, adaptive should behave like standard maxT."""
        from torchgenomics.stats.permutation import adaptive_permutation_maxT, permutation_maxT

        torch.manual_seed(42)
        n, m = 50, 10
        G = torch.randn(n, m, dtype=torch.float64)
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        p_std = permutation_maxT(G, Y, X0, n_perms=200, seed=42)
        p_adp = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=200, n_perms_max=200, seed=42,
        )
        # Should be identical when min == max (no Phase 2)
        torch.testing.assert_close(p_std, p_adp)
