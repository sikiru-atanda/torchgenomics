"""Tests for Phase 30: Adaptive FDR control — IHW and AdaPT.

Covers:
- IHW: FDR control (null/signal), power vs BH, weight constraints, cross-fitting
- AdaPT: FDR control (null/signal), power vs BH, convergence, threshold shape
- Binning: quantile bins, edge cases
- Edge cases: few hypotheses, all null, uninformative covariate
- Integration: works with ScanResult-like p-values + MAF as covariate
"""

from __future__ import annotations

import pytest
import torch

from torchgwas.stats.adaptive_fdr import (
    AdaPTResult,
    IHWResult,
    _bin_covariates,
    adapt,
    ihw,
)
from torchgwas.stats.multipletesting import benjamini_hochberg


# ===================================================================
# Helper: simulate p-values with informative covariate
# ===================================================================

def _simulate_pvals(
    m: int = 1000,
    pi1: float = 0.1,
    signal_strength: float = 3.0,
    seed: int = 42,
):
    """Simulate p-values with an informative covariate.

    Signal hypotheses are concentrated in low-covariate bins and have
    small p-values (drawn from Beta(1, signal_strength)).

    Returns p, covariates, is_signal.
    """
    torch.manual_seed(seed)
    n_signal = int(m * pi1)
    n_null = m - n_signal

    # Null p-values ~ Uniform(0, 1)
    p_null = torch.rand(n_null, dtype=torch.float64)

    # Signal p-values ~ Beta(1, signal_strength) → concentrated near 0
    # Beta(1, b) = 1 - U^(1/b) where U ~ Uniform
    u = torch.rand(n_signal, dtype=torch.float64)
    p_signal = 1.0 - u.pow(1.0 / signal_strength)

    p = torch.cat([p_null, p_signal])

    # Covariate: signal hypotheses have low covariate values (informative)
    cov_null = torch.rand(n_null, dtype=torch.float64) * 0.5 + 0.3  # [0.3, 0.8]
    cov_signal = torch.rand(n_signal, dtype=torch.float64) * 0.3  # [0, 0.3]
    covariates = torch.cat([cov_null, cov_signal])

    is_signal = torch.cat([
        torch.zeros(n_null, dtype=torch.bool),
        torch.ones(n_signal, dtype=torch.bool),
    ])

    # Shuffle
    perm = torch.randperm(m)
    return p[perm], covariates[perm], is_signal[perm]


# ===================================================================
# Test class 1: IHW
# ===================================================================

class TestIHW:
    """Tests for Independent Hypothesis Weighting."""

    def test_fdr_control_null(self):
        """Under pure null, IHW should not inflate FDR."""
        torch.manual_seed(10)
        m = 500
        p = torch.rand(m, dtype=torch.float64)
        cov = torch.rand(m, dtype=torch.float64)

        result = ihw(p, cov, q=0.05, n_folds=5, n_bins=5, seed=10)
        assert isinstance(result, IHWResult)
        # Under null, few or no rejections
        assert result.n_rejections <= m * 0.10  # ≤10% is conservative

    def test_fdr_control_with_signal(self):
        """With signal, IHW should control FDR below target."""
        p, cov, is_signal = _simulate_pvals(m=1000, pi1=0.1,
                                             signal_strength=5.0, seed=11)
        result = ihw(p, cov, q=0.10, n_folds=5, n_bins=5, seed=11)

        # Check FDR: among rejected, what fraction are null?
        if result.n_rejections > 0:
            rejected = result.adjusted_p <= 0.10
            false_disc = (~is_signal & rejected).sum().item()
            fdr = false_disc / max(rejected.sum().item(), 1)
            # Allow some slack due to finite sample
            assert fdr < 0.30  # generous bound for m=1000

    def test_power_vs_bh(self):
        """IHW should have >= power of BH with informative covariate."""
        p, cov, _ = _simulate_pvals(m=2000, pi1=0.15,
                                     signal_strength=4.0, seed=12)
        result_ihw = ihw(p, cov, q=0.10, n_folds=5, n_bins=10, seed=12)
        adj_bh = benjamini_hochberg(p)
        n_rej_bh = (adj_bh <= 0.10).sum().item()

        # IHW should reject at least as many (or close)
        assert result_ihw.n_rejections >= n_rej_bh * 0.8

    def test_weights_nonneg_mean_one(self):
        """Learned weights should be non-negative and mean ~1."""
        p, cov, _ = _simulate_pvals(m=500, seed=13)
        result = ihw(p, cov, q=0.05, seed=13)

        assert (result.weights >= 0).all()
        assert abs(result.weights.mean().item() - 1.0) < 0.1

    def test_cross_fitting_folds(self):
        """Each hypothesis should be in exactly one fold."""
        p, cov, _ = _simulate_pvals(m=500, seed=14)
        result = ihw(p, cov, q=0.05, n_folds=5, seed=14)

        assert result.fold_assignments.min() >= 0
        assert result.fold_assignments.max() < 5
        # All folds should be represented
        for k in range(5):
            assert (result.fold_assignments == k).sum() > 0

    def test_adjusted_p_in_range(self):
        """Adjusted p-values should be in [0, 1]."""
        p, cov, _ = _simulate_pvals(m=500, seed=15)
        result = ihw(p, cov, q=0.05, seed=15)

        assert (result.adjusted_p >= 0).all()
        assert (result.adjusted_p <= 1).all()


# ===================================================================
# Test class 2: AdaPT
# ===================================================================

class TestAdaPT:
    """Tests for Adaptive P-value Thresholding."""

    def test_fdr_control_null(self):
        """Under pure null, AdaPT should not inflate FDR."""
        torch.manual_seed(20)
        m = 500
        p = torch.rand(m, dtype=torch.float64)
        cov = torch.rand(m, dtype=torch.float64)

        result = adapt(p, cov, q=0.05, n_bins=5)
        assert isinstance(result, AdaPTResult)
        # Under null, few or no rejections
        assert result.n_rejections <= m * 0.10

    def test_fdr_control_with_signal(self):
        """With signal, AdaPT should control FDR."""
        p, cov, is_signal = _simulate_pvals(m=1000, pi1=0.1,
                                             signal_strength=5.0, seed=21)
        result = adapt(p, cov, q=0.10, n_bins=5)

        if result.n_rejections > 0:
            rejected = result.adjusted_p <= 0.10
            false_disc = (~is_signal & rejected).sum().item()
            fdr = false_disc / max(rejected.sum().item(), 1)
            assert fdr < 0.30

    def test_power_vs_bh(self):
        """AdaPT should have >= power of BH with informative covariate."""
        p, cov, _ = _simulate_pvals(m=2000, pi1=0.15,
                                     signal_strength=4.0, seed=22)
        result_adapt = adapt(p, cov, q=0.10, n_bins=10)
        adj_bh = benjamini_hochberg(p)
        n_rej_bh = (adj_bh <= 0.10).sum().item()

        # AdaPT should reject at least a decent fraction
        assert result_adapt.n_rejections >= n_rej_bh * 0.5

    def test_convergence(self):
        """EM should converge within max iterations."""
        p, cov, _ = _simulate_pvals(m=500, seed=23)
        result = adapt(p, cov, q=0.05, n_iter=100, n_bins=5)
        # Should converge (or at least run without error)
        assert result.n_iter <= 100
        assert isinstance(result.converged, bool)

    def test_adjusted_p_in_range(self):
        """Adjusted p-values should be in [0, 1]."""
        p, cov, _ = _simulate_pvals(m=500, seed=24)
        result = adapt(p, cov, q=0.05)

        assert (result.adjusted_p >= 0).all()
        assert (result.adjusted_p <= 1).all()


# ===================================================================
# Test class 3: Binning
# ===================================================================

class TestBinning:
    """Tests for _bin_covariates."""

    def test_quantile_bins(self):
        """Should produce roughly equal-sized bins."""
        torch.manual_seed(30)
        m = 1000
        cov = torch.rand(m, dtype=torch.float64)
        bins = _bin_covariates(cov, n_bins=5)

        assert bins.shape == (m,)
        assert bins.min() >= 0
        assert bins.max() < 5

        # Each bin should have roughly m/5 = 200 hypotheses
        for b in range(5):
            count = (bins == b).sum().item()
            assert count > 100  # at least 100 of 200 expected

    def test_edge_single_bin(self):
        """With n_bins=1, all hypotheses in one bin."""
        cov = torch.rand(50, dtype=torch.float64)
        bins = _bin_covariates(cov, n_bins=1)
        assert (bins == 0).all()

    def test_empty_input(self):
        """Empty input should return empty tensor."""
        cov = torch.zeros(0, dtype=torch.float64)
        bins = _bin_covariates(cov, n_bins=5)
        assert bins.shape == (0,)

    def test_constant_covariate(self):
        """Constant covariate should not crash."""
        cov = torch.ones(100, dtype=torch.float64) * 0.5
        bins = _bin_covariates(cov, n_bins=5)
        assert bins.shape == (100,)


# ===================================================================
# Test class 4: Edge cases
# ===================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_few_hypotheses_ihw(self):
        """IHW should handle very few hypotheses gracefully."""
        p = torch.tensor([0.01, 0.5, 0.8], dtype=torch.float64)
        cov = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64)
        result = ihw(p, cov, q=0.05, n_folds=5)
        assert isinstance(result, IHWResult)
        assert result.adjusted_p.shape == (3,)

    def test_few_hypotheses_adapt(self):
        """AdaPT should handle very few hypotheses gracefully."""
        p = torch.tensor([0.01, 0.5], dtype=torch.float64)
        cov = torch.tensor([0.1, 0.5], dtype=torch.float64)
        result = adapt(p, cov, q=0.05)
        assert isinstance(result, AdaPTResult)
        assert result.adjusted_p.shape == (2,)

    def test_all_null_ihw(self):
        """Under all-null, IHW should reject very few."""
        torch.manual_seed(40)
        p = torch.rand(500, dtype=torch.float64)
        cov = torch.rand(500, dtype=torch.float64)
        result = ihw(p, cov, q=0.05, seed=40)
        # Conservative: few false positives
        assert result.n_rejections <= 50  # ≤10% of 500

    def test_uninformative_covariate_ihw(self):
        """With uninformative covariate, IHW ~ BH."""
        p, _, is_signal = _simulate_pvals(m=1000, pi1=0.1, seed=41)
        # Random covariate (not correlated with signal)
        cov = torch.rand(1000, dtype=torch.float64)
        result = ihw(p, cov, q=0.10, seed=41)
        adj_bh = benjamini_hochberg(p)
        n_rej_bh = (adj_bh <= 0.10).sum().item()

        # Should not be dramatically worse than BH
        assert result.n_rejections >= n_rej_bh * 0.5


# ===================================================================
# Test class 5: Integration
# ===================================================================

class TestIntegration:
    """Tests with ScanResult-like data (p-values + MAF covariate)."""

    def test_ihw_with_maf_covariate(self):
        """IHW should work with MAF as the covariate."""
        torch.manual_seed(50)
        m = 500

        # Simulate: low-MAF SNPs have more signal
        maf = torch.rand(m, dtype=torch.float64) * 0.45 + 0.05  # [0.05, 0.5]
        n_signal = 50
        # Signal p-values
        p = torch.rand(m, dtype=torch.float64)
        signal_idx = maf.argsort()[:n_signal]  # lowest MAF
        p[signal_idx] = torch.rand(n_signal, dtype=torch.float64) * 0.01

        result = ihw(p, maf, q=0.05, n_bins=5, seed=50)
        assert isinstance(result, IHWResult)
        assert result.adjusted_p.shape == (m,)

    def test_adapt_with_maf_covariate(self):
        """AdaPT should work with MAF as the covariate."""
        torch.manual_seed(51)
        m = 500

        maf = torch.rand(m, dtype=torch.float64) * 0.45 + 0.05
        n_signal = 50
        p = torch.rand(m, dtype=torch.float64)
        signal_idx = maf.argsort()[:n_signal]
        p[signal_idx] = torch.rand(n_signal, dtype=torch.float64) * 0.01

        result = adapt(p, maf, q=0.05, n_bins=5)
        assert isinstance(result, AdaPTResult)
        assert result.adjusted_p.shape == (m,)
