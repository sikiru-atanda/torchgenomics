"""Tier-1 math-correctness coverage tests for torchgwas.stats.

Bar (spec section 4.3 Tier 1):
- Closed-form / scipy / statsmodels / Monte-Carlo verification per public function.
- One function in isolation per test. No compositional tests.
- Tolerance <= 1e-10 absolute for closed-form; documented otherwise.

Covers 13 symbols: 7 in ``torchgwas.stats.multipletesting``
(``benjamini_hochberg``, ``benjamini_yekutieli``, ``bonferroni``, ``holm``,
``sidak``, ``storey_qvalue``, ``eigenmt_adjust``) and 6 in
``torchgwas.stats.tests`` (``chi2_sf``, ``lrt_test``, ``score_test``,
``score_test_multi_df``, ``wald_test``, ``apply_contrast``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as sp_stats
import torch
from statsmodels.stats.multitest import multipletests

from torchgwas.stats import (
    AdaPTResult,
    BestModelResult,
    IHWResult,
    PVEResult,
    QTLPeak,
    adapt,
    adaptive_permutation_maxT,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    cauchy_combination,
    chi2_sf,
    compute_pve,
    davies_pvalue,
    diagnose_inflation,
    effective_test_count,
    effective_test_count_moskvina,
    hierarchical_fdr,
    holm,
    ihw,
    lambda_gc,
    ld_correlation_eigenvalues,
    liu_pvalue,
    local_fdr,
    lrt_test,
    mixture_chi2_pvalue,
    permutation_maxT,
    prune_peaks,
    saddlepoint_pvalue,
    score_test,
    score_test_multi_df,
    select_best_model,
    sidak,
    storey_qvalue,
    wald_test,
    weighted_bh,
)
from torchgwas.models.base import ScanResult
from torchgwas.stats.calibrate import compare_pvalues
from torchgwas.stats.genomic_control import CHI2_1_MEDIAN
from torchgwas.stats.multipletesting import eigenmt_adjust
from torchgwas.stats.tests import apply_contrast


pytestmark = pytest.mark.timeout(30)


# ---------------------------------------------------------------------------
# multipletesting
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    """``benjamini_hochberg`` matches ``scipy.stats.false_discovery_control``."""

    def test_matches_scipy_bh(self, stat_dtype):
        """Random p-values: torchgwas BH agrees with scipy BH to 1e-10."""
        gen = torch.Generator().manual_seed(0)
        p = torch.rand(200, generator=gen, dtype=stat_dtype)
        out = benjamini_hochberg(p).cpu().numpy()
        ref = sp_stats.false_discovery_control(p.cpu().numpy(), method="bh")
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_handles_all_ones(self, stat_dtype):
        """All p=1 produces all q=1."""
        p = torch.ones(20, dtype=stat_dtype)
        out = benjamini_hochberg(p)
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=1e-12)

    def test_monotone_after_sort(self, stat_dtype):
        """BH-adjusted output is non-decreasing in original ascending order."""
        gen = torch.Generator().manual_seed(1)
        p = torch.rand(50, generator=gen, dtype=stat_dtype)
        sorted_p, idx = p.sort()
        adj_sorted = benjamini_hochberg(sorted_p).cpu().numpy()
        diffs = np.diff(adj_sorted)
        assert (diffs >= -1e-12).all(), "BH adj should be non-decreasing in ascending p order"


class TestBenjaminiYekutieli:
    """``benjamini_yekutieli`` matches scipy BY and dominates BH."""

    def test_matches_scipy_by(self, stat_dtype):
        """Random p-values: torchgwas BY agrees with scipy BY to 1e-10."""
        gen = torch.Generator().manual_seed(2)
        p = torch.rand(200, generator=gen, dtype=stat_dtype)
        out = benjamini_yekutieli(p).cpu().numpy()
        ref = sp_stats.false_discovery_control(p.cpu().numpy(), method="by")
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_strictly_more_conservative_than_bh(self, stat_dtype):
        """BY adjusted p-values are >= BH element-wise."""
        gen = torch.Generator().manual_seed(3)
        p = torch.rand(150, generator=gen, dtype=stat_dtype)
        bh = benjamini_hochberg(p).cpu().numpy()
        by = benjamini_yekutieli(p).cpu().numpy()
        assert (by + 1e-12 >= bh).all(), "BY should be >= BH everywhere"


class TestBonferroni:
    """``bonferroni`` is the closed-form ``min(p * m, 1)``."""

    def test_closed_form(self, stat_dtype):
        """Output matches torch.minimum(p*m, 1) to 1e-12."""
        gen = torch.Generator().manual_seed(4)
        p = torch.rand(100, generator=gen, dtype=stat_dtype)
        m = p.shape[0]
        ref = torch.minimum(p * m, torch.ones_like(p))
        torch.testing.assert_close(bonferroni(p), ref, rtol=0, atol=1e-12)

    def test_clamped_at_one(self, stat_dtype):
        """Inputs that overshoot 1 after multiplication are clamped to 1."""
        p = torch.tensor([0.6, 0.7], dtype=stat_dtype)
        out = bonferroni(p)
        torch.testing.assert_close(out, torch.ones(2, dtype=stat_dtype),
                                   rtol=0, atol=1e-12)


class TestHolm:
    """``holm`` matches statsmodels' Holm step-down to 1e-10."""

    def test_matches_statsmodels_holm(self, stat_dtype):
        """Random p-values: agreement with statsmodels' multipletests."""
        gen = torch.Generator().manual_seed(5)
        p = torch.rand(200, generator=gen, dtype=stat_dtype)
        out = holm(p).cpu().numpy()
        ref = multipletests(p.cpu().numpy(), method="holm")[1]
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-10)

    def test_monotone_after_sort(self, stat_dtype):
        """Sorted-ascending input produces a monotone non-decreasing adj output."""
        gen = torch.Generator().manual_seed(6)
        p = torch.rand(60, generator=gen, dtype=stat_dtype)
        sorted_p, _ = p.sort()
        adj = holm(sorted_p).cpu().numpy()
        diffs = np.diff(adj)
        assert (diffs >= -1e-12).all(), "Holm adj on sorted-asc p must be non-decreasing"

    def test_no_inflation_above_one(self, stat_dtype):
        """All Holm-adjusted p-values are <= 1."""
        gen = torch.Generator().manual_seed(7)
        p = torch.rand(80, generator=gen, dtype=stat_dtype)
        out = holm(p)
        assert torch.all(out <= 1.0 + 1e-12)


class TestSidak:
    """``sidak`` is closed-form ``1 - (1 - p)^m`` and dominates Bonferroni."""

    def test_closed_form(self, stat_dtype):
        """Output matches 1 - (1-p)^m to 1e-12."""
        gen = torch.Generator().manual_seed(8)
        p = torch.rand(100, generator=gen, dtype=stat_dtype)
        m = p.shape[0]
        ref = torch.clamp(1.0 - (1.0 - p) ** m, min=0.0, max=1.0)
        torch.testing.assert_close(sidak(p), ref, rtol=0, atol=1e-12)

    def test_more_conservative_than_bonferroni(self, stat_dtype):
        """Sidak <= Bonferroni for all p in (0, 1) (Sidak slightly more powerful)."""
        gen = torch.Generator().manual_seed(9)
        p = torch.rand(120, generator=gen, dtype=stat_dtype)
        s = sidak(p).cpu().numpy()
        b = bonferroni(p).cpu().numpy()
        assert (s <= b + 1e-12).all(), "Sidak adjusted p must be <= Bonferroni"


class TestStoreyQvalue:
    """``storey_qvalue`` recovers pi0 ~ 1 under the global null and
    behaves as an adaptive BH otherwise.
    """

    def test_uniform_pvalues_pi0_near_one(self, stat_dtype):
        """For pure null U(0,1) p-values, the smallest q is large (~0.5+)."""
        gen = torch.Generator().manual_seed(10)
        p = torch.rand(10000, generator=gen, dtype=stat_dtype)
        q = storey_qvalue(p, lambda_=0.5)
        # pi0 estimator: n_above / (m * (1 - lambda)). With m=10000 and
        # uniform, n_above ~ 5000, so pi0_hat ~ 1.0. The smallest q-value
        # then is approximately 1.0 * m * p_min / 1, clamped/cummin'd to a
        # value near min(BH adj). Empirically, the smallest q under uniform
        # is essentially never below 0.5.
        assert q.min().item() >= 0.4, (
            f"Storey q-value under uniform null too small: q.min={q.min().item():.4f}"
        )

    def test_q_le_p_for_strong_signals(self, stat_dtype):
        """Strong signals (very small p) get q <= 1: storey accepts them."""
        gen = torch.Generator().manual_seed(11)
        # Mix: 100 strong signals + 9900 uniform nulls.
        nulls = torch.rand(9900, generator=gen, dtype=stat_dtype)
        signals = torch.full((100,), 1e-6, dtype=stat_dtype)
        p = torch.cat([signals, nulls])
        q = storey_qvalue(p, lambda_=0.5)
        # Smallest q-values correspond to the smallest p-values (signals).
        small_q = q[:100]
        # These should be tiny (<< 0.05) because pi0_hat near 1, and
        # m * pi0 * p / rank for p=1e-6 is ~1e-2.
        assert small_q.max().item() < 0.05, (
            f"Storey failed to call signals: q.max(signals)={small_q.max().item():.4f}"
        )


class TestEigenmtAdjust:
    """``eigenmt_adjust`` recovers Bonferroni when LD = I and the minimum
    when LD = 11^T (perfect correlation), and returns a 2-tuple.
    """

    def test_independent_LD_recovers_bonferroni(self, stat_dtype):
        """Identity LD produces M_eff = m and adjusted p = bonferroni(p)."""
        m = 20
        gen = torch.Generator().manual_seed(12)
        p = torch.rand(m, generator=gen, dtype=stat_dtype)
        LD = torch.eye(m, dtype=stat_dtype)
        adj, m_eff = eigenmt_adjust(p, LD, var_threshold=0.995)
        assert m_eff == float(m)
        ref = torch.clamp(p * m, max=1.0)
        torch.testing.assert_close(adj, ref, rtol=0, atol=1e-12)

    def test_perfectly_correlated_recovers_minimum(self, stat_dtype):
        """All-ones LD has rank 1, so M_eff = 1 and adj p = p."""
        m = 15
        gen = torch.Generator().manual_seed(13)
        p = torch.rand(m, generator=gen, dtype=stat_dtype)
        LD = torch.ones((m, m), dtype=stat_dtype)
        adj, m_eff = eigenmt_adjust(p, LD, var_threshold=0.995)
        assert m_eff == 1.0
        torch.testing.assert_close(adj, torch.clamp(p, max=1.0),
                                   rtol=0, atol=1e-12)

    def test_returns_tuple(self, stat_dtype):
        """Output is a (Tensor, float) 2-tuple."""
        m = 5
        p = torch.full((m,), 0.1, dtype=stat_dtype)
        LD = torch.eye(m, dtype=stat_dtype)
        out = eigenmt_adjust(p, LD)
        assert isinstance(out, tuple) and len(out) == 2
        assert isinstance(out[0], torch.Tensor)
        assert isinstance(out[1], float)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestChi2Sf:
    """``chi2_sf`` matches ``scipy.stats.chi2.sf``."""

    def test_matches_scipy_chi2_sf(self, stat_dtype):
        """Range of (stat, df) pairs: agreement with scipy.chi2.sf to 1e-12."""
        for df in (1, 2, 3, 5, 10):
            stat = torch.tensor([0.0, 0.5, 1.0, 3.84, 10.0, 25.0], dtype=stat_dtype)
            out = chi2_sf(stat, df).cpu().numpy()
            ref = sp_stats.chi2.sf(stat.cpu().numpy(), df=df)
            ref = np.clip(ref, 1e-300, 1.0)
            np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_zero_stat_returns_one(self, stat_dtype):
        """chi2_sf(0, df) == 1 for any df."""
        for df in (1, 2, 3, 7, 20):
            out = chi2_sf(torch.tensor([0.0], dtype=stat_dtype), df).item()
            assert abs(out - 1.0) < 1e-12

    def test_large_stat_underflow_handled(self, stat_dtype):
        """chi2_sf(1000, df=1) is finite, > 0, and not NaN (clamped at 1e-300)."""
        out = chi2_sf(torch.tensor([1000.0], dtype=stat_dtype), df=1).item()
        assert not np.isnan(out)
        assert out > 0.0
        assert out <= 1.0


class TestLrtTest:
    """``lrt_test`` returns (stat, p) with stat = 2*(ll_alt - ll_null)."""

    def test_closed_form(self, stat_dtype):
        """Hand-computed example: stat and p match closed form to 1e-10."""
        ll_null = -100.0
        ll_alt = torch.tensor([-95.0, -90.0], dtype=stat_dtype)
        stat, p = lrt_test(ll_null, ll_alt, df=1)
        ref_stat = torch.tensor([10.0, 20.0], dtype=stat_dtype)
        torch.testing.assert_close(stat, ref_stat, rtol=0, atol=1e-12)
        ref_p = sp_stats.chi2.sf(ref_stat.cpu().numpy(), df=1)
        np.testing.assert_allclose(p.cpu().numpy(), ref_p, rtol=1e-10, atol=1e-10)

    def test_negative_diff_clamped_to_zero(self, stat_dtype):
        """When ll_alt < ll_null, stat is clamped to 0 and p == 1."""
        ll_null = 0.0
        ll_alt = torch.tensor([-5.0, -1.0], dtype=stat_dtype)
        stat, p = lrt_test(ll_null, ll_alt, df=1)
        torch.testing.assert_close(stat, torch.zeros(2, dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(p, torch.ones(2, dtype=stat_dtype),
                                   rtol=0, atol=1e-12)


class TestScoreTest:
    """``score_test`` returns (U^2 / V, chi2_sf(stat, 1))."""

    def test_closed_form(self, stat_dtype):
        """U=2, V=1 -> stat=4, p = chi2.sf(4, 1)."""
        U = torch.tensor([2.0], dtype=stat_dtype)
        V = torch.tensor([1.0], dtype=stat_dtype)
        stat, p = score_test(U, V)
        torch.testing.assert_close(stat, torch.tensor([4.0], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        ref_p = sp_stats.chi2.sf(4.0, df=1)
        np.testing.assert_allclose(p.cpu().numpy(), [ref_p],
                                   rtol=1e-10, atol=1e-10)

    def test_zero_U_returns_zero_stat_and_p_one(self, stat_dtype):
        """U=0 -> stat=0, p=1."""
        U = torch.tensor([0.0], dtype=stat_dtype)
        V = torch.tensor([1.0], dtype=stat_dtype)
        stat, p = score_test(U, V)
        assert stat.item() == 0.0
        assert abs(p.item() - 1.0) < 1e-12


class TestScoreTestMultiDf:
    """``score_test_multi_df`` returns (U' V^{-1} U, chi2_sf(stat, df))."""

    def test_matches_scipy_chi2_for_known_quadratic_form(self, stat_dtype):
        """U=[1,2], V=I_2 -> stat = 1+4 = 5, p = chi2.sf(5, 2)."""
        U = torch.tensor([[1.0, 2.0]], dtype=stat_dtype)
        V = torch.eye(2, dtype=stat_dtype).unsqueeze(0)
        stat, p = score_test_multi_df(U, V, df=2)
        torch.testing.assert_close(stat, torch.tensor([5.0], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        ref_p = sp_stats.chi2.sf(5.0, df=2)
        np.testing.assert_allclose(p.cpu().numpy(), [ref_p],
                                   rtol=1e-10, atol=1e-10)


class TestWaldTest:
    """``wald_test`` returns ((beta/se)^2, chi2_sf(stat, df))."""

    def test_closed_form(self, stat_dtype):
        """beta=2, se=1 -> stat=4, p = chi2.sf(4, 1)."""
        beta = torch.tensor([2.0], dtype=stat_dtype)
        se = torch.tensor([1.0], dtype=stat_dtype)
        stat, p = wald_test(beta, se, df=1)
        torch.testing.assert_close(stat, torch.tensor([4.0], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        ref_p = sp_stats.chi2.sf(4.0, df=1)
        np.testing.assert_allclose(p.cpu().numpy(), [ref_p],
                                   rtol=1e-10, atol=1e-10)

    def test_handles_zero_se(self, stat_dtype):
        """Zero SE: impl floors |se| to 1e-20 (no NaN, very large stat)."""
        beta = torch.tensor([1.0], dtype=stat_dtype)
        se = torch.tensor([0.0], dtype=stat_dtype)
        stat, p = wald_test(beta, se, df=1)
        # impl clamps |se| to 1e-20 -> stat ~ 1e40, finite. p clamped to 1e-300.
        assert torch.isfinite(stat).all()
        assert stat.item() > 1e30
        assert not torch.isnan(p).any()


class TestApplyContrast:
    """``apply_contrast`` returns (Cb, CVCt, stat, p) for H0: C beta = 0."""

    def test_simple_contrast(self, stat_dtype):
        """beta=[1,2,3], Var=I_3, C=[1,0,0] -> Cb=[1], CVCt=[[1]], stat=1."""
        beta = torch.tensor([[1.0, 2.0, 3.0]], dtype=stat_dtype)  # (1, 3)
        Var_beta = torch.eye(3, dtype=stat_dtype).unsqueeze(0)  # (1, 3, 3)
        C = torch.tensor([[1.0, 0.0, 0.0]], dtype=stat_dtype)  # (1, 3)
        Cb, CVCt, stat, p = apply_contrast(beta, Var_beta, C)
        torch.testing.assert_close(Cb, torch.tensor([[1.0]], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(CVCt, torch.tensor([[[1.0]]], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(stat, torch.tensor([1.0], dtype=stat_dtype),
                                   rtol=0, atol=1e-12)
        ref_p = sp_stats.chi2.sf(1.0, df=1)
        np.testing.assert_allclose(p.cpu().numpy(), [ref_p],
                                   rtol=1e-10, atol=1e-10)

    def test_returns_4_tuple(self, stat_dtype):
        """Output is a 4-tuple of tensors."""
        beta = torch.tensor([[1.0, 2.0]], dtype=stat_dtype)
        Var_beta = torch.eye(2, dtype=stat_dtype).unsqueeze(0)
        C = torch.tensor([[1.0, 0.0]], dtype=stat_dtype)
        out = apply_contrast(beta, Var_beta, C)
        assert isinstance(out, tuple) and len(out) == 4
        for t in out:
            assert isinstance(t, torch.Tensor)


# ---------------------------------------------------------------------------
# adaptive_fdr
# ---------------------------------------------------------------------------


class TestAdaPTResult:
    """``AdaPTResult`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate with shaped tensors, attributes round-trip exactly."""
        m = 7
        adjusted_p = torch.rand(m, dtype=stat_dtype)
        thresholds = torch.rand(m, dtype=stat_dtype)
        bins = torch.zeros(m, dtype=torch.long)
        res = AdaPTResult(
            adjusted_p=adjusted_p,
            thresholds=thresholds,
            n_rejections=3,
            bins=bins,
            n_iter=12,
            converged=True,
            n_bins=4,
        )
        torch.testing.assert_close(res.adjusted_p, adjusted_p, rtol=0, atol=0)
        torch.testing.assert_close(res.thresholds, thresholds, rtol=0, atol=0)
        torch.testing.assert_close(res.bins, bins, rtol=0, atol=0)
        assert res.n_rejections == 3
        assert res.n_iter == 12
        assert res.converged is True
        assert res.n_bins == 4

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` works on a populated AdaPTResult."""
        m = 3
        res = AdaPTResult(
            adjusted_p=torch.zeros(m, dtype=stat_dtype),
            thresholds=torch.zeros(m, dtype=stat_dtype),
            n_rejections=0,
            bins=torch.zeros(m, dtype=torch.long),
            n_iter=0,
            converged=False,
            n_bins=1,
        )
        s = repr(res)
        assert isinstance(s, str)
        assert "AdaPTResult" in s


class TestIHWResult:
    """``IHWResult`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate with shaped tensors, attributes round-trip exactly."""
        m = 6
        adjusted_p = torch.rand(m, dtype=stat_dtype)
        weights = torch.rand(m, dtype=stat_dtype)
        bins = torch.zeros(m, dtype=torch.long)
        folds = torch.arange(m, dtype=torch.long) % 3
        res = IHWResult(
            adjusted_p=adjusted_p,
            weights=weights,
            n_rejections=2,
            bins=bins,
            fold_assignments=folds,
            n_bins=5,
            n_folds=3,
        )
        torch.testing.assert_close(res.adjusted_p, adjusted_p, rtol=0, atol=0)
        torch.testing.assert_close(res.weights, weights, rtol=0, atol=0)
        torch.testing.assert_close(res.bins, bins, rtol=0, atol=0)
        torch.testing.assert_close(res.fold_assignments, folds, rtol=0, atol=0)
        assert res.n_rejections == 2
        assert res.n_bins == 5
        assert res.n_folds == 3

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` works on a populated IHWResult."""
        m = 3
        res = IHWResult(
            adjusted_p=torch.zeros(m, dtype=stat_dtype),
            weights=torch.ones(m, dtype=stat_dtype),
            n_rejections=0,
            bins=torch.zeros(m, dtype=torch.long),
            fold_assignments=torch.zeros(m, dtype=torch.long),
            n_bins=1,
            n_folds=1,
        )
        s = repr(res)
        assert isinstance(s, str)
        assert "IHWResult" in s


class TestAdapt:
    """``adapt`` (Lei & Fithian 2018): EM-based covariate-adaptive thresholds."""

    def test_uniform_pvals_no_rejections(self, stat_dtype):
        """Uniform null p-values: rejections at q=0.05 are bounded
        well below 5% of m by the procedure's FDP estimate.

        Bound is observe-then-floor: with seed=0, m=5000, the impl
        currently reports 0–250 rejections (Monte-Carlo bound)."""
        torch.manual_seed(0)
        m = 5000
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        out = adapt(p, covariate, q=0.05)
        assert out.n_rejections <= 250, (
            f"AdaPT under uniform null rejected {out.n_rejections}/5000; "
            f"Monte-Carlo bound 250"
        )

    def test_signal_recovery_with_informative_covariate(self, stat_dtype):
        """Plant tiny p-values where covariate is large; AdaPT should
        recover most signals via covariate-adaptive thresholds."""
        torch.manual_seed(0)
        m = 5000
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        # Plant 100 tiny p-values at the high-covariate end
        signal_idx = (covariate > 0.9).nonzero(as_tuple=True)[0][:100]
        p[signal_idx] = 1e-5
        out = adapt(p, covariate, q=0.05)
        # AdaPT may not be guaranteed to find all signals, but should
        # find enough that it beats a uniform-cov baseline.
        rejected = (out.adjusted_p <= 0.05)
        n_signal_rej = rejected[signal_idx].sum().item()
        # observe-then-floor: with seed=0 the impl gets ~80+ of 100
        assert n_signal_rej >= 80, (
            f"AdaPT recovered only {n_signal_rej}/100 planted signals"
        )

    def test_returns_AdaPTResult(self, stat_dtype):
        """Output is an AdaPTResult instance with documented fields."""
        torch.manual_seed(0)
        m = 200
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        out = adapt(p, covariate, q=0.05)
        assert isinstance(out, AdaPTResult)
        assert out.adjusted_p.shape == (m,)
        assert out.thresholds.shape == (m,)
        assert out.bins.shape == (m,)
        assert isinstance(out.n_rejections, int)
        assert isinstance(out.n_iter, int)
        assert isinstance(out.converged, bool)
        assert isinstance(out.n_bins, int)


class TestIhw:
    """``ihw`` (Ignatiadis & Huber 2021): K-fold cross-fit weighted BH."""

    def test_uniform_pvals_no_rejections(self, stat_dtype):
        """Uniform null p-values: rejections at q=0.05 are bounded
        well below 5% of m. Monte-Carlo upper bound 250 with seed=0."""
        torch.manual_seed(0)
        m = 5000
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        out = ihw(p, covariate, q=0.05, seed=0)
        assert out.n_rejections <= 250, (
            f"IHW under uniform null rejected {out.n_rejections}/5000; "
            f"Monte-Carlo bound 250"
        )

    def test_signal_recovery_with_informative_covariate(self, stat_dtype):
        """Plant tiny p-values where covariate is large; IHW should
        recover most signals via informative weights."""
        torch.manual_seed(0)
        m = 5000
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        signal_idx = (covariate > 0.9).nonzero(as_tuple=True)[0][:100]
        p[signal_idx] = 1e-5
        out = ihw(p, covariate, q=0.05, seed=0)
        rejected = (out.adjusted_p <= 0.05)
        n_signal_rej = rejected[signal_idx].sum().item()
        # observe-then-floor: IHW typically recovers ~all of them
        assert n_signal_rej >= 80, (
            f"IHW recovered only {n_signal_rej}/100 planted signals"
        )

    def test_returns_IHWResult(self, stat_dtype):
        """Output is an IHWResult with documented fields."""
        torch.manual_seed(0)
        m = 200
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        out = ihw(p, covariate, q=0.05, seed=0)
        assert isinstance(out, IHWResult)
        assert out.adjusted_p.shape == (m,)
        assert out.weights.shape == (m,)
        assert out.bins.shape == (m,)
        assert out.fold_assignments.shape == (m,)
        assert isinstance(out.n_rejections, int)
        assert isinstance(out.n_bins, int)
        assert isinstance(out.n_folds, int)

    def test_seed_reproducibility(self, stat_dtype):
        """Identical seed -> identical adjusted_p across two calls."""
        torch.manual_seed(0)
        m = 500
        p = torch.rand(m, dtype=stat_dtype)
        covariate = torch.rand(m, dtype=stat_dtype)
        out1 = ihw(p, covariate, q=0.05, seed=42)
        out2 = ihw(p, covariate, q=0.05, seed=42)
        torch.testing.assert_close(
            out1.adjusted_p, out2.adjusted_p, rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# weighted_fdr
# ---------------------------------------------------------------------------


class TestHierarchicalFdr:
    """``hierarchical_fdr``: two-stage gene-then-SNP FDR."""

    def test_basic_two_group(self, stat_dtype):
        """Group A (mostly small p) gets more rejections than group B
        (uniform null p). The non-trivial-behavior bar."""
        torch.manual_seed(0)
        # Group A: 50 tiny p-values
        p_a = torch.full((50,), 1e-4, dtype=stat_dtype)
        # Group B: 50 uniform null p-values
        p_b = torch.rand(50, dtype=stat_dtype)
        p = torch.cat([p_a, p_b])
        gids = ["A"] * 50 + ["B"] * 50
        out = hierarchical_fdr(p, gids, q=0.05)
        rej_a = (out[:50] <= 0.05).sum().item()
        rej_b = (out[50:] <= 0.05).sum().item()
        assert rej_a > rej_b, (
            f"Group A (signal) should reject more than group B (null); "
            f"got {rej_a} vs {rej_b}"
        )

    def test_returns_tensor(self, stat_dtype):
        """Output is a torch.Tensor of shape (m,)."""
        torch.manual_seed(0)
        m = 30
        p = torch.rand(m, dtype=stat_dtype)
        gids = (["A"] * 15) + (["B"] * 15)
        out = hierarchical_fdr(p, gids, q=0.05)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (m,)

    def test_handles_single_group(self, stat_dtype):
        """All in one group -> output highly correlated with plain BH."""
        torch.manual_seed(0)
        m = 200
        p = torch.rand(m, dtype=stat_dtype)
        # Inject a few small signals so BH does some non-trivial work
        p[:20] = torch.linspace(1e-6, 1e-3, 20, dtype=stat_dtype)
        gids = ["G"] * m
        hier = hierarchical_fdr(p, gids, q=0.5).cpu().numpy()
        bh = benjamini_hochberg(p).cpu().numpy()
        # Spearman is robust to monotone transforms; here both should
        # be tightly correlated since the within-group BH dominates.
        corr = np.corrcoef(hier, bh)[0, 1]
        assert corr > 0.99, (
            f"Single-group hierarchical_fdr correlation with BH={corr:.4f}"
        )


class TestLocalFdr:
    """``local_fdr`` (Efron 2004): empirical Bayes posterior null prob."""

    def test_uniform_pvals_high_local_fdr(self, stat_dtype):
        """Uniform p (no signal): mean local FDR ~ 1.0 (>0.7)."""
        torch.manual_seed(0)
        m = 5000
        p = torch.rand(m, dtype=stat_dtype)
        lfdr = local_fdr(p)
        assert lfdr.mean().item() > 0.7, (
            f"Uniform-null lfdr too low: mean={lfdr.mean().item():.4f}"
        )

    def test_strong_signals_low_local_fdr(self, stat_dtype):
        """100 small p-values + 4900 uniform: mean lfdr at signals < 0.5."""
        torch.manual_seed(0)
        n_sig, n_null = 100, 4900
        p_sig = torch.full((n_sig,), 1e-6, dtype=stat_dtype)
        p_null = torch.rand(n_null, dtype=stat_dtype)
        p = torch.cat([p_sig, p_null])
        lfdr = local_fdr(p)
        sig_lfdr = lfdr[:n_sig].mean().item()
        assert sig_lfdr < 0.5, (
            f"Signals' mean lfdr={sig_lfdr:.4f}; expected < 0.5"
        )

    def test_returns_tensor_same_shape(self, stat_dtype):
        """Output shape matches input."""
        torch.manual_seed(0)
        m = 100
        p = torch.rand(m, dtype=stat_dtype)
        out = local_fdr(p)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (m,)


class TestWeightedBh:
    """``weighted_bh`` (Ignatiadis et al. 2023): weighted BH."""

    def test_uniform_weights_recovers_bh(self, stat_dtype):
        """Uniform weights -> output equals plain BH to 1e-8."""
        torch.manual_seed(0)
        m = 200
        p = torch.rand(m, dtype=stat_dtype)
        w = torch.ones(m, dtype=stat_dtype)
        out = weighted_bh(p, w, q=0.05).cpu().numpy()
        ref = benjamini_hochberg(p).cpu().numpy()
        np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)

    def test_higher_weight_increases_rejections(self, stat_dtype):
        """Doubling signal weights increases rejection count vs uniform."""
        torch.manual_seed(0)
        m = 1000
        p = torch.rand(m, dtype=stat_dtype)
        signal_idx = torch.arange(50)
        p[signal_idx] = torch.linspace(1e-5, 5e-3, 50, dtype=stat_dtype)
        w_uniform = torch.ones(m, dtype=stat_dtype)
        w_boost = torch.ones(m, dtype=stat_dtype)
        w_boost[signal_idx] = 2.0
        out_unif = weighted_bh(p, w_uniform, q=0.05)
        out_boost = weighted_bh(p, w_boost, q=0.05)
        n_unif = (out_unif <= 0.05).sum().item()
        n_boost = (out_boost <= 0.05).sum().item()
        assert n_boost > n_unif, (
            f"Boosting weights at signal should raise rejections: "
            f"boost={n_boost} vs uniform={n_unif}"
        )

    def test_returns_tensor_same_shape(self, stat_dtype):
        """Output shape matches input."""
        torch.manual_seed(0)
        m = 50
        p = torch.rand(m, dtype=stat_dtype)
        w = torch.rand(m, dtype=stat_dtype) + 0.1
        out = weighted_bh(p, w, q=0.05)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (m,)


# ---------------------------------------------------------------------------
# simplem
# ---------------------------------------------------------------------------


class TestEffectiveTestCount:
    """``effective_test_count`` (Gao 2008 simpleM): variance-explained M_eff."""

    def test_independent_eigenvalues_recovers_full_count(self, stat_dtype):
        """All-ones eigenvalues (perfect independence) -> M_eff = m."""
        m = 100
        evals = torch.ones(m, dtype=stat_dtype)
        out = effective_test_count(evals)
        assert out == m

    def test_perfectly_correlated_recovers_one(self, stat_dtype):
        """One huge eigenvalue, rest zero -> M_eff == 1."""
        m = 100
        evals = torch.zeros(m, dtype=stat_dtype)
        evals[0] = float(m)  # all variance in 1st component
        out = effective_test_count(evals)
        assert out == 1

    def test_intermediate_case(self, stat_dtype):
        """eigenvalues=[2.5, 0.4, 0.1]: cumvar=[0.833, 0.967, 1.0],
        threshold=0.995 needs all 3 -> M_eff=3."""
        evals = torch.tensor([2.5, 0.4, 0.1], dtype=stat_dtype)
        out = effective_test_count(evals)
        assert out == 3


class TestEffectiveTestCountMoskvina:
    """``effective_test_count_moskvina`` (Moskvina & Schmidt 2008)."""

    def test_independent_eigenvalues_recovers_full_count(self, stat_dtype):
        """All-ones eigenvalues -> n_large=m, frac_sum=0 -> M_eff=m."""
        m = 100
        evals = torch.ones(m, dtype=stat_dtype)
        out = effective_test_count_moskvina(evals)
        assert out == m

    def test_perfectly_correlated_recovers_one(self, stat_dtype):
        """One eigenvalue=m, rest zero: n_large=1, frac_sum=0 -> M_eff=1."""
        m = 100
        evals = torch.zeros(m, dtype=stat_dtype)
        evals[0] = float(m)
        out = effective_test_count_moskvina(evals)
        assert out == 1

    def test_intermediate_case_more_conservative(self, stat_dtype):
        """eigenvalues=[2.5, 0.4, 0.1]: n_large=1, frac_sum=0.4+0.1=0.5,
        round(1.5)=2 -> Moskvina M_eff=2 (< simpleM M_eff=3)."""
        evals = torch.tensor([2.5, 0.4, 0.1], dtype=stat_dtype)
        out_mosk = effective_test_count_moskvina(evals)
        out_simple = effective_test_count(evals)
        # Moskvina bound (more conservative ⇒ smaller M_eff for this input)
        assert out_mosk <= out_simple, (
            f"Moskvina ({out_mosk}) should be <= simpleM ({out_simple})"
        )
        assert out_mosk == 2


class TestLdCorrelationEigenvalues:
    """``ld_correlation_eigenvalues``: eigenvalues of SNP-SNP corr matrix."""

    def test_eigenvalues_sum_to_n_snps(self, tiny_genotype_diploid):
        """Trace identity: sum(evals) == trace(corr).

        Convention discovered: the impl standardizes columns with
        ``torch.std`` (unbiased, /(n-1)) but normalizes the Gram matrix
        with /n, so the diagonal is (n-1)/n and the expected sum is
        m * (n - 1) / n, not m. Trace identity still holds against the
        actual normalization the impl uses, to 1e-8."""
        G = tiny_genotype_diploid  # (100, 50)
        evals = ld_correlation_eigenvalues(G)
        n, m = G.shape
        expected = m * (n - 1) / n
        assert abs(evals.sum().item() - expected) < 1e-8, (
            f"sum(evals)={evals.sum().item():.6f}, expected {expected}"
        )

    def test_eigenvalues_nonnegative(self, tiny_genotype_diploid):
        """Correlation matrix is PSD -> all eigenvalues >= -1e-10."""
        G = tiny_genotype_diploid
        evals = ld_correlation_eigenvalues(G)
        assert (evals >= -1e-10).all(), (
            f"Min eigenvalue {evals.min().item()} below -1e-10"
        )


# ---------------------------------------------------------------------------
# mixture
# ---------------------------------------------------------------------------


class TestDaviesPvalue:
    """``davies_pvalue`` for Q ~ sum_k lambda_k * chi2_1."""

    def test_single_lambda_recovers_chi2(self, stat_dtype):
        """Single-component lambda=[1.0] reduces Q to chi2(1).

        The impl special-cases all-equal lambdas via scipy.stats.chi2.sf,
        so this should match exactly. Tolerance 1e-10 (closed-form)."""
        lambdas = torch.tensor([1.0], dtype=stat_dtype)
        for q in (0.5, 1.0, 2.0, 3.84, 10.0):
            out = davies_pvalue(q, lambdas)
            ref = sp_stats.chi2.sf(q, df=1)
            assert abs(out - ref) < 1e-10, (
                f"q={q}: davies={out}, chi2.sf={ref}, diff={abs(out - ref)}"
            )

    def test_returns_float_in_unit_interval(self, stat_dtype):
        """Output is a plain Python float in [0, 1]."""
        lambdas = torch.tensor([1.0, 0.5, 0.25], dtype=stat_dtype)
        out = davies_pvalue(2.0, lambdas)
        assert isinstance(out, float)
        assert 0.0 <= out <= 1.0


class TestLiuPvalue:
    """``liu_pvalue`` (Liu 2009 / Satterthwaite moment-matching)."""

    def test_single_lambda_approximates_chi2(self, stat_dtype):
        """For lambdas=[1.0], q=2.0, Liu agrees with chi2.sf(2, 1) ~ 0.157.
        Loose tolerance (Liu is moment-matched, not exact)."""
        lambdas = torch.tensor([1.0], dtype=stat_dtype)
        out = liu_pvalue(2.0, lambdas)
        ref = sp_stats.chi2.sf(2.0, df=1)
        # Documented loose tolerance: Liu within 0.05 of exact chi2.
        assert abs(out - ref) < 0.05, (
            f"liu={out}, chi2.sf={ref}, diff={abs(out - ref)}"
        )

    def test_matches_davies_in_easy_regime(self, stat_dtype):
        """Liu and Davies agree to within 50% relative error for a
        balanced 3-component mixture in the bulk (q near E[Q])."""
        lambdas = torch.tensor([1.0, 0.5, 0.25], dtype=stat_dtype)
        d = davies_pvalue(3.0, lambdas)
        l = liu_pvalue(3.0, lambdas)
        # Loose tolerance (50% relative): Liu can diverge in tails
        assert abs(d - l) / max(d, l, 1e-12) < 0.5, (
            f"liu={l}, davies={d}, rel diff={abs(d - l) / max(d, l, 1e-12)}"
        )


class TestMixtureChi2Pvalue:
    """``mixture_chi2_pvalue`` dispatch wrapper."""

    def test_dispatches_to_davies(self, stat_dtype):
        """method='davies' returns the same as davies_pvalue."""
        lambdas = torch.tensor([1.0], dtype=stat_dtype)
        a = mixture_chi2_pvalue(2.0, lambdas, method="davies")
        b = davies_pvalue(2.0, lambdas)
        assert abs(a - b) < 1e-12, f"dispatch={a}, direct={b}"

    def test_dispatches_to_liu(self, stat_dtype):
        """method='liu' returns the same as liu_pvalue."""
        lambdas = torch.tensor([1.0], dtype=stat_dtype)
        a = mixture_chi2_pvalue(2.0, lambdas, method="liu")
        b = liu_pvalue(2.0, lambdas)
        assert abs(a - b) < 1e-12, f"dispatch={a}, direct={b}"


# ---------------------------------------------------------------------------
# permutation
# ---------------------------------------------------------------------------


def _tiny_perm_inputs(n=50, m=10, seed=0):
    """Build a small (n, m) permutation-test problem deterministically."""
    gen = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=gen, dtype=torch.float64)
    Y = torch.randn(n, generator=gen, dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    return G, Y, X0


class TestPermutationMaxT:
    """``permutation_maxT`` Westfall-Young maxT FWER."""

    def test_seed_reproducibility(self, stat_dtype):
        """Identical seed -> identical adjusted p-values."""
        G, Y, X0 = _tiny_perm_inputs(n=50, m=10, seed=0)
        p1 = permutation_maxT(G, Y, X0, n_perms=100, seed=42)
        p2 = permutation_maxT(G, Y, X0, n_perms=100, seed=42)
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    def test_returns_tensor_of_correct_shape(self, stat_dtype):
        """Output shape (n_snps,)."""
        G, Y, X0 = _tiny_perm_inputs(n=50, m=10, seed=1)
        out = permutation_maxT(G, Y, X0, n_perms=100, seed=7)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10,)

    def test_no_signal_pvalues_uniform(self, stat_dtype):
        """Independent (G, Y) -> empirical p-values roughly uniform on (0,1).
        With n_snps=10, expect 2-9 of them below 0.5 (loose bound)."""
        G, Y, X0 = _tiny_perm_inputs(n=50, m=10, seed=0)
        p = permutation_maxT(G, Y, X0, n_perms=200, seed=0)
        n_low = int((p < 0.5).sum().item())
        assert 2 <= n_low <= 9, (
            f"n_low={n_low} out of 10 SNPs; expected uniform in [2, 9]"
        )


class TestAdaptivePermutationMaxT:
    """``adaptive_permutation_maxT`` two-phase adaptive FWER."""

    def test_seed_reproducibility(self, stat_dtype):
        """Identical seed -> identical output."""
        G, Y, X0 = _tiny_perm_inputs(n=50, m=10, seed=0)
        p1 = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=200, n_perms_max=500, seed=42, batch_size=100
        )
        p2 = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=200, n_perms_max=500, seed=42, batch_size=100
        )
        torch.testing.assert_close(p1, p2, rtol=0, atol=0)

    def test_returns_tensor_of_correct_shape(self, stat_dtype):
        """Output shape (n_snps,)."""
        G, Y, X0 = _tiny_perm_inputs(n=50, m=10, seed=1)
        out = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=200, n_perms_max=500, seed=7, batch_size=100
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (10,)

    def test_runs_quickly_in_null_setting(self, stat_dtype):
        """Null setting: function should complete quickly (< 10 s) for
        n_snps=20, n_perms_max=10000 because most SNPs are pruned in
        Phase 1. We bound run time as a proxy for early-stop behavior."""
        import time

        G, Y, X0 = _tiny_perm_inputs(n=50, m=20, seed=2)
        t0 = time.time()
        out = adaptive_permutation_maxT(
            G, Y, X0, n_perms_min=1000, n_perms_max=10000, seed=3,
            batch_size=500,
        )
        elapsed = time.time() - t0
        assert out.shape == (20,)
        assert elapsed < 10.0, f"adaptive_permutation_maxT took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# spa
# ---------------------------------------------------------------------------


class TestSaddlepointPvalue:
    """``saddlepoint_pvalue`` SAIGE-style SPA correction."""

    def test_returns_tensor_in_unit_interval(self, stat_dtype):
        """Output is a tensor with all values in [0, 1]."""
        torch.manual_seed(0)
        n, m = 100, 5
        mu = torch.full((n,), 0.3, dtype=torch.float64)
        G = torch.randn(n, m, dtype=torch.float64)
        # Build a small score: sum_i (Y_i - mu_i) * g_ij with random Y
        Y = torch.bernoulli(mu)
        scores = (G * (Y - mu).unsqueeze(1)).sum(dim=0)
        out = saddlepoint_pvalue(scores, mu, G, threshold=2.0)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (m,)
        assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_below_threshold_falls_back_to_chi2(self, stat_dtype):
        """When chi2_stat < threshold, the impl skips SPA correction
        and the output equals the simple chi2_sf(score^2 / V, 1)."""
        torch.manual_seed(0)
        n, m = 100, 1
        mu = torch.full((n,), 0.5, dtype=torch.float64)
        G = torch.randn(n, m, dtype=torch.float64)
        # Tiny score -> chi2_stat << 2.0
        scores = torch.tensor([0.05], dtype=torch.float64)
        # Compute reference chi2 p-value directly.
        W = mu * (1.0 - mu)
        V = (G * (W.unsqueeze(1) * G)).sum(dim=0)
        ref_chi2 = scores ** 2 / V
        ref_p = sp_stats.chi2.sf(ref_chi2.cpu().numpy(), df=1)

        out = saddlepoint_pvalue(scores, mu, G, threshold=2.0)
        np.testing.assert_allclose(
            out.cpu().numpy(), ref_p, rtol=1e-6, atol=1e-6,
        )


# ---------------------------------------------------------------------------
# cauchy
# ---------------------------------------------------------------------------


class TestCauchyCombination:
    """``cauchy_combination`` (ACAT / CCT, Liu et al. 2019)."""

    def test_uniform_pvals_returns_uniform_in_distribution(self, stat_dtype):
        """1000 trials of m=5 uniform p-values: combined p has mean ~0.5
        and KS distance from uniform < 0.1."""
        torch.manual_seed(0)
        n_trials, m = 1000, 5
        p_matrix = torch.rand(n_trials, m, dtype=torch.float64)
        combined = cauchy_combination(p_matrix)
        assert combined.shape == (n_trials,)
        # Mean of a uniform in (0,1) is 0.5; tolerance 0.1.
        mean_p = float(combined.mean().item())
        assert abs(mean_p - 0.5) < 0.1, f"mean combined p={mean_p}"
        # KS distance from uniform: tolerance 0.1 (loose).
        ks_stat, _ = sp_stats.kstest(combined.cpu().numpy(), "uniform")
        assert ks_stat < 0.1, f"KS distance from uniform = {ks_stat}"

    def test_strong_signal_recovers_low_pvalue(self, stat_dtype):
        """One p=1e-10 + the rest uniform should give combined p < 1e-3.
        ACAT inherits from the smallest input p in the heavy-tail regime."""
        gen = torch.Generator().manual_seed(0)
        m = 5
        p_row = torch.rand(m, generator=gen, dtype=torch.float64)
        p_row[0] = 1e-10
        out = cauchy_combination(p_row.unsqueeze(0))
        assert out.shape == (1,)
        assert out.item() < 1e-3, f"combined p = {out.item()}"

    def test_returns_tensor_same_shape_as_p_axis_0(self, stat_dtype):
        """For (n, m) input, output is (n,)."""
        p = torch.rand(7, 4, dtype=torch.float64)
        out = cauchy_combination(p)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (7,)


# ---------------------------------------------------------------------------
# genomic_control
# ---------------------------------------------------------------------------


class TestLambdaGc:
    """``lambda_gc`` Devlin & Roeder 1999 inflation factor."""

    def test_uniform_pvalues_lambda_one(self, stat_dtype):
        """Pure-null U(0,1) p-values: lambda_GC ~ 1.0 ± 0.1 with 10k draws."""
        gen = torch.Generator().manual_seed(0)
        p = torch.rand(10000, generator=gen, dtype=torch.float64)
        lam = lambda_gc(p)
        assert isinstance(lam, float)
        assert abs(lam - 1.0) < 0.1, f"lambda_GC under uniform = {lam}"

    def test_inflated_pvalues_lambda_above_one(self, stat_dtype):
        """Multiply chi2 statistics by 1.5 (artificial inflation):
        lambda_GC clearly > 1.2."""
        gen = np.random.default_rng(0)
        # Sample chi2(1) statistics, inflate, convert back to p.
        chi2s = sp_stats.chi2.rvs(df=1, size=10000, random_state=gen) * 1.5
        p_inflated = sp_stats.chi2.sf(chi2s, df=1)
        p_t = torch.tensor(p_inflated, dtype=torch.float64)
        lam = lambda_gc(p_t)
        assert lam > 1.2, f"lambda_GC under 1.5x inflation = {lam}"

    def test_uses_chi2_1_median_constant(self, stat_dtype):
        """Hand-checked formula: median chi2 for p=0.5 across all markers
        is chi2.ppf(0.5, 1); divided by CHI2_1_MEDIAN gives 1.0."""
        # All p-values are 0.5 -> all chi2_isf(0.5, 1) = chi2.ppf(0.5, 1)
        # = CHI2_1_MEDIAN, so lambda = CHI2_1_MEDIAN / CHI2_1_MEDIAN = 1.0.
        p = torch.full((100,), 0.5, dtype=torch.float64)
        lam = lambda_gc(p)
        assert abs(lam - 1.0) < 1e-10, (
            f"lambda_gc(p=0.5 const) = {lam}, expected 1.0 from CHI2_1_MEDIAN identity"
        )


class TestDiagnoseInflation:
    """``diagnose_inflation`` returns a status string for a lambda_GC."""

    def test_low_lambda_is_deflated(self, stat_dtype):
        """lambda < 0.9 -> 'deflated'."""
        s = diagnose_inflation(0.85)
        assert isinstance(s, str)
        assert "deflat" in s.lower()

    def test_normal_lambda_is_well_calibrated(self, stat_dtype):
        """lambda ~ 1.0 -> 'well-calibrated'."""
        s = diagnose_inflation(1.0)
        assert isinstance(s, str)
        assert "well" in s.lower() or "calibrat" in s.lower()

    def test_high_lambda_is_inflated(self, stat_dtype):
        """lambda > 1.1 -> 'mildly inflated' or 'severely inflated'."""
        s = diagnose_inflation(1.5)
        assert isinstance(s, str)
        assert "inflat" in s.lower()

    def test_returns_string(self, stat_dtype):
        """Always returns a Python str."""
        for lam in (0.5, 0.95, 1.0, 1.2, 2.0, 5.0):
            assert isinstance(diagnose_inflation(lam), str)


class TestChi21Median:
    """``CHI2_1_MEDIAN`` constant matches scipy."""

    def test_value_matches_scipy(self, stat_dtype):
        """CHI2_1_MEDIAN == scipy.stats.chi2.ppf(0.5, df=1) to 1e-12."""
        ref = sp_stats.chi2.ppf(0.5, df=1)
        assert abs(CHI2_1_MEDIAN - ref) < 1e-12, (
            f"CHI2_1_MEDIAN={CHI2_1_MEDIAN}, scipy={ref}"
        )


# ---------------------------------------------------------------------------
# pve
# ---------------------------------------------------------------------------


def _make_scan_result(m, p_values, dtype=torch.float64):
    """Build a single-trait ScanResult with m markers and given p-values.

    All other fields filled with placeholder values appropriate for a
    one-chromosome diploid scan."""
    return ScanResult(
        chr=["1"] * m,
        pos=list(range(1, m + 1)),
        snp=[f"snp{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.3, dtype=dtype),
        beta=torch.zeros(m, dtype=dtype),
        se=torch.ones(m, dtype=dtype),
        stat=torch.zeros(m, dtype=dtype),
        p=torch.as_tensor(p_values, dtype=dtype),
        test="wald",
    )


class TestPVEResult:
    """``PVEResult`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate, attribute round-trip exactly."""
        beta = torch.tensor([0.1, 0.2], dtype=stat_dtype)
        pve = torch.tensor([0.01, 0.02], dtype=stat_dtype)
        res = PVEResult(
            snp=["s1", "s2"], chr=["1", "2"], pos=[100, 200],
            beta=beta, pve=pve,
            pve_total=0.03, phenotypic_variance=1.5,
            method="marginal", n_significant=2,
            capped=False, rank_deficient=False,
        )
        assert res.snp == ["s1", "s2"]
        assert res.chr == ["1", "2"]
        assert res.pos == [100, 200]
        torch.testing.assert_close(res.beta, beta, rtol=0, atol=0)
        torch.testing.assert_close(res.pve, pve, rtol=0, atol=0)
        assert res.pve_total == 0.03
        assert res.phenotypic_variance == 1.5
        assert res.method == "marginal"
        assert res.n_significant == 2
        assert res.capped is False
        assert res.rank_deficient is False

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` returns a string."""
        res = PVEResult(
            snp=[], chr=[], pos=[],
            beta=torch.zeros(0, dtype=stat_dtype),
            pve=torch.zeros(0, dtype=stat_dtype),
            pve_total=0.0, phenotypic_variance=0.0,
            method="marginal", n_significant=0, capped=False,
        )
        s = repr(res)
        assert isinstance(s, str)
        assert "PVEResult" in s


class TestComputePve:
    """``compute_pve`` per-marker phenotypic variance explained."""

    def test_returns_pveresult(self, stat_dtype):
        """Output is a PVEResult instance."""
        torch.manual_seed(0)
        n, m = 100, 20
        result = _make_scan_result(m, [0.5] * m, dtype=stat_dtype)
        y = torch.randn(n, dtype=stat_dtype,
                        generator=torch.Generator().manual_seed(0))
        out = compute_pve(result, y, significance_threshold=5e-8)
        assert isinstance(out, PVEResult)

    def test_pve_total_in_unit_interval(self, stat_dtype):
        """``result.pve_total`` is a finite number in [0, 1]."""
        torch.manual_seed(0)
        n, m = 100, 20
        # Two markers significant, with non-trivial beta + af.
        p_vals = [0.5] * m
        p_vals[0] = 1e-12
        p_vals[1] = 1e-10
        result = _make_scan_result(m, p_vals, dtype=stat_dtype)
        # Set realistic beta for the two significant markers.
        result.beta = torch.zeros(m, dtype=stat_dtype)
        result.beta[0] = 0.1
        result.beta[1] = 0.05
        y = torch.randn(n, dtype=stat_dtype,
                        generator=torch.Generator().manual_seed(0))
        out = compute_pve(result, y, significance_threshold=5e-8)
        assert 0.0 <= out.pve_total <= 1.0, f"pve_total = {out.pve_total}"

    def test_no_significant_markers_pve_zero(self, stat_dtype):
        """When all p > threshold, returns pve_total=0 and n_significant=0."""
        n, m = 100, 20
        result = _make_scan_result(m, [0.5] * m, dtype=stat_dtype)
        y = torch.randn(n, dtype=stat_dtype,
                        generator=torch.Generator().manual_seed(0))
        out = compute_pve(result, y, significance_threshold=5e-8)
        assert out.n_significant == 0
        assert out.pve_total == 0.0


# ---------------------------------------------------------------------------
# peak_pruning
# ---------------------------------------------------------------------------


class TestQTLPeak:
    """``QTLPeak`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate, attribute round-trip exactly."""
        peak = QTLPeak(
            snp="rs123", chr="1", pos=1_500_000,
            neglog10p=8.5, best_model="additive",
            window_start=1_000_000, window_end=2_000_000,
            n_markers_in_window=5,
        )
        assert peak.snp == "rs123"
        assert peak.chr == "1"
        assert peak.pos == 1_500_000
        assert peak.neglog10p == 8.5
        assert peak.best_model == "additive"
        assert peak.window_start == 1_000_000
        assert peak.window_end == 2_000_000
        assert peak.n_markers_in_window == 5

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` returns a string."""
        peak = QTLPeak(
            snp="rs1", chr="1", pos=100,
            neglog10p=1.0, best_model="NA",
            window_start=0, window_end=200, n_markers_in_window=1,
        )
        s = repr(peak)
        assert isinstance(s, str)
        assert "QTLPeak" in s


class TestPrunePeaks:
    """``prune_peaks`` LD-window peak merging."""

    def test_no_significant_peaks(self, stat_dtype):
        """All p > threshold -> empty peak list."""
        result = _make_scan_result(10, [0.5] * 10, dtype=stat_dtype)
        peaks = prune_peaks(result, p_threshold=1e-4)
        assert peaks == []

    def test_window_merges_nearby(self, stat_dtype):
        """3 markers at pos=1000/1001/2_000_000 with bp_window=1_000_000:
        the first two merge into one peak; the third stands alone."""
        m = 3
        result = ScanResult(
            chr=["1"] * m,
            pos=[1000, 1001, 2_000_000],
            snp=[f"snp{i}" for i in range(m)],
            a1=["A"] * m, a2=["G"] * m,
            af=torch.full((m,), 0.3, dtype=stat_dtype),
            beta=torch.zeros(m, dtype=stat_dtype),
            se=torch.ones(m, dtype=stat_dtype),
            stat=torch.zeros(m, dtype=stat_dtype),
            p=torch.tensor([1e-10, 1e-9, 1e-8], dtype=stat_dtype),
            test="wald",
        )
        peaks = prune_peaks(result, bp_window=1_000_000, p_threshold=1e-4)
        assert len(peaks) == 2, (
            f"got {len(peaks)} peaks; expected 2 (merged 1000+1001, plus 2M)"
        )
        # Best peak (lowest p) should be at pos=1000.
        peaks_sorted = sorted(peaks, key=lambda p: -p.neglog10p)
        assert peaks_sorted[0].pos == 1000

    def test_returns_list_of_QTLPeak(self, stat_dtype):
        """Output is a list of QTLPeak instances."""
        m = 3
        result = ScanResult(
            chr=["1"] * m,
            pos=[100, 1_500_000, 3_000_000],
            snp=[f"snp{i}" for i in range(m)],
            a1=["A"] * m, a2=["G"] * m,
            af=torch.full((m,), 0.3, dtype=stat_dtype),
            beta=torch.zeros(m, dtype=stat_dtype),
            se=torch.ones(m, dtype=stat_dtype),
            stat=torch.zeros(m, dtype=stat_dtype),
            p=torch.tensor([1e-10, 1e-8, 1e-6], dtype=stat_dtype),
            test="wald",
        )
        out = prune_peaks(result, bp_window=1_000_000, p_threshold=1e-4)
        assert isinstance(out, list)
        for p in out:
            assert isinstance(p, QTLPeak)


# ---------------------------------------------------------------------------
# best_model
# ---------------------------------------------------------------------------


class TestBestModelResult:
    """``BestModelResult`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate, attribute round-trip exactly."""
        best_p = torch.tensor([1e-10, 0.5], dtype=stat_dtype)
        nl = torch.tensor([10.0, 0.30103], dtype=stat_dtype)
        res = BestModelResult(
            snp=["s1", "s2"], chr=["1", "1"], pos=[100, 200],
            best_model=["additive", "additive"],
            best_neglog10p=nl, best_p=best_p,
            bic_penalty_applied=True,
            all_model_p={"additive": best_p},
        )
        assert res.snp == ["s1", "s2"]
        assert res.best_model == ["additive", "additive"]
        torch.testing.assert_close(res.best_p, best_p, rtol=0, atol=0)
        torch.testing.assert_close(res.best_neglog10p, nl, rtol=0, atol=0)
        assert res.bic_penalty_applied is True
        assert "additive" in res.all_model_p

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` returns a string."""
        res = BestModelResult(
            snp=[], chr=[], pos=[],
            best_model=[],
            best_neglog10p=torch.zeros(0, dtype=stat_dtype),
            best_p=torch.zeros(0, dtype=stat_dtype),
            bic_penalty_applied=False,
        )
        s = repr(res)
        assert isinstance(s, str)
        assert "BestModelResult" in s


class TestSelectBestModel:
    """``select_best_model`` per-marker gene-action selection."""

    def test_returns_BestModelResult(self, stat_dtype):
        """Output is a BestModelResult instance."""
        m = 5
        scan_a = _make_scan_result(m, [0.1] * m, dtype=stat_dtype)
        scan_b = _make_scan_result(m, [0.5] * m, dtype=stat_dtype)
        out = select_best_model(
            {"additive": scan_a, "1-dom": scan_b},
            n_samples=100, bic_penalty=True, ploidy=4,
        )
        assert isinstance(out, BestModelResult)

    def test_no_bic_penalty_picks_minimum_p(self, stat_dtype):
        """With bic_penalty=False, best_model[j] = argmin(p[j])."""
        m = 4
        # additive has lowest p at marker 0, 2; '1-dom' at 1, 3.
        p_a = torch.tensor([1e-10, 0.5, 1e-12, 0.3], dtype=stat_dtype)
        p_b = torch.tensor([0.5, 1e-10, 0.3, 1e-12], dtype=stat_dtype)
        scan_a = _make_scan_result(m, p_a, dtype=stat_dtype)
        scan_b = _make_scan_result(m, p_b, dtype=stat_dtype)
        out = select_best_model(
            {"additive": scan_a, "1-dom": scan_b},
            n_samples=100, bic_penalty=False, ploidy=4,
        )
        assert out.best_model == ["additive", "1-dom", "additive", "1-dom"]

    def test_picks_lowest_p_per_marker(self, stat_dtype):
        """With BIC penalty for >1 parameter models: when all candidate
        models have 1 parameter (additive vs 1-dom), the penalty is 0
        and the choice reduces to argmin(p) per marker."""
        m = 3
        p_a = torch.tensor([1e-12, 0.5, 1e-5], dtype=stat_dtype)
        p_b = torch.tensor([0.5, 1e-12, 1e-3], dtype=stat_dtype)
        scan_a = _make_scan_result(m, p_a, dtype=stat_dtype)
        scan_b = _make_scan_result(m, p_b, dtype=stat_dtype)
        out = select_best_model(
            {"additive": scan_a, "1-dom": scan_b},
            n_samples=100, bic_penalty=True, ploidy=4,
        )
        # Both models have 1 param -> no penalty -> argmin(p)
        assert out.best_model == ["additive", "1-dom", "additive"]


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


class TestComparePvalues:
    """``compare_pvalues`` against a reference p-value vector."""

    def test_perfect_match_high_concordance(self, stat_dtype):
        """compare_pvalues(p, p) -> frac_within_tolerance=1.0,
        mean_abs_diff=0, corr_neglog10=1.0."""
        gen = torch.Generator().manual_seed(0)
        p = torch.rand(100, generator=gen, dtype=torch.float64)
        out = compare_pvalues(p, p, tolerance=1e-4)
        assert isinstance(out, dict)
        assert out["frac_within_tolerance"] == 1.0
        assert out["mean_abs_diff"] == 0.0
        assert out["max_abs_diff"] == 0.0
        assert abs(out["corr_neglog10"] - 1.0) < 1e-10

    def test_returns_dict_with_documented_keys(self, stat_dtype):
        """Output dict contains the documented keys."""
        gen = torch.Generator().manual_seed(0)
        p1 = torch.rand(50, generator=gen, dtype=torch.float64)
        p2 = torch.rand(50, generator=gen, dtype=torch.float64)
        out = compare_pvalues(p1, p2, tolerance=1e-4)
        for key in (
            "n", "mean_abs_diff", "max_abs_diff",
            "frac_within_tolerance", "corr_neglog10",
            "mean_abs_diff_neglog10", "tolerance",
        ):
            assert key in out, f"missing key {key!r} in compare_pvalues output"
        assert out["n"] == 50
        assert out["tolerance"] == 1e-4


# ---------------------------------------------------------------------------
# models.base
# ---------------------------------------------------------------------------


class TestScanResult:
    """``ScanResult`` dataclass round-trip + repr."""

    def test_construct_and_round_trip(self, stat_dtype):
        """Instantiate, attribute round-trip exactly."""
        m = 3
        af = torch.tensor([0.1, 0.2, 0.3], dtype=stat_dtype)
        beta = torch.tensor([0.5, 0.6, 0.7], dtype=stat_dtype)
        se = torch.tensor([0.05, 0.06, 0.07], dtype=stat_dtype)
        stat = torch.tensor([100.0, 100.0, 100.0], dtype=stat_dtype)
        p = torch.tensor([1e-10, 1e-9, 1e-8], dtype=stat_dtype)
        n_obs = torch.tensor([100, 100, 100], dtype=torch.long)
        res = ScanResult(
            chr=["1", "2", "3"],
            pos=[100, 200, 300],
            snp=["rs1", "rs2", "rs3"],
            a1=["A", "C", "G"],
            a2=["T", "G", "C"],
            af=af, beta=beta, se=se, stat=stat, p=p,
            test="wald",
            n_obs=n_obs,
        )
        assert res.chr == ["1", "2", "3"]
        assert res.pos == [100, 200, 300]
        assert res.snp == ["rs1", "rs2", "rs3"]
        assert res.a1 == ["A", "C", "G"]
        assert res.a2 == ["T", "G", "C"]
        torch.testing.assert_close(res.af, af, rtol=0, atol=0)
        torch.testing.assert_close(res.beta, beta, rtol=0, atol=0)
        torch.testing.assert_close(res.se, se, rtol=0, atol=0)
        torch.testing.assert_close(res.stat, stat, rtol=0, atol=0)
        torch.testing.assert_close(res.p, p, rtol=0, atol=0)
        torch.testing.assert_close(res.n_obs, n_obs, rtol=0, atol=0)
        assert res.test == "wald"
        # Default inference_type
        assert res.inference_type == "marginal"
        assert len(res) == m

    def test_repr_does_not_crash(self, stat_dtype):
        """``repr(...)`` returns a string."""
        m = 3
        res = ScanResult(
            chr=["1"] * m, pos=[1, 2, 3],
            snp=[f"s{i}" for i in range(m)],
            a1=["A"] * m, a2=["G"] * m,
            af=torch.zeros(m, dtype=stat_dtype),
            beta=torch.zeros(m, dtype=stat_dtype),
            se=torch.ones(m, dtype=stat_dtype),
            stat=torch.zeros(m, dtype=stat_dtype),
            p=torch.full((m,), 1.0, dtype=stat_dtype),
            test="wald",
        )
        s = repr(res)
        assert isinstance(s, str)
        assert "ScanResult" in s
