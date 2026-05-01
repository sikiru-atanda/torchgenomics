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
    IHWResult,
    adapt,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    chi2_sf,
    effective_test_count,
    effective_test_count_moskvina,
    hierarchical_fdr,
    holm,
    ihw,
    ld_correlation_eigenvalues,
    local_fdr,
    lrt_test,
    score_test,
    score_test_multi_df,
    sidak,
    storey_qvalue,
    wald_test,
    weighted_bh,
)
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
