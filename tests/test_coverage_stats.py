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
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    chi2_sf,
    holm,
    lrt_test,
    score_test,
    score_test_multi_df,
    sidak,
    storey_qvalue,
    wald_test,
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
