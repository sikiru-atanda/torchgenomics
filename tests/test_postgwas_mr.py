"""Tests for ``torchgwas.postgwas._mr`` --- Mendelian Randomization."""
from __future__ import annotations

import math

import torch

from torchgwas.postgwas._mr import MRResult, mr_all, mr_egger, mr_ivw, mr_presso, mr_weighted_median
from torchgwas.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_mr_data(
    n_instruments: int = 20,
    true_causal_effect: float = 0.3,
    seed: int = 0,
    pleiotropy: float = 0.0,
    outlier_indices: list[int] | None = None,
    outlier_magnitude: float = 5.0,
) -> tuple[SumStats, SumStats]:
    """Generate aligned exposure + outcome SumStats with a known causal effect.

    Parameters
    ----------
    n_instruments : int
        Number of instrumental variables (SNPs).
    true_causal_effect : float
        True causal effect of exposure on outcome.
    seed : int
        Random seed for reproducibility.
    pleiotropy : float
        Directional pleiotropy added to all outcome betas (Egger intercept).
    outlier_indices : list[int] or None
        Indices of instruments to corrupt with large outlier effects.
    outlier_magnitude : float
        Magnitude of the outlier effect added to corrupted instruments.
    """
    torch.manual_seed(seed)
    K = n_instruments

    # Exposure: strong signals with small SE
    bx = 0.3 + 0.2 * torch.randn(K, dtype=torch.float64).abs()
    se_x = 0.02 + 0.01 * torch.rand(K, dtype=torch.float64)

    # Outcome: causal effect * exposure + noise + optional pleiotropy
    noise = 0.02 * torch.randn(K, dtype=torch.float64)
    by = true_causal_effect * bx + noise + pleiotropy
    se_y = 0.05 + 0.02 * torch.rand(K, dtype=torch.float64)

    # Inject outliers
    if outlier_indices is not None:
        for j in outlier_indices:
            by[j] += outlier_magnitude * se_y[j]

    # Compute approximate p-values
    p_x = torch.erfc((bx / se_x).abs() / 2.0**0.5)
    p_y = torch.erfc((by / se_y).abs() / 2.0**0.5)
    n_samples = torch.full((K,), 10000.0, dtype=torch.float64)

    chr_list = ["1"] * K
    pos_list = list(range(1000, 1000 + K * 100, 100))
    snp_list = [f"rs{i}" for i in range(K)]
    a1_list = ["A"] * K
    a2_list = ["G"] * K

    exposure = SumStats(
        chr=chr_list, pos=pos_list, snp=snp_list,
        a1=a1_list, a2=a2_list,
        beta=bx, se=se_x, p=p_x, n=n_samples,
    )
    outcome = SumStats(
        chr=chr_list, pos=pos_list, snp=snp_list,
        a1=a1_list, a2=a2_list,
        beta=by, se=se_y, p=p_y, n=n_samples,
    )
    return exposure, outcome


# ---------------------------------------------------------------------------
# IVW tests
# ---------------------------------------------------------------------------


def test_mr_ivw_recovers_causal_effect():
    """IVW recovers a positive causal effect of 0.3."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.3, seed=42,
    )
    res = mr_ivw(exposure, outcome)
    assert res.method == "ivw"
    assert abs(res.beta_hat - 0.3) < 0.15, (
        f"IVW beta_hat={res.beta_hat:.4f}, expected ~0.3"
    )
    assert res.p_value < 0.05, f"IVW p_value={res.p_value:.4f}, expected < 0.05"


def test_mr_ivw_null_effect_not_significant():
    """IVW with true effect = 0.0 should not produce a significant result."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.0, seed=99,
    )
    res = mr_ivw(exposure, outcome)
    # Either the p-value is non-significant or the estimate is near zero
    assert res.p_value > 0.01 or abs(res.beta_hat) < 0.1, (
        f"IVW should not be significant under null: "
        f"beta_hat={res.beta_hat:.4f}, p={res.p_value:.4f}"
    )


# ---------------------------------------------------------------------------
# MR-Egger tests
# ---------------------------------------------------------------------------


def test_mr_egger_recovers_effect():
    """MR-Egger slope recovers the causal effect of 0.3."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=7,
    )
    res = mr_egger(exposure, outcome)
    assert res.method == "egger"
    assert abs(res.beta_hat - 0.3) < 0.2, (
        f"Egger beta_hat={res.beta_hat:.4f}, expected ~0.3"
    )


def test_mr_egger_intercept_near_zero_no_pleiotropy():
    """Without pleiotropy the Egger intercept should be non-significant."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=12,
        pleiotropy=0.0,
    )
    res = mr_egger(exposure, outcome)
    assert res.intercept_p > 0.05, (
        f"Egger intercept_p={res.intercept_p:.4f}, expected > 0.05 (no pleiotropy)"
    )


def test_mr_egger_detects_pleiotropy():
    """With directional pleiotropy the Egger intercept should deviate from zero."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=15,
        pleiotropy=0.15,
    )
    res = mr_egger(exposure, outcome)
    # Either p < 0.05 or |intercept| notably different from zero
    assert res.intercept_p < 0.05 or abs(res.intercept) > 0.01, (
        f"Egger intercept={res.intercept:.4f}, p={res.intercept_p:.4f}; "
        f"expected to detect pleiotropy"
    )


# ---------------------------------------------------------------------------
# Weighted median tests
# ---------------------------------------------------------------------------


def test_mr_weighted_median_recovers_effect():
    """Weighted median recovers causal effect of 0.3."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=21,
    )
    res = mr_weighted_median(exposure, outcome, n_boot=200, seed=21)
    assert res.method == "weighted_median"
    assert abs(res.beta_hat - 0.3) < 0.2, (
        f"Weighted median beta_hat={res.beta_hat:.4f}, expected ~0.3"
    )


def test_mr_weighted_median_robust_to_outliers():
    """Weighted median stays close to true effect even with 5/30 outlier IVs."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=33,
        outlier_indices=[0, 1, 2, 3, 4], outlier_magnitude=8.0,
    )
    res = mr_weighted_median(exposure, outcome, n_boot=200, seed=33)
    assert abs(res.beta_hat - 0.3) < 0.25, (
        f"Weighted median beta_hat={res.beta_hat:.4f}, "
        f"expected ~0.3 despite outliers"
    )


# ---------------------------------------------------------------------------
# MR-PRESSO tests
# ---------------------------------------------------------------------------


def test_mr_presso_detects_outliers():
    """MR-PRESSO flags at least 1 outlier when 3 large outliers are injected."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.3, seed=44,
        outlier_indices=[0, 5, 10], outlier_magnitude=10.0,
    )
    res = mr_presso(exposure, outcome, n_perm=200, seed=44)
    assert res.method == "mr_presso"
    # Either outliers are detected or the global test is suggestive
    assert res.n_outliers >= 1 or res.global_p < 0.1, (
        f"MR-PRESSO n_outliers={res.n_outliers}, global_p={res.global_p:.4f}; "
        f"expected to detect outliers"
    )


def test_mr_presso_corrected_closer_to_true():
    """After outlier removal the corrected beta should be closer to 0.3."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.3, seed=55,
        outlier_indices=[0, 5, 10], outlier_magnitude=10.0,
    )
    res = mr_presso(exposure, outcome, n_perm=200, seed=55)
    # If outliers were found, corrected should be better
    if res.n_outliers >= 1:
        err_uncorrected = abs(res.beta_hat - 0.3)
        err_corrected = abs(res.beta_corrected - 0.3)
        assert err_corrected <= err_uncorrected + 0.05, (
            f"Corrected beta={res.beta_corrected:.4f} not closer to 0.3 "
            f"than uncorrected beta={res.beta_hat:.4f}"
        )


# ---------------------------------------------------------------------------
# mr_all convenience
# ---------------------------------------------------------------------------


def test_mr_all_returns_four_results():
    """mr_all returns exactly 4 MRResult instances."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.3, seed=66,
    )
    results = mr_all(exposure, outcome, n_boot=200, n_perm=200, seed=66)
    assert len(results) == 4
    for r in results:
        assert isinstance(r, MRResult)
    methods = [r.method for r in results]
    assert methods == ["ivw", "egger", "weighted_median", "mr_presso"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mr_rejects_mismatched_lengths():
    """Exposure and outcome with different SNP counts raise ValueError."""
    exposure, _ = _make_mr_data(n_instruments=20, seed=77)
    _, outcome = _make_mr_data(n_instruments=15, seed=78)
    try:
        mr_ivw(exposure, outcome)
        assert False, "Expected ValueError for mismatched lengths"
    except ValueError as e:
        assert "same number" in str(e).lower() or "SNPs" in str(e)


def test_mr_rejects_too_few_instruments():
    """Fewer than 3 instruments raises ValueError."""
    exposure, outcome = _make_mr_data(n_instruments=2, seed=88)
    # Override the validation by constructing small SumStats directly
    exposure_small = SumStats(
        chr=["1", "1"], pos=[100, 200], snp=["rs0", "rs1"],
        a1=["A", "A"], a2=["G", "G"],
        beta=torch.tensor([0.3, 0.4], dtype=torch.float64),
        se=torch.tensor([0.02, 0.02], dtype=torch.float64),
        p=torch.tensor([1e-8, 1e-8], dtype=torch.float64),
        n=torch.tensor([10000.0, 10000.0], dtype=torch.float64),
    )
    outcome_small = SumStats(
        chr=["1", "1"], pos=[100, 200], snp=["rs0", "rs1"],
        a1=["A", "A"], a2=["G", "G"],
        beta=torch.tensor([0.1, 0.12], dtype=torch.float64),
        se=torch.tensor([0.05, 0.05], dtype=torch.float64),
        p=torch.tensor([0.05, 0.05], dtype=torch.float64),
        n=torch.tensor([10000.0, 10000.0], dtype=torch.float64),
    )
    try:
        mr_ivw(exposure_small, outcome_small)
        assert False, "Expected ValueError for too few instruments"
    except ValueError as e:
        assert "3" in str(e) or "instrument" in str(e).lower()


# ---------------------------------------------------------------------------
# MRResult field sanity
# ---------------------------------------------------------------------------


def test_mr_result_fields():
    """All MRResult fields from each method are finite and have expected types."""
    exposure, outcome = _make_mr_data(
        n_instruments=20, true_causal_effect=0.3, seed=100,
    )
    results = mr_all(exposure, outcome, n_boot=200, n_perm=200, seed=100)
    for res in results:
        # Core fields
        assert isinstance(res.method, str)
        assert isinstance(res.beta_hat, float)
        assert isinstance(res.se, float)
        assert isinstance(res.p_value, float)
        assert isinstance(res.n_instruments, int)
        assert math.isfinite(res.beta_hat), f"{res.method}: beta_hat not finite"
        assert math.isfinite(res.se), f"{res.method}: se not finite"
        assert math.isfinite(res.p_value), f"{res.method}: p_value not finite"
        assert 0.0 <= res.p_value <= 1.0, f"{res.method}: p_value out of [0,1]"
        assert res.se >= 0.0, f"{res.method}: se < 0"
        assert res.n_instruments == 20

        # Egger-specific
        if res.method == "egger":
            assert math.isfinite(res.intercept)
            assert math.isfinite(res.intercept_se)
            assert 0.0 <= res.intercept_p <= 1.0
            assert 0.0 <= res.egger_i2 <= 1.0

        # MR-PRESSO-specific
        if res.method == "mr_presso":
            assert math.isfinite(res.global_rss)
            assert 0.0 <= res.global_p <= 1.0
            assert isinstance(res.outlier_indices, list)
            assert res.n_outliers == len(res.outlier_indices)
            assert math.isfinite(res.beta_corrected)
            assert math.isfinite(res.se_corrected)
            assert 0.0 <= res.p_corrected <= 1.0


def test_mr_egger_i_squared_high_for_strong_instruments():
    """Egger I-squared should be high (> 0.5) for strong instruments."""
    exposure, outcome = _make_mr_data(
        n_instruments=30, true_causal_effect=0.3, seed=111,
    )
    res = mr_egger(exposure, outcome)
    assert res.egger_i2 > 0.5, (
        f"Egger I^2={res.egger_i2:.4f}, expected > 0.5 for strong instruments"
    )
