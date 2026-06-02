"""Tests for ``torchgenomics.pgs.validation`` -- PGS predictive performance metrics."""
from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.pgs.validation import validate_pgs

# ---------------------------------------------------------------------------
# Continuous-trait tests
# ---------------------------------------------------------------------------


def test_perfect_continuous_prediction():
    """pgs == y (perfect predictor) => R^2 ~ 1, pearson_r ~ 1."""
    torch.manual_seed(0)
    y = torch.randn(100, dtype=torch.float64)
    pgs = y.clone()
    result = validate_pgs(pgs, y, trait_type="continuous")
    assert result.r2 == pytest.approx(1.0, abs=1e-6)
    assert result.pearson_r == pytest.approx(1.0, abs=1e-6)
    assert result.trait_type == "continuous"
    assert result.n == 100


def test_zero_correlation_continuous():
    """pgs independent of y => R^2 < 0.1, |pearson_r| < 0.15."""
    torch.manual_seed(42)
    n = 200
    y = torch.randn(n, dtype=torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    result = validate_pgs(pgs, y, trait_type="continuous")
    assert result.r2 < 0.1
    assert abs(result.pearson_r) < 0.15


def test_incremental_r2_with_covariates():
    """y = 0.5*cov + 0.5*pgs + noise => 0 < r2_incremental < r2."""
    torch.manual_seed(7)
    n = 200
    cov = torch.randn(n, dtype=torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    noise = torch.randn(n, dtype=torch.float64) * 0.3
    y = 0.5 * cov + 0.5 * pgs + noise
    covariates = cov.unsqueeze(-1)
    result = validate_pgs(pgs, y, covariates=covariates, trait_type="continuous")
    assert result.r2_incremental > 0.0
    assert result.r2_incremental < result.r2


def test_incremental_r2_no_covariates():
    """Without covariates, r2_incremental == r2."""
    torch.manual_seed(1)
    n = 100
    pgs = torch.randn(n, dtype=torch.float64)
    y = 2.0 * pgs + torch.randn(n, dtype=torch.float64) * 0.5
    result = validate_pgs(pgs, y, trait_type="continuous")
    assert result.r2_incremental == pytest.approx(result.r2, abs=1e-10)


def test_spearman_monotonic_relationship():
    """pgs = y^3 (monotonic, nonlinear) => spearman_r > 0.9, pearson_r < spearman_r."""
    torch.manual_seed(3)
    n = 200
    y = torch.randn(n, dtype=torch.float64)
    pgs = y.pow(3)
    result = validate_pgs(pgs, y, trait_type="continuous")
    assert result.spearman_r > 0.9
    assert result.pearson_r < result.spearman_r


# ---------------------------------------------------------------------------
# Binary-trait tests
# ---------------------------------------------------------------------------


def test_binary_auc_perfect():
    """Cases all have higher PGS than controls => AUC = 1.0."""
    n = 100
    y = torch.cat([torch.zeros(50), torch.ones(50)]).to(torch.float64)
    # Controls get low PGS, cases get high PGS (no overlap).
    pgs = torch.cat([torch.linspace(-2, -0.1, 50), torch.linspace(0.1, 2, 50)]).to(
        torch.float64
    )
    result = validate_pgs(pgs, y, trait_type="binary")
    assert result.auc == pytest.approx(1.0, abs=1e-6)


def test_binary_auc_random():
    """PGS independent of case/control => AUC ~ 0.5."""
    torch.manual_seed(99)
    n = 200
    y = torch.cat([torch.zeros(100), torch.ones(100)]).to(torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    result = validate_pgs(pgs, y, trait_type="binary")
    assert abs(result.auc - 0.5) < 0.1


def test_binary_auc_good_discrimination():
    """Moderate separation => AUC between 0.6 and 0.9."""
    torch.manual_seed(11)
    n = 200
    y = torch.cat([torch.zeros(100), torch.ones(100)]).to(torch.float64)
    # Cases shifted up by 0.8 std => moderate discrimination.
    pgs = torch.randn(n, dtype=torch.float64)
    pgs[100:] += 0.8
    result = validate_pgs(pgs, y, trait_type="binary")
    assert 0.6 < result.auc < 0.9


def test_binary_nagelkerke_r2():
    """Nagelkerke R^2 should be finite and in [0, 1]."""
    torch.manual_seed(5)
    n = 200
    y = torch.cat([torch.zeros(100), torch.ones(100)]).to(torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    pgs[100:] += 1.0
    result = validate_pgs(pgs, y, trait_type="binary")
    assert result.nagelkerke_r2 is not None
    assert math.isfinite(result.nagelkerke_r2)
    assert 0.0 <= result.nagelkerke_r2 <= 1.0


def test_binary_liability_r2():
    """With prevalence, liability_r2 is finite and >= 0."""
    torch.manual_seed(8)
    n = 200
    y = torch.cat([torch.zeros(100), torch.ones(100)]).to(torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    pgs[100:] += 1.0
    result = validate_pgs(pgs, y, trait_type="binary", prevalence=0.05)
    assert result.liability_r2 is not None
    assert math.isfinite(result.liability_r2)
    assert result.liability_r2 >= 0.0


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_rejects_mismatched_lengths():
    """pgs and y with different lengths => ValueError."""
    pgs = torch.randn(50, dtype=torch.float64)
    y = torch.randn(60, dtype=torch.float64)
    with pytest.raises(ValueError, match="same length"):
        validate_pgs(pgs, y)


def test_rejects_invalid_trait_type():
    """trait_type='invalid' => ValueError."""
    pgs = torch.randn(20, dtype=torch.float64)
    y = torch.randn(20, dtype=torch.float64)
    with pytest.raises(ValueError, match="trait_type"):
        validate_pgs(pgs, y, trait_type="invalid")


def test_rejects_non_binary_for_binary_type():
    """y with 3 unique values and trait_type='binary' => ValueError."""
    n = 30
    y = torch.tensor([0.0, 1.0, 2.0] * 10, dtype=torch.float64)
    pgs = torch.randn(n, dtype=torch.float64)
    with pytest.raises(ValueError, match="binary"):
        validate_pgs(pgs, y, trait_type="binary")


def test_continuous_result_has_no_binary_fields():
    """For continuous traits, auc / nagelkerke_r2 / liability_r2 are all None."""
    torch.manual_seed(0)
    y = torch.randn(50, dtype=torch.float64)
    pgs = torch.randn(50, dtype=torch.float64)
    result = validate_pgs(pgs, y, trait_type="continuous")
    assert result.auc is None
    assert result.nagelkerke_r2 is None
    assert result.liability_r2 is None
