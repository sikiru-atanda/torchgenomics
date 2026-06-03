"""Tests for winner's curse correction module.

Covers three correction methods (conditional likelihood, FIQT, bootstrap)
and the dispatch wrapper.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.postgwas._winners_curse import (
    WinnersCurseResult,
    bootstrap_correction,
    conditional_likelihood,
    correct_winners_curse,
    fiqt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQRT2PI = (2.0 * math.pi) ** 0.5


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x ** 2) / _SQRT2PI


def _Phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / 2.0 ** 0.5)


# ---------------------------------------------------------------------------
# A. Conditional likelihood
# ---------------------------------------------------------------------------


def test_cl_shrinks_toward_zero():
    """Significant positive beta should shrink toward zero after CL."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.08], dtype=torch.float64)  # z = 8
    res = conditional_likelihood(beta, se, alpha=5e-8)

    assert res.method == "conditional_likelihood"
    assert res.beta_adjusted[0] < beta[0], "CL should shrink positive beta toward zero"
    assert res.beta_adjusted[0] > 0.0, "CL should not overshoot past zero"
    assert res.n_corrected == 1


def test_cl_large_effect_minimal_shrinkage():
    """Very large z (z=20) should have shrinkage close to 1."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.20], dtype=torch.float64)  # z = 20
    res = conditional_likelihood(beta, se, alpha=5e-8)

    assert res.shrinkage_factor[0] > 0.99, (
        f"Shrinkage for z=20 should be ~1, got {res.shrinkage_factor[0]:.6f}"
    )


def test_cl_shrinkage_bounded():
    """All shrinkage factors should be in [0, 1] for positive betas."""
    torch.manual_seed(42)
    se = torch.full((50,), 0.01, dtype=torch.float64)
    # z-scores ranging from 6 to 15
    z_vals = torch.linspace(6.0, 15.0, 50, dtype=torch.float64)
    beta = z_vals * se
    res = conditional_likelihood(beta, se, alpha=5e-8)

    assert (res.shrinkage_factor >= -1e-10).all(), "Shrinkage should be >= 0"
    assert (res.shrinkage_factor <= 1.0 + 1e-10).all(), "Shrinkage should be <= 1"


def test_cl_nonsignificant_unchanged():
    """Non-significant variants should be left uncorrected."""
    se = torch.tensor([0.01, 0.01], dtype=torch.float64)
    beta = torch.tensor([0.02, 0.03], dtype=torch.float64)  # z = 2, 3: not significant
    res = conditional_likelihood(beta, se, alpha=5e-8)

    assert torch.allclose(res.beta_adjusted, beta), (
        "Non-significant betas should be unchanged"
    )
    assert res.n_corrected == 0


def test_cl_known_value():
    """Hand-computed CL correction for z=6, c=5.4507 (alpha=5e-8).

    correction = se * [phi(c - z) - phi(-c - z)] / [Phi(z - c) + Phi(-z - c)]
    With z=6, c ~ 5.4507:
      phi(c - z) = phi(-0.5493)
      phi(-c - z) = phi(-11.4507)  ~ 0
      Phi(z - c) = Phi(0.5493)
      Phi(-z - c) = Phi(-11.4507) ~ 0
    So correction ~ se * phi(-0.5493) / Phi(0.5493)
    """
    from scipy.stats import norm  # type: ignore[import-untyped]

    alpha = 5e-8
    c = norm.ppf(1.0 - alpha / 2.0)  # ~5.4507
    z = 6.0
    se_val = 0.01

    num = _phi(c - z) - _phi(-c - z)
    den = _Phi(z - c) + _Phi(-z - c)
    expected_correction = se_val * (num / den)
    expected_beta_adj = z * se_val - expected_correction

    beta = torch.tensor([z * se_val], dtype=torch.float64)
    se = torch.tensor([se_val], dtype=torch.float64)
    res = conditional_likelihood(beta, se, alpha=alpha)

    assert abs(res.beta_adjusted[0].item() - expected_beta_adj) < 1e-8, (
        f"Expected {expected_beta_adj:.10f}, got {res.beta_adjusted[0].item():.10f}"
    )


# ---------------------------------------------------------------------------
# B. FIQT
# ---------------------------------------------------------------------------


def test_fiqt_null_shrinks():
    """Pure null z-scores should be shrunk closer to zero by FIQT."""
    torch.manual_seed(123)
    z = torch.randn(1000, dtype=torch.float64)
    se = torch.ones(1000, dtype=torch.float64)
    res = fiqt(z, se)

    assert res.method == "fiqt"
    # Mean absolute corrected z should be smaller than original
    assert res.beta_adjusted.abs().mean() < z.abs().mean(), (
        "FIQT should shrink null z-scores toward zero"
    )


def test_fiqt_strong_signal_preserved():
    """A very strong z-score (z=10) should be largely preserved."""
    torch.manual_seed(456)
    # Mostly null + one strong signal
    z = torch.randn(500, dtype=torch.float64)
    z[0] = 10.0
    se = torch.ones(500, dtype=torch.float64)
    res = fiqt(z, se)

    # The corrected value for the strong signal should be close to original
    assert res.beta_adjusted[0].item() > 8.0, (
        f"Strong signal z=10 should be mostly preserved, got {res.beta_adjusted[0].item():.2f}"
    )


def test_fiqt_pi0_estimation():
    """On a known mixture, estimated pi0 should be reasonable."""
    torch.manual_seed(789)
    m = 2000
    # 90% null, 10% signal
    n_null = int(0.9 * m)
    z_null = torch.randn(n_null, dtype=torch.float64)
    z_signal = 5.0 + torch.randn(m - n_null, dtype=torch.float64)
    z = torch.cat([z_null, z_signal])
    se = torch.ones(m, dtype=torch.float64)

    # We don't directly get pi0 from the result, but we can check that the
    # method runs and produces reasonable shrinkage on null variants
    res = fiqt(z, se)

    # Null variants (first n_null) should be shrunk; signal should be preserved
    null_shrinkage = res.shrinkage_factor[:n_null].abs().mean()
    signal_shrinkage = res.shrinkage_factor[n_null:].abs().mean()
    assert null_shrinkage < signal_shrinkage, (
        "Null variants should be shrunk more than signals"
    )


def test_fiqt_custom_bandwidth():
    """FIQT should run without error with a custom bandwidth."""
    torch.manual_seed(101)
    z = torch.randn(200, dtype=torch.float64)
    se = torch.ones(200, dtype=torch.float64)

    res = fiqt(z, se, bandwidth=0.5)
    assert isinstance(res, WinnersCurseResult)
    assert res.beta_adjusted.shape == (200,)


# ---------------------------------------------------------------------------
# C. Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_direction():
    """Positive significant beta should be corrected downward by bootstrap."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.07], dtype=torch.float64)  # z = 7, significant
    res = bootstrap_correction(beta, se, alpha=5e-8, n_boot=50000, seed=42)

    assert res.method == "bootstrap"
    assert res.beta_adjusted[0] < beta[0], (
        "Bootstrap should correct positive bias downward"
    )
    assert res.beta_adjusted[0] > 0.0, (
        "Bootstrap should not overshoot past zero for z=7"
    )


def test_bootstrap_seed_determinism():
    """Same seed should produce identical results."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.08], dtype=torch.float64)

    res1 = bootstrap_correction(beta, se, alpha=5e-8, n_boot=5000, seed=99)
    res2 = bootstrap_correction(beta, se, alpha=5e-8, n_boot=5000, seed=99)

    assert torch.allclose(res1.beta_adjusted, res2.beta_adjusted), (
        "Same seed should give identical bootstrap results"
    )
    assert torch.allclose(res1.shrinkage_factor, res2.shrinkage_factor)


def test_bootstrap_large_effect():
    """Very large |z| should have minimal bootstrap bias correction."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.20], dtype=torch.float64)  # z = 20
    res = bootstrap_correction(beta, se, alpha=5e-8, n_boot=10000, seed=7)

    # At z=20 almost all bootstraps will be significant, bias is tiny
    relative_change = abs(
        (res.beta_adjusted[0] - beta[0]) / beta[0]
    ).item()
    assert relative_change < 0.01, (
        f"Large z=20 should have <1% correction, got {relative_change:.4f}"
    )


# ---------------------------------------------------------------------------
# D. Dispatch wrapper
# ---------------------------------------------------------------------------


def test_dispatch_conditional_likelihood():
    """correct_winners_curse with method='conditional_likelihood' dispatches correctly."""
    se = torch.tensor([0.01], dtype=torch.float64)
    beta = torch.tensor([0.08], dtype=torch.float64)

    res = correct_winners_curse(beta, se, method="conditional_likelihood")
    res_direct = conditional_likelihood(beta, se)

    assert res.method == "conditional_likelihood"
    assert torch.allclose(res.beta_adjusted, res_direct.beta_adjusted)


def test_dispatch_invalid_method():
    """Unknown method should raise ValueError."""
    beta = torch.tensor([0.05], dtype=torch.float64)
    se = torch.tensor([0.01], dtype=torch.float64)

    with pytest.raises(ValueError, match="Unknown method"):
        correct_winners_curse(beta, se, method="nonexistent")
