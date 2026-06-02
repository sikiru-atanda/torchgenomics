"""Tests for the HESS regional heritability module."""

import math

import pytest
import torch

from torchgenomics.postgwas._hess import (
    HESSResult,
    hess_local_h2,
    hess_local_rg,
)

# ---------------------------------------------------------------------------
# 1. Identity LD: h2 near 0 under the null
# ---------------------------------------------------------------------------

def test_hess_identity_ld():
    """With R = I and z ~ N(0, 1), h2_hat ~ (sum(z^2) - m) / N ~ 0."""
    torch.manual_seed(42)
    m = 50
    N = 10_000
    z = torch.randn(m, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64)

    result = hess_local_h2(z, R, N, [(0, m)])

    assert isinstance(result, HESSResult)
    assert result.n_regions == 1
    assert result.n_snps_total == m

    # Under null: E[h2] = 0, check within 3 SE
    region = result.regions[0]
    assert region.n_snps == m
    assert region.n_eigenvalues_kept == m
    assert abs(region.h2_local) < 3 * region.h2_local_se + 0.05


# ---------------------------------------------------------------------------
# 2. Known h2 recovery
# ---------------------------------------------------------------------------

def test_hess_known_h2():
    """z ~ N(0, (1 + N*h2/m)*I) should give h2_hat ~ h2."""
    torch.manual_seed(42)
    m = 100
    N = 10_000
    h2_true = 0.5
    scale = math.sqrt(1.0 + N * h2_true / m)  # sqrt(51)

    z = torch.randn(m, dtype=torch.float64) * scale
    R = torch.eye(m, dtype=torch.float64)

    result = hess_local_h2(z, R, N, [(0, m)])
    h2_hat = result.regions[0].h2_local

    # h2_hat = (sum(z_i^2) - m) / N where z_i ~ N(0, 1+N*h2/m)
    # E[h2_hat] = h2, Var(h2_hat) = 2*m*(1+N*h2/m)^2 / N^2
    # For m=100, the sampling SD is ~ sqrt(2*100*51^2)/10000 ~ 0.072
    # Allow 3 sampling SDs
    sampling_sd = math.sqrt(2.0 * m * scale ** 4) / N
    assert abs(h2_hat - h2_true) < 3 * sampling_sd, (
        f"h2_hat={h2_hat:.4f} too far from h2_true={h2_true} "
        f"(sampling_sd={sampling_sd:.4f})"
    )


# ---------------------------------------------------------------------------
# 3. Eigenvalue regularization
# ---------------------------------------------------------------------------

def test_hess_eigenvalue_regularization():
    """LD matrix with near-singular structure; some eigenvalues pruned."""
    torch.manual_seed(42)
    m = 30
    N = 5_000

    # Build rank-deficient R: R = 0.9 * ones + 0.1 * I
    # Eigenvalues: 0.9*m + 0.1 (once), 0.1 (m-1 times)
    rho = 0.9
    R = rho * torch.ones(m, m, dtype=torch.float64) + (1 - rho) * torch.eye(
        m, dtype=torch.float64
    )
    z = torch.randn(m, dtype=torch.float64)

    # With threshold=1.0, eigenvalues of 0.1 are dropped
    result = hess_local_h2(z, R, N, [(0, m)], eigenvalue_threshold=1.0)
    region = result.regions[0]

    # Only 1 eigenvalue (0.9*30 + 0.1 = 27.1) survives
    assert region.n_eigenvalues_kept == 1
    assert region.n_snps == m

    # With lower threshold, all eigenvalues kept
    result_all = hess_local_h2(z, R, N, [(0, m)], eigenvalue_threshold=0.01)
    assert result_all.regions[0].n_eigenvalues_kept == m


# ---------------------------------------------------------------------------
# 4. Sum of regions equals total
# ---------------------------------------------------------------------------

def test_hess_sum_equals_total():
    """h2_total = sum of per-region h2_local."""
    torch.manual_seed(42)
    m = 80
    N = 10_000
    z = torch.randn(m, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64)

    bounds = [(0, 30), (30, 80)]
    result = hess_local_h2(z, R, N, bounds)

    h2_sum = sum(r.h2_local for r in result.regions)
    assert result.h2_total == pytest.approx(h2_sum, abs=1e-12)
    assert result.n_regions == 2
    assert result.n_snps_total == 80


# ---------------------------------------------------------------------------
# 5. Negative h2 allowed for individual regions
# ---------------------------------------------------------------------------

def test_hess_negative_h2_allowed():
    """Individual regions may have negative h2 estimates (noise)."""
    torch.manual_seed(42)
    m = 20
    N = 100_000
    # z near zero -> z^T z < m -> h2_r < 0
    z = torch.randn(m, dtype=torch.float64) * 0.5
    R = torch.eye(m, dtype=torch.float64)

    result = hess_local_h2(z, R, N, [(0, m)])
    # With z scaled by 0.5, E[z^T z] = 0.25*m, so h2 = (0.25*m - m)/N < 0
    assert result.regions[0].h2_local < 0


# ---------------------------------------------------------------------------
# 6. SE formula verification
# ---------------------------------------------------------------------------

def test_hess_se_formula():
    """For R = I, SE = sqrt(2*m) / N."""
    torch.manual_seed(42)
    m = 60
    N = 8_000
    z = torch.randn(m, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64)

    result = hess_local_h2(z, R, N, [(0, m)])
    se = result.regions[0].h2_local_se

    # tr(R^{-2}) = tr(I) = m for R = I
    expected_se = math.sqrt(2.0 * m) / N
    assert se == pytest.approx(expected_se, rel=1e-10)


# ---------------------------------------------------------------------------
# 7. Local rg with self equals h2
# ---------------------------------------------------------------------------

def test_hess_local_rg_self_equals_h2():
    """hess_local_rg(z, z, R, n, n, ...) matches hess_local_h2(z, R, n, ...)."""
    torch.manual_seed(42)
    m = 40
    N = 10_000
    z = torch.randn(m, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64)
    bounds = [(0, 20), (20, 40)]

    h2_result = hess_local_h2(z, R, N, bounds)
    rg_result = hess_local_rg(z, z, R, N, N, bounds)

    # rg with self: z^T R^{-1} z / sqrt(N*N) = z^T R^{-1} z / N
    # h2: (z^T R^{-1} z - m) / N
    # These differ by m/N, so rg_local = h2_local + m_r/N
    for h2_reg, rg_reg in zip(h2_result.regions, rg_result.regions):
        m_r = h2_reg.n_snps
        expected_rg = h2_reg.h2_local + m_r / N
        assert rg_reg.h2_local == pytest.approx(expected_rg, abs=1e-10)


# ---------------------------------------------------------------------------
# 8. Single region covering all SNPs
# ---------------------------------------------------------------------------

def test_hess_single_region():
    """Single region that covers all SNPs."""
    torch.manual_seed(42)
    m = 50
    N = 10_000
    z = torch.randn(m, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64)

    result = hess_local_h2(z, R, N, [(0, m)], region_labels=["whole_genome"])

    assert result.n_regions == 1
    assert result.n_snps_total == m
    assert result.regions[0].region_id == "whole_genome"
    assert result.regions[0].start == 0
    assert result.regions[0].end == m


# ---------------------------------------------------------------------------
# 9. Empty region_bounds raises ValueError
# ---------------------------------------------------------------------------

def test_hess_empty_region_raises():
    """Empty region_bounds must raise ValueError."""
    z = torch.randn(10, dtype=torch.float64)
    R = torch.eye(10, dtype=torch.float64)

    with pytest.raises(ValueError, match="At least one region"):
        hess_local_h2(z, R, 1000, [])

    with pytest.raises(ValueError, match="At least one region"):
        hess_local_rg(z, z, R, 1000, 1000, [])
