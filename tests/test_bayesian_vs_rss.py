"""Tier 1 unit tests for BayesianVSRss (SuSiE-RSS algorithm).

Per NA1 design spec section 5.1. Tests ground truth in the Zou et al. 2022
PLOS Genet equations 8-10 (SER posterior) and equation 11 (IBSS update).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchgwas.models.bayesian_vs_rss import (
    BayesianVSRss,
    BayesianVSRssResult,
    ser_posterior,
)


def test_ser_posterior_closed_form_p_equals_3():
    """SER posterior closed-form on a tiny p=3 fixture.

    Per Zou et al. 2022 [C4] equations 8-10:
        sigma_lj^2 = 1 / (R_jj / sigma_prior^2 + n)
        mu_lj = sigma_lj^2 * z_j * sqrt(n)
        log_BF_lj = 0.5 * log(sigma_lj^2 / sigma_prior^2)
                  + 0.5 * mu_lj^2 / sigma_lj^2
    """
    z = torch.tensor([1.5, -2.0, 0.3], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)  # diagonal R: R_jj = 1
    n = 1000
    sigma_prior_sq = 0.04

    sigma_sq, mu, log_bf = ser_posterior(z, R, n, sigma_prior_sq)

    # Closed-form expected values
    R_diag = torch.diag(R)
    expected_sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    expected_mu = expected_sigma_sq * z * math.sqrt(n)
    expected_log_bf = 0.5 * torch.log(expected_sigma_sq / sigma_prior_sq) + \
                      0.5 * expected_mu ** 2 / expected_sigma_sq

    assert torch.allclose(sigma_sq, expected_sigma_sq, atol=1e-12)
    assert torch.allclose(mu, expected_mu, atol=1e-12)
    assert torch.allclose(log_bf, expected_log_bf, atol=1e-12)


def test_ser_posterior_with_nondiagonal_R():
    """SER posterior uses R_jj from the diagonal regardless of off-diagonal structure."""
    z = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    # Diagonal R values are different
    R = torch.tensor(
        [[1.0, 0.5, 0.3], [0.5, 0.8, 0.1], [0.3, 0.1, 0.6]],
        dtype=torch.float64,
    )
    n = 500
    sigma_prior_sq = 0.1

    sigma_sq, mu, log_bf = ser_posterior(z, R, n, sigma_prior_sq)

    R_diag = torch.diag(R)
    expected_sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    assert torch.allclose(sigma_sq, expected_sigma_sq, atol=1e-12)


def test_bayesian_vs_rss_class_constructs():
    """BayesianVSRss constructs with default args."""
    model = BayesianVSRss(
        max_num_causal=10,
        coverage=0.95,
        purity=0.5,
    )
    assert model.max_num_causal == 10
    assert model.coverage == 0.95
    assert model.purity == 0.5


from torchgwas.models.bayesian_vs_rss import compute_alpha


def test_softmax_uniform_prior_recovers_softmax_over_bf():
    """With uniform prior, alpha is exact softmax over log_BF."""
    log_bf = torch.tensor([0.0, 1.0, 2.0, 0.5], dtype=torch.float64)
    alpha = compute_alpha(log_bf, prior_pi=None)
    expected = torch.softmax(log_bf, dim=0)
    assert torch.allclose(alpha, expected, atol=1e-12)


def test_softmax_uniform_prior_sums_to_one():
    """alpha sums to 1.0."""
    log_bf = torch.tensor([0.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    alpha = compute_alpha(log_bf, prior_pi=None)
    assert torch.isclose(alpha.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_extreme_prior_dominates_alpha():
    """An extreme per-SNP prior on one SNP drives alpha toward that SNP."""
    log_bf = torch.zeros(4, dtype=torch.float64)  # uniform BFs
    prior_pi = torch.tensor([0.99, 1e-3, 1e-3, 1e-3 - 1e-12], dtype=torch.float64)
    # Normalize prior_pi to sum to 1.0 inside compute_alpha
    alpha = compute_alpha(log_bf, prior_pi=prior_pi)
    # Approx: alpha[0] ~ 0.99 / (0.99 + 3e-3) ~ 0.997
    assert alpha[0] > 0.99


def test_per_snp_prior_normalized_internally():
    """Unnormalized prior is normalized internally; equivalent to normalized input."""
    log_bf = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    prior_unnorm = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
    prior_norm = prior_unnorm / prior_unnorm.sum()
    alpha_unnorm = compute_alpha(log_bf, prior_pi=prior_unnorm)
    alpha_norm = compute_alpha(log_bf, prior_pi=prior_norm)
    assert torch.allclose(alpha_unnorm, alpha_norm, atol=1e-12)


from torchgwas.models.bayesian_vs_rss import ibss_residual_update


def test_ibss_residual_subtracts_other_layers():
    """IBSS residual: tilde_z_l = z - R @ sum_{l' != l} b_{l'}.

    Per Zou et al. 2022 [C4] eq. 11.
    """
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    R = torch.tensor(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]],
        dtype=torch.float64,
    )
    # Two layers, each with effects b_l = alpha_l * mu_l
    b = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        dtype=torch.float64,
    )  # shape (L=2, p=3)
    # Tilde z for layer 0: z - R @ b[1] = z - R @ [0, 0.2, 0]
    expected_l0 = z - R @ b[1]
    tilde_z_l0 = ibss_residual_update(z, R, b, layer_idx=0)
    assert torch.allclose(tilde_z_l0, expected_l0, atol=1e-12)

    # Tilde z for layer 1: z - R @ b[0] = z - R @ [0.1, 0, 0]
    expected_l1 = z - R @ b[0]
    tilde_z_l1 = ibss_residual_update(z, R, b, layer_idx=1)
    assert torch.allclose(tilde_z_l1, expected_l1, atol=1e-12)


def test_ibss_elbo_monotone_non_decreasing():
    """ELBO is non-decreasing per IBSS iteration (within numerical tolerance).

    Per Zou et al. 2022 [C4] section A.2 supplementary.
    """
    rng = np.random.default_rng(13)
    p = 30
    n = 500
    # Synthetic z-scores with one strong signal
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64))
    z[10] += 5.0  # planted causal at index 10
    # Identity R (independent SNPs) — IBSS should converge in few iterations
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=3, max_iter=50, tol=1e-8)
    result = model.fit_rss(z=z, R=R, n=n)

    elbo_history = result.elbo_history
    # Within numerical noise (1e-9), ELBO is non-decreasing
    diffs = elbo_history[1:] - elbo_history[:-1]
    assert (diffs >= -1e-9).all(), f"ELBO decreased: min diff = {diffs.min().item()}"


def test_ibss_elbo_monotone_non_decreasing_ar1_R():
    """ELBO is non-decreasing per IBSS iteration on AR(1) LD matrix.

    Critical regression test: the original prescribed _compute_elbo was
    monotone only on R=I (the original fixture). The reviewer of Task 7
    empirically demonstrated non-monotonicity (min diff = -5.35) on
    non-identity R, which would silently fail Tier 2 parity vs susieR.
    This test catches that regression.
    """
    rng = np.random.default_rng(31)
    p = 30
    n = 500
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64))
    z[10] += 5.0  # planted causal at index 10
    # AR(1) R with rho=0.7
    rho = 0.7
    indices = torch.arange(p, dtype=torch.float64)
    R = rho ** torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    # Add small jitter for numerical stability of torch.linalg.solve
    R = R + 1e-6 * torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=3, max_iter=50, tol=1e-8)
    result = model.fit_rss(z=z, R=R, n=n)

    elbo_history = result.elbo_history
    diffs = elbo_history[1:] - elbo_history[:-1]
    # Allow tiny numerical noise (1e-6) but no real regression
    assert (diffs >= -1e-6).all(), \
        f"ELBO decreased on AR(1) R: min diff = {diffs.min().item()}"


def test_ibss_recovers_planted_causal():
    """IBSS recovers a planted causal SNP at high PIP."""
    rng = np.random.default_rng(17)
    p = 50
    n = 1000
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64) * 0.5)
    causal_idx = 25
    z[causal_idx] = 6.0  # very strong signal
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=5)
    result = model.fit_rss(z=z, R=R, n=n)

    assert result.pip[causal_idx] > 0.9, \
        f"Planted causal at index {causal_idx} had PIP {result.pip[causal_idx]:.4f}"


def test_pip_formula():
    """PIP = 1 - prod_l (1 - alpha[l]) per Wang et al. 2020 [C1] eq. 12."""
    p = 5
    L = 3
    rng = np.random.default_rng(23)
    alpha = torch.from_numpy(rng.uniform(0.0, 0.3, (L, p)).astype(np.float64))
    expected_pip = 1.0 - torch.prod(1.0 - alpha, dim=0)

    # Build a fixture result with known alpha
    z = torch.zeros(p, dtype=torch.float64)
    R = torch.eye(p, dtype=torch.float64)
    model = BayesianVSRss(max_num_causal=L, max_iter=1)
    result = model.fit_rss(z=z, R=R, n=100)
    # The internal pip computation must match the formula on result.alpha
    actual_pip = 1.0 - torch.prod(1.0 - result.alpha, dim=0)
    assert torch.allclose(result.pip, actual_pip, atol=1e-12)


def test_credible_set_single_strong_layer():
    """A layer with one dominant alpha produces a credible set of one variant."""
    p = 5
    alpha = torch.zeros(1, p, dtype=torch.float64)
    alpha[0, 2] = 0.99
    alpha[0] += 0.01 / p
    alpha[0, 2] -= 0.01 / p  # keep total = 1.0 (approx)
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=1, coverage=0.95, purity=0.5)
    cs = model._build_credible_sets(alpha, R)
    assert len(cs) == 1
    layer_idx, members = cs[0]
    assert layer_idx == 0
    assert members == [2]


def test_credible_set_purity_filters_low_correlation():
    """If purity check fails, the credible set is dropped (or shrunk)."""
    p = 4
    alpha = torch.zeros(1, p, dtype=torch.float64)
    # Two SNPs share alpha but are uncorrelated
    alpha[0, 0] = 0.55
    alpha[0, 3] = 0.45
    R = torch.eye(p, dtype=torch.float64)  # zero off-diagonal correlation

    model = BayesianVSRss(max_num_causal=1, coverage=0.95, purity=0.5)
    cs = model._build_credible_sets(alpha, R)
    # The naive cumulative-coverage set would be {0, 3} but purity = 0
    # (R[0, 3] = 0), so the credible set should be empty (purity-failed)
    # OR truncated to just the top variant. Per susieR, the entire CS is
    # dropped when purity fails; mirror that.
    assert len(cs) == 0 or (len(cs) == 1 and len(cs[0][1]) <= 1)


def test_credible_set_sorted_by_pip():
    """Variants within a credible set are ordered by descending alpha."""
    p = 4
    alpha = torch.zeros(2, p, dtype=torch.float64)
    alpha[0, 0] = 0.5
    alpha[0, 1] = 0.4
    alpha[0, 2] = 0.1
    alpha[1, 3] = 0.95
    alpha[1, 0] = 0.05
    R = torch.tensor(
        [[1.0, 0.7, 0.0, 0.0],
         [0.7, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    model = BayesianVSRss(max_num_causal=2, coverage=0.9, purity=0.5)
    cs = model._build_credible_sets(alpha, R)

    # Layer 0: cumsum 0.5 + 0.4 = 0.9 → CS = [0, 1]; purity |R[0,1]| = 0.7 ≥ 0.5 → kept
    # Layer 1: alpha[3] = 0.95 ≥ 0.9 → CS = [3]
    # Both layers should produce a credible set
    assert len(cs) == 2
    # Within CS, members ordered by descending alpha
    layers = {layer_idx: members for layer_idx, members in cs}
    assert layers[0] == [0, 1]  # 0.5 > 0.4
    assert layers[1] == [3]
