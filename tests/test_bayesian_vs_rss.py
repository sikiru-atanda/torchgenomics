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
