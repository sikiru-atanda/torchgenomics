"""Tests for GLM link function abstraction."""

from __future__ import annotations

import math
import pytest
import torch

from torchgwas.models.glm_link import (
    LogitLink, ProbitLink, CumulativeLogitLink, IdentityLink,
)


class TestLogitLink:
    def test_inverse_of_link(self):
        link = LogitLink()
        mu = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        eta = link.link(mu)
        mu_recovered = link.inverse(eta)
        assert torch.allclose(mu, mu_recovered, atol=1e-10)

    def test_derivative_positive(self):
        link = LogitLink()
        mu = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float64)
        d = link.derivative(mu)
        assert (d > 0).all()

    def test_variance_correct(self):
        link = LogitLink()
        mu = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
        v = link.variance(mu)
        expected = mu * (1.0 - mu)
        assert torch.allclose(v, expected, atol=1e-12)


class TestProbitLink:
    def test_inverse_of_link(self):
        link = ProbitLink()
        mu = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        eta = link.link(mu)
        mu_recovered = link.inverse(eta)
        assert torch.allclose(mu, mu_recovered, atol=1e-6)

    def test_symmetric_at_half(self):
        link = ProbitLink()
        mu = torch.tensor([0.5], dtype=torch.float64)
        assert abs(link.link(mu).item()) < 1e-10


class TestCumulativeLogitLink:
    def test_cumulative_probs_ordered(self):
        link = CumulativeLogitLink(n_categories=4)
        link.thresholds = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
        eta = torch.zeros(5, dtype=torch.float64)
        gamma = link.cumulative_probs(eta)
        # Cumulative probs should be increasing across columns
        for j in range(gamma.shape[1] - 1):
            assert (gamma[:, j] <= gamma[:, j + 1] + 1e-10).all()

    def test_category_probs_sum_to_one(self):
        link = CumulativeLogitLink(n_categories=3)
        link.thresholds = torch.tensor([-0.5, 0.5], dtype=torch.float64)
        eta = torch.randn(10, dtype=torch.float64)
        pi = link.category_probs(eta)
        sums = pi.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_init_thresholds(self):
        link = CumulativeLogitLink(n_categories=3)
        Y = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.float64)
        link.init_thresholds(Y)
        assert link.thresholds.shape == (2,)
        assert link.thresholds[0] < link.thresholds[1]
