"""GLM link function abstraction for binary, ordinal, and multinomial models.

Provides link(), inverse() (mean function), derivative(), and variance()
methods needed by IRLS fitting and score test computation.

Link functions:
- IdentityLink: y = eta (existing GLM, Gaussian)
- LogitLink: log(mu/(1-mu)) = eta (binary logistic)
- ProbitLink: Phi^{-1}(mu) = eta (binary probit)
- CumulativeLogitLink: logit(P(Y<=j)) = alpha_j - eta (ordinal, proportional odds)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..config import STAT_DTYPE

# ===================================================================
# Base link function
# ===================================================================

class LinkFunction:
    """Base class for GLM link functions."""

    name: str = "base"

    def link(self, mu: Tensor) -> Tensor:
        """Map mean to linear predictor: eta = g(mu)."""
        raise NotImplementedError

    def inverse(self, eta: Tensor) -> Tensor:
        """Map linear predictor to mean: mu = g^{-1}(eta)."""
        raise NotImplementedError

    def derivative(self, mu: Tensor) -> Tensor:
        """Derivative of link: d(eta)/d(mu)."""
        raise NotImplementedError

    def variance(self, mu: Tensor) -> Tensor:
        """Variance function V(mu) for the GLM family."""
        raise NotImplementedError


# ===================================================================
# Identity link (Gaussian)
# ===================================================================

class IdentityLink(LinkFunction):
    """Identity link: eta = mu. Gaussian family."""

    name = "identity"

    def link(self, mu: Tensor) -> Tensor:
        return mu

    def inverse(self, eta: Tensor) -> Tensor:
        return eta

    def derivative(self, mu: Tensor) -> Tensor:
        return torch.ones_like(mu)

    def variance(self, mu: Tensor) -> Tensor:
        return torch.ones_like(mu)


# ===================================================================
# Logit link (binary)
# ===================================================================

class LogitLink(LinkFunction):
    """Logit link: eta = log(mu/(1-mu)). Binomial family."""

    name = "logit"

    def link(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return torch.log(mu_c / (1.0 - mu_c))

    def inverse(self, eta: Tensor) -> Tensor:
        return torch.sigmoid(eta)

    def derivative(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return mu_c * (1.0 - mu_c)


# ===================================================================
# Probit link (binary)
# ===================================================================

class ProbitLink(LinkFunction):
    """Probit link: eta = Phi^{-1}(mu). Binomial family."""

    name = "probit"

    def link(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return math.sqrt(2.0) * torch.erfinv(2.0 * mu_c - 1.0)

    def inverse(self, eta: Tensor) -> Tensor:
        return 0.5 * (1.0 + torch.erf(eta / math.sqrt(2.0)))

    def derivative(self, mu: Tensor) -> Tensor:
        # d(eta)/d(mu) = 1 / phi(Phi^{-1}(mu)) where phi is std normal PDF
        eta = self.link(mu)
        phi = torch.exp(-0.5 * eta ** 2) / math.sqrt(2.0 * math.pi)
        return 1.0 / torch.clamp(phi, min=1e-20)

    def variance(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return mu_c * (1.0 - mu_c)


# ===================================================================
# Cumulative logit link (ordinal / proportional odds)
# ===================================================================

class CumulativeLogitLink(LinkFunction):
    """Cumulative logit link for ordinal regression (proportional odds).

    Model: logit(P(Y <= j)) = alpha_j - X'beta for j = 1, ..., J-1.

    The thresholds alpha are parameters estimated during IRLS.
    This link stores the current thresholds as state.

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories J.
    thresholds : Tensor, optional
        Initial thresholds (J-1,). If None, initialised later.
    """

    name = "cumulative_logit"

    def __init__(self, n_categories: int, thresholds: Tensor | None = None):
        self.n_categories = n_categories
        self.thresholds = thresholds  # (J-1,)

    def cumulative_probs(self, eta: Tensor) -> Tensor:
        """Compute cumulative probabilities P(Y <= j) for each j.

        Parameters
        ----------
        eta : (n,) linear predictor X'beta (without thresholds).

        Returns
        -------
        gamma : (n, J-1) cumulative probabilities.
        """
        if self.thresholds is None:
            raise ValueError("Thresholds not set. Call init_thresholds() first.")
        # gamma_ij = expit(alpha_j - eta_i)
        alpha = self.thresholds.to(dtype=eta.dtype, device=eta.device)
        # (n, J-1) = alpha[None, :] - eta[:, None]
        logits = alpha.unsqueeze(0) - eta.unsqueeze(1)
        return torch.sigmoid(logits)

    def category_probs(self, eta: Tensor) -> Tensor:
        """Compute per-category probabilities P(Y = j).

        Parameters
        ----------
        eta : (n,) linear predictor.

        Returns
        -------
        pi : (n, J) category probabilities.
        """
        gamma = self.cumulative_probs(eta)  # (n, J-1)
        n = eta.shape[0]
        J = self.n_categories

        pi = torch.zeros(n, J, dtype=eta.dtype, device=eta.device)
        # P(Y=0) = gamma_0
        pi[:, 0] = gamma[:, 0]
        # P(Y=j) = gamma_j - gamma_{j-1} for j=1..J-2
        for j in range(1, J - 1):
            pi[:, j] = gamma[:, j] - gamma[:, j - 1]
        # P(Y=J-1) = 1 - gamma_{J-2}
        pi[:, J - 1] = 1.0 - gamma[:, J - 2]

        return torch.clamp(pi, min=1e-10, max=1.0 - 1e-10)

    def init_thresholds(self, Y: Tensor, dtype=STAT_DTYPE, device=None):
        """Initialise thresholds from observed category frequencies.

        Uses probit quantiles of cumulative frequencies.
        """
        J = self.n_categories
        device = device or Y.device
        counts = torch.zeros(J, dtype=dtype, device=device)
        for j in range(J):
            counts[j] = (j == Y).sum().float()
        freq = counts / counts.sum()
        cum_freq = torch.cumsum(freq, dim=0)[:-1]  # (J-1,)
        cum_freq = torch.clamp(cum_freq, min=0.01, max=0.99)
        # Inverse logit of cumulative frequencies
        self.thresholds = torch.log(cum_freq / (1.0 - cum_freq)).to(dtype)

    # For compatibility with IRLS, these operate on the binary (cumulative) level
    def link(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return torch.log(mu_c / (1.0 - mu_c))

    def inverse(self, eta: Tensor) -> Tensor:
        return torch.sigmoid(eta)

    def derivative(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return 1.0 / (mu_c * (1.0 - mu_c))

    def variance(self, mu: Tensor) -> Tensor:
        mu_c = torch.clamp(mu, min=1e-10, max=1.0 - 1e-10)
        return mu_c * (1.0 - mu_c)
