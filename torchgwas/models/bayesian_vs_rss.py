"""SuSiE-RSS fine-mapping on summary statistics.

Implements the algorithm of Zou, Carbonetto, Wang, Stephens (2022) PLOS Genet
18(7):e1010299 - SuSiE on summary statistics + LD reference. This is a new
class separate from BayesianVS (raw-G SuSiE) to preserve all 30+ existing
test_bayesian_vs.py parity tests with zero modification.

Per NA1 design spec docs/superpowers/specs/2026-05-11-na1-susie-streaming-design.md.

References:
  [C1] Wang et al. 2020 JRSS-B 82(5):1273-1300 - SuSiE / IBSS
  [C4] Zou et al. 2022 PLOS Genet 18(7):e1010299 - SuSiE-RSS canonical derivation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math
import torch


@dataclass
class BayesianVSRssResult:
    """Return type of BayesianVSRss.fit_rss.

    Attributes:
        alpha: Per-layer per-variant inclusion probabilities, shape (L, p).
        mu: Per-layer per-variant posterior effect means, shape (L, p).
        sigma_sq: Per-layer per-variant posterior effect variances, shape (L, p).
        pip: Per-variant posterior inclusion probabilities, shape (p,).
            pip[j] = 1 - prod_l(1 - alpha[l, j]) per Wang et al. 2020 [C1] eq. 12.
        beta_mean: Per-variant posterior effect means, shape (p,).
            beta_mean[j] = sum_l alpha[l, j] * mu[l, j].
        beta_sd: Per-variant posterior effect standard deviations, shape (p,).
        elbo: ELBO at convergence (scalar).
        elbo_history: ELBO trace per IBSS iteration, shape (T,).
        credible_sets: List of (layer_idx, list_of_variant_idx) per credible set.
        converged: True if IBSS converged within max_iter.
        n_iter: Number of IBSS iterations performed.
    """
    alpha: torch.Tensor
    mu: torch.Tensor
    sigma_sq: torch.Tensor
    pip: torch.Tensor
    beta_mean: torch.Tensor
    beta_sd: torch.Tensor
    elbo: float
    elbo_history: torch.Tensor
    credible_sets: list[Tuple[int, list[int]]]
    converged: bool
    n_iter: int


def ser_posterior(
    z: torch.Tensor,
    R: torch.Tensor,
    n: int,
    sigma_prior_sq: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-variant Single Effect Regression posterior on summary stats.

    Per Zou et al. 2022 [C4] eq. 8-10:
        sigma_lj^2 = 1 / (R_jj / sigma_prior^2 + n)
        mu_lj = sigma_lj^2 * z_j * sqrt(n)
        log_BF_lj = 0.5 * log(sigma_lj^2 / sigma_prior^2)
                  + 0.5 * mu_lj^2 / sigma_lj^2

    Args:
        z: Per-variant z-scores, shape (p,) or (..., p).
        R: LD correlation matrix, shape (p, p). Only the diagonal is used here;
            off-diagonal enters via the IBSS residual update (Task 6).
        n: GWAS sample size.
        sigma_prior_sq: Prior variance of the single effect (sigma_lprior^2).

    Returns:
        (sigma_sq, mu, log_bf) tuple, each shape matching z.
    """
    R_diag = torch.diag(R)
    sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    mu = sigma_sq * z * math.sqrt(n)
    log_bf = (
        0.5 * torch.log(sigma_sq / sigma_prior_sq)
        + 0.5 * mu ** 2 / sigma_sq
    )
    return sigma_sq, mu, log_bf


@dataclass
class BayesianVSRss:
    """SuSiE-RSS fine-mapping on summary statistics.

    Per NA1 design spec section 3.4 public API. The fit_rss method (Task 6+)
    runs IBSS to convergence using the SER posterior from ser_posterior above.

    Args:
        max_num_causal: L (single-effect layers, default 10 per [C4]).
        coverage: credible-set coverage threshold (default 0.95 per [C1]).
        purity: minimum |R_jk| within a credible set (default 0.5,
            matches susieR::susie_rss(min_abs_corr=0.5)).
        sigma_prior_sq: prior variance of single effects (default 0.04).
        max_iter: IBSS max iterations (default 100).
        tol: ELBO convergence tolerance (default 1e-6).
        block_size_threshold: max p per block before decomposition kicks in
            (default 5000 per PolyFun [C5] / ldetect convention).
    """
    max_num_causal: int = 10
    coverage: float = 0.95
    purity: float = 0.5
    sigma_prior_sq: float = 0.04
    max_iter: int = 100
    tol: float = 1e-6
    block_size_threshold: int = 5000

    # Methods fit_rss, _run_ibss, _compute_elbo, _build_credible_sets
    # are added in Tasks 6-9.


def compute_alpha(
    log_bf: torch.Tensor,
    prior_pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-variant inclusion probabilities via prior-weighted softmax.

    Per Zou et al. 2022 [C4] eq. 10 and the D3 shim from Phase 59 PolyFun spec:
        alpha_lj = pi_j * exp(log_BF_lj) / sum_j' pi_j' * exp(log_BF_lj')

    With prior_pi=None, uses uniform 1/p — recovers exact softmax over log_BF.
    With prior_pi provided as a tensor, normalizes internally to sum to 1.0
    within the locus, then weights the softmax accordingly.

    Args:
        log_bf: Per-variant log Bayes factors, shape (p,).
        prior_pi: Optional per-variant prior inclusion probabilities, shape (p,).
            If None, uniform 1/p is used.

    Returns:
        alpha: Per-variant inclusion probabilities, shape (p,), summing to 1.0.
    """
    if prior_pi is None:
        return torch.softmax(log_bf, dim=0)

    # Normalize prior_pi to sum to 1.0 within the locus
    prior_pi_normalized = prior_pi / prior_pi.sum()
    log_prior = torch.log(prior_pi_normalized.clamp_min(1e-300))
    return torch.softmax(log_bf + log_prior, dim=0)


def ibss_residual_update(
    z: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute the IBSS residual for a given layer.

    Per Zou et al. 2022 [C4] eq. 11:
        tilde_z_l = z - R @ sum_{l' != l} b_{l'}

    Args:
        z: Per-variant z-scores, shape (p,).
        R: LD correlation matrix, shape (p, p).
        b: Per-layer effect vectors, shape (L, p), where b[l] = alpha[l] * mu[l].
        layer_idx: Index of the layer being updated (excluded from the sum).

    Returns:
        tilde_z_l: Residual z-scores for layer layer_idx, shape (p,).
    """
    # Sum over all layers except layer_idx
    mask = torch.ones(b.shape[0], dtype=torch.bool, device=b.device)
    mask[layer_idx] = False
    other_layers_sum = b[mask].sum(dim=0)  # shape (p,)
    return z - R @ other_layers_sum
