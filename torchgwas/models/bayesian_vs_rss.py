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
