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

from dataclasses import dataclass
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


class BayesianVSRss:
    """SuSiE-RSS fine-mapping on summary statistics.

    Per NA1 design spec section 3.4 public API. The fit_rss method runs IBSS
    to convergence using the SER posterior from ser_posterior above.

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

    def __init__(
        self,
        max_num_causal: int = 10,
        coverage: float = 0.95,
        purity: float = 0.5,
        sigma_prior_sq: float = 0.04,
        max_iter: int = 100,
        tol: float = 1e-6,
        block_size_threshold: int = 5000,
    ):
        self.max_num_causal = max_num_causal
        self.coverage = coverage
        self.purity = purity
        self.sigma_prior_sq = sigma_prior_sq
        self.max_iter = max_iter
        self.tol = tol
        self.block_size_threshold = block_size_threshold

    def fit_rss(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        prior_pi_per_snp: Optional[torch.Tensor] = None,
    ) -> BayesianVSRssResult:
        """Run SuSiE-RSS IBSS to convergence on a single locus.

        Args:
            z: Per-variant z-scores, shape (p,).
            R: LD correlation matrix, shape (p, p).
            n: GWAS sample size.
            prior_pi_per_snp: Optional per-SNP prior, shape (p,). D3 shim from
                Phase 59 PolyFun spec; if None, uses uniform 1/p.

        Returns:
            BayesianVSRssResult with alpha, mu, sigma_sq, pip, beta_mean,
            beta_sd, elbo, elbo_history, credible_sets, converged, n_iter.
        """
        z = z.to(torch.float64)
        R = R.to(torch.float64)
        p = z.shape[0]
        L = self.max_num_causal
        sqrt_n = math.sqrt(n)

        # Internal effects work on the "scaled" basis b_eff = sqrt(n) * beta so
        # that the IBSS residual update z - R @ b_eff_others is dimensionally
        # consistent with the z-scale data (z = R @ b_eff + noise). This matches
        # the convention used by susieR::susie_rss internally. Returned mu and
        # sigma_sq are converted back to the regression-coefficient (beta) scale.
        alpha = torch.zeros(L, p, dtype=torch.float64)
        mu = torch.zeros(L, p, dtype=torch.float64)        # beta-scale (returned)
        sigma_sq = torch.zeros(L, p, dtype=torch.float64)  # beta-scale (returned)
        b_eff = torch.zeros(L, p, dtype=torch.float64)     # z-scale: b_eff = sqrt(n)*beta

        elbo_history: list[float] = []

        for iteration in range(self.max_iter):
            for l in range(L):
                # Residual on z-scale: tilde_z_l = z - R @ sum_{l'!=l} b_eff_{l'}
                tilde_z_l = ibss_residual_update(z, R, b_eff, layer_idx=l)
                # SER posterior (beta-scale)
                sigma_sq_l, mu_l, log_bf_l = ser_posterior(
                    tilde_z_l, R, n, self.sigma_prior_sq
                )
                # Softmax with per-SNP prior (D3 shim)
                alpha_l = compute_alpha(log_bf_l, prior_pi=prior_pi_per_snp)

                alpha[l] = alpha_l
                mu[l] = mu_l
                sigma_sq[l] = sigma_sq_l
                # b_eff = sqrt(n) * beta-scale posterior mean for this layer
                b_eff[l] = alpha_l * (sqrt_n * mu_l)

            # Compute ELBO at end of iteration (uses scaled effects internally)
            elbo = self._compute_elbo(z, R, n, alpha, mu, sigma_sq)
            elbo_history.append(elbo)

            # Convergence check
            if iteration > 0:
                elbo_diff = elbo_history[-1] - elbo_history[-2]
                if abs(elbo_diff) < self.tol:
                    converged = True
                    break
        else:
            converged = False

        elbo_history_tensor = torch.tensor(elbo_history, dtype=torch.float64)

        # Per-variant PIP, beta_mean, beta_sd
        pip = 1.0 - torch.prod(1.0 - alpha, dim=0)
        beta_mean = (alpha * mu).sum(dim=0)
        # Per-variant variance: var(beta) = sum_l (alpha_lj * (mu_lj^2 + sigma_lj^2)) - beta_mean^2
        # This is the law of total variance applied across the L single-effect components.
        second_moment = (alpha * (mu ** 2 + sigma_sq)).sum(dim=0)
        beta_var = second_moment - beta_mean ** 2
        beta_var = beta_var.clamp_min(0.0)  # numerical floor
        beta_sd = beta_var.sqrt()

        credible_sets = self._build_credible_sets(alpha, R)

        return BayesianVSRssResult(
            alpha=alpha,
            mu=mu,
            sigma_sq=sigma_sq,
            pip=pip,
            beta_mean=beta_mean,
            beta_sd=beta_sd,
            elbo=float(elbo_history_tensor[-1]),
            elbo_history=elbo_history_tensor,
            credible_sets=credible_sets,
            converged=converged,
            n_iter=iteration + 1,
        )

    def _build_credible_sets(
        self,
        alpha: torch.Tensor,
        R: torch.Tensor,
    ) -> list[Tuple[int, list[int]]]:
        """Construct credible sets per layer, with purity filtering.

        Per Wang et al. 2020 [C1] section 3.4:
        1. Sort alpha[l] descending.
        2. Take the smallest set whose cumulative alpha >= self.coverage.
        3. Enforce purity: minimum pairwise |R_jk| within the set >= self.purity.
           If purity check fails, drop the credible set.

        Args:
            alpha: Per-layer per-variant inclusion probabilities, shape (L, p).
            R: LD correlation matrix, shape (p, p).

        Returns:
            List of (layer_idx, sorted_member_indices) tuples for credible sets
            that pass the purity check.
        """
        L, p = alpha.shape
        cs_list: list[Tuple[int, list[int]]] = []

        for l in range(L):
            alpha_l = alpha[l]
            # Sort descending
            sorted_alpha, sorted_indices = torch.sort(alpha_l, descending=True)
            cumsum = torch.cumsum(sorted_alpha, dim=0)
            # First index where cumsum >= coverage
            mask = cumsum >= self.coverage
            if not mask.any():
                continue  # Not enough alpha mass to form a credible set
            cutoff = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
            members = sorted_indices[:cutoff].tolist()

            # Purity check: min pairwise |R_jk| over members
            if len(members) > 1:
                sub_R = R[members][:, members]
                # Off-diagonal absolute values
                off_diag_mask = ~torch.eye(len(members), dtype=torch.bool, device=R.device)
                min_abs_corr = sub_R.abs()[off_diag_mask].min().item()
                if min_abs_corr < self.purity:
                    continue  # Purity failed; drop CS

            cs_list.append((l, members))

        return cs_list

    def _compute_elbo(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
    ) -> float:
        """Compute the ELBO for SuSiE-RSS.

        Per Zou et al. 2022 [C4] section A.2 supplementary. The ELBO has the
        form ELBO = E[log p(z | b, sigma^2)] - KL[q(b, sigma^2) || p(b, sigma^2)].

        For the per-layer single-effect prior, the KL term decomposes per layer:
            KL_l = sum_j alpha_lj * (log(alpha_lj * p) - 0.5 * (1 + log(sigma_lj^2 / sigma_prior^2) - mu_lj^2 / sigma_prior^2 - sigma_lj^2 / sigma_prior^2))

        Returns:
            elbo: Scalar ELBO value (float).
        """
        p = z.shape[0]
        L = alpha.shape[0]

        # SuSiE-RSS model (Zou 2022 [C4] eq. 2-3):
        #   z | beta ~ N(sqrt(n) R beta, R)
        # Internally we work with the scaled effect b_eff = sqrt(n) * beta so
        # that the model becomes z ~ N(R b_eff, R) and the likelihood quadratic
        # form aligns directly with the IBSS residual semantics (z - R b_eff_others).
        sqrt_n = math.sqrt(n)
        mu_eff = sqrt_n * mu              # scaled posterior mean (z-scale)
        sigma_eff_sq = n * sigma_sq       # scaled posterior variance (z-scale)

        # Per-layer scaled posterior mean
        b_l = alpha * mu_eff               # (L, p)
        b_total = b_l.sum(dim=0)           # (p,)
        z_pred = R @ b_total

        # Likelihood term: E_q[log p(z | beta)] = -0.5 * E_q[(z - R b_eff)^T R^{-1} (z - R b_eff)] + const.
        # Per Zou 2022 [C4] §A.2 the canonical SuSiE-RSS likelihood under z | beta ~ N(R b_eff, R)
        # uses the R^{-1} Mahalanobis weighting. Solve R^{-1}(z - z_pred) via torch.linalg.solve
        # for numerical stability (avoids explicit matrix inverse; equivalent but better-conditioned).
        residual = z - z_pred
        R_inv_residual = torch.linalg.solve(R, residual)
        squared_residual = (residual * R_inv_residual).sum()

        # Variance trace term under the R^{-1}-weighted likelihood:
        #   E_q[(R b_l)^T R^{-1} (R b_l)] = E[b_l]^T R E[b_l] + tr(R Var_q(b_l))
        # Decomposing the second moment over layers gives the variance penalty
        #   sum_l (E_q[b_l^T R b_l] - E[b_l]^T R E[b_l]) = sum_l tr(R Var_q(b_l)) >= 0.
        # Under the per-layer SER mean-field factorization:
        #   Var_q(b_l[j]) = alpha_lj * (mu_eff_lj^2 + sigma_eff_lj^2) - (alpha_lj * mu_eff_lj)^2
        # and tr(R diag(v)) = sum_j R_jj * v_j.
        R_diag = torch.diag(R)  # (p,)
        b_eff_var = (
            alpha * (mu_eff ** 2 + sigma_eff_sq) - (alpha * mu_eff) ** 2
        )  # (L, p)
        var_penalty = (R_diag * b_eff_var).sum()
        log_lik = (-0.5 * (squared_residual + var_penalty)).item()

        # KL for the categorical inclusion (alpha vs uniform 1/p)
        # KL_cat = sum_l sum_j alpha_lj * log(alpha_lj * p)
        alpha_safe = alpha.clamp_min(1e-300)
        kl_cat = (alpha * (torch.log(alpha_safe) + math.log(p))).sum().item()

        # KL for the Gaussian prior on the effect, per layer
        # KL_gauss_l = 0.5 * sum_j alpha_lj * (mu_lj^2 / sigma_prior^2 + sigma_lj^2 / sigma_prior^2 - 1 - log(sigma_lj^2 / sigma_prior^2))
        sigma_prior_sq = self.sigma_prior_sq
        kl_gauss = (
            0.5 * alpha * (
                mu ** 2 / sigma_prior_sq
                + sigma_sq / sigma_prior_sq
                - 1.0
                - torch.log(sigma_sq / sigma_prior_sq)
            )
        ).sum().item()

        return log_lik - kl_cat - kl_gauss


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
        b: Per-layer effect vectors on z-score scale, shape (L, p), where
            b[l] = sqrt(n) * alpha[l] * mu[l] (the IBSS convention used by
            fit_rss; mu and alpha returned in BayesianVSRssResult are on the
            beta scale and must be rescaled by sqrt(n) before passing here).
        layer_idx: Index of the layer being updated (excluded from the sum).

    Returns:
        tilde_z_l: Residual z-scores for layer layer_idx, shape (p,).
    """
    # Sum over all layers except layer_idx
    mask = torch.ones(b.shape[0], dtype=torch.bool, device=b.device)
    mask[layer_idx] = False
    other_layers_sum = b[mask].sum(dim=0)  # shape (p,)
    return z - R @ other_layers_sum
