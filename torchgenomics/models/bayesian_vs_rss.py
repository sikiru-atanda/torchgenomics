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

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .._dispatch import native_disabled
from .._native import HAS_NATIVE_IBSS, _ibss_native


@dataclass
class BayesianVSRssResult:
    """Return type of BayesianVSRss.fit_rss.

    Attributes:
        alpha: Per-layer per-variant inclusion probabilities, shape (L, p).
        mu: Per-layer per-variant posterior effect means, shape (L, p).
        sigma_sq: Per-layer per-variant posterior effect variances, shape (L, p).
        V: Per-layer estimated prior variance, shape (L,). When
            estimate_prior_variance=True (default), V_l is updated each
            iteration via the EM M-step. Layers without real signal get
            V_l → 0 ("shut off"). Mirrors susieR's `fit$V`.
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
    V: torch.Tensor
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


def apply_z_score_adjustment(z: torch.Tensor, n: int) -> torch.Tensor:
    """Map sample-based marginal z-scores onto SuSiE-RSS's assumed Var(y)=1 scale.

    The SuSiE-RSS likelihood (Zhu & Stephens 2017; Zou et al. 2022 [C4]) is
    derived under the assumption that the marginal z-scores are on the
    asymptotic-normal scale with Var(y) = 1. When the user supplies z-scores
    computed from a finite-sample marginal regression where Var(y) is
    estimated from the data (so the underlying statistic follows a
    t-distribution with df = n - 2), the marginal z is on a slightly
    inflated scale relative to what the model assumes. The canonical
    correction (susieR `susie_rss()`, lines ~36-39 in source 0.14.2) is

        adj   = (n - 1) / (z^2 + n - 2)
        z_adj = sqrt(adj) * z

    For large n the factor approaches 1 and the correction vanishes; for
    finite n with large |z| the correction shrinks z toward 0 by a small
    amount (e.g., ~5 percent for |z| = 5, n = 279). On the MDP real-data
    fixture this single correction takes the β_sd Pearson vs susieR from
    0.99441 to 0.999999 (see docs/validation_findings.md, "Update
    2026-05-12: structural-residual investigation").

    Args:
        z: Per-variant z-scores, shape (p,).
        n: GWAS sample size used to compute z.

    Returns:
        z_adj: Adjusted z-scores on the assumed-Var(y)=1 scale, same shape as z.
    """
    adj = (n - 1.0) / (z ** 2 + n - 2.0)
    return torch.sqrt(adj) * z


def find_optimal_V(
    z: torch.Tensor,
    R: torch.Tensor,
    n: int,
    V_init: float,
    prior_pi: Optional[torch.Tensor] = None,
    upper_bound_factor: float = 100.0,
) -> float:
    """Find V that maximizes the marginal log-likelihood for one SER layer.

    Mirrors susieR's `estimate_prior_variance="optim"` path. Per Wang et al.
    2020 [C1] §3.2 and Zou et al. 2022 [C4] eq. 8-10, the marginal log-
    likelihood for a single-effect layer at prior variance V is

        L(V) = log( sum_j pi_j * BF_j(V) )
             = logsumexp_j( log_BF_j(V) + log pi_j )

    with the convention L(0) = 0 (BF_j(0) = 1 since the prior collapses to
    a delta at zero). The optim path computes V_l = argmax_V L(V) directly
    via 1D bounded optimization on log V and snaps to V = 0 when the null
    hypothesis maximizes (L(V_opt) <= 0). This is an exact-evidence test
    and avoids the snap-threshold tradeoff of the EM path: noise-floor V
    fixed points around p^{-1}/n still register positive marginal evidence
    under EM but lose to V = 0 under optim.

    Args:
        z: Residual z-scores for this layer, shape (p,).
        R: LD correlation matrix, shape (p, p). Only diag(R) is read here.
        n: GWAS sample size.
        V_init: Initial V scale (e.g., previous-iteration V_l, or
            `sigma_prior_sq`). Sets the search upper bound.
        prior_pi: Optional per-SNP prior, shape (p,). Defaults to uniform 1/p.
        upper_bound_factor: Search [eps, V_init * factor]. Default 100x is
            generous yet keeps the 1D optim well-conditioned.

    Returns:
        V_opt: argmax of L(V). Returns 0.0 when V = 0 maximizes L (null layer).
    """
    from scipy.optimize import minimize_scalar

    p = z.shape[0]
    if prior_pi is None:
        log_prior = torch.full((p,), -math.log(p), dtype=z.dtype)
    else:
        log_prior = torch.log(prior_pi.to(z.dtype).clamp_min(1e-300))

    # Search a NARROW log-V window around V_init, mirroring susieR's
    # `optimize_prior_variance()` which uses `optim(method="Brent",
    # lower=log(V0)-10, upper=log(V0)+10)`. A wider range (e.g., down to
    # log(1e-12)) confuses Brent's bracketing because L(V) is flat at zero
    # for V << V_init and the optimizer misidentifies the flat region as a
    # converged minimum at the lower bound.
    log_V_init = math.log(max(float(V_init), 1e-12))
    log_V_lo = log_V_init - 10.0
    log_V_hi = log_V_init + math.log(upper_bound_factor)

    def neg_log_marginal(log_V: float) -> float:
        V = math.exp(log_V)
        _, _, log_bf = ser_posterior(z, R, n, V)
        return -float(torch.logsumexp(log_bf + log_prior, dim=0))

    # xatol matches susieR's default `tol = .Machine$double.eps^0.25 ≈ 1.2e-4`.
    result = minimize_scalar(
        neg_log_marginal,
        bounds=(log_V_lo, log_V_hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    L_opt = -float(result.fun)
    V_opt = math.exp(float(result.x))

    # Compare against L(V=0) = 0 (the null). Snap to 0 when null is favored.
    return V_opt if L_opt > 0.0 else 0.0


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
        estimate_prior_variance: bool = True,
        estimate_prior_method: str = "optim",
        prior_variance_tol: float = 1e-3,
        z_adjustment: bool = True,
    ):
        self.max_num_causal = max_num_causal
        self.coverage = coverage
        self.purity = purity
        self.sigma_prior_sq = sigma_prior_sq
        self.max_iter = max_iter
        self.tol = tol
        self.block_size_threshold = block_size_threshold
        # Per-layer prior-variance update (matches susieR default).
        # When False, all layers use the fixed sigma_prior_sq for all iterations
        # (the pre-V-update behavior; useful for back-compat tests).
        self.estimate_prior_variance = estimate_prior_variance
        # Method for the V update when estimate_prior_variance=True:
        #   "optim" — per-layer 1D maximization of L(V) (susieR default; the
        #     principled fix that mirrors the upstream implementation; gives
        #     V=0 EXACTLY for null layers without a snap threshold).
        #   "EM"    — closed-form M-step plus snap-to-zero at prior_variance_tol
        #     (cheaper but introduces a noise-floor V fixed point on realistic
        #     LD that registers as a small β_sd Pearson residual vs susieR;
        #     see findings ledger 2026-05-12).
        if estimate_prior_method not in ("optim", "EM"):
            raise ValueError(
                f"estimate_prior_method must be 'optim' or 'EM'; got "
                f"{estimate_prior_method!r}"
            )
        self.estimate_prior_method = estimate_prior_method
        # Apply the canonical SuSiE-RSS z-score adjustment (Zhu & Stephens
        # 2017 / Zou 2022 [C4]; mirrors susieR `susie_rss()` default). See
        # `apply_z_score_adjustment()` for the formula and rationale. Default
        # True matches susieR. Set False to operate on raw z (e.g., when z
        # is already on the assumed-Var(y)=1 scale).
        self.z_adjustment = z_adjustment
        # Threshold below which a layer's V is snapped to zero (layer "shuts off").
        # The pure EM update has a non-zero fixed point at ~p^{-1} * sigma^2_l,j
        # for noise-floor layers (because uniform alpha + small mu still
        # contributes positively). susieR's "optim" estimator picks V=0 when
        # the data favor a null layer; we approximate with a snap-to-zero
        # threshold.
        #
        # Default 1e-3 is empirically calibrated:
        #   - Synthetic strong signals: noise-floor V ~ 1e-4 to 5e-5 → snaps cleanly,
        #     β_sd Pearson = 1.000 vs susieR.
        #   - Realistic LD (e.g., MDP maize): noise-floor V ~ 5e-3 → does NOT snap,
        #     β_sd Pearson ≈ 0.994 vs susieR (still well above the 0.993 floor).
        #   - Bumping to 1e-2 closes MDP gap but kills MDP's real weak secondary
        #     signals (β_mean Pearson drops 0.999 → 0.986). The fundamental fix
        #     for the MDP-class residual is susieR's "optim" path (per-layer
        #     marginal-evidence comparison) — Tier C work.
        self.prior_variance_tol = prior_variance_tol

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
        # Map z onto SuSiE-RSS's assumed Var(y)=1 scale (Zou 2022 [C4]).
        # Default-on; mirrors susieR. Vanishes for large n.
        if self.z_adjustment:
            z = apply_z_score_adjustment(z, n)
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

        # Per-layer prior variance V_l. Init all layers to sigma_prior_sq.
        # When estimate_prior_variance=True, updated each iteration via EM:
        #     V_l = sum_j alpha_l,j * (mu_l,j^2 + sigma_l,j^2)
        # Layers that don't fit a real signal get V_l → 0 and effectively
        # turn off (mu → 0, sigma_sq → 0). Matches susieR's fit$V semantics.
        # Investigation 2026-05-12: without this update, all L layers stay
        # active with fixed sigma_prior_sq, producing spurious noise-floor
        # PIPs and BETA_SD ~10x larger than susieR at noise variants.
        V = torch.full((L,), self.sigma_prior_sq, dtype=torch.float64)

        elbo_history: list[float] = []
        # PIP-stability convergence criterion mirrors susieR's primary convergence
        # check (max |delta_alpha|), which is more robust than ELBO tolerance to
        # the transient ELBO drops introduced by the per-layer V update (the V
        # update + alpha update are decoupled coordinate-ascent sub-steps, so
        # ELBO can oscillate by ~1e-3 even after PIPs are stable).
        prev_pip: Optional[torch.Tensor] = None

        # Native C++ inner-sweep is selected when the build is present, the
        # tensors live on CPU and are float64, and TORCHGENOMICS_DISABLE_NATIVE
        # is unset. One call per outer iteration runs all L sequential layer
        # updates (residual + Brent V optimiser + SER + softmax + EM) in place.
        # The Python outer loop still drives the ELBO + convergence check.
        # Below the threshold the per-iteration overhead of the C++ marshalling
        # does not pay back; the Python reference body runs unchanged.
        use_native_sweep = (
            HAS_NATIVE_IBSS
            and not native_disabled()
            and z.device.type == "cpu"
            and R.device.type == "cpu"
            and z.dtype == torch.float64
            and R.dtype == torch.float64
            and p >= 128
        )
        if use_native_sweep:
            # Precompute R_diag once; constant across iterations.
            R_diag_np = torch.diag(R).contiguous().numpy()
            R_np = R.detach().contiguous().numpy()
            # log_prior matches find_optimal_V's convention exactly (uniform
            # = -log(p); user-supplied prior_pi taken as-is). Softmax is
            # shift-invariant, so this single array drives both the V
            # maximisation and the per-layer alpha softmax with bit-equal
            # results vs the Python body.
            if prior_pi_per_snp is None:
                log_prior_np = torch.full(
                    (p,), -math.log(p), dtype=torch.float64
                ).numpy()
            else:
                log_prior_np = torch.log(
                    prior_pi_per_snp.to(torch.float64).clamp_min(1e-300)
                ).contiguous().numpy()

        V_method_int = 0 if self.estimate_prior_method == "optim" else 1
        sigma_prior_sq_f = float(self.sigma_prior_sq)
        prior_variance_tol_f = float(self.prior_variance_tol)

        # Precompute the Cholesky factor of R once. R is constant across
        # the outer IBSS iterations, but _compute_elbo solves R^{-1}·r
        # every iteration — caching the factor turns each ELBO call from
        # O(p³) to O(p²). On a non-PD R (numerical fluke; rare on real LD
        # matrices) we fall back to the unfactored linalg.solve path with
        # a warning-free None.
        try:
            cholesky_L = torch.linalg.cholesky(R)
        except Exception:
            cholesky_L = None

        for iteration in range(self.max_iter):
            if use_native_sweep:
                # In-place sweep: the contiguous CPU torch tensors share
                # storage with their .numpy() views, so alpha / mu /
                # sigma_sq / b_eff / V are updated directly.
                z_np = z.contiguous().numpy()
                alpha_t = alpha.contiguous()
                mu_t = mu.contiguous()
                sigma_sq_t = sigma_sq.contiguous()
                b_eff_t = b_eff.contiguous()
                V_t = V.contiguous()
                _ibss_native.ibss_inner_sweep(
                    z_np,
                    R_np,
                    R_diag_np,
                    alpha_t.numpy(),
                    mu_t.numpy(),
                    sigma_sq_t.numpy(),
                    b_eff_t.numpy(),
                    V_t.numpy(),
                    log_prior_np,
                    int(n),
                    bool(self.estimate_prior_variance),
                    V_method_int,
                    prior_variance_tol_f,
                    sigma_prior_sq_f,
                    1e-4,
                )
                alpha, mu, sigma_sq, b_eff, V = (
                    alpha_t, mu_t, sigma_sq_t, b_eff_t, V_t,
                )
            else:
                for l in range(L):
                    # Residual on z-scale: tilde_z_l = z - R @ sum_{l'!=l} b_eff_{l'}
                    tilde_z_l = ibss_residual_update(z, R, b_eff, layer_idx=l)

                    # V update — two methods, both mirror susieR options:
                    #   "optim": per-layer 1D maximization of L(V) BEFORE the SER.
                    #     Returns V=0 EXACTLY when null is favored. No snap threshold.
                    #   "EM":    closed-form M-step AFTER the SER (uses this layer's
                    #     fresh alpha/mu/sigma_sq) with snap-to-zero at
                    #     prior_variance_tol. Cheaper but has noise-floor fixed point.
                    if (
                        self.estimate_prior_variance
                        and self.estimate_prior_method == "optim"
                    ):
                        V[l] = find_optimal_V(
                            z=tilde_z_l,
                            R=R,
                            n=n,
                            V_init=max(V[l].item(), self.sigma_prior_sq),
                            prior_pi=prior_pi_per_snp,
                        )

                    # Layer is null (V_l == 0): SER posterior collapses to a delta
                    # at zero. Skip computation.
                    if self.estimate_prior_variance and V[l].item() == 0.0:
                        alpha[l] = 1.0 / p  # uniform (irrelevant; mu is zero)
                        mu[l] = 0.0
                        sigma_sq[l] = 0.0
                        b_eff[l] = 0.0
                        continue

                    # SER posterior (beta-scale) using PER-LAYER prior variance V_l
                    sigma_sq_l, mu_l, log_bf_l = ser_posterior(
                        tilde_z_l, R, n, V[l].item()
                    )
                    # Softmax with per-SNP prior (D3 shim)
                    alpha_l = compute_alpha(log_bf_l, prior_pi=prior_pi_per_snp)

                    alpha[l] = alpha_l
                    mu[l] = mu_l
                    sigma_sq[l] = sigma_sq_l
                    # b_eff = sqrt(n) * beta-scale posterior mean for this layer
                    b_eff[l] = alpha_l * (sqrt_n * mu_l)

                    # EM M-step for V_l: posterior expected squared effect.
                    # Mirrors susieR's estimate_prior_variance="EM" path. Snap-to-
                    # zero closes the β_sd parity gap on noise variants but breaks
                    # strict ELBO monotonicity at the snap (PIP-stability
                    # convergence handles it). For the cleaner null treatment use
                    # estimate_prior_method="optim" (default).
                    if (
                        self.estimate_prior_variance
                        and self.estimate_prior_method == "EM"
                    ):
                        V_new = (alpha_l * (mu_l ** 2 + sigma_sq_l)).sum().clamp_min(0.0).item()
                        V[l] = 0.0 if V_new < self.prior_variance_tol else V_new

            # Compute ELBO at end of iteration (uses scaled effects internally)
            elbo = self._compute_elbo(z, R, n, alpha, mu, sigma_sq, V, cholesky_L=cholesky_L)
            elbo_history.append(elbo)

            # Current PIPs for convergence check
            current_pip = 1.0 - torch.prod(1.0 - alpha, dim=0)

            # Convergence check: either ELBO change is small AND positive,
            # OR PIPs are stable (mirrors susieR's primary convergence test).
            # PIP-stability is the more robust criterion because the V-update
            # can introduce transient ELBO drops even when PIPs are converged.
            if iteration > 0:
                elbo_diff = elbo_history[-1] - elbo_history[-2]
                elbo_converged = 0.0 <= elbo_diff < self.tol
                # PIP-stability threshold mirrors susieR's primary check.
                # Default 1e-3 (=0.1% PIP movement) is what susieR's `tol`
                # parameter controls; our `self.tol` (default 1e-6) is the
                # ELBO tolerance which is tighter than what's achievable
                # under V-update transients. Practical convergence is
                # PIP-stability, not ELBO-equality.
                pip_tol = max(self.tol, 1e-3)
                pip_converged = (prev_pip is not None
                                 and (current_pip - prev_pip).abs().max().item() < pip_tol)
                if elbo_converged or pip_converged:
                    converged = True
                    prev_pip = current_pip
                    break
            prev_pip = current_pip
        else:
            converged = False

        elbo_history_tensor = torch.tensor(elbo_history, dtype=torch.float64)

        # Per-variant PIP, beta_mean, beta_sd
        pip = 1.0 - torch.prod(1.0 - alpha, dim=0)
        beta_mean = (alpha * mu).sum(dim=0)
        # Per-variant variance under SuSiE mean-field q (layers independent):
        #     Var[beta_j] = sum_l Var[b_l,j]
        # where Var[b_l,j] = alpha_lj * (mu_lj^2 + sigma_lj^2) - (alpha_lj * mu_lj)^2
        # is the per-layer variance of the categorical Bernoulli * Normal mixture.
        #
        # Matches susieR's susie_get_posterior_sd() (CRAN), which computes
        # sqrt(colSums(alpha*mu2 - (alpha*mu)^2)) on the per-layer matrices.
        # See validation/external/susieR/ Tier 2 parity findings (2026-05-12)
        # for the head-to-head investigation that surfaced the prior bug
        # (sum-of-(squared-mean) instead of sum-of-per-layer-variance).
        per_layer_var = alpha * (mu ** 2 + sigma_sq) - (alpha * mu) ** 2
        beta_var = per_layer_var.sum(dim=0).clamp_min(0.0)
        beta_sd = beta_var.sqrt()

        credible_sets = self._build_credible_sets(alpha, R)

        return BayesianVSRssResult(
            alpha=alpha,
            mu=mu,
            sigma_sq=sigma_sq,
            V=V.clone(),
            pip=pip,
            beta_mean=beta_mean,
            beta_sd=beta_sd,
            elbo=float(elbo_history_tensor[-1]),
            elbo_history=elbo_history_tensor,
            credible_sets=credible_sets,
            converged=converged,
            n_iter=iteration + 1,
        )

    def fit_rss_blocked(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        blocks: list,
        prior_pi_per_snp: Optional[torch.Tensor] = None,
    ) -> BayesianVSRssResult:
        """Run SuSiE-RSS per block and concatenate results.

        Per NA1 design spec section 2.6: block decomposition makes per-block
        memory bounded by O(p_block_max^2) instead of O(p^2) total, enabling
        biobank-scale fine-mapping where dense R does not fit in RAM.

        IMPORTANT — divergence from dense fit is REAL, not just numerical:

        Even on strictly block-diagonal R (zero off-block LD), `fit_rss_blocked`
        does NOT exactly equal `fit_rss` because per-block IBSS recalibrates
        its softmax denominator over a smaller candidate pool (p_block) than
        dense IBSS does (p_total). Concretely, background-noise PIPs differ
        by O(1/p_block) - O(1/p_total). The TRUE invariants that hold:

        1. PIPs at high-signal variants (PIP > 0.5) agree between blocked and
           dense to ~1e-2 absolute when L per block matches the per-block
           causal count.
        2. Credible-set membership of true causals matches between methods.
        3. Per-block independence: variants in block A do not affect
           posteriors of variants in block B (true by construction since
           the off-block R is zero).

        The Tier 2 parity test against susieR (NA1 plan Task 14) compares
        against susieR's dense per-locus fit and absorbs the per-block
        recalibration divergence under the spec section 5.2 tolerance
        (floor + observed * 2 for MVP scope).

        Args:
            z: Per-variant z-scores, shape (p,).
            R: LD correlation matrix, shape (p, p) — only the per-block
                sub-matrices R[block][:, block] are accessed.
            n: GWAS sample size.
            blocks: List of BlockSpec defining the block partition.
            prior_pi_per_snp: Optional per-SNP prior, shape (p,).

        Returns:
            BayesianVSRssResult covering all p variants, with per-block
            results concatenated.
        """
        p = z.shape[0]
        L = self.max_num_causal

        alpha_full = torch.zeros(L, p, dtype=torch.float64)
        mu_full = torch.zeros(L, p, dtype=torch.float64)
        sigma_sq_full = torch.zeros(L, p, dtype=torch.float64)
        pip_full = torch.zeros(p, dtype=torch.float64)
        beta_mean_full = torch.zeros(p, dtype=torch.float64)
        beta_sd_full = torch.zeros(p, dtype=torch.float64)
        elbo_total = 0.0
        all_credible_sets: list[Tuple[int, list[int]]] = []
        all_converged = True
        max_n_iter = 0
        V_blocks: list[torch.Tensor] = []

        for b_idx, block in enumerate(blocks):
            start, stop = block.start, block.stop
            z_block = z[start:stop]
            R_block = R[start:stop, start:stop]
            prior_block = (
                prior_pi_per_snp[start:stop]
                if prior_pi_per_snp is not None
                else None
            )

            # Run dense fit on the block
            block_result = self.fit_rss(
                z=z_block,
                R=R_block,
                n=n,
                prior_pi_per_snp=prior_block,
            )

            alpha_full[:, start:stop] = block_result.alpha
            mu_full[:, start:stop] = block_result.mu
            sigma_sq_full[:, start:stop] = block_result.sigma_sq
            pip_full[start:stop] = block_result.pip
            beta_mean_full[start:stop] = block_result.beta_mean
            beta_sd_full[start:stop] = block_result.beta_sd
            elbo_total += block_result.elbo
            # Re-index credible-set members from block-local to global indices
            for layer_idx, members in block_result.credible_sets:
                global_members = [m + start for m in members]
                all_credible_sets.append((layer_idx, global_members))
            all_converged = all_converged and block_result.converged
            max_n_iter = max(max_n_iter, block_result.n_iter)
            # block_result.V may be a per-block (L,) vector for unblocked
            # fits or stacked (n_inner_blocks, L) — we always reduce to (L,)
            # by taking the per-layer max across inner blocks (the most
            # active layer wins; same-V across blocks is the common case).
            block_V = block_result.V
            if block_V.ndim == 2:
                block_V = block_V.amax(dim=0)
            V_blocks.append(block_V)

        # Concatenated V is a stack of per-block V vectors (each length L).
        # We expose them as a (n_blocks, L) tensor for diagnostics.
        V_all = torch.stack(V_blocks, dim=0) if V_blocks else torch.zeros((0, self.max_num_causal), dtype=torch.float64)

        return BayesianVSRssResult(
            alpha=alpha_full,
            mu=mu_full,
            sigma_sq=sigma_sq_full,
            V=V_all,
            pip=pip_full,
            beta_mean=beta_mean_full,
            beta_sd=beta_sd_full,
            elbo=elbo_total,
            elbo_history=torch.tensor([elbo_total], dtype=torch.float64),
            credible_sets=all_credible_sets,
            converged=all_converged,
            n_iter=max_n_iter,
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
        V: Optional[torch.Tensor] = None,
        cholesky_L: Optional[torch.Tensor] = None,
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
        # uses the R^{-1} Mahalanobis weighting. R is constant across IBSS
        # iterations, so when the caller passes a precomputed Cholesky
        # factor cholesky_L (R = L L^T) we reuse it via torch.cholesky_solve
        # — O(p²) per call vs O(p³) for the unfactored torch.linalg.solve
        # path. The unfactored path remains the algorithmic spec / default
        # when cholesky_L is None (single-call usage outside fit_rss, or
        # when the cached factor is unavailable due to a non-PD R).
        residual = z - z_pred
        if cholesky_L is not None:
            R_inv_residual = torch.cholesky_solve(
                residual.unsqueeze(-1), cholesky_L
            ).squeeze(-1)
        else:
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

        # KL for the Gaussian prior on the effect, per layer.
        # KL_gauss_l = 0.5 * sum_j alpha_lj * (
        #     mu_lj^2 / V_l + sigma_lj^2 / V_l - 1 - log(sigma_lj^2 / V_l) )
        # When V is None (back-compat), use the fixed sigma_prior_sq for all
        # layers. When V is provided as a (L,) tensor, use V_l per layer.
        # For shut-off layers (V_l ≈ 0), the per-layer KL is zero by
        # convention — the prior collapses to a delta at zero and the
        # posterior (mu=0, sigma=0) matches it exactly.
        if V is None:
            V_per_layer = torch.full((alpha.shape[0],), self.sigma_prior_sq, dtype=torch.float64)
        else:
            V_per_layer = V
        kl_gauss = 0.0
        tol = self.prior_variance_tol
        for l in range(alpha.shape[0]):
            V_l = V_per_layer[l].item() if isinstance(V_per_layer[l], torch.Tensor) else float(V_per_layer[l])
            if V_l < tol:
                continue  # shut-off layer; KL = 0 (degenerate posterior matches degenerate prior)
            sigma_sq_safe = sigma_sq[l].clamp_min(1e-300)
            kl_l = (
                0.5 * alpha[l] * (
                    mu[l] ** 2 / V_l
                    + sigma_sq[l] / V_l
                    - 1.0
                    - torch.log(sigma_sq_safe / V_l)
                )
            ).sum().item()
            kl_gauss += kl_l

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


import csv
from pathlib import Path


def write_results_tsv(
    path: Path | str,
    result: BayesianVSRssResult,
    snp_meta: dict,
) -> None:
    """Write SuSiE-RSS results to a Phase 59 PolyFun-compatible TSV.

    Output columns (per NA1 design spec section 6):
        SNP, CHR, BP, A1, A2, Z, N, PIP, BETA_MEAN, BETA_SD, CREDIBLE_SET

    CREDIBLE_SET = integer index of the credible set the variant belongs to
    (0 = not in any CS; 1, 2, ... for first, second, ... CS in returned order).
    Matches PolyFun convention for downstream aggregation compatibility.

    Args:
        path: Output TSV path.
        result: BayesianVSRssResult to serialize.
        snp_meta: Dict with keys snp, chr, bp, a1, a2, z, n; each a list of
            length p in variant order matching result tensors.
    """
    p = result.pip.shape[0]
    # Build a SNP -> CS index map
    cs_index = [0] * p
    for cs_id, (layer_idx, members) in enumerate(result.credible_sets, start=1):
        for m in members:
            # If a variant appears in multiple CSes (rare), keep the first
            if cs_index[m] == 0:
                cs_index[m] = cs_id

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["SNP", "CHR", "BP", "A1", "A2", "Z", "N",
             "PIP", "BETA_MEAN", "BETA_SD", "CREDIBLE_SET"]
        )
        for j in range(p):
            writer.writerow([
                snp_meta["snp"][j],
                snp_meta["chr"][j],
                snp_meta["bp"][j],
                snp_meta["a1"][j],
                snp_meta["a2"][j],
                f"{snp_meta['z'][j]:.6g}",
                snp_meta["n"][j],
                f"{float(result.pip[j]):.6g}",
                f"{float(result.beta_mean[j]):.6g}",
                f"{float(result.beta_sd[j]):.6g}",
                cs_index[j],
            ])
