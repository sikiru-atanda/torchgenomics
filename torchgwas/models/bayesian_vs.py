"""Bayesian Variable Selection GWAS with Spike-and-Slab Prior (Phase 18).

Two inference methods:
1. **SuSiE** (default): Sum of Single Effects regression (Wang et al., JRSSB
   2020). Models L independent single-effect layers where SNPs compete via
   softmax within each layer, preventing LD-driven PIP inflation.
2. **CAVI**: Mean-field coordinate ascent with per-SNP sigmoid inclusion.
   Simpler but spreads PIP across correlated SNPs.

Both operate under the LMM framework using a pre-fitted NullFit. In the
rotated eigenspace V = diag(d_i) is diagonal, enabling efficient updates.

References:
    - Wang et al. (2020). A simple new approach to variable selection
      in regression, with application to genetic fine-mapping (SuSiE).
    - Carbonetto & Stephens (2012). Scalable variational inference for
      Bayesian variable selection in regression.
    - Quickdraws (Nat Genet 2025). GPU-accelerated spike-and-slab VI.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import HAS_NATIVE_CAVI, _cavi_native
from ..config import STAT_DTYPE
from ..linalg.eigh import rotate
from .base import NullFit, VariantMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BayesianVSResult:
    """Bayesian variable selection results per SNP."""

    # Variant identifiers
    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]

    # Allele frequencies
    af: Tensor  # (p,)

    # Bayesian results
    pip: Tensor  # (p,) posterior inclusion probabilities
    beta_mean: Tensor  # (p,) posterior mean E_q[beta_j] = gamma_j * mu_j
    beta_sd: Tensor  # (p,) posterior SD

    # Credible sets for fine-mapping
    credible_sets: list[list[int]] | None = None

    # ELBO convergence trace
    elbo_trace: list[float] = field(default_factory=list)
    converged: bool = False

    # Learned hyperparameters
    prior_pi: float = 0.01
    prior_sig2_beta: float = 0.1

    # SuSiE-specific fields
    alpha: Tensor | None = None  # (L, p) per-layer inclusion probs
    n_signals: int = 0  # number of active signals detected
    method: str = "cavi"  # "cavi" or "susie"

    def __len__(self) -> int:
        return len(self.snp)


# ---------------------------------------------------------------------------
# BayesianVS
# ---------------------------------------------------------------------------

class BayesianVS:
    """Bayesian variable selection under LMM (SuSiE or CAVI).

    Operates on a pre-computed NullFit from SingleTraitLMM.fit_null().
    Works in the rotated eigenspace where V is diagonal for efficient
    updates. Default method is SuSiE (sum of single effects).

    Parameters
    ----------
    null_fit : NullFit
        Must contain eigenvectors, eigenvalues, Y_rot, X0_rot, sig2_g,
        sig2_e, b0.
    ploidy : int
        Ploidy level for allele frequency computation (default 2).
    """

    def __init__(self, null_fit: NullFit, ploidy: int = 2) -> None:
        self.null_fit = null_fit
        self.ploidy = ploidy

        nf = null_fit
        # Diagonal of V in rotated space: d_i = eigenvalue_i * sig2_g + sig2_e
        lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
        self._d = nf.eigenvalues * lam + 1.0  # H diagonal (unitless)
        # Precision weights: w_i = 1 / (eigenvalue_i * sig2_g + sig2_e)
        self._w = 1.0 / (nf.eigenvalues * nf.sig2_g + max(nf.sig2_e, 1e-20))
        # Fixed-effect estimates from null
        self._alpha_hat = nf.b0  # (c,)

    def fit(
        self,
        G: Tensor,
        variant_meta: VariantMeta,
        *,
        max_iter: int = 1000,
        tol: float = 1e-6,
        prior_pi: float = 0.01,
        prior_sig2_beta: float = 0.1,
        learn_hyperparams: bool = True,
        em_interval: int = 10,
        elbo_freq: int = 5,
        credible_set_coverage: float = 0.95,
        ld_threshold: float = 0.5,
        method: str = "susie",
        n_signals: int = 10,
    ) -> BayesianVSResult:
        """Run Bayesian variable selection to compute PIPs.

        Parameters
        ----------
        G : Tensor, shape (n, p)
            Full genotype matrix for the region of interest.
        variant_meta : VariantMeta
            Variant identifiers.
        max_iter : int
            Maximum iterations.
        tol : float
            ELBO relative convergence tolerance.
        prior_pi : float
            Initial prior inclusion probability (CAVI) or ignored (SuSiE).
        prior_sig2_beta : float
            Initial prior slab variance.
        learn_hyperparams : bool
            If True, update hyperparameters via empirical Bayes.
        em_interval : int
            Update hyperparams every this many iterations.
        elbo_freq : int
            Compute ELBO every this many iterations.
        credible_set_coverage : float
            Target coverage for credible sets (default 0.95).
        ld_threshold : float
            r^2 threshold for LD grouping (CAVI only).
        method : str
            "susie" (default) or "cavi".
        n_signals : int
            Number of single-effect layers L (SuSiE only, default 10).

        Returns
        -------
        BayesianVSResult
        """
        if method not in ("susie", "cavi"):
            raise ValueError(
                f"Unknown method: {method!r}. Use 'susie' or 'cavi'."
            )

        G = G.to(STAT_DTYPE)
        nf = self.null_fit
        n, p = G.shape

        # Shared preamble: allele freq, centering, rotation, residual
        af = G.mean(dim=0) / self.ploidy
        G_centered = G - G.mean(dim=0, keepdim=True)
        U = nf.eigenvectors
        G_rot = rotate(G_centered, U)

        Y_rot = nf.Y_rot.to(STAT_DTYPE)
        X0_rot = nf.X0_rot.to(STAT_DTYPE)
        alpha_hat = self._alpha_hat.to(STAT_DTYPE)
        r = Y_rot - X0_rot @ alpha_hat
        w = self._w.to(STAT_DTYPE)

        # Auto-scale prior slab variance to the phenotype scale.
        # The user-supplied prior_sig2_beta is a *scaled* prior variance
        # (fraction of total phenotypic variance), following the convention
        # in the SuSiE R package (scaled_prior_variance=0.2).  This ensures
        # the prior covers plausible effect sizes regardless of phenotype
        # units.  For standardized phenotypes (sig2_g+sig2_e≈1) this is a
        # no-op; for raw phenotypes (e.g., MDP EarHT with var≈254) it
        # scales the slab ~250x wider to match the data.
        total_var = max(nf.sig2_g + nf.sig2_e, 1e-10)
        prior_sig2_beta_scaled = prior_sig2_beta * total_var
        logger.debug(
            "Auto-scaled prior_sig2_beta: %.4e * %.4e (total_var) = %.4e",
            prior_sig2_beta, total_var, prior_sig2_beta_scaled,
        )

        common_kwargs = dict(
            chr=variant_meta.chr, pos=variant_meta.pos,
            snp=variant_meta.snp, a1=variant_meta.a1, a2=variant_meta.a2,
            af=af,
        )

        if method == "susie":
            return self._fit_susie(
                G, G_rot, r, w, n, p,
                n_signals=n_signals,
                prior_sig2_beta=prior_sig2_beta_scaled,
                max_iter=max_iter, tol=tol,
                learn_hyperparams=learn_hyperparams,
                em_interval=em_interval, elbo_freq=elbo_freq,
                credible_set_coverage=credible_set_coverage,
                **common_kwargs,
            )
        else:
            return self._fit_cavi(
                G, G_rot, r, w, p,
                prior_pi=prior_pi,
                prior_sig2_beta=prior_sig2_beta_scaled,
                max_iter=max_iter, tol=tol,
                learn_hyperparams=learn_hyperparams,
                em_interval=em_interval, elbo_freq=elbo_freq,
                credible_set_coverage=credible_set_coverage,
                ld_threshold=ld_threshold,
                **common_kwargs,
            )

    def _fit_cavi(
        self, G, G_rot, r, w, p, *,
        prior_pi, prior_sig2_beta,
        max_iter, tol, learn_hyperparams, em_interval, elbo_freq,
        credible_set_coverage, ld_threshold, **common,
    ) -> BayesianVSResult:
        """CAVI spike-and-slab inference path."""
        gamma, mu, sigma2 = self._initialize_from_marginal(
            G_rot, w, r, prior_pi, prior_sig2_beta,
        )
        gamma, mu, sigma2, elbo_trace, converged, pi, sig2_beta = self._cavi_loop(
            G_rot=G_rot, r=r, w=w,
            gamma=gamma, mu=mu, sigma2=sigma2,
            pi=prior_pi, sig2_beta=prior_sig2_beta,
            max_iter=max_iter, tol=tol,
            learn_hyperparams=learn_hyperparams,
            em_interval=em_interval, elbo_freq=elbo_freq,
        )
        pip = gamma
        beta_mean = gamma * mu
        beta_var = gamma * sigma2 + gamma * (1.0 - gamma) * mu ** 2
        beta_sd = torch.sqrt(torch.clamp(beta_var, min=1e-30))
        credible_sets = self._build_credible_sets(
            G, gamma, credible_set_coverage, ld_threshold,
        )
        return BayesianVSResult(
            **common, pip=pip, beta_mean=beta_mean, beta_sd=beta_sd,
            credible_sets=credible_sets, elbo_trace=elbo_trace,
            converged=converged, prior_pi=pi, prior_sig2_beta=sig2_beta,
            method="cavi",
        )

    def _fit_susie(
        self, G, G_rot, r, w, n, p, *,
        n_signals, prior_sig2_beta,
        max_iter, tol, learn_hyperparams, em_interval, elbo_freq,
        credible_set_coverage, **common,
    ) -> BayesianVSResult:
        """SuSiE (sum of single effects) inference path."""
        alpha_L, mu_L, sigma2_L, elbo_trace, converged, sig2_l = self._susie_loop(
            G_rot=G_rot, r=r, w=w,
            n_signals=n_signals,
            prior_sig2_beta=prior_sig2_beta,
            max_iter=max_iter, tol=tol,
            learn_hyperparams=learn_hyperparams,
            em_interval=em_interval, elbo_freq=elbo_freq,
        )
        # PIP = 1 - prod_l(1 - alpha_lj)
        pip = 1.0 - torch.prod(1.0 - alpha_L, dim=0)
        # Posterior mean = sum_l E_q[b_l] = sum_l (alpha_l * mu_l)
        beta_mean = (alpha_L * mu_L).sum(dim=0)
        # Posterior variance
        E_b2 = (alpha_L * (mu_L ** 2 + sigma2_L)).sum(dim=0)
        beta_var = E_b2 - beta_mean ** 2
        beta_sd = torch.sqrt(torch.clamp(beta_var, min=1e-30))
        # Credible sets: per-layer
        credible_sets = self._build_susie_credible_sets(
            alpha_L, credible_set_coverage,
        )
        # Active signals: layers whose max alpha > 2/p
        active = int((alpha_L.max(dim=1).values > 2.0 / p).sum().item())

        return BayesianVSResult(
            **common, pip=pip, beta_mean=beta_mean, beta_sd=beta_sd,
            credible_sets=credible_sets, elbo_trace=elbo_trace,
            converged=converged, prior_pi=1.0 / p,
            prior_sig2_beta=float(sig2_l.mean().item()),
            alpha=alpha_L, n_signals=active, method="susie",
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_from_marginal(
        self,
        G_rot: Tensor,
        w: Tensor,
        r: Tensor,
        prior_pi: float,
        prior_sig2_beta: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Initialize variational parameters conservatively.

        Marginal z-scores are computed independently per SNP and would cause
        divergence if applied jointly (sum of independent effects >> actual
        signal). Instead, initialize gamma at the prior and mu = 0. The
        posterior variance is set to its analytical form.

        Returns
        -------
        gamma : (p,) initial inclusion probabilities (all = prior_pi)
        mu : (p,) initial posterior means (all = 0)
        sigma2 : (p,) initial posterior variances
        """
        p = G_rot.shape[1]

        # Weighted inner products: g_j^T W g_j (constant, used for sigma2)
        wG = w.unsqueeze(1) * G_rot  # (n, p)
        gWg = (G_rot * wG).sum(dim=0)  # (p,)

        # Posterior variance at initialization
        sigma2 = 1.0 / (gWg + 1.0 / prior_sig2_beta)  # (p,)

        # Conservative: gamma = prior_pi, mu = 0
        gamma = torch.full((p,), prior_pi, dtype=G_rot.dtype, device=G_rot.device)
        mu = torch.zeros(p, dtype=G_rot.dtype, device=G_rot.device)

        return gamma, mu, sigma2

    # ------------------------------------------------------------------
    # CAVI loop
    # ------------------------------------------------------------------

    def _cavi_loop(
        self,
        G_rot: Tensor,
        r: Tensor,
        w: Tensor,
        gamma: Tensor,
        mu: Tensor,
        sigma2: Tensor,
        pi: float,
        sig2_beta: float,
        max_iter: int,
        tol: float,
        learn_hyperparams: bool,
        em_interval: int,
        elbo_freq: int,
    ) -> tuple[Tensor, Tensor, Tensor, list[float], bool, float, float]:
        """Run coordinate ascent variational inference.

        Each iteration sweeps through all p SNPs, updating gamma_j, mu_j,
        sigma2_j one at a time while incrementally maintaining the residual.

        Returns
        -------
        gamma, mu, sigma2, elbo_trace, converged, pi, sig2_beta
        """
        n, p = G_rot.shape
        elbo_trace: list[float] = []
        converged = False
        stable_count = 0

        # Precompute g_j^T W g_j for all j (constant across iterations)
        wG = w.unsqueeze(1) * G_rot  # (n, p)
        gWg = (G_rot * wG).sum(dim=0)  # (p,)

        # Native C++ inner-sweep is selected when the build is present, the
        # tensors live on CPU and are float64, and ``TORCHGWAS_DISABLE_NATIVE``
        # is unset. Each call runs one full p-SNP coordinate-update sweep
        # in place; the outer iteration / ELBO / EM loop stays in Python.
        use_native_sweep = (
            HAS_NATIVE_CAVI
            and not native_disabled()
            and G_rot.device.type == "cpu"
            and G_rot.dtype == torch.float64
            and r.dtype == torch.float64
            and gamma.dtype == torch.float64
        )

        for iteration in range(max_iter):
            if use_native_sweep:
                # In-place sweep — numpy arrays share memory with the
                # contiguous CPU torch tensors below, so updates land
                # directly in `r`, `gamma`, `mu`, `sigma2`.
                G_np = G_rot.detach().contiguous().numpy()
                wG_np = wG.detach().contiguous().numpy()
                gWg_np = gWg.detach().contiguous().numpy()
                r_t = r.detach().contiguous()
                gamma_t = gamma.detach().contiguous()
                mu_t = mu.detach().contiguous()
                sigma2_t = sigma2.detach().contiguous()
                _cavi_native.cavi_sweep(
                    G_np,
                    wG_np,
                    gWg_np,
                    r_t.numpy(),
                    gamma_t.numpy(),
                    mu_t.numpy(),
                    sigma2_t.numpy(),
                    float(pi),
                    float(sig2_beta),
                )
                r = r_t
                gamma = gamma_t
                mu = mu_t
                sigma2 = sigma2_t
            else:
                # Sweep through all SNPs (algorithmic reference).
                for j in range(p):
                    g_j = G_rot[:, j]  # (n,)
                    wg_j = wG[:, j]  # (n,) = w * g_j

                    # Add back contribution of SNP j to residual
                    old_effect = gamma[j] * mu[j]
                    r = r + g_j * old_effect

                    # Posterior variance
                    sigma2_j = 1.0 / (gWg[j] + 1.0 / sig2_beta)

                    # Posterior mean
                    mu_j = sigma2_j * torch.dot(wg_j, r)

                    # Log Bayes factor
                    log_bf = 0.5 * math.log(sigma2_j.item() / sig2_beta) + \
                        0.5 * mu_j ** 2 / sigma2_j
                    log_bf = torch.clamp(log_bf, -30.0, 30.0)

                    # Inclusion probability
                    log_odds_prior = math.log(pi / (1.0 - pi))
                    gamma_j = torch.sigmoid(log_odds_prior + log_bf)
                    gamma_j = torch.clamp(gamma_j, 1e-10, 1.0 - 1e-10)

                    # Update parameters
                    gamma[j] = gamma_j
                    mu[j] = mu_j
                    sigma2[j] = sigma2_j

                    # Subtract new contribution from residual
                    new_effect = gamma_j * mu_j
                    r = r - g_j * new_effect

            # Empirical Bayes hyperparameter update
            if learn_hyperparams and (iteration + 1) % em_interval == 0:
                pi, sig2_beta = self._update_hyperparams(
                    gamma, mu, sigma2, pi, sig2_beta,
                )

            # ELBO check
            if (iteration + 1) % elbo_freq == 0:
                elbo = self._compute_elbo(
                    G_rot, r, w, gamma, mu, sigma2, pi, sig2_beta,
                )
                elbo_trace.append(elbo)

                if len(elbo_trace) >= 2:
                    prev = elbo_trace[-2]
                    rel_change = abs(elbo - prev) / (abs(prev) + 1e-10)
                    if rel_change < tol:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if stable_count >= 3:
                        converged = True
                        logger.info(
                            "CAVI converged at iteration %d (ELBO=%.4f, "
                            "rel_change=%.2e)",
                            iteration + 1, elbo, rel_change,
                        )
                        break

        if not converged:
            logger.warning(
                "CAVI did not converge after %d iterations (ELBO=%.4f)",
                max_iter,
                elbo_trace[-1] if elbo_trace else float("nan"),
            )

        return gamma, mu, sigma2, elbo_trace, converged, pi, sig2_beta

    # ------------------------------------------------------------------
    # ELBO computation
    # ------------------------------------------------------------------

    def _compute_elbo(
        self,
        G_rot: Tensor,
        r: Tensor,
        w: Tensor,
        gamma: Tensor,
        mu: Tensor,
        sigma2: Tensor,
        pi: float,
        sig2_beta: float,
    ) -> float:
        """Compute the Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(y|beta)] - KL(q(beta) || p(beta))

        The likelihood term uses the diagonal form in rotated space.
        """
        n = r.shape[0]

        # --- Likelihood term ---
        # E_q[||W^{1/2} (r + G_rot @ E_q[beta] - G_rot @ beta)||^2]
        # Since r already has E_q[beta] subtracted:
        # E_q[||W^{1/2} r||^2] + sum_j gamma_j * sigma2_j * (g_j^T W g_j)
        # The second term accounts for variance of beta_j under q
        wG = w.unsqueeze(1) * G_rot
        gWg = (G_rot * wG).sum(dim=0)  # (p,)

        # Residual quadratic: r^T W r (r already excludes E_q[beta])
        rWr = torch.dot(r, w * r)

        # Expected variance contribution
        var_term = torch.sum(gamma * sigma2 * gWg)

        # Log-determinant of V (constant across iterations, but needed for ELBO)
        nf = self.null_fit
        d_full = nf.eigenvalues * nf.sig2_g + max(nf.sig2_e, 1e-20)
        log_det = torch.sum(torch.log(d_full))

        # Likelihood: -0.5 * (n*log(2pi) + log|V| + r^T V^{-1} r + var_term)
        ll = -0.5 * (n * math.log(2.0 * math.pi) + log_det + rWr + var_term)

        # --- KL divergence ---
        # KL(q(beta_j) || p(beta_j)) for spike-and-slab
        # = gamma_j * [log(gamma_j/pi) + 0.5*(log(sig2_beta/sigma2_j) - 1
        #              + (sigma2_j + mu_j^2)/sig2_beta)]
        #   + (1-gamma_j) * log((1-gamma_j)/(1-pi))
        eps = 1e-10
        g = gamma
        log_g = torch.log(g + eps)
        log_1mg = torch.log(1.0 - g + eps)
        log_pi = math.log(pi + eps)
        log_1mpi = math.log(1.0 - pi + eps)

        kl_slab = g * (
            log_g - log_pi
            + 0.5 * (math.log(sig2_beta) - torch.log(sigma2 + eps) - 1.0
                      + (sigma2 + mu ** 2) / sig2_beta)
        )
        kl_spike = (1.0 - g) * (log_1mg - log_1mpi)
        kl = torch.sum(kl_slab + kl_spike)

        elbo = ll.item() - kl.item()
        return elbo

    # ------------------------------------------------------------------
    # Empirical Bayes hyperparameter updates
    # ------------------------------------------------------------------

    def _update_hyperparams(
        self,
        gamma: Tensor,
        mu: Tensor,
        sigma2: Tensor,
        pi: float,
        sig2_beta: float,
    ) -> tuple[float, float]:
        """EM M-step: update prior pi and slab variance sig2_beta.

        Returns
        -------
        (pi_new, sig2_beta_new)
        """
        p = gamma.shape[0]

        # pi = mean(gamma)
        pi_new = float(gamma.mean().item())
        pi_new = max(1e-6, min(pi_new, 1.0 - 1e-6))  # clamp

        # sig2_beta = sum(gamma * (mu^2 + sigma2)) / sum(gamma)
        sum_gamma = float(gamma.sum().item())
        if sum_gamma > 1e-6:
            sig2_beta_new = float(
                (gamma * (mu ** 2 + sigma2)).sum().item() / sum_gamma
            )
            sig2_beta_new = max(sig2_beta_new, 1e-6)  # floor
        else:
            sig2_beta_new = sig2_beta  # keep old if no inclusions

        logger.debug(
            "EM update: pi=%.4e -> %.4e, sig2_beta=%.4e -> %.4e",
            pi, pi_new, sig2_beta, sig2_beta_new,
        )
        return pi_new, sig2_beta_new

    # ------------------------------------------------------------------
    # Credible sets
    # ------------------------------------------------------------------

    def _build_credible_sets(
        self,
        G: Tensor,
        gamma: Tensor,
        coverage: float,
        ld_threshold: float,
    ) -> list[list[int]]:
        """Build credible sets by grouping SNPs in LD and accumulating PIPs.

        1. Compute pairwise r^2 between SNPs with PIP > 0.01.
        2. Group into LD-connected components (r^2 > threshold).
        3. Within each group, sort by PIP descending and accumulate until
           cumulative PIP >= coverage.

        Parameters
        ----------
        G : (n, p) original (unrotated) genotypes for LD computation.
        gamma : (p,) posterior inclusion probabilities.
        coverage : float, target cumulative PIP (default 0.95).
        ld_threshold : float, r^2 threshold for LD grouping.

        Returns
        -------
        list of lists of SNP indices (0-based).
        """
        p = gamma.shape[0]

        # Only consider SNPs with non-trivial PIP
        candidates = torch.where(gamma > 0.01)[0].tolist()
        if len(candidates) == 0:
            return []

        # Compute pairwise r^2 among candidates
        G_cand = G[:, candidates].to(STAT_DTYPE)
        # Standardize
        G_centered = G_cand - G_cand.mean(dim=0, keepdim=True)
        G_std = G_centered / (G_centered.std(dim=0, keepdim=True) + 1e-10)
        R = (G_std.T @ G_std) / G_std.shape[0]  # correlation matrix
        R2 = R ** 2  # r-squared

        # Union-find for LD-connected components
        parent = list(range(len(candidates)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if R2[i, j].item() > ld_threshold:
                    union(i, j)

        # Group into components
        groups: dict[int, list[int]] = {}
        for i, cand_idx in enumerate(candidates):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(cand_idx)

        # Build credible sets within each group
        credible_sets: list[list[int]] = []
        for group_indices in groups.values():
            # Sort by PIP descending
            pips = [(idx, gamma[idx].item()) for idx in group_indices]
            pips.sort(key=lambda x: -x[1])

            # Accumulate until coverage
            cs: list[int] = []
            cumsum = 0.0
            for idx, pip_val in pips:
                cs.append(idx)
                cumsum += pip_val
                if cumsum >= coverage:
                    break

            if cs:
                credible_sets.append(cs)

        # Sort credible sets by lead PIP (descending)
        credible_sets.sort(key=lambda cs: -gamma[cs[0]].item())

        return credible_sets

    # ------------------------------------------------------------------
    # SuSiE: IBSS loop
    # ------------------------------------------------------------------

    def _susie_loop(
        self,
        G_rot: Tensor,
        r: Tensor,
        w: Tensor,
        n_signals: int,
        prior_sig2_beta: float,
        max_iter: int,
        tol: float,
        learn_hyperparams: bool,
        em_interval: int,
        elbo_freq: int,
    ) -> tuple[Tensor, Tensor, Tensor, list[float], bool, Tensor]:
        """Iterative Bayesian Stepwise Selection (IBSS) for SuSiE.

        Maintains L independent single-effect layers. Within each layer,
        SNPs compete via softmax (not sigmoid), so per-layer inclusion
        probabilities sum to 1.

        When ``learn_hyperparams`` is True, the per-layer prior variance V_l
        is updated at every layer update using the EM estimator from susieR:

            V_l = sum_j alpha_lj * mu_lj^2

        This is applied after each layer's variational update (not at fixed
        EM intervals), matching the susieR reference implementation.  The EM
        estimator naturally adapts V_l to the data scale without the V=0
        attractor that plagues marginal-likelihood optimization on
        genome-wide data (where the uniform 1/p prior dilutes evidence for
        any single SNP).

        Returns
        -------
        alpha_L : (L, p) per-layer inclusion probabilities
        mu_L : (L, p) per-layer posterior means
        sigma2_L : (L, p) per-layer posterior variances
        elbo_trace : list of ELBO values
        converged : bool
        sig2_l : (L,) per-layer slab variances
        """
        n, p = G_rot.shape
        L = n_signals
        dtype = G_rot.dtype
        device = G_rot.device

        # Precompute weighted genotypes and gWg
        wG = w.unsqueeze(1) * G_rot  # (n, p)
        gWg = (G_rot * wG).sum(dim=0)  # (p,)

        # Initialize L layers uniformly
        alpha_L = torch.full((L, p), 1.0 / p, dtype=dtype, device=device)
        mu_L = torch.zeros(L, p, dtype=dtype, device=device)
        sigma2_L = torch.zeros(L, p, dtype=dtype, device=device)
        sig2_l = torch.full((L,), prior_sig2_beta, dtype=dtype, device=device)

        # Xb[l] = G_rot @ (alpha_l * mu_l)  -- cached predictions per layer
        Xb = torch.zeros(L, n, dtype=dtype, device=device)

        # V floor: 1e-4 of the initial prior variance prevents complete
        # layer collapse while allowing large shrinkage for null layers.
        V_floor = 1e-4 * prior_sig2_beta

        elbo_trace: list[float] = []
        converged = False
        stable_count = 0

        for iteration in range(max_iter):
            for l in range(L):
                # Partial residual: add back layer l's contribution
                r_l = r + Xb[l]

                # Posterior variance: sigma2_lj = 1 / (gWg_j + 1/sig2_l)
                inv_sig2 = 1.0 / sig2_l[l]
                sigma2_lj = 1.0 / (gWg + inv_sig2)  # (p,)

                # Posterior mean: mu_lj = sigma2_lj * (wG^T @ r_l)_j
                scores = wG.T @ r_l  # (p,)  -- one matmul for all SNPs
                mu_lj = sigma2_lj * scores  # (p,)

                # Log Bayes factors
                log_bf = 0.5 * torch.log(sigma2_lj * inv_sig2) + \
                    0.5 * mu_lj ** 2 / sigma2_lj
                log_bf = torch.clamp(log_bf, -500.0, 500.0)

                # Softmax for inclusion (uniform prior 1/p)
                log_alpha = math.log(1.0 / p) + log_bf
                alpha_lj = torch.softmax(log_alpha, dim=0)  # (p,) sums to 1

                # Store updates
                alpha_L[l] = alpha_lj
                mu_L[l] = mu_lj
                sigma2_L[l] = sigma2_lj

                # Update cached prediction and residual
                Xb[l] = G_rot @ (alpha_lj * mu_lj)  # (n,)
                r = r_l - Xb[l]

            # Per-layer EM for V_l at fixed intervals.
            # Updating at intervals (not every layer) lets the variational
            # parameters alpha/mu stabilize before V is re-estimated,
            # yielding better V estimates — especially when p >> signals.
            if learn_hyperparams and (iteration + 1) % em_interval == 0:
                for l_em in range(L):
                    V_new = (alpha_L[l_em] * (
                        mu_L[l_em] ** 2 + sigma2_L[l_em]
                    )).sum().item()
                    sig2_l[l_em] = max(V_new, V_floor)

            # ELBO check
            if (iteration + 1) % elbo_freq == 0:
                elbo = self._susie_compute_elbo(
                    G_rot, r, w, alpha_L, mu_L, sigma2_L, sig2_l, Xb,
                )
                elbo_trace.append(elbo)

                if len(elbo_trace) >= 2:
                    prev = elbo_trace[-2]
                    rel_change = abs(elbo - prev) / (abs(prev) + 1e-10)
                    if rel_change < tol:
                        stable_count += 1
                    else:
                        stable_count = 0

                    if stable_count >= 3:
                        converged = True
                        logger.info(
                            "SuSiE converged at iteration %d (ELBO=%.4f, "
                            "rel_change=%.2e)",
                            iteration + 1, elbo, rel_change,
                        )
                        break

        if not converged:
            logger.warning(
                "SuSiE did not converge after %d iterations (ELBO=%.4f)",
                max_iter,
                elbo_trace[-1] if elbo_trace else float("nan"),
            )

        return alpha_L, mu_L, sigma2_L, elbo_trace, converged, sig2_l

    # ------------------------------------------------------------------
    # SuSiE: ELBO computation
    # ------------------------------------------------------------------

    def _susie_compute_elbo(
        self,
        G_rot: Tensor,
        r: Tensor,
        w: Tensor,
        alpha_L: Tensor,
        mu_L: Tensor,
        sigma2_L: Tensor,
        sig2_l: Tensor,
        Xb: Tensor,
    ) -> float:
        """Compute the SuSiE Evidence Lower Bound.

        ELBO = E_q[log p(y|b_1,...,b_L)] - sum_l KL(q(b_l) || p(b_l))

        The likelihood is evaluated in the rotated eigenspace where V is
        diagonal. Each layer has a Multinomial-Gaussian KL.
        """
        n = r.shape[0]
        L, p = alpha_L.shape

        # --- Likelihood term ---
        # r already = Y_rot - X0_rot @ alpha_hat - sum_l Xb[l]
        rWr = torch.dot(r, w * r)

        # Variance contribution from each layer
        wG_sq = (w.unsqueeze(1) * G_rot) * G_rot  # (n, p) elementwise
        # For each layer, var = sum_j alpha_lj * sigma2_lj * gWg_j
        #   + sum_j alpha_lj * (1-alpha_lj) * mu_lj^2 * gWg_j
        #   (but cross-terms within layer sum to zero since only one effect)
        gWg = wG_sq.sum(dim=0)  # (p,)
        var_term = 0.0
        for l in range(L):
            # E[b_l^T W b_l] - (E[b_l])^T W (E[b_l])
            # = sum_j alpha_lj*(mu_lj^2 + sigma2_lj)*gWg_j - Xb[l]^T W Xb[l]/...
            # Simplified: just the variance part
            var_l = (alpha_L[l] * (sigma2_L[l] + mu_L[l] ** 2) * gWg).sum()
            mean_sq_l = torch.dot(Xb[l], w * Xb[l])
            var_term += (var_l - mean_sq_l).item()

        nf = self.null_fit
        d_full = nf.eigenvalues * nf.sig2_g + max(nf.sig2_e, 1e-20)
        log_det = torch.sum(torch.log(d_full))

        ll = -0.5 * (n * math.log(2.0 * math.pi) + log_det + rWr.item()
                      + var_term)

        # --- KL divergence per layer ---
        # KL(q(b_l) || p(b_l)) = KL_categorical + E_alpha[KL_gaussian]
        # KL_cat = sum_j alpha_lj * log(alpha_lj / (1/p))
        # KL_gauss = sum_j alpha_lj * 0.5 * (log(sig2_l/sigma2_lj) - 1
        #            + (sigma2_lj + mu_lj^2) / sig2_l)
        eps = 1e-30
        kl_total = 0.0
        for l in range(L):
            a = alpha_L[l]  # (p,)
            # Categorical KL
            kl_cat = (a * (torch.log(a + eps) - math.log(1.0 / p))).sum()

            # Weighted Gaussian KL
            s2_prior = sig2_l[l]
            kl_gauss = (a * 0.5 * (
                math.log(s2_prior.item() + eps) - torch.log(sigma2_L[l] + eps)
                - 1.0
                + (sigma2_L[l] + mu_L[l] ** 2) / s2_prior
            )).sum()

            kl_total += kl_cat.item() + kl_gauss.item()

        return ll - kl_total

    # ------------------------------------------------------------------
    # SuSiE: credible sets (per-layer)
    # ------------------------------------------------------------------

    def _build_susie_credible_sets(
        self,
        alpha_L: Tensor,
        coverage: float,
    ) -> list[list[int]]:
        """Build per-layer credible sets from SuSiE alpha.

        For each active layer, sort alpha_l descending and accumulate
        until >= coverage. Skip inactive layers (max alpha < 2/p).

        Returns
        -------
        list of lists of SNP indices (0-based), sorted by lead alpha.
        """
        L, p = alpha_L.shape
        threshold = 2.0 / p  # activity threshold
        credible_sets: list[list[int]] = []

        for l in range(L):
            a = alpha_L[l]
            if a.max().item() < threshold:
                continue  # inactive layer

            # Sort by alpha descending
            sorted_vals, sorted_idx = torch.sort(a, descending=True)
            cs: list[int] = []
            cumsum = 0.0
            for i in range(p):
                cs.append(sorted_idx[i].item())
                cumsum += sorted_vals[i].item()
                if cumsum >= coverage:
                    break
            credible_sets.append(cs)

        # Sort credible sets by lead alpha (descending)
        credible_sets.sort(
            key=lambda cs: -alpha_L[:, cs[0]].max().item(),
        )
        return credible_sets
