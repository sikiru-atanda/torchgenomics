"""Multi-ancestry GWAS meta-analysis.

Standard fixed/random-effects meta-analysis (in ``_meta.py``) treats
between-study heterogeneity as nuisance. In multi-ancestry contexts,
effect-size heterogeneity across populations is *expected* (different LD,
different allele frequencies, different environmental modifiers), and the
goal is to borrow strength while respecting genuine differences.

Two methods are provided:

* **MR-MEGA** (Magi et al. 2017) -- Meta-Regression of Multi-AncEstry
  Genetic Association. Models effect heterogeneity as a function of
  axes of genetic variation (principal components of an allele-frequency
  correlation matrix across populations). The residual heterogeneity
  test separates genuine ancestry-driven effect modification from
  unexplained heterogeneity. Fixed-effect and random-effect variants.

* **MANTRA** (Morris 2011) -- Meta-ANalysis of Transethnic Association
  studies. Bayesian clustering: each SNP's effects across populations
  are modelled as arising from a mixture of a "consistent" (shared
  effect) model and a "heterogeneous" (population-specific effects)
  model, with a Bayes factor quantifying evidence for association.
  Uses a simplified conjugate-normal model for tractability.

**Polyploid compatibility.** Both methods operate on summary statistics
(beta, SE, p-values) and allele frequencies. AF is in [0, 1] regardless
of ploidy, so the PCA on per-population AF and the WLS / Bayesian
inference are valid for diploid, tetraploid, and higher ploidy organisms.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ._sumstats import SumStats, align_sumstats


@dataclass
class MultiAncestryResult:
    """Multi-ancestry meta-analysis result."""

    # Per-SNP results
    beta_meta: Tensor  # (m,) meta-analytic effect
    se_meta: Tensor  # (m,) SE
    p_meta: Tensor  # (m,) association p-value
    p_heterogeneity: Tensor  # (m,) heterogeneity test p-value

    # Method identifier
    method: str  # "mr_mega" or "mantra"

    # MR-MEGA specific
    p_ancestry: Tensor | None = None  # (m,) ancestry-correlated het. p
    p_residual: Tensor | None = None  # (m,) residual het. p
    n_axes: int = 0  # number of PCs used

    # MANTRA specific
    log10_bf: Tensor | None = None  # (m,) log10 Bayes factor
    posterior_effect: Tensor | None = None  # (m,) posterior mean effect

    n_populations: int = 0
    n_snps: int = 0


# ── Chi-squared survival for integer df ───────────────────────────────


def _chi2_sf(stat: Tensor, df: int) -> Tensor:
    """Chi-squared survival function for a fixed integer df."""
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=stat.dtype, device=stat.device)


def _chi2_sf_variable_df(stat: Tensor, df: Tensor) -> Tensor:
    """Chi-squared survival function with per-element df (both 1-D)."""
    import scipy.stats as sp_stats

    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    df_np = df.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.chi2.sf(stat_np, df=df_np)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=stat.dtype, device=stat.device)


def _z_to_p(z: Tensor) -> Tensor:
    """Two-sided p-value from z-score."""
    p = torch.erfc(z.abs() / 2.0**0.5)
    return p.clamp(min=1e-300, max=1.0)


# ── SNP alignment helper ─────────────────────────────────────────────


def _align_populations(
    sumstats_list: list[SumStats],
) -> list[SumStats]:
    """Align summary statistics across populations by SNP ID.

    Uses :func:`align_sumstats` for inner-join alignment with allele
    flipping.  Warns if significant SNP loss occurs (>50% dropped).

    Parameters
    ----------
    sumstats_list : list[SumStats]
        Per-population summary statistics.

    Returns
    -------
    list[SumStats]
        Aligned summary statistics with identical SNP sets and order.
    """
    total_snps = [ss.m for ss in sumstats_list]
    aligned = align_sumstats(sumstats_list, match_alleles=True)
    m_aligned = aligned[0].m

    max_input = max(total_snps)
    if m_aligned < max_input * 0.5:
        warnings.warn(
            f"Only {m_aligned} of up to {max_input} SNPs remain after "
            f"alignment ({m_aligned / max_input:.1%}). Consider checking "
            f"SNP ID consistency across populations.",
            stacklevel=3,
        )

    return aligned


# ── MR-MEGA ───────────────────────────────────────────────────────────


def mr_mega(
    sumstats_list: list[SumStats],
    n_axes: int = 4,
    random_effects: bool = False,
) -> MultiAncestryResult:
    """MR-MEGA multi-ancestry meta-regression (Magi et al. 2017).

    Models effect-size heterogeneity across populations as a function of
    axes of genetic variation (PCs of the cross-population allele-frequency
    correlation matrix).  Decomposes heterogeneity into an ancestry-
    correlated component and a residual component.

    Parameters
    ----------
    sumstats_list : list[SumStats]
        Per-population GWAS summary statistics.  Must contain at least 2
        populations.
    n_axes : int
        Number of principal-component axes of genetic variation to include
        in the meta-regression.  Capped at ``K - 1`` where ``K`` is the
        number of populations.
    random_effects : bool
        If True, estimate between-study residual variance (tau^2) via
        DerSimonian-Laird and inflate per-study variances accordingly.

    Returns
    -------
    MultiAncestryResult
        Contains ``p_meta`` (association), ``p_ancestry`` (ancestry
        heterogeneity), ``p_residual`` (residual heterogeneity), and
        ``p_heterogeneity`` (total heterogeneity).

    Raises
    ------
    ValueError
        If fewer than 2 populations or ``n_axes`` >= ``K``.
    """
    K = len(sumstats_list)
    if K < 2:
        raise ValueError(
            f"MR-MEGA requires at least 2 populations, got {K}."
        )
    if n_axes >= K:
        raise ValueError(
            f"n_axes ({n_axes}) must be < K ({K}).  "
            f"At most K-1 = {K - 1} axes are identifiable."
        )
    n_axes = min(n_axes, K - 1)

    # ── 1. Align SNPs ────────────────────────────────────────────────
    aligned = _align_populations(sumstats_list)
    m = aligned[0].m
    device = aligned[0].beta.device
    dtype = torch.float64

    # ── 2. Build allele-frequency matrix F: (K, m) ───────────────────
    af_list = []
    for ss in aligned:
        if ss.af is not None:
            af_list.append(ss.af.to(dtype))
        else:
            af_list.append(
                torch.full((m,), 0.5, dtype=dtype, device=device)
            )
    F = torch.stack(af_list, dim=0)  # (K, m)

    # ── 3. PCA on F across populations ───────────────────────────────
    # Center F across SNPs (each population's AF centred to mean 0).
    F_centered = F - F.mean(dim=1, keepdim=True)  # (K, m)

    # Correlation matrix across populations: (K, K)
    # Use the sample covariance of the centred rows, then normalise.
    cov = (F_centered @ F_centered.T) / max(m - 1, 1)  # (K, K)
    std = cov.diag().sqrt().clamp(min=1e-12)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

    # Eigendecompose the correlation matrix
    eigvals, eigvecs = torch.linalg.eigh(corr)
    # eigh returns ascending order; reverse for descending
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)

    # Take top n_axes eigenvectors as PC loadings: (K, n_axes). Centre each
    # column so the PC axes are orthogonal to the intercept — otherwise the
    # intercept ↔ PC1 near-collinearity inflates (X'WX)^{-1}[0,0] and the
    # intercept Wald test loses power even on strongly shared signals.
    PC = eigvecs[:, :n_axes]  # (K, n_axes)
    if n_axes > 0:
        PC = PC - PC.mean(dim=0, keepdim=True)

    # ── 4. Stack per-population effects and SEs ──────────────────────
    beta_all = torch.stack(
        [ss.beta.to(dtype) for ss in aligned], dim=1
    )  # (m, K)
    se_all = torch.stack(
        [ss.se.to(dtype) for ss in aligned], dim=1
    )  # (m, K)

    # ── 5. Per-SNP meta-regression via WLS ───────────────────────────
    # Design matrix X: (K, 1 + n_axes) — intercept + PC axes
    X = torch.cat(
        [torch.ones(K, 1, dtype=dtype, device=device), PC], dim=1
    )  # (K, p) where p = 1 + n_axes
    p_coef = X.shape[1]

    # Pre-allocate outputs
    beta_meta = torch.zeros(m, dtype=dtype, device=device)
    se_meta = torch.zeros(m, dtype=dtype, device=device)
    gamma_coef = torch.zeros(m, n_axes, dtype=dtype, device=device)

    # Heterogeneity statistics
    q_total = torch.zeros(m, dtype=dtype, device=device)
    q_ancestry = torch.zeros(m, dtype=dtype, device=device)
    q_residual = torch.zeros(m, dtype=dtype, device=device)

    # Degrees of freedom
    df_total = K - 1
    df_ancestry = n_axes
    df_residual = max(K - 1 - n_axes, 0)

    for j in range(m):
        y = beta_all[j]  # (K,)
        se = se_all[j]  # (K,)
        w = 1.0 / (se**2)  # (K,)

        # ── Optional random-effects tau^2 estimation ─────────────
        tau2 = torch.tensor(0.0, dtype=dtype, device=device)
        if random_effects:
            # Initial fixed-effect estimate for DL tau^2
            w_sum = w.sum()
            beta_fe = (w * y).sum() / w_sum
            resid_fe = y - beta_fe
            q_fe = (w * resid_fe**2).sum()
            if q_fe > df_total:
                w_sq_sum = (w**2).sum()
                c = w_sum - w_sq_sum / w_sum
                tau2 = ((q_fe - df_total) / c).clamp(min=0.0)
            w = 1.0 / (se**2 + tau2)

        # ── WLS: (X'WX)^{-1} X'Wy ───────────────────────────────
        W = torch.diag(w)  # (K, K)
        XtW = X.T @ W  # (p, K)
        XtWX = XtW @ X  # (p, p)
        XtWy = XtW @ y  # (p,)

        # Solve via Cholesky for numerical stability
        try:
            L = torch.linalg.cholesky(XtWX)
            coef = torch.cholesky_solve(XtWy.unsqueeze(1), L).squeeze(1)
            XtWX_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            # Fallback to lstsq if XtWX is singular
            coef = torch.linalg.lstsq(XtWX, XtWy).solution
            XtWX_inv = torch.linalg.pinv(XtWX)

        mu = coef[0]
        se_mu = XtWX_inv[0, 0].sqrt()
        beta_meta[j] = mu
        se_meta[j] = se_mu

        if n_axes > 0:
            gamma_coef[j] = coef[1:]

        # ── Heterogeneity decomposition ──────────────────────────
        # Fitted values and residuals
        fitted = X @ coef  # (K,)
        resid = y - fitted  # (K,)

        # Total Q: weighted SS of deviations from the intercept-only fit
        beta_intercept = (w * y).sum() / w.sum()
        resid_total = y - beta_intercept
        q_total[j] = (w * resid_total**2).sum()

        # Residual Q: weighted SS of residuals from the full model
        q_residual[j] = (w * resid**2).sum()

        # Ancestry Q: difference
        q_ancestry[j] = q_total[j] - q_residual[j]

    # ── 6. Compute p-values ──────────────────────────────────────────
    # Association: Wald test for mu = 0
    z_meta = beta_meta / se_meta.clamp(min=1e-300)
    p_meta = _z_to_p(z_meta)

    # Total heterogeneity: chi2 with K-1 df
    p_heterogeneity = _chi2_sf(q_total, df=df_total)

    # Ancestry-correlated heterogeneity: chi2 with n_axes df
    if n_axes > 0:
        p_ancestry = _chi2_sf(q_ancestry, df=df_ancestry)
    else:
        p_ancestry = torch.ones(m, dtype=dtype, device=device)

    # Residual heterogeneity: chi2 with K-1-n_axes df
    if df_residual > 0:
        p_residual = _chi2_sf(q_residual, df=df_residual)
    else:
        p_residual = torch.ones(m, dtype=dtype, device=device)

    return MultiAncestryResult(
        beta_meta=beta_meta,
        se_meta=se_meta,
        p_meta=p_meta,
        p_heterogeneity=p_heterogeneity,
        method="mr_mega",
        p_ancestry=p_ancestry,
        p_residual=p_residual,
        n_axes=n_axes,
        log10_bf=None,
        posterior_effect=None,
        n_populations=K,
        n_snps=m,
    )


# ── MANTRA ────────────────────────────────────────────────────────────


def mantra(
    sumstats_list: list[SumStats],
    prior_sigma2: float = 0.04,
    prior_het_sigma2: float = 0.04,
    prior_prob_consistent: float = 0.5,
) -> MultiAncestryResult:
    """MANTRA Bayesian trans-ethnic meta-analysis (Morris 2011).

    Computes per-SNP Bayes factors under a mixture of a consistent-effect
    model (all populations share a common effect) and a heterogeneous-effect
    model (each population has an independent effect).  Uses conjugate
    normal priors for tractability.

    Parameters
    ----------
    sumstats_list : list[SumStats]
        Per-population GWAS summary statistics (>= 2 populations).
    prior_sigma2 : float
        Prior variance for the shared effect under the consistent model.
    prior_het_sigma2 : float
        Prior variance for each population-specific effect under the
        heterogeneous model.
    prior_prob_consistent : float
        Prior probability of the consistent (shared-effect) model vs the
        heterogeneous model, in [0, 1].

    Returns
    -------
    MultiAncestryResult
        Contains ``log10_bf`` (log10 Bayes factor for association),
        ``posterior_effect`` (posterior mean under the consistent model),
        ``p_meta`` (approximate p-value from BF), and ``p_heterogeneity``
        (BF-derived heterogeneity indicator).

    Raises
    ------
    ValueError
        If fewer than 2 populations or invalid prior parameters.
    """
    K = len(sumstats_list)
    if K < 2:
        raise ValueError(
            f"MANTRA requires at least 2 populations, got {K}."
        )
    if prior_sigma2 <= 0:
        raise ValueError(
            f"prior_sigma2 must be positive, got {prior_sigma2}."
        )
    if prior_het_sigma2 <= 0:
        raise ValueError(
            f"prior_het_sigma2 must be positive, got {prior_het_sigma2}."
        )
    if not 0.0 <= prior_prob_consistent <= 1.0:
        raise ValueError(
            f"prior_prob_consistent must be in [0, 1], "
            f"got {prior_prob_consistent}."
        )

    # ── 1. Align SNPs ────────────────────────────────────────────────
    aligned = _align_populations(sumstats_list)
    m = aligned[0].m
    device = aligned[0].beta.device
    dtype = torch.float64

    # Stack effects and SEs: (m, K)
    beta_all = torch.stack(
        [ss.beta.to(dtype) for ss in aligned], dim=1
    )
    se_all = torch.stack(
        [ss.se.to(dtype) for ss in aligned], dim=1
    )

    # ── 2. Consistent model log-BF ───────────────────────────────────
    # Prior: mu ~ N(0, prior_sigma2)
    # Likelihood: beta_k | mu ~ N(mu, se_k^2)  independently
    # Marginal: beta | H_c is multivariate normal.
    # log BF_c = 0.5 * log(V_post / prior_sigma2) + 0.5 * mu_post^2 / V_post
    #
    # V_post = 1 / (1/prior_sigma2 + sum_k 1/se_k^2)
    # mu_post = V_post * sum_k (beta_k / se_k^2)

    var_all = se_all**2  # (m, K)
    precision_sum = (1.0 / var_all).sum(dim=1)  # (m,)

    V_post_c = 1.0 / (1.0 / prior_sigma2 + precision_sum)  # (m,)
    mu_post_c = V_post_c * (beta_all / var_all).sum(dim=1)  # (m,)

    log_bf_c = 0.5 * torch.log(V_post_c / prior_sigma2) + (
        0.5 * mu_post_c**2 / V_post_c
    )

    # ── 3. Heterogeneous model log-BF ────────────────────────────────
    # Prior: mu_k ~ N(0, prior_het_sigma2) independently for each k
    # Likelihood: beta_k | mu_k ~ N(mu_k, se_k^2)
    # Per-population marginal BF, then sum in log space.
    #
    # V_k = 1 / (1/prior_het_sigma2 + 1/se_k^2)
    # mu_k = V_k * (beta_k / se_k^2)
    # log BF_k = 0.5 * log(V_k / prior_het_sigma2) + 0.5 * mu_k^2 / V_k

    V_k = 1.0 / (1.0 / prior_het_sigma2 + 1.0 / var_all)  # (m, K)
    mu_k = V_k * (beta_all / var_all)  # (m, K)

    log_bf_k = 0.5 * torch.log(V_k / prior_het_sigma2) + (
        0.5 * mu_k**2 / V_k
    )
    log_bf_h = log_bf_k.sum(dim=1)  # (m,)

    # ── 4. Combined BF via mixture prior ─────────────────────────────
    # BF = pi_c * exp(log_bf_c) + (1 - pi_c) * exp(log_bf_h)
    # Compute in log space for numerical stability using log-sum-exp.

    pi_c = prior_prob_consistent
    log_pi_c = torch.tensor(pi_c, dtype=dtype, device=device).log()
    log_pi_h = torch.tensor(1.0 - pi_c, dtype=dtype, device=device).log()

    # log(BF) = logsumexp(log_pi_c + log_bf_c, log_pi_h + log_bf_h)
    term_c = log_pi_c + log_bf_c  # (m,)
    term_h = log_pi_h + log_bf_h  # (m,)
    stacked = torch.stack([term_c, term_h], dim=1)  # (m, 2)
    log_bf = torch.logsumexp(stacked, dim=1)  # (m,)

    # Convert to log10
    log10_bf = log_bf / torch.log(torch.tensor(10.0, dtype=dtype, device=device))

    # ── 5. Posterior mean effect (under consistent model) ────────────
    posterior_effect = mu_post_c

    # ── 6. Approximate p-value from BF ───────────────────────────────
    # Use the Wakefield approximation: 2 * log(BF) ~ chi2(1) under H1.
    # This gives a rough frequentist p-value for comparison.
    # For negative log-BF (evidence for null), clamp to p = 1.
    stat_approx = (2.0 * log_bf).clamp(min=0.0)
    p_meta = _chi2_sf(stat_approx, df=1)

    # ── 7. Heterogeneity indicator ───────────────────────────────────
    # Compare consistent vs heterogeneous BF: if het model dominates,
    # there is evidence for effect heterogeneity.
    # p_heterogeneity: approximate p from 2*(log_bf_h - log_bf_c) ~ chi2(K-1)
    het_stat = (2.0 * (log_bf_h - log_bf_c)).clamp(min=0.0)
    p_heterogeneity = _chi2_sf(het_stat, df=K - 1)

    # ── 8. Meta-analytic effect and SE (from consistent model) ───────
    beta_meta = posterior_effect
    se_meta = V_post_c.sqrt()

    return MultiAncestryResult(
        beta_meta=beta_meta,
        se_meta=se_meta,
        p_meta=p_meta,
        p_heterogeneity=p_heterogeneity,
        method="mantra",
        p_ancestry=None,
        p_residual=None,
        n_axes=0,
        log10_bf=log10_bf,
        posterior_effect=posterior_effect,
        n_populations=K,
        n_snps=m,
    )
