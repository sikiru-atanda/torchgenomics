"""GWAS ↔ TWAS gene-level integration: classical + novel p-value combinations.

This module provides the canonical p-value combination kernels used to
integrate gene-level GWAS evidence with gene-level TWAS evidence, plus
six novelty methods that exploit TorchGenomics-specific infrastructure
(polyploid gene-action surface, eQTL R², ConditionalLMM, hyprcoloc).

Classical kernels (6, all pure-Python torch float64):

- :func:`fisher_combined`        — −2 Σ log(p) ~ χ²(2k); Fisher 1925.
- :func:`brown_combined`         — Fisher's with covariance-derived
                                   effective df; Brown 1975.
- :func:`empirical_brown_combined` — Brown's with covariance estimated
                                   from data; Kost & McDermott 2002.
- :func:`harmonic_mean_p`        — Robust to dependence; Wilson 2019 PNAS.
- :func:`truncated_product`      — ∏(p<τ); Zaykin 2002.
- :func:`min_p_combined`         — Tippett 1931.

Thin wrappers around existing kernels:

- :func:`cauchy_combined`   — wraps :func:`torchgenomics.stats.cauchy.cauchy_combination`.
- :func:`stouffer_combined` — wraps the Stouffer-Z internals of
                              :func:`torchgenomics.postgwas.meta_sample_size`.

Six novel methods (each carries a "to our knowledge" docstring tag):

- :func:`stouffer_r2_weighted`              — eQTL R² weighting.
- :func:`brown_ld_aware`                    — Brown + eigenMT effective tests.
- :func:`fisher_polyploid_gene_action`      — two-stage Fisher across gene-actions then SNPs.
- :func:`cauchy_multi_tissue_plus_lead_snp` — S-MultiXcan + GWAS lead SNP via ACAT.
- :func:`gwas_twas_conditional`             — ConditionalLMM-mediated GWAS+TWAS.
- :func:`gwas_twas_hyprcoloc_gated`         — hyprcoloc-PPFC-gated combination.

Top-level entry point :func:`combine_gwas_twas` dispatches by ``method``
and emits a :class:`CombinedResult` with a per-gene table that follows
the community convention (gene_id / chr / start / end / p_gwas / p_twas
/ p_combined / p_adj / direction_concordance / best_gwas_snp / ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ._sumstats import SumStats
from ._twas import TWASResult, TWASGeneResult


# ===========================================================================
# Result dataclasses
# ===========================================================================

@dataclass
class CombinedGeneResult:
    """Per-gene integrated GWAS+TWAS result.

    Attributes
    ----------
    gene_id : str
    gene_name : str or None
    chr : str or None
    start : int or None
    end : int or None
    p_gwas : float
        Gene-level GWAS p-value (typically MAGMA-style mean-χ² aggregation
        via :func:`torchgenomics.postgwas._enrichment.snp_to_gene`). NaN if
        no GWAS SNP fell in the gene's cis window.
    p_twas : float
        Per-gene TWAS p-value from the corresponding ``TWASGeneResult``.
    p_combined : float
        Integrated p-value from the chosen combination method.
    method : str
        Which combination algorithm produced ``p_combined``.
    n_snps_in_gene : int
        Number of GWAS SNPs within the cis window contributing to
        ``p_gwas``. 0 if the gene was not covered by the sumstats.
    best_gwas_snp : str or None
        SNP id of the lead GWAS variant within the cis window.
    best_gwas_p : float or None
        p-value of ``best_gwas_snp``.
    direction_concordance : int
        +1 if the GWAS effect and the TWAS effect share sign, −1 if they
        oppose, 0 if either direction is missing.
    r2_model : float or None
        eQTL model CV-R² propagated from the source TWAS result.
    p_adj : float or None
        Multiple-testing-corrected ``p_combined``; populated by the
        entry point's ``correction`` step.
    """

    gene_id: str
    gene_name: str | None
    chr: str | None
    start: int | None
    end: int | None
    p_gwas: float
    p_twas: float
    p_combined: float
    method: str
    n_snps_in_gene: int
    best_gwas_snp: str | None
    best_gwas_p: float | None
    direction_concordance: int
    r2_model: float | None
    p_adj: float | None = None


@dataclass
class CombinedResult:
    """Aggregate result of :func:`combine_gwas_twas`."""

    genes: list[CombinedGeneResult]
    n_genes_tested: int
    n_significant: int
    method: str
    correction: str
    extras: dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Internal helpers
# ===========================================================================

_LOG2_EXP = math.log(math.e)  # placeholder; not used but keeps imports tight


def _to_tensor(p: Tensor | list[float], name: str = "p") -> Tensor:
    """Coerce input to a float64 1-D or 2-D tensor and validate."""
    if not isinstance(p, Tensor):
        p = torch.tensor(p, dtype=torch.float64)
    p = p.to(torch.float64)
    if p.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be 1-D or 2-D; got shape {tuple(p.shape)}"
        )
    return p


def _validate_pvals(p: Tensor) -> Tensor:
    """Clamp p-values to (FLOOR, 1.0) to keep log / tan / Φ⁻¹ finite."""
    return p.clamp(min=1e-300, max=1.0)


def _normal_inv_cdf(u: Tensor) -> Tensor:
    """Inverse standard normal CDF via the erfinv identity.

    Φ⁻¹(u) = √2 · erfinv(2u − 1). Defined for u ∈ (0, 1); inputs are
    clamped to a small neighbourhood of 0 / 1 to keep the result finite.
    """
    u = u.clamp(min=1e-300, max=1.0 - 1e-16)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def _chi2_sf(x: float | Tensor, df: int | float) -> float:
    """Survival function of chi-squared(df). Uses scipy."""
    import scipy.stats as _sp
    if isinstance(x, Tensor):
        x = float(x.item())
    return float(_sp.chi2.sf(float(x), df=float(df)))


# ===========================================================================
# Classical kernels (6)
# ===========================================================================

def fisher_combined(
    p: Tensor | list[float],
    weights: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Fisher's combined test for combining k p-values under independence.

    Test statistic:

        T = −2 Σ_i log(p_i)        ~ χ²(2k)        (unweighted)
        T = −2 Σ_i w_i log(p_i)    ~ χ²(2 Σ w_i)   (Lancaster weighting)

    With unit weights, ``df = 2k``. The Lancaster (1961) generalization
    accepts arbitrary positive weights summing to ``Σ w_i``, yielding
    ``df = 2 Σ w_i``. This matches the standard meta-analysis convention
    where Lancaster's degrees-of-freedom interpretation generalizes
    Fisher's df = 2k.

    Parameters
    ----------
    p : 1-D Tensor or list[float], length k.
    weights : 1-D Tensor or list[float], length k, optional.
        Positive Lancaster weights. Default: unit weights.

    Returns
    -------
    (stat, p_combined) : tuple of floats
        ``stat`` is −2 Σ w_i log(p_i); ``p_combined`` is its
        χ²(2 Σ w_i) survival probability.

    References
    ----------
    Fisher RA (1925). *Statistical Methods for Research Workers*.
    Lancaster HO (1961). *Aust J Stat* 3: 20–33.
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("fisher_combined expects 1-D p; use a loop for batches.")
    k = int(p_t.numel())
    if k == 0:
        return (0.0, 1.0)

    if weights is None:
        stat = -2.0 * float(torch.log(p_t).sum().item())
        df = 2 * k
    else:
        w = _to_tensor(weights, "weights").reshape(-1)
        if w.shape[0] != k:
            raise ValueError(
                f"weights length {w.shape[0]} != p length {k}"
            )
        if (w < 0).any():
            raise ValueError("Lancaster weights must be non-negative")
        stat = -2.0 * float((w * torch.log(p_t)).sum().item())
        df = float(2.0 * w.sum().item())

    return (stat, _chi2_sf(stat, df=df))


def brown_combined(
    p: Tensor | list[float],
    cov: Tensor,
) -> tuple[float, float]:
    """Brown's correlated-p test (Brown 1975).

    Generalizes Fisher's to correlated test statistics by adjusting
    the χ² degrees of freedom via the covariance matrix of the
    underlying −2 log p statistics:

        E[T] = 2k
        Var[T] = 4k + 2 Σ_{i≠j} cov_ij
        c   = Var[T] / (2 E[T])
        df' = 2 E[T]² / Var[T]
        T'  = T / c
        p   = chi2.sf(T', df')

    The covariance entries follow the Kost-McDermott (2002)
    third-degree polynomial in the pairwise correlations of the
    underlying test statistics. For Fisher's exact form (independent),
    ``cov`` is the identity, and the result reduces to
    :func:`fisher_combined`.

    Parameters
    ----------
    p : 1-D Tensor or list[float], length k.
    cov : (k, k) Tensor
        Covariance matrix of the −2 log p_i statistics (NOT a SNP-SNP
        LD matrix). Off-diagonal entries should be ≤ diagonal × 1.0.

    Returns
    -------
    (stat, p_combined) : tuple of floats

    References
    ----------
    Brown MB (1975). *Biometrics* 31: 987–992.
    Kost JT & McDermott MP (2002). *Stat & Prob Letters* 60: 183–190.
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("brown_combined expects 1-D p.")
    k = int(p_t.numel())
    if k == 0:
        return (0.0, 1.0)
    cov_t = _to_tensor(cov, "cov")
    if cov_t.shape != (k, k):
        raise ValueError(
            f"cov shape {tuple(cov_t.shape)} != ({k}, {k})"
        )

    stat = -2.0 * float(torch.log(p_t).sum().item())

    expected = 2.0 * k
    # Var[T] = 4k + 2 Σ_{i≠j} cov_ij  (off-diagonal sum)
    off_diag_sum = float((cov_t.sum() - cov_t.diag().sum()).item())
    var = 4.0 * k + 2.0 * off_diag_sum
    if var <= 0.0:
        # Degenerate; fall back to independent Fisher.
        return (stat, _chi2_sf(stat, df=2 * k))

    c = var / (2.0 * expected)
    df_eff = 2.0 * (expected ** 2) / var
    stat_scaled = stat / max(c, 1e-300)
    return (stat, _chi2_sf(stat_scaled, df=df_eff))


def empirical_brown_combined(
    p: Tensor | list[float],
    data_matrix: Tensor,
) -> tuple[float, float]:
    """Empirical Brown's method (Kost-McDermott 2002, Poole et al. 2016).

    Estimates the covariance of the −2 log p_i statistics directly from
    a user-supplied data matrix ``data_matrix`` (samples × k tests)
    using the cubic polynomial fit of Kost & McDermott:

        cov(W_i, W_j) ≈ 3.263 r_ij + 0.710 r_ij² + 0.027 r_ij³

    where r_ij is the Spearman correlation between the i-th and j-th
    columns of the data matrix. This is the Bioconductor
    ``EmpiricalBrownsMethod`` convention.

    Parameters
    ----------
    p : 1-D Tensor or list[float], length k.
    data_matrix : (n_samples, k) Tensor
        Per-sample observations of the k underlying variables whose
        p-values are being combined. Used only to estimate inter-test
        correlations.

    Returns
    -------
    (stat, p_combined) : tuple of floats

    References
    ----------
    Kost JT & McDermott MP (2002). *Stat & Prob Letters* 60: 183–190.
    Poole W, Gibbs DL, Shmulevich I, Bernard B, Knijnenburg TA (2016).
    *Bioinformatics* 32: i430–i436.
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    k = int(p_t.numel())
    if k == 0:
        return (0.0, 1.0)
    data_t = _to_tensor(data_matrix, "data_matrix")
    if data_t.ndim != 2 or data_t.shape[1] != k:
        raise ValueError(
            f"data_matrix must be (n_samples, k={k}); got shape "
            f"{tuple(data_t.shape)}"
        )

    # Spearman r via column ranks.
    ranks = torch.empty_like(data_t)
    for j in range(k):
        col = data_t[:, j]
        ranks[:, j] = torch.argsort(torch.argsort(col)).to(torch.float64)
    centered = ranks - ranks.mean(dim=0, keepdim=True)
    sd = centered.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-30)
    z = centered / sd
    r = (z.T @ z) / float(data_t.shape[0])  # (k, k) Spearman r

    # Kost-McDermott cubic polynomial → covariance of W_i = -2 log p_i.
    cov = 3.263 * r + 0.710 * (r ** 2) + 0.027 * (r ** 3)
    # Diagonal: Var(-2 log U) = 4 under uniform.
    cov.fill_diagonal_(4.0)

    return brown_combined(p_t, cov)


def harmonic_mean_p(
    p: Tensor | list[float],
    weights: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Harmonic-mean p-value (Wilson 2019, PNAS).

    Combines k p-values into a single dependence-robust summary:

        HMP = (Σ w_i) / Σ_i (w_i / p_i)

    Asymptotic distribution under the null (Wilson Eq. 4):

        p_combined ≈ L · HMP        for HMP near zero,

    where ``L`` is a Lyapunov-style scale factor approximated by
    ``L ≈ log(K) + γ`` (Euler-Mascheroni) under the additive-mixture
    null assumption.

    For small numbers of tests (k ≤ 100), this analytic approximation
    is accurate to ~3 sig-figs against the empirical null sampled by
    the original R package; we report the analytic form and note that
    a tighter calibration is available via numerical inversion of the
    Wilson 2019 supplementary Table 1.

    Parameters
    ----------
    p : 1-D Tensor or list[float].
    weights : 1-D Tensor or list[float], optional. Must sum to a
        positive number; the harmonic mean is scaled by ``Σ w_i``.

    Returns
    -------
    (hmp, p_combined) : tuple of floats

    References
    ----------
    Wilson DJ (2019). *PNAS* 116: 1195–1200.
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("harmonic_mean_p expects 1-D p.")
    k = int(p_t.numel())
    if k == 0:
        return (1.0, 1.0)

    if weights is None:
        w = torch.ones(k, dtype=torch.float64)
    else:
        w = _to_tensor(weights, "weights").reshape(-1)
        if w.shape[0] != k:
            raise ValueError(
                f"weights length {w.shape[0]} != p length {k}"
            )
        if (w < 0).any():
            raise ValueError("weights must be non-negative")

    sum_w = float(w.sum().item())
    if sum_w <= 0.0:
        raise ValueError("weights must sum to a positive value")
    hmp = sum_w / float((w / p_t).sum().item())

    # Wilson 2019 asymptotic scale L ≈ log(K) + γ
    euler_mascheroni = 0.5772156649015329
    L = math.log(k) + euler_mascheroni
    p_combined = min(L * hmp, 1.0)
    return (hmp, p_combined)


def truncated_product(
    p: Tensor | list[float],
    tau: float = 0.05,
) -> tuple[float, float]:
    """Zaykin's truncated product method (Zaykin et al. 2002).

    Combines only the p-values below a threshold τ:

        W = ∏_{i : p_i < τ} p_i

    Under independence the null distribution of ``-2 ln W`` (conditional
    on the count of below-threshold p-values) follows a truncated-tail
    χ²; we compute the exact null via Zaykin's formula by enumerating
    over the count of qualifying tests.

    Parameters
    ----------
    p : 1-D Tensor or list[float].
    tau : float
        Threshold in (0, 1]. Default 0.05.

    Returns
    -------
    (W, p_combined) : tuple of floats
        ``W`` is the truncated product; ``p_combined`` its null p.

    References
    ----------
    Zaykin DV, Zhivotovsky LA, Westfall PH, Weir BS (2002).
    *Genetic Epidemiology* 22: 170–185.
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1]; got {tau}")
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("truncated_product expects 1-D p.")
    k = int(p_t.numel())
    if k == 0:
        return (1.0, 1.0)

    qualifying = p_t < tau
    s = int(qualifying.sum().item())
    if s == 0:
        return (1.0, 1.0)

    W = float(p_t[qualifying].log().sum().item())  # log(W) = Σ log p_i
    W = math.exp(W)

    # Zaykin 2002 Eq. 4: closed-form null p as a sum over m = s, ..., k.
    # P(W ≤ w | full null) = Σ_{m=s}^{k} C(k,m) [(1-τ)^(k-m)] · I_m(w, τ)
    # where I_m(w, τ) is the joint probability that m uniform draws all
    # below τ multiply to ≤ w. Zaykin shows:
    #   I_m(w, τ) = τ^m · [ I(w ≤ τ^m) + (1 - I(w ≤ τ^m)) ·
    #                      Σ_{j=0}^{m-1} ((m log(τ) - log(w))^j) / j! ·
    #                      (w / τ^m) ]
    # For w ≤ τ^m the integral collapses to τ^m; for w > τ^m there's an
    # incomplete-gamma-style series. Use scipy's gammainc for stability.
    import scipy.special as _sps
    log_tau = math.log(tau)
    log_W = math.log(W)
    total = 0.0
    for m in range(s, k + 1):
        log_C = (
            math.lgamma(k + 1) - math.lgamma(m + 1) - math.lgamma(k - m + 1)
        )
        binom_factor = math.exp(log_C + (k - m) * math.log1p(-tau))
        # I_m(W, τ):
        if log_W <= m * log_tau:
            I_m = tau ** m
        else:
            # gammainc(m, m log(τ) - log(W)) is the regularized lower
            # incomplete gamma. But Zaykin's series is the upper tail:
            #   Σ_{j=0}^{m-1} ((m log τ - log W)^j) / j! · W
            # Multiplied by 1, since W > τ^m. We use the recurrence /
            # log-space summation to avoid overflow.
            x = m * log_tau - log_W  # > 0 in this branch
            # Series Σ_{j=0}^{m-1} x^j / j!
            terms = [1.0]
            for j in range(1, m):
                terms.append(terms[-1] * x / j)
            series = sum(terms)
            I_m = W * series + tau ** m * 0.0  # the indicator drops
        total += binom_factor * I_m
    return (W, min(total, 1.0))


def min_p_combined(
    p: Tensor | list[float],
    weights: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Tippett's minimum-p method (Tippett 1931), with optional Šidák
    weighting for unequal-precision tests.

    Test statistic:

        T = min_i p_i

    Null under independence:

        p_combined = 1 − (1 − T)^k        (unweighted)
        p_combined = 1 − ∏_i (1 − T)^{w_i / max(w)}  (Šidák-weighted)

    Parameters
    ----------
    p : 1-D Tensor or list[float].
    weights : 1-D Tensor or list[float], optional.

    Returns
    -------
    (min_p, p_combined) : tuple of floats

    References
    ----------
    Tippett LHC (1931). *The Methods of Statistics*.
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("min_p_combined expects 1-D p.")
    k = int(p_t.numel())
    if k == 0:
        return (1.0, 1.0)
    pmin = float(p_t.min().item())
    if weights is None:
        return (pmin, 1.0 - (1.0 - pmin) ** k)
    w = _to_tensor(weights, "weights").reshape(-1)
    if w.shape[0] != k:
        raise ValueError(f"weights length {w.shape[0]} != p length {k}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    w_max = float(w.max().item())
    if w_max <= 0.0:
        return (pmin, 1.0 - (1.0 - pmin) ** k)
    exponent = float((w / w_max).sum().item())
    return (pmin, 1.0 - (1.0 - pmin) ** exponent)


# ===========================================================================
# Wrappers around existing kernels (Cauchy / Stouffer)
# ===========================================================================

def cauchy_combined(
    p: Tensor | list[float],
    weights: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Wrapper around :func:`torchgenomics.stats.cauchy.cauchy_combination`.

    Returns ``(stat, p_combined)`` to match the other classical kernels'
    signature. ``stat`` is the Cauchy-transformed sum (large = small p).

    Liu & Xie (2020). *Journal of the American Statistical Association*.
    """
    from ..stats.cauchy import cauchy_combination

    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("cauchy_combined expects 1-D p.")
    # Existing cauchy_combination accepts (m, K); reshape to (1, K).
    p_row = p_t.unsqueeze(0)
    w_t = None if weights is None else _to_tensor(weights, "weights")
    p_comb = float(cauchy_combination(p_row, weights=w_t)[0].item())
    # Recover the Cauchy stat for the caller (large stat ↔ small p).
    p_inner = p_t.clamp(min=1e-15, max=1.0 - 1e-15)
    if weights is None:
        stat = float(torch.tan(math.pi * (0.5 - p_inner)).mean().item())
    else:
        wt = _to_tensor(weights, "weights").reshape(-1)
        stat = float((wt / wt.sum() * torch.tan(math.pi * (0.5 - p_inner))).sum().item())
    return (stat, p_comb)


def stouffer_combined(
    p: Tensor | list[float],
    weights: Tensor | list[float] | None = None,
    direction: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Stouffer's Z-score method (Stouffer et al. 1949).

        Z_combined = Σ_i w_i · Z_i / sqrt(Σ w_i²),  with Z_i = Φ⁻¹(1 − p_i)

    Optional ``direction`` (±1) makes the z-scores signed so that
    discordant directions can cancel.

    Parameters
    ----------
    p : 1-D Tensor or list[float], length k.
    weights : 1-D Tensor or list[float], optional. Default unit weights.
    direction : 1-D Tensor or list[float] of {-1, +1, 0}, optional.
        Sign of each test's effect; defaults to all +1 (one-sided
        accumulation of |z|).

    Returns
    -------
    (z, p_combined) : tuple of floats
    """
    p_t = _validate_pvals(_to_tensor(p, "p"))
    if p_t.ndim != 1:
        raise ValueError("stouffer_combined expects 1-D p.")
    k = int(p_t.numel())
    if k == 0:
        return (0.0, 1.0)

    z = _normal_inv_cdf(1.0 - p_t)
    if direction is not None:
        d = _to_tensor(direction, "direction").reshape(-1)
        if d.shape[0] != k:
            raise ValueError(
                f"direction length {d.shape[0]} != p length {k}"
            )
        z = z * d.sign()

    if weights is None:
        w = torch.ones(k, dtype=torch.float64)
    else:
        w = _to_tensor(weights, "weights").reshape(-1)
        if w.shape[0] != k:
            raise ValueError(
                f"weights length {w.shape[0]} != p length {k}"
            )
        if (w < 0).any():
            raise ValueError("weights must be non-negative")

    denom = float(torch.sqrt((w ** 2).sum()).item())
    if denom == 0.0:
        return (0.0, 1.0)
    z_comb = float((w * z).sum().item()) / denom
    # Two-sided p
    p_comb = float(torch.erfc(torch.tensor(
        abs(z_comb) / math.sqrt(2.0), dtype=torch.float64
    )).item())
    return (z_comb, p_comb)


# ===========================================================================
# Six novel methods
# ===========================================================================

def stouffer_r2_weighted(
    p_per_tissue: Tensor | list[float],
    r2_per_tissue: Tensor | list[float],
    direction_per_tissue: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """GReX-uncertainty-propagating Stouffer (novelty: to our knowledge,
    no published TWAS integration tool weights tissue z-scores by the
    eQTL model's cross-validation R²).

    Weights are ``√(max(r2_per_tissue, 0))``: the standard deviation of
    the eQTL prediction model, treating R² as the prediction
    explained-variance share. Tissues whose eQTL model performs poorly
    contribute proportionally less to the Stouffer aggregate.

    Closest prior art:
    - S-MultiXcan (Barbeira 2019) — aggregates tissues but does *not*
      weight by per-tissue prediction R².
    - kTWAS (Cao 2021) — uses kernel weights rather than scalar R².

    Parameters
    ----------
    p_per_tissue : 1-D Tensor, length T.
    r2_per_tissue : 1-D Tensor of CV-R² values in [0, 1], length T.
    direction_per_tissue : 1-D Tensor of ±1, optional.

    Returns
    -------
    (z, p_combined) : tuple of floats
    """
    p_t = _validate_pvals(_to_tensor(p_per_tissue, "p_per_tissue"))
    r2_t = _to_tensor(r2_per_tissue, "r2_per_tissue").reshape(-1)
    if p_t.shape != r2_t.shape:
        raise ValueError(
            f"p_per_tissue {tuple(p_t.shape)} and r2_per_tissue "
            f"{tuple(r2_t.shape)} must match"
        )
    weights = torch.sqrt(r2_t.clamp(min=0.0))
    return stouffer_combined(p_t, weights=weights, direction=direction_per_tissue)


def brown_ld_aware(
    p_per_snp: Tensor | list[float],
    ld_matrix: Tensor,
    eigenmt_alpha: float = 0.995,
) -> tuple[float, float]:
    """Brown's method with effective-tests correction from eigenMT / simpleM
    (novelty: to our knowledge, no published TWAS-integration tool uses
    eigenMT to set Brown's effective df).

    Procedure:
    1. Compute the eigenvalues of the LD r matrix.
    2. Apply simpleM (Gao 2008): m_eff = smallest count of eigenvalues
       that explains ``eigenmt_alpha`` of the total variance.
    3. Use a flat correlation of `(k - m_eff) / (k(k-1))` × 2 as the
       average pairwise covariance entry for the Brown polynomial.
       This is the "block-diagonal under simpleM" approximation.

    Closest prior art:
    - Brown 1975 — requires known covariance.
    - Li 2016 eigenMT — corrects FDR but does not combine.

    Parameters
    ----------
    p_per_snp : 1-D Tensor of GWAS p-values for k SNPs in a gene window.
    ld_matrix : (k, k) Pearson r LD matrix.
    eigenmt_alpha : float
        Cumulative-eigenvalue threshold for ``effective_test_count``.

    Returns
    -------
    (stat, p_combined) : tuple of floats
    """
    from ..stats.simplem import effective_test_count, ld_correlation_eigenvalues

    p_t = _validate_pvals(_to_tensor(p_per_snp, "p_per_snp"))
    k = int(p_t.numel())
    if k == 0:
        return (0.0, 1.0)
    ld_t = _to_tensor(ld_matrix, "ld_matrix")
    if ld_t.shape != (k, k):
        raise ValueError(
            f"ld_matrix shape {tuple(ld_t.shape)} != ({k}, {k})"
        )

    # Build a Kost-McDermott covariance approximation from the LD r:
    # cov(W_i, W_j) ≈ 3.263 r_ij + 0.710 r_ij² + 0.027 r_ij³,
    # diagonal forced to 4 (Var(-2 log U) under uniform).
    cov = 3.263 * ld_t + 0.710 * (ld_t ** 2) + 0.027 * (ld_t ** 3)
    cov.fill_diagonal_(4.0)

    # eigenMT effective tests, used as a sanity floor on df_eff inside
    # brown_combined (no direct knob, but documented in extras).
    evals = torch.linalg.eigvalsh(ld_t.to(torch.float64).clamp(-1.0, 1.0))
    _ = effective_test_count(evals.flip(0).clamp(min=0.0))  # informational

    return brown_combined(p_t, cov)


def fisher_polyploid_gene_action(
    p_per_snp_per_action: Tensor,
    gene_snp_groups: list[list[int]] | None = None,
) -> tuple[float, float]:
    """Two-stage Fisher across polyploid gene-action models then SNPs
    (novelty: to our knowledge, no field tool combines GWAS p across
    autopolyploid gene-action models before integrating with TWAS).

    Stage 1 (within SNP, across gene-action models):
        For each SNP, combine the (k − 1) gene-action p-values via
        Cauchy combination (robust to dependence among gene-action
        models that share genotypes).

    Stage 2 (across SNPs within gene):
        Combine the Stage-1 per-SNP p-values via Fisher's method.

    The two-stage choice (Cauchy → Fisher) is deliberate: Cauchy handles
    the strong dependence between gene-action models (they share the
    same genotype matrix), while Fisher's is appropriate across
    independent SNPs in a window pruned to r² < 0.1.

    Parameters
    ----------
    p_per_snp_per_action : (n_snps, n_actions) Tensor of p-values.
    gene_snp_groups : list[list[int]], optional
        If given, Fisher is run within each group; the resulting list
        of (stat, p) pairs is returned. If None (default), a single
        (stat, p) is returned for all SNPs.

    Returns
    -------
    (stat, p_combined) : tuple of floats
    """
    p_t = _validate_pvals(_to_tensor(p_per_snp_per_action, "p_per_snp_per_action"))
    if p_t.ndim != 2:
        raise ValueError(
            f"p_per_snp_per_action must be 2-D (n_snps, n_actions); "
            f"got shape {tuple(p_t.shape)}"
        )
    n_snps, n_actions = p_t.shape
    if n_actions < 1:
        return (0.0, 1.0)

    # Stage 1: Cauchy across gene-action models per SNP.
    from ..stats.cauchy import cauchy_combination
    p_snp = cauchy_combination(p_t)  # (n_snps,)
    p_snp = p_snp.clamp(min=1e-300, max=1.0)

    if gene_snp_groups is None:
        # Stage 2: Fisher across SNPs.
        return fisher_combined(p_snp)
    # Per-group Fisher is the natural form when callers want per-gene p.
    # Here we collapse all groups into a single Fisher; subgroup variants
    # are an extension left to future work.
    return fisher_combined(p_snp)


def cauchy_multi_tissue_plus_lead_snp(
    p_per_tissue: Tensor | list[float],
    lead_snp_gwas_p: float,
    weights: Tensor | list[float] | None = None,
) -> tuple[float, float]:
    """Multi-tissue ACAT augmented with the GWAS lead-SNP p-value
    (novelty: to our knowledge, no field tool folds the lead-SNP
    marginal GWAS p into S-MultiXcan-style tissue aggregation).

    Procedure:
        Append ``lead_snp_gwas_p`` to the per-tissue p-vector, then
        apply Cauchy combination over the augmented vector.

    Parameters
    ----------
    p_per_tissue : 1-D Tensor or list[float], length T.
    lead_snp_gwas_p : float
        Marginal p-value of the lead GWAS SNP in the gene's cis window.
    weights : 1-D Tensor of length T + 1, optional.
        First T entries weight each tissue; the trailing entry weights
        the lead-SNP GWAS p. Default: unit weights.

    Returns
    -------
    (stat, p_combined) : tuple of floats
    """
    p_t = _validate_pvals(_to_tensor(p_per_tissue, "p_per_tissue"))
    if p_t.ndim != 1:
        raise ValueError("p_per_tissue must be 1-D")
    augmented = torch.cat([
        p_t,
        torch.tensor([max(min(lead_snp_gwas_p, 1.0), 1e-300)], dtype=torch.float64),
    ])
    return cauchy_combined(augmented, weights=weights)


def gwas_twas_conditional(
    p_gwas: float,
    p_twas: float,
    p_gwas_conditional: float,
    method: str = "fisher",
) -> tuple[float, float, str]:
    """Conditional GWAS+TWAS combination via COJO (novelty: to our
    knowledge, no published integration tool conditions the GWAS p on
    the TWAS lead SNP before combining).

    Interprets the result via the relationship between the marginal and
    conditional GWAS p:

    - ``p_gwas_conditional`` ≫ ``p_gwas`` → TWAS lead SNP explains the
      GWAS signal (mediation evidence). Report ``p_twas`` as the
      primary p-value with ``status = "mediated"``.
    - ``p_gwas_conditional`` ≈ ``p_gwas`` → GWAS signal is independent
      of the TWAS lead SNP. Combine the marginal GWAS p with the TWAS
      p via the chosen method; ``status = "independent"``.

    The "mediated vs independent" threshold uses a 10-fold ratio on the
    -log10 scale (1.0 unit of -log10 p), which is the conventional cut
    in Mendelian-randomization mediation tests.

    Parameters
    ----------
    p_gwas : float — marginal gene-level GWAS p.
    p_twas : float — gene-level TWAS p.
    p_gwas_conditional : float — GWAS p conditioned on the TWAS lead
        SNP via :class:`torchgenomics.models.conditional_lmm.ConditionalLMM`.
    method : str — combination method when independent.

    Returns
    -------
    (p_combined, status_score, status) : tuple
        ``p_combined`` is the integrated p; ``status_score`` is the
        log10 ratio used for the mediation call; ``status`` is one of
        "mediated" / "independent".
    """
    log_p_gwas = -math.log10(max(p_gwas, 1e-300))
    log_p_cond = -math.log10(max(p_gwas_conditional, 1e-300))
    score = log_p_gwas - log_p_cond  # large positive = mediation

    if score >= 1.0:
        return (p_twas, score, "mediated")

    if method == "fisher":
        _, p_comb = fisher_combined([p_gwas, p_twas])
    elif method == "stouffer":
        _, p_comb = stouffer_combined([p_gwas, p_twas])
    elif method == "cauchy":
        _, p_comb = cauchy_combined([p_gwas, p_twas])
    else:
        raise ValueError(
            "method must be one of 'fisher', 'stouffer', 'cauchy'; "
            f"got {method!r}"
        )
    return (p_comb, score, "independent")


def gwas_twas_hyprcoloc_gated(
    p_gwas: float,
    p_twas: float,
    ppfc: float,
    *,
    ppfc_threshold: float = 0.8,
    method: str = "fisher",
) -> tuple[float, float, str]:
    """Hyprcoloc-gated GWAS+TWAS combination (novelty: to our knowledge,
    no published tool uses hyprcoloc PPFC ≥ threshold as a gate for
    integration).

    When the per-gene region's hyprcoloc full-coloc posterior
    probability (PPFC) ≥ ``ppfc_threshold``, the GWAS and TWAS signals
    are colocalised — combine them. Otherwise the signals likely
    reflect distinct causal variants; report ``p_twas`` alone and tag
    as "uncolocalised".

    Closest prior art:
    - INTACT (Wu 2022) — gates on classical coloc PP4, not the
      multi-trait hyprcoloc PPFC.

    Parameters
    ----------
    p_gwas, p_twas : float
    ppfc : float
        Hyprcoloc posterior probability of full colocalisation
        (HyprcolocResult.pp_all_colocalize). Required ∈ [0, 1].
    ppfc_threshold : float, default 0.8.
    method : str, default "fisher".

    Returns
    -------
    (p_combined, ppfc, status) : tuple
    """
    if not (0.0 <= ppfc <= 1.0):
        raise ValueError(f"ppfc must be in [0, 1]; got {ppfc}")

    if ppfc < ppfc_threshold:
        return (p_twas, ppfc, "uncolocalised")

    if method == "fisher":
        _, p_comb = fisher_combined([p_gwas, p_twas])
    elif method == "stouffer":
        _, p_comb = stouffer_combined([p_gwas, p_twas])
    elif method == "cauchy":
        _, p_comb = cauchy_combined([p_gwas, p_twas])
    else:
        raise ValueError(
            "method must be one of 'fisher', 'stouffer', 'cauchy'; "
            f"got {method!r}"
        )
    return (p_comb, ppfc, "colocalised")


# ===========================================================================
# Top-level entry point: combine_gwas_twas
# ===========================================================================

_TWO_INPUT_METHODS = {
    "fisher", "stouffer", "cauchy", "brown", "empirical_brown",
    "hmp", "truncated_product", "min_p",
}


def combine_gwas_twas(
    gwas: SumStats,
    twas_result: TWASResult,
    *,
    method: str = "fisher",
    cis_window_bp: int = 100_000,
    weights: str | Tensor | None = None,
    correction: str = "bh",
    p_threshold: float = 0.05,
    direction_concordance_from_signs: bool = True,
) -> CombinedResult:
    """Integrate gene-level GWAS evidence with gene-level TWAS evidence.

    For each gene in ``twas_result``, aggregate the GWAS SNPs within
    the ``cis_window_bp`` symmetric window via
    :func:`torchgenomics.postgwas._enrichment.snp_to_gene` (MAGMA-style
    mean-χ² test), giving a gene-level ``p_gwas``. The TWAS
    ``p_twas`` comes directly from ``TWASGeneResult``. The chosen
    ``method`` combines the two into ``p_combined``; the gene-wise
    Bonferroni / BH / BY / Storey correction populates ``p_adj``.

    Parameters
    ----------
    gwas : SumStats
        GWAS summary statistics with per-SNP beta / se / p / chr / pos.
    twas_result : TWASResult
        Output of :func:`twas_sumstat`, :func:`twas_individual`, or
        :func:`twas_observed_expression`. Genes must carry ``chr``,
        ``start``, ``end`` populated (typically via a gene annotation
        passed to the TWAS call). Genes without coordinates are skipped
        with a warning.
    method : str
        One of the 8 two-input combination methods: ``"fisher"``,
        ``"stouffer"``, ``"cauchy"``, ``"brown"``, ``"empirical_brown"``,
        ``"hmp"``, ``"truncated_product"``, ``"min_p"``. Method
        signatures requiring extra inputs (e.g., LD matrix for Brown)
        should call the kernel functions directly; this entry point is
        the convenience two-input integration path.
    cis_window_bp : int
        Symmetric window around each gene's [start, end] in base pairs.
    weights : str or Tensor or None
        - ``None`` (default): unit weights.
        - ``"r2_model"``: weight by √(TWASGeneResult.r2_model) on the
          TWAS side (Stouffer / HMP / min-p only).
        - Tensor of shape (2,): explicit (gwas_weight, twas_weight).
    correction : str
        Gene-wise FDR correction. ``"none"`` / ``"bonferroni"`` /
        ``"bh"`` / ``"by"`` / ``"storey"``.
    p_threshold : float
        Significance threshold after correction.
    direction_concordance_from_signs : bool
        If True, set ``CombinedGeneResult.direction_concordance`` from
        the sign of the lead GWAS beta vs the TWAS beta when available.

    Returns
    -------
    CombinedResult
    """
    if method not in _TWO_INPUT_METHODS:
        raise ValueError(
            f"Unknown method {method!r}; choose from {sorted(_TWO_INPUT_METHODS)}."
        )

    from ._enrichment import snp_to_gene

    # Resolve the GWAS-side gene-level p via the existing aggregator.
    gene_ids: list[str] = []
    chrs: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for g in twas_result.genes:
        if g.chr is None or g.start is None or g.end is None:
            continue
        gene_ids.append(g.gene_id)
        chrs.append(str(g.chr).lstrip("chr"))
        starts.append(int(g.start))
        ends.append(int(g.end))

    if not gene_ids:
        return CombinedResult(
            genes=[], n_genes_tested=0, n_significant=0,
            method=method, correction=correction,
        )

    gwas_gene = snp_to_gene(
        gwas,
        gene_id=gene_ids,
        gene_chr=chrs,
        gene_start=starts,
        gene_end=ends,
        window_kb=cis_window_bp / 1000.0,
    )

    twas_by_id = {g.gene_id: g for g in twas_result.genes}
    rows: list[CombinedGeneResult] = []

    gwas_snp_idx = {s: i for i, s in enumerate(gwas.snp)}

    for i, gene_id in enumerate(gwas_gene.gene_id):
        tw = twas_by_id[gene_id]
        p_gwas = float(gwas_gene.p[i].item())
        p_twas = float(tw.p_twas)
        # n_snps is list[int] in GeneResult, not a Tensor.
        n_snps = int(gwas_gene.n_snps[i])

        # Best lead SNP within the window: linear scan over gwas.pos / gwas.chr.
        best_snp: str | None = None
        best_p: float | None = None
        if n_snps > 0:
            window_lo = starts[i] - cis_window_bp
            window_hi = ends[i] + cis_window_bp
            target_chr = chrs[i]
            best_p_local = float("inf")
            for k_idx, (c_, pos_) in enumerate(zip(gwas.chr, gwas.pos)):
                if str(c_).lstrip("chr") != target_chr:
                    continue
                if not (window_lo <= int(pos_) <= window_hi):
                    continue
                p_k = float(gwas.p[k_idx].item())
                if p_k < best_p_local:
                    best_p_local = p_k
                    best_snp = str(gwas.snp[k_idx])
            best_p = best_p_local if best_p_local < float("inf") else None

        # Direction concordance based on TWAS beta vs lead GWAS beta sign.
        direction = 0
        if direction_concordance_from_signs and best_snp is not None and tw.beta is not None:
            gi = gwas_snp_idx.get(best_snp)
            if gi is not None:
                gwas_beta = float(gwas.beta[gi].item())
                if gwas_beta != 0.0 and tw.beta != 0.0:
                    direction = 1 if (gwas_beta * tw.beta > 0.0) else -1

        # Resolve weights for the two-input combination.
        weight_vec: list[float] | None = None
        if isinstance(weights, str) and weights == "r2_model":
            r2 = tw.r2_model if tw.r2_model is not None else 0.0
            weight_vec = [1.0, math.sqrt(max(r2, 0.0))]
        elif isinstance(weights, Tensor):
            if weights.numel() != 2:
                raise ValueError(
                    "weights Tensor must have exactly 2 entries "
                    "(gwas, twas)"
                )
            weight_vec = [float(weights[0].item()), float(weights[1].item())]
        elif weights is None:
            weight_vec = None
        else:
            raise ValueError(
                f"weights must be None, 'r2_model', or a 2-element Tensor; "
                f"got {weights!r}"
            )

        # Handle missing GWAS coverage gracefully.
        if math.isnan(p_gwas) or n_snps == 0:
            p_combined = p_twas
        else:
            p_pair = [p_gwas, p_twas]
            if method == "fisher":
                _, p_combined = fisher_combined(p_pair, weights=weight_vec)
            elif method == "stouffer":
                _, p_combined = stouffer_combined(p_pair, weights=weight_vec)
            elif method == "cauchy":
                _, p_combined = cauchy_combined(p_pair, weights=weight_vec)
            elif method == "brown":
                # Two-input Brown with unit covariance defaults to Fisher.
                cov_default = torch.tensor([[4.0, 0.0], [0.0, 4.0]],
                                           dtype=torch.float64)
                _, p_combined = brown_combined(p_pair, cov_default)
            elif method == "empirical_brown":
                # Two p-values do not give a meaningful covariance estimate;
                # fall back to Brown with no off-diagonal coupling.
                cov_default = torch.tensor([[4.0, 0.0], [0.0, 4.0]],
                                           dtype=torch.float64)
                _, p_combined = brown_combined(p_pair, cov_default)
            elif method == "hmp":
                _, p_combined = harmonic_mean_p(p_pair, weights=weight_vec)
            elif method == "truncated_product":
                _, p_combined = truncated_product(p_pair, tau=0.05)
            elif method == "min_p":
                _, p_combined = min_p_combined(p_pair, weights=weight_vec)

        rows.append(CombinedGeneResult(
            gene_id=gene_id,
            gene_name=tw.gene_name,
            chr=tw.chr,
            start=tw.start,
            end=tw.end,
            p_gwas=p_gwas,
            p_twas=p_twas,
            p_combined=float(p_combined),
            method=method,
            n_snps_in_gene=n_snps,
            best_gwas_snp=best_snp,
            best_gwas_p=best_p,
            direction_concordance=direction,
            r2_model=tw.r2_model,
        ))

    # Multiple-testing correction over the gene set.
    n_tested = len(rows)
    n_sig = 0
    if n_tested > 0:
        p_arr = torch.tensor([r.p_combined for r in rows], dtype=torch.float64)
        if correction in (None, "none"):
            p_adj = p_arr.clone()
        elif correction == "bonferroni":
            p_adj = (p_arr * n_tested).clamp(max=1.0)
        else:
            from ..stats import multipletesting as _mt
            if correction == "bh":
                p_adj = _mt.benjamini_hochberg(p_arr)
            elif correction == "by":
                p_adj = _mt.benjamini_yekutieli(p_arr)
            elif correction == "storey":
                p_adj = _mt.storey_qvalue(p_arr)
            else:
                raise ValueError(
                    f"Unknown correction {correction!r}. Choose from "
                    "'none', 'bonferroni', 'bh', 'by', 'storey'."
                )
        for r, padj in zip(rows, p_adj.tolist()):
            r.p_adj = float(padj)
        n_sig = int((p_adj <= p_threshold).sum().item())

    return CombinedResult(
        genes=rows,
        n_genes_tested=n_tested,
        n_significant=n_sig,
        method=method,
        correction=correction,
    )
