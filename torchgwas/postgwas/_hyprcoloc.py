"""Hyprcoloc: multi-trait colocalization (Foley et al. 2021).

Generalizes Giambartolomei et al. (2014) two-trait ``coloc`` to K >= 2 traits
and returns the posterior probability that all traits in a query subset
share a single causal variant in the region.

Core formula (Wakefield 2009 approximate Bayes factor per SNP per trait):

    log_ABF_jk = 0.5 * log(1 - r_jk) + 0.5 * r_jk * z_jk^2
    r_jk       = W / (W + se_jk^2)

where ``W`` is the prior variance on the standardized effect size (default
``W = 0.15^2`` per the Giambartolomei paper). The regional Bayes factor for
the hypothesis "traits in subset ``S`` all colocalize at a single causal SNP
in the region" is

    BF_S = sum_j prod_{k in S} ABF_jk

and the null / independent hypothesis uses the per-trait marginals
``BF_null_k = sum_j ABF_jk``.

This module exposes:

* :func:`hyprcoloc` — full analysis: computes per-trait BFs, enumerates all
  non-singleton subsets of traits (2^K - K - 1 combinations), and returns the
  posterior probabilities and the most likely colocalizing cluster.
* :func:`coloc_pairwise` — classical two-trait coloc (Giambartolomei 2014)
  posterior probability (H0..H4) as a convenience for users who want the
  original decomposition.

Input sumstats are passed as :class:`~torchgwas.postgwas.SumStats` objects;
the caller is responsible for restricting them to the region of interest and
for ensuring allele alignment.

**Polyploid compatibility.** The Wakefield ABF uses ``z = beta / se`` and
``r = W / (W + se^2)``. Both are ploidy-invariant — when per-allele
effects and SEs scale together (as they do for higher ploidy), the
z-score and ratio ``r`` remain unchanged. Colocalization posteriors are
therefore valid for diploid, tetraploid, hexaploid, and arbitrary ploidy.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import HAS_NATIVE_HYPRCOLOC, _hyprcoloc_native
from ._sumstats import SumStats


@dataclass
class HyprcolocResult:
    """Result of a :func:`hyprcoloc` run."""

    # Marginal per-subset posterior, keyed by the sorted tuple of trait
    # indices included in the subset. Always sums to <= 1 (the complement
    # is the null hypothesis "no shared causal variant").
    subset_posteriors: dict[tuple[int, ...], float]
    # Posterior that *all* K traits share a single causal variant.
    pp_all_colocalize: float
    # Posterior of the null hypothesis (no subset of size >= 2 colocalizes).
    pp_null: float
    # Most-likely colocalizing cluster and its posterior.
    best_cluster: tuple[int, ...]
    best_cluster_posterior: float
    # Candidate SNP (max posterior contributor within best_cluster), given
    # by its index in the common sumstats table, and its per-SNP posterior
    # (PP_ixc) within the best cluster.
    candidate_snp: int
    candidate_snp_posterior: float
    # Per-trait marginal log-evidence (natural log).
    log_bf_marginal: list[float]
    # List of trait names if supplied.
    trait_names: list[str] | None = field(default=None)


@dataclass
class ColocPairwiseResult:
    """Result of a classical two-trait :func:`coloc_pairwise` run."""

    pp_h0: float  # no causal variant in either trait
    pp_h1: float  # causal variant in trait 1 only
    pp_h2: float  # causal variant in trait 2 only
    pp_h3: float  # distinct causal variants
    pp_h4: float  # shared causal variant
    candidate_snp: int  # index of the best candidate SNP under H4


# ---------------------------------------------------------------------------
# Wakefield approximate Bayes factor
# ---------------------------------------------------------------------------


def _wakefield_log_abf(
    beta: Tensor, se: Tensor, prior_w: float
) -> Tensor:
    """Per-SNP Wakefield log-ABF for a single trait.

    ``prior_w`` is the prior variance on beta (not log-beta). Returns a
    (m,) float64 tensor in natural log units.
    """
    beta = beta.to(torch.float64)
    se = se.to(torch.float64)
    v = se**2
    r = prior_w / torch.clamp(prior_w + v, min=1e-30)
    z = beta / torch.clamp(se, min=1e-12)
    # log_ABF = 0.5 * log(1 - r) + 0.5 * r * z^2
    return 0.5 * torch.log(torch.clamp(1.0 - r, min=1e-300)) + 0.5 * r * z**2


def _log_sum_exp(x: Tensor) -> float:
    """Stable log-sum-exp reducing a 1-D tensor to a Python float."""
    m = float(x.max().item())
    if not math.isfinite(m):
        return m
    return m + float(torch.log(torch.exp(x - m).sum()).item())


def _log_diff_exp(a: float, b: float) -> float:
    """Stable log(exp(a) - exp(b)) for a >= b.

    Used by ``coloc_pairwise`` to compute log of the H3 sum
    `(sum_j BF1_j)(sum_j' BF2_j') - sum_j BF1_j*BF2_j` — i.e. the
    outer product over distinct (j, j') pairs only, with the
    diagonal that already accrues to H4 subtracted. See F3 #2 in
    ``docs/validation_findings.md`` (2026-05-15) for the bug history.
    """
    if not math.isfinite(a):
        return a
    if a < b:
        # Numerical safety: a >= b should hold by construction. If
        # rounding produces a < b by a tiny amount, treat the diff
        # as zero (log(0) = -inf).
        return float("-inf")
    if a == b:
        return float("-inf")
    return a + math.log1p(-math.exp(b - a))


# ---------------------------------------------------------------------------
# Multi-trait hyprcoloc
# ---------------------------------------------------------------------------


def hyprcoloc(
    sumstats_list: list[SumStats],
    prior_1: float = 1e-4,
    prior_2: float = 0.98,
    prior_w: float = 0.15**2,
    trait_names: list[str] | None = None,
) -> HyprcolocResult:
    """Multi-trait colocalization posterior (Foley et al. 2021).

    Enumerates every non-singleton subset of the ``K`` traits and scores the
    hypothesis "all traits in the subset share a single causal variant in
    the region". Feasible up to ``K <= ~12``.

    Parameters
    ----------
    sumstats_list : list[SumStats]
        One SumStats per trait, **already restricted to the region of
        interest** and **aligned on a common SNP set** (same length, same
        order). Only ``beta`` and ``se`` (1-D) are read.
    prior_1 : float
        Prior probability that a single trait is associated in the region.
        Default 1e-4 matches Giambartolomei (2014).
    prior_2 : float
        Prior probability that two associated traits share the same causal
        variant (given both are associated). Default 0.98 matches Foley
        (2021) when strong prior belief in sharing is appropriate.
    prior_w : float
        Prior variance for the Wakefield ABF, on the effect-size scale.
        Default 0.15**2 matches the coloc paper.
    trait_names : list[str] or None
        Optional human-readable labels, attached to the result.

    Returns
    -------
    HyprcolocResult
    """
    K = len(sumstats_list)
    if K < 2:
        raise ValueError("hyprcoloc requires at least two traits")
    m = sumstats_list[0].beta.shape[0]
    if m < 1:
        raise ValueError("sumstats must contain at least one SNP")
    for k, ss in enumerate(sumstats_list):
        if ss.beta.shape[0] != m:
            raise ValueError(
                f"all sumstats must share the same SNP count; trait {k} has "
                f"{ss.beta.shape[0]} vs expected {m}"
            )
        if ss.beta.ndim != 1 or ss.se.ndim != 1:
            raise ValueError("hyprcoloc only supports 1-D beta/se per trait")
    if not 0.0 < prior_1 < 1.0:
        raise ValueError("prior_1 must be in (0, 1)")
    if not 0.0 < prior_2 < 1.0:
        raise ValueError("prior_2 must be in (0, 1)")

    # Per-SNP per-trait log-ABF, stacked into a (K, m) float64 tensor on CPU.
    log_abf = torch.stack(
        [
            _wakefield_log_abf(ss.beta.cpu(), ss.se.cpu(), prior_w)
            for ss in sumstats_list
        ],
        dim=0,
    )

    # Regional marginal per trait: log( sum_j ABF_jk ).
    log_bf_marginal = [float(_log_sum_exp(log_abf[k])) for k in range(K)]

    # Subset evidence: for subset S the regional BF (summing over the single
    # putative causal SNP j) is log( sum_j exp( sum_{k in S} log_abf_jk ) ).
    # We enumerate every non-singleton subset up to the full set; at K = 10
    # that is 1013 subsets, easily handled.
    #
    # Native C++ shortcut: enumerate 2^K bitmasks with per-subset trait-row
    # sum + per-subset logsumexp inline, OpenMP across subsets. Output is a
    # (2^K,) array indexed by bitmask; we read out the non-singleton entries
    # into the same dict the Python body produces, preserving the downstream
    # contract bit-for-bit. The Python body below is the algorithmic spec
    # and runs unchanged on fallthrough.
    subset_log_bf: dict[tuple[int, ...], float] = {}
    if (
        HAS_NATIVE_HYPRCOLOC
        and not native_disabled()
        and log_abf.device.type == "cpu"
        and K >= 6
        and K <= 30
    ):
        log_abf_np = log_abf.contiguous().numpy()
        out_np = np.empty(1 << K, dtype=np.float64)
        _hyprcoloc_native.enumerate_subset_log_bf(log_abf_np, out_np)
        for size in range(2, K + 1):
            for subset in itertools.combinations(range(K), size):
                bitmask = 0
                for k in subset:
                    bitmask |= 1 << k
                subset_log_bf[tuple(subset)] = float(out_np[bitmask])
    else:
        for size in range(2, K + 1):
            for subset in itertools.combinations(range(K), size):
                stacked = log_abf[list(subset)].sum(dim=0)  # (m,)
                subset_log_bf[tuple(subset)] = _log_sum_exp(stacked)

    # Prior structure -- Foley et al. (2021) Eq. 2 (corrected per F3 #1
    # patch, 2026-05-15). For a candidate cluster S of size |S| >= 2
    # in a panel of K traits:
    #
    #     Pr(H_S) = prior_1 * prior_2^(|S| - 1) * (1 - prior_2)^(K - |S|)
    #
    # where:
    #   prior_1 (~1e-4) is the prior probability that *some* colocalization
    #     architecture is true at this region (drawn once per cluster).
    #   prior_2 (~0.98, Foley's `c`) is the conditional sharing probability
    #     -- given that a cluster exists, the probability that one more
    #     trait joins it.
    #   (1 - prior_2) penalises each trait that is *not* in the cluster,
    #     i.e., the exclusion mass that the cluster must "pay" for the
    #     remaining (K - |S|) traits being un-associated with this cluster.
    #
    # Pre-patch TG used the PRODUCT prior `prior_1^|S| * prior_2^(|S|-1)
    # * (1 - prior_1)^(K - |S|)`. With prior_1 = 1e-4, that imposed a
    # ~1e-4 penalty *per additional trait* in the cluster, so the
    # |S| = 3 / |S| = 2 prior ratio was ~1e-4 instead of Foley's
    # ~prior_2 / (1 - prior_2) ~= 49. The result was that strong
    # 3-trait shared-causal architectures were assigned to a 2-trait
    # subset by TG while R hyprcoloc correctly selected the 3-trait
    # cluster. See docs/validation_findings.md "Tier 1 A4" for the
    # full diagnosis + numerical reproduction.
    log_p1 = math.log(prior_1)
    log_p2 = math.log(prior_2)
    log_1mp2 = math.log1p(-prior_2)

    def _log_prior(subset_size: int) -> float:
        return (
            log_p1
            + (subset_size - 1) * log_p2
            + (K - subset_size) * log_1mp2
        )

    # Null prior: no subset of size >= 2 colocalizes. This is the residual
    # after subtracting the non-singleton prior mass.
    non_null_log_priors: list[float] = []
    subset_log_prior: dict[tuple[int, ...], float] = {}
    for subset in subset_log_bf:
        lp = _log_prior(len(subset))
        subset_log_prior[subset] = lp
        non_null_log_priors.append(lp)

    # log(1 - sum_subset prior_subset) with log-sum-exp trick.
    max_lp = max(non_null_log_priors)
    non_null_mass = math.exp(max_lp) * float(
        torch.tensor(
            [math.exp(lp - max_lp) for lp in non_null_log_priors],
            dtype=torch.float64,
        ).sum()
    )
    null_mass = max(1.0 - non_null_mass, 1e-300)
    log_p_null = math.log(null_mass)

    # Posteriors: softmax over {null} ∪ {all non-singleton subsets}, with the
    # null hypothesis carrying BF = 1 (log 0) by construction.
    log_weights = [log_p_null]  # null hypothesis, log BF = 0
    keys: list[tuple[int, ...] | None] = [None]
    for subset, lbf in subset_log_bf.items():
        log_weights.append(subset_log_prior[subset] + lbf)
        keys.append(subset)

    log_weights_t = torch.tensor(log_weights, dtype=torch.float64)
    log_norm = _log_sum_exp(log_weights_t)
    posteriors = [math.exp(lw - log_norm) for lw in log_weights]

    pp_null = posteriors[0]
    subset_posteriors: dict[tuple[int, ...], float] = {}
    for key, pp in zip(keys[1:], posteriors[1:]):
        assert key is not None
        subset_posteriors[key] = pp

    pp_all = subset_posteriors.get(tuple(range(K)), 0.0)

    # Best cluster = subset with the largest posterior (ties broken by size,
    # preferring the *largest* colocalizing cluster, which matches the paper).
    best_cluster: tuple[int, ...] = tuple(range(K))
    best_pp = pp_all
    for subset, pp in subset_posteriors.items():
        if pp > best_pp or (pp == best_pp and len(subset) > len(best_cluster)):
            best_pp = pp
            best_cluster = subset

    # Candidate SNP within the best cluster: argmax of the per-SNP joint
    # log-ABF across the traits in the cluster. The per-SNP posterior (PP_ixc)
    # is the softmax of the joint log-ABF over SNPs in the region.
    joint = log_abf[list(best_cluster)].sum(dim=0)
    cand_idx = int(torch.argmax(joint).item())
    joint_log_norm = _log_sum_exp(joint)
    cand_pp = float(math.exp(float(joint[cand_idx].item()) - joint_log_norm))

    return HyprcolocResult(
        subset_posteriors=subset_posteriors,
        pp_all_colocalize=pp_all,
        pp_null=pp_null,
        best_cluster=best_cluster,
        best_cluster_posterior=best_pp,
        candidate_snp=cand_idx,
        candidate_snp_posterior=cand_pp,
        log_bf_marginal=log_bf_marginal,
        trait_names=list(trait_names) if trait_names else None,
    )


# ---------------------------------------------------------------------------
# Classical two-trait coloc (Giambartolomei 2014) for reference
# ---------------------------------------------------------------------------


def coloc_pairwise(
    ss1: SumStats,
    ss2: SumStats,
    prior_1: float = 1e-4,
    prior_2: float = 1e-4,
    prior_12: float = 1e-5,
    prior_w: float = 0.15**2,
) -> ColocPairwiseResult:
    """Two-trait colocalization posterior (Giambartolomei et al. 2014).

    Computes the PP(H0..H4) decomposition:

    * H0: no causal variant in either trait
    * H1: causal variant in trait 1 only
    * H2: causal variant in trait 2 only
    * H3: distinct causal variants
    * H4: shared causal variant

    Uses the Wakefield approximate Bayes factor and assumes a single causal
    variant per hypothesis.

    Parameters
    ----------
    ss1, ss2 : SumStats
        Aligned sumstats for the two traits (same SNP count and order).
    prior_1, prior_2 : float
        Per-SNP prior probability that trait 1 / trait 2 has a causal
        variant. Default 1e-4 from the paper.
    prior_12 : float
        Per-SNP prior probability that both traits share the same causal
        variant. Default 1e-5 from the paper.
    prior_w : float
        Wakefield prior variance on beta. Default 0.15**2.

    Returns
    -------
    ColocPairwiseResult
    """
    if ss1.beta.shape[0] != ss2.beta.shape[0]:
        raise ValueError("ss1 and ss2 must have the same number of SNPs")

    log_abf1 = _wakefield_log_abf(ss1.beta.cpu(), ss1.se.cpu(), prior_w)
    log_abf2 = _wakefield_log_abf(ss2.beta.cpu(), ss2.se.cpu(), prior_w)
    m = log_abf1.shape[0]

    # Regional sum-BF under each hypothesis.
    # H1: trait 1 causal, trait 2 null -> sum_j ABF1_j
    # H2: symmetric
    # H3: distinct causal variants -> (sum_j ABF1_j) * (sum_j' ABF2_j')
    # H4: shared causal -> sum_j ABF1_j * ABF2_j
    log_sum1 = _log_sum_exp(log_abf1)
    log_sum2 = _log_sum_exp(log_abf2)
    log_sum12 = _log_sum_exp(log_abf1 + log_abf2)

    # The "per-SNP BF" formulation of Giambartolomei folds priors in as:
    #   PP(H0) ∝ 1
    #   PP(H1) ∝ prior_1 * sum_j BF1_j
    #   PP(H2) ∝ prior_2 * sum_j BF2_j
    #   PP(H3) ∝ prior_1 * prior_2 * (sum_j BF1_j) * (sum_j' BF2_j')
    #   PP(H4) ∝ prior_12 * sum_j BF1_j * BF2_j
    # (per Wallace 2020 erratum). We work entirely in log space.
    log_p1 = math.log(prior_1)
    log_p2 = math.log(prior_2)
    log_p12 = math.log(prior_12)

    # F3 #2 patch (2026-05-15): two corrections vs the pre-patch formula.
    #
    #   (a) Diagonal subtraction on H3. H3 is the outer product over
    #       DISTINCT pairs (j, j') with j != j', i.e.
    #       (sum_j BF1_j)(sum_j' BF2_j') - sum_j BF1_j*BF2_j. The
    #       diagonal sum (j == j') already accrues to H4 and must be
    #       subtracted to avoid double-counting.
    #
    #   (b) Drop the `-log m` per-SNP normalisation. R coloc::coloc.abf
    #       (Wallace 2020 erratum) absorbs per-SNP normalisation into
    #       the prior parameterisation directly: `prior_1` is already
    #       the marginal-SNP prior, so summing BF1_j over j gives the
    #       per-region weight without a further 1/m factor. Pre-patch
    #       TG carried `-log m` on H1/H2/H4 and `-2*log m` on H3,
    #       which shifted ~14% mass from H3 to H1 in the distinct
    #       scenario when H4 had stopped dominating.
    #
    # Numerically verified by the Tier 1 A3 dispatch (2026-05-15): the
    # corrected closed form reproduces R coloc.abf at FP precision on
    # the planted shared / distinct / null scenarios. See
    # docs/validation_findings.md "Tier 1 A3" entry for the full
    # derivation.
    log_h0 = 0.0
    log_h1 = log_p1 + log_sum1
    log_h2 = log_p2 + log_sum2
    log_h3_outer = log_sum1 + log_sum2
    log_h3_outer_minus_diag = _log_diff_exp(log_h3_outer, log_sum12)
    log_h3 = log_p1 + log_p2 + log_h3_outer_minus_diag
    log_h4 = log_p12 + log_sum12

    log_weights = torch.tensor(
        [log_h0, log_h1, log_h2, log_h3, log_h4], dtype=torch.float64
    )
    log_norm = _log_sum_exp(log_weights)
    pp = [math.exp(float(lw.item()) - log_norm) for lw in log_weights]

    # Candidate SNP under H4: argmax of per-SNP shared log-ABF.
    cand_idx = int(torch.argmax(log_abf1 + log_abf2).item())

    return ColocPairwiseResult(
        pp_h0=pp[0],
        pp_h1=pp[1],
        pp_h2=pp[2],
        pp_h3=pp[3],
        pp_h4=pp[4],
        candidate_snp=cand_idx,
    )
