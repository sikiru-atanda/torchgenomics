"""Fine-mapping post-processing utilities.

This module provides post-hoc analysis tools for Bayesian fine-mapping
results produced by ``torchgenomics.models.BayesianVS``. It does *not*
re-implement the fine-mapping inference itself -- that lives in
``torchgenomics.models.bayesian_vs`` -- but rather provides utilities for
interpreting, annotating, and reporting the results:

* **Credible set extraction** -- given PIPs, extract minimal sets of
  variants that capture a target posterior mass (e.g. 95%).
* **PIP annotation** -- merge PIPs and credible set membership onto
  summary statistics for export.
* **Locus summary** -- per-locus report with lead SNP, credible set size,
  total PIP, and signal count.
* **Colocalization-ready export** -- reshape fine-mapping results into
  the format expected by ``coloc_pairwise`` / ``hyprcoloc``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..models.bayesian_vs import BayesianVSResult
    from ._sumstats import SumStats


# ---------------------------------------------------------------------------
# Credible set dataclass
# ---------------------------------------------------------------------------


@dataclass
class CredibleSet:
    """A single credible set from fine-mapping.

    Attributes
    ----------
    signal_index : int
        Which signal this set corresponds to (0-based).
    snp_indices : list[int]
        Variant indices in decreasing PIP order.
    snp_ids : list[str]
        SNP identifiers corresponding to *snp_indices*.
    pip : Tensor
        Per-variant PIP within this set, same order as *snp_indices*.
    cumulative_coverage : float
        Sum of PIPs across all variants in the set.
    lead_snp_index : int
        Index of the highest-PIP variant.
    lead_snp_id : str
        SNP ID of the highest-PIP variant.
    lead_pip : float
        PIP of the lead variant.
    """

    signal_index: int
    snp_indices: list[int]
    snp_ids: list[str]
    pip: Tensor
    cumulative_coverage: float
    lead_snp_index: int
    lead_snp_id: str
    lead_pip: float


# ---------------------------------------------------------------------------
# Annotated summary statistics
# ---------------------------------------------------------------------------


@dataclass
class AnnotatedSumStats:
    """Summary statistics annotated with fine-mapping results.

    Attributes
    ----------
    chr, pos, snp, a1, a2 : list
        Variant identifiers (same as the input ``SumStats``).
    beta, se, p : Tensor
        Association statistics from the input ``SumStats``.
    pip : Tensor
        Posterior inclusion probability per variant (0 for variants
        not present in the fine-mapping result).
    in_credible_set : list[int]
        Signal index for variants belonging to a credible set, or -1
        if the variant is not in any set.
    beta_posterior : Tensor
        Posterior mean effect from the fine-mapping model.
    beta_posterior_sd : Tensor
        Posterior standard deviation of the effect.
    """

    chr: list[str]
    pos: list[int]
    snp: list[str]
    a1: list[str]
    a2: list[str]
    beta: Tensor
    se: Tensor
    p: Tensor
    pip: Tensor
    in_credible_set: list[int]
    beta_posterior: Tensor
    beta_posterior_sd: Tensor

    def as_table(self) -> dict[str, list]:
        """Column-oriented dict suitable for ``pandas.DataFrame`` or Parquet.

        Returns
        -------
        dict[str, list]
            Keys are column names; values are Python lists of scalars.
        """
        return {
            "chr": list(self.chr),
            "pos": list(self.pos),
            "snp": list(self.snp),
            "a1": list(self.a1),
            "a2": list(self.a2),
            "beta": self.beta.tolist(),
            "se": self.se.tolist(),
            "p": self.p.tolist(),
            "pip": self.pip.tolist(),
            "in_credible_set": list(self.in_credible_set),
            "beta_posterior": self.beta_posterior.tolist(),
            "beta_posterior_sd": self.beta_posterior_sd.tolist(),
        }


# ---------------------------------------------------------------------------
# Locus summary
# ---------------------------------------------------------------------------


@dataclass
class LocusSummary:
    """Per-locus fine-mapping summary.

    Each element in the lists corresponds to one locus (one
    ``BayesianVSResult``).

    Attributes
    ----------
    locus_chr : list[str]
        Chromosome of each locus (taken from the first variant).
    locus_start : list[int]
        Start position (minimum ``pos``) of each locus.
    locus_end : list[int]
        End position (maximum ``pos``) of each locus.
    n_variants : list[int]
        Number of variants in each locus.
    n_signals : list[int]
        Number of credible sets (active signals) in each locus.
    credible_sets : list[list[CredibleSet]]
        Credible sets per locus.
    lead_snp : list[str]
        SNP ID of the highest-PIP variant in each locus.
    lead_pip : list[float]
        PIP of the lead variant in each locus.
    total_pip : list[float]
        Sum of PIPs across all variants in the locus.
    """

    locus_chr: list[str] = field(default_factory=list)
    locus_start: list[int] = field(default_factory=list)
    locus_end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    n_signals: list[int] = field(default_factory=list)
    credible_sets: list[list[CredibleSet]] = field(default_factory=list)
    lead_snp: list[str] = field(default_factory=list)
    lead_pip: list[float] = field(default_factory=list)
    total_pip: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# extract_credible_sets
# ---------------------------------------------------------------------------


def extract_credible_sets(
    result: BayesianVSResult,
    coverage: float = 0.95,
    min_pip: float = 0.0,
    deduplicate: bool = True,
) -> list[CredibleSet]:
    """Extract credible sets from a fine-mapping result.

    Parameters
    ----------
    result : BayesianVSResult
        Output of ``BayesianVS.scan`` (or equivalent).
    coverage : float, optional
        Target posterior mass for each credible set (default 0.95).
    min_pip : float, optional
        Only include variants with PIP >= this threshold (default 0.0).
    deduplicate : bool, optional
        If True, once a variant is assigned to a credible set it is
        excluded from subsequent sets (default True).

    Returns
    -------
    list[CredibleSet]
        One ``CredibleSet`` per active signal, ordered by signal index.

    Raises
    ------
    ValueError
        If *coverage* is not in (0, 1].
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    p = len(result.snp)
    used: set[int] = set()  # indices already assigned (for dedup)
    sets: list[CredibleSet] = []

    use_susie = (
        getattr(result, "method", "cavi") == "susie"
        and getattr(result, "alpha", None) is not None
    )

    if use_susie:
        alpha = result.alpha.to(dtype=torch.float64)  # (L, p)
        n_layers = alpha.shape[0]
        uniform_threshold = 1.0 / p  # signal is "active" if max > 1/p

        for l_idx in range(n_layers):
            layer_probs = alpha[l_idx]  # (p,)
            if layer_probs.max().item() <= uniform_threshold:
                continue  # inactive signal layer

            # Sort variants by layer probability descending
            sorted_vals, sorted_idx = torch.sort(layer_probs, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=0)

            snp_indices: list[int] = []
            snp_ids: list[str] = []
            pips: list[float] = []
            cum_cov = 0.0

            for rank in range(p):
                idx = sorted_idx[rank].item()
                prob = sorted_vals[rank].item()

                if prob < min_pip:
                    continue
                if deduplicate and idx in used:
                    continue

                snp_indices.append(idx)
                snp_ids.append(result.snp[idx])
                pips.append(prob)
                cum_cov += prob

                if cum_cov >= coverage:
                    break

            if not snp_indices:
                continue

            pip_tensor = torch.tensor(pips, dtype=torch.float64)

            # Lead is the first (highest-prob) variant in the set
            lead_idx = snp_indices[0]
            cs = CredibleSet(
                signal_index=len(sets),
                snp_indices=snp_indices,
                snp_ids=snp_ids,
                pip=pip_tensor,
                cumulative_coverage=cum_cov,
                lead_snp_index=lead_idx,
                lead_snp_id=result.snp[lead_idx],
                lead_pip=pips[0],
            )
            sets.append(cs)

            if deduplicate:
                used.update(snp_indices)
    else:
        # CAVI or missing alpha: single credible set from global PIP
        pip = result.pip.to(dtype=torch.float64)
        sorted_vals, sorted_idx = torch.sort(pip, descending=True)

        snp_indices = []
        snp_ids = []
        pips = []
        cum_cov = 0.0

        for rank in range(p):
            idx = sorted_idx[rank].item()
            prob = sorted_vals[rank].item()

            if prob < min_pip:
                continue

            snp_indices.append(idx)
            snp_ids.append(result.snp[idx])
            pips.append(prob)
            cum_cov += prob

            if cum_cov >= coverage:
                break

        if snp_indices:
            pip_tensor = torch.tensor(pips, dtype=torch.float64)
            lead_idx = snp_indices[0]
            cs = CredibleSet(
                signal_index=0,
                snp_indices=snp_indices,
                snp_ids=snp_ids,
                pip=pip_tensor,
                cumulative_coverage=cum_cov,
                lead_snp_index=lead_idx,
                lead_snp_id=result.snp[lead_idx],
                lead_pip=pips[0],
            )
            sets.append(cs)

    return sets


# ---------------------------------------------------------------------------
# annotate_sumstats
# ---------------------------------------------------------------------------


def annotate_sumstats(
    ss: SumStats,
    fm: BayesianVSResult,
) -> AnnotatedSumStats:
    """Merge fine-mapping PIPs and credible set membership onto summary stats.

    Variants present in *ss* but absent from *fm* receive ``pip=0``,
    ``in_credible_set=-1``, and zero posterior effect / SD. Matching is
    performed by SNP ID.

    Parameters
    ----------
    ss : SumStats
        GWAS summary statistics.
    fm : BayesianVSResult
        Fine-mapping result (from ``BayesianVS``).

    Returns
    -------
    AnnotatedSumStats
        Annotated copy of the input summary statistics.
    """
    m = len(ss.snp)

    # Build lookup from fm SNP ID -> index in fm
    fm_lookup: dict[str, int] = {}
    for i, sid in enumerate(fm.snp):
        fm_lookup[sid] = i

    # Extract credible sets to determine membership
    cs_list = extract_credible_sets(fm)
    # Map fm index -> signal index
    fm_idx_to_signal: dict[int, int] = {}
    for cs in cs_list:
        for idx in cs.snp_indices:
            if idx not in fm_idx_to_signal:
                fm_idx_to_signal[idx] = cs.signal_index

    pip_out = torch.zeros(m, dtype=torch.float64)
    beta_post = torch.zeros(m, dtype=torch.float64)
    beta_post_sd = torch.zeros(m, dtype=torch.float64)
    in_cs: list[int] = [-1] * m

    fm_pip = fm.pip.to(dtype=torch.float64)
    fm_beta = fm.beta_mean.to(dtype=torch.float64)
    fm_sd = fm.beta_sd.to(dtype=torch.float64)

    for ss_i, sid in enumerate(ss.snp):
        fm_i = fm_lookup.get(sid)
        if fm_i is None:
            continue
        pip_out[ss_i] = fm_pip[fm_i]
        beta_post[ss_i] = fm_beta[fm_i]
        beta_post_sd[ss_i] = fm_sd[fm_i]
        if fm_i in fm_idx_to_signal:
            in_cs[ss_i] = fm_idx_to_signal[fm_i]

    return AnnotatedSumStats(
        chr=list(ss.chr),
        pos=list(ss.pos),
        snp=list(ss.snp),
        a1=list(ss.a1),
        a2=list(ss.a2),
        beta=ss.beta.to(dtype=torch.float64),
        se=ss.se.to(dtype=torch.float64),
        p=ss.p.to(dtype=torch.float64),
        pip=pip_out,
        in_credible_set=in_cs,
        beta_posterior=beta_post,
        beta_posterior_sd=beta_post_sd,
    )


# ---------------------------------------------------------------------------
# locus_summary
# ---------------------------------------------------------------------------


def locus_summary(
    results: list[BayesianVSResult],
    coverage: float = 0.95,
) -> LocusSummary:
    """Aggregate fine-mapping results across multiple loci.

    Each element of *results* is treated as one independent locus.

    Parameters
    ----------
    results : list[BayesianVSResult]
        One fine-mapping result per locus.
    coverage : float, optional
        Target posterior mass for credible set extraction (default 0.95).

    Returns
    -------
    LocusSummary
        Per-locus summary with lead SNP, credible set count, and total PIP.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results must be a non-empty list")

    summary = LocusSummary()

    for res in results:
        p = len(res.snp)
        pip = res.pip.to(dtype=torch.float64)

        # Chromosome: take the first variant's chromosome
        summary.locus_chr.append(res.chr[0] if res.chr else "?")
        summary.locus_start.append(min(res.pos) if res.pos else 0)
        summary.locus_end.append(max(res.pos) if res.pos else 0)
        summary.n_variants.append(p)

        # Total PIP
        summary.total_pip.append(float(pip.sum().item()))

        # Lead SNP (highest PIP)
        lead_rank = int(pip.argmax().item())
        summary.lead_snp.append(res.snp[lead_rank])
        summary.lead_pip.append(float(pip[lead_rank].item()))

        # Credible sets
        cs_list = extract_credible_sets(res, coverage=coverage)
        summary.credible_sets.append(cs_list)
        summary.n_signals.append(len(cs_list))

    return summary


# ---------------------------------------------------------------------------
# to_coloc_sumstats
# ---------------------------------------------------------------------------


def to_coloc_sumstats(
    fm: BayesianVSResult,
    ss: SumStats,
) -> SumStats:
    """Subset and align summary statistics to match a fine-mapping result.

    Returns a new ``SumStats`` whose variants are in the same order as
    *fm*, ready for ``coloc_pairwise`` or ``hyprcoloc``. Only variants
    present in both *fm* and *ss* (matched by SNP ID) are retained.

    Parameters
    ----------
    fm : BayesianVSResult
        Fine-mapping result defining the target variant order.
    ss : SumStats
        GWAS summary statistics to subset.

    Returns
    -------
    SumStats
        A new ``SumStats`` aligned to the variants in *fm*.

    Raises
    ------
    ValueError
        If no overlapping variants are found.
    """
    # Lazy import to avoid circular dependency at module load time
    from ._sumstats import SumStats as _SumStats

    # Build lookup from ss SNP ID -> index
    ss_lookup: dict[str, int] = {}
    for i, sid in enumerate(ss.snp):
        ss_lookup[sid] = i

    # Collect indices in fm order, keeping only shared variants
    fm_keep: list[int] = []  # indices into fm
    ss_keep: list[int] = []  # indices into ss
    for fm_i, sid in enumerate(fm.snp):
        ss_i = ss_lookup.get(sid)
        if ss_i is not None:
            fm_keep.append(fm_i)
            ss_keep.append(ss_i)

    if not ss_keep:
        raise ValueError(
            "No overlapping variants between fine-mapping result and "
            "summary statistics"
        )

    ss_idx = torch.tensor(ss_keep, dtype=torch.long)

    # Build the aligned SumStats using fm's variant identifiers for
    # the kept positions (they should be identical, but fm defines order)
    chr_out = [fm.chr[i] for i in fm_keep]
    pos_out = [fm.pos[i] for i in fm_keep]
    snp_out = [fm.snp[i] for i in fm_keep]
    a1_out = [fm.a1[i] for i in fm_keep]
    a2_out = [fm.a2[i] for i in fm_keep]

    beta_out = ss.beta.to(dtype=torch.float64)[ss_idx]
    se_out = ss.se.to(dtype=torch.float64)[ss_idx]
    p_out = ss.p.to(dtype=torch.float64)[ss_idx]
    n_out = ss.n.to(dtype=torch.float64)[ss_idx]

    af_out = None
    if ss.af is not None:
        af_out = ss.af.to(dtype=torch.float64)[ss_idx]

    return _SumStats(
        chr=chr_out,
        pos=pos_out,
        snp=snp_out,
        a1=a1_out,
        a2=a2_out,
        beta=beta_out,
        se=se_out,
        p=p_out,
        n=n_out,
        af=af_out,
    )
