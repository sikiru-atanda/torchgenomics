"""Coloc-based prefilter for ``scan_mediation`` candidate pairs.

Reuses ``torchgwas.postgwas._hyprcoloc.coloc_pairwise`` unchanged: for each
mediator feature, we compute PP(H4) between the outcome GWAS and the feature's
QTL summary stats within the cis window around that feature. Pairs belonging
to a feature whose PP(H4) is below the threshold are dropped before the
mediation fit runs.

This is a *coarse* filter — it operates per feature, not per (SNP, feature)
pair — because classical coloc tests are region-level. It is extremely
cheap (one ABF pass per feature) and typically shrinks the candidate set
10-100x before the per-pair mediation work kicks in.
"""

from __future__ import annotations

import numpy as np

from ..postgwas._hyprcoloc import coloc_pairwise
from ..postgwas._sumstats import SumStats


def _slice_sumstats(ss: SumStats, idx: np.ndarray) -> SumStats:
    """Return a SumStats restricted to row indices ``idx`` (numpy int64)."""
    idx_list = [int(i) for i in idx]
    import torch
    return SumStats(
        chr=[ss.chr[i] for i in idx_list],
        pos=[ss.pos[i] for i in idx_list],
        snp=[ss.snp[i] for i in idx_list],
        a1=[ss.a1[i] for i in idx_list],
        a2=[ss.a2[i] for i in idx_list],
        beta=ss.beta[torch.tensor(idx_list, dtype=torch.long)].clone(),
        se=ss.se[torch.tensor(idx_list, dtype=torch.long)].clone(),
        p=ss.p[torch.tensor(idx_list, dtype=torch.long)],
        n=ss.n[torch.tensor(idx_list, dtype=torch.long)],
        af=None if ss.af is None else ss.af[torch.tensor(idx_list, dtype=torch.long)],
    )


def coloc_prefilter_pairs(
    pairs: list[tuple[int, int]],
    y_sumstats: SumStats,
    m_sumstats: SumStats,
    *,
    snp_chrom: list[str] | None = None,
    snp_pos: list[int] | None = None,
    feature_chrom: list[str] | None = None,
    feature_pos: list[int] | None = None,
    cis_window_bp: int = 1_000_000,
    coloc_threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], dict[int, float]]:
    """Filter (snp_idx, feature_idx) pairs by per-feature coloc PP.H4.

    ``y_sumstats`` is the outcome-phenotype GWAS summary; ``m_sumstats`` is
    the mediator-QTL summary for the *same* SNP panel, with matching row
    order (caller responsibility; both are typically aligned to the panel
    by SNP ID before calling this function).

    For each unique feature appearing in ``pairs``, the cis region is the
    set of SNPs on ``feature_chrom[j]`` within ``cis_window_bp`` of
    ``feature_pos[j]``; the prefilter runs ``coloc_pairwise`` on the two
    sumstats restricted to that region and retains the feature iff
    ``PP.H4 >= coloc_threshold``.

    Returns ``(kept_pairs, pp_h4_by_feature)`` — the filtered pair list and
    a dict mapping feature index to its computed PP(H4).
    """
    if y_sumstats.beta.shape[0] != m_sumstats.beta.shape[0]:
        raise ValueError(
            "y_sumstats and m_sumstats must be aligned to the same SNP panel."
        )
    if snp_chrom is None or snp_pos is None:
        raise ValueError("snp_chrom and snp_pos are required for cis-region slicing.")
    if feature_chrom is None or feature_pos is None:
        raise ValueError("feature_chrom and feature_pos are required.")

    snp_pos_np = np.asarray(snp_pos, dtype=np.int64)
    snp_chr_np = np.asarray(snp_chrom, dtype=object)

    # Group features by their cis region signature (chrom, pos).
    feat_indices = sorted({j for _, j in pairs})
    pp_h4_by_feature: dict[int, float] = {}
    kept_features: set[int] = set()

    for j in feat_indices:
        fc = feature_chrom[j]
        fp = int(feature_pos[j])
        lo, hi = fp - int(cis_window_bp), fp + int(cis_window_bp)
        mask = (snp_chr_np == fc) & (snp_pos_np >= lo) & (snp_pos_np <= hi)
        idx = np.nonzero(mask)[0]
        if len(idx) < 2:
            # Need at least 2 SNPs for a meaningful coloc computation.
            pp_h4_by_feature[j] = 0.0
            continue
        y_slice = _slice_sumstats(y_sumstats, idx)
        m_slice = _slice_sumstats(m_sumstats, idx)
        res = coloc_pairwise(y_slice, m_slice)
        pp_h4_by_feature[j] = float(res.pp_h4)
        if res.pp_h4 >= coloc_threshold:
            kept_features.add(j)

    kept_pairs = [(i, j) for (i, j) in pairs if j in kept_features]
    return kept_pairs, pp_h4_by_feature
