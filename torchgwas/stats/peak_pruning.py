"""LD-window peak pruning for QTL detection (charter Section 9g).

Merges nearby significant markers on the same chromosome within a
base-pair window, keeping only independent QTL peaks. Equivalent
to GWASpoly's ``get.QTL`` function.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.base import ScanResult


@dataclass
class QTLPeak:
    """A single QTL peak after LD-window pruning."""

    snp: str
    chr: str
    pos: int
    neglog10p: float
    best_model: str  # gene-action model name (or "NA" if not from best-model)
    window_start: int
    window_end: int
    n_markers_in_window: int


def prune_peaks(
    scan_result: ScanResult,
    bp_window: int = 1_000_000,
    p_threshold: float = 1e-4,
    best_models: list[str] | None = None,
) -> list[QTLPeak]:
    """Merge nearby significant markers into independent QTL peaks.

    Algorithm (greedy, matches GWASpoly ``get.QTL``):
    1. Filter to markers with p < p_threshold.
    2. Compute -log10(p) for each.
    3. Sort by -log10(p) descending.
    4. Take the top marker as a peak, remove all markers within
       bp_window on the same chromosome, repeat.

    Parameters
    ----------
    scan_result : ScanResult
        GWAS scan output with chr, pos, snp, p fields.
    bp_window : int
        Window size in base pairs (default: 1 Mb).
    p_threshold : float
        Significance threshold for including a marker.
    best_models : list[str] | None
        Per-marker best gene-action model names (from BestModelResult).
        If None, all peaks labeled "NA".

    Returns
    -------
    list[QTLPeak]
        Independent peaks sorted by -log10(p) descending.
    """
    p = scan_result.p.to(torch.float64)
    m = p.shape[0]

    if best_models is not None and len(best_models) != m:
        raise ValueError(
            f"best_models length {len(best_models)} != number of markers {m}"
        )

    # Filter to significant markers
    sig_mask = (p < p_threshold) & torch.isfinite(p) & (p > 0)
    sig_indices = torch.where(sig_mask)[0].tolist()

    if not sig_indices:
        return []

    # Compute -log10(p) for significant markers
    neglog10p = -torch.log10(torch.clamp(p, min=1e-300))

    # Build candidate list: (index, chr, pos, score)
    candidates = []
    for idx in sig_indices:
        candidates.append({
            "idx": idx,
            "chr": scan_result.chr[idx],
            "pos": scan_result.pos[idx],
            "score": neglog10p[idx].item(),
            "snp": scan_result.snp[idx],
            "model": best_models[idx] if best_models else "NA",
        })

    # Sort by score descending
    candidates.sort(key=lambda x: -x["score"])

    # Greedy pruning
    peaks = []
    used = set()

    for cand in candidates:
        if cand["idx"] in used:
            continue

        # This marker becomes a peak — count and remove neighbors
        peak_chr = cand["chr"]
        peak_pos = cand["pos"]
        window_start = peak_pos - bp_window // 2
        window_end = peak_pos + bp_window // 2
        n_in_window = 0

        for other in candidates:
            if other["idx"] in used:
                continue
            if other["chr"] == peak_chr and abs(other["pos"] - peak_pos) <= bp_window:
                used.add(other["idx"])
                n_in_window += 1

        peaks.append(QTLPeak(
            snp=cand["snp"],
            chr=peak_chr,
            pos=peak_pos,
            neglog10p=cand["score"],
            best_model=cand["model"],
            window_start=max(0, window_start),
            window_end=window_end,
            n_markers_in_window=n_in_window,
        ))

    return peaks
