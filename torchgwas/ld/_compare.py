"""Block comparison and concordance metrics.

Compares block detection results across methods or against PLINK
reference outputs.  Provides Jaccard index, boundary agreement,
coverage concordance, and summary statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ._blocks import LDBlock

logger = logging.getLogger(__name__)


@dataclass
class BlockComparisonResult:
    """Concordance metrics between two sets of blocks."""

    # Counts
    n_blocks_a: int
    n_blocks_b: int
    n_variants_a: int  # total variant coverage (union of all blocks)
    n_variants_b: int

    # SNP-level agreement
    jaccard_index: float        # |intersection| / |union| of covered variant indices
    overlap_coefficient: float  # |intersection| / min(|A|, |B|)
    dice_coefficient: float     # 2*|intersection| / (|A| + |B|)

    # Block-level agreement
    n_matched_a: int  # blocks in A that overlap ≥50% with some block in B
    n_matched_b: int  # blocks in B that overlap ≥50% with some block in A
    match_rate_a: float  # n_matched_a / n_blocks_a
    match_rate_b: float  # n_matched_b / n_blocks_b

    # Boundary agreement
    mean_boundary_dist: float  # mean distance (bp) between nearest boundaries
    median_boundary_dist: float

    # Size statistics
    mean_size_a: float
    mean_size_b: float
    mean_r2_a: float
    mean_r2_b: float

    # Method names
    method_a: str
    method_b: str

    # Per-chromosome breakdown
    per_chr: dict[str, dict] = field(default_factory=dict)


def compare_blocks(
    blocks_a: list[LDBlock],
    blocks_b: list[LDBlock],
    *,
    overlap_threshold: float = 0.5,
    method_a: str = "",
    method_b: str = "",
) -> BlockComparisonResult:
    """Compare two sets of detected blocks.

    Parameters
    ----------
    blocks_a, blocks_b : list[LDBlock]
        Two sets of blocks to compare.
    overlap_threshold : float
        Minimum reciprocal overlap fraction to count as "matched".
    method_a, method_b : str
        Labels for the two methods (auto-detected from blocks if empty).

    Returns
    -------
    BlockComparisonResult
    """
    if not method_a and blocks_a:
        method_a = blocks_a[0].method
    if not method_b and blocks_b:
        method_b = blocks_b[0].method

    # Collect covered variant indices
    set_a: set[int] = set()
    for b in blocks_a:
        set_a.update(b.variant_indices)
    set_b: set[int] = set()
    for b in blocks_b:
        set_b.update(b.variant_indices)

    intersection = set_a & set_b
    union = set_a | set_b

    n_inter = len(intersection)
    n_union = len(union)
    n_a = len(set_a)
    n_b = len(set_b)

    jaccard = n_inter / n_union if n_union > 0 else 1.0
    overlap_coeff = n_inter / min(n_a, n_b) if min(n_a, n_b) > 0 else 1.0
    dice = 2 * n_inter / (n_a + n_b) if (n_a + n_b) > 0 else 1.0

    # Block-level matching: A block in A "matches" B if it overlaps
    # ≥ overlap_threshold of its variants with some block in B
    n_matched_a = _count_matched(blocks_a, blocks_b, overlap_threshold)
    n_matched_b = _count_matched(blocks_b, blocks_a, overlap_threshold)

    n_blocks_a = len(blocks_a)
    n_blocks_b = len(blocks_b)
    match_rate_a = n_matched_a / n_blocks_a if n_blocks_a > 0 else 1.0
    match_rate_b = n_matched_b / n_blocks_b if n_blocks_b > 0 else 1.0

    # Boundary agreement
    boundaries_a = _extract_boundaries(blocks_a)
    boundaries_b = _extract_boundaries(blocks_b)
    mean_bd, median_bd = _boundary_distances(boundaries_a, boundaries_b)

    # Size and quality stats
    mean_size_a = sum(b.n_variants for b in blocks_a) / n_blocks_a if n_blocks_a > 0 else 0.0
    mean_size_b = sum(b.n_variants for b in blocks_b) / n_blocks_b if n_blocks_b > 0 else 0.0
    mean_r2_a = sum(b.mean_r2 for b in blocks_a) / n_blocks_a if n_blocks_a > 0 else 0.0
    mean_r2_b = sum(b.mean_r2 for b in blocks_b) / n_blocks_b if n_blocks_b > 0 else 0.0

    # Per-chromosome breakdown
    per_chr = _per_chr_comparison(blocks_a, blocks_b, overlap_threshold)

    return BlockComparisonResult(
        n_blocks_a=n_blocks_a,
        n_blocks_b=n_blocks_b,
        n_variants_a=n_a,
        n_variants_b=n_b,
        jaccard_index=jaccard,
        overlap_coefficient=overlap_coeff,
        dice_coefficient=dice,
        n_matched_a=n_matched_a,
        n_matched_b=n_matched_b,
        match_rate_a=match_rate_a,
        match_rate_b=match_rate_b,
        mean_boundary_dist=mean_bd,
        median_boundary_dist=median_bd,
        mean_size_a=mean_size_a,
        mean_size_b=mean_size_b,
        mean_r2_a=mean_r2_a,
        mean_r2_b=mean_r2_b,
        method_a=method_a,
        method_b=method_b,
        per_chr=per_chr,
    )


def compare_all_methods(
    blocks_by_method: dict[str, list[LDBlock]],
    *,
    reference: str | None = None,
    overlap_threshold: float = 0.5,
) -> dict[tuple[str, str], BlockComparisonResult]:
    """Pairwise comparison of multiple block detection methods.

    Parameters
    ----------
    blocks_by_method : dict[str, list[LDBlock]]
        Method name -> list of detected blocks.
    reference : str, optional
        If provided, only compare each method against this reference.
        Otherwise, compute all pairwise comparisons.
    overlap_threshold : float
        Minimum overlap fraction for block matching.

    Returns
    -------
    dict[(method_a, method_b) -> BlockComparisonResult]
    """
    methods = sorted(blocks_by_method.keys())
    results: dict[tuple[str, str], BlockComparisonResult] = {}

    if reference:
        if reference not in blocks_by_method:
            raise ValueError(f"Reference method '{reference}' not found.")
        ref_blocks = blocks_by_method[reference]
        for m in methods:
            if m == reference:
                continue
            results[(m, reference)] = compare_blocks(
                blocks_by_method[m], ref_blocks,
                overlap_threshold=overlap_threshold,
                method_a=m, method_b=reference,
            )
    else:
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                results[(m1, m2)] = compare_blocks(
                    blocks_by_method[m1], blocks_by_method[m2],
                    overlap_threshold=overlap_threshold,
                    method_a=m1, method_b=m2,
                )

    return results


def format_comparison_table(
    results: dict[tuple[str, str], BlockComparisonResult],
) -> str:
    """Format pairwise comparison results as a readable table.

    Returns
    -------
    str
        Tab-separated table string.
    """
    header = (
        "Method_A\tMethod_B\tBlocks_A\tBlocks_B\t"
        "Jaccard\tDice\tOverlap_Coeff\t"
        "Match_Rate_A\tMatch_Rate_B\t"
        "Mean_Boundary_Dist\t"
        "Mean_Size_A\tMean_Size_B\t"
        "Mean_R2_A\tMean_R2_B"
    )
    lines = [header]
    for (m_a, m_b), r in sorted(results.items()):
        lines.append(
            f"{r.method_a}\t{r.method_b}\t{r.n_blocks_a}\t{r.n_blocks_b}\t"
            f"{r.jaccard_index:.4f}\t{r.dice_coefficient:.4f}\t{r.overlap_coefficient:.4f}\t"
            f"{r.match_rate_a:.4f}\t{r.match_rate_b:.4f}\t"
            f"{r.mean_boundary_dist:.1f}\t"
            f"{r.mean_size_a:.1f}\t{r.mean_size_b:.1f}\t"
            f"{r.mean_r2_a:.4f}\t{r.mean_r2_b:.4f}"
        )
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────


def _count_matched(
    blocks_query: list[LDBlock],
    blocks_target: list[LDBlock],
    threshold: float,
) -> int:
    """Count blocks in query that overlap ≥threshold with any target block."""
    # Pre-build target sets by chromosome for efficiency
    target_by_chr: dict[str, list[set[int]]] = {}
    for b in blocks_target:
        target_by_chr.setdefault(b.region.chr, []).append(set(b.variant_indices))

    matched = 0
    for b in blocks_query:
        query_set = set(b.variant_indices)
        if not query_set:
            continue
        targets = target_by_chr.get(b.region.chr, [])
        for t_set in targets:
            overlap = len(query_set & t_set)
            # Reciprocal overlap: fraction of query that overlaps
            if overlap / len(query_set) >= threshold:
                matched += 1
                break
    return matched


def _extract_boundaries(blocks: list[LDBlock]) -> dict[str, list[int]]:
    """Extract block boundaries (start, end bp) by chromosome."""
    boundaries: dict[str, list[int]] = {}
    for b in blocks:
        chrom = b.region.chr
        boundaries.setdefault(chrom, [])
        boundaries[chrom].append(b.region.start)
        boundaries[chrom].append(b.region.end)
    for chrom in boundaries:
        boundaries[chrom] = sorted(set(boundaries[chrom]))
    return boundaries


def _boundary_distances(
    boundaries_a: dict[str, list[int]],
    boundaries_b: dict[str, list[int]],
) -> tuple[float, float]:
    """Compute mean and median nearest-boundary distance (bp).

    For each boundary in A, find the nearest boundary in B (same chr).
    """
    dists: list[float] = []
    all_chrs = set(boundaries_a) | set(boundaries_b)

    for chrom in all_chrs:
        ba = boundaries_a.get(chrom, [])
        bb = boundaries_b.get(chrom, [])
        if not ba or not bb:
            continue

        # For each boundary in A, find nearest in B
        for bp in ba:
            min_dist = min(abs(bp - b) for b in bb)
            dists.append(float(min_dist))
        # Symmetric: for each boundary in B, find nearest in A
        for bp in bb:
            min_dist = min(abs(bp - a) for a in ba)
            dists.append(float(min_dist))

    if not dists:
        return 0.0, 0.0

    dists.sort()
    mean_d = sum(dists) / len(dists)
    n = len(dists)
    median_d = dists[n // 2] if n % 2 == 1 else (dists[n // 2 - 1] + dists[n // 2]) / 2.0
    return mean_d, median_d


def _per_chr_comparison(
    blocks_a: list[LDBlock],
    blocks_b: list[LDBlock],
    overlap_threshold: float,
) -> dict[str, dict]:
    """Per-chromosome comparison breakdown."""
    # Group blocks by chromosome
    a_by_chr: dict[str, list[LDBlock]] = {}
    for b in blocks_a:
        a_by_chr.setdefault(b.region.chr, []).append(b)
    b_by_chr: dict[str, list[LDBlock]] = {}
    for b in blocks_b:
        b_by_chr.setdefault(b.region.chr, []).append(b)

    all_chrs = sorted(set(a_by_chr) | set(b_by_chr))
    per_chr: dict[str, dict] = {}

    for chrom in all_chrs:
        ca = a_by_chr.get(chrom, [])
        cb = b_by_chr.get(chrom, [])

        set_a = set()
        for bl in ca:
            set_a.update(bl.variant_indices)
        set_b = set()
        for bl in cb:
            set_b.update(bl.variant_indices)

        inter = set_a & set_b
        union = set_a | set_b
        jaccard = len(inter) / len(union) if union else 1.0

        per_chr[chrom] = {
            "n_blocks_a": len(ca),
            "n_blocks_b": len(cb),
            "n_variants_a": len(set_a),
            "n_variants_b": len(set_b),
            "jaccard": jaccard,
        }

    return per_chr
