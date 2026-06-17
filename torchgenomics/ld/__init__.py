"""Haplotype block detection and pairwise LD statistics.

Thirteen block detection methods:

Classical (PLINK-style):
  - ``gabriel``: Gabriel et al. (2002) D' confidence interval method
  - ``four_gamete``: Four-gamete test (Wang et al. 2002)
  - ``spine``: Solid spine of LD (D' threshold from block anchors)
  - ``r2``: Simple r-squared threshold on adjacent pairs

Novel scientific:
  - ``gwas_aligned``: Phenotype-free, DP-optimized for downstream GWAS power
  - ``uncertainty``: GP/DS-corrected LD for imputed/low-depth data
  - ``cross_pop``: Multi-population consensus via spectral partitioning
  - ``graphical``: Conditional-independence blocks via graphical Lasso
  - ``changepoint``: Recombination / LD-decay change-point detection

Literature:
  - ``big_ld``: Big-LD interval graph + MWIS (Kim et al. 2018)
  - ``dp_optimize``: DP-optimal haplotype diversity / tag-SNP blocks
  - ``cc_graph``: Connected-component graph blocks with tag-SNP selection

Diagnostics:
  - ``wall_pritchard``: Wall & Pritchard blockiness score

Public API
----------
detect_blocks        High-level block detection (dispatches to method)
compute_pairwise_ld  Windowed pairwise LD computation (r², D', D' CI)
compute_r2_matrix    Full r² matrix for a genotype submatrix
compute_dprime_matrix Full D' matrix for a genotype submatrix
save_blocks_bed      Write blocks to BED file
LDBlock              Detected haplotype block dataclass
PairwiseLD           Pairwise LD statistics dataclass
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor

from ._blocks import (
    LDBlock,
    PairwiseLD,
    detect_blocks_four_gamete,
    detect_blocks_gabriel,
    detect_blocks_r2,
    detect_blocks_spine,
)
from ._blocks_diagnostics import compute_wall_pritchard_diagnostics
from ._blocks_literature import (
    detect_blocks_big_ld,
    detect_blocks_cc_graph,
    detect_blocks_dp_optimize,
)
from ._blocks_novel import (
    detect_blocks_changepoint,
    detect_blocks_cross_pop,
    detect_blocks_graphical,
    detect_blocks_gwas_aligned,
    detect_blocks_uncertainty,
)
from ._compare import (
    BlockComparisonResult,
    compare_all_methods,
    compare_blocks,
    format_comparison_table,
)
from ._dprime_ci import dprime_confidence_interval
from ._em_haplotype import build_genotype_counts
from ._pairwise import (
    compute_dprime_phased,
    compute_dprime_unphased,
    compute_r2_matrix,
    compute_r2_pairs,
)
from ._plink_compat import (
    PLINKBlock,
    ldblocks_to_plink_det,
    load_plink_blocks_det,
    plink_blocks_to_ldblocks,
)

__all__ = [
    "detect_blocks",
    "compute_pairwise_ld",
    "compute_r2_matrix",
    "compute_dprime_matrix",
    "save_blocks_bed",
    "save_blocks_det",
    "save_blocks_summary",
    "compare_blocks",
    "compare_all_methods",
    "format_comparison_table",
    "load_plink_blocks_det",
    "plink_blocks_to_ldblocks",
    "ldblocks_to_plink_det",
    "LDBlock",
    "PairwiseLD",
    "PLINKBlock",
    "BlockComparisonResult",
]

logger = logging.getLogger(__name__)


# ── High-level API ──────────────────────────────────────────────────

def detect_blocks(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    method: str = "gabriel",
    haplotypes: Tensor | None = None,
    max_kb: float = 200.0,
    maf_min: float = 0.05,
    device: torch.device | None = None,
    # Advanced method inputs
    gp_probs: Tensor | None = None,
    population_ids: list[str] | None = None,
    population_weights: dict[str, float] | None = None,
    recomb_map_cm: list[float] | None = None,
    ploidy: int = 2,
    **method_kwargs,
) -> list[LDBlock]:
    """Detect haplotype blocks from genotype data.

    Parameters
    ----------
    G : (n, m) Tensor
        Genotype dosage matrix.
    variant_pos : list[int]
        Base-pair positions for each variant.
    variant_chr : list[str]
        Chromosome labels for each variant.
    variant_ids : list[str], optional
        Variant IDs (for block naming).
    method : str
        Detection method.  Classical: "gabriel", "four_gamete", "spine",
        "r2".  Novel: "gwas_aligned", "uncertainty", "cross_pop",
        "graphical", "changepoint".  Literature: "big_ld", "cc_graph",
        "dp_optimize".  Diagnostics: "wall_pritchard".
    haplotypes : (n, ploidy, m) Tensor, optional
        Phased haplotype data (required for four_gamete, improves D').
    max_kb : float
        Maximum distance (kb) for pairwise LD computation.
    maf_min : float
        MAF filter for informative pairs.
    device : torch.device, optional
        Device for computation.
    gp_probs : (n, m, k+1) Tensor, optional
        Genotype probabilities (for uncertainty method).
    population_ids : list[str], optional
        Per-sample population labels (for cross_pop method).
    population_weights : dict, optional
        Population -> weight mapping (for cross_pop method).
    recomb_map_cm : list[float], optional
        CentiMorgan positions per variant (for changepoint method).
    ploidy : int
        Ploidy level (default 2, used by uncertainty method).
    **method_kwargs
        Method-specific parameters.

    Returns
    -------
    list[LDBlock]
    """
    # Filter method_kwargs to only pass recognized parameters
    _METHOD_KWARGS = {
        # Classical
        "four_gamete": {"freq_threshold", "min_block_snps"},
        "gabriel": {"ci_low", "ci_high", "rec_high", "strong_pct", "rec_max_pct", "min_block_snps"},
        "spine": {"d_prime_threshold", "min_block_snps"},
        "r2": {"r2_threshold", "min_block_snps", "tolerance"},
        # Novel
        "gwas_aligned": {"max_block_snps", "condition_penalty", "min_block_snps"},
        "uncertainty": {"base_method", "min_dosage_rsq", "min_block_snps",
                        "r2_threshold", "d_prime_threshold"},
        "cross_pop": {"n_blocks_hint", "stability_threshold", "min_block_snps", "aggregation"},
        "graphical": {"l1_penalty", "window_size", "overlap", "min_block_snps", "method_variant"},
        "changepoint": {"penalty", "min_segment_size", "cost_model", "use_recomb_map"},
        # Literature
        "big_ld": {"r2_threshold", "min_block_snps", "window_size"},
        "cc_graph": {"r2_threshold", "min_block_snps", "window", "include_singletons"},
        "dp_optimize": {"objective", "penalty", "min_block_snps", "max_block_snps",
                        "hap_freq_threshold", "tag_r2_threshold"},
        # Diagnostics
        "wall_pritchard": {"d_prime_threshold", "n_permutations"},
    }

    if method not in _METHOD_KWARGS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {', '.join(sorted(_METHOD_KWARGS))}"
        )

    allowed = _METHOD_KWARGS[method]
    filtered_kwargs = {k: v for k, v in method_kwargs.items() if k in allowed}

    # ── Classical methods ──────────────────────────────────────────
    if method == "four_gamete":
        if haplotypes is None:
            raise ValueError(
                "four_gamete method requires phased haplotypes. "
                "Provide haplotypes=(n, ploidy, m) tensor."
            )
        return detect_blocks_four_gamete(
            haplotypes, variant_pos, variant_chr, variant_ids,
            **filtered_kwargs,
        )

    if method in {"gabriel", "spine", "r2"}:
        compute_ci = method == "gabriel"
        pld = compute_pairwise_ld(
            G, variant_pos, variant_chr,
            haplotypes=haplotypes,
            max_kb=max_kb,
            maf_min=maf_min,
            compute_ci=compute_ci,
            device=device,
        )
        n_snps = len(variant_pos)

        if method == "gabriel":
            return detect_blocks_gabriel(
                pld, variant_pos, variant_chr, variant_ids,
                **filtered_kwargs,
            )
        elif method == "spine":
            return detect_blocks_spine(
                pld, variant_pos, variant_chr, variant_ids,
                n_snps=n_snps, **filtered_kwargs,
            )
        else:  # r2
            return detect_blocks_r2(
                pld, variant_pos, variant_chr, variant_ids,
                n_snps=n_snps, **filtered_kwargs,
            )

    # ── Novel methods ──────────────────────────────────────────────
    if method == "gwas_aligned":
        return detect_blocks_gwas_aligned(
            G, variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "uncertainty":
        if gp_probs is None:
            raise ValueError(
                "uncertainty method requires gp_probs=(n, m, k+1) tensor."
            )
        return detect_blocks_uncertainty(
            G, gp_probs, ploidy, variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "cross_pop":
        if population_ids is None:
            raise ValueError(
                "cross_pop method requires population_ids (per-sample labels)."
            )
        return detect_blocks_cross_pop(
            G, population_ids, population_weights,
            variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "graphical":
        return detect_blocks_graphical(
            G, variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "changepoint":
        return detect_blocks_changepoint(
            G, variant_pos, variant_chr, variant_ids,
            recomb_map_cm=recomb_map_cm,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    # ── Literature methods ─────────────────────────────────────────
    if method == "big_ld":
        return detect_blocks_big_ld(
            G, variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "cc_graph":
        return detect_blocks_cc_graph(
            G, variant_pos, variant_chr, variant_ids,
            max_kb=max_kb, device=device, **filtered_kwargs,
        )

    if method == "dp_optimize":
        return detect_blocks_dp_optimize(
            G, variant_pos, variant_chr, variant_ids,
            haplotypes=haplotypes, max_kb=max_kb, device=device,
            **filtered_kwargs,
        )

    # ── Diagnostics ────────────────────────────────────────────────
    # method == "wall_pritchard"
    pld = compute_pairwise_ld(
        G, variant_pos, variant_chr,
        haplotypes=haplotypes,
        max_kb=max_kb, maf_min=maf_min,
        compute_ci=False, device=device,
    )
    return compute_wall_pritchard_diagnostics(
        pld, variant_pos, variant_chr, variant_ids,
        **filtered_kwargs,
    )


def compute_pairwise_ld(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    *,
    haplotypes: Tensor | None = None,
    max_kb: float = 200.0,
    maf_min: float = 0.05,
    compute_ci: bool = True,
    device: torch.device | None = None,
) -> PairwiseLD:
    """Compute pairwise LD statistics within a distance window.

    Processes per-chromosome, window-by-window for memory efficiency.

    Parameters
    ----------
    G : (n, m) Tensor
    variant_pos : list[int]
    variant_chr : list[str]
    haplotypes : optional (n, ploidy, m) for phased D'
    max_kb : float
        Window size in kilobases.
    maf_min : float
        Skip pairs where either SNP has MAF < maf_min.
    compute_ci : bool
        Whether to compute D' confidence intervals (needed for Gabriel).
    device : torch.device, optional

    Returns
    -------
    PairwiseLD
    """
    if device is None:
        device = G.device
    G = G.to(device=device, dtype=torch.float64)

    n, m = G.shape
    max_bp = max_kb * 1000.0

    # MAF filter
    af = G.mean(dim=0) / 2.0  # assume diploid dosage [0, 2]
    maf = torch.min(af, 1.0 - af)
    maf_ok = maf >= maf_min  # (m,)

    # Group by chromosome
    chr_set = sorted(set(variant_chr))
    all_idx_i, all_idx_j = [], []
    all_r2, all_dp = [], []
    all_ci_low, all_ci_high = [], []

    for chrom in chr_set:
        # Get indices for this chromosome
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < 2:
            continue

        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=device)
        chr_pos = [variant_pos[i] for i in chr_mask]

        # Generate pairs within window
        pair_i_list = []
        pair_j_list = []
        for local_i in range(len(chr_mask)):
            if not maf_ok[chr_mask[local_i]]:
                continue
            pos_i = chr_pos[local_i]
            for local_j in range(local_i + 1, len(chr_mask)):
                if not maf_ok[chr_mask[local_j]]:
                    continue
                if chr_pos[local_j] - pos_i > max_bp:
                    break
                pair_i_list.append(local_i)
                pair_j_list.append(local_j)

        if not pair_i_list:
            continue

        # Convert to global indices for genotype extraction
        local_i_t = torch.tensor(pair_i_list, dtype=torch.long, device=device)
        local_j_t = torch.tensor(pair_j_list, dtype=torch.long, device=device)
        global_i = chr_idx[local_i_t]
        global_j = chr_idx[local_j_t]

        # Compute r²
        r2 = compute_r2_pairs(G, global_i, global_j)

        # Compute D'
        if haplotypes is not None:
            dp = compute_dprime_phased(haplotypes.to(device), global_i, global_j)
        else:
            dp = compute_dprime_unphased(G, global_i, global_j)

        # D' confidence intervals
        if compute_ci:
            counts = build_genotype_counts(G, global_i, global_j)
            _, ci_low, ci_high = dprime_confidence_interval(counts)
        else:
            ci_low = torch.full_like(dp, -1.0)
            ci_high = torch.full_like(dp, 1.0)

        # Store with LOCAL indices (for block detection within chromosome)
        all_idx_i.append(local_i_t.cpu())
        all_idx_j.append(local_j_t.cpu())
        all_r2.append(r2.cpu())
        all_dp.append(dp.abs().cpu())
        all_ci_low.append(ci_low.cpu())
        all_ci_high.append(ci_high.cpu())

    if not all_idx_i:
        empty = torch.tensor([], dtype=torch.float64)
        empty_long = torch.tensor([], dtype=torch.long)
        return PairwiseLD(
            idx_i=empty_long, idx_j=empty_long,
            r2=empty, dprime=empty,
            dprime_ci_low=empty, dprime_ci_high=empty,
        )

    return PairwiseLD(
        idx_i=torch.cat(all_idx_i),
        idx_j=torch.cat(all_idx_j),
        r2=torch.cat(all_r2),
        dprime=torch.cat(all_dp),
        dprime_ci_low=torch.cat(all_ci_low),
        dprime_ci_high=torch.cat(all_ci_high),
    )


def compute_dprime_matrix(
    G: Tensor,
    *,
    haplotypes: Tensor | None = None,
) -> Tensor:
    """Compute full D' matrix for a small set of variants.

    Parameters
    ----------
    G : (n, m) Tensor
    haplotypes : (n, ploidy, m) optional

    Returns
    -------
    dprime : (m, m) float64
    """
    m = G.shape[1]
    device = G.device
    dprime = torch.zeros(m, m, dtype=torch.float64, device=device)

    if m < 2:
        return dprime

    # Generate all upper-triangle pairs
    idx = torch.triu_indices(m, m, offset=1, device=device)
    idx_i = idx[0]
    idx_j = idx[1]

    if haplotypes is not None:
        dp = compute_dprime_phased(haplotypes, idx_i, idx_j)
    else:
        dp = compute_dprime_unphased(G, idx_i, idx_j)

    dprime[idx_i, idx_j] = dp.abs()
    dprime[idx_j, idx_i] = dp.abs()

    return dprime


def save_blocks_bed(
    blocks: list[LDBlock],
    output_path: str | Path,
) -> None:
    """Write detected blocks to a BED file with standard fixed columns.

    All 13 columns are always written (NA for missing optional fields),
    ensuring consistent format across all 13 detection methods.

    Columns
    -------
    1. chr
    2. start (0-based inclusive)
    3. end (0-based exclusive)
    4. block_id
    5. n_variants
    6. mean_r2
    7. mean_dprime
    8. method
    9. blockiness_score (or NA)
    10. concentration_ratio (or NA)
    11. population_stability (or NA)
    12. uncertainty_metric (or NA)
    13. changepoint_confidence (or NA)
    """
    _HEADER = (
        "#chr\tstart\tend\tblock_id\tn_variants\tmean_r2\tmean_dprime\t"
        "method\tblockiness_score\tconcentration_ratio\t"
        "population_stability\tuncertainty_metric\tchangepoint_confidence"
    )

    def _fmt(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "NA"

    path = Path(output_path)
    with open(path, "w") as f:
        f.write(_HEADER + "\n")
        for block in blocks:
            r = block.region
            f.write(
                f"{r.chr}\t{r.start}\t{r.end}\t{r.region_id}\t"
                f"{block.n_variants}\t{block.mean_r2:.4f}\t"
                f"{block.mean_dprime:.4f}\t{block.method}\t"
                f"{_fmt(block.blockiness_score)}\t"
                f"{_fmt(block.concentration_ratio)}\t"
                f"{_fmt(block.population_stability)}\t"
                f"{_fmt(block.uncertainty_metric)}\t"
                f"{_fmt(block.changepoint_confidence)}\n"
            )

    logger.info("Saved %d blocks to %s", len(blocks), path)


def save_blocks_det(
    blocks: list[LDBlock],
    output_path: str | Path,
    variant_ids: list[str] | None = None,
) -> None:
    """Write blocks in PLINK-compatible .blocks.det format.

    Columns: CHR  BP1  BP2  KB  NSNPS  MEAN_R2  MEAN_DPRIME  METHOD  SNPS

    BP1/BP2 are 1-based inclusive (PLINK convention).
    SNPS is a pipe-delimited list of variant IDs.
    """
    path = Path(output_path)
    with open(path, "w") as f:
        f.write("CHR\tBP1\tBP2\tKB\tNSNPS\tMEAN_R2\tMEAN_DPRIME\tMETHOD\tSNPS\n")
        for block in blocks:
            r = block.region
            bp1 = r.start + 1   # 0-based → 1-based inclusive
            bp2 = r.end          # half-open → inclusive (PLINK convention)
            kb = (bp2 - bp1 + 1) / 1000.0

            if variant_ids and block.variant_indices:
                snps = "|".join(variant_ids[i] for i in block.variant_indices)
            else:
                snps = r.region_id

            f.write(
                f"{r.chr}\t{bp1}\t{bp2}\t{kb:.3f}\t"
                f"{block.n_variants}\t{block.mean_r2:.4f}\t"
                f"{block.mean_dprime:.4f}\t{block.method}\t{snps}\n"
            )

    logger.info("Saved %d blocks to %s (det format)", len(blocks), path)


def save_blocks_summary(
    blocks: list[LDBlock],
    output_path: str | Path,
) -> None:
    """Write block detection summary statistics.

    Reports method, total blocks, size distribution, LD statistics,
    genomic coverage, and per-chromosome breakdown.
    """
    import statistics

    path = Path(output_path)

    if not blocks:
        with open(path, "w") as f:
            f.write("No blocks detected.\n")
        return

    method = blocks[0].method
    sizes = [b.n_variants for b in blocks]
    r2s = [b.mean_r2 for b in blocks]
    spans_kb = [(b.region.end - b.region.start) / 1000.0 for b in blocks]

    # Per-chromosome counts
    chr_counts: dict[str, int] = {}
    chr_variants: dict[str, int] = {}
    for b in blocks:
        c = b.region.chr
        chr_counts[c] = chr_counts.get(c, 0) + 1
        chr_variants[c] = chr_variants.get(c, 0) + b.n_variants

    total_variants = sum(sizes)

    with open(path, "w") as f:
        f.write("Block Detection Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Method:             {method}\n")
        f.write(f"Total blocks:       {len(blocks)}\n")
        f.write(f"Total variants:     {total_variants}\n")
        f.write("\n")
        f.write("Block Size (variants)\n")
        f.write(f"{'-' * 30}\n")
        f.write(f"  Min:              {min(sizes)}\n")
        f.write(f"  Max:              {max(sizes)}\n")
        f.write(f"  Mean:             {statistics.mean(sizes):.1f}\n")
        f.write(f"  Median:           {statistics.median(sizes):.1f}\n")
        if len(sizes) > 1:
            f.write(f"  Std:              {statistics.stdev(sizes):.1f}\n")
        f.write("\n")
        f.write("Block Span (kb)\n")
        f.write(f"{'-' * 30}\n")
        f.write(f"  Min:              {min(spans_kb):.2f}\n")
        f.write(f"  Max:              {max(spans_kb):.2f}\n")
        f.write(f"  Mean:             {statistics.mean(spans_kb):.2f}\n")
        f.write(f"  Median:           {statistics.median(spans_kb):.2f}\n")
        f.write("\n")
        f.write("Within-Block LD (mean r2)\n")
        f.write(f"{'-' * 30}\n")
        f.write(f"  Min:              {min(r2s):.4f}\n")
        f.write(f"  Max:              {max(r2s):.4f}\n")
        f.write(f"  Mean:             {statistics.mean(r2s):.4f}\n")
        f.write("\n")
        f.write("Per-Chromosome Breakdown\n")
        f.write(f"{'-' * 30}\n")
        f.write(f"  {'CHR':<8}{'BLOCKS':>8}{'VARIANTS':>10}\n")
        for c in sorted(chr_counts):
            f.write(f"  {c:<8}{chr_counts[c]:>8}{chr_variants[c]:>10}\n")

    logger.info("Saved block summary to %s", path)
