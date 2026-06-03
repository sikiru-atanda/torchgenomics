"""Literature-based block detection methods.

  - Big-LD (Kim et al. 2018): interval graph + maximum-weight independent set
  - DP optimization: minimize haplotype diversity or tag-SNP objective
  - CC-Graph: connected-component graph pruning / block detection
"""

from __future__ import annotations

import os
from .._dispatch import native_disabled

import numpy as np
import torch
from torch import Tensor

from .._native import (
    HAS_NATIVE_BIG_LD,
    HAS_NATIVE_CC_GRAPH,
    HAS_NATIVE_DP_OPTIMIZE,
    _big_ld_native,
    _cc_graph_native,
    _dp_optimize_native,
)
from ..io.regions import Region
from ._blocks import LDBlock
from ._pairwise import compute_r2_matrix

_EPS = 1e-10


def _big_ld_native_enabled() -> bool:
    """Whether the native C++ Big-LD accelerator should be used."""
    return HAS_NATIVE_BIG_LD and not native_disabled()


def _dp_optimize_native_enabled() -> bool:
    """Whether the native C++ DP-optimize accelerator should be used."""
    return HAS_NATIVE_DP_OPTIMIZE and not native_disabled()


def _cc_graph_native_enabled() -> bool:
    """Whether the native C++ cc-graph accelerator should be used."""
    return HAS_NATIVE_CC_GRAPH and not native_disabled()


# ── Big-LD ─────────────────────────────────────────────────────────

def detect_blocks_big_ld(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    r2_threshold: float = 0.5,
    min_block_snps: int = 2,
    window_size: int = 200,
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks using the Big-LD algorithm (Kim et al. 2018).

    1. Compute windowed r² matrix.
    2. For each SNP pair with r² > threshold, create a "correlation
       interval" [i, j].
    3. Find the maximum-weight independent set (MWIS) on the interval
       graph, where weight = span * mean_r².
    4. Each selected interval becomes a block.

    Parameters
    ----------
    r2_threshold : float
        Minimum r² to create a correlation interval.
    window_size : int
        Maximum index span for candidate intervals.
    """
    from ._graph_utils import greedy_mwis

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []
    max_bp = max_kb * 1000.0

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        m_chr = len(chr_mask)
        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)
        G_chr = G[:, chr_idx]
        chr_pos = [variant_pos[i] for i in chr_mask]

        # Compute r² matrix
        R = compute_r2_matrix(G_chr)  # (m_chr, m_chr)

        # Build correlation intervals.
        # Native C++ shortcut: 2D-prefix-sum O(m·W) candidate scan.
        # The pure-Python loop below remains the canonical reference and
        # is exercised when the extension is missing or
        # TORCHGENOMICS_DISABLE_NATIVE=1 is set.
        intervals: list[tuple[int, int, float]] = []
        if _big_ld_native_enabled():
            R_np = R.detach().to(torch.float64).cpu().numpy()
            chr_pos_np = np.asarray(chr_pos, dtype=np.int64)
            intervals = [
                (int(i), int(j), float(w))
                for (i, j, w) in _big_ld_native.big_ld_intervals(
                    R_np, chr_pos_np,
                    float(r2_threshold),
                    int(min_block_snps),
                    int(window_size),
                    float(max_bp),
                )
            ]
        else:
            for i in range(m_chr):
                for j in range(i + min_block_snps - 1, min(i + window_size, m_chr)):
                    # Physical distance constraint
                    if chr_pos[j] - chr_pos[i] > max_bp:
                        break
                    # Check if all adjacent pairs in [i..j] exceed threshold
                    # (more permissive: check mean r² in the subblock)
                    R_sub = R[i:j + 1, i:j + 1]
                    mask = torch.ones_like(R_sub, dtype=torch.bool)
                    mask.fill_diagonal_(False)
                    if mask.sum() == 0:
                        continue
                    mean_r2 = R_sub[mask].mean().item()
                    if mean_r2 >= r2_threshold:
                        span = j - i + 1
                        weight = span * mean_r2
                        intervals.append((i, j, weight))

        if not intervals:
            continue

        # MWIS: select non-overlapping intervals with highest total weight
        selected = greedy_mwis(intervals)

        for start_local, end_local, weight in selected:
            if end_local - start_local + 1 < min_block_snps:
                continue

            start_global = chr_mask[start_local]
            end_global = chr_mask[end_local]
            chr_val = variant_chr[start_global]

            block_id = (
                f"{variant_ids[start_global]}-{variant_ids[end_global]}"
                if variant_ids
                else f"{chr_val}:{variant_pos[start_global]}-{variant_pos[end_global]}"
            )
            region = Region(
                region_id=block_id, chr=chr_val,
                start=variant_pos[start_global],
                end=variant_pos[end_global] + 1,
            )
            indices = list(range(start_global, end_global + 1))

            R_sub = R[start_local:end_local + 1, start_local:end_local + 1]
            block_mask = torch.ones_like(R_sub, dtype=torch.bool)
            block_mask.fill_diagonal_(False)
            mean_r2 = R_sub[block_mask].mean().item() if block_mask.sum() > 0 else 1.0

            all_blocks.append(LDBlock(
                region=region,
                n_variants=end_local - start_local + 1,
                variant_indices=indices,
                method="big_ld",
                mean_r2=mean_r2,
                mean_dprime=0.0,
            ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))


# ── DP Optimization Blocks ─────────────────────────────────────────

def detect_blocks_dp_optimize(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    objective: str = "haplotype_diversity",
    penalty: float = 0.0,
    min_block_snps: int = 2,
    max_block_snps: int = 50,
    hap_freq_threshold: float = 0.01,
    tag_r2_threshold: float = 0.8,
    max_kb: float = 200.0,
    haplotypes: Tensor | None = None,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks by minimizing an objective via dynamic programming.

    Two objectives:
      - ``"haplotype_diversity"``: minimize total number of distinct
        haplotype patterns across blocks (favors internally homogeneous
        blocks).
      - ``"tag_snp"``: minimize total number of tag SNPs needed to
        capture all common variants (at r² > ``tag_r2_threshold``).

    DP recurrence:
      OPT[j] = min_{i} { OPT[i] + cost(i+1, j) + penalty }
    """
    import math

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []
    max_bp = max_kb * 1000.0

    # Auto penalty
    eff_penalty = penalty if penalty > 0 else math.log(max(m, 2))

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        m_chr = len(chr_mask)
        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)
        G_chr = G[:, chr_idx]
        chr_pos = [variant_pos[i] for i in chr_mask]

        # Compute r² matrix (needed for tag_snp objective)
        R = compute_r2_matrix(G_chr)

        # Native C++ shortcut: runs the per-chromosome DP + both cost
        # branches in a single GIL-released call. The pure-Python DP
        # below remains the canonical algorithmic reference and is
        # exercised when the extension is missing or
        # TORCHGENOMICS_DISABLE_NATIVE=1 is set.
        if _dp_optimize_native_enabled():
            import numpy as np
            G_chr_np = G_chr.detach().to(torch.float64).cpu().numpy()
            R_np = R.detach().to(torch.float64).cpu().numpy()
            chr_pos_np = np.asarray(chr_pos, dtype=np.int64)
            obj_id = 0 if objective == "haplotype_diversity" else (
                1 if objective == "tag_snp" else -1
            )
            if obj_id < 0:
                raise ValueError(f"Unknown objective: {objective}")
            blocks_indices = [
                (int(s), int(e))
                for (s, e) in _dp_optimize_native.dp_optimize_segment(
                    G_chr_np, R_np, chr_pos_np,
                    obj_id, float(eff_penalty),
                    int(min_block_snps), int(max_block_snps),
                    float(max_bp),
                    float(hap_freq_threshold),
                    float(tag_r2_threshold),
                )
            ]
        else:
            # DP: opt[j] = min cost for segmenting chr_mask[0:j]
            INF = float("inf")
            opt = [INF] * (m_chr + 1)
            opt[0] = 0.0
            backtrack = [0] * (m_chr + 1)

            for j in range(min_block_snps, m_chr + 1):
                for i in range(max(0, j - max_block_snps), j - min_block_snps + 1):
                    # Physical distance constraint
                    if chr_pos[j - 1] - chr_pos[i] > max_bp:
                        continue
                    if opt[i] == INF:
                        continue

                    cost = _block_cost(
                        G_chr[:, i:j], R[i:j, i:j],
                        objective=objective,
                        hap_freq_threshold=hap_freq_threshold,
                        tag_r2_threshold=tag_r2_threshold,
                    )

                    total = opt[i] + cost + eff_penalty
                    if total < opt[j]:
                        opt[j] = total
                        backtrack[j] = i

            # Backtrack
            blocks_indices: list[tuple[int, int]] = []
            pos = m_chr
            while pos > 0:
                start = backtrack[pos]
                if pos - start >= min_block_snps:
                    blocks_indices.append((start, pos - 1))
                pos = start

            blocks_indices.reverse()

        for start_local, end_local in blocks_indices:
            start_global = chr_mask[start_local]
            end_global = chr_mask[end_local]
            chr_val = variant_chr[start_global]

            block_id = (
                f"{variant_ids[start_global]}-{variant_ids[end_global]}"
                if variant_ids
                else f"{chr_val}:{variant_pos[start_global]}-{variant_pos[end_global]}"
            )
            region = Region(
                region_id=block_id, chr=chr_val,
                start=variant_pos[start_global],
                end=variant_pos[end_global] + 1,
            )
            indices = list(range(start_global, end_global + 1))

            R_sub = R[start_local:end_local + 1, start_local:end_local + 1]
            mask = torch.ones_like(R_sub, dtype=torch.bool)
            mask.fill_diagonal_(False)
            mean_r2 = R_sub[mask].mean().item() if mask.sum() > 0 else 1.0

            all_blocks.append(LDBlock(
                region=region,
                n_variants=end_local - start_local + 1,
                variant_indices=indices,
                method="dp_optimize",
                mean_r2=mean_r2,
                mean_dprime=0.0,
                metadata={"objective": objective},
            ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))


def _block_cost(
    G_block: Tensor,
    R_block: Tensor,
    *,
    objective: str,
    hap_freq_threshold: float,
    tag_r2_threshold: float,
) -> float:
    """Compute the cost of a candidate block under the chosen objective.

    Parameters
    ----------
    G_block : (n, w) genotype submatrix for the block.
    R_block : (w, w) r² submatrix.
    objective : "haplotype_diversity" or "tag_snp".
    """
    n, w = G_block.shape

    if objective == "haplotype_diversity":
        # Count distinct haplotype patterns above frequency threshold
        # Round dosages to integer genotypes for pattern counting
        patterns = G_block.round().long().clamp(0, 2)
        # Hash each row as a tuple
        unique_count = 0
        seen: set[tuple[int, ...]] = set()
        for row_idx in range(n):
            pat = tuple(patterns[row_idx].cpu().tolist())
            seen.add(pat)
        # Filter by frequency
        from collections import Counter
        pattern_list = [
            tuple(patterns[i].cpu().tolist()) for i in range(n)
        ]
        counts = Counter(pattern_list)
        threshold_count = max(1, int(hap_freq_threshold * n))
        n_common = sum(1 for c in counts.values() if c >= threshold_count)
        return float(n_common)

    elif objective == "tag_snp":
        # Minimum tag SNPs to capture all variants at r² > threshold
        # Greedy set cover on the r² matrix
        captured = [False] * w
        n_tags = 0
        while not all(captured):
            # Find the SNP that captures the most uncaptured SNPs
            best_idx = -1
            best_count = -1
            for j in range(w):
                if captured[j]:
                    continue
                count = sum(
                    1 for k in range(w)
                    if not captured[k] and R_block[j, k].item() >= tag_r2_threshold
                )
                if count > best_count:
                    best_count = count
                    best_idx = j
            if best_idx < 0:
                break
            # Tag this SNP and all it captures
            for k in range(w):
                if R_block[best_idx, k].item() >= tag_r2_threshold:
                    captured[k] = True
            n_tags += 1
        return float(n_tags)

    else:
        raise ValueError(f"Unknown objective: {objective}")


# ── Connected-Component Graph Blocks ─────────────────────────────

def detect_blocks_cc_graph(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    r2_threshold: float = 0.5,
    min_block_snps: int = 1,
    window: int = 100,
    max_kb: float = 200.0,
    include_singletons: bool = True,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks via connected components on an r²-thresholded graph.

    Adapted from the graph-based LD pruning approach:

    1. Compute windowed pairwise r² between SNPs.
    2. Build an undirected graph: edge (i,j) iff r²(i,j) > threshold.
    3. Find connected components — each component defines an LD block
       (SNPs transitively linked by high r²).
    4. Per block: identify the tag SNP with highest MAF, stored in
       ``metadata["tag_snp"]``.
    5. Singletons (SNPs not in any high-r² pair) become single-SNP
       blocks if ``include_singletons=True``.

    This captures *transitive* LD relationships that contiguous-run
    methods miss: if A↔B and B↔C both exceed threshold, {A,B,C} form
    one block even if A↔C is below threshold.

    Parameters
    ----------
    r2_threshold : float
        Minimum r² to place an edge in the graph.
    min_block_snps : int
        Minimum component size to report as a block (default 1).
    window : int
        Maximum index distance for candidate pairs (limits O(m²) work).
    max_kb : float
        Maximum physical distance (kb) for candidate pairs.
    include_singletons : bool
        If True, SNPs not in any high-r² pair are reported as size-1
        blocks.  If False, only multi-SNP components are returned.
    """
    from ._graph_utils import connected_components

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []
    max_bp = max_kb * 1000.0

    # Precompute MAF for tag-SNP selection
    freq = G.mean(dim=0) / 2.0
    maf = torch.min(freq, 1.0 - freq)

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        m_chr = len(chr_mask)
        if m_chr == 0:
            continue

        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)
        G_chr = G[:, chr_idx]
        chr_pos = [variant_pos[i] for i in chr_mask]

        # Build windowed r² adjacency matrix.
        # Native C++ shortcut: pre-center G_chr in one vectorized op,
        # then call the C++ kernel which computes r² for every (i, j)
        # pair within the window without per-pair Python overhead.
        # The pure-Python double loop below remains the canonical
        # algorithmic reference and is exercised when the extension is
        # missing or TORCHGENOMICS_DISABLE_NATIVE=1 is set.
        if _cc_graph_native_enabled():
            import numpy as np
            G_centered = (G_chr - G_chr.mean(dim=0, keepdim=True))
            col_var = (G_centered * G_centered).mean(dim=0)
            G_centered_np = G_centered.detach().to(torch.float64).cpu().numpy()
            col_var_np = col_var.detach().to(torch.float64).cpu().numpy()
            chr_pos_np = np.asarray(chr_pos, dtype=np.int64)
            adj_np = _cc_graph_native.cc_graph_windowed_adj(
                G_centered_np, col_var_np, chr_pos_np,
                int(window), float(max_bp), float(r2_threshold),
            )
            adj = torch.from_numpy(adj_np).to(device=G.device, dtype=torch.float64)
        else:
            adj = torch.zeros(m_chr, m_chr, dtype=torch.float64, device=G.device)
            for i in range(m_chr):
                j_end = min(m_chr, i + window + 1)
                for j in range(i + 1, j_end):
                    if chr_pos[j] - chr_pos[i] > max_bp:
                        break
                    # r² between SNPs i and j
                    gi = G_chr[:, i]
                    gj = G_chr[:, j]
                    gi_c = gi - gi.mean()
                    gj_c = gj - gj.mean()
                    cov = (gi_c * gj_c).mean()
                    var_i = (gi_c * gi_c).mean()
                    var_j = (gj_c * gj_c).mean()
                    denom = var_i * var_j
                    if denom > _EPS:
                        r2 = (cov * cov / denom).item()
                    else:
                        r2 = 0.0
                    if r2 > r2_threshold:
                        adj[i, j] = r2
                        adj[j, i] = r2

        # Find connected components on the thresholded graph
        components = connected_components(adj, threshold=0.0)

        # Track which local indices are assigned to multi-SNP components
        assigned: set[int] = set()

        for comp in components:
            if len(comp) < max(min_block_snps, 2):
                continue

            assigned.update(comp)

            # Component may span non-contiguous indices; use min/max
            start_local = min(comp)
            end_local = max(comp)
            start_global = chr_mask[start_local]
            end_global = chr_mask[end_local]
            chr_val = variant_chr[start_global]

            # Variant indices: the actual component members (may be non-contiguous)
            global_indices = sorted(chr_mask[k] for k in comp)

            block_id = (
                f"{variant_ids[global_indices[0]]}-{variant_ids[global_indices[-1]]}"
                if variant_ids
                else f"{chr_val}:{variant_pos[global_indices[0]]}-{variant_pos[global_indices[-1]]}"
            )
            region = Region(
                region_id=block_id, chr=chr_val,
                start=variant_pos[global_indices[0]],
                end=variant_pos[global_indices[-1]] + 1,
            )

            # Mean r² within component (from adjacency, including zeros)
            comp_t = torch.tensor(comp, dtype=torch.long, device=G.device)
            R_sub = adj[comp_t][:, comp_t]
            n_pairs = len(comp) * (len(comp) - 1)
            mean_r2 = R_sub.sum().item() / n_pairs if n_pairs > 0 else 1.0

            # Tag SNP: component member with highest MAF
            maf_comp = maf[torch.tensor(global_indices, dtype=torch.long, device=G.device)]
            tag_local = maf_comp.argmax().item()
            tag_global = global_indices[tag_local]
            tag_id = variant_ids[tag_global] if variant_ids else str(tag_global)

            all_blocks.append(LDBlock(
                region=region,
                n_variants=len(comp),
                variant_indices=global_indices,
                method="cc_graph",
                mean_r2=mean_r2,
                mean_dprime=0.0,
                metadata={
                    "tag_snp": tag_id,
                    "tag_snp_index": tag_global,
                    "tag_snp_maf": maf_comp[tag_local].item(),
                    "component_contiguous": (end_local - start_local + 1 == len(comp)),
                },
            ))

        # Singletons: SNPs not in any multi-SNP component
        if include_singletons and min_block_snps <= 1:
            for k in range(m_chr):
                if k in assigned:
                    continue
                g_idx = chr_mask[k]
                chr_val = variant_chr[g_idx]
                sid = variant_ids[g_idx] if variant_ids else str(g_idx)
                region = Region(
                    region_id=sid, chr=chr_val,
                    start=variant_pos[g_idx],
                    end=variant_pos[g_idx] + 1,
                )
                all_blocks.append(LDBlock(
                    region=region,
                    n_variants=1,
                    variant_indices=[g_idx],
                    method="cc_graph",
                    mean_r2=1.0,
                    mean_dprime=0.0,
                    metadata={
                        "tag_snp": sid,
                        "tag_snp_index": g_idx,
                        "tag_snp_maf": maf[g_idx].item(),
                        "component_contiguous": True,
                    },
                ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))
