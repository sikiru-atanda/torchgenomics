"""Novel scientific block detection methods.

Five methods that go beyond classical PLINK-style block detection:
  1. GWAS-aligned block discovery (DP + concentration ratio)
  2. Uncertainty-aware LD blocks (corrected r² from GP/DS)
  3. Cross-population consensus blocks (multi-graph spectral)
  4. Conditional-independence blocks (graphical Lasso)
  5. Recombination / change-point blocks (DP segmentation)
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import (
    HAS_NATIVE_GWAS_ALIGNED,
    HAS_NATIVE_UNCERTAINTY_BLOCKS,
    _gwas_aligned_native,
    _uncertainty_blocks_native,
)
from ..io.regions import Region
from ._blocks import LDBlock
from ._pairwise import compute_r2_matrix, compute_r2_pairs

_EPS = 1e-10


def _gwas_aligned_native_enabled() -> bool:
    return HAS_NATIVE_GWAS_ALIGNED and not native_disabled()


def _uncertainty_blocks_native_enabled() -> bool:
    return HAS_NATIVE_UNCERTAINTY_BLOCKS and not os.environ.get(
        "TORCHGENOMICS_DISABLE_NATIVE"
    )


# ── 1. GWAS-Aligned Block Discovery ───────────────────────────────

def detect_blocks_gwas_aligned(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    max_block_snps: int = 50,
    condition_penalty: float = 0.1,
    min_block_snps: int = 2,
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Partition variants to maximize within-block genotypic concentration.

    For each candidate block [i..j], computes:
      - **concentration**: ratio of leading eigenvalue to trace of the
        correlation submatrix (fraction of variance in first PC).
      - **penalty**: condition number of the submatrix (high = multicollinear).

    Uses DP over ordered variants to find the optimal partition.

    Parameters
    ----------
    G : (n, m) genotype dosage matrix.
    max_block_snps : int
        Maximum number of SNPs per block.
    condition_penalty : float
        Weight for the condition-number penalty term.
    min_block_snps : int
        Minimum block size.
    max_kb : float
        Maximum physical span per block (kilobases).
    """
    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    if m < min_block_snps:
        return []

    max_bp = max_kb * 1000.0

    # Process per chromosome
    all_blocks: list[LDBlock] = []
    chr_set = sorted(set(variant_chr))

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        chr_idx = torch.tensor(chr_mask, dtype=torch.long)
        G_chr = G[:, chr_idx]
        m_chr = len(chr_mask)
        chr_pos = [variant_pos[i] for i in chr_mask]

        # Compute full r² matrix for this chromosome
        R = compute_r2_matrix(G_chr)  # (m_chr, m_chr)

        if _gwas_aligned_native_enabled():
            R_np = R.detach().to(torch.float64).cpu().numpy()
            chr_pos_np = np.asarray(chr_pos, dtype=np.int64)
            blocks_indices = [
                (int(s), int(e)) for (s, e) in
                _gwas_aligned_native.gwas_aligned_dp(
                    R_np, chr_pos_np,
                    int(max_block_snps), float(condition_penalty),
                    int(min_block_snps), float(max_bp),
                )
            ]
        else:
            # Precompute block scores for all candidate blocks
            # score(i,j) = concentration - condition_penalty * log(cond_number)
            # Use DP: opt[j] = max over i of { opt[i] + score(i+1, j) }

            # opt[i] = best score for partitioning chr_mask[0:i]
            opt = torch.full((m_chr + 1,), -float("inf"), dtype=torch.float64)
            opt[0] = 0.0
            backtrack = [0] * (m_chr + 1)

            for j in range(min_block_snps, m_chr + 1):
                for i in range(max(0, j - max_block_snps), j - min_block_snps + 1):
                    # Check physical distance constraint
                    if chr_pos[j - 1] - chr_pos[i] > max_bp:
                        continue
                    if opt[i] == -float("inf"):
                        continue

                    # Extract submatrix R[i:j, i:j]
                    R_sub = R[i:j, i:j]
                    block_size = j - i

                    # Concentration: leading eigenvalue / trace
                    eigenvalues = torch.linalg.eigvalsh(R_sub)
                    trace_val = eigenvalues.sum().clamp(min=_EPS)
                    leading = eigenvalues[-1]
                    concentration = (leading / trace_val).item()

                    # Condition number penalty
                    min_eig = eigenvalues[0].clamp(min=_EPS)
                    cond_num = (leading / min_eig).clamp(min=1.0)
                    penalty = condition_penalty * cond_num.log().item()

                    score = opt[i].item() + concentration - penalty
                    if score > opt[j].item():
                        opt[j] = score
                        backtrack[j] = i

            # Backtrack to recover partition
            # Find the rightmost position with a finite optimal value
            blocks_indices: list[tuple[int, int]] = []
            pos = m_chr
            while pos > 0 and opt[pos] == -float("inf"):
                pos -= 1
            while pos > 0:
                start = backtrack[pos]
                if pos - start >= min_block_snps:
                    blocks_indices.append((start, pos - 1))
                pos = start

            blocks_indices.reverse()

        # Build LDBlock objects
        for start_local, end_local in blocks_indices:
            start_global = chr_mask[start_local]
            end_global = chr_mask[end_local]

            # Compute concentration ratio for the block
            R_sub = R[start_local:end_local + 1, start_local:end_local + 1]
            eigenvalues = torch.linalg.eigvalsh(R_sub)
            trace_val = eigenvalues.sum().clamp(min=_EPS)
            concentration = (eigenvalues[-1] / trace_val).item()

            r2_mean = R_sub.mean().item()
            indices = list(range(start_global, end_global + 1))

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
            all_blocks.append(LDBlock(
                region=region,
                n_variants=end_local - start_local + 1,
                variant_indices=indices,
                method="gwas_aligned",
                mean_r2=r2_mean,
                mean_dprime=0.0,
                concentration_ratio=concentration,
            ))

    return all_blocks


# ── 2. Uncertainty-Aware LD Blocks ─────────────────────────────────

def detect_blocks_uncertainty(
    G: Tensor,
    gp_probs: Tensor,
    ploidy: int,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    base_method: str = "r2",
    min_dosage_rsq: float = 0.3,
    min_block_snps: int = 2,
    r2_threshold: float = 0.2,
    d_prime_threshold: float = 0.8,
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks using uncertainty-corrected LD statistics.

    Computes measurement-error corrected r² from genotype probabilities,
    then delegates to an existing block detection method.

    Parameters
    ----------
    G : (n, m) expected dosage matrix.
    gp_probs : (n, m, k+1) genotype probability tensor.
    ploidy : int
    base_method : str
        Which classical method to apply on corrected LD ("r2" or "spine").
    min_dosage_rsq : float
        Exclude markers with dosage R² below this threshold.
    """
    from ..preprocess.dosage_uncertainty import dosage_rsq
    from ._uncertainty_ld import compute_corrected_r2, compute_edge_weights

    if device:
        G = G.to(device=device, dtype=torch.float64)
        gp_probs = gp_probs.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)
        gp_probs = gp_probs.to(dtype=torch.float64)

    n, m = G.shape

    # Compute quality and filter markers
    q = dosage_rsq(gp_probs, ploidy)  # (m,)
    quality_ok = q >= min_dosage_rsq

    # Compute corrected r² matrix
    r2_corrected = compute_corrected_r2(G, gp_probs, ploidy)
    weights = compute_edge_weights(r2_corrected, gp_probs, ploidy)

    # Build PairwiseLD from weighted adjacency for the base method
    # Process per-chromosome
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []
    max_bp = max_kb * 1000.0

    # Native shortcut: do the per-chromosome adjacent-r2 break detection +
    # block-mean computation in C++. The pure-Python loop below remains the
    # canonical algorithmic reference and is used when the extension is
    # missing or TORCHGENOMICS_DISABLE_NATIVE=1 is set.
    if _uncertainty_blocks_native_enabled():
        weights_np = weights.detach().to(torch.float64).cpu().numpy()
        quality_np = q.detach().to(torch.float64).cpu().numpy()
        quality_ok_np = quality_ok.detach().cpu().numpy().astype(bool)
        for chrom in chr_set:
            chr_mask_list = [i for i, c in enumerate(variant_chr) if c == chrom]
            if len(chr_mask_list) < min_block_snps:
                continue
            chr_mask_np = np.asarray(chr_mask_list, dtype=np.int64)
            chr_pos_np = np.asarray(
                [variant_pos[i] for i in chr_mask_list], dtype=np.int64
            )
            block_tuples = _uncertainty_blocks_native.uncertainty_per_chrom_blocks(
                weights_np,
                quality_np,
                quality_ok_np,
                chr_mask_np,
                chr_pos_np,
                float(max_bp),
                float(r2_threshold),
                int(min_block_snps),
            )
            for start_local, end_local, mean_r2, q_block in block_tuples:
                start_global = chr_mask_list[start_local]
                end_global = chr_mask_list[end_local]
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
                all_blocks.append(LDBlock(
                    region=region,
                    n_variants=end_local - start_local + 1,
                    variant_indices=indices,
                    method="uncertainty",
                    mean_r2=float(mean_r2),
                    mean_dprime=0.0,
                    uncertainty_metric=float(q_block),
                ))
        return all_blocks

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        chr_pos = [variant_pos[i] for i in chr_mask]
        m_chr = len(chr_mask)

        # Build pairs within window, filtered by quality
        pair_i, pair_j, pair_r2 = [], [], []
        for li in range(m_chr):
            gi = chr_mask[li]
            if not quality_ok[gi]:
                continue
            for lj in range(li + 1, m_chr):
                gj = chr_mask[lj]
                if not quality_ok[gj]:
                    continue
                if chr_pos[lj] - chr_pos[li] > max_bp:
                    break
                pair_i.append(li)
                pair_j.append(lj)
                pair_r2.append(weights[gi, gj].item())

        if not pair_i:
            continue

        # Use r2-threshold method on corrected r²
        # Find adjacent-pair breakpoints
        block_start = 0
        for k in range(m_chr - 1):
            gi = chr_mask[k]
            gj = chr_mask[k + 1]
            adj_r2 = weights[gi, gj].item() if (quality_ok[gi] and quality_ok[gj]) else 0.0
            if adj_r2 < r2_threshold:
                if k - block_start + 1 >= min_block_snps:
                    block = _make_uncertainty_block(
                        block_start, k, chr_mask, variant_pos, variant_chr,
                        variant_ids, weights, q,
                    )
                    all_blocks.append(block)
                block_start = k + 1

        # Last segment
        if m_chr - 1 - block_start + 1 >= min_block_snps:
            block = _make_uncertainty_block(
                block_start, m_chr - 1, chr_mask, variant_pos, variant_chr,
                variant_ids, weights, q,
            )
            all_blocks.append(block)

    return all_blocks


def _make_uncertainty_block(
    start_local: int,
    end_local: int,
    chr_mask: list[int],
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None,
    weights: Tensor,
    quality: Tensor,
) -> LDBlock:
    """Create an LDBlock with uncertainty metrics."""
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

    # Mean corrected r² within block
    global_indices = [chr_mask[k] for k in range(start_local, end_local + 1)]
    n_pairs = 0
    r2_sum = 0.0
    for ii in range(len(global_indices)):
        for jj in range(ii + 1, len(global_indices)):
            r2_sum += weights[global_indices[ii], global_indices[jj]].item()
            n_pairs += 1
    mean_r2 = r2_sum / max(n_pairs, 1)

    # Mean quality within block
    q_block = quality[global_indices].mean().item()

    return LDBlock(
        region=region,
        n_variants=end_local - start_local + 1,
        variant_indices=indices,
        method="uncertainty",
        mean_r2=mean_r2,
        mean_dprime=0.0,
        uncertainty_metric=q_block,
    )


# ── 3. Cross-Population Consensus Blocks ──────────────────────────

def detect_blocks_cross_pop(
    G: Tensor,
    population_ids: list[str],
    population_weights: dict[str, float] | None,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    n_blocks_hint: int = 0,
    stability_threshold: float = 0.5,
    min_block_snps: int = 2,
    aggregation: str = "mean",
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks stable across multiple populations.

    Builds per-population r² graphs, aggregates into a consensus
    adjacency matrix, and applies spectral partitioning.

    Parameters
    ----------
    G : (n, m) genotype dosage matrix (all populations concatenated).
    population_ids : list[str]
        Per-sample population label (length n).
    population_weights : dict mapping pop_id -> weight (default: equal).
    n_blocks_hint : int
        Approximate number of desired blocks (0 = auto-detect via
        eigengap heuristic).
    stability_threshold : float
        Minimum stability score to retain a block.
    aggregation : str
        How to combine per-pop matrices: "mean", "min", "median".
    """
    from ._graph_utils import spectral_partition

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    if m < min_block_snps:
        return []

    # Split samples by population
    pops = sorted(set(population_ids))
    if population_weights is None:
        population_weights = {p: 1.0 / len(pops) for p in pops}

    # Process per chromosome
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        m_chr = len(chr_mask)
        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)

        # Compute per-population r² matrices
        pop_r2: dict[str, Tensor] = {}
        for pop in pops:
            pop_mask = [s for s in range(n) if population_ids[s] == pop]
            if len(pop_mask) < 2:
                continue
            pop_idx = torch.tensor(pop_mask, dtype=torch.long, device=G.device)
            G_pop = G[pop_idx][:, chr_idx]
            pop_r2[pop] = compute_r2_matrix(G_pop)  # (m_chr, m_chr)

        if not pop_r2:
            continue

        # Aggregate into consensus matrix
        if aggregation == "mean":
            total_weight = sum(population_weights.get(p, 1.0) for p in pop_r2)
            A = sum(
                population_weights.get(p, 1.0) * r2
                for p, r2 in pop_r2.items()
            ) / max(total_weight, _EPS)
        elif aggregation == "min":
            stack = torch.stack(list(pop_r2.values()))
            A = stack.min(dim=0).values
        elif aggregation == "median":
            stack = torch.stack(list(pop_r2.values()))
            A = stack.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Determine number of clusters
        if n_blocks_hint > 0:
            k = min(n_blocks_hint, m_chr)
        else:
            # Eigengap heuristic
            from ._graph_utils import normalized_laplacian
            L = normalized_laplacian(A)
            eigvals = torch.linalg.eigvalsh(L)
            gaps = eigvals[1:] - eigvals[:-1]
            k = max(2, (gaps[:m_chr // 2].argmax() + 2).item())
            k = min(k, m_chr // min_block_snps)

        clusters = spectral_partition(A, n_clusters=k, min_cluster_size=min_block_snps)

        # Native shortcut: collapses the per-cluster, per-pop, per-pair
        # ``r2[..].item()`` round-trips into a single C++ pass. The Python
        # body below remains as the algorithmic spec; both paths are
        # exercised in the test suite.
        from .._native import HAS_NATIVE_CROSS_POP, _cross_pop_native
        cluster_stabilities: list[float] | None = None
        if (
            HAS_NATIVE_CROSS_POP and not native_disabled()
            and G.device.type == "cpu" and len(clusters) > 0
        ):
            import numpy as _np
            pop_stack = _np.stack(
                [r2.detach().cpu().numpy().astype(_np.float64, copy=False)
                 for r2 in pop_r2.values()],
                axis=0,
            )
            starts = [0]
            flat: list[int] = []
            for cl in clusters:
                flat.extend(int(x) for x in cl)
                starts.append(len(flat))
            cs_np = _np.asarray(starts, dtype=_np.int64)
            ci_np = _np.asarray(flat, dtype=_np.int64)
            stab_np = _cross_pop_native.cross_pop_stability(
                pop_stack, cs_np, ci_np,
            )
            cluster_stabilities = stab_np.tolist()

        # Convert clusters to contiguous blocks and compute stability
        for ci_, cluster in enumerate(clusters):
            if len(cluster) < min_block_snps:
                continue

            # Make contiguous: use min..max range
            start_local = min(cluster)
            end_local = max(cluster)

            if cluster_stabilities is not None:
                stability = cluster_stabilities[ci_]
            else:
                # Compute stability: min/max of per-pop mean r² within cluster
                pop_means = []
                for pop, r2 in pop_r2.items():
                    r2_vals = []
                    for ii in range(len(cluster)):
                        for jj in range(ii + 1, len(cluster)):
                            r2_vals.append(r2[cluster[ii], cluster[jj]].item())
                    if r2_vals:
                        pop_means.append(sum(r2_vals) / len(r2_vals))
                if pop_means:
                    stability = min(pop_means) / max(max(pop_means), _EPS)
                else:
                    stability = 0.0

            if stability < stability_threshold:
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
            mean_r2 = A[start_local:end_local + 1, start_local:end_local + 1].mean().item()

            all_blocks.append(LDBlock(
                region=region,
                n_variants=end_local - start_local + 1,
                variant_indices=indices,
                method="cross_pop",
                mean_r2=mean_r2,
                mean_dprime=0.0,
                population_stability=stability,
            ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))


# ── 4. Conditional-Independence Blocks (Graphical Model) ──────────

def detect_blocks_graphical(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    l1_penalty: float = 0.1,
    window_size: int = 100,
    overlap: int = 20,
    min_block_snps: int = 2,
    method_variant: str = "glasso",
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks via sparse precision matrix estimation.

    Estimates a sparse precision matrix (inverse covariance) using the
    graphical Lasso, then finds communities in the resulting
    conditional-dependence graph.

    Parameters
    ----------
    l1_penalty : float
        L1 regularization strength for the graphical Lasso.
    window_size : int
        Number of variants per processing window.
    overlap : int
        Overlap between adjacent windows for merging.
    method_variant : str
        "glasso" (graphical Lasso) or "neighborhood" (Meinshausen-Bühlmann).
    """
    from ._graph_utils import connected_components

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    if m < min_block_snps:
        return []

    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < min_block_snps:
            continue

        m_chr = len(chr_mask)
        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)
        G_chr = G[:, chr_idx]

        # Process in overlapping windows
        # Accumulate a global adjacency from precision estimates
        adj = torch.zeros(m_chr, m_chr, dtype=torch.float64, device=G.device)

        for w_start in range(0, m_chr, window_size - overlap):
            w_end = min(w_start + window_size, m_chr)
            if w_end - w_start < min_block_snps:
                continue

            G_w = G_chr[:, w_start:w_end]
            w = w_end - w_start

            # Sample covariance
            G_c = G_w - G_w.mean(dim=0, keepdim=True)
            S = G_c.T @ G_c / (n - 1)
            S = S + _EPS * torch.eye(w, dtype=torch.float64, device=G.device)

            # Graphical Lasso
            Theta = _graphical_lasso(S, l1_penalty)

            # Extract conditional-dependence adjacency
            # Off-diagonal elements of Theta indicate conditional dependencies
            Theta_abs = Theta.abs()
            Theta_abs.fill_diagonal_(0.0)

            adj[w_start:w_end, w_start:w_end] = torch.max(
                adj[w_start:w_end, w_start:w_end], Theta_abs
            )

        # Find connected components in the precision graph
        components = connected_components(adj, threshold=_EPS)

        for comp in components:
            if len(comp) < min_block_snps:
                continue

            start_local = min(comp)
            end_local = max(comp)
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

            # Mean r² within block
            R_sub = compute_r2_matrix(G_chr[:, start_local:end_local + 1])
            mean_r2 = R_sub.mean().item()

            all_blocks.append(LDBlock(
                region=region,
                n_variants=end_local - start_local + 1,
                variant_indices=indices,
                method="graphical",
                mean_r2=mean_r2,
                mean_dprime=0.0,
                metadata={"l1_penalty": l1_penalty},
            ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))


def _graphical_lasso(
    S: Tensor,
    alpha: float,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> Tensor:
    """Graphical Lasso via proximal gradient descent.

    Solves: min_Theta { -log det(Theta) + tr(S @ Theta) + alpha * ||Theta||_1 }
    subject to Theta positive definite.

    Parameters
    ----------
    S : (p, p) sample covariance matrix.
    alpha : float  L1 penalty strength.

    Returns
    -------
    Theta : (p, p) sparse precision matrix estimate.
    """
    p = S.shape[0]
    device = S.device
    dtype = S.dtype

    # Initialize with diagonal of inverse variances
    Theta = torch.diag(1.0 / S.diag().clamp(min=_EPS))
    step_size = 1.0 / (torch.linalg.norm(S).item() + alpha + _EPS)

    for iteration in range(max_iter):
        Theta_old = Theta.clone()

        # Gradient: S - Theta^{-1}
        try:
            Sigma = torch.linalg.inv(Theta)
        except torch.linalg.LinAlgError:
            Sigma = torch.linalg.pinv(Theta)
        grad = S - Sigma

        # Proximal gradient step
        Theta = Theta - step_size * grad

        # Soft-thresholding (L1 proximal operator) on off-diagonal
        diag_vals = Theta.diag().clone()
        Theta = torch.sign(Theta) * torch.clamp(Theta.abs() - step_size * alpha, min=0.0)
        # Restore diagonal (no penalty on diagonal)
        Theta.fill_diagonal_(0.0)
        Theta = Theta + torch.diag(diag_vals.clamp(min=_EPS))

        # Symmetrize
        Theta = (Theta + Theta.T) / 2.0

        # Convergence check
        change = (Theta - Theta_old).abs().max().item()
        if change < tol:
            break

    return Theta


# ── 5. Recombination / Change-Point Blocks ────────────────────────

def detect_blocks_changepoint(
    G: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    penalty: float = 0.0,
    min_segment_size: int = 2,
    cost_model: str = "gaussian",
    use_recomb_map: bool = False,
    recomb_map_cm: list[float] | None = None,
    max_kb: float = 200.0,
    device: torch.device | None = None,
) -> list[LDBlock]:
    """Detect blocks via change-point analysis on LD decay.

    Computes a per-variant LD summary signal (mean r² to nearest
    neighbors), then finds optimal change-points using DP.

    Parameters
    ----------
    penalty : float
        Per-changepoint penalty (0 = auto-select via BIC: 2*log(m)).
    min_segment_size : int
        Minimum segment length.
    cost_model : str
        "gaussian" or "poisson" cost function.
    use_recomb_map : bool
        If True and ``recomb_map_cm`` is provided, use cM-based signal.
    """
    import math

    from ._changepoint import dp_changepoint, ld_decay_signal

    if device:
        G = G.to(device=device, dtype=torch.float64)
    else:
        G = G.to(dtype=torch.float64)

    n, m = G.shape
    chr_set = sorted(set(variant_chr))
    all_blocks: list[LDBlock] = []

    for chrom in chr_set:
        chr_mask = [i for i, c in enumerate(variant_chr) if c == chrom]
        if len(chr_mask) < 2 * min_segment_size:
            continue

        m_chr = len(chr_mask)
        chr_idx = torch.tensor(chr_mask, dtype=torch.long, device=G.device)
        G_chr = G[:, chr_idx]

        # Build the signal to segment
        if use_recomb_map and recomb_map_cm is not None:
            # Use inter-marker recombination rate as signal
            cm_vals = [recomb_map_cm[i] for i in chr_mask]
            recomb_rate = torch.zeros(m_chr, dtype=torch.float64)
            for k in range(1, m_chr):
                recomb_rate[k] = cm_vals[k] - cm_vals[k - 1]
            signal = recomb_rate
        else:
            # Compute LD decay signal: mean r² to neighbors
            # Build adjacent pairs
            idx_i = torch.arange(m_chr - 1, device=G.device)
            idx_j = torch.arange(1, m_chr, device=G.device)
            r2_pairs = compute_r2_pairs(G_chr, idx_i, idx_j)

            signal = ld_decay_signal(r2_pairs, idx_i, idx_j, m_chr)

        # Auto-select penalty if not provided
        eff_penalty = penalty if penalty > 0 else 2.0 * math.log(max(m_chr, 2))

        # Find change-points
        cps = dp_changepoint(
            signal, eff_penalty,
            min_seg=min_segment_size,
            cost_fn=cost_model,
        )

        # Convert change-points to blocks
        boundaries = [0] + cps + [m_chr]
        for seg_idx in range(len(boundaries) - 1):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1] - 1
            if seg_end - seg_start + 1 < min_segment_size:
                continue

            start_global = chr_mask[seg_start]
            end_global = chr_mask[seg_end]
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

            # Compute mean r² within segment
            G_seg = G_chr[:, seg_start:seg_end + 1]
            if G_seg.shape[1] >= 2:
                R_seg = compute_r2_matrix(G_seg)
                mean_r2 = R_seg.mean().item()
            else:
                mean_r2 = 1.0

            # Confidence: how strong is the change at boundaries
            confidence = None
            if seg_idx > 0 and seg_idx < len(boundaries) - 1:
                # Use signal difference at boundary
                left_mean = signal[seg_start:seg_end + 1].mean().item()
                if seg_end + 1 < m_chr:
                    right_val = signal[seg_end + 1].item()
                    confidence = abs(left_mean - right_val)

            all_blocks.append(LDBlock(
                region=region,
                n_variants=seg_end - seg_start + 1,
                variant_indices=indices,
                method="changepoint",
                mean_r2=mean_r2,
                mean_dprime=0.0,
                changepoint_confidence=confidence,
            ))

    return sorted(all_blocks, key=lambda b: (b.region.chr, b.region.start))
