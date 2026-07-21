"""Haplotype block detection algorithms.

Implements four standard methods:
  - Gabriel et al. (2002): D' confidence interval classification
  - Four-gamete test (Wang et al. 2002): recombination breakpoint detection
  - Solid spine of LD: contiguous D' threshold from block anchors
  - r-squared threshold: simple correlation-based blocks

All methods operate on pre-computed pairwise LD statistics (CPU-side logic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import (
    HAS_NATIVE_GABRIEL,
    HAS_NATIVE_SPINE,
    _gabriel_native,
    _spine_native,
)
from ..io.regions import Region


def _gabriel_native_enabled() -> bool:
    """Whether the native C++ Gabriel accelerator should be used."""
    return HAS_NATIVE_GABRIEL and not native_disabled()


def _spine_native_enabled() -> bool:
    """Whether the native C++ spine accelerator should be used."""
    return HAS_NATIVE_SPINE and not native_disabled()


@dataclass
class PairwiseLD:
    """Pairwise LD statistics for SNP pairs."""

    idx_i: Tensor           # (P,) first SNP index (within-chromosome)
    idx_j: Tensor           # (P,) second SNP index
    r2: Tensor              # (P,)
    dprime: Tensor          # (P,)
    dprime_ci_low: Tensor   # (P,) 95% CI lower bound
    dprime_ci_high: Tensor  # (P,) 95% CI upper bound


@dataclass
class LDBlock:
    """A detected haplotype block."""

    region: Region
    n_variants: int
    variant_indices: list[int]
    method: str
    mean_r2: float
    mean_dprime: float
    # Optional fields for advanced methods (backward-compatible)
    blockiness_score: float | None = None
    uncertainty_metric: float | None = None
    population_stability: float | None = None
    concentration_ratio: float | None = None
    changepoint_confidence: float | None = None
    metadata: dict | None = None


# ── Gabriel et al. (2002) ───────────────────────────────────────────

def detect_blocks_gabriel(
    pld: PairwiseLD,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    ci_low: float = 0.70,
    ci_high: float = 0.98,
    rec_high: float = 0.90,
    strong_pct: float = 0.95,
    rec_max_pct: float = 0.04,
    min_block_snps: int = 2,
) -> list[LDBlock]:
    """Detect haplotype blocks using Gabriel et al. (2002) criteria.

    A pair is classified as:
      - "strong LD" if CI_lower > ci_low AND CI_upper > ci_high
      - "strong recombination" if CI_upper < rec_high
      - otherwise "uninformative"

    A block [i..j] is accepted if among informative pairs:
      - fraction of strong LD >= strong_pct (default 95%)
      - fraction of strong recombination < rec_max_pct (default 4%)
    """
    n_snps = len(variant_pos)
    if n_snps < 2:
        return []

    # Classify pairs
    strong_ld = (pld.dprime_ci_low > ci_low) & (pld.dprime_ci_high > ci_high)
    strong_rec = pld.dprime_ci_high < rec_high
    informative = strong_ld | strong_rec

    # Native C++ shortcut. The pure-Python body below remains the
    # canonical algorithmic reference and is exercised when the
    # extension is missing or TORCHGENOMICS_DISABLE_NATIVE=1 is set.
    if _gabriel_native_enabled() and pld.idx_i.device.type == "cpu":
        idx_i_np = pld.idx_i.detach().cpu().numpy().astype(np.int64, copy=False)
        idx_j_np = pld.idx_j.detach().cpu().numpy().astype(np.int64, copy=False)
        sld_np = strong_ld.detach().cpu().numpy().astype(np.uint8, copy=False)
        srec_np = strong_rec.detach().cpu().numpy().astype(np.uint8, copy=False)
        inf_np = informative.detach().cpu().numpy().astype(np.uint8, copy=False)
        r2_np = pld.r2.detach().to(torch.float64).cpu().numpy()
        dp_np = pld.dprime.detach().to(torch.float64).cpu().numpy()
        accepted = _gabriel_native.gabriel_blocks(
            int(n_snps),
            idx_i_np, idx_j_np,
            sld_np, srec_np, inf_np,
            r2_np, dp_np,
            float(strong_pct), float(rec_max_pct),
            int(min_block_snps),
        )
        blocks: list[LDBlock] = []
        for start, end, mean_r2, mean_dp in accepted:
            chr_val = variant_chr[start]
            block_id = (
                f"{chr_val}:{variant_pos[start]}-{variant_pos[end]}"
                if variant_ids is None
                else f"{variant_ids[start]}-{variant_ids[end]}"
            )
            region = Region(
                region_id=block_id,
                chr=chr_val,
                start=variant_pos[start],
                end=variant_pos[end] + 1,
            )
            blocks.append(LDBlock(
                region=region,
                n_variants=end - start + 1,
                variant_indices=list(range(start, end + 1)),
                method="gabriel",
                mean_r2=float(mean_r2),
                mean_dprime=float(mean_dp),
            ))
        blocks.sort(key=lambda b: (b.region.chr, b.region.start))
        return blocks

    # Build lookup: for each pair (i, j), store classification
    # Use a dict keyed by (i, j) for O(1) lookup
    idx_i = pld.idx_i.cpu().tolist()
    idx_j = pld.idx_j.cpu().tolist()
    sld = strong_ld.cpu().tolist()
    srec = strong_rec.cpu().tolist()
    inf = informative.cpu().tolist()
    r2_vals = pld.r2.cpu().tolist()
    dp_vals = pld.dprime.cpu().tolist()

    pair_sld = {}
    pair_srec = {}
    pair_inf = {}
    pair_r2 = {}
    pair_dp = {}
    for k in range(len(idx_i)):
        key = (idx_i[k], idx_j[k])
        pair_sld[key] = sld[k]
        pair_srec[key] = srec[k]
        pair_inf[key] = inf[k]
        pair_r2[key] = r2_vals[k]
        pair_dp[key] = dp_vals[k]

    # Find blocks: greedy, longest first
    # Try all candidate blocks [start, end], sorted by length descending
    used = [False] * n_snps
    blocks = []

    # Generate candidates sorted by decreasing length
    candidates = []
    for start in range(n_snps):
        for end in range(start + min_block_snps - 1, n_snps):
            candidates.append((end - start, start, end))
    candidates.sort(reverse=True)

    for length, start, end in candidates:
        # Skip if any SNP already in a block
        if any(used[k] for k in range(start, end + 1)):
            continue

        # Count informative, strong LD, and strong rec pairs
        n_info = 0
        n_sld = 0
        n_srec = 0
        r2_sum = 0.0
        dp_sum = 0.0
        n_pairs = 0

        for i in range(start, end + 1):
            for j in range(i + 1, end + 1):
                key = (i, j)
                if key not in pair_inf:
                    continue
                n_pairs += 1
                r2_sum += pair_r2.get(key, 0.0)
                dp_sum += pair_dp.get(key, 0.0)
                if pair_inf[key]:
                    n_info += 1
                    if pair_sld[key]:
                        n_sld += 1
                    if pair_srec[key]:
                        n_srec += 1

        if n_info == 0:
            continue

        frac_sld = n_sld / n_info
        frac_srec = n_srec / n_info

        if frac_sld >= strong_pct and frac_srec < rec_max_pct:
            for k in range(start, end + 1):
                used[k] = True

            chr_val = variant_chr[start]
            block_id = (
                f"{chr_val}:{variant_pos[start]}-{variant_pos[end]}"
                if variant_ids is None
                else f"{variant_ids[start]}-{variant_ids[end]}"
            )
            region = Region(
                region_id=block_id,
                chr=chr_val,
                start=variant_pos[start],
                end=variant_pos[end] + 1,  # BED half-open
            )
            indices = list(range(start, end + 1))
            mean_r2 = r2_sum / max(n_pairs, 1)
            mean_dp = dp_sum / max(n_pairs, 1)

            blocks.append(LDBlock(
                region=region,
                n_variants=end - start + 1,
                variant_indices=indices,
                method="gabriel",
                mean_r2=mean_r2,
                mean_dprime=mean_dp,
            ))

    # Sort by position
    blocks.sort(key=lambda b: (b.region.chr, b.region.start))
    return blocks


# ── Four-Gamete Test ────────────────────────────────────────────────

def detect_blocks_four_gamete(
    haplotypes: Tensor,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    freq_threshold: float = 0.01,
    min_block_snps: int = 2,
) -> list[LDBlock]:
    """Detect blocks using the four-gamete test (Wang et al. 2002).

    For each adjacent pair, checks if all four gametic types exist
    above ``freq_threshold``.  If so, a recombination breakpoint is
    inferred.  Blocks are maximal segments between breakpoints.

    Parameters
    ----------
    haplotypes : (n, ploidy, m) Tensor
        Phased binary haplotype data.
    """
    n, ploidy, m = haplotypes.shape
    if m < 2:
        return []

    H = haplotypes.reshape(n * ploidy, m).to(torch.float64)
    N = H.shape[0]

    # Test all adjacent pairs simultaneously
    h_i = H[:, :-1]  # (N, m-1)
    h_j = H[:, 1:]   # (N, m-1)

    n_00 = ((1 - h_i) * (1 - h_j)).sum(dim=0)
    n_01 = ((1 - h_i) * h_j).sum(dim=0)
    n_10 = (h_i * (1 - h_j)).sum(dim=0)
    n_11 = (h_i * h_j).sum(dim=0)

    # All four gametes present above threshold
    threshold_count = freq_threshold * N
    breakpoint = (
        (n_00 > threshold_count)
        & (n_01 > threshold_count)
        & (n_10 > threshold_count)
        & (n_11 > threshold_count)
    )  # (m-1,) bool — True = recombination between SNP i and i+1

    breakpoints = breakpoint.cpu().tolist()

    # Build blocks as maximal segments between breakpoints
    blocks = []
    block_start = 0

    for i, is_break in enumerate(breakpoints):
        if is_break:
            # Block is [block_start .. i]
            if i - block_start + 1 >= min_block_snps:
                blocks.append(_make_block(
                    block_start, i, variant_pos, variant_chr, variant_ids,
                    method="four_gamete",
                ))
            block_start = i + 1

    # Last block
    if m - 1 - block_start + 1 >= min_block_snps:
        blocks.append(_make_block(
            block_start, m - 1, variant_pos, variant_chr, variant_ids,
            method="four_gamete",
        ))

    return blocks


# ── Solid Spine of LD ───────────────────────────────────────────────

def detect_blocks_spine(
    pld: PairwiseLD,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    d_prime_threshold: float = 0.8,
    min_block_snps: int = 2,
    n_snps: int = 0,
) -> list[LDBlock]:
    """Detect blocks using the solid spine of LD method.

    A block [start..end] is valid if D'(start, k) >= threshold for
    all k in [start+1..end] AND D'(k, end) >= threshold for all
    k in [start..end-1].
    """
    if n_snps == 0:
        n_snps = len(variant_pos)
    if n_snps < 2:
        return []

    blocks = []

    if _spine_native_enabled() and pld.idx_i.device.type == "cpu":
        # Native path: build dense |D'|, r², and presence matrices directly
        # from the sparse pair tensors via vectorized numpy scatter. This
        # skips the Python dict intermediate that dominated wall time at
        # small m on the previous implementation (bench showed 0.7× — native
        # losing to Python because the dict→dense rebuild ate the C++ win).
        idx_i_np = pld.idx_i.cpu().numpy().astype(np.int64)
        idx_j_np = pld.idx_j.cpu().numpy().astype(np.int64)
        dp_np = pld.dprime.abs().cpu().numpy().astype(np.float64)
        r2_np = pld.r2.cpu().numpy().astype(np.float64)
        dp_dense = np.zeros((n_snps, n_snps), dtype=np.float64)
        r2_dense = np.zeros((n_snps, n_snps), dtype=np.float64)
        present = np.zeros((n_snps, n_snps), dtype=bool)
        dp_dense[idx_i_np, idx_j_np] = dp_np
        r2_dense[idx_i_np, idx_j_np] = r2_np
        present[idx_i_np, idx_j_np] = True
        intervals = _spine_native.spine_partition(
            dp_dense, int(n_snps),
            float(d_prime_threshold), int(min_block_snps),
        )
        for s, e in intervals:
            blocks.append(_make_block_from_dense(
                int(s), int(e), variant_pos, variant_chr, variant_ids,
                method="spine",
                dp_dense=dp_dense, r2_dense=r2_dense, present=present,
            ))
        return blocks

    # Pure-Python path: build D' / r² dict lookups for the greedy loop.
    idx_i = pld.idx_i.cpu().tolist()
    idx_j = pld.idx_j.cpu().tolist()
    dp_vals = pld.dprime.abs().cpu().tolist()
    r2_vals = pld.r2.cpu().tolist()

    dp_map = {}
    r2_map = {}
    for k in range(len(idx_i)):
        key = (idx_i[k], idx_j[k])
        dp_map[key] = dp_vals[k]
        r2_map[key] = r2_vals[k]

    used = [False] * n_snps

    # Greedy: try extending from each start position
    start = 0
    while start < n_snps:
        if used[start]:
            start += 1
            continue

        end = start
        # Extend rightward while spine conditions hold
        while end + 1 < n_snps:
            candidate = end + 1
            # Check D'(start, candidate) >= threshold
            key_start = (start, candidate)
            if key_start not in dp_map or dp_map[key_start] < d_prime_threshold:
                break
            # Check D'(candidate, end) for existing block — actually check
            # D'(k, candidate) for k=start..end (spine from candidate back)
            # Simplified: check D'(start, candidate) AND D'(end, candidate)
            # The full spine condition checks ALL pairs to first and last.
            # For efficiency, check the anchor conditions:
            ok = True
            for k in range(start, candidate):
                key_k = (k, candidate)
                if key_k not in dp_map or dp_map[key_k] < d_prime_threshold:
                    ok = False
                    break
            if not ok:
                break
            end = candidate

        if end - start + 1 >= min_block_snps:
            for k in range(start, end + 1):
                used[k] = True
            blocks.append(_make_block(
                start, end, variant_pos, variant_chr, variant_ids,
                method="spine",
                dp_map=dp_map, r2_map=r2_map,
            ))
        start = end + 1

    return blocks


# ── r-squared Threshold ─────────────────────────────────────────────

def detect_blocks_r2(
    pld: PairwiseLD,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    r2_threshold: float = 0.2,
    min_block_snps: int = 2,
    n_snps: int = 0,
    tolerance: int = 0,
) -> list[LDBlock]:
    """Detect blocks using an r-squared threshold on adjacent pairs.

    A block boundary is placed when the adjacent-pair r² between
    successive SNPs drops below ``r2_threshold``. By default
    (``tolerance=0``) the block closes on the very first below-threshold
    pair (legacy behavior). When ``tolerance>0`` the walker tolerates up
    to ``tolerance`` consecutive below-threshold adjacent pairs before
    closing the block, and the block is closed only after
    ``tolerance + 1`` consecutive failures. A subsequent pair that meets
    or exceeds ``r2_threshold`` resets the failure counter.

    This matches the SelectionTools R package
    (https://github.com/PBGLMichaelHall/SelectionTools) ``hbd.bfile``
    haplo-block detector, which the plant-breeding community uses with
    ``r2=0.7`` and ``t=2`` (e.g., Pandit et al., *Theoretical and
    Applied Genetics* 2026, barley leaf rust). The tolerance is the
    parameter that lets adjacent low-r² gaps inside an otherwise
    high-LD region — common with imputed dosages or genotyping artifacts
    — be absorbed into the surrounding block rather than splitting it.

    Parameters
    ----------
    pld : PairwiseLD
        Pre-computed pairwise LD statistics.
    variant_pos : list[int]
        Base-pair positions (genome-wide, per chromosome).
    variant_chr : list[str]
        Chromosome label per variant.
    variant_ids : list[str], optional
        Variant IDs for block naming.
    r2_threshold : float
        Minimum r² for adjacent SNPs to stay in the same block.
    min_block_snps : int
        Minimum block size in SNPs to emit a block.
    n_snps : int
        Number of variants (0 = infer from ``variant_pos``).
    tolerance : int, default 0
        Number of consecutive below-threshold adjacent pairs to absorb
        before closing a block. ``tolerance=0`` reproduces the strict
        legacy behavior. ``tolerance=2`` is the SelectionTools default.

    Returns
    -------
    list[LDBlock]
    """
    if n_snps == 0:
        n_snps = len(variant_pos)
    if n_snps < 2:
        return []
    if tolerance < 0:
        raise ValueError(
            f"tolerance must be non-negative; got {tolerance}."
        )

    # Build r2 lookup for adjacent pairs
    idx_i = pld.idx_i.cpu().tolist()
    idx_j = pld.idx_j.cpu().tolist()
    r2_vals = pld.r2.cpu().tolist()
    dp_vals = pld.dprime.cpu().tolist()

    r2_map = {}
    dp_map = {}
    for k in range(len(idx_i)):
        key = (idx_i[k], idx_j[k])
        r2_map[key] = r2_vals[k]
        dp_map[key] = dp_vals[k]

    # Find breakpoints: adjacent pairs with r2 below threshold.
    # ``consec_fail`` is the running count of consecutive below-threshold
    # adjacent pairs; the block closes only when this count exceeds
    # ``tolerance`` (i.e. after tolerance + 1 consecutive failures).
    blocks = []
    block_start = 0
    consec_fail = 0

    for i in range(n_snps - 1):
        key = (i, i + 1)
        adj_r2 = r2_map.get(key, 0.0)
        if adj_r2 < r2_threshold:
            consec_fail += 1
            if consec_fail > tolerance:
                # The terminating failure run spans adjacent pairs
                # ``[i - consec_fail + 1, ..., i]``. The block ends at
                # the LEFT endpoint of the first failing pair, which is
                # SNP index ``i - consec_fail + 1``. The new block
                # starts at the RIGHT endpoint of the last failing pair,
                # which is SNP ``i + 1``. SNPs strictly between
                # ``i - consec_fail + 2`` and ``i`` are interior to the
                # gap and belong to neither block. For ``tolerance=0``
                # (``consec_fail = 1``) this reduces to the legacy
                # behavior: ``block_end = i`` and ``block_start = i+1``.
                block_end = i - consec_fail + 1
                if block_end >= block_start and (
                    block_end - block_start + 1 >= min_block_snps
                ):
                    blocks.append(_make_block(
                        block_start, block_end,
                        variant_pos, variant_chr, variant_ids,
                        method="r2", dp_map=dp_map, r2_map=r2_map,
                    ))
                block_start = i + 1
                consec_fail = 0
        else:
            # A passing pair resets the failure counter — the absorbed
            # gap is now bridged on both sides by ≥ threshold pairs.
            consec_fail = 0

    # Last segment. If the chromosome ended mid-failure-run, trim those
    # trailing failures off the block before emitting.
    last_end = n_snps - 1 - consec_fail
    if last_end >= block_start and (
        last_end - block_start + 1 >= min_block_snps
    ):
        blocks.append(_make_block(
            block_start, last_end,
            variant_pos, variant_chr, variant_ids,
            method="r2", dp_map=dp_map, r2_map=r2_map,
        ))

    return blocks


# ── Helpers ─────────────────────────────────────────────────────────

def _make_block(
    start: int,
    end: int,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None,
    method: str,
    dp_map: dict | None = None,
    r2_map: dict | None = None,
) -> LDBlock:
    """Create an LDBlock from start/end indices."""
    chr_val = variant_chr[start]
    block_id = (
        f"{chr_val}:{variant_pos[start]}-{variant_pos[end]}"
        if variant_ids is None
        else f"{variant_ids[start]}-{variant_ids[end]}"
    )
    region = Region(
        region_id=block_id,
        chr=chr_val,
        start=variant_pos[start],
        end=variant_pos[end] + 1,
    )
    indices = list(range(start, end + 1))

    # Compute mean LD within block
    mean_r2 = 0.0
    mean_dp = 0.0
    n_pairs = 0
    if r2_map and dp_map:
        for i in range(start, end + 1):
            for j in range(i + 1, end + 1):
                key = (i, j)
                if key in r2_map:
                    mean_r2 += r2_map[key]
                    mean_dp += abs(dp_map.get(key, 0.0))
                    n_pairs += 1
    if n_pairs > 0:
        mean_r2 /= n_pairs
        mean_dp /= n_pairs

    return LDBlock(
        region=region,
        n_variants=end - start + 1,
        variant_indices=indices,
        method=method,
        mean_r2=mean_r2,
        mean_dprime=mean_dp,
    )


def _make_block_from_dense(
    start: int,
    end: int,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None,
    method: str,
    dp_dense: np.ndarray,
    r2_dense: np.ndarray,
    present: np.ndarray,
) -> LDBlock:
    """Create an LDBlock from dense |D'|/r²/presence matrices.

    Equivalent to ``_make_block`` but reads block statistics from the
    dense numpy matrices instead of a per-pair Python dict. Only pairs
    marked in ``present`` contribute to the means, matching the
    ``if key in r2_map`` branch of ``_make_block`` exactly.
    """
    chr_val = variant_chr[start]
    block_id = (
        f"{chr_val}:{variant_pos[start]}-{variant_pos[end]}"
        if variant_ids is None
        else f"{variant_ids[start]}-{variant_ids[end]}"
    )
    region = Region(
        region_id=block_id,
        chr=chr_val,
        start=variant_pos[start],
        end=variant_pos[end] + 1,
    )
    indices = list(range(start, end + 1))

    b = end - start + 1
    mean_r2 = 0.0
    mean_dp = 0.0
    if b >= 2:
        iu = np.triu_indices(b, k=1)
        sub_present = present[start:end + 1, start:end + 1][iu]
        n_pairs = int(sub_present.sum())
        if n_pairs > 0:
            sub_r2 = r2_dense[start:end + 1, start:end + 1][iu][sub_present]
            sub_dp = dp_dense[start:end + 1, start:end + 1][iu][sub_present]
            mean_r2 = float(sub_r2.mean())
            mean_dp = float(sub_dp.mean())

    return LDBlock(
        region=region,
        n_variants=b,
        variant_indices=indices,
        method=method,
        mean_r2=mean_r2,
        mean_dprime=mean_dp,
    )
