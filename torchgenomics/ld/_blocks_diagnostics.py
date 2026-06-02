"""Wall & Pritchard blockiness diagnostics.

Quantifies how block-like a genomic region is, using statistics from
Wall (2003) and Pritchard & Przeworski (2001).  Useful for model
criticism: deciding whether haplotype-block assumptions are appropriate
for a given region.
"""

from __future__ import annotations

import os

import torch

from .._native import HAS_NATIVE_WALL_PRITCHARD, _wall_pritchard_native
from ..io.regions import Region
from ._blocks import LDBlock, PairwiseLD

_EPS = 1e-10


def _wall_pritchard_native_enabled() -> bool:
    """Whether the native C++ Wall-Pritchard permutation accelerator should be used."""
    return HAS_NATIVE_WALL_PRITCHARD and not os.environ.get(
        "TORCHGENOMICS_DISABLE_NATIVE"
    )


def compute_wall_pritchard_diagnostics(
    pld: PairwiseLD,
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None = None,
    *,
    d_prime_threshold: float = 0.7,
    n_permutations: int = 0,
) -> list[LDBlock]:
    """Compute Wall & Pritchard blockiness diagnostics.

    Returns a single ``LDBlock`` spanning the entire region with the
    ``blockiness_score`` field populated.

    Statistics computed:
      - **Q** (Wall's Q): fraction of pairs with |D'| > ``d_prime_threshold``.
      - **Q_adj**: ratio of adjacent-pair LD to all-pair LD.  In true
        block structure, even distant within-block pairs have high LD,
        so Q_adj ≈ 1.  In decay-like LD, Q_adj << 1.
      - **blockiness_score**: ``Q * Q_adj`` — high when the region
        exhibits strong, uniform LD consistent with block structure.

    Parameters
    ----------
    pld : PairwiseLD
        Pre-computed pairwise LD statistics.
    d_prime_threshold : float
        Threshold for "significant" |D'| in Wall's Q statistic.
    n_permutations : int
        If > 0, compute a permutation p-value for the blockiness score
        (stored in ``metadata["perm_pvalue"]``).

    Returns
    -------
    list[LDBlock]
        Single-element list with the diagnostic block.
    """
    n_snps = len(variant_pos)
    if n_snps < 2 or pld.r2.shape[0] == 0:
        region = _make_diagnostic_region(variant_pos, variant_chr, variant_ids)
        return [LDBlock(
            region=region,
            n_variants=n_snps,
            variant_indices=list(range(n_snps)),
            method="wall_pritchard",
            mean_r2=0.0,
            mean_dprime=0.0,
            blockiness_score=0.0,
            metadata={"Q": 0.0, "Q_adj": 0.0},
        )]

    dp = pld.dprime.abs().cpu()
    r2 = pld.r2.cpu()
    idx_i = pld.idx_i.cpu()
    idx_j = pld.idx_j.cpu()

    n_pairs = dp.shape[0]

    # Wall's Q: fraction of pairs with |D'| > threshold
    Q = (dp > d_prime_threshold).float().mean().item()

    # Q_adj: ratio of adjacent-pair mean r² to all-pair mean r²
    is_adjacent = (idx_j - idx_i) == 1
    if is_adjacent.any():
        adj_mean_r2 = r2[is_adjacent].mean().item()
    else:
        adj_mean_r2 = 0.0
    all_mean_r2 = r2.mean().item() if n_pairs > 0 else _EPS
    Q_adj = min(adj_mean_r2 / max(all_mean_r2, _EPS), 1.0)

    blockiness = Q * Q_adj

    # Optional permutation test
    meta: dict = {"Q": Q, "Q_adj": Q_adj}
    if n_permutations > 0:
        # Native C++ shortcut. The pure-Python loop below remains the
        # canonical algorithmic reference and runs when the extension is
        # missing or TORCHGENOMICS_DISABLE_NATIVE=1 is set. The C++ path uses
        # std::mt19937_64 instead of torch.randperm so it is statistically
        # equivalent rather than bit-for-bit identical to the Python path.
        if _wall_pritchard_native_enabled():
            seed = int(torch.randint(0, 2**62 - 1, (1,)).item())
            dp_np = dp.detach().to(torch.float64).cpu().numpy()
            r2_np = r2.detach().to(torch.float64).cpu().numpy()
            is_adj_np = is_adjacent.detach().cpu().numpy().astype(bool)
            perm_count = _wall_pritchard_native.wall_pritchard_perm(
                dp_np, r2_np, is_adj_np,
                float(d_prime_threshold),
                float(blockiness),
                int(n_permutations),
                int(seed),
            )
        else:
            perm_count = 0
            for _ in range(n_permutations):
                perm = torch.randperm(n_pairs)
                dp_perm = dp[perm]
                Q_perm = (dp_perm > d_prime_threshold).float().mean().item()
                r2_perm = r2[perm]
                adj_mean_perm = r2_perm[is_adjacent].mean().item() if is_adjacent.any() else 0.0
                all_mean_perm = r2_perm.mean().item() if n_pairs > 0 else _EPS
                Q_adj_perm = min(adj_mean_perm / max(all_mean_perm, _EPS), 1.0)
                if Q_perm * Q_adj_perm >= blockiness:
                    perm_count += 1
        meta["perm_pvalue"] = (perm_count + 1) / (n_permutations + 1)

    region = _make_diagnostic_region(variant_pos, variant_chr, variant_ids)
    mean_r2 = r2.mean().item() if n_pairs > 0 else 0.0
    mean_dp = dp.mean().item() if n_pairs > 0 else 0.0

    return [LDBlock(
        region=region,
        n_variants=n_snps,
        variant_indices=list(range(n_snps)),
        method="wall_pritchard",
        mean_r2=mean_r2,
        mean_dprime=mean_dp,
        blockiness_score=blockiness,
        metadata=meta,
    )]


def _make_diagnostic_region(
    variant_pos: list[int],
    variant_chr: list[str],
    variant_ids: list[str] | None,
) -> Region:
    """Create a Region spanning all variants."""
    chr_val = variant_chr[0] if variant_chr else "unknown"
    start = variant_pos[0] if variant_pos else 0
    end = variant_pos[-1] + 1 if variant_pos else 1
    if variant_ids:
        region_id = f"{variant_ids[0]}-{variant_ids[-1]}"
    else:
        region_id = f"{chr_val}:{start}-{end}"
    return Region(region_id=region_id, chr=chr_val, start=start, end=end)
