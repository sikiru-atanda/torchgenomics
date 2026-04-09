"""DP-based change-point detection with PELT-style pruning.

Used by the ``changepoint`` block detection method to find boundaries
in LD decay profiles or recombination rate signals.

Algorithm: exact segmentation via dynamic programming with a penalty
per segment.  PELT (Pruned Exact Linear Time, Killick et al. 2012)
prunes candidate change-points that can no longer be optimal,
giving O(n) expected runtime for well-behaved signals.
"""

from __future__ import annotations

import math
import os

import torch
from torch import Tensor

from .._native import (
    HAS_NATIVE_LD_DECAY_SIGNAL,
    HAS_NATIVE_PELT,
    _ld_decay_signal_native,
    _pelt_native,
)


def _native_enabled() -> bool:
    """Whether the native C++ PELT accelerator should be used.

    Disabled when the extension was not built or when the user sets
    ``TORCHGWAS_DISABLE_NATIVE=1`` in the environment.
    """
    return HAS_NATIVE_PELT and not os.environ.get("TORCHGWAS_DISABLE_NATIVE")


def _ld_decay_signal_native_enabled() -> bool:
    """Whether the native C++ ld_decay_signal accelerator should be used."""
    return HAS_NATIVE_LD_DECAY_SIGNAL and not os.environ.get(
        "TORCHGWAS_DISABLE_NATIVE"
    )


def dp_changepoint(
    signal: Tensor,
    penalty: float,
    *,
    min_seg: int = 2,
    cost_fn: str = "gaussian",
) -> list[int]:
    """Find optimal change-points in a 1-D signal.

    Minimizes ``sum_segments cost(segment) + penalty * n_changepoints``
    using exact DP with PELT pruning.

    Parameters
    ----------
    signal : (m,) 1-D signal (e.g. mean r² per variant).
    penalty : float
        Per-changepoint penalty (BIC-style: ``2 * log(m)`` is a
        reasonable default).
    min_seg : int
        Minimum segment length (default 2).
    cost_fn : str
        Cost model: "gaussian" (sum of squared deviations from segment
        mean) or "poisson" (negative log-likelihood under Poisson).

    Returns
    -------
    list[int]
        Change-point indices (positions where a new segment starts).
        Does NOT include index 0 or m.
    """
    x = signal.cpu().double()
    n = x.shape[0]

    if n < 2 * min_seg:
        return []

    # Native C++ shortcut. The pure-Python body below remains the canonical
    # algorithmic reference and is exercised when the extension is missing
    # or TORCHGWAS_DISABLE_NATIVE=1 is set.
    if _native_enabled():
        x_np = x.numpy()
        return list(
            _pelt_native.pelt_dp_changepoint(
                x_np, float(penalty), int(min_seg), str(cost_fn)
            )
        )

    # Precompute cumulative sums for O(1) segment cost evaluation
    cumsum = torch.zeros(n + 1, dtype=torch.float64)
    cumsum[1:] = x.cumsum(dim=0)
    cumsumsq = torch.zeros(n + 1, dtype=torch.float64)
    cumsumsq[1:] = (x ** 2).cumsum(dim=0)

    def _segment_cost(start: int, end: int) -> float:
        """Cost of segment [start, end) under the chosen model."""
        length = end - start
        if length < 1:
            return 0.0
        seg_sum = (cumsum[end] - cumsum[start]).item()
        if cost_fn == "gaussian":
            seg_sumsq = (cumsumsq[end] - cumsumsq[start]).item()
            mean_val = seg_sum / length
            # Sum of squared deviations = sumsq - n*mean^2
            return seg_sumsq - length * mean_val ** 2
        elif cost_fn == "poisson":
            seg_mean = max(seg_sum / length, 1e-10)
            # -loglik = n*mean - sum*log(mean)  (up to constant)
            return length * seg_mean - seg_sum * math.log(seg_mean)
        else:
            raise ValueError(f"Unknown cost_fn: {cost_fn}")

    # DP arrays
    # opt[j] = minimum cost of segmenting signal[0:j]
    opt = [0.0] * (n + 1)
    # last_cp[j] = last change-point before j in the optimal segmentation
    last_cp = [0] * (n + 1)

    # PELT candidate set — initially just {0}
    candidates = {0}

    for j in range(min_seg, n + 1):
        best_cost = float("inf")
        best_t = 0

        pruned = set()
        for t in candidates:
            if j - t < min_seg:
                continue
            cost = opt[t] + _segment_cost(t, j) + penalty
            if cost < best_cost:
                best_cost = cost
                best_t = t
            # PELT pruning: if opt[t] + cost(t,j) > opt[j], then t
            # can never be optimal for any future j' > j
            if opt[t] + _segment_cost(t, j) > best_cost + penalty:
                pruned.add(t)

        opt[j] = best_cost
        last_cp[j] = best_t
        candidates -= pruned
        candidates.add(j)

    # Backtrack to recover change-points
    cps: list[int] = []
    pos = n
    while pos > 0:
        cp = last_cp[pos]
        if cp > 0:
            cps.append(cp)
        pos = cp

    cps.sort()
    return cps


def ld_decay_signal(
    r2_pairs: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    n_variants: int,
    *,
    k_neighbors: int = 5,
) -> Tensor:
    """Compute an LD-decay summary signal per variant.

    For each variant, computes the mean r² to its ``k_neighbors``
    nearest neighbors (by index distance).  The resulting signal
    can be fed to ``dp_changepoint()`` to detect LD boundaries.

    Parameters
    ----------
    r2_pairs : (P,) r² values for pre-computed pairs.
    idx_i, idx_j : (P,) pair indices.
    n_variants : int
        Total number of variants.
    k_neighbors : int
        Number of nearest neighbors for the local LD summary.

    Returns
    -------
    signal : (n_variants,) mean r² to nearest neighbors.
    """
    device = r2_pairs.device

    if _ld_decay_signal_native_enabled():
        import numpy as np
        ii_np = idx_i.detach().to(torch.int64).cpu().numpy()
        jj_np = idx_j.detach().to(torch.int64).cpu().numpy()
        r2_np = r2_pairs.detach().to(torch.float64).cpu().numpy()
        out_np = _ld_decay_signal_native.ld_decay_signal(
            ii_np, jj_np, r2_np, int(n_variants), int(k_neighbors),
        )
        return torch.from_numpy(out_np).to(device=device, dtype=torch.float64)

    # Accumulate r² contributions per variant
    r2_sum = torch.zeros(n_variants, dtype=torch.float64, device=device)
    r2_count = torch.zeros(n_variants, dtype=torch.float64, device=device)

    # For each pair, contribute to both variants
    ii = idx_i.long()
    jj = idx_j.long()
    dist = (jj - ii).abs()

    # Only use k nearest neighbors (smallest index distance)
    # Sort by distance and keep top-k per variant
    for v in range(n_variants):
        mask_i = ii == v
        mask_j = jj == v
        mask = mask_i | mask_j
        if not mask.any():
            continue
        r2_v = r2_pairs[mask]
        dist_v = dist[mask]
        # Keep k nearest
        k = min(k_neighbors, r2_v.shape[0])
        _, topk_idx = dist_v.topk(k, largest=False)
        r2_sum[v] = r2_v[topk_idx].sum()
        r2_count[v] = k

    r2_count = r2_count.clamp(min=1.0)
    return r2_sum / r2_count
