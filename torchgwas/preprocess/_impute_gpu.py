"""Torch-GPU kernels for the imputation hot loops.

These are dedicated GPU implementations of ``impute_mode`` / ``impute_knn``
/ ``impute_ld`` written entirely in vectorized torch ops. They are
selected by ``torchgwas._dispatch.select_path`` when:

- the input dosage matrix is already on a CUDA device, *and*
- the problem is large enough to amortize kernel-launch overhead, *and*
- ``TORCHGWAS_DISABLE_GPU`` is unset.

They are *not* selected for CPU tensors — the C++ extensions and the
pure-Python reference cover that case. Crucially, none of these kernels
proactively pulls data to host or to GPU; they assume the caller has
already chosen to keep the dosage matrix on CUDA.

The semantics match the pure-Python reference in ``impute.py`` to the
limit of float-summation order. The reference implementations remain in
``impute.py`` and stay the algorithmic spec.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# impute_mode — per-column histogram via scatter_add
# ---------------------------------------------------------------------------


def impute_mode_gpu(
    G_int: Tensor,
    max_dosage: int,
    n_cols: int,
) -> Tensor:
    """Vectorized per-column mode of an integer dosage matrix.

    Parameters
    ----------
    G_int : Tensor, shape (n, m), dtype long
        Integer dosages with ``-1`` marking missing entries.
    max_dosage : int
        Maximum non-missing dosage; histogram has ``max_dosage + 1`` bins.
    n_cols : int
        Number of columns ``m``. Passed explicitly for symmetry with the
        Python loop signature.

    Returns
    -------
    Tensor, shape (m,), dtype float64
        Per-column mode (most frequent dosage). All-missing columns get
        ``0.0``. Tie-break is first-wins (matches ``torch.argmax``).
    """
    bins = max_dosage + 1
    valid = G_int >= 0  # (n, m) bool

    # Replace -1 with 0 so the scatter index is in range; the contribution
    # of those cells is killed by ``valid.long()`` in the src term.
    safe_idx = torch.where(valid, G_int, torch.zeros_like(G_int))

    counts = torch.zeros(
        (bins, n_cols), dtype=torch.long, device=G_int.device
    )
    counts.scatter_add_(0, safe_idx, valid.long())

    # ``argmax`` over a column of all-zero counts returns 0 — exactly what
    # the Python reference produces for an all-missing column.
    return counts.argmax(dim=0).to(torch.float64)


# ---------------------------------------------------------------------------
# impute_knn — chunked fancy gather + masked weighted reduce
# ---------------------------------------------------------------------------


def impute_knn_gpu(
    G: Tensor,
    K: Tensor,
    knn_idx: Tensor,
    mask: Tensor,
    *,
    chunk_cols: int = 4096,
) -> Tensor:
    """Vectorized KNN imputation kernel for CUDA tensors.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN markers.
    K : Tensor, shape (n, n)
        Kinship / GRM matrix.
    knn_idx : Tensor, shape (n, k), dtype long
        Top-k neighbour indices per sample (already excludes self).
    mask : Tensor, shape (n, m), dtype bool
        ``True`` where ``G`` is NaN.
    chunk_cols : int
        Column-chunk size to bound the (n, k, chunk_cols) intermediate.

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN positions filled. The output dtype matches
        ``G``.
    """
    n, m = G.shape
    k = knn_idx.shape[1]

    # Per-sample weights live on (n, k); positive-clamp matches the
    # Python reference.
    weights_nk = K.gather(1, knn_idx).clamp(min=0.0).to(G.dtype)

    # Per-column observed mean for the all-neighbours-NaN fallback. This
    # is the *global* observed mean (over all rows), matching what the
    # Python reference computes inside its else-branch.
    n_obs = (~mask).sum(dim=0).clamp(min=1).to(G.dtype)
    G_zero = torch.where(mask, torch.zeros_like(G), G)
    col_means = G_zero.sum(dim=0) / n_obs
    col_means = torch.where(
        (~mask).any(dim=0), col_means, torch.zeros_like(col_means)
    )

    G_out = G.clone()

    for start in range(0, m, chunk_cols):
        end = min(m, start + chunk_cols)
        ch = end - start

        # Gather a (n, k, ch) slab of neighbour values without
        # materializing the full (n, k, m) intermediate.
        G_chunk = G[:, start:end]                # (n, ch)
        nbr_vals = G_chunk[knn_idx]              # (n, k, ch)
        nvalid_mask = ~torch.isnan(nbr_vals)     # (n, k, ch)

        # Per-(i, t, j) weight = K-weight * (neighbour valid).
        w = weights_nk.unsqueeze(-1) * nvalid_mask.to(G.dtype)
        nbr_clean = torch.where(
            nvalid_mask, nbr_vals, torch.zeros_like(nbr_vals)
        )

        wsum = w.sum(dim=1)                       # (n, ch)
        wnum = (w * nbr_clean).sum(dim=1)         # (n, ch)

        # Tier 1: weighted mean
        primary = wnum / wsum.clamp(min=1e-300)

        # Tier 2: simple mean of valid neighbours when all weights are 0
        nvalid = nvalid_mask.sum(dim=1).to(G.dtype)
        equal_mean = nbr_clean.sum(dim=1) / nvalid.clamp(min=1.0)

        # Tier 3: per-column observed mean when no neighbours are valid
        cm_chunk = col_means[start:end].unsqueeze(0).expand(n, -1)

        result = torch.where(
            wsum > 0,
            primary,
            torch.where(nvalid > 0, equal_mean, cm_chunk),
        )

        chunk_mask = mask[:, start:end]
        G_out[:, start:end] = torch.where(
            chunk_mask, result.to(G.dtype), G_out[:, start:end]
        )

    return G_out


# ---------------------------------------------------------------------------
# impute_ld — per-column matvec over the centred matrix
# ---------------------------------------------------------------------------


def impute_ld_gpu(
    G: Tensor,
    G_complete: Tensor,
    mask: Tensor,
    window_size: int,
) -> Tensor:
    """Vectorized LD-window imputation kernel for CUDA tensors.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Original dosage matrix with NaN markers.
    G_complete : Tensor, shape (n, m)
        Mean-imputed copy of ``G`` (used for centred dot products and as
        the prediction source).
    mask : Tensor, shape (n, m), dtype bool
        ``True`` where ``G`` is NaN.
    window_size : int
        Flanking window size (each side).

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN positions filled.
    """
    n, m = G.shape
    tiny = 1e-10
    G_out = G.clone()

    col_means = G_complete.mean(dim=0)
    col_stds = G_complete.std(dim=0)              # ddof=1, matches torch default
    centered = G_complete - col_means.unsqueeze(0)
    denom_n = float(max(n - 1, 1))

    # Only iterate columns that actually contain a missing entry.
    miss_cols = mask.any(dim=0).nonzero(as_tuple=False).squeeze(-1)
    if miss_cols.numel() == 0:
        return G_out

    for j in miss_cols.tolist():
        rows_missing = mask[:, j].nonzero(as_tuple=False).squeeze(-1)
        if rows_missing.numel() == 0:
            continue

        left = max(0, j - window_size)
        right = min(m, j + window_size + 1)

        # Window indices excluding the target SNP.
        if right - left <= 1:
            G_out[rows_missing, j] = G_complete[rows_missing, j]
            continue

        idxs = torch.cat(
            [
                torch.arange(left, j, device=G.device, dtype=torch.long),
                torch.arange(j + 1, right, device=G.device, dtype=torch.long),
            ]
        )

        tstd = col_stds[j]
        if float(tstd.item()) < tiny:
            G_out[rows_missing, j] = G_complete[rows_missing, j]
            continue

        # Centred dot products: (centered[:, idxs])^T @ centered[:, j]
        dots = centered[:, idxs].T @ centered[:, j]
        denom = denom_n * tstd * col_stds[idxs].clamp(min=tiny)
        corrs = dots / denom
        weights = corrs.abs()
        wsum = weights.sum()

        if float(wsum.item()) > tiny:
            preds = G_complete[rows_missing][:, idxs] @ weights / wsum
            G_out[rows_missing, j] = preds
        else:
            G_out[rows_missing, j] = G_complete[rows_missing, j]

    return G_out


__all__ = [
    "impute_mode_gpu",
    "impute_knn_gpu",
    "impute_ld_gpu",
]
