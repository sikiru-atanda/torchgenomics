"""Built-in imputation methods: mean, mode, KNN (using GRM), LD-based.

The flat in-memory entry points (``impute_mean`` / ``impute_mode`` /
``impute_knn`` / ``impute_ld``) below are the algorithmic reference;
the streaming-friendly variants (``compute_column_means_streaming``,
``compute_column_modes_streaming``, ``impute_chunk_with_means``,
``impute_chunk_with_modes``, etc.) at the bottom of this module are
the building blocks the streaming CLI uses, and reduce to the same
math chunk-by-chunk.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor

from .._dispatch import native_disabled, select_path
from .._native import (
    HAS_NATIVE_IMPUTE_KNN,
    HAS_NATIVE_IMPUTE_LD,
    HAS_NATIVE_IMPUTE_MODE,
    _impute_knn_native,
    _impute_ld_native,
    _impute_mode_native,
)
from ._impute_gpu import impute_knn_gpu, impute_ld_gpu, impute_mode_gpu

# ---------------------------------------------------------------------------
# Path-selection helpers
# ---------------------------------------------------------------------------
#
# These thin shims keep backward compatibility with the existing test
# suite (which imports them by name) while delegating the real decision
# to ``torchgwas._dispatch.select_path``. They answer the *global* "is
# the C++ build present and not disabled?" question — the dispatchers
# below additionally consult ``select_path`` with the live tensor so the
# device and problem size also enter the decision.


def _impute_mode_native_enabled() -> bool:
    """Whether the native C++ impute_mode accelerator should be used.

    This reflects the build / environment state only; the per-call
    dispatcher additionally checks the input tensor's device and size
    via :func:`torchgwas._dispatch.select_path`.
    """
    return HAS_NATIVE_IMPUTE_MODE and not native_disabled()


def _impute_knn_native_enabled() -> bool:
    """Whether the native C++ impute_knn accelerator should be used."""
    return HAS_NATIVE_IMPUTE_KNN and not native_disabled()


def _impute_ld_native_enabled() -> bool:
    """Whether the native C++ impute_ld accelerator should be used."""
    return HAS_NATIVE_IMPUTE_LD and not native_disabled()


def impute_mean(G: Tensor) -> Tensor:
    """Replace NaN entries with per-SNP mean dosage.

    This is the default imputation method matching GEMMA's convention.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN replaced by per-SNP means.
    """
    mask = torch.isnan(G)
    if not mask.any():
        return G

    G_out = G.clone()

    # Compute per-SNP means ignoring NaN
    G_zero = torch.where(mask, torch.zeros_like(G), G)
    n_obs = (~mask).sum(dim=0).to(G.dtype)
    n_obs = torch.clamp(n_obs, min=1.0)
    col_means = G_zero.sum(dim=0) / n_obs  # (m,)

    # Broadcast means into NaN positions
    G_out[mask] = col_means.unsqueeze(0).expand_as(G)[mask]

    return G_out


def impute_mode(G: Tensor) -> Tensor:
    """Replace NaN entries with per-SNP mode (most frequent dosage).

    For genotype data, dosages are typically integers (0, 1, 2 for diploid),
    so mode is well-defined. Ties are broken by choosing the lower value.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN replaced by per-SNP modes.
    """
    mask = torch.isnan(G)
    if not mask.any():
        return G

    G_out = G.clone()
    n, m = G.shape

    # Compute mode per column using vectorized bincount
    # Replace NaN with -1 so we can work with integers
    G_int = torch.where(mask, torch.tensor(-1, dtype=torch.long, device=G.device),
                        torch.round(G).long())
    max_dosage = int(G_int[G_int >= 0].max().item()) if (G_int >= 0).any() else 2

    # Three-way dispatch: native C++ on CPU-resident data, generic torch
    # on whatever device the tensor lives (the "python" branch — torch
    # ops are device-agnostic so this naturally stays on the GPU when G
    # is a CUDA tensor), and a Python loop as the algorithmic reference.
    # We never proactively pull a CUDA tensor to host for the C++ path.
    path = select_path(
        G, has_native=HAS_NATIVE_IMPUTE_MODE, has_gpu_kernel=True
    )
    if path == "gpu":
        modes = impute_mode_gpu(G_int, max_dosage, m).to(G.dtype)
    elif path == "native":
        G_int_np = G_int.detach().to(torch.int64).cpu().numpy()
        modes_np = _impute_mode_native.impute_mode_columns(
            G_int_np, int(max_dosage)
        )
        modes = torch.from_numpy(modes_np).to(device=G.device, dtype=G.dtype)
    else:
        modes = torch.zeros(m, dtype=G.dtype, device=G.device)
        for j in range(m):
            col = G_int[:, j]
            valid = col[col >= 0]
            if len(valid) == 0:
                modes[j] = 0.0
                continue
            counts = torch.bincount(valid, minlength=max_dosage + 1)
            modes[j] = float(counts.argmax().item())

    G_out[mask] = modes.unsqueeze(0).expand_as(G)[mask]
    return G_out


def impute_knn(G: Tensor, K: Tensor, k: int = 5) -> Tensor:
    """KNN imputation using GRM-derived similarity.

    For each missing entry G[i, j], find the k samples most similar to
    sample i (by GRM), and impute as the weighted average of their
    observed values at SNP j.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.
    K : Tensor, shape (n, n)
        Kinship / GRM matrix (higher = more similar).
    k : int
        Number of nearest neighbours.

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN imputed by KNN.
    """
    mask = torch.isnan(G)
    if not mask.any():
        return G

    G_out = G.clone()
    n, m = G.shape

    # For each sample, precompute k nearest neighbours from GRM
    # Set self-similarity to -inf to exclude self
    K_mod = K.clone()
    K_mod.fill_diagonal_(-float("inf"))

    # Top-k neighbours per sample: (n, k)
    _, knn_idx = torch.topk(K_mod, k=min(k, n - 1), dim=1)

    # Three-way dispatch — see ``impute_mode`` for the rationale.
    path = select_path(
        G, has_native=HAS_NATIVE_IMPUTE_KNN, has_gpu_kernel=True
    )
    if path == "gpu":
        return impute_knn_gpu(G, K, knn_idx, mask)
    if path == "native":
        # Precompute per-column observed mean for the all-neighbours-missing
        # fallback so the inner C++ loop never has to scan G[:, j] itself.
        col_mask = ~mask
        n_obs = col_mask.sum(dim=0).to(torch.float64)
        n_obs_safe = torch.clamp(n_obs, min=1.0)
        G_zero = torch.where(mask, torch.zeros_like(G), G)
        col_means = (G_zero.sum(dim=0).to(torch.float64) / n_obs_safe)
        col_means = torch.where(n_obs > 0, col_means, torch.zeros_like(col_means))

        G_in_np = G.detach().to(torch.float64).cpu().numpy()
        G_out_np = G_in_np.copy()
        knn_np = knn_idx.detach().to(torch.int64).cpu().numpy()
        K_np = K.detach().to(torch.float64).cpu().numpy()
        cm_np = col_means.detach().cpu().numpy()
        _impute_knn_native.impute_knn_fill(
            G_in_np, G_out_np, knn_np, K_np, cm_np
        )
        return torch.from_numpy(G_out_np).to(device=G.device, dtype=G.dtype)

    # Impute each missing entry
    miss_i, miss_j = torch.where(mask)
    for idx in range(len(miss_i)):
        i = miss_i[idx].item()
        j = miss_j[idx].item()

        # Get neighbours' values at this SNP
        neighbours = knn_idx[i]
        neighbour_vals = G[neighbours, j]

        # Use only non-NaN neighbours
        valid = ~torch.isnan(neighbour_vals)
        if valid.any():
            # Weight by GRM similarity
            weights = K[i, neighbours][valid]
            weights = torch.clamp(weights, min=0.0)  # ensure non-negative
            w_sum = weights.sum()
            if w_sum > 0:
                G_out[i, j] = (neighbour_vals[valid] * weights).sum() / w_sum
            else:
                G_out[i, j] = neighbour_vals[valid].mean()
        else:
            # Fallback: mean of observed values for this SNP
            col = G[:, j]
            observed = col[~torch.isnan(col)]
            G_out[i, j] = observed.mean() if len(observed) > 0 else 0.0

    return G_out


def impute_ld(G: Tensor, window_size: int = 50) -> Tensor:
    """LD-based sliding-window imputation.

    For each missing entry G[i, j], uses a local regression on the
    surrounding window of SNPs to predict the missing value.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix with NaN for missing values.
    window_size : int
        Number of flanking SNPs on each side to use.

    Returns
    -------
    Tensor, shape (n, m)
        Dosage matrix with NaN imputed by LD regression.
    """
    mask = torch.isnan(G)
    if not mask.any():
        return G

    # First pass: mean-impute to get complete matrix for correlation
    G_complete = impute_mean(G)
    G_out = G.clone()

    n, m = G.shape

    # Three-way dispatch — see ``impute_mode`` for the rationale.
    path = select_path(
        G, has_native=HAS_NATIVE_IMPUTE_LD, has_gpu_kernel=True
    )
    if path == "gpu":
        return impute_ld_gpu(G, G_complete, mask, int(window_size))
    if path == "native":
        col_means = G_complete.mean(dim=0).to(torch.float64)
        # torch.std default is unbiased (ddof=1), matching the Python loop.
        col_stds = G_complete.std(dim=0).to(torch.float64)
        Gc_np = G_complete.detach().to(torch.float64).cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype("uint8")
        cm_np = col_means.detach().cpu().numpy()
        cs_np = col_stds.detach().cpu().numpy()
        G_out_np = G_out.detach().to(torch.float64).cpu().numpy()
        _impute_ld_native.impute_ld_fill(
            Gc_np, mask_np, cm_np, cs_np, int(window_size), G_out_np
        )
        return torch.from_numpy(G_out_np).to(device=G.device, dtype=G.dtype)

    miss_i, miss_j = torch.where(mask)

    for idx in range(len(miss_i)):
        i = miss_i[idx].item()
        j = miss_j[idx].item()

        # Window boundaries
        left = max(0, j - window_size)
        right = min(m, j + window_size + 1)

        # Exclude the target SNP
        window_cols = list(range(left, j)) + list(range(j + 1, right))
        if not window_cols:
            G_out[i, j] = G_complete[i, j]
            continue

        # Use correlation-weighted average from window
        target_col = G_complete[:, j]
        window_data = G_complete[:, window_cols]

        # Compute correlation of each window SNP with target
        target_centered = target_col - target_col.mean()
        window_centered = window_data - window_data.mean(dim=0)

        target_std = target_centered.std()
        window_std = window_centered.std(dim=0)

        if target_std < 1e-10:
            G_out[i, j] = G_complete[i, j]
            continue

        corrs = (target_centered @ window_centered) / ((n - 1) * target_std * window_std.clamp(min=1e-10))
        weights = corrs.abs()  # use absolute correlation as weight

        # Weighted prediction
        w_sum = weights.sum()
        if w_sum > 1e-10:
            G_out[i, j] = (G_complete[i, window_cols] * weights).sum() / w_sum
        else:
            G_out[i, j] = G_complete[i, j]

    return G_out


# ---------------------------------------------------------------------------
# Streaming-friendly building blocks (Efficiency E4)
# ---------------------------------------------------------------------------
#
# These helpers let the CLI ``impute`` subcommand do a two-pass scan
# over the genotype reader without ever materializing the full
# ``(n_samples × n_variants)`` tensor. Pass 1 accumulates per-column
# statistics (sums for mean, integer-class histograms for mode); pass
# 2 streams chunks again and writes per-chunk imputed dosages to a
# streaming output sink.
#
# The math here is identical to the flat-tensor reference functions
# above — we just split the column-statistic compute and the fill
# step into two separate passes that consume an ``iter_chunks``
# iterator. For mean/mode this is a clean two-pass; for KNN we do a
# single pass with the full GRM (built streaming via
# ``grm_vanraden_streaming``); for LD we do a sliding-window
# buffered single pass with documented buffer cost.


def compute_column_means_streaming(
    chunk_iter: Iterator[tuple[Tensor, object]],
) -> Tensor:
    """Pass 1 of streaming mean imputation.

    Accumulates per-column ``(sum, n_observed)`` over the chunk
    iterator and returns the per-column means. Memory cost is
    ``O(n_variants)`` for the running sums — independent of
    ``n_samples`` — so this fits even at biobank scale.

    Parameters
    ----------
    chunk_iter
        Yields ``(G_chunk, vmeta)`` tuples. ``G_chunk`` may have NaN
        for missing entries.

    Returns
    -------
    Tensor, shape (m,)
        Per-column means in the dtype of the first chunk seen.
    """
    sums_list: list[Tensor] = []
    counts_list: list[Tensor] = []
    dtype = None

    for G_chunk, _ in chunk_iter:
        if dtype is None:
            dtype = G_chunk.dtype
        mask = torch.isnan(G_chunk)
        G_zero = torch.where(mask, torch.zeros_like(G_chunk), G_chunk)
        sums_list.append(G_zero.sum(dim=0))
        counts_list.append((~mask).sum(dim=0).to(dtype))

    if not sums_list:
        return torch.zeros(0, dtype=torch.float64)

    sums = torch.cat(sums_list, dim=0)
    counts = torch.cat(counts_list, dim=0).clamp(min=1.0)
    return sums / counts


def impute_chunk_with_means(
    G_chunk: Tensor, col_means: Tensor, col_offset: int,
) -> Tensor:
    """Pass 2 of streaming mean imputation.

    Takes a single chunk, slices the global ``col_means`` for the
    chunk's column window, and replaces NaN entries with the
    corresponding per-column mean. Returns the imputed chunk.

    Parameters
    ----------
    G_chunk : Tensor, shape (n, m_chunk)
    col_means : Tensor, shape (m_total,)
        Global per-column means from
        :func:`compute_column_means_streaming`.
    col_offset : int
        Starting column index in the global G this chunk represents.
    """
    mask = torch.isnan(G_chunk)
    if not mask.any():
        return G_chunk
    out = G_chunk.clone()
    m_chunk = G_chunk.shape[1]
    means_slice = col_means[col_offset : col_offset + m_chunk]
    fill = means_slice.unsqueeze(0).expand_as(G_chunk).to(G_chunk.dtype)
    out[mask] = fill[mask]
    return out


def compute_column_modes_streaming(
    chunk_iter: Iterator[tuple[Tensor, object]],
    max_dosage: int,
) -> Tensor:
    """Pass 1 of streaming mode imputation.

    Accumulates per-column class histograms (integer dosage classes
    ``0..max_dosage``) across chunks and returns the per-column mode
    (lowest-value tie-break, matching :func:`impute_mode`). Memory
    cost is ``O(n_variants × (max_dosage + 1))`` — for diploid
    (max_dosage=2) that's three times the column count, well within
    biobank-scale memory.

    Parameters
    ----------
    chunk_iter
        Yields ``(G_chunk, vmeta)`` tuples.
    max_dosage : int
        Highest dosage value in the dataset. For ploidy-k organisms,
        pass ``k``.
    """
    counts_per_class_list: list[Tensor] = []  # list of (max_dosage+1, m_chunk)
    dtype = None

    for G_chunk, _ in chunk_iter:
        if dtype is None:
            dtype = G_chunk.dtype
        mask = torch.isnan(G_chunk)
        G_int = torch.where(
            mask,
            torch.tensor(-1, dtype=torch.long, device=G_chunk.device),
            torch.round(G_chunk).long(),
        )
        m_chunk = G_chunk.shape[1]
        # Per-column class counts. (max_dosage+1, m_chunk).
        cnts = torch.zeros(
            max_dosage + 1, m_chunk, dtype=torch.long, device=G_chunk.device,
        )
        for c in range(max_dosage + 1):
            cnts[c] = (G_int == c).sum(dim=0)
        counts_per_class_list.append(cnts)

    if not counts_per_class_list:
        return torch.zeros(0, dtype=torch.float64)

    # Concat along columns -> (max_dosage+1, m_total).
    cnts_global = torch.cat(counts_per_class_list, dim=1)
    # argmax over class axis. Lowest-index tie-break is the default.
    modes = cnts_global.argmax(dim=0).to(dtype or torch.float64)
    return modes


def impute_chunk_with_modes(
    G_chunk: Tensor, col_modes: Tensor, col_offset: int,
) -> Tensor:
    """Pass 2 of streaming mode imputation."""
    mask = torch.isnan(G_chunk)
    if not mask.any():
        return G_chunk
    out = G_chunk.clone()
    m_chunk = G_chunk.shape[1]
    modes_slice = col_modes[col_offset : col_offset + m_chunk]
    fill = modes_slice.unsqueeze(0).expand_as(G_chunk).to(G_chunk.dtype)
    out[mask] = fill[mask]
    return out


def impute_chunk_with_knn(
    G_chunk: Tensor, K: Tensor, k: int = 5,
) -> Tensor:
    """KNN imputation applied to a single chunk.

    Once the kinship matrix ``K`` (built once via
    :func:`grm_vanraden_streaming`) is available, KNN imputation is
    fundamentally per-chunk: each chunk's missing entries only need
    other rows of the same chunk (column ``j`` is restricted to the
    chunk's column window). We delegate to :func:`impute_knn`.

    Memory cost: ``O(n_samples² × 8 B)`` for K (held once across the
    whole imputation), plus ``O(n_samples × m_chunk × 8 B)`` for the
    chunk. For biobank-scale (n=500K) the GRM is ~1 TB float64; this
    is the documented hard limit for the KNN streaming path.

    Parameters
    ----------
    G_chunk : Tensor, shape (n, m_chunk)
    K : Tensor, shape (n, n)
        Kinship matrix (held once across all chunks).
    k : int
        Neighbour count.
    """
    return impute_knn(G_chunk, K, k=k)


def impute_chunk_with_ld_window(
    G_chunk: Tensor,
    G_left_buffer: Tensor | None,
    G_right_buffer: Tensor | None,
    window_size: int,
    chunk_col_offset: int,
) -> Tensor:
    """LD-based imputation on a single chunk with buffered flanks.

    The reference :func:`impute_ld` looks at each missing entry's
    flanking ``window_size`` SNPs on each side. Streaming with
    ``window_size > chunk_size`` requires holding ``window_size``
    flanking SNPs in memory on each side of the active chunk —
    documented as the buffer cost.

    Parameters
    ----------
    G_chunk : Tensor, shape (n, m_chunk)
        The active chunk (NaN allowed).
    G_left_buffer, G_right_buffer : Tensor or None
        Flanking SNPs from the previous / next chunks. Each of shape
        ``(n, ≤ window_size)``. The caller is responsible for
        windowing.
    window_size : int
    chunk_col_offset : int
        Number of columns in ``G_left_buffer`` (used to remap target
        column indices into the merged tensor).
    """
    pieces: list[Tensor] = []
    if G_left_buffer is not None and G_left_buffer.shape[1] > 0:
        pieces.append(G_left_buffer)
    pieces.append(G_chunk)
    if G_right_buffer is not None and G_right_buffer.shape[1] > 0:
        pieces.append(G_right_buffer)

    G_merged = torch.cat(pieces, dim=1) if len(pieces) > 1 else G_chunk
    G_imp = impute_ld(G_merged, window_size=window_size)

    # Slice out the chunk's range from the imputed merged tensor.
    return G_imp[:, chunk_col_offset : chunk_col_offset + G_chunk.shape[1]].clone()
