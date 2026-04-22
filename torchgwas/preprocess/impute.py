"""Built-in imputation methods: mean, mode, KNN (using GRM), LD-based."""

from __future__ import annotations

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
