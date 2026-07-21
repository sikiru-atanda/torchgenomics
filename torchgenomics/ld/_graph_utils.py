"""Graph utilities for LD-based block detection.

Provides graph Laplacian construction, spectral partitioning,
connected component detection, and maximum-weight independent set
for interval graphs.  All tensor ops use float64 for consistency
with the LD module.
"""

from __future__ import annotations

import os
from collections import deque

import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import (
    HAS_NATIVE_GRAPH,
    HAS_NATIVE_GREEDY_MWIS,
    _graph_native,
    _greedy_mwis_native,
)

_EPS = 1e-10


def _native_enabled() -> bool:
    """Whether the native C++ graph-utils accelerator should be used.

    Disabled when the extension was not built or when the user sets
    ``TORCHGENOMICS_DISABLE_NATIVE=1`` in the environment.
    """
    return HAS_NATIVE_GRAPH and not native_disabled()


def _greedy_mwis_native_enabled() -> bool:
    """Whether the native C++ greedy_mwis accelerator should be used."""
    return HAS_NATIVE_GREEDY_MWIS and not os.environ.get(
        "TORCHGENOMICS_DISABLE_NATIVE"
    )


# ── Laplacian construction ─────────────────────────────────────────

def adjacency_to_laplacian(A: Tensor) -> Tensor:
    """Unnormalized graph Laplacian  L = D - A.

    Parameters
    ----------
    A : (m, m) symmetric non-negative adjacency matrix.

    Returns
    -------
    L : (m, m) Laplacian (row-sums zero).
    """
    D = A.sum(dim=1)
    return torch.diag(D) - A


def normalized_laplacian(A: Tensor) -> Tensor:
    """Symmetric normalized Laplacian  I - D^{-1/2} A D^{-1/2}.

    Eigenvalues lie in [0, 2].  Isolated nodes (degree 0) get
    self-loop = 1 on the diagonal to avoid division by zero.
    """
    D = A.sum(dim=1)
    D_inv_sqrt = torch.where(D > _EPS, D.pow(-0.5), torch.zeros_like(D))
    # D^{-1/2} A D^{-1/2}
    DAD = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    m = A.shape[0]
    return torch.eye(m, dtype=A.dtype, device=A.device) - DAD


# ── Spectral partitioning ──────────────────────────────────────────

def spectral_partition(
    A: Tensor,
    n_clusters: int = 2,
    *,
    min_cluster_size: int = 1,
) -> list[list[int]]:
    """Spectral clustering on a symmetric adjacency matrix.

    For ``n_clusters=2`` uses the Fiedler vector (2nd-smallest eigenvector
    of the Laplacian).  For k > 2, uses the k smallest eigenvectors
    followed by k-means on the rows of the embedding.

    Parameters
    ----------
    A : (m, m) symmetric non-negative adjacency.
    n_clusters : int
        Desired number of clusters.
    min_cluster_size : int
        Clusters smaller than this are merged into the nearest neighbor.

    Returns
    -------
    list[list[int]]
        Each inner list contains vertex indices belonging to one cluster,
        sorted in ascending order.
    """
    m = A.shape[0]
    if m <= n_clusters:
        return [[i] for i in range(m)]

    # First, find connected components — disconnected components are
    # natural clusters that spectral methods can miss due to arbitrary
    # eigenvector signs.
    components = connected_components(A, threshold=0.0)
    if len(components) >= n_clusters:
        # Already have enough clusters from connectivity alone
        result = [c for c in components if len(c) >= min_cluster_size]
        # Merge tiny components into nearest neighbor
        for comp in components:
            if len(comp) < min_cluster_size and result:
                best_cluster = max(result, key=lambda c: sum(
                    A[i, j].item() for i in comp for j in c
                ))
                best_cluster.extend(comp)
                best_cluster.sort()
        return sorted(result, key=lambda c: c[0])

    # Graph is connected (or fewer components than desired clusters):
    # apply spectral bisection / k-way clustering
    L = normalized_laplacian(A)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    embed = eigenvectors[:, 1:n_clusters + 1]  # (m, n_clusters)

    if n_clusters == 2:
        fiedler = embed[:, 0]
        labels = (fiedler >= 0).long()
    else:
        labels = _simple_kmeans(embed, n_clusters)

    # Group indices by label
    clusters: dict[int, list[int]] = {}
    for i in range(m):
        lab = labels[i].item()
        clusters.setdefault(lab, []).append(i)

    # Merge tiny clusters into nearest neighbor
    result = []
    for indices in clusters.values():
        if len(indices) >= min_cluster_size:
            result.append(sorted(indices))
        else:
            best_cluster = None
            best_score = -1.0
            for c in result:
                score = sum(A[i, j].item() for i in indices for j in c)
                if score > best_score:
                    best_score = score
                    best_cluster = c
            if best_cluster is not None:
                best_cluster.extend(indices)
                best_cluster.sort()
            else:
                result.append(sorted(indices))

    return sorted(result, key=lambda c: c[0])


def _simple_kmeans(X: Tensor, k: int, max_iter: int = 50) -> Tensor:
    """Minimal k-means for spectral clustering embeddings.

    Parameters
    ----------
    X : (n, d) data matrix.
    k : int number of clusters.

    Returns
    -------
    labels : (n,) long tensor of cluster assignments.
    """
    n, d = X.shape
    # Initialize with k evenly-spaced indices
    idx = torch.linspace(0, n - 1, k, device=X.device).long()
    centers = X[idx].clone()  # (k, d)

    labels = torch.zeros(n, dtype=torch.long, device=X.device)
    for _ in range(max_iter):
        # Assign each point to nearest center
        dists = torch.cdist(X, centers)  # (n, k)
        new_labels = dists.argmin(dim=1)

        if (new_labels == labels).all():
            break
        labels = new_labels

        # Recompute centers
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(dim=0)

    return labels


# ── Connected components ───────────────────────────────────────────

def connected_components(
    A: Tensor,
    threshold: float = 0.0,
) -> list[list[int]]:
    """Find connected components in a thresholded adjacency matrix.

    Parameters
    ----------
    A : (m, m) adjacency matrix.
    threshold : float
        Edges with weight <= threshold are ignored.

    Returns
    -------
    list[list[int]]
        Each list contains sorted vertex indices of one component.
    """
    m = A.shape[0]

    # Native C++ shortcut. The pure-Python BFS body below remains the
    # canonical algorithmic reference and is exercised when the extension
    # is missing or TORCHGENOMICS_DISABLE_NATIVE=1 is set.
    # Device discipline: only take the C++ path for CPU tensors; a CUDA input
    # runs the torch/Python body instead of being force-copied to host for the
    # native path (CLAUDE.md invariant; this algorithm has no GPU kernel).
    if _native_enabled() and m > 0 and A.device.type == "cpu":
        A_np = A.detach().to(torch.float64).cpu().numpy()
        return [list(c) for c in _graph_native.connected_components(A_np, float(threshold))]

    visited = [False] * m
    components: list[list[int]] = []

    # Build sparse neighbor lists on CPU for BFS
    A_cpu = A.cpu()
    for start in range(m):
        if visited[start]:
            continue
        component: list[int] = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in range(m):
                if not visited[neighbor] and A_cpu[node, neighbor] > threshold:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(sorted(component))

    return sorted(components, key=lambda c: c[0])


# ── Maximum-weight independent set on interval graph ───────────────

def greedy_mwis(
    intervals: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Greedy maximum-weight independent set for an interval graph.

    Each interval is ``(start, end, weight)``.  Returns a subset of
    non-overlapping intervals with high total weight.

    Greedy strategy: sort by weight descending, greedily add intervals
    that do not overlap with any already-selected interval.

    Parameters
    ----------
    intervals : list of (start, end, weight)
        Intervals are closed: [start, end].

    Returns
    -------
    list of selected (start, end, weight), sorted by start.
    """
    if not intervals:
        return []

    if _greedy_mwis_native_enabled():
        import numpy as np
        starts_np = np.fromiter((iv[0] for iv in intervals), dtype=np.int64,
                                count=len(intervals))
        ends_np = np.fromiter((iv[1] for iv in intervals), dtype=np.int64,
                              count=len(intervals))
        weights_np = np.fromiter((iv[2] for iv in intervals), dtype=np.float64,
                                 count=len(intervals))
        sel_s, sel_e, sel_w = _greedy_mwis_native.greedy_mwis(
            starts_np, ends_np, weights_np,
        )
        return [
            (int(sel_s[i]), int(sel_e[i]), float(sel_w[i]))
            for i in range(sel_s.shape[0])
        ]

    # Sort by weight descending
    ranked = sorted(intervals, key=lambda x: -x[2])
    selected: list[tuple[int, int, float]] = []

    for start, end, weight in ranked:
        # Check overlap with already-selected intervals
        overlaps = False
        for s_start, s_end, _ in selected:
            if start <= s_end and end >= s_start:
                overlaps = True
                break
        if not overlaps:
            selected.append((start, end, weight))

    return sorted(selected, key=lambda x: x[0])
