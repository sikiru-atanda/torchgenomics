"""Build per-individual similarity kernels from a continuous feature matrix M (n, f).

Used as a context kernel alongside the SNP GRM in `mkernel_h2`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_expression_kernel(
    M: Tensor,
    method: str = "linear",
    bandwidth: float | None = None,
    standardise: bool = True,
) -> Tensor:
    """Build an (n, n) PSD kernel from a (n, f) abundance matrix.

    Methods
    -------
    linear   : (M_s @ M_s.T) / f, with M_s column-standardised when ``standardise``.
    gaussian : exp(-||x_i - x_j||^2 / (2 * h^2)); h = median pairwise dist if None.
    cosine   : row-normalised inner product (diag = 1).
    """
    if M.ndim != 2:
        raise ValueError(f"M must be (n, f); got shape {tuple(M.shape)}.")

    M = M.to(torch.float64)
    n, f = M.shape
    if f == 0:
        raise ValueError("M has zero features.")

    if method == "linear":
        Ms = _standardise(M) if standardise else M
        return (Ms @ Ms.T) / float(f)

    if method == "gaussian":
        Ms = _standardise(M) if standardise else M
        # squared pairwise distances
        sq = (Ms * Ms).sum(dim=1, keepdim=True)
        d2 = sq + sq.T - 2.0 * (Ms @ Ms.T)
        d2 = d2.clamp(min=0.0)
        if bandwidth is None:
            # median heuristic over upper-triangular off-diagonal distances
            iu = torch.triu_indices(n, n, offset=1)
            vals = d2[iu[0], iu[1]].sqrt()
            h = vals.median().item() if vals.numel() > 0 else 1.0
            if h <= 0.0:
                h = 1.0
        else:
            h = float(bandwidth)
        return torch.exp(-d2 / (2.0 * h * h))

    if method == "cosine":
        norms = M.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Mn = M / norms
        return Mn @ Mn.T

    raise ValueError(f"Unknown kernel method: {method!r}")


def _standardise(M: Tensor) -> Tensor:
    mu = M.mean(dim=0, keepdim=True)
    sd = M.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-12)
    return (M - mu) / sd
