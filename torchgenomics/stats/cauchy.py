"""Cauchy combination test for combining p-values across methods/traits.

Implements the Aggregated Cauchy Association Test (ACAT) from Liu & Xie (2020).
The Cauchy combination works for correlated tests without needing a correlation
matrix, making it ideal for combining multi-trait or multi-method GWAS results.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def cauchy_combination(
    p_matrix: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Combine p-values across methods or traits via the Cauchy distribution.

    For each test (row), transforms p-values to Cauchy statistics
    T_i = tan((0.5 - p_i) * pi), computes a weighted sum, and back-transforms
    to a combined p-value.

    Parameters
    ----------
    p_matrix : Tensor, shape (m, k) or (m,)
        P-values for m tests across k methods/traits.
        If 1-D (m,), treated as a single set of m p-values to combine into
        one scalar (k=m, returns a single combined p-value).
    weights : Tensor, shape (k,), optional
        Non-negative weights. Normalized to sum to 1 internally.
        If None, uses equal weights (1/k).

    Returns
    -------
    Tensor, shape (m,) or scalar
        Combined p-values, one per test. Clamped to [0, 1].
    """
    scalar_mode = p_matrix.ndim == 1

    if scalar_mode:
        # Single set of p-values → combine all into one scalar
        p_matrix = p_matrix.unsqueeze(0)  # (1, m)

    m, k = p_matrix.shape

    if weights is None:
        weights = torch.ones(k, dtype=p_matrix.dtype, device=p_matrix.device) / k
    else:
        weights = weights.to(dtype=p_matrix.dtype, device=p_matrix.device)
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            # All-zero weights: return 0.5 (uninformative)
            result = torch.full((m,), 0.5, dtype=p_matrix.dtype, device=p_matrix.device)
            return result.squeeze() if scalar_mode else result

    # Clamp p-values away from exact 0 and 1 to avoid infinite tan values
    p_clamped = torch.clamp(p_matrix, min=1e-300, max=1.0 - 1e-15)

    # Transform to Cauchy: T = tan((0.5 - p) * pi). For very small p, the naive
    # tan loses all precision because (0.5 - p) rounds to 0.5 in float64, so
    # tan(pi/2 - pi*p) = cot(pi*p) ~ 1/(pi*p) is used instead (Liu & Xie 2020,
    # ACAT; matches their reference implementation). Without this branch the
    # combined p-value saturates at ~5.5e-17 regardless of how small the input.
    is_small = p_clamped < 1e-15
    T_tan = torch.tan((0.5 - p_clamped) * math.pi)  # (m, k)
    T_small = 1.0 / (p_clamped * math.pi)  # (m, k)
    T = torch.where(is_small, T_small, T_tan)

    # Weighted sum across methods/traits
    T_combined = (T * weights.unsqueeze(0)).sum(dim=1)  # (m,)

    # Back-transform to p-value. For a large combined statistic (tiny p) the
    # survival function 0.5 - arctan(T)/pi saturates because arctan(T) -> pi/2;
    # use the tail approximation p = 1/(pi*T) there instead.
    is_large = T_combined > 1e15
    p_normal = 0.5 - torch.arctan(T_combined) / math.pi
    p_tail = 1.0 / (math.pi * torch.clamp(T_combined, min=1e-300))
    p_combined = torch.where(is_large, p_tail, p_normal)

    # Clamp to [0, 1]
    p_combined = torch.clamp(p_combined, min=0.0, max=1.0)

    return p_combined.squeeze() if scalar_mode else p_combined
