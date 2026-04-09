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

    # Transform to Cauchy: T = tan((0.5 - p) * pi)
    T = torch.tan((0.5 - p_clamped) * math.pi)  # (m, k)

    # Weighted sum across methods/traits
    T_combined = (T * weights.unsqueeze(0)).sum(dim=1)  # (m,)

    # Back-transform to p-value
    p_combined = 0.5 - torch.arctan(T_combined) / math.pi

    # Clamp to [0, 1]
    p_combined = torch.clamp(p_combined, min=0.0, max=1.0)

    return p_combined.squeeze() if scalar_mode else p_combined
