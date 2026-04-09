"""Multiple testing correction: Bonferroni, Sidak, Holm, BH, BY, Storey q-value.

All functions accept a 1-D tensor of raw p-values and return adjusted p-values
of the same shape and device.  Adjusted p-values are clamped to [0, 1].
"""

from __future__ import annotations

import torch
from torch import Tensor


def bonferroni(p: Tensor) -> Tensor:
    """Bonferroni correction: p_adj = min(p * m, 1)."""
    m = p.shape[0]
    return torch.clamp(p * m, max=1.0)


def sidak(p: Tensor) -> Tensor:
    """Sidak correction: p_adj = 1 - (1 - p)^m.

    Slightly less conservative than Bonferroni; exact when tests are independent.
    """
    m = p.shape[0]
    return torch.clamp(1.0 - (1.0 - p) ** m, min=0.0, max=1.0)


def holm(p: Tensor) -> Tensor:
    """Holm step-down correction (uniformly more powerful than Bonferroni).

    Algorithm:
      1. Sort p-values ascending: p_(1) <= p_(2) <= ... <= p_(m)
      2. Adjusted: p_adj_(i) = max_{j<=i} min(p_(j) * (m - j + 1), 1)
      3. Map back to original order.
    """
    m = p.shape[0]
    sorted_p, sort_idx = p.sort()
    multipliers = torch.arange(m, 0, -1, dtype=p.dtype, device=p.device)
    adjusted = torch.clamp(sorted_p * multipliers, max=1.0)

    # Enforce monotonicity (cumulative max)
    adjusted = torch.cummax(adjusted, dim=0).values

    # Unsort
    result = torch.empty_like(p)
    result[sort_idx] = adjusted
    return result


def benjamini_hochberg(p: Tensor) -> Tensor:
    """Benjamini-Hochberg (BH) FDR correction.

    Algorithm:
      1. Sort p-values ascending: p_(1) <= ... <= p_(m)
      2. Adjusted: p_adj_(i) = min_{j>=i} p_(j) * m / j
      3. Clamp to [0, 1] and map back.

    Matches R's ``p.adjust(method="BH")``.
    """
    m = p.shape[0]
    sorted_p, sort_idx = p.sort()
    ranks = torch.arange(1, m + 1, dtype=p.dtype, device=p.device)

    adjusted = sorted_p * m / ranks

    # Enforce monotonicity from right (cumulative min from the right)
    adjusted = _cummin_reverse(adjusted)
    adjusted = torch.clamp(adjusted, max=1.0)

    result = torch.empty_like(p)
    result[sort_idx] = adjusted
    return result


def benjamini_yekutieli(p: Tensor) -> Tensor:
    """Benjamini-Yekutieli (BY) FDR correction (valid under arbitrary dependence).

    Same as BH but with an additional harmonic-number factor:
      c_m = sum_{i=1}^{m} 1/i

    Adjusted: p_adj = BH_adj * c_m (then monotonicity + clamp).
    """
    m = p.shape[0]
    c_m = sum(1.0 / i for i in range(1, m + 1))

    sorted_p, sort_idx = p.sort()
    ranks = torch.arange(1, m + 1, dtype=p.dtype, device=p.device)

    adjusted = sorted_p * m * c_m / ranks
    adjusted = _cummin_reverse(adjusted)
    adjusted = torch.clamp(adjusted, max=1.0)

    result = torch.empty_like(p)
    result[sort_idx] = adjusted
    return result


def storey_qvalue(p: Tensor, lambda_: float = 0.5) -> Tensor:
    """Storey q-value with pi0 estimation.

    Estimates the proportion of true nulls (pi0) from the p-value distribution,
    then applies an adaptive BH-like procedure.

    Parameters
    ----------
    p : (m,) — raw p-values
    lambda_ : float — threshold for pi0 estimation (default 0.5)
        pi0_hat = #{p_i > lambda} / (m * (1 - lambda))

    Returns
    -------
    (m,) — q-values (adjusted p-values)
    """
    m = p.shape[0]

    # Estimate pi0
    n_above = (p > lambda_).sum().item()
    pi0 = n_above / (m * (1.0 - lambda_))
    pi0 = min(pi0, 1.0)

    if pi0 <= 0:
        return torch.zeros_like(p)

    sorted_p, sort_idx = p.sort()
    ranks = torch.arange(1, m + 1, dtype=p.dtype, device=p.device)

    adjusted = pi0 * m * sorted_p / ranks
    adjusted = _cummin_reverse(adjusted)
    adjusted = torch.clamp(adjusted, max=1.0)

    result = torch.empty_like(p)
    result[sort_idx] = adjusted
    return result


def _cummin_reverse(x: Tensor) -> Tensor:
    """Cumulative minimum from right to left."""
    flipped = x.flip(0)
    cummin_vals = torch.cummin(flipped, dim=0).values
    return cummin_vals.flip(0)
