"""P-value calibration checks versus reference tools (GEMMA, GAPIT)."""

from __future__ import annotations

import torch
from torch import Tensor


def compare_pvalues(ours: Tensor, reference: Tensor, tolerance: float = 1e-4) -> dict[str, float | int]:
    """Compare p-values against a reference tool.

    Non-finite pairs are dropped before computing metrics. The returned
    dictionary is intentionally plain Python scalars so it can be serialized
    directly in validation reports.
    """
    ours_t = torch.as_tensor(ours, dtype=torch.float64).reshape(-1)
    ref_t = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    if ours_t.numel() != ref_t.numel():
        raise ValueError(
            f"ours and reference must have the same number of p-values; "
            f"got {ours_t.numel()} and {ref_t.numel()}."
        )

    finite = torch.isfinite(ours_t) & torch.isfinite(ref_t)
    n_total = int(ours_t.numel())
    n = int(finite.sum().item())
    if n == 0:
        raise ValueError("No finite p-value pairs to compare.")

    diff = (ours_t[finite] - ref_t[finite]).abs()
    within = diff <= float(tolerance)
    ours_f = ours_t[finite]
    ref_f = ref_t[finite]
    ours_has_variance = float(ours_f.std(unbiased=False).item()) > 0.0
    ref_has_variance = float(ref_f.std(unbiased=False).item()) > 0.0
    if n > 1 and ours_has_variance and ref_has_variance:
        corr = float(torch.corrcoef(torch.stack([ours_f, ref_f]))[0, 1].item())
    else:
        corr = float("nan")

    return {
        "n_total": n_total,
        "n_finite": n,
        "n_within_tolerance": int(within.sum().item()),
        "fraction_within_tolerance": float(within.to(torch.float64).mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "pearson_r": corr,
        "tolerance": float(tolerance),
    }
