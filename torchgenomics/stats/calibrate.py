"""P-value calibration checks versus reference tools (GEMMA, GAPIT)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compare_pvalues(ours: Tensor, reference: Tensor, tolerance: float = 1e-4) -> dict:
    """Compare a vector of p-values against a reference tool's output.

    Reports concordance metrics on the original p-value scale and on the
    -log10 scale (the latter is what the GWAS literature uses for plot
    agreement: Manhattan / QQ comparisons).

    Parameters
    ----------
    ours : Tensor, shape (m,)
        TorchGenomics p-values.
    reference : Tensor, shape (m,)
        Reference tool's p-values, paired index-by-index with ``ours``.
    tolerance : float
        Absolute tolerance on raw p-values for the
        ``frac_within_tolerance`` summary. Default 1e-4 (the charter
        section 4 reference-equivalence target).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``n``: int — number of paired markers used in the comparison
          (equivalent to ``n_finite``; kept for backward compatibility).
        - ``n_total``: int — total pair count, including NaN/inf rows.
        - ``n_finite``: int — pairs where *both* ``ours`` and
          ``reference`` are finite (not NaN, not inf). The summary
          metrics below are computed over these pairs only.
        - ``n_within_tolerance``: int — count of finite pairs with
          ``|ours - reference| <= tolerance``.
        - ``mean_abs_diff``: float — mean |ours - reference| over
          finite pairs.
        - ``max_abs_diff``: float — max |ours - reference| over finite
          pairs.
        - ``frac_within_tolerance``: float — fraction with
          ``|ours - reference| <= tolerance``, taken over finite pairs.
        - ``corr_neglog10``: float — Pearson correlation of -log10(p)
          (NaN-safe; returns 1.0 when both vectors are constant and
          equal). Reflects QQ / Manhattan agreement.
        - ``mean_abs_diff_neglog10``: float — mean |ΔlogP| on the
          -log10 scale.
        - ``tolerance``: float — echo of the tolerance argument.
    """
    o = ours.detach().to(torch.float64).reshape(-1)
    r = reference.detach().to(torch.float64).reshape(-1)
    if o.shape != r.shape:
        raise ValueError(
            f"compare_pvalues: shape mismatch ours={tuple(o.shape)} "
            f"reference={tuple(r.shape)}"
        )

    n_total = int(o.numel())
    if n_total == 0:
        return {
            "n": 0,
            "n_total": 0,
            "n_finite": 0,
            "n_within_tolerance": 0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "frac_within_tolerance": 1.0,
            "corr_neglog10": 1.0,
            "mean_abs_diff_neglog10": 0.0,
            "tolerance": float(tolerance),
        }

    # Pair is finite iff *both* p-values are finite (not NaN, not inf).
    finite_mask = torch.isfinite(o) & torch.isfinite(r)
    n_finite = int(finite_mask.sum().item())

    if n_finite == 0:
        return {
            "n": n_total,
            "n_total": n_total,
            "n_finite": 0,
            "n_within_tolerance": 0,
            "mean_abs_diff": float("nan"),
            "max_abs_diff": float("nan"),
            "frac_within_tolerance": float("nan"),
            "corr_neglog10": float("nan"),
            "mean_abs_diff_neglog10": float("nan"),
            "tolerance": float(tolerance),
        }

    o_finite = o[finite_mask]
    r_finite = r[finite_mask]
    abs_diff = (o_finite - r_finite).abs()
    mean_abs_diff = float(abs_diff.mean().item())
    max_abs_diff = float(abs_diff.max().item())
    within_mask = abs_diff <= tolerance
    n_within = int(within_mask.sum().item())
    frac_within = float(within_mask.to(torch.float64).mean().item())
    # Keep n for the historical "n_finite-equivalent" reading so existing
    # callers that interpret n as "the comparison count" remain correct.
    n = n_finite

    # -log10 scale: clamp to floor so a 0 p-value does not wreck the
    # correlation. The floor matches the canonical chi2_sf clamp of 1e-300.
    floor = 1e-300
    nl_o = -torch.log10(o_finite.clamp_min(floor))
    nl_r = -torch.log10(r_finite.clamp_min(floor))

    nl_diff = (nl_o - nl_r).abs()
    mean_abs_diff_nl = float(nl_diff.mean().item())

    if n == 1:
        # Pearson correlation undefined for a single point; report 1.0
        # iff the values match, else NaN.
        corr_nl = 1.0 if math.isclose(nl_o.item(), nl_r.item()) else float("nan")
    else:
        var_o = float(nl_o.var(unbiased=False).item())
        var_r = float(nl_r.var(unbiased=False).item())
        if var_o == 0.0 and var_r == 0.0:
            # Both constant on the -log10 scale.
            corr_nl = 1.0 if math.isclose(
                float(nl_o.mean().item()), float(nl_r.mean().item())
            ) else float("nan")
        elif var_o == 0.0 or var_r == 0.0:
            corr_nl = float("nan")
        else:
            cov = float(((nl_o - nl_o.mean()) * (nl_r - nl_r.mean())).mean().item())
            corr_nl = cov / math.sqrt(var_o * var_r)

    return {
        "n": n,
        "n_total": n_total,
        "n_finite": n_finite,
        "n_within_tolerance": n_within,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "frac_within_tolerance": frac_within,
        "corr_neglog10": corr_nl,
        "mean_abs_diff_neglog10": mean_abs_diff_nl,
        "tolerance": float(tolerance),
    }
