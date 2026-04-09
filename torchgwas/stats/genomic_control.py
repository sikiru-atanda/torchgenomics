"""Lambda_GC computation and inflation diagnostics.

Genomic control (Devlin & Roeder, 1999): lambda_GC = median(chi2) / 0.4549,
where 0.4549 is the median of the chi-squared(1) distribution.

lambda_GC ≈ 1.0 indicates well-calibrated test statistics.
lambda_GC > 1.0 suggests systematic inflation (population structure, etc.).
lambda_GC < 1.0 suggests deflation (over-correction or low power).
"""

from __future__ import annotations

import torch
from torch import Tensor

import scipy.stats as sp_stats


# Median of chi2(1) distribution
CHI2_1_MEDIAN = 0.4549364231195724  # scipy.stats.chi2.ppf(0.5, df=1)


def lambda_gc(p: Tensor) -> float:
    """Genomic control inflation factor.

    Parameters
    ----------
    p : (m,) — p-values from a GWAS scan

    Returns
    -------
    lambda_GC : float — inflation factor (1.0 = well-calibrated)
    """
    # Convert p-values to chi2(1) statistics
    p_np = p.detach().cpu().numpy()
    chi2_stats = sp_stats.chi2.isf(p_np.clip(1e-300, 1.0), df=1)

    median_chi2 = float(torch.tensor(chi2_stats).median().item())
    return median_chi2 / CHI2_1_MEDIAN


def diagnose_inflation(lambda_val: float) -> str:
    """Return a diagnostic string for the given lambda_GC value."""
    if lambda_val < 0.9:
        return "deflated"
    elif lambda_val <= 1.1:
        return "well-calibrated"
    elif lambda_val <= 1.5:
        return "mildly inflated"
    else:
        return "severely inflated"
