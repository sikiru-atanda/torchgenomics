"""Uncertainty-aware LD computation from genotype probabilities.

When genotypes are imputed or from low-depth sequencing, point dosages
carry measurement error that attenuates observed correlations.  This
module computes corrected r² using regression calibration and weights
edges by imputation quality (dosage R²).

Uses ``torchgenomics.preprocess.dosage_uncertainty`` for E[g] and Var[g].
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-10


def compute_corrected_r2(
    G: Tensor,
    gp_probs: Tensor,
    ploidy: int = 2,
) -> Tensor:
    """Compute measurement-error corrected r² matrix.

    Naive r² from expected dosages is attenuated by dosage uncertainty.
    The correction uses the regression-calibration identity:

        r²_true ≈ r²_naive / (reliability_i * reliability_j)

    where ``reliability_j = 1 - E[Var[g_j]] / Var[E[g_j]]``.

    Parameters
    ----------
    G : (n, m) expected dosage matrix (E[g]).
    gp_probs : (n, m, k+1) genotype probability tensor.
    ploidy : int

    Returns
    -------
    r2_corrected : (m, m) corrected r² matrix, symmetric.
    """
    from ..preprocess.dosage_uncertainty import dosage_variance

    n, m = G.shape
    device = G.device
    G = G.to(dtype=torch.float64, device=device)

    # Per-sample per-marker dosage variance from GP
    var_g = dosage_variance(gp_probs.to(dtype=torch.float64, device=device), ploidy)  # (n, m)

    # Variance of expected dosage across samples
    var_Eg = G.var(dim=0, correction=1)  # (m,)
    # Mean dosage variance (measurement error variance)
    mean_var_g = var_g.mean(dim=0)  # (m,)

    # Reliability = 1 - measurement_error_var / total_var
    reliability = (1.0 - mean_var_g / var_Eg.clamp(min=_EPS)).clamp(min=_EPS, max=1.0)

    # Naive r² from expected dosages
    G_c = G - G.mean(dim=0, keepdim=True)
    std = G_c.std(dim=0, correction=1).clamp(min=_EPS)
    G_norm = G_c / std
    r_naive = G_norm.T @ G_norm / (n - 1)  # (m, m)
    r2_naive = r_naive ** 2

    # Corrected r²
    rel_outer = reliability.unsqueeze(1) * reliability.unsqueeze(0)  # (m, m)
    r2_corrected = (r2_naive / rel_outer.clamp(min=_EPS)).clamp(0.0, 1.0)

    # Diagonal should be 1
    r2_corrected.fill_diagonal_(1.0)

    return r2_corrected


def compute_edge_weights(
    r2_corrected: Tensor,
    gp_probs: Tensor,
    ploidy: int = 2,
) -> Tensor:
    """Weight LD edges by imputation quality.

    Edge weight = r²_corrected(i,j) * sqrt(q_i * q_j), where
    q = per-marker dosage R² (imputation quality in [0, 1]).

    Parameters
    ----------
    r2_corrected : (m, m) corrected r² matrix.
    gp_probs : (n, m, k+1) genotype probability tensor.
    ploidy : int

    Returns
    -------
    weights : (m, m) quality-weighted adjacency matrix.
    """
    from ..preprocess.dosage_uncertainty import dosage_rsq

    q = dosage_rsq(gp_probs.to(dtype=torch.float64), ploidy)  # (m,)
    # Geometric mean of quality scores
    q_outer = (q.unsqueeze(1) * q.unsqueeze(0)).clamp(min=0.0).sqrt()  # (m, m)
    return r2_corrected * q_outer
