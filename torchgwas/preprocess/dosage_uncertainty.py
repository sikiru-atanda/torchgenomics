"""Dosage probability propagation for polyploid imputation quality.

For polyploids, dosage calling from sequencing data is uncertain (especially
at low depth). Tools like polyRAD and updog produce posterior probability
vectors P(dosage=0..k) per sample per marker. This module computes expected
dosage and dosage variance from those probabilities.
"""

from __future__ import annotations

import torch
from torch import Tensor


def expected_dosage(probs: Tensor, ploidy: int) -> Tensor:
    """Compute expected dosage E[d] from probability tensor.

    Parameters
    ----------
    probs : Tensor, shape (n, m, k+1)
        Posterior probability P(dosage = d) for d in 0..k, per sample per marker.
        Must sum to 1 along the last dimension.
    ploidy : int
        Organism ploidy level (k). probs.shape[-1] must equal k+1.

    Returns
    -------
    Tensor, shape (n, m)
        Expected dosage E[d] = sum_d d * P(d).
    """
    if probs.shape[-1] != ploidy + 1:
        raise ValueError(
            f"probs last dim ({probs.shape[-1]}) must equal ploidy+1 ({ploidy + 1})."
        )

    d_vals = torch.arange(ploidy + 1, dtype=probs.dtype, device=probs.device)
    return (probs * d_vals).sum(dim=-1)


def dosage_variance(probs: Tensor, ploidy: int) -> Tensor:
    """Compute Var[d] per sample per marker from probability tensor.

    High variance indicates uncertain dosage calls — useful as an
    imputation quality metric. Var[d] = E[d²] - E[d]².

    Parameters
    ----------
    probs : Tensor, shape (n, m, k+1)
    ploidy : int

    Returns
    -------
    Tensor, shape (n, m)
        Dosage variance per sample per marker.
    """
    if probs.shape[-1] != ploidy + 1:
        raise ValueError(
            f"probs last dim ({probs.shape[-1]}) must equal ploidy+1 ({ploidy + 1})."
        )

    d_vals = torch.arange(ploidy + 1, dtype=probs.dtype, device=probs.device)
    e_d = (probs * d_vals).sum(dim=-1)
    e_d2 = (probs * d_vals ** 2).sum(dim=-1)
    return e_d2 - e_d ** 2


def dosage_rsq(probs: Tensor, ploidy: int) -> Tensor:
    """Compute dosage R² (imputation quality) per marker.

    R² = 1 - mean(Var[d]) / (p * (1-p) * ploidy), where p is the allele
    frequency estimated from expected dosages. This approximates the Rsq/DR2
    metric used by BEAGLE and Minimac.

    Parameters
    ----------
    probs : Tensor, shape (n, m, k+1)
    ploidy : int

    Returns
    -------
    Tensor, shape (m,)
        Imputation quality per marker in [0, 1].
    """
    e_d = expected_dosage(probs, ploidy)  # (n, m)
    var_d = dosage_variance(probs, ploidy)  # (n, m)

    af = e_d.mean(dim=0) / ploidy  # (m,)
    expected_var = ploidy * af * (1.0 - af)
    expected_var = torch.clamp(expected_var, min=1e-10)

    mean_var = var_d.mean(dim=0)  # (m,)
    rsq = 1.0 - mean_var / expected_var
    return torch.clamp(rsq, min=0.0, max=1.0)
