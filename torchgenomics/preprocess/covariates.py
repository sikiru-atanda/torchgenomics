"""Covariate construction: intercept, PCs, batch effects."""

from __future__ import annotations

import torch
from torch import Tensor


def build_covariate_matrix(
    *,
    n_samples: int,
    covariates: Tensor | None = None,
    n_pcs: int = 0,
    eigenvectors: Tensor | None = None,
) -> Tensor:
    """Build the full covariate matrix X0 with intercept as first column.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    covariates : Tensor, shape (n, c1), optional
        User-supplied covariates (sex, age, batch, etc.).
    n_pcs : int
        Number of principal components to include from eigenvectors.
    eigenvectors : Tensor, shape (n, k), optional
        Eigenvectors from GRM decomposition. First *n_pcs* columns are used.

    Returns
    -------
    Tensor, shape (n, c)
        Full covariate matrix where c = 1 + c1 + n_pcs.
        First column is always the intercept (all 1s).
    """
    parts: list[Tensor] = []

    # Intercept (always first)
    intercept = torch.ones(n_samples, 1, dtype=torch.float64)
    parts.append(intercept)

    # User covariates
    if covariates is not None:
        if covariates.shape[0] != n_samples:
            raise ValueError(
                f"Covariate row count ({covariates.shape[0]}) does not match "
                f"n_samples ({n_samples})."
            )
        parts.append(covariates.to(torch.float64))

    # Principal components
    if n_pcs > 0:
        if eigenvectors is None:
            raise ValueError("n_pcs > 0 but no eigenvectors provided.")
        if eigenvectors.shape[0] != n_samples:
            raise ValueError(
                f"Eigenvector row count ({eigenvectors.shape[0]}) does not match "
                f"n_samples ({n_samples})."
            )
        if n_pcs > eigenvectors.shape[1]:
            raise ValueError(
                f"Requested {n_pcs} PCs but only {eigenvectors.shape[1]} eigenvectors available."
            )
        parts.append(eigenvectors[:, :n_pcs].to(torch.float64))

    return torch.cat(parts, dim=1)
