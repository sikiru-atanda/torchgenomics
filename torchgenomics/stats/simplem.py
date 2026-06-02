"""simpleM / M_eff: spectral decomposition for effective test count.

Reference: Gao et al. (2008) "A Multiple Testing Correction Method for Genetic
Association Studies Using Correlated Single Nucleotide Polymorphisms."

M_eff is estimated from the eigenvalue spectrum of the LD correlation matrix.
The idea: if all SNPs were independent, M_eff = M. With LD, M_eff < M.
Bonferroni with M_eff instead of M is less conservative but still controls FWER.
"""

from __future__ import annotations

import torch
from torch import Tensor


def effective_test_count(eigenvalues: Tensor) -> int:
    """Estimate M_eff from eigenvalue spectrum of the LD correlation matrix.

    Uses the simpleM method (Gao et al. 2008): M_eff is the number of
    eigenvalues needed to explain a specified fraction of total variance.

    Parameters
    ----------
    eigenvalues : (m,) — eigenvalues of the LD correlation matrix,
        in descending order. Should sum to m (trace of correlation matrix).

    Returns
    -------
    M_eff : int — effective number of independent tests
    """
    # Sort descending
    evals = eigenvalues.sort(descending=True).values
    evals = torch.clamp(evals, min=0.0)  # remove numerical negatives

    total = evals.sum()
    if total <= 0:
        return 1

    # Cumulative proportion of variance explained
    cumvar = torch.cumsum(evals, dim=0) / total

    # M_eff = number of eigenvalues needed to explain 99.5% of variance
    threshold = 0.995
    above = torch.where(cumvar >= threshold)[0]
    if len(above) == 0:
        return int(evals.shape[0])

    m_eff = above[0].item() + 1  # +1 because 0-indexed
    return max(m_eff, 1)


def effective_test_count_moskvina(eigenvalues: Tensor) -> int:
    """Estimate M_eff using the Moskvina & Schmidt (2008) method.

    M_eff = sum_i I(lambda_i >= 1) + sum_i (lambda_i - floor(lambda_i))
            for eigenvalues with lambda_i < 1.

    This is the default M.eff algorithm used by GWASpoly. It typically
    produces a slightly smaller M_eff than simpleM (Gao 2008), giving
    marginally more power.

    Parameters
    ----------
    eigenvalues : (m,) — eigenvalues of the LD correlation matrix.

    Returns
    -------
    M_eff : int — effective number of independent tests
    """
    evals = torch.clamp(eigenvalues, min=0.0)

    if evals.numel() == 0:
        return 1

    # Count eigenvalues >= 1
    n_large = int((evals >= 1.0).sum().item())

    # For eigenvalues < 1, sum their fractional parts
    small = evals[evals < 1.0]
    frac_sum = (small - torch.floor(small)).sum().item()

    m_eff = n_large + frac_sum
    return max(int(round(m_eff)), 1)


def ld_correlation_eigenvalues(G: Tensor) -> Tensor:
    """Compute eigenvalues of the SNP-SNP correlation matrix.

    Parameters
    ----------
    G : (n, m) — genotype matrix (n samples, m SNPs)

    Returns
    -------
    eigenvalues : (m,) — eigenvalues in descending order
    """
    n, m = G.shape

    # Standardize columns (mean 0, var 1)
    G_centered = G - G.mean(dim=0, keepdim=True)
    std = G_centered.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-10)
    G_std = G_centered / std

    # Correlation matrix = (1/n) * G_std^T @ G_std
    # For eigenvalues only, use SVD on G_std for efficiency when m > n
    if m > n:
        # Eigenvalues of G^T G / n = singular values^2 / n of G_std
        # Use the (n, n) Gram matrix instead
        gram = G_std @ G_std.T / n  # (n, n)
        evals_small = torch.linalg.eigvalsh(gram)  # (n,)
        # Pad with zeros for the m - n zero eigenvalues
        evals = torch.zeros(m, dtype=G.dtype, device=G.device)
        evals[:n] = evals_small.flip(0)  # descending
    else:
        corr = G_std.T @ G_std / n  # (m, m)
        evals = torch.linalg.eigvalsh(corr)  # ascending
        evals = evals.flip(0)  # descending

    return evals
