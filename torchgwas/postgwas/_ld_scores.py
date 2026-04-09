"""Per-SNP LD score computation for LDSC regression.

LD score for SNP j is the sum of r-squared values between j and all SNPs k
within a specified genomic window on the same chromosome:

    l_j = sum_{k in window} r^2(j, k)

This includes r^2(j, j) = 1, so independent SNPs have l_j = 1.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..ld._pairwise import compute_r2_matrix


def compute_ld_scores(
    G: Tensor,
    pos: list[int] | Tensor,
    chr_labels: list[str] | list[int],
    window_kb: float = 1000.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute per-SNP LD scores within a sliding genomic window.

    For each SNP j, the LD score l_j = sum of r^2(j, k) for all k on the
    same chromosome within ``window_kb`` kilobases.

    Parameters
    ----------
    G : (n, m) Tensor
        Genotype dosage matrix.
    pos : list[int] or Tensor
        Physical positions in base pairs for each of m SNPs.
    chr_labels : list[str] or list[int]
        Chromosome label for each SNP.
    window_kb : float
        One-sided window size in kilobases (default 1000 = 1 Mb).
    device : torch.device or str
        Device for computation.

    Returns
    -------
    ld_scores : (m,) float64
        Per-SNP LD scores.
    """
    G = G.to(torch.float64).to(device)
    m = G.shape[1]

    if isinstance(pos, Tensor):
        pos_arr = pos.cpu().tolist()
    else:
        pos_arr = list(pos)

    chr_arr = [str(c) for c in chr_labels]
    window_bp = window_kb * 1000.0

    ld_scores = torch.ones(m, dtype=torch.float64, device=device)

    # Group SNPs by chromosome
    chr_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chr_arr):
        chr_to_indices.setdefault(c, []).append(i)

    for chrom, snp_indices in chr_to_indices.items():
        if len(snp_indices) <= 1:
            continue  # l_j = 1 (self only)

        idx = torch.tensor(snp_indices, dtype=torch.long)
        G_chr = G[:, idx]  # (n, m_chr)
        pos_chr = [pos_arr[i] for i in snp_indices]

        # Compute full r² matrix for this chromosome
        r2_chr = compute_r2_matrix(G_chr)  # (m_chr, m_chr)

        # Apply window mask
        m_chr = len(snp_indices)
        pos_t = torch.tensor(pos_chr, dtype=torch.float64, device=device)
        # (m_chr, m_chr) distance matrix
        dist = (pos_t.unsqueeze(1) - pos_t.unsqueeze(0)).abs()
        window_mask = dist <= window_bp  # includes self (dist=0)

        # Sum r² within window for each SNP
        r2_windowed = r2_chr.to(device) * window_mask.to(torch.float64)
        scores_chr = r2_windowed.sum(dim=1)  # (m_chr,)

        ld_scores[idx] = scores_chr

    return ld_scores


def compute_cross_ld_scores(
    G: Tensor,
    pos: list[int] | Tensor,
    chr_labels: list[str] | list[int],
    window_kb: float = 1000.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Compute cross-trait LD scores for genetic correlation.

    For LDSC rg, the cross-trait LD score uses the same formula as
    univariate LD scores since both traits share the same LD structure.
    This is a convenience alias that returns the same result as
    ``compute_ld_scores`` but makes the intent explicit.

    Parameters
    ----------
    G, pos, chr_labels, window_kb, device
        Same as :func:`compute_ld_scores`.

    Returns
    -------
    ld_scores : (m,) float64
    """
    return compute_ld_scores(G, pos, chr_labels, window_kb, device)
