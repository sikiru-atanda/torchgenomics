"""Apply PGS weights to target genotypes.

Given a :class:`PGSResult` (per-SNP weights with effect alleles) and a
target genotype matrix, compute ``PGS_i = sum_j w_j * g_ij`` with:

- SNP-id intersection between weights and target,
- allele alignment (effect-allele matching with sign flip on swap),
- missing-genotype handling (``mean`` or ``drop``),
- optional standardization,
- optional chunked streaming over SNPs (keeps GPU memory bounded).

The function is intentionally simple: target genotypes are provided as
an ``(n_target, m_target)`` tensor (e.g. loaded via
``torchgenomics.io``). Streaming from disk formats is handled by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .base import PGSResult, _alleles_match


@dataclass
class ScoringResult:
    """Per-individual polygenic score plus alignment audit."""

    pgs: Tensor  # (n_target,)
    n_snp_used: int
    n_snp_missing: int
    n_snp_flipped: int
    snp_used: list[str]


def score_individuals(
    G: Tensor,
    target_snp: list[str],
    target_a1: list[str],
    target_a2: list[str],
    pgs_result: PGSResult,
    standardize: bool = False,
    handle_missing: Literal["mean", "drop"] = "mean",
    chunk_size: int = 50_000,
) -> ScoringResult:
    """Compute ``PGS_i = sum_j w_j * g_ij`` on aligned target genotypes.

    Parameters
    ----------
    G : Tensor of shape (n_target, m_target)
        Target genotype dosages in ``[0, ploidy]``. NaNs are allowed and
        handled via ``handle_missing``.
    target_snp, target_a1, target_a2 : list[str]
        Length-``m_target`` metadata for the target SNPs.
    pgs_result : PGSResult
        PGS weights with effect-allele annotation.
    standardize : bool
        If True, standardize each used SNP column to unit variance before
        scoring (helpful for comparing magnitudes across cohorts).
    handle_missing : "mean" | "drop"
        ``"mean"`` imputes NaN dosages with the column mean; ``"drop"``
        excludes SNPs that contain any NaN in the target.
    chunk_size : int
        SNP-chunk size for matmul. Reduces peak memory on large ``m``.

    Returns
    -------
    ScoringResult
    """
    if G.dim() != 2:
        raise ValueError(f"G must be 2D; got shape {tuple(G.shape)}.")
    n_target, m_target = G.shape
    if len(target_snp) != m_target:
        raise ValueError(
            f"target_snp length {len(target_snp)} != G.shape[1]={m_target}."
        )

    # Build index into target by SNP id
    target_idx: dict[str, int] = {s: i for i, s in enumerate(target_snp)}

    # Align weight SNPs to target
    used_target_idx: list[int] = []
    used_weight: list[float] = []
    used_snp: list[str] = []
    n_flipped = 0
    n_missing_snp = 0

    weight_np = pgs_result.weight.detach().cpu()

    for j, snp in enumerate(pgs_result.snp):
        ti = target_idx.get(snp)
        if ti is None:
            n_missing_snp += 1
            continue
        matched, sign = _alleles_match(
            pgs_result.a1[j],
            pgs_result.a2[j],
            target_a1[ti],
            target_a2[ti],
        )
        if not matched:
            n_missing_snp += 1
            continue
        if sign == -1:
            n_flipped += 1
        used_target_idx.append(ti)
        used_weight.append(float(weight_np[j].item()) * sign)
        used_snp.append(snp)

    if not used_target_idx:
        return ScoringResult(
            pgs=torch.zeros(n_target, dtype=G.dtype, device=G.device),
            n_snp_used=0,
            n_snp_missing=n_missing_snp,
            n_snp_flipped=n_flipped,
            snp_used=[],
        )

    idx_t = torch.tensor(used_target_idx, dtype=torch.long, device=G.device)
    w_t = torch.tensor(used_weight, dtype=G.dtype, device=G.device)

    G_used = G.index_select(1, idx_t)

    if handle_missing == "drop":
        # Drop columns with any NaN
        col_ok = ~torch.isnan(G_used).any(dim=0)
        G_used = G_used[:, col_ok]
        w_t = w_t[col_ok]
        used_snp = [used_snp[i] for i in range(len(used_snp)) if bool(col_ok[i].item())]
    else:  # mean imputation
        nan_mask = torch.isnan(G_used)
        if nan_mask.any():
            col_means = torch.where(
                nan_mask, torch.zeros_like(G_used), G_used
            ).sum(dim=0) / (~nan_mask).sum(dim=0).clamp(min=1).to(G_used.dtype)
            G_used = torch.where(nan_mask, col_means.unsqueeze(0).expand_as(G_used), G_used)

    if standardize:
        means = G_used.mean(dim=0, keepdim=True)
        sds = G_used.std(dim=0, keepdim=True).clamp(min=1e-12)
        G_used = (G_used - means) / sds

    # Chunked matmul over SNPs
    m_used = int(G_used.shape[1])
    pgs = torch.zeros(n_target, dtype=G.dtype, device=G.device)
    for start in range(0, m_used, chunk_size):
        end = min(start + chunk_size, m_used)
        pgs = pgs + G_used[:, start:end] @ w_t[start:end]

    return ScoringResult(
        pgs=pgs,
        n_snp_used=int(G_used.shape[1]),
        n_snp_missing=n_missing_snp,
        n_snp_flipped=n_flipped,
        snp_used=list(used_snp),
    )
