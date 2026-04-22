"""PLINK-style LD clumping for identifying independent loci.

Greedy algorithm: sort SNPs by p-value, select the most significant as
an index SNP, remove all SNPs in LD (r^2 > threshold) within a genomic
window, repeat until no significant SNPs remain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..ld._pairwise import compute_r2_matrix


@dataclass
class ClumpResult:
    """Result from LD clumping."""

    index_snps: list[int]  # indices of index SNPs in original order
    index_p: Tensor  # (n_index,) p-values of index SNPs
    clump_members: list[list[int]]  # SNP indices clumped with each index
    n_clumps: int


def ld_clump(
    p: Tensor,
    G: Tensor,
    pos: list[int] | Tensor,
    chr_labels: list[str] | list[int],
    r2_threshold: float = 0.1,
    p_threshold: float = 5e-8,
    window_kb: float = 250.0,
    device: torch.device | str = "cpu",
) -> ClumpResult:
    """PLINK-style greedy LD clumping.

    Parameters
    ----------
    p : (m,) Tensor
        P-values for each SNP.
    G : (n, m) Tensor
        Genotype dosage matrix (for r^2 computation).
    pos : list[int] or Tensor
        Physical positions in base pairs.
    chr_labels : list[str] or list[int]
        Chromosome label for each SNP.
    r2_threshold : float
        r^2 threshold for clumping (default 0.1).
    p_threshold : float
        Only consider SNPs with p < p_threshold (default 5e-8).
    window_kb : float
        One-sided window in kilobases (default 250).
    device : torch.device or str
        Device for computation.

    Returns
    -------
    ClumpResult
    """
    p = p.to(torch.float64).to(device)
    G = G.to(torch.float64).to(device)
    m = p.shape[0]

    if isinstance(pos, Tensor):
        pos_arr = pos.cpu().tolist()
    else:
        pos_arr = list(pos)

    chr_arr = [str(c) for c in chr_labels]
    window_bp = window_kb * 1000.0

    # Group SNPs by chromosome
    chr_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chr_arr):
        chr_to_indices.setdefault(c, []).append(i)

    # Pre-compute r^2 matrices per chromosome
    chr_r2: dict[str, Tensor] = {}
    chr_local_idx: dict[str, dict[int, int]] = {}  # global -> local index
    chr_order: list[str] = list(chr_to_indices.keys())
    chr_block_id: dict[str, int] = {c: i for i, c in enumerate(chr_order)}
    for chrom, snp_indices in chr_to_indices.items():
        idx_t = torch.tensor(snp_indices, dtype=torch.long, device=device)
        G_chr = G[:, idx_t]
        chr_r2[chrom] = compute_r2_matrix(G_chr).to(device)
        chr_local_idx[chrom] = {g: l for l, g in enumerate(snp_indices)}

    # Native fast path: reuse the C+T greedy clumping kernel by mapping each
    # chromosome to a "block" of the block-diagonal LD reference. The
    # algorithm is purely deterministic so the two paths produce identical
    # ClumpResults; the C++ path eliminates the per-pair r2.item() calls.
    from .._dispatch import native_disabled
    from .._native import HAS_NATIVE_CT, _ct_native

    if (
        HAS_NATIVE_CT and not native_disabled()
        and p.device.type == "cpu" and G.device.type == "cpu"
    ):
        import numpy as _np

        p_np = p.detach().cpu().numpy().astype(_np.float64, copy=False)
        chr_codes_np = _np.asarray(
            [chr_block_id[c] for c in chr_arr], dtype=_np.int64
        )
        pos_np = _np.asarray(pos_arr, dtype=_np.int64)
        block_index_np = chr_codes_np  # one block per chromosome
        local_index_np = _np.empty(m, dtype=_np.int64)
        for chrom, snp_indices in chr_to_indices.items():
            for local_i, global_i in enumerate(snp_indices):
                local_index_np[global_i] = local_i
        # The C++ kernel computes r2 as ``R[i,j] * R[i,j]`` (it expects a
        # correlation matrix, not r-squared), so feed it sqrt(r2). Sign is
        # irrelevant since the kernel squares the values back.
        R_blocks_np = [
            chr_r2[chrom].clamp(min=0.0).sqrt().detach().cpu()
            .numpy().astype(_np.float64, copy=False)
            for chrom in chr_order
        ]

        idx_arr, off_arr, mem_arr = _ct_native.clump_block(
            p_np, R_blocks_np,
            block_index_np, local_index_np,
            chr_codes_np, pos_np,
            float(p_threshold), float(r2_threshold), float(window_bp),
        )

        index_snps_n: list[int] = [int(x) for x in idx_arr.tolist()]
        if not index_snps_n:
            return ClumpResult(
                index_snps=[], index_p=torch.tensor([]),
                clump_members=[], n_clumps=0,
            )
        offsets_n = off_arr.tolist()
        members_flat = mem_arr.tolist()
        clump_members_n: list[list[int]] = []
        for k in range(len(index_snps_n)):
            s = int(offsets_n[k])
            e = int(offsets_n[k + 1])
            clump_members_n.append([int(x) for x in members_flat[s:e]])
        index_p_n = p[torch.tensor(index_snps_n, dtype=torch.long, device=device)]
        return ClumpResult(
            index_snps=index_snps_n,
            index_p=index_p_n,
            clump_members=clump_members_n,
            n_clumps=len(index_snps_n),
        )

    # Filter to significant SNPs and sort by p-value
    sig_mask = p < p_threshold
    sig_indices = sig_mask.nonzero(as_tuple=True)[0].cpu().tolist()

    if len(sig_indices) == 0:
        return ClumpResult(
            index_snps=[], index_p=torch.tensor([]), clump_members=[], n_clumps=0
        )

    # Sort by p-value (ascending)
    sig_indices.sort(key=lambda i: p[i].item())

    available = set(range(m))  # all SNPs available for clumping
    index_snps: list[int] = []
    clump_members: list[list[int]] = []

    for idx in sig_indices:
        if idx not in available:
            continue

        # This SNP becomes an index SNP
        index_snps.append(idx)
        available.discard(idx)

        # Find and remove SNPs in LD within window on same chromosome
        chrom = chr_arr[idx]
        members: list[int] = []
        local_i = chr_local_idx[chrom][idx]

        for other_global in list(chr_to_indices[chrom]):
            if other_global == idx or other_global not in available:
                continue

            # Check window
            dist = abs(pos_arr[other_global] - pos_arr[idx])
            if dist > window_bp:
                continue

            # Check r^2
            local_j = chr_local_idx[chrom][other_global]
            r2_val = chr_r2[chrom][local_i, local_j].item()

            if r2_val >= r2_threshold:
                members.append(other_global)
                available.discard(other_global)

        clump_members.append(members)

    index_p = p[torch.tensor(index_snps, dtype=torch.long, device=device)]

    return ClumpResult(
        index_snps=index_snps,
        index_p=index_p,
        clump_members=clump_members,
        n_clumps=len(index_snps),
    )
