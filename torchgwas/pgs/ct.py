"""Clumping + Thresholding (C+T) PGS method.

A simple but consistently strong baseline: for a given p-value
threshold, perform LD clumping using the supplied :class:`LDReference`
and assign each retained index SNP its marginal effect from sumstats.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .._native import HAS_NATIVE_CT, _ct_native
from ..postgwas._sumstats import SumStats
from .base import BasePGSMethod, LDReference, PGSResult


def _native_enabled() -> bool:
    """Whether the native C++ C+T clumping accelerator should be used.

    Disabled when the extension was not built or when the user sets
    ``TORCHGWAS_DISABLE_NATIVE=1`` in the environment.
    """
    return HAS_NATIVE_CT and not os.environ.get("TORCHGWAS_DISABLE_NATIVE")


def _clump_with_ld_reference(
    p: Tensor,
    ld_ref: LDReference,
    chr_labels: list[str],
    pos: list[int],
    r2_threshold: float = 0.1,
    p_threshold: float = 5e-8,
    window_kb: float = 250.0,
) -> tuple[list[int], list[list[int]]]:
    """Greedy LD clumping using an LDReference instead of raw genotypes.

    Returns ``(index_snps, clump_members)`` where indices are positions
    into the harmonized LD-reference / sumstats SNP order.
    """
    m = int(p.shape[0])
    if m == 0:
        return [], []
    sig_mask = p < p_threshold
    sig_indices = sig_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    if not sig_indices:
        return [], []

    # Sort candidate SNPs by ascending p-value
    sig_indices.sort(key=lambda i: float(p[i].item()))

    window_bp = window_kb * 1000.0

    # Helper that returns r^2 between two SNPs in LD reference order
    def _r2(i: int, j: int) -> float:
        if ld_ref.mode == "full":
            assert ld_ref.R_full is not None
            return float(ld_ref.R_full[i, j].item() ** 2)
        # block mode: only nonzero if same block
        assert ld_ref.R_blocks is not None and ld_ref.block_index is not None
        bi = int(ld_ref.block_index[i].item())
        bj = int(ld_ref.block_index[j].item())
        if bi != bj:
            return 0.0
        # within-block local indices
        block_mask = ld_ref.block_index == bi
        local_pos = torch.nonzero(block_mask, as_tuple=False).flatten()
        local_i = int((local_pos == i).nonzero(as_tuple=False).item())
        local_j = int((local_pos == j).nonzero(as_tuple=False).item())
        return float(ld_ref.R_blocks[bi][local_i, local_j].item() ** 2)

    # Pre-compute chromosome groups for window filtering
    chr_to_idx: dict[str, list[int]] = {}
    for i, c in enumerate(chr_labels):
        chr_to_idx.setdefault(str(c), []).append(i)

    available = set(range(m))
    index_snps: list[int] = []
    clump_members: list[list[int]] = []

    for idx in sig_indices:
        if idx not in available:
            continue
        index_snps.append(idx)
        available.discard(idx)

        members: list[int] = []
        chrom = str(chr_labels[idx])
        for other in list(chr_to_idx.get(chrom, ())):
            if other == idx or other not in available:
                continue
            dist = abs(int(pos[other]) - int(pos[idx]))
            if dist > window_bp:
                continue
            if _r2(idx, other) >= r2_threshold:
                members.append(other)
                available.discard(other)
        clump_members.append(members)

    return index_snps, clump_members


def _clump_with_ld_reference_dispatch(
    p: Tensor,
    ld_ref: LDReference,
    chr_labels: list[str],
    pos: list[int],
    r2_threshold: float = 0.1,
    p_threshold: float = 5e-8,
    window_kb: float = 250.0,
) -> tuple[list[int], list[list[int]]]:
    """Dispatcher: prefer the native C++ C+T greedy clumping when available;
    otherwise fall through to the pure-Python reference implementation in
    :func:`_clump_with_ld_reference`.

    Both paths produce identical clumping results (the algorithm is purely
    deterministic given the inputs); the C++ path just avoids per-element
    ``.item()`` round-trips that dominate the Python implementation.
    """
    m = int(p.shape[0])
    if m == 0 or not _native_enabled():
        return _clump_with_ld_reference(
            p, ld_ref,
            chr_labels=chr_labels, pos=pos,
            r2_threshold=r2_threshold,
            p_threshold=p_threshold,
            window_kb=window_kb,
        )

    p_np = p.detach().to(torch.float64).cpu().numpy()
    chr_to_code: dict[str, int] = {}
    chr_codes_list: list[int] = []
    for c in chr_labels:
        key = str(c)
        code = chr_to_code.get(key)
        if code is None:
            code = len(chr_to_code)
            chr_to_code[key] = code
        chr_codes_list.append(code)
    chr_codes_np = np.asarray(chr_codes_list, dtype=np.int64)
    pos_np = np.asarray(pos, dtype=np.int64)
    window_bp = float(window_kb) * 1000.0

    if ld_ref.mode == "full":
        assert ld_ref.R_full is not None
        R_np = ld_ref.R_full.detach().to(torch.float64).cpu().numpy()
        idx_arr, off_arr, mem_arr = _ct_native.clump_full(
            p_np, R_np, chr_codes_np, pos_np,
            float(p_threshold), float(r2_threshold), window_bp,
        )
    else:
        assert ld_ref.R_blocks is not None and ld_ref.block_index is not None
        R_blocks_np = [
            R_b.detach().to(torch.float64).cpu().numpy() for R_b in ld_ref.R_blocks
        ]
        block_index_np = ld_ref.block_index.detach().to(torch.int64).cpu().numpy()
        # Per-SNP local position within its block (precomputed once).
        local_index_np = np.empty(m, dtype=np.int64)
        counters = np.zeros(len(R_blocks_np), dtype=np.int64)
        for j in range(m):
            b = int(block_index_np[j])
            local_index_np[j] = counters[b]
            counters[b] += 1
        idx_arr, off_arr, mem_arr = _ct_native.clump_block(
            p_np, R_blocks_np,
            block_index_np, local_index_np,
            chr_codes_np, pos_np,
            float(p_threshold), float(r2_threshold), window_bp,
        )

    index_snps = [int(x) for x in idx_arr.tolist()]
    members_flat = mem_arr.tolist()
    offsets = off_arr.tolist()
    clump_members: list[list[int]] = []
    for k in range(len(index_snps)):
        s = int(offsets[k])
        e = int(offsets[k + 1])
        clump_members.append([int(x) for x in members_flat[s:e]])
    return index_snps, clump_members


class ClumpingThresholding(BasePGSMethod):
    """Clumping + Thresholding PGS method.

    Parameters at ``fit`` time:
        p_threshold (float): p-value cutoff
        r2_threshold (float): clumping r^2 cutoff
        window_kb (float): one-sided clumping window in kb
    """

    name = "ct"

    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        p_threshold: float = 5e-8,
        r2_threshold: float = 0.1,
        window_kb: float = 250.0,
        **kwargs: Any,
    ) -> PGSResult:
        ss, ref, audit = self._harmonize(sumstats, ld_ref)

        index_snps, clump_members = _clump_with_ld_reference_dispatch(
            ss.p,
            ref,
            chr_labels=ss.chr,
            pos=ss.pos,
            r2_threshold=r2_threshold,
            p_threshold=p_threshold,
            window_kb=window_kb,
        )

        # Build a dense weight vector at LD-reference resolution: zero
        # everywhere except at retained index SNPs.
        m = ss.m
        weight = torch.zeros(m, dtype=ss.beta.dtype, device=ss.beta.device)
        if index_snps:
            idx_t = torch.tensor(index_snps, dtype=torch.long, device=ss.beta.device)
            weight[idx_t] = ss.beta[idx_t]

        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=weight,
            af=ss.af,
            grid={
                "p_threshold": float(p_threshold),
                "r2_threshold": float(r2_threshold),
                "window_kb": float(window_kb),
                "n_index_snps": len(index_snps),
            },
            converged=True,
            **audit,
        )
