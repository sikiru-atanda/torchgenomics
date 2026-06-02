"""Build LD reference panels (full or block-diagonal) for PGS methods.

Provides:
    - build_ld_reference(G, snp_meta, mode, ...): construct LDReference
    - save_ld_reference(ld, path) / load_ld_reference(path)
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .base import LDReference

SnpMeta = dict[str, list]  # {"snp": [...], "chr": [...], "pos": [...], "a1": [...], "a2": [...]}


def _standardize_genotypes(G: Tensor) -> tuple[Tensor, Tensor]:
    """Mean-center and unit-variance scale genotype columns.

    Returns ``(Gs, af)`` where ``af`` is the allele frequency assuming
    biallelic dosages in [0, 2]. Constant columns are kept as zero
    (their LD with anything is undefined; downstream Ledoit-Wolf
    shrinkage / ridge handles this).
    """
    n = G.shape[0]
    mean = G.mean(dim=0)
    af = mean / 2.0
    centered = G - mean
    var = (centered * centered).sum(dim=0) / max(n - 1, 1)
    std = torch.sqrt(var)
    safe_std = torch.where(std > 1e-12, std, torch.ones_like(std))
    Gs = centered / safe_std
    Gs[:, std <= 1e-12] = 0.0
    return Gs, af


def _ledoit_wolf_shrink(R: Tensor, intensity: float) -> Tensor:
    """Shrink correlation matrix toward identity by ``intensity``.

    ``R_shrunk = (1 - intensity) * R + intensity * I``.
    """
    if intensity <= 0.0:
        return R
    intensity = float(min(max(intensity, 0.0), 1.0))
    eye = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
    return (1.0 - intensity) * R + intensity * eye


def _full_ld_from_standardized(Gs: Tensor) -> Tensor:
    """Compute the full ``(m, m)`` correlation matrix from standardized G."""
    n = Gs.shape[0]
    R = (Gs.T @ Gs) / max(n - 1, 1)
    # Numerical symmetrize and clamp diagonal
    R = 0.5 * (R + R.T)
    diag_idx = torch.arange(R.shape[0], device=R.device)
    R[diag_idx, diag_idx] = 1.0
    return R


def build_ld_reference(
    G: Tensor,
    snp_meta: SnpMeta,
    mode: Literal["full", "block"] = "block",
    block_assignments: Tensor | list[int] | None = None,
    shrinkage: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> LDReference:
    """Construct an :class:`LDReference` from a genotype matrix.

    Parameters
    ----------
    G : Tensor of shape (n, m)
        Genotype dosages in [0, 2] for ``n`` reference samples and
        ``m`` SNPs (in genomic order).
    snp_meta : dict
        Must contain keys ``snp``, ``chr``, ``pos``, ``a1``, ``a2``,
        each a list of length ``m``.
    mode : "full" | "block"
        ``"full"`` builds an ``(m, m)`` correlation matrix; ``"block"``
        builds one ``(b_k, b_k)`` correlation matrix per block.
    block_assignments : Tensor or list[int] or None
        Required when ``mode="block"``. Length-``m`` block id per SNP.
        Block ids must be non-negative integers; SNPs sharing an id are
        grouped into the same block.
    shrinkage : float in [0, 1]
        Ledoit-Wolf-style shrinkage toward the identity matrix applied
        per block (or to the full matrix).
    device, dtype :
        Output device and dtype.

    Returns
    -------
    LDReference
    """
    if G.dim() != 2:
        raise ValueError(f"G must be 2D (n, m); got shape {tuple(G.shape)}.")
    n, m = G.shape
    for key in ("snp", "chr", "pos", "a1", "a2"):
        if key not in snp_meta:
            raise ValueError(f"snp_meta missing required key {key!r}.")
        if len(snp_meta[key]) != m:
            raise ValueError(
                f"snp_meta[{key!r}] length {len(snp_meta[key])} != m={m}."
            )

    G_dev = G.to(device=device, dtype=dtype)
    Gs, af = _standardize_genotypes(G_dev)

    if mode == "full":
        R = _full_ld_from_standardized(Gs)
        R = _ledoit_wolf_shrink(R, shrinkage)
        return LDReference(
            snp=list(snp_meta["snp"]),
            chr=list(snp_meta["chr"]),
            pos=list(snp_meta["pos"]),
            a1=list(snp_meta["a1"]),
            a2=list(snp_meta["a2"]),
            af=af,
            mode="full",
            R_full=R,
            n_ref=n,
            device=device,
            dtype=dtype,
        )

    # block mode
    if block_assignments is None:
        raise ValueError("mode='block' requires block_assignments.")
    if isinstance(block_assignments, list):
        block_assignments = torch.tensor(block_assignments, dtype=torch.long)
    block_assignments = block_assignments.to(device=device, dtype=torch.long)
    if int(block_assignments.shape[0]) != m:
        raise ValueError(
            f"block_assignments length {int(block_assignments.shape[0])} != m={m}."
        )

    unique_blocks = torch.unique(block_assignments, sorted=True)
    R_blocks: list[Tensor] = []
    new_block_index = torch.empty(m, dtype=torch.long, device=device)
    for new_k, old_k in enumerate(unique_blocks.tolist()):
        idx = torch.nonzero(block_assignments == old_k, as_tuple=False).flatten()
        Gs_b = Gs[:, idx]
        R_b = _full_ld_from_standardized(Gs_b)
        R_b = _ledoit_wolf_shrink(R_b, shrinkage)
        R_blocks.append(R_b)
        new_block_index[idx] = new_k

    return LDReference(
        snp=list(snp_meta["snp"]),
        chr=list(snp_meta["chr"]),
        pos=list(snp_meta["pos"]),
        a1=list(snp_meta["a1"]),
        a2=list(snp_meta["a2"]),
        af=af,
        mode="block",
        R_blocks=R_blocks,
        block_index=new_block_index,
        n_ref=n,
        device=device,
        dtype=dtype,
    )


def save_ld_reference(ld: LDReference, path: str) -> None:
    """Serialize an LDReference via ``torch.save``."""
    payload = {
        "snp": ld.snp,
        "chr": ld.chr,
        "pos": ld.pos,
        "a1": ld.a1,
        "a2": ld.a2,
        "af": ld.af.cpu(),
        "mode": ld.mode,
        "R_full": ld.R_full.cpu() if ld.R_full is not None else None,
        "R_blocks": (
            [b.cpu() for b in ld.R_blocks] if ld.R_blocks is not None else None
        ),
        "block_index": (
            ld.block_index.cpu() if ld.block_index is not None else None
        ),
        "n_ref": ld.n_ref,
        "dtype": str(ld.dtype),
    }
    torch.save(payload, path)


def load_ld_reference(
    path: str,
    device: torch.device | str = "cpu",
) -> LDReference:
    """Load an LDReference saved with :func:`save_ld_reference`."""
    payload = torch.load(path, weights_only=False, map_location=device)
    R_full = payload.get("R_full")
    R_blocks = payload.get("R_blocks")
    block_index = payload.get("block_index")
    if R_full is not None:
        R_full = R_full.to(device)
    if R_blocks is not None:
        R_blocks = [b.to(device) for b in R_blocks]
    if block_index is not None:
        block_index = block_index.to(device)
    dtype_str = payload.get("dtype", "torch.float64")
    dtype = getattr(torch, dtype_str.split(".")[-1], torch.float64)
    return LDReference(
        snp=list(payload["snp"]),
        chr=list(payload["chr"]),
        pos=list(payload["pos"]),
        a1=list(payload["a1"]),
        a2=list(payload["a2"]),
        af=payload["af"].to(device),
        mode=payload["mode"],
        R_full=R_full,
        R_blocks=R_blocks,
        block_index=block_index,
        n_ref=int(payload.get("n_ref", 0)),
        device=device,
        dtype=dtype,
    )
