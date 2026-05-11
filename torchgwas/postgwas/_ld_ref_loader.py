"""Multi-format LD reference loader.

Per NA1 design spec section 3.1 and Decision 4: supports .pt (existing
TorchGWAS format) and .npz (PolyFun format) at Tier A. Each loaded
reference carries its LDReferenceMetadata for cohort-mismatch detection
at bayes-scan-rss invocation.

In-sample LD computation from a genotype panel (--geno mode) is in Task 3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchgwas.postgwas._ld_ref_metadata import LDReferenceMetadata


class LDReferenceUnsupportedFormatError(ValueError):
    """Raised when an LD reference file extension is not supported."""


def save_ld_reference(
    path: Path | str,
    R: torch.Tensor,
    snp_ids: list[str],
    meta: LDReferenceMetadata,
) -> None:
    """Save an LD reference to disk in the format inferred from extension.

    Args:
        path: Output path. Extension determines format: .pt or .npz.
        R: LD correlation matrix, shape (p, p), dtype float64 recommended.
        snp_ids: List of p SNP identifiers in order matching R rows/columns.
        meta: LDReferenceMetadata to stamp into the file.

    Raises:
        LDReferenceUnsupportedFormatError: if extension is not .pt or .npz.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        torch.save(
            {
                "R": R.cpu(),
                "snp_ids": snp_ids,
                "metadata": meta.to_dict(),
            },
            str(path),
        )
    elif suffix == ".npz":
        np.savez(
            str(path),
            R=R.cpu().numpy(),
            snp_ids=np.array(snp_ids, dtype=object),
            metadata_json=json.dumps(meta.to_dict()),
        )
    else:
        raise LDReferenceUnsupportedFormatError(
            f"Unsupported LD reference format: {suffix!r}. "
            f"Tier A supports .pt and .npz only; .bcor (FINEMAP) is Tier B."
        )


def load_ld_reference(
    path: Path | str,
) -> tuple[torch.Tensor, list[str], LDReferenceMetadata]:
    """Load an LD reference from disk.

    Args:
        path: Input path; format inferred from extension.

    Returns:
        (R, snp_ids, metadata) tuple.

    Raises:
        LDReferenceUnsupportedFormatError: if extension is not .pt or .npz.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        R = d["R"]
        snp_ids = list(d["snp_ids"])
        meta = LDReferenceMetadata.from_dict(d["metadata"])
        return R, snp_ids, meta

    if suffix == ".npz":
        d = np.load(str(path), allow_pickle=True)
        R = torch.from_numpy(d["R"])
        snp_ids = list(d["snp_ids"])
        # metadata_json is stored as JSON; parse safely (no eval -- eval with
        # {"__builtins__": {}} is NOT a sandbox; an attacker-controlled .npz
        # would otherwise enable RCE via .__class__.__mro__ escapes).
        meta_dict = json.loads(str(d["metadata_json"]))
        meta = LDReferenceMetadata.from_dict(meta_dict)
        return R, snp_ids, meta

    raise LDReferenceUnsupportedFormatError(
        f"Unsupported LD reference format: {suffix!r}. "
        f"Tier A supports .pt and .npz only; .bcor (FINEMAP) is Tier B."
    )


def compute_in_sample_ld(G: torch.Tensor) -> torch.Tensor:
    """Compute the LD correlation matrix from an in-sample genotype panel.

    Args:
        G: Genotype matrix, shape (n, p), dtype float64 (or upcast internally).

    Returns:
        R: LD correlation matrix, shape (p, p), dtype matching G after upcast.

    Notes:
        Constant columns (zero variance) produce NaN correlations on the
        corresponding rows/columns, matching numpy.corrcoef's behavior.
        Caller should handle NaN downstream (e.g., drop constant SNPs).
    """
    # Upcast to float64 for numerical stability of the centering + scaling
    G64 = G.to(torch.float64)
    # Center columns
    G_centered = G64 - G64.mean(dim=0, keepdim=True)
    # Compute standard deviations (ddof=1 for sample std; matches numpy.corrcoef)
    n = G64.shape[0]
    if n < 2:
        raise ValueError(
            f"compute_in_sample_ld requires n >= 2 samples; got n={n}"
        )
    std = G_centered.std(dim=0, keepdim=True, unbiased=True)
    # Scale (NaN where std is 0 — matches numpy.corrcoef)
    G_scaled = G_centered / std
    # R = (G_scaled^T @ G_scaled) / (n - 1)
    R = (G_scaled.T @ G_scaled) / (n - 1)
    return R


@dataclass(frozen=True)
class BlockSpec:
    """Specifies an LD block as a half-open interval [start, stop) over SNP indices."""
    start: int
    stop: int


def decompose_into_blocks(
    R: torch.Tensor,
    snp_ids: list[str],
    regions: list[BlockSpec] | None,
    max_block_size: int = 5000,
) -> list[BlockSpec]:
    """Decompose an LD reference into per-block sub-references.

    Per NA1 spec section 2.6: when the locus exceeds the block-size
    threshold (default 5000 per PolyFun / ldetect convention), the LD ref
    is decomposed into LD blocks. Per-block IBSS is mathematically equivalent
    to dense IBSS when block boundaries are at near-zero-LD positions.

    SIGNATURE NOTE: ``torchgwas.ld.detect_blocks`` requires raw genotypes +
    positions + chromosomes, which are unavailable for ``--ld-ref`` mode
    (we only have R + snp_ids). Tier A therefore uses fixed-size tiling
    as the fallback path when ``regions`` is None and ``p > max_block_size``.
    Genotype-aware auto-detection (``--geno`` mode) is wired into the CLI
    layer in Task 11, which calls ``detect_blocks`` directly on G and
    passes the result as explicit ``regions`` to this function.

    Args:
        R: LD correlation matrix, shape (p, p).
        snp_ids: List of p SNP identifiers in order.
        regions: Explicit block specs; if None and p > max_block_size, fall
            back to fixed-size tiling. If None and p <= max_block_size,
            returns a single block covering [0, p).
        max_block_size: Threshold above which auto-detection / fallback
            tiling kicks in.

    Returns:
        List of BlockSpec covering [0, p) without overlap or gaps.
    """
    p = R.shape[0]
    if regions is not None:
        return list(regions)

    if p <= max_block_size:
        return [BlockSpec(start=0, stop=p)]

    # Fixed-size tiling fallback (Tier A path)
    blocks = []
    start = 0
    while start < p:
        stop = min(start + max_block_size, p)
        blocks.append(BlockSpec(start=start, stop=stop))
        start = stop
    return blocks
