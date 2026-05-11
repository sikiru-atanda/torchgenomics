"""Multi-format LD reference loader.

Per NA1 design spec section 3.1 and Decision 4: supports .pt (existing
TorchGWAS format) and .npz (PolyFun format) at Tier A. Each loaded
reference carries its LDReferenceMetadata for cohort-mismatch detection
at bayes-scan-rss invocation.

In-sample LD computation from a genotype panel (--geno mode) is in Task 3.
"""
from __future__ import annotations

import json
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
