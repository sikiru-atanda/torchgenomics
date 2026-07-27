"""Reader for ancestry-partitioned genotype dosages (Tractor ExtractTracts).

Local ancestry inference (RFMix2) and dosage extraction (Tractor
ExtractTracts.py) are UPSTREAM and out of scope: this module consumes their
output. Tractor writes one file per ancestry, `anc{k}.dosage.txt`, rows =
variants, cols = samples. We also accept `.npy`/`.pt` stacks of shape (K,n,m).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import glob
import re
import numpy as np
import torch
from ..models.base import VariantMeta


def _slice_variant_meta(vmeta: VariantMeta, sl: slice) -> VariantMeta:
    """Slice every field of a `VariantMeta` by the same variant range."""
    return VariantMeta(
        snp=vmeta.snp[sl],
        chr=vmeta.chr[sl],
        pos=vmeta.pos[sl],
        a1=vmeta.a1[sl],
        a2=vmeta.a2[sl],
    )


def _ancestry_sort_key(path: str):
    """Numeric sort key parsed from an `anc{k}` filename.

    Falls back to a tuple that sorts lexicographically (after all numerically
    keyed entries) when the filename doesn't match the `anc{int}` pattern, so
    K >= 10 ancestries (anc10, anc11, ...) sort after anc9 rather than before
    it.
    """
    m = re.search(r"anc(\d+)", path)
    if m is not None:
        return (0, int(m.group(1)), path)
    return (1, 0, path)


@dataclass
class AncestryDosages:
    dosages: torch.Tensor        # (K, n, m)
    ancestry_names: list[str]
    variant_meta: VariantMeta | None = None

    def n_ancestries(self) -> int:
        return self.dosages.shape[0]

    def iter_chunks(self, chunk_size: int) -> Iterator[tuple["AncestryDosages", slice]]:
        m = self.dosages.shape[2]
        for start in range(0, m, chunk_size):
            sl = slice(start, min(start + chunk_size, m))
            chunk_meta = (
                _slice_variant_meta(self.variant_meta, sl)
                if self.variant_meta is not None else None
            )
            yield AncestryDosages(self.dosages[:, :, sl], self.ancestry_names, chunk_meta), sl


def _check_ancestry_names(ancestry_names: list[str] | None, k: int) -> None:
    if ancestry_names is not None and len(ancestry_names) != k:
        raise ValueError(
            f"ancestry_names has length {len(ancestry_names)} but {k} "
            f"ancestries were found; lengths must match."
        )


def read_ancestry_dosages(prefix: str, ancestry_names: list[str] | None = None,
                          fmt: str = "auto") -> AncestryDosages:
    if fmt in ("npy", "pt") or prefix.endswith((".npy", ".pt")):
        arr = (np.load(prefix) if prefix.endswith(".npy")
               else torch.load(prefix, map_location="cpu").numpy())
        t = torch.as_tensor(arr, dtype=torch.float64)
        _check_ancestry_names(ancestry_names, t.shape[0])
        names = ancestry_names or [f"anc{k}" for k in range(t.shape[0])]
        return AncestryDosages(t, names)
    files = sorted(glob.glob(f"{prefix}*.dosage.txt"), key=_ancestry_sort_key)
    if not files:
        raise FileNotFoundError(f"no ancestry dosage files at {prefix}*.dosage.txt")
    _check_ancestry_names(ancestry_names, len(files))
    mats = [torch.as_tensor(np.loadtxt(f), dtype=torch.float64).T for f in files]  # -> (n, m)
    dosages = torch.stack(mats, dim=0)  # (K, n, m)
    names = ancestry_names or [f"anc{k}" for k in range(len(files))]
    return AncestryDosages(dosages, names)
