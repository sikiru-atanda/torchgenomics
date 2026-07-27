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
import numpy as np
import torch
from ..models.base import VariantMeta


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
            yield AncestryDosages(self.dosages[:, :, sl], self.ancestry_names), sl


def read_ancestry_dosages(prefix: str, ancestry_names: list[str] | None = None,
                          fmt: str = "auto") -> AncestryDosages:
    if fmt in ("npy", "pt") or prefix.endswith((".npy", ".pt")):
        arr = (np.load(prefix) if prefix.endswith(".npy")
               else torch.load(prefix).numpy())
        t = torch.as_tensor(arr, dtype=torch.float64)
        names = ancestry_names or [f"anc{k}" for k in range(t.shape[0])]
        return AncestryDosages(t, names)
    files = sorted(glob.glob(f"{prefix}*.dosage.txt"))
    if not files:
        raise FileNotFoundError(f"no ancestry dosage files at {prefix}*.dosage.txt")
    mats = [torch.as_tensor(np.loadtxt(f), dtype=torch.float64).T for f in files]  # -> (n, m)
    dosages = torch.stack(mats, dim=0)  # (K, n, m)
    names = ancestry_names or [f"anc{k}" for k in range(len(files))]
    return AncestryDosages(dosages, names)
