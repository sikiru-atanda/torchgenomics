"""BGEN reader (.bgen + .sample + .bgi index)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch import Tensor

from ..models.base import VariantMeta
from .base import GenotypeReader

logger = logging.getLogger(__name__)

try:
    from bgen_reader import open_bgen  # type: ignore[import-untyped]
    _HAS_BGEN = True
except ImportError:
    _HAS_BGEN = False


class BGENReader:
    """Reader for BGEN format with optional .bgi index.  Implements GenotypeReader.

    Converts genotype probabilities to expected dosage:
      dosage = 0*P(AA) + 1*P(AB) + 2*P(BB)

    Requires ``bgen-reader`` package: ``pip install bgen-reader``
    """

    def __init__(self, path: str | Path, *, sample_path: str | None = None) -> None:
        if not _HAS_BGEN:
            raise ImportError(
                "BGENReader requires the 'bgen-reader' package. "
                "Install it with: pip install bgen-reader"
            )

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"BGEN file not found: {p}")

        self._bgen = open_bgen(str(p), verbose=False)

        # Sample IDs
        if sample_path is not None:
            self._sample_ids = _parse_sample_file(sample_path)
        else:
            self._sample_ids = list(self._bgen.samples)

        self._n_samples = len(self._sample_ids)
        self._n_variants = self._bgen.nvariants

        # Cache variant metadata
        self._rsids = list(self._bgen.rsids)
        self._chroms = [str(c) for c in self._bgen.chromosomes]
        self._positions = [int(p) for p in self._bgen.positions]
        self._a1_alleles = list(self._bgen.allele_ids)  # "A,G" format

        logger.info(
            "BGENReader: %d samples, %d variants from %s",
            self._n_samples, self._n_variants, p,
        )

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def n_variants(self) -> int:
        return self._n_variants

    @property
    def sample_ids(self) -> list[str]:
        return list(self._sample_ids)

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[Tuple[Tensor, VariantMeta]]:
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)

            # Read genotype probabilities for this range
            probs = self._bgen.read(slice(start, end))  # (m, n, 3) typically

            # Convert to dosage: 0*P(AA) + 1*P(AB) + 2*P(BB)
            if probs.ndim == 3 and probs.shape[2] == 3:
                dosage = probs[:, :, 1] + 2.0 * probs[:, :, 2]
            else:
                # Fallback: assume dosage is directly stored
                dosage = probs.squeeze()

            dosage = dosage.astype(np.float64)
            G_chunk = torch.from_numpy(dosage.T.copy()).to(torch.float64)

            # Parse alleles
            a1_list, a2_list = [], []
            for i in range(start, end):
                alleles = self._a1_alleles[i].split(",")
                a1_list.append(alleles[1] if len(alleles) > 1 else ".")
                a2_list.append(alleles[0] if alleles else ".")

            vmeta = VariantMeta(
                snp=self._rsids[start:end],
                chr=self._chroms[start:end],
                pos=self._positions[start:end],
                a1=a1_list,
                a2=a2_list,
            )
            yield G_chunk, vmeta


def _parse_sample_file(path: str) -> list[str]:
    """Parse BGEN .sample file (Oxford format) → list of sample IDs."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Sample file not found: {p}")

    lines = p.read_text().strip().splitlines()
    # First two lines are header in Oxford .sample format
    if len(lines) < 3:
        raise ValueError(f"Sample file too short: {p}")

    sample_ids = []
    for line in lines[2:]:
        parts = line.split()
        if parts:
            sample_ids.append(parts[0])

    return sample_ids
