"""PLINK 2.x .pgen/.pvar/.psam reader with dosage and phased support."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)

try:
    import pgenlib  # type: ignore[import-untyped]
    _HAS_PGENLIB = True
except ImportError:
    _HAS_PGENLIB = False


class Plink2PgenReader:
    """Reader for PLINK 2 .pgen fileset.  Implements GenotypeReader.

    Uses ``pgenlib`` for efficient access to .pgen binary format.
    Falls back to error if pgenlib is unavailable.

    Requires ``pgenlib`` package: ``pip install Pgenlib``
    """

    def __init__(self, prefix: str | Path) -> None:
        if not _HAS_PGENLIB:
            raise ImportError(
                "Plink2PgenReader requires the 'pgenlib' package. "
                "Install it with: pip install Pgenlib"
            )

        prefix = Path(prefix)
        if prefix.suffix in {".pgen", ".pvar", ".psam"}:
            prefix = prefix.with_suffix("")

        pgen_path = prefix.with_suffix(".pgen")
        pvar_path = prefix.with_suffix(".pvar")
        psam_path = prefix.with_suffix(".psam")

        for p, label in [(pgen_path, ".pgen"), (pvar_path, ".pvar"), (psam_path, ".psam")]:
            if not p.is_file():
                raise FileNotFoundError(f"Missing PLINK2 {label} file: {p}")

        # Parse .psam → sample IDs
        self._sample_ids = _parse_psam(psam_path)
        self._n_samples = len(self._sample_ids)

        # Parse .pvar → variant metadata
        self._variant_meta = _parse_pvar(pvar_path)
        self._n_variants = len(self._variant_meta.snp)

        # Open pgen reader
        self._pgen = pgenlib.PgenReader(
            bytes(str(pgen_path), encoding="utf-8"),
            raw_sample_ct=self._n_samples,
        )

        logger.info(
            "Plink2PgenReader: %d samples, %d variants from %s",
            self._n_samples, self._n_variants, prefix,
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

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[tuple[Tensor, VariantMeta]]:
        buf = np.empty(self._n_samples, dtype=np.int8)

        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)
            m = end - start

            dosage_arr = np.empty((m, self._n_samples), dtype=np.float64)

            for i, vid in enumerate(range(start, end)):
                self._pgen.read(vid, buf)
                # pgenlib encodes: 0=hom_ref, 1=het, 2=hom_alt, -9=missing
                row = buf.astype(np.float64)
                row[buf == -9] = np.nan
                dosage_arr[i] = row

            G_chunk = torch.from_numpy(dosage_arr.T.copy()).to(torch.float64)

            vmeta = VariantMeta(
                snp=self._variant_meta.snp[start:end],
                chr=self._variant_meta.chr[start:end],
                pos=self._variant_meta.pos[start:end],
                a1=self._variant_meta.a1[start:end],
                a2=self._variant_meta.a2[start:end],
            )
            yield G_chunk, vmeta

    def __del__(self) -> None:
        if hasattr(self, "_pgen"):
            self._pgen.close()


def _parse_psam(path: Path) -> list[str]:
    """Parse PLINK2 .psam file → sample IDs.

    .psam has a header line starting with #IID or #FID IID.
    """
    sample_ids: list[str] = []
    header_seen = False

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                header_seen = True
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if not parts:
                continue
            # With header (#FID IID ...): IID is column 1
            # Without header: assume FID IID format like .fam, IID is column 1
            if len(parts) > 1:
                sample_ids.append(parts[1])
            else:
                sample_ids.append(parts[0])

    return sample_ids


def _parse_pvar(path: Path) -> VariantMeta:
    """Parse PLINK2 .pvar file → VariantMeta.

    .pvar columns: #CHROM  POS  ID  REF  ALT  [QUAL  FILTER  INFO]
    """
    chrs, snps, positions, a1s, a2s = [], [], [], [], []

    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                parts = line.strip().split()
            if len(parts) < 5:
                continue

            chrs.append(parts[0])
            positions.append(int(parts[1]))
            snps.append(parts[2])
            a2s.append(parts[3])  # REF
            a1s.append(parts[4])  # ALT (effect allele)

    if not snps:
        raise ValueError(f"No variants found in .pvar file: {path}")

    return VariantMeta(snp=snps, chr=chrs, pos=positions, a1=a1s, a2=a2s)
