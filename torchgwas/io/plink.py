"""PLINK .bed/.bim/.fam memmap reader, SNP-major chunk iterator."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)

# PLINK BED magic bytes: 0x6C 0x1B 0x01 (SNP-major mode)
_BED_MAGIC = bytes([0x6C, 0x1B, 0x01])

# PLINK 1.9 BED 2-bit decoding — dosage = count of A1 (BIM column 5),
# matching PLINK 1.9 `--glm` and regenie, which both report BETA on the A1
# effect-allele convention.
#
#   0b00 -> 2.0 (homozygous A1 / 2 copies of A1)
#   0b01 -> NaN (missing)
#   0b10 -> 1.0 (heterozygous)
#   0b11 -> 0.0 (homozygous A2 / 0 copies of A1)
#
# Prior to 2026-05-13 this table was [0.0, NaN, 1.0, 2.0] (counting A2),
# which produced sign-flipped BETA on every BED-derived sumstats output
# vs PLINK / regenie. NA3 Track B parity vs regenie surfaced the bug
# (raw β Pearson = -1.000). VCF (`vcf.py:74`) and PLINK 2.0 (`plink2.py`)
# readers were already on the correct convention; only the BED reader
# was inverted.
_GENO_DECODE = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float64)


class PlinkBedReader:
    """Memory-mapped reader for PLINK 1.x binary fileset (.bed/.bim/.fam).

    Implements :class:`GenotypeReader`.  Reads SNP-major .bed files via
    ``numpy.memmap``, decodes 2-bit genotypes to float dosage (0/1/2 or NaN),
    and yields chunks for streaming scan.
    """

    def __init__(self, prefix: str | Path) -> None:
        prefix = Path(prefix)
        # Strip extension if user passes e.g. "data.bed"
        if prefix.suffix in {".bed", ".bim", ".fam"}:
            prefix = prefix.with_suffix("")

        bed_path = prefix.with_suffix(".bed")
        bim_path = prefix.with_suffix(".bim")
        fam_path = prefix.with_suffix(".fam")

        for p, label in [(bed_path, ".bed"), (bim_path, ".bim"), (fam_path, ".fam")]:
            if not p.is_file():
                raise FileNotFoundError(f"Missing PLINK {label} file: {p}")

        # Parse .fam → sample IDs
        self._sample_ids, self._n_samples = _parse_fam(fam_path)

        # Parse .bim → variant metadata
        self._variant_meta = _parse_bim(bim_path)
        self._n_variants = len(self._variant_meta.snp)

        # Validate .bed magic bytes
        with open(bed_path, "rb") as fh:
            magic = fh.read(3)
        if magic != _BED_MAGIC:
            raise ValueError(
                f"Invalid .bed magic bytes ({magic.hex()}). "
                "Expected SNP-major format (0x6C1B01). "
                "Individual-major .bed files are not supported — "
                "convert with: plink --bfile <prefix> --make-bed --out <prefix>"
            )

        # Bytes per SNP in SNP-major mode: ceil(n_samples / 4)
        self._bytes_per_snp = (self._n_samples + 3) // 4

        # Memory-map the .bed file (skip 3-byte header)
        expected_size = 3 + self._bytes_per_snp * self._n_variants
        actual_size = bed_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f".bed file size mismatch: expected {expected_size} bytes "
                f"({self._n_samples} samples × {self._n_variants} variants), "
                f"got {actual_size} bytes."
            )

        self._bed_mmap = np.memmap(
            bed_path, dtype=np.uint8, mode="r", offset=3,
            shape=(self._n_variants, self._bytes_per_snp),
        )

        logger.info(
            "PlinkBedReader: %d samples, %d variants from %s",
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
        return self._sample_ids

    @property
    def variant_meta(self) -> VariantMeta:
        """Full variant metadata for the entire file."""
        return self._variant_meta

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[tuple[Tensor, VariantMeta]]:
        """Yield (G_chunk, variant_meta) where G_chunk is (n_samples, m) float64 dosage.

        Decoding: each byte holds 4 genotypes (2 bits each, LSB first).
        PLINK coding: 00=hom_A1(0), 01=missing(NaN), 10=het(1), 11=hom_A2(2).
        """
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)
            m = end - start

            # Read raw bytes for this chunk: (m, bytes_per_snp)
            raw = np.array(self._bed_mmap[start:end], dtype=np.uint8)

            # Decode 2-bit genotypes → (m, n_samples) dosage
            dosage = _decode_bed_chunk(raw, self._n_samples)

            # Convert to tensor: (n_samples, m) — transpose from SNP-major to sample-major
            G_chunk = torch.from_numpy(dosage.T.copy()).to(torch.float64)

            # Slice variant metadata
            vmeta = VariantMeta(
                snp=self._variant_meta.snp[start:end],
                chr=self._variant_meta.chr[start:end],
                pos=self._variant_meta.pos[start:end],
                a1=self._variant_meta.a1[start:end],
                a2=self._variant_meta.a2[start:end],
            )

            yield G_chunk, vmeta


def _decode_bed_chunk(raw: np.ndarray, n_samples: int) -> np.ndarray:
    """Decode a (m, bytes_per_snp) uint8 array to (m, n_samples) float64 dosage.

    Each byte encodes 4 genotypes in 2-bit pairs, LSB first:
      byte & 0x03 → sample 4k
      (byte >> 2) & 0x03 → sample 4k+1
      (byte >> 4) & 0x03 → sample 4k+2
      (byte >> 6) & 0x03 → sample 4k+3
    """
    m, bps = raw.shape

    # Unpack all 4 genotypes per byte at once
    g0 = raw & 0x03
    g1 = (raw >> 2) & 0x03
    g2 = (raw >> 4) & 0x03
    g3 = (raw >> 6) & 0x03

    # Interleave: for each byte position, 4 consecutive samples
    # Shape: (m, bps * 4)
    unpacked = np.empty((m, bps * 4), dtype=np.uint8)
    unpacked[:, 0::4] = g0
    unpacked[:, 1::4] = g1
    unpacked[:, 2::4] = g2
    unpacked[:, 3::4] = g3

    # Trim to actual sample count (last byte may have padding)
    unpacked = unpacked[:, :n_samples]

    # Map 2-bit codes to dosage via lookup table
    dosage = _GENO_DECODE[unpacked]

    return dosage


def _parse_fam(path: Path) -> tuple[list[str], int]:
    """Parse PLINK .fam file → (sample_ids, n_samples).

    .fam columns: FID IID father mother sex phenotype
    We use IID (column 1) as the sample identifier.
    """
    sample_ids: list[str] = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sample_ids.append(parts[1])  # IID

    if not sample_ids:
        raise ValueError(f"Empty .fam file: {path}")

    # Check for duplicate IIDs
    if len(sample_ids) != len(set(sample_ids)):
        seen: set[str] = set()
        dups = [s for s in sample_ids if s in seen or seen.add(s)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"Duplicate IIDs in .fam file: {dups[:5]}{'...' if len(dups) > 5 else ''}"
        )

    return sample_ids, len(sample_ids)


def _parse_bim(path: Path) -> VariantMeta:
    """Parse PLINK .bim file → VariantMeta.

    .bim columns: chr  snp  cm  pos  a1  a2
    """
    chrs: list[str] = []
    snps: list[str] = []
    positions: list[int] = []
    a1s: list[str] = []
    a2s: list[str] = []

    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                parts = line.strip().split()
            if len(parts) < 6:
                continue
            chrs.append(parts[0])
            snps.append(parts[1])
            # parts[2] = cM (ignored)
            positions.append(int(parts[3]))
            a1s.append(parts[4])
            a2s.append(parts[5])

    if not snps:
        raise ValueError(f"Empty .bim file: {path}")

    return VariantMeta(snp=snps, chr=chrs, pos=positions, a1=a1s, a2=a2s)
