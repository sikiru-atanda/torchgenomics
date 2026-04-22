"""VCF/BCF streaming reader (GT and DS fields)."""

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
    import cyvcf2  # type: ignore[import-untyped]
    _HAS_CYVCF2 = True
except ImportError:
    _HAS_CYVCF2 = False


class VCFReader:
    """Streaming reader for VCF and BCF files.  Implements GenotypeReader.

    Extracts either the GT (genotype) or DS (dosage) field depending on
    availability.  DS is preferred when present.

    Requires ``cyvcf2`` package: ``pip install cyvcf2``
    """

    def __init__(self, path: str | Path, *, field: str = "auto") -> None:
        if not _HAS_CYVCF2:
            raise ImportError(
                "VCFReader requires the 'cyvcf2' package. "
                "Install it with: pip install cyvcf2"
            )

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"VCF/BCF file not found: {p}")

        self._path = str(p)
        self._field = field  # "auto", "GT", or "DS"

        # Open once to get sample list and count variants
        vcf = cyvcf2.VCF(self._path)
        self._sample_ids = list(vcf.samples)
        self._n_samples = len(self._sample_ids)

        # Count variants (requires full scan for VCF; indexed BCF can be faster)
        self._n_variants = 0
        for _ in vcf:
            self._n_variants += 1
        vcf.close()

        logger.info(
            "VCFReader: %d samples, %d variants from %s",
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

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[tuple[Tensor, VariantMeta]]:
        vcf = cyvcf2.VCF(self._path)

        buffer_dosage: list[np.ndarray] = []
        buffer_snp: list[str] = []
        buffer_chr: list[str] = []
        buffer_pos: list[int] = []
        buffer_a1: list[str] = []
        buffer_a2: list[str] = []

        for variant in vcf:
            # Extract dosage
            dosage = self._extract_dosage(variant)
            buffer_dosage.append(dosage)
            buffer_snp.append(variant.ID or f"{variant.CHROM}:{variant.POS}")
            buffer_chr.append(str(variant.CHROM))
            buffer_pos.append(variant.POS)
            # REF = a2 (other allele), ALT[0] = a1 (effect allele) — PLINK convention
            buffer_a1.append(variant.ALT[0] if variant.ALT else ".")
            buffer_a2.append(variant.REF)

            if len(buffer_dosage) >= chunk_size:
                yield self._flush_buffer(
                    buffer_dosage, buffer_snp, buffer_chr, buffer_pos,
                    buffer_a1, buffer_a2,
                )
                buffer_dosage, buffer_snp, buffer_chr = [], [], []
                buffer_pos, buffer_a1, buffer_a2 = [], [], []

        if buffer_dosage:
            yield self._flush_buffer(
                buffer_dosage, buffer_snp, buffer_chr, buffer_pos,
                buffer_a1, buffer_a2,
            )

        vcf.close()

    def _extract_dosage(self, variant) -> np.ndarray:
        """Extract per-sample dosage from a VCF variant record.

        Preference: DS field > GT field.
        Returns shape (n_samples,) float64 array.
        """
        use_ds = self._field == "DS" or (self._field == "auto")

        if use_ds:
            try:
                ds = variant.format("DS")
                if ds is not None:
                    return ds[:, 0].astype(np.float64)
            except Exception:
                pass

        # Fall back to GT via cyvcf2's .genotypes property
        # cyvcf2 .genotypes returns list of [allele1, allele2, is_phased]
        gt_list = variant.genotypes  # list of [a1, a2, phased] per sample
        gt = np.array(gt_list, dtype=np.int32)  # (n_samples, 3)

        a1 = gt[:, 0].astype(np.float64)
        a2 = gt[:, 1].astype(np.float64)

        # -1 = missing in cyvcf2
        dosage = a1 + a2
        missing = (gt[:, 0] < 0) | (gt[:, 1] < 0)
        dosage[missing] = np.nan

        return dosage

    def _flush_buffer(
        self,
        dosage_list: list[np.ndarray],
        snp: list[str], chr_: list[str], pos: list[int],
        a1: list[str], a2: list[str],
    ) -> tuple[Tensor, VariantMeta]:
        """Stack buffered variants into a chunk."""
        dosage_arr = np.stack(dosage_list, axis=0)  # (m, n)
        G_chunk = torch.from_numpy(dosage_arr.T.copy()).to(torch.float64)

        vmeta = VariantMeta(snp=snp, chr=chr_, pos=pos, a1=a1, a2=a2)
        return G_chunk, vmeta
