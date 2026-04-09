"""Zarr genotype block reader (cloud/object-store ready).

Reads chunked Zarr stores without loading the full dosage matrix into memory.
Each call to ``iter_chunks`` slices the on-disk array, so memory usage stays
bounded regardless of dataset size.

Expected Zarr store layout::

    root/
      calldata/
        dosage    (n_samples, n_variants) float32/64
      samples     (n_samples,) str
      variants/
        id        (n_variants,) str        — rs IDs
        chrom     (n_variants,) str        — chromosome
        pos       (n_variants,) int        — base-pair position
        ref       (n_variants,) str        — reference allele
        alt       (n_variants,) str        — alternate allele
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)


class ZarrReader:
    """Reader for Zarr genotype stores.  Implements GenotypeReader.

    Designed for out-of-core and cloud-native workflows.  The dosage array
    is never fully loaded — slices are read on demand in ``iter_chunks``.
    """

    def __init__(self, store_path: str | Path) -> None:
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "ZarrReader requires the 'zarr' package. "
                "Install it with: pip install 'torchgwas[zarr]'"
            )

        store_path = str(store_path)
        root = zarr.open_group(store_path, mode="r")

        # --- Dosage array (not loaded, just a reference) ---
        if "calldata" in root and "dosage" in root["calldata"]:
            self._dosage = root["calldata"]["dosage"]
        elif "dosage" in root:
            self._dosage = root["dosage"]
        else:
            raise ValueError(
                f"Zarr store '{store_path}' missing dosage array. "
                "Expected 'calldata/dosage' or 'dosage'."
            )

        self._n_samples, self._n_variants = self._dosage.shape

        # --- Sample IDs ---
        if "samples" in root:
            self._sample_ids = [str(s) for s in root["samples"][:]]
        elif "sample_ids" in root.attrs:
            self._sample_ids = list(root.attrs["sample_ids"])
        else:
            self._sample_ids = [f"SAMPLE_{i}" for i in range(self._n_samples)]
            logger.warning(
                "Zarr store has no 'samples' array — using auto-generated IDs."
            )

        if len(self._sample_ids) != self._n_samples:
            raise ValueError(
                f"Sample ID count ({len(self._sample_ids)}) does not match "
                f"dosage rows ({self._n_samples})."
            )

        # --- Variant metadata ---
        variants = root.get("variants", None)
        if variants is not None:
            self._snp_ids = [str(s) for s in variants["id"][:]] if "id" in variants else [f"VAR_{j}" for j in range(self._n_variants)]
            self._chroms = [str(s) for s in variants["chrom"][:]] if "chrom" in variants else ["0"] * self._n_variants
            self._positions = [int(p) for p in variants["pos"][:]] if "pos" in variants else list(range(self._n_variants))
            self._ref = [str(s) for s in variants["ref"][:]] if "ref" in variants else ["N"] * self._n_variants
            self._alt = [str(s) for s in variants["alt"][:]] if "alt" in variants else ["N"] * self._n_variants
        else:
            logger.warning("Zarr store has no 'variants' group — using placeholder metadata.")
            self._snp_ids = [f"VAR_{j}" for j in range(self._n_variants)]
            self._chroms = ["0"] * self._n_variants
            self._positions = list(range(self._n_variants))
            self._ref = ["N"] * self._n_variants
            self._alt = ["N"] * self._n_variants

        logger.info(
            "ZarrReader: %d samples, %d variants from %s",
            self._n_samples, self._n_variants, store_path,
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
        """Full variant metadata for the entire store."""
        return VariantMeta(
            snp=self._snp_ids,
            chr=self._chroms,
            pos=self._positions,
            a1=self._ref,
            a2=self._alt,
        )

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[Tuple[Tensor, VariantMeta]]:
        """Yield (G_chunk, variant_meta) where G_chunk is (n_samples, m) float64 dosage.

        Only the requested slice is read from disk/cloud — memory usage stays
        bounded at O(n_samples * chunk_size).
        """
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)

            # Zarr slicing: reads only this column range from disk
            chunk_np = self._dosage[:, start:end]
            G_chunk = torch.from_numpy(chunk_np.copy()).to(torch.float64)

            vmeta = VariantMeta(
                snp=self._snp_ids[start:end],
                chr=self._chroms[start:end],
                pos=self._positions[start:end],
                a1=self._ref[start:end],
                a2=self._alt[start:end],
            )

            yield G_chunk, vmeta
