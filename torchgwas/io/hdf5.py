"""HDF5 genotype reader with chunked on-demand I/O.

Reads HDF5 files without loading the full dosage matrix into memory.
Each call to ``iter_chunks`` slices the on-disk dataset, so memory usage
stays bounded regardless of dataset size.

Expected HDF5 layout::

    /calldata/dosage    (n_samples, n_variants) float32/64
    /samples            (n_samples,) str
    /variants/id        (n_variants,) str
    /variants/chrom     (n_variants,) str
    /variants/pos       (n_variants,) int
    /variants/ref       (n_variants,) str
    /variants/alt       (n_variants,) str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)


class HDF5Reader:
    """Reader for HDF5 genotype files.  Implements GenotypeReader.

    The dosage dataset is accessed via h5py slicing — only the requested
    column range is read from disk per chunk.
    """

    def __init__(self, path: str | Path) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "HDF5Reader requires the 'h5py' package. "
                "Install it with: pip install 'torchgwas[hdf5]'"
            )

        self._path = str(path)
        self._file = h5py.File(self._path, "r")

        # --- Dosage dataset (not loaded, just a reference) ---
        if "calldata" in self._file and "dosage" in self._file["calldata"]:
            self._dosage = self._file["calldata"]["dosage"]
        elif "dosage" in self._file:
            self._dosage = self._file["dosage"]
        else:
            self._file.close()
            raise ValueError(
                f"HDF5 file '{path}' missing dosage dataset. "
                "Expected '/calldata/dosage' or '/dosage'."
            )

        self._n_samples, self._n_variants = self._dosage.shape

        # --- Sample IDs ---
        if "samples" in self._file:
            raw = self._file["samples"][:]
            self._sample_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
        elif "sample_ids" in self._file.attrs:
            self._sample_ids = list(self._file.attrs["sample_ids"])
        else:
            self._sample_ids = [f"SAMPLE_{i}" for i in range(self._n_samples)]
            logger.warning(
                "HDF5 file has no 'samples' dataset — using auto-generated IDs."
            )

        if len(self._sample_ids) != self._n_samples:
            self._file.close()
            raise ValueError(
                f"Sample ID count ({len(self._sample_ids)}) does not match "
                f"dosage rows ({self._n_samples})."
            )

        # --- Variant metadata ---
        if "variants" in self._file:
            vg = self._file["variants"]
            self._snp_ids = _read_str_dataset(vg, "id", self._n_variants, "VAR_")
            self._chroms = _read_str_dataset(vg, "chrom", self._n_variants, "0", default_val="0")
            self._positions = [int(p) for p in vg["pos"][:]] if "pos" in vg else list(range(self._n_variants))
            self._ref = _read_str_dataset(vg, "ref", self._n_variants, "N", default_val="N")
            self._alt = _read_str_dataset(vg, "alt", self._n_variants, "N", default_val="N")
        else:
            logger.warning("HDF5 file has no 'variants' group — using placeholder metadata.")
            self._snp_ids = [f"VAR_{j}" for j in range(self._n_variants)]
            self._chroms = ["0"] * self._n_variants
            self._positions = list(range(self._n_variants))
            self._ref = ["N"] * self._n_variants
            self._alt = ["N"] * self._n_variants

        logger.info(
            "HDF5Reader: %d samples, %d variants from %s",
            self._n_samples, self._n_variants, self._path,
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
        return VariantMeta(
            snp=self._snp_ids,
            chr=self._chroms,
            pos=self._positions,
            a1=self._ref,
            a2=self._alt,
        )

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[Tuple[Tensor, VariantMeta]]:
        """Yield (G_chunk, variant_meta) where G_chunk is (n_samples, m) float64 dosage.

        Only the requested slice is read from disk — memory usage stays
        bounded at O(n_samples * chunk_size).
        """
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)

            # h5py slicing: reads only this column range from disk
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

    def close(self) -> None:
        """Close the underlying HDF5 file handle."""
        try:
            if hasattr(self, "_file") and self._file.id.valid:
                self._file.close()
        except Exception:
            pass  # Suppress errors during interpreter shutdown

    def __del__(self) -> None:
        self.close()


def _read_str_dataset(group, name: str, n: int, prefix: str, default_val: str = "") -> list[str]:
    """Read a string dataset from an HDF5 group, handling bytes decoding."""
    if name not in group:
        if default_val:
            return [default_val] * n
        return [f"{prefix}{j}" for j in range(n)]
    raw = group[name][:]
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
