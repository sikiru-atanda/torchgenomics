"""CSV/TSV/TXT numeric dosage reader with auto-detect orientation and delimiter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..models.base import VariantMeta
from .base import GenotypeReader
from .map_file import MapInfo, discover_map_file, read_map_file

logger = logging.getLogger(__name__)


class NumericDosageReader:
    """Reader for CSV/TSV/TXT numeric dosage files.  Implements GenotypeReader.

    Auto-detects delimiter, orientation (markers-as-rows vs samples-as-rows),
    and joins with a companion map file for marker metadata (SNP, Chr, Pos).

    Orientation heuristic: if the first column/row looks like sample IDs
    (non-numeric strings) and the bulk of the data is numeric, we infer
    samples-as-rows. Otherwise, markers-as-rows (GAPIT convention: rows =
    markers, columns = samples).
    """

    def __init__(self, path: str | Path, *, map_path: str | None = None) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Numeric dosage file not found: {p}")

        # Detect delimiter
        with open(p, "r") as _fh:
            first_line = _fh.readline()
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        else:
            sep = r"\s+"

        # Read full file
        df = pd.read_csv(p, sep=sep, index_col=0)

        # Detect orientation: if index entries are mostly non-numeric → they are IDs
        # If column headers are mostly non-numeric → they are IDs
        idx_numeric = _fraction_numeric(df.index.astype(str).tolist())
        col_numeric = _fraction_numeric([str(c) for c in df.columns])

        # Markers-as-rows: index = marker IDs, columns = sample IDs → transpose needed
        # Samples-as-rows: index = sample IDs, columns = marker IDs → no transpose
        if idx_numeric < 0.5 and col_numeric < 0.5:
            # Both look like IDs — assume GAPIT convention: markers as rows
            self._markers_as_rows = True
        elif idx_numeric < col_numeric:
            # Index has fewer numerics → index = marker/sample names
            self._markers_as_rows = True
        else:
            self._markers_as_rows = False

        if self._markers_as_rows:
            # Rows = markers, columns = samples
            self._marker_ids = df.index.astype(str).tolist()
            self._sample_ids = [str(c) for c in df.columns]
            self._dosage = df.values.astype(np.float64)  # (m, n)
        else:
            # Rows = samples, columns = markers
            self._sample_ids = df.index.astype(str).tolist()
            self._marker_ids = [str(c) for c in df.columns]
            self._dosage = df.values.astype(np.float64).T  # transpose to (m, n)

        self._n_samples = len(self._sample_ids)
        self._n_variants = len(self._marker_ids)

        # Load map file for variant metadata
        if map_path is None:
            map_path = discover_map_file(p)

        if map_path is not None:
            mi = read_map_file(map_path)
            # Match map entries to our marker IDs
            map_lookup = {s: i for i, s in enumerate(mi.snp)}
            self._vmeta = VariantMeta(
                snp=self._marker_ids,
                chr=[mi.chr[map_lookup[s]] if s in map_lookup else "0" for s in self._marker_ids],
                pos=[mi.pos[map_lookup[s]] if s in map_lookup else 0 for s in self._marker_ids],
                a1=["A"] * self._n_variants,  # placeholder — numeric files lack allele info
                a2=["B"] * self._n_variants,
            )
        else:
            # No map file: use marker IDs, fill placeholders
            self._vmeta = VariantMeta(
                snp=self._marker_ids,
                chr=["0"] * self._n_variants,
                pos=list(range(1, self._n_variants + 1)),
                a1=["A"] * self._n_variants,
                a2=["B"] * self._n_variants,
            )
            logger.warning(
                "No map file found for %s — variant positions are placeholders. "
                "Provide a .map file or use --map for proper metadata.",
                p,
            )

        logger.info(
            "NumericDosageReader: %d samples, %d variants from %s (orientation=%s)",
            self._n_samples, self._n_variants, p,
            "markers-as-rows" if self._markers_as_rows else "samples-as-rows",
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

            # dosage is (m, n), we want (n, m)
            chunk = self._dosage[start:end, :]
            G_chunk = torch.from_numpy(chunk.T.copy()).to(torch.float64)

            vmeta = VariantMeta(
                snp=self._vmeta.snp[start:end],
                chr=self._vmeta.chr[start:end],
                pos=self._vmeta.pos[start:end],
                a1=self._vmeta.a1[start:end],
                a2=self._vmeta.a2[start:end],
            )
            yield G_chunk, vmeta


def _fraction_numeric(values: list[str]) -> float:
    """Return fraction of values that parse as float."""
    if not values:
        return 0.0
    count = 0
    for v in values:
        try:
            float(v)
            count += 1
        except (ValueError, TypeError):
            pass
    return count / len(values)
