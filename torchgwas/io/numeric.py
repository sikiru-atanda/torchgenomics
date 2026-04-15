"""CSV/TSV/TXT/Excel numeric dosage reader with strict 3-column rule.

**Recommended format (3-column rule)**: The first three columns of a CSV, TXT,
or XLSX genotype file MUST be:

1. **Marker/SNP ID** (column 0, read as the DataFrame index).
   Aliases (case-insensitive): snp, marker, rs, rs#, variant_id, id, name,
   snp_id, marker_id, locus.
2. **Chromosome** (column 1, first data column after the index).
   Aliases (case-insensitive): chr, chrom, chromosome, #chrom, chr_id.
3. **Position** (column 2, second data column after the index).
   Aliases (case-insensitive): pos, position, bp, position_bp, basepair,
   base_pair, base_position, map_position.

Optional 4th/5th columns may contain allele codes (a1/ref, a2/alt).
All remaining columns are sample dosage values.

Example::

    SNP,         Chromosome, Position_BP, SAMPLE1, SAMPLE2, ...
    rs_00001,    1,          71683,       2,       0,       ...
    rs_00002,    1,          249691,      0,       1,       ...

**Legacy fallback**: If the first two data columns do NOT match chromosome and
position aliases, the reader falls back to orientation auto-detection and
companion ``.map`` file lookup.  A warning is emitted recommending the
3-column format.

Excel files (``.xlsx``, ``.xls``) are read transparently via ``pandas``.
"""

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

# Column-name aliases (lower-cased) for the required first-three columns.
_MARKER_ALIASES = {"snp", "marker", "rs", "rs#", "variant_id", "id", "name", "snp_id", "marker_id", "locus"}
_CHR_ALIASES = {"chr", "chrom", "chromosome", "#chrom", "chr_id", "chromsome"}  # incl. common typo
_POS_ALIASES = {"pos", "position", "bp", "position_bp", "basepair", "base_pair", "base_position", "map_position"}
_ALLELE1_ALIASES = {"a1", "allele1", "effect_allele", "ref", "reference"}
_ALLELE2_ALIASES = {"a2", "allele2", "other_allele", "alt", "alternate"}


class NumericDosageReader:
    """Reader for CSV/TSV/TXT/Excel numeric dosage files.  Implements GenotypeReader.

    Two modes of operation:

    **Structured mode (3-column rule)** -- activated when the first two data
    columns (after the index) match chromosome and position aliases.  Marker
    IDs come from the index (column 0 in the original file), chromosome from
    column 1, position from column 2.  Optional allele columns (columns 3-4)
    are detected by alias.  All remaining columns are sample dosages.  The
    layout is always markers-as-rows.

    **Legacy mode** -- activated when the first two data columns do *not*
    match chromosome/position aliases.  Falls back to orientation
    auto-detection (markers-as-rows vs samples-as-rows) and companion ``.map``
    file lookup, matching the original behaviour.
    """

    def __init__(self, path: str | Path, *, map_path: str | None = None) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Numeric dosage file not found: {p}")

        df = _read_tabular(p)

        # -----------------------------------------------------------------
        # Check if the file follows the 3-column rule.
        # After _read_tabular, column 0 of the original file is the index
        # (marker IDs).  The remaining columns are in df.columns.
        # We check whether the first two data columns match chr/pos aliases.
        # -----------------------------------------------------------------
        col_lower = [str(c).strip().lower() for c in df.columns]

        has_structured_header = (
            len(col_lower) >= 2
            and col_lower[0] in _CHR_ALIASES
            and col_lower[1] in _POS_ALIASES
        )

        if has_structured_header:
            # =============================================================
            # STRUCTURED FORMAT (3-column rule)
            # =============================================================
            # Index = marker IDs, col[0] = chromosome, col[1] = position
            # Optional: col[2] = allele1, col[3] = allele2
            # Remaining columns = sample dosages
            # =============================================================
            chr_col_name = df.columns[0]
            pos_col_name = df.columns[1]

            embedded_chr = df[chr_col_name].astype(str).tolist()
            embedded_pos = df[pos_col_name].astype(int).tolist()

            drop_cols = [chr_col_name, pos_col_name]
            embedded_a1, embedded_a2 = None, None

            remaining_start = 2
            if len(col_lower) > 2 and col_lower[2] in _ALLELE1_ALIASES:
                embedded_a1 = df.iloc[:, 2].astype(str).tolist()
                drop_cols.append(df.columns[2])
                remaining_start = 3
                if len(col_lower) > 3 and col_lower[3] in _ALLELE2_ALIASES:
                    embedded_a2 = df.iloc[:, 3].astype(str).tolist()
                    drop_cols.append(df.columns[3])
                    remaining_start = 4

            df = df.drop(columns=drop_cols)

            # Layout is always markers-as-rows in structured mode.
            self._markers_as_rows = True
            self._marker_ids = df.index.astype(str).tolist()
            self._sample_ids = [str(c) for c in df.columns]
            self._dosage = df.values.astype(np.float64)  # (m, n)

            self._n_samples = len(self._sample_ids)
            self._n_variants = len(self._marker_ids)

            if self._n_samples == 0:
                raise ValueError(
                    f"No sample columns found after extracting marker/chromosome/"
                    f"position columns from {p}. Expected format: "
                    f"Marker | Chr | Pos | Sample1 | Sample2 | ..."
                )

            self._vmeta = VariantMeta(
                snp=self._marker_ids,
                chr=embedded_chr,
                pos=embedded_pos,
                a1=embedded_a1 if embedded_a1 else ["A"] * self._n_variants,
                a2=embedded_a2 if embedded_a2 else ["B"] * self._n_variants,
            )

            logger.info(
                "Structured genotype file: %d samples, %d variants, "
                "chr/pos extracted from columns (%s, %s)",
                self._n_samples, self._n_variants, chr_col_name, pos_col_name,
            )

        else:
            # =============================================================
            # LEGACY FORMAT (no structured header)
            # =============================================================
            # Fall back to orientation auto-detection + companion map file.
            # =============================================================
            logger.warning(
                "File %s does not follow the recommended 3-column format "
                "(Marker | Chromosome | Position | Sample1 | Sample2 | ...). "
                "Falling back to orientation auto-detection. For best results, "
                "restructure your file with marker ID as the first column, "
                "chromosome as the second, and position as the third.",
                p,
            )

            # --- Orientation detection ---
            idx_numeric = _fraction_numeric(df.index.astype(str).tolist())
            col_numeric = _fraction_numeric([str(c) for c in df.columns])

            # If a companion map file is available, use it as a ground-truth
            # disambiguator: whichever axis overlaps more with the map's
            # marker IDs is the marker axis.  This is more reliable than
            # the numeric-heuristic when both axes hold string IDs (GAPIT
            # mdp_numeric.txt has samples-as-rows but "33-16" / "PZB00859.1"
            # both fail the numeric test).
            _map_hint_path = map_path if map_path is not None else discover_map_file(p)
            _orientation_resolved = False
            if _map_hint_path is not None:
                try:
                    _mi_hint = read_map_file(_map_hint_path)
                    _map_snps = set(str(s) for s in _mi_hint.snp)
                    _idx_hit = sum(1 for x in df.index.astype(str) if x in _map_snps)
                    _col_hit = sum(1 for c in df.columns if str(c) in _map_snps)
                    if _idx_hit > 0 or _col_hit > 0:
                        self._markers_as_rows = _idx_hit >= _col_hit
                        _orientation_resolved = True
                except Exception:
                    pass

            if not _orientation_resolved:
                if idx_numeric < 0.5 and col_numeric < 0.5:
                    # Both look like IDs -- assume GAPIT convention: markers as rows
                    self._markers_as_rows = True
                elif idx_numeric < col_numeric:
                    self._markers_as_rows = True
                else:
                    self._markers_as_rows = False

            if self._markers_as_rows:
                self._marker_ids = df.index.astype(str).tolist()
                self._sample_ids = [str(c) for c in df.columns]
                self._dosage = df.values.astype(np.float64)  # (m, n)
            else:
                self._sample_ids = df.index.astype(str).tolist()
                self._marker_ids = [str(c) for c in df.columns]
                self._dosage = df.values.astype(np.float64).T  # transpose to (m, n)

            self._n_samples = len(self._sample_ids)
            self._n_variants = len(self._marker_ids)

            # --- Build VariantMeta from companion map file ---
            if map_path is None:
                map_path = discover_map_file(p)

            if map_path is not None:
                mi = read_map_file(map_path)
                map_lookup = {s: i for i, s in enumerate(mi.snp)}
                self._vmeta = VariantMeta(
                    snp=self._marker_ids,
                    chr=[mi.chr[map_lookup[s]] if s in map_lookup else "0" for s in self._marker_ids],
                    pos=[mi.pos[map_lookup[s]] if s in map_lookup else 0 for s in self._marker_ids],
                    a1=["A"] * self._n_variants,
                    a2=["B"] * self._n_variants,
                )
            else:
                self._vmeta = VariantMeta(
                    snp=self._marker_ids,
                    chr=["0"] * self._n_variants,
                    pos=list(range(1, self._n_variants + 1)),
                    a1=["A"] * self._n_variants,
                    a2=["B"] * self._n_variants,
                )
                logger.warning(
                    "No map file found for %s -- variant positions are "
                    "placeholders. Provide a .map file or restructure the "
                    "file to follow the 3-column rule "
                    "(Marker | Chromosome | Position | Sample1 | ...).",
                    p,
                )

        logger.info(
            "NumericDosageReader: %d samples, %d variants from %s (mode=%s)",
            self._n_samples, self._n_variants, p,
            "structured" if has_structured_header else "legacy",
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


def _read_tabular(p: Path) -> pd.DataFrame:
    """Read a tabular file into a DataFrame with index_col=0.

    Handles CSV, TSV, TXT (auto-detects delimiter) and Excel (.xlsx, .xls).
    """
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p, index_col=0)

    # Text formats -- detect delimiter
    with open(p, "r") as _fh:
        first_line = _fh.readline()
    if "\t" in first_line:
        sep = "\t"
    elif "," in first_line:
        sep = ","
    else:
        sep = r"\s+"

    return pd.read_csv(p, sep=sep, index_col=0)


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
