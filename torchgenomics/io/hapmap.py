"""HapMap format reader with letter genotype decoding (IUPAC + polyploid integer)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..models.base import VariantMeta

logger = logging.getLogger(__name__)

# IUPAC ambiguity codes → constituent bases
_IUPAC: dict[str, tuple[str, str]] = {
    "R": ("A", "G"), "Y": ("C", "T"), "S": ("G", "C"),
    "W": ("A", "T"), "K": ("G", "T"), "M": ("A", "C"),
}


class HapMapReader:
    """Reader for HapMap-format genotype files.  Implements GenotypeReader.

    Expected header columns (case-insensitive):
      rs#  alleles  chrom  pos  strand  assembly#  ...  <sample_1>  <sample_2>  ...

    Genotype values per sample: letter pairs (AA, AT, TT) or IUPAC codes (R, Y, etc.)
    or numeric integers for polyploid (0, 1, 2, 3, 4).
    """

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"HapMap file not found: {p}")

        # Read full file; HapMap files are typically small enough
        with open(p) as _fh:
            _first = _fh.readline()
        sep = "\t" if "\t" in _first else r"\s+"
        self._df = pd.read_csv(p, sep=sep, dtype=str)

        # Normalise only known metadata column names to lowercase;
        # preserve original case for sample ID columns.
        _meta_names_lower = {"rs#", "alleles", "chrom", "pos", "strand", "assembly#",
                             "center", "protlsid", "assaylsid", "panellsid", "qccode"}
        col_map = {}
        for c in self._df.columns:
            stripped = c.strip()
            if stripped.lower() in _meta_names_lower:
                col_map[c] = stripped.lower()
            else:
                col_map[c] = stripped  # preserve original case
        self._df.rename(columns=col_map, inplace=True)

        # Identify mandatory metadata columns
        for required in ["rs#", "alleles", "chrom", "pos"]:
            if required not in self._df.columns:
                raise ValueError(f"HapMap file missing required column: {required}")

        # Sample columns: everything not in known metadata set
        self._sample_cols = [c for c in self._df.columns if c not in _meta_names_lower]
        if not self._sample_cols:
            raise ValueError("No sample columns found in HapMap file.")

        self._sample_ids = self._sample_cols
        self._n_samples = len(self._sample_ids)
        self._n_variants = len(self._df)

        logger.info(
            "HapMapReader: %d samples, %d variants from %s",
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
        for start in range(0, self._n_variants, chunk_size):
            end = min(start + chunk_size, self._n_variants)
            chunk_df = self._df.iloc[start:end]

            alleles_col = chunk_df["alleles"].tolist()
            geno_data = chunk_df[self._sample_cols].values  # (m, n) string array

            dosage = np.full((end - start, self._n_samples), np.nan, dtype=np.float64)

            for i in range(end - start):
                allele_str = alleles_col[i]  # e.g. "A/T"
                a1, a2 = _parse_alleles(allele_str)
                for j in range(self._n_samples):
                    dosage[i, j] = _decode_genotype(geno_data[i, j], a1, a2)

            G_chunk = torch.from_numpy(dosage.T.copy()).to(torch.float64)

            vmeta = VariantMeta(
                snp=chunk_df["rs#"].tolist(),
                chr=chunk_df["chrom"].tolist(),
                pos=[int(x) for x in chunk_df["pos"].tolist()],
                a1=[_parse_alleles(a)[0] for a in alleles_col],
                a2=[_parse_alleles(a)[1] for a in alleles_col],
            )
            yield G_chunk, vmeta


def _parse_alleles(allele_str: str) -> tuple[str, str]:
    """Parse alleles string like 'A/T' into (a1, a2)."""
    parts = allele_str.split("/")
    if len(parts) != 2:
        logger.warning("Malformed alleles field '%s' — expected 'A1/A2' format.", allele_str)
        return ("?", "?")
    return (parts[0].strip(), parts[1].strip())


def _decode_genotype(geno: str, a1: str, a2: str) -> float:
    """Decode a single genotype string to dosage count of a2.

    Handles: letter pairs (AA, AT, TT), IUPAC codes (R, Y, ...),
    numeric strings (0, 1, 2, ...), and missing values (N, NN, NA, -).
    """
    geno = str(geno).strip().upper()

    # Missing values
    if geno in {"N", "NN", "NA", "NAN", "-", ".", "--", ""}:
        return np.nan

    # Numeric (polyploid integer encoding)
    try:
        return float(geno)
    except ValueError:
        pass

    a1u, a2u = a1.upper(), a2.upper()

    # IUPAC single-letter code
    if len(geno) == 1 and geno in _IUPAC:
        bases = _IUPAC[geno]
        if set(bases) == {a1u, a2u}:
            return 1.0  # heterozygote
        elif bases[0] == bases[1] == a2u:
            return 2.0
        elif bases[0] == bases[1] == a1u:
            return 0.0
        return np.nan

    # Two-letter genotype (e.g., "AA", "AT", "TT")
    if len(geno) == 2:
        count = sum(1 for b in geno if b == a2u)
        return float(count)

    return np.nan
