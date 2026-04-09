"""Auto-detect genotype file format from magic bytes, headers, and extension."""

from __future__ import annotations

import gzip
from pathlib import Path

# Magic bytes for binary formats
_PLINK_BED_MAGIC = bytes([0x6C, 0x1B, 0x01])
_BGEN_OFFSET_SIZE = 4  # first 4 bytes = offset; real check is header parse
_PGEN_MAGIC = bytes([0x6C, 0x1B, 0x02])  # PLINK2 pgen magic

# Extension mapping (lower-cased)
_EXT_MAP: dict[str, str] = {
    ".bed": "bed",
    ".pgen": "pgen",
    ".vcf": "vcf",
    ".vcf.gz": "vcf",
    ".bcf": "bcf",
    ".bgen": "bgen",
    ".zarr": "zarr",
    ".hdf5": "hdf5",
    ".h5": "hdf5",
}

# HapMap header sentinel columns
_HAPMAP_SENTINELS = {"rs#", "alleles", "chrom"}


def detect_format(path: str | Path) -> str:
    """Return format identifier for the given genotype file.

    Priority chain (per charter Section 6c):
    1. Binary magic bytes: .bed (0x6C1B01), .pgen (0x6C1B02)
    2. File header: VCF (##fileformat=VCFv4), HapMap (rs#/alleles/chrom)
    3. Extension mapping
    4. Heuristic for CSV/TSV/TXT: numeric-only first data row → numeric dosage

    Returns one of: ``"bed"``, ``"pgen"``, ``"vcf"``, ``"bcf"``, ``"bgen"``,
    ``"hapmap"``, ``"csv"``, ``"zarr"``, ``"hdf5"``.

    Raises ``ValueError`` if the format cannot be determined.
    """
    p = Path(path)

    # --- Zarr is a directory ---
    if p.is_dir():
        # Check for .zarray/.zattrs (zarr v2) or zarr.json (zarr v3) sentinel
        if (p / ".zarray").exists() or (p / ".zattrs").exists() or (p / "zarr.json").exists():
            return "zarr"
        raise ValueError(
            f"Directory '{p}' does not appear to be a Zarr store. "
            "Specify --format explicitly."
        )

    if not p.is_file():
        raise FileNotFoundError(f"Genotype path does not exist: {p}")

    # --- 1. Magic bytes (binary formats) ---
    with open(p, "rb") as fh:
        header = fh.read(16)

    if header[:3] == _PLINK_BED_MAGIC:
        return "bed"
    if header[:3] == _PGEN_MAGIC:
        return "pgen"

    # --- 2. Text header inspection ---
    first_lines = _read_first_lines(p, n=2)
    if first_lines:
        line0 = first_lines[0]
        # VCF
        if line0.startswith("##fileformat=VCF"):
            return "vcf"
        # HapMap: header row contains rs#, alleles, chrom
        header_cols = {c.strip().lower() for c in line0.split("\t")}
        if _HAPMAP_SENTINELS.issubset(header_cols):
            return "hapmap"

    # --- 3. Extension mapping ---
    # Handle compound extensions like .vcf.gz
    suffixes = "".join(p.suffixes).lower()
    for ext, fmt in _EXT_MAP.items():
        if suffixes.endswith(ext):
            return fmt

    # --- 4. Heuristic: CSV/TSV/TXT with numeric data ---
    ext_lower = p.suffix.lower()
    if ext_lower in {".csv", ".tsv", ".txt"}:
        if first_lines and len(first_lines) > 1:
            # Check if first data row is mostly numeric
            data_row = first_lines[1].strip().split()
            if not data_row:
                # Try comma/tab split
                for sep in [",", "\t"]:
                    data_row = first_lines[1].strip().split(sep)
                    if len(data_row) > 1:
                        break
            numeric_count = sum(1 for v in data_row if _is_numeric(v))
            if len(data_row) > 0 and numeric_count / len(data_row) > 0.5:
                return "csv"
        # Even without heuristic match, these extensions default to csv
        return "csv"

    raise ValueError(
        f"Cannot determine genotype format for '{p}'. "
        f"Detected characteristics: extension={p.suffix}, "
        f"magic_bytes={header[:4].hex() if header else 'empty'}. "
        "Please specify --format explicitly."
    )


def _read_first_lines(path: Path, n: int = 2) -> list[str]:
    """Read up to *n* lines from a text file, handling gzip transparently."""
    try:
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        lines: list[str] = []
        with opener(path, "rt", errors="replace") as fh:  # type: ignore[call-overload]
            for i, line in enumerate(fh):
                if i >= n:
                    break
                lines.append(line.rstrip("\n\r"))
        return lines
    except Exception:
        return []


def _is_numeric(s: str) -> bool:
    """Return True if *s* can be parsed as a float (including NA/NaN/.)."""
    if s.upper() in {"NA", "NAN", ".", "-", ""}:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False
