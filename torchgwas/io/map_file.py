"""Companion map-file reader (SNP, Chr, Pos); auto-discovery logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MapInfo:
    """Parsed marker metadata from a companion map file."""

    snp: list[str]
    chr: list[str]
    pos: list[int]
    cm: list[float] | None = None  # centiMorgan genetic distances


def read_map_file(path: str | Path) -> MapInfo:
    """Read a map file and return structured marker metadata.

    Accepts two layouts:
    - PLINK .map (4 columns): chr  snp  cm  pos
    - Simple 3-column: snp  chr  pos  (or chr  snp  pos with header)

    Heuristic: if the file has a header row with recognised column names
    (SNP/rs, Chr/Chrom, Pos/Position/BP), use those; otherwise assume
    PLINK .map format.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Map file not found: {p}")

    lines = p.read_text().strip().splitlines()
    if not lines:
        raise ValueError(f"Empty map file: {p}")

    # Detect header
    first_tokens = lines[0].split("\t") if "\t" in lines[0] else lines[0].split()
    header_lower = [t.strip().lower() for t in first_tokens]

    # Known header column names
    snp_names = {"snp", "rs", "rs#", "marker", "variant_id", "id"}
    chr_names = {"chr", "chrom", "chromosome", "#chrom"}
    pos_names = {"pos", "position", "bp", "basepair"}

    has_header = bool(
        (set(header_lower) & snp_names) and (set(header_lower) & chr_names)
    )

    if has_header:
        return _parse_with_header(lines, header_lower, snp_names, chr_names, pos_names)
    else:
        return _parse_plink_map(lines)


def _parse_with_header(
    lines: list[str],
    header_lower: list[str],
    snp_names: set[str],
    chr_names: set[str],
    pos_names: set[str],
) -> MapInfo:
    """Parse map file using detected header columns."""
    snp_idx = next(i for i, h in enumerate(header_lower) if h in snp_names)
    chr_idx = next(i for i, h in enumerate(header_lower) if h in chr_names)
    pos_idx = next(i for i, h in enumerate(header_lower) if h in pos_names)

    snps, chrs, positions = [], [], []
    sep = "\t" if "\t" in lines[0] else None
    for line in lines[1:]:
        parts = line.split(sep)
        if len(parts) <= max(snp_idx, chr_idx, pos_idx):
            continue
        snps.append(parts[snp_idx].strip())
        chrs.append(parts[chr_idx].strip())
        positions.append(int(parts[pos_idx].strip()))

    return MapInfo(snp=snps, chr=chrs, pos=positions)


def _parse_plink_map(lines: list[str]) -> MapInfo:
    """Parse PLINK .map format: chr  snp  cm  pos (4 cols) or chr snp pos (3 cols)."""
    snps, chrs, positions, cm_values = [], [], [], []
    has_cm = False
    for line in lines:
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 4:
            # PLINK .map: chr snp cm pos
            chrs.append(parts[0])
            snps.append(parts[1])
            cm_values.append(float(parts[2]))
            positions.append(int(parts[3]))
            has_cm = True
        elif len(parts) == 3:
            # Minimal: chr snp pos
            chrs.append(parts[0])
            snps.append(parts[1])
            positions.append(int(parts[2]))
        else:
            continue

    if not snps:
        raise ValueError("No valid rows found in map file.")

    return MapInfo(snp=snps, chr=chrs, pos=positions,
                   cm=cm_values if has_cm else None)


def discover_map_file(genotype_path: str | Path) -> str | None:
    """Attempt to find a companion map file adjacent to the genotype file.

    Searches for files with the same stem and extensions: .map, .snpinfo,
    _map.txt, _snpinfo.txt in the same directory.
    """
    p = Path(genotype_path)
    stem = p.stem
    # Strip double extensions like .vcf from .vcf.gz
    if stem.endswith(".vcf"):
        stem = stem[:-4]

    parent = p.parent
    candidates = [
        parent / f"{stem}.map",
        parent / f"{stem}.bim",
        parent / f"{stem}_map.txt",
        parent / f"{stem}_snpinfo.txt",
        parent / f"{stem}.snpinfo",
    ]

    for c in candidates:
        if c.is_file():
            logger.info("Discovered companion map file: %s", c)
            return str(c)

    return None
