"""Region I/O: BED file parsing, region-to-variant mapping, SKAT weights.

Provides the infrastructure for set-based association tests (SKAT/Burden/SKAT-O)
to define gene regions and map them to variant column indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE

logger = logging.getLogger(__name__)


@dataclass
class Region:
    """A genomic region (gene, pathway element, LD block)."""

    region_id: str
    chr: str
    start: int  # 0-based, inclusive (BED convention)
    end: int  # 0-based, exclusive (BED convention)


def load_regions(bed_path: str | Path) -> list[Region]:
    """Parse a BED3+ file into Region objects.

    Expects tab-separated columns: chr, start, end, [name, ...].
    If a 4th column exists, it is used as region_id; otherwise
    the ID is generated as 'chr:start-end'.

    Parameters
    ----------
    bed_path : str or Path
        Path to BED file.

    Returns
    -------
    list[Region]
    """
    regions = []
    path = Path(bed_path)

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning("Skipping malformed line %d: %s", line_num, line)
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                logger.warning("Skipping line %d with non-integer coords: %s", line_num, line)
                continue

            region_id = parts[3] if len(parts) >= 4 else f"{chrom}:{start}-{end}"
            regions.append(Region(region_id=region_id, chr=chrom, start=start, end=end))

    logger.info("Loaded %d regions from %s", len(regions), path)
    return regions


def map_regions_to_variants(
    regions: list[Region],
    variant_chr: list[str],
    variant_pos: list[int],
) -> dict[str, list[int]]:
    """Map regions to variant column indices.

    For each region, find all variants where chr matches and
    start <= pos < end (BED convention: half-open interval).
    Regions with zero variants are dropped with a warning.

    Parameters
    ----------
    regions : list[Region]
    variant_chr : list[str] — chromosome per variant
    variant_pos : list[int] — position per variant

    Returns
    -------
    dict[region_id -> list[variant_col_index]]
    """
    # Build chr -> [(pos, idx)] lookup for efficiency
    chr_to_variants: dict[str, list[tuple[int, int]]] = {}
    for idx, (c, p) in enumerate(zip(variant_chr, variant_pos)):
        chr_to_variants.setdefault(c, []).append((p, idx))

    # Sort by position within each chromosome
    for c in chr_to_variants:
        chr_to_variants[c].sort()

    mapping: dict[str, list[int]] = {}
    n_empty = 0

    for region in regions:
        variants_on_chr = chr_to_variants.get(region.chr, [])
        indices = [
            idx for pos, idx in variants_on_chr
            if region.start <= pos < region.end
        ]
        if not indices:
            n_empty += 1
            continue
        mapping[region.region_id] = indices

    if n_empty > 0:
        logger.warning("%d regions had zero variants and were dropped", n_empty)

    logger.info(
        "Mapped %d regions to variants (mean %.1f variants/region)",
        len(mapping),
        np.mean([len(v) for v in mapping.values()]) if mapping else 0,
    )
    return mapping


def compute_skat_weights(
    allele_freq: Tensor,
    a1: float = 1.0,
    a2: float = 25.0,
) -> Tensor:
    """Beta(MAF; a1, a2) density weights for SKAT.

    Standard SKAT weighting upweights rare variants.
    w_j = dbeta(MAF_j, a1, a2)^2

    Parameters
    ----------
    allele_freq : (m,) allele frequencies in [0, 1].
    a1, a2 : Beta distribution shape parameters (default: 1, 25).

    Returns
    -------
    (m,) weight values (squared Beta density).
    """
    from scipy.stats import beta as beta_dist

    af = allele_freq.detach().cpu().numpy().astype(np.float64)
    maf = np.minimum(af, 1.0 - af)
    maf = np.clip(maf, 1e-10, 0.5 - 1e-10)  # avoid boundary issues

    # Beta density squared
    density = beta_dist.pdf(maf, a1, a2)
    weights = density ** 2

    return torch.tensor(weights, dtype=STAT_DTYPE, device=allele_freq.device)
