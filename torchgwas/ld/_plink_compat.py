"""PLINK block output parsing and format conversion.

Reads PLINK 1.9 .blocks.det files and converts between PLINK and
torchgwas block representations for concordance analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..io.regions import Region
from ._blocks import LDBlock

logger = logging.getLogger(__name__)


@dataclass
class PLINKBlock:
    """A block from PLINK .blocks.det output."""
    chr: str
    bp1: int        # start position (1-based inclusive in PLINK)
    bp2: int        # end position (1-based inclusive in PLINK)
    kb: float       # block span in kb
    n_snps: int     # number of SNPs in block
    snps: list[str] # SNP IDs within block


def load_plink_blocks_det(path: str | Path) -> list[PLINKBlock]:
    """Parse a PLINK .blocks.det file.

    PLINK 1.9 .blocks.det format (whitespace-delimited, header line):
        CHR  BP1  BP2  KB  NSNPS  SNPS

    Where SNPS is a pipe-delimited list of variant IDs.

    Parameters
    ----------
    path : str or Path
        Path to .blocks.det file.

    Returns
    -------
    list[PLINKBlock]
    """
    blocks = []
    p = Path(path)

    with open(p, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header line
            if line_num == 1 and line.upper().startswith("CHR"):
                continue

            parts = line.split()
            if len(parts) < 6:
                logger.warning("Skipping malformed line %d: %s", line_num, line)
                continue

            try:
                chrom = parts[0]
                bp1 = int(parts[1])
                bp2 = int(parts[2])
                kb = float(parts[3])
                n_snps = int(parts[4])
                snps = parts[5].split("|")
                blocks.append(PLINKBlock(
                    chr=chrom, bp1=bp1, bp2=bp2, kb=kb,
                    n_snps=n_snps, snps=snps,
                ))
            except (ValueError, IndexError) as e:
                logger.warning("Skipping line %d: %s (%s)", line_num, line, e)
                continue

    logger.info("Loaded %d PLINK blocks from %s", len(blocks), p)
    return blocks


def plink_blocks_to_ldblocks(
    plink_blocks: list[PLINKBlock],
    variant_ids: list[str] | None = None,
) -> list[LDBlock]:
    """Convert PLINK blocks to LDBlock objects.

    Parameters
    ----------
    plink_blocks : list[PLINKBlock]
    variant_ids : list[str], optional
        Full list of variant IDs for index resolution. If provided,
        ``variant_indices`` will be populated by matching SNP IDs.

    Returns
    -------
    list[LDBlock]
    """
    # Build ID -> index map if variant_ids provided
    id_to_idx: dict[str, int] = {}
    if variant_ids:
        id_to_idx = {vid: i for i, vid in enumerate(variant_ids)}

    result: list[LDBlock] = []
    for pb in plink_blocks:
        # PLINK uses 1-based inclusive; convert to BED 0-based half-open
        start = pb.bp1 - 1  # 0-based inclusive
        end = pb.bp2        # 0-based exclusive (since PLINK bp2 is inclusive)

        region_id = f"{pb.chr}:{pb.bp1}-{pb.bp2}"
        region = Region(region_id=region_id, chr=pb.chr, start=start, end=end)

        # Resolve variant indices from SNP IDs
        indices: list[int] = []
        if id_to_idx:
            indices = sorted(
                id_to_idx[sid] for sid in pb.snps if sid in id_to_idx
            )

        result.append(LDBlock(
            region=region,
            n_variants=pb.n_snps,
            variant_indices=indices,
            method="plink",
            mean_r2=0.0,   # PLINK .blocks.det doesn't report r²
            mean_dprime=0.0,
        ))

    return sorted(result, key=lambda b: (b.region.chr, b.region.start))


def ldblocks_to_plink_det(
    blocks: list[LDBlock],
    variant_ids: list[str] | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Convert LDBlocks to PLINK .blocks.det format string.

    Parameters
    ----------
    blocks : list[LDBlock]
    variant_ids : list[str], optional
        Full list of variant IDs for SNP name resolution.
    output_path : str or Path, optional
        If provided, write to file.

    Returns
    -------
    str
        The formatted .blocks.det content.
    """
    lines = ["CHR\tBP1\tBP2\tKB\tNSNPS\tSNPS"]
    for block in blocks:
        r = block.region
        bp1 = r.start + 1  # convert to 1-based
        bp2 = r.end         # PLINK end is inclusive
        kb = (bp2 - bp1 + 1) / 1000.0

        # Resolve SNP names
        if variant_ids and block.variant_indices:
            snp_names = [variant_ids[i] for i in block.variant_indices]
        else:
            snp_names = [r.region_id]
        snps_str = "|".join(snp_names)

        lines.append(
            f"{r.chr}\t{bp1}\t{bp2}\t{kb:.3f}\t"
            f"{block.n_variants}\t{snps_str}"
        )

    content = "\n".join(lines) + "\n"

    if output_path:
        p = Path(output_path)
        with open(p, "w") as f:
            f.write(content)
        logger.info("Saved %d blocks to %s (PLINK .blocks.det format)", len(blocks), p)

    return content
