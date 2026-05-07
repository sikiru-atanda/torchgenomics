"""Format conversion utilities."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def convert(
    input_path: str,
    output_path: str,
    output_format: str,
    *,
    sample_path: str | None = None,
    map_path: str | None = None,
) -> None:
    """Convert genotype data between formats.

    Reads the input using the appropriate reader, then writes to the
    target format.  Currently supports output to: bed, zarr, vcf.

    Parameters
    ----------
    input_path : str
        Path to input genotype file.
    output_path : str
        Path for output file (prefix for PLINK filesets).
    output_format : str
        Target format: "bed", "zarr", or "vcf".
    sample_path : str, optional
        Path to .sample file (for BGEN input).
    map_path : str, optional
        Path to companion map file (for CSV input).
    """
    from .detect import detect_format
    from .validate import _open_reader

    fmt = detect_format(input_path)
    reader = _open_reader(input_path, fmt)

    if output_format == "bed":
        _write_plink_bed(reader, output_path)
    elif output_format == "zarr":
        _write_zarr(reader, output_path)
    elif output_format == "vcf":
        _write_vcf(reader, output_path)
    else:
        raise ValueError(
            f"Conversion to '{output_format}' not yet supported. "
            f"Supported: bed, zarr, vcf"
        )

    logger.info("Converted %s (%s) → %s (%s)", input_path, fmt, output_path, output_format)


def _write_plink_bed(reader, output_prefix: str) -> None:
    """Write genotype data to PLINK BED/BIM/FAM format."""
    prefix = Path(output_prefix)
    if prefix.suffix in {".bed", ".bim", ".fam"}:
        prefix = prefix.with_suffix("")

    # Collect all chunks
    all_dosage = []
    all_meta_snp, all_meta_chr, all_meta_pos = [], [], []
    all_meta_a1, all_meta_a2 = [], []

    for G_chunk, vmeta in reader.iter_chunks(chunk_size=reader.n_variants):
        all_dosage.append(G_chunk.numpy())  # (n, m)
        all_meta_snp.extend(vmeta.snp)
        all_meta_chr.extend(vmeta.chr)
        all_meta_pos.extend(vmeta.pos)
        all_meta_a1.extend(vmeta.a1)
        all_meta_a2.extend(vmeta.a2)

    dosage = np.concatenate(all_dosage, axis=1)  # (n, m)
    n, m = dosage.shape

    # Write .fam
    with open(prefix.with_suffix(".fam"), "w") as fh:
        for sid in reader.sample_ids:
            fh.write(f"{sid} {sid} 0 0 0 -9\n")

    # Write .bim
    with open(prefix.with_suffix(".bim"), "w") as fh:
        for i in range(m):
            fh.write(f"{all_meta_chr[i]}\t{all_meta_snp[i]}\t0\t{all_meta_pos[i]}\t{all_meta_a1[i]}\t{all_meta_a2[i]}\n")

    # Write .bed (SNP-major)
    bytes_per_snp = (n + 3) // 4
    with open(prefix.with_suffix(".bed"), "wb") as fh:
        # Magic bytes
        fh.write(bytes([0x6C, 0x1B, 0x01]))

        for j in range(m):
            col = dosage[:, j]
            byte_arr = np.zeros(bytes_per_snp, dtype=np.uint8)

            for i in range(n):
                val = col[i]
                if np.isnan(val):
                    code = 0b01  # missing
                elif val == 0.0:
                    code = 0b00  # hom A1
                elif val == 1.0:
                    code = 0b10  # het
                elif val == 2.0:
                    code = 0b11  # hom A2
                else:
                    # Round to nearest integer genotype
                    rounded = round(val)
                    code = [0b00, 0b10, 0b11][min(max(rounded, 0), 2)]

                byte_idx = i // 4
                bit_offset = (i % 4) * 2
                byte_arr[byte_idx] |= code << bit_offset

            fh.write(byte_arr.tobytes())

    logger.info("Wrote PLINK BED: %s (.bed/.bim/.fam), %d samples × %d variants", prefix, n, m)


def _write_zarr(reader, output_path: str) -> None:
    """Write genotype data to Zarr store."""
    try:
        import zarr
    except ImportError:
        raise ImportError("Zarr output requires: pip install zarr")

    all_dosage = []
    for G_chunk, _ in reader.iter_chunks(chunk_size=reader.n_variants):
        all_dosage.append(G_chunk.numpy())

    dosage = np.concatenate(all_dosage, axis=1)  # (n, m)

    store = zarr.open(output_path, mode="w")
    store.create_dataset("dosage", data=dosage, chunks=(min(1000, dosage.shape[0]), min(1000, dosage.shape[1])))
    store.attrs["n_samples"] = dosage.shape[0]
    store.attrs["n_variants"] = dosage.shape[1]
    store.attrs["sample_ids"] = reader.sample_ids

    logger.info("Wrote Zarr store: %s, %d samples × %d variants", output_path, *dosage.shape)


def _write_vcf(reader, output_path: str) -> None:
    """Write dosage data as VCF with GT and DS fields.

    The internal convention stores dosage as ALT/effect-allele count where
    ``VariantMeta.a1`` is ALT and ``a2`` is REF. Integer dosages are rendered
    as GT; fractional dosages keep a best-guess GT and preserve the value in DS.
    """
    p = Path(output_path)
    opener = gzip.open if p.suffix == ".gz" else open

    with opener(p, "wt") as fh:
        fh.write("##fileformat=VCFv4.3\n")
        fh.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Best-guess genotype\">\n")
        fh.write("##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"ALT allele dosage\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for sid in reader.sample_ids:
            fh.write(f"\t{sid}")
        fh.write("\n")

        for G_chunk, vmeta in reader.iter_chunks(chunk_size=1024):
            dosage_np = G_chunk.numpy()  # (n, m)
            for j in range(len(vmeta)):
                fh.write(f"{vmeta.chr[j]}\t{vmeta.pos[j]}\t{vmeta.snp[j]}\t")
                fh.write(f"{vmeta.a2[j]}\t{vmeta.a1[j]}\t.\tPASS\t.\tGT:DS")
                for i in range(dosage_np.shape[0]):
                    val = dosage_np[i, j]
                    if np.isnan(val):
                        sample = "./.:."
                    else:
                        clipped = min(max(float(val), 0.0), 2.0)
                        rounded = int(round(clipped))
                        if rounded == 0:
                            gt = "0/0"
                        elif rounded == 1:
                            gt = "0/1"
                        else:
                            gt = "1/1"
                        sample = f"{gt}:{clipped:.6g}"
                    fh.write(f"\t{sample}")
                fh.write("\n")

    logger.info("Wrote VCF: %s", p)


def _write_vcf_stub(reader, output_path: str) -> None:
    """Backward-compatible alias for the supported VCF writer."""
    _write_vcf(reader, output_path)
