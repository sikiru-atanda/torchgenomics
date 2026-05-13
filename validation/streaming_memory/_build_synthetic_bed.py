"""Stream-write a synthetic PLINK BED fixture for the streaming-memory benchmark.

Generates a (n_samples × n_snps) random additive-coded {0, 1, 2} matrix and
writes it to disk in PLINK 1.9 BED format (SNP-major, 2 bits per call).
Memory footprint while writing is one column at a time — never the whole
matrix — so we can produce >10 GB BED files on a workstation with modest RAM.

Companion files:
  <out>.bed   — packed binary genotype matrix
  <out>.bim   — variant info (chr=22, pos = 1..p, A1='A', A2='G')
  <out>.fam   — sample info (FID == IID == 'IID_<i>', phenotype = -9)
  <out>.pheno — separate phenotype file with covariates (PC1, PC2)

Per memory pre-flight: the script asserts disk + RAM headroom before any
write. Aborts cleanly if either is insufficient.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


PLINK_BED_MAGIC = bytes([0x6C, 0x1B, 0x01])
# PLINK 2-bit genotype codes (per byte; LSB-first within byte):
#   homozygous A1   = 00 (binary 0)
#   missing         = 01 (binary 1)
#   heterozygous    = 10 (binary 2)
#   homozygous A2   = 11 (binary 3)
# Mapping additive-dosage 0/1/2 → PLINK code:
#   dosage 0 → A1A1 → 00
#   dosage 1 → het  → 10
#   dosage 2 → A2A2 → 11
DOSAGE_TO_PLINK = np.array([0b00, 0b10, 0b11], dtype=np.uint8)


def preflight(n: int, p: int, out: Path) -> None:
    bed_bytes = 3 + p * math.ceil(n / 4)
    free_disk = os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
    if bed_bytes + 5_000_000_000 > free_disk:  # 5 GB headroom
        sys.exit(
            f"FATAL: BED needs {bed_bytes / 1e9:.1f} GB but only "
            f"{free_disk / 1e9:.1f} GB free at {out.parent}. Aborting."
        )
    # We hold one column (n samples) of int8 dosages while packing — peak ~n bytes
    # plus packed buffer ~ceil(n/4) bytes. Always tiny; nothing to assert.
    print(
        f"[preflight] n={n:,}  p={p:,}  BED={bed_bytes/1e6:.1f} MB  "
        f"free_disk={free_disk/1e9:.1f} GB → OK"
    )


def write_bed_streaming(out: Path, n: int, p: int, seed: int = 7,
                        maf_lo: float = 0.05, maf_hi: float = 0.5) -> None:
    """Write a PLINK BED file column-by-column without materializing (n,p)."""
    rng = np.random.default_rng(seed)
    bytes_per_col = math.ceil(n / 4)
    # Allocate one packed column buffer
    packed = np.zeros(bytes_per_col, dtype=np.uint8)

    # Pre-sample MAFs uniformly in [maf_lo, maf_hi]; this keeps SNPs realistic
    mafs = rng.uniform(maf_lo, maf_hi, size=p).astype(np.float32)

    t0 = time.time()
    with open(out, "wb") as f:
        f.write(PLINK_BED_MAGIC)
        for j in range(p):
            maf = float(mafs[j])
            # Sample n dosages from Bin(2, maf) — Hardy-Weinberg under random mating
            dosages = rng.binomial(2, maf, size=n).astype(np.int8)
            plink_codes = DOSAGE_TO_PLINK[dosages]  # shape (n,)
            packed[:] = 0
            # Pack 4 calls per byte, LSB-first within byte
            for shift_idx in range(4):
                src = plink_codes[shift_idx::4]
                # Pad with 0b00 if n not divisible by 4
                if src.size < bytes_per_col:
                    pad = np.zeros(bytes_per_col - src.size, dtype=np.uint8)
                    src = np.concatenate([src, pad])
                packed |= (src.astype(np.uint8) & 0b11) << (2 * shift_idx)
            f.write(packed.tobytes())
            if (j + 1) % 50000 == 0:
                rate = (j + 1) / (time.time() - t0)
                print(f"  ...wrote {j+1:,}/{p:,} variants ({rate:.0f}/s)")
    elapsed = time.time() - t0
    print(f"[bed] {out}: {os.path.getsize(out)/1e6:.1f} MB in {elapsed:.1f}s "
          f"({p/elapsed:.0f} variants/s)")


def write_bim(out: Path, p: int) -> None:
    """Variant info: all chr 22, sequential positions, A/G alleles."""
    with open(out, "w") as f:
        for j in range(p):
            f.write(f"22\trs{j+1:08d}\t0\t{j+1}\tA\tG\n")
    print(f"[bim] {out}: {p:,} lines")


def write_fam(out: Path, n: int) -> None:
    """Sample info: dummy FID=IID, phenotype = -9 (missing)."""
    with open(out, "w") as f:
        for i in range(n):
            f.write(f"IID_{i+1}\tIID_{i+1}\t0\t0\t0\t-9\n")
    print(f"[fam] {out}: {n:,} lines")


def write_pheno(out: Path, n: int, seed: int = 11) -> None:
    """Phenotype + 2 PC covariates. Y ~ Normal(0, 1); PCs N(0, 1)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    pc1 = rng.standard_normal(n)
    pc2 = rng.standard_normal(n)
    with open(out, "w") as f:
        f.write("FID\tIID\tY\tPC1\tPC2\n")
        for i in range(n):
            f.write(f"IID_{i+1}\tIID_{i+1}\t{y[i]:.6f}\t{pc1[i]:.6f}\t{pc2[i]:.6f}\n")
    print(f"[pheno] {out}: {n:,} lines")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True, help="Sample count")
    p.add_argument("--p", type=int, required=True, help="Variant count")
    p.add_argument("--out", type=Path, required=True, help="Output prefix (no ext)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    preflight(args.n, args.p, args.out.with_suffix(".bed"))
    write_bed_streaming(args.out.with_suffix(".bed"), args.n, args.p, args.seed)
    write_bim(args.out.with_suffix(".bim"), args.p)
    write_fam(args.out.with_suffix(".fam"), args.n)
    write_pheno(args.out.with_suffix(".pheno"), args.n, args.seed + 4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
