"""Convert the MDP maize numeric fixture (`benchmark/data/mdp_*`) to PLINK BED.

Inputs (from `benchmark/data/`):
  - mdp_numeric.txt           — TSV: header is taxa + SNP IDs; rows are samples;
                                values are 0/1/2 dosage (NA = missing).
  - mdp_traits.txt            — TSV: Taxa, EarHT, dpoll, EarDia.
  - mdp_SNP_information.txt   — TSV: SNP, Chromosome, Position.

Outputs (in --out-dir):
  - mdp.bed/.bim/.fam   — PLINK 1.x binary fileset (SNP-major), 282 samples × 3093 SNPs.
  - mdp_pheno.txt       — PLINK-style phenotype: FID IID EarHT dpoll.

Conventions matching how the upstream conversion script in
`benchmark/convert_mdp_to_bimbam.py` and the GEMMA fixture treat alleles:
  - A1 = "A" (the dosage-counted allele; dosage 2 = AA homozygote).
  - A2 = "G".
  - PLINK BED 2-bit codes (SNP-major), per sample:
        00 → hom A1 A1   (we map dosage 2 here — i.e. A1 is the COUNTED allele
                          in the .bim, and we want the homozygous A1 genotype to
                          decode back to dosage 2 under TorchGWAS' decoder)
        11 → hom A2 A2   (dosage 0)
        10 → het         (dosage 1)
        01 → missing     (NaN)
    NB: TorchGWAS' `PlinkBedReader._GENO_DECODE = [0, NaN, 1, 2]` decodes:
        00 → 0, 01 → NaN, 10 → 1, 11 → 2.
    So to round-trip dosage d ∈ {0,1,2} back to the SAME numeric value d,
    we encode 0 → 00, 1 → 10, 2 → 11. That makes A1 the *minor* / "B" allele
    in PLINK semantics, but it's the allele whose count equals our dosage.

This script is invoked by fetch_data.sh; it has no Pillar B pre-flight of its
own (the parent shell script does that).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def encode_bed(dosage: np.ndarray) -> bytes:
    """Encode (n_samples, n_variants) dosage array to PLINK BED SNP-major bytes.

    Mapping (TorchGWAS-compatible round-trip):
      dosage 0   -> 2-bit code 0b00
      dosage 1   -> 2-bit code 0b10
      dosage 2   -> 2-bit code 0b11
      NaN/missing-> 2-bit code 0b01
    """
    n_samples, n_variants = dosage.shape
    bytes_per_snp = (n_samples + 3) // 4

    # Build per-sample 2-bit code table (SNP-major: outer=SNP, inner=sample)
    code = np.full((n_variants, n_samples), 0b01, dtype=np.uint8)  # default missing

    # Use .T to get (n_variants, n_samples) view of dosage
    d = dosage.T  # (n_variants, n_samples)
    nonnan = ~np.isnan(d)

    # Round dosage to nearest int for assignment
    di = np.where(nonnan, np.rint(d).astype(np.int8), -1)

    code = np.where(di == 0, 0b00, code)
    code = np.where(di == 1, 0b10, code)
    code = np.where(di == 2, 0b11, code)

    # Pack 4 samples per byte (LSB first)
    pad = bytes_per_snp * 4 - n_samples
    if pad:
        code = np.concatenate(
            [code, np.full((n_variants, pad), 0b01, dtype=np.uint8)], axis=1
        )

    code = code.reshape(n_variants, bytes_per_snp, 4)
    packed = (
        code[:, :, 0]
        | (code[:, :, 1] << 2)
        | (code[:, :, 2] << 4)
        | (code[:, :, 3] << 6)
    ).astype(np.uint8)

    magic = bytes([0x6C, 0x1B, 0x01])
    return magic + packed.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mdp-numeric", required=True, help="benchmark/data/mdp_numeric.txt")
    ap.add_argument("--mdp-traits", required=True, help="benchmark/data/mdp_traits.txt")
    ap.add_argument("--mdp-snpmap", required=True, help="benchmark/data/mdp_SNP_information.txt")
    ap.add_argument("--out-dir", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load -----------------------------------------------------------------
    geno = pd.read_csv(args.mdp_numeric, sep="\t")
    traits = pd.read_csv(args.mdp_traits, sep="\t")
    snpmap = pd.read_csv(args.mdp_snpmap, sep="\t")

    geno["taxa"] = geno["taxa"].astype(str)
    traits["Taxa"] = traits["Taxa"].astype(str)

    # Align: keep only samples present in geno AND with at least one trait observed
    # (we keep NaN traits — PLINK will skip them per-trait)
    sample_ids = list(geno["taxa"])
    n = len(sample_ids)
    snp_names = [c for c in geno.columns if c != "taxa"]
    m = len(snp_names)

    print(f"[convert_mdp] {n} samples, {m} SNPs")

    # --- Build dosage matrix --------------------------------------------------
    dosage = geno[snp_names].to_numpy(dtype=np.float64)
    assert dosage.shape == (n, m), f"shape mismatch: {dosage.shape}"

    # --- BED ------------------------------------------------------------------
    bed_bytes = encode_bed(dosage)
    bed_path = out / "mdp.bed"
    bed_path.write_bytes(bed_bytes)
    print(f"[convert_mdp] wrote {bed_path} ({bed_path.stat().st_size:,} bytes)")

    # --- BIM ------------------------------------------------------------------
    snpmap_idx = snpmap.set_index("SNP")
    bim_path = out / "mdp.bim"
    with bim_path.open("w") as fh:
        for snp in snp_names:
            if snp in snpmap_idx.index:
                row = snpmap_idx.loc[snp]
                chrom = int(row["Chromosome"])
                pos = int(row["Position"])
            else:
                chrom = 0
                pos = 0
            # Cols: chr  snp  cM  pos  A1(counted)  A2(other)
            fh.write(f"{chrom}\t{snp}\t0\t{pos}\tA\tG\n")
    print(f"[convert_mdp] wrote {bim_path}")

    # --- FAM ------------------------------------------------------------------
    fam_path = out / "mdp.fam"
    with fam_path.open("w") as fh:
        for sid in sample_ids:
            # FID IID PID MID SEX PHENO
            fh.write(f"{sid}\t{sid}\t0\t0\t0\t-9\n")
    print(f"[convert_mdp] wrote {fam_path}")

    # --- Phenotype ------------------------------------------------------------
    pheno_path = out / "mdp_pheno.txt"
    traits_idx = traits.set_index("Taxa")
    with pheno_path.open("w") as fh:
        fh.write("FID\tIID\tEarHT\tdpoll\n")
        for sid in sample_ids:
            if sid in traits_idx.index:
                row = traits_idx.loc[sid]
                ear = row["EarHT"]
                dpoll = row["dpoll"]
            else:
                ear = float("nan")
                dpoll = float("nan")
            ear_s = "NA" if pd.isna(ear) else f"{ear}"
            dp_s = "NA" if pd.isna(dpoll) else f"{dpoll}"
            fh.write(f"{sid}\t{sid}\t{ear_s}\t{dp_s}\n")
    print(f"[convert_mdp] wrote {pheno_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
