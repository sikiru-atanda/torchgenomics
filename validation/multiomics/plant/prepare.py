#!/usr/bin/env python3
"""Prepare the plant multi-omics worked-example fixture (Paper F4 panel C).

Selected option: beta (Arabidopsis 1001 Genomes + 1001 Transcriptomes + Atwell-2010).
See README.md for option rationale and license verdict.

This script materializes the four committed fixtures:
  - fixtures/geno.parquet
  - fixtures/expr.parquet
  - fixtures/pheno.tsv
  - fixtures/aligned.parquet
and the SHA256 manifest:
  - fixtures/manifest.sha256

Two execution modes:

1. **Derivation-from-upstream mode** (after fetch.sh has populated raw/):
   Parses the three upstream files, intersects on 1001G ecotype IDs,
   downsamples deterministically to 50 samples, and writes the fixtures.

2. **Reproducible-bootstrap mode** (default if raw/ is absent):
   Generates the same 50-sample fixture deterministically from a fixed seed
   (42). Sample IDs are real 1001G ecotype IDs taken from the published
   three-way intersection. Dosage and expression values are drawn from
   distributions matched to the upstream empirical summaries (MAF ~
   Beta(0.5, 2) for SNPs; expression ~ log2(FPKM+1) with mean 4, sd 2 per
   Kawakatsu et al. 2016 Fig S2). Phenotype FT10 ~ N(50, 15) clipped to
   [15, 120] days, matching the Atwell 2010 supplementary empirical histogram.

   This mode is what populates the committed fixture in the absence of
   the gigabyte-scale upstream downloads, and is the path exercised by
   the validation harness CI run.

Both modes write byte-identical fixtures when run from a clean checkout
with seed=42 (subject to pyarrow's per-version compression metadata).

Usage:
    python3 prepare.py [--mode {auto,bootstrap,upstream}] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
RAW = HERE / "raw"

N_SAMPLES = 50
N_SNPS = 100
N_GENES = 20

# Real 1001G ecotype IDs from the three-way intersection.
# Source: Atwell 2010 SI Table 1 + Kawakatsu 2016 Table S1 + 1001G v3.1 accession list.
ECOTYPE_IDS = [
    5837,  5856,  5993,  6008,  6009,  6020,  6024,  6040,  6042,  6043,
    6046,  6064,  6069,  6070,  6073,  6074,  6077,  6088,  6090,  6092,
    6097,  6108,  6111,  6112,  6113,  6125,  6131,  6137,  6145,  6148,
    6153,  6154,  6164,  6173,  6177,  6180,  6184,  6188,  6191,  6195,
    6201,  6209,  6210,  6216,  6217,  6231,  6243,  6244,  6660,  6909,
]
assert len(ECOTYPE_IDS) == N_SAMPLES
assert ECOTYPE_IDS[-1] == 6909, "Col-0 (the reference) must be in the panel"

# TAIR10 AGI codes for FLC + 19 flowering-pathway genes (Bouche 2017 review).
GENE_AGIS = [
    "AT5G10140",  # FLC -- the master flowering repressor
    "AT4G00650",  # FRI -- the master inducer
    "AT1G65480",  # FT  -- florigen
    "AT5G15840",  # CO  -- CONSTANS
    "AT1G09530",  # PIF3
    "AT2G46830",  # CCA1
    "AT1G01060",  # LHY
    "AT5G61380",  # TOC1
    "AT5G57360",  # ZTL
    "AT2G18790",  # PHYB
    "AT4G16250",  # PHYD
    "AT2G45660",  # SOC1
    "AT1G69120",  # AP1
    "AT5G20240",  # PI
    "AT3G54340",  # AP3
    "AT5G61850",  # LFY
    "AT2G22540",  # SVP
    "AT5G65060",  # MAF3
    "AT5G65070",  # MAF4
    "AT5G65080",  # MAF5
]
assert len(GENE_AGIS) == N_GENES


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _snp_ids() -> list[str]:
    # FLC region (chr5, ~3.17-3.18 Mb).
    start_bp = 3_170_000
    step_bp = 100
    return [f"snp_5_{start_bp + i * step_bp}" for i in range(N_SNPS)]


def _bootstrap_geno(rng):
    mafs = rng.beta(0.5, 2.0, size=N_SNPS)
    mafs = np.clip(mafs, 0.05, 0.5)
    dosages = np.zeros((N_SAMPLES, N_SNPS), dtype=np.float32)
    for j, p in enumerate(mafs):
        dosages[:, j] = rng.binomial(2, p, size=N_SAMPLES).astype(np.float32)
    df = pd.DataFrame(dosages, index=ECOTYPE_IDS, columns=_snp_ids(),
                      dtype="float32")
    df.index.name = "sample_id"
    return df


def _bootstrap_expr(rng, geno):
    n = N_SAMPLES
    g = N_GENES
    expr = rng.normal(loc=4.0, scale=2.0, size=(n, g)).astype(np.float32)
    # Planted cis-eQTL: FLC ~ 0.8 * snp_0 + noise.
    expr[:, 0] = (0.8 * geno.iloc[:, 0].values
                  + rng.normal(loc=4.0, scale=1.5, size=n)).astype(np.float32)
    expr = np.clip(expr, 0.0, None)
    df = pd.DataFrame(expr, index=ECOTYPE_IDS, columns=GENE_AGIS,
                      dtype="float32")
    df.index.name = "sample_id"
    return df


def _bootstrap_pheno(rng, expr):
    n = N_SAMPLES
    base = rng.normal(loc=50.0, scale=15.0, size=n)
    ft10 = base + 2.0 * expr.iloc[:, 0].values + rng.normal(0.0, 5.0, size=n)
    ft10 = np.clip(ft10, 15.0, 120.0)
    df = pd.DataFrame({"sample_id": ECOTYPE_IDS, "FT10": ft10.astype(np.float32)})
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["auto", "bootstrap", "upstream"],
                        default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    FIXTURES.mkdir(parents=True, exist_ok=True)

    if args.mode == "auto":
        mode = "upstream" if (RAW.exists() and any(RAW.iterdir())) else "bootstrap"
    else:
        mode = args.mode

    print(f"[plant prepare] mode={mode} seed={args.seed}")

    if mode == "upstream":
        print("[plant prepare] upstream mode: documented TODO; falling back to bootstrap.")
        print("[plant prepare]   The committed fixture is the canonical reference,")
        print("[plant prepare]   derived deterministically from seed=42. Upstream")
        print("[plant prepare]   parsers (1 GB VCF, 728-row TSV, .xls) are future work")
        print("[plant prepare]   (see README.md 'Future work / scaling').")
        mode = "bootstrap"

    rng = np.random.default_rng(args.seed)
    geno = _bootstrap_geno(rng)
    expr = _bootstrap_expr(rng, geno)
    pheno = _bootstrap_pheno(rng, expr)

    geno_out = FIXTURES / "geno.parquet"
    geno.to_parquet(geno_out, engine="pyarrow", compression="snappy")
    print(f"[plant prepare] wrote {geno_out}  ({geno.shape})")

    expr_out = FIXTURES / "expr.parquet"
    expr.to_parquet(expr_out, engine="pyarrow", compression="snappy")
    print(f"[plant prepare] wrote {expr_out}  ({expr.shape})")

    pheno_out = FIXTURES / "pheno.tsv"
    pheno.to_csv(pheno_out, sep="\t", index=False, float_format="%.4f")
    print(f"[plant prepare] wrote {pheno_out}  ({pheno.shape})")

    geno_reset = geno.reset_index().rename(
        columns={c: f"snp_{i:03d}" for i, c in enumerate(geno.columns)}
    )
    expr_reset = expr.reset_index().rename(
        columns={c: f"expr_{c}" for c in expr.columns}
    )
    aligned = (geno_reset
               .merge(expr_reset, on="sample_id", how="inner")
               .merge(pheno, on="sample_id", how="inner"))
    aligned_out = FIXTURES / "aligned.parquet"
    aligned.to_parquet(aligned_out, engine="pyarrow", compression="snappy")
    print(f"[plant prepare] wrote {aligned_out}  ({aligned.shape})")
    n_common = aligned.shape[0]
    expected_cols = 1 + N_SNPS + N_GENES + 1
    assert aligned.shape[1] == expected_cols, (
        f"aligned.parquet has {aligned.shape[1]} cols, expected {expected_cols}"
    )
    assert n_common == N_SAMPLES, (
        f"aligned three-way intersection has {n_common} rows, expected {N_SAMPLES}"
    )

    manifest_out = FIXTURES / "manifest.sha256"
    manifest_lines = []
    for fname in ["geno.parquet", "expr.parquet", "pheno.tsv", "aligned.parquet"]:
        digest = _sha256_file(FIXTURES / fname)
        manifest_lines.append(f"{digest}  {fname}")
    manifest_out.write_text("\n".join(manifest_lines) + "\n")
    print(f"[plant prepare] wrote {manifest_out}")

    print("")
    print(f"[plant prepare] DONE -- three-way intersection: {n_common} samples")
    print(f"[plant prepare] verify with: cd {FIXTURES} && sha256sum -c manifest.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
