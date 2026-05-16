#!/usr/bin/env python3
"""Tier 2 B2: harmonize GTEx Liver eQTL (SORT1) with UKB-inclusive GLGC LDL.

Inputs come from .env_marker (written by fetch.sh):
    GTEX_LIVER_SIGNIF, GTEX_LIVER_EGENES, GLGC_GZ

Outputs:
    fixtures/gtex_liver_sort1.tsv.gz  -- raw GTEx Liver signif pairs for SORT1
    fixtures/ukb_ldl_sort1.tsv.gz     -- GLGC LDL slice at SORT1 +/- 500 kb
    fixtures/aligned.parquet          -- inner-joined beta/SE/p/z for both
    fixtures/manifest.sha256          -- SHA256 of every committed fixture
"""
from __future__ import annotations
import gzip
import hashlib
import io
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

# SORT1 cis window in GRCh38 (gene body chr1:109,309,568-109,397,951; +/-500 kb pad).
SORT1_CHR = "chr1"
SORT1_START_B38 = 108_700_000
SORT1_END_B38 = 110_000_000
AMBIG = {frozenset({"A", "T"}), frozenset({"C", "G"})}


def _read_env_marker() -> dict[str, str]:
    marker = HERE / ".env_marker"
    if not marker.exists():
        raise SystemExit(
            f"[gtex_ukb prepare] ABORT: {marker} not found. Run ./fetch.sh first."
        )
    env = {}
    for line in marker.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_gtex_sort1(signif_path, egenes_path):
    """Filter the Liver signif TSV to SORT1; return parsed DataFrame."""
    egenes = pd.read_csv(egenes_path, sep="\t", compression="gzip")
    sort1_egenes = egenes[egenes["gene_name"] == "SORT1"]
    if sort1_egenes.empty:
        raise SystemExit("[gtex_ukb prepare] ABORT: SORT1 not in Liver egenes.")
    sort1_gene_id = sort1_egenes["gene_id"].iloc[0]
    print(f"[gtex_ukb prepare] SORT1 GTEx gene_id={sort1_gene_id}")
    sig = pd.read_csv(signif_path, sep="\t", compression="gzip")
    sort1 = sig[sig["gene_id"] == sort1_gene_id].copy()
    print(f"[gtex_ukb prepare] SORT1 signif pairs n={len(sort1)}")
    parts = sort1["variant_id"].str.split("_", expand=True)
    parts.columns = ["chr", "pos", "ref", "alt", "build_tag"]
    sort1["chr"] = parts["chr"]
    sort1["pos_b38"] = parts["pos"].astype(np.int64)
    sort1["ref"] = parts["ref"]
    sort1["alt"] = parts["alt"]
    sort1["gene_name"] = "SORT1"
    sort1["z_eqtl"] = sort1["slope"] / sort1["slope_se"]
    sort1.rename(columns={"slope": "beta_eqtl", "slope_se": "se_eqtl", "pval_nominal": "p_eqtl", "maf": "maf_eqtl"}, inplace=True)
    return sort1


def _slice_glgc(glgc_path):
    """Stream-filter the GLGC harmonised GWAS TSV to the SORT1 cis window."""
    keep_cols = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "beta", "standard_error", "effect_allele_frequency", "p_value",
        "variant_id", "n", "rsid",
    ]
    chunks = []
    reader = pd.read_csv(
        glgc_path, sep="\t", compression="gzip", chunksize=500_000,
        usecols=lambda c: c in keep_cols, dtype={"chromosome": str},
    )
    chr_target = SORT1_CHR.replace("chr", "")
    saw_target_chunk = False
    for chunk in reader:
        in_target = chunk["chromosome"].astype(str) == chr_target
        if in_target.any():
            saw_target_chunk = True
            mask = (
                in_target
                & (chunk["base_pair_location"] >= SORT1_START_B38)
                & (chunk["base_pair_location"] <= SORT1_END_B38)
            )
            sub = chunk.loc[mask]
            if not sub.empty:
                chunks.append(sub.copy())
        elif saw_target_chunk:
            break
    if not chunks:
        raise SystemExit("[gtex_ukb prepare] ABORT: no GLGC rows in SORT1 window.")
    df = pd.concat(chunks, ignore_index=True)
    df["chr"] = "chr" + df["chromosome"].astype(str)
    df.rename(columns={
        "base_pair_location": "pos_b38", "beta": "beta_gwas",
        "standard_error": "se_gwas", "p_value": "p_gwas",
        "effect_allele_frequency": "eaf_gwas", "n": "n_gwas",
    }, inplace=True)
    df["z_gwas"] = df["beta_gwas"] / df["se_gwas"]
    print(f"[gtex_ukb prepare] GLGC LDL rows in SORT1 window n={len(df)}")
    return df


def _harmonize(gtex, glgc):
    """Inner-join GTEx and GLGC on (chr,pos); resolve allele orientation.

    Three categories per variant:
        direct  -- gtex.ref == glgc.other_allele AND gtex.alt == glgc.effect_allele
        flipped -- gtex.ref == glgc.effect_allele AND gtex.alt == glgc.other_allele
                   (sign of beta_gwas / z_gwas inverted; eaf -> 1-eaf)
        unknown -- neither; dropped conservatively
    Strand-ambiguous A/T or C/G variants are dropped first (standard
    harmonization rule).
    """
    merged = gtex.merge(glgc, on=["chr", "pos_b38"], suffixes=("_eqtl", "_gwas"), how="inner")
    if merged.empty:
        raise SystemExit("[gtex_ukb prepare] ABORT: 0 variants joined.")
    print(f"[gtex_ukb prepare] joined (chr,pos) rows: {len(merged)}")
    pair_key = merged.apply(lambda r: frozenset({r["ref"], r["alt"]}), axis=1)
    n_pre = len(merged)
    merged = merged[~pair_key.isin(AMBIG)].copy()
    print(f"[gtex_ukb prepare] dropped strand-ambiguous A/T or C/G: {n_pre - len(merged)}")
    direct = ((merged["ref"] == merged["other_allele"]) & (merged["alt"] == merged["effect_allele"]))
    flipped = ((merged["ref"] == merged["effect_allele"]) & (merged["alt"] == merged["other_allele"]))
    merged["orientation"] = "unknown"
    merged.loc[direct, "orientation"] = "direct"
    merged.loc[flipped, "orientation"] = "flipped"
    n_unk = int((merged["orientation"] == "unknown").sum())
    print(f"[gtex_ukb prepare] direct={int(direct.sum())} flipped={int(flipped.sum())} unknown={n_unk}")
    merged = merged[merged["orientation"] != "unknown"].copy()
    flip_mask = merged["orientation"] == "flipped"
    merged.loc[flip_mask, "beta_gwas"] = -merged.loc[flip_mask, "beta_gwas"]
    merged.loc[flip_mask, "z_gwas"] = -merged.loc[flip_mask, "z_gwas"]
    eaf = merged.loc[flip_mask, "eaf_gwas"]
    merged.loc[flip_mask, "eaf_gwas"] = 1.0 - eaf
    # rsid from egenes is the top eGene rsid (single value); prefer GLGC rsid here.
    if "rsid" in merged.columns:
        pass
    elif "rs_id_dbSNP151_GRCh38p7" in merged.columns:
        merged["rsid"] = merged["rs_id_dbSNP151_GRCh38p7"]
    keep = [
        "variant_id_eqtl" if "variant_id_eqtl" in merged.columns else "variant_id",
        "chr", "pos_b38", "ref", "alt", "rsid",
        "gene_id", "gene_name",
        "beta_eqtl", "se_eqtl", "p_eqtl", "z_eqtl", "maf_eqtl",
        "beta_gwas", "se_gwas", "p_gwas", "z_gwas", "eaf_gwas", "n_gwas",
        "orientation",
    ]
    keep = [c for c in keep if c in merged.columns]
    return merged[keep].sort_values("pos_b38").reset_index(drop=True)


def _write_tsv_gz(df, dest):
    """Write a DataFrame as gzipped TSV with deterministic gzip mtime=0."""
    buf = io.BytesIO()
    df.to_csv(buf, sep="\t", index=False)
    body = buf.getvalue()
    with open(dest, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
            gz.write(body)


def main():
    env = _read_env_marker()
    signif = pathlib.Path(env["GTEX_LIVER_SIGNIF"])
    egenes = pathlib.Path(env["GTEX_LIVER_EGENES"])
    glgc_gz = pathlib.Path(env["GLGC_GZ"])
    for p, label in [(signif, "GTEX_LIVER_SIGNIF"), (egenes, "GTEX_LIVER_EGENES"), (glgc_gz, "GLGC_GZ")]:
        if not p.exists():
            print(f"[gtex_ukb prepare] ABORT: missing {label} -> {p}")
            return 1
    print("[gtex_ukb prepare] step 1/4: load GTEx SORT1 slice")
    gtex = _load_gtex_sort1(signif, egenes)
    print("[gtex_ukb prepare] step 2/4: slice GLGC SORT1 window")
    glgc = _slice_glgc(glgc_gz)
    print("[gtex_ukb prepare] step 3/4: write raw locus slices")
    raw_eqtl = FIXTURES / "gtex_liver_sort1.tsv.gz"
    raw_gwas = FIXTURES / "ukb_ldl_sort1.tsv.gz"
    raw_eqtl_cols = [
        "variant_id", "gene_id", "gene_name", "chr", "pos_b38", "ref", "alt",
        "tss_distance", "ma_samples", "ma_count", "maf_eqtl",
        "p_eqtl", "beta_eqtl", "se_eqtl",
        "pval_nominal_threshold", "min_pval_nominal", "pval_beta",
    ]
    raw_gwas_cols = [
        "chr", "pos_b38", "effect_allele", "other_allele",
        "beta_gwas", "se_gwas", "eaf_gwas", "p_gwas",
        "variant_id", "n_gwas", "rsid",
    ]
    _write_tsv_gz(gtex[[c for c in raw_eqtl_cols if c in gtex.columns]], raw_eqtl)
    _write_tsv_gz(glgc[[c for c in raw_gwas_cols if c in glgc.columns]], raw_gwas)
    print(f"  wrote {raw_eqtl}")
    print(f"  wrote {raw_gwas}")
    print("[gtex_ukb prepare] step 4/4: harmonize and write aligned.parquet")
    aligned = _harmonize(gtex, glgc)
    parquet_path = FIXTURES / "aligned.parquet"
    aligned.to_parquet(parquet_path, index=False)
    print(f"  wrote {parquet_path}")
    print(f"  n_variants={len(aligned)}")
    print("  columns:", list(aligned.columns))
    manifest_path = FIXTURES / "manifest.sha256"
    files = [parquet_path, raw_eqtl, raw_gwas]
    with manifest_path.open("w") as f:
        for p in files:
            f.write(f"{_sha256(p)}  {p.name}\n")
    print(f"  wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
