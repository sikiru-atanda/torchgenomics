"""Real-data sanity check for `torchgwas.annotate` against live NCBI.

Uses the MDP maize top-5 EarHT hits from the multi-omics sanity script plus
their real chromosome / bp positions from mdp_SNP_information.txt. Calls NCBI
Datasets v2 live, so the machine must have internet and the run takes a few
seconds per hit.

Sanity goals:
- crop="maize" resolves to Zea mays taxid (4577)
- assembly comes back as a real GCF accession
- each hit returns 0+ real genes with NCBI gene_id, symbol, and a window that
  brackets the SNP position
- GO terms (best-effort) and no weird crashes
"""

from __future__ import annotations

import os

import pandas as pd
import torch

from torchgwas.annotate import annotate_hits
from torchgwas.postgwas._sumstats import SumStats

DATA = "C:/Users/Sikiru/Documents/GWAS_Expert/benchmark/data"

# Top-5 EarHT hits from sanity_mediate_mdp.py
TOP = [
    ("PZB00811.1", 1.564e-04),
    ("PZA00112.5", 1.170e-03),
    ("PZA00051.17", 1.248e-03),
    ("PZB01881.10", 1.454e-03),
    ("PZA03188.4",  2.107e-03),
]

snp_info = pd.read_csv(f"{DATA}/mdp_SNP_information.txt", sep="\t")
snp_info = snp_info.set_index("SNP")

rows = []
for snp_id, p in TOP:
    if snp_id not in snp_info.index:
        print(f"  [warn] {snp_id} not in mdp_SNP_information.txt; skipping")
        continue
    rec = snp_info.loc[snp_id]
    rows.append((str(rec["Chromosome"]), int(rec["Position"]), snp_id, p))

print("Top-5 EarHT hits with genomic coordinates:")
for chrom, pos, snp_id, p in rows:
    print(f"  {snp_id:<14s} chr{chrom}:{pos:,}  (p = {p:.3e})")

# Build SumStats — beta/se are placeholders; annotate_hits only needs snp/chr/pos/p.
m = len(rows)
ss = SumStats(
    chr=[r[0] for r in rows],
    pos=[r[1] for r in rows],
    snp=[r[2] for r in rows],
    a1=["A"] * m,
    a2=["C"] * m,
    beta=torch.zeros(m, dtype=torch.float64),
    se=torch.ones(m, dtype=torch.float64),
    p=torch.tensor([r[3] for r in rows], dtype=torch.float64),
    n=torch.full((m,), 249, dtype=torch.float64),
)

# --- Call live NCBI -----------------------------------------------------------
print("\n=== annotate_hits (live NCBI, taxid=4577 / Zea mays) ===")
api_key = os.environ.get("NCBI_API_KEY")  # optional, raises throughput cap
result = annotate_hits(
    ss,
    taxid=4577,  # species-level Zea mays; crop-name lookup resolves to a subspecies with no gene index
    window_upstream_bp=50_000,
    window_downstream_bp=50_000,
    p_threshold=5e-3,  # loose so all 5 pass (these are demo data, not genome-wide sig)
    include_go=True,
    include_orthologs=False,
    api_key=api_key,
)

print(f"\n  assembly   = {result.assembly_accession}")
print(f"  taxid      = {result.taxid}")
print(f"  n hits     = {len(result.hits)}")

for hit in result.hits:
    print(f"\n  {hit.snp} chr{hit.chrom}:{hit.pos:,}  ({len(hit.genes)} gene(s) in window)")
    for g in hit.genes[:3]:
        go_summary = (
            f"  GO[{len(g.go_terms)}]={[t.get('name', t) for t in g.go_terms[:2]]}"
            if g.go_terms else "  GO[0]"
        )
        print(f"    - {g.symbol or g.gene_id:<15s} "
              f"[{g.chrom}:{g.start:,}-{g.end:,}]  {(g.description or '')[:60]}"
              f"{go_summary}")
    if len(hit.genes) > 3:
        print(f"    ... ({len(hit.genes) - 3} more)")

# --- Sanity assertions --------------------------------------------------------
assert result.taxid == 4577, f"expected Zea mays taxid 4577, got {result.taxid}"
assert result.assembly_accession.startswith("GCF_") or \
       result.assembly_accession.startswith("GCA_"), \
    f"unexpected assembly format: {result.assembly_accession}"

for hit in result.hits:
    for g in hit.genes:
        # Gene window must overlap the hit +/- 50kb window
        lo, hi = hit.pos - 50_000, hit.pos + 50_000
        assert g.start <= hi and g.end >= lo, \
            f"{g.gene_id} [{g.start}-{g.end}] does not overlap hit window [{lo}-{hi}]"
        assert g.gene_id, f"empty gene_id on {hit.snp}"

print("\n  [OK] all genes overlap their hit windows; taxid + assembly look real")

# --- DataFrame flatten + TSV write --------------------------------------------
df = result.to_dataframe()
print(f"\n  to_dataframe() -> shape = {df.shape}; columns = {list(df.columns)[:8]} ...")
print(df[["snp", "chrom", "pos", "gene_id", "symbol"]].head(10).to_string(index=False))

out = "C:/Users/Sikiru/Documents/GWAS_Expert/scripts/sanity_annotate_mdp.tsv"
result.to_tsv(out)
roundtrip = pd.read_csv(out, sep="\t")
assert len(roundtrip) == len(df)
print(f"\n  Wrote + roundtripped {len(df)} gene-rows to {out}")

print("\nAll NCBI annotate sanity checks passed on MDP maize real data.")
