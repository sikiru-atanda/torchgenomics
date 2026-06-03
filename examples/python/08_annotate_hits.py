"""NCBI gene annotation of GWAS hits (live NCBI query).

Demonstrates: build a toy SumStats with two 'hits' on maize chromosome 1, then
call `annotate_hits` to pull genes in +/- 50 kb windows (with GO terms).

Phases covered: 48 (NCBI gene annotation; Datasets v2 + E-utilities).

Runtime: < 30 s on a warm NCBI connection (network-dependent). Skipped
gracefully when offline or rate-limited.
"""
from __future__ import annotations

from pathlib import Path

import torch

from torchgenomics.annotate import annotate_hits
from torchgenomics.postgwas._sumstats import SumStats

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Toy sumstats: two hits on maize chr 1 at genomic positions known to
    # fall within annotated gene-rich regions of B73 RefGen v5.
    m = 4
    ss = SumStats(
        chr=["1", "1", "2", "3"],
        pos=[30_000_000, 120_000_000, 50_000_000, 90_000_000],
        snp=["rs_chr1_30M", "rs_chr1_120M", "rs_chr2_50M", "rs_chr3_90M"],
        a1=["A", "C", "G", "T"],
        a2=["G", "T", "A", "C"],
        beta=torch.tensor([0.4, -0.3, 0.05, 0.02], dtype=torch.float64),
        se=torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float64),
        p=torch.tensor([1e-10, 5e-9, 1e-3, 0.5], dtype=torch.float64),
        n=torch.full((m,), 5000.0, dtype=torch.float64),
    )

    print(f"SumStats: {m} SNPs; {int((ss.p <= 5e-8).sum())} at genome-wide sig.")

    try:
        result = annotate_hits(
            ss,
            crop="maize",
            window_upstream_bp=50_000,
            window_downstream_bp=50_000,
            p_threshold=5e-8,
            include_go=True,
            include_orthologs=False,
        )
    except Exception as exc:  # noqa: BLE001 — network is inherently flaky
        print(f"[skip] annotate_hits failed (network/NCBI issue): {type(exc).__name__}: {exc}")
        print("       Set NCBI_API_KEY and retry, or run with internet access.")
        return

    print(f"Assembly: {result.assembly_accession} ({result.assembly_name})")
    print(f"Taxid   : {result.taxid}")
    print(f"Hits    : {len(result.hits)}  (window +/-{result.window_upstream_bp/1e3:.0f} kb)")

    df = result.to_dataframe()
    out = OUT_DIR / "08_annotate_hits.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}  ({len(df)} hit-gene rows)")

    # Brief preview
    for hit in result.hits:
        print(f"\n{hit.snp} (chr{hit.chrom}:{hit.pos}, p={hit.p:.2e}) status={hit.status}")
        for g in hit.genes[:3]:
            print(f"  {g.symbol or '-':<15} {g.gene_id:<15} {g.gene_type:<15} "
                  f"d={g.genomic_distance_to_snp:+d} bp  GO={len(g.go_terms)}")


if __name__ == "__main__":
    main()
