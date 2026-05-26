"""S-PrediXcan / FUSION TWAS using a SQLite ``.db`` weights file.

Demonstrates the canonical S-PrediXcan workflow: GWAS summary statistics
+ pre-trained cis-eQTL weights (PrediXcan / FUSION / MetaXcan-format
SQLite ``.db``) → per-gene TWAS z-score. The new
``torchgwas.io.read_predixcan_db`` reader returns a ``PredixcanModel``
dataclass that plugs directly into ``twas_sumstat``.

Phases covered:
- TWAS sumstat (Phase 45)
- PrediXcan / FUSION .db reader (added 2026-05-26)
- Multi-tissue stacking + S-MultiXcan-style aggregation

Runtime: < 3 s on CPU (3 simulated tissues × 100 genes).
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import torch

from torchgwas.io import read_predixcan_db, list_genes_in_db
from torchgwas.postgwas import (
    SumStats,
    twas_multi_tissue_aggregate,
    twas_multi_tissue_stack,
    twas_sumstat,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_predixcan_db(path: Path, n_genes: int = 30,
                       snps_per_gene: int = 6, seed: int = 11) -> list[str]:
    """Build a tiny PrediXcan-style ``.db`` matching GTEx schema."""
    rng = np.random.default_rng(seed)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE weights (rsid TEXT, gene TEXT, weight REAL, "
        "ref_allele TEXT, eff_allele TEXT)"
    )
    cur.execute(
        'CREATE TABLE extra (gene TEXT, genename TEXT, '
        '"n.snps.in.model" INTEGER, "pred.perf.R2" REAL, '
        '"pred.perf.pval" REAL, "pred.perf.qval" REAL)'
    )
    snp_ids = []
    for g in range(n_genes):
        gene = f"ENSG{g:08d}"
        r2 = float(rng.uniform(0.05, 0.40))
        rows = []
        for s in range(snps_per_gene):
            rsid = f"rs{g:04d}{s:02d}"
            snp_ids.append(rsid)
            weight = float(rng.normal(0.0, 0.3))
            rows.append((rsid, gene, weight, "A", "G"))
        cur.executemany(
            "INSERT INTO weights (rsid, gene, weight, ref_allele, eff_allele) "
            "VALUES (?, ?, ?, ?, ?)", rows,
        )
        cur.execute(
            'INSERT INTO extra (gene, genename, "n.snps.in.model", '
            '"pred.perf.R2", "pred.perf.pval", "pred.perf.qval") '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (gene, f"GENE_{g:04d}", snps_per_gene, r2, 1e-4, 1e-3),
        )
    conn.commit()
    conn.close()
    return snp_ids


def simulate_gwas(snp_ids: list[str], causal_snp_idx: int = 30,
                  effect: float = 0.4, seed: int = 42) -> SumStats:
    """Build GWAS sumstats with one planted causal SNP."""
    rng = np.random.default_rng(seed)
    m = len(snp_ids)
    beta = rng.normal(0.0, 0.02, size=m)
    beta[causal_snp_idx] = effect
    se = np.full(m, 0.05)
    z = beta / se
    p = 2 * np.array([math.erfc(abs(z_i) / math.sqrt(2.0)) for z_i in z]) / 2
    return SumStats(
        chr=["1"] * m,
        pos=list(range(1, m + 1)),
        snp=snp_ids,
        a1=["A"] * m, a2=["G"] * m,
        beta=torch.tensor(beta, dtype=torch.float64),
        se=torch.tensor(se, dtype=torch.float64),
        p=torch.tensor(p, dtype=torch.float64),
        n=torch.full((m,), 5000.0, dtype=torch.float64),
    )


def main() -> None:
    # ----- Build three "tissue" .db files (e.g., Liver / Blood / Brain) ----
    tissues = ["Liver", "Blood", "Brain"]
    snp_ids_all: set[str] = set()
    tissue_models: dict[str, object] = {}
    for tissue in tissues:
        db_path = OUT_DIR / f"model_{tissue}.db"
        snp_ids = write_predixcan_db(db_path, n_genes=30, seed=hash(tissue) % 2**31)
        model = read_predixcan_db(db_path)
        print(f"[{tissue:5s}] loaded {model.n_genes} genes, "
              f"{model.n_weights} weights from {db_path.name}")
        print(f"       sample gene_names: "
              f"{list(model.gene_names.items())[:2]}")
        tissue_models[tissue] = model
        snp_ids_all.update(snp_ids)

    # ----- One GWAS sumstats vector across the union of all tissues' SNPs --
    snp_ids_sorted = sorted(snp_ids_all)
    gwas = simulate_gwas(snp_ids_sorted, causal_snp_idx=30)

    # ----- Run S-PrediXcan TWAS per tissue ----------------------------------
    print()
    print("=" * 72)
    print("Per-tissue S-PrediXcan TWAS")
    print("=" * 72)
    per_tissue: dict[str, object] = {}
    for tissue, model in tissue_models.items():
        result = twas_sumstat(
            gwas=gwas,
            weights=model.weights,
            snp_lists=model.snp_lists,
            r2_models=model.r2_models,
            correction="bonferroni",
        )
        per_tissue[tissue] = result
        sig = [g for g in result.genes if g.p_twas < 0.05 / result.n_genes_tested]
        print(f"  {tissue:5s}: {result.n_genes_tested} genes tested, "
              f"{len(sig)} Bonferroni-significant. "
              f"Most extreme z = "
              f"{max(g.z_twas for g in result.genes):.3f}")

    # ----- Multi-tissue aggregation (S-MultiXcan-style approximation) ------
    print()
    print("=" * 72)
    print("Multi-tissue aggregation (S-MultiXcan-style χ² approximation)")
    print("=" * 72)
    long_rows = twas_multi_tissue_stack(per_tissue)
    summaries = twas_multi_tissue_aggregate(per_tissue)
    print(f"  Long-format rows: {len(long_rows)}")
    print(f"  Per-gene aggregated rows: {len(summaries)}")
    print(f"  Top 5 multi-tissue genes by p_multixcan:")
    for s in summaries[:5]:
        print(
            f"    {s.gene_id}  n_tissues={s.n_tissues}  "
            f"χ²={s.chi2_multixcan:6.2f}  p={s.p_multixcan:.3e}  "
            f"driver={s.driver_tissue}"
        )

    # ----- Persist the aggregated table -------------------------------------
    out_path = OUT_DIR / "12_twas_sumstat_predixcan_db.tsv"
    with open(out_path, "w") as f:
        f.write("gene_id\tn_tissues\tz_mean\tz_sd\tp_min\tdriver_tissue\t"
                "chi2_multixcan\tp_multixcan\n")
        for s in summaries:
            f.write(
                f"{s.gene_id}\t{s.n_tissues}\t{s.z_mean:+.4f}\t{s.z_sd:.4f}\t"
                f"{s.p_min:.3e}\t{s.driver_tissue}\t"
                f"{s.chi2_multixcan:.4f}\t{s.p_multixcan:.3e}\n"
            )
    print(f"\nResults written to {out_path}")
    print(f"  Inspect models via: list_genes_in_db('{OUT_DIR / 'model_Liver.db'}')")


if __name__ == "__main__":
    main()
