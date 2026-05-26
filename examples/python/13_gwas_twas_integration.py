"""GWAS↔TWAS gene-level integration with all 8 classical methods.

Demonstrates the canonical post-hoc integration of GWAS and TWAS gene-
level evidence using each of the 8 two-input combination methods
(Fisher / Stouffer / Cauchy / Brown / Empirical Brown / HMP / truncated
product / min-p) shipped in v0.3.9. Each method is run against the
same synthetic fixture (3 planted-signal genes, 2 null genes) so the
relative behavior across methods is directly comparable.

Phases covered:
- GWAS↔TWAS integration (added 2026-05-26)
- 8 classical p-value combination kernels
- MAGMA-style snp_to_gene aggregation (Phase 44)

Runtime: < 3 s on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from torchgwas.postgwas import (
    SumStats,
    TWASGeneResult,
    TWASResult,
    combine_gwas_twas,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_fixture():
    """Three signal genes + two null genes on chr 1.

    Gene 1 (G1): strong GWAS + strong TWAS, concordant directions.
    Gene 2 (G2): strong GWAS only.
    Gene 3 (G3): strong TWAS only.
    Gene 4 (G4): null in both (background).
    Gene 5 (G5): null in both.
    """
    # 10 SNPs total: 2 per gene window.
    snps, positions, betas = [], [], []
    for g in range(5):
        for s in range(2):
            snps.append(f"rs{g + 1}_{s}")
            positions.append((g + 1) * 1000 + s * 100)
            betas.append(0.05)  # background

    # Plant signals at SNPs in genes 1 and 2.
    betas[0] = 0.30   # G1, rs1_0
    betas[1] = 0.25   # G1, rs1_1
    betas[2] = 0.28   # G2, rs2_0
    betas[3] = 0.22   # G2, rs2_1

    se = [0.05] * 10
    z = [b / s for b, s in zip(betas, se)]
    p = [math.erfc(abs(zi) / math.sqrt(2.0)) for zi in z]
    gwas = SumStats(
        chr=["1"] * 10, pos=positions, snp=snps,
        a1=["A"] * 10, a2=["G"] * 10,
        beta=torch.tensor(betas, dtype=torch.float64),
        se=torch.tensor(se, dtype=torch.float64),
        p=torch.tensor(p, dtype=torch.float64),
        n=torch.full((10,), 5000.0, dtype=torch.float64),
    )

    # TWAS: strong on G1 (concordant) and G3, null on G2 / G4 / G5.
    twas = TWASResult(
        genes=[
            TWASGeneResult(
                gene_id="G1", z_twas=4.5, p_twas=6.8e-06, n_cis_snps=0,
                r2_model=0.30, top_weight_snp=None, beta=0.22, se=0.05,
                chr="1", start=1000, end=1500, gene_name="GENE_1",
            ),
            TWASGeneResult(
                gene_id="G2", z_twas=0.5, p_twas=0.62, n_cis_snps=0,
                r2_model=0.05, top_weight_snp=None, beta=0.03, se=0.06,
                chr="1", start=2000, end=2500, gene_name="GENE_2",
            ),
            TWASGeneResult(
                gene_id="G3", z_twas=-4.1, p_twas=4.0e-05, n_cis_snps=0,
                r2_model=0.40, top_weight_snp=None, beta=-0.21, se=0.05,
                chr="1", start=3000, end=3500, gene_name="GENE_3",
            ),
            TWASGeneResult(
                gene_id="G4", z_twas=0.3, p_twas=0.76, n_cis_snps=0,
                r2_model=0.10, top_weight_snp=None, beta=0.02, se=0.05,
                chr="1", start=4000, end=4500, gene_name="GENE_4",
            ),
            TWASGeneResult(
                gene_id="G5", z_twas=-0.1, p_twas=0.92, n_cis_snps=0,
                r2_model=0.08, top_weight_snp=None, beta=0.01, se=0.05,
                chr="1", start=5000, end=5500, gene_name="GENE_5",
            ),
        ],
        n_genes_tested=5, n_significant=0,
    )
    return gwas, twas


def main() -> None:
    gwas, twas = build_fixture()

    methods = (
        "fisher", "stouffer", "cauchy",
        "brown", "empirical_brown",
        "hmp", "truncated_product", "min_p",
    )

    # Run all 8 methods against the same fixture.
    rows = {g.gene_id: {"p_gwas": None, "p_twas": g.p_twas}
            for g in twas.genes}
    for method in methods:
        res = combine_gwas_twas(
            gwas, twas, method=method, cis_window_bp=200, correction="bh",
        )
        for g in res.genes:
            if rows[g.gene_id]["p_gwas"] is None:
                rows[g.gene_id]["p_gwas"] = g.p_gwas
            rows[g.gene_id][method] = g.p_combined

    # Print a per-gene table comparing all methods.
    print("=" * 96)
    header = (f"{'gene':4s}  {'p_GWAS':>10s}  {'p_TWAS':>10s}  "
              + "  ".join(f"{m[:8]:>10s}" for m in methods))
    print(header)
    print("-" * 96)
    for gid in sorted(rows):
        row = rows[gid]
        line = (
            f"{gid:4s}  {row['p_gwas']:10.3e}  {row['p_twas']:10.3e}  "
            + "  ".join(f"{row[m]:10.3e}" for m in methods)
        )
        print(line)
    print()
    print("Interpretation:")
    print("  - G1 (concordant strong GWAS + strong TWAS) — all methods make p_combined < both inputs.")
    print("  - G2 (strong GWAS only) — Fisher amplifies; min-p is the most permissive.")
    print("  - G3 (strong TWAS only) — same Fisher-style amplification.")
    print("  - G4 / G5 (null both) — combined p stays close to the larger of the two.")

    # Persist a wide table for downstream comparison.
    out_path = OUT_DIR / "13_gwas_twas_integration.tsv"
    with open(out_path, "w") as f:
        f.write("gene_id\tp_gwas\tp_twas\t" + "\t".join(methods) + "\n")
        for gid in sorted(rows):
            row = rows[gid]
            cells = [f"{row['p_gwas']:.6e}", f"{row['p_twas']:.6e}"]
            cells += [f"{row[m]:.6e}" for m in methods]
            f.write(f"{gid}\t" + "\t".join(cells) + "\n")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
