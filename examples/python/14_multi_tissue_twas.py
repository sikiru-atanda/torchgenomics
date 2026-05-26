"""Multi-tissue TWAS aggregation + gene-level visualization.

Demonstrates the multi-tissue surface added in v0.3.9:

  - ``twas_multi_tissue_stack`` — long-format ``{tissue: TWASResult}``
    → flat list of (gene, tissue) rows for easy DataFrame work.
  - ``twas_multi_tissue_aggregate`` — per-gene S-MultiXcan-style χ²
    summary with driver-tissue identification.
  - ``manhattan_twas`` / ``qq_twas`` / ``genomic_inflation_factor_twas``
    — gene-level visualizations including a Beta-CI band on the QQ
    plot.

Phases covered:
- Multi-tissue TWAS (added 2026-05-26)
- Gene-level TWAS plots (added 2026-05-26)

Runtime: < 5 s on CPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchgwas.postgwas import (
    TWASGeneResult,
    TWASResult,
    twas_multi_tissue_aggregate,
    twas_multi_tissue_stack,
)
from torchgwas.viz import (
    genomic_inflation_factor_twas,
    manhattan_twas,
    qq_twas,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate_tissue_twas(n_genes: int = 200, n_strong: int = 4,
                        tissue_name: str = "Liver", seed: int = 0) -> TWASResult:
    """Generate a tissue's TWAS table with a small number of strong
    signals planted on top of a roughly-normal z-score background."""
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(n_genes, generator=gen, dtype=torch.float64)
    if n_strong > 0:
        # Plant signals at fixed indices so we can verify multi-tissue
        # aggregation finds them.
        z[:n_strong] = z[:n_strong] + 5.5
    from scipy.special import erfc
    p = np.clip(erfc(z.abs().numpy() / math.sqrt(2.0)), 1e-300, 1.0)

    genes = []
    for i in range(n_genes):
        genes.append(TWASGeneResult(
            gene_id=f"G{i:04d}",
            z_twas=float(z[i].item()),
            p_twas=float(p[i]),
            n_cis_snps=0, r2_model=float(np.clip(0.25 - 0.0003 * i, 0.05, 0.4)),
            top_weight_snp=None,
            beta=float(z[i].item()) * 0.1, se=0.1,
            chr=f"chr{1 + (i % 22)}",
            start=(i + 1) * 1_000_000,
            end=(i + 1) * 1_000_000 + 50_000,
            gene_name=f"NAME_G{i:04d}",
        ))
    return TWASResult(genes=genes, n_genes_tested=n_genes, n_significant=n_strong)


def main() -> None:
    # ----- Simulate 4 tissues with shared + tissue-specific signals --------
    per_tissue = {
        "Liver":  simulate_tissue_twas(200, n_strong=4, tissue_name="Liver",  seed=1),
        "Blood":  simulate_tissue_twas(200, n_strong=3, tissue_name="Blood",  seed=2),
        "Brain":  simulate_tissue_twas(200, n_strong=5, tissue_name="Brain",  seed=3),
        "Muscle": simulate_tissue_twas(200, n_strong=2, tissue_name="Muscle", seed=4),
    }

    # ----- Long-format stack (one row per gene per tissue) -----------------
    long_rows = twas_multi_tissue_stack(per_tissue)
    print(f"Long-format rows: {len(long_rows)}  (4 tissues × 200 genes)")
    print(f"First 4 rows:")
    for r in long_rows[:4]:
        print(f"  {r.gene_id} {r.tissue:6s}  z={r.z_twas:+6.3f}  "
              f"p={r.p_twas:.3e}  r²={r.r2_model:.3f}")

    # ----- Per-gene S-MultiXcan-style aggregation --------------------------
    summaries = twas_multi_tissue_aggregate(per_tissue)
    print(f"\nPer-gene aggregated rows: {len(summaries)}")
    print(f"Top 5 genes by p_multixcan:")
    for s in summaries[:5]:
        print(f"  {s.gene_id}  n_tissues={s.n_tissues}  "
              f"χ²={s.chi2_multixcan:6.2f}  p={s.p_multixcan:.3e}  "
              f"driver={s.driver_tissue}")

    # ----- λ_TWAS per tissue ------------------------------------------------
    print(f"\nGenomic inflation factor λ_TWAS per tissue:")
    for tissue, result in per_tissue.items():
        lam = genomic_inflation_factor_twas(result)
        print(f"  {tissue:6s}  λ_TWAS = {lam:.3f}  "
              f"({'OK' if 0.85 < lam < 1.15 else 'inflated'})")

    # ----- Plot Liver's Manhattan + QQ -------------------------------------
    liver = per_tissue["Liver"]

    fig, ax = plt.subplots(figsize=(11, 4))
    manhattan_twas(liver, ax=ax, label_top=5)
    fig.tight_layout()
    manhattan_path = OUT_DIR / "14_multi_tissue_twas_manhattan_liver.png"
    fig.savefig(manhattan_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    qq_twas(liver, ax=ax, confidence=True)
    fig.tight_layout()
    qq_path = OUT_DIR / "14_multi_tissue_twas_qq_liver.png"
    fig.savefig(qq_path, dpi=120)
    plt.close(fig)

    print(f"\nPlots written to:")
    print(f"  {manhattan_path}")
    print(f"  {qq_path}")

    # ----- Persist the aggregated table ------------------------------------
    out_path = OUT_DIR / "14_multi_tissue_twas.tsv"
    with open(out_path, "w") as f:
        f.write("gene_id\tn_tissues\tz_min\tz_max\tz_mean\tz_sd\t"
                "p_min\tchi2_multixcan\tp_multixcan\tdriver_tissue\n")
        for s in summaries:
            f.write(
                f"{s.gene_id}\t{s.n_tissues}\t{s.z_min:+.4f}\t{s.z_max:+.4f}\t"
                f"{s.z_mean:+.4f}\t{s.z_sd:.4f}\t{s.p_min:.3e}\t"
                f"{s.chi2_multixcan:.4f}\t{s.p_multixcan:.3e}\t"
                f"{s.driver_tissue}\n"
            )
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
