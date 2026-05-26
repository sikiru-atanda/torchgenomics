"""Showcase the six novel GWAS↔TWAS integration methods.

Demonstrates each of the six novel combination kernels added in v0.3.9,
illustrating how each one exploits a specific piece of TorchGWAS
infrastructure that no published TWAS-integration tool currently uses
(to our knowledge):

  1. ``stouffer_r2_weighted``                — eQTL CV-R² weighting.
  2. ``brown_ld_aware``                      — Brown's via eigenMT.
  3. ``fisher_polyploid_gene_action``        — autopolyploid gene-action.
  4. ``cauchy_multi_tissue_plus_lead_snp``   — S-MultiXcan + lead SNP.
  5. ``gwas_twas_conditional``               — COJO-mediated.
  6. ``gwas_twas_hyprcoloc_gated``           — hyprcoloc PPFC gate.

Each section runs the method on a tiny synthetic fixture and prints
the result. Code is intentionally compact so a reader can see how
each function's signature works in isolation.

Runtime: < 2 s on CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch

from torchgwas.postgwas import (
    brown_ld_aware,
    cauchy_multi_tissue_plus_lead_snp,
    fisher_polyploid_gene_action,
    gwas_twas_conditional,
    gwas_twas_hyprcoloc_gated,
    stouffer_combined,
    stouffer_r2_weighted,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. stouffer_r2_weighted — propagate eQTL prediction uncertainty
    # ------------------------------------------------------------------
    section("1. R²-weighted Stouffer (GReX-uncertainty-propagating)")
    # Three tissues; tissue 2 has very low CV-R² (poor expression model)
    # so its TWAS contribution should be down-weighted in the combination.
    p_per_tissue = [1e-6, 1e-6, 0.5]
    r2_models   = [0.30, 0.02, 0.30]
    z_eq, p_eq = stouffer_combined(p_per_tissue)
    z_r2, p_r2 = stouffer_r2_weighted(p_per_tissue, r2_models)
    print(f"  Unweighted Stouffer: z = {z_eq:+.4f}, p = {p_eq:.3e}")
    print(f"  R²-weighted   z = {z_r2:+.4f}, p = {p_r2:.3e}  "
          f"(tissue 2 down-weighted because R² = 0.02)")

    # ------------------------------------------------------------------
    # 2. brown_ld_aware — Brown's correlated-p test with eigenMT
    # ------------------------------------------------------------------
    section("2. LD-aware Brown's via eigenMT")
    p_per_snp = [0.01, 0.01, 0.01, 0.01]
    # Two LD regimes: independent SNPs and strongly-LD-linked SNPs.
    ld_indep = torch.eye(4, dtype=torch.float64)
    ld_strong = torch.full((4, 4), 0.85, dtype=torch.float64)
    ld_strong.fill_diagonal_(1.0)
    _, p_indep  = brown_ld_aware(p_per_snp, ld_indep)
    _, p_strong = brown_ld_aware(p_per_snp, ld_strong)
    print(f"  Independent LD (eye(4)):  p_combined = {p_indep:.4e}")
    print(f"  Strong LD (r = 0.85):     p_combined = {p_strong:.4e}  "
          "(correctly inflated — fewer effective tests)")

    # ------------------------------------------------------------------
    # 3. fisher_polyploid_gene_action — two-stage Cauchy → Fisher
    # ------------------------------------------------------------------
    section("3. Polyploid gene-action Fisher (autopolyploid GWAS)")
    # 5 SNPs × 3 gene-action models (e.g., additive, simplex-dominant,
    # duplex-dominant at k=4 ploidy). One SNP shows signal across all
    # three models, four are background.
    p_per_snp_per_action = torch.tensor([
        [1e-5, 1e-4, 1e-5],   # SNP 0: signal in all 3 gene actions
        [0.50, 0.45, 0.55],   # SNP 1: background
        [0.30, 0.35, 0.40],   # SNP 2: background
        [0.60, 0.70, 0.65],   # SNP 3: background
        [0.20, 0.25, 0.30],   # SNP 4: weakly elevated
    ], dtype=torch.float64)
    stat, p_combined = fisher_polyploid_gene_action(p_per_snp_per_action)
    print(f"  Stage 1: Cauchy across {p_per_snp_per_action.shape[1]} "
          "gene-action models per SNP")
    print(f"  Stage 2: Fisher across {p_per_snp_per_action.shape[0]} SNPs")
    print(f"  Combined: stat = {stat:.3f}, p_gene = {p_combined:.4e}")

    # ------------------------------------------------------------------
    # 4. cauchy_multi_tissue_plus_lead_snp — S-MultiXcan + lead-SNP
    # ------------------------------------------------------------------
    section("4. Multi-tissue ACAT + lead-SNP GWAS")
    p_tissues = [0.05, 0.10, 0.20]
    # Two scenarios: weak vs strong lead-SNP GWAS evidence.
    _, p_weak   = cauchy_multi_tissue_plus_lead_snp(p_tissues, 0.30)
    _, p_strong = cauchy_multi_tissue_plus_lead_snp(p_tissues, 1e-8)
    print(f"  Tissues only (no GWAS):  ACAT-only p = approx similar shape")
    print(f"  + weak GWAS lead (0.30):   p_combined = {p_weak:.4e}")
    print(f"  + strong GWAS lead (1e-8): p_combined = {p_strong:.4e}  "
          "(lead SNP drives the combination)")

    # ------------------------------------------------------------------
    # 5. gwas_twas_conditional — COJO-mediated combination
    # ------------------------------------------------------------------
    section("5. Conditional GWAS+TWAS (mediation vs independence)")
    # Scenario A: TWAS lead SNP fully explains the GWAS signal
    # (conditional GWAS p drops from 1e-8 to 0.5).
    p, score, status = gwas_twas_conditional(
        p_gwas=1e-8, p_twas=1e-6, p_gwas_conditional=0.5,
    )
    print(f"  A. p_GWAS=1e-8, p_TWAS=1e-6, p_GWAS|TWAS=0.5")
    print(f"     -> status={status:11s}  p_reported = {p:.3e}  "
          f"(log10 ratio = {score:.2f})")
    # Scenario B: TWAS lead SNP does NOT explain GWAS (independent signals).
    p, score, status = gwas_twas_conditional(
        p_gwas=1e-5, p_twas=1e-3, p_gwas_conditional=1e-5,
    )
    print(f"  B. p_GWAS=1e-5, p_TWAS=1e-3, p_GWAS|TWAS=1e-5")
    print(f"     -> status={status:11s}  p_combined  = {p:.3e}  "
          "(Fisher's used because signals are independent)")

    # ------------------------------------------------------------------
    # 6. gwas_twas_hyprcoloc_gated — PPFC gate
    # ------------------------------------------------------------------
    section("6. Hyprcoloc-PPFC-gated combination")
    # Same GWAS / TWAS p, two coloc scenarios.
    p_lo, _, status_lo = gwas_twas_hyprcoloc_gated(
        p_gwas=1e-5, p_twas=1e-4, ppfc=0.20, ppfc_threshold=0.80,
    )
    p_hi, _, status_hi = gwas_twas_hyprcoloc_gated(
        p_gwas=1e-5, p_twas=1e-4, ppfc=0.95, ppfc_threshold=0.80,
    )
    print(f"  Low PPFC (0.20):  status={status_lo:13s}  p_reported = {p_lo:.3e}  "
          "(uncolocalised — report p_TWAS only)")
    print(f"  High PPFC (0.95): status={status_hi:13s}  p_combined = {p_hi:.3e}  "
          "(colocalised — combine via Fisher)")

    # ------------------------------------------------------------------
    # Persist a per-method summary table.
    # ------------------------------------------------------------------
    out_path = OUT_DIR / "15_combine_methods_comparison.tsv"
    with open(out_path, "w") as f:
        f.write("scenario\tmethod\tp_combined\n")
        f.write(f"1_low_R2_tissue_2\tstouffer_r2_weighted\t{p_r2:.3e}\n")
        f.write(f"1_unweighted_3tissue\tstouffer\t{p_eq:.3e}\n")
        f.write(f"2_independent_LD\tbrown_ld_aware\t{p_indep:.3e}\n")
        f.write(f"2_r0.85_LD\tbrown_ld_aware\t{p_strong:.3e}\n")
        f.write(f"3_polyploid_5x3\tfisher_polyploid_gene_action\t{p_combined:.3e}\n")
        f.write(f"4_weak_lead\tcauchy_multi_tissue_plus_lead_snp\t{p_weak:.3e}\n")
        f.write(f"4_strong_lead\tcauchy_multi_tissue_plus_lead_snp\t{p_strong:.3e}\n")
        f.write(f"5_low_PPFC\tgwas_twas_hyprcoloc_gated\t{p_lo:.3e}\n")
        f.write(f"5_high_PPFC\tgwas_twas_hyprcoloc_gated\t{p_hi:.3e}\n")
    print(f"\nResults table written to {out_path}")


if __name__ == "__main__":
    main()
