"""Observed-expression TWAS on simulated RNA-seq + phenotype.

Demonstrates the FUSION measured-expression workflow: when the user has
already-normalized gene expression in their discovery cohort (no
pre-trained eQTL weights needed), test gene-trait associations directly.
Shows both the OLS path (no kinship) and the LMM path (kinship-corrected)
through the same ``twas_observed_expression`` entry point.

Phases covered:
- TWAS observed-expression (added 2026-05-26)
- LMM machinery (Phase 4 SingleTraitLMM reuse)
- Multiple-testing correction (BH)

Runtime: < 5 s on CPU (200 samples × 20 genes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from torchgwas.postgwas import twas_observed_expression
from torchgwas.preprocess import inverse_normal_transform, quantile_normalize

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def simulate(n=200, n_genes=20, causal_idx=7, effect=0.5,
             with_population_structure=True, seed=42):
    """Plant a single causal gene; optionally add population structure."""
    gen = torch.Generator().manual_seed(seed)
    expression = torch.randn(n, n_genes, generator=gen, dtype=torch.float64)
    covariates = torch.cat([
        torch.randn(n, 3, generator=gen, dtype=torch.float64),  # 3 PCs
        torch.randint(0, 2, (n, 1), generator=gen, dtype=torch.float64),  # sex
    ], dim=1)

    if with_population_structure:
        # Build a kinship matrix from a "background" genotype block, then
        # add a polygenic random effect tied to it. This is the signal a
        # kinship-corrected LMM is designed to absorb.
        bg_geno = torch.randint(
            0, 3, (n, 500), generator=gen, dtype=torch.float64,
        )
        bg_geno_c = bg_geno - bg_geno.mean(dim=0, keepdim=True)
        sd = bg_geno_c.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-9)
        bg_geno_s = bg_geno_c / sd
        K = (bg_geno_s @ bg_geno_s.T) / 500.0
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        u = L @ torch.randn(n, generator=gen, dtype=torch.float64)
    else:
        K = None
        u = torch.zeros(n, dtype=torch.float64)

    y = (
        effect * expression[:, causal_idx]
        + 0.2 * covariates[:, 0]
        + 0.6 * u
        + 0.3 * torch.randn(n, generator=gen, dtype=torch.float64)
    )
    gene_ids = [f"ENSG_{i:04d}" for i in range(n_genes)]
    return expression, y, gene_ids, covariates, K


def main() -> None:
    expr, y, gene_ids, covariates, K = simulate()

    # ----- Optional GTEx-style preprocessing pipeline -----------------------
    # The canonical chain is quantile-normalize → rank-INT → PEER residualize.
    # PEER requires R + the 'peer' package; here we just demonstrate the
    # first two pure-Python transforms.
    expr_qn = quantile_normalize(expr)
    expr_int = inverse_normal_transform(expr_qn)

    # ----- OLS path (no kinship) -------------------------------------------
    print("=" * 72)
    print("OLS path (no kinship correction)")
    print("=" * 72)
    res_ols = twas_observed_expression(
        expr_int, y, gene_ids,
        covariates=covariates,
        correction="bh",
        p_threshold=0.05,
    )
    print(f"  n_genes_tested = {res_ols.n_genes_tested}")
    print(f"  n_significant (BH q < 0.05) = {res_ols.n_significant}")
    print(f"  causal gene (ENSG_0007): β = {res_ols.genes[7].beta:+.4f}, "
          f"p = {res_ols.genes[7].p_twas:.3e}")

    # ----- LMM path (kinship-corrected) -------------------------------------
    print()
    print("=" * 72)
    print("LMM path with kinship K (handles population stratification)")
    print("=" * 72)
    res_lmm = twas_observed_expression(
        expr_int, y, gene_ids,
        covariates=covariates,
        kinship=K,
        correction="bh",
        p_threshold=0.05,
    )
    print(f"  n_genes_tested = {res_lmm.n_genes_tested}")
    print(f"  n_significant (BH q < 0.05) = {res_lmm.n_significant}")
    print(f"  causal gene (ENSG_0007): β = {res_lmm.genes[7].beta:+.4f}, "
          f"p = {res_lmm.genes[7].p_twas:.3e}")

    # ----- Side-by-side dump -----------------------------------------------
    out_path = OUT_DIR / "11_twas_observed_expression.tsv"
    with open(out_path, "w") as f:
        f.write("gene_id\tols_beta\tols_p\tlmm_beta\tlmm_p\n")
        for g_ols, g_lmm in zip(res_ols.genes, res_lmm.genes):
            f.write(
                f"{g_ols.gene_id}\t{g_ols.beta:+.6f}\t{g_ols.p_twas:.4e}\t"
                f"{g_lmm.beta:+.6f}\t{g_lmm.p_twas:.4e}\n"
            )
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
