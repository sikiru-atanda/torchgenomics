# TorchGWAS

**GPU-accelerated Genome-Wide Association Studies with PyTorch.**

TorchGWAS is a modular Python library that brings GPU acceleration to GWAS pipelines.
It replicates and extends established tools like GEMMA and GAPIT, achieving
4th-decimal-place p-value agreement while adding novel statistical models, native
C++ accelerators, and support for both diploid and polyploid organisms.

## Status

- **Version**: 0.3.9 (Alpha)
- **Tests**: 3038 pass, 492 skipped (full suite green)
- **CLI subcommands**: 40
- **Python**: 3.10, 3.11, 3.12, 3.13
- **Platforms**: Linux + macOS + Windows (CPU all platforms; GPU Linux + macOS)
- **License**: MIT

## What you can do with TorchGWAS

**Variant-level GWAS**

- **Classical** GWAS models (GLM, single- and multi-trait LMM, FarmCPU, BLINK)
- **Mixed-model extensions** — multi-kernel, GxE / heteroscedastic, orthogonal
  cross-fit (with optional ridge_quadratic nonlinear nuisance learner),
  knockoff FDR, genotype-uncertainty, leave-region-out LOCO
- **Categorical traits** — binary / ordinal / multinomial GLM and GLMM
  (PQL, SPA), multi-environment GLMM, Bermann-2026 threshold-linear
- **Multi-environment GWAS** — reaction-norm, FA(k), multi-kernel,
  MT-MET separable Kronecker
- **Longitudinal traits** via random regression (Legendre / B-spline /
  spatio-temporal 2D P-spline)
- **Polyploid GWAS** at arbitrary ploidy *k* (dosage calling via updog,
  F1 phasing via PolyOrigin, per-SNP gene-action testing)

**Haplotype layer**

- 13 LD-block detection methods (Gabriel, four-gamete, BigLD, CC-graph,
  spine, change-point, …; PLINK 1.9 `--blocks` validated)
- 9 haplotype-GWAS tests: HTR / block / window / SKAT + 5 novel methods
  (PCHT / HHCT / HSKAT / HapGxE / BayesHap)
- Multi-env / multi-trait / MT-MET haplotype extensions

**Transcriptome-wide association (TWAS)**

- **Three TWAS workflows**:
  - `twas_sumstat()` — S-PrediXcan-style on GWAS sumstats + pre-trained eQTL weights
  - `twas_individual()` — PrediXcan-style on genotypes + weights → GReX → OLS
  - `twas_observed_expression()` — FUSION measured-expression mode on
    already-normalized RNA-seq counts, supports OLS + LMM (kinship-corrected)
- **PrediXcan / FUSION `.db` reader** (`torchgwas.io.read_predixcan_db`) for
  GTEx / PsychENCODE / UTMOST model packs
- **Multi-tissue stacking + S-MultiXcan-style aggregation** —
  `twas_multi_tissue_stack`, `twas_multi_tissue_aggregate`
- **Preprocessing helpers** — rank-INT (Blom), quantile normalization,
  PEER residualization (R subprocess wrapper)
- **Gene-level visualizations** — `manhattan_twas`, `qq_twas`,
  `genomic_inflation_factor_twas`

**GWAS↔TWAS integration**

- `combine_gwas_twas()` entry point with 14 combination methods:
  - **8 classical kernels**: Fisher's combined test, Brown 1975, Empirical
    Brown (Kost-McDermott), harmonic mean p (Wilson 2019), truncated product
    (Zaykin 2002), min-p (Tippett), Cauchy / ACAT (Liu & Xie 2020), Stouffer
  - **6 novel methods** (each "to our knowledge"-tagged): R²-weighted
    Stouffer, LD-aware Brown via eigenMT, polyploid gene-action Fisher,
    multi-tissue Cauchy + lead-SNP, conditional GWAS+TWAS via COJO,
    hyprcoloc-PPFC-gated integration

**Post-GWAS**

- LDSC h²/rg, S-LDSC, meta-analysis (IVW/DL/Stouffer/RE2), LD clumping
- Coloc, HyPrColoc, MR (IVW/Egger/weighted-median/MR-PRESSO), SMR + HEIDI
  with LD-weighted variance, TWAS, fine-mapping (SuSiE, SuSiE-RSS), gene-set
  enrichment (MAGMA-style + custom), HESS regional heritability
- **Polygenic scores** — C+T, LDpred2 (Inf / Grid / Auto), PRS-CS
- **NCBI gene annotation** (`torchgwas annotate` CLI)

**Multi-omics integration**

- GRM-corrected causal mediation (`mediate_lmm`, `scan_mediation`),
  multi-kernel heritability, coloc prefilter

## Quickstart

```bash
pip install torchgwas
torchgwas lmm-scan --genotype data.bed --phenotype pheno.txt --correction bh
```

See [Getting Started](getting-started/installation.md) for the full path from
install to first scan.

## Validation

TorchGWAS is validated against GEMMA 0.98.5, GAPIT3, and GWASpoly on the MDP
maize and GWASpoly tetraploid potato fixtures. See the [Validation page](validation.md)
for the Section-16 tolerances and the current golden-data CI gate.

## Where to go next

- [Installation](getting-started/installation.md) — CPU / GPU / Windows install paths
- [Quickstart](getting-started/quickstart.md) — run your first LMM scan in under 5 minutes
- [CLI Reference](cli.md) — every subcommand with examples
- [API Reference](api/index.md) — module-level auto-generated documentation
- [Tutorials](tutorials/single-trait-lmm.md) — end-to-end walkthroughs
