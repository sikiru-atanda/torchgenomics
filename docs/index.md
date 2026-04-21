# TorchGWAS

**GPU-accelerated Genome-Wide Association Studies with PyTorch.**

TorchGWAS is a modular Python library that brings GPU acceleration to GWAS pipelines.
It replicates and extends established tools like GEMMA and GAPIT, achieving
4th-decimal-place p-value agreement while adding novel statistical models, native
C++ accelerators, and support for both diploid and polyploid organisms.

## Status

- **Version**: 0.1.1 (Alpha)
- **Tests**: 2201 pass, 38 skip across 49 development phases
- **Python**: 3.10, 3.11, 3.12
- **Platforms**: Linux, Windows, macOS
- **License**: MIT

## What you can do with TorchGWAS

- Run **classical** GWAS models (GLM, single- and multi-trait LMM, FarmCPU, BLINK)
- Fit **mixed-model extensions** — multi-kernel, GxE / heteroscedastic, orthogonal
  cross-fit, knockoff FDR, genotype-uncertainty, leave-region-out LOCO
- Handle **categorical traits** — binary / ordinal / multinomial GLM and GLMM
  (PQL, SPA), multi-environment GLMM, Bermann-2026 threshold-linear
- Perform **multi-environment GWAS** — reaction-norm, FA(k), multi-kernel,
  MT-MET separable Kronecker
- Analyze **longitudinal** traits with random regression (Legendre / B-spline /
  spatio-temporal 2D P-spline)
- Scan **haplotypes** with HTR / block / window / SKAT plus 5 novel methods
- Construct **polygenic scores** — C+T, LDpred2 (Inf / Grid / Auto), PRS-CS
- Run **post-GWAS** — LDSC h²/rg, S-LDSC, meta-analysis, coloc / HyPrColoc, MR,
  SMR/HEIDI, TWAS, fine-mapping, gene-set enrichment
- Integrate **multi-omics** — GRM-corrected causal mediation, multi-kernel
  heritability, coloc prefilter

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
