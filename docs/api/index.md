# API Reference

TorchGWAS is organized into a set of focused modules. Each page below is auto-
generated from the module-level and function-level docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/).

| Module                    | What it does                                                |
|---------------------------|-------------------------------------------------------------|
| `torchgwas.io`            | Genotype + phenotype readers (PLINK, VCF, BGEN, HapMap, Zarr, CSV) |
| `torchgwas.preprocess`    | Imputation, QC, standardization, polyploid encoding         |
| `torchgwas.linalg`        | GRM, eigendecomposition, batched Cholesky                   |
| `torchgwas.models`        | All GWAS models (BaseModel protocol)                        |
| `torchgwas.scan`          | `UnifiedScanner` streaming chunks through any model         |
| `torchgwas.stats`         | Multiple testing, SPA, PVE                                  |
| `torchgwas.ld`            | LD block detection (13 methods), pairwise LD                |
| `torchgwas.optim`         | 6-mode optimizer stack                                      |
| `torchgwas.pgs`           | Polygenic score construction + validation                   |
| `torchgwas.postgwas`      | LDSC, meta-analysis, coloc, MR, TWAS, enrichment            |
| `torchgwas.multiomics`    | GRM-corrected causal mediation + multi-kernel h²            |
| `torchgwas.viz`           | Manhattan / QQ / Miami / Circos / Haploview / trumpet       |
| `torchgwas.annotate`      | NCBI gene annotation                                        |
| `torchgwas.results`       | Result containers + I/O                                     |

## Top-level package

::: torchgwas
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
