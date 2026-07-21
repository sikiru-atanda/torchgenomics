# API Reference

TorchGenomics is organized into a set of focused modules. Each page below is auto-
generated from the module-level and function-level docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/).

| Module                    | What it does                                                |
|---------------------------|-------------------------------------------------------------|
| `torchgenomics.io`            | Genotype + phenotype readers (PLINK, VCF, BGEN, HapMap, Zarr, CSV) |
| `torchgenomics.preprocess`    | Imputation, QC, standardization, polyploid encoding         |
| `torchgenomics.linalg`        | GRM, eigendecomposition, batched Cholesky                   |
| `torchgenomics.models`        | All GWAS models (BaseModel protocol)                        |
| `torchgenomics.scan`          | `UnifiedScanner` streaming chunks through any model         |
| `torchgenomics.stats`         | Multiple testing, SPA, PVE                                  |
| `torchgenomics.ld`            | LD block detection (13 methods; `r2` accepts SelectionTools-style `tolerance` from 2026-06-17), pairwise LD |
| `torchgenomics.optim`         | 6-mode optimizer stack                                      |
| `torchgenomics.pgs`           | Polygenic score construction + validation + `api.lgebv` (Local GEBV per haplo-block; consumes pre-computed BLUEs) |
| `torchgenomics.postgwas`      | LDSC, meta-analysis, coloc, MR, TWAS, enrichment, `iclass` G×E clustering on FA loadings (Smith 2015/2021) |
| `torchgenomics.multiomics`    | GRM-corrected causal mediation + multi-kernel h²            |
| `torchgenomics.viz`           | Manhattan / QQ / Miami / Circos / Haploview / trumpet       |
| `torchgenomics.annotate`      | NCBI gene annotation                                        |
| `torchgenomics.results`       | Result containers + I/O                                     |

## Top-level package

::: torchgenomics
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: false
