# rTorchGenomics

[![R-CMD-check](https://github.com/sikiru-atanda/torchgenomics/actions/workflows/rcheck.yml/badge.svg)](https://github.com/sikiru-atanda/torchgenomics/actions/workflows/rcheck.yml)
[![r-universe](https://sikiru-atanda.r-universe.dev/badges/rTorchGenomics)](https://sikiru-atanda.r-universe.dev/rTorchGenomics)

GPU-accelerated statistical and quantitative genomics on PyTorch — GWAS +
post-GWAS + PGS + LD + imputation + annotation + visualization, exposed
as a clean R interface via [reticulate](https://rstudio.github.io/reticulate/).

Same engine as the Python [`torchgenomics`](https://pypi.org/project/torchgenomics/)
package (v0.4.0), with R-idiomatic ergonomics: S4 result classes, tibble
top-hits tables, and native [ggplot2](https://ggplot2.tidyverse.org/) plots.

## What's new in 0.4.1

- `tg_lgebv()` — Local Genomic Estimated Breeding Values per haplo-block
  (Endelman 2011 rrBLUP; Pandit et al. 2026 breeding-program workflow).
- `tg_iclass()` — G x E classification on factor-analytic loadings
  (Smith et al. 2015 / 2021); environments are clustered by polarity
  pattern.
- `tg_ld_blocks(..., tolerance = N)` — new `tolerance` parameter on
  `method = "r2"` (SelectionTools-style; Wittenburg et al. 2024).
- `vignette("breeding-program-haplotype-workflow")` — barley-style
  end-to-end workflow that chains the three additions.

## Install

```r
install.packages(
  "rTorchGenomics",
  repos = c("https://sikiru-atanda.r-universe.dev", getOption("repos"))
)
library(rTorchGenomics)

# One-time Python runtime setup. Idempotent.
tg_install()
```

`tg_install()` creates a Python virtualenv called `r-rtorchgenomics` and
installs `torchgenomics[mcp]==0.4.0` into it. If you don't have Python
>= 3.10 on `PATH`, reticulate installs it first. Use
`tg_install(method = "conda")` if you prefer a conda env, or
`tg_install(envname = "...")` to pin a custom env name.

## Quickstart

```r
library(rTorchGenomics)

r <- tg_lmm_scan(
  genotype  = "data.bed",
  phenotype = "pheno.tsv"
)
print(r)
#> ScanRun (SingleTraitLMM, test=wald, correction=bh)
#>   Samples: 2500   Variants tested: 489102
#>   Significant @ alpha=5.00e-08: 12
#>   lambda_GC: 1.012
#>   Output dir: /tmp/torchgenomics_run_lmm_2026.../

top_hits(r)                    # tibble of top hits by p-value
output_files(r)$tsv            # path to the full result table on disk
tg_manhattan(r)                # ggplot2 Manhattan plot
tg_qq(r)                       # ggplot2 Q-Q plot
```

All scan wrappers return an S4 `ScanRun` (or `LDBlocksResult`, `PGSFit`,
`MetaResult`, ...) — see `?ScanRun` and `vignette("result-classes")`.

## What's available

**51 functions total: 13 hand-crafted polished + 32 auto-generated tier-2 + 2 plots + 4 install / bridge / codegen utilities.**

**13 hand-crafted polished wrappers** (rich docs, defaults, validation):

- Data: `tg_validate`, `tg_convert`, `tg_impute`
- Scans: `tg_lmm_scan`, `tg_glm_scan`
- LD + post-GWAS: `tg_ld_blocks`, `tg_clump`, `tg_meta`
- PGS: `tg_pgs_fit`, `tg_pgs_score`
- Annotation: `tg_annotate_hits`
- Breeding-program haplotype workflow (new in 0.4.1): `tg_lgebv`, `tg_iclass`

**32 auto-generated tier-2 wrappers** (codegen from Python manifest):

- Multi-trait / multi-env: `tg_mvlmm_scan`, `tg_mtmet_scan`, `tg_met_scan`
- Polyploid: `tg_poly_scan`, `tg_dosage_call`, `tg_phase_poly`
- GLMM family: `tg_glmm_scan`, `tg_me_glmm_scan`
- Specialty mixed models: `tg_set_scan`, `tg_bayes_scan`,
  `tg_bayes_scan_rss`, `tg_threshold_scan`, `tg_family_scan`,
  `tg_conditional_scan`, `tg_ocf_scan`, `tg_knockoff_scan`, `tg_gu_scan`,
  `tg_lro_scan`, `tg_survival_scan`, `tg_rr_scan`, `tg_rr_met_scan`
- Iterative: `tg_farmcpu_scan`, `tg_blink_scan`
- GxE: `tg_gxe_scan`
- Multi-kernel: `tg_mklmm_scan`
- TWAS: `tg_twas_scan`, `tg_combine_gwas_twas`
- LDSC: `tg_ldsc`, `tg_ldsc_rg`
- Mediation: `tg_mediate`, `tg_mediate_scan`
- Pipeline: `tg_pipeline`

**2 ggplot2 plot helpers:**

- `tg_manhattan` — Manhattan plot from a `ScanRun` or summary-stats tibble
- `tg_qq` — Q-Q plot with reference line and 95% confidence band

See `?<function>` for argument docs, or the
[pkgdown site](https://sikiru-atanda.github.io/torchgenomics/r/) for the
full reference.

## Architecture

R wraps the Python `torchgenomics` engine via
[reticulate](https://rstudio.github.io/reticulate/). The data flow is:

1. R user calls e.g. `tg_lmm_scan(...)`.
2. The bridge (`R/bridge.R`) marshals R arguments to Python via reticulate.
3. Python `torchgenomics.api.lmm_scan` runs the scan (CPU or GPU).
4. The Python `ScanRun` object's `.to_dict()` is returned to R.
5. R's `ScanRun$new_from_dict()` reconstructs an S4 result object with
   tibble fields and the same `output_files()` / `top_hits()` / plotting
   API as the Python side.

The MCP server (`torchgenomics-mcp` console script) is for LLM tool
clients (Claude, GPT) — **not for R**. The R bridge talks directly to
the in-process Python interpreter via reticulate, no MCP, no JSON-RPC,
no subprocess.

See `vignette("reticulate-bridge")` for the full implementation details.

## Versioning

`rTorchGenomics` is version-locked to `torchgenomics`:

| rTorchGenomics | Python torchgenomics | Status |
| -------------- | -------------------- | ------ |
| 0.4.1          | 0.4.0+ (incl. iclass / lgebv) | current |
| 0.4.0          | 0.4.0                | previous |

`tg_install()` pins `torchgenomics[mcp]==0.4.0` to keep the two sides in
lock-step. Bumping the R package version also bumps the pinned Python
version.

## Citation

If you use rTorchGenomics in a publication, please cite:

> Atanda, S. (2026). TorchGenomics: GPU-accelerated statistical and
> quantitative genomics on PyTorch. Manuscript in preparation.

A BibTeX entry is available via `citation("rTorchGenomics")` after
install.

## License

MIT. See [`LICENSE`](LICENSE).
