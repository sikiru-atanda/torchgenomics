# TorchGenomics — Python example scripts

Numbered, runnable demonstrations of each major capability in
TorchGenomics. Every script is self-contained: simulates or loads a small
fixture, runs the relevant scan / integration / visualization, and
writes results (TSV + PNG) under `results/` next to the script.

## Run convention

The scripts assume TorchGenomics resolves on the Python path. From the
repository root:

```bash
pip install -e .               # once
python examples/python/01_single_trait_lmm.py
```

Or, without an editable install:

```bash
PYTHONPATH=. python examples/python/01_single_trait_lmm.py
```

Each script is also CLI-runnable through `torchgenomics <subcommand>` —
see the comments at the top of each file for the CLI equivalent.

## Index

### Variant-level GWAS

| # | Script | What it shows |
|---|---|---|
| 01 | `01_single_trait_lmm.py` | Single-trait LMM on a self-contained synthetic fixture, two ways: the one-call `tg.gwas` API and the low-level `SingleTraitLMM`; BH correction; Manhattan + QQ. |
| 02 | `02_multi_trait_lmm.py` | mvLMM with PX-EM + AI-REML on two correlated MDP traits. |
| 03 | `03_met_scan.py` | Multi-environment GWAS (`MET`) with reaction-norm + FA(k) variance structures. |
| 04 | `04_threshold_linear.py` | Bermann-2026 threshold-linear multi-trait model for ordinal + continuous traits. |
| 09 | `09_polyploid_gwas.py` | Polyploid GWAS at arbitrary ploidy *k* with per-SNP gene-action testing. |
| 10 | `10_haplotype_gwas.py` | Haplotype-trend regression + window scan on a maize block. |

### Polygenic scores + fine-mapping

| # | Script | What it shows |
|---|---|---|
| 05 | `05_pgs_construction.py` | Compose LDpred2-Auto + PRS-CS PGS from sumstats + LD ref. |
| 06 | `06_fine_mapping.py` | SuSiE / SuSiE-RSS credible-set fine-mapping. |

### Multi-omics, annotation, mediation

| # | Script | What it shows |
|---|---|---|
| 07 | `07_mediation.py` | GRM-corrected causal mediation (`mediate_lmm`) with Imai ρ-sensitivity. |
| 08 | `08_annotate_hits.py` | NCBI gene annotation for GWAS hits (`torchgenomics annotate`). |

### TWAS + GWAS↔TWAS integration (2026-05-26)

| # | Script | What it shows |
|---|---|---|
| 11 | `11_twas_observed_expression.py` | FUSION measured-expression workflow on normalized RNA-seq counts. Compares OLS vs LMM (kinship-corrected) paths through `twas_observed_expression`. Demonstrates the GTEx-style rank-INT + quantile-norm preprocessing chain. |
| 12 | `12_twas_sumstat_predixcan_db.py` | S-PrediXcan TWAS reading three tissues' weights from PrediXcan-format `.db` files via `torchgenomics.io.read_predixcan_db`, then multi-tissue stacking + S-MultiXcan-style χ² aggregation. |
| 13 | `13_gwas_twas_integration.py` | All 8 classical combination methods (Fisher / Stouffer / Cauchy / Brown / Empirical Brown / HMP / truncated product / min-p) run against the same 5-gene fixture so the relative behavior across methods is directly comparable. Wide TSV output. |
| 14 | `14_multi_tissue_twas.py` | 4-tissue TWAS with `twas_multi_tissue_stack` + `twas_multi_tissue_aggregate`; gene-level Manhattan + QQ plots + λ_TWAS via `torchgenomics.viz.{manhattan_twas, qq_twas, genomic_inflation_factor_twas}`. |
| 15 | `15_combine_methods_comparison.py` | Showcase of the **six novel methods**: R²-weighted Stouffer, LD-aware Brown via eigenMT, polyploid gene-action Fisher, multi-tissue ACAT + lead-SNP, conditional GWAS+TWAS via COJO, hyprcoloc-PPFC-gated combination. Each section is compact so the calling shape of each function is visible in one place. |

## Outputs

Every script writes its artifacts under `results/` next to this README.
TSVs are tab-separated with a header row; PNGs are 120 dpi by default
(override at the matplotlib `savefig` call).

## R examples

Parallel R-side examples live under `examples/R/`. The TWAS + GWAS↔TWAS
integration work in `examples/python/11_*` – `15_*` is Python-only at
present (no R port yet); the underlying integration kernels (Fisher,
Stouffer, etc.) have R-side references in `validation/external/metap/`
for users who want to compare TorchGenomics outputs against `metap` /
`EmpiricalBrownsMethod`.
