# S2 — Complete software-capability inventory

F1 of the main figures (`paper/reproducibility/render_figures/f1_capability_map.py`) asserts 121 capabilities across 11 clusters; the manifest cell `F1.numbers.cluster_summary` enumerates them. S2 is the full per-capability table. CLI subcommands trace back to the dispatch dict in `torchgwas/cli.py` (lines 103-145, 41 handlers). Module paths follow the package layout under `torchgwas/`. Phase numbers reference the Phase Index in `CLAUDE.md`. Test file column names the closest-matching `tests/test_*.py` file; validation harness column names the `validation/external/<tool>/` or `validation/specialty/<model>/` harness wired for the cluster (multiple capabilities may share one harness — e.g. all polyploid scans inherit `gwaspoly`).

## Capability table

### Cluster 1 — Variant-level GWAS (10 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| GLM | `glm-scan` | `torchgwas.models.glm` | 0 | `tests/test_glm.py` | `validation/external/gapit/` |
| Single-trait LMM | `lmm-scan` | `torchgwas.models.single_trait_lmm` | 4 | `tests/test_lmm.py` | `validation/external/gemma/`, `regenie/`, `gapit/` |
| Multi-trait mvLMM | `mvlmm-scan` | `torchgwas.models.multi_trait_lmm` | 5 | `tests/test_mvlmm.py` | `validation/external/gemma/` |
| FarmCPU | `farmcpu-scan` | `torchgwas.models.farmcpu` | 8 | `tests/test_farmcpu.py` | `validation/external/gapit/` |
| BLINK | `blink-scan` | `torchgwas.models.blink` | 9 | `tests/test_blink.py` | `validation/external/gapit/` |
| GxE LMM | `gxe-scan` | `torchgwas.models.gxe_lmm` | 15 | `tests/test_gxe.py` | self-paired (Phase 15 internal) |
| Multi-kernel LMM | `mklmm-scan` | `torchgwas.models.multi_kernel_lmm` | 14 | `tests/test_multi_kernel.py` | self-paired |
| Set-based (SKAT/Burden/SKAT-O) | `set-scan` | `torchgwas.models.set_based` | 16 | `tests/test_set_based.py` | self-paired (closed-form SKAT) |
| Bayesian VS (SuSiE + CAVI) | `bayes-scan` | `torchgwas.models.bayesian_vs` | 17 | `tests/test_bayesian_vs.py` | `validation/external/susieR/` |
| Bayesian VS-RSS | `bayes-scan-rss` | `torchgwas.models.bayesian_vs_rss` | NA1 (post-Phase 17) | `tests/test_bayesian_vs_rss.py` | `validation/external/susieR/` |

### Cluster 2 — Haplotype layer (22 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| LD blocks: Gabriel | `ld-blocks --method gabriel` | `torchgwas.ld.blocks` | 20 | `tests/test_ld_blocks.py` | `validation/external/plink2/` (--blocks) |
| LD blocks: 4-gamete | `ld-blocks --method four_gamete` | `torchgwas.ld.blocks` | 20 | `tests/test_ld_blocks.py` | self-paired |
| LD blocks: r2 | `ld-blocks --method r2` | `torchgwas.ld.blocks` | 20 | `tests/test_ld_blocks.py` | self-paired |
| LD blocks: spine | `ld-blocks --method spine` | `torchgwas.ld.blocks` | 20 | `tests/test_ld_blocks.py` | self-paired |
| LD blocks: BigLD | `ld-blocks --method big_ld` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (literature) |
| LD blocks: CC-graph | `ld-blocks --method cc_graph` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (literature) |
| LD blocks: DP-optimize | `ld-blocks --method dp_optimize` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (literature) |
| LD blocks: change-point | `ld-blocks --method changepoint` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (novel) |
| LD blocks: cross-pop | `ld-blocks --method cross_pop` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (novel) |
| LD blocks: graphical | `ld-blocks --method graphical` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (novel) |
| LD blocks: GWAS-aligned | `ld-blocks --method gwas_aligned` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (novel) |
| LD blocks: uncertainty | `ld-blocks --method uncertainty` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (novel) |
| LD blocks: Wall-Pritchard diag. | `ld-blocks --method wall_pritchard` | `torchgwas.ld.blocks` | 21 | `tests/test_ld_blocks.py` | self-paired (diagnostic) |
| Haplotype HTR | hap subcommand (Phase 46) | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | `validation/external/hapref/` |
| Haplotype block | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | `validation/external/hapref/` |
| Haplotype window | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | `validation/external/hapref/` |
| Haplotype SKAT | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired |
| PCHT (score test) | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired (novel) |
| HHCT (hierarchical) | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired (novel) |
| HSKAT (similarity kernel) | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired (novel) |
| HapGxE | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired (novel) |
| BayesHap | hap subcommand | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_gwas.py` | self-paired (novel) |

### Cluster 3 — Multi-omics integration (8 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| TWAS (S-PrediXcan) | (via `mediate-scan` adapter) | `torchgwas.postgwas.twas_sumstat` | 45 | `tests/test_postgwas_twas.py` | `validation/external/metaxcan/` |
| SMR + HEIDI | (via `mediate-scan` adapter) | `torchgwas.postgwas.smr_test` | 45 | `tests/test_postgwas_smr.py` | `validation/external/smr/` |
| Coloc (2-trait) | (programmatic API; PostGWAS) | `torchgwas.postgwas.coloc_pairwise` | 42 | `tests/test_postgwas_coloc.py` | `validation/external/coloc/` |
| Hyprcoloc | (programmatic API; PostGWAS) | `torchgwas.postgwas.hyprcoloc` | 42 | `tests/test_postgwas_hyprcoloc.py` | `validation/external/hyprcoloc/` |
| GRM-corrected mediation | `mediate`, `mediate-scan` | `torchgwas.multiomics.mediate_lmm` | 49 | `tests/test_multiomics_mediate.py` | `validation/external/soymd/` |
| Multi-kernel h2 | (programmatic API) | `torchgwas.multiomics.mkernel_h2` | 49b | `tests/test_multiomics_mkernel.py` | `validation/external/ldsc/` (rg/h2 anchor) |
| Gene-set Wald | (via `mediate-scan` aggregator) | `torchgwas.multiomics.mediate_gene_set` | 49b | `tests/test_multiomics_gene_set.py` | self-paired |
| eigenMT FDR | (via stats.eigenmt) | `torchgwas.multiomics.eigenmt_fdr` | 49b | `tests/test_stats_eigenmt.py` | self-paired |

### Cluster 4 — Polyploid pipeline (6 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| Dosage call (updog/polyrad) | `dosage-call` | `torchgwas.preprocess.dosage_call` | 55 | `tests/test_dosage_call.py` | self-paired (HW posterior surrogate; updog wrapper) |
| F1 phasing (PolyOrigin) | `phase-poly` | `torchgwas.preprocess.polyploid_phase` | 56 | `tests/test_phase_polyorigin*.py` | self-paired (Julia subprocess) |
| Polyploid GWAS scan | `poly-scan` | `torchgwas.models.single_trait_lmm` (k-aware) | 0-13 | `tests/test_polyploid_scan.py` | `validation/external/gwaspoly/` |
| Polyploid LD blocks | `ld-blocks` (ploidy-aware) | `torchgwas.ld.blocks` | 20 | `tests/test_ld_blocks_polyploid.py` | self-paired |
| Polyploid haplotype GWAS | hap subcommand (k-aware) | `torchgwas.models.haplotype_gwas` | 46 | `tests/test_haplotype_polyploid.py` | self-paired |
| Gene-action encoding | (preprocess; per-SNP test) | `torchgwas.preprocess.gene_action` | 0-13 | `tests/test_gene_action.py` | `validation/external/gwaspoly/` |

### Cluster 5 — Specialty models (11 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| Survival (Cox PH frailty) | `survival-scan` | `torchgwas.models.survival_glmm` | 35 | `tests/test_survival.py` | `validation/specialty/survival/` (coxme) |
| Random regression LMM | `rr-scan` | `torchgwas.models.random_regression_lmm` | 38 | `tests/test_rr_lmm.py` | `validation/specialty/rr/` (lme4) |
| RR x multi-env | `rr-met-scan` | `torchgwas.models.random_regression_met` | 39 | `tests/test_rr_met.py` | self-paired |
| Spatio-temporal RR | `rr-scan` (spatial flag) | `torchgwas.models.spatio_temporal_rr` | 38 | `tests/test_spatio_temporal.py` | self-paired |
| Within-family LMM | `family-scan` | `torchgwas.models.within_family_lmm` | 23 | `tests/test_within_family.py` | `validation/specialty/family/` (paper-OLS) |
| Threshold-linear (multi-trait) | `threshold-scan` | `torchgwas.models.threshold_linear` | 22 | `tests/test_threshold.py` | `validation/specialty/threshold/` (BLUPF90+) |
| Knockoff LMM (FDR) | `knockoff-scan` | `torchgwas.models.knockoff_lmm` | 27 | `tests/test_knockoff.py` | `validation/specialty/knockoff/` (R knockoff) |
| OCF LMM (DML) | `ocf-scan` | `torchgwas.models.ocf_lmm` | 26 | `tests/test_ocf_lmm.py` | `validation/specialty/ocf/` (hand-coded DML2) |
| Genotype-uncertainty LMM | `gu-scan` | `torchgwas.models.gu_lmm` | 28 | `tests/test_gu_lmm.py` | `validation/specialty/gu/` (internal-consistency) |
| Leave-region-out LMM | `lro-scan` | `torchgwas.models.lro_lmm` | 29 | `tests/test_lro_lmm.py` | `validation/specialty/lro/` (internal-consistency) |
| Conditional LMM (COJO) | `conditional-scan` | `torchgwas.models.conditional_lmm` | 24 | `tests/test_conditional.py` | self-paired |

### Cluster 6 — GLM / GLMM family (7 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| Binary GLM | `glm-scan --family binary` | `torchgwas.models.binary_glm` | 31 | `tests/test_binary_glm.py` | self-paired (statsmodels) |
| Ordinal GLM | `glm-scan --family ordinal` | `torchgwas.models.ordinal_glm` | 32 | `tests/test_ordinal_glm.py` | self-paired |
| Multinomial GLM | `glm-scan --family multinomial` | `torchgwas.models.multinomial_glm` | 33 | `tests/test_multinomial_glm.py` | self-paired |
| Binary GLMM (PQL) | `glmm-scan --family binary` | `torchgwas.models.binary_glmm` | 31 | `tests/test_binary_glmm.py` | `validation/external/saige/` (infra-blocker doc) |
| Ordinal GLMM (PQL) | `glmm-scan --family ordinal` | `torchgwas.models.ordinal_glmm` | 32 | `tests/test_ordinal_glmm.py` | self-paired |
| Multinomial GLMM (PQL) | `glmm-scan --family multinomial` | `torchgwas.models.multinomial_glmm` | 33 | `tests/test_multinomial_glmm.py` | self-paired |
| Multi-env GLMM | `me-glmm-scan` | `torchgwas.models.me_glmm` | 34 | `tests/test_me_glmm.py` | self-paired |

### Cluster 7 — Multi-env / multi-trait (5 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| MET (reaction-norm + FA(k)) | `met-scan` | `torchgwas.models.met_lmm` | 19 | `tests/test_met.py` | self-paired |
| MT-MET LMM (Kronecker) | `mtmet-scan` | `torchgwas.models.mtmet_lmm` | 25, 36 | `tests/test_mtmet.py` | self-paired |
| Multi-env haplotype GWAS | (composed; Phase 47) | `torchgwas.models.haplotype_multi_env_gwas` | 47 | `tests/test_haplotype_multi_env.py` | self-paired |
| Multi-trait haplotype GWAS | (composed; Phase 47) | `torchgwas.models.haplotype_multi_trait_gwas` | 47 | `tests/test_haplotype_multi_trait.py` | self-paired |
| MT-MET haplotype GWAS | (composed; Phase 47) | `torchgwas.models.haplotype_mtmet_gwas` | 47 | `tests/test_haplotype_mtmet.py` | self-paired |

### Cluster 8 — Post-GWAS (16 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| LDSC h2 | `ldsc` | `torchgwas.postgwas.ldsc_h2` | 37 | `tests/test_postgwas_ldsc.py` | `validation/external/ldsc/` |
| LDSC genetic correlation | `ldsc-rg` | `torchgwas.postgwas.ldsc_rg` | 37 | `tests/test_postgwas_ldsc.py` | `validation/external/ldsc/` |
| S-LDSC (partitioned) | (programmatic) | `torchgwas.postgwas.sldsc_h2_partitioned` | 42 | `tests/test_sldsc.py` | self-paired (Finucane 2015 anchor) |
| Meta-analysis (IVW/DL/Stouffer/RE2) | `meta` | `torchgwas.postgwas.meta_analysis` | 37 | `tests/test_meta.py` | self-paired |
| LD clumping | `clump` | `torchgwas.postgwas.ld_clump` | 37 | `tests/test_clump.py` | `validation/external/plink2/` |
| Fine-mapping | (programmatic) | `torchgwas.postgwas.fine_mapping` | 44 | `tests/test_fine_mapping.py` | `validation/external/susieR/` |
| MR (IVW/Egger/median/PRESSO) | (programmatic) | `torchgwas.postgwas.mr` | 44 | `tests/test_mr.py` | `validation/external/twosamplemr/` |
| HESS regional h2 | (programmatic) | `torchgwas.postgwas.hess` | 45 | `tests/test_hess.py` | self-paired |
| Multi-ancestry (MR-MEGA/MANTRA) | (programmatic) | `torchgwas.postgwas.multi_ancestry` | 44 | `tests/test_multi_ancestry.py` | self-paired |
| Power analysis | (programmatic) | `torchgwas.postgwas.power` | 45 | `tests/test_power.py` | self-paired |
| Winners-curse correction | (programmatic) | `torchgwas.postgwas.winners_curse` | 45 | `tests/test_winners_curse.py` | self-paired |
| PGS: C+T | `pgs-fit --method ct` | `torchgwas.pgs.ct` | 40 | `tests/test_pgs_ct.py` | self-paired |
| PGS: LDpred2-Inf/Grid/Auto | `pgs-fit --method ldpred2-*` | `torchgwas.pgs.ldpred2` | 40 | `tests/test_pgs_ldpred2.py` | self-paired |
| PGS: PRS-CS | `pgs-fit --method prs-cs` | `torchgwas.pgs.prs_cs` | 40 | `tests/test_pgs_prscs.py` | self-paired |
| PGS scoring + validation | `pgs-score` | `torchgwas.pgs.score_individuals` | 40 | `tests/test_pgs_score.py` | self-paired |
| NCBI gene annotation | `annotate` | `torchgwas.annotate.ncbi` | 48 | `tests/test_annotate.py` | self-paired (network) |

### Cluster 9 — Multiple testing (14 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| Bonferroni | `--correction bonferroni` | `torchgwas.stats.multiple_testing` | 11 | `tests/test_multiple_testing.py` | self-paired |
| Holm | `--correction holm` | `torchgwas.stats.multiple_testing` | 11 | `tests/test_multiple_testing.py` | self-paired |
| Sidak | `--correction sidak` | `torchgwas.stats.multiple_testing` | 11 | `tests/test_multiple_testing.py` | self-paired |
| BH | `--correction bh` | `torchgwas.stats.multiple_testing` | 11 | `tests/test_multiple_testing.py` | self-paired |
| BY | `--correction by` | `torchgwas.stats.multiple_testing` | 11 | `tests/test_multiple_testing.py` | self-paired |
| Storey q-value | `--correction qvalue` | `torchgwas.stats.qvalue` | 11 | `tests/test_qvalue.py` | self-paired |
| simpleM / M_eff | (programmatic) | `torchgwas.stats.simpleM` | 11 | `tests/test_simpleM.py` | self-paired |
| eigenMT | (programmatic) | `torchgwas.stats.eigenmt` | 11 | `tests/test_stats_eigenmt.py` | self-paired |
| IHW | `--correction ihw` | `torchgwas.stats.ihw` | 30 | `tests/test_ihw.py` | self-paired |
| AdaPT | `--correction adapt` | `torchgwas.stats.adapt` | 30 | `tests/test_adapt.py` | self-paired |
| Weighted FDR | (programmatic) | `torchgwas.stats.weighted_fdr` | 30 | `tests/test_weighted_fdr.py` | self-paired |
| Cauchy combination | (programmatic) | `torchgwas.stats.cauchy_combination` | 30 | `tests/test_cauchy.py` | self-paired |
| Local FDR | (programmatic) | `torchgwas.stats.local_fdr` | 30 | `tests/test_local_fdr.py` | self-paired |
| Permutation max-T | (programmatic) | `torchgwas.stats.permutation` | 11 | `tests/test_permutation.py` | self-paired |

### Cluster 10 — Visualization (6 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| Manhattan (linear) | (viz API; figure-renderer driver) | `torchgwas.viz.manhattan` | 43 | `tests/test_viz_manhattan.py` | self-paired |
| Circos Manhattan | (viz API) | `torchgwas.viz.circos` | 43 | `tests/test_viz_circos.py` | self-paired |
| QQ plot | (viz API) | `torchgwas.viz.qq` | 43 | `tests/test_viz_qq.py` | self-paired |
| Miami plot | (viz API) | `torchgwas.viz.miami` | 43 | `tests/test_viz_miami.py` | self-paired |
| Haploview LD triangle | (viz API) | `torchgwas.viz.haploview` | 43 | `tests/test_viz_haploview.py` | self-paired |
| Trumpet plot | (viz API) | `torchgwas.viz.trumpet` | 45 | `tests/test_viz_trumpet.py` | self-paired |

### Cluster 11 — I/O + preprocessing (16 capabilities)

| Capability | CLI subcommand | Module path | Phase | Test file | Validation harness |
|---|---|---|---|---|---|
| PLINK BED reader | (auto-detect) | `torchgwas.io.plink_bed` | 1 | `tests/test_io_plink.py` | `validation/external/plink2/` |
| PLINK2 PGEN reader | (auto-detect) | `torchgwas.io.plink_pgen` | 1 | `tests/test_io_pgen.py` | `validation/external/plink2/` |
| VCF/BCF reader | (auto-detect) | `torchgwas.io.vcf` | 1 | `tests/test_io_vcf.py` | self-paired |
| BGEN reader | (auto-detect) | `torchgwas.io.bgen` | 1 | `tests/test_io_bgen.py` | self-paired |
| HapMap reader | (auto-detect) | `torchgwas.io.hapmap` | 1 | `tests/test_io_hapmap.py` | self-paired |
| Zarr/HDF5 reader | (auto-detect) | `torchgwas.io.zarr_h5` | 1 | `tests/test_io_zarr.py` | self-paired (zarr 3.1.5) |
| Numeric/CSV dosage reader | (auto-detect) | `torchgwas.io.csv` | 1 | `tests/test_io_csv.py` | self-paired |
| Format auto-detect + convert | `convert` | `torchgwas.io.convert` | 1 | `tests/test_io_convert.py` | self-paired (round-trip) |
| Imputation: mean/KNN/LD | `impute --method mean|knn|ld` | `torchgwas.preprocess.impute` | 2 | `tests/test_impute.py` | self-paired |
| Imputation: GPU Li-Stephens | `impute --method li-stephens` | `torchgwas.preprocess.li_stephens_hmm` | 18 | `tests/test_li_stephens.py` | self-paired |
| Imputation: deep autoencoder | `impute --method deep-learning` | `torchgwas.preprocess.deep_impute` | 18 | `tests/test_deep_impute.py` | self-paired |
| Imputation: BEAGLE/IMPUTE5/Minimac4 | `impute --method beagle|impute5|minimac4` | `torchgwas.preprocess.external_imputers` | 2 | `tests/test_external_imputers.py` | self-paired (subprocess) |
| Phasing | (programmatic) | `torchgwas.preprocess.phasing` | 2 | `tests/test_phasing.py` | self-paired |
| QC (MAF/HWE/call-rate) | (programmatic; pre-scan) | `torchgwas.preprocess.qc` | 2 | `tests/test_qc.py` | self-paired (parquet output gates) |
| Standardization (ploidy-aware) | (programmatic; pre-scan) | `torchgwas.preprocess.standardize` | 2 | `tests/test_standardize.py` | self-paired |
| Dosage uncertainty (R^2) | (programmatic) | `torchgwas.preprocess.dosage_uncertainty` | 28 | `tests/test_dosage_uncertainty.py` | `validation/specialty/gu/` |

## Totals

| Cluster | n_capabilities |
|---|---|
| Variant-level GWAS | 10 |
| Haplotype layer | 22 |
| Multi-omics integration | 8 |
| Polyploid pipeline | 6 |
| Specialty models | 11 |
| GLM / GLMM family | 7 |
| Multi-env / multi-trait | 5 |
| Post-GWAS | 16 |
| Multiple testing | 14 |
| Visualization | 6 |
| I/O + preprocessing | 16 |
| **TOTAL** | **121** |

Matches the F1 renderer assertion in `paper/reproducibility/render_figures/f1_capability_map.py` (`n_capabilities = 121`, `n_clusters = 11`). Source-of-truth: `paper/reproducibility/manifest.json -> F1.numbers.cluster_summary`.

## CLI dispatch backstop

The CLI handler dict at `torchgwas/cli.py:103-145` exposes 41 subcommands. The Phase Index in `CLAUDE.md` (line `CLI subcommand count: 38 as of Phase 56`) is the conservative documented count; the dispatch dict adds three NA1 / NA3 / convert-impute variants (`bayes-scan-rss`, `convert`, `impute`) for a total of 41. Each subcommand maps to one or more rows above; clusters 5 / 6 / 7 share `--family` and `--vg-structure` flags, so capability rows may share a subcommand.
