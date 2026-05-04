# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TorchGWAS** is a modular Python library for GPU-accelerated Genome-Wide Association Studies (GWAS) using PyTorch. It replicates and extends functionality from established tools like GEMMA and GAPIT, targeting 4th-decimal-place agreement with their p-values. Supports both diploid and polyploid organisms.

**Status**: Active development. Version 0.1.1 (Alpha). 2192 tests pass, 45 skipped. V1 core (Phases 0–13) complete with GEMMA/GAPIT reference equivalence; post-V1 extensions implemented through Phase 56. See the **Phase Index** below for scope, and `git log --grep="Phase NN"` for per-phase details (every phase was shipped as a labeled commit).

## Planned Architecture

Data flow: **format detection → imputation/phasing → QC/preprocessing → genotype encoding → GRM → eigendecomposition → null fitting → scan loop → multiple testing → reporting**.

### Core Modules

- **`torchgwas.io`** — Auto-detect and read all major formats (PLINK BED/PGEN, VCF/BCF, BGEN, HapMap, Zarr, CSV dosage). Format validation and conversion utilities.
- **`torchgwas.preprocess`** — Imputation (built-in mean/KNN/LD-based + GPU Li-and-Stephens HMM + deep learning autoencoder + external BEAGLE/IMPUTE5/Minimac4 wrappers), phasing, QC (MAF/HWE/call rate), standardization (ploidy-aware), polyploid gene-action model encoding.
- **`torchgwas.linalg`** — Kinship/GRM computation (diploid + polyploid, LOCO), EED, batched Cholesky.
- **`torchgwas.models`** — `BaseModel` protocol with `fit_null()` and `score_chunk()`:
  - Core: `GLM`, `SingleTraitLMM`, `MultiTraitLMM`, `FarmCPU`, `BLINK`
  - Mixed-model extensions: `MultiKernelLMM`, `GxELMM` / `HetLMM`, `SetBasedScanner`, `BayesianVS` (SuSiE + CAVI), `ThresholdLinearModel` (Bermann et al. 2026), `WithinFamilyLMM` (Young et al. 2022), `ConditionalLMM`, `MultiTraitMultiEnvLMM`, `OCFLMM` (DML cross-fit), `KnockoffLMM` (Sesia et al. 2020), `GULM` (dosage-variance score test), `LROLMM` (block-level LOCO)
  - GLM / GLMM: `BinaryGLM`, `OrdinalGLM`, `MultinomialGLM`; `BinaryGLMM`, `OrdinalGLMM`, `MultinomialGLMM` (PQL null, SAIGE-style)
  - Specialty: `SurvivalGLMM` (Cox PH frailty), `RandomRegressionLMM` + `SpatioTemporalRR` (Phase 38), `RandomRegressionMultiEnvLMM` (Phase 39), `HaplotypeGWAS` (Phase 46), `HaplotypeMultiEnvGWAS` / `HaplotypeMultiTraitGWAS` / `HaplotypeMTMETGWAS` (Phase 47)
- **`torchgwas.scan`** — `UnifiedScanner` streams chunks through any model via adapters.
- **`torchgwas.stats`** — Multiple testing: Bonferroni, Holm, BH, BY, Storey q-value, simpleM/M_eff, GPU-accelerated permutation, weighted FDR, Cauchy combination, local FDR, IHW, AdaPT, eigenMT.
- **`torchgwas.ld`** — LD block detection: 13 methods (4 classical, 5 novel, 3 literature, 1 diagnostic). Pairwise LD (r², D', CI), graph utilities (Laplacian, spectral partition, MWIS), change-point detection (PELT), uncertainty-corrected LD, PLINK .blocks.det compatibility. Validated against PLINK 1.9 `--blocks`. Diploid and arbitrary polyploid.
- **`torchgwas.optim`** — 6-mode optimizer stack with automatic fallback: PX-EM → AI-REML → LBFGS-autograd → MM algorithm → PCG iterative → derivative-free rescue.
- **`torchgwas.pgs`** — Polygenic score construction: C+T, LDpred2 (Inf / Grid / Auto), PRS-CS; allele harmonization; `score_individuals`; MCMC diagnostics; validation metrics.
- **`torchgwas.postgwas`** — LDSC h²/rg + S-LDSC, meta-analysis (IVW/DL/Stouffer/RE2), LD clumping, coloc (two-trait + hyprcoloc), MR (IVW/Egger/weighted median/MR-PRESSO), fine-mapping utilities, gene-set enrichment, multi-ancestry (MR-MEGA / MANTRA), power, winner's curse, SMR/HEIDI, TWAS, HESS.
- **`torchgwas.multiomics`** — GRM-corrected causal mediation (`mediate_lmm`, `scan_mediation`, `mediate_gene_set`), multi-kernel heritability (`mkernel_h2`), eigenMT FDR, coloc prefilter. GPU-batched scan.
- **`torchgwas.viz`** — Manhattan (linear + Circos), QQ, Miami, Haploview LD triangle, trumpet plot. Matplotlib-only.
- **`torchgwas.annotate`** — NCBI gene annotation (Datasets v2 + E-utilities); crop name or assembly accession → genes in ±window with GO terms and optional orthologs. Network-required.
- **`torchgwas._native`** — 24 pybind11 C++ extensions (optional build) + device-aware dispatcher `_dispatch.select_path`. Dedicated torch-GPU kernels for imputation. OpenMP parallelization where it pays off. See `csrc/`, `tests/test_native_*.py`, `bench/native_speedups.md`.

## Key Design Decisions

- **Hardware-agnostic tensors**: All operations use `torch.Tensor` with `.to(device)` for CPU/GPU portability.
- **Polyploid-first**: Genotypes stored as float tensors in [0, k] for arbitrary ploidy k. Gene-action models (additive, 1-dom through k-1-dom, diplo-additive, overdominant) tested per SNP.
- **Format intelligence**: Auto-detect, validate, and convert between all major genotype formats. Pre-flight checks before any GWAS run. Deterministic sample alignment with three-way ID intersection and alignment manifest output.
- **Variant QC Parquet output**: Per-variant MAF, heterozygosity, missingness, HWE p-value, and filter status written to Parquet before scan — enables QA against GEMMA/GAPIT defaults for both diploid and polyploid.
- **Integrated imputation**: Built-in simple imputation + pluggable external tools (BEAGLE, IMPUTE5, Minimac4, STITCH) with quality tracking.
- **Layered optimizer stack**: Adapted from ASReml-class charter + PyTorch autograd extensions.
- **Precision management**: AMP for FP16/BF16 in I/O and GRM; FP64 mandatory for statistical inference.

## Repo Conventions

These conventions are invariants across the codebase — when editing a hot loop or adding a new accelerator, preserve them:

- **Python-as-spec, native-as-shortcut**: Every function that has a C++ accelerator keeps its pure-Python / torch body intact as the algorithmic reference. The native shortcut is wired *at the top* of the function; if the extension is missing or `TORCHGWAS_DISABLE_NATIVE=1`, execution falls through to the Python body unchanged. This applies to all 26 native extensions.
- **GPU kernels follow the same dispatch discipline** under `TORCHGWAS_DISABLE_GPU=1`. Device-aware routing is centralized in `torchgwas._dispatch.select_path` — **never proactively move a CUDA tensor to host** for a C++ path; if no dedicated GPU kernel exists, the torch body runs on-device instead.
- **Optional C++ build**: The pybind11 extensions in `csrc/` are `optional=True` in `setup.py`, so `pip install` succeeds without a C++ toolchain. OpenMP is opt-in per platform and can be disabled with `TORCHGWAS_DISABLE_OPENMP=1`.
- **Every phase ships as a commit with "Phase N" in the message.** To recover per-phase detail, read `git log --grep="Phase N"` (the commit body is the authoritative changelog) and the corresponding `tests/test_*.py` + module docstrings.
- **Novelty claims** are qualified with "to our knowledge."
- **Validation gates** use numerical tolerance, not bitwise identity (GPU non-determinism).
- **CLI subcommand count**: 37 as of Phase 56 (adds `phase-poly`; see CLI Commands section, plus `rr-scan`, `rr-met-scan`, `pgs-fit`, `pgs-score`, `annotate`, `mediate`, `mediate-scan` which are wired but may not appear in the examples below).

## Phase Index

**V1 core** (GEMMA/GAPIT reference equivalence):
- **Phases 0–13** — GLM, SingleTraitLMM, MultiTraitLMM, FarmCPU, BLINK, kinship/EED, layered optimizer stack

**Post-V1 novel models**:
- **Phases 14–19** — Multi-kernel LMM, GxE-LMM, set-based tests, BayesianVS (SuSiE + CAVI), GPU imputation, multi-env GWAS (MET; reaction-norm + FA(k) + multi-kernel)
- **Phases 20–21** — LD block detection module (13 methods; PLINK-validated)
- **Phase 22** — Multi-trait threshold-linear GWAS (Bermann et al. 2026); T-EM / T-NR / SQUAREM / ssGWAS
- **Phase 23** — Within-family LMM (Young et al. 2022); dual-scan attenuation diagnostics
- **Phase 24** — LD-conditional GWAS (GCTA-COJO-style stepwise; persistence metrics)
- **Phase 25** — Multi-trait multi-environment LMM (separable Kronecker Vg = Vg_trait ⊗ Vg_env)
- **Phase 26** — Orthogonal Cross-Fit LMM (DML-valid; Chernozhukov et al. 2018)
- **Phase 27** — Knockoff Mixed Model (Sesia et al. 2020; group knockoffs + knockoff+ filter)
- **Phase 28** — Genotype-Uncertainty LMM (dosage-variance–corrected score test)
- **Phase 29** — Leave-Region-Out LMM (block-level LOCO)
- **Phase 30** — Adaptive FDR (IHW + AdaPT)
- **Phases 31–33** — GLM family (binary / ordinal / multinomial) + GLMM via PQL (SAIGE-style)
- **Phase 34** — Multi-Environment GLMM (binary / ordinal with structured Σ_g)
- **Phase 35** — Survival GWAS (Cox PH frailty; Breslow-Clayton PQL + SPACox SPA)
- **Phase 36** — MT-MET scaling (Kronecker EED diagonal precision; d=50 / E=20)
- **Phase 37** — Post-GWAS foundation: LDSC h²/rg, meta-analysis (IVW/DL/Stouffer/RE2), LD clumping
- **Phase 38** — Random Regression LMM (Legendre / B-spline); SpatioTemporalRR; `rr-scan` CLI
- **Phase 39** — Random Regression × Multi-Environment LMM; 9 RR-aware contrast tests; `rr-met-scan` CLI
- **Phase 40** — Polygenic Score construction (C+T, LDpred2-Inf/Grid/Auto, PRS-CS); `pgs-fit` / `pgs-score` CLI
- **Phases 41a–41af** — Native C++ accelerators (24 pybind11 extensions, OpenMP, dedicated torch-GPU kernels, device-aware dispatcher, realistic-size benchmark suite). Speedups range from 1.5× to >12000×. Sources in `csrc/`, tests in `tests/test_native_*.py`, numbers in `bench/native_speedups.md`. Full per-sub-phase detail: `git log --grep="Phase 41"`.
- **Phase 42** — S-LDSC (partitioned; Finucane 2015), hyprcoloc (Foley 2021), classical two-trait coloc (Giambartolomei 2014)
- **Phase 43** — GWAS visualization (`torchgwas.viz`: Manhattan / Circos / Miami / QQ / Haploview); per-marker PVE
- **Phase 44** — Mendelian Randomization (IVW / Egger / weighted median / MR-PRESSO); gene-set enrichment (MAGMA-style); fine-mapping utilities; PGS validation; multi-ancestry meta-analysis (MR-MEGA / MANTRA)
- **Phase 45** — Power analysis; winner's curse correction; trumpet plot; SMR/HEIDI; TWAS (S-PrediXcan + PrediXcan); HESS regional heritability
- **Phase 46** — Haplotype GWAS (HTR / block / window / SKAT) + 5 novel methods (PCHT / HHCT / HSKAT / HapGxE / BayesHap)
- **Phase 47** — Multi-env / multi-trait / MT-MET haplotype GWAS (thin composition over Phase 25 / 39 / 46)
- **Phase 48** — NCBI gene annotation (`torchgwas.annotate`); `annotate` CLI; `requests` runtime dep added
- **Phases 49 + 49b** — GRM-corrected causal mediation (`torchgwas.multiomics`); multi-kernel h²; GPU-batched scan; gene-set Wald; eigenMT FDR; coloc prefilter; `mediate` / `mediate-scan` CLI
- **Phase 55** — Polyploid allele dosage calling (`torchgwas.preprocess.dosage_call`); updog wrapper; `dosage-call` CLI; 28 Tier 1 + 4 Tier 2 tests
- **Phase 56** — Polyploid F1 phasing (`torchgwas.preprocess.polyploid_phase`); PolyOrigin Julia wrapper; pedigree/map/probs pipeline; `phase-poly` CLI; 43 Tier 1 + 4 Tier 2 scaffolds

## Canonical Reference

The full project charter is at `TorchGWAS_AI_Agent_Handoff_Charter.md` (Version 4.0, 23 sections). Key sections:
- Preamble c/d/e: three-tier scope split — V1 Core (Gaussian quantitative-trait GWAS) / Platform Commitments / Post-V1 Extensions (incl. platform-scoped categorical-trait support)
- Section 4: GEMMA algorithmic reference — exact mathematics to replicate
- Sections 6–9: data formats, imputation/phasing, multiple testing, polyploid GWAS
- Section 5b: multi-trait threshold-linear models for categorical traits (Bermann et al. 2026) — post-V1
- Section 13: 6-mode optimizer stack with automatic fallback hierarchy
- Section 14: 23-phase gated roadmap
- Section 17: novel scientific models
- Section 18: LD block detection — 13 methods across 4 categories

**V1 scope**: V1 statistical deliverable focuses on Gaussian quantitative-trait GWAS for reference-tool equivalence. Binary and ordinal traits are in scope for the platform, scientifically anchored in Bermann et al. (2026) threshold-linear models, but implementation is deferred to Phase 16. Multimodal data integration (omics, environmental, image) is a distinct later extension. Novelty claims are qualified with "to our knowledge." Validation gates use numerical tolerance (not bitwise identity) to account for GPU non-determinism.

## Target Dependencies

torch (>=2.0), numpy, pandas, scipy, matplotlib, requests (>=2.28)

Optional: zarr, h5py, pyarrow, seaborn

## Validation campaign

A function-by-function validation campaign runs out of `docs/superpowers/` against `validation/pillar-{A,B,C,D}-coverage` branches. Pillar A (coverage audit + tiered fill) is complete on `validation/pillar-A-coverage`.

- **Spec:** `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`
- **Per-pillar plans:** `docs/superpowers/plans/2026-04-30-pillar-{A,B,C,D}-plan.md`
- **Findings ledger:** `docs/validation_findings.md` — every divergence with F3 classification + resolution.
- **Reviewer prompt templates:** `docs/superpowers/reviewer_prompts/{code_review,rerun_verification}.md`

Re-run a coverage audit (which symbols have direct tests):

```bash
python3 scripts/audit_public_coverage.py
# Output: docs/validation_findings/coverage_audit.json with per-symbol rows + summary.
```

Generate the per-package worklist for a given tier (consumed by Pillar B/C/D fill subagents):

```bash
python3 scripts/build_tier_worklist.py --tier 1 --package linalg --output /tmp/worklist.json
```

When Pillar B is wired, run external-reference comparisons:

```bash
pytest -m external
```

Pillar A delivered 5 V1-core fix-now production fixes surfaced from validation work (3 unimplemented model stubs, 1 unimplemented stats helper, 1 unimplemented optim solver) plus 3 V1-platform fix-now fixes (zarr v3 API compat, 2 silent shape-truncation guards in postgwas LD scoring + clumping). All 8 are recorded in the findings ledger.

## CLI Commands

```bash
pip install -e ".[dev]"                    # Install in dev mode
pytest tests/ -v                           # Run all tests

# --- Data management ---
torchgwas validate --genotype data.bed --phenotype pheno.txt
torchgwas convert --input data.vcf.gz --output data --format bed
torchgwas impute --genotype data.bed --method mean --output imp.pt
torchgwas impute --genotype data.vcf.gz --method beagle --ref-panel 1000G.vcf.gz --output imp.vcf.gz
torchgwas impute --genotype data.vcf.gz --method li-stephens --ploidy 4 --output imp.pt
torchgwas impute --genotype data.vcf.gz --method deep-learning --output imp.pt

# --- Polyploid allele dosage calling (Phase 55) ---
torchgwas dosage-call --vcf calls.vcf.gz --output out/dcall --ploidy 4 --model norm

# --- Polyploid F1 phasing (Phase 56) ---
torchgwas phase-poly --probs out/dcall.probs.pt --pedigree ped.tsv \
    --map markers.tsv --output out/phased --ploidy 4

# --- GWAS scans ---
torchgwas glm-scan --genotype data.bed --phenotype pheno.txt --output results
torchgwas lmm-scan --genotype data.bed --phenotype pheno.txt --test wald --correction bh
torchgwas mvlmm-scan --genotype data.bed --phenotype pheno.txt --traits Y1,Y2,Y3
torchgwas poly-scan --genotype data.csv --phenotype pheno.txt --ploidy 4 --gene-action all
torchgwas mklmm-scan --genotype data.bed --phenotype pheno.txt --kernels additive,dominance
torchgwas gxe-scan --genotype data.bed --phenotype pheno.txt --env env.tsv --gxe-model het
torchgwas set-scan --genotype data.bed --phenotype pheno.txt --regions genes.bed --set-test skat
torchgwas bayes-scan --genotype data.bed --phenotype pheno.txt --method susie --n-signals 10
torchgwas farmcpu-scan --genotype data.bed --phenotype pheno.txt --max-qtns 20
torchgwas blink-scan --genotype data.bed --phenotype pheno.txt --cutoff 0.01

# --- Multi-environment GWAS ---
torchgwas met-scan --genotype data.bed --phenotype pheno_env.txt --parameterization reaction_norm
torchgwas met-scan --genotype data.bed --phenotype pheno_env.txt --vg-structure "fa(2)"
torchgwas met-scan --genotype data.bed --phenotype pheno_env.txt --kernel-files dominance.npy

# --- Threshold-linear GWAS ---
torchgwas threshold-scan --genotype data.bed --phenotype pheno.txt --trait-types ordinal,continuous --n-categories 3,0
torchgwas threshold-scan --genotype data.bed --phenotype pheno.txt --trait-types ordinal,ordinal,continuous --n-categories 2,3,0 --solver em --R-matrix R.csv --G-matrix G.csv

# --- Within-family GWAS ---
torchgwas family-scan --genotype data.bed --phenotype pheno.txt --family-col FID
torchgwas family-scan --genotype data.bed --phenotype pheno.txt --family-col FID --min-family-size 3 --confound-threshold 0.3

# --- LD-conditional GWAS ---
torchgwas conditional-scan --genotype data.bed --phenotype pheno.txt --ld-method r2
torchgwas conditional-scan --genotype data.bed --phenotype pheno.txt --ld-method gabriel --max-conditioning 3 --sig-threshold 5e-8

# --- Multi-trait multi-environment GWAS ---
torchgwas mtmet-scan --genotype data.bed --phenotype pheno.txt --traits yield,protein --env-cols E1,E2,E3
torchgwas mtmet-scan --genotype data.bed --phenotype pheno.txt --traits Y1,Y2 --env-cols E1,E2 --vg-structure unstructured
torchgwas mtmet-scan --genotype data.bed --phenotype pheno.txt --traits Y1,Y2,Y3 --env-cols E1,E2,E3 --vg-structure "fa(2)"

# --- Orthogonal Cross-Fit LMM (debiased inference) ---
torchgwas ocf-scan --genotype data.bed --phenotype pheno.txt --n-folds 5
torchgwas ocf-scan --genotype data.bed --phenotype pheno.txt --n-folds 10 --variance-type HC --seed 42

# --- Knockoff FDR-controlled GWAS ---
torchgwas knockoff-scan --genotype data.bed --phenotype pheno.txt --fdr-level 0.05
torchgwas knockoff-scan --genotype data.bed --phenotype pheno.txt --fdr-level 0.1 --ld-method r2 --aggregation sum_sq

# --- Genotype-Uncertainty LMM (GP-aware score test) ---
torchgwas gu-scan --genotype data.bed --phenotype pheno.txt --dosage-var dosage_var.pt
torchgwas gu-scan --genotype data.bed --phenotype pheno.txt --dosage-var dosage_var.npy --test score

# --- Leave-Region-Out LMM (block-level LOCO) ---
torchgwas lro-scan --genotype data.bed --phenotype pheno.txt --ld-method r2
torchgwas lro-scan --genotype data.bed --phenotype pheno.txt --ld-method gabriel --test score

# --- GLM family (binary/ordinal/multinomial, no random effects) ---
torchgwas glm-scan --genotype data.bed --phenotype pheno.txt --family binary --firth
torchgwas glm-scan --genotype data.bed --phenotype pheno.txt --family ordinal --n-categories 3
torchgwas glm-scan --genotype data.bed --phenotype pheno.txt --family multinomial --n-categories 4

# --- GLMM (binary/ordinal/multinomial with random effects) ---
torchgwas glmm-scan --genotype data.bed --phenotype pheno.txt --family binary
torchgwas glmm-scan --genotype data.bed --phenotype pheno.txt --family ordinal --n-categories 3
torchgwas glmm-scan --genotype data.bed --phenotype pheno.txt --family multinomial --n-categories 4

# --- Survival GWAS (Cox PH frailty model) ---
torchgwas survival-scan --genotype data.bed --phenotype pheno_surv.txt
torchgwas survival-scan --genotype data.bed --phenotype pheno_surv.txt --no-spa --pql-max-iter 50

# --- Adaptive FDR (IHW / AdaPT via --correction) ---
torchgwas lmm-scan --genotype data.bed --phenotype pheno.txt --correction ihw
torchgwas lmm-scan --genotype data.bed --phenotype pheno.txt --correction adapt --fdr-covariate covariates.tsv

# --- LD block detection ---
torchgwas ld-blocks --genotype data.bed --method gabriel --output blocks
torchgwas ld-blocks --genotype data.bed --method big_ld --r2-threshold 0.5
torchgwas ld-blocks --genotype data.bed --method cc_graph --ld-window 100
torchgwas ld-blocks --genotype data.bed --method changepoint --cp-penalty 0
torchgwas ld-blocks --genotype data.bed --method graphical --l1-penalty 0.1
torchgwas ld-blocks --genotype data.bed --method gabriel --compare-plink plink.blocks.det

# --- Random regression (longitudinal) ---
torchgwas rr-scan --genotype data.bed --phenotype pheno_long.txt --basis legendre --order 2
torchgwas rr-met-scan --genotype data.bed --phenotype pheno_long.txt --env-col ENV --vg-structure separable

# --- Polygenic scores (Phase 40) ---
torchgwas pgs-fit --sumstats ss.tsv --ld-ref ld.pt --method ldpred2-auto --output weights.tsv
torchgwas pgs-score --weights weights.tsv --genotype target.bed --output scores.tsv

# --- Causal mediation (Phase 49) ---
torchgwas mediate --y pheno.npy --snp snp.npy --mediator mediator.npy --kinship K.npy
torchgwas mediate-scan --y pheno.npy --genotype data.bed --mediator-matrix mediators.npy --kinship K.npy --fdr bh

# --- NCBI gene annotation (Phase 48) ---
torchgwas annotate --sumstats hits.tsv --crop maize --p-threshold 5e-8 --window-up 50000 --window-down 50000

# --- Full pipeline ---
torchgwas pipeline --genotype data.vcf.gz --impute beagle --model lmm --test wald
torchgwas pipeline --genotype data.bed --phenotype pheno.txt --model met --env-cols E1,E2,E3
```
