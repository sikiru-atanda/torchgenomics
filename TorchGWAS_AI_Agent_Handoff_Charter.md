# TorchGWAS — AI Agent Handoff Charter and Stepwise Build Blueprint

Prepared for a project that aims to build, to our knowledge, the first PyTorch-native GWAS library
capable of single-trait, multi-trait, and multi-locus association analysis with statistical
gold-standard accuracy and order-of-magnitude speed improvements over existing tools.

**Version 5.0 — Updated 2026-04-09 to cover Phases 0-47 (2102 tests pass, 44 skipped). Release v0.1.0 (Alpha)

---

| Field | Value |
|---|---|
| **Primary target** | Reproduce GEMMA/GAPIT statistical outputs (p-values to 4th decimal) while delivering 10-50x wall-clock speedups via GPU acceleration, batched tensor algebra, and mixed-precision pipelines. |
| **Preferred stack** | Python front-end + PyTorch computational core. All tensor operations use `torch.Tensor` with `.to(device)` for seamless CPU/GPU/multi-GPU portability. |
| **Execution rule** | Build in gated phases: exact correctness first (match reference tools), GPU acceleration second, approximate large-scale methods third. |
| **Golden rule** | Correctness must be demonstrated against trusted reference software (GEMMA, GAPIT, PLINK) before any performance optimization or scope widening is attempted. |

---

## a. Core Mission and Design Philosophy

TorchGWAS must be a statistically rigorous, computationally efficient GWAS library for plant and animal
breeding, human genetics, and biobank-scale association studies. It must deliver GEMMA/GAPIT-class
statistical output while exploiting modern GPU hardware through PyTorch's ecosystem: automatic
differentiation, batched linear algebra, mixed precision, and out-of-core streaming.

The architecture must separate data I/O from linear algebra kernels from model fitting from scan loops
from statistical reporting, so that each layer can be independently optimized, tested, and extended.

## b. Competitive Landscape and Novelty Justification

### Current tool limitations that TorchGWAS addresses

| Tool | Key limitation TorchGWAS overcomes |
|---|---|
| **GEMMA** | CPU-only; O(n^3) eigendecomposition bottleneck; no streaming I/O; single-threaded scan loop; last major release v0.98.5 (August 2021), limited ongoing development. |
| **GAPIT3** (R) | R interpreter overhead; cubic complexity in n for most methods except FarmCPU/BLINK; no GPU path; serial SNP scanning. |
| **BOLT-LMM** | CPU-only; miscalibrated for unbalanced binary traits; requires ~50 GB RAM for biobank data; no multi-trait support. |
| **SAIGE / SAIGE-GPU** | GPU-accelerated model fitting (5x speedup) but single-trait generalized LMM only; no multi-trait, no multi-locus (FarmCPU/BLINK); saddle-point approximation adds complexity. |
| **REGENIE** | Highly scalable CPU tool (151x faster than BOLT-LMM); but no GPU path, no GEMMA-style multivariate Kronecker mvLMM, no multi-locus methods. |
| **fastGWA** | Large-scale approximate LMM via sparse GRM; CPU-only; limited model flexibility. |
| **rMVP** | Parallelized R tool; limited GPU support; no autograd-based optimization. |
| **TensorQTL** | Closest precedent — PyTorch GPU-accelerated eQTL mapping (Taylor-Weiner et al. 2019). But GLM only (no mixed models, no GRM, no LMM, no multi-locus). Demonstrates viability of PyTorch for genetic association. |
| **BLUPF90 suite** | Only existing implementation of multi-trait threshold-linear MAP models (Bermann et al. 2026); Fortran 90/95, CPU-only, no GPU path; primarily designed for breeding evaluations, not GWAS scanning; closed-source (blup90iod3 available only under research agreement). |
| **GWASpoly** | R-only polyploid GWAS with P3D mixed model, gene-action models, M.eff threshold, QTL detection; but no GPU, no multi-trait, no categorical-trait support, no non-additive GRM methods, no FarmCPU/BLINK for polyploids. |
| **AGHmatrix** | R package for polyploid relationship matrices (VanRaden, Slater, Endelman digenic, Vitezica/Su dominance, Yang, pedigree A/H-matrix); GRM-only, no GWAS scanning. |
| **GAPIT3** | R GWAS with GLM/MLM/FarmCPU/BLINK; accepts polyploid dosage as numeric input but has NO ploidy-aware GRM normalization, gene-action models, or polyploid-specific QC. |

### TorchGWAS novelty claims

1. **First PyTorch-native GWAS library** (to our knowledge): Autograd-enabled REML optimization eliminates hand-derived gradient code. New variance-component models can be prototyped by writing only the log-likelihood — PyTorch computes exact gradients automatically. To our knowledge, no existing GWAS tool offers this.

2. **GPU-accelerated multi-trait mvLMM**: To our knowledge, no existing tool provides GPU-batched Kronecker-structured mvLMM scanning. TorchGWAS implements the Efficient Eigen-Decomposition (EED) trick with batched `(n, d, d)` Cholesky solves on GPU, reducing per-SNP complexity from O(n^3 d^3) to O(n d^3) with GPU-parallel execution across SNP chunks.

3. **GPU-accelerated multi-locus methods (FarmCPU/BLINK)**: These iterative methods have linear time complexity in n (the most scalable GWAS models), but to our knowledge, no existing implementation exploits GPU. TorchGWAS batches the FEM/REM iterations and BIC selection on GPU tensors.

4. **Mixed-precision GWAS pipeline**: Automatic Mixed Precision (AMP) uses FP32/FP16/BF16 for heavy compute steps (GRM construction, genotype chunk matmuls, streaming transforms) while keeping the statistical path (likelihood, GLS coefficients, SEs, test statistics, p-values) in FP64 for numerical fidelity. To our knowledge, no existing GWAS tool implements this split-precision strategy.

5. **Unified scan architecture**: One `UnifiedScanner` class streams genotype chunks through any model (GLM, single-trait LMM, mvLMM, FarmCPU, BLINK) via adapters, with consistent output schema. Users switch methods by changing one argument, not learning a new tool.

6. **Out-of-core streaming with compute/IO overlap**: Pinned-memory staging + async CUDA streams overlap genotype loading (PLINK memmap, Zarr, HDF5) with GPU computation. To our knowledge, no existing Python GWAS tool implements this pipeline.

7. **Interpretation-safety layers (routine novelty)**: Every TorchGWAS run produces a covariate audit (detecting heritable covariates / colliders; Aschard et al. 2015 AJHG), LD-block attribution, and per-SNP causal-safety diagnostics (LD proxy risk, confounding sensitivity, within-family attenuation when available). To our knowledge, no existing GWAS tool ships these diagnostics as default output.

8. **Orthogonal cross-fit LMM (OCF-LMM)**: DML-valid inference (Chernozhukov et al. 2018) applied to the mixed-model GWAS setting — cross-fitting + Neyman-orthogonal score yields calibrated chi-squared(1) tests even when nuisance models (polygenic background) are complex. To our knowledge, no existing GWAS tool formalizes the debiasing that predict-then-test pipelines (REGENIE) implicitly approximate.

9. **Group-knockoff FDR-controlled GWAS (KO-LMM)**: LD-block-level group knockoffs (Sesia et al. 2020 JASA) integrated with the LMM framework for explicit FDR control at the locus level. To our knowledge, no existing tool combines knockoff guarantees with mixed-model random effects.

10. **GPU-accelerated Random Regression LMM for longitudinal + spatio-temporal GWAS (Phase 38)**: To our knowledge, the first PyTorch GWAS tool to handle repeated-measures longitudinal phenotypes (growth curves, yield-over-seasons, repeated biomarkers) via Legendre / B-spline basis projection reduced to a multi-trait LMM, with five-test contrast catalog (joint / intercept / slope / time-varying / per-time-point β(t)) and a 2D tensor-product P-spline `SpatioTemporalRR` field-trial spatial smoother. Existing tools (BLUPF90+, ASReml-R) ship random regression for breeding evaluation but not as a GPU-batched GWAS scanner.

11. **Random Regression × Multi-Environment LMM (RR-MET, Phase 39)**: To our knowledge, the first GPU-batched longitudinal-GWAS tool that supports cross-environment genetic structure for repeated-measures phenotypes measured across multiple trials / years / locations. Implemented as thin composition over Phase 25 MT-MET with separable Kronecker `Vg = K_coef ⊗ Vg_env` (default), unstructured (`bE ≤ 12` guard) and `fa(k)` alternatives; 9 RR-aware contrast tests per SNP including the genuinely-new joint stable-vs-GxE (`I_b ⊗ D_E`) and intercept-only-GxE (`e_0^T ⊗ D_E`) tests; pooled-time correctness invariant guarantees cross-environment contrasts have consistent physical meaning. **Validated against BLUPF90+ multi-trait RR with cookbook init** (n=400, T=8, E=2, Legendre order=1): TorchGWAS K_coef rel=0.051 vs BLUPF90 0.852 (~17× better); Vg_env rel=0.101 vs 0.276 (~2.7× better). FA mode validated against BLUPF90+ XFA at n=300, T=8, E=4, fa_rank=2: full-Vg rel=0.241 vs BLUPF90 XFA 0.599 (~2.5× better).

12. **Unified PGS Construction Framework (Phase 40)**: To our knowledge, the first GWAS tool to integrate Tier 1 (C+T, LDpred2-Inf/Grid/Auto) and Tier 2 (PRS-CS) polygenic score methods within the same GPU-accelerated ecosystem, with allele harmonization (palindromic A-T/C-G handling), LD reference construction (Ledoit-Wolf shrinkage), MCMC diagnostics (R-hat, ESS), and individual-level scoring — all connected via CLI subcommands `pgs-fit` and `pgs-score`.

13. **26 Native C++ Accelerators with Python Fallback (Phase 41)**: A systematic pybind11 acceleration layer covering every identified hot loop — from PRS-CS Gibbs (96×) to Gabriel block detection (10717×) to PELT change-point (12466×). Each kernel preserves the Python reference below it as the algorithmic spec, is toggleable via `TORCHGWAS_DISABLE_NATIVE=1`, and includes OpenMP parallelization where applicable. A central `select_path()` dispatcher routes between GPU, native C++, and Python based on device, problem size, and build capability.

14. **Haplotype-Based GWAS with Multi-Dimensional Extensions (Phases 46-47)**: To our knowledge, the first GPU-accelerated tool combining haplotype-level association testing (HTR, block-scan, window-scan, haplotype-SKAT) with five novel methods (PCHT phase-uncertainty correction, HHCT hierarchical collapsing, HSKAT Hamming-kernel SKAT, HapGxE haplotype×environment interaction, BayesHap SuSiE fine-mapping) and multi-env/multi-trait/MT-MET extensions via thin composition over existing null-fitting infrastructure.

## c. Non-Negotiable Project Directives (V1 Core)

- Match GEMMA/GAPIT p-values to the 4th decimal place on canonical test datasets before claiming correctness.
- Use FP64 for all statistical inference paths (likelihood, coefficients, SEs, test statistics, p-values). Mixed precision is only for I/O, GRM construction, and streaming transforms.
- Implement in gated phases: exact CPU correctness first, then GPU acceleration, then approximate/large-scale methods.
- One unified scan API that routes to multiple model backends without changing user code.
- Do not widen V1 scope casually. Anything outside the approved scope must be marked as later-phase work.
- **V1 statistical deliverable focuses on Gaussian quantitative-trait GWAS** (GLM, LMM, mvLMM, FarmCPU, BLINK) for reference-tool equivalence. Platform scope, however, explicitly includes future support for mixed outcome types — continuous, binary, and ordinal phenotypes — through multi-trait threshold-linear models based on Bermann et al. (2026; see Section 5b). Binary and ordinal traits are **in scope for the platform, deferred for implementation, and scientifically anchored** in the threshold-linear manuscript. Broader multimodal data integration (e.g., combining genotype with omics, environmental, image, or other feature modalities) is a distinct later-phase extension and is not implied by the threshold-linear work. If a user passes a binary or ordinal phenotype to a V1 model, TorchGWAS must warn that results assume a linear model and may be miscalibrated for categorical data, and point the user to the planned threshold-linear model support (Phase 16).

## d. Platform Commitments (V1-adjacent, phased in after core models)

These capabilities are committed features of the TorchGWAS platform but are not part of the V1
statistical-model deliverable. They are built into the architecture from Phase 0 so they do not
require redesign, but their full implementation is phased after the core models are validated.

- **Multi-format I/O with validation**: Accept PLINK BED/PGEN, VCF/BCF, BGEN, HapMap, and CSV/TSV/TXT with mandatory marker metadata (SNP, Chr, Pos). Auto-detect format, validate structure, report pre-flight summary. (Phase 1 deliverable.)
- **Built-in simple imputation**: Mean, mode, and KNN imputation available without external tools. (Phase 2 deliverable.)
- **Imputation/phasing orchestration layer**: TorchGWAS provides an integrated orchestration layer for imputation and phasing, with native simple imputers and wrapper-based support for best-in-class external tools (BEAGLE, IMPUTE5, Minimac4, STITCH). TorchGWAS does not reimplement these algorithms; it manages format conversion, invocation, quality tracking, and pipeline chaining. (Phase 2 for wrappers; Phase 20 for novel GPU imputation research.)
- **Core multiple-testing methods**: Bonferroni, Holm, BH, BY, Storey q-value, simpleM/M_eff, genomic control. (Phase 8 deliverable.)

## e. Post-V1 Research Extensions

These are research-grade extensions that build on the V1 platform. They are documented in this
charter for architectural foresight but are **not V1 commitments**. Each requires its own
validation campaign and may be prioritized based on scientific impact and user demand.

- **Polyploid GWAS**: Ploidy-aware genotype encoding, gene-action models, polyploid GRM, multi-trait polyploid mvLMM. (Phase 9.)
- **GPU acceleration**: AMP pipeline, compute/IO overlap, GPU permutation testing. (Phase 10.)
- **Advanced multiple testing**: Weighted BH, adaptive GPU permutation, Cauchy combination, hierarchical FDR, local FDR. (Phase 14.)
- **Novel statistical models**: Multi-kernel LMM, GxE-LMM, set-based tests (SKAT/Burden + LDJ-LMM), Bayesian variable selection (spike-and-slab + CSMM debiased path), GPU-native imputation. (Phases 15-19.)
- **Novel inference models (Guardian charter)**: OCF-LMM (DML-valid cross-fit GWAS), KO-LMM (group knockoff FDR control), within-family GWAS, LD-conditional score GWAS, HetLMM (single-trait GxE), DML causal-inference module. (Phases 21-26.)
- **Interpretation-safety layers**: Covariate audit mode (heritable covariate / collider detection), causal-safety diagnostics (LD proxy risk, confounding sensitivity, conditional persistence), LD-block attribution. (Gate C deliverable, available for all models.)
- **Categorical-trait threshold-linear models (platform-scoped)**: Multi-trait threshold-linear model (NR/EM MAP algorithms) for joint analysis of multiple categorical and continuous traits, based on Bermann et al. (2026). GPU-accelerated per-individual NR computations, ssGWAS p-values for categorical traits, and polyploid categorical-trait extension. Threshold models are the principled approach for ordinal/binary phenotypes in the mixed-model framework. Binary and ordinal traits are in scope for the platform and scientifically anchored in the Bermann et al. manuscript — this is deferred implementation, not a philosophical exclusion. Full algorithmic detail in Section 5b. (Phase 16.)
- **Multimodal data integration (distinct from threshold-linear)**: Combining genotype with omics (transcriptomic, proteomic), environmental covariates, image features, or other modalities is a separate research direction not implied by the threshold-linear manuscript. (Not yet phased.)

---

## 1. Purpose of This Document

This document is a project handoff pack for any future AI agent or human contributor. It captures the
project identity, fixed decisions, competitive positioning, phased roadmap, validation gates, and
immediate work packages. A new agent can start productive work without prior conversation history.

## 2. Project Identity

- **Project objective**: Build a GPU-accelerated GWAS library that matches or exceeds the statistical accuracy of GEMMA/GAPIT while delivering 10-50x speedups on GPU hardware.
- **Primary benchmark**: GEMMA (single-trait LMM, multi-trait mvLMM) and GAPIT3 (FarmCPU, BLINK) for statistical correctness.
- **Secondary benchmark**: SAIGE-GPU and REGENIE for computational performance at biobank scale.
- **Core philosophy**: Exact core first, GPU acceleration second, approximate large-scale methods third.
- **Development style**: Gated phases with formal validation against reference outputs at every gate.

## 3. Language and Implementation Decision

This project is Python-first with a PyTorch computational core.

- **Python** is the primary interface because the target audience (geneticists, breeders, bioinformaticians) increasingly works in Python, and PyTorch's ecosystem (autograd, AMP, CUDA streams, distributed) provides the computational substrate.
- **PyTorch** is the numerical core for all tensor operations, eigendecomposition, batched linear algebra, optimization, and GPU dispatch.
- **No C++ core is required for V1** — PyTorch's backend (ATen/cuBLAS/cuSOLVER) provides compiled performance. C++/CUDA custom kernels may be added in later phases for specific bottlenecks.
- **R bindings** are out of scope for V1 but the architecture should not prevent them.

## 4. GEMMA Algorithmic Reference (Primary Benchmark)

TorchGWAS must faithfully replicate GEMMA's mathematical machinery before extending it. This section
documents the exact algorithms that must be matched. Misunderstanding any detail here will produce
silent p-value drift.

### 4a. Single-trait LMM (Zhou & Stephens, Nat Genet 2012)

**Model**: y = Xb + u + e, where u ~ N(0, K * sig_g^2), e ~ N(0, I * sig_e^2).

**The eigendecomposition trick** (core of GEMMA's speed):

1. Compute the relatedness matrix K = X_geno @ X_geno.T / p (centered, optionally scaled).
2. Eigendecompose K once: K = U @ diag(evals) @ U.T. Cost: O(n^3), done once.
3. Rotate all data into eigenspace: y_rot = U.T @ y, X_rot = U.T @ X, g_rot = U.T @ g (per SNP).
4. In rotated space, the covariance becomes diagonal: V_rot = diag(evals * sig_g^2 + sig_e^2).
5. This makes per-SNP operations O(n) instead of O(n^3) — the key to GEMMA being n times faster than EMMA.

**Variance component estimation (REML)**:
- GEMMA uses Newton-Raphson on the REML log-likelihood, parameterized by lambda = sig_g^2 / sig_e^2.
- In rotated space, the REML log-likelihood and its first/second derivatives reduce to O(n) sums over eigenvalues.
- TorchGWAS must match this: AI-REML or Newton-Raphson for the 2-parameter (sig_g, sig_e) case.

**Per-SNP scan** (after null fitting):
- For each SNP g, rotate: g_rot = U.T @ g.
- Compute GLS estimates of the SNP effect beta_g using the pre-computed diagonal V_rot^{-1}.
- Wald test: beta_g^2 / Var(beta_g) ~ chi^2(1).
- LRT: 2 * (ll_full - ll_null) ~ chi^2(1), where ll is evaluated at the fixed null variance components.
- Score test: gradient of the log-likelihood under the null, evaluated cheaply with no refitting.

**Critical preprocessing that GEMMA applies** (must match exactly for p-value equivalence):
- Genotype coding: 0/1/2 dosage from PLINK .bed (minor allele count).
- Missing genotypes: mean-imputed per SNP (replace NaN with SNP mean).
- Centering: subtract per-SNP mean after imputation.
- MAF filtering: remove SNPs below threshold before scan.
- Covariates: intercept always included as first column of X. Additional covariates appended.
- Sample alignment: intersection of .fam and phenotype file by IID; samples with missing phenotype dropped.
- Phenotype centering: NOT done by GEMMA (the intercept absorbs the mean).

### 4b. Multi-trait mvLMM (Zhou & Stephens, Nat Methods 2014)

**Model**: Y (n x d) = X @ B + G + E, where:
- vec(G) ~ N(0, Vg kron K), vec(E) ~ N(0, Ve kron I)
- Total covariance: V = Vg kron K + Ve kron I

**The Efficient Eigen-Decomposition (EED) trick**:
1. Eigendecompose K once (same as single-trait).
2. In rotated space, per-individual covariance becomes: Sigma_i = evals_i * Vg + Ve (d x d matrix).
3. This decouples the n-dimensional problem into n independent d-dimensional problems.
4. Per-SNP cost drops from O(n^3 * d^3) to O(n * d^3).

**Null fitting for mvLMM**:
- Estimate Vg (d x d) and Ve (d x d) via REML, parameterized through Cholesky factors (SPD guarantee).
- GEMMA uses a Newton-like approach; TorchGWAS provides two paths:
  - **LBFGS-autograd**: Minimize -REML(Vg, Ve) via torch.optim.LBFGS on Cholesky factors. Stable, general, autograd-enabled.
  - **AI-REML**: Average Information updates using batched d x d blocks from EED. Fast for d <= ~6.

**Per-SNP mvLMM scan** (Schur complement approach):
1. Precompute null quantities: M00 (normal equation matrix under null), b0 (null RHS), W_i = Sigma_i^{-1}.
2. For each SNP chunk (m SNPs):
   - Compute SNP-specific blocks: M11, M01, b1 using batched tensor ops over W_i.
   - Schur complement: S = M11 - M10 @ M00^{-1} @ M01.
   - SNP effect: beta_snp = S^{-1} @ (b1 - M10 @ M00^{-1} @ b0).
   - Joint Wald test: beta_snp.T @ S @ beta_snp ~ chi^2(d).
   - Per-trait Wald: beta_t^2 / S_inv_tt ~ chi^2(1).
3. All operations are batched over the m SNPs in a chunk — this is where GPU parallelism delivers massive speedup.

### 4c. Why tiny preprocessing differences destroy p-value equivalence

P-values in GWAS span 60+ orders of magnitude (from 1.0 down to 1e-300). Even 1e-6 relative error in intermediate quantities can shift p-values by several decimal places. The most common sources of mismatch with GEMMA are:

1. **Genotype centering convention**: GEMMA centers to mean 0 but does NOT divide by std. Some tools divide by sqrt(2 * maf * (1-maf)). This changes the GRM and all downstream quantities.
2. **Missing value handling**: GEMMA mean-imputes before centering. Imputing after centering gives different results.
3. **Sample ordering**: Different sort orders for IIDs can lead to different matrix layouts, which with floating-point non-commutativity produces different results.
4. **Covariate column order**: The intercept must be the first column of X to match GEMMA's convention.
5. **Eigenvalue precision**: Using FP32 for eigendecomposition loses ~7 decimal digits. GEMMA uses FP64. TorchGWAS must too.

## 5. Statistical Model Scope (V1)

### Models

| Model | Description | Reference |
|---|---|---|
| **GLM** | Standard linear regression with covariates, batched scan | Baseline / PLINK |
| **Single-trait LMM** | y = Xb + Zg + e with K-based random effects; REML via AI-REML + LBFGS; Score/Wald/LRT tests | EMMA / GEMMA (Zhou & Stephens 2012) |
| **Multi-trait mvLMM** | Y = XB + G + E with Kronecker covariance V = K x Vg + I x Ve; EED trick; Wald/LRT | GEMMA mvLMM (Zhou & Stephens 2014) |
| **FarmCPU** | Iterative FEM + REM with binning and pseudo-QTN selection; linear in n | Liu et al. 2016 |
| **BLINK** | LD-clustering + BIC-based multi-locus selection; no kinship in REM; linear in n | Huang et al. 2019 |

### Test statistics

- **Wald test**: beta / SE, chi-square(df) — provides effect sizes and standard errors
- **Likelihood Ratio Test (LRT)**: 2(ll_1 - ll_0), chi-square(df) — fixed-VC fast scan
- **Score test**: Null-only, fastest scan; no beta/SE but calibrated p-values

### Variance component estimation

- **Primary**: AI-REML (Average Information) with damped Newton updates for d <= ~6 traits
- **Secondary/fallback**: LBFGS on Cholesky-parameterized log-likelihood via `torch.autograd` — SPD guaranteed, stable, autograd-enabled
- **Warm-start**: PX-EM or EM-REML for first 3-10 iterations when initial values are poor

### Selection-aware inference labeling (required)

Methods that perform variable selection during modeling produce post-selection statistics. TorchGWAS must label outputs accordingly and not claim calibrated marginal p-values unless debiasing is applied.

| Method | Default output meaning | When p-values are claimable as calibrated |
|---|---|---|
| FarmCPU / BLINK | Post-selection association summaries | Only if debiasing / orthogonal inference is applied and passes null calibration gates |
| CSMM (Section 17) | Sparse mixed-model estimates | Only with debiased / orthogonal score inference |
| KO-LMM (Section 17) | FDR-controlled block/locus discoveries | Uses knockoff guarantees; not interpreted as marginal p-values |
| LD-Conditional (Section 17) | Conditional associations within LD blocks | Calibrated if conditioning set is pre-specified and gates pass |

### Default policies

- **LOCO (Leave-One-Chromosome-Out)**: Default `loco="auto"` for univariate LMM GWAS when K is built from genome-wide SNPs. If K is derived from genome-wide variants and chromosome labels are available, TorchGWAS builds per-chromosome K by excluding all variants on the tested chromosome. If K is user-supplied or chromosome labels are unavailable, LOCO is disabled and recorded in metadata.
- **Variance-component fitting**: Default: fit VC once under null, scan with fixed VC. Optional: refit VC per SNP (labeled "strict/experimental").
- **Estimand labeling**: Any covariate adjustment that changes estimand (direct vs total effect) must be labeled in outputs.

## 5b. Multi-trait Threshold-Linear Models for Categorical Traits (Post-V1)

**Reference**: Bermann, Legarra, Misztal & Lourenco (2026). "Methods for joint genetic prediction of multiple ordinal categorical and continuous traits." _Genetics_, DOI: 10.1093/genetics/iyag086.

**Motivation**: Categorical traits (disease resistance, fertility scores, lodging, survival) are central to breeding and clinical genetics, but violate normality assumptions. Threshold models postulate an unobserved Gaussian liability that is discretized into observed categories via thresholds (Wright 1934; Gianola & Foulley 1983). Prior MAP methods could only handle **one** categorical trait + many continuous traits. Bermann et al. (2026) generalize to **arbitrary numbers** of categorical + continuous traits jointly, solving a decades-old methodological gap.

### 5b-i. Model specification

```
[l; y2] = [W1  0 ] [θ1] + [e1]
           [ 0  W2] [θ2]   [e2]
```

where `l` is the vector of unobserved liabilities for c categorical traits, `y2` contains continuous phenotypes, `W_i = [X_i Z_i]` are incidence matrices, `θ'_i = [b'_i u'_i]` with `u ~ N(0, G)`, and `e ~ N(0, R⊗I)` with `R = [R11 R12; R21 R22]`.

For individual i with categorical observations `y_{1i}`, the probability of observing category vector **k** is:

```
P(y_{1i} = k) = ∫_{T_k} φ(l_i; μ_i, Σ_i) dl_i
```

where `T_k` is the hyperrectangle determined by threshold vectors, `μ_i = W_{1i}θ_1 + R12 R22^{-1}(y_{2i} - W_{2i}θ_2)` is the conditional liability mean, and `Σ_i = R11 - R12 R22^{-1} R21` is the conditional liability covariance.

### 5b-ii. Newton-Raphson MAP algorithm

The log posterior is maximized via Newton-Raphson: `θ_{j+1} = θ_j - H(j)^{-1} ∇(j)`.

**Per-individual gradient contribution**:
```
∇_i = W'_i z_i,  where z_i = [Δ_i; -R22^{-1} R21 Δ_i + R22^{-1}(y_{2i} - W_{2i}θ_i)]
```
with `Δ_i = Σ_i^{-1}(E_T[l_i] - μ_i)` — the multivariate analog of the univariate δ_i. `E_T[l_i]` is the mean of a multivariate truncated normal (no closed form; computed numerically via Manjunath & Wilhelm 2021).

**Per-individual Hessian contribution**:
```
H_i = -W'_i R̃_i^{-1} W_i
```
where `R̃_i^{-1}` contains `Γ_i = Σ_i^{-1} - Σ_i^{-1} Var_T(l_i) Σ_i^{-1}` — the multivariate analog of γ_i. `Var_T(l_i)` is the covariance of the truncated multivariate normal (computed via Manjunath & Wilhelm 2021, Eq. 45).

**Iterative system** (resembles standard MME with modified residuals):
```
(W' R̃^{-1} W + S) θ_{j+1} = W' R̃^{-1} ỹ
```
where `ỹ_i = [Γ_i^{-1} Δ_i + W_{1i}θ_i + R12 R22^{-1}(y_{2i} - W_{2i}θ_i); y_{2i}]` are pseudo-observations.

**Critical optimization**: `R̃^{-1}` is precomputed once per NR iteration (does not change during MME solving), dramatically reducing cost in iteration-on-data schemes.

**Convergence**: `||e||_2 / ||W'R̃^{-1}ỹ||_2 < 10^{-12}` where e is the MME residual.

### 5b-iii. Expectation-Maximization MAP algorithm

- **E-step**: Compute expected liabilities `l̃_i = E_T[l_i]` — the expectation of l_i under a truncated multivariate normal with mean μ_i, variance Σ_i, and truncation region T_k. Computed numerically via Eq. (44) of Manjunath & Wilhelm (2021): `E_T[l_i] = μ_i + Σ_i(F_i(t_{k-1}) - F_i(t_k))`.
- **M-step**: Solve standard MME `(W'R^{-1}W + S) θ_{j+1} = W'R^{-1} ẏ` using `ẏ = (l̃, y2)` as phenotypes — a regular linear mixed model solve.
- **SQUAREM acceleration** (Varadhan & Roland 2008) applied every other round to improve EM convergence.
- EM requires only `E_T[l_i]` (not `Var_T(l_i)`), making each iteration cheaper than NR but requiring more iterations.

### 5b-iv. Performance benchmarks (Bermann et al. 2026)

| Method | Iterations | Time (1.8M animals, 4 traits) | Correlation with Gibbs |
|---|---|---|---|
| Newton-Raphson | 7 | 29 minutes | >0.99 |
| EM + SQUAREM | 24 | 1.1 hours | >0.99 |
| Gibbs sampler | 30,000 | 3.14 days | reference |

On real data (1.25M pedigree, 150K genotyped, 4 traits): NR 2 hrs, EM 22 hrs, Gibbs infeasible.

### 5b-v. TorchGWAS GPU implementation strategy

The NR algorithm is **embarrassingly parallel** at the per-individual level:
1. **Parallel Δ_i and Γ_i computation**: Each individual's truncated multivariate normal moments are independent — batch across all n individuals on GPU.
2. **GPU-accelerated MVN CDF**: The Miwa et al. (2003) recursive integration for multivariate normal probabilities can be batched and parallelized on GPU tensors.
3. **R̃^{-1} precomputation**: Batch-invert all per-individual R̃_i matrices in a single GPU kernel.
4. **MME solving**: The modified MME `(W'R̃^{-1}W + S) θ = W'R̃^{-1}ỹ` is solved using PCG (Preconditioned Conjugate Gradient) on GPU, leveraging the same infrastructure as the LMM scan.
5. **PyTorch autograd for variance components**: Unlike the Fortran BLUPF90 implementation which assumes known variance components, TorchGWAS can use autograd to differentiate through the NR iterations for joint estimation of variance components and thresholds — a potential novel extension.
6. **ssGWAS extension**: Posterior variances from `(W'R̃^{-1}W + S)^{-1}` at convergence enable marker effect p-values for single-step GWAS on categorical traits (Aguilar et al. 2019; Leite et al. 2024).

### 5b-vi. Scope and dependencies

- **Post-V1 implementation, in-scope for platform**: Binary and ordinal traits are part of TorchGWAS's platform vision, scientifically anchored in Bermann et al. (2026). Implementation is deferred to Phase 16, after V1 Gaussian-trait models are validated.
- **Prerequisites**: Requires V1 LMM infrastructure (GRM, eigendecomposition, MME solvers, PCG).
- **Module**: `torchgwas.models.ThresholdLinearModel` implementing `BaseModel` protocol with `fit_null()` solving the NR/EM iteration and `score_chunk()` computing ssGWAS p-values.

## 6. Data Format Intelligence and Preprocessing Pipeline

### 6a-0. Internal Data Contract (Enforceable)

All supported genotype inputs must be normalized into a single internal representation so that QC, imputation, GRM construction, and association testing are consistent across formats.

**Dosage tensor**: Core tensor `G` with shape `(n_samples, n_variants)` in dosage units. Dosage is the ALT-allele count for biallelic variants. **Invariant**: internal dosage always counts ALT (A2/ALT); any "effect allele" choice for reporting is separate and recorded in artifacts. Allowed values: hard-call `{0,1,...,P}` or fractional dosage (DS/GP) in `[0, P]`. Ploidy `P` defaults to 2 unless specified. Inference requires float64; staging may use float32 but must cast back for inference.

**Missing encoding**: Raw readers preserve missingness as NaN (preferred) or sentinel value (PLINK1 may use -1). QC must be computed on raw data prior to imputation.

**Allele conventions**: Each variant must have canonical `A1` (REF) and `A2` (ALT) fields. Variant IDs use `CHR:POS[:REF:ALT]` when rsID is absent. A/T and C/G strand ambiguity must be flagged when harmonizing.

**Multiallelic handling**: Split to biallelic (default) or drop with explicit reason. `VariantHarmonizationReport` records the mapping. If a multiallelic site carries DS/GP, it is dropped by default unless `--allow_multiallelic_split_ds` is enabled with a validated dosage splitter.

**Polyploid sanity checks**: Validate dosage in `[0, P]` up to `eps_dosage` (default 1e-6, configurable). Homozygote thresholds: `dosage <= eps` = homo-REF, `dosage >= P-eps` = homo-ALT. Warn/error on systematic violations.

**GRM standardization (ploidy-aware)**: Diploid: `2p(1-p)`; ploidy P: `Pp(1-p)` where `p = mean(dosage)/P`. GRM accumulation records exact standardization policy, SNP set, and LOCO partitioning.

### 6a. The format problem in GWAS

Users arrive with data in many formats, and most GWAS tools silently fail or produce wrong results
when the format is subtly incorrect (e.g., wrong allele coding, mismatched sample IDs, wrong ploidy
encoding, missing required columns). TorchGWAS must handle this gracefully.

### 6b. Supported genotype formats

| Format | Description | Read | Write | Notes |
|---|---|---|---|---|
| **PLINK BED/BIM/FAM** | Binary PLINK 1.x; 2-bit genotypes, SNP-major | Yes (V1) | Yes | Most common GWAS format. Diploid only (0/1/2 + missing). |
| **PLINK2 PGEN/PVAR/PSAM** | PLINK 2.0 format; supports dosages, multiallelic, phased | Yes (V1) | Yes | More flexible than BED; stores dosage probabilities. |
| **VCF / BCF** | Variant Call Format (text / binary compressed) | Yes (V1) | No | Standard for sequence data. Extract GT or DS fields. |
| **BGEN** | Oxford format; stores genotype probabilities | Yes (V1) | No | UK Biobank standard. Must handle .bgen + .sample + .bgi index. |
| **HapMap** | Tab-delimited genotype format (GAPIT/TASSEL standard) | Yes (V1) | Yes | Very common in plant genetics. Details in 6e below. |
| **Numeric CSV/TSV/TXT** | Plain-text dosage or genotype matrix with marker metadata | Yes (V1) | Yes | Flexible; must contain required marker metadata columns. Details in 6f below. |
| **Zarr** | Chunked array store for cloud/out-of-core | Yes (Phase 12) | Yes | Ideal for large datasets and cloud storage. |
| **HDF5** | Hierarchical data format | Yes (Phase 12) | Yes | Alternative to Zarr for chunked access. |

### 6e. HapMap format specification

The HapMap format is widely used in plant genetics (GAPIT, TASSEL, rMVP). TorchGWAS must accept
both the standard HapMap format and common variants found in practice.

**Standard HapMap columns (tab-delimited):**

| Column | Name | Required | Description | Example |
|---|---|---|---|---|
| 1 | `rs#` or `SNP` | **Yes** | Variant/marker ID | `S1_12345` |
| 2 | `alleles` | **Yes** | Allele pair separated by `/` | `A/G` |
| 3 | `chrom` or `chr` | **Yes** | Chromosome identifier | `1` or `chr1` |
| 4 | `pos` | **Yes** | Physical position (bp) on chromosome | `12345` |
| 5 | `strand` | No | Strand orientation (`+` or `-`) | `+` |
| 6 | `assembly#` | No | Genome assembly version | `NA` |
| 7 | `center` | No | Genotyping center | `NA` |
| 8 | `protLSID` | No | Protocol LS ID | `NA` |
| 9 | `assayLSID` | No | Assay LS ID | `NA` |
| 10 | `panelLSID` | No | Panel LS ID | `NA` |
| 11 | `QCcode` | No | Quality control code | `NA` |
| 12+ | Sample genotypes | **Yes** | One column per sample; genotype coded as `AA`, `AG`, `GG`, `NN` (missing), or IUPAC codes (`A`, `M`, `R`, `W`, etc.) | `AG` |

**Example HapMap file:**
```
rs#     alleles chrom   pos     strand  assembly#  center  protLSID  assayLSID  panelLSID  QCcode  Sample1  Sample2  Sample3
S1_100  A/G     1       100     +       NA         NA      NA        NA         NA         NA      AA       AG       GG
S1_250  C/T     1       250     +       NA         NA      NA        NA         NA         NA      CC       CT       NN
S2_500  A/T     2       500     +       NA         NA      NA        NA         NA         NA      AT       TT       AA
```

**TorchGWAS HapMap parsing rules:**
1. Detect HapMap format by checking if the first line header contains `rs#` or `rs` and `alleles` and `chrom`.
2. **Required columns**: `rs#` (or `SNP`), `alleles`, `chrom` (or `chr`), `pos`. All others are optional and will be skipped if present.
3. Columns 5-11 (strand through QCcode) are metadata — read but not required for GWAS. If absent, the parser must handle the case where genotype columns start at column 5 instead of column 12.
4. **Genotype decoding**: Convert letter genotypes to numeric dosage:
   - Homozygous reference (`AA` if alleles=`A/G`): dosage = 0
   - Heterozygous (`AG` or `GA`): dosage = 1
   - Homozygous alternate (`GG`): dosage = 2
   - Missing (`NN`, `NA`, `--`, empty): dosage = NaN (to be imputed)
   - IUPAC single-letter codes: `M`=AC het, `R`=AG het, `W`=AT het, `S`=CG het, `Y`=CT het, `K`=GT het. Decode using the `alleles` column.
5. **Polyploid HapMap**: Some polyploid pipelines encode dosage directly as integers (`0`, `1`, `2`, `3`, `4` for tetraploid) in place of letter genotypes. TorchGWAS must detect this automatically (all genotype values are single digits) and parse accordingly.
6. **Chromosome naming normalization**: Accept `chr1`, `Chr1`, `CHR1`, `1`, `01` — normalize to consistent format internally.

### 6f. Numeric CSV / TSV / TXT format specification

For users who have pre-processed genotype data (common in polyploid breeding, custom genotyping
arrays, or dosage called from sequencing), TorchGWAS accepts plain-text numeric files.

**Two accepted layouts:**

**Layout A — Marker-metadata-plus-genotype (recommended, GAPIT numeric style):**

The genotype file contains ONLY the numeric dosage matrix (samples as columns, markers as rows OR
samples as rows, markers as columns). A **separate map file** provides marker metadata.

Genotype file (`genotype.csv` or `.txt` or `.tsv`):
```
Sample1,Sample2,Sample3,Sample4
0,1,2,0
1,1,0,2
2,0,1,1
```

Map file (`map.csv` or `.txt` or `.tsv`) — **required** companion file:

| Column | Name | Required | Description | Example |
|---|---|---|---|---|
| 1 | `SNP` or `Marker` or `rs` | **Yes** | Variant/marker ID | `S1_12345` |
| 2 | `Chr` or `Chromosome` or `chrom` | **Yes** | Chromosome identifier | `1` |
| 3 | `Pos` or `Position` or `bp` | **Yes** | Physical position (bp) | `12345` |

Optional map columns: `Allele1` / `A1`, `Allele2` / `A2` (reference and alternate alleles),
`MAF`, `cM` (genetic distance in centiMorgans).

**Layout B — Self-contained (all metadata embedded in the genotype file):**

A single file with marker metadata columns followed by genotype columns:
```
SNP,Chr,Pos,Allele1,Allele2,Sample1,Sample2,Sample3
S1_100,1,100,A,G,0,1,2
S1_250,1,250,C,T,1,1,0
S2_500,2,500,A,T,2,0,1
```

**Mandatory metadata columns** (must be present in EITHER the genotype file or a companion map file):

| Column | Accepted names (case-insensitive) | Required | Why |
|---|---|---|---|
| **Marker ID** | `SNP`, `Marker`, `rs`, `rs#`, `MarkerName`, `ID`, `Name` | **Yes** | Unique variant identifier for results output and LD computations |
| **Chromosome** | `Chr`, `Chromosome`, `chrom`, `chr#`, `CHR` | **Yes** | Required for Manhattan plots, LOCO GRM, LD-based imputation, chromosome-aware multiple testing |
| **Position** | `Pos`, `Position`, `bp`, `BasePair`, `pos`, `POS`, `map` | **Yes** | Required for Manhattan plots, LD computation, regional association plots, physical-position-based clumping |

**Optional but recommended metadata columns:**

| Column | Accepted names | Why useful |
|---|---|---|
| Allele 1 | `A1`, `Allele1`, `REF`, `Reference` | Needed for allele-frequency reporting, strand alignment, imputation |
| Allele 2 | `A2`, `Allele2`, `ALT`, `Alternate` | Same as above |
| Genetic distance | `cM`, `GeneticDist`, `cm` | Used for recombination-aware LD estimation |

**TorchGWAS CSV/TSV/TXT parsing rules:**
1. **Auto-detect delimiter**: Try tab first, then comma, then space. Count consistent columns.
2. **Auto-detect orientation**: If the first row looks like sample IDs and the first column looks like marker IDs, infer markers-as-rows. If the first column has chromosome-like values (integers 1-30, or "chr1"-"chrX"), infer samples-as-rows. Ask user to confirm via `--orientation rows|cols` if ambiguous.
3. **Auto-detect companion map file**: If the genotype file has no chromosome/position columns, look for `<basename>.map`, `<basename>_map.csv`, `<basename>_map.txt` in the same directory. If not found, raise a clear error: `"Genotype file lacks chromosome and position columns. Provide a map file via --map <path> with columns: SNP, Chr, Pos"`.
4. **Validate mandatory columns**: Before any computation, check that SNP, Chr, and Pos are present (either in the genotype file or the map file). If any are missing, raise a clear error listing exactly which columns are missing and suggesting the accepted column names.
5. **Handle common encoding variants**:
   - Numeric dosage: `0`, `1`, `2` (diploid) or `0`-`k` (polyploid) — parse as float
   - Letter genotypes in CSV: `AA`, `AG`, `GG` — decode like HapMap
   - `-1` or `NA` or `NaN` or empty or `.` or `--` — all treated as missing (NaN)
   - Continuous dosage from imputation: `0.0`, `1.37`, `1.98` — parse as float directly
6. **Column name matching**: Case-insensitive, strip whitespace, strip quotes. Accept any of the synonym lists above.

### 6g. Phenotype and covariate file format

Phenotype and covariate files follow the same flexible parsing:

**Phenotype file** (CSV/TSV/TXT):

| Column | Accepted names | Required | Description |
|---|---|---|---|
| Sample ID | `IID`, `ID`, `Sample`, `SampleID`, `Genotype`, `Taxa` | **Yes** | Must match sample IDs in genotype file |
| Trait(s) | User-specified names | **Yes** | One or more numeric columns; `NA`/`NaN`/empty = missing |

**Covariate file** (CSV/TSV/TXT, or embedded in phenotype file):

| Column | Accepted names | Required | Description |
|---|---|---|---|
| Sample ID | Same as phenotype | **Yes** | Must match genotype and phenotype files |
| Covariate(s) | User-specified names | **Yes** | Numeric or categorical (auto-encoded to dummy variables) |

**Sample alignment rules (deterministic enforcement):**

Sample alignment is a critical production edge — mismatches between .fam/.psam IDs and phenotype/covariate IDs silently produce wrong results in other tools. TorchGWAS must enforce alignment deterministically:

1. **Load all ID sets explicitly**: Extract sample IDs from genotype source (.fam IID for PLINK, sample column for VCF/BGEN, row/column headers for HapMap/CSV), phenotype file, and covariate file.
2. **Compute three-way intersection**: `aligned_ids = geno_ids ∩ pheno_ids ∩ covar_ids`. Use exact string matching (no case folding, no whitespace trimming — mismatches are bugs in upstream data).
3. **Sort deterministically**: Always sort `aligned_ids` lexicographically. This guarantees identical row ordering regardless of input file order, platform, or locale. Never rely on input file order.
4. **Reindex all matrices to aligned order**: Genotype tensor, phenotype vector(s), and covariate matrix must all be reindexed to the sorted `aligned_ids` order before any computation. Store the mapping `original_index -> aligned_index` for traceability.
5. **Drop samples with missing phenotype**: For each trait being analyzed, further drop samples where the phenotype is NA/NaN. For multi-trait mvLMM, follow GEMMA convention: drop sample only if ALL traits are missing; partially observed samples are retained.
6. **Hard errors on critical mismatches**:
   - If `geno_ids ∩ pheno_ids` is empty → raise error, not a warning.
   - If any covariate file ID is not in `geno_ids ∩ pheno_ids` → raise error (covariate file has samples not in the analysis set — likely a file mix-up).
   - If >50% of genotype samples have no phenotype → raise error (likely wrong file pairing).
7. **Structured warnings for non-critical drops**:
   - If 10-50% of samples are dropped → warn with counts: "N genotype-only, M phenotype-only, K covariate-only samples excluded."
   - If <10% dropped → info-level log.
8. **Report in pre-flight summary**: Final aligned sample count, per-source counts (genotype N, phenotype N, covariate N), intersection N, and dropped counts by reason.
9. **Write alignment manifest**: Output `alignment_manifest.tsv` with columns `[IID, in_genotype, in_phenotype, in_covariate, included, drop_reason]` — enables post-hoc auditing of which samples entered the analysis and why others were excluded.

### 6c. Automatic format detection and validation

TorchGWAS must implement a `DataLoader` that:

1. **Auto-detects format** using this priority chain:
   - Binary magic bytes: `.bed` (0x6C1B01), `.bgen` (BGEN magic), `.pgen` (PGEN magic)
   - File header: VCF (`##fileformat=VCFv4.x`), HapMap (header contains `rs#` + `alleles` + `chrom`)
   - Extension: `.vcf`, `.vcf.gz`, `.bcf`, `.bgen`, `.hapmap`, `.hmp.txt`
   - For `.csv`, `.tsv`, `.txt`: inspect header row for mandatory columns (SNP/Chr/Pos) or HapMap-style columns (rs#/alleles/chrom). If numeric-only first data row, treat as numeric dosage and look for companion map file.
   - If format cannot be determined, raise a clear error listing detected characteristics and asking user to specify `--format`.
2. **Validates structure** before any computation:
   - Checks for matching sample counts across genotype/phenotype/covariate files.
   - Verifies mandatory metadata: every variant must have a marker ID, chromosome, and position. If missing, raise an error with specific remediation instructions.
   - Verifies allele coding (biallelic for diploid GWAS; multi-dosage for polyploid).
   - Detects and reports chromosome naming inconsistencies (e.g., "chr1" vs "1") and normalizes.
   - Checks for duplicate sample IDs or variant IDs.
   - Warns about excessive missingness (per-sample and per-variant).
   - For HapMap/CSV: validates that genotype encoding matches expected ploidy.
3. **Converts between formats** on-the-fly when needed:
   - VCF -> internal tensor representation (extract GT or DS field).
   - BGEN -> dosage tensor (sum genotype probabilities).
   - HapMap -> numeric dosage tensor (letter genotypes decoded via alleles column).
   - CSV/TSV/TXT -> dosage tensor (with map file join if metadata is separate).
   - Polyploid dosage matrix -> appropriate tensor encoding.
   - All conversions preserve marker ID, chromosome, position, and allele information.
4. **Reports a pre-flight summary** before any GWAS run:
   ```
   TorchGWAS Pre-flight Check:
     Format:     PLINK BED (SNP-major)
     Samples:    5,000 (4,832 after phenotype intersection)
     Variants:   500,000 (487,221 after MAF > 0.01 filter)
     Ploidy:     Diploid (2x)
     Traits:     3 (Y1, Y2, Y3)
     Covariates: 2 (age, sex) + intercept
     Missing:    0.3% genotype, 0 phenotype
     Device:     cuda:0 (NVIDIA A100, 40GB)
   ```

### 6h. Format conversion utilities

```bash
# Convert VCF to PLINK BED
torchgwas convert --input data.vcf.gz --output data --format bed

# Convert HapMap to PLINK BED
torchgwas convert --input data.hmp.txt --output data --format bed

# Convert numeric CSV (with separate map) to PLINK BED
torchgwas convert --input genotype.csv --map marker_map.txt --output data --format bed

# Convert BGEN to Zarr (for out-of-core processing)
torchgwas convert --input data.bgen --sample data.sample --output data.zarr --format zarr

# Validate a dataset without running GWAS (works with any format)
torchgwas validate --genotype data.hmp.txt --phenotype pheno.txt --report report.json
torchgwas validate --genotype dosage.csv --map snp_info.txt --phenotype pheno.csv --report report.json
```

### 6i. Variant QC filters and allele-frequency Parquet output

TorchGWAS must compute per-variant QC statistics during preprocessing and persist them to a Parquet file. This serves three purposes: (1) reproducible QC — the exact filter decisions are auditable; (2) QA against GAPIT/GEMMA defaults — users can verify that TorchGWAS computes identical allele frequencies and filter counts; (3) downstream analysis — allele-frequency columns are required for meta-analysis, LD score regression, and polyploid gene-action model selection.

**Per-variant statistics (computed during Phase 2 preprocessing):**

| Column | Definition | Diploid | Polyploid (ploidy k) |
|---|---|---|---|
| `SNP` | Marker ID | from .bim / VCF ID / HapMap rs# | same |
| `CHR` | Chromosome | from .bim / VCF CHROM | same |
| `POS` | Base-pair position | from .bim / VCF POS | same |
| `A1` | Effect allele (minor) | from .bim col 5 | same |
| `A2` | Reference allele (other) | from .bim col 6 | same |
| `N_OBS` | Number of non-missing genotype calls | count of non-NaN samples | same |
| `N_MISS` | Number of missing genotype calls | count of NaN samples | same |
| `MISS_RATE` | Per-variant missingness rate | N_MISS / N_TOTAL | same |
| `AF` | Allele frequency of A1 | mean(dosage) / 2 | mean(dosage) / k |
| `MAF` | Minor allele frequency | min(AF, 1 - AF) | min(AF, 1 - AF) |
| `HET` | Observed heterozygosity | freq(dosage == 1) | freq(0 < dosage < k) |
| `HET_EXP` | Expected heterozygosity under HWE | 2 * AF * (1 - AF) | generalized HW (De Silva et al. 2005) |
| `HET_EXCESS` | Excess heterozygosity | (HET - HET_EXP) / HET_EXP | same formula |
| `HWE_P` | Hardy-Weinberg equilibrium p-value | chi-square exact test | polyploid HW test (Levene 1949 extension) |
| `IMPUTED` | Whether the variant was imputed | bool | bool |
| `IMP_RSQ` | Imputation quality (Rsq/DR2) if imputed | float, NaN if genotyped | same |

**Filter thresholds (CLI-configurable, with GEMMA/GAPIT-matching defaults):**

| Filter | Default (diploid) | Default (polyploid) | GEMMA default | GAPIT default |
|---|---|---|---|---|
| `--maf-min` | 0.01 | 0.01 | 0.01 | 0.05 |
| `--miss-max` | 0.10 | 0.10 | 0.05 | 0.30 |
| `--hwe-p-min` | 1e-6 | 1e-4 | not applied | not applied |
| `--het-excess-max` | none | none | not applied | not applied |
| `--imp-rsq-min` | 0.3 | 0.3 | n/a | n/a |

**Filter application order** (matches GEMMA convention):
1. Remove variants with `MISS_RATE > --miss-max`
2. Remove variants with `MAF < --maf-min`
3. Remove variants with `HWE_P < --hwe-p-min` (optional, off by default for case-control to avoid removing true associations)
4. Remove variants with `HET_EXCESS > --het-excess-max` (optional, useful for detecting genotyping artifacts)
5. Remove imputed variants with `IMP_RSQ < --imp-rsq-min`

**Parquet output:**

After computing all statistics and applying filters, TorchGWAS writes a Parquet file (`{prefix}_variant_qc.parquet`) containing all columns above plus a `FILTER_PASS` boolean column. This file is written **before** the GWAS scan begins and is referenced in the results output.

```python
# Parquet schema (via pyarrow)
variant_qc_schema = pa.schema([
    ('SNP', pa.string()),
    ('CHR', pa.string()),
    ('POS', pa.int64()),
    ('A1', pa.string()),
    ('A2', pa.string()),
    ('N_OBS', pa.int32()),
    ('N_MISS', pa.int32()),
    ('MISS_RATE', pa.float32()),
    ('AF', pa.float64()),
    ('MAF', pa.float64()),
    ('HET', pa.float64()),
    ('HET_EXP', pa.float64()),
    ('HET_EXCESS', pa.float64()),
    ('HWE_P', pa.float64()),
    ('IMPUTED', pa.bool_()),
    ('IMP_RSQ', pa.float64()),
    ('FILTER_PASS', pa.bool_()),
    ('FILTER_REASON', pa.string()),  # e.g., "MAF<0.01", "MISS>0.10", "PASS"
])
```

**Why Parquet**: Columnar format with efficient compression; trivially loadable in pandas/polars/R arrow for QA comparisons; supports per-column statistics for fast filtering; standard in bioinformatics pipelines (Hail, REGENIE output). Parquet is already implied by the pyarrow optional dependency.

**Polyploid-aware QC definitions**: `AF = mean(dosage)/P`, `MAF = min(AF, 1-AF)`, `MAC = P * N_nonmiss * MAF`, `HET_RATE = mean(0 < dosage < P | non-missing)` (generalizes diploid heterozygosity; optionally exported as `NONHOM_RATE` alias), `MONO` if `MAF==0` or numeric variance ~0. QC must be computed from raw genotypes prior to analysis-mode imputation.

**QC inclusion policy**: Default: variants failing QC excluded from inference. If `write_filtered=True`, all variants written but failing variants have statistics set to NaN and `PASS_QC=false`. Monomorphic variants always excluded. Per-sample missingness filtering is opt-in; dropped sample IDs recorded.

### 6j. Covariate Audit Mode (pre-GWAS diagnostic; default warn-only)

Before running GWAS, TorchGWAS can run a lightweight covariate audit (enabled by default as warn-only):
- Run quick SNP→covariate GWAS + covariate→trait association
- Detect covariates that are heritable / genetically influenced (Aschard et al. 2015, AJHG)
- Label covariates as likely confounders vs mediators vs collider risks
- Default: warn-only (no auto-dropping unless explicitly requested by user)
- Output: `CovariateAuditReport` (Parquet + JSON summary) with per-covariate flags

**Rationale**: Conditioning on a heritable mediator biases GWAS results; conditioning on a collider induces spurious associations. Most GWAS tools silently accept any covariate list. The audit helps users identify problematic covariates before they corrupt results.

### 6k. Causal-Safety Diagnostics (per-SNP / per-locus; always available)

For each SNP and/or locus, TorchGWAS produces diagnostic components:
- **LD proxy risk**: LD block ID, block size, r²-to-lead, conditional persistence (whether significance survives conditioning on lead SNP)
- **Confounding sensitivity**: stability across model families (GLM → LMM → multi-locus); within-family attenuation when pedigree/sibship data available
- **Covariate-collider risk**: flags covariates likely downstream/heritable; labels estimand as "direct effect" when relevant
- **Redundancy attribution**: multiple SNPs/covariates carrying similar information at block level

These are exported as individual Parquet columns (`LD_PROXY_SCORE`, `CONFOUND_SENS_SCORE`, `COV_COLLIDER_RISK`). Each component is independently interpretable. TorchGWAS does NOT combine them into a single composite "causal credibility" score — combining heterogeneous diagnostics into one number lacks statistical justification and invites misinterpretation. Users may combine components for their own purposes.

**GEMMA/GAPIT QA workflow**: Users can load `_variant_qc.parquet` alongside GEMMA's `.log.txt` (which reports MAF/missingness counts) or GAPIT's `GAPIT..Genotype.Frequency.csv` and verify column-by-column agreement. For polyploid, compare against GWASpoly's allele frequency output.

## 7. Genotype Imputation and Phasing

### 7a. Why imputation must be integrated

Most real-world GWAS datasets have missing genotypes (from failed assays, low coverage sequencing,
or merging arrays with different SNP content). Users currently must run external tools (BEAGLE,
IMPUTE5, Minimac4) separately before GWAS — a manual, error-prone step. TorchGWAS integrates
imputation as an optional pre-GWAS pipeline stage.

### 7b. Imputation tiers (inference safety contract)

TorchGWAS classifies imputation methods by their suitability for downstream inference:

**Tier A — Inference-safe (default for analysis mode)**
- Mean dosage imputation (per variant) — standard for GWAS, preserves allele frequency
- Mode / major-allele imputation (per variant)
- Applies to all input formats.

**Tier B — QC convenience (optional, labeled)**
- KNN over samples using GRM (small n only; disabled by default)
- LD-based local imputation using sliding windows
- Must be labeled and recorded in run artifacts.

**Tier C — Simulation-only (blocked by default in inference)**
- Random HWE sampling and similar stochastic imputers
- Requires explicit opt-in (`--allow_noninferential_impute`); outputs labeled "simulation-only".

### 7b-2. Imputation backends

TorchGWAS does not reimplement BEAGLE's Li-and-Stephens HMM from scratch (that would be a multi-year
effort). Instead, it provides:

1. **Built-in simple imputation** (Tier A, always available, no external dependencies):
   - Mean imputation (per-SNP mean, standard for most GWAS)
   - Mode imputation (most common genotype)
   - K-nearest-neighbor imputation using the GRM (Tier B: novel, use kinship to inform imputation)
   - LD-based local imputation using sliding windows (Tier B: lightweight, no reference panel needed)

2. **External tool wrappers** (pluggable, user selects backend):

   | Tool | Capabilities | When to use |
   |---|---|---|
   | **BEAGLE 5.5** | Phasing, imputation with reference panel, IBD detection, high accuracy for common variants | Default recommendation; best speed/accuracy tradeoff |
   | **IMPUTE5** | Very accurate for rare variants, large reference panels | When rare variant accuracy is critical |
   | **Minimac4** | Low memory usage, fast | Memory-constrained environments |
   | **STITCH** | Reference-free imputation from low-coverage sequencing | Low-pass sequencing data without reference panel |

   TorchGWAS wraps these tools with:
   - Automatic format conversion (TorchGWAS internal format -> tool's required input -> parse output back).
   - Unified interface: `torchgwas impute --method beagle|impute5|minimac4|stitch --genotype data.vcf`
   - Reference panel management: download, index, and cache standard panels (1000 Genomes, HRC).
   - Quality metrics: per-SNP imputation R-squared (DR2/Rsq), INFO score filtering.
   - **ExternalImputeReport** (required when external imputation is used): tool name + version, command/parameter hash, reference panel name + version/hash, genetic map version/hash, input/output file hashes, phasing status, sample/variant counts before/after.

### 7b-3. Format-aware imputation decision matrix (required enforcement)

External haplotype-aware imputation requires VCF/BCF. TorchGWAS must enforce:

| Input format | Internal Tier A (mean/mode) | External (e.g., Beagle) | Notes |
|---|---|---|---|
| PLINK1 (.bed/.bim/.fam) | Yes | Via conversion only | Conversion must preserve REF/ALT and biallelic normalization |
| VCF/BCF | Yes | Yes | Prefer DS/GP when present |
| HapMap | Yes (if dosage derivable) | Via conversion only | Must provide allele fields and unambiguous biallelic mapping |
| TXT/TSV/CSV dosage | Yes | Via conversion only | Requires CHR/POS + REF/ALT per variant; if absent, external imputation disabled |

Conversion to VCF is attempted only when: variants can be normalized to biallelic, REF/ALT are defined and harmonized, sample IDs are stable, output VCF passes validation. If prerequisites fail: fall back to Tier A or error (never silently proceed with invalid external imputation).

3. **Novel GPU-accelerated imputation** (Phase 19, **DONE**):
   - **Li-and-Stephens HMM** (`impute_li_stephens`): Genotype-level first-order HMM where hidden
     states = {0, 1, ..., ploidy} and transitions encode LD between adjacent markers. Forward-backward
     runs in log-space with all samples batched on GPU in parallel. Laplace-smoothed transition matrices,
     windowed processing for memory control, arbitrary ploidy support.
     Validated on GWASpoly potato ground truth (958 samples, 9888 markers, ploidy=4):
     corr=0.906, concordance=0.668, MAE=0.417 at 5% masking.
   - **Deep learning masked autoencoder** (`impute_deep_learning`): Encoder-decoder architecture
     trained with masked reconstruction (random masking of observed entries). Learns non-adjacent LD
     patterns. Validated on GWASpoly: corr=0.917 on 200-marker subset.
   - Both methods outperform mean imputation (0.90 vs 0.83 correlation on real data).
   - CLI: `torchgwas impute --method li-stephens|deep-learning --ploidy K`
   - Reference: Li & Stephens (2003, Genetics); Naito et al. (J Hum Genet 2023) for DL imputation.

### 7c. Phasing

Phasing (resolving which alleles are on the same haplotype) is required before imputation and is
useful for haplotype-based association tests. TorchGWAS provides:

- **Wrapper for BEAGLE phasing**: BEAGLE 5.5's two-stage phasing algorithm handles both common
  and rare variants efficiently. TorchGWAS calls BEAGLE, parses phased output, and stores
  haplotypes as a (n, 2, p) tensor (two haplotypes per diploid individual).
- **Phased-aware downstream analysis**: Once phased, haplotype-based tests (haplotype regression,
  parent-of-origin effects) become possible in future phases.

### 7d. Imputation quality control

After imputation, TorchGWAS automatically:
- Filters variants with imputation quality below threshold (default: Rsq > 0.3 for GWAS, > 0.8 for fine-mapping).
- Reports imputation quality distribution (histogram of Rsq/INFO scores).
- Tags imputed vs. directly genotyped variants in the output (column `imputed: true/false`).

### 7e. CLI interface for imputation

```bash
# Simple mean imputation (built-in)
torchgwas impute --genotype data.bed --method mean --output data_imputed.bed

# BEAGLE imputation with reference panel
torchgwas impute --genotype data.vcf.gz --method beagle \
  --ref-panel 1000G_phase3.vcf.gz --output data_imputed.vcf.gz

# Impute + phase + run GWAS in one pipeline
torchgwas pipeline --genotype data.vcf.gz \
  --impute beagle --ref-panel 1000G.vcf.gz \
  --phase beagle \
  --model lmm --test wald \
  --phenotype pheno.txt --trait Y1 \
  --output results.tsv
```

## 8. Multiple Testing Correction Framework

### 8a. Why a comprehensive framework matters

GWAS tests hundreds of thousands to millions of SNPs. Naive Bonferroni correction (p < 0.05 / M)
is widely used but is overly conservative because SNPs in LD are not independent. More sophisticated
methods improve power without inflating false positives. TorchGWAS must provide a complete toolkit.

### 8b. Implemented methods (V1)

| Method | Type | Description | When to use |
|---|---|---|---|
| **Bonferroni** | FWER | p < alpha / M; simplest, most conservative | Quick threshold; always report alongside others |
| **Sidak** | FWER | p < 1 - (1-alpha)^(1/M); slightly less conservative than Bonferroni | Alternative FWER when tests are independent |
| **Holm step-down** | FWER | Sequential rejection procedure; uniformly more powerful than Bonferroni | Default FWER recommendation |
| **Benjamini-Hochberg (BH)** | FDR | Controls expected proportion of false discoveries at level q | Standard for GWAS discovery; default recommendation |
| **Benjamini-Yekutieli (BY)** | FDR | BH variant valid under arbitrary dependence | Conservative FDR when LD structure is complex |
| **Storey q-value** | FDR | Estimates proportion of true nulls (pi0) for adaptive FDR | More powerful than BH when many true signals exist |
| **Permutation (maxT)** | FWER | Westfall-Young maxT; resamples phenotype labels to build null distribution | Gold standard for FWER; accounts for LD exactly; GPU-accelerable |
| **simpleM / M_eff** | FWER | Spectral decomposition of LD matrix to estimate effective number of independent tests; adjusted Bonferroni: p < alpha / M_eff | Recommended when permutation is too slow; much less conservative than naive Bonferroni |
| **Genomic control (lambda_GC)** | Calibration | Median chi-square / 0.4549; inflation factor for systematic bias detection | Always compute; flag if lambda_GC > 1.1 |

### 8c. Novel / advanced methods (post-V1)

| Method | Type | Description | Novelty |
|---|---|---|---|
| **Weighted BH (wBH)** | FDR | Weight p-values by prior information (MAF, functional annotation, LD score) — prioritizes biologically plausible SNPs | Implements wBHa (Ignatiadis et al. 2023) for MAF-aware weighting |
| **Adaptive permutation** | FWER | Run more permutations only for SNPs near the threshold; GPU-batched to make 10K+ permutations fast | GPU parallelism makes permutation testing practical for genome-wide use |
| **COMBI** | Hybrid | Train SVM on SNP effects, select candidate set, then apply corrected threshold to the subset | ML-assisted multiple testing (Brzyski et al., Sci Rep 2016) |
| **Cauchy combination test** | Meta | Combine p-values across correlated tests using Cauchy distribution — avoids LD matrix estimation | Useful for combining multi-trait or multi-method results |
| **Hierarchical FDR** | FDR | Two-level FDR: first test gene-level significance, then SNP-level within significant genes | More powerful for gene-centric analyses |
| **Local FDR (lfdr)** | FDR | Posterior probability that each SNP is a true null, estimated from the p-value distribution | Enables probabilistic ranking of discoveries |

### 8d. GPU-accelerated permutation testing

Permutation testing is the gold standard for multiple testing correction because it exactly accounts
for the LD structure without any distributional assumptions. The bottleneck is cost: 10,000
permutations x 500,000 SNPs = 5 billion test statistics. TorchGWAS makes this practical:

1. Permute the phenotype vector (shuffle Y, keep X and G fixed) — maintains LD structure.
2. Recompute the full scan for each permutation — the same batched scan loop from Phase 7.
3. Track the maximum test statistic per permutation — this gives the maxT null distribution.
4. p_adj = (number of permutations where maxT >= observed stat) / (total permutations).

On GPU with batched scanning, a single permutation takes ~the same time as the original scan.
10,000 permutations on GPU could complete in the time that 100 permutations take on CPU.

## 9. Polyploid GWAS Support

### 9a. Why polyploid support is critical

Many economically important crops are polyploid: potato (tetraploid, 4x), wheat (hexaploid, 6x),
sugarcane (octaploid+, 8x-12x), strawberry (octaploid, 8x), sweet potato (hexaploid, 6x),
cotton (allotetraploid, 4x), oat (hexaploid, 6x), and many forage grasses. The only dedicated
polyploid GWAS tool is **GWASpoly** (Rosyara et al. 2016, R package), which is:
- R-only, slow, cannot handle large datasets
- Limited to autopolyploids
- No GPU support
- No multi-trait capability
- No integration with imputation or phasing tools
- No FarmCPU/BLINK for polyploids

TorchGWAS aims to be, to our knowledge, the first GPU-accelerated, multi-model polyploid GWAS tool.

### 9a-2. PloidyConfig (user-facing, enforceable)

TorchGWAS exposes a public configuration object that determines how ploidy is interpreted and validated.

**Fields**:
- `ploidy`: one of `{2, 4, 6}` or `"from_vcf"` or `"per_variant_map"`
- `crop`: optional shortcut (e.g., `"potato"`, `"alfalfa"` → P=4; `"wheat"` → P=6). Must never override an explicit `ploidy`.
- `strict_ploidy`: boolean (default `True`). If `True`, ploidy mismatch triggers ERROR.
- `ploidy_mismatch_policy`: `"error"` (default) or `"warn+override"`
- `per_variant_map_path`: path to mapping file when `ploidy="per_variant_map"`

**Resolution order**: (1) explicit `ploidy` integer, (2) `per_variant_map`, (3) `from_vcf` (derive from VCF headers/GT counts), (4) `crop` shortcut, (5) fallback P=2.

**Required run metadata**: `ploidy_source`, `ploidy_value`, `ploidy_mismatch_summary`, `per_variant_map_hash` (when used).

**Polyploid format constraints**: PLINK1 `.bed` is diploid-first; polyploid workflows should prefer VCF/BCF with DS/GP or dosage matrices. If PLINK1 is used for polyploid, TorchGWAS requires explicit user confirmation and strict validation.

### 9b. Ploidy-aware genotype encoding

For a k-ploid organism with biallelic SNPs, genotype dosage ranges from 0 to k:

| Ploidy | Dosage range | Example organism |
|---|---|---|
| Diploid (2x) | 0, 1, 2 | Human, maize, rice |
| Tetraploid (4x) | 0, 1, 2, 3, 4 | Potato, alfalfa, cotton |
| Hexaploid (6x) | 0, 1, 2, 3, 4, 5, 6 | Wheat, sweet potato, oat |
| Octaploid (8x) | 0, 1, 2, ..., 8 | Sugarcane, strawberry |

TorchGWAS stores genotypes as float tensors with values in [0, k]. This representation:
- Naturally extends the diploid 0/1/2 coding.
- Supports continuous dosage from imputation (e.g., 2.37 for a tetraploid).
- Works with all existing tensor operations (centering, scaling, matmul) without code changes.
- The only code that changes is GRM construction and gene-action model encoding.

### 9c. Polyploid gene-action models

Unlike diploids where only additive effects are typically modeled, polyploids require multiple
gene-action models because dominance relationships depend on dosage thresholds. GWASpoly
introduced these models; TorchGWAS generalizes them to arbitrary ploidy k:

| Model | Encoding function f(d) for dosage d in [0,k] | Description |
|---|---|---|
| **General** | d (raw dosage) | No assumption about gene action; equivalent to regression on dosage |
| **Additive** | d - k/2 (centered) | Standard additive model; linear dose-response |
| **Simplex dominant (1-dom)** | 1 if d >= 1, else 0 | One copy of the minor allele is sufficient for full effect |
| **Duplex dominant (2-dom)** | 1 if d >= 2, else 0 | Two copies needed for full effect (tetraploid+) |
| **Triplex dominant (3-dom)** | 1 if d >= 3, else 0 | Three copies needed (hexaploid+) |
| **k-1 dominant** | 1 if d >= k-1, else 0 | Recessive model: nearly all copies needed |
| **Diplo-additive** | min(d, k-d) | Heterozygosity-based; peaks at d=k/2 |
| **Overdominant** | 1 if 0 < d < k, else 0 | Any heterozygous state |

For each SNP, TorchGWAS can test all gene-action models and report the best-fitting model
alongside the standard additive result. This is critical because the true gene action is unknown.

**Diplo-general model**: GWASpoly implements a "diplo-general" model that first diploidizes genotypes ({0->0, 1..k-1->1, k->2}) then encodes as a factor (one-hot). TorchGWAS should implement this as an additional multi-column model alongside the existing "general" model. Note: TorchGWAS's `diplo-additive` uses `min(d, k-d)` which provides finer allelic resolution than GWASpoly's diploidized encoding — both are valid choices for different biological hypotheses.

### 9d. Polyploid GRM / kinship matrix

The kinship matrix K must account for ploidy. TorchGWAS implements multiple GRM methods, benchmarked against AGHmatrix (Amadeu et al. 2016) and GWASpoly:

#### Currently implemented

1. **VanRaden-style polyploid GRM** (generalization):
   ```
   K = (X - 2P)(X - 2P).T / sum(2 * p_j * (1 - p_j) * k/2)
   ```
   where X is the (n x m) dosage matrix, P is the matrix of allele frequencies (each multiplied by k/2),
   and the denominator normalizes by expected variance under Hardy-Weinberg for ploidy k.

2. **Separate GRMs per gene-action model**: For dominant models, X is recoded via f(d) before
   computing the GRM, giving a model-specific relatedness matrix.

3. **LOCO (Leave-One-Chromosome-Out)**: Compute K excluding the chromosome being tested,
   as in GWASpoly. Essential for avoiding proximal contamination.

4. **Zhang (GEMMA-style)** centered-only GRM: implemented for diploid; extends to polyploid.

#### Additional GRM methods to implement (from AGHmatrix / polyBreedR literature)

5. **Slater et al. (2016)**: Polyploid-specific GRM with modified centering that accounts for the combinatorial structure of polyploid genotypes. Uses a different scaling than VanRaden for ploidy > 2. AGHmatrix method `"Slater"`.

6. **Endelman & Jannink (2012) digenic interaction GRM**: Tetraploid-specific (ploidy=4 only). Computes non-additive relationship using `6p²q²` normalization — captures digenic epistatic effects.
   ```
   W_ij = 6p²(1-p)² normalization
   X_recoded = 6p² - 3p*d + 0.5*d*(d-1)
   K_digenic = X_recoded @ X_recoded.T / sum(6p²q²)
   ```
   This is relevant for traits where heterozygous-class interactions matter (e.g., disease resistance in potato). From AGHmatrix method `"Endelman"`.

7. **Vitezica et al. (2013) dominance GRM**: Genomic dominance relationship matrix — proper orthogonal partition separating additive and dominance variance components. For diploid: encodes heterozygotes as `2pq`, homozygotes as `-(2p²)` or `-(2q²)`. AGHmatrix method `"Vitezica"`. Extension to polyploid: encode based on heterozygosity classes.

8. **Su et al. (2012) dominance GRM**: Alternative dominance coding — heterozygotes coded as `1 - 2pq`, homozygotes as `0 - 2pq`. Normalizer = `sum(pq * (1 - pq))`. AGHmatrix method `"Su"`.

9. **Yang et al. (2010) GCTA-style GRM**: Per-marker normalization (each marker standardized individually before computing outer product), yielding a GRM where diagonal elements directly estimate individual inbreeding. AGHmatrix method `"Yang"`.

10. **Weighted GRM**: Marker-weight-aware computation `K = X @ diag(w) @ X.T / normalizer`, supporting LD-weighted or GWAS-informed kinship (for iterative GWAS like FarmCPU in polyploid context).

11. **Pseudo-diploid GRM**: Convert polyploid dosages to pseudo-diploid (0/1/2) before GRM computation — useful as a fallback when polyploid-specific methods produce unstable estimates (small n, high ploidy).

12. **Ploidy-correction toggle**: Option to normalize by `K*p*(1-p)` (ploidy-corrected, Hardy-Weinberg-based) vs `var(x)` (empirical variance-based). AGHmatrix supports both via `ploidy.correction` flag.

#### GRM scaling and transformation

13. **ASV (Additive Standardized Variance) transformation**: Scale GRM so that average diagonal equals 1: `K_asv = K / (trace(K) / (n - 1))`. This ensures the GRM diagonal approximates individual inbreeding coefficients. Recommended by Feldmann et al. (2022) for comparability across GRM methods. Applied post-hoc to any GRM.

#### Epistatic and IBD-based GRM methods

14. **Epistatic GRM via Hadamard products**: Following Muñoz et al. (2014) and Su et al. (2012), construct epistatic relationship matrices as element-wise products of additive (G) and dominance (D) GRMs:
    - `G#G` — additive × additive epistasis
    - `D#D` — dominance × dominance epistasis
    - `D#G` or `G#D` — additive × dominance epistasis
    These capture non-additive genetic effects beyond single-locus dominance. Implemented in polyBreedR.

15. **GIBD (Genomic IBD-based kinship)**: polyBreedR method for computing genomic kinship from IBD (identity-by-descent) probabilities estimated from low-depth sequencing data. Uses posterior IBD state probabilities (e.g., from software like polyRAD or updog) rather than dosage point estimates. Relevant for polyploid species where dosage calling uncertainty is high.

#### Pedigree and blended relationship matrices (Post-V1)

16. **Pedigree A-matrix for polyploids**: Kerr et al. (2012) method for any even ploidy, with optional double reduction parameter `w` (Slater et al. 2014 for tetraploids). From AGHmatrix `Amatrix`.

17. **H-matrix (single-step blending)**: Legarra et al. (2009) / Martini et al. (2018) — blends pedigree A-matrix and genomic G-matrix for single-step GBLUP. Relevant for breeding populations where pedigree information supplements genomic data. From AGHmatrix `Hmatrix`.

#### Benchmark targets

- **AGHmatrix**: Benchmark all GRM methods above against AGHmatrix v2.1.4. VanRaden GRM should match exactly; Slater, Endelman, Vitezica, Su, Yang methods should agree to 4th decimal.
- **polyBreedR**: Benchmark digenic dominance GRM for tetraploids (polyBreedR implements Endelman's method independently).
- **GWASpoly kinship**: Already benchmarked — TorchGWAS VanRaden matches perfectly (r=1.000, Section 13b).

### 9e. Multi-trait polyploid GWAS (novel)

To our knowledge, no existing tool supports multi-trait GWAS for polyploids. TorchGWAS extends the mvLMM (Section 4b)
to polyploid dosage by:
- Using polyploid-encoded genotype tensors (dosage 0..k) in the EED-rotated scan.
- Computing the polyploid GRM for the Kronecker covariance structure.
- Testing all gene-action models across all traits jointly.

To our knowledge, this would be a novel scientific contribution: multi-trait polyploid GWAS with GPU acceleration.

**Categorical-trait extension (Post-V1, Phase 16)**: Combined with the multi-trait threshold-linear model (Section 5b, Bermann et al. 2026), TorchGWAS can extend polyploid GWAS to categorical traits — disease resistance scores in tetraploid potato, lodging resistance in hexaploid wheat, survival traits in polyploid aquaculture species. The threshold-linear model's liability framework operates on the same polyploid GRM and gene-action models, requiring only that the NR/EM pseudo-observations replace continuous phenotypes with liability estimates for categorical traits. To our knowledge, no existing tool supports categorical-trait GWAS for polyploid organisms.

### 9f. Polyploid-specific preprocessing

- **Dosage calling quality**: For polyploids, dosage calling is less certain than for diploids.
  TorchGWAS accepts probability vectors P(d=0), P(d=1), ..., P(d=k) per sample-SNP and
  computes expected dosage E[d] = sum(i * P(d=i)) for the scan, while retaining full
  probabilities for imputation quality assessment.
- **Allele frequency estimation**: p = mean(dosage) / k (not /2 as in diploids).
- **MAF filter**: MAF = min(p, 1-p), same definition but computed from polyploid frequencies.
- **Hardy-Weinberg test**: Generalized HW for polyploids (Levene 1949; extended by De Silva et al. 2005).

### 9g. Polyploid QTL detection, model options, and best-model selection

GWASpoly implements a structured post-GWAS workflow for polyploids that TorchGWAS should replicate:

#### Post-encoding genotype frequency filter (max.geno.freq)

After applying gene-action encoding, filter markers where the most common encoded genotype class exceeds a frequency threshold (default: `1 - 5/N` where N is sample size). This is critical for polyploid gene-action models because encoding can collapse multiple dosage classes into one — e.g., under `1-dom` encoding, a marker with MAF > 5/N may become nearly monomorphic (all 1's) after the `dose >= 1` threshold. Without this filter, such markers inflate false positives. GWASpoly applies this via the `geno.freq` parameter in `set.params()`.

**Implementation**: Apply after `recode_gene_action()`, before the scan. For multi-column models (general, diplo-general), check max frequency of each column independently.

#### P3D toggle (per-marker variance re-estimation)

GWASpoly supports P3D=TRUE (default) and P3D=FALSE modes:
- **P3D=TRUE**: Estimate variance components (σ²_g, σ²_e) once from the null model, then scan all markers with fixed variance components. This is the standard approach and matches TorchGWAS's current behavior.
- **P3D=FALSE**: Re-estimate variance components for every marker tested. Slower (n_markers × REML fits) but more accurate when individual markers explain large fractions of trait variance. Relevant for oligogenic traits in polyploids.

TorchGWAS should support both via a `p3d: bool = True` parameter on the scan method.

#### Marker effect estimates in scan output

GWASpoly reports marker effect estimates (betas) alongside p-values. TorchGWAS's `ScanResult` already includes `beta` and `se` fields for Wald and LRT tests, but these are NaN for score tests. Ensure:
1. Wald test always populates `beta` and `se` (already implemented)
2. For polyploid multi-column models (general, diplo-general), report per-column betas as a vector
3. Score test output should clearly indicate that betas are unavailable (NaN), not zero

#### Best-model selection per SNP

After scanning all gene-action models, select the best model per SNP by:
1. For each marker, compare -log10(p) across all models tested
2. Report the model with the highest score alongside additive results
3. Apply BIC penalty for multi-column models (general, diplo-general) vs single-column (additive, j-dom)

This is critical because the true gene action varies by locus — a trait may be simplex-dominant at one QTL and additive at another.

#### LD-window peak pruning (from GWASpoly `get.QTL`)

After significance thresholding, merge nearby significant markers into QTL peaks:
1. Within each chromosome, sort significant markers by score (descending)
2. For each marker, check if any higher-scoring marker exists within `bp.window` (default: 1 Mb)
3. If so, discard the lower-scoring marker (it's in LD with the peak)
4. Report only independent QTL peaks with their best gene-action model

This prevents reporting multiple markers in LD as separate QTL.

#### Effective number of tests for polyploid (M.eff)

TorchGWAS should support two M.eff algorithms for polyploid multiple-testing correction:

**1. Li & Ji (2005) spectral M.eff** (already planned via `simpleM`):
1. Compute r² (LD) matrix per chromosome from the genotype matrix
2. Eigendecompose each per-chromosome r² matrix
3. Estimate effective number of independent tests: M_eff = sum over eigenvalues of f(lambda)
4. Use M_eff instead of total marker count for Bonferroni correction

**2. Moskvina & Schmidt (2008) M.eff** (used by GWASpoly):
1. Same eigendecomposition of the per-chromosome LD correlation matrix
2. Different aggregation: M_eff = sum_i I(lambda_i >= 1) + sum_i (lambda_i - floor(lambda_i)) for lambda_i < 1
3. Generally produces a slightly smaller M_eff than Li & Ji, giving marginally more power

Both should be available; Moskvina-Schmidt should be the default for polyploid analyses to match GWASpoly behavior. Ensure polyploid-aware r² computation (using recoded genotypes from the gene-action model) is supported.

#### Multi-QTL joint model fitting (from GWASpoly `fit.QTL`)

After identifying independent QTL peaks:
1. Fit a joint mixed model with all QTL simultaneously as fixed effects
2. Use the Z incidence matrix for multi-env data
3. Report per-QTL: R² (variance explained), LRT p-value, effect size, gene-action model
4. Account for confounding between linked QTL
5. Backward elimination: iteratively drop non-significant QTL (LRT p > threshold)

This is equivalent to a backward elimination / forward selection procedure and is more statistically rigorous than reporting marginal p-values.

#### Polyploid visualization extensions

- **Per-chromosome QQ plots**: Generate separate QQ plots for each chromosome to diagnose chromosome-specific inflation (e.g., from population structure or large-effect QTL on a specific chromosome). GWASpoly supports this via `qq.plot(data, chrom=...)`.
- **LD decay plot**: Plot pairwise r² vs physical distance with a shape-constrained spline fit (monotone decreasing). Useful for determining appropriate `bp.window` for peak pruning. Should support both raw dosage and gene-action-encoded r² computation.

### 9h. Polyploid GWAS benchmarking targets

| Reference tool | Version | Benchmark | Status |
|---------------|---------|-----------|--------|
| **GWASpoly** | v2.14 | P-values for additive, j-dom, diplo-additive on potato tetraploid | **DONE** (Section 13b): r > 0.999 for matching models |
| **AGHmatrix** | v2.1.4 | GRM methods: VanRaden, Slater, Endelman, Yang for polyploid | Pending: implement methods, benchmark element-wise |
| **polyBreedR** | latest | Digenic dominance GRM, epistatic Hadamard GRM, GIBD kinship | Pending: implement methods, cross-validate |
| **GAPIT3** | v3 | Additive-only baseline (GAPIT treats dosage as continuous, no ploidy-aware normalization) | Pending: easy cross-check |

**GAPIT3 polyploid note**: GAPIT3 can accept polyploid dosage as numeric input, but it does NOT have ploidy-aware GRM normalization (uses diploid `2p(1-p)` regardless), no gene-action models, and no polyploid-specific QC. It is useful only as an additive-model baseline. TorchGWAS should match GAPIT3's additive p-values when using the same (diploid-style) GRM normalization.

## 10. Architectural Data Flow

```
Raw Data Input (PLINK/PLINK2/VCF/BGEN/HapMap/Zarr/CSV)
    |
    v
Format Detection & Validation (auto-detect, pre-flight checks, ploidy detection)
    |
    v
Imputation & Phasing [optional] (built-in mean/KNN or external BEAGLE/IMPUTE5)
    |
    v
QC & Preprocessing (MAF filter, HWE, call rate, missingness, sample alignment)
    |
    v
Genotype Encoding (diploid 0/1/2 or polyploid 0..k; gene-action model recoding)
    |
    v
Standardization (center, scale, ploidy-aware allele frequency)
    |
    v
GRM Construction (streaming X @ X.T / p, AMP-enabled, polyploid-aware)
    |
    v
Eigendecomposition (K = U diag(evals) U^T)
    |
    v
Rotation (Y_rot = U^T Y, X_rot = U^T X)
    |
    v
Null Model Fitting (REML: PX-EM -> AI-REML / LBFGS-autograd -> MM fallback)
    |
    v
Scan Loop (stream genotype chunks -> rotate -> batched score/wald/lrt)
    |
    v
Multiple Testing Correction (Bonferroni, BH, simpleM, permutation, weighted FDR)
    |
    v
Statistics & Reporting (p-values, q-values, Manhattan/QQ plots, TSV/Parquet)
```

## 11. Module Architecture

```
torchgwas/
  __init__.py
  config.py                     # dtype policy (FP64 stats, FP32/16 IO), AMP guards, eps
  
  io/
    base.py                     # GenotypeReader protocol: yields (G_chunk, variant_meta)
    detect.py                   # Auto-detect format from magic bytes + extension
    validate.py                 # Pre-flight validation: sample counts, allele coding, ploidy
    convert.py                  # Format conversion utilities
    plink.py                    # .bed/.bim/.fam memmap reader, SNP-major chunk iterator
    plink2.py                   # .pgen/.pvar/.psam reader (dosage + phased support)
    vcf.py                      # VCF/BCF streaming reader (GT and DS fields)
    bgen.py                     # BGEN reader (.bgen + .sample + .bgi index)
    hapmap.py                   # HapMap format reader (letter genotype decoding, IUPAC, polyploid integer)
    numeric.py                  # CSV/TSV/TXT numeric dosage reader (auto-detect orientation + delimiter)
    map_file.py                 # Companion map file reader (SNP, Chr, Pos); auto-discovery
    zarr.py                     # Zarr genotype block reader (cloud/object-store ready)
    hdf5.py                     # HDF5 genotype reader
    phenotype.py                # phenotype/covariate loaders, sample alignment

  preprocess/
    standardize.py              # center, variance-scale, MAF filter (ploidy-aware)
    impute.py                   # mean/mode/KNN/LD-based imputation (built-in)
    impute_external.py          # wrappers for BEAGLE, IMPUTE5, Minimac4, STITCH
    phase.py                    # phasing wrapper (BEAGLE) + haplotype tensor storage
    covariates.py               # intercept, PCs, batch effects
    polyploid.py                # dosage encoding, gene-action model recoding, ploidy detection
    qc.py                       # HWE test (diploid + polyploid), call rate, het excess

  linalg/
    kinship.py                  # GRM builders: streaming GPU GEMM, VanRaden (diploid + polyploid)
    kinship_polyploid.py        # Polyploid-specific GRM: per-gene-action-model, LOCO
    eigh.py                     # exact eigendecomp, LOBPCG, randomized SVD
    woodbury.py                 # low-rank update helpers for Woodbury identity
    batched.py                  # batched Cholesky/solve for (n,d,d) blocks
    safe.py                     # safe_cholesky with adaptive jitter, logdet, symmetrize

  models/
    base.py                     # BaseModel protocol: fit_null(), score_chunk()
    glm.py                      # GLM batched scan via residualization
    lmm_single.py               # SingleTraitLMM: AI-REML, score/wald/lrt
    lmm_multi.py                # MultiTraitLMM: EED rotation, Kronecker structure
    lmm_multi_fit.py            # mvLMM null fitting: LBFGS-autograd + AI-REML
    iterative.py                # shared framework for multi-locus iterative loops
    farmcpu.py                  # FEM/REM iteration, binning, pseudo-QTN
    blink.py                    # LD clustering + BIC, no-kinship REM
    # --- Novel models (post-V1) ---
    lmm_multikernel.py          # Multi-kernel LMM (additive + dominance + epistasis)
    lmm_gxe.py                  # Gene-environment interaction LMM
    set_based.py                # SKAT / Burden / SKAT-O under LMM
    bayesian_vs.py              # Bayesian variable selection (spike-and-slab)
    threshold_linear.py         # Multi-trait threshold-linear model (NR/EM MAP; Bermann et al. 2026)

  scan/
    unified.py                  # UnifiedScanner: reader -> model adapter -> stats -> output
    adapters.py                 # adapters making all models look identical to scanner
    strategies.py               # null-VC-fixed, per-SNP-refit, hybrid heuristics

  stats/
    tests.py                    # chi2_sf, t_sf via torch.special; Wald/LRT/Score dispatch
    multipletesting.py          # Bonferroni, Sidak, Holm, BH, BY, Storey q-value
    permutation.py              # GPU-accelerated maxT permutation testing
    simplem.py                  # simpleM / M_eff: spectral decomposition for effective test count
    weighted_fdr.py             # Weighted BH, adaptive FDR, local FDR
    cauchy.py                   # Cauchy combination test for multi-method/multi-trait p-values
    genomic_control.py          # lambda_GC computation and inflation diagnostics
    plots.py                    # Manhattan, QQ (matplotlib)
    calibrate.py                # p-value calibration checks vs reference tools

  optim/
    controller.py               # Optimizer controller: auto-fallback A->B/C->D->F
    ai_reml.py                  # Mode B: Average Information REML (single-trait 2-param, multi-trait)
    lbfgs_reml.py               # Mode C: LBFGS on autograd log-likelihood (Cholesky param)
    em_warmstart.py             # Mode A: PX-EM / EM-REML warm-start (3-10 iters)
    mm_reml.py                  # Mode D: MM (Minorization-Maximization) algorithm
    fisher_scoring.py           # Mode D alt: Fisher scoring with expected information
    pcg_solver.py               # Mode E: PCG iterative solver for large-scale REML

  cli.py                        # Entry point: glm-scan, lmm-scan, mvlmm-scan subcommands

  results/
    tables.py                   # result assembly, top hits, clumping
    export.py                   # TSV, Parquet, JSON metadata

tests/
  golden/                       # GEMMA/GAPIT reference datasets + expected outputs
  test_glm.py
  test_lmm_single.py
  test_lmm_multi.py
  test_farmcpu.py
  test_blink.py
  test_io_plink.py
  test_grm.py
  test_scan_unified.py
  test_golden_gemma.py          # 4th-decimal p-value regression tests
  test_golden_gapit.py
```

## 12. BaseModel Contract

Every model must implement:

```python
class BaseModel(Protocol):
    def fit_null(self, Y, X0, K=None, **kwargs) -> NullFit:
        """Fit null model (no SNP effect). Return fitted variance components + cached transforms."""
        ...

    def score_chunk(self, G_chunk: Tensor, null_fit: NullFit) -> ScanResult:
        """Score a genotype chunk (n, m). Return effect sizes, SEs, test stats, p-values per SNP."""
        ...
```

### ScanResult schema (per-variant output, consistent across all models)

| Field | Shape | Description |
|---|---|---|
| `chr` | (m,) | Chromosome |
| `pos` | (m,) | Base-pair position |
| `snp` | (m,) | Variant ID |
| `a1` / `a2` | (m,) | Alleles |
| `af` | (m,) | Allele frequency |
| `beta` | (m,) or (m,d) | Effect size(s) — NaN for score test |
| `se` | (m,) or (m,d) | Standard error(s) — NaN for score test |
| `stat` | (m,) | Test statistic (chi-square) |
| `p` | (m,) | P-value |
| `test` | str | "score", "wald", or "lrt" |

## 13. Optimizer Stack and Numerical Policy

A single optimizer is not enough. The optimizer stack is part of the charter and must not be
simplified to a single engine unless explicitly approved. This layered strategy is adapted from
the ASReml-class mixed model charter (which uses the same principle for breeding models) and
extended with PyTorch-specific autograd capabilities.

| Mode | When used | Method | Rationale |
|---|---|---|---|
| **A. Warm start** | First 3-10 iterations to stabilize weak initial values and boundary cases | PX-EM (Parameter-Expanded EM) or EM-REML | Monotonic convergence guaranteed; keeps parameters in valid space. PX-EM converges faster than plain EM (Foulley & Van Dyk 2000). |
| **B. Primary exact (small d)** | After warm start, for d <= ~6 traits | AI-REML (Average Information) with damped Newton updates, backtracking line-search for positivity | Standard in GEMMA and ASReml. Quadratic convergence near optimum. AI matrix is cheaper than full Fisher information. |
| **C. Primary robust (general)** | Default path for any d, or when B is unavailable | LBFGS on autograd REML log-likelihood, Cholesky parameterization (SPD guaranteed by construction) | Novel to TorchGWAS: `torch.autograd` computes exact gradients — no hand-derived score equations needed. Cholesky factors ensure Vg, Ve stay positive definite. Scales to large d without implementing AI matrix. |
| **D. Robust fallback** | If B or C oscillates or repeatedly hits constraints | MM algorithm (Minorization-Maximization) or Fisher scoring | MM is monotone and globally convergent (Zhou et al. 2019, PLoS Comp Bio). Prioritized for models with many variance components. Fisher scoring uses expected information — more stable than observed information when the model is misspecified. |
| **E. Large-scale iterative** | When direct solves become too expensive (n > 100K with multiple random effects) | Augmented AI-REML with PCG (Preconditioned Conjugate Gradient) for the linear system solve, instead of direct Cholesky | Target model remains exact, but the linear system inside each REML iteration is solved iteratively. Follows BLUPF90 scaling strategy. |
| **F. Rescue / diagnostic** | Last resort when all above fail | Derivative-free optimizer (Nelder-Mead, Powell) or generic box-constrained optimizer (L-BFGS-B with explicit bounds) | Never the primary production engine. Used only for diagnosis or when gradient information is unreliable. |
| **G. TRIAD-REML** | Multi-trait REML when d >= 5, or when Mode C (LBFGS) converges to suboptimal points | Trust-Region Inexact Autograd-Differentiated REML: CG-Steihaug trust-region with exact Hessian-vector products via `torch.autograd`, log-Cholesky parameterization, double eigendecomposition (EED) for O(d³ + n·d²) per iteration | Novel to TorchGWAS. Uses exact second-order curvature (not the Average Information approximation) — no other GWAS tool computes exact Hv products. Trust-region globalization is more robust than line-search for ill-conditioned multi-trait landscapes. 24% closer to GEMMA variance components than Mode C on 5-trait benchmarks. |

### Optimizer controller logic

The optimizer controller should implement automatic fallback:
1. Start with Mode A (PX-EM, 3-10 iterations).
2. Switch to Mode B (AI-REML) if d <= 6, else Mode C (LBFGS-autograd). For d >= 5, Mode G (TRIAD-REML) may be selected via `multi_trait_reml_method="triad"`.
3. If the primary mode oscillates (likelihood decreases) or hits boundary 3+ times, switch to Mode D (MM). Mode G falls back to Mode C (LBFGS) on failure.
4. If Mode D also fails, try Mode F (derivative-free) and flag the fit as "rescue mode."
5. Log the optimizer trajectory (mode switches, likelihood trace) for diagnostics.

### Why this stack matters vs. single-optimizer tools

- **GEMMA** uses only Newton-Raphson on lambda = sig_g^2 / sig_e^2. Works well for single-trait but can fail for multi-trait or boundary cases.
- **ASReml** uses AI-REML as primary with PX-EM warm start — the gold standard for breeding models. TorchGWAS adopts this pattern.
- **SAIGE** uses AI-REML with PCG — TorchGWAS includes this as Mode E for large-scale.
- **TorchGWAS adds Mode C** (LBFGS-autograd) which no other tool has: it lets users define new log-likelihoods and get optimization "for free" via autograd.
- **TorchGWAS adds Mode G** (TRIAD-REML) which goes further: exact Hessian-vector products via autograd double-backward, combined with trust-region globalization. No other GWAS tool uses exact second-order curvature — GEMMA, GCTA, and ASReml all rely on the Average Information approximation, which is inaccurate far from the optimum.

### Numerical policy

- **FP64 mandatory** for: eigendecomposition, REML log-likelihood evaluation, GLS coefficient computation, variance of estimates, test statistics, p-values.
- **FP32 allowed** for: GRM accumulation (with FP64 final cast), genotype chunk matmuls during streaming, I/O decode.
- **FP16/BF16 allowed** for: GRM GEMM inner loop (with FP32 accumulator), genotype standardization staging.
- **Jitter policy**: Add deterministic diagonal jitter (eps = 1e-6 * trace/n) before Cholesky; never invert large matrices explicitly except tiny d x d blocks.
- **Deterministic mode**: Optional `torch.use_deterministic_algorithms(True)` for regression test reproducibility.

### Extension: Threshold-linear model optimizer modes (Post-V1)

The multi-trait threshold-linear model (Section 5b, Bermann et al. 2026) introduces two specialized MAP solvers that integrate into the optimizer stack:

| Mode | Method | Mapping to stack | Key properties |
|---|---|---|---|
| **T-NR** | Newton-Raphson on threshold-linear log posterior | Extension of Mode B (Newton-type) | Modified MME with R̃^{-1} and pseudo-observations ỹ; 7 iterations for 1.8M animals; requires E_T[l_i] and Var_T(l_i) from truncated MVN moments |
| **T-EM** | EM with SQUAREM acceleration | Extension of Mode A (EM-type) | E-step computes expected liabilities, M-step solves standard MME; simpler per-iteration but ~3x more iterations than NR; SQUAREM accelerator halves iteration count |

**Controller logic for threshold models**: Start with T-EM (3-5 iterations for warm start of liabilities), then switch to T-NR. If T-NR fails to converge (rare — see Bermann et al. sensitivity analysis), fall back to T-EM with SQUAREM. Variance components and thresholds are assumed known for V1 of threshold support (estimated via prior Gibbs sampling or REML on continuous subset); autograd-based joint estimation is a future extension.

## 14. Stepwise Roadmap

The phases below are the canonical execution order. A new AI agent must not reorder them.
Exact phases establish truth. GPU acceleration comes after truth is validated. A phase is not done
until its outputs match trusted reference software within declared tolerance.

| Phase | Title | Primary Deliverables | Policy | Done When |
|---|---|---|---|---|
| Phase | Title | Primary Deliverables | Policy | Done When |
|---|---|---|---|---|
| **0** | Architecture freeze | Module skeleton, type contracts, config policy, validation matrix, ploidy-aware types | No implementation beyond skeletons | All protocols, schemas, and test stubs exist |
| **1** | I/O substrate + format intelligence | Auto-detect format (BED/PGEN/VCF/BGEN/HapMap/CSV); PLINK memmap reader; PLINK2 pgen reader; VCF/BGEN readers; format validation and pre-flight checks; phenotype loader with sample alignment | Correctness over speed; all formats must produce numerically equivalent internal tensor representation (within declared tolerances) | Reader outputs match GEMMA's preprocessing on test data; all format readers tested |
| **2** | Preprocessing + QC + imputation | Built-in imputation (mean/mode/KNN/LD-based); external imputation wrappers (BEAGLE, IMPUTE5, Minimac4); phasing wrapper; MAF/HWE/call-rate QC; standardization (ploidy-aware) | Imputation Rsq/DR2 quality metrics tracked; imputed variants tagged in output | Mean-imputed results match GEMMA; BEAGLE wrapper produces correct phased output |
| **3** | GRM + eigendecomposition | Streaming GRM builder (GPU GEMM); polyploid GRM (VanRaden generalization); eigendecomposition; rotation utilities; LOCO support | FP64 for eigen; FP32 GRM accumulation OK; polyploid GRM tested for tetraploid | GRM matches GEMMA's centered relatedness (diploid); polyploid GRM matches GWASpoly |
| **4** | Single-trait LMM (CPU, exact) | AI-REML 2-param optimizer, score/wald/lrt scan, P-operator, full optimizer stack (PX-EM -> AI-REML -> MM fallback) | No GPU optimization yet; no approximate methods | Variance components, beta, SE, p-values match GEMMA within tolerance |
| **5** | Multi-trait mvLMM (CPU, exact) | EED rotation, Kronecker batched likelihood, LBFGS-autograd + AI-REML null fit, Wald/LRT scan | Float64 throughout; match GEMMA mvLMM | mvLMM p-values match GEMMA on canonical multi-trait dataset |
| **6** | GLM (CPU) | Batched GLM via residualization, t-test and F-test | Fast, simple, reference for non-LMM users | GLM results match PLINK --linear on test data |
| **7** | Multi-locus: FarmCPU + BLINK (CPU) | FEM/REM iteration, binning, BIC-LD selection, pseudo-QTN update | Match GAPIT3 FarmCPU/BLINK outputs | Detected QTNs and p-values match GAPIT3 on simulated data |
| **8** | Multiple testing framework | Bonferroni, Sidak, Holm, BH, BY, Storey q-value, simpleM/M_eff, genomic control; all built-in | Standard methods first; novel methods in later phases | BH matches R `p.adjust`; simpleM matches R `simpleM`; lambda_GC correctly computed |
| **9** | Polyploid GWAS | Polyploid genotype encoding (0..k); gene-action models (additive, j-dom, diplo-additive, diplo-general, overdominant); polyploid GRM (VanRaden, Slater, Endelman digenic, Vitezica/Su dominance, Yang, weighted, pseudo-diploid); per-model scan; best-model selection; LD-window QTL peak pruning; M.eff polyploid threshold; multi-QTL joint fitting | Test on tetraploid potato and hexaploid wheat; GRM benchmark against AGHmatrix; additive cross-check against GAPIT3 | Results match GWASpoly (r>0.999, **DONE** for p-values); GRM matches AGHmatrix; multi-model scan works for 4x and 6x |
| **10** | GPU acceleration | Move all scan loops, GRM construction, eigendecomposition to GPU; AMP pipeline; compute/IO overlap; GPU-accelerated permutation testing | Fit semantics must not change; GPU vs CPU outputs must agree within FP tolerance | Same p-values as CPU path; documented speedup on benchmark; 10K permutations demonstrated |
| **11** | CLI + reporting + pipeline | `glm-scan`, `lmm-scan`, `mvlmm-scan`, `poly-scan` subcommands; `impute`, `convert`, `validate`, `pipeline` commands; TSV/Parquet output; Manhattan/QQ plots | User-facing polish | **DONE**: 15 subcommands wired (glm-scan, lmm-scan, mvlmm-scan, poly-scan, mklmm-scan, gxe-scan, set-scan, bayes-scan, met-scan, farmcpu-scan, blink-scan, validate, convert, impute, pipeline); all 9 imputation methods connected; pipeline supports 8 models; 591 tests pass |
| **12** | Streaming + scale | Zarr/HDF5 readers; pinned-memory async prefetch; multi-GPU data-parallel scan | Performance changes only; no new statistical methods | Demonstrated on datasets exceeding system RAM |
| **13** | Approximate backends | Randomized SVD / Nystrom for GRM; stochastic trace estimation; sparse GRM (fastGWA-style); PCG-REML; SparseLMM score test | Approximate fits must be visibly labeled | **DONE**: Approximation error quantified — sparse GRM 0.953 corr with exact; RSVD k=270 0.993 corr; all NullFit objects labeled `approximate=True` |
| **14** | Advanced multiple testing | GPU-batched adaptive permutation; weighted BH (MAF/annotation-aware); Cauchy combination; hierarchical FDR; local FDR | Novel contributions; compare power to standard methods | Weighted BH finds more associations than BH with controlled FDR on benchmark |
| **15** | Novel: Multi-kernel LMM (additive + dominance + epistasis) | Multi-kernel model: V = K_a * sig_a^2 + K_d * sig_d^2 + K_aa * sig_aa^2 + I * sig_e^2; partitioned heritability; per-kernel Wald tests; polyploid multi-kernel | New scientific contribution; validate on simulated truth-known data | Variance components recovered within tolerance on simulated data with known additive/dominance/epistasis |
| **16** | Novel: Gene-environment interaction LMM (GxE-LMM) | Structured GxE model with environment-specific variance components; multi-environment Kronecker structure; interaction scan | New scientific contribution; compare to StructLMM, MAGEE, SPAGxECCT | GxE interaction p-values calibrated on simulated data |
| **17** | Novel: Set-based association tests (SKAT/Burden under LMM) | Gene-level and region-based tests (SKAT, Burden, SKAT-O) integrated with LMM framework; GPU-batched kernel tests; multi-trait support | Extends SAIGE-GENE approach with GPU acceleration | Set-based p-values match SKAT on standard examples |
| **18** | Novel: Bayesian variable selection GWAS (spike-and-slab) | Spike-and-slab prior on SNP effects within LMM; variational inference via autograd; posterior inclusion probabilities (PIPs); **SuSiE** (Sum of Single Effects, IBSS) with per-layer softmax and credible sets; CAVI mean-field baseline | Competes with Quickdraws, BGWAS; autograd makes VI straightforward | **DONE**: PIPs well-calibrated on simulated data and MDP real data; SuSiE + CAVI both validated against GEMMA BSLMM on MDP (279 samples, 3093 SNPs); CLI `--method susie|cavi`; 23 tests pass; 8 benchmarks pass |
| **19** | Novel: GPU-accelerated imputation | Li-and-Stephens HMM forward-backward with batched tensor ops on GPU; deep learning masked autoencoder imputation model | Research contribution; benchmark against BEAGLE 5.5 | **DONE**: Li-Stephens corr=0.906, concordance=0.668 on GWASpoly potato ground truth (958 samples, 9888 markers, ploidy=4); DL imputer corr=0.917 on 200-marker subset; Li-Stephens outperforms mean imputation (0.90 vs 0.83 corr); 31 tests pass; 7 benchmarks pass |
| **20** | Threshold-linear GWAS | Multi-trait threshold-linear MAP (NR/EM) for categorical + continuous traits (Bermann et al. 2026); ssGWAS p-values | Verify NR/EM breeding values match Gibbs (corr >0.99) | Calibrated on simulated threshold data; ssGWAS p-values validated vs BLUPF90 |
| **21** | OCF-LMM (Orthogonal Cross-Fit LMM) | DML-valid inference with cross-fitting + Neyman-orthogonal score; LOCO-K integration | Must pass null calibration at genome-wide alpha | Calibrated chi-squared(1) null; improved power vs standard LMM on semi-synthetic benchmarks |
| **22** | KO-LMM (Knockoff Mixed Model) | Group knockoffs per LD block; LMM-aware importance statistics; FDR control at target q | Knockoff exchangeability documented; FDR verified | Achieved FDR <= target q on null + semi-synthetic; discoveries replicate |
| **23** | Within-family GWAS | Family fixed effects / within-cluster demeaning; attenuation metrics; integration with causal-safety diagnostics | Requires pedigree/sibship data | Within-family effects on UK Biobank sib pairs or simulated families; attenuation flags calibrated |
| **24** | LD-Conditional Score GWAS | Stepwise conditional analysis within LD blocks; marginal + conditional p-values; conditional persistence metric | Pre-specified conditioning sets | Conditional p-values match GCTA-COJO on test data |
| **25** | HetLMM (single-trait G×E) | Random slopes interaction model; 1-df main, 1-df interaction, 2-df joint | Lightweight alternative to full GxE-LMM (Phase 16) | Calibrated on simulated GxE data |
| **26** | DML module (causal inference) | Double/debiased ML for modifiable exposures; cross-fitting; diagnostics | NOT a GWAS model; separate causal-inference extension | Valid coverage of CI; diagnostics functional |
| **38** | Random Regression LMM (longitudinal + spatio-temporal) | `RandomRegressionLMM` (Legendre / B-spline / projection mode + stacked verification mode); `K_coef` structures unstructured / diagonal / fa(k) / +PE block; five test types per SNP (joint / intercept / slope / time-varying / per-time-point β(t)); `SpatioTemporalRR` 2D tensor-product P-spline field-trial smoother; `rr-scan` CLI (Section 12n) | First GPU-batched longitudinal GWAS scanner (to our knowledge) | **DONE**: 97 RR tests pass; cross-validated against BLUPF90+ on simulated longitudinal data (Legendre b=3 TorchGWAS↔BLUPF90 rel=0.078; B-spline b=5 rel=0.162); `SpatioTemporalRR` validated against R `sommer` (corr=0.994 on simulated field) |
| **39** | Random Regression × Multi-Environment LMM (RR-MET) | `RandomRegressionMultiEnvLMM` thin composition over Phase 25 MT-MET, separable Kronecker `Vg = K_coef ⊗ Vg_env` default + unstructured (`bE ≤ 12` guard) + `fa(k)` + `diagonal_coef`; pooled-time correctness invariant; 9 RR-aware contrast tests (joint / per-env / intercept-per-env / time-varying-per-env / stable-per-coef / joint stable-vs-GxE / mean-curve / intercept-only-GxE / per-time × per-env); time-domain interpretive surfaces; `update_null` warm-start; `rr-met-scan` CLI (Section 12o) | First GPU-batched longitudinal × multi-environment GWAS scanner (to our knowledge) | **DONE**: 36 RR-MET tests pass (1350 total); cross-validated against BLUPF90+ multi-trait RR with cookbook init (n=400, E=2): TorchGWAS K_coef rel=0.051 vs BLUPF90 0.852 (~17×), Vg_env rel=0.101 vs 0.276 (~2.7×); FA mode validated against BLUPF90+ XFA(2) at n=300, E=4: TorchGWAS Vg_full rel=0.241 vs BLUPF90 0.599 (~2.5×) |
| **27** | Genotype-Uncertainty LMM (GU-LMM) | `GULM` — dosage-variance–corrected score test for GP-aware GWAS; diagonal V_dosage correction; `GUResult` output | Extends score test for imputation uncertainty | **DONE**: 17 tests pass |
| **28** | Leave-Region-Out LMM (LRO-LMM) | `LROLMM` — block-level LOCO for proximal decontamination; per-block kinship subtraction; `LROResult` output | Finer-grained LOCO than per-chromosome | **DONE**: 17 tests pass |
| **29** | Adaptive FDR Control | IHW (Ignatiadis & Huber 2021) covariate-adaptive weighting + AdaPT (Lei & Fithian 2018) covariate-adaptive thresholding | Novel multiple testing beyond standard BH/BY | **DONE**: 21 tests pass |
| **30** | GLM Family (Binary/Ordinal/Multinomial) | `BinaryGLM`, `OrdinalGLM`, `MultinomialGLM` — score tests + SPA + Firth correction; no random effects | Platform scope for categorical traits | **DONE**: 46 tests pass |
| **31** | GLMM (Binary/Ordinal with random effects) | `BinaryGLMM`, `OrdinalGLMM` — PQL null fitting (SAIGE-style); score test + SPA | Extends GLM with kinship correction | **DONE**: 20 tests pass |
| **32** | Multinomial GLMM + Integration | `MultinomialGLMM` — PQL for multi-class with random effects; cross-model integration tests | Complete categorical GLMM family | **DONE**: 22 tests pass |
| **33** | Multi-Environment GLMM | `MultiEnvGLMM` — binary/ordinal categorical phenotypes across E environments; structured genetic covariance Σ_g; multi-env PQL; joint/per-env/homogeneity/reaction-norm score tests; per-env SPA | Multi-env extension of GLMM | **DONE**: 22 tests pass |
| **34** | Survival GWAS | `SurvivalGLMM` — Cox PH frailty model; Breslow-Clayton PQL; martingale residual score test; SPACox-style SPA; polyploid support | Time-to-event phenotypes | **DONE**: 25 tests pass |
| **35** | MT-MET Scaling | Kronecker EED diagonal precision for d=50/E=20; score test tiered dispatch | Scale MT-MET to large d×E | **DONE**: 23 tests pass |
| **36** | Post-GWAS (LDSC, Meta-analysis, Clumping) | `torchgwas.postgwas` — LDSC h²/rg, meta-analysis (IVW/DL/Stouffer/RE2), LD clumping | Essential downstream analysis | **DONE**: 30 tests pass |
| **40** | Polygenic Score (PGS) Construction | `torchgwas.pgs` — C+T, LDpred2 (Inf/Grid/Auto), PRS-CS; allele harmonization; `score_individuals`; MCMC diagnostics; `pgs-fit` + `pgs-score` CLI | Tier 1 + Tier 2 PGS methods | **DONE**: 75 tests pass |
| **41a-41z** | Native C++ Accelerators | 26 pybind11 C++ extensions for hot loops (PRS-CS, LDpred2, C+T, PELT, ESS, Gabriel, Big-LD, DP-optimize, CC-graph, GWAS-aligned, Spine, LD-decay-signal, greedy-MWIS, uncertainty-blocks, Wall-Pritchard, impute-mode/knn/ld, CAVI, SPA, LDSC-jackknife, HWE/HWE-DR, cross-pop, postgwas-clump); OpenMP parallelization for 6 kernels; dispatch helper `select_path`; GPU imputation kernels; 12/24 kernels show >50× speedup | Performance acceleration with Python fallback | **DONE**: ~250 native tests pass; all toggleable via `TORCHGWAS_DISABLE_NATIVE=1` |
| **42** | S-LDSC + Colocalization | `torchgwas.postgwas._sldsc` (partitioned LDSC, Finucane 2015) + `_hyprcoloc` (multi-trait coloc, Foley 2021) + classical `coloc_pairwise` (Giambartolomei 2014) | Functional enrichment + causal variant sharing | **DONE**: 17 tests pass |
| **43** | Visualization + PVE | `torchgwas.viz` — Manhattan, Circos Manhattan, Miami, QQ, Haploview LD triangle; `torchgwas.stats.pve` — marginal + joint PVE | Pub-quality plots + variance explained | **DONE**: 37 tests pass |
| **44** | Post-GWAS Extensions | MR (IVW/Egger/weighted-median/PRESSO), gene-set enrichment (MAGMA-style), fine-mapping utilities, PGS validation, multi-ancestry meta-analysis (MR-MEGA + MANTRA) | Comprehensive downstream analysis | **DONE**: 67 tests pass |
| **45** | Post-GWAS Extensions II | Power analysis (Sham & Purcell 2014), winner's curse correction (conditional likelihood/FIQT/bootstrap), trumpet plot, SMR/HEIDI, TWAS (S-PrediXcan/PrediXcan), HESS regional h² | Advanced downstream + visualization | **DONE**: 67 tests pass |
| **46** | Haplotype-Based GWAS | `HaplotypeGWAS` — HTR F-test/LRT, block-based scan (13 LD block methods), moving-window, haplotype-SKAT; multi-locus EM; LMM correction; 5 novel methods (PCHT, HHCT, HSKAT, HapGxE, BayesHap) in `haplotype_novel.py` | Haplotype-level association testing | **DONE**: 48 tests pass |
| **47** | Multi-Env/Multi-Trait Haplotype GWAS | `HaplotypeMultiEnvGWAS`, `HaplotypeMultiTraitGWAS`, `HaplotypeMTMETGWAS` — thin composition over existing null-fitting; per-block GLS Wald; contrast algebra for joint/per-hap/per-env/per-trait/GxE tests | Multi-dimensional haplotype GWAS | **DONE**: 33 tests pass; 2102 total |

### Stage Gates (scope control, must be passed in order)

- **Gate A (Parity Core)**: Gaussian GLM + univariate LMM parity vs GEMMA/GAPIT; strict alignment/QC; TSV+Parquet output; golden tests. Encompasses Phases 0-7.
- **Gate B (mvLMM Parity)**: mvLMM EED likelihood + AI-REML + df=d/df=1 tests; golden tests vs GEMMA mvLMM. Encompasses Phase 5.
- **Gate C (Interpretation Safety)**: Covariate Audit Mode + Causal-Safety diagnostics (LD blocks + stability metrics + within-family attenuation). First deployment of Section 6j-6k.
- **Gate D (First "Outperform" Claim)**: At least one novel model (recommended: LDJ-LMM or OCF-LMM) demonstrates improved power/robustness on the benchmark suite (Section 16a) with matched calibration. No "outperforms" claim permitted before this gate passes.

## 15. What Each Phase Must Contain

### Phase 0 — Architecture freeze
- Write module skeletons with Protocol/ABC definitions
- Write config.py with dtype policy and AMP guards
- Write ScanResult and NullFit dataclass schemas
- Write validation matrix: which datasets, which tools, which tolerances
- Write test stubs for every subsequent phase

### Phase 1 — I/O substrate + format intelligence
- Auto-detect format from magic bytes and file extension
- PLINK .bed reader: memmap, SNP-major, byte-decode to dosage tensor
- PLINK2 .pgen reader, VCF/BCF reader, BGEN reader, HapMap reader, CSV dosage reader
- Pre-flight validation: sample count consistency, allele coding, chromosome naming, duplicate detection
- Phenotype/covariate loader with deterministic sample alignment: three-way ID intersection, lexicographic sort, hard errors on empty intersection or >50% drop, alignment manifest TSV output
- Format conversion utilities (VCF->BED, BGEN->Zarr, etc.)
- Verify: sample order, allele coding, and missing-value handling match GEMMA conventions across all formats; alignment manifest is correct and reproducible

### Phase 2 — Preprocessing + QC + imputation
- Built-in imputation: mean, mode, KNN (using GRM), LD-based sliding window
- External imputation wrappers: BEAGLE 5.5, IMPUTE5, Minimac4, STITCH
- Phasing wrapper for BEAGLE (haplotype output as (n, 2, p) tensor)
- QC filters: MAF, missingness, HWE (diploid + polyploid), heterozygosity excess, imputation Rsq — all CLI-configurable with GEMMA/GAPIT-matching defaults
- Variant QC Parquet output: per-variant AF, MAF, HET, HET_EXP, HWE_P, MISS_RATE, IMP_RSQ, FILTER_PASS, FILTER_REASON — written before scan for QA auditing
- Standardization: center, optional variance-scale (ploidy-aware: divide by k not 2)
- Imputation quality tracking: Rsq/DR2/INFO per variant; post-imputation filtering
- Verify: mean-imputed output matches GEMMA; allele frequencies in Parquet match GEMMA .log.txt and GAPIT frequency output; BEAGLE wrapper round-trips correctly

### Phase 3 — GRM + eigendecomposition
- Streaming GRM: accumulate X @ X.T / p in chunks (FP32 GEMM, FP64 accumulator)
- Polyploid GRM: VanRaden generalization with ploidy-appropriate normalization
- Gene-action-model-specific GRMs for polyploid (recode X via f(d) before GRM)
- LOCO (Leave-One-Chromosome-Out) GRM support
- Eigendecomposition: `torch.linalg.eigh` with negative-eigenvalue clamping
- Rotation utilities: U^T Y, U^T X, U^T G
- Verify: K matches GEMMA's centered relatedness (diploid); polyploid GRM matches GWASpoly

### Phases 4-7 — Exact model buildout
- Phase 4 proves single-trait LMM correctness (the core statistical engine) with full optimizer stack
- Phase 5 extends to multi-trait with Kronecker/EED machinery
- Phase 6 adds GLM for completeness
- Phase 7 adds iterative multi-locus models (FarmCPU, BLINK)

### Phase 8 — Multiple testing framework
- Implement all V1 methods: Bonferroni, Sidak, Holm, BH, BY, Storey q-value
- simpleM / M_eff: spectral decomposition of LD correlation matrix for effective test count
- Genomic control (lambda_GC) with automatic inflation warnings
- Verify: BH q-values match R `p.adjust(method="BH")`; simpleM matches R `simpleM` package

### Phase 9 — Polyploid GWAS
- Polyploid genotype encoding (dosage 0..k for arbitrary ploidy k)
- All gene-action models: general, additive, 1-dom through (k-1)-dom, diplo-additive, diplo-general, overdominant
- Polyploid GRM and LOCO; per-gene-action-model scan
- Additional GRM methods: Slater, Endelman digenic (ploidy=4), Vitezica dominance, Su dominance, Yang GCTA-style, weighted GRM, pseudo-diploid, ASV transformation, epistatic Hadamard (G#G, D#D, D#G), GIBD (Section 9d). Benchmark against AGHmatrix v2.1.4 and polyBreedR.
- Post-encoding genotype frequency filter (max.geno.freq) — critical for polyploid gene-action models (Section 9g)
- P3D=FALSE mode: per-marker variance component re-estimation option
- Best-model selection with BIC across gene-action models per SNP
- LD-window peak pruning for QTL detection (bp.window parameter, GWASpoly `get.QTL` equivalent)
- M.eff polyploid-aware significance threshold: both Li & Ji (2005) and Moskvina-Schmidt (2008) algorithms
- Multi-QTL joint model fitting with per-QTL R² and LRT (GWASpoly `fit.QTL` equivalent)
- Per-chromosome QQ plots; LD decay plot with shape-constrained spline
- Multi-trait polyploid mvLMM (novel: to our knowledge, no existing tool)
- Verify: match GWASpoly on published tetraploid potato dataset; hexaploid wheat tested
- Verify: GRM methods match AGHmatrix to 4th decimal on potato data
- Verify: additive-model p-values cross-check against GAPIT3 with dosage input
- **DONE**: GWASpoly p-value benchmark passed — r > 0.999 for additive/1-dom/2-dom/3-dom on potato (957 genotypes, 9888 markers, 6 envs). See Section 13b. Golden tests: `tests/test_golden_gwaspoly.py` (6/6 pass).

### Phase 10 — GPU acceleration
- Profile CPU bottlenecks; move hot paths to GPU
- Implement AMP context managers around GRM and genotype streaming
- Implement async prefetch: CPU loads next chunk while GPU computes current
- GPU-accelerated permutation testing (maxT with batched scan)
- Verify: GPU path produces p-values within declared tolerance (≤1e-4 relative error) of CPU path

### Phases 11-13 — CLI, streaming, approximate
- Phase 11 delivers full CLI with `pipeline` command chaining impute->scan->correct
- Phase 12 enables datasets larger than RAM via streaming backends (Zarr/HDF5 readers, streaming GRM, async prefetch, PCG solver)
- Phase 13 adds approximate methods for biobank-scale data. **Recommended production path**: sparse GRM + PCG-REML (`--approx-method sparse`), achieving 0.953 correlation with exact methods while avoiding O(n³) eigendecomposition. Also provides randomized SVD, LOBPCG, and Nystrom alternatives for truncated spectral approximation.

### Phases 14-20 — Novel scientific contributions (established extensions)
- Phase 14: Advanced multiple testing (weighted BH, adaptive permutation, Cauchy, hierarchical FDR)
- Phase 15: Multi-kernel LMM (additive + dominance + epistasis)
- Phase 16: GxE-LMM (gene-environment interaction; multi-trait Kronecker + single-trait HetLMM)
- Phase 17: Set-based association tests (SKAT/Burden/SKAT-O under LMM + LDJ-LMM block-level extension)
- Phase 18: Bayesian variable selection GWAS (spike-and-slab VI via autograd + CSMM debiased path). **DONE**: Two inference methods — SuSiE (IBSS with L softmax layers, per-layer credible sets, auto-scaling prior) and CAVI (mean-field with sigmoid inclusion). CLI `--method susie|cavi --n-signals L`. Validated against GEMMA BSLMM on MDP real data (SuSiE PIP=0.17, CAVI PIP=0.53, GEMMA BSLMM PIP=0.37 on top SNP). 23 tests, 8 benchmarks (including three-way real-data comparison).
- Phase 19: GPU-accelerated imputation (Li-and-Stephens HMM on GPU; deep learning imputation). **DONE**: Li-Stephens genotype-level HMM with log-space forward-backward, Laplace-smoothed transitions, windowed processing, batched samples. Deep learning masked autoencoder. Ground-truth validation on GWASpoly potato (958 samples, 9888 markers, ploidy=4, zero original missingness): Li-Stephens corr=0.906, concordance=0.668; DL corr=0.917 on subset; both outperform mean imputation. CLI `--method li-stephens|deep-learning --ploidy K`. 31 tests, 7 benchmarks.
- Phase 20: Multi-trait threshold-linear models for categorical GWAS (Bermann et al. 2026)
  - NR MAP solver with GPU-parallel per-individual Δ_i, Γ_i, R̃_i^{-1} computations
  - EM MAP solver with SQUAREM acceleration
  - GPU-batched multivariate truncated normal moments (Manjunath & Wilhelm 2021)
  - ssGWAS p-values for categorical traits via posterior variance extraction
  - Polyploid categorical-trait extension (disease resistance, lodging, survival in polyploids)
  - Verify: NR/EM breeding values match Gibbs sampling (correlation >0.99); ssGWAS p-values validated against BLUPF90 on simulated threshold data

### Phases 21-26 — Novel inference models (Guardian charter innovations)
- Phase 21: OCF-LMM (DML-valid cross-fit GWAS; formalizes REGENIE-style residualization with calibrated inference)
- Phase 22: KO-LMM (group knockoff FDR-controlled GWAS under LMM)
- Phase 23: Within-family / family-aware GWAS (causal credibility via family fixed effects)
- Phase 24: LD-conditional score GWAS (stepwise conditional analysis with persistence metrics)
- Phase 25: HetLMM single-trait interaction scan (lightweight G×E; complements Phase 16 full GxE-LMM)
- Phase 26: DML module for modifiable exposures (causal inference extension, not GWAS)

## 16. Validation Gates and Definition of Done

No phase is complete until it passes a formal gate. Correctness must be demonstrated, not assumed.

| Validation type | Examples | Reason |
|---|---|---|
| **Golden tests** | Tiny canonical datasets (balanced/unbalanced) with saved GEMMA/GAPIT outputs | Fast truth checks for log-likelihood, beta, SE, variance components, p-values |
| **Reference equivalence** | GEMMA (LMM, mvLMM), GAPIT3 (FarmCPU, BLINK), PLINK (GLM), GWASpoly (polyploid) | Confirms p-values match to 4th decimal where models overlap |
| **Polyploid equivalence** | Tetraploid potato (GWASpoly published dataset), hexaploid wheat | Confirms polyploid gene-action models and GRM produce correct results |
| **Imputation quality** | BEAGLE-imputed vs directly genotyped; Rsq distribution | Confirms imputation pipeline does not introduce systematic error |
| **Format round-trip** | VCF->internal->scan vs BED->internal->scan on same data | Confirms all format readers produce numerically equivalent statistical results (within declared tolerances) |
| **Simulated truth-known** | Synthetic genotype/phenotype with planted QTLs | Checks power, FPR, and identifiability under controlled truth |
| **CPU/GPU equivalence** | Same dataset, CPU vs GPU path | Ensures GPU acceleration doesn't alter statistical results |
| **Stress tests** | Near-singular K, monomorphic SNPs, all-missing columns, rank-deficient X | Ensures jitter, filters, and boundary handling are safe |
| **Performance benchmarks** | Wall-clock time and peak memory on standard datasets (1K, 10K, 100K samples) | Separates correctness from speed claims; tracks regression |

### Numerical tolerances

| Quantity | Tolerance vs reference |
|---|---|
| Log-likelihood | < 1e-4 relative |
| Variance components | < 1e-4 relative |
| Beta (effect size) | < 1e-4 relative |
| SE | < 1e-4 relative |
| P-value | Match to 4th decimal (< 5e-5 absolute for p > 1e-4) |
| GRM elements | < 1e-6 relative |

### 16a. Benchmark Charter (How We Prove "Outperforms")

Novel models (Phases 21+) must demonstrate improvement via a reproducible benchmarking suite. No "outperforms" claim is permitted without passing Gate D.

**Baselines to include**: GEMMA, GAPIT3, BOLT-LMM, fastGWA, SAIGE, REGENIE, KnockoffGWAS (group knockoffs), MTAG (summary-stat multi-trait). Baselines run via wrappers in `bench/runners/` with explicit version capture and configuration logging.

**Dataset tiers** (to avoid self-delusion):
1. **Pure null**: real genotypes; simulated phenotype with no genetic effect
2. **Semi-synthetic**: real genotypes; simulated phenotypes with known QTNs — single causal per locus, allelic heterogeneity (2-5 causal in LD block), polygenic + sparse large effects, stratification + relatedness, heterogeneity/GxE, rare variant/low MAC regimes. **Must stratify results by MAF/MAC bins.**
3. **Real-world replication**: split-sample replication, cross-environment replication (breeding MET), cross-ancestry replication when available

**Metrics** (define "outperform" precisely):
- **Calibration**: empirical type-I error at alpha in {1e-2, 1e-3, 1e-4, 5e-8}; QQ deviation + lambda_GC stratified by MAF/MAC bins
- **Power**: TPR at fixed FDR or fixed genome-wide alpha; recall/precision for causal loci; KO-LMM: achieved FDR vs target q
- **Resolution**: distance to true causal; conditional persistence; credible-set-lite size
- **Robustness**: sensitivity across model families; within-family attenuation; environment/stratum stability
- **Efficiency**: runtime, SNP/s throughput, peak memory, device utilization

**Acceptance gates** (minimum bar for "outperforms" claim):
- Well-calibrated null (no systematic inflation)
- Competitive or improved power at matched calibration
- Clear win in at least one targeted scenario (e.g., OCF-LMM: richer nuisance models; LDJ-LMM: allelic heterogeneity; KO-LMM: FDR control; HetLMM: context-specific loci)

**External baseline runner contract**: Each runner in `bench/runners/` must capture exact command, parameters, versions, input hashes; emit standardized Parquet with schema: `chrom, pos, snp, ref, alt, beta, se, stat, p, wall_time_s, peak_rss_mb, tool, tool_version, command, config_hash`. Schema validators must fail fast on deviation.

**Reserved tier**: When binary/ordinal GLMM is introduced, the suite must include severe case-control imbalance null calibration.

### 16b. API & Artifact Schema Versioning

- Semantic versioning required (vMAJOR.MINOR.PATCH).
- All output artifacts must include: artifact schema version, TorchGWAS version, dependency versions (torch, numpy, IO backends), device info, dtype policy.
- Breaking artifact schema changes require MAJOR version bump.

## 17. Novel Scientific Contributions (Beyond V1 Replication)

These models represent TorchGWAS's contribution to the broader genetics community — not just faster
implementations of existing methods, but new statistical models enabled by the PyTorch-native
architecture (autograd, GPU batching, flexible likelihood specification).

### 12a. Multi-kernel LMM: Additive + Dominance + Epistasis (Phase 11)

**Scientific gap**: Most GWAS tools model only additive genetic effects (y = Xb + u_a + e). But complex
traits are influenced by dominance (d), additive-by-additive epistasis (aa), and higher-order interactions.
Studies show non-additive heritability can exceed additive heritability for some traits (Zhu et al. 2015,
Sci Rep). GEMMA's eigendecomposition trick does not apply to models with more than one random effect
(acknowledged in Zhou & Stephens 2012).

**TorchGWAS model**:
```
y = Xb + u_a + u_d + u_aa + e

u_a  ~ N(0, K_a * sig_a^2)     # additive GRM
u_d  ~ N(0, K_d * sig_d^2)     # dominance GRM (from heterozygote coding)
u_aa ~ N(0, K_a #K_a * sig_aa^2) # epistatic GRM (Hadamard product)
e    ~ N(0, I * sig_e^2)

V = K_a * sig_a^2 + K_d * sig_d^2 + (K_a # K_a) * sig_aa^2 + I * sig_e^2
```

**Why TorchGWAS can do this**: The multi-kernel V cannot be diagonalized by a single eigendecomposition,
which is why GEMMA cannot support it. But TorchGWAS's LBFGS-autograd optimizer (Mode C) doesn't need
the eigendecomposition trick — it optimizes the REML log-likelihood directly via autograd through
`torch.linalg.slogdet` and `torch.linalg.solve`. For n < ~30K this is tractable on GPU; for larger n,
Mode E (PCG iterative solver) scales further.

**Outputs**: Partitioned heritability (h^2_a, h^2_d, h^2_aa), per-kernel SNP effects, improved power
for traits with substantial non-additive architecture.

**Novelty**: No existing GPU-accelerated tool supports multi-kernel GWAS with automatic variance
component estimation. The 2024 Gordon Bell Prize finalist work (Slim et al., SC24) demonstrated
mixed-precision kernel ridge regression for epistasis on the Frontier supercomputer — TorchGWAS
brings a related capability to single-GPU commodity hardware via PyTorch.

### 12b. Gene-Environment Interaction LMM (GxE-LMM) (Phase 17)

**Scientific gap**: Gene-environment interactions (GxE) are critical for understanding why the same
genotype produces different phenotypes in different environments (e.g., crop yields across locations,
drug response across populations). Existing GxE tools are limited:
- **StructLMM** (Nat Genet 2019): Cannot account for sample relatedness.
- **MAGEE**: R/C++ package, no GPU, limited to score tests.
- **SPAGxECCT** (Nat Commun 2025): Fast but uses saddlepoint approximation, single-trait only.
- **LEMMA**: Bayesian whole-genome regression for GxE, CPU-only, slow.
- No tool offers GPU-accelerated multi-trait GxE with structured variance components.

**TorchGWAS model** (multi-environment with Kronecker structure):
```
Y (n x d) = X @ B + G + GxE + E

V = K kron Vg + K kron Vge * diag(env) + I kron Ve
```
For structured environments (e.g., multi-environment trials in breeding):
```
V = K kron Vg + (K # Z_env Z_env.T) kron Vge + I kron Ve
```
Where Z_env encodes environmental groupings.

**Why TorchGWAS can do this**: The Kronecker-structured GxE model extends the mvLMM framework
(Phase 4) with an additional interaction kernel. The LBFGS-autograd path handles the extra variance
components without any new hand-derived derivatives. The GxE interaction test per SNP becomes a
Wald test on the interaction effect vector — batched over SNPs using the same Schur complement
machinery as mvLMM.

**Outputs**: Main genetic effects + GxE interaction effects per SNP, environment-specific effect sizes,
interaction p-values, partitioned V_g vs V_ge variance.

**Single-trait HetLMM specialization**: For single-trait interaction screening (lighter than the full multi-trait Kronecker model), implement `y = Xβ + gγ + (g⊙z)η + u + ε` with 1-df main, 1-df interaction, and 2-df joint tests. In rotated eigenspace this is a 2-column Schur complement — same computational cost as a 2-covariate LMM scan. See also Section 12k.

### 12b2. Multi-Environment Trial (MET) GWAS — **DONE**

**Scientific gap**: Multi-environment trials (MET) are fundamental in plant and animal breeding — genotypes are evaluated across multiple locations, years, or management practices. Existing MET-GWAS tools are limited:
- **GEMMA mvLMM**: Treats environments as pseudo-traits but only supports unstructured Vg covariance. For E > 10 environments, the E(E+1)/2 free parameters become ill-conditioned.
- **ASReml-R**: Supports FA(k) covariance structures but is commercial, CPU-only, and not designed for GWAS scanning.
- **No tool** offers GPU-accelerated MET-GWAS with factor analytic covariance, multi-kernel support, and reaction-norm decomposition.

**TorchGWAS implementation** (three complementary model families):

1. **Per-Environment mvLMM** (default): Y(n x E) = XB + U + E, vec(U) ~ N(0, K kron Vg), vec(E) ~ N(0, I kron Ve). Per-env betas with joint chi-squared(E), homogeneity chi-squared(E-1), and per-env marginal chi-squared(1) tests.

2. **Reaction-Norm Decomposition**: Reparameterizes per-env betas into stable effect alpha = mean(beta_e) and GxE deviations delta_e = beta_e - alpha (sum-to-zero constraint). Three tests: stable H0: alpha=0 (1 df), GxE H0: delta=0 (E-1 df), joint H0: all=0 (E df). Extracted via contrast matrices on Var(beta) — no duplicate GLS code.

3. **FA(k) Variance Structure**: Vg = Lambda @ Lambda^T + diag(psi) where Lambda is (E, k). Arbitrary k < E (hard error if k >= E). Lower-triangular Lambda with log-diagonal for identifiability (ASReml-R convention). Free params = E*k - k(k-1)/2 + E. For E=10, k=2: 30 params vs 55 unstructured.

**Multi-kernel MET**: V = sum_j K_j kron Vg_j + I kron Ve. Uses direct V^{-1} path when multiple kernels provided (e.g., additive + dominance). O(n^3 E^3) — feasible for moderate n and E.

**Validation**: Vg/Ve match GEMMA mvLMM to 6+ decimal places on MDP maize data (276 samples, 2926 SNPs, 2 environments). -log10(P) correlation = 0.998.

**CLI**: `torchgwas met-scan --parameterization reaction_norm|per_env --vg-structure "fa(k)" --kernel-files dom.npy`

**Tests**: 591 tests pass (19 MET-specific: 8 reaction-norm, 19 FA(k), 5 multi-kernel).

### 12c. Set-Based Association Tests Under LMM (Phase 18)

**Scientific gap**: Single-SNP tests miss rare-variant signals and aggregate effects within genes or
pathways. SKAT, Burden, and SKAT-O tests aggregate SNP effects within regions but existing
implementations (SKAT R package, SAIGE-GENE) are CPU-bound and single-trait only. No tool offers
GPU-accelerated set-based tests under a multi-trait LMM.

**TorchGWAS approach**:
- Implement SKAT (variance-component test), Burden (mean-effect test), and SKAT-O (optimal combination)
  within the LMM framework using the null model's P-operator (projection matrix).
- The set-based test statistic Q = (Py).T @ G_set @ W @ G_set.T @ (Py) can be computed using the
  same rotated-space quantities cached during the null fit.
- Batch across gene regions using GPU parallelism: score hundreds of genes simultaneously.
- Extend to multi-trait: test whether a gene region affects any of d traits jointly.

**Novelty**: To our knowledge, the first GPU-accelerated, multi-trait set-based GWAS tests under an LMM framework.

**LDJ-LMM extension (LD-Block Joint LMM — "fine-mapping lite")**: Beyond gene-region tests, extend the set-based framework to systematic LD-block testing. Partition genome into LD blocks b; for each block compute block score quadratic form T_b = U_b^T I_b^{+} U_b ~ chi-squared(r_b) where r_b = effective rank of block's score information matrix (not simply the number of SNPs; a regularized pseudoinverse is required when the block is nearly collinear). Optional within-block low-rank basis (randomized SVD / block-PCs) bounds df and computation. This provides higher locus-level power and more stable discovery than single-SNP tests under allelic heterogeneity. Output includes block-level p-values plus per-SNP conditional persistence scores (Section 12j).

### 12d. Bayesian Variable Selection GWAS (Phase 18) — **DONE**

**Scientific gap**: Frequentist single-SNP tests suffer from multiple testing burden and cannot model
joint SNP effects. Bayesian approaches (BOLT-LMM's mixture prior, Quickdraws' spike-and-slab) offer
more power but are computationally expensive. Quickdraws (Nat Genet 2025) showed 5% more
associations than REGENIE using variational inference with GPU matrix ops — but it's a closed system,
not extensible.

**TorchGWAS implementation** (two inference methods):

1. **CAVI (Coordinate Ascent Variational Inference)**: Mean-field spike-and-slab prior on SNP effects
   within the LMM: `beta_j ~ pi * N(0, sig_beta^2) + (1-pi) * delta(0)`. Sigmoid inclusion
   probabilities. EM for hyperparameters (pi, sig2_beta). ELBO convergence monitoring.

2. **SuSiE (Sum of Single Effects)** (Wang et al., JRSSB 2020): Models L independent "single-effect"
   layers where SNPs compete within each layer via softmax (not sigmoid). This prevents LD-driven
   PIP inflation that plagues mean-field methods. Key implementation details:
   - IBSS (Iterative Bayesian Stepwise Selection) update: per layer, partial residual → vectorized
     Bayes factors → softmax for alpha (inclusion) → EM for per-layer prior variance.
   - PIP = `1 - prod_l(1 - alpha_lj)` across L layers.
   - Per-layer 95% credible sets from sorted alpha accumulation.
   - Auto-scaling: `prior_sig2_beta *= (sig2_g + sig2_e)` to handle raw (unstandardized) phenotypes.
   - V_floor: `1e-4 * prior_sig2_beta` prevents complete layer collapse during EM.
   - Interval-based EM (every `em_interval` iterations) for stable V estimation.

**Outputs**: Posterior Inclusion Probabilities (PIPs) per SNP, per-layer credible sets (SuSiE),
posterior effect means, ELBO trace, method tag ("susie"/"cavi"), number of active layers.

**CLI**: `torchgwas bayes-scan --method susie|cavi --n-signals 10 ...`

**Validation results**:
- MDP real data (279 samples, 3093 SNPs, EarHT trait), validated against GEMMA BSLMM: CAVI top PIP=0.53, GEMMA BSLMM=0.37, SuSiE=0.17. SuSiE is inherently more conservative due to softmax dilution with large p.
- Simulated data with known causal SNPs: Both methods detect causal loci; SuSiE provides sharper credible sets under LD.
- 23 tests pass, 8 benchmarks pass (including three-way real-data comparison).

**Key insight**: SuSiE's conservatism on highly polygenic traits (like MDP EarHT) is a feature, not a bug — it avoids spreading PIP across LD-correlated non-causal SNPs. On traits with fewer, stronger QTLs, SuSiE's credible sets are more interpretable.

**CSMM variant (Calibrated Sparse Mixed Model)**: The spike-and-slab model above can also be fit via penalized likelihood (L1) + debiased/orthogonal inference, yielding frequentist calibrated p-values rather than Bayesian PIPs. When using L1 penalization under the LMM, the debiasing correction must account for the mixed-model covariance structure — the precision matrix involves V^{-1} = (sig2_g K + sig2_e I)^{-1}, not just I. This requires n >> s*log(p) (Javanmard & Montanari 2014 conditions, generalized to correlated errors). The CSMM is NOT a separate model from the spike-and-slab — it is the same model family with an alternative inference path.

### 12e. Mixed-Precision Kernel Ridge Regression for Epistasis (Future)

**Scientific context**: The 2024 Gordon Bell Prize finalist (Slim et al., "Toward Capturing Genetic
Epistasis from Multivariate GWAS Using Mixed-Precision Kernel Ridge Regression," SC24) demonstrated
that mixed-precision tensor-core arithmetic can make epistasis GWAS tractable at scale — achieving
1.805 ExaOp/s on the Alps supercomputer for 305K UK Biobank patients. Their four-precision
Cholesky solver outperformed REGENIE by five orders of magnitude.

**TorchGWAS opportunity**: Bring the core algorithmic ideas (INT8/FP16 tensor cores for distance
matrices, mixed-precision Cholesky solver) to commodity single-GPU hardware. TorchGWAS's AMP
pipeline (Phase 7) and the Cholesky-parameterized optimizer (Mode C) already provide the substrate.
The epistasis kernel K_aa = K_a # K_a (Hadamard product) from Phase 11 is the mathematical
prerequisite. This positions TorchGWAS at the intersection of HPC genomics and accessible tooling.

### 12f. GPU-Accelerated Multi-trait Threshold-Linear GWAS (Phase 16)

**Scientific gap**: Categorical traits (disease resistance, fertility scores, lodging, survival, psychiatric disorder status) are ubiquitous in breeding and clinical genetics but violate Gaussian assumptions. For over two decades, MAP methods for threshold models were limited to a single categorical trait. Bermann et al. (2026, _Genetics_) solved this decades-old gap with Newton-Raphson and EM algorithms for jointly analyzing multiple categorical and continuous traits under threshold-linear models. However, their implementation (BLUPF90 suite, Fortran) is CPU-only and designed for breeding evaluations, not GWAS scanning.

**TorchGWAS contribution**: To our knowledge, the first GPU-accelerated implementation of multi-trait threshold-linear GWAS:
- **GPU-parallel NR iteration**: Per-individual computations of Δ_i (truncated MVN means), Γ_i (truncated MVN variance-based Hessian weights), and R̃_i^{-1} are embarrassingly parallel — batch across all n individuals on GPU tensors.
- **GPU-batched MVN CDF**: The multivariate normal probability integrals (Miwa et al. 2003) required for truncated normal moments are batched across individuals using GPU kernels, replacing serial Fortran loops.
- **ssGWAS for categorical traits**: Posterior variances from `(W'R̃^{-1}W + S)^{-1}` at NR convergence yield marker effect p-values for single-step GWAS on categorical traits (Aguilar et al. 2019; Leite et al. 2024) — enabling GWAS for disease resistance, fertility, and other ordinal phenotypes.
- **Autograd-based threshold and variance component estimation**: Unlike BLUPF90 which assumes known variance components and thresholds (estimated via prior Gibbs sampling), TorchGWAS can leverage PyTorch autograd to differentiate through the NR iterations for joint estimation — a potential methodological advance.
- **Polyploid categorical-trait GWAS**: Combined with TorchGWAS's ploidy-aware genotype encoding (Section 9), this enables threshold-linear GWAS for polyploid organisms — to our knowledge, entirely novel. Disease resistance scoring in tetraploid potato, lodging scores in hexaploid wheat, and survival traits in polyploid aquaculture species become accessible.

**Novelty**: To our knowledge, the first open-source, GPU-accelerated tool combining multi-trait threshold-linear MAP models with GWAS scanning, polyploid support, and autograd-based parameter estimation.

### 12g. OCF-LMM — Orthogonal Cross-Fit LMM (Phase 21)

**Scientific gap**: Modern "predict-then-test" pipelines (REGENIE, BOLT-LMM) residualize phenotypes on a polygenic predictor before scanning. But when the nuisance model is rich (e.g., many covariates, non-linear terms), standard plug-in residualization produces biased standard errors because the prediction error is correlated with the test statistic. The Double/Debiased Machine Learning framework (Chernozhukov et al. 2018, Econometrica) solves this with cross-fitting + Neyman-orthogonal scores, but has not been formalized for LMM-based GWAS.

**TorchGWAS model**:
```
y = alpha_j * g_j + f(W) + u + epsilon,  u ~ N(0, sig2_g * K)
```
- Cross-fitting: split samples into K folds; fit nuisance f(W) on K-1 folds, predict on held-out fold.
- Neyman-orthogonal score for alpha_j yields valid chi-squared(1) null even when f(W) converges slowly.
- Optional LOCO-K and/or genotype nuisance residualization E[g_j | W] to mitigate structure leakage.

**Expected wins**: Improved power at fixed type-I error when polygenic background is well modeled; robust to "predictive proxy" leakage. Formalizes what REGENIE approximates.

**Novelty**: To our knowledge, the first GWAS tool to implement DML-valid inference within the mixed-model framework with explicit cross-fitting guarantees.

### 12h. KO-LMM — Knockoff Mixed Model (Phase 22)

**Scientific gap**: Standard GWAS p-value thresholding + clumping provides no formal FDR guarantee under LD. Model-X knockoffs (Candès et al. 2018, JRSS-B) and group knockoffs (Sesia et al. 2020, JASA; KnockoffGWAS) provide exact FDR control but are not available under the mixed-model framework.

**TorchGWAS model**:
- Partition variants into LD blocks b.
- Generate group knockoff genotypes G̃ per block (multivariate Gaussian approximation conditional on covariates/PCs).
- Compute LMM-aware importance statistics: W_b = |Z_b| - |Z̃_b| where Z are scan statistics.
- Apply knockoff filter to control FDR at target q.

**Important qualification**: Standard knockoff generation assumes exchangeability of (G, G̃) conditional on covariates. Under relatedness (when K is needed), the exchangeability condition must hold conditionally on covariates but NOT on the random effect u — the knockoffs are generated marginally with respect to the random effects, following the approach in KnockoffGWAS. This is an approximation whose validity depends on adequate covariate adjustment for population structure. The charter records this assumption explicitly.

**Computational cost**: Knockoff generation is O(p_b^3) per LD block for the conditional covariance matrix. This is tractable for typical LD blocks (p_b < 500) but the charter should flag blocks with p_b > 1000 for special handling (subsampling or rank reduction).

**Expected wins**: Improved replicability and controlled false discoveries vs p-value threshold + clumping.

### 12i. Within-Family / Family-Aware GWAS (Phase 23)

**Scientific gap**: Standard GWAS associations can be confounded by population structure, indirect genetic effects (dynastic effects from parents), and assortative mating — even after kinship correction (Young et al. 2022, Nat Genet). Within-family designs (sibling comparisons, family fixed effects) remove all between-family confounding and provide the strongest causal credibility for genetic associations.

**TorchGWAS approach**:
- Implement within-family scan: family fixed effects or within-cluster demeaning (subtract family mean from all variables).
- After demeaning, run standard LMM scan on within-family residuals.
- Output attenuation metrics: ratio of within-family to standard effect sizes per SNP.
- Integrate attenuation into the causal-safety diagnostics (Section 6k): SNPs with large between- vs within-family discrepancy are flagged for confounding.

**Why this matters**: Within-family GWAS is now standard for causal credibility assessment (sib-GWAS in UK Biobank, within-family PGS evaluation). No existing GPU-accelerated tool provides integrated within-family vs standard GWAS comparison.

### 12j. LD-Conditional Score GWAS (Phase 24)

**Scientific gap**: Single-SNP GWAS identifies loci but cannot distinguish causal from proxy variants within LD blocks. Full Bayesian fine-mapping (SuSiE, FINEMAP) is powerful but computationally expensive and requires careful specification. Stepwise conditional analysis (GCTA-COJO, Yang et al. 2012) is a lighter alternative but has not been integrated with LMM scanning or systematic LD-block partitioning.

**TorchGWAS approach**:
- Within each LD block, compute conditional score tests (condition on lead SNP(s) or block-PCs).
- Output both marginal and conditional p-values per SNP.
- **Conditional persistence**: binary indicator of whether a SNP remains genome-wide significant (or retains >50% of its -log10(p)) after conditioning on all other significant SNPs in the block. This is a simple, well-defined metric — not a posterior probability.

**Expected wins**: Reduces proxy-as-causal interpretability errors; improves locus refinement without full fine-mapping.

### 12k. HetLMM — Heterogeneity-Aware Single-Trait LMM (Phase 25)

**Scientific gap**: GxE interactions are critical but the multi-trait Kronecker GxE-LMM (Section 12b) is heavy for single-trait screening. A lighter single-trait model with random slopes for a single environment/context variable provides a fast scan for heterogeneity.

**TorchGWAS model**:
```
y = X*beta + g*gamma + (g ⊙ z)*eta + u + epsilon
```
where z is an environment/context variable (continuous or categorical indicator). Tests: main (1 df on gamma), interaction (1 df on eta), combined (2 df joint). In rotated eigenspace, this is a 2-column Schur complement — same cost as a 2-covariate LMM scan.

**Expected wins**: Finds context-specific loci missed by standard GWAS; essential for multi-environment trials (MET) in breeding. Lightweight alternative to full GxE-LMM.

### 12l. DML Module for Modifiable Exposures (Phase 26, Causal Inference Extension)

**Scientific context**: This is NOT a GWAS model but a causal inference module that uses GWAS results and the TorchGWAS infrastructure. It estimates causal effects of modifiable exposures T on outcomes Y using Double/Debiased ML (Chernozhukov et al. 2018), distinct from OCF-LMM (which tests SNP effects with debiased nuisance estimation).

**TorchGWAS approach**:
- Fit nuisance models E[Y|W] and E[T|W] using GPU-accelerated regression (leveraging existing LMM/GLM infrastructure).
- Cross-fitting + orthogonalization → point estimate tau_hat, SE, CI, p-value.
- Diagnostics: overlap assessment, nuisance model quality, sensitivity to unobserved confounding.

**Important distinction**: This module answers "what is the causal effect of exposure T on outcome Y?" — a fundamentally different question from GWAS ("which SNPs associate with Y?"). It is placed in a separate causal-inference section, not in the GWAS model family.

### 12m. TRIAD-REML: Trust-Region Inexact Autograd-Differentiated REML (V1, Optimizer Mode G)

**Scientific gap**: All existing GWAS variance-component optimizers (GEMMA, GCTA, ASReml, SAIGE) use the Average Information (AI) matrix as a Hessian proxy. The AI matrix is the average of the observed and expected Fisher information — it is computationally cheap but is an *approximation*. Far from the optimum, the AI approximation gives inaccurate Newton directions, causing convergence to suboptimal points or requiring many damped steps. This problem worsens as the number of traits d grows, because the parameter space dimension scales as d(d+1) and the curvature landscape becomes increasingly ill-conditioned.

**Key insight**: PyTorch's autograd system supports double-backward (`torch.autograd.grad` of a gradient), which computes *exact* Hessian-vector products (Hvp) without ever forming the full Hessian matrix. This is a capability no other GWAS tool has — it requires a differentiable programming framework, which traditional GWAS tools (written in C/C++ with hand-derived derivatives) cannot provide.

**TRIAD-REML algorithm**:

1. **Log-Cholesky parameterization**: Represent each SPD variance component V = LLᵀ where L is lower-triangular with log-transformed diagonal. The parameter vector θ ∈ ℝᵖ (p = d(d+1)) is unconstrained — every θ maps to valid (Vg, Ve) ≻ 0. No projection, no boundary handling, no PD checks needed.

2. **Double eigendecomposition (EED) forward pass**: Given Cholesky factors Lg, Le:
   - Compute M = Le⁻¹Lg, eigendecompose MMᵀ = US²Uᵀ
   - Per-individual covariance becomes diagonal: H[i,k] = λᵢ·sₖ² + 1
   - All n individuals share the same d×d eigendecomposition → O(d³ + n·d²) per evaluation instead of O(n·d³)

3. **Exact Hessian-vector products**: Given gradient g = ∂f/∂θ (from autograd backward), compute Hv = ∂(gᵀv)/∂θ for any direction v via a second autograd call. Cost: one additional forward+backward pass. No AI approximation, no hand-derived second derivatives.

4. **CG-Steihaug trust-region solver**: Approximately solve the trust-region subproblem min‖g + Hs‖ s.t. ‖s‖ ≤ Δ using truncated conjugate gradient with:
   - Early termination on negative curvature (step to trust-region boundary)
   - Early termination when step reaches trust-region boundary
   - Adaptive trust radius based on actual-vs-predicted reduction ratio (ρ)

5. **Trust-region update**: Accept/reject step based on ρ = (f(θ) − f(θ+s)) / (predicted reduction). Expand radius if ρ > η₂ (good agreement), shrink if ρ < η₁ (poor agreement). This globalization strategy is more robust than line-search for non-convex landscapes.

**Computational complexity per iteration**:
- Forward pass (EED): O(d³ + n·d²) — one d×d eigendecomposition + n scalar operations per trait
- Gradient (autograd backward): O(d³ + n·d²) — same cost as forward
- Each CG iteration (one Hvp): O(d³ + n·d²) — autograd double-backward
- Total per outer iteration with k CG steps: O(k · (d³ + n·d²)), typically k ≤ 2p

**Benchmark results (MDP maize dataset, 279 samples)**:

| Method | Traits | Log-likelihood | Vg max|diff| vs GEMMA | Time |
|--------|--------|---------------|----------------------|------|
| LBFGS (Mode C) | 5 | −3845.459 | 1.27 | 3.5s |
| **TRIAD (Mode G)** | **5** | **−3845.498** | **0.96** | **7.6s** |
| PX-EM+NR (GEMMA-style) | 5 | −3847.245 | 23.18 | 0.5s |
| LBFGS (Mode C) | 7 | — | — | — |
| **TRIAD (Mode G)** | **7** | — | p-corr 0.9859 vs 0.9844 (LBFGS) | — |

EED optimization speedup (forward + gradient + 1 Hvp):

| n | d | Naive Cholesky | EED | Speedup |
|---|---|---------------|-----|---------|
| 5,000 | 5 | 239 ms | 10 ms | 23× |
| 5,000 | 7 | 166 ms | 14 ms | 12× |

**When to use TRIAD over LBFGS**:
- d ≥ 5 traits: curvature landscape becomes ill-conditioned; exact Hessian outperforms L-BFGS approximation
- When variance component accuracy matters more than speed (e.g., heritability estimation, genetic correlation studies)
- When LBFGS converges to suboptimal points (detectable by comparing final log-likelihoods)

**Implementation**: `torchgwas/optim/triad_reml.py`. Selected via `NumericalConfig(multi_trait_reml_method="triad")`. Falls back to LBFGS on failure.

**Novelty claim**: To our knowledge, TRIAD-REML is the first GWAS variance-component optimizer to use exact Hessian-vector products via automatic differentiation combined with trust-region globalization. All prior tools use either the AI approximation (GEMMA, GCTA, ASReml), quasi-Newton with approximate curvature (L-BFGS), or first-order methods (EM). The combination of exact second-order information with trust-region methods provides both stronger convergence guarantees and better empirical accuracy on multi-trait problems.

### 12n. Random Regression LMM — Longitudinal + Spatio-Temporal GWAS (Phase 38) — **DONE**

**Scientific gap**: Longitudinal phenotypes (growth curves, yield-over-seasons, repeated biomarkers, NDVI time series) and field-trial spatial structure are standard in plant and animal breeding but absent from GEMMA / GAPIT. BLUPF90+ and ASReml-R ship random regression for breeding evaluation but neither is GPU-batched nor designed as a GWAS scanner. To our knowledge, no existing tool offers GPU-accelerated longitudinal GWAS via basis projection.

**TorchGWAS implementation**:

1. **Bases**: `torchgwas/linalg/basis.py` ships Legendre polynomials, B-splines, and 2D tensor-product P-splines, plus an `evaluate_basis_at` helper for predicting at arbitrary times.
2. **`RandomRegressionLMM`** (`torchgwas/models/rr_lmm.py`): Two modes —
   - `mode="projection"` (default, production): per-individual OLS projection of long-format observations onto an `(n, b)` basis-coefficient matrix, then routed through `MultiTraitLMM` with `n_traits := b`.
   - `mode="stacked"` (verification only): T-trait MultiTraitLMM then `Φ⁺`-project `Vg_T` → `K_coef`. Used to verify the projection path on balanced data.
3. **K_coef structures** (`torchgwas/optim/rr_reml.py`): `unstructured`, `diagonal` (per-channel EMMA), and `fa(k)` (`Vg = Λ Λᵀ + diag(ψ)`).
4. **Permanent environment block** (`include_pe=True`): `Ve = K_pe + σ²_e_residual·mean((Φᵢᵀ Φᵢ)⁻¹)`, PSD-projected via eigh.
5. **Five test types per SNP**: joint χ²(b), intercept χ²(1), slope χ²(1), time-varying χ²(b−1), and per-time-point β(t) χ²(1) when `eval_times` is supplied.
6. **`SpatioTemporalRR`** (`torchgwas/models/rr_spatial.py`): Two-step deconfounding wrapper — penalized 2D P-spline fit on (row, col) field coordinates removes a fixed-effect spatial surface from `Y_long` *before* the temporal RR-LMM. Random-effect smoother path is deferred.
7. **CLI**: `torchgwas rr-scan` (29th overall subcommand). Outputs `.rr.tsv` (per-SNP all four base tests), `.rr_at_t.tsv` (per-SNP × time when `--eval-times` given), `.rr_null.json`.

**External validation against BLUPF90+ (single-env RR)** on simulated balanced longitudinal data (n=250, T=10, h²=0.6, polygenic GRM, planted-truth K_coef):

| basis | b | Truth↔TorchGWAS rel | Truth↔BLUPF90 rel | TorchGWAS↔BLUPF90 rel |
|---|---|---|---|---|
| Legendre (order=2) | 3 | 0.123 | 0.130 | **0.078** |
| B-spline (degree=3, 1 interior knot) | 5 | 0.227 | 0.227 | **0.162** |

In both bases TorchGWAS and BLUPF90+ track each other ~30-50% more tightly than either matches the planted truth — the diagnostic fingerprint of two correct REML procedures landing on the same likelihood surface with sampling noise. σ²_g(t) curves agree within ~3-8% across all 10 timepoints. Harness: `benchmark/benchmark_rr_blupf90.py`. **2D P-spline `SpatioTemporalRR` is separately validated against R `sommer`** (15×20 simulated field, planted Gaussian + sinusoidal surface): TorchGWAS↔sommer corr=0.994, RMSE=0.078 — substantially tighter than either matches truth (corr=0.984/0.983).

**Critical BLUPF90 gotcha** (worth recording): GRM file must be passed as `RANDOM_TYPE user_file_inv` — that means "file contains K, blupf90 inverts it"; `user_file` means "file is already K⁻¹". Getting it backwards gives 69% relative error vs truth.

### 12o. Random Regression × Multi-Environment LMM (RR-MET, Phase 39) — **DONE**

**Scientific gap**: Plant breeders measure spectral / longitudinal phenotypes (NDVI, canopy temperature, growth curves, repeated yields) on the same plots across multiple environments / trials / years. The single-env Phase 38 RR-LMM ignores cross-environment genetic correlation, has no GxE × time tests, and cannot share strength across envs. BLUPF90+ supports multi-trait random regression but does not exploit separable Kronecker structure and cannot batch the scan on GPU.

**TorchGWAS implementation** — `RandomRegressionMultiEnvLMM` (`torchgwas/models/rr_met.py`, ~918 LOC), thin composition over Phase 25 `MultiTraitMultiEnvLMM`:

1. **Per-(individual, environment) projection** of long-format observations to `(n, b, E)` basis-coefficient tensor, flattened in basis-major order `k·E + e` to match MT-MET's trait-major column convention. The basis-coef axis plays the role of the "trait" dimension.
2. **Critical correctness invariant (load-bearing)**: all E environments share a single global `(t_min, t_max)` and (for B-spline) a single global knot vector placed on the **pooled** times across envs. Otherwise `β_{k,e}` has different physical meaning per env and across-env contrasts (stable / GxE / per-time × per-env) become meaningless. `project_multi_env` owns the pooling — caller does not pass per-env time refs.
3. **Vg structure menu** (mirrors MT-MET):
   - `separable` (default): `Vg = K_coef ⊗ Vg_env`, free params `b(b+1)/2 + E(E+1)/2 - 1` instead of `bE(bE+1)/2`. Cost `O(n·b³ + n·E³)`.
   - `unstructured`: full `bE × bE` Vg, hard guard at `bE ≤ 12`.
   - `fa(k)`: factor-analytic `Vg = Λ Λᵀ + diag(ψ)`, scales to large E.
   - `diagonal_coef`: per-coef diagonal.
4. **9 RR-aware contrast tests per SNP** (all linear contrasts of `vec(β)` in basis-major order, single `apply_contrast` call each):
   1. Joint χ²(bE) — `β = 0`
   2. Per env e χ²(b) — `β_{:,e} = 0`
   3. Intercept per env χ²(1) — `β_{0,e} = 0`
   4. Time-varying per env χ²(b−1) — `β_{1:,e} = 0`
   5. Stable per coef k χ²(E−1) — `β_{k,1}=…=β_{k,E}` via successive-differences `D_E`
   6. Joint stable-vs-GxE χ²(b(E−1)) — full curve constant in env: `I_b ⊗ D_E`
   7. Mean curve χ²(b) — `(1/E)Σ_e β_{:,e} = 0`
   8. Intercept-only GxE χ²(E−1) — `e_0^T ⊗ D_E`
   9. Per-time × per-env χ²(1) (when `eval_times` given) — `φ(t)^T ⊗ e_e^T`
   Tests 6 and 8 are the only genuinely-new pieces of math; everything else is composition.
5. **Per-SNP scan** dispatches to `_gls_wald_scan` for `bE ≤ 64` else `_score_test_scan_ked` (KED diagonal-precision from Phase 36).
6. **Interpretive helpers**: `genetic_variance_surface(t)` per env over a query grid, `heritability_surface(t)`, `genetic_correlation_between_envs_at_time(t)`, `env_specific_eigenfunctions(e)`.
7. **`update_null` warm-start** via `is_rr_met` flag preserves K_coef, Vg_env, basis_kind/params, b, E_envs, env_index, Phi_list_per_env across SNP-by-SNP refits.
8. **CLI**: `torchgwas rr-met-scan` (30th overall subcommand).

**External validation against BLUPF90+ MET — honest cookbook init** (n=400, T=8, E=2, h²=0.6, Legendre order=1, b=2, polygenic GRM, planted separable Kronecker truth):

| factor | Truth↔TorchGWAS rel | Truth↔BLUPF90 rel | TorchGWAS advantage |
|---|---|---|---|
| K_coef (b×b basis cov) | **0.051** | 0.852 | **~17×** better |
| Vg_env (E×E env cov)   | **0.101** | 0.276 | **~2.7×** better |

**Critical fairness note**: an earlier version of this benchmark warm-started the BLUPF90 par file at the planted truth `(CO)VARIANCES` block, which produced misleadingly favourable BLUPF90 numbers (rel < 0.02 on both factors). The corrected protocol initializes both methods from the cookbook neutral start `0.5 · diag(var(Y_e))` per environment — no truth peeking. Under this honest comparison TorchGWAS substantially beats BLUPF90: the separable structural prior (6 free Vg parameters) is far more sample-efficient than BLUPF90's unstructured 10-parameter Vg fit, which converges to a local optimum where all cross-env / cross-coef covariances collapse to zero. A seed sweep at n=400 over `[0,1,7,42,99]` confirms TorchGWAS recovers cross-env Vg_env off-diagonals correctly across all seeds (V rel ~0.07–0.15, K rel ~0.01–0.09); the seed=42 / n=200 collapse seen in early runs was a small-sample artifact, not a solver bug.

**Factor-analytic head-to-head** (n=300, T=8, E=4, b=2, fa_rank=2, separable Kron planted truth, BLUPF90 par file gets `OPTION cov_structure 1 XFA 2`; the shipped 64bit_old build *does* accept the XFA option):

| comparison | rel diff on full bE×bE Vg |
|---|---|
| Truth ↔ TorchGWAS `fa(2)` | **0.241** |
| Truth ↔ BLUPF90 `XFA 2`   | 0.599 |
| Torch ↔ BLUPF90           | 0.638 |

TorchGWAS `fa(2)` is ~2.5× closer to truth than BLUPF90 XFA on the same data. Both methods fit a rank-2 FA approximation; neither can hit zero against separable Kron truth, but TorchGWAS gets dramatically closer. Harness: `benchmark/benchmark_rr_blupf90.py --mode met` and `--mode met-fa --fa-envs E --fa-rank k`. The harness gracefully falls back to a BLUPF90 unstructured fit if a future build rejects the XFA option.

**Pushback / non-goals** (per Phase 39 plan):
- No "stacked mode" — Phase 38 stacked was verification-only; MET analogue would require per-env balanced grids and gain nothing.
- No `include_pe` (permanent environment) — adds a fourth `(bE, bE)` variance component; identifiability needs `T_{i,e} ≥ 2` per env. Deferred.
- No "combined reaction-norm flag" — reaction-norm interpretation lives entirely in the contrast machinery (tests 5, 6, 7).
- `MT-MET.score_chunk` is **not** reused verbatim — its per-trait / per-env tests interpret traits as biological traits, not basis coefs (per-basis-coef test is uninterpretable). RR-MET reuses `_gls_wald_scan` / `_score_test_scan_ked` as private backends but writes its own public `score_chunk`.

**Tests**: 36 RR-MET tests (11 projection, 22 LMM, 3 CLI) on top of 97 RR-LMM tests; **1350 total pass, 36 skipped**.

### 12p. Genotype-Uncertainty LMM (GU-LMM, Phase 27) — **DONE**

**Scientific gap**: Imputed genotypes carry uncertainty (posterior genotype probabilities from BEAGLE/IMPUTE5/Minimac4), but standard GWAS tools treat dosages as known values, ignoring the dosage variance. This inflates test statistics for poorly imputed variants and produces miscalibrated p-values (Zheng et al. 2011).

**TorchGWAS model**: `GULM` in `torchgwas/models/gu_lmm.py`. Implements a dosage-variance–corrected score test where the variance of the test statistic accounts for imputation uncertainty via `diag(V_dosage)`. Reuses the LMM null fit and P-operator infrastructure.

**Tests**: 17 tests in `tests/test_gu_lmm.py`.

### 12q. Leave-Region-Out LMM (LRO-LMM, Phase 28) — **DONE**

**Scientific gap**: Standard LOCO (leave-one-chromosome-out) removes only the target chromosome from the GRM. But for fine-mapping within LD blocks, the target block's variants still contribute to the kinship estimate when other blocks on the same chromosome are included. This creates proximal contamination (Listgarten et al. 2012).

**TorchGWAS model**: `LROLMM` in `torchgwas/models/lro_lmm.py`. Per-block kinship subtraction — for each LD block, subtracts that block's contribution from the full GRM and re-fits the null. Finer-grained proximal decontamination than per-chromosome LOCO.

**Tests**: 17 tests in `tests/test_lro_lmm.py`.

### 12r. Adaptive FDR Control (Phase 29) — **DONE**

**Scientific gap**: Standard BH/BY FDR methods ignore informative covariates (MAF, LD score, functional annotation) that could improve power. IHW (Ignatiadis & Huber 2021, Nat Methods) and AdaPT (Lei & Fithian 2018, JRSS-B) exploit these covariates for adaptive thresholding.

**TorchGWAS implementation**: `torchgwas/stats/ihw.py` and `torchgwas/stats/adapt.py`. IHW bins p-values by covariate strata and optimizes per-stratum weights. AdaPT uses EM to learn a covariate-dependent rejection threshold. Both integrated into the `--correction` CLI flag.

**Tests**: 21 tests in `tests/test_adaptive_fdr.py`.

### 12s. GLM + GLMM Family (Phases 30-33) — **DONE**

**Scientific gap**: Categorical traits (disease status, ordinal severity, multi-class phenotypes) require non-Gaussian likelihoods. SAIGE handles binary traits with SPA but is single-trait and has no ordinal/multinomial support. No existing tool offers GPU-accelerated ordinal or multinomial GWAS with random effects.

**TorchGWAS implementation**:
- **Phase 30**: `BinaryGLM` (logistic), `OrdinalGLM` (proportional odds), `MultinomialGLM` (multinomial logit) — score tests + SPA + Firth correction. No random effects.
- **Phase 31**: `BinaryGLMM`, `OrdinalGLMM` — PQL null fitting (SAIGE-style) with score test + SPA.
- **Phase 32**: `MultinomialGLMM` — PQL for multi-class with random effects; cross-model integration tests.
- **Phase 33**: `MultiEnvGLMM` — binary/ordinal categorical phenotypes across E environments; structured genetic covariance Σ_g; joint/per-env/homogeneity/reaction-norm score tests.

**Tests**: 88 tests across `tests/test_binary_glm.py`, `tests/test_ordinal_glm.py`, `tests/test_multinomial_glm.py`, `tests/test_binary_glmm.py`, `tests/test_ordinal_glmm.py`, `tests/test_multinomial_glmm.py`, `tests/test_multi_env_glmm.py`.

### 12t. Survival GWAS (Phase 34) — **DONE**

**Scientific gap**: Time-to-event phenotypes (disease onset, survival, time-to-flowering) require Cox PH or accelerated failure time models. COXMEG, SPACox, and GATE handle this but are R/C++ CPU-only tools with no GPU path.

**TorchGWAS model**: `SurvivalGLMM` in `torchgwas/models/survival_glmm.py`. Cox PH frailty model with Breslow-Clayton PQL for null fitting, martingale residual score test for scanning, and SPACox-style SPA for extreme p-values. Polyploid support included.

**Tests**: 25 tests in `tests/test_survival_glmm.py`.

### 12u. Post-GWAS Infrastructure (Phases 36-37, 42, 44-45) — **DONE**

**TorchGWAS implementation** in `torchgwas/postgwas/`:
- **LDSC** (`_ldsc.py`): univariate h² and bivariate rg with block jackknife SE; partitioned S-LDSC (Finucane 2015) for functional enrichment.
- **Meta-analysis** (`_meta.py`): IVW fixed-effects, DerSimonian-Laird random-effects, Stouffer weighted Z, RE2 (Lee 2017).
- **LD clumping** (`_clump.py`): per-chromosome greedy clumping with configurable r² and window.
- **Colocalization** (`_hyprcoloc.py`): multi-trait hyprcoloc (Foley 2021) + classical two-trait coloc (Giambartolomei 2014).
- **Mendelian Randomization** (`_mr.py`): IVW, MR-Egger, weighted median, MR-PRESSO.
- **Gene-set enrichment** (`_enrichment.py`): MAGMA-style snp_to_gene + competitive gene_set_enrichment.
- **Fine-mapping utilities** (`_finemapping.py`): credible set extraction, PIP annotation, locus summary.
- **Multi-ancestry meta** (`_multi_ancestry.py`): MR-MEGA (Mägi 2017) + MANTRA (Morris 2011).
- **SMR/HEIDI** (`_smr.py`): summary-data-based Mendelian randomization (Zhu 2016).
- **TWAS** (`_twas.py`): S-PrediXcan + PrediXcan transcriptome-wide association.
- **HESS** (`_hess.py`): regional heritability + local genetic correlation (Shi 2016).
- **Power analysis** (`_power.py`): `gwas_power`, `power_curve`, `required_n`.
- **Winner's curse** (`_winners_curse.py`): conditional likelihood, FIQT, parametric bootstrap.

**Tests**: ~130 tests across 10 test files.

### 12v. Polygenic Score Construction (Phase 40) — **DONE**

**Scientific gap**: PGS methods (C+T, LDpred2, PRS-CS) are typically available as standalone R/Python packages that don't integrate with the GWAS pipeline. No tool offers a unified PGS framework with allele harmonization, LD reference construction, and individual scoring within the same GPU-accelerated ecosystem.

**TorchGWAS implementation** in `torchgwas/pgs/`:
- **Shared infrastructure**: `BasePGSMethod`, `LDReference` (full + block-diagonal with Ledoit-Wolf shrinkage), `PGSResult`, `load_pgs_sumstats` (alias resolution + OR→log-OR), allele harmonization (exact/swapped/strand-complement, palindromic A-T/C-G handling).
- **Tier 1**: `ClumpingThresholding` (C+T), `LDpred2Inf` (closed-form), `LDpred2Grid` (Gibbs over (p, h²) grid), `LDpred2Auto` (multi-chain hyperparameter learning with R-hat).
- **Tier 2**: `PRSCS` (Strawderman-Berger continuous shrinkage prior, block Cholesky joint β update, scipy-backed GIG sampler).
- **Scoring**: `score_individuals` for applying weights to target genotypes with allele alignment + missing-handling + chunked matmul.
- **Diagnostics**: `rhat`, `ess`, `check_convergence` for MCMC chains.
- **Validation**: `validate_pgs` with R², AUC, Nagelkerke R², liability-scale R².
- **CLI**: `pgs-fit` and `pgs-score` subcommands.

**Tests**: 75 tests across 9 files in `tests/`.

### 12w. Native C++ Accelerators (Phase 41a-41z) — **DONE**

**Scientific gap**: Python's per-element `.item()` round-trips and Python-level `for` loops create O(m) or O(n²) overhead in hot inner loops (Gibbs samplers, LD block detectors, imputation, QC). This limits practical scalability for datasets with millions of SNPs.

**TorchGWAS implementation**: 26 pybind11 C++ extension modules in `csrc/` and `torchgwas/_native/`, each providing a drop-in replacement for a specific Python hot loop:
- **PGS Gibbs**: PRS-CS (`_prscs_native`), LDpred2 (`_ldpred2_native`), C+T clumping (`_ct_native`)
- **LD block detectors**: PELT (`_pelt_native`), Gabriel (`_gabriel_native`), Big-LD (`_big_ld_native`), DP-optimize (`_dp_optimize_native`), CC-graph (`_cc_graph_native`), GWAS-aligned (`_gwas_aligned_native`), Spine (`_spine_native`), uncertainty (`_uncertainty_blocks_native`), cross-pop (`_cross_pop_native`)
- **LD utilities**: `_ld_decay_signal_native`, `_greedy_mwis_native`, `_wall_pritchard_native`
- **Imputation**: `_impute_mode_native`, `_impute_knn_native`, `_impute_ld_native`
- **Statistics**: `_spa_native` (saddlepoint), `_ldsc_native` (block jackknife), `_hwe_native` (HWE + double reduction), `_cavi_native` (BayesianVS), `_ess_native` (Geyer ESS), `_graph_native` (connected components)
- **Post-GWAS**: `_ct_native` reused for `ld_clump`

**Key design invariants**:
- Every C++ kernel preserves the Python reference below it — the `.py` file remains the algorithmic spec.
- All toggleable via `TORCHGWAS_DISABLE_NATIVE=1`.
- OpenMP parallelization for 6 kernels (HWE, impute_knn/mode/ld, SPA, DP-optimize).
- GPU imputation kernels (`_impute_gpu.py`) for CUDA tensors.
- Central dispatch via `select_path()` in `torchgwas/_dispatch.py`.

**Benchmark**: 12/24 kernels show >50× speedup (PELT 12466×, Gabriel 10717×, Big-LD 3558×).

**Tests**: ~250 tests across `tests/test_native_*.py` files.

### 12x. Visualization + PVE (Phase 43) — **DONE**

**TorchGWAS implementation** in `torchgwas/viz/`:
- `manhattan_plot`: linear per-SNP scatter with alternating chromosome colors, threshold lines, highlight/annotate overlays.
- `circos_manhattan_plot`: polar Circos-style Manhattan with angular chromosome arcs.
- `miami_plot`: two-trait back-to-back Manhattan (mirrored).
- `qq_plot`: observed vs expected -log10(p) with 95% Beta confidence band and λ_GC annotation.
- `haploview_plot`: 45°-rotated triangular LD heatmap with optional block overlays.
- `trumpet_plot`: AF-vs-effect-size scatter with power curve overlays (Garcia-Gonzalez 2023).

`torchgwas/stats/pve.py`: per-marker PVE (marginal GAPIT-compatible + joint semi-partial R²).

**Tests**: 37 tests in `tests/test_viz.py` + `tests/test_pve.py`.

### 12y. Haplotype-Based GWAS (Phases 46-47) — **DONE**

**Scientific gap**: Single-SNP GWAS cannot capture multi-allelic haplotype effects where combinations of alleles on a chromosome segment influence the trait. TorchGWAS had 13 LD block detection methods but no haplotype association testing. Motivated by livestock genetics showing haplotype GWAS detecting signals invisible to single-SNP analysis.

**TorchGWAS implementation**:
- **Phase 46** (`torchgwas/models/haplotype_gwas.py`): Four methods — HTR (Zaykin et al. 2002) F-test/LRT, block-based scan (uses any of 13 `detect_blocks()` methods), moving-window scan, haplotype-SKAT (Davies mixture chi-squared). Multi-locus EM for haplotype inference from unphased genotypes (extends Hill 1974 EM). LMM correction via rotated eigenspace WLS and P-operator.
- **Phase 46 novel** (`torchgwas/models/haplotype_novel.py`): PCHT (phase uncertainty correction), HHCT (hierarchical collapsing with Meinshausen 2008 FWER), HSKAT (exponential-decay Hamming kernel SKAT), HapGxE (haplotype × environment interaction F-tests), BayesHap (SuSiE IBSS on haplotype dosages).
- **Phase 47** (`torchgwas/models/haplotype_multi.py`): Three thin-composition classes extending haplotype testing to multi-dimensional phenotypes: `HaplotypeMultiEnvGWAS` (→ MultiEnvLMM), `HaplotypeMultiTraitGWAS` (→ MultiTraitLMM), `HaplotypeMTMETGWAS` (→ MultiTraitMultiEnvLMM). Core `_haplotype_gls_wald()` builds `(c*p + h*p) x (c*p + h*p)` GLS per block; contrast algebra via `_wald_subblock()` and `_wald_contrast()`.

**Tests**: 48 tests in `tests/test_haplotype_gwas.py` + `tests/test_haplotype_novel.py`, 33 tests in `tests/test_haplotype_multi.py`.

### 13a. Phase 13 Approximate Backends — Implementation and Benchmark Results

**Charter deliverable**: "Randomized SVD / Nystrom for GRM; stochastic trace estimation; sparse GRM (fastGWA-style). Approximate fits must be visibly labeled. Approximation error quantified against exact Phase 4-5 benchmarks."

#### Implemented Methods

| Method | File | Complexity | Memory |
|--------|------|-----------|--------|
| Randomized SVD (Halko-Martinsson-Tropp 2011) | `linalg/randomized.py` | O(n·k²) | O(n·k) |
| LOBPCG (iterative GPU eigensolver) | `linalg/randomized.py` | O(n·k·iters) | O(n·k) |
| Nystrom GRM approximation (streaming) | `linalg/nystrom.py` | O(n·l²) per chunk | O(n·l) |
| Sparse GRM (fastGWA-style threshold) | `linalg/sparse_grm.py` | O(n²) build, O(nnz) matvec | O(nnz) |
| Hutchinson trace estimator | `optim/stochastic_trace.py` | O(n·probes) | O(n) |
| Stochastic Lanczos quadrature (log-det) | `optim/stochastic_trace.py` | O(n·probes·lanczos_k) | O(n) |
| PCG-based REML (sparse path) | `optim/sparse_reml.py` | O(nnz·iters) per eval | O(nnz) |
| SparseLMM (score test) | `models/sparse_lmm.py` | O(nnz·iters) per SNP chunk | O(nnz) |

All approximate methods return `EigenDecomp` objects (or, for the sparse path, use PCG solves), so the downstream REML + scan pipeline is reused without modification. All `NullFit` objects from approximate paths carry `approximate=True`, `approx_method`, and `approx_config` fields for audit traceability.

#### Benchmark Results (MDP maize dataset, 279 samples, 3093 SNPs, EarHT)

**Eigenvalue approximation quality (vs exact `torch.linalg.eigh` on VanRaden GRM):**

| Method | Max rel. error | Mean rel. error | Spectral energy captured |
|--------|---------------|----------------|--------------------------|
| RSVD k=100 | 5.65e-02 | 1.08e-02 | 68.1% |
| RSVD k=200 | 3.30e-02 | 2.49e-03 | 92.9% |
| RSVD k=250 | 6.67e-03 | 1.16e-04 | 98.9% |
| RSVD k=270 | 4.91e-15 | 4.56e-16 | 99.8% |
| LOBPCG k=80 | 3.25e-11 | 3.18e-12 | 61.3% |
| Nystrom L=270 | — | — | eigenvalue corr 0.999 |

**P-value agreement (Pearson r on -log10(p) vs exact VanRaden LMM, and vs GAPIT MLM):**

| Method | vs Exact VanRaden r | vs GAPIT r | h² estimate | Notes |
|--------|-------------------|------------|-------------|-------|
| Exact Zhang (GAPIT ref) | 0.99999 | 1.00000 | 0.626 | Phase 4 baseline |
| Exact VanRaden | 1.00000 | 0.99999 | 0.569 | Phase 12 baseline |
| RSVD k=270 | **0.993** | 0.993 | 0.480 | 97% of n, near-exact eigenvalues |
| RSVD k=250 | 0.959 | 0.958 | 0.414 | 98.9% energy |
| **Sparse GRM (θ=0.05)** | **0.953** | **0.952** | 0.369 | **Recommended production path** |
| Nystrom L=270 | 0.928 | 0.928 | 0.598 | streaming, no full GRM |
| RSVD k=200 | 0.785 | 0.784 | 0.540 | |
| Nystrom L=200 | 0.538 | 0.537 | 0.369 | |
| RSVD k=100 | 0.458 | 0.456 | 0.957 | h² overestimated (too few components) |

#### Recommended Production Path: Sparse GRM + PCG-REML

For biobank-scale data (n > 50K), the **sparse GRM path** (`--approx-method sparse`) is the recommended approach:

1. **No eigendecomposition required** — avoids the O(n³) bottleneck entirely
2. **Bounded memory** — sparse GRM stores only entries above the relatedness threshold (typically < 1% of n² for population cohorts)
3. **PCG-based REML** — variance components estimated via Brent optimization over λ = σ²_g/σ²_e, where each λ evaluation uses PCG solves (O(nnz·iters)) and stochastic trace/log-det estimators
4. **Score test scan** — per-SNP score test T = (g^T P y)² / (σ²_e · g^T P g), requiring only one precomputed P·y and per-chunk PCG solves
5. **0.953 correlation** with exact VanRaden on MDP benchmark — strong agreement despite fully avoiding eigendecomposition

CLI usage: `torchgwas lmm-scan --genotype data.zarr --phenotype pheno.txt --approx-method sparse --sparse-threshold 0.05`

**Trade-off**: The sparse path supports only the score test (no Wald/LRT), does not produce per-SNP beta/SE, and the stochastic log-determinant introduces ~10-15% estimation noise in the REML likelihood. For exact results on datasets that fit in memory, the standard eigendecomposition path remains preferred.

#### Known Limitation: Truncated REML Bias

When using randomized SVD or Nystrom with k << n, the REML fit operates in a k-dimensional eigenspace. The "lost" variance from the n−k dropped eigenvectors is absorbed into σ²_e, causing systematic underestimation of h² (e.g., h²=0.96 at k=100 on n=279 data with true h²=0.57). This is a known limitation of all truncated spectral methods (BOLT-LMM mitigates this with a two-component correction). For biobank-scale data where the GRM eigenspectrum decays rapidly, k=2000-5000 typically captures > 99% of genetic variance and the bias is negligible.

### 13b. Phase 13 Polyploid Benchmark — TorchGWAS vs GWASpoly Reference

**Charter deliverable** (Phase 9): "Verify: match GWASpoly on published tetraploid potato dataset."

#### Dataset

GWASpoly built-in potato tetraploid dataset:
- 957 unique genotypes, 9888 SNP markers (dosage 0-4)
- vine.maturity trait across 6 environments (1249 total observations)
- GWASpoly v2.14 run with `set.K(LOCO=FALSE)`, `set.params(geno.freq=1-5/N, fixed="env", fixed.type="factor")`
- P3D approach: variance components estimated once with additive kinship, then per-model scan

#### Gene-Action Model Mapping

| GWASpoly Model | Encoding | TorchGWAS Equivalent |
|----------------|----------|---------------------|
| additive | dosage as-is (0..k) | `additive` |
| j-dom-alt | 1 if dose >= j | `j-dom` |
| j-dom-ref | 1 if dose > k-j | `(k-j+1)-dom` (approximate) |
| diplo-additive | {0->0, 1..k-1->1, k->2} (diploidized) | **differs**: TorchGWAS uses min(d, k-d) |
| diplo-general | factor(diplo-additive) | multi-column; no direct LMM comparison |
| general | factor(dosage) | multi-column; no direct LMM comparison |

For tetraploid: `1-dom-alt` = TorchGWAS `1-dom`, `2-dom-alt` = TorchGWAS `2-dom`, `2-dom-ref` = TorchGWAS `3-dom`.

#### Multi-Environment Model

GWASpoly uses all 1249 observations with a Z incidence matrix mapping observations to 957 genotypes: V = sig2_g * Z @ K @ Z^T + sig2_e * I. TorchGWAS replicates this by expanding K to observation level (Z @ K_geno @ Z^T) and including env factor dummies in X0.

#### Benchmark Results (P3D mode — matching GWASpoly)

| TorchGWAS Model | GWASpoly Ref | n_common | r(-log10p) | Top-50% Overlap |
|----------------|-------------|----------|-----------|-----------------|
| additive | additive | 9888 | **0.9999** | **100%** |
| 1-dom | 1-dom-alt | 6114 | **1.0000** | **100%** |
| 2-dom | 2-dom-alt | 7784 | **0.9999** | **100%** |
| 3-dom | 2-dom-ref | 8529 | **1.0000** | **100%** |
| diplo-additive | diplo-additive | 9888 | 0.488 | 18% |

**Result**: All gene-action models with matching encoding achieve r > 0.999 and 100% top-50 overlap with GWASpoly. The diplo-additive difference is expected — GWASpoly uses a "diploidized" encoding ({0->0, 1..3->1, 4->2}) while TorchGWAS uses min(d, k-d), which is a deliberate design choice supporting finer allelic resolution.

#### Model-Specific vs P3D Kinship

Running with model-specific kinship (each gene-action model gets its own K) produces lower agreement (r = 0.86-0.92 for dom models), confirming that GWASpoly uses P3D (additive K for all). Both approaches are statistically valid — P3D is faster, model-specific K may be more powerful for non-additive signals.

#### GRM Comparison

VanRaden GRM computed by TorchGWAS (`grm_vanraden` with ploidy=4) shows:
- Diagonal correlation with GWASpoly K: 1.000
- Off-diagonal correlation: 1.000
- Relative Frobenius difference: 8.7% (scaling difference only)

#### Verification Artifacts

- Reference data: `benchmark/gwaspoly_results/gwaspoly_*.csv` (per-model p-values)
- Benchmark script: `benchmark/benchmark_polyploid.py`
- R reference: `benchmark/run_gwaspoly.R`
- Golden tests: `tests/test_golden_gwaspoly.py` (6 tests, all passing)

### 13c. GRM Benchmark — TorchGWAS vs AGHmatrix (Polyploid Kinship)

**Dataset**: Potato tetraploid (957 genotypes, 9888 markers)

| Reference Method | Diag corr | Off-diag corr | Rel. Frob diff | Notes |
|-----------------|-----------|--------------|----------------|-------|
| **AGHmatrix VanRaden (ploidy correction)** | **1.000000** | **1.000000** | **0.000000** | **Perfect match** — identical GRM |
| AGHmatrix VanRaden (no ploidy correction) | 1.000000 | 1.000000 | 0.086 | Same structure, variance-based normalization |
| GWASpoly kinship | 1.000000 | 1.000000 | 0.087 | Same structure, GWASpoly-specific scaling |
| AGHmatrix Slater (2016) | 0.896 | 0.887 | 1.087 | Different method — combinatorial polyploid encoding |
| AGHmatrix Endelman digenic | 0.888 | 0.166 | 1.832 | Non-additive interaction GRM — captures digenic effects |
| AGHmatrix pseudo-diploid | 0.927 | 0.914 | 0.472 | Collapsed to 0/1/2 before GRM |

**Key result**: TorchGWAS `grm_vanraden(G, ploidy=4)` produces **identical** results to AGHmatrix `Gmatrix(G, method="VanRaden", ploidy=4, ploidy.correction=TRUE)`. The Slater, Endelman, and pseudo-diploid methods capture fundamentally different genetic relationships and are targets for future implementation (Section 9d items 5-11).

**Reference data**: `benchmark/aghmatrix_results/aghmatrix_*.csv`

## 18. Known Scaling Risks and Mitigations

| Risk | Scale threshold | Mitigation |
|---|---|---|
| GRM does not fit in GPU memory | n > ~50K (n x n FP64 = n^2 * 8 bytes) | Block-streaming GRM: compute tiles on GPU, accumulate on CPU or write to disk. Phase 9 deliverable. |
| GRM does not fit in CPU RAM | n > ~500K (~2 TB in FP64) | **Phase 13 sparse GRM** (recommended): threshold sparsification + PCG-REML avoids full n×n matrix. Alternatives: Nystrom low-rank (O(n·l) memory), randomized SVD on materialized K. |
| Eigendecomposition is O(n^3) | n > ~100K | **Phase 13 sparse path**: PCG-REML + score test bypasses eigendecomposition entirely. Alternative: randomized SVD / LOBPCG for truncated top-k eigenpairs in O(n·k²). |
| Simple GLM may not beat PLINK 2.0 on CPU | Any n | PLINK 2.0 uses hand-tuned SIMD intrinsics for GLM. TorchGWAS GLM should target GPU advantage; do not claim CPU GLM parity with PLINK 2.0. |
| GPU non-determinism | Golden tests | Use `torch.use_deterministic_algorithms(True)` for regression tests. Accept FP-tolerance differences between CPU and GPU paths. |

## 19. Rules Any Future AI Agent Must Follow

1. **Do not broaden scope without explicit approval.** V1 is GLM + LMM + mvLMM + FarmCPU + BLINK. Nothing else.
2. **Do not skip the exact CPU phase.** GPU acceleration comes after CPU correctness is proven.
3. **Do not use approximate methods in exact-core phases.** No stochastic trace, no Nystrom, no sparse GRM until Phase 10.
4. **Do not break the dtype policy.** Statistical inference paths must be FP64. Violations will produce silent p-value drift.
5. **Do not let I/O-specific code leak into model classes.** Models receive tensors; they do not know about PLINK or Zarr.
6. **Do not call a phase complete without its validation gate.**
7. **Do not change the ScanResult schema** without explicit approval — downstream tools depend on it.
8. **Match GEMMA's preprocessing exactly** when targeting equivalence: genotype centering, mean imputation, MAF filtering, covariate ordering, sample alignment. Tiny differences here cause 3rd-5th decimal p-value drift.
9. **Use deterministic algorithms for golden tests** (`torch.use_deterministic_algorithms(True)`).
10. **Prefer solves over inversions.** Never invert matrices larger than d x d (traits). Use `torch.linalg.solve` or `torch.linalg.cholesky_solve`.
11. **Never run GWAS without marker metadata.** Every variant must have a marker ID, chromosome, and physical position before entering the scan loop. If the input format lacks these (e.g., a numeric CSV without a companion map file), raise a clear error — do not silently assign dummy values.
12. **Accept all common column name synonyms** for marker metadata, phenotype IDs, etc. (as listed in Section 6). Match case-insensitively. Never reject valid data because of a column naming convention difference.
13. **All format readers must produce numerically equivalent internal representations.** The same dataset in HapMap, CSV, VCF, and PLINK format must yield dosage tensors and metadata that are numerically equivalent under declared tolerances (≤1e-12 for dosage values, exact match for metadata) after parsing. Test this with format round-trip equivalence tests.

## 20. Immediate Work Packages

A new agent starting from scratch should begin with these work packages in order:

| Work Package | Phase | Required Outputs | Pass Gate |
|---|---|---|---|
| **A — Phase 0: Architecture** | 0 | Module skeletons with Protocol/ABC definitions; config.py; ScanResult + NullFit schemas; validation matrix; test stubs; ploidy-aware types | All contracts are typed, importable, and documented |
| **B — Phase 1: I/O + format intelligence** | 1 | Format auto-detection; PlinkReader (memmap .bed); PLINK2 pgen reader; VCF reader; BGEN reader; HapMap reader (letter genotype decoding); CSV/TSV/TXT numeric reader (auto-detect orientation/delimiter/companion map); phenotype/covariate loader with deterministic sample alignment (three-way ID intersection, lexicographic sort, alignment manifest TSV); format validation and pre-flight checks | All readers produce numerically equivalent internal tensors (≤1e-12 tolerance) for same data in different formats; HapMap letter decoding matches GAPIT; CSV map-file auto-discovery works; sample alignment is deterministic and reproducible across runs |
| **C — Phase 2: Preprocessing + imputation** | 2 | Built-in imputation (mean/mode/KNN/LD); external wrappers (BEAGLE, IMPUTE5); QC filters (MAF/missingness/HWE/het-excess/Rsq) with CLI-configurable thresholds; variant QC Parquet output with per-variant AF, MAF, HET, HWE_P, MISS_RATE, FILTER_PASS; ploidy-aware standardization | Imputed output correct; BEAGLE wrapper round-trips; QC filter counts match expected; allele frequencies in Parquet match GEMMA .log.txt and GAPIT frequency output |
| **D — Phase 3: GRM + Eigen** | 3 | Streaming GRM builder; polyploid GRM; eigendecomposition; rotation utilities; LOCO | GRM and eigenvalues match GEMMA within 1e-6 |
| **E — Phase 4: Single-trait LMM** | 4 | AI-REML optimizer, score/wald/lrt scan, P-operator, full optimizer stack | P-values match GEMMA on mouse dataset to 4th decimal |

## 21. Repository Layout

```
torchgwas/                      # installable package
  __init__.py
  config.py
  _dispatch.py                  # GPU/native/Python execution path selector
  _native/                      # pybind11 C++ extension dispatchers
  io/                           # Format readers (PLINK, VCF, BGEN, HapMap, CSV, Zarr)
  preprocess/                   # Imputation, QC, standardization
  linalg/                       # GRM, eigendecomposition, basis functions
  models/                       # 30+ statistical models (GLM → MT-MET → Haplotype)
  scan/                         # UnifiedScanner streaming infrastructure
  stats/                        # Multiple testing, SPA, mixture distributions, PVE
  optim/                        # 6-mode optimizer stack + TRIAD-REML
  ld/                           # 13 LD block detection methods + diagnostics
  postgwas/                     # LDSC, meta-analysis, coloc, MR, enrichment, SMR, TWAS
  pgs/                          # PGS construction (C+T, LDpred2, PRS-CS) + validation
  viz/                          # Manhattan, Circos, QQ, Haploview, Miami, Trumpet plots
  results/                      # Result formatting
  cli.py                        # 47 CLI subcommands
csrc/                           # C++ source for native accelerators
  pgs/                          # PRS-CS, LDpred2, C+T native kernels
  ld/                           # LD block detection native kernels
  stats/                        # SPA, LDSC, HWE, ESS native kernels
  impute/                       # Imputation native kernels
  models/                       # CAVI native kernel
tests/                          # 128 test files, 2102 tests pass
  golden/                       # GEMMA/GAPIT/GWASpoly reference datasets
  test_*.py
bench/                          # Benchmark scripts and results
  native_speedups.py            # Native accelerator benchmark suite
  native_speedups.md            # Speedup results table
pyproject.toml                  # build config, entry points, dependencies
setup.py                        # C++ extension build (optional=True)
README.md
CLAUDE.md                       # agent guidance (comprehensive, up-to-date)
TorchGWAS_AI_Agent_Handoff_Charter.md  # this document (Version 5.0)
```

## 22. Key Dependencies

| Package | Role | Required |
|---|---|---|
| `torch` (>= 2.0) | Tensor operations, autograd, GPU, AMP | Yes |
| `numpy` | Array interop, memmap for PLINK | Yes |
| `pandas` | Phenotype/covariate I/O | Yes |
| `scipy` | Fallback distributions, sparse utilities | Yes |
| `matplotlib` | Manhattan/QQ plots | Yes |
| `zarr` | Zarr genotype reader | Optional (Phase 9) |
| `h5py` | HDF5 genotype reader | Optional (Phase 9) |
| `pyarrow` | Parquet output | Optional |
| `seaborn` | Enhanced plotting | Optional |

## 23. CLI Interface

TorchGWAS provides **47 subcommands** covering data management, GWAS scans (all model families), post-GWAS analysis, PGS construction, and end-to-end pipeline orchestration. All scan commands share common arguments: `--genotype`, `--phenotype`, `--test` (wald/score/lrt), `--correction` (bonferroni/holm/bh/by/storey/simplem/weighted-bh/lfdr/hierarchical/ihw/adapt/none), `--maf-min`, `--miss-max`, `--output`, `--device`, `--chunk-size`, `--max-iter`. See CLAUDE.md for the full CLI command listing.

```bash
# Install
pip install -e ".[dev]"

# ═══════════════════════════════════════════════════
# DATA MANAGEMENT (3 commands)
# ═══════════════════════════════════════════════════

# Validate a dataset (pre-flight check)
torchgwas validate --genotype data.bed --phenotype pheno.txt

# Convert between formats (supported output: bed, vcf, zarr)
torchgwas convert --input data.vcf.gz --output data --format bed
torchgwas convert --input data.bgen --sample data.sample --output data.zarr --format zarr

# Impute missing genotypes (9 methods)
# Built-in simple methods
torchgwas impute --genotype data.bed --method mean --output imp.pt
torchgwas impute --genotype data.bed --method mode --output imp.pt
torchgwas impute --genotype data.bed --method knn --output imp.pt
torchgwas impute --genotype data.bed --method ld --output imp.pt
# GPU-accelerated methods (Phase 19)
torchgwas impute --genotype data.vcf.gz --method li-stephens --ploidy 4 --output imp.pt
torchgwas impute --genotype data.vcf.gz --method deep-learning --output imp.pt
# External tool wrappers (require tool installation)
torchgwas impute --genotype data.vcf.gz --method beagle \
  --ref-panel 1000G.vcf.gz --output data_imputed.vcf.gz
torchgwas impute --genotype data.vcf.gz --method impute5 \
  --ref-panel 1000G.vcf.gz --output data_imputed.vcf.gz
torchgwas impute --genotype data.vcf.gz --method minimac4 \
  --ref-panel 1000G.msav --output data_imputed.vcf.gz

# ═══════════════════════════════════════════════════
# GWAS SCANS — V1 CORE (5 commands)
# ═══════════════════════════════════════════════════

# GLM (no kinship)
torchgwas glm-scan \
  --genotype data.bed --phenotype pheno.txt --output results

# Single-trait LMM (GEMMA-equivalent)
torchgwas lmm-scan \
  --genotype data.bed --phenotype pheno.txt \
  --test wald --correction bh --device cuda:0

# Multi-trait mvLMM
torchgwas mvlmm-scan \
  --genotype data.bed --phenotype pheno.txt \
  --traits Y1,Y2,Y3 --test wald

# FarmCPU multi-locus (iterative FEM/REM)
torchgwas farmcpu-scan \
  --genotype data.bed --phenotype pheno.txt \
  --p-threshold 0.01 --max-qtns 20 \
  --bin-sizes 500000,5000000,50000000

# BLINK multi-locus (LD-clustering + BIC selection)
torchgwas blink-scan \
  --genotype data.bed --phenotype pheno.txt \
  --cutoff 0.01 --ld-threshold 0.7

# ═══════════════════════════════════════════════════
# GWAS SCANS — POLYPLOID (1 command)
# ═══════════════════════════════════════════════════

# Tetraploid potato (tests all gene-action models)
torchgwas poly-scan \
  --genotype potato_dosage.csv --phenotype pheno.txt \
  --ploidy 4 --gene-action all --test wald

# Hexaploid wheat with specific gene-action models
torchgwas poly-scan \
  --genotype wheat.vcf.gz --phenotype pheno.txt \
  --ploidy 6 --gene-action additive,1-dom,2-dom

# ═══════════════════════════════════════════════════
# GWAS SCANS — NOVEL MODELS (5 commands, Phases 15-19)
# ═══════════════════════════════════════════════════

# Multi-kernel LMM (additive + dominance + epistasis)
torchgwas mklmm-scan \
  --genotype data.bed --phenotype pheno.txt \
  --kernels additive,dominance --ploidy 2

# Gene-environment interaction (single-trait HetLMM or multi-trait GxELMM)
torchgwas gxe-scan \
  --genotype data.bed --phenotype pheno.txt \
  --env env.tsv --gxe-model het

# Set-based tests (SKAT/Burden/SKAT-O under LMM)
torchgwas set-scan \
  --genotype data.bed --phenotype pheno.txt \
  --regions genes.bed --set-test skat

# Bayesian variable selection (SuSiE or CAVI)
torchgwas bayes-scan \
  --genotype data.bed --phenotype pheno.txt \
  --method susie --n-signals 10

# Multi-environment trial (MET) GWAS
# Per-environment effects (default)
torchgwas met-scan \
  --genotype data.bed --phenotype pheno_env.txt \
  --env-cols E1,E2,E3

# Reaction-norm decomposition (stable alpha + GxE delta)
torchgwas met-scan \
  --genotype data.bed --phenotype pheno_env.txt \
  --parameterization reaction_norm

# FA(k) variance structure for large E
torchgwas met-scan \
  --genotype data.bed --phenotype pheno_env.txt \
  --vg-structure "fa(2)"

# Multi-kernel MET (additive + dominance)
torchgwas met-scan \
  --genotype data.bed --phenotype pheno_env.txt \
  --kernel-files dominance.npy epistasis.pt

# ═══════════════════════════════════════════════════
# PIPELINE (1 command — end-to-end orchestration)
# ═══════════════════════════════════════════════════

# Standard LMM pipeline
torchgwas pipeline \
  --genotype data.vcf.gz --phenotype pheno.txt \
  --impute mean --model lmm --test wald --correction bh \
  --device cuda:0

# Multi-environment pipeline
torchgwas pipeline \
  --genotype data.bed --phenotype pheno_env.txt \
  --model met --env-cols E1,E2,E3 --parameterization reaction_norm

# Pipeline supports all models: glm, lmm, mvlmm, farmcpu, blink, mklmm, gxe, met
# Pipeline supports all imputation: mean, beagle, impute5, minimac4, li-stephens, deep-learning

# ═══════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════

pytest tests/ -v                           # Full suite (2102 tests pass, 44 skipped)
pytest tests/test_golden_gemma.py -v       # GEMMA reference tests
pytest tests/test_golden_gapit.py -v       # GAPIT reference tests
pytest tests/test_golden_gwaspoly.py -v    # GWASpoly polyploid reference tests
pytest tests/test_format_roundtrip.py -v   # Format equivalence tests
pytest tests/test_met_reaction_norm.py -v  # MET reaction-norm tests
pytest tests/test_met_fa_reml.py -v        # MET FA(k) tests
pytest tests/test_met_multi_kernel.py -v   # MET multi-kernel tests
```

---

*This charter is the canonical reference for the TorchGWAS project. A new AI agent should treat it
as binding unless the user explicitly approves a change.*
