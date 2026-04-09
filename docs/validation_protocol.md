# TorchGWAS Validation Protocol

Reference: TorchGWAS AI Agent Handoff Charter, Section 16.

## Validation Types

### 1. Golden Tests

Tiny canonical datasets with saved reference outputs for fast truth checks.

| Dataset | Description | Samples | Variants | Reference Tool |
|---------|-------------|---------|----------|----------------|
| `golden/gemma_mouse_tiny` | Mouse BMI, balanced, diploid | 100 | 5,000 | GEMMA |
| `golden/gemma_lmm_unbalanced` | Unbalanced design, missing phenotypes | 80 | 3,000 | GEMMA |
| `golden/gemma_mvlmm_2trait` | 2 correlated continuous traits | 100 | 5,000 | GEMMA |
| `golden/gapit_farmcpu` | FarmCPU reference run | 200 | 10,000 | GAPIT3 |
| `golden/gapit_blink` | BLINK reference run | 200 | 10,000 | GAPIT3 |
| `golden/gwaspoly_potato` | Tetraploid potato chip color | 150 | 8,000 | GWASpoly |
| `golden/gwaspoly_hexaploid` | Hexaploid wheat yield | 100 | 5,000 | GWASpoly |

**Stored outputs per dataset**: log-likelihood, variance components (sig2_g, sig2_e, Vg, Ve), beta, SE, p-values, GRM (or GRM diagonal + sample off-diagonals).

### 2. Numerical Tolerances

| Quantity | Tolerance | Type |
|----------|-----------|------|
| Log-likelihood | < 1e-4 | Relative |
| Variance components (sig2_g, sig2_e) | < 1e-4 | Relative |
| Beta (effect size) | < 1e-4 | Relative |
| Standard error (SE) | < 1e-4 | Relative |
| P-value (p > 1e-4) | < 5e-5 | Absolute (match to 4th decimal) |
| P-value (p <= 1e-4) | < 1e-4 | Relative |
| GRM elements | < 1e-6 | Relative |
| Dosage values (format round-trip) | < 1e-12 | Absolute |
| Metadata (sample IDs, variant IDs) | exact | Exact match |

### 3. Reference Equivalence

| Reference Tool | Models Compared | Dataset Scale |
|----------------|----------------|---------------|
| GEMMA | Single-trait LMM (Wald, Score, LRT), mvLMM | 1K–10K samples |
| GAPIT3 | FarmCPU, BLINK, GLM | 1K–10K samples |
| PLINK 2.0 | GLM (logistic/linear) | 1K–10K samples |
| GWASpoly | Polyploid models (additive, 1-dom, diplo-additive, general) | 100–1K samples |

### 4. Format Round-Trip

Same dataset encoded in multiple formats must produce statistically equivalent scan results.

| Source Format | Target Format | Comparison |
|---------------|---------------|------------|
| PLINK BED | VCF | Dosage tensor ≤ 1e-12; p-values match to 4th decimal |
| PLINK BED | HapMap | Dosage tensor ≤ 1e-12; p-values match to 4th decimal |
| PLINK BED | CSV dosage | Dosage tensor ≤ 1e-12; p-values match to 4th decimal |
| PLINK BED | BGEN | Dosage tensor ≤ 1e-12; p-values match to 4th decimal |
| VCF | Zarr | Dosage tensor ≤ 1e-12; p-values match to 4th decimal |

### 5. Simulated Truth-Known

Synthetic genotype/phenotype with planted QTLs for power and FPR assessment.

| Scenario | Causal SNPs | Heritability | Expected |
|----------|-------------|--------------|----------|
| Simple additive | 5 of 10K | h2 = 0.5 | All 5 detected at 5e-8; FPR < 0.05 |
| Polygenic | 100 of 50K | h2 = 0.8 | Top hits overlap true set; lambda_gc ~ 1.0 after correction |
| No signal (null) | 0 | h2 = 0.0 | Uniform p-value distribution; lambda_gc ~ 1.0 |
| Population structure | 5 of 10K | h2 = 0.3 + stratification | LMM controls inflation; GLM does not |

### 6. CPU/GPU Equivalence

| Test | Tolerance |
|------|-----------|
| GRM: CPU vs GPU | < 1e-6 relative |
| Scan p-values: CPU vs GPU | ≤ 1e-4 relative |
| Variance components: CPU vs GPU | ≤ 1e-4 relative |
| Deterministic mode: run-to-run | Bitwise identical (with `torch.use_deterministic_algorithms(True)`) |

### 7. Stress Tests

| Scenario | Expected Behavior |
|----------|-------------------|
| Near-singular GRM (eigenvalue < 1e-10) | Jitter applied; no crash; warning logged |
| Monomorphic SNPs (MAF = 0) | Filtered out; NaN p-value or excluded from results |
| All-missing column (100% missingness) | Filtered out before scan |
| Rank-deficient covariate matrix | Detected and reported; graceful error or automatic column drop |
| Single sample per group | Warning; scan proceeds if statistically valid |
| Very large p-values (p → 1.0) | No overflow; correct rounding |
| Very small p-values (p < 1e-300) | Clamped to pvalue_floor; no underflow to 0 |

### 8. Performance Benchmarks

Wall-clock time and peak memory on standard datasets. Not correctness gates, but tracked for regression detection.

| Dataset Scale | Samples | Variants | Target (GPU) | Target (CPU) |
|---------------|---------|----------|--------------|--------------|
| Small | 1,000 | 10,000 | < 10s | < 60s |
| Medium | 10,000 | 100,000 | < 2min | < 30min |
| Large | 100,000 | 500,000 | < 30min | Baseline |

## Phase-to-Validation Mapping

| Phase | Required Validations |
|-------|---------------------|
| 0 — Architecture | All contracts importable; test stubs exist |
| 1 — I/O | Format detection; PLINK reader correctness; format round-trip |
| 2 — Preprocessing | Imputation correctness; QC filter behavior; standardization |
| 3 — GRM/Eigen | GRM symmetry, PSD; eigendecomposition reconstruction; GEMMA GRM match |
| 4 — LMM (CPU) | GEMMA golden tests (variance components, beta, SE, p-values) |
| 5 — LOCO | LOCO p-values match GEMMA LOCO output |
| 6 — mvLMM | GEMMA mvLMM golden tests (Vg, Ve, multi-trait p-values) |
| 7 — FarmCPU/BLINK | GAPIT3 golden tests |
| 8 — Polyploid | GWASpoly golden tests (all gene-action models) |
| 9 — Extra formats | Zarr, HDF5, BGEN readers; format round-trip |
| 10 — GPU | CPU/GPU equivalence; AMP dtype enforcement; deterministic mode |
| 11 — Permutation | Permutation p-value range; FWER control |
| 12 — Plots | Manhattan/QQ render without error |
| 13 — CLI/Export | CLI parsing; TSV/Parquet export correctness |
| 14 — Adv. testing | Weighted BH, adaptive permutation, Cauchy combination |
| 15 — Multi-kernel | Variance component recovery on simulated additive/dominance/epistasis |
| 16 — GxE-LMM | GxE interaction p-value calibration on simulated data |
| 17 — Set-based | SKAT/Burden/SKAT-O p-values match R SKAT reference |
| 18 — Bayesian VS | **DONE**: SuSiE + CAVI PIPs calibrated; validated against GEMMA BSLMM on MDP data; 23 tests, 8 benchmarks |
| 19 — GPU impute | **DONE**: Li-Stephens corr=0.906 on GWASpoly ground truth; DL corr=0.917; 31 tests, 7 benchmarks |
