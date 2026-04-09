# CHARTER_TorchGWAS_Guardian.md
**Project:** TorchGWAS  
**AI Agent Codename:** *TorchGWAS Guardian*  
**Version:** v0.3.5  
**Date:** 2026-04-03  
**Owner:** TorchGWAS Core Team (Computational Biology / Quantitative Genetics / HPC)

---

## v0.3 Redline Summary (what changed since v0.2)
This revision tightens statistical contracts and makes “worldwide robustness” concrete without expanding scope uncontrollably.

**Added / clarified**
- **Supported genotype formats (official):** PLINK1 (.bed/.bim/.fam), **VCF**, **BCF**, **HapMap** (.hmp/.hmp.txt/.hmp.gz), and **delimited text** genotype matrices (TSV/CSV).  
- **Explicit HapMap/TXT schema contracts:**
- **PloidyConfig (user-facing):** explicit public schema for ploidy/crop shortcuts, strictness, mismatch policy, and per-variant ploidy maps with required metadata fields.
- **Dosage convention invariant:** internal dosage **always counts ALT (A2/ALT)**; effect allele for reporting is a separate choice recorded in artifacts.
- **Multiallelic DS/GP enforcement:** multiallelic sites with DS/GP are **dropped by default** unless a validated dosage-splitting mode is explicitly enabled.
- **Declared `eps_dosage` default + scope:** default tolerance and where it is applied (QC, harmonization, heterozygosity/non-homozygote classification).
- **Benchmark runners contract:** explicit `bench/runners/*.py` wrappers and a standardized Parquet schema for all external baselines. required columns, encoding rules, and VCF-conversion eligibility for external imputation.
- **Format-aware imputation policy:** internal inference-safe imputation available for all formats; **external haplotype-aware imputation (e.g., Beagle) requires VCF/BCF** (TorchGWAS may convert HapMap/TXT → VCF when possible; otherwise warns/errors).
- **Trait family contract:** v0.x is **Gaussian** (continuous) for strict parity; binary/ordinal enter only when a calibrated GLMM strategy + acceptance gates are specified.
- **LOCO policy:** explicit default for LMM GWAS to mitigate proximal contamination.
- **Variant normalization & allele alignment:** multiallelic policy, allele flips/strand ambiguity flags, canonical REF/ALT representation.
- **Imputation tiers:** inference-safe vs QC convenience vs simulation-only; simulation-only is blocked by default in inference.
- **Stage gates:** explicit Gate A–D to prevent feature creep and to control “outperforms” claims.
- **Benchmarks:** added required stratification by **MAF/MAC bins** (rare variants) and a reserved tier for severe case-control imbalance (when GLMM is introduced).
- **API & artifact schema versioning:** reproducible artifacts require schema version and toolchain versions.

---

## v0.3.1 Patch Summary (tightening contracts; no scope expansion)
This patch adds enforceable internal contracts and reproducibility requirements, and removes ambiguity around LOCO, allele conventions, QC behavior, and external adapters.

**Added**
- **Internal Data Contract** (dosage/allele conventions; IDs; missing encoding; K/LOCO requirements)
- **LOCO “auto” definition** (exact behavior; when disabled; artifact fields)
- **External imputation reproducibility contract** (tool/version/params/reference panel hashes)
- **Polyploid-aware QC definitions**
- `AF = mean(dosage)/P`, `MAF = min(AF, 1-AF)`, `MAC = P * N_nonmiss * MAF`
- `HET_RATE = mean(0 < dosage < P | non-missing)` (generalizes diploid heterozygosity)
- `MONO` if `MAF==0` or numeric variance ~0

**QC inclusion policy** (PASS_QC vs analysis inclusion; monomorphic always dropped; sample-missingness opt-in)
- **Selection-aware inference labeling table** (FarmCPU/BLINK/CSMM/KO-LMM)
- **Benchmark harness version capture** for external baselines (BOLT/SAIGE/REGENIE/GEMMA/etc.)

## v0.3.2 Patch Summary (polyploid support + remaining tightening; no scope expansion)
This patch adds an explicit **PloidyConfig** (diploid/tetraploid/hexaploid), updates the internal dosage/QC contracts from `[0,2]` to `[0,P]`, and incorporates remaining execution-tightening recommendations:
- Ploidy-aware IO validation (VCF DS/GP/GT; dosage matrices)
- Ploidy-aware QC formulas (AF/MAF/MAC/HET/MONO) and PASS_QC inclusion rules
- Explicit `crop` shortcut mapping to ploidy defaults (potato/alfalfa=4; wheat=6)
- Explicit multiallelic handling rule and required VariantHarmonizationReport fields
- Explicit error/warn policy for ploidy mismatches and allele ambiguity
- Benchmark tier expansion to include tetraploid and hexaploid null + semi-synthetic scenarios


## v0.3.3 Patch Summary (format-aware imputation matrix + internal data contract formalized)
This patch resolves remaining ambiguity around **dosage conventions**, **polyploid handling**, and **format-specific imputation constraints** (no scope expansion; only enforceable contracts).

**Added**
- A dedicated **Internal Data Contract** section (dosage range \([0,P]\), missing encoding, allele conventions, multiallelic rules, fractional dosage policy).
- **Imputation Decision Matrix by Format**:
  - inference-safe internal imputation works for all formats,
  - **external haplotype-aware imputation (e.g., Beagle) requires VCF/BCF**,
  - TorchGWAS conversion rules (HapMap/TXT/PLINK → VCF) with explicit prerequisites and failure modes.
- **Polyploid GRM standardization rule** (\(\mathrm{Var}(dosage)\approx P p(1-p)\)) and tolerance checks for fractional dosage.
- Clarified that “HET_RATE” in polyploids is a **non-homozygote rate** (dosage not near 0 or \(P\)); optionally exported as `NONHOM_RATE` alias for clarity.

## 1. Mission
Build and maintain a modular, GPU-accelerated GWAS library (**TorchGWAS**) that:
1) matches statistical gold standards (GEMMA, GAPIT and modern LMM toolchains) for association testing,  
2) adds **routine “interpretation safety” novelty** (alignment/QC/covariate audit/causal-safety reporting) even when users run established GWAS models, and  
3) produces reproducible artifacts (Parquet + metadata) suitable for worldwide scientific use.

**North Star:** *Every result is both (a) statistically correct and (b) interpretation-safe.*

---

## 2. Scope

### 2.1 Core Association Engines (Must Be Correct)
These are the baseline engines TorchGWAS must run with high fidelity and calibration.

- **Univariate GWAS (Gaussian)**
  - GLM (OLS/GLS)
  - LMM/MLM with kinship/GRM
  - Tests: **Score**, **Wald**, **LRT** (fixed-VC default; refit-VC optional; see LOCO policy)
- **Multivariate GWAS (mvLMM; Gaussian)**
  - GEMMA-style Efficient Eigen-Decomposition (EED): \(V = K\otimes V_g + I\otimes V_e\)
  - Tests:
    - **Joint** df = d (trait-specific SNP effects)
    - **Common-effect** df = 1 (shared effect across traits)
- **Multi-locus GWAS (Established)**
  - FarmCPU (FEM/REM iteration; binning; pseudo-QTN selection)
  - BLINK (BIC-based selection + LD pruning logic)
  - Must log and export iteration history (selection steps, bins/LD pruning, pseudo-QTNs).

> **Routine novelty requirement (always on by default as warnings/diagnostics):** strict alignment, QC, covariate audit, and causal-safety reporting must be available regardless of which model is run.

### 2.2 Trait Families & Likelihood Contract (v0.x)
To protect correctness claims and avoid under-calibrated inference:
- **v0.x guarantees strict correctness for Gaussian traits** (continuous) under GLM/LMM/mvLMM.
- **Binary/ordinal/count traits (GLMM)** are **out of scope until** the charter specifies:
  - the exact approximation strategy (e.g., calibrated score test with robust variance, saddlepoint/SPA-style correction, or other well-defined method),
  - the null calibration acceptance gates (including severe imbalance regimes),
  - and the benchmark harness includes those regimes.

### 2.3 Default Policies (must be explicit)
- **LOCO (Leave-One-Chromosome-Out) policy:** default `loco="auto"` for univariate LMM GWAS when K is built from genome-wide SNPs to reduce proximal contamination.  
  - Artifacts must record: LOCO enabled/disabled; how K was built per chromosome; hash of SNP set used.
- **Definition of `loco="auto"`:** if K is derived from genome-wide variants and chromosome labels are available, TorchGWAS builds a per-chromosome `K_chr` by excluding **all** variants on the tested chromosome. If K is user-supplied (not variant-derived) or chromosome labels are unavailable, LOCO is disabled and this is recorded in metadata.
- **Variance-component fitting policy:**
  - default: fit VC once under null; scan with fixed VC (fast, stable; closest to standard practice),
  - optional: refit VC per SNP (slow; labeled “strict/experimental”).
- **Estimand labeling policy:**
  - any covariate adjustment that changes estimand (direct vs total) must be labeled in outputs.

---

## 3. New Model Families (Intended to Outperform on Specific Axes)
These models are designed to outperform existing SOTA on at least one of:
**power at fixed calibration**, **resolution under allelic heterogeneity**, **replicability/FDR control**, **robustness under confounding/colliders**, or **G×E / heterogeneity detection**.
All claims must be backed by the benchmark suite in Section 9.

### 3.1 OCF-LMM — Orthogonal Cross-Fit LMM (DML-valid inference; REGENIE-speed target)
**Goal:** achieve strong power and scalability from modern “predict-then-test” pipelines while keeping calibrated inference when nuisance models become rich.
- Structure: \(y = \alpha_j g_j + f(W) + u + \varepsilon\), with \(u\sim N(0,\sigma_g^2 K)\).
- Use cross-fitting + Neyman-orthogonal score to test \(\alpha_j\) with valid \(\chi^2_1\) null.
- Optional LOCO-K and/or genotype nuisance residualization \(E[g_j\mid W]\) to mitigate structure leakage.

**Expected wins:** improved power at fixed type-I error when polygenic background is well modeled; robust to “predictive proxy” leakage.

### 3.2 LDJ-LMM — LD-Block Joint LMM (Valid multi-locus power; “fine-mapping lite”)
**Goal:** increase power and stability in the presence of allelic heterogeneity without selection-induced inference pitfalls.
- Partition variants into LD blocks \(b\); test block effect \(\theta_b\) in:
  \(y = X\beta + G_b\theta_b + u + \varepsilon\).
- Use block score quadratic form with df \(r_b=\text{rank}(I_b)\):
  \(T_b = U_b^\top I_b^{{+}} U_b \sim \chi^2_{{r_b}}\).
- Optional within-block low-rank basis (randomized SVD / block-PCs) to bound df and computation.

**Expected wins:** higher locus-level power and more stable locus discovery than single-SNP tests under allelic heterogeneity.

### 3.3 KO-LMM — Knockoff Mixed Model (Group knockoffs by LD blocks; FDR guarantees)
**Goal:** provide discoveries with explicit FDR control under LD at the block/locus level.
- Generate knockoff genotypes \(\tilde G\) (group knockoffs per LD block).
- Compute LMM-aware importance statistics (e.g., \(W_b = |Z_b| - |\tilde Z_b|\)).
- Apply knockoff filter to control FDR at target \(q\).

**Expected wins:** improved replicability and controlled false discoveries vs p-value threshold + clumping.

### 3.4 CSMM — Calibrated Sparse Mixed Model (Multi-locus sparse effects with LMM calibration)
**Goal:** jointly model sparse large effects and polygenic background with correct inference.
- Model: \(y=X\beta + G\alpha + u + \varepsilon\), \(u\sim N(0,\sigma_g^2 K)\), with sparse \(\alpha\).
- Fit sparse effects in rotated space with either:
  - spike-and-slab (Bayesian/variational), or
  - penalized likelihood (L1) + **debiased / orthogonal** inference.

**Expected wins:** power and resolution gains in multi-causal regions while retaining calibration.

### 3.5 HetLMM — Heterogeneity-Aware LMM (random slopes / interaction GWAS)
**Goal:** detect G×E / ancestry / batch heterogeneity with valid inference.
- Model main effect + interaction:
  \(y = X\beta + g\gamma + (g\odot z)\eta + u + \varepsilon\).
- Tests: main (1 df), interaction (1 df), combined (2 df).

**Expected wins:** finds context-specific loci missed by standard GWAS; essential for MET and breeding pipelines.

### 3.6 LD-Conditional Score GWAS (Conditional persistence; proxy-risk reduction)
**Goal:** improve interpretability and local resolution without full Bayesian fine-mapping.
- Within each LD block, compute conditional score tests (condition on lead SNP(s) or block-PCs).
- Output marginal vs conditional p-values; “conditional persistence score”.

**Expected wins:** reduces proxy-as-causal interpretability errors; improves locus refinement.

### 3.7 Within-Family / Family-Aware GWAS (stratification & indirect effects diagnostics)
**Goal:** diagnose and mitigate stratification/indirect genetic effects when pedigrees/sibships exist.
- Implement within-family scan (family fixed effects or within-cluster demeaning).
- Output attenuation metrics vs unrelated GWAS and integrate into causal-safety scoring.

**Expected wins:** improved causal credibility; reduces confounding from population structure and indirect effects.

### 3.8 DML-GWAS for Modifiable Exposures (causal effect estimation module)
**Goal:** estimate causal effects of modifiable exposures \(T\) on \(Y\) using double/debiased ML with valid inference.
- Fit nuisance models \(E[Y\mid W]\), \(E[T\mid W]\) (GPU-accelerated).
- Cross-fitting + orthogonalization → \(\hat\tau\), SE, CI, p-value + diagnostics.

**Expected wins:** answers “what can I change?” with explicit assumptions and valid inference.

---

## 4. Routine Novelty Layers (Always Available; Default Warn-Only)
These features are **not optional add-ons**; they are routine novelty shipped with TorchGWAS, even for established models.

### 4.1 Causal-Safety Report (per SNP / per locus)
For each SNP and/or locus produce a report containing:
- **LD proxy risk**: LD block ID, block size, r²-to-lead, LD proxy score, conditional persistence
- **Confounding sensitivity**: stability across model families (GLM → LMM → MLM variants); within-family attenuation when available
- **Covariate-collider risk**: flags covariates likely downstream/heritable; labels estimand as “direct effect” when relevant
- **Redundancy attribution**: multiple SNPs/covariates carry similar information
- **Overall causal credibility**: heuristic 0–1 (non-causal; weights stored)

**Contract for “causal credibility”**
- It is a **risk/robustness score**, not probability.
- Components must be exported with explicit weights and invariants. Example invariant:
  - If within-family attenuation exceeds a high-risk threshold (configurable), overall credibility cannot be “high”.

### 4.2 Covariate Audit Mode (pre-GWAS; warn-only by default)
Before running GWAS:
- run quick **SNP→covariate GWAS** + **covariate→trait association**
- detect covariates that are heritable / genetically influenced
- label covariates as likely confounders vs mediators vs collider risks
- default: warn-only (no auto-dropping unless explicitly requested)

### 4.3 LD-Redundancy–Aware Explainability (LD-SHAP / block attribution)
For polygenic predictors and multi-locus models:
- cluster SNPs into LD blocks
- attribute importance at block level by default
- optionally distribute within block using conditional tests / credible-set-lite heuristics
- export block-level artifact + per-SNP linkage

---

## 5. Production Hardening (Must Be Robust Worldwide)


### 5.0 Internal Data Contract (Enforceable)
TorchGWAS must normalize *all* supported genotype inputs into a single internal representation so that QC, imputation, GRM construction, and association testing are consistent across formats.

**5.0.1 Dosage tensor**
- Core tensor: `G` with shape `(n_samples, n_variants)` in **dosage units**.
- Dosage is the **ALT-allele count** for a **biallelic** variant.
- **Invariant:** internal dosage always counts ALT (A2/ALT) alleles; any “effect allele” choice for reporting is separate and must be recorded.
- Allowed values:
  - **hard-call** dosage: integers in `{0,1,...,P}`
  - **fractional dosage** (e.g., DS/GP expectations): real values in `[0, P]`
- **Ploidy:** `PloidyConfig.P = P` (default 2 unless specified or inferred by `crop` shortcut).
- Inference requires **float64**; staging/transforms may use float32, but must cast back to float64 for inference.

**5.0.2 Missing encoding**
- Raw readers must preserve missingness using **exactly one** of:
  - `NaN` (preferred for DS/VCF/Zarr/HDF5/TXT dosage matrices), or
  - a sentinel value (PLINK1 raw mode may use `-1`).
- QC (missingness, MAF/MAC, etc.) must be computed on **raw** data prior to analysis-mode imputation.

**5.0.3 Allele conventions**
- Each variant must have canonical allele fields:
  - `A1` = REF, `A2` = ALT (or explicitly recorded alternative convention).
- Variant IDs must be stable; if rsID is missing use `CHR:POS[:REF:ALT]` as configured.
- **Ambiguity handling:** A/T and C/G strand ambiguity must be flagged when harmonizing across sources or references.

**5.0.4 Multiallelic handling**
- Multiallelic variants must be either:
  - split into biallelic records (default), or
  - dropped with explicit reason.
- The `VariantHarmonizationReport` must record the mapping from original site to derived biallelic records.
- **Dosage-aware constraint (DS/GP):** if a multiallelic site carries DS/GP (or any fractional dosage field), TorchGWAS must **drop the site by default** unless the user enables `--allow_multiallelic_split_ds` **and** a validated dosage splitter is configured. Rationale: naïve split can silently corrupt dosages and invalidate inference.

**5.0.5 Polyploid / fractional dosage sanity checks**
- For any analysis run, TorchGWAS must validate:
  - dosage range within `[0,P]` up to tolerance `eps_dosage`,
  - **default:** `eps_dosage = 1e-6` (configurable; recommend `1e-4` for noisy dosages),
  - **applied in:** (i) QC sanity checks, (ii) variant harmonization/validation, and (iii) genotype class rules (e.g., non-homozygote / heterozygosity-like rates),
  - homozygote thresholds use tolerance: `dosage <= eps` => homo-REF; `dosage >= P-eps` => homo-ALT,
  - warn/error on systematic violations (configurable policy).
- If input format cannot represent polyploid dosage correctly (e.g., diploid-only allele strings), TorchGWAS must warn and require an alternative format (VCF/BCF DS/GP or dosage matrix).

**5.0.6 GRM construction (ploidy-aware standardization)**
- For diploid, standardize with `2p(1-p)`; for ploidy `P`, standardize with `P p(1-p)` where `p = mean(dosage)/P`.
- GRM accumulation must record the exact standardization policy and SNP set used (and LOCO partitioning if applicable).



### 5.1 PloidyConfig (User-facing; enforceable)
TorchGWAS exposes a public configuration object that determines how ploidy is interpreted and validated.

**Fields**
- `ploidy`: one of `{2, 4, 6}` **or** `"from_vcf"` **or** `"per_variant_map"`.
- `crop`: optional shortcut (e.g., `"potato"`, `"alfalfa"`, `"wheat"`). If set, it may set default `ploidy` but must never override an explicit `ploidy`.
- `strict_ploidy`: boolean (default `True`). If `True`, any ploidy mismatch triggers ERROR (see mismatch policy).
- `ploidy_mismatch_policy`: `"error"` (default) or `"warn+override"` where override applies only when `ploidy_source` is explicit and consistent.
- `per_variant_map_path`: path/URI to a mapping file when `ploidy="per_variant_map"` (keyed by `CHR:POS[:REF:ALT]` or `SNP` ID).

**Resolution order**
1) explicit `ploidy` (integer)  
2) `ploidy="per_variant_map"` (map required)  
3) `ploidy="from_vcf"` (derive from VCF/BCF headers/GT token counts; DS ranges validated)  
4) `crop` shortcut default  
5) fallback `P=2`

**Required run metadata**
- `ploidy_source` (explicit / crop / from_vcf / per_variant_map / default)
- `ploidy_value` (scalar or “mixed”)
- `ploidy_mismatch_summary` (counts by variant/sample where detectable)
- `per_variant_map_hash` (when used) + number of mapped variants


### 5.1 Deterministic Alignment (Non-negotiable)
- **Strict alignment (deterministic):** genotype ↔ phenotype ↔ covariate sample orders are enforced and validated.
- Alignment must be logged and artifacts must contain:
  - input sample IDs hashes,
  - intersection size,
  - dropped samples + reasons (missing phenotype, missing covariates, sample-missingness threshold).

### 5.2 Missingness Checks (Pheno/Covariates/Genotypes)
- phenotype: DROP (default) or ERROR
- covariates: IMPUTE mean (default), DROP, or ERROR
- genotype:
  - per-variant missingness and per-sample missingness must be computed in **raw** mode before any imputation
  - optional sample filter by missingness threshold (explicit opt-in)

### 5.3 Genotype QC (Required outputs)
For each variant:
- AF, MAF, MAC, missingness rate, heterozygosity rate, monomorphic flag
- PASS_QC + FAIL_REASON bitmask
- QC thresholds must be recorded in run metadata.
**QC must be computed from raw genotypes**, prior to analysis-mode imputation.
**Polyploid-aware QC definitions**
- `AF = mean(dosage)/P`, `MAF = min(AF, 1-AF)`, `MAC = P * N_nonmiss * MAF`
- `HET_RATE = mean(0 < dosage < P | non-missing)` (generalizes diploid heterozygosity)
- `MONO` if `MAF==0` or numeric variance ~0

**QC inclusion policy**
- Default behavior: variants failing QC are excluded from inference (`write_filtered=False`) and are not scanned.
- If `write_filtered=True`, all variants are written, but failing variants must have association statistics set to NaN and `PASS_QC=false`.
- Monomorphic variants are always excluded from inference in default mode.
- Per-sample missingness filtering is opt-in; dropped sample IDs must be recorded.

### 5.4 Variant Normalization & Allele Alignment (Required)
TorchGWAS must implement and/or record:
- **Canonical allele representation** (REF/ALT, A1/A2) per format
- Multiallelic policy: split-to-biallelic (default) or drop with explicit reason
- Allele flip / strand ambiguity flags (A/T, C/G) when external reference is available or when merging sources
- Variant ID normalization (CHR:POS when rsID missing)
- Optional: export a “variant harmonization table” for reproducibility.

### 5.5 Genotype Formats (Official Support)
**Polyploid note:** PLINK1 `.bed` is treated as *diploid-first*. Polyploid workflows should prefer VCF/BCF with DS/GP or dosage matrices. If PLINK1 is used for polyploid, TorchGWAS requires explicit user confirmation and strict validation.
TorchGWAS must accept:
- **PLINK1**: .bed/.bim/.fam (SNP-major, chunked)
- **VCF**: .vcf / .vcf.gz
- **BCF**: .bcf
- **HapMap**: .hmp, .hmp.txt, .hmp.gz
- **Delimited text** genotype matrices: .txt/.tsv/.csv (explicit schema required; see below)

Each format must map into a unified internal dosage abstraction:
- hard calls (0..P; diploid special case P=2),
- or dosage floats (DS),
- with explicit missing representation (NaN or sentinel).



### 5.5.1 HapMap input contract (required schema + genotype encoding)
TorchGWAS accepts HapMap genotype files (e.g., `.hmp`, `.hmp.txt`, `.hmp.gz`) **only when the following schema requirements are met**.

**Required variant columns (names or accepted synonyms)**
- `rs#` (synonyms: `SNP`, `rsid`, `marker`, `Marker`)
- `alleles` (synonyms: `Alleles`, `A1/A2`, `A1A2`), typically in the form `A/G` (biallelic)  
- `chrom` (synonyms: `CHR`, `Chromosome`)
- `pos` (synonyms: `POS`, `Position`)

**Accepted sample genotype encodings (diploid default)**
- Two-letter calls: `AA`, `AG`, `GG`
- One-letter IUPAC heterozygotes: `R,Y,S,W,K,M` (mapped using the `alleles` field)
- Missing: `N`, `NN`, `.`, `-`, `NA` (configurable)

**Multiallelic markers**
- If `alleles` implies >2 alleles or if sample calls are inconsistent with biallelic encoding, markers are treated as multiallelic and handled by the global multiallelic policy (default: split-to-biallelic when possible; otherwise drop with explicit FAIL_REASON).

**Dosage mapping (internal contract)**
- Diploid: dosage = ALT allele count in `{0,1,2}`.
- ALT is defined as the **second allele** in the `alleles` field by default (configurable), and the mapping is recorded in the VariantHarmonizationReport.
- Polyploid HapMap is **not assumed**; polyploid workflows should supply VCF/BCF with DS/GP or a dosage matrix.

**External imputation eligibility**
- HapMap can be routed to external haplotype-aware imputers (e.g., Beagle) **only after conversion to VCF/BCF**.
- Conversion requires unambiguous biallelic alleles and valid coordinates (`chrom`, `pos`). If REF/ALT cannot be harmonized (e.g., strand ambiguity without a reference), TorchGWAS must refuse external imputation and fall back to Tier A inference-safe imputation or error based on user policy.

---

### 5.5.2 Delimited text genotype matrices (TXT/TSV/CSV) contract
TorchGWAS accepts `.txt/.tsv/.csv` genotype matrices **only with an explicit schema** so that downstream QC, allele alignment, and optional VCF conversion are well-defined.

**Required schema fields**
A TXT genotype source MUST define:
- **Orientation:** `samples×variants` or `variants×samples`
- **Sample identifiers:** either:
  - a leading `IID` column (preferred), or
  - a separate `samples` file listing IIDs in row order
- **Variant identifiers:** either:
  - VCF-like columns: `CHROM`, `POS`, optional `ID`, and allele columns (`REF`,`ALT`) **or** (`A1`,`A2`), or
  - a single `VARIANT_ID` column plus an `ALLELES` column (e.g., `A/G`) and optional `CHROM`,`POS`

**Accepted genotype encodings**
- **Dosage numeric** (recommended): real-valued dosage in `[0..P]` (diploid default `P=2`), allowing fractional values (e.g., dosages from imputation).
- **Hard-call genotype strings**: `0/0`, `0/1`, `1/1` (and polyploid forms like `0/0/1/1` when `P>2`)
- **Allele-pair strings**: `AA`, `AG`, `GG` (requires allele columns to interpret)
- Missing: `.`, `NA`, `N`, `-` (configurable)

**Ploidy**
- Default ploidy is `P=2`. If `P != 2`, the user must specify ploidy explicitly and TorchGWAS will validate dosage ranges and/or GT token counts accordingly.

**Allele alignment + VCF conversion**
- Internal association testing only requires a consistent A1/A2 (or REF/ALT) convention recorded in artifacts.
- **External imputation (e.g., Beagle) requires VCF/BCF** and therefore requires that TorchGWAS can emit a valid VCF:
  - `CHROM` and `POS` present,
  - biallelic alleles defined (`REF`,`ALT`) or (`A1`,`A2`) convertible to REF/ALT,
  - no unresolved strand ambiguity (A/T or C/G) unless a reference FASTA or an explicit harmonization map is provided.
- If these prerequisites are not met, TorchGWAS must not attempt external imputation and will fall back to Tier A inference-safe imputation or error depending on user policy.

**Schema persistence**
- The resolved TXT schema (including inferred defaults and column mappings) must be written into run artifacts to guarantee reproducibility.

---

### 5.5.3 External imputation decision matrix (format-aware enforcement)
TorchGWAS must enforce the following:
- **VCF/BCF**: eligible for external imputation directly (subject to biallelic normalization and metadata completeness).
- **HapMap**: eligible only after conversion to VCF/BCF and only if alleles/coordinates can be harmonized.
- **Delimited TXT/TSV/CSV**: eligible only after conversion to VCF/BCF and only if REF/ALT (or A1/A2) + coordinates are present and harmonizable.
- **PLINK1**: internal Tier A imputation supported; external imputation requires a reliable VCF conversion with valid REF/ALT + coordinates (not guaranteed without reference/harmonization), therefore external imputation is **disabled by default** unless the user provides the required harmonization inputs.


### 5.6 Format-Aware Imputation (Three Tiers)
TorchGWAS supports **imputation policies**, but must clearly separate their intended use:

**Tier A — Inference-safe (default for analysis mode)**
- MEAN dosage imputation (per variant)
- MODE / “major allele” imputation (per variant)
Applies to **all** input formats.

**Tier B — QC convenience (optional)**
- lightweight KNN over samples (small n only; disabled by default)
- other deterministic convenience methods
Must be labeled and recorded.

**Tier C — Simulation-only / non-inferential (blocked by default in inference)**
- RANDOM_HWE and similar stochastic imputations
Requires explicit opt-in flag (e.g., `--allow_noninferential_impute`) and must label outputs as simulation-only.

**External imputation adapters (optional; format constrained)**
- Haplotype-aware imputation tools (e.g., Beagle) are supported only via an adapter interface and **require VCF/BCF** inputs.
  - If user provides HapMap/TXT, TorchGWAS may **convert to VCF** (biallelic normalization required) before calling external imputation.
  - If conversion is not possible (e.g., ambiguous alleles, missing required metadata), TorchGWAS must warn and fall back to Tier A or error based on configuration.

- **ExternalImputeReport (required when external imputation is used):**
    - tool name + version, command/parameter hash, reference panel name + version/hash, genetic map version/hash (if applicable),
    - input file hashes (pre- and post-normalization), output file hash,
    - phasing status and sample/variant counts before/after.

> Important constraint: external imputation tools may not accept certain formats directly (e.g., “TXT” matrices). TorchGWAS must not claim otherwise; it must convert to a supported format (typically VCF/BCF) or skip external imputation.

#### 5.6.1 Imputation Decision Matrix by Input Format (Required)
TorchGWAS must choose an imputation path that is both *compatible* with the file format and *appropriate* for inference.

| Input format | Native representation | Internal Tier A (mean/mode) | External haplotype-aware (e.g., Beagle) | Notes / prerequisites |
|---|---|---:|---:|---|
| PLINK1 (.bed/.bim/.fam) | hard calls (diploid-first) | ✅ | ⚠️ via conversion | External requires conversion to VCF; conversion must preserve REF/ALT and biallelic normalization. |
| VCF (.vcf/.vcf.gz) | GT + optional DS/GP | ✅ | ✅ | Prefer DS/GP when present; otherwise compute dosage from GT. |
| BCF (.bcf) | GT + optional DS/GP | ✅ | ✅ | Same as VCF; BCF recommended for large datasets. |
| HapMap (.hmp/.hmp.txt/.hmp.gz) | allele strings (often diploid) | ✅ (if dosage derivable) | ⚠️ via conversion | External requires conversion to VCF; HapMap must provide allele fields and unambiguous biallelic mapping. Polyploid HapMap is generally unsupported unless explicit dosage encoding is present. |
| TXT/TSV/CSV dosage matrix | numeric dosage or calls | ✅ | ❌ (direct) / ⚠️ via conversion | External tools typically do not accept raw matrices; conversion to VCF requires: chromosome/position and REF/ALT alleles per variant (or a trusted reference mapping). If absent, external imputation must be disabled. |

**Conversion rules**
- TorchGWAS may convert PLINK/HapMap/TXT → VCF for external imputation only if:
  1) variants can be normalized to biallelic records,
  2) REF/ALT alleles are defined and harmonized,
  3) sample IDs are stable and consistent,
  4) output VCF passes strict validation (dosage range, allele consistency).
- If prerequisites fail, TorchGWAS must:
  - fall back to Tier A, or
  - error (based on user config), but never silently proceed with an invalid external imputation.


### 5.7 Out-of-Core IO + Hardware-Agnostic Compute
- unified chunk iterator contract yielding \((n, m)\) genotype blocks with metadata
- torch.Tensor everywhere; `.to(device)` works end-to-end
- memory-safe: chunked IO; no assumption that genotype fits in RAM/VRAM
- deterministic numerics: stable jitter policy, symmetry enforcement, seeded stochastic components

---

## 6. Non-Negotiables

### 6.1 Statistical Correctness
- **float64** for inference: REML/ML, solves, logdets, test statistics, p-values
- AMP permitted only for non-inferential operations (GRM accumulation, staging, standardization) with explicit cast back to float64.
- Covariate adjustment changes estimand: outputs must label **total vs direct** effect implications when flagged by Covariate Audit.
- Any procedure that performs selection (multi-locus, model-family screening) must use:
  - debiased/orthogonal inference, or
  - cross-fitting, or
  - knockoff-style calibration,
  otherwise TorchGWAS must not claim calibrated p-values for post-selection statistics.

### 6.1.1 Selection-aware inference labeling (required)
Some methods select variables during modeling; TorchGWAS must label outputs accordingly:

| Method | Default output meaning | When p-values are claimable as calibrated |
|---|---|---|
| FarmCPU / BLINK | Post-selection association summaries | Only if debiasing / orthogonal inference is used and passes null calibration gates |
| CSMM | Sparse mixed-model estimates | Only with debiased / orthogonal score inference (Gate D+) |
| KO-LMM | FDR-controlled block/locus discoveries | Uses knockoff guarantees; not interpreted as marginal p-values |
| LD-Conditional | Conditional associations within LD blocks | Calibrated if conditioning set is pre-specified and gates pass |

### 6.2 Communication Safety
- Never claim causality from association.
- “Causal credibility” is a **heuristic risk score**, not a probability.

### 6.3 API Stability & Artifact Schema Versioning
- Semantic versioning required (vMAJOR.MINOR.PATCH).
- Artifacts must include:
  - artifact schema version,
  - TorchGWAS version,
  - dependency versions (torch, numpy, IO backends),
  - device info and dtype policy.
- Breaking artifact schema changes require MAJOR bump.

---

## 7. Deliverables

### 7.1 Code Deliverables
1) Models:
   - SingleTraitLMM (AI-REML, Score/Wald/LRT; LOCO policy)
   - MultiTraitLMM (EED likelihood, AI-REML, df=d and df=1 tests)
   - FarmCPU, BLINK iterative framework (with selection logs and explicit inference labeling)
   - OCF-LMM, LDJ-LMM, KO-LMM, CSMM, HetLMM, LD-conditional tests, within-family GWAS, DML module
2) IO:
   - PLINK1 bed SNP-major reader (chunked + sample subset)
   - VCF/BCF reader (GT + DS; chunked)
   - HapMap reader (biallelic mapping + metadata extraction)
   - Delimited text genotype reader (schema-driven)
   - Zarr backend (optional but supported)
   - unified loader factory `--geno` with auto format detection
3) Pipelines:
   - TorchGWASConfig unified config object
   - one-call workflows (format-agnostic): `TorchGWAS.run_gwas(geno=..., ...)`
4) Reports:
   - CovariateAuditReport (Parquet + JSON summary)
   - CausalSafetyReport (per SNP / locus; Parquet columns)
   - LDBlockReport (block attribution + redundancy stats)
   - VariantHarmonizationReport (allele alignment / split / flip flags)
5) Output artifacts (TSV or Parquet; Parquet recommended):
   - SNP metadata: CHR, POS, A1/A2, CM, SNP
   - association: BETA, SE, CHI2, P (+ U/VarU for score test)
   - QC: AF, MAF, MAC, N_MISS, MISS_RATE, HET_RATE, MONO, PASS_QC, FAIL_REASON
   - causal-safety: LD_PROXY_SCORE, STRAT_SENS_SCORE, COV_COLLIDER_RISK, CAUSAL_CREDIBILITY
   - schema metadata: config, VC, convergence, timings, versions, device, LOCO status.

### 7.2 Stage Gates (scope control)
- **Gate A (Parity Core):** Gaussian GLM + univariate LMM parity vs reference tools; strict alignment/QC; TSV+Parquet; golden tests.
- **Gate B (mvLMM Parity):** mvLMM EED likelihood + AI-REML + df=d/df=1 tests; golden tests.
- **Gate C (Interpretation Safety Baseline):** LD blocks + conditional tests + Covariate Audit + Causal-Safety report foundations.
- **Gate D (First “Outperform” Claim):** LDJ-LMM (recommended first) or OCF-LMM shows improved power/robustness under benchmark charter with matched calibration.

---

## 8. Validation Deliverables
- Golden tests vs GEMMA/GAPIT for overlapping methods:
  - \(|p_{torch} - p_{ref}| \le 5\times10^{-5}\) (4th decimal stable)
  - variance components within tight tolerances
- Integration tests:
  - strict alignment mismatch detection
  - QC + imputation consistency across **all supported formats** via dosage abstraction
  - allele alignment and multiallelic policy tests
- Benchmarks:
  - SNP/s throughput
  - GRM build throughput
  - GPU memory profile by chunk size

---

## 9. Benchmark Charter (How We Prove “Outperforms”)
TorchGWAS must include a reproducible benchmarking suite that compares:
- established baselines (association + multi-locus + scalable biobank LMMs)
- TorchGWAS baselines (GLM/LMM/mvLMM/FarmCPU/BLINK)
- new model families (OCF-LMM, LDJ-LMM, KO-LMM, CSMM, HetLMM, LD-conditional, within-family, DML)

### 9.1 Baselines to Include (SOTA Reference Set)
- Biobank-scale LMM association: BOLT-LMM, fastGWA, SAIGE, REGENIE, scalable variational/ML-flavored methods (Quickdraws-style)
- Multi-locus heuristics: FarmCPU, BLINK
- Conditional/FDR-controlled discovery: KnockoffGWAS (group knockoffs)
- Multi-trait: mvLMM (GEMMA-style) and MTAG (summary-stat based multi-trait)

> Baselines may be run via wrappers/subprocess in the benchmark harness, with explicit version capture and configuration logging.

**Benchmark harness requirement (versions & commands)**
- For every external baseline run, artifacts must include: tool name + version, command/parameter hash, input file hashes, and runtime environment (CPU/GPU, threads).

### 9.2 Dataset Tiers (to avoid self-delusion)
1) **Pure null** (real genotypes; simulated phenotype with no genetic effect)
2) **Semi-synthetic** (real genotypes; simulated phenotypes with known QTNs)
   - single causal per locus
   - **allelic heterogeneity** (2–5 causal variants in LD block)
   - polygenic background + sparse large effects
   - stratification + relatedness
   - heterogeneity / G×E scenarios
   - **rare variant / low MAC regimes** (must stratify results by MAF/MAC bins)
3) **Real-world replication**
   - split-sample replication
   - cross-environment replication (breeding MET)
   - cross-ancestry replication when available

> **Reserved tier for GLMM introduction:** when binary/ordinal is added, the suite must include severe case-control imbalance null calibration.

### 9.3 Metrics (define “outperform” precisely)
**Calibration**
- empirical type-I error at \(\alpha\in\{10^{-2},10^{-3},10^{-4},5\times10^{-8}\}\)
- QQ deviation metrics + \(\lambda_{GC}\) (and stratified \(\lambda\) by MAF/MAC bins)
**Power**
- TPR at fixed FDR (or fixed genome-wide \(\alpha\))
- recall/precision for causal loci (semi-synthetic)
- KO-LMM: achieved FDR vs target q and power
**Resolution**
- distance to true causal, credible-set-lite size, conditional persistence
**Robustness**
- sensitivity across model families (GLM → LMM variants)
- within-family attenuation diagnostics
- environment/stratum stability
**Efficiency**
- runtime, SNP/s throughput, peak memory, device utilization

### 9.4 Acceptance Gates (minimum bar)
- Well-calibrated null (no systematic inflation)
- Competitive or improved power at matched calibration
- Clear wins in at least one targeted scenario:
  - OCF-LMM: richer nuisance models without calibration loss
  - LDJ-LMM/CSMM: allelic heterogeneity power and stability
  - KO-LMM: FDR-controlled reliable discoveries
  - HetLMM: improved detection of context-specific loci

---


### 9.5 External baseline runner contract (operational; non-optional)
To prevent ad hoc benchmarking, TorchGWAS must ship explicit wrappers for external baselines under:

- `bench/runners/gemma.py`
- `bench/runners/gapit.R` (or `bench/runners/gapit.py` invoking R)
- `bench/runners/saige.py`
- `bench/runners/regenie.py`
- `bench/runners/bolt_lmm.py` (where licensed/available)
- `bench/runners/fastgwa.py` (where available)
- `bench/runners/knockoffgwas.py` (or group-knockoff baseline)

**Runner requirements**
- Each runner must:
  - capture exact command, parameters, versions, and inputs hashes,
  - emit a standardized **Parquet** results file with the TorchGWAS benchmark schema:
    - variant keys: `chrom,pos,snp,ref,alt`
    - association: `beta,se,stat,p` (and df where applicable)
    - runtime: `wall_time_s, peak_rss_mb`
    - provenance: `tool, tool_version, command, config_hash`
- The benchmark harness must include schema validators to fail fast if a runner deviates.


## 10. Quality Gates (Definition of Done)
A feature is “done” only if:
1) unit + integration tests pass  
2) artifacts (TSV/Parquet + metadata) are correct and complete  
3) float64 inference is used and numerics are stable (no silent NaN propagation)  
4) interpretation-safety outputs are labeled heuristic and do not imply causality.

---

## 11. Operating Model for the AI Agent
- Plan → implement → test → document in small increments.
- Start with correctness, then optimize.
- Favor composable modules and stable APIs.
- Defaults policy: if behavior affects estimands or could surprise users, default to **warn-only** and require explicit opt-in for automatic changes.
- Every run produces: results (TSV or Parquet), JSON summary, and clear logs (alignment, missingness, QC, sensitivity suite).

---

## 12. Roadmap (Recommended Execution Order)
1) Gate A: Core univariate correctness + format-agnostic IO + strict pipeline + TSV/Parquet + golden tests  
2) Gate B: mvLMM likelihood + AI-REML + df=d/df=1 tests  
3) Gate C: Covariate Audit Mode + Causal-Safety foundations (LD blocks + stability metrics)  
4) Gate D: LDJ-LMM + LD-conditional tests (fine-mapping lite)  
5) OCF-LMM (cross-fit polygenic residualization)  
6) CSMM (calibrated sparse mixed model)  
7) KO-LMM (group knockoffs)  
8) HetLMM + within-family mode  
9) DML module for modifiable exposures
