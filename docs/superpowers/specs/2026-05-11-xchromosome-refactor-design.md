# Sex-Chromosome (X / Y / PAR / MT) Handling — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven)
- **Date:** 2026-05-11
- **Branch:** `modernization/specs`
- **Master spec:** [`2026-05-11-modernization-master-design.md`](2026-05-11-modernization-master-design.md)
- **Research brief:** [`../research/xchrom-research.md`](../research/xchrom-research.md)
- **Phase number:** none — horizontal refactor track
- **Scope class:** FULL PARITY (per master spec §2)
- **Living spec.** Tier A is fixed; Tier B / C may be revised at implementation kickoff.

---

## 1. Goal & scope

Add complete sex-chromosome (chrX, chrY, PAR1/PAR2, mtDNA) support to TorchGWAS so it can perform human GWAS without silently dropping or mis-handling non-autosomal variants. Today the codebase has zero sex-chromosome handling — `_parse_fam` in `torchgwas/io/plink.py:172–193` discards the `.fam` sex column entirely, and no chr-aware code path exists anywhere under `torchgwas/`. This is a credibility gap: any human-GWAS user who runs the toolkit on a chrX variant gets results that silently use autosomal MAF / HWE / additive coding, which is statistically wrong (Clayton 2008 [C1]).

**In scope (FULL PARITY):**

- chrX non-PAR with all four PLINK 2.0 `--xchr-model` values (0=skip / 1=random skewing / 2=uniform skewing / 3=sex-by-genotype interaction); citation: PLINK 2.0 docs [C4].
- PAR1 and PAR2 (autosomal-style handling) with per-build coordinate tables (GRCh37 / GRCh38); citation: Ensembl Human PARs [C16].
- chrY (males-only, hemizygous additive); citation: PLINK 2.0 sex filters [C7].
- mtDNA / chrM (haploid additive, optional heteroplasmy filtering); citation: Yonova-Doing et al. 2021 [C19], Hägg et al. 2021 [C20].
- Sex-aware MAF (denominator $2n_F + n_M$ for chrX); citation: PLINK 2.0 `--freq` [C9].
- Sex-aware HWE (females-only for chrX, undefined for chrY/MT); citation: PLINK 2.0 `--hardy` [C10].
- Two GRM strategies for chrX: (i) **default for LMM scanners** — separate X kernel via `MultiKernelLMM` (the GCTA approach [C14]); (ii) **fallback for stepwise scanners (FarmCPU, BLINK)** — single-kernel autosomal GRM with chrX scanned as fixed-effect SNPs (the BOLT-LMM approach [C12]).
- Sex-stratified meta-analysis as an alternative to the dosage-coding models (Stouffer / Z-score combination of male-only and female-only scans); citation: König et al. 2014 [C3] §"Sex-stratified analysis".
- Per-chromosome multiple-testing pool option (separate Bonferroni for chrX); citation: König et al. 2014 [C3].
- CLI flags on every V1 scan command: `--xchr-model {0,1,2,3}`, `--split-par {hg19,hg38,coords}`, `--filter-males`, `--filter-females`, `--xchr-grm-kernel {separate,joint,none}`, `--mt-heteroplasmy-cutoff <float>`.
- Visualization: chrX/Y/MT colored distinctly in Manhattan plots; chrX flagged when LD blocks are computed (LD block algorithms assume diploid HWE — chrX must be warned-and-skipped).

**Explicitly out of scope (deferred):**

- Gene-level XCI escape per Tukiainen et al. 2017 [C2] Supplementary Tables 8–13 — Tier C nice-to-have only; default uses uniform-skewing.
- XTR (X-transposed region) special handling beyond chrX defaults — Tier C.
- Sex-by-environment-by-genotype three-way interactions — out of scope (requires separate model class).
- Non-human polyploid sex chromosomes (e.g., ZW systems in birds, fish) — out of scope; the refactor uses a human-default chr table but the canonicalization layer is generic enough to extend later.
- PolyOrigin-style polyploid F1 phasing on chrX — N/A (PolyOrigin is autopolyploid-only per Phase 56 spec).

## 2. Algorithm reference

### 2.1 X-inactivation models (chrX non-PAR)

Notation: $g_{ij} \in \{0,1,2\}$ is the additive allele-count dosage at SNP $j$ for individual $i$; $s_i \in \{F, M\}$ is sex; $\beta_j$ is the per-allele effect.

**Uniform skewing model (PLINK `--xchr-model 2`, default):** male hemizygote treated as homozygous-equivalent. This is the BOLT-LMM and PLINK 2.0 `--glm` default. Equation from Clayton 2008 [C1] equation (2):

$$
\tilde g_{ij} = \begin{cases} g_{ij}, & s_i = F \\ 2\,g_{ij}, & s_i = M \end{cases}
$$

Resulting effective dosages: females $\in \{0, 1, 2\}$, males $\in \{0, 2\}$. Same per-allele effect $\beta_j$ across sexes.

**Random skewing model (PLINK `--xchr-model 1`):** no dosage compensation; male hemizygote = single allele, female heterozygote = single allele. Equation from Clayton 2008 [C1] equation (1):

$$
\tilde g_{ij} = g_{ij}
$$

Resulting effective dosages: females $\in \{0, 1, 2\}$, males $\in \{0, 1\}$.

**Skip model (PLINK `--xchr-model 0`):** all chrX variants dropped from the regression.

**Sex-by-genotype interaction model (PLINK `--xchr-model 3`):**

$$
E[y_i] = \alpha + \beta_j g_{ij} + \gamma\, s_i + \delta_j (s_i \times g_{ij})
$$

with $\delta_j$ as the per-SNP sex-genotype interaction term. PLINK 2.0 docs [C4].

**Default choice:** `--xchr-model 2` (uniform skewing). This is the default in BOLT-LMM v2.4 [C12] and PLINK 2.0 `--glm` [C4]; matches the most common published practice. The Clayton 2008 paper [C1] shows uniform-skewing has higher power under XCI than random-skewing for the typical case where one X allele is silenced.

### 2.2 PAR coordinates (per build)

Pinned per assembly per Ensembl Human PARs [C16]:

| Region | Build | chrX coords (1-based) | chrY coords (1-based) |
|---|---|---|---|
| PAR1 | GRCh37 / hg19 | 60,001 – 2,699,520 | 10,001 – 2,649,520 |
| PAR1 | GRCh38 / hg38 | 10,001 – 2,781,479 | 10,001 – 2,781,479 |
| PAR2 | GRCh37 / hg19 | 154,931,044 – 155,260,560 | 59,034,050 – 59,363,566 |
| PAR2 | GRCh38 / hg38 | 155,701,383 – 156,030,895 | 56,887,903 – 57,217,415 |

These tables live in `torchgwas/io/chrom.py` as constants `PAR1_GRCH37`, `PAR1_GRCH38`, `PAR2_GRCH37`, `PAR2_GRCH38`. The `is_par(chr, pos, build)` predicate uses them. Spec MUST cite Ensembl source URL in the docstring of each constant.

### 2.3 chrY model (males-only haploid)

For males, $g_{ij} \in \{0, 1\}$ (haploid additive). Females receive `NA` for chrY variants. No GRM contribution from chrY (every reference tool — GCTA [C14], BOLT [C12], REGENIE [C11], SAIGE [C13] — excludes chrY from the GRM). Y-PAR variants are deduplicated onto X-PAR (PLINK convention; PLINK 2.0 `--split-par` [C5]).

### 2.4 mtDNA model (haploid, optional heteroplasmy filtering)

For all individuals (mtDNA is maternally inherited), $g_{ij} \in \{0, 1\}$ if a hard call is made, or $g_{ij} \in [0, 1]$ if heteroplasmy fraction is preserved as continuous dosage. No GRM contribution from mtDNA itself; the autosomal GRM is reused for the random-effect covariance. mtDNA PCs may be added as fixed-effect covariates to control for haplogroup structure. Reference: Hägg et al. 2021 [C20] §Methods.

Heteroplasmy filtering (Yonova-Doing et al. 2021 [C19] §Methods): variants with heteroplasmy fraction > `--mt-heteroplasmy-cutoff` (default `0.05` per Yonova-Doing) are dropped, OR retained with continuous dosage if `--mt-heteroplasmy-mode continuous`.

### 2.5 MAF formula on sex chromosomes

Autosomal MAF: $\hat p = \sum_i g_{ij} / (2n)$.

chrX non-PAR MAF (PLINK 2.0 `--freq` [C9]):

$$
\hat p_X = \frac{\sum_{i:\,s_i=F} g_{ij} + \sum_{i:\,s_i=M} g_{ij}}{2 n_F + n_M}
$$

chrY MAF (males-only):

$$
\hat p_Y = \frac{\sum_{i:\,s_i=M} g_{ij}}{n_M}
$$

mtDNA MAF (treats all samples as single-allele):

$$
\hat p_{MT} = \frac{\sum_i g_{ij}}{n}
$$

PAR variants use the autosomal $2n$ denominator (males are diploid in PAR; PLINK 2.0 `--split-par` semantics [C5]).

### 2.6 HWE on sex chromosomes

PLINK 2.0 `--hardy` [C10]:
- chrX non-PAR: HWE computed on females-only (males cannot be heterozygous).
- chrY, mtDNA: HWE undefined; emit `NA`.
- PAR: HWE computed on full sample (autosomal-style).

### 2.7 GRM contributions

**Default for LMM scanners (GCTA [C14] approach):** two-kernel.
- Autosomal kernel $G_{auto}$ from autosomal SNPs (existing `linalg/kinship.grm_vanraden`).
- X kernel $G_X$ from chrX non-PAR SNPs via new `linalg/kinship.grm_vanraden_xchr(G, sample_sex, model="uniform"|"random")`. The `model` argument controls dosage compensation per Clayton 2008 [C1].
- Both fed to `MultiKernelLMM` as separate variance components: $V = \sigma^2_{auto} G_{auto} + \sigma^2_X G_X + \sigma^2_e I$.
- Justification: GCTA explicitly recommends a two-GRM mixed model so the X variance component is decomposed and the autosomal GRM is not contaminated; see GCTA docs [C14] section "Making a GRM from the X chromosome".

**Fallback for stepwise scanners (FarmCPU, BLINK):** single-kernel.
- Only autosomal $G_{auto}$ is built.
- chrX scanned as fixed-effect SNPs with the chosen `--xchr-model` dosage coding.
- Justification: FarmCPU and BLINK are iterative pseudo-QTN methods (Liu et al. 2016 BLINK; Liu et al. 2016 FarmCPU); they don't natively accept multiple variance components. The BOLT-LMM v2.4 manual [C12] §"Best practices" makes the same recommendation: exclude chrX from the GRM and scan as fixed effects.

User-exposed flag: `--xchr-grm-kernel {separate,joint,none}`. `separate` = two-kernel (default for LMM scanners); `joint` = single-kernel autosomal-only with chrX fixed-effect-scanned (default for FarmCPU/BLINK; legal for LMM scanners as a BOLT/REGENIE-equivalent override); `none` = chrX excluded from any kernel and from any scan (equivalent to `--xchr-model 0`).

### 2.8 Sex-stratified meta-analysis (alternative pathway)

Per König et al. 2014 [C3] §"Sex-stratified analysis": run the chrX scan twice (males-only and females-only), then combine per-SNP $p$-values via Stouffer Z-score combination:

$$
Z_{combined} = \frac{Z_M \sqrt{N_M} + Z_F \sqrt{N_F}}{\sqrt{N_M + N_F}}
$$

with $Z_M = \Phi^{-1}(1 - p_M)$ and analogously for females. This is the model-free alternative to the dosage-coding models in §2.1 and is the conservative choice when XCI status of the variant is unknown. Exposed via flag `--xchr-strategy {dosage,stratified}`; default `dosage` (Clayton-style).

## 3. Module layout

### 3.1 New files

| Path | Purpose |
|---|---|
| `torchgwas/io/chrom.py` | Chr canonicalization + PAR coordinate tables + `is_par`/`is_xchr_nonpar`/`is_y`/`is_mt` predicates. |
| `torchgwas/linalg/kinship_xchr.py` | `grm_vanraden_xchr(G, sample_sex, model)` and helpers. Separate file (not in `kinship.py`) to keep autosomal V1-equivalence isolated from X-chrom changes. |
| `torchgwas/scan/chunk_context.py` | `ChunkContext` dataclass passed to `BaseModel.score_chunk`. |
| `tests/fixtures/xchrom_diploid.bed/bim/fam` | Mini fixture with PAR1, chrX non-PAR, PAR2, chrY, chrM variants. |
| `tests/test_xchrom_invariance.py` | R-MOD-1 regression net: `sample_sex=None` + autosomal labels reproduces pre-refactor outputs. |
| `tests/test_xchrom_parity_plink2.py` | Tier 2 parity vs PLINK 2.0 `--glm --xchr-model 2`. |
| `tests/test_xchrom_qc.py` | Tier 1 unit tests on sex-aware MAF / HWE formulae. |

### 3.2 Public-API extensions

`torchgwas/models/base.py` — add `SampleMeta` dataclass mirroring the existing `VariantMeta`. Carries `sample_ids: list[str]`, `sample_sex: Tensor[int8] | None` (1=male, 2=female, 0=unknown per PLINK convention), `sample_pheno: Tensor | None`. Backward-compat: existing call sites that don't pass `SampleMeta` get `sample_sex=None` and the scan proceeds as autosome-only (zero behavior change vs pre-refactor).

`torchgwas/scan/base.py` — `BaseModel.score_chunk` protocol gains a `chunk_context: ChunkContext` keyword argument with a default value. The default `ChunkContext(is_xchr_nonpar=False, is_y=False, is_mt=False, is_par=False, sample_sex=None)` makes existing calls behave exactly as before.

### 3.3 Files modified (per research brief §11; refined count)

- **io** (10 files modified, 1 new): `plink.py`, `plink2.py`, `vcf.py`, `bgen.py`, `hapmap.py`, `zarr.py`, `hdf5.py`, `numeric.py`, `detect.py`, `validate.py`, `regions.py`. New: `chrom.py`. Each reader adds sex parsing + chr canonicalization.
- **preprocess** (6 files): `qc.py` (sex-aware MAF / HWE), `standardize.py` (per-sample-per-variant ploidy tensor support; sex-aware male-doubling for chrX), `covariates.py` (auto-add sex when chrX is in scan), `dosage_call.py` / `dosage_uncertainty.py` (verify haploid-aware EM), `phase.py` (mark N/A for human chrX).
- **linalg** (4 files): `kinship.py` (no behavior change for autosomes; entry hook for X kernel), `kinship_advanced.py` (LROLMM × chrX interplay), `kinship_polyploid.py` (verify not invoked on human chrX), `sparse_grm.py` (sparse-GRM X kernel for SAIGE-style scans). New: `kinship_xchr.py`.
- **models** (~35 files): every `BaseModel` subclass that takes `score_chunk(G, ...)` gets a new `chunk_context: ChunkContext = ChunkContext()` kw-arg. Tier A: V1 core + categorical (17 models from research brief §10). Tier B: haplotype/RR/MET families. Tier C: polyploid-specific FA(k) fits marked "diploid-X-aware not applicable" with explicit `NotImplementedError("X-chromosome handling not applicable for autopolyploid model")`.
- **scan** (3 files): `unified.py` (chrom-boundary chunk splits via `split_at_chrom_change=True`), `adapters.py` (sex-mask helpers), `strategies.py` (sex-stratified scan strategy).
- **stats** (2 files): `multipletesting.py` (separate chrX pool option), `genomic_control.py` (per-chr λGC).
- **cli.py** (1 file): X-chrom flags on every V1 scan subcommand.
- **results** (1 file): `tables.py` emits per-chr summary rows + flags chrX with model used.
- **viz** (1 file): `_manhattan.py` colors chrX/Y/MT distinctly.
- **ld** (1 file): `_blocks.py` warns and skips chrX (LD block algorithms assume diploid HWE — Wall & Pritchard 2003 model is autosomal).

Total: ~65 files touched (~30 substantive + ~35 mechanical model-file plumbing).

### 3.4 Files NOT touched

- `torchgwas/_native/`, `csrc/` — kernels are sex-agnostic; sex masking happens in the Python wrapper before the native call. No native-side change required.
- `torchgwas/multiomics/`, `torchgwas/postgwas/`, `torchgwas/pgs/`, `torchgwas/annotate/` — downstream of the scan; consume per-chr sumstats without sex-chr-internal changes.

## 4. Data flow

**Sample sex propagation:**

```
.fam / .psam / .sample (BGEN) / VCF PEDIGREE block
   ↓ (per-reader parser)
SampleMeta.sample_sex: Tensor[int8] of shape (N,)
   ↓ (UnifiedScanner)
ChunkContext.sample_sex passed to every score_chunk call
   ↓ (BaseModel.score_chunk)
Per-model sex-aware logic: male doubling, female-only HWE, etc.
```

**Variant chr canonicalization:**

```
.bim / .pvar / VCF #CHROM
   ↓ (per-reader parser, raw string preserved)
VariantMeta.chr: list[str] (heterogeneous: "23", "X", "chrX", "PAR1", ...)
   ↓ (io/chrom.py: canonicalize_chr)
Internal canonical labels: {"1".."22", "X", "Y", "XY", "MT"}
   ↓ (io/chrom.py: is_par, is_xchr_nonpar, is_y, is_mt)
Per-variant region predicates passed via ChunkContext
```

**Per-chromosome chunk dispatch:**

```
UnifiedScanner with split_at_chrom_change=True
   ↓
Each chunk has uniform chr → ChunkContext.is_xchr_nonpar etc. is uniform per chunk
   ↓
Model applies per-chunk sex-aware logic without per-row branching
```

## 5. Validation gates

### 5.1 Tier 1 — math correctness (synthetic, deterministic)

Required tests in `tests/test_xchrom_qc.py` and `tests/test_xchrom_genotype.py`:

- **MAF formulae** (per §2.5): hand-computed expected values on a 6-sample fixture with 2 males / 4 females; assert ≤ 1e-10 absolute on chrX, chrY, mtDNA, and PAR.
- **HWE-on-females** (per §2.6): hand-computed expected $\chi^2$ on a 4-female fixture; assert against `scipy.stats.chi2.sf` to ≤ 1e-10.
- **Uniform-skewing dosage** (per §2.1): hand-computed expected $\tilde g$ matrix; assert exact equality (deterministic).
- **Random-skewing dosage** (per §2.1): same.
- **PAR detection** (per §2.2): hand-coded boundary tests at PAR1/PAR2 endpoints in both GRCh37 and GRCh38; assert exact `is_par` truth.
- **Chr canonicalization** (per §3): table-driven test mapping `"23"`, `"X"`, `"chrX"`, `"PAR1"`, `"X_nonPAR"` → canonical labels; assert exact equality.

### 5.2 Tier 2 — smoke parity vs PLINK 2.0 (small fixture)

Required test in `tests/test_xchrom_parity_plink2.py`:

- **Fixture**: `tests/fixtures/xchrom_diploid.bed/bim/fam` — 200 samples (100 male / 100 female), 1000 variants distributed across chr1, PAR1, chrX non-PAR, PAR2, chrY, chrM.
- **Reference run**: `plink2 --bfile xchrom_diploid --glm --xchr-model 2 --pheno pheno.txt`. Output: per-variant β, SE, p-value.
- **Tolerance** (per master spec §5; MVP-equivalent for additive uniform-skewing): `floor + observed × 2`. Floor: 1e-6 absolute on β and SE, 1e-3 relative on p-value.
- **Tolerance for `--xchr-model 0/1/3`** (full-parity scope): `floor + observed × 1.5` per master spec §5.
- **Failure mode**: any variant with divergence outside tolerance → log to `docs/validation_findings.md` per F3 severity policy. Phase does NOT block on documented divergences (post-V1 per master spec §7).
- **Pre-flight**: PLINK 2.0 install + 200-sample fixture generation < 100 MB total. Memory headroom check before run per `feedback_preflight`.

### 5.3 Tier 2 — smoke parity vs REGENIE chrX scan (Tier B)

Tier B-only test, gated on Phase 57 REGENIE wrapper landing (REGENIE itself depends on this refactor — see master spec §3 — but a *parity test against REGENIE* requires our wrapper to exist). When that gate clears: same fixture as §5.2, REGENIE step-1 + step-2 with `--chr 23`, same tolerance rule (`floor + observed × 2`).

### 5.4 Tier 3 — full-scale empirical (deferred per NA3)

Documented assertion (not run): on UKB chrX scan (~700K samples, ~200K chrX variants), our outputs match REGENIE step-2 to within `floor + observed × 2` tolerance. Deferred to NA3 task per `project_next_agent_tasks` memory.

### 5.5 R-MOD-1 invariant — autosomal regression net

Required test in `tests/test_xchrom_invariance.py`:

- Run the FULL existing V1 parity suite (GEMMA, GAPIT, GWASpoly comparisons in `tests/test_*.py` marked `@pytest.mark.golden`) with `sample_sex=None` and all autosomal chr labels.
- Assert byte-for-byte equality (within existing tolerance) with pre-refactor outputs.
- This test runs FIRST and MUST pass before any other Tier 1 / Tier 2 X-chrom test.
- Failure → halt the refactor track per master spec §7.

## 6. CLI integration

X-chrom flags added to: `glm-scan`, `lmm-scan`, `mvlmm-scan`, `farmcpu-scan`, `blink-scan`, `binary-glm-scan`, `binary-glmm-scan`, `survival-scan`. Implemented as a shared `argparse` group `xchrom_flags()` to avoid per-subcommand duplication.

| Flag | Default | Values | Source |
|---|---|---|---|
| `--xchr-model` | `2` | `{0, 1, 2, 3}` | PLINK 2.0 `--xchr-model` [C4] |
| `--split-par` | `hg38` | `{hg19, hg38, none, coords:<X1>:<X2>:<Y1>:<Y2>}` | PLINK 2.0 `--split-par` [C5] |
| `--xchr-grm-kernel` | `separate` (LMM) / `joint` (FarmCPU/BLINK) | `{separate, joint, none}` | GCTA [C14] / BOLT [C12] |
| `--xchr-strategy` | `dosage` | `{dosage, stratified}` | König 2014 [C3] |
| `--filter-males` | `false` | flag | PLINK 2.0 [C7] |
| `--filter-females` | `false` | flag | PLINK 2.0 [C7] |
| `--mt-heteroplasmy-cutoff` | `0.05` | float ∈ [0, 1] | Yonova-Doing 2021 [C19] |
| `--mt-heteroplasmy-mode` | `filter` | `{filter, continuous}` | Hägg 2021 [C20] |

**Streaming compliance** (per `feedback_streaming` memory): every modified scan command continues to use `iter_chunks`. The chrom-boundary split (`split_at_chrom_change=True`) is set by default when any X-chrom flag is active. `tests/test_streaming_memory.py` extended to cover chrX-active scans.

## 7. Native acceleration scope (deferred)

Per `Python-as-spec, native-as-shortcut` convention (`CLAUDE.md` §"Repo Conventions"), no native code is committed in the initial implementation. Pure-torch path is the spec.

Candidate hot loops for a future Phase-41-style sub-phase:
- Per-variant uniform-skewing transform (vectorized; expected ~5× speedup on torch GPU; no native needed).
- Sex-aware MAF + HWE (low compute relative to scan loop; native unlikely to pay off).
- chrX two-kernel GRM eigendecomposition (likely already covered by existing `_native/eigh.cpp`; needs verification).

Native sub-phase deferred until profiling against UKB-scale workloads identifies an actual bottleneck. Speedup targets not committed.

## 8. F3 severity application

Per master spec §7 (post-V1 → divergences documented, regressions in existing paths halt the refactor):

| Event | Action |
|---|---|
| Tier 1 test failure | Halt; bug in our code; fix before any further Tier work |
| Tier 2 divergence vs PLINK 2.0 within tolerance | Pass; no findings entry |
| Tier 2 divergence vs PLINK 2.0 outside tolerance | Document in `docs/validation_findings.md`; investigate root cause; phase does not block (post-V1) |
| `tests/test_xchrom_invariance.py` failure | **HALT the refactor track**; revert and re-approach. R-MOD-1 invariant is non-negotiable. |
| Any GEMMA / GAPIT / GWASpoly autosomal parity test failure during refactor | **HALT**; same reason. |
| ≥ 3 unrelated divergences during one Tier B implementation step | **C4 emergency stop**; regroup with user. |

## 9. Risks & open questions

### 9.1 Risks (from research brief §10, mitigations made concrete)

| ID | Risk | Mitigation |
|---|---|---|
| R-XC-1 | Per-sample-per-variant ploidy tensor extension breaks polyploid V1 parity | Tier A test: explicit polyploid V1 parity re-run with the new tensor-typed `ploidy` parameter set to a constant; assert byte-equal to pre-refactor outputs |
| R-XC-2 | Sex-column propagation across 7 readers introduces format-specific bugs | Tier 1: per-reader `test_<format>_sex_parsing.py` with hand-crafted fixtures covering: missing sex column (fall through to `None`), `0`/`1`/`2` PLINK encoding, VCF `PEDIGREE` block, BGEN `.sample` file |
| R-XC-3 | `ChunkContext` protocol change breaks third-party `BaseModel` subclasses | Default value on `chunk_context: ChunkContext = ChunkContext()` keyword preserves backward compat; documented in CLAUDE.md and CHANGELOG |
| R-XC-4 | Heterogeneous chr labels cause silent mis-classification | Tier 1 table-driven canonicalization tests; refusal-mode default for unrecognized labels (raise `ValueError("unrecognized chromosome label: ..."); pass `--allow-unknown-chr` to fall back to autosomal-treatment) |
| R-XC-5 | PLINK 2.0 floats and our floats may diverge at the bit level (different std lib, different SIMD) | Tolerance per §5.2 is `floor + observed × 2` — generous enough to absorb PLINK / our minor numerical differences |
| R-XC-6 | UKB-scale chrX scan reveals memory/time issues that local tests miss | Documented in §5.4 deferred to NA3; not a Tier A blocker but tracked |
| R-XC-7 | LD block algorithms applied to chrX produce nonsense | `ld/_blocks.py` warns and skips chrX; documented in module docstring with `Wall & Pritchard 2003` citation |
| R-XC-8 | mtDNA scan with REGENIE / BOLT fixtures fails (tools don't support mtDNA) | Tier 2 mtDNA parity uses PLINK 2.0 `--glm --chr 26` only; REGENIE / BOLT mtDNA explicitly out-of-scope |

### 9.2 Open questions (deferred to implementation kickoff)

- **OQ-XC-1**: For the per-sample-per-variant ploidy tensor, should we use `int8` (1 byte/cell, 1.5 GB for 1M variants × 1.5K samples) or pack into a bit-vector (slower, smaller)? Default lean: `int8` for clarity; revisit if memory becomes a Tier 3 issue.
- **OQ-XC-2**: When `--xchr-grm-kernel separate` is used with a non-MultiKernel-aware downstream model (e.g., FarmCPU), should we silently fall through to `joint` or raise? Default lean: raise; explicit user choice required.
- **OQ-XC-3**: Should `--check-sex` / `--impute-sex` be implemented (PLINK 2.0 [C8])? Default lean: defer to Tier C; users can pre-process with PLINK 2.0 directly.
- **OQ-XC-4**: For the König 2014 [C3] sex-stratified pathway, should it run as a separate CLI subcommand (`xchrom-stratified-scan`) or as a flag (`--xchr-strategy stratified`)? Default lean: flag (lower CLI surface); parallel runs handled internally via `iter_chunks` per sex.
- **OQ-XC-5**: REGENIE has documented X-chromosome behavior changes between v3.x and v4.x (research brief §9). Which version do we pin for Tier 2 parity? Default lean: pin to v4.2+ (after Issue #678 sparse-variant SE bug is fixed); document in test docstring.

## 10. Implementation phasing

Per existing campaign convention (Tier A = minimum-viable shipping unit; Tier B = parity-tightening; Tier C = nice-to-haves).

### 10.1 Tier A — minimum viable shipping unit

Shippable when: chrX additive scan with uniform skewing + autosomal R-MOD-1 invariant + chrY males-only + PAR autosomal-style + chrM haploid all work end-to-end with PLINK 2.0 Tier 2 parity within tolerance.

| Step | Files | Test |
|---|---|---|
| **A0** Tier-0 prerequisite — sex parsing across all 7 readers | All 7 `io/*.py` readers + `models/base.py` (`SampleMeta`) | `test_<format>_sex_parsing.py` per reader |
| **A1** Chr canonicalization layer | `io/chrom.py` (new) | `tests/test_chrom_canonicalize.py` |
| **A2** Per-sample-per-variant ploidy tensor support in QC + standardize | `preprocess/qc.py`, `preprocess/standardize.py` | Tier 1 unit tests + R-XC-1 mitigation test |
| **A3** Sex-aware MAF + HWE | `preprocess/qc.py` | `tests/test_xchrom_qc.py` |
| **A4** `ChunkContext` protocol + `UnifiedScanner` chrom-boundary splits | `scan/chunk_context.py` (new), `scan/unified.py`, `models/base.py` | `tests/test_chunk_context.py` |
| **A5** Uniform-skewing dosage transform on chrX in V1 scans | All 5 V1 scan models in `models/` | Tier 2 PLINK 2.0 parity (subset) |
| **A6** chrY males-only scan | All 5 V1 scan models | `tests/test_xchrom_y.py` |
| **A7** chrM haploid scan + heteroplasmy filter | All 5 V1 scan models, `preprocess/qc.py` | `tests/test_xchrom_mt.py` |
| **A8** PAR detection + autosomal-style routing | `io/chrom.py`, every V1 scan | `tests/test_xchrom_par.py` |
| **A9** R-MOD-1 invariant test | `tests/test_xchrom_invariance.py` (new) | this test |
| **A10** CLI flag wiring (V1 scans only) | `cli.py` | smoke test in `tests/test_cli.py` |
| **A11** Streaming regression net extension | `tests/test_streaming_memory.py` | this test |

### 10.2 Tier B — parity-tightening

Shippable when: all four PLINK `--xchr-model` values + sex-stratified meta-analysis + two-kernel GRM all work + Tier 2 parity vs PLINK 2.0 across all four xchr-model values within full-parity tolerance.

| Step | Files | Test |
|---|---|---|
| **B1** PLINK `--xchr-model 0` (skip) | scan dispatch | parity test |
| **B2** PLINK `--xchr-model 1` (random skewing) | dosage transform | parity test |
| **B3** PLINK `--xchr-model 3` (sex-by-genotype interaction) | model fit refactor | parity test |
| **B4** Sex-stratified meta-analysis (König 2014) | `scan/strategies.py` | `tests/test_xchrom_stratified.py` |
| **B5** Two-kernel GRM via `MultiKernelLMM` | `linalg/kinship_xchr.py` (new), `models/multi_kernel_lmm.py` | `tests/test_xchrom_grm_two_kernel.py` |
| **B6** GLMM family extensions (chrX-aware binary/ordinal/multinomial GLMM) | 6 GLM/GLMM models | per-model parity test |
| **B7** Per-chr Bonferroni / multiple-testing pool | `stats/multipletesting.py` | `tests/test_xchrom_multipletesting.py` |
| **B8** Tier 2 parity vs REGENIE chrX (gated on Phase 57) | `tests/test_xchrom_parity_regenie.py` | gated test |
| **B9** Manhattan plot color-by-chrX/Y/MT | `viz/_manhattan.py` | smoke test |
| **B10** LD block warn-and-skip for chrX | `ld/_blocks.py` | unit test |

### 10.3 Tier C — nice-to-haves

| Step | Files | Test |
|---|---|---|
| **C1** Gene-level XCI escape per Tukiainen 2017 [C2] | new `preprocess/xci_escape.py` with Supp Tables 8–13 ingestion | parity vs published gene list |
| **C2** XTR-aware analysis (Mueller 2008 [C18]) | `io/chrom.py` adds `is_xtr` predicate | unit test |
| **C3** PLINK `--check-sex` / `--impute-sex` | new `preprocess/check_sex.py` | parity vs PLINK |
| **C4** Sex-by-environment-by-genotype interaction (LMM with three-way) | `models/lmm_gxe.py` extension | unit test |
| **C5** Tier 2 parity vs BOLT-LMM chrX, SAIGE chrX | `tests/test_xchrom_parity_bolt.py`, `tests/test_xchrom_parity_saige.py` | gated tests |
| **C6** Native acceleration of uniform-skewing transform | `csrc/xchrom_dosage.cpp` | parity + bench |

## 11. Citations

All claims in this spec MUST trace to a primary source. Citations consolidated here for cross-reference; each is also cited inline where the claim appears.

- **[C1] Clayton, D. (2008).** *Testing for association on the X chromosome.* *Biostatistics* 9(4):593–600. <https://doi.org/10.1093/biostatistics/kxn007>. Equations (1)–(2) define random-skewing and uniform-skewing male dosage models.
- **[C2] Tukiainen, T. et al. (2017).** *Landscape of X chromosome inactivation across human tissues.* *Nature* 550:244–248. <https://doi.org/10.1038/nature24265>. Supplementary Tables 8–13: per-gene XCI escape status.
- **[C3] König, I. R., Loley, C., Erdmann, J., Ziegler, A. (2014).** *How to include chromosome X in your genome-wide association study.* *Genet. Epidemiol.* 38(2):97–103. <https://doi.org/10.1002/gepi.21782>. §"Sex-stratified analysis".
- **[C4] PLINK 2.0 docs — `--glm` / `--xchr-model`.** <https://www.cog-genomics.org/plink/2.0/assoc#glm>. `--xchr-model 0/1/2/3` semantics.
- **[C5] PLINK 2.0 docs — `--split-par` / `--merge-par`.** <https://www.cog-genomics.org/plink/2.0/data#split_par>.
- **[C6] PLINK 2.0 docs — chromosome set / `--chr-set`.** <https://www.cog-genomics.org/plink/2.0/data#chr_set>.
- **[C7] PLINK 2.0 docs — sex filters.** <https://www.cog-genomics.org/plink/2.0/filter#sex>.
- **[C8] PLINK 2.0 docs — `--check-sex` / `--impute-sex`.** <https://www.cog-genomics.org/plink/2.0/basic_stats#check_sex>.
- **[C9] PLINK 2.0 docs — `--freq` / `--maf` (sex-chr handling).** <https://www.cog-genomics.org/plink/2.0/basic_stats#freq>. MAF denominator $2n_F + n_M$ for chrX.
- **[C10] PLINK 2.0 docs — `--hardy` (sex-chr behaviour).** <https://www.cog-genomics.org/plink/2.0/basic_stats#hardy>.
- **[C11] REGENIE docs — Step 1 / Step 2 ("X chromosome" section).** <https://rgcgithub.github.io/regenie/options/>.
- **[C12] BOLT-LMM v2.4 manual.** <https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>.
- **[C13] SAIGE Step 2 docs.** <https://saigegit.github.io/SAIGE-doc/docs/single_step2.html>.
- **[C14] GCTA docs — Making a GRM (autosomal and X-chromosome).** <https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM>.
- **[C15] GCTA docs — MLM association (`--dc`).** <https://yanglab.westlake.edu.cn/software/gcta/#MLMassociationanalysis>.
- **[C16] Ensembl Human PAR coordinates.** <https://www.ensembl.org/info/genome/genebuild/human_PARS.html>.
- **[C17] NCBI Genome Reference Consortium — Human assembly summary.** <https://www.ncbi.nlm.nih.gov/grc/human>.
- **[C18] Mueller, J. L. et al. (2008).** *Chromosomal mapping reveals deeply conserved human genes on the X-transposed region.* *Nature Genetics* 40(7):794–799. <https://doi.org/10.1038/ng.272>.
- **[C19] Yonova-Doing, E. et al. (2021).** *An atlas of mitochondrial DNA genotype-phenotype associations in the UK Biobank.* *Nat. Genet.* 53:982–993. <https://doi.org/10.1038/s41588-021-00868-1>. Methods: heteroplasmy filtering.
- **[C20] Hägg, S. et al. (2021).** *Deciphering the genetic and epidemiological landscape of mitochondrial DNA abundance.* *Nat. Commun.* 12:6147. <https://doi.org/10.1038/s41467-021-26424-3>. Methods: mtDNA scan protocol.

## 12. Approval & next step

This sub-spec is committed once approved alongside the master and the other three sub-specs. Implementation is DEFERRED until validation campaign Pillars A–D and the three NA tasks (NA1, NA2, NA3) close, per master spec §3. At implementation kickoff, this spec is fed to `superpowers:writing-plans` to produce the per-Tier implementation plan.
