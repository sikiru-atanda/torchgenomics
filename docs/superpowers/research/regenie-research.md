# REGENIE — Research Brief

- **Author:** Claude Code (desk research; no install performed for this brief)
- **Date:** 2026-05-11
- **Audience:** TorchGWAS Phase 57 sub-spec drafter
- **Scope of this brief:** REGENIE step 1 + step 2, **MVP integration** — quantitative + binary-with-SPA only. No Firth, no time-to-event (Cox), no gene/region/SKAT/SBAT, no interaction, no conditional, no joint. Per master spec [`2026-05-11-modernization-master-design.md`](../specs/2026-05-11-modernization-master-design.md) §2.

---

## 1. Upstream tool overview

| Field | Value | Citation |
|---|---|---|
| Repository | `rgcgithub/regenie` (C++) | https://github.com/rgcgithub/regenie |
| Latest release | **v4.1**, published **2025-01-27** | GitHub Releases API: `https://api.github.com/repos/rgcgithub/regenie/releases/latest` (queried 2026-05-11) |
| License | **MIT** (verified by reading raw `LICENSE` at `master`) | https://github.com/rgcgithub/regenie/blob/master/LICENSE — copyright "(c) 2020-2021 Joelle Mbatchou, Andrey Ziyatdinov & Jonathan Marchini" |
| Paper | Mbatchou, J. *et al.* "Computationally efficient whole-genome regression for quantitative and binary traits." *Nat Genet* **53**, 1097–1103 (2021). | DOI: https://doi.org/10.1038/s41588-021-00870-7 ; PubMed 34017140 |
| Maintainer | Regeneron Genetics Center | README §"developed and supported by a team of scientists at the Regeneron Genetics Center" — https://github.com/rgcgithub/regenie#readme |
| Static binaries | **Yes**, statically-linked Linux x86_64 binary published per release. v4.1 ships `regenie_v4.1.gz_x86_64_Linux.zip` (3.6 MB) and an MKL-linked variant (8.8 MB). | https://github.com/rgcgithub/regenie/releases/tag/v4.1 (assets listing via GitHub API, queried 2026-05-11) |

The `LICENSE` file additionally documents that REGENIE links the BGEN library under the **Boost Software License 1.0** (compatible with MIT).

---

## 2. Algorithm summary

### 2.1 Step 1 — whole-genome ridge regression (stacked, LOCO)

Per the official Overview docs at https://rgcgithub.github.io/regenie/overview/, Step 1 has two levels.

**Level 0 (block-wise ridge):** Markers are partitioned into consecutive blocks of size `--bsize`. For each block $b$ with genotype submatrix $X_b \in \mathbb{R}^{N \times M_b}$ and phenotype $y \in \mathbb{R}^N$, $J$ ridge predictors are computed across a fixed grid of shrinkage values $\{\lambda_j\}_{j=1}^{J}$:

$$
\hat\beta_{b,j} = (X_b^\top X_b + \lambda_j I)^{-1} X_b^\top y
\qquad j = 1,\dots,J
$$

Stacking the per-block, per-shrinkage predictions $\hat y_{b,j} = X_b \hat\beta_{b,j}$ column-wise produces the **Level 0 design matrix**

$$
W \in \mathbb{R}^{N \times (B \cdot J)}.
$$

**Level 1 (cross-validated stacking ridge):** A second ridge regression combines the $B \cdot J$ Level-0 columns with cross-validation across `--cv` folds:

- Quantitative trait: $\hat y = W \hat\alpha$ with $\hat\alpha = \arg\min_\alpha \|y - W\alpha\|^2 + \tau \|\alpha\|^2$, picking $\tau$ from a grid by CV MSE.
- Binary trait: log-odds $\eta = W\alpha + Cc$ fit by **logistic ridge regression**, again CV across $\tau$.

Cited verbatim from the Overview page: *"a set of ridge regression predictors are calculated for a small range of shrinkage parameters (using `--l0` option [default is 5])"* and *"we use a logistic ridge regression model to combine the predictors in W."*

**LOCO predictions:** Once $\hat\alpha$ is selected, per-chromosome predictions are constructed by zeroing out the columns of $W$ derived from chromosome $k$:

$$
\hat y_{-k} = W_{-k}\, \hat\alpha_{-k},
\qquad k = 1,\dots,K
$$

These are written to one `.loco` file per phenotype (see §4).

### 2.2 Step 2 — single-variant association testing

Per Overview docs, with $\hat y_{-k}$ from Step 1 used as an **offset**:

**Quantitative trait — score test of marker $g$ on chromosome $k$:**

$$
\tilde y = y - \hat y_{-k} - C \hat c, \qquad
T_g = \frac{g^\top \tilde y}{\sqrt{\widehat{\mathrm{Var}}(g^\top \tilde y)}}, \qquad
T_g^2 \sim \chi^2_1
$$

where $C$ is the covariate matrix and $\hat c$ are the Step-2 covariate effects.

**Binary trait — logistic score test:** the LOCO offset enters the linear predictor:

$$
\mathrm{logit}\,\Pr(Y_i = 1) = C_i \hat c + \hat y_{-k,i} + g_i \beta
$$

The score statistic and its variance are computed under the null $\beta=0$ to give a normal-approximation p-value $p_{\mathrm{norm}}$. **SPA fallback:** when $p_{\mathrm{norm}} < $ `--pThresh` (default **0.05**), recompute via saddlepoint approximation (see §11 for derivation source). REGENIE uses the **fastSPA** variant — confirmed by reading the source: the `spa_data` struct carries a `fastSPA` flag and the SPA routines branch on `fastSPA` for sparse-genotype acceleration (`src/Step2_Models.cpp` lines 2068–2088, https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.cpp).

### 2.3 Dimension table

| Symbol | Meaning | Typical biobank value | Source |
|---|---|---|---|
| $N$ | samples | $5\times 10^5$ (UKB) | Mbatchou 2021 abstract |
| $M$ | array variants used in Step 1 | $\sim 5\times 10^5$ | recommendations docs §"Suggested usage" |
| $K$ | LOCO blocks | **23** chromosomes (autosomes + X collapsed into chr 23 row) | Output docs (LOCO file has 23 rows) — see §4 |
| $B$ | Step-1 blocks per genome | $\lceil M / \text{bsize} \rceil$, e.g. $\sim 500$ at `--bsize 1000` | Overview docs |
| $J$ | Level-0 ridge grid size | **5** (default) | `--l0` default per https://hpc.nih.gov/apps/regenie.html |
| (Level 1 grid) | $J_1$ | **5** (default) | `--l1` default per https://hpc.nih.gov/apps/regenie.html |
| `--cv` | CV folds | **5** (default) | Options docs https://rgcgithub.github.io/regenie/options/ |

### 2.4 Default hyperparameters (consolidated)

| Flag | Default | Source |
|---|---|---|
| `--l0` | 5 ridge values: $\{0.01, 0.25, 0.5, 0.75, 0.99\}$ | https://hpc.nih.gov/apps/regenie.html |
| `--l1` | 5 ridge values: $\{0.01, 0.25, 0.5, 0.75, 0.99\}$ | same |
| `--bsize` (Step 1) | required, no default; recommended 1000 | Recommendations docs https://rgcgithub.github.io/regenie/recommendations/ |
| `--bsize` (Step 2) | required, no default; recommended 200–400 | same |
| `--cv` | 5 | Options docs |
| `--pThresh` (SPA / Firth fallback trigger) | 0.05 | Options docs §Step 2 |
| `--minMAC` (Step 2 variant filter) | 5 | Options docs §Step 2 |
| `--minINFO` | none unless set | Options docs §Step 2 |

UNVERIFIED: The exact ridge grid values $\{0.01, 0.25, 0.5, 0.75, 0.99\}$ come from a third-party HPC docs page (NIH Biowulf), not from the upstream regenie docs (the Options docs page does not display the `--l0`/`--l1` defaults explicitly). The Phase 57 sub-spec must reconfirm by either running `regenie --help` after install or grepping the `cxxopts` parameter declarations in `src/Regenie.cpp` / `src/Pheno.cpp`.

---

## 3. Input formats

All schemas verbatim from https://rgcgithub.github.io/regenie/options/ and from inspecting the example fixtures at https://github.com/rgcgithub/regenie/tree/master/example.

### 3.1 Phenotype file (`--phenoFile`)

*"Line 1 : Header with FID, IID and phenotypes names. Followed by lines of values. Space/tab separated."* For binary traits: `0=control, 1=case, NA=missing` unless `--1` is passed.

Example header (from `example/phenotype.txt`, fetched via raw.githubusercontent):

```
FID IID Y1 Y2
1 1 1.64818554321186 2.2765234736685
2 2 -2.67352013711554 -1.53680421614647
```

### 3.2 Covariate file (`--covarFile`)

*"Line 1 : Header with FID, IID and covariate names. Followed by lines of values. Space/tab separated."*

Example header (from `example/covariates.txt`):

```
FID IID V1 V2 V3
1 1 1.46837294454993 1.93779743016325 0.152887004505393
```

For categorical covariates, REGENIE accepts a separate column with `--catCovarList` and string levels (example `example/covariates_wBin.txt`):

```
FID	IID	V1	V2	V3	V4	V5
1	1	1.46837294454993	1.93779743016325	0.152887004505393	1	urban
```

### 3.3 Genotype formats

| Format | Flag | Files expected | Source |
|---|---|---|---|
| PLINK1 BED | `--bed PREFIX` | `PREFIX.bed`, `PREFIX.bim`, `PREFIX.fam` | Options docs |
| PLINK2 PGEN | `--pgen PREFIX` | `PREFIX.pgen`, `PREFIX.pvar`, `PREFIX.psam` | Options docs |
| BGEN | `--bgen FILE` (+ optional `--sample`, `--bgi`) | `.bgen`, optional `.sample`, optional `.bgi` index | Options docs |

Notes from issue tracker (queried via GitHub API 2026-05-11):
- Issue #666 (closed): REGENIE *requires* FID in `.psam` even though the PLINK2 spec allows it to be optional. Implication for our `io/plink2.py` adapter: must back-fill FID column when emitting REGENIE-bound files.
- Issue #677 (open): no native VCF/BCF support — converters are required.

---

## 4. Output formats

### 4.1 Step 1 — `.loco` predictions

Per https://rgcgithub.github.io/regenie/options/#output:

> *"Line 1 starts with `FID_IID` followed by N sample identifiers. It is followed by 23 lines containing the genetic predictions for each chromosome (sex chromosomes are collapsed into chromosome 23)."*
>
> *"Each line has N+1 values which are the chromosome number followed by the N leave-one chromosome out (LOCO) predictions for each individual."*

**Layout:** one `.loco` file per phenotype. **Header row** lists $N$ sample IDs; **23 data rows** (one per chromosome). Each data row: chromosome integer + $N$ float predictions. Total file shape: $24 \times (N+1)$.

A companion `_pred.list` file (one line per phenotype) maps phenotype name → `.loco` file path. This is what `--pred` consumes in Step 2.

**Streaming-readiness:** the LOCO file is *row-streamable per chromosome* — each Step-2 chromosome scan only needs one row. See §11 for our implementation implication.

### 4.2 Step 2 — single-variant summary stats

The exact column header verified by reading `example/example.test_bin_out_firth_Y1.regenie` (raw.githubusercontent.com):

```
CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ INFO N TEST BETA SE CHISQ LOG10P
1 1 1 2 1 0.214575 1 494 ADD 0.0775674 0.230001 0.113736 0.133163
1 2 2 2 1 0.218623 1 494 ADD 0.131068 0.239808 0.29872 0.233077
```

| Column | Meaning |
|---|---|
| `CHROM` | chromosome (integer or string) |
| `GENPOS` | base-pair position |
| `ID` | variant ID |
| `ALLELE0` | reference allele (allele 0) |
| `ALLELE1` | alternative allele (allele 1) — effect allele |
| `A1FREQ` | frequency of `ALLELE1` |
| `INFO` | imputation INFO score (BGEN/PGEN dosage only; "1" for hard-call BED) |
| `N` | sample size used at this variant |
| `TEST` | test performed: `ADD` / `DOM` / `REC` |
| `BETA` | effect size for `ALLELE1` on the original scale |
| `SE` | standard error of `BETA` |
| `CHISQ` | chi-square test statistic |
| `LOG10P` | $-\log_{10}(p)$ |

Optional alternative format: `--htp COHORT` emits HTP-format summary stats per remeta convention (https://rgcgithub.github.io/remeta/file_formats/#-htp). Not in MVP scope.

A failure indicator column is appended when Firth/SPA correction fails (Options docs §Output).

---

## 5. Key flags / options for the MVP scope

The 80%-subset for the **quantitative + binary-with-SPA** MVP, drawn from https://rgcgithub.github.io/regenie/options/.

**Step 1 (whole-genome ridge):**

| Flag | Required? | Notes |
|---|---|---|
| `--step 1` | yes | mode selector |
| `--bed` / `--pgen` / `--bgen` | yes | genotype input |
| `--phenoFile FILE` | yes | FID/IID + phenotype columns |
| `--covarFile FILE` | optional | FID/IID + covariate columns |
| `--bsize INT` | yes | block size; recommended 1000 for ~500K array SNPs |
| `--bt` | binary only | flag binary traits |
| `--cv INT` | optional | CV folds, default 5 |
| `--lowmem` + `--lowmem-prefix` | optional | spill Level-0 predictions to disk for many phenotypes |
| `--threads INT` | optional | default `all-1` |
| `--phenoColList` / `--covarColList` | optional | column subsetting |
| `--apply-rint` | optional | RINT for QT (must also pass in Step 2) |
| `--out PREFIX` | yes | output prefix |

**Step 2 (per-variant scan):**

| Flag | Required? | Notes |
|---|---|---|
| `--step 2` | yes | mode selector |
| `--bed` / `--pgen` / `--bgen` | yes | typically imputed dosage data |
| `--pred FILE` | **yes** | the `_pred.list` from Step 1 |
| `--phenoFile FILE` | yes | same as Step 1 |
| `--covarFile FILE` | optional | same as Step 1 |
| `--bsize INT` | yes | recommended 200–400 |
| `--bt` | binary only | |
| `--spa` | binary only — **MVP** | enable SPA fallback |
| `--pThresh FLOAT` | optional | SPA invocation cutoff, default 0.05 |
| `--minMAC FLOAT` | optional | default 5 |
| `--minINFO FLOAT` | optional | for dosage data |
| `--chr` / `--range` / `--extract` / `--exclude` | optional | region filters |
| `--apply-rint` | optional | mirror Step-1 setting |
| `--out PREFIX` | yes | output prefix |

**Out of scope of MVP:** `--firth`, `--firth-se`, `--approx`, `--t2e`, `--eventColList`, gene/region (`--anno-file`, `--mask-def`, `--set-list`), interaction (`--interaction`), conditional (`--condition-list`, `--condition-file`), joint tests (SBAT/NNLS/SKAT/SKATO/ACAT), `--htp`.

---

## 6. Reference data requirements

| Use case | Dataset | URL | Approximate size |
|---|---|---|---|
| **Smoke** (Tier 2) | REGENIE bundled `example/` (BED + BGEN + PGEN + 2 QTs + 1 binary surrogate; **500 samples × 1000 variants** verified by `wc -l` on raw `example/example.fam` → 500, `example/example.bim` → 1000) | https://github.com/rgcgithub/regenie/tree/master/example | ~600 KB total per format |
| **Mid-scale parity** | HapMap3 chr22 sample (used in our existing Tier-2 fixtures per master spec §5) | https://www.internationalgenome.org/category/hapmap-3/ | ~10–50 MB |
| **Mid-scale parity** | 1000 Genomes phase-3 chr22 subset | https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ | ~80 MB compressed |
| **Biobank-scale** (Tier 3, deferred) | UK Biobank — requires application; ~488K samples × 11.4M imputed variants (the figure used in Mbatchou 2021 §Results) | https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access | quoted as 2 TB+ for genotypes |
| **Public biobank-scale alternative** (Tier 3, deferred) | All of Us short-read WGS subset, or HRC-imputed 1KG on synthetic phenotypes | varies | varies |

For the MVP, the smoke fixture (§8) plus a 1KG chr22 mid-scale fixture is sufficient. UK Biobank is gated by NA3 in `project_next_agent_tasks` memory and is out of scope for the Phase 57 implementation gate.

---

## 7. Install prerequisites

Per https://rgcgithub.github.io/regenie/install/ and the Releases assets listing:

- **Compiler (for source builds):** GCC ≥ 5.1 (Linux) or Clang ≥ 3.3 (macOS); GFortran required.
- **Pre-compiled binary (Linux x86_64):** statically linked with **GLIBC ≥ 2.22**. Asset: `regenie_v4.1.gz_x86_64_Linux.zip` (3.6 MB compressed; ~9 MB extracted). MKL-linked variant: `regenie_v4.1.gz_x86_64_Linux_mkl.zip` (8.8 MB compressed; ~25 MB extracted).
- **Boost Iostream:** *optional*, only required for gzip-compressed input — `make HAS_BOOST_IOSTREAM=1`. **Specific Boost version is not pinned in the install docs.** UNVERIFIED: required Boost minor version. The Phase 57 sub-spec should pin against whatever the conda recipe declares (`conda search -c bioconda regenie --info`).
- **Intel MKL:** optional; speeds up the Step-1 ridge solves.
- **BGEN library:** required for source builds (separate download, edit `BGEN_PATH` in Makefile). The static binary already bundles it.
- **Conda one-liner:** `conda create -n regenie_env -c conda-forge -c bioconda regenie` then `conda activate regenie_env`.

**Install footprint estimate:** static binary < 30 MB. Conda env (regenie + dependencies) ~500 MB–1 GB depending on MKL inclusion. **No glibc concerns on the project's RHEL 9.6 host** (`uname -r` shows kernel 5.14, glibc ≥ 2.34 ≫ 2.22 requirement).

**Apache Spark / GLOW integration:** REGENIE is supported in Spark via the GLOW project (README §"It is ideally suited for implementation in Apache Spark"). Out of scope for our Python toolkit but worth flagging because GLOW's REGENIE bindings are an existence proof that the binary can be driven from a higher-level dataframe layer.

---

## 8. Smoke fixture suggestion

**Use the bundled `example/` fixture from the REGENIE repo** (https://github.com/rgcgithub/regenie/tree/master/example). Verified by raw fetches:

- `example/example.fam` — **500 samples** (`wc -l` on raw file)
- `example/example.bim` — **1000 variants**
- `example/example.bed` — 125 KB (also `.bgen`, `.pgen` provided for format-coverage tests)
- `example/phenotype.txt` — 2 quantitative traits Y1, Y2 (header `FID IID Y1 Y2`)
- `example/covariates.txt` — 3 quantitative covariates V1–V3
- `example/covariates_wBin.txt` — adds V4 (binary integer) and V5 (categorical string) — exercises `--catCovarList`
- `example/example.test_bin_out_firth_Y1.regenie` — a checked-in expected-output file the upstream test suite compares against (excellent oracle for Tier-2 column-by-column parity)

**Coverage:** the fixture exercises both Step 1 (with `--bt` toggled by recoding Y1 to binary, or by adding a binary surrogate phenotype) and Step 2. Two QTs let us sanity-check multi-phenotype LOCO file emission.

**Estimated runtime on CPU:** << 1 minute for full Step 1 + Step 2 round trip on a single core. The upstream `test/test_bash.sh` (https://github.com/rgcgithub/regenie/blob/master/test/test_bash.sh — 4 KB) is a reusable parity reference: it runs `regenie` on the bundled fixture and `diff`s against checked-in expected outputs.

**Tier-2 plan:** `pytest -m golden` test that (a) calls the upstream binary on `example/`, (b) runs our forthcoming `regenie-scan` CLI on the same inputs, (c) joins on `(CHROM, GENPOS, ID, ALLELE0, ALLELE1, TEST)`, (d) asserts BETA / SE / LOG10P agree within tolerance (per master-spec §5 "MVP scope = floor + observed × 2").

---

## 9. Published validation tolerances

The Mbatchou 2021 Nature Genetics paper does not, in the publicly accessible Overview / Methods page, quote raw Pearson correlation values for REGENIE vs BOLT-LMM / SAIGE. UNVERIFIED beyond the qualitative summaries below — the full Methods is paywalled (Nature returned 303 redirects on every WebFetch attempt 2026-05-11) and the bioRxiv preprint v2/v3 returned 403. The quantitative tables sit in the paper Methods + Supplementary Tables behind the Nature paywall.

What is verifiable from the upstream **Performance docs** (https://rgcgithub.github.io/regenie/performance/):

- **Quantitative traits (50 traits, 11.4M SNPs tested):** REGENIE Manhattan plots show *"good agreement"* with BOLT-LMM and stronger signals than fastGWA at known peaks. Memory: REGENIE 12.9 GB vs BOLT-LMM 50 GB vs fastGWA 2 GB.
- **Binary traits (50 traits, 11.4M SNPs):** *"all four approaches show very good agreement for the most balanced trait (CAD)"* — but BOLT-LMM inflates as case-control ratio worsens, while REGENIE+SPA/Firth and SAIGE remain calibrated. Memory: REGENIE uses ~40% of SAIGE's memory.
- **Speed:** REGENIE 151× faster than BOLT-LMM in Step 1 elapsed time, 11.5× in Step 2; 350× faster than SAIGE in Step 1 for binary traits.

Independent empirical comparison — Loesch *et al.* 2022 (Frontiers in Genetics, "Evaluation of GENESIS, SAIGE, REGENIE and fastGWA-GLMM…"; https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.897210/full) — reports rank-correlation between SAIGE and GENESIS sparse-/full-GRM SPA tests of **Spearman ρ ≈ 0.97–0.98** on NECS imputed and LLFS WGS data. The same paper reports REGENIE *over-corrects* in family-based data (see §10).

**Implication for our Tier-2 tolerance:** since no upstream paper publishes a hard "REGENIE vs BOLT-LMM" Pearson on shared simulated data, our Tier-2 must compute the observed divergence on our own fixture and floor it (per `feedback_validation_spec` and master-spec §5), not borrow a number from literature.

---

## 10. Known issues / gotchas

Pulled from the live GitHub issues list (https://api.github.com/repos/rgcgithub/regenie/issues, queried 2026-05-11) and from Loesch *et al.* 2022.

### 10.1 Confirmed bug — sparse-variant SE deflation in subset analyses

**Issue #678 (closed):** *"Incorrect SE deflation for sparse variants in subset analyses (Step 2 approximation error)"* — affects v4.1. When Step 2 runs on a sample subset (e.g. males-only), sparse variants take a fast variance approximation in `src/Step2_Models.cpp::compute_score_qt` that uses `XtG_ss` computed on the **full sample** rather than the subset, deflating SEs and producing a bimodal SE distribution. The reporter notes the correct projection is *"currently commented out in the source code immediately below the approximation."* https://github.com/rgcgithub/regenie/issues/678

> Direct implication for our Tier-2 parity gate: if we run REGENIE v4.1 on a subset fixture and our pure-Python score test diverges, the reference may be wrong, not us. Tier-2 should pin REGENIE to a version after this is fixed (track v4.2+) or limit subset-mode parity tests until upstream resolution.

### 10.2 Open — version-to-version BETA inflation (v2 → v4) for logistic

**Issue #684 (open):** *"Inflation of beta-values from logistic analysis - it is version specific"* — same input data, BETAs differ ~13× between v2 and v4 (e.g. 0.019 → 0.260) while SEs are comparable. CHISQ / LOG10P correspondingly inflate. Reporter shows side-by-side headers. https://github.com/rgcgithub/regenie/issues/684

> Implication: **REGENIE itself is not version-stable on the BETA scale for binary traits.** Our Tier-2 parity gate must pin a specific REGENIE version and treat it as the moving oracle. The `floor + observed × 2` tolerance rule absorbs this cleanly; our Phase 57 sub-spec must record the pinned version explicitly.

### 10.3 Open — covariate-scale-driven inflation

**Issue #679 (open):** Continuous covariates with units in millions of mm³ (intracranial volume) drive λ_GC from 1.16 → 1.24. Rescaling to cm³ or z-scoring restores calibration. Indicates Step-1 ridge regression is not scale-invariant under the published defaults. https://github.com/rgcgithub/regenie/issues/679

> Implication: our Step-1 implementation must standardize covariates before the Level-0 ridge solve (or document the same caveat). We can choose to be more robust than upstream here — that is *not* a divergence we owe upstream.

### 10.4 Documented — over-conservative in family-based data

Loesch *et al.* 2022 (Frontiers in Genetics): *"REGENIE may over-correct p-values in family-based data… with possible loss of power."* Documented for both NECS imputed and LLFS WGS data, with REGENIE *"substantially more conservative"* than GENESIS or SAIGE with a full GRM. https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.897210/full

> Implication: REGENIE's whole-genome-ridge polygenic prediction acts as a substitute for the explicit kinship matrix used by SAIGE/BOLT/GENESIS. In high-relatedness fixtures the substitution is imperfect. **Tier-2 must use a low-relatedness fixture (1KG-style unrelateds), not a family pedigree fixture, or the observed divergence will reflect the upstream behavior gap, not a bug in our implementation.**

### 10.5 Other tracker themes (lower priority)

- #677 (open): VCF/BCF input not natively supported → conversion required upstream of REGENIE.
- #623 (open): potential inflation in time-to-event analysis (out of MVP scope, but a flag for future Cox extension).
- #659, #530 (closed): guidance on imputed-genotype use in Step 1; recommendation is array genotypes only (not imputed) for the whole-genome ridge.

---

## 11. Risk flags for our integration

Each flag below targets a specific TorchGWAS module rather than a generic concern.

### R-REG-1 — Step-1 LOCO predictions file is row-streamable per chromosome, BUT the column dimension is N

The `.loco` file is shaped $24 \times (N+1)$ — header row of $N$ sample IDs, then 23 chromosome rows each with $N$ floats (Output docs §4.1). At UKB scale ($N \approx 5\times 10^5$, FP64), one chromosome row is **4 MB**; the full file per phenotype is **~92 MB**. With 50 phenotypes, ~4.6 GB on disk — manageable.

For Step 2, **one chromosome row at a time** is sufficient: each Step-2 chunk that we hand to `UnifiedScanner` is restricted to a single chromosome (the scanner already groups by chromosome via `iter_chunks` in `torchgwas/io/{bgen,vcf}.py`), so we only need to load the corresponding LOCO row. **No re-materialization of the full predictions matrix is required.** Our streaming-first invariant (`feedback_streaming` memory) is preserved.

**Concrete mitigation:** Phase 57 sub-spec must include a `RegenieLOCOReader` adapter that:
- parses the `_pred.list` file once at scan start;
- on each chunk of variants, loads only the matching chromosome row from each phenotype's `.loco` file (single seek + 1 line read per chromosome change);
- yields the offset vector to the Step-2 score-test path.

This sidesteps R-MOD-2 in the master risk register.

### R-REG-2 — Conceptual overlap with `models/lro_lmm.py` is real but small

`torchgwas/models/lro_lmm.py` (read for this brief) implements **block-level** LOCO over LD blocks for fine-grained proximal contamination control: $K_{-b} = K_{\text{full}} - K_b$ per LD block, fresh REML per block. REGENIE step 1 is **chromosome-level** LOCO over a learned polygenic predictor, not over a kinship matrix.

| Axis | `lro_lmm` | REGENIE |
|---|---|---|
| LOCO granularity | LD blocks | chromosomes |
| Random effect representation | explicit GRM $K_{-b}$ + REML per block | learned ridge predictor $\hat y_{-k}$ used as offset |
| Step-2 statistic | LMM Wald | OLS / logistic score with offset |
| Per-block fit cost | one EED + REML per block | single Step-1 cross-validated solve, reused for all chromosomes |

**Implementation overlap is the chromosome-grouping of variants and the per-chromosome offset machinery.** Code reuse opportunity: factor a `ChromosomeOffsetProvider` interface that both modules consume. **Risk:** none if we keep them as separate model classes; do not collapse them.

### R-REG-3 — REGENIE's binary path uses fastSPA, not vanilla SPA

We have `torchgwas/stats/spa.py` (Lugannani-Rice on the exact CGF, modeled on Zhou *et al.* 2018 SAIGE) and `torchgwas/optim/pql.py` (PQL inner loop, also SAIGE-style, citing Breslow & Clayton 1993). Source inspection of `src/Step2_Models.cpp` (lines 2068–2280, https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.cpp) confirms REGENIE branches on a `fastSPA` flag and uses `compute_K_fast_snp` / `compute_K1_fast_snp` / `compute_K2_fast_snp` for sparse genotypes — i.e. the **SAIGE fastSPA variant** that exploits sparse-vector inner products by partitioning the CGF into "carrier" and "non-carrier" terms.

**Difference from our existing SPA path:**
- Our `_cgf` operates on the dense $g$ vector (line 19 of `stats/spa.py`).
- REGENIE's fastSPA splits $K(t) = \log\sum (q\,e^{-\mu g t} + \mu\,e^{q g t})$ into the carrier sum (computed exactly over `Gsparse`'s nonzeros) plus a closed-form non-carrier residual using `val_b`, `val_c`, `val_d` cached on the SPA struct.
- For the MVP, our pure-torch dense-CGF path will give numerically equivalent answers up to the rounding induced by the sparse re-organization. **Tier-2 may show small (1e-4 to 1e-3 relative) divergences on rare variants where REGENIE's sparse path uses a different evaluation order than our dense path.** That is an algorithmic-equivalent-but-numerically-different situation, and it should be absorbed by the `floor + observed × 2` tolerance.

**Mitigation:** if Tier-2 divergence on rare-variant SPA exceeds the tolerance, the right response is **not** to rewrite our SPA — it is to add a fastSPA variant of `stats/spa.py` (separate function) that mirrors REGENIE's sparse decomposition for parity-mode runs, while keeping the dense path as the spec body per `Repo Conventions`. Phase 57 sub-spec should explicitly defer fastSPA to a follow-up sub-phase unless Tier-2 forces the issue.

### R-REG-4 — Sex-chromosome handling assumed

REGENIE collapses chrX into "chromosome 23" in the LOCO file (Output docs §4.1) and uses `--par-region` with build-aware bounds for PAR1/PAR2 (Options docs §Step 2). This implies that our Phase 57 implementation cannot ship until the X-chromosome refactor track lands — sequencing already enforced by master-spec §3 (X-chrom first, then Phase 57).

### R-REG-5 — Step-1 ridge is not scale-invariant under defaults (issue #679)

Our implementation should standardize covariates inside Step-1 fitting to be more robust than upstream defaults. Document the divergence-on-purpose in the Phase 57 sub-spec.

### R-REG-6 — Version-to-version reference instability (issue #684)

The Phase 57 sub-spec must pin a specific REGENIE version (proposed: v4.1, latest as of 2026-05-11) for Tier-2 parity, document the pin in `tests/conftest.py`, and re-baseline tolerances when bumping the pin.

### R-REG-7 — Family-based fixture would reflect the upstream over-correction

Tier-2 fixture choice must be unrelated samples (1KG / HapMap3 chr22 unrelateds), not a pedigree, or our agreement gate would be measuring REGENIE's published over-conservatism (Loesch 2022) rather than implementation parity.

---

## 12. Citations

Primary sources consulted for this brief (queried 2026-05-11). Each citation is direct (paper section, repo path, docs page) — none are paraphrased from training memory.

1. Mbatchou, J., Barnard, L., Backman, J. *et al.* (2021). "Computationally efficient whole-genome regression for quantitative and binary traits." *Nature Genetics* **53**, 1097–1103. **DOI:** https://doi.org/10.1038/s41588-021-00870-7 (Nature paywalled the full Methods on every WebFetch attempt; bioRxiv preprint at https://www.biorxiv.org/content/10.1101/2020.06.19.162354 returned 403 — flagged in §9, §11 R-REG-3.)
2. PubMed entry for Mbatchou *et al.* 2021: https://pubmed.ncbi.nlm.nih.gov/34017140/
3. REGENIE GitHub repository, master branch: https://github.com/rgcgithub/regenie
4. REGENIE LICENSE (raw): https://raw.githubusercontent.com/rgcgithub/regenie/master/LICENSE — verifies MIT + Boost-licensed BGEN library.
5. REGENIE Overview docs: https://rgcgithub.github.io/regenie/overview/ — algorithm description, Level 0 / Level 1 / LOCO.
6. REGENIE Install docs: https://rgcgithub.github.io/regenie/install/ — GLIBC ≥ 2.22, Boost iostream, MKL, conda recipe.
7. REGENIE Options docs: https://rgcgithub.github.io/regenie/options/ — phenotype/covariate schema, all flags, Step 2 column header description.
8. REGENIE Recommendations docs: https://rgcgithub.github.io/regenie/recommendations/ — `--bsize` 1000 (Step 1) / 200–400 (Step 2) recommendations, `--pThresh 0.01` example.
9. REGENIE Performance docs: https://rgcgithub.github.io/regenie/performance/ — memory tables, runtime tables, qualitative agreement statements vs BOLT-LMM/SAIGE/fastGWA.
10. REGENIE GitHub Releases (latest = v4.1, 2025-01-27): https://github.com/rgcgithub/regenie/releases/tag/v4.1 — verified asset names via `https://api.github.com/repos/rgcgithub/regenie/releases`.
11. Source: `src/Step2_Models.cpp`: https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.cpp — fastSPA branch (lines 2068–2280) and sparse-variant `compute_score_qt` approximation (line 381 onwards) — both load-bearing for §10.1 and §11 R-REG-3.
12. Source: `src/Step2_Models.hpp`: https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.hpp — SPA struct and function signatures (lines 41–101).
13. Bundled fixture: `example/` — https://github.com/rgcgithub/regenie/tree/master/example. Verified line counts via `https://raw.githubusercontent.com/rgcgithub/regenie/master/example/{example.fam,example.bim,phenotype.txt,covariates.txt,covariates_wBin.txt,example.test_bin_out_firth_Y1.regenie}` (500 samples, 1000 variants, header with exact column names).
14. Issue #678 (sparse-variant SE deflation, Step 2 approximation error): https://github.com/rgcgithub/regenie/issues/678
15. Issue #684 (version-specific BETA inflation in logistic analysis): https://github.com/rgcgithub/regenie/issues/684
16. Issue #679 (covariate-scale-driven inflation): https://github.com/rgcgithub/regenie/issues/679
17. Loesch, D.P. *et al.* (2022). "Evaluation of GENESIS, SAIGE, REGENIE and fastGWA-GLMM for genome-wide association studies of binary traits in correlated data." *Frontiers in Genetics*. https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.897210/full — quantifies REGENIE's over-conservatism in family-based data (§10.4).
18. NIH Biowulf REGENIE app docs: https://hpc.nih.gov/apps/regenie.html — third-party source for the `--l0` / `--l1` default ridge grid values (flagged UNVERIFIED in §2.4).
