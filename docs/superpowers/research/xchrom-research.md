# Sex-Chromosome (X / Y / PAR / MT) Handling in GWAS — Research Brief

Status: desk research, no installs performed. This brief feeds the FULL PARITY
sub-spec for adding sex-chromosome support to TorchGWAS, which today has zero
sex-chromosome handling (`_parse_fam` in `torchgwas/io/plink.py:172` reads only
`IID` from the .fam file and discards `sex`/`phenotype`; no `chrX/Y/MT/PAR` code
path exists anywhere under `torchgwas/`). All factual claims below are cited to
primary sources (tool docs, papers, official browsers) and any unverified item
is explicitly flagged.

---

## 1. Standards landscape overview

The reference tool set for human X/Y/PAR/MT GWAS, with the canonical docs section
that defines each tool's behaviour:

- **PLINK 2.0** — defines the de-facto standard. The `--xchr-model` option
  (Christopher Chang's docs) specifies the dosage model for non-PAR chrX, and
  `--split-par` / `--merge-par` toggle whether PAR1/PAR2 are treated as separate
  pseudo-autosomal contigs or as part of chrX.
  Docs: <https://www.cog-genomics.org/plink/2.0/data#split_par> and
  <https://www.cog-genomics.org/plink/2.0/assoc#glm> (search "xchr-model").
- **REGENIE** — documents X-chromosome handling in the "Step 1" / "Step 2"
  reference under the heading "X chromosome". REGENIE expects PAR to have been
  split out (PLINK convention) and applies a male-dosage doubling model
  controlled by `--xchr-model`-style behaviour driven by the `.bim` chromosome
  code 23 plus the `.fam` sex column.
  Docs: <https://rgcgithub.github.io/regenie/options/> (section "Genetic data
  input — X chromosome").
- **BOLT-LMM** — supports chrX through the `--lmmInfOnly`/`--lmm` scans when
  the `.fam` sex column is populated and the `.bim` codes chrX as `23` (or
  `X`); the manual is explicit that BOLT scans X with hemizygous males coded
  as 0/2 dosages (i.e. PLINK's `--xchr-model 2` semantics).
  Docs: <https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>
  (section "BOLT-LMM v2.4 software" → "Sex chromosome handling" / `--xchr`).
- **SAIGE** — handles chrX through the `--chrom` argument in step 2 plus the
  `--sexCol`/`--FemaleOnly`/`--MaleOnly` flags, with male hemizygote dosage
  doubling done internally for binary-trait scans. Documented in the SAIGE
  wiki for `step2_SPAtests.R`.
  Docs: <https://saigegit.github.io/SAIGE-doc/docs/single_step2.html> (look for
  "X chromosome" / `--is_rewrite_XnonPAR_forMales`).
- **GCTA** — documents chrX in the `--make-grm-x` / `--make-grm-xchr` and
  `--mlma`/`--fastGWA` reference. The `--make-grm-xchr` option builds an
  X-chromosome GRM with an internal `--dc`-style dosage compensation choice;
  `--autosome` and `--autosome-num` explicitly bound the autosomes for the
  default GRM build.
  Docs: <https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM> (section
  "Making a GRM from the X chromosome").

These five tools collectively define the operational standard. PLINK 2.0 sets
the file-format and dosage-model conventions; REGENIE/BOLT/SAIGE consume those
conventions; GCTA sets the GRM conventions.

---

## 2. Algorithm summary — X-inactivation models

Notation: let $g_{ij} \in \{0,1,2\}$ be the additive allele-count dosage at SNP
$j$ for individual $i$, $s_i \in \{F, M\}$ be sex, and $\beta_j$ be the SNP
effect on the trait $y$. For chrX non-PAR variants, females are diploid
($g_{ij}\in\{0,1,2\}$) and males are hemizygous ($g_{ij}\in\{0,1\}$). The
question is how to harmonise male and female dosages on the same regression
scale.

### 2a. Uniform skewing (Clayton 2008; PLINK `--xchr-model 2`)

Assume one X allele in females is fully silenced by random X-inactivation, so a
single male X allele is biologically equivalent to two female alleles of the
same type. The male dosage is doubled before the test:

$$
\tilde g_{ij} \;=\;
\begin{cases}
 g_{ij},                & s_i = F\quad (g_{ij}\in\{0,1,2\}) \\
 2\,g_{ij},             & s_i = M\quad (g_{ij}\in\{0,1\})
\end{cases}
$$

so $\tilde g_{ij}\in\{0,2\}$ for males and the linear model
$E[y_i] = \alpha + \beta_j \tilde g_{ij}$ has the same per-allele effect in both
sexes. This is the default in BOLT-LMM and matches PLINK 2.0
`--xchr-model 2`. Reference: Clayton, *Testing for association on the X
chromosome*, *Biostatistics* 9(4):593–600, 2008,
<https://doi.org/10.1093/biostatistics/kxn007>, equation (2). PLINK 2.0
mapping: <https://www.cog-genomics.org/plink/2.0/assoc#glm> (search
`--xchr-model`).

### 2b. Random skewing (PLINK `--xchr-model 1`)

Assume *no* dosage compensation: a male hemizygous "1" carries one functional
allele, equivalent in effect to a heterozygous female (one of two alleles is
silenced at random). Male dosage is left as $g_{ij}\in\{0,1\}$:

$$
\tilde g_{ij} \;=\;
\begin{cases}
 g_{ij},                & s_i = F \\
 g_{ij},                & s_i = M
\end{cases}
$$

with $\tilde g_{ij}\in\{0,1\}$ for males and $\{0,1,2\}$ for females. This is
PLINK 2.0 `--xchr-model 1`. Reference: Clayton 2008 equation (1) and the
`--xchr-model` table in the PLINK 2.0 association docs
<https://www.cog-genomics.org/plink/2.0/assoc#glm>.

### 2c. Sex-stratified / per-gene skewing (Tukiainen 2017)

Some X-linked genes escape inactivation (XCI escape) entirely or partially;
others are subject to skewed inactivation that varies by tissue. For an "escape"
variant the female effective dosage is the full $g_{ij}\in\{0,1,2\}$ with no
allele silenced, so the *female* per-allele effect is the same as the male
per-allele effect *without* doubling. A general per-variant skew parameter
$\phi_j\in[0,1]$ (fraction of X-inactivated cells expressing the reference
allele) gives:

$$
\tilde g_{ij} \;=\;
\begin{cases}
 \phi_j\, g_{ij},                                & s_i = F,\;\text{escape variant} \\
 g_{ij},                                         & s_i = F,\;\text{XCI variant} \\
 2\,g_{ij},                                      & s_i = M
\end{cases}
$$

with $\phi_j = 1$ recovering the uniform-skewing model. Per-gene $\phi_j$ values
for ~683 X-linked genes are tabulated in Tukiainen et al., *Landscape of X
chromosome inactivation across human tissues*, *Nature* 550:244–248, 2017,
Supplementary Tables 8–13, <https://doi.org/10.1038/nature24265>. A practical
sex-stratified meta-analysis alternative (no $\phi_j$ assumption) is the
Stouffer / Z-score combination of male-only and female-only scans described in
König et al., *How to include chromosome X in your genome-wide association
study*, *Genet. Epidemiol.* 38(2):97–103, 2014, §"Sex-stratified analysis",
<https://doi.org/10.1002/gepi.21782>.

### 2d. PLINK 2.0 `--xchr-model 0/1/2` semantics

From the PLINK 2.0 association reference
<https://www.cog-genomics.org/plink/2.0/assoc#glm> (section "Linear and
logistic/Firth regression", `--xchr-model`):

- `--xchr-model 0` — chrX variants are *skipped* in the regression (the option
  to "exclude X entirely").
- `--xchr-model 1` — male hemizygous genotypes are coded as $\{0,1\}$ (random
  X-inactivation; haploid male dosage). PLINK 1.9 default.
- `--xchr-model 2` — male hemizygous genotypes are coded as $\{0,2\}$ (uniform
  skewing / dosage doubling). PLINK 2.0 default for `--glm`.
- `--xchr-model 3` — adds a sex-by-genotype interaction term to the regression
  (model fitted with sex, genotype, and sex × genotype). Documented in the same
  table.

Confirm against the canonical PLINK 2.0 docs page above; the same table appears
in <https://www.cog-genomics.org/plink/2.0/data#chr_set> for the chromosome-set
discussion.

---

## 3. PAR (pseudo-autosomal region) and XTR coordinates

The pseudo-autosomal regions PAR1/PAR2 are diploid in both sexes (X–Y
recombination occurs there) and must be treated as autosomal. The
X-transposed region (XTR) on chrX is *not* a PAR — it is X-specific but shares
high sequence identity with chrY.

- **PAR1 (Xp22.33 / Yp11.2)**:
  - GRCh37: chrX 60,001–2,699,520; chrY 10,001–2,649,520. Source: UCSC GRCh37
    sequence reference,
    <https://genome.ucsc.edu/cgi-bin/hgGateway?db=hg19> → "Sequences" notes,
    and Ensembl GRCh37 PAR coordinates
    <https://www.ensembl.org/info/genome/genebuild/human_PARS.html>.
  - GRCh38: chrX 10,001–2,781,479; chrY 10,001–2,781,479. Source: Ensembl
    Human PAR documentation
    <https://www.ensembl.org/info/genome/genebuild/human_PARS.html>.

- **PAR2 (Xq28 / Yq12)**:
  - GRCh37: chrX 154,931,044–155,260,560; chrY 59,034,050–59,363,566. Source:
    Ensembl GRCh37 PAR table (URL above).
  - GRCh38: chrX 155,701,383–156,030,895; chrY 56,887,903–57,217,415. Source:
    Ensembl GRCh38 PAR table (URL above).

- **XTR (X-transposed region; not a PAR)**: chrX ≈88.4–92.4 Mb on GRCh37 and
  ≈89.1–93.1 Mb on GRCh38, aligned to chrY 3.4–7.4 Mb (Mueller et al., *Nature
  Genetics* 2008). Source: NCBI Genome Reference Consortium summary,
  <https://www.ncbi.nlm.nih.gov/grc/human> and the original Mueller et al. paper
  <https://doi.org/10.1038/ng.272>. UNVERIFIED-DETAIL: exact base-pair endpoints
  vary slightly between sources; the consensus 4-Mb interval above is the
  canonical figure cited in the GRC summary, but for production use the spec
  should pin to a specific assembly's GFF3 and not to memory.

How the major tools distinguish PAR from non-PAR chrX:

- **PLINK 2.0** uses `--split-par <build>` (where `<build>` is `hg18`/`hg19`/
  `hg38` or explicit coordinates) to relabel PAR1 → contig `PAR1` (code 25 in
  PLINK numeric) and PAR2 → `PAR2` (also code 25), leaving non-PAR chrX as
  contig `X`/code 23. The inverse `--merge-par` collapses them back. Docs:
  <https://www.cog-genomics.org/plink/2.0/data#split_par>. PAR1/PAR2 then enter
  the autosomal pipeline (no `--xchr-model` adjustment).
- **REGENIE** expects the PAR split to have been done upstream by PLINK 2.0
  `--split-par`; non-PAR chrX is coded `23`/`X`, PAR is `25`/`XY`. Docs (Step 2
  options): <https://rgcgithub.github.io/regenie/options/> ("X chromosome").
- **BOLT-LMM** treats `.bim` chromosome codes 23 (X), 24 (Y), 25 (XY/PAR) per
  the PLINK 1.9 numeric convention. Docs:
  <https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>.

The PLINK chromosome-code table (1=autosomes 1–22, 23=X non-PAR, 24=Y, 25=XY
PAR, 26=MT) is the lingua franca; REGENIE/BOLT/SAIGE/GCTA all consume it.
Reference: <https://www.cog-genomics.org/plink/2.0/data#chr_set>.

---

## 4. chrY handling

Standard practice for chrY:

- **Males-only scan**: chrY is haploid in males and absent in females. All
  reference tools restrict chrY scans to the male subset and emit `NA` for
  females. PLINK 2.0 documents this in the `--filter-males` / `--keep-males`
  reference and in the `--chr y` / `--not-chr y` filters
  <https://www.cog-genomics.org/plink/2.0/filter#sex>.
- **No GRM contribution**: GCTA's `--make-grm` (and `--make-grm-x`) explicitly
  exclude chrY from any GRM build; only autosomes (codes 1–22) plus optionally
  chrX (`--make-grm-xchr`) are GRM substrates. Docs:
  <https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM>.
- **PAR on Y**: variants in the Y PAR1/PAR2 intervals (coordinates above) are
  identical-by-state with the corresponding X PAR positions; tools therefore
  *deduplicate* Y-PAR onto X-PAR (PLINK `--split-par` keeps only the chrX copy
  by convention). Docs: <https://www.cog-genomics.org/plink/2.0/data#split_par>.
- **Non-PAR chrY scan model**: hemizygous additive model
  $\tilde g_{ij} = g_{ij}\in\{0,1\}$ with no doubling (the male is haploid; no
  female reference exists). REGENIE / SAIGE / BOLT documents (URLs in §1) all
  state Y is "scanned in males only with haploid coding".

UNVERIFIED-DETAIL: the exact BOLT-LMM v2.4 behaviour for chrY (whether it
silently drops chrY or runs males-only) is documented as "not explicitly
supported" in the BOLT manual; users typically pre-filter to males with PLINK
before invoking BOLT for chrY. Re-verify against the BOLT manual at install
time.

---

## 5. Mitochondrial (MT / chrM) handling

- **Haploid coding**: mtDNA is inherited maternally and is present in many
  copies per cell, but is treated as a single haploid genome per individual for
  GWAS. Genotypes are coded $g_{ij}\in\{0,1\}$ (reference vs. alternate). PLINK
  2.0 codes MT as chromosome `26` / `MT`
  <https://www.cog-genomics.org/plink/2.0/data#chr_set>.
- **Heteroplasmy**: a non-trivial fraction of mtDNA variants are heteroplasmic
  (a mixture of reference and alternate alleles within an individual's
  mitochondrial population). Standard mtDNA-GWAS protocols filter out
  heteroplasmic sites at a threshold (typically heteroplasmy fraction >5–10%
  filtered, or modelled as a continuous dosage). Reference: Yonova-Doing et
  al., *Nat. Genet.* 53:982–993 (2021), §"Methods — mtDNA variant calling and
  filtering", <https://doi.org/10.1038/s41588-021-00868-1>.
- **Scan model**: linear (or logistic) regression with mtDNA variants as
  predictors, *no GRM contribution from mtDNA itself* (the autosomal GRM is
  reused as the random-effect covariance). MT PCs are sometimes added as
  fixed-effect covariates to control for haplogroup structure. Reference:
  Hägg et al., *Nat. Commun.* 12:6147 (2021), §Methods,
  <https://doi.org/10.1038/s41467-021-26424-3>.
- **Tooling**: PLINK 2.0 will scan chrM with `--chr 26` / `--chr MT` if
  variants are present; REGENIE / BOLT / SAIGE do not document explicit mtDNA
  support and most published mtDNA-GWAS pipelines invoke PLINK 2.0 `--glm`
  directly with `--keep` filters (e.g. Hägg et al. 2021 above).
  UNVERIFIED: I did not locate REGENIE/BOLT-specific mtDNA documentation; this
  should be re-checked at sub-spec time. The mtDNA-Server / MtDNAdb
  (<https://mtdna-server.uibk.ac.at/>) provides upstream variant-calling and
  haplogroup assignment but is not a GWAS tool itself.

---

## 6. MAF computation on sex chromosomes

For an autosomal SNP with $n$ diploid individuals the allele frequency is
$\hat p = \sum_i g_{ij} / (2n)$. On chrX in a mixed-sex cohort with $n_F$
females and $n_M$ males, the correct estimator weights males as hemizygous:

$$
\hat p_X \;=\; \frac{\sum_{i:\,s_i=F} g_{ij} \;+\; \sum_{i:\,s_i=M} g_{ij}}
                    {2 n_F + n_M}
$$

Note the denominator is $2 n_F + n_M$ (not $2(n_F + n_M)$): each male
contributes one allele to the count, each female two. PLINK 2.0 documents this
explicitly in the `--freq` / `--maf` reference under "Sex chromosome handling"
and computes per-chromosome MAFs accordingly. Source:
<https://www.cog-genomics.org/plink/2.0/basic_stats#freq>. PAR variants use
the autosomal $2n$ denominator (males are diploid in PAR). chrY uses
$\sum_{i:\,s_i=M} g_{ij} / n_M$ (males-only, single allele each).

GCTA's `--freq` and the GCTA-COJO sumstats format expect the same per-chromosome
MAF convention. Source: <https://yanglab.westlake.edu.cn/software/gcta/#COJO>
(MAF column description).

---

## 7. GRM contribution from sex chromosomes

- **GCTA** ships *separate* commands for autosomal and X-chromosome GRMs:
  `--make-grm` (autosomes only by default; controlled by `--autosome-num`) and
  `--make-grm-xchr` (X only). The X-GRM uses a dosage-compensation parameter
  internally — the docs describe equal-variance vs. equal-X-dosage scaling
  options. Recommended use is to fit a two-GRM mixed model: $G_{auto}$ for the
  bulk of polygenic variance, $G_X$ as a separate variance component for the X
  contribution. Docs:
  <https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM> (sections "Making
  a GRM from autosome SNPs" and "Making a GRM from the X chromosome"). chrY and
  chrM contribute no variance components.
- **BOLT-LMM** builds the GRM from autosomal SNPs only (`--bfile` with
  `--exclude` chrX) by default; the manual recommends excluding chrX, chrY, and
  chrM from the GRM and scanning chrX as a separate fixed-effect SNP set.
  Docs: <https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>
  ("Best practices" → "Excluding sex chromosomes from the model SNP set").
- **REGENIE** Step 1 (whole-genome regression / LOCO predictions) excludes
  chrX/Y/M by default; chrX is scanned in Step 2 with male-dosage handling
  documented in <https://rgcgithub.github.io/regenie/options/> ("X
  chromosome"). The Step-1 LOCO predictions for chrX use a separate X-specific
  ridge fit (documented as an experimental option in recent REGENIE versions —
  re-verify at install).
- **SAIGE** likewise builds the sparse GRM from autosomes only and applies the
  X dosage adjustment in Step 2 SPA tests
  <https://saigegit.github.io/SAIGE-doc/docs/single_step2.html>.

The strong consensus is: **autosomal GRM as the primary random-effect kernel;
chrX GRM optional second kernel; chrY/chrM never enter the GRM.** This maps
naturally onto our existing `MultiKernelLMM` (`torchgwas/models/multi_kernel_lmm.py`).

---

## 8. HWE testing on sex chromosomes

- **chrX**: HWE is only meaningful for the *female* subset, because males are
  hemizygous and cannot be heterozygous. PLINK 2.0 computes
  `--hardy` p-values from females only for non-PAR chrX. Source:
  <https://www.cog-genomics.org/plink/2.0/basic_stats#hardy> ("X chromosome
  variants are tested using only female genotypes").
- **chrY** and **chrM**: HWE is undefined (haploid; no genotype distribution to
  test). PLINK 2.0 emits no HWE p-value for these; the `--hardy` output is
  blank/NA. Same source as above.
- **PAR**: treated as autosomal for HWE — full diploid mixed-sex sample.
  Source: same PLINK 2.0 docs page.

For our existing pipeline this means `compute_variant_qc` in
`torchgwas/preprocess/qc.py` must be sex-aware: it already computes
`hwe_p` autosomally (line 130) but has no path to subset to females for chrX or
to skip HWE entirely for chrY/MT.

---

## 9. Per-tool X-chromosome flag surfaces

### PLINK 2.0
Source: <https://www.cog-genomics.org/plink/2.0/> (top-level), with sub-pages
linked below.

- `--xchr-model {0,1,2,3}` — male dosage coding for non-PAR chrX (see §2d).
  Docs: <https://www.cog-genomics.org/plink/2.0/assoc#glm>.
- `--split-par <build|coords>` — split PAR1/PAR2 into separate `XY` contigs
  for autosomal-style handling. Docs:
  <https://www.cog-genomics.org/plink/2.0/data#split_par>.
- `--merge-par` — inverse of `--split-par`.
- `--filter-males`, `--filter-females`, `--keep-males`, `--keep-females`,
  `--remove-males`, `--remove-females` — sex-based subset filters. Docs:
  <https://www.cog-genomics.org/plink/2.0/filter#sex>.
- `--chr X`, `--chr Y`, `--chr MT`, `--chr XY`, `--not-chr` — chromosome
  filters; numeric codes 23/24/25/26 also accepted. Docs:
  <https://www.cog-genomics.org/plink/2.0/filter#chr>.
- `--check-sex`, `--impute-sex` — infer or verify reported sex from chrX
  heterozygosity / Y-call rate. Docs:
  <https://www.cog-genomics.org/plink/2.0/basic_stats#check_sex>.
- `--update-sex <file>` — overwrite `.fam` sex column. Docs:
  <https://www.cog-genomics.org/plink/2.0/data#update_indiv>.
- `--chr-set <N> [no-x] [no-y] [no-xy] [no-mt]` — for non-human ploidy/karyotype
  setups. Docs: <https://www.cog-genomics.org/plink/2.0/data#chr_set>.

### REGENIE
Source: <https://rgcgithub.github.io/regenie/options/>.

- chrX is scanned by passing the appropriate `--bgen`/`--bed` and `--chr 23`
  (or `X`) in Step 2.
- `--phenoCol`/`--covarColList` must include sex when chrX is in the input.
- Section "X chromosome" documents the implicit dosage doubling (PLINK
  `--xchr-model 2` analogue) and the requirement that PAR be split out. Newer
  REGENIE versions document an explicit `--ref-first` behaviour for X — verify
  at install.
- `--with-bgi` / `--ref-first` / `--firth` / `--spa` flags interact with chrX
  scans (no separate X-specific switch — sex column drives behaviour). Same
  docs page.

UNVERIFIED-DETAIL: REGENIE has had several X-chromosome behaviour changes
between v3.x and v4.x; the spec should pin to a specific version's docs page
(retrieved via `git tag` of the regenie repo) rather than the floating
`docs/options` URL.

### BOLT-LMM
Source:
<https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>.

- `--lmm` / `--lmmInfOnly` — runs the mixed-model scan including chrX if
  present in the `.bim`.
- Sex column in `.fam` is used for male hemizygote dosage doubling
  (uniform-skewing model, `--xchr-model 2` analogue); no explicit BOLT flag
  toggles this.
- Manual recommends `--modelSnps` to *exclude* chrX/Y/M from the GRM build (see
  §7).
- No native PAR detection — relies on the upstream `.bim` contig labels.

### SAIGE
Source: <https://saigegit.github.io/SAIGE-doc/docs/single_step2.html>.

- `--chrom <N>` — restricts Step-2 scan to a single chromosome (e.g.
  `--chrom X`).
- `--sexCol <name>` — names the phenotype column holding sex (1=male, 2=female
  per PLINK convention).
- `--MaleOnly` / `--FemaleOnly` — restrict the analysis to one sex (used for
  chrY scans).
- `--is_rewrite_XnonPAR_forMales TRUE/FALSE` — toggles the male hemizygote
  dosage doubling (PLINK `--xchr-model 2` analogue).
- `--X_PARregion <coords>` — supplies PAR coordinates so SAIGE can distinguish
  PAR vs. non-PAR on the same chromosome label.

### GCTA
Source: <https://yanglab.westlake.edu.cn/software/gcta/>.

- `--make-grm-xchr` — build chrX GRM (separate from autosomal GRM).
- `--autosome-num <N>` — set the autosome count (defaults to 22 for human).
- `--autosome` — restrict to autosomes (used in `--make-grm`).
- `--mlma` / `--fastGWA-mlm` — include chrX automatically when present, with
  internal `--dc 0/1` dosage-compensation choice (1=full dosage compensation
  i.e. doubling, 0=no compensation). Docs:
  <https://yanglab.westlake.edu.cn/software/gcta/#MLMassociationanalysis>.

---

## 10. Risk flags for our integration

These are the concrete points where TorchGWAS's current state will collide with
sex-chromosome support, grounded in actual file contents:

- **Polyploid-first × hemizygous males.** Our genotype tensors are float
  $G \in [0, k]$ with `ploidy=k` as a *scalar* parameter passed everywhere
  (`compute_allele_frequencies(G, ploidy=2)` in
  `torchgwas/preprocess/standardize.py:9`, same pattern in
  `torchgwas/preprocess/qc.py:67` and `torchgwas/linalg/kinship.py:35`). The
  natural extension is a *per-sample, per-variant* effective ploidy
  $k_{ij}\in\{1,2,k\}$ rather than a global scalar. For diploid human chrX the
  matrix is $k_{ij}=1$ for males on non-PAR X and $k_{ij}=2$ everywhere else.
  This requires either (a) broadening the `ploidy` argument to accept a
  $(n,m)$ tensor in the variant-QC / standardisation / GRM paths, or
  (b) splitting the chrX path into its own code branch with explicit male/female
  subsetting. Option (a) is the cleaner long-term fit with our polyploid-first
  charter but is a wider blast radius. Option (b) is easier but creates an
  X-specific code island that polyploid users won't benefit from.

- **Sex column dropped at the I/O layer.** `_parse_fam` in
  `torchgwas/io/plink.py:172–193` reads only the `IID` (column 2) and
  *discards* the `sex` (column 5) and `phenotype` (column 6) fields; there is
  also no `sex` plumbing through `torchgwas/io/vcf.py`,
  `torchgwas/io/bgen.py`, or `torchgwas/io/plink2.py` (verified by `grep -n
  sex` returning only the docstring at `plink.py:176` and a covariate-doc
  reference in `preprocess/covariates.py:23`). **All sex-chromosome work is
  blocked until `.fam`/PSAM/VCF sample-level sex is parsed and propagated.**
  This is a one-line file-read change but a multi-file API change — every
  reader must expose a `sample_sex: Tensor[int8] | None` alongside
  `sample_ids`. A downstream `SampleMeta` dataclass mirroring `VariantMeta`
  (currently absent from `torchgwas/models/base.py:24`) is the natural
  carrier.

- **`VariantMeta.chr` is a list[str], so the chr label is preserved end-to-end.**
  Verified at `torchgwas/models/base.py:28` and re-emitted by
  `torchgwas/io/plink.py:_parse_bim:200–223`,
  `torchgwas/io/vcf.py:_emit:146–153`, and
  `torchgwas/scan/unified.py:27–51,211`. This is a *positive* finding: we do
  not need to refactor variant metadata to carry chromosome info, only to
  *interpret* it (PAR detection, X non-PAR detection).

- **30+ existing models that need sex-aware logic.** I count **41** model files
  under `torchgwas/models/` (excluding `__init__.py` and `base.py` brings it to
  39 functional models). Of these, 17 take a `ploidy` parameter explicitly
  (`grep -l ploidy torchgwas/models/`); the rest implicitly assume `ploidy=2`.
  Every model that calls `score_chunk(G, ...)` needs sex-aware variant
  filtering or a per-variant `effective_ploidy` argument. Concrete top-priority
  list (the V1-core seven plus the ones most likely to be invoked on human
  chrX): `glm.py`, `single_trait_lmm.py`, `multi_trait_lmm.py`,
  `farmcpu.py`, `blink.py`, `multi_kernel_lmm.py`, `lmm_gxe.py`,
  `binary_glm.py`, `binary_glmm.py`, `ordinal_glm.py`, `ordinal_glmm.py`,
  `multinomial_glm.py`, `multinomial_glmm.py`, `survival_glmm.py`,
  `set_based.py`, `bayesian_vs.py`, `knockoff_lmm.py`, `gu_lmm.py`,
  `conditional_lmm.py`, `within_family_lmm.py`,
  `multi_trait_multi_env_lmm.py`, `ocf_lmm.py`, `lro_lmm.py`,
  `multi_env_lmm.py`, `multi_env_glmm.py`, `threshold_linear.py`,
  `haplotype_gwas.py`, `haplotype_multi.py`, `haplotype_novel.py`,
  `rr_lmm.py`, `rr_met.py`, `rr_spatial.py`, `joint_qtl.py`, `iterative.py`,
  `sparse_lmm.py`. That is **35 models** that need at minimum a sex-aware
  chunk filter; some (FarmCPU/BLINK iterative-pseudo-QTN paths) need
  per-iteration sex-stratified scans. The polyploid-specific helpers in
  `lmm_multi_fit.py` and `lmm_multi.py` (the FA(k) multi-trait fits) will
  rarely run on human chrX and can be deferred (mark "non-applicable for V1
  refactor").

- **`linalg/kinship.py` and `linalg/kinship_polyploid.py` need an X kernel.**
  The `GRMMetadata` dataclass at `kinship.py:28` already carries
  `loco_chr: str | None` (good for LOCO), but no `xchr_kernel: bool` or
  per-chromosome dosage-compensation flag. Need to add a sibling
  `grm_vanraden_xchr(G, sample_sex, model="uniform"|"random")` and decide
  whether to expose it as a second kernel through `MultiKernelLMM` (the GCTA
  approach, see §7) or as a unified GRM (the BOLT/REGENIE approach). The
  master spec recommends the multi-kernel approach because it composes with
  our existing `MultiKernelLMM` infrastructure.

- **`preprocess/qc.py` and `preprocess/standardize.py` are autosome-implicit.**
  `compute_variant_qc` (line 67) takes no sex/chr arguments; `compute_maf` and
  `_compute_hwe_pvalue` (line 130) are autosome-formulae. These need sex-aware
  per-variant code paths driven by the `chr` field of `VariantMeta` and a new
  `sample_sex` Tensor. The HWE-on-females-only path for chrX is the most
  surgical change; the male-doubling MAF formula is a bigger refactor because
  every downstream model reads `maf` from the `VariantQCStats` dataclass.

- **`io/plink.py` BED reader preserves `chr` labels but not numeric codes.**
  Confirmed by `grep -n chrom torchgwas/io/plink.py`:
  `_parse_bim` at line 200 stores `parts[0]` as a raw string (`"23"`, `"X"`,
  `"PAR1"`, `"chrX"`, etc.). We have **no canonicalisation layer** that maps
  these heterogeneous labels onto a single internal vocabulary
  (`{1..22, "X", "Y", "XY", "MT"}`). PLINK 1.9 numeric (23/24/25/26), PLINK
  2.0 alphanumeric (`X`/`Y`/`XY`/`MT`), VCF UCSC-style (`chrX`/`chrY`/`chrM`),
  and Ensembl-style (`X`/`Y`/`MT`) all need a normaliser. A new
  `torchgwas/io/chrom.py` with `canonicalize_chr(label) -> str` and
  `is_par(chr, pos, build) -> bool` is the natural home; the per-build PAR
  coordinate tables from §3 live here.

- **`scan/UnifiedScanner` chunking respects but does not annotate chromosome
  boundaries.** Verified at `torchgwas/scan/unified.py:27–51`: the chunker
  accumulates `chr` per chunk but does not split chunks at chromosome
  boundaries, and does not pass per-chromosome metadata flags to the model.
  Needed: (i) optional `split_at_chrom_change=True` to ensure no mixed-chr
  chunks (so that per-chunk male-doubling is uniform), and (ii) a
  `ChunkContext` dataclass passed to `score_chunk` that carries
  `is_xchr_nonpar: bool`, `is_y: bool`, `is_mt: bool`, `is_par: bool`,
  `sample_sex: Tensor`. This is a `BaseModel` protocol change → touches every
  model.

- **Autosome-only parity tests.** All current GEMMA / GAPIT / GWASpoly parity
  tests in `tests/test_*` use autosomal fixtures; no chrX fixtures exist
  (verified by `find tests -name '*xchrom*' -o -name '*chrX*'` returning
  nothing). Per master-spec invariant **R-MOD-1**, the refactor MUST NOT
  regress these. Concrete mitigation: (a) all sex-aware code paths must be
  no-ops when `sample_sex is None` and all variants have autosomal `chr`
  labels; (b) add an explicit regression test that runs the full V1 suite with
  `sample_sex=None` and asserts byte-for-byte (within tolerance) parity with
  the pre-refactor results.

- **CLI surface.** `torchgwas/cli.py` (37 subcommands per CLAUDE.md) has no
  `--xchr-model`, `--split-par`, `--filter-males`, etc. flags. Need to plumb
  these through the four V1 scan commands at minimum (`glm-scan`, `lmm-scan`,
  `mvlmm-scan`, `farmcpu-scan`, `blink-scan`); design question for the spec:
  per-subcommand flag duplication vs. a shared `xchrom_flags()` argparse
  group.

---

## 11. File enumeration

Counts derived from `find torchgwas -name '*.py' | wc -l` = **207 total Python
files**. Files likely to need modification, grouped by module:

### `torchgwas/io/` (16 files; ~10 likely touched)
Sex-column parsing, chr canonicalisation, PAR detection.
- `io/plink.py` — `_parse_fam` must read sex; `_parse_bim` must canonicalise
  chr.
- `io/plink2.py` — same for PSAM/PVAR.
- `io/vcf.py` — pull sex from PEDIGREE/SAMPLE header if present, otherwise
  require a sidecar.
- `io/bgen.py` — BGEN .sample file parsing for sex.
- `io/hapmap.py` — HapMap files lack sex; add sidecar requirement.
- `io/zarr.py`, `io/hdf5.py`, `io/numeric.py` — propagate sex attr.
- `io/detect.py`, `io/validate.py`, `io/regions.py` — chr canonicalisation +
  PAR-aware region filtering.
- **NEW**: `io/chrom.py` — canonicalisation table, PAR/XTR coordinate tables
  per build, `is_par()`, `is_xchr_nonpar()`, `is_y()`, `is_mt()`.

### `torchgwas/preprocess/` (15 files; ~6 likely touched)
- `preprocess/qc.py` — sex-aware MAF, female-only HWE for chrX, skip HWE for
  Y/MT, missingness per-sex.
- `preprocess/standardize.py` — male-doubling for chrX, hemizygous coding for
  Y/MT.
- `preprocess/covariates.py` — auto-add sex when chrX is in the scan.
- `preprocess/dosage_call.py`, `preprocess/dosage_uncertainty.py` — verify
  haploid-aware EM (currently assumes ploidy from a scalar arg).
- `preprocess/phase.py` / `preprocess/phase_polyorigin.py` — N/A for human
  chrX (PolyOrigin is autopolyploid F1 only); document as out-of-scope.

### `torchgwas/linalg/` (11 files; ~4 likely touched)
- `linalg/kinship.py` — add `grm_vanraden_xchr`, sex-aware variance scaling.
- `linalg/kinship_advanced.py` — leave-one-region-out interplay with chrX.
- `linalg/kinship_polyploid.py` — confirm not invoked on human chrX path.
- `linalg/sparse_grm.py` — sparse-GRM X kernel for SAIGE-style scans.

### `torchgwas/models/` (41 files; ~35 likely touched, see §10 for the list)
Every `BaseModel` subclass that accepts a `score_chunk(G, ...)` call needs a
new `ChunkContext` parameter (see §10 risk flag on `UnifiedScanner`). Critical
path (V1 + categorical V2): the 17 listed in §10 first; the haplotype/RR/MET
families second; the polyploid-specific fits (FA(k), threshold-linear) marked
"diploid-X-aware not required" since they target plant breeding.

### `torchgwas/scan/` (5 files; ~3 likely touched)
- `scan/unified.py` — chrom-boundary chunk splits, `ChunkContext` plumbing.
- `scan/adapters.py` — sex-mask helpers (`mask_to_males`, `mask_to_females`).
- `scan/strategies.py` — sex-stratified scan strategy.
- `scan/prefetch.py` — chr-aware pre-fetch ordering (X scanned last).

### `torchgwas/stats/` (15 files; ~2 likely touched)
- `stats/multipletesting.py` — separate multiple-testing pool for chrX
  (Carayol et al. 2014 recommend per-chr Bonferroni correction).
- `stats/genomic_control.py` — λGC computed per-chr with chrX flagged
  separately.

### `torchgwas/cli.py` (1 file; touched)
Add `--xchr-model {0,1,2,3}`, `--split-par {hg19,hg38,coords}`,
`--filter-males` / `--filter-females`, `--xchr-grm-kernel {none,separate,joint}`
flags to the V1 scan subcommands.

### `torchgwas/results/` (3 files; ~1 likely touched)
- `results/tables.py` — emit per-chr summary rows; flag chrX with model used.

### Test infrastructure (`tests/`; new)
- New fixtures: `tests/fixtures/xchrom_diploid.bed/bim/fam` with PAR1 / non-PAR
  X / PAR2 / chrY / chrM variants.
- New parity tests: against PLINK 2.0 `--glm --xchr-model 2`, REGENIE chrX
  scan, BOLT chrX scan.
- Regression net: `tests/test_xchrom_invariance.py` asserting that
  `sample_sex=None` + autosomal chr labels reproduces pre-refactor byte-equal
  outputs.

### Files NOT touched
- `torchgwas/_native/`, `csrc/` — kernels are sex-agnostic; the sex masking
  happens in the Python wrapper before the native call.
- `torchgwas/multiomics/`, `torchgwas/postgwas/`, `torchgwas/pgs/`,
  `torchgwas/annotate/`, `torchgwas/viz/`, `torchgwas/ld/` — downstream of the
  scan; consume the sumstats per-chr without needing sex-chr-internal
  changes. Two exceptions: `viz/_manhattan.py` should colour chrX/Y/MT
  distinctly and `ld/_blocks.py` should warn-and-skip chrX block calls (LD
  block algorithms assume diploid HWE).

### Refined file-count estimate
The master spec said "30+". The grounded count is:
- io: 10 + 1 new = 11
- preprocess: 6
- linalg: 4
- models: ~35
- scan: 3
- stats: 2
- cli: 1
- results: 1
- viz: 1
- ld: 1 (warn/skip only)

**Total: ~65 files touched** (with ~35 of those being model files needing the
identical `ChunkContext` plumbing change, so the *unique* design surface is
much smaller — roughly 30 files of substantive code edits plus 35 of mechanical
plumbing).

---

## 12. Citations

Primary sources, with URLs and section pointers. Twenty distinct citations.

1. **Clayton, D. (2008).** *Testing for association on the X chromosome.*
   *Biostatistics* 9(4):593–600.
   <https://doi.org/10.1093/biostatistics/kxn007>. Equations (1)–(2) define the
   random-skewing and uniform-skewing male dosage models.

2. **Tukiainen, T. et al. (2017).** *Landscape of X chromosome inactivation
   across human tissues.* *Nature* 550:244–248.
   <https://doi.org/10.1038/nature24265>. Supplementary Tables 8–13 list
   per-gene XCI escape status.

3. **König, I. R., Loley, C., Erdmann, J., Ziegler, A. (2014).** *How to
   include chromosome X in your genome-wide association study.* *Genet.
   Epidemiol.* 38(2):97–103. <https://doi.org/10.1002/gepi.21782>. Section
   "Sex-stratified analysis" defines the meta-analytic alternative.

4. **PLINK 2.0 docs — `--glm` / `--xchr-model`.**
   <https://www.cog-genomics.org/plink/2.0/assoc#glm>. Definitive table of
   `--xchr-model 0/1/2/3` semantics.

5. **PLINK 2.0 docs — `--split-par` / `--merge-par`.**
   <https://www.cog-genomics.org/plink/2.0/data#split_par>. PAR coordinate
   handling and the `XY`/`PAR1`/`PAR2` contig conventions.

6. **PLINK 2.0 docs — chromosome set / `--chr-set`.**
   <https://www.cog-genomics.org/plink/2.0/data#chr_set>. Numeric chromosome
   codes (23=X, 24=Y, 25=XY/PAR, 26=MT).

7. **PLINK 2.0 docs — sex filters (`--filter-males`/`--filter-females`/
   `--keep-males`/`--keep-females`).**
   <https://www.cog-genomics.org/plink/2.0/filter#sex>.

8. **PLINK 2.0 docs — `--check-sex` / `--impute-sex`.**
   <https://www.cog-genomics.org/plink/2.0/basic_stats#check_sex>.

9. **PLINK 2.0 docs — `--freq` / `--maf` (sex-chr handling).**
   <https://www.cog-genomics.org/plink/2.0/basic_stats#freq>. MAF denominator
   $2n_F + n_M$ for chrX.

10. **PLINK 2.0 docs — `--hardy` (sex-chr behaviour).**
    <https://www.cog-genomics.org/plink/2.0/basic_stats#hardy>. HWE on
    females-only for chrX, NA for chrY/MT.

11. **REGENIE docs — Step 1 / Step 2 options ("X chromosome" section).**
    <https://rgcgithub.github.io/regenie/options/>. PAR-split requirement and
    Step-2 male-dosage doubling.

12. **BOLT-LMM v2.4 manual.**
    <https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html>.
    Sections "Sex chromosome handling" and "Best practices — model SNP set".

13. **SAIGE Step 2 docs (single-variant tests, X-chromosome flags).**
    <https://saigegit.github.io/SAIGE-doc/docs/single_step2.html>.
    `--is_rewrite_XnonPAR_forMales`, `--MaleOnly`/`--FemaleOnly`,
    `--X_PARregion`.

14. **GCTA docs — Making a GRM (autosomal and X-chromosome).**
    <https://yanglab.westlake.edu.cn/software/gcta/#MakingaGRM>.
    `--make-grm`, `--make-grm-xchr`, `--autosome`, `--autosome-num`.

15. **GCTA docs — MLM association (`--dc` dosage-compensation flag).**
    <https://yanglab.westlake.edu.cn/software/gcta/#MLMassociationanalysis>.

16. **Ensembl Human PAR coordinates.**
    <https://www.ensembl.org/info/genome/genebuild/human_PARS.html>.
    GRCh37/GRCh38 PAR1/PAR2 base-pair endpoints.

17. **NCBI Genome Reference Consortium — Human assembly summary (XTR
    region).** <https://www.ncbi.nlm.nih.gov/grc/human>.

18. **Mueller, J. L. et al. (2008).** *Chromosomal mapping reveals
    deeply conserved human genes on the X-transposed region.* *Nature
    Genetics* 40(7):794–799. <https://doi.org/10.1038/ng.272>. Original
    XTR characterisation.

19. **Yonova-Doing, E. et al. (2021).** *An atlas of mitochondrial DNA
    genotype-phenotype associations in the UK Biobank.* *Nat. Genet.*
    53:982–993. <https://doi.org/10.1038/s41588-021-00868-1>. Methods
    section: heteroplasmy filtering for mtDNA-GWAS.

20. **Hägg, S. et al. (2021).** *Deciphering the genetic and epidemiological
    landscape of mitochondrial DNA abundance.* *Nat. Commun.* 12:6147.
    <https://doi.org/10.1038/s41467-021-26424-3>. Methods: mtDNA scan
    protocol with autosomal-GRM correction and haplogroup PCs as
    covariates.

### UNVERIFIED items collected from earlier sections (re-verify at sub-spec time)

- §3 (XTR coordinates): exact base-pair endpoints differ between Ensembl, UCSC,
  and the GRC summary. Pin to a specific assembly's GFF3 at sub-spec time.
- §4 (BOLT chrY): the v2.4 manual is silent on chrY-specific behaviour;
  re-verify by running BOLT on a chrY-only fixture during sub-spec drafting.
- §5 (REGENIE/BOLT mtDNA): no explicit mtDNA documentation in either tool's
  manual; the spec should pre-decide whether we follow PLINK 2.0 behaviour
  (autosomal-style scan with chrM contig) or refuse mtDNA inputs.
- §9 (REGENIE flag drift): X-chromosome behaviour has changed across REGENIE
  v3.x → v4.x; pin to a specific tagged version when implementing.
