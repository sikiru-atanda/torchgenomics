# PolyFun Research Brief (Phase 59 — FULL PARITY)

**Author:** TorchGWAS modernization — desk research only (no installs).
**Date:** 2026-05-11.
**Scope:** Background research feeding `2026-05-11-phase-59-polyfun-design.md`. Per the modernization master spec, Phase 59 is the only one of the four modernization items at **FULL PARITY** scope, with Tier-2 tolerance `floor + observed × 1.5`.
**Method:** Primary sources only — Weissbrod et al. 2020 *Nature Genetics* paper (PMC mirror), the omerwe/polyfun GitHub repo source files and Wiki, and AWS Open Data registry. No paraphrase from training memory.

---

## 1. Upstream tool overview

| Item | Value | Source |
|---|---|---|
| Repo URL | `https://github.com/omerwe/polyfun` | GitHub repo home [C1] |
| License | MIT | GitHub repo license entry [C1] |
| Latest release / tag | **None published** ("No releases are published according to the repository page"). Development is on `master` only; users pin to `master` HEAD. | GitHub releases tab [C1] |
| Primary paper | Weissbrod O., Hormozdiari F., Benner C., Cui R., Ulirsch J., Gazal S., Schoech A. P., van de Geijn B., Reshef Y., Márquez-Luna C., O'Connor L., Pirinen M., Finucane H. K., Price A. L. (2020). "Functionally informed fine-mapping and polygenic localization of complex trait heritability." *Nature Genetics* 52, 1355–1363. | Nature Genetics [C2]; PMC mirror [C3] |
| DOI | `10.1038/s41588-020-00735-5` | Nature page URL & PMC PMC7710571 [C2, C3] |
| Publication date | November 2020 (online); print Dec 2020 | Nature Genetics record [C2] |
| Bundled tools in same repo | **PolyFun** (functional-prior fine-mapping), **PolyLoc** (polygenic localization), **PolyPred** (cross-ancestry PRS — separate paper, Weissbrod*, Kanai*, Shi* et al. 2022, *Nat Genet*, DOI `10.1038/s41588-022-01036-9`) | README [C1] |

**Top-level Python entry-point scripts (verbatim from repo root):** `polyfun.py`, `finemapper.py`, `polyloc.py`, `polypred.py`, `ldsc.py`, `munge_polyfun_sumstats.py`, `aggregate_finemapper_results.py`, `compute_ldscores.py`, `compute_ldscores_from_ld.py`, `create_finemapper_jobs.py`, `extract_annotations.py`, `extract_snpvar.py`, `polyfun_utils.py`, `test_polyfun.py` [C1].

---

## 2. Algorithm summary (full parity scope)

PolyFun is a **five-step procedure** (the paper itself describes it as five steps; the wiki sometimes condenses to four). The output is a per-SNP prior causal-effect variance ("SNPVAR") that is then injected into SuSiE / FINEMAP in place of the uniform prior. [C3, Methods]

### Dimension table

| Symbol | Meaning | Typical value (UKB analysis) |
|---|---|---|
| $M$ | total SNPs across genome | ~19 million (UKB imputed, MAF > 0.1%) [C5] |
| $A$ | annotations in baseline-LF v2.2 | **187** overlapping annotations (10 common-MAF bins, 10 low-frequency bins, LD-related, binary, continuous) [C3, Methods] |
| $B$ | bins for per-SNP $h^2$ partition | **20** ("Partition all SNPs into 20 bins with similar values of $\widehat{\mathrm{var}}[\beta_i \mid \mathbf{a}_i]$") [C3, Methods] |
| $L$ | SuSiE single-effect layers | up to `--max-num-causal` (paper uses $L=10$) [C3] |
| $K$ | credible sets returned | ≤ $L$ (one per active layer) |
| $n$ | GWAS sample size | $N = 337{,}491$ unrelated British UKB [C3, Results] |

### Step 1 — L2-regularized S-LDSC on even chromosomes
PolyFun fits S-LDSC restricted to **even chromosomes** with an **L2 penalty** to estimate per-annotation coefficients $\hat{\tau}^{\text{even}}_c$:

$$
\min_{\boldsymbol{\tau}, b}\; \sum_i \Bigl(\chi_i^2 - n \sum_{c=1}^{A} \tau_c\, \ell(i,c) - n\, b - 1\Bigr)^2 + \lambda \sum_{c=1}^{A} \tau_c^2
$$

where $\ell(i,c)$ is the per-SNP LD score for annotation $c$, $\chi_i^2$ is the GWAS chi-square at SNP $i$, $b$ is the LDSC intercept term, and $\lambda$ is the L2 penalty. [C3, Methods, eqn. quoted verbatim by author from paper]

### Step 2 — Per-SNP heritability for the held-out (odd) chromosomes
The fitted even-chromosome $\hat{\tau}^{\text{even}}_c$ are used to compute per-SNP heritabilities for SNPs on **odd** chromosomes (and vice versa, by symmetry):

$$
\widehat{\mathrm{var}}[\beta_i \mid \mathbf{a}_i] \;=\; \sum_{c=1}^{A} \hat{\tau}^{\text{even}}_c\, a_{ic}
$$

This is the verbatim formula `var[β_i | a_i] = Σ_c τ_c a_ic` reported in the Methods section [C3]. The cross-chromosome split is the explicit mechanism PolyFun uses to avoid winner's curse: *"PolyFun avoids winner's curse by using different data for partitioning SNPs and for per-bin heritability estimation."* [C3]

### Step 3 — Partition into 20 bins
SNPs are sorted by $\widehat{\mathrm{var}}[\beta_i \mid \mathbf{a}_i]$ and partitioned into $B = 20$ bins of approximately equal heritability via the **Ckmeans.1d.dp** R package (a one-dimensional optimal $k$-medians). When R/Ckmeans is unavailable, PolyFun falls back to scikit-learn 1-D K-means via `--skip-Ckmedian`. [C3, Methods; C6, polyfun.py flag table]

### Step 4 — Re-fit S-LDSC on the 20 bins
S-LDSC is re-fit using the 20 bins as the new annotation set — i.e., one $\tau_b$ per bin — to obtain robust per-bin per-SNP heritability $\hat{\sigma}^2_{r,b}$. This step is non-parametric in $\mathbf{a}_i$: it only uses the bin assignment, not the original 187 annotations. [C3, Methods]

### Step 5 — Prior causal probability
Per-SNP prior causal probability is set proportional to per-SNP heritability:

$$
P(\beta_i \neq 0 \mid \mathbf{a}_i) \;\propto\; \hat{\sigma}^2_{r,b(i)} \quad \text{normalized within each fine-mapping locus to } \sum_{i \in \text{locus}} P(\beta_i \neq 0) = 1.0
$$

The paper notes: *"PolyFun avoids directly estimating the proportionality factor $1/\mathrm{var}[\beta_i \mid \beta_i \neq 0]$ by constraining the prior causal probabilities $P(\beta_i \neq 0 \mid \mathbf{a}_i)$ in each tested locus to sum to 1.0."* [C3]

### Step 6 (in `finemapper.py`) — SuSiE / FINEMAP with per-SNP priors
SuSiE's per-layer single-effect prior is replaced from uniform $1/p$ to the normalized PolyFun prior, and the inference proceeds as in Wang et al. 2020 SuSiE / Benner et al. 2016 FINEMAP. [C7]

### "polyfun-imp" vs "polyfun-ldsc" — terminology check
The terminology "polyfun-imp" and "polyfun-ldsc" used in the prompt does **not appear verbatim** in the Weissbrod 2020 paper. **UNVERIFIED** as named modes — the paper presents one PolyFun pipeline. What the **wiki** distinguishes is three *approaches* to obtain SNPVAR [C5, Wiki §1]:

1. **Pre-computed**: download SNPVAR for the ~19M UKB-imputed SNPs (meta-analysis over 15 traits).
2. **L2-regularized S-LDSC** (`--compute-h2-L2 --no-partitions`): use the eqn-1 / eqn-2 result directly as the prior; this is what the prompt likely calls "polyfun-imp" since it imputes SNPVAR from L2 S-LDSC without binning.
3. **Non-parametric / robust** (`--compute-h2-L2` then `--compute-h2-bins`): the full five-step pipeline with binning. Wiki: *"the most robust approach, but it is computationally intensive."* This is what the prompt likely calls "polyfun-ldsc."

We adopt the wiki's approach naming (`pre-computed` / `L2-only` / `non-parametric / binned`) for the spec. The prompt's `polyfun-imp` / `polyfun-ldsc` labels are flagged for the user to confirm.

---

## 3. Input formats

### Summary statistics (input to `munge_polyfun_sumstats.py`, then to `polyfun.py` / `finemapper.py`)
**Required columns** (from wiki §1, §3, and `finemapper.py` source) [C5, C7, C8]:

| Column | Description |
|---|---|
| `CHR` | chromosome (integer; UKB analysis used hg19) |
| `BP` | base-pair position (hg19 coordinates per the wiki) |
| `SNP` | RSID or chr:pos:a1:a2 |
| `A1` | effect allele |
| `A2` | non-effect allele |
| `Z` | z-score (required for `finemapper.py`) |
| `N` | sample size (or supply via `--n`) |
| `SNPVAR` | per-SNP heritability prior (output of `polyfun.py` Step 5; consumed by `finemapper.py`) |

Munged output is **parquet** (preferred) or `.gz` text. Example header from wiki [C5]:
```
CHR  SNP          BP        A1  A2  SNPVAR      Z            N
22   rs139069276  16866502  G   A   4.5173e-06  -1.2188e-02  383290
```

### Baseline-LF annotation files
**Format:** `.parquet` (recommended) or `.gz` text. **Per-chromosome split** (one file per chromosome 1–22). Wiki §2 [C9].

Required columns: `CHR, BP, SNP, A1, A2`, then one column per annotation (187 in baseline-LF v2.2.UKB). Companion files per chromosome:
- `<prefix>.<chr>.annot.parquet` — annotation matrix
- `<prefix>.<chr>.l2.ldscore.parquet` — partitioned LD scores (one column per annotation)
- `<prefix>.<chr>.l2.M` — single-line whitespace-delimited sums of annotation columns
- `<prefix>.<chr>.l2.M_5_50` — same but restricted to MAF 5%–50% (optional)

### Reference LD format
Three modes (mutually exclusive at the locus level) [C7, C9]:
1. **In-sample LD from PLINK BED** via `--geno <plink_prefix>` (LD computed on the fly).
2. **In-sample LD from BGEN** via `--geno <bgen>` + `--sample-file` (uses LDstore 2.0 `--ldstore2`).
3. **Pre-computed LD from UKB** via `--ld <prefix>` — **NumPy sparse `.npz` format** or **BCOR 1.1** binary; the public S3 dataset is split into 2,763 contiguous ~3 Mb regions covering the genome [C10].

---

## 4. Output formats

### `polyfun.py` SNPVAR output
After `--compute-h2-L2 --no-partitions` (or after `--compute-h2-bins`), per-chromosome parquet with columns [C5]:
```
CHR  BP        SNP                    A1        A2  SNPVAR
1    10000006  rs186077422            G         A   4.0733e-09
1    10000179  1:10000179_AAAAAAAC_A  AAAAAAAC  A   4.0733e-09
```

### `finemapper.py` output (per region)
Tab-separated `.gz` (or `.txt`) with columns [C7, C8]:

| Column | Description |
|---|---|
| `SNP, CHR, BP, A1, A2` | variant ID |
| `SNPVAR` | input prior carried through |
| `Z, N` | input statistics |
| `P` | two-sided p-value |
| `PIP` | posterior inclusion probability |
| `BETA_MEAN, BETA_SD` | posterior effect-size mean and SD |
| `CREDIBLE_SET` | integer index of credible set (0 = not in any CS) |

### `aggregate_finemapper_results.py` output
Concatenated table over all regions and chromosomes; same column set as `finemapper.py` plus optional per-allele rescaling when `--adjust-beta-freq` is set [C8].

### Heritability output (S-LDSC enrichment)
From `ldsc.py --h2 ... --overlap-annot`: standard S-LDSC `.results` file with `Category, Prop._SNPs, Prop._h2, Prop._h2_std_error, Enrichment, Enrichment_std_error, Enrichment_p, Coefficient, Coefficient_std_error, Coefficient_z-score`. [C11]

---

## 5. All flags / options (FULL PARITY surface)

### 5.1 `polyfun.py` — heritability estimation (the "polyfun-ldsc" mode)
Source: argparse in `polyfun.py` head [C6]. **Full surface (24 flags):**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--compute-h2-L2` | flag | False | Step 1: L2-regularized S-LDSC. |
| `--compute-h2-bins` | flag | False | Step 4: re-estimate per-bin h². |
| `--compute-ldscores` | flag | False | Compute LD scores for SNP bins. |
| `--no-partitions` | flag | False | Skip binning (use L2-only SNPVAR directly). |
| `--num-bins` | int | None (BIC-selected) | Number of partition bins (default 20 in paper). |
| `--anno` | str | None | Comma-delimited annotation subset. |
| `--skip-Ckmedian` | flag | False | Use sklearn K-means instead of R Ckmeans.1d.dp. |
| `--chr` | int | None (all) | Target chromosome for LD-score computation. |
| `--ld-wind-cm` | float | None | LD-score window in cM. |
| `--ld-wind-kb` | int | None | LD-score window in kb. |
| `--ld-wind-snps` | int | None | LD-score window in SNPs. |
| `--chunk-size` | int | 50 | Chunk size for LD-score calc. |
| `--keep` | str | None | Subset of individuals for LD-score calc. |
| `--q` | float | 100 | Max ratio largest:smallest truncated per-SNP h². |
| `--sumstats` | str | required | Input sumstats (parquet preferred). |
| `--ref-ld-chr` | str | None | Prefix(es) of LD-score / annotation files (comma-sep allowed per FAQ). |
| `--w-ld-chr` | str | None | Prefix of weights LD-score files. |
| `--bfile-chr` | str | None | PLINK reference panel prefix. |
| `--ld-ukb` | flag | False | Use precomputed UKB LD matrices. |
| `--ld-dir` | str | None | Local cache for downloaded UKB LD. |
| `--output-prefix` | str | required | Prefix for outputs. |
| `--allow-missing` | flag | False | Continue if some SNPs lack annotations. |
| `--num-chr` | int | 22 | Number of chromosomes (organism). |
| `--nnls-exact` | flag | False | Exact (slow) NNLS for non-negative tau. |

### 5.2 `finemapper.py` — the SuSiE / FINEMAP wrapper
Source: argparse at end of `finemapper.py` [C7]. **Full surface (32 flags):**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--method` | str | required | `susie` or `finemap`. |
| `--sumstats` | str | required | Sumstats with `Z` (and `SNPVAR` for functional priors). |
| `--n` | int | required | GWAS sample size. |
| `--chr` | int | required | Target chromosome. |
| `--start` | int | required | Locus start bp. |
| `--end` | int | required | Locus end bp. |
| `--max-num-causal` | int | required | Max causal (FINEMAP) / fixed L (SuSiE). |
| `--out` | str | required | Output filename. |
| `--geno` | str | None | PLINK prefix or `.bgen` (mutually exclusive with `--ld`). |
| `--ld` | str | None | Pre-computed LD prefix (npz / BCOR). |
| `--sample-file` | str | None | SNPTEST2 sample (required with bgen). |
| `--incl-samples` | str | None | Subset IDs file. |
| `--cache-dir` | str | None | Cache for LD matrices. |
| `--ldstore2` | str | None | Path to LDstore 2.0 binary. |
| `--finemap-exe` | str | None | Path to FINEMAP v1.4(.1) binary. |
| `--finemap-dir` | str | None | Working dir for FINEMAP files. |
| `--non-funct` | flag | False | Force uniform prior (ignore SNPVAR). |
| `--allow-missing` | flag | False | Drop SNPs missing from LD. |
| `--allow-swapped-indel-alleles` | flag | False | Keep indels with swapped alleles. |
| `--no-sort-pip` | flag | False | Preserve sumstats order in output. |
| `--memory` | int | 1 | Max GB for LDstore. |
| `--threads` | int | None | LDstore CPU threads. |
| `--debug-dir` | str | None | Dump intermediate state. |
| `--verbose` | flag | False | Verbose logging. |
| `--susie-outfile` | str | None | Save SuSiE R object. |
| `--susie-resvar` | float | None | Fix residual variance. |
| `--susie-resvar-init` | float | None | Initial residual variance. |
| `--susie-resvar-hess` | flag | False | Init residual variance from HESS. |
| `--susie-max-iter` | int | 100 | SuSiE IBSS max iters. |
| `--hess` | flag | False | Use HESS estimate for prior effect-size variance. |
| `--hess-iter` | int | 100 | HESS averaging iterations. |
| `--hess-min-h2` | float | None | HESS h² threshold for SNP inclusion. |

### 5.3 `aggregate_finemapper_results.py`
Source: argparse [C8]. **Full surface (7 flags):**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--sumstats` | str | required | Sumstats file. |
| `--out-prefix` | str | required | Prefix used by per-region jobs. |
| `--out` | str | required | Aggregated output path. |
| `--allow-missing-jobs` | flag | False | Continue when some regions are missing. |
| `--regions-file` | str | DEFAULT_REGIONS_FILE | Region list (default: bundled UKB 2,763-region file). |
| `--chr` | int | None | Restrict to one chromosome. |
| `--pvalue-cutoff` | float | None | Skip regions where no SNP passes p-cutoff. |
| `--adjust-beta-freq` | flag | False | Output β on per-allele scale instead of standardized. |

### 5.4 Auxiliary scripts (also part of full-parity surface)

| Script | Key flags | Source |
|---|---|---|
| `munge_polyfun_sumstats.py` | `--sumstats`, `--out`, `--n`, `--min-info` (0.6), `--min-maf` (0.001), `--remove-strand-ambig`, `--chi2-cutoff` (30), `--keep-hla`, `--no-neff` | [C12] |
| `extract_snpvar.py` | `--sumstats`, `--out`, `--allow-missing`, `--q` (100) | [C13] |
| `compute_ldscores_from_ld.py` | `--annot`, `--out`, `--ukb`, `--n`, `--gz-out`, `--ld-dir`, `--no-cache`, positional `bcor` files | [C14] |
| `extract_annotations.py` | `--pips`, `--annot`, `--out`, `--allow-missing`, `--pip-cutoff` (0.95) | [C15] |
| `create_finemapper_jobs.py` | builds shell jobs for batched per-region runs | wiki §3 [C7] |
| `ldsc.py` (PolyFun's wrapper) | `--h2`, `--ref-ld-chr`, `--w-ld-chr`, `--out`, `--overlap-annot`, `--not-M-5-50`, `--frqfile-chr` | [C11] |

---

## 6. Reference data requirements

| Resource | URL | Size | Source |
|---|---|---|---|
| baseline-LF v2.2.UKB pre-computed SNPVAR for ~19M UKB-imputed SNPs | `https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.polyfun.tar.gz` | **~30 GB** ("WARNING: this is a large download, requiring 30GB" — wiki §2) | [C9] |
| baseline-LF v2.2.UKB raw annotations (no SNPVAR) | `https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.tar.gz` | UNVERIFIED exact size — wiki §2 mentions but does not quote a number; comparable to the .polyfun bundle | [C9, C16] |
| UKB LD matrices (per-3 Mb-region npz/BCOR), 2,763 regions, $N \approx 337{,}000$ British | `s3://broad-alkesgroup-ukbb-ld/UKBB_LD/` (no-sign-request OK) | UNVERIFIED total — AWS Open Data registry page does not state the dataset size; community reports place it at ~1.5–2 TB but I could not confirm from primary sources today | [C16, C17] |
| Bundled `example_data/` for smoke tests | repo `example_data/` directory | small (annotations.{1..22}.parquet, weights.*.l2.*, reference.{1..14}.{bed,bim,fam}, RBC.sumstats.small.parquet, chr1.finemap_sumstats.txt.gz, posterior_betas.{gz,parquet}) | [C18] |

> **Storage pre-flight (per `feedback_preflight`):** Phase 59 sub-spec must assert ≥35 GB free for code + baseline-LF v2.2.UKB.polyfun bundle alone. Full UKB LD matrix download is multi-TB and SHOULD NOT be a CI/test prerequisite; design it as an opt-in user-supplied resource with a small bundled smoke fixture.

---

## 7. Install prerequisites

From `polyfun.yml` conda environment file [C19]:

**Python stack (versions where pinned):**
- Python 3.8
- numpy ≥1.20, scipy, pandas (≥0.25.0 per FAQ [C20])
- pyarrow ≥3.0
- scikit-learn, networkx, pandas-plink, bitarray, tqdm, packaging, pip
- rpy2 with version exclusion: `<3.5.7 or ≥3.5.9`

**R stack (Bioconductor not used; CRAN only):**
- r-base
- **r-susier == 0.11.92** (pinned — this is the SuSiE R package, *not* LDpred2; see "LDpred2 dep" gotcha below)
- r-ckmeans.1d.dp, r-wavethresh, r-lattice, r-stringi, r-matrixstats, r-devtools, r-expm

**pip-installed:**
- bgen == 1.2.10

**External binaries (optional but used in production):**
- **FINEMAP v1.4.1** — pre-compiled tarball, Linux/MacOS [C1]
- **LDstore 2.0** — pre-compiled tarball, Linux/MacOS [C1]

**Note on the prompt's "LDpred2 dep is critical":** PolyFun's Weissbrod 2020 paper and the polyfun.yml environment file do **not** depend on LDpred2 (`bigsnpr`). The PolyPred 2022 follow-up paper does combine PolyFun-derived priors with LDpred2 polygenic scores, but that is a separate codepath in `polypred.py`, not the fine-mapping pipeline. **The polyfun-finemap surface in Phase 59 has no LDpred2 dependency.** Flagging back to the user — this looks like a question carried over from PRS-CSx scoping.

**Estimated total install footprint (code + baseline-LF .polyfun bundle + smoke fixtures, no UKB LD):**
| Component | Size |
|---|---|
| polyfun source + Python deps + R-base + r-susier | ~3 GB |
| baseline-LF v2.2.UKB.polyfun.tar.gz expanded | ~30 GB |
| Bundled `example_data/` | <100 MB |
| **Total minimum** | **~33 GB** |
| **Plus optional UKB LD (full)** | **+~1.5–2 TB (UNVERIFIED)** |

---

## 8. Smoke fixture suggestion

The repo ships a self-contained smoke set in `example_data/` [C18]. The wiki §1 / §3 walkthroughs use it end-to-end; we can mirror that for our V&V harness:

**Three-step end-to-end smoke test:**
1. **Step 1 (S-LDSC + SNPVAR):** `polyfun.py --compute-h2-L2 --no-partitions --output-prefix out/test --sumstats example_data/sumstats.parquet --ref-ld-chr example_data/annotations. --w-ld-chr example_data/weights.` (uses bundled 22-chr annotation parquet + weights — toy chr1 only is fine).
2. **Step 2 (binning):** `polyfun.py --compute-h2-bins --output-prefix out/test --sumstats example_data/sumstats.parquet --w-ld-chr example_data/weights.`
3. **Step 3 (fine-mapping):** `finemapper.py --geno example_data/chr1 --sumstats example_data/chr1.finemap_sumstats.txt.gz --n 383290 --chr 1 --start 46000001 --end 49000001 --method susie --max-num-causal 5 --cache-dir LD_cache --out out/finemap.1.46Mb.gz` [C7].

**Estimated CPU runtime (single 8-core CPU, no GPU):** Step 1 ≈ 2–5 min (22 chrs × small SNP count), Step 2 ≈ 1 min, Step 3 ≈ 30–60 s for the bundled 3 Mb region. **End-to-end ~5–10 minutes**, all-CPU. UNVERIFIED on exact wall time — needs an empirical run during sub-spec drafting, but the bundled fixture is explicitly designed for fast end-to-end demo per the wiki.

---

## 9. Published validation tolerances

| Claim | Reported value | Source |
|---|---|---|
| Power gain — simulated, PolyFun + SuSiE vs SuSiE alone | **">25% more PIP > 0.95 causal SNPs"** in main chr-1 simulations [C3, Results]; the abstract states **">20% more variants with PIP > 0.95"** in pooled simulations [C2 abstract] | [C2, C3] |
| FDR / calibration — simulated | *"No method except CAVIARBF2− and CAVIARBF2 had significantly inflated false discovery rates"*, evaluated at PIP thresholds 0.5 and 0.95 with FDR ≈ 1 − PIP-threshold (conservative) [C3] | [C3] |
| Real-data improvement, 49 UKB traits, $\bar N$ = 318 K | **3,025 PIP>0.95 fine-mapped variant–trait pairs**, **>32% more than SuSiE alone** [C2 abstract; C3 Results] | [C2, C3] |
| Empirical credible-set coverage vs nominal | **UNVERIFIED:** the Weissbrod 2020 paper does not explicitly tabulate empirical coverage at nominal 95% (the calibration claim is FDR-based). | [C3] |
| Replication PIP correlation across cohorts | **UNVERIFIED in original paper**: not reported by Weissbrod 2020 [C3]. The follow-up *Funmap* paper (Wang et al., *Bioinformatics* 2025, building on PolyFun) reports **15.5%–26.2% improvement in replication rate** of PolyFun-style methods over the runner-up in independent-cohort replication [C21]. The PolyPred 2022 paper [C4] reports cross-population PRS R² gains but not PIP correlations per se. | [C4, C21] |
| Sample-size requirement / bias caveat | The paper's analyses use $N = 337{,}491$. PolyFun requires "non-overlapping LD reference panel from the target population spanning ≥10% of target sample size" for the LD-mismatched scenario [C3]. Also: *"PolyFun avoids winner's curse by using different data for partitioning SNPs and for per-bin heritability estimation"* — winner's-curse mitigation built in by chromosome split [C3]. | [C3] |

**Implication for our Tier-2 tolerance (`floor + observed × 1.5`):** PolyFun + SuSiE PIPs vs reference PolyFun + SuSiE PIPs should agree to within `1.5 × observed_max_diff` after both run on the same fixture; SNPVAR values should agree exponentially (per the L2-regularized S-LDSC fit) modulo Ckmeans vs sklearn binning differences (test only with `--skip-Ckmedian` for byte-deterministic comparison).

---

## 10. Known issues / gotchas

From the omerwe/polyfun GitHub issue tracker and wiki FAQ [C1, C20, C22]:

1. **Missing annotations** — wiki FAQ: *"The best solution is to create annotations for all your SNPs. Otherwise you might miss truly causal SNPs simply because they did not have annotations info. However, if you're willing to take this risk you can omit such SNPs by providing the flag `--allow-missing`."* Behavior when annotations are sparse is to silently exclude rather than error, which is a quiet failure mode we must replicate / detect. [C20]
2. **Pandas version** — *"Before reporting an error, please make sure that you have updated versions of all the required packages. In particular, you should have pandas version >=0.25.0."* [C20]
3. **Custom annotations on top of baseline-LF** — *"`--ref-ld-chr` accepts a comma-separated list of file name prefixes, just like standard S-LDSC."* Allows additive layering without reconstructing the whole baseline. [C20]
4. **LD-coordinate mismatches** — Issue #213 (open): *"polyfun.py is not loading all the downloaded LD files to --compute-ldscores"* — unresolved as of latest snapshot. Issue #144: *"The link to the pre-computed LD matrixes for UK Biobank data doesn't work, and it seems that the https://alkesgroup.broadinstitute.org/UKBB_LD is down."* — primary alkesgroup mirror is dead; we must use the AWS S3 mirror exclusively [C16, C22, C23]. **Implication:** all our docs and tests must point at the S3 URL, never the broadinstitute.org one.
5. **Open SuSiE-API churn** — Issue #209 mentions `susie_rss` replacing `susie_suff_stat`. Our wrapper must pin `r-susier == 0.11.92` (per polyfun.yml) or implement directly to avoid drift. [C19, C22]
6. **BGEN input bug** — Issue #187: *"Bgen input with an error caused by an incorrect variable invocation."* Open. Our re-implementation can avoid this by routing through PLINK/BED via existing `torchgwas.io`. [C22]
7. **Sample-size bias / winner's curse** — Built-in mitigation via even/odd chromosome split (see §2 step 1 vs step 2). However, when users supply only a single chromosome's sumstats, the cross-chromosome split breaks down. UNVERIFIED in the paper how PolyFun handles this — likely silent fall-back to single-chromosome S-LDSC.
8. **Strand-ambiguous SNPs** — `munge_polyfun_sumstats.py --remove-strand-ambig` is opt-in; default keeps them. Our preprocessor should default to removing per modern best practice. [C12]
9. **HLA region** — `munge_polyfun_sumstats.py` removes the HLA region by default (`--keep-hla` is opt-in to keep). PolyFun's published UKB analyses also exclude HLA. [C12]
10. **SuSiE convergence in dense LD blocks** — `--susie-max-iter` defaults to 100; our `bayesian_vs.py` SuSiE implementation defaults to 1000. We should preserve the higher default for parity with our own existing tests but expose the flag.

---

## 11. Risk flags for our integration

### R59-1 — `models/bayesian_vs.py` lacks per-SNP prior input ⚠️ **HIGH**
Our existing `BayesianVS.fit()` signature accepts only **scalar** `prior_pi: float = 0.01` and `prior_sig2_beta: float = 0.1` [C24]. PolyFun's per-SNP `SNPVAR` produces a **vector** $\boldsymbol{\pi} \in \mathbb{R}^p$ that replaces uniform $1/p$ in the SuSiE single-effect step. **Required refactor:**

- Promote `prior_pi` to `Union[float, Tensor]`. When tensor of length $p$, normalize to sum to 1.0 *per locus* (per Step 5 above).
- The SuSiE `_susie_loop` softmax line `# Softmax for inclusion (uniform prior 1/p)` (bayesian_vs.py:786) becomes `softmax(log_bf + log_prior)`.
- The CAVI path's `log_odds_prior = math.log(pi / (1.0 - pi))` (line 443) becomes `log_odds_prior_j = log(pi_j / (1 - pi_j))` per SNP.
- Backward compatibility: scalar input remains a special case (broadcast to vector).
- New unit test: per-SNP prior with one extreme prior should drive PIP toward the prior's argmax even on noisy z-scores — verifies the prior is actually wired in.

### R59-2 — `postgwas/_sldsc.py` is OLS, not L2-regularized ⚠️ **HIGH**
Our `sldsc_h2_partitioned()` uses `_weighted_lstsq` (ordinary weighted least squares with two-step outlier filter and block jackknife) [C25]. PolyFun's Step 1 uses **L2-regularized S-LDSC**:

$$
\min_{\boldsymbol{\tau}, b}\; \sum_i \Bigl(\chi_i^2 - n \sum_c \tau_c \ell(i,c) - n b - 1\Bigr)^2 + \lambda \sum_c \tau_c^2
$$

The penalty is essential — paper: *"To address the first limitation, PolyFun incorporates an L2-regularized extension of S-LDSC."* [C3] Without L2, $\hat\tau$ for the 187-annotation baseline-LF is unstable.

**Required refactor:**
- Add `lambda_l2: float = 0.0` parameter to `sldsc_h2_partitioned`. The paper does not give an explicit $\lambda$ value — UNVERIFIED, must read source `polyfun.py` `_compute_h2_L2` to recover it (deferred to sub-spec drafting where we install the package).
- Optional `nnls: bool = False` flag mirroring `--nnls-exact`.
- Per-SNP h² accessor: a method `predict_per_snp_h2(annot_matrix)` that returns $\hat\sigma^2_i = \sum_c \hat\tau_c a_{ic}$ — currently absent.

### R59-3 — `postgwas/_finemapping.py` does not own SuSiE/FINEMAP execution ⚠️ **MEDIUM**
Our `_finemapping.py` exposes `extract_credible_sets`, `annotate_sumstats`, `locus_summary`, `to_coloc_sumstats` — purely **post-processing** utilities operating on the output of `BayesianVS`. PolyFun's `finemapper.py` is the **driver** that orchestrates LD construction → prior loading → SuSiE/FINEMAP fit → credible-set extraction.

**Resolution:** add a new module `torchgwas/postgwas/_polyfun_finemap.py` (or extend `_finemapping.py` with a `polyfun_finemap_locus(...)` driver function). It should:
- Accept a region (chr, start, end), a sumstats DataFrame with `SNPVAR`, and an LD source (path to BED, on-the-fly LD, or pre-computed npz).
- Build $\mathbf R$ (LD matrix) for the region via `torchgwas.io` + `torchgwas.linalg`.
- Call `BayesianVS.fit(...)` with the per-SNP `pi` derived from `SNPVAR`.
- Reuse `extract_credible_sets()` (already exists).

### R59-4 — `annotate/` is wrong home for baseline-LF ingestion ⚠️ **MEDIUM**
`torchgwas/annotate/` is exclusively NCBI gene-annotation (Phase 48): `_client.py`, `_genes.py`, `_resolve.py`, `_types.py` — REST-API-driven, low-volume (kilobytes per query). Baseline-LF v2.2 is a **dense $19{,}000{,}000 \times 187$ float matrix** (~30 GB on disk, ~28 GB in float32 if fully materialized). This belongs in **a new module** — recommend `torchgwas/postgwas/_baseline_lf.py` or `torchgwas/io/_annot_parquet.py`. Co-locate with S-LDSC since that's the only consumer.

**Streaming requirement (per `feedback_streaming`):** the baseline-LF annotation matrix MUST be loaded as a **chunked parquet stream** (per chromosome, per row-group), not materialized whole. PyArrow's `ParquetFile.iter_batches()` is the right primitive. Memory-regression test must verify peak RSS bounded by chunk size, not 19M-row size.

### R59-5 — Genotype streaming claim is moot ✅ **CONFIRMED**
The prompt's note *"Streams genotype chunks — but PolyFun is summary-stats-based, so no genotype streaming"* is correct: `finemapper.py` either consumes pre-computed LD npz (no genotype touched) or uses PLINK/BGEN solely to build the local LD matrix (one region's worth of variants, ~10 K rows, fits in RAM). **No `iter_chunks` integration needed at the fine-mapping driver level.** The streaming concern relocates to (a) baseline-LF loading (R59-4) and (b) the S-LDSC genome-wide chi² regression (already chunked in our `_sldsc.py`).

### R59-6 — Reference-data download is multi-TB, but not all is required ⚠️ **MEDIUM**
The full UKB LD matrix dump is community-reported at ~1.5–2 TB (UNVERIFIED). It is **not** required for non-UKB users, who can use in-sample LD via `--geno`. Our spec must:
- Default to in-sample LD (no UKB LD download) for the main test path.
- Provide a `torchgwas polyfun-download-ld --regions chr22:46Mb-49Mb` opt-in CLI for users who specifically want UKB LD for a region.
- Pre-flight gate (per `feedback_preflight`): refuse to proceed if free disk < download size + 20% headroom.

### R59-7 — Tier-2 tolerance bite zone ⚠️ **MEDIUM**
With the FULL PARITY tolerance `floor + observed × 1.5`, our golden-file tests will compare:
- SNPVAR values per SNP (Step 5 output) — should agree to ~5 sig figs because L2 closed-form is deterministic given $\lambda$ and $\hat\tau$.
- PIP per SNP (Step 6 output) — softmax over BFs is sensitive; expect ~1e-2 absolute agreement after 1000 IBSS iters.
- Credible-set membership — likely identical given identical PIP and CS-purity threshold.

**Risk:** Ckmeans.1d.dp vs sklearn K-means produce different bin assignments for borderline SNPs, which propagate to per-bin h² and thence to SNPVAR. Mitigation: in our reference-equivalence test, force `--skip-Ckmedian` and use sklearn's K-means with a fixed `random_state`.

---

## 12. Citations

| ID | Citation | URL / DOI | Section / page reference |
|---|---|---|---|
| C1 | omerwe/polyfun GitHub repository (root README + license + script list) | https://github.com/omerwe/polyfun | repo home, README §Installation, MIT LICENSE |
| C2 | Weissbrod O. *et al.* (2020). "Functionally informed fine-mapping and polygenic localization of complex trait heritability." *Nature Genetics* 52, 1355–1363. | https://www.nature.com/articles/s41588-020-00735-5 — DOI 10.1038/s41588-020-00735-5 | Abstract, main text |
| C3 | Same paper, PMC mirror (open access full text) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7710571/ | Methods §"Computing per-SNP heritabilities", Results, Figs 2–3 |
| C4 | Weissbrod*, Kanai*, Shi* *et al.* (2022). "Leveraging fine-mapping and multipopulation training data to improve cross-population polygenic risk scores." *Nature Genetics* 54, 450–458. (PolyPred) | https://www.nature.com/articles/s41588-022-01036-9 — DOI 10.1038/s41588-022-01036-9 | Whole paper |
| C5 | PolyFun Wiki §1 — "Computing prior causal probabilities with PolyFun" | https://github.com/omerwe/polyfun/wiki/1.-Computing-prior-causal-probabilities-with-PolyFun | Approach descriptions, polyfun.py examples, output column table |
| C6 | `polyfun.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/polyfun.py | argparse block |
| C7 | PolyFun Wiki §3 — "Functionally informed fine mapping with finemapper" | https://github.com/omerwe/polyfun/wiki/3.-Functionally-informed-fine-mapping-with-finemapper | Examples, flag table |
| C8 | `aggregate_finemapper_results.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/aggregate_finemapper_results.py | argparse block |
| C9 | PolyFun Wiki §2 — "Using and creating functional annotations" | https://github.com/omerwe/polyfun/wiki/2.-Using-and-creating-functional-annotations | baseline-LF download URL & 30 GB warning |
| C10 | UK Biobank Linkage Disequilibrium Matrices on AWS Open Data | https://registry.opendata.aws/ukbb-ld/ | S3 ARN, `--no-sign-request` access |
| C11 | PolyFun Wiki §5 — "Estimating functional enrichment using S-LDSC" | https://github.com/omerwe/polyfun/wiki/5.-Estimating-functional-enrichment-using-S-LDSC | ldsc.py wrapper flags |
| C12 | `munge_polyfun_sumstats.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/munge_polyfun_sumstats.py | argparse block |
| C13 | `extract_snpvar.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/extract_snpvar.py | argparse block |
| C14 | `compute_ldscores_from_ld.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/compute_ldscores_from_ld.py | argparse block |
| C15 | `extract_annotations.py` source (argparse) | https://github.com/omerwe/polyfun/blob/master/extract_annotations.py | argparse block |
| C16 | omerwe/polyfun Issue #144 — "Pre-computed LD matrix link" | https://github.com/omerwe/polyfun/issues/144 | broadinstitute.org URL is dead; use S3 mirror |
| C17 | PolyFun Wiki home | https://github.com/omerwe/polyfun/wiki | overview / typical workflow |
| C18 | omerwe/polyfun `example_data/` directory listing | https://github.com/omerwe/polyfun/tree/master/example_data | bundled smoke fixtures |
| C19 | `polyfun.yml` conda environment | https://github.com/omerwe/polyfun/blob/master/polyfun.yml | exact version pins (Python 3.8, r-susier 0.11.92, pyarrow ≥3.0, etc.) |
| C20 | PolyFun Wiki §7 — FAQ | https://github.com/omerwe/polyfun/wiki/7.-FAQ | `--allow-missing`, pandas version, comma-sep `--ref-ld-chr` |
| C21 | Wang *et al.* (2025). "Funmap: integrating high-dimensional functional annotations to improve fine-mapping." *Bioinformatics* 41, btaf017. (Replication-rate follow-up) | https://academic.oup.com/bioinformatics/article/41/1/btaf017/7952015 | Replication improvement 15.5%–26.2% |
| C22 | omerwe/polyfun GitHub Issues page | https://github.com/omerwe/polyfun/issues | Issues #213, #209, #187, #144 |
| C23 | omerwe/polyfun Issue #17 — LD directly from alkesgroup server | https://github.com/omerwe/polyfun/issues/17 | Confirms migration to S3 |
| C24 | TorchGWAS `models/bayesian_vs.py` (in-tree, lines 115–224 — `BayesianVS.fit` signature with scalar `prior_pi`) | repo file `torchgwas/models/bayesian_vs.py` | Confirms current API has no per-SNP prior |
| C25 | TorchGWAS `postgwas/_sldsc.py` (in-tree, `sldsc_h2_partitioned` signature) | repo file `torchgwas/postgwas/_sldsc.py` | Confirms current S-LDSC is OLS not L2 |

---

## Summary of `UNVERIFIED:` markers

1. **§2** — "polyfun-imp" / "polyfun-ldsc" mode names from the prompt do not appear in the paper. We adopted the wiki's three-approach naming. **Risk:** if the user has an external doc using these names, we need their definition.
2. **§6** — Exact size of `baselineLF_v2.2.UKB.tar.gz` (raw, non-polyfun bundle): wiki §2 doesn't quote a number.
3. **§6** — Total UKB LD matrix dump size: AWS Open Data registry doesn't quote it; community estimates ~1.5–2 TB but unconfirmed from a primary source.
4. **§8** — Empirical CPU runtime for the bundled smoke test: needs real run during sub-spec drafting.
5. **§9** — Empirical credible-set coverage at nominal 95%: not tabulated in Weissbrod 2020 (calibration claim is FDR-based).
6. **§9** — Replication PIP correlation across cohorts: not in Weissbrod 2020; closest available is Funmap (Wang 2025) replication-rate metric.
7. **§10 #7** — Behavior of PolyFun on single-chromosome sumstats (winner's-curse split breaks): not documented in paper.
8. **R59-2** — Exact $\lambda$ for L2-regularized S-LDSC: paper gives the equation but not the chosen value; need to read `polyfun.py` source during sub-spec drafting.
