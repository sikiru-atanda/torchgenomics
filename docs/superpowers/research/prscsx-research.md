# PRS-CSx Research Brief (MVP scope)

- **Status:** desk research, no installs performed
- **Audience:** authors of `docs/superpowers/specs/2026-05-11-phase-58-prscsx-design.md`
- **Scope contract (from master spec §2):** PRS-CSx MVP = the multi-population coupled-shrinkage Gibbs sampler only. No `--meta` posterior averaging, no `--write_pst`, no φ auto-learn, no PRS-CSx-mult / PRS-CSx-meta variants beyond the core sampler. The validation workflow that linearly combines per-population scores in a hold-out set is treated as a downstream user concern, not part of the MVP code surface.

---

## 1. Upstream tool overview

- **Repo:** `getian107/PRScsx` — `https://github.com/getian107/PRScsx`. Plain-Python script (`PRScsx.py`) plus support modules (`mcmc_gtb.py`, `parse_genet.py`, `gigrnd.py`); no setup.py, no packaging — git-clone-and-run.
- **Latest tagged release:** v1.1.0, 2023-08-11 — `https://github.com/getian107/PRScsx/releases/tag/v1.1.0`. The README's own version-history block lists later in-place updates (the most recent being 2024-11-21 for `--write_pst` under `--meta=True`, and 2024-05-14 for replacing scipy calls with numpy), with no further tagged release as of this brief — i.e., the master branch is ahead of the most recent tag.
- **Primary paper:** Ruan, Y., Lin, Y.-F., Feng, Y.-C.A., et al. "Improving polygenic prediction in ancestrally diverse populations." *Nature Genetics* 54:573–580 (2022). DOI **10.1038/s41588-022-01054-7** — `https://www.nature.com/articles/s41588-022-01054-7`. Open-access mirror: PMC9117455 — `https://pmc.ncbi.nlm.nih.gov/articles/PMC9117455/`.
- **License:** MIT, "Copyright (c) 2019 Tian Ge" — `https://github.com/getian107/PRScsx/blob/master/LICENSE`. Compatible with TorchGWAS (no copyleft obligation; we may port equations and re-implement freely).
- **Maintainer contact:** Tian Ge `tge1@mgh.harvard.edu` (per README "Support" section).

---

## 2. Algorithm summary

### 2.1 Single-population PRS-CS (for contrast with PRS-CSx)

PRS-CS (Ge et al. 2019, *Nature Communications* 10:1776 — `https://doi.org/10.1038/s41467-019-09718-0`) places a Strawderman–Berger continuous-shrinkage prior on each per-variant standardized effect:

$$
\beta_j \mid \psi_j \sim \mathcal{N}\!\left(0,\, \tfrac{\sigma^2}{N}\,\psi_j\right), \qquad
\psi_j \mid \delta_j \sim \mathrm{Gamma}(a, \delta_j), \qquad
\delta_j \mid \phi \sim \mathrm{Gamma}(b, \phi).
$$

Defaults: $a = 1$, $b = 1/2$ (Strawderman–Berger), $\phi$ either fixed by the user or learned via a Gamma(1, 1) hyperprior (the "auto" mode). This is what `torchgwas/pgs/prscs.py` implements verbatim (lines 6–13 of the docstring confirm this prior hierarchy).

### 2.2 Multi-population PRS-CSx coupling

Ruan et al. 2022 Methods, "PRS-CSx" subsection (paraphrased from PMC9117455): for population $k \in \{1, \dots, K\}$ with sample size $N_k$ and standardized marginal effect $\hat\beta_{jk}$,

$$
\beta_{jk} \mid \psi_j \sim \mathcal{N}\!\left(0,\, \tfrac{\sigma_k^2}{N_k}\,\psi_j\right), \qquad
\psi_j \mid \delta_j \sim \mathrm{Gamma}(a, \delta_j), \qquad
\delta_j \mid \phi \sim \mathrm{Gamma}(b, \phi).
$$

The crucial coupling — quoting Methods verbatim — is "**when SNP $j$ is available in multiple GWAS summary statistics, the continuous shrinkage prior is shared across populations (i.e., both $\phi$ and $\Psi_j$ do not depend on $k$)**." So $\beta_{jk}$ is population-specific, but $\psi_j$ and $\delta_j$ borrow strength across all populations in which variant $j$ is observed.

The posterior block update for population $k$, conditional on $\Psi = \mathrm{diag}(\psi_1, \dots, \psi_M)$, reduces to the population's own ridge-regularized normal equations:

$$
\mathbb{E}[\beta_k \mid \hat\beta_k, \Psi] = (D_k + \Psi^{-1})^{-1}\,\hat\beta_k,
$$

where $D_k$ is the per-population block-diagonal LD correlation matrix (population-specific).

### 2.3 Gibbs sampler (per iteration; from `mcmc_gtb.py`)

Using the variable names in `mcmc_gtb.py`:

1. **For each population $k$:** block update $\beta_k$ via Cholesky of $\bigl(D_k + \mathrm{diag}(1/\psi)\bigr)$ — exactly the per-population precision matrix that the existing `_prscs_gibbs_block` builds for the single-population case in `torchgwas/pgs/prscs.py:182–195`. Then update $\sigma_k^2 \sim \mathrm{Gamma}\bigl((N_k + p_k)/2,\,1/\mathrm{err}_k\bigr)$ (one residual variance per population).
2. **Shared $\delta$ update:** $\delta_j \sim \mathrm{Gamma}(a + b,\, 1/(\psi_j + \phi))$ — a single $\delta$ vector across all populations.
3. **Shared $\psi$ update with cross-population borrowing:**
   $$\psi_j \sim \mathrm{GIG}\!\left(\lambda = a - \tfrac{1}{2} n_{\text{grp},j},\ \chi = 2\delta_j,\ \psi_{\text{GIG}} = x_j\right),$$
   where $n_{\text{grp},j}$ is the **count of populations in which variant $j$ appears** (line: `n_grp[jj]` in `mcmc_gtb.py`) and $x_j = \sum_{k:\, j \in k} \beta_{jk}^2 / \sigma_k^2$. This is the precise mechanism by which evidence is pooled across ancestries.
4. **Optional $\phi$ update (auto mode):** $\phi \sim \mathrm{Gamma}\bigl(M b + 1/2,\,1/(\sum_j \delta_j + w)\bigr)$ with hyperprior $w \sim \mathrm{Gamma}(1, 1)$.

After burn-in, the post-burn-in $\beta_{jk}$ samples are averaged (with thinning factor $T_{\text{thin}} = 5$) to produce the per-population posterior mean effect sizes $\bar\beta_{jk}$.

### 2.4 Dimension table

| Symbol | Meaning | Typical value (example) |
|---|---|---|
| $K$ | number of discovery populations | 2–5 (paper screens up to all 5) |
| $M$ | total union of QC'd HapMap3 variants across the $K$ populations | ~1.0–1.1 M (HapMap3) |
| $M_k$ | variants observed in population $k$ | $\le M$; varies by GWAS coverage |
| $N_k$ | per-population GWAS sample size | $10^4$–$10^6$ |
| $T$ | total MCMC iterations | $1{,}000 \times K$ (default) |
| $B$ | burn-in iterations | $500 \times K$ (default) |
| $T_{\text{thin}}$ | thinning factor | 5 (default) |
| $a, b$ | Strawderman–Berger prior shape | 1, 0.5 |
| $\phi$ | global shrinkage; either fixed (e.g. $10^{-2}$) or learned | grid: $\{10^{-6}, 10^{-4}, 10^{-2}, 1\}$ |

Sources: README "Optional Flags" section; Ruan et al. 2022 Methods §"PRS-CSx".

---

## 3. Input formats

### 3.1 Per-population summary statistics (from README, "Input Formats")

Whitespace- or tab-separated; one of two header schemas:

**BETA/OR + SE (recommended as of 2023-08-10 update):**

```
SNP          A1   A2   BETA      SE
rs4970383    C    A    -0.0064   0.0090
rs4475691    C    T    -0.0145   0.0094
```

**BETA/OR + P (legacy; p-values < 1e-323 are truncated, per README's 2023-08-10 note):**

```
SNP          A1   A2   BETA      P
rs4970383    C    A    -0.0064   0.4778
```

The README also accepts `OR` in place of `BETA` (in which case `SE` is the SE of $\log\mathrm{OR}$). `parse_sumstats()` in `parse_genet.py` standardizes effects internally:

- with SE: $\tilde\beta = \beta / (\mathrm{SE}\sqrt{N})$
- with P: $\tilde\beta = \mathrm{sign}(\beta)\,|\Phi^{-1}(P/2)| / \sqrt{N}$

The README explicitly notes that under the +P schema "BETA/OR is only used to determine the direction of an association," so $\pm 1$ direction tokens are accepted there.

### 3.2 LD reference panel format (per `parse_genet.parse_ldblk`)

Per-population HDF5 (`.hdf5`) files inside one of:
- `ldblk_1kg_{afr|amr|eas|eur|sas}/` (1000 Genomes phase 3 panels), or
- `ldblk_ukbb_{afr|amr|eas|eur|sas}/` (UK Biobank panels).

Each chromosome is one HDF5 file with key structure `'blk_N/ldblk'` (the LD correlation matrix for block N, dense float) and `'blk_N/snplist'` (variant IDs for block N). `parse_ldblk()` subsets each block to the variants present in the per-population sumstats, applies allele-flip sign corrections, and symmetrizes via SVD (this is the 2021-04-06 change "added LD matrix projection to nearest non-negative definite matrix").

### 3.3 Cross-population variant intersection

`align_ldblk()` keeps the **union** of variants that appear in `ref_dict` (the master HapMap3 panel SNP list) AND in **at least one** per-population sumstats file. Variants missing from a given population's sumstats are simply absent from that population's posterior update — the sampler tracks per-variant population-membership counts via `n_grp[jj]` (see §2.3 step 3).

### 3.4 Master SNP info file

A single `snpinfo_mult_1kg_hm3` (or `snpinfo_mult_ukbb_hm3`) ASCII file with columns: `CHR SNP BP A1 A2 FRQ_AFR FRQ_AMR FRQ_EAS FRQ_EUR FRQ_SAS FLP_AFR FLP_AMR FLP_EAS FLP_EUR FLP_SAS` (15 columns, indices 0–14 per `parse_ref()`). The FLP columns are per-population reference-panel allele-flip flags.

---

## 4. Output format

PRS-CSx writes one file per (population, chromosome) under `<out_dir>/<out_name>_{POP}_pst_eff_a{a}_b{b}_phi{phi}_chr{C}.txt` (filename pattern reconstructed from the `--out_name` documentation; the README states "PRS-CSx writes posterior SNP effect size estimates for each chromosome to the user-specified directory").

**Schema** (per README "Output Format" section):

```
<chr>  <rsid>  <bp>  <A1>  <A2>  <posterior_effect_size>
```

No header row. Whitespace-separated. Example:

```
22  rs5746647  16051249  C  G  -1.234e-05
22  rs62224609 16059471  T  C   3.456e-05
```

Downstream scoring is delegated to PLINK 1.9 `--score ... sum`, with chromosome files concatenated and a per-population score produced for each $(K, \phi)$ combination; the user is expected to learn a linear blend across populations on a held-out validation set (the "Approach 1" recommended in the README, "Recommended Analysis Approaches"). With `--meta=True` PRS-CSx additionally writes an inverse-variance-weighted across-population posterior file — **out of MVP scope** per master spec §2.

---

## 5. Key flags / options for the MVP scope

The 80% MVP surface, per the README "Required Flags" and "Optional Flags" sections:

**Required:**
- `--ref_dir=PATH` — directory containing the SNP info file + per-population HDF5 panels.
- `--bim_prefix=PATH` — prefix of the target dataset PLINK `.bim` (used purely as a variant filter; no genotypes are read).
- `--sst_file=A.tsv,B.tsv,...` — comma-separated per-population sumstats; **order is load-bearing** (must align with `--n_gwas` and `--pop`).
- `--n_gwas=NA,NB,...` — comma-separated per-population sample sizes; order matches `--sst_file`.
- `--pop=EUR,EAS,...` — comma-separated population codes; order matches `--sst_file`. Allowed values: AFR, AMR, EAS, EUR, SAS.
- `--out_dir=PATH`, `--out_name=PREFIX`.

**Optional but in MVP:**
- `--phi=FLOAT` — fix the global shrinkage. README guidance: "fixing phi to 1e-2 (for highly polygenic traits) or 1e-4 (for less polygenic traits), or doing a small-scale grid search (e.g., phi=1e-6, 1e-4, 1e-2, 1)". MVP must support a fixed-φ run; auto-learn is post-MVP per master spec.
- `--n_iter=INT` (default `1000 * K`), `--n_burnin=INT` (default `500 * K`) — README requires both to be specified together to override defaults.
- `--thin=INT` (default 5).
- `--chrom=1,3,...` (default: 1–22). README: "Parallel computation for the 22 autosomes is recommended."
- `--seed=INT`.
- `--a=FLOAT` (default 1), `--b=FLOAT` (default 0.5).

**Out of MVP** (explicit, by master spec): `--meta`, `--write_pst`, φ auto-learn (the no-`--phi` code path), and the validation-set linear-combination workflow.

---

## 6. Reference data requirements

Per the README "LD Reference Panels & Downloads" tables. All hosted on Dropbox (with a Broad mirror at `https://personal.broadinstitute.org/hhuang//public//PRS-CSx/Reference`). Sizes are the compressed `.tar.gz`:

**1000 Genomes phase 3 panels**

| Pop | Size | URL |
|---|---|---|
| AFR | ~4.44 GB | https://www.dropbox.com/s/mq94h1q9uuhun1h/ldblk_1kg_afr.tar.gz?dl=0 |
| AMR | ~3.84 GB | https://www.dropbox.com/s/uv5ydr4uv528lca/ldblk_1kg_amr.tar.gz?dl=0 |
| EAS | ~4.33 GB | https://www.dropbox.com/s/7ek4lwwf2b7f749/ldblk_1kg_eas.tar.gz?dl=0 |
| EUR | ~4.56 GB | https://www.dropbox.com/s/mt6var0z96vb6fv/ldblk_1kg_eur.tar.gz?dl=0 |
| SAS | ~5.60 GB | https://www.dropbox.com/s/hsm0qwgyixswdcv/ldblk_1kg_sas.tar.gz?dl=0 |

**UK Biobank panels**

| Pop | Size | URL |
|---|---|---|
| AFR | ~4.93 GB | https://www.dropbox.com/s/dtccsidwlb6pbtv/ldblk_ukbb_afr.tar.gz?dl=0 |
| AMR | ~4.10 GB | https://www.dropbox.com/s/y7ruj364buprkl6/ldblk_ukbb_amr.tar.gz?dl=0 |
| EAS | ~5.80 GB | https://www.dropbox.com/s/fz0y3tb9kayw8oq/ldblk_ukbb_eas.tar.gz?dl=0 |
| EUR | ~6.25 GB | https://www.dropbox.com/s/t9opx2ty6ucrpib/ldblk_ukbb_eur.tar.gz?dl=0 |
| SAS | ~7.37 GB | https://www.dropbox.com/s/nto6gdajq8qfhh0/ldblk_ukbb_sas.tar.gz?dl=0 |

**Master SNP info files** (one per panel family; both can coexist):
- 1000G HapMap3: `snpinfo_mult_1kg_hm3` ~106 MB — https://www.dropbox.com/s/rhi806sstvppzzz/snpinfo_mult_1kg_hm3?dl=0
- UKBB HapMap3: `snpinfo_mult_ukbb_hm3` ~108 MB — https://www.dropbox.com/s/oyn5trwtuei27qj/snpinfo_mult_ukbb_hm3?dl=0

For a 2-ancestry MVP install (e.g., EUR + EAS, 1000G), uncompressed footprint ~20–25 GB after extraction (compressed 8.9 GB; HDF5 expands ~2.5×). For all 5 ancestries × both panel families, plan ~110 GB. README explicitly notes UKBB and 1000G panels share file format and can be co-mounted.

---

## 7. Install prerequisites

From README "Dependencies" + observed source:

- Python (no version pinned in README; the May-2024 changelog entry "Replaced scipy functions with numpy" suggests upstream is targeting Python 3.x with current scipy — UNVERIFIED: minimum Python version is not stated by upstream).
- `scipy` (used in `gigrnd.py` for the GIG sampler and previously in `mcmc_gtb.py`; partly migrated to numpy in 2024-05).
- `h5py` (LD block panels are HDF5).
- `numpy` (transitively, via scipy).

Implicit runtime expectation: PLINK 1.9 for downstream scoring (README "Output Format"). Not needed for our integration since we use `torchgwas/pgs/scoring.py`.

**Estimated total install footprint** (code + LD panels for 2 ancestries, 1000G family):
- Code: ~50 KB (single-script clone).
- Compressed panels (EUR + EAS): ~8.9 GB.
- Uncompressed panels: ~22 GB.
- Plus the master `snpinfo_mult_1kg_hm3`: ~106 MB.

So **~9 GB download, ~22 GB on disk** for a minimum-viable 2-ancestry install. Pre-flight gate per `feedback_preflight` should require ≥ 30 GB free for the canonical Tier-2 fixture path.

**Threading caveat (README "Computational Efficiency"):** PRS-CSx's numpy implicitly uses all cores via MKL, which "may interfere with other jobs." The recommended mitigation is `MKL_NUM_THREADS / NUMEXPR_NUM_THREADS / OMP_NUM_THREADS` env-var pinning. Our wrapper / parity harness must set these explicitly when shelling out to upstream — otherwise reproducibility timing comparisons are noisy.

---

## 8. Smoke fixture suggestion

Upstream provides a built-in test fixture (README "Test Data"): EUR + EAS sumstats and a `.bim` for **1,000 SNPs on chromosome 22**, runnable as

```
python PRScsx.py --ref_dir=$REF --bim_prefix=$BIM/test \
  --sst_file=$SS/EUR_sumstats.txt,$SS/EAS_sumstats.txt \
  --n_gwas=200000,100000 --pop=EUR,EAS --chrom=22 --phi=1e-2 \
  --out_dir=$OUT --out_name=test
```

with the README quoting **"approximately 1 min when using 8 GB of RAM."** This is the right Tier-2 fixture for our parity harness — it is the one upstream documents and tests against, and it sits comfortably under the 5-minute Tier-2 budget set by master spec §5 ("sized so each smoke run completes in < 5 min on CPU"). LD panels needed for this fixture: only `ldblk_1kg_eur` + `ldblk_1kg_eas` chr 22 slices (≈ 200–400 MB each, *not* the full 4–5 GB tarballs, if we extract per-chromosome).

---

## 9. Published validation tolerances

From Ruan et al. 2022 (PMC9117455 / DOI 10.1038/s41588-022-01054-7):

- **(a) PRS-CSx vs single-population PRS-CS** (UK Biobank target, Figure 3a):
  - **EAS target, vs PRS-CS trained on EAS GWAS:** "median relative increase in R²: 52.3%."
  - **EAS target, vs PRS-CS-mult** (the meta-analysis baseline that runs PRS-CS on a fixed-effect meta of all populations): "median improvement of 10.5% (P_wilcoxon = 3.90e-4)."
  - **Taiwan Biobank target** (Figure 3b): vs single-pop PRS-CS, "median improvement of 39.5%"; vs PRS-CS-mult, "8.2%."
- **(b) PRS-CSx vs other multi-ancestry methods:**
  - **Schizophrenia, EAS cohorts, Figure 4a:** vs LDpred2 trained on EAS GWAS, "relative increase of 45.4% (from 0.043 to 0.063 R²)" on the liability scale; vs PT-meta (P+T on the meta-analyzed sumstats), "relative increase of 135.9%."
  - **N/A:** the paper does not include head-to-head comparisons against XPASS or PolyPred in the sections we extracted from PMC9117455 — those comparisons are made in the **Kachuri et al. 2024 review** (*Nat Rev Genet* 25:8–25, the README's recommended further reading) but not in Ruan 2022 itself.

These numbers are not "tolerances" in the parity-test sense — they are *expected effect-size improvements* over baselines. Our Tier-2 parity test is against PRS-CSx itself (i.e., do we reproduce upstream's per-variant posterior effects to within `floor + 2 × observed`, per master spec §5 MVP rule), not against the absolute R² targets in the paper.

---

## 10. Known issues / gotchas (from `getian107/PRScsx` GitHub Issues)

The visible issue tracker is small (~12 issues displayed), so the following is the full population, not a sample:

- **Issue #75 (open, 2026-01-10) — "Mismatch: PRScsx.py logic does not support multi-population input as described in README":** alleges that the `main()` function "rigidly looks for a single reference file (e.g., snpinfo_mult_1kg_hm3) and lacks the necessary loop (for pp in range(n_pop)) to handle multiple populations." No maintainer response yet. **UNVERIFIED:** we could not corroborate this from the source we read — `parse_genet.parse_ref()` does take a `chrom` filter but appears single-ref-file by design, with multi-population support handled inside `parse_sumstats / parse_ldblk / mcmc_gtb` loops over `range(n_pop)`. We recommend treating this issue as an unconfirmed user complaint, not a known bug, until the maintainer triages it. **Risk for us:** if the complaint *is* valid, our parity harness against upstream would inherit the bug. Mitigation: re-derive multi-pop behavior from the paper, not from upstream behavior.
- **Issue #76 (open, 2026-01-10) — "Validation set for tuning and weight estimation":** statistical question about reusing one validation set for both φ selection AND linear-combination weight fitting. No response. **Relevance to MVP: none** — we explicitly exclude the linear-combination workflow from MVP.
- **Issue #73 (closed, 2026-01-08) — "PRS-CSx ancestry selection in admixed samples":** asks how to handle Brazilian admixed cohorts (~65% EUR / 25% AFR / 10% AMR). No maintainer guidance captured. **Relevance:** our docs should warn that PRS-CSx assumes target individuals are mappable to one of the 5 super-populations; admixed targets are an open methodological question.
- **Issue #67 (closed) — "Small non-EUR GWAS sample sizes":** asks for a minimum-N rule of thumb. No maintainer response captured. **Relevance:** sample-size imbalance is an open user question; no upstream guidance to inherit.
- **Issue #68 (closed) — "PRSCSx taking so long":** user reported the sampler "stuck at the first iteration" with `n_iter=300, n_burnin=100` on chr 21 / 2 populations. No documented resolution. **Relevance:** suggests cold-start cost is non-trivial; reinforces our need for a wall-time regression net per `feedback_regression_nets`.
- **Issue #72 (closed) — "independent LD blocks for constructing my own LD ref panels":** user asks for ldetect-derived blocks for SAS / AMR. No maintainer response captured. **Relevance:** if a user wants TorchGWAS to ingest a non-1000G / non-UKBB panel, they currently need to roll their own block partition with ldetect. We should document that our `pgs/ld_ref.py` block mode can substitute, but the variant set must still match a HapMap3-equivalent SNP universe.

**Specific topics requested in the brief contract:**
- **Numerical stability at extreme φ:** N/A from issues — none of the 12 visible issues raises this. The README recommends a φ grid bottoming at $10^{-6}$, which is already mild; the paper screens $\{10^{-6}, 10^{-4}, 10^{-2}, 1\}$ without stability caveats.
- **Convergence diagnostics:** N/A from issues. The README and paper do not document a Gelman–Rubin or ESS check; convergence is assumed at $1000K$ iterations / $500K$ burn-in. **Risk:** our existing `pgs/diagnostics.py` already computes R-hat for `pgs/prscs.py`; we should reuse it for PRS-CSx, but note that single-chain runs (the upstream default) cannot produce R-hat — multi-chain wrapping is our addition, not parity-against-upstream.
- **Iteration count recommendation:** **defaults are $n_{\text{iter}} = 1000K$, $n_{\text{burnin}} = 500K$** per the README (and the paper's Methods); we should mirror this default scaling in our MVP.
- **LD panel ancestry matching:** README does not state a hard rule, but the paper's Methods uses 1000G or UKBB panels matched to each discovery population's super-population label. **Risk:** there is no built-in fallback if a user supplies a discovery population whose super-pop label is unmapped; we must validate `--pop` against `{AFR, AMR, EAS, EUR, SAS}` early, with a clear error.

---

## 11. Risk flags for our integration

### 11.1 Extend `pgs/prscs.py` vs new `pgs/prscsx.py` — **RECOMMENDATION: NEW FILE `pgs/prscsx.py`.**

Reasoning, weighted against the existing code at `torchgwas/pgs/prscs.py`:

1. **Coupled-shrinkage MCMC has different state shape.** Single-pop `_prscs_gibbs_block` (lines 141–228) carries $(\beta, \psi, \delta, \sigma^2)$ with $\beta, \psi, \delta$ of shape $(b\text{-size},)$. Multi-pop coupling in PRS-CSx requires per-population $(\beta_k, \sigma_k^2)$ of shape $(K, b\text{-size})$ but **shared** $(\psi, \delta)$ of shape $(b\text{-size},)$ with a per-variant population-membership count $n_{\text{grp}}$. Squeezing this into the existing function signature would mean adding `K`-aware branches throughout the inner loop and would obscure the algorithmic spec (the existing docstring in `prscs.py` is exactly the canonical PRS-CS prior hierarchy from Ge 2019; mutating it to multi-pop changes its provenance).
2. **Native dispatch surface is single-pop.** The native C++ extension is wired through `_prscs_gibbs_block_dispatch` (line 329) and the hybrid path `_prscs_gibbs_block_hybrid` (line 240). Both signatures are single-pop — adding a new `K` axis would force re-spec'ing both the C++ entry point `prscs_gibbs_block` and the per-iteration `prscs_sample_psi_delta`. Per master spec §6 ("each sub-spec lists candidate hot loops with rough speedup targets but does NOT commit to native code in the initial implementation"), PRS-CSx native acceleration is deferred — but if we extend in-place we destabilize the *existing* native path before its V1-equivalence regression even gets a chance to settle.
3. **Master spec already leans this way.** Master spec §10 row R-MOD-3 reads "default lean: separate file to avoid V1-equivalence regressions in `prscs.py`." Our research confirms that lean is correct.
4. **Code reuse is still cheap with separate files.** `pgs/prscsx.py` can import and re-use `sample_gig` and `_block_iter` from `prscs.py` and `ldpred2.py`, the `BasePGSMethod` / `LDReference` / `PGSResult` from `base.py`, and the standardized-effect helper `_marginal_beta_std` from `ldpred2.py`. The only genuinely new code is the multi-population Gibbs kernel and the multi-pop harmonization wrapper. The duplication is intentional — the *spec* is duplicated (PRS-CS vs PRS-CSx are two separate published methods), so the *implementation* should be too.

**Counter-argument considered and rejected:** "Single-pop is just $K=1$ of multi-pop; one function suffices." True mathematically, but at $K=1$ the existing code has a 4-mode dispatcher (Python / C++-full / C++-hybrid / native-disabled) and observed-then-floored parity tolerances against PRS-CS upstream. Collapsing to the multi-pop generalization would require re-running every parity test in `tests/test_prscs.py` and risks invalidating Tier-2 PRS-CS gates. Not worth it.

### 11.2 LD panel format mismatch — **RISK: significant.**

`torchgwas/pgs/ld_ref.py` (`LDReference` dataclass) stores LD as **either** a single `(m, m)` `R_full` torch tensor **or** a list of `(b_k, b_k)` `R_blocks` torch tensors with a `block_index`, plus per-variant SNP / chr / pos / a1 / a2 / af lists, persisted via `torch.save`.

PRS-CSx panels are **HDF5 `'blk_N/ldblk'` + `'blk_N/snplist'`** per chromosome, **per ancestry**. Differences:

- HDF5 vs torch.save: requires `h5py` import path in the loader.
- Per-ancestry: PRS-CSx panels are a *family* of single-ancestry panels, not one multi-ancestry object. We need a new `MultiAncestryLDReference` container (a `dict[str, LDReference]` keyed by ancestry code, plus a shared variant-universe SNP info table) — or thread a `population` argument through everything.
- Variant alignment: PRS-CSx's master `snpinfo_mult_*` file is the **shared variant universe** with per-pop allele-flip flags pre-computed. Our `pgs/ld_ref.py` has no equivalent — alignment happens at sumstats-load time via `_harmonize` in `BasePGSMethod`. We need a multi-pop harmonization step that intersects the union with the target `.bim` and then dispatches per-population alignment.

**Recommendation:** add `pgs/ld_ref_multi.py` (or extend `pgs/ld_ref.py` with a `load_multipop_ld_reference(ancestry_dirs: dict[str, Path])` constructor that returns a `dict[str, LDReference]` plus the shared SNP-info DataFrame). Either way, this is a real new module-level surface, not a one-line addition.

### 11.3 Architectural overlap with `postgwas/_multi_ancestry.py` — **RISK: low; methods are genuinely separate.**

`postgwas/_multi_ancestry.py` implements **MR-MEGA and MANTRA**, which are *meta-analysis* methods producing per-variant pooled effect / Bayes factor + heterogeneity tests on summary stats. PRS-CSx is a *posterior-effect-estimation* method producing per-(population, variant) shrunk posterior $\beta$. The two operate at different layers (meta-analysis vs PGS construction), have different I/O contracts (per-variant scalar vs per-(population, variant) vector), and target different downstream consumers (reporting / fine-mapping vs PLINK `--score`).

We should NOT try to share code across these. The only architectural touchpoint is a possible future "use MR-MEGA's allele-frequency PCA to infer ancestry assignments for admixed targets" feature — explicitly out of MVP.

### 11.4 Streaming compatibility with `iter_chunks` — **RISK: medium; PRS-CSx is not naturally streaming.**

`feedback_streaming` (and master spec §6) requires every new CLI scan command to stream chunks via `iter_chunks`. PRS-CSx works **only on summary statistics + LD panels**, *not* on a genotype matrix — so the `iter_chunks` contract that applies to scan paths (GLM/LMM/etc.) does not literally apply here. The existing `pgs/prscs.py` already operates without genotype streaming (it iterates LD blocks via `_block_iter(ref)`, which is a streaming-equivalent that yields one block at a time).

The relevant streaming question for PRS-CSx is: **does the MCMC require all $K$ populations' LD blocks for variant $j$ in RAM simultaneously?** Answer: **yes, within one block.** The shared $\psi_j$ update at step 3 of §2.3 needs $\sum_k \beta_{jk}^2 / \sigma_k^2$ — i.e., it needs the current $\beta_{jk}$ from every population in which variant $j$ appears. So per-block memory scales as $O(K \cdot b_{\max}^2)$ for the LD matrices plus $O(K \cdot b_{\max})$ for the per-pop effects. With $b_{\max} \le 5000$ (HapMap3 LD blocks rarely exceed this; PRS-CS uses ldetect blocks averaging ~1500 SNPs) and $K=2$, this is trivially in RAM (~200 MB / block in float64).

Streaming **across** blocks is fine — `_block_iter` already does this and PRS-CSx can mirror that pattern unmodified. Streaming **within** a block would break the MCMC update (block-Cholesky requires the full $D_k$).

The genotype-streaming regression net is therefore **not directly applicable** to PRS-CSx. We still need a memory regression test, but it should assert per-block RAM ceiling, not per-chunk genotype RAM.

### 11.5 Other risks

- **GIG sampler reuse:** `sample_gig` in `pgs/prscs.py` (line 108) uses scipy's `geninvgauss` and is loop-over-scalar-draws. PRS-CSx will hit it $M$ times per iteration (vs single-pop $m_b$ times), and $M \approx 10^6$ for HapMap3 across ancestries. **Risk:** the scipy GIG path becomes a dominant bottleneck. The native `_prscs_native.prscs_sample_psi_delta` C++ entry already exists and could be re-used **if** we generalize its interface to accept the per-variant population count $n_{\text{grp}}$ as an extra argument. Master spec §6 defers native to a sub-phase, so MVP can ship on the scipy path and accept the slowdown — but Tier-3 will need the C++ path.
- **Per-population $\sigma_k^2$ initialization:** existing PRS-CS holds $\sigma^2 = 1.0$ in the hybrid path (line 273); the multi-pop sampler updates each $\sigma_k^2$ via Gamma. We need to *not* short-circuit this in the MVP just because the single-pop path does — the cross-pop borrowing relies on $\sigma_k^2$ being faithful per population.
- **`thin` factor:** existing PRS-CS uses no thinning (every post-burn-in sample contributes; `prscs.py:219–221`). PRS-CSx defaults to `thin=5`. We must either honor `thin` (preferred — matches upstream) or document the divergence (not recommended, breaks parity at the per-variant level).

---

## 12. Citations

All accessed 2026-05-11 unless noted.

1. PRS-CSx GitHub repository — `https://github.com/getian107/PRScsx` (README, version history, usage block).
2. PRS-CSx LICENSE (MIT) — `https://github.com/getian107/PRScsx/blob/master/LICENSE`.
3. PRS-CSx release v1.1.0, 2023-08-11 — `https://github.com/getian107/PRScsx/releases/tag/v1.1.0`.
4. PRS-CSx `PRScsx.py` main entry script — `https://raw.githubusercontent.com/getian107/PRScsx/master/PRScsx.py` (argument parsing, default scaling of `n_iter` / `n_burnin` to $1000K$ / $500K$).
5. PRS-CSx `mcmc_gtb.py` Gibbs sampler source — `https://raw.githubusercontent.com/getian107/PRScsx/master/mcmc_gtb.py` (per-population $\beta_k$, $\sigma_k^2$ updates; shared $\delta$ and $\psi$ updates with `n_grp[jj]` cross-population count; auto-φ Gamma update).
6. PRS-CSx `parse_genet.py` I/O source — `https://raw.githubusercontent.com/getian107/PRScsx/master/parse_genet.py` (`parse_ref`, `parse_bim`, `parse_sumstats`, `parse_ldblk`, `align_ldblk` signatures and column layouts).
7. Ruan, Y. et al. (2022). "Improving polygenic prediction in ancestrally diverse populations." *Nature Genetics* 54:573–580. DOI: 10.1038/s41588-022-01054-7 — `https://www.nature.com/articles/s41588-022-01054-7` (Methods §"PRS-CSx" prior hierarchy; Figures 3a/3b/4a validation results).
8. Open-access mirror of [7]: PMC9117455 — `https://pmc.ncbi.nlm.nih.gov/articles/PMC9117455/`.
9. Ge, T. et al. (2019). "Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors." *Nature Communications* 10:1776. DOI: 10.1038/s41467-019-09718-0 — original PRS-CS prior hierarchy that PRS-CSx generalizes.
10. Ge, T. et al. (2022). "Development and validation of a trans-ancestry polygenic risk score for type 2 diabetes in diverse populations." *Genome Medicine* 14:70 — applied PRS-CSx reference (cited by upstream README).
11. Kachuri, L. et al. (2024). "Principles and methods for transferring polygenic risk scores across global populations." *Nature Reviews Genetics* 25:8–25 — best-practices review covering PRS-CSx vs MultiPRS / PolyPred (cited by upstream README; not used here for primary numbers, only for context).
12. Issue #67 — small non-EUR GWAS sample sizes — `https://github.com/getian107/PRScsx/issues/67`.
13. Issue #68 — "PRSCSx taking so long" — `https://github.com/getian107/PRScsx/issues/68`.
14. Issue #72 — independent LD blocks for custom panels — `https://github.com/getian107/PRScsx/issues/72`.
15. Issue #73 — PRS-CSx ancestry selection in admixed samples — `https://github.com/getian107/PRScsx/issues/73`.
16. Issue #75 — alleged single-population logic regression — `https://github.com/getian107/PRScsx/issues/75`.
17. Issue #76 — validation set for tuning and weight estimation — `https://github.com/getian107/PRScsx/issues/76`.
18. PRS-CSx posterior weights data release (Dropbox) — `https://www.dropbox.com/sh/5v1bzlukxoor9fi/AACA580wl_gNKapqWvx3siOza?dl=0` (referenced from README "Data Releases").
19. PRS-CSx Broad mirror for LD panels — `https://personal.broadinstitute.org/hhuang//public//PRS-CSx/Reference` (alternate-host mirror per README).

**Internal repo references** (for the integration-risk section, not cited as primary external sources):
- `torchgwas/pgs/prscs.py` (existing single-population implementation; lines cited inline above).
- `torchgwas/pgs/ld_ref.py` (existing LD reference container).
- `torchgwas/pgs/sumstats_io.py` (existing PGS sumstats loader).
- `torchgwas/postgwas/_multi_ancestry.py` (existing MR-MEGA / MANTRA module).
- `docs/superpowers/specs/2026-05-11-modernization-master-design.md` (master spec, for scope contract).
