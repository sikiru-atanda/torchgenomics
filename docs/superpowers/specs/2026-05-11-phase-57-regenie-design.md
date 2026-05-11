# Phase 57 — REGENIE Step-1 + Step-2 Parity (MVP) — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven)
- **Date:** 2026-05-11
- **Branch:** `modernization/specs`
- **Master spec:** [`2026-05-11-modernization-master-design.md`](2026-05-11-modernization-master-design.md)
- **Research brief:** [`../research/regenie-research.md`](../research/regenie-research.md)
- **Phase number:** 57
- **Scope class:** MVP (per master spec §2)
- **Sequencing constraint:** depends on X-chromosome refactor track landing first (per master spec §3 and brief R-REG-4)
- **Living spec.** Tier A is fixed; Tier B / C may be revised at implementation kickoff.

---

## 1. Goal & scope

Add reference-equivalent REGENIE step-1 (whole-genome ridge with chromosome-level LOCO predictors) and step-2 (single-variant scan using LOCO predictors as offsets) to TorchGWAS. REGENIE (Mbatchou et al. 2021 [C1]) is the modern biobank-default LMM workflow — published as the UK Biobank reference pipeline and adopted by Regeneron's PheWAS at scale. Its absence from the toolkit is the single most-likely "what about REGENIE?" question reviewers will ask, per master spec §1.

**In scope (MVP):**

- **Step 1 — quantitative trait** whole-genome ridge with cross-validated stacking (Level-0 block ridge + Level-1 stacking ridge), producing per-chromosome LOCO offsets. Spec from REGENIE Overview docs [C5] §"Step 1".
- **Step 1 — binary trait** logistic ridge with the same Level-0 / Level-1 / LOCO structure. Spec from REGENIE Overview docs [C5] §"Step 1, binary".
- **Step 2 — quantitative score test** with LOCO offset $\hat y_{-k}$ as fixed effect. Per Mbatchou 2021 [C1] equation (3); see §2.2 for our notation.
- **Step 2 — binary score test** with LOCO offset entering the linear predictor + **SPA fallback** when normal-approximation $p < $ `--pThresh` (default 0.05). SPA backed by our existing `torchgwas/stats/spa.py` (dense Lugannani-Rice — see §2.3 for divergence note vs REGENIE's fastSPA).
- **CLI:** new subcommand `regenie-scan` exposing the Step 1 + Step 2 flow as a single end-to-end command, plus separate `regenie-step1` / `regenie-step2` subcommands for users who want to reuse Step 1 across multiple Step 2 runs.
- **LOCO file I/O:** read REGENIE's published `.loco` format (per [C5] §Output) so users can run REGENIE step 1 externally and consume the predictions in our step 2; write the same format so users can do the inverse.
- **Tier 2 parity** vs REGENIE upstream binary on the bundled `example/` fixture (per brief §8) within the `floor + observed × 2` tolerance per master spec §5.

**Explicitly out of scope (deferred — see brief §5 "Out of scope of MVP"):**

- **Firth correction** (`--firth`, `--firth-se`, `--approx`). Defer to a follow-on phase.
- **Time-to-event / Cox regression** (`--t2e`, `--eventColList`). Defer.
- **Gene / region / SKAT / SBAT burden tests** (`--anno-file`, `--mask-def`, `--set-list`). Defer.
- **Interaction tests** (`--interaction`).
- **Conditional analyses** (`--condition-list`, `--condition-file`).
- **Joint tests** (NNLS / SBAT / SKATO / ACAT).
- **HTP output format** (`--htp`). Stick to standard REGENIE per-variant output for V1.
- **Native VCF/BCF input.** REGENIE does not natively support VCF (per brief Issue #677); users must convert upstream. We mirror this constraint and document it.
- **Imputed-genotype Step 1.** Per brief §10.5 (issues #659, #530), REGENIE recommends array-genotype-only Step 1; we mirror this recommendation in the docstring.
- **fastSPA.** Defer to a follow-on sub-phase. We use our existing dense Lugannani-Rice SPA from `stats/spa.py`. See §2.3 for tolerance impact.

## 2. Algorithm reference

### 2.1 Step 1 — whole-genome ridge regression (Level 0 + Level 1 + LOCO)

Per REGENIE Overview docs [C5] §"Step 1":

**Level 0 (block-wise ridge).** Markers partitioned into consecutive blocks of size `--bsize`. For each block $b$ with genotype submatrix $X_b \in \mathbb{R}^{N \times M_b}$ and phenotype $y \in \mathbb{R}^N$, $J$ ridge predictors are computed across a fixed grid of shrinkage values $\{\lambda_j\}_{j=1}^J$:

$$
\hat\beta_{b,j} = (X_b^\top X_b + \lambda_j I)^{-1} X_b^\top y, \qquad j = 1,\dots,J
$$

Stacking the per-block per-shrinkage predictions $\hat y_{b,j} = X_b \hat\beta_{b,j}$ column-wise produces the **Level 0 design matrix**:

$$
W \in \mathbb{R}^{N \times (B \cdot J)}
$$

with $B = \lceil M / \text{bsize} \rceil$.

**Level 1 (cross-validated stacking ridge).** A second ridge regression combines the $B \cdot J$ Level-0 columns. For quantitative trait:

$$
\hat\alpha = \arg\min_\alpha \|y - W\alpha\|^2 + \tau \|\alpha\|^2
$$

with $\tau$ selected from a grid by `--cv`-fold CV MSE.

For binary trait, log-odds $\eta = W\alpha + Cc$ fit by **logistic ridge regression**, again CV across $\tau$.

**LOCO predictions.** Per-chromosome predictions are constructed by zeroing out the columns of $W$ derived from chromosome $k$ before applying $\hat\alpha$:

$$
\hat y_{-k} = W_{-k}\, \hat\alpha_{-k}, \qquad k = 1,\dots,K
$$

Written to one `.loco` file per phenotype. File format per [C5] §Output: $24 \times (N+1)$ matrix (1 header row of $N$ sample IDs + 23 chromosome rows; sex chromosomes collapsed into chromosome 23 row by REGENIE convention).

### 2.2 Step 2 — single-variant association testing

Per Overview docs [C5] §"Step 2":

**Quantitative trait — score test of marker $g$ on chromosome $k$:**

$$
\tilde y = y - \hat y_{-k} - C \hat c, \qquad
T_g = \frac{g^\top \tilde y}{\sqrt{\widehat{\mathrm{Var}}(g^\top \tilde y)}}, \qquad
T_g^2 \sim \chi^2_1
$$

with $C$ the covariate matrix and $\hat c$ the Step-2 covariate effects (re-fit; not from Step 1).

**Binary trait — logistic score test:** the LOCO offset enters the linear predictor:

$$
\mathrm{logit}\,\Pr(Y_i = 1) = C_i \hat c + \hat y_{-k,i} + g_i \beta
$$

Score statistic and variance computed under the null $\beta = 0$ to give a normal-approximation p-value $p_{\mathrm{norm}}$.

**SPA fallback:** when $p_{\mathrm{norm}} < $ `--pThresh` (default 0.05 per [C7] Options docs), recompute via saddlepoint approximation. Our implementation calls `torchgwas/stats/spa.py::lugannani_rice_pvalue` (existing module, derived from Daniels 1954 [C19] and Lugannani-Rice 1980 [C20], same as SAIGE's Zhou 2018 [C21] reference).

### 2.3 Algorithmic note: dense vs fast SPA

REGENIE uses **fastSPA** — the SAIGE variant that exploits sparse-genotype structure by partitioning the cumulant generating function into "carrier" + "non-carrier" terms (verified by source inspection `src/Step2_Models.cpp:2068-2280` per brief R-REG-3). Our `stats/spa.py` uses the **dense Lugannani-Rice on the full $g$ vector**.

The two are algorithmically equivalent but evaluate the CGF in different orders. For typical variants ($\text{MAF} > 0.01$) this produces $p$-value differences in the 1e-6 to 1e-5 absolute range, well within our `floor + observed × 2` tolerance. For rare variants ($\text{MAF} < 0.001$) the difference can reach 1e-3 relative, still within tolerance for MVP scope.

**If Tier 2 reveals rare-variant divergences exceeding tolerance**, the response is NOT to rewrite our existing SPA — it is to add a `fastSPA` variant of `stats/spa.py` (separate function) that mirrors REGENIE's sparse decomposition for parity-mode runs, while keeping the dense path as the spec body per `Repo Conventions` (Python-as-spec, native-as-shortcut). This is deferred to a follow-on sub-phase.

### 2.4 Dimension table

| Symbol | Meaning | Typical value | Source |
|---|---|---|---|
| $N$ | samples | $5\times 10^5$ (UKB) | Mbatchou 2021 [C1] abstract |
| $M$ | array variants used in Step 1 | $\sim 5\times 10^5$ | REGENIE recommendations [C8] |
| $K$ | LOCO blocks (chromosomes) | 23 (autosomes + chrX collapsed) | [C5] §Output |
| $B$ | Step-1 blocks per genome | $\lceil M / \text{bsize} \rceil$ ≈ 500 at `--bsize 1000` | [C5] |
| $J$ | Level-0 ridge grid size | 5 (default) | [C7] Options docs `--l0` |
| $J_1$ | Level-1 ridge grid size | 5 (default) | [C7] Options docs `--l1` |
| `--cv` | CV folds | 5 (default) | [C7] Options docs |
| `--pThresh` | SPA invocation cutoff | 0.05 (default) | [C7] Options docs §Step 2 |
| `--minMAC` | Step 2 variant filter | 5 (default) | [C7] Options docs §Step 2 |

### 2.5 Default ridge grid

UNVERIFIED (per brief §2.4 — flagged as third-party-sourced): $\{0.01, 0.25, 0.5, 0.75, 0.99\}$ for both `--l0` and `--l1`. Source: NIH Biowulf REGENIE docs [C18]. Phase 57 implementation MUST reconfirm at install time by either running `regenie --help` or grepping `cxxopts` parameter declarations in `src/Regenie.cpp`. If the actual defaults differ, this spec is amended and the test fixtures are regenerated.

### 2.6 Covariate standardization (deliberate divergence from upstream)

Per brief §11 R-REG-5 and upstream issue #679 [C16]: REGENIE's Step-1 ridge regression is **not scale-invariant** under default settings (continuous covariates with units in millions of mm³ inflate $\lambda_{GC}$). Our implementation MUST internally z-score continuous covariates before feeding them to the Level-0 ridge solve, which is more robust than upstream defaults.

This is a **deliberate divergence-on-purpose** documented in the docstring of `RegenieStep1.fit_null()` and called out in `docs/validation_findings.md` at implementation time as "not a bug; intentional improvement; see issue #679 for the upstream behavior we deviate from."

## 3. Module layout

### 3.1 New files

| Path | Purpose |
|---|---|
| `torchgwas/models/regenie_step1.py` | `RegenieStep1` class — Level 0 + Level 1 + LOCO predictor producer. Implements `BaseModel.fit_null()` returning `RegenieStep1Result` containing per-phenotype LOCO predictions. |
| `torchgwas/models/regenie_step2.py` | `RegenieStep2` class — score test with LOCO offset. Implements `BaseModel.score_chunk()`. Quantitative + binary-with-SPA. |
| `torchgwas/io/regenie_loco.py` | `RegenieLOCOReader` (streamed per-chromosome row reader) and `RegenieLOCOWriter` for round-trip with REGENIE upstream. Supports REGENIE's published `.loco` format ($24 \times (N+1)$ per [C5] §Output). |
| `torchgwas/io/regenie_pred_list.py` | `_pred.list` parser (phenotype name → `.loco` file path mapping). |
| `tests/fixtures/regenie/` | Symlink to REGENIE's bundled `example/` fixture (downloaded once during `pytest --setup-only` setup; pre-flight disk check per `feedback_preflight`). |
| `tests/test_regenie_step1.py` | Tier 1 unit tests on Level 0 + Level 1 ridge solves. |
| `tests/test_regenie_step2.py` | Tier 1 unit tests on score test + SPA fallback path. |
| `tests/test_regenie_parity_upstream.py` | Tier 2 parity vs REGENIE upstream binary. |
| `tests/test_regenie_loco_io.py` | Round-trip read/write of `.loco` format. |

### 3.2 Modified files

| Path | Change |
|---|---|
| `torchgwas/cli.py` | Add `regenie-scan`, `regenie-step1`, `regenie-step2` subcommands. Reuse existing `xchrom_flags()` group from X-chrom refactor. |
| `torchgwas/scan/unified.py` | No change required (chrom-boundary chunk splits already added by X-chrom refactor as Tier A item A4). |
| `torchgwas/stats/spa.py` | No change required (existing `lugannani_rice_pvalue` is reused). May add a docstring note pointing to the REGENIE fastSPA divergence in §2.3 above. |
| `torchgwas/models/base.py` | No change required (`SampleMeta`, `ChunkContext` already added by X-chrom refactor). |
| `tests/test_streaming_memory.py` | Extend to cover `regenie-scan` per `feedback_streaming` memory. |
| `bench/native_speedups.py` | Extend wall-time gate to include `regenie-scan` per `feedback_regression_nets` memory. |
| `docs/cli.md` | Add `regenie-scan` documentation. |
| `CLAUDE.md` | Update CLI subcommand count (37 → 40, adding `regenie-scan`, `regenie-step1`, `regenie-step2`). Update Phase Index with Phase 57 entry. |

### 3.3 Public API surface

```python
# torchgwas/models/regenie_step1.py
class RegenieStep1(BaseModel):
    """REGENIE step-1 whole-genome ridge regression with LOCO predictors.

    Implements Mbatchou et al. 2021 [C1] step 1: Level-0 block ridge + Level-1
    cross-validated stacking ridge + per-chromosome LOCO predictors.

    Args:
        bsize: Step-1 block size. Required; recommended 1000 per
            REGENIE recommendations [C8].
        l0_grid: Level-0 ridge shrinkage grid (default
            (0.01, 0.25, 0.5, 0.75, 0.99) per [C18]; UNVERIFIED — confirm
            against `regenie --help` at implementation time).
        l1_grid: Level-1 ridge shrinkage grid (same default).
        cv: Cross-validation folds (default 5 per [C7]).
        bt: Binary trait flag (default False = quantitative).
        standardize_covariates: Internally z-score continuous covariates
            before Level-0 fit. Default True (deliberate divergence from
            upstream — see §2.6 and issue #679 [C16]).

    Returns:
        RegenieStep1Result with per-phenotype `.loco` arrays and CV-selected
        `tau` for Level 1.
    """
    ...

# torchgwas/models/regenie_step2.py
class RegenieStep2(BaseModel):
    """REGENIE step-2 single-variant score test using LOCO offsets.

    Args:
        loco: RegenieStep1Result OR RegenieLOCOReader OR path to _pred.list.
        bt: Binary trait flag.
        spa: Enable SPA fallback for binary (default True).
        p_thresh: SPA invocation cutoff (default 0.05 per [C7]).
        min_mac: Minimum minor-allele-count filter (default 5 per [C7]).
    """
    ...
```

## 4. Data flow

### 4.1 End-to-end (Step 1 + Step 2)

```
.bed/.bim/.fam (or .pgen/.pvar/.psam, or .bgen + .sample)
   ↓ (existing torchgwas/io readers)
G: Tensor[N × M_array], pheno: Tensor[N × P], covar: Tensor[N × C]
   ↓ (RegenieStep1.fit_null with bsize, l0_grid, l1_grid, cv)
RegenieStep1Result {
    loco: Tensor[N × P × K=23],
    tau_per_pheno: Tensor[P],
    cv_mse: Tensor[P × |tau_grid|],
}
   ↓ (RegenieLOCOWriter — optional, for round-trip with upstream)
.loco files + _pred.list (REGENIE-format on disk)

Step 2 (independent):
.bgen / .pgen with imputed dosages
   ↓ (existing torchgwas/io chunk readers, iter_chunks per chromosome)
G_chunk: Tensor[N × M_chunk] (single chromosome k)
   ↓ (RegenieStep2.score_chunk with loco offset for chromosome k)
Per-variant β, SE, χ², log10p (REGENIE-format columns per [C5] §4.2)
   ↓ (UnifiedScanner.collect)
results/regenie_step2.tsv
```

### 4.2 Streaming compliance

**LOCO file streaming.** The `RegenieLOCOReader` reads ONE chromosome row per Step-2 chunk (single seek + 1 line read on chromosome change). At UKB scale ($N = 5 \times 10^5$, FP64) one chromosome row is ~4 MB; full file per phenotype is ~92 MB. With 50 phenotypes, ~4.6 GB on disk total — manageable. **No re-materialization of the full $N \times K$ predictions matrix is required.** Brief R-REG-1 explicitly resolves master spec R-MOD-2.

**Genotype streaming.** Step 2 inherits `iter_chunks` from `UnifiedScanner` per `feedback_streaming` memory. `tests/test_streaming_memory.py` extended to cover `regenie-scan`.

**Step 1 memory:** the Level-0 design matrix $W \in \mathbb{R}^{N \times (B \cdot J)}$ at biobank scale ($N = 5 \times 10^5$, $B = 500$, $J = 5$) = $5 \times 10^5 \times 2500$ = $1.25 \times 10^9$ FP64 elements = 10 GB. **Too large for default RAM.** Mitigation: support REGENIE's `--lowmem` mode via per-block disk spill (Level-0 outputs per block written to disk, Level-1 streams them back). Implementation: Tier B item, not Tier A.

## 5. Validation gates

### 5.1 Tier 1 — math correctness (synthetic, deterministic)

Required tests in `tests/test_regenie_step1.py` and `tests/test_regenie_step2.py`:

- **Level-0 ridge solve** (per §2.1): hand-crafted $50 \times 100$ fixture with known closed-form $\hat\beta = (X^\top X + \lambda I)^{-1} X^\top y$. Compare against `scipy.linalg.solve` with the same regularizer to ≤ 1e-10 absolute.
- **Level-1 stacking ridge solve**: same closed-form check on $W \in \mathbb{R}^{50 \times 250}$.
- **CV fold MSE**: hand-coded 5-fold split; assert exact MSE matches sklearn `KFold` + `Ridge` on the same fixture.
- **LOCO column zeroing**: verify $\hat y_{-k}$ uses only non-chromosome-$k$ columns of $W$.
- **Score test statistic** (per §2.2): closed-form on a $100 \times 1$ fixture with known $\tilde y$ and $g$; compare to `scipy.stats.chi2.sf` to ≤ 1e-10.
- **SPA fallback trigger**: assert that $p_{\mathrm{norm}} \geq $ `pThresh` skips SPA and $p_{\mathrm{norm}} < $ `pThresh` invokes `lugannani_rice_pvalue` (mock the SPA call and assert call count).
- **Logistic ridge null log-likelihood**: closed-form on a 100-sample binary fixture; compare to `sklearn.linear_model.LogisticRegression(penalty='l2', C=1/(2*tau))` to ≤ 1e-6 (the `1/(2*tau)` parametrization conversion documented).

### 5.2 Tier 2 — smoke parity vs REGENIE upstream binary

Required test in `tests/test_regenie_parity_upstream.py`:

- **Fixture**: REGENIE's bundled `example/` fixture (per brief §8) — 500 samples, 1000 variants, BED + BGEN + PGEN, 2 quantitative phenotypes (Y1, Y2) + binary surrogate, 3 quantitative covariates + 1 binary + 1 categorical. Total ~600 KB. **Use the upstream test suite's checked-in expected outputs as our oracle** (file: `example/example.test_bin_out_firth_Y1.regenie`).
- **Pre-flight install check**: `regenie --version` returns expected pinned version. Disk + RAM headroom check per `feedback_preflight`.
- **REGENIE upstream pin**: v4.2 if available (after brief §10.1 Issue #678 fix); else v4.1 with explicit subset-mode parity restriction (only test on full sample, not a male-only or female-only subset, to avoid the SE deflation bug). Pin recorded in `tests/conftest.py::REGENIE_PINNED_VERSION`.
- **Reference run**: `regenie --step 1 --bed example/example --phenoFile example/phenotype.txt --covarFile example/covariates.txt --bsize 100 --out step1` followed by `regenie --step 2 --bed example/example --phenoFile example/phenotype.txt --covarFile example/covariates.txt --pred step1_pred.list --bsize 100 --out step2`.
- **Our run**: same inputs through `torchgwas regenie-scan ...` with matching flag mappings.
- **Comparison**: join on `(CHROM, GENPOS, ID, ALLELE0, ALLELE1, TEST)` and assert per-variant agreement.
- **Tolerance** (per master spec §5; MVP scope): `floor + observed × 2`. Floor: 1e-6 absolute on β and SE, 1e-3 relative on log10p. Observed divergence captured at first run, used to set the actual tolerance.
- **Failure mode**: any variant outside tolerance → log to `docs/validation_findings.md` per F3 severity policy. Phase does NOT block on documented divergences (post-V1 per master spec §7).
- **Unrelated samples requirement** (per brief §11 R-REG-7 and §10.4): the bundled `example/` fixture contains unrelated samples (verified by `wc -l example/example.fam` showing 500 distinct FIDs). Family-pedigree fixtures would test REGENIE's published over-correction, not our implementation parity.

### 5.3 Tier 2 — binary-with-SPA parity

Same fixture, recoded to binary phenotype:

- Run upstream `regenie --step 2 --bt --spa --pThresh 0.05 ...`.
- Run our `torchgwas regenie-scan --bt --spa --p-thresh 0.05 ...`.
- Same join and tolerance rule.
- Expect 1e-6 to 1e-5 absolute divergences on common variants (per §2.3). Rare variants ($\text{MAF} < 0.001$) may show 1e-3 relative divergence — within tolerance.

### 5.4 Tier 3 — full-scale empirical (deferred per NA3)

Documented assertion (not run): on UK Biobank chr22 scan (~500K samples, ~200K variants), our outputs match REGENIE step-2 to within `floor + observed × 2` tolerance. Deferred to NA3 task.

### 5.5 LOCO file round-trip

Required test in `tests/test_regenie_loco_io.py`:

- Generate LOCO from our `RegenieStep1` on bundled fixture.
- Write via `RegenieLOCOWriter`.
- Read back via REGENIE upstream's `--pred` flag in step 2.
- Assert step-2 output matches our step-2 output within tolerance (closes the loop).

## 6. CLI integration

### 6.1 New subcommands

```bash
torchgwas regenie-step1 \
    --bed data/array \
    --pheno data/pheno.tsv \
    --covar data/covar.tsv \
    --bsize 1000 \
    --bt \              # if binary
    --cv 5 \
    --out out/step1

torchgwas regenie-step2 \
    --bgen data/imputed.bgen \
    --sample data/imputed.sample \
    --pheno data/pheno.tsv \
    --covar data/covar.tsv \
    --pred out/step1_pred.list \
    --bsize 200 \
    --bt --spa \        # if binary
    --p-thresh 0.05 \
    --min-mac 5 \
    --out out/step2

# Convenience wrapper (Step 1 + Step 2 in one command)
torchgwas regenie-scan \
    --array-bed data/array \
    --imputed-bgen data/imputed.bgen \
    --pheno data/pheno.tsv \
    --covar data/covar.tsv \
    --bsize-step1 1000 \
    --bsize-step2 200 \
    --bt --spa \
    --out out/regenie
```

### 6.2 Flag mapping (ours → REGENIE upstream)

| Ours | REGENIE upstream | Notes |
|---|---|---|
| `--bed` | `--bed` | identical |
| `--pgen` | `--pgen` | identical |
| `--bgen` | `--bgen` | identical |
| `--sample` | `--sample` | BGEN only |
| `--pheno` | `--phenoFile` | identical (we use shorter form per existing CLI convention) |
| `--covar` | `--covarFile` | identical |
| `--bsize` | `--bsize` | identical |
| `--bt` | `--bt` | identical |
| `--cv` | `--cv` | identical |
| `--spa` | `--spa` | identical |
| `--p-thresh` | `--pThresh` | identical (kebab-case per our CLI convention) |
| `--min-mac` | `--minMAC` | identical |
| `--min-info` | `--minINFO` | identical |
| `--apply-rint` | `--apply-rint` | identical |
| `--pred` | `--pred` | identical (Step 2 only) |
| `--threads` | `--threads` | identical |
| `--out` | `--out` | identical |
| `--xchr-model` | (none — REGENIE uses `--par-region` + sex-driven default) | inherits from X-chrom refactor; we expose explicitly per master spec convention |

### 6.2.1 Deferred upstream flags (NotImplementedError in MVP)

Users running `regenie-scan --help` will see the flags below listed as DEFERRED with a one-line explanation. Passing any of them raises `NotImplementedError` with the deferral pointer. This makes the MVP scope boundary explicit in the user surface.

| Upstream flag | Deferred to | Reason |
|---|---|---|
| `--firth`, `--firth-se`, `--approx`, `--write-null-firth` | follow-on phase | Firth correction explicitly out of MVP per §1 |
| `--t2e`, `--eventColList` | follow-on phase | Time-to-event / Cox out of MVP per §1 |
| `--anno-file`, `--mask-def`, `--set-list`, `--mask-lovo`, `--vc-tests`, `--rgc-gene-p` | follow-on phase | Gene/region/SKAT/SBAT burden out of MVP per §1 |
| `--interaction`, `--interaction-snp`, `--interaction-prs` | follow-on phase | Interaction tests out of MVP per §1 |
| `--condition-list`, `--condition-file` | follow-on phase | Conditional analyses out of MVP per §1 |
| `--htp` | follow-on phase | HTP output format out of MVP per §1 |
| `--lowmem`, `--lowmem-prefix` | Tier B (per §10.2 B2) | Per-block disk spill for Step 1 — deferred from Tier A but in-scope for Phase 57 |
| `--gz` | Tier B | Output compression — defer to Tier B; MVP writes uncompressed |

**Streaming compliance**: `regenie-step2` and `regenie-scan` both use `iter_chunks` per `feedback_streaming` memory. Step 1 reads the array-genotype matrix in `bsize`-sized blocks (Level-0 fit per block), which is naturally streaming-friendly.

## 7. Native acceleration scope (deferred)

Per `Python-as-spec, native-as-shortcut` convention. Pure-torch path is the spec.

Candidate hot loops for a future Phase-41-style sub-phase:

- **Level-0 ridge solves** ($B \cdot J$ small ridge regressions). Already vectorizable in torch via batched `torch.linalg.solve`. May not need native.
- **Level-1 stacking ridge** ($B \cdot J$-column Cholesky solve). Existing `_native/eigh.cpp` likely covers it.
- **Step-2 score test inner loop** (vector dot products over chunks). Already covered by existing `_dispatch.select_path` machinery for our other LMM scanners.
- **fastSPA sparse-genotype CGF** — listed in §2.3 as a separate consideration; native dispatch would be the natural home if/when added.

## 8. F3 severity application

Per master spec §7:

| Event | Action |
|---|---|
| Tier 1 test failure | Halt; fix the bug |
| Tier 2 quantitative divergence outside tolerance | Document; investigate (may be R-REG-1 LOCO read bug, R-REG-5 covariate scaling difference, or rare-variant SPA per §2.3); does not block |
| Tier 2 binary divergence outside tolerance on rare variant | Document with note "expected per §2.3 fastSPA difference"; does not block |
| Tier 2 LOCO round-trip failure | **Halt** — round-trip is fundamental correctness, not a parity question |
| Pinned REGENIE version becomes unavailable | Re-pin to nearest version, re-baseline tolerance, document |
| ≥ 3 unrelated divergences during Tier B | C4 emergency stop |

## 9. Risks & open questions

### 9.1 Risks (from brief §11, mitigations made concrete)

| ID | Risk | Mitigation |
|---|---|---|
| R-P57-1 | LOCO file format change in future REGENIE versions | Pin REGENIE version per §5.2; format-version detection in `RegenieLOCOReader` raising `ValueError` on unknown header |
| R-P57-2 | Step-1 `W` matrix doesn't fit in RAM at biobank scale (10 GB at UKB) | Tier B `--lowmem` per-block disk spill |
| R-P57-3 | Scale-dependent covariate inflation (issue #679) | Mitigated by §2.6 internal z-scoring; documented divergence-on-purpose |
| R-P57-4 | SPA fastSPA-vs-dense divergence on rare variants | Tolerance absorbs; if exceeded, Tier C fastSPA implementation |
| R-P57-5 | REGENIE version-to-version BETA inflation (issue #684) | Pinned version per §5.2; tests re-baselined on bumps |
| R-P57-6 | Family-pedigree fixture would reflect REGENIE over-correction (Loesch 2022) | Fixture choice constraint per §5.2 |
| R-P57-7 | UNVERIFIED ridge grid defaults (§2.5) | Confirm at implementation time via `regenie --help` |
| R-P57-8 | Sex-chromosome handling depends on X-chrom refactor (R-REG-4) | Sequencing per master spec §3 — X-chrom first |
| R-P57-9 | Level-1 CV $\tau$ selection differs from REGENIE due to RNG differences in fold splitting | Decision criterion: at implementation time, check whether `regenie --step 1 --debug` exposes the per-fold sample assignments (search `src/Pheno.cpp` for `cv_idx` or `fold_idx` allocation). **If exposed:** mirror REGENIE's fold assignment by reading the debug output. **If not exposed:** fix `random_state=42` in our `KFold(n_splits=cv, shuffle=True, random_state=42)` call and document the fold-assignment divergence in `docs/validation_findings.md` as expected-bounded (≤ 1e-4 absolute on per-variant β under the `floor + observed × 2` tolerance). |
| R-P57-10 | REGENIE's `_pred.list` format may include absolute paths that don't survive being moved | `RegenieLOCOReader` resolves relative paths from the `_pred.list` directory |

### 9.2 Open questions (deferred to implementation kickoff)

- **OQ-P57-1**: Should we expose REGENIE's MKL-linked binary as the upstream pin (faster but bigger install) or the static binary? Default lean: static for simplicity; user can override with `TORCHGWAS_REGENIE_BINARY=path/to/mkl/binary`.
- **OQ-P57-2**: Multi-phenotype LOCO file emission — should we write one `.loco` per phenotype (REGENIE convention) or a single `.h5`/`.npz` with all phenotypes (more efficient on disk)? Default lean: REGENIE convention for round-trip compatibility; HDF5 as opt-in via `--loco-format hdf5`.
- **OQ-P57-3**: When the same fixture is used to test both quantitative and binary paths, do we use the bundled binary surrogate phenotype or recode Y1 to binary? Default lean: bundled binary surrogate (whatever upstream test suite uses, for direct oracle comparison).
- **OQ-P57-4**: REGENIE Step 1 supports `--apply-rint` (rank-based inverse normal transform) for QTs. Do we implement this or require user to pre-transform? Default lean: implement (it's a one-line `scipy.stats.rankdata` + `norm.ppf`); MVP must mirror REGENIE on this.
- **OQ-P57-5**: Should the `regenie-scan` convenience wrapper allow caching of Step 1 across multiple Step 2 invocations on different imputed datasets? Default lean: yes, via `--reuse-step1 path/to/cached/loco/` flag.

## 10. Implementation phasing

### 10.1 Tier A — minimum viable shipping unit

Shippable when: `regenie-scan` runs end-to-end on the bundled `example/` fixture for both quantitative and binary phenotypes, produces output matching REGENIE upstream within tolerance, and the LOCO round-trip closes.

| Step | Files | Test |
|---|---|---|
| **A0** Pre-flight: confirm X-chrom refactor merged | (sequencing gate) | n/a |
| **A1** `RegenieLOCOReader` + `RegenieLOCOWriter` | `io/regenie_loco.py`, `io/regenie_pred_list.py` | `test_regenie_loco_io.py` |
| **A2** Pure-torch Level 0 + Level 1 ridge solves | `models/regenie_step1.py` | `test_regenie_step1.py` Tier 1 unit tests |
| **A3** Cross-validated Level 1 $\tau$ selection | `models/regenie_step1.py` | unit test on closed-form CV-MSE |
| **A4** LOCO predictor construction | `models/regenie_step1.py` | unit test on column-zeroing |
| **A5** Quantitative Step 2 score test with LOCO offset | `models/regenie_step2.py` | `test_regenie_step2.py` Tier 1 unit tests |
| **A6** Binary logistic ridge in Step 1 | `models/regenie_step1.py` | unit test vs `sklearn.LogisticRegression(penalty='l2')` |
| **A7** Binary score test + SPA fallback in Step 2 | `models/regenie_step2.py` | unit test on SPA invocation |
| **A8** CLI subcommands | `cli.py` | `test_cli.py` smoke |
| **A9** Tier 2 parity vs upstream on bundled fixture (quantitative) | `test_regenie_parity_upstream.py` | this test |
| **A10** Tier 2 parity vs upstream (binary-with-SPA) | `test_regenie_parity_upstream.py` | this test |
| **A11** LOCO round-trip closure test | `test_regenie_loco_io.py` | this test |
| **A12** Streaming regression net | `tests/test_streaming_memory.py` | this test |

### 10.2 Tier B — parity-tightening + scaling

Shippable when: multi-phenotype + multi-chromosome scaling work correctly, `--lowmem` enables Step 1 on 100K+ samples, full Tier 2 parity passes on 1KG chr22 fixture.

| Step | Files | Test |
|---|---|---|
| **B1** Multi-phenotype Step 1 (one Level-1 fit per phenotype, batched LOCO emission) | `models/regenie_step1.py` | parity with upstream multi-pheno run |
| **B2** `--lowmem` per-block disk spill in Step 1 | `models/regenie_step1.py` | memory regression test at 100K samples |
| **B3** Mid-scale Tier 2 parity on 1KG phase-3 chr22 (~500 samples × 100K variants) | `test_regenie_parity_upstream.py` extended | this test |
| **B4** `--apply-rint` (rank-based inverse normal transform) | `models/regenie_step1.py` + `models/regenie_step2.py` | parity test |
| **B5** Categorical covariate handling (`--catCovarList` analogue) | `models/regenie_step1.py` | parity test |
| **B6** Per-chromosome `iter_chunks` in Step 2 (already inherited from X-chrom refactor; verify) | `scan/unified.py` (no change expected) | confirmation test |
| **B7** Wall-time regression net | `bench/native_speedups.py` | wall-time gate |

### 10.3 Tier C — nice-to-haves & follow-on

| Step | Files | Test |
|---|---|---|
| **C1** fastSPA variant in `stats/spa.py` (separate function; sparse-genotype CGF decomposition) | `stats/spa.py` extension | rare-variant parity test |
| **C2** Native dispatch for Level-0 ridge solves | `_native/regenie_l0.cpp` + dispatcher | parity + bench |
| **C3** GPU dispatch for Level-1 stacking solve | torch GPU body | bench |
| **C4** HDF5 LOCO format option | `io/regenie_loco.py` extension | format round-trip test |
| **C5** Step-1 caching across multiple Step 2 runs (`--reuse-step1`) | `cli.py` + `models/regenie_step1.py` | smoke test |
| **C6** Documented preparation for follow-on phases (Firth, Cox, gene burden) | docs only | n/a |

## 11. Citations

All claims in this spec MUST trace to a primary source.

- **[C1] Mbatchou, J., Barnard, L., Backman, J. et al. (2021).** *Computationally efficient whole-genome regression for quantitative and binary traits.* *Nature Genetics* 53:1097–1103. <https://doi.org/10.1038/s41588-021-00870-7>. PubMed: <https://pubmed.ncbi.nlm.nih.gov/34017140/>. **Methods Section paywalled** (per brief §9); equations transcribed from REGENIE Overview docs [C5].
- **[C2] REGENIE GitHub repository (master branch).** <https://github.com/rgcgithub/regenie>.
- **[C3] REGENIE LICENSE.** MIT + Boost-licensed BGEN library. <https://raw.githubusercontent.com/rgcgithub/regenie/master/LICENSE>.
- **[C4] REGENIE GitHub Releases — v4.1 (2025-01-27).** <https://github.com/rgcgithub/regenie/releases/tag/v4.1>. Static Linux x86_64 binary.
- **[C5] REGENIE Overview docs.** <https://rgcgithub.github.io/regenie/overview/>. Algorithm description (Level 0 / Level 1 / LOCO); LOCO file format §Output.
- **[C6] REGENIE Install docs.** <https://rgcgithub.github.io/regenie/install/>. GLIBC ≥ 2.22, Boost iostream, MKL, conda recipe.
- **[C7] REGENIE Options docs.** <https://rgcgithub.github.io/regenie/options/>. All flags, default values, output column header description.
- **[C8] REGENIE Recommendations docs.** <https://rgcgithub.github.io/regenie/recommendations/>. `--bsize` 1000 (Step 1) / 200–400 (Step 2).
- **[C9] REGENIE Performance docs.** <https://rgcgithub.github.io/regenie/performance/>. Memory and runtime tables vs BOLT-LMM/SAIGE/fastGWA.
- **[C10] REGENIE source `src/Step2_Models.cpp`.** <https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.cpp>. fastSPA branch (lines 2068–2280); sparse-variant `compute_score_qt` (line 381).
- **[C11] REGENIE source `src/Step2_Models.hpp`.** <https://raw.githubusercontent.com/rgcgithub/regenie/master/src/Step2_Models.hpp>. SPA struct and function signatures.
- **[C12] REGENIE bundled fixture.** <https://github.com/rgcgithub/regenie/tree/master/example>. 500 samples × 1000 variants; QT + binary phenotypes; expected-output oracle file `example.test_bin_out_firth_Y1.regenie`.
- **[C13] REGENIE upstream test suite.** <https://github.com/rgcgithub/regenie/blob/master/test/test_bash.sh>. Reusable parity reference.
- **[C14] Issue #678 — Sparse-variant SE deflation in subset analyses.** <https://github.com/rgcgithub/regenie/issues/678>. Affects v4.1; pin to v4.2+ when available.
- **[C15] Issue #684 — Version-to-version BETA inflation in logistic.** <https://github.com/rgcgithub/regenie/issues/684>. v2 vs v4 BETAs differ ~13× on same data.
- **[C16] Issue #679 — Covariate-scale-driven inflation.** <https://github.com/rgcgithub/regenie/issues/679>. Motivates §2.6 internal z-scoring.
- **[C17] Loesch, D.P. et al. (2022).** *Evaluation of GENESIS, SAIGE, REGENIE and fastGWA-GLMM for genome-wide association studies of binary traits in correlated data.* *Frontiers in Genetics*. <https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.897210/full>. REGENIE over-correction in family-based data.
- **[C18] NIH Biowulf REGENIE app docs.** <https://hpc.nih.gov/apps/regenie.html>. Source of the `--l0`/`--l1` default ridge grid (UNVERIFIED — to confirm at implementation time).
- **[C19] Daniels, H. E. (1954).** *Saddlepoint approximations in statistics.* *Annals of Mathematical Statistics* 25:631–650. Original SPA derivation.
- **[C20] Lugannani, R., Rice, S. (1980).** *Saddle point approximation for the distribution of the sum of independent random variables.* *Advances in Applied Probability* 12:475–490. Lugannani-Rice formula used in our `stats/spa.py`.
- **[C21] Zhou, W. et al. (2018).** *Efficiently controlling for case-control imbalance and sample relatedness in large-scale genetic association studies (SAIGE).* *Nature Genetics* 50:1335–1341. <https://doi.org/10.1038/s41588-018-0184-x>. fastSPA derivation our follow-on phase will mirror.

## 12. Approval & next step

This sub-spec is committed once approved alongside the master and the other three sub-specs. Implementation is DEFERRED until validation campaign Pillars A–D and the three NA tasks (NA1, NA2, NA3) close, AND the X-chromosome refactor track is merged, per master spec §3 + brief R-REG-4. At implementation kickoff, this spec is fed to `superpowers:writing-plans` to produce the per-Tier implementation plan.
