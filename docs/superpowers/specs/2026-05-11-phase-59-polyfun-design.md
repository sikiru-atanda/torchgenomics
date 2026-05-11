# Phase 59 — PolyFun Functional-Annotation–Informed Fine-Mapping (FULL PARITY) — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven)
- **Date:** 2026-05-11
- **Branch:** `modernization/specs`
- **Master spec:** [`2026-05-11-modernization-master-design.md`](2026-05-11-modernization-master-design.md)
- **Research brief:** [`../research/polyfun-research.md`](../research/polyfun-research.md)
- **Phase number:** 59
- **Scope class:** **FULL PARITY** (per master spec §2 — only of the four with this stricter tolerance class)
- **Tier 2 tolerance:** `floor + observed × 1.5` (tighter than the MVPs at × 2)
- **Sequencing constraint:** independent of X-chrom refactor (PolyFun is summary-stats based; no genotype matrix). Can be implemented in parallel with Phase 58 PRS-CSx per master spec §3.
- **Living spec.** Tier A is fixed; Tier B / C may be revised at implementation kickoff.

---

## 1. Goal & scope

Add functional-annotation–informed fine-mapping (PolyFun, Weissbrod et al. 2020 [C2]) to TorchGWAS at full parity. PolyFun layers per-SNP causal-effect priors derived from L2-regularized S-LDSC over the baseline-LF v2.2 functional annotations (187 categories) onto SuSiE / FINEMAP, sharpening credible sets relative to uniform-prior fine-mapping. Reported gains in Weissbrod 2020: **>20% more variants with PIP > 0.95** in pooled simulations; **>32% more PIP > 0.95 fine-mapped variant–trait pairs** in 49 UK Biobank traits ($\bar N$ = 318 K).

**This is the only one of the four modernization items at FULL PARITY scope** per master spec §2 — meaning we implement the entire upstream surface (88 flags across 7 scripts), not just the 80% MVP subset. Justification: PolyFun has a smaller surface than REGENIE/PRS-CSx and is layered on infrastructure we already have (SuSiE in `models/bayesian_vs.py`, S-LDSC in `postgwas/_sldsc.py`, fine-mapping wrappers in `postgwas/_finemapping.py`). Full-parity is achievable without explosive scope.

**In scope (FULL PARITY):**

- **All three SNPVAR approaches** per [C5] Wiki §1 (NOT the "polyfun-imp" / "polyfun-ldsc" prompt naming, which doesn't appear in the paper — see §2.6):
  1. **Pre-computed SNPVAR** — download baseline-LF v2.2.UKB pre-computed SNPVAR for ~19M UKB-imputed SNPs.
  2. **L2-only S-LDSC** — `--compute-h2-L2 --no-partitions`: use eqn (1)→(2) result directly as the prior; no binning step.
  3. **Non-parametric / binned (full pipeline)** — `--compute-h2-L2` followed by `--compute-h2-bins`: the full five-step pipeline. The most robust approach per [C5] but computationally intensive.
- **All `polyfun.py` flags** (24 flags per brief §5.1).
- **All `finemapper.py` flags** (32 flags per brief §5.2), with both `susie` and `finemap` backends.
- **All `aggregate_finemapper_results.py` flags** (7 flags per brief §5.3).
- **All auxiliary scripts**: `munge_polyfun_sumstats.py` (8 flags), `extract_snpvar.py` (5 flags), `compute_ldscores_from_ld.py` (8 flags), `extract_annotations.py` (7 flags), and `ldsc.py` wrapper (per brief §5.4).
- **Per-SNP prior injection into our SuSiE** (`models/bayesian_vs.py`) via D3 backward-compat shim (new `prior_pi_per_snp` keyword preserves existing scalar `prior_pi` API).
- **L2-regularized S-LDSC** added to `postgwas/_sldsc.py` (currently OLS-only).
- **Baseline-LF v2.2 ingestion** via new module `postgwas/_baseline_lf.py` (chunked PyArrow streaming for the 19M × 187 dense annotation matrix per `feedback_streaming`).
- **Fine-mapping driver** via new module `postgwas/_polyfun_finemap.py` orchestrating LD construction → prior loading → SuSiE/FINEMAP fit → credible-set extraction.
- **Custom-annotation layering on top of baseline-LF** via comma-separated `--ref-ld-chr` (per [C20] Wiki FAQ).
- **CLI:** new subcommand `polyfun-finemap` (combines polyfun.py + finemapper.py + aggregate_finemapper_results.py for the common case); plus expose individual scripts as `polyfun-h2-l2`, `polyfun-h2-bins`, `polyfun-finemap-locus`, `polyfun-aggregate`, `polyfun-munge-sumstats`, `polyfun-extract-snpvar`, `polyfun-compute-ldscores-from-ld`, `polyfun-extract-annotations`.
- **Tier 2 parity** vs PolyFun upstream Python on the bundled `example_data/` fixture per [C18] within `floor + observed × 1.5` tolerance per master spec §5.

**Explicitly out of scope (deferred):**

- **PolyPred** (cross-ancestry PRS extension, Weissbrod*, Kanai*, Shi* 2022 [C4]) — separate paper, separate codepath in `polypred.py`. Phase 59 is fine-mapping only.
- **PolyLoc** (polygenic localization to per-region heritability, bundled with polyfun source). Defer to follow-on phase.
- **FINEMAP v1.4(.1) external binary integration.** SuSiE backend is implemented natively (via our existing `models/bayesian_vs.py`); FINEMAP backend requires shelling out to the external binary — implement as Tier C nice-to-have.
- **LDstore 2.0 binary integration.** Required only for BGEN input with on-the-fly LD; defer to Tier C. PLINK BED + pre-computed npz LD are fully supported in MVP.
- **Custom user-built baseline-LF (non-published) annotations.** Users supply their own annotation parquet; the published baseline-LF v2.2.UKB is the canonical reference fixture. Custom annotation construction is documented but not auto-generated by us.
- **R-package dependency for SuSiE.** Upstream PolyFun uses R-susier 0.11.92 [C19]; we use our existing pure-Python `models/bayesian_vs.py` SuSiE. This is a **deliberate divergence on a known-equivalent algorithm**, documented in §2.7 below.

## 2. Algorithm reference

### 2.1 Five-step PolyFun pipeline

Per Weissbrod et al. 2020 [C3] Methods §"Computing per-SNP heritabilities":

**Step 1 — L2-regularized S-LDSC on even chromosomes.** Fit per-annotation coefficients $\hat\tau^{\text{even}}_c$ for $c = 1, \dots, A = 187$ baseline-LF v2.2 annotations using the weighted-LDSC objective (LDSC weights $w_i$ per Bulik-Sullivan et al. 2015 [C28] and Finucane et al. 2015 [C29] correct for heteroskedasticity from per-SNP $\chi^2$ variance scaling with LD):

$$
\min_{\boldsymbol\tau, b}\; \sum_i w_i \Bigl(\chi_i^2 - n \sum_{c=1}^{A} \tau_c \ell(i,c) - n b - 1\Bigr)^2 + \lambda \sum_{c=1}^{A} \tau_c^2
$$

where $\ell(i,c)$ is the per-SNP LD score for annotation $c$, $\chi_i^2$ is the GWAS chi-square at SNP $i$, $b$ is the LDSC intercept, $w_i$ is the LDSC heteroskedasticity weight per [C29] (typically $w_i = 1 / [\ell(i,\text{base}) \cdot (1 + n \ell(i,\text{base}) / M)^2 ]$ where $\ell(i,\text{base})$ is the unpartitioned LD score and $M$ is the genome-wide SNP count), and $\lambda$ is the L2 penalty added by PolyFun on top of standard S-LDSC. Per [C3]: *"To address the first limitation, PolyFun incorporates an L2-regularized extension of S-LDSC."* The weighted normal-equation form $\hat\tau = (X^\top W X + \lambda I)^{-1} X^\top W \chi^2$ in §2.4 below is the closed-form solution to this objective.

**Step 2 — Per-SNP heritability for held-out (odd) chromosomes.** The fitted even-chromosome $\hat\tau^{\text{even}}_c$ are applied to odd-chromosome SNPs (and vice versa, by symmetry):

$$
\widehat{\mathrm{var}}[\beta_i \mid \mathbf a_i] = \sum_{c=1}^{A} \hat\tau^{\text{even}}_c\, a_{ic}
$$

**Per [C3]:** *"PolyFun avoids winner's curse by using different data for partitioning SNPs and for per-bin heritability estimation."* This cross-chromosome split is the explicit winner's-curse mitigation mechanism.

**Step 3 — Partition into B = 20 bins.** SNPs are sorted by $\widehat{\mathrm{var}}[\beta_i \mid \mathbf a_i]$ and partitioned into 20 bins of approximately equal heritability via the **Ckmeans.1d.dp** R package (1-D optimal $k$-medians). When R/Ckmeans unavailable, PolyFun falls back to scikit-learn 1-D K-means via `--skip-Ckmedian`.

**Step 4 — Re-fit S-LDSC on the 20 bins.** S-LDSC is re-fit using the 20 bins as the new annotation set (one $\tau_b$ per bin) to obtain robust per-bin per-SNP heritability $\hat\sigma^2_{r,b}$. This step is non-parametric in $\mathbf a_i$ — only the bin assignment is used.

**Step 5 — Prior causal probability.** Per-SNP prior causal probability is set proportional to per-SNP heritability:

$$
P(\beta_i \neq 0 \mid \mathbf a_i) \propto \hat\sigma^2_{r,b(i)}, \qquad \sum_{i \in \text{locus}} P(\beta_i \neq 0) = 1.0
$$

(Normalized within each fine-mapping locus.)

**Step 6 (in `finemapper.py`) — SuSiE/FINEMAP with per-SNP priors.** SuSiE's per-layer single-effect prior is replaced from uniform $1/p$ to the normalized PolyFun prior, and inference proceeds per Wang et al. 2020 SuSiE [C26] / Benner et al. 2016 FINEMAP [C27].

### 2.2 The L2 penalty value

The paper gives the equation (per §2.1 step 1) but does NOT specify the value of $\lambda$ used in the published analyses. Per brief §11 R59-2 / UNVERIFIED #8: must recover from `polyfun.py` source at implementation time.

**Implementation requirement:** the spec MUST cite the recovered $\lambda$ value at implementation time. If it's a hyperparameter (e.g., chosen by leave-one-out CV on a held-out chromosome), implement that selection. If it's a fixed value, document the source line.

### 2.3 Per-SNP prior injection into SuSiE (D3 decision: backward-compat shim)

Our existing `BayesianVS.fit()` accepts only **scalar** `prior_pi: float = 0.01`. PolyFun requires a **vector** $\boldsymbol\pi \in \mathbb{R}^p$ that replaces uniform $1/p$ in SuSiE's single-effect step.

Per **Decision D3** from brainstorming (2026-05-11): use a backward-compat shim — add a new `prior_pi_per_snp: Optional[Tensor] = None` keyword while preserving scalar `prior_pi: float = 0.01`. Existing call sites that pass scalar `prior_pi` get exactly the same behavior as before; new call sites that pass `prior_pi_per_snp=tensor` get PolyFun-style per-SNP priors.

**Required code changes in `models/bayesian_vs.py`:**

- `fit()` signature gains `prior_pi_per_snp: Optional[Tensor] = None` keyword. Validation: if both `prior_pi` (default 0.01) and `prior_pi_per_snp` are provided non-trivially, raise `ValueError` with explicit message.
- The CAVI path's `log_odds_prior = math.log(pi / (1 - pi))` (line 443 per brief §11 R59-1) becomes per-SNP `log_odds_prior_j = log(pi_j / (1 - pi_j))` when vector prior is provided.
- The SuSiE `_susie_loop` softmax (line 786) becomes `softmax(log_bf + log_prior_per_snp)` when vector prior is provided.
- Normalization: per-SNP prior is normalized within the locus per Step 5 above (sum to 1.0). If user supplies an unnormalized vector, normalize internally with a logged warning.

**New unit test:** per-SNP prior with one extreme prior ($\pi_j = 0.99$ for SNP $j$) should drive PIP toward SNP $j$ even on noisy z-scores. Verifies the prior is actually wired in.

### 2.4 L2-regularized S-LDSC in `postgwas/_sldsc.py`

Per brief §11 R59-2: existing `sldsc_h2_partitioned()` uses `_weighted_lstsq` (ordinary weighted least squares with two-step outlier filter and block jackknife). Add **L2 regularization** option.

**Required code changes:**

- `sldsc_h2_partitioned()` gains `lambda_l2: float = 0.0` parameter. Default `0.0` preserves existing OLS behavior — backward-compat.
- When `lambda_l2 > 0.0`: solve $\hat\tau = (X^\top W X + \lambda I)^{-1} X^\top W \chi^2$ instead of unregularized $\hat\tau = (X^\top W X)^{-1} X^\top W \chi^2$. (Where $X$ is the LD-score design matrix with intercept column, $W$ is the LDSC weight matrix.)
- Optional `nnls: bool = False` flag mirroring upstream `--nnls-exact` for non-negativity-constrained $\hat\tau$.
- Per-SNP h² accessor: new method `predict_per_snp_h2(annot_matrix: Tensor) -> Tensor` returning $\hat\sigma^2_i = \sum_c \hat\tau_c a_{ic}$. Currently absent.

### 2.5 Baseline-LF ingestion (R59-4: new module)

Per brief §11 R59-4: the existing `annotate/` module is for low-volume NCBI REST queries (~kilobytes per query). Baseline-LF v2.2 is a dense $19 \times 10^6 \times 187$ float matrix (~30 GB on disk, ~28 GB float32 if fully materialized). Wrong place.

**New module `torchgwas/postgwas/_baseline_lf.py`** with chunked parquet streaming via PyArrow:

```python
def iter_baseline_lf_chunks(
    annot_dir: Path,
    chrom: int,
    chunk_size: int = 50_000,  # SNPs per chunk
) -> Iterator[Tuple[pd.DataFrame, Tensor]]:
    """Yield (variant_metadata, annotation_matrix) chunks from baseline-LF v2.2.

    Uses pyarrow.parquet.ParquetFile.iter_batches() to bound peak memory
    by chunk_size, not by total variant count (~19M).

    Args:
        annot_dir: directory containing baselineLF.<chr>.annot.parquet files.
        chrom: target chromosome (1-22).
        chunk_size: SNPs per yielded chunk.

    Yields:
        (meta_df, annot_tensor) where annot_tensor has shape (chunk_size, 187).
    """
    ...
```

**Memory regression test** (`tests/test_streaming_memory.py` extension): assert peak RSS < 1 GB during a full-genome baseline-LF iteration with `chunk_size=50_000`, even though full materialization would require ~28 GB.

### 2.6 Terminology disambiguation: the brief's "polyfun-imp" / "polyfun-ldsc"

Per brief §2.6 / UNVERIFIED #1: the prompt's "polyfun-imp" / "polyfun-ldsc" mode names DO NOT appear in the Weissbrod 2020 paper. The wiki distinguishes three approaches per §1 above:
1. **Pre-computed** (download SNPVAR for ~19M UKB-imputed SNPs)
2. **L2-only** (`--compute-h2-L2 --no-partitions`)
3. **Non-parametric / binned** (full five-step pipeline)

Our spec uses the wiki's three-approach naming. The CLI maps:
- Approach 1 → `polyfun-finemap --snpvar-source precomputed --baseline-lf-dir <path>`
- Approach 2 → `polyfun-finemap --snpvar-source l2-ldsc`
- Approach 3 → `polyfun-finemap --snpvar-source non-parametric` (default)

The prompt's labels were a misremembered shorthand from the brainstorming author. No user-facing impact.

### 2.7 Deliberate divergence: pure-Python SuSiE instead of R-susier 0.11.92

Upstream PolyFun calls R-susier 0.11.92 via rpy2 [C19]. We use our existing pure-Python `models/bayesian_vs.py` SuSiE (CAVI + IBSS). This is a **deliberate divergence on a known-equivalent algorithm** documented in `docs/validation_findings.md` at implementation time.

**Justification:**
- No rpy2 dependency for our users (reduces install footprint by ~2 GB and removes R-Python interop friction).
- Our SuSiE is already in production (Phase 17 BayesianVS) and has its own parity tests against the original SuSiE R package — separately validated.
- The algorithm itself (Wang et al. 2020 [C26] IBSS) is identical between implementations; only execution path differs.

**Tier 2 implication:** per-SNP PIPs from our implementation vs upstream-PolyFun-with-R-susier may differ at the 1e-3 to 1e-2 absolute level due to (a) IBSS convergence criteria differences, (b) numerical solver differences (R LAPACK vs torch). Within `floor + observed × 1.5` tolerance — verify at first run.

### 2.8 Dimension table

| Symbol | Meaning | UKB analysis value |
|---|---|---|
| $M$ | total SNPs | ~19 M (UKB imputed, MAF > 0.1%) |
| $A$ | annotations in baseline-LF v2.2 | 187 |
| $B$ | bins for per-SNP $h^2$ partition | 20 |
| $L$ | SuSiE single-effect layers | 10 (default `--max-num-causal`) |
| $K$ | credible sets per locus | ≤ $L$ |
| $n$ | GWAS sample size | 337,491 (unrelated British UKB per [C3]) |
| $\lambda$ | L2 penalty for S-LDSC | UNVERIFIED — recover from source |

## 3. Module layout

### 3.1 New files

| Path | Purpose |
|---|---|
| `torchgwas/postgwas/_baseline_lf.py` | Baseline-LF v2.2 ingestion with chunked PyArrow streaming. `iter_baseline_lf_chunks`, `load_full_baseline_lf` (only call for small fixtures), `load_precomputed_snpvar`. |
| `torchgwas/postgwas/_polyfun_h2.py` | The five-step PolyFun pipeline orchestrator. `compute_h2_l2`, `compute_h2_bins`, `compute_per_snp_snpvar`. |
| `torchgwas/postgwas/_polyfun_finemap.py` | Fine-mapping driver. `polyfun_finemap_locus`, `polyfun_finemap_genome`, `aggregate_polyfun_results`. |
| `torchgwas/postgwas/_polyfun_sumstats.py` | Sumstats munging mirror of `munge_polyfun_sumstats.py`. |
| `torchgwas/postgwas/_polyfun_extract.py` | `extract_snpvar`, `extract_annotations` mirrors. |
| `torchgwas/postgwas/_polyfun_ldscores.py` | `compute_ldscores_from_ld` mirror. |
| `tests/fixtures/polyfun/` | Symlink to PolyFun's bundled `example_data/` fixture (downloaded once via `pytest --setup-only`). |
| `tests/test_polyfun_h2.py` | Tier 1 unit tests on L2 S-LDSC, binning, SNPVAR. |
| `tests/test_polyfun_finemap.py` | Tier 1 unit tests on the fine-mapping driver. |
| `tests/test_polyfun_parity_upstream.py` | Tier 2 parity vs PolyFun upstream Python on bundled fixture. |
| `tests/test_polyfun_per_snp_prior.py` | Tier 1 unit test for the per-SNP prior injection in `bayesian_vs.py`. |
| `tests/test_baseline_lf_streaming.py` | Memory regression test for chunked ingestion. |

### 3.2 Modified files (existing modules extended)

| Path | Change | Backward compat |
|---|---|---|
| `torchgwas/models/bayesian_vs.py` | Add `prior_pi_per_snp: Optional[Tensor] = None` keyword to `BayesianVS.fit()`. Vector prior path in CAVI + SuSiE loops. | YES — scalar `prior_pi` API unchanged; existing tests pass |
| `torchgwas/postgwas/_sldsc.py` | Add `lambda_l2: float = 0.0`, `nnls: bool = False` to `sldsc_h2_partitioned()`. New method `predict_per_snp_h2()`. | YES — `lambda_l2=0.0` default = existing OLS behavior |
| `torchgwas/postgwas/_finemapping.py` | No change to existing `extract_credible_sets`, `annotate_sumstats`, `locus_summary`, `to_coloc_sumstats`. The new `_polyfun_finemap.py` driver imports and reuses them. | YES — no edits |
| `torchgwas/cli.py` | Add 9 subcommands: `polyfun-finemap` (combined), `polyfun-h2-l2`, `polyfun-h2-bins`, `polyfun-finemap-locus`, `polyfun-aggregate`, `polyfun-munge-sumstats`, `polyfun-extract-snpvar`, `polyfun-compute-ldscores-from-ld`, `polyfun-extract-annotations`. | YES — additive |
| `tests/test_streaming_memory.py` | Add memory regression for `polyfun-finemap` and baseline-LF streaming. | YES — additive |
| `bench/native_speedups.py` | Add wall-time gate for `polyfun-finemap`. | YES — additive |
| `docs/cli.md` | Add 9 new subcommand entries. | YES |
| `CLAUDE.md` | Update CLI subcommand count. Update Phase Index with Phase 59 entry. | YES |

### 3.3 Files NOT touched (per brief §11.1)

- `torchgwas/annotate/` — wrong home for baseline-LF ingestion (per R59-4); we use `postgwas/_baseline_lf.py` instead. No edits.
- `torchgwas/pgs/prscs.py`, `torchgwas/pgs/prscsx.py` — separate concern (PRS construction); PolyFun is fine-mapping only. No edits.
- `torchgwas/postgwas/_finemapping.py` (existing utilities) — preserved as-is; new driver imports them.

### 3.4 Public API surface

```python
# torchgwas/postgwas/_polyfun_h2.py
def compute_h2_l2(
    sumstats: pd.DataFrame,                   # Z, N, CHR, BP, SNP, A1, A2
    baseline_lf_dir: Path,                    # baseline-LF v2.2 annotation parquet
    weights_ld_chr: Path,                     # weights LD-score files
    bfile_chr: Optional[Path] = None,         # PLINK reference panel (or use precomputed LD)
    lambda_l2: float = ...,                   # recovered from upstream source
    nnls_exact: bool = False,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Step 1+2 of PolyFun: L2-regularized S-LDSC + per-SNP h^2 imputation.

    Returns per-SNP DataFrame with SNPVAR column.
    """

def compute_h2_bins(
    sumstats_with_snpvar: pd.DataFrame,
    weights_ld_chr: Path,
    num_bins: int = 20,
    skip_ckmedian: bool = False,              # use sklearn KMeans if True
) -> pd.DataFrame:
    """Step 3+4 of PolyFun: 20-bin partition + per-bin S-LDSC re-fit.

    Returns per-SNP DataFrame with updated SNPVAR (bin-based).
    """

# torchgwas/postgwas/_polyfun_finemap.py
def polyfun_finemap_locus(
    sumstats: pd.DataFrame,                   # Z, N, SNP, CHR, BP, A1, A2, SNPVAR
    chr: int, start: int, end: int,
    geno: Optional[Path] = None,              # PLINK BED for in-sample LD
    ld: Optional[Path] = None,                # OR pre-computed LD npz
    method: Literal["susie", "finemap"] = "susie",
    max_num_causal: int = 10,
    susie_max_iter: int = 100,
    non_funct: bool = False,                  # ignore SNPVAR if True (uniform prior)
    ...,                                      # all 32 finemapper.py flags
) -> pd.DataFrame:
    """Step 6 of PolyFun: SuSiE/FINEMAP with per-SNP PolyFun priors over a single locus."""

# torchgwas/postgwas/_baseline_lf.py
def iter_baseline_lf_chunks(...):  # see §2.5
def load_precomputed_snpvar(snpvar_dir: Path, chrom: int) -> pd.DataFrame: ...

# torchgwas/postgwas/_sldsc.py (existing module, extended)
def sldsc_h2_partitioned(
    chi2: Tensor,
    ldscores: Tensor,
    weights: Tensor,
    annot_matrix: Tensor,
    n: int,
    *,
    lambda_l2: float = 0.0,                   # NEW (default = existing OLS)
    nnls: bool = False,                        # NEW
    block_jackknife: bool = True,
) -> SLDSCResult: ...

def predict_per_snp_h2(                        # NEW method
    self,
    annot_matrix: Tensor,
) -> Tensor: ...

# torchgwas/models/bayesian_vs.py (existing module, extended via D3 shim)
class BayesianVS:
    def fit(
        self,
        sumstats: ...,
        ld: ...,
        *,
        prior_pi: float = 0.01,                # EXISTING (unchanged behavior)
        prior_pi_per_snp: Optional[Tensor] = None,  # NEW (D3 shim)
        prior_sig2_beta: float = 0.1,
        ...,
    ) -> BayesianVSResult: ...
```

## 4. Data flow

### 4.1 End-to-end for the non-parametric (full) approach

```
Per-trait sumstats (CHR, BP, SNP, A1, A2, BETA, SE, [P], N)
   ↓ (polyfun_munge_sumstats: filter MAF, INFO, strand-ambig, HLA, etc.)
Munged sumstats (with Z column)
   ↓ + baseline-LF v2.2.UKB.polyfun.tar.gz (chunked PyArrow ingestion)
   ↓ (compute_h2_l2: L2 S-LDSC even-vs-odd cross-fit per §2.1)
Sumstats with SNPVAR column (per-SNP h^2 from L2 step)
   ↓ (compute_h2_bins: 20-bin partition + per-bin S-LDSC re-fit)
Sumstats with refined SNPVAR column
   ↓ + LD source (in-sample BED, or pre-computed npz)
   ↓ (polyfun_finemap_locus per region; loops over loci)
Per-locus outputs: PIP, BETA_MEAN, BETA_SD, CREDIBLE_SET
   ↓ (aggregate_polyfun_results: concatenate over regions)
Genome-wide fine-mapping table (~10K SNPs in credible sets per published UKB analysis)
```

### 4.2 End-to-end for the L2-only approach

```
Per-trait sumstats
   ↓ (compute_h2_l2 with no_partitions=True)
SNPVAR column (no binning step)
   ↓ + LD source
   ↓ (polyfun_finemap_locus per region)
Per-locus PIP outputs
```

### 4.3 End-to-end for the pre-computed approach

```
Per-trait sumstats
   ↓ + pre-computed UKB SNPVAR file (one of 22 chr files, ~30 GB total)
   ↓ (load_precomputed_snpvar per chromosome; join on CHR + BP + A1 + A2)
Sumstats with SNPVAR column (no S-LDSC needed)
   ↓ + LD source
   ↓ (polyfun_finemap_locus per region)
Per-locus PIP outputs
```

### 4.4 Memory footprint

- **Baseline-LF iteration:** chunked at 50K SNPs × 187 annotations × float32 = 38 MB per chunk. Peak RSS bounded by ~1 GB even on full-genome runs (per §2.5 streaming).
- **L2 S-LDSC fit:** per-chromosome LD-score regression on ~1M SNPs × 187 annotations = ~750 MB float32 design matrix. Within RAM budget.
- **Fine-mapping per locus:** ~10K SNPs × ~10K SNPs LD matrix = ~800 MB float64. Within budget.
- **Pre-computed SNPVAR ingestion:** chunked Parquet read; bounded by chunk size, not total file size.

### 4.5 Streaming compliance

Per `feedback_streaming` and brief R59-5:
- PolyFun is summary-stats based; no genotype `iter_chunks` integration needed at the fine-mapping driver level.
- Baseline-LF ingestion uses chunked PyArrow streaming per §2.5 — covered by `tests/test_streaming_memory.py` extension (test name `test_baseline_lf_streaming_peak_rss`).
- The S-LDSC genome-wide chi² regression in `postgwas/_sldsc.py` is already chunked (existing).

## 5. Validation gates

### 5.1 Tier 1 — math correctness (synthetic, deterministic)

Required tests:

**`tests/test_polyfun_h2.py`:**
- **L2 S-LDSC analytical check.** $50 \times 5$ design matrix (50 LD scores, 5 annotations) with hand-computed $\hat\tau = (X^\top W X + \lambda I)^{-1} X^\top W \chi^2$. Compare to `torch.linalg.solve` with the same regularizer to ≤ 1e-10 absolute.
- **Even-odd cross-fit.** Hand-crafted 2-chromosome fixture with known $\hat\tau^{\text{even}}$ and $\hat\tau^{\text{odd}}$; assert per-SNP SNPVAR uses the cross-chromosome $\hat\tau$.
- **20-bin partition reproducibility.** With `--skip-Ckmedian` (sklearn KMeans + fixed `random_state`), assert exact bin assignment matches a checked-in expected file for the bundled fixture.
- **Per-bin S-LDSC re-fit.** Closed-form check on a 20-bin × 100-SNP fixture.
- **Step-5 prior normalization.** Hand-coded SNPVAR vector for 100 SNPs in a locus; assert `polyfun_finemap_locus` normalizes to `sum(pi) == 1.0` within 1e-10.

**`tests/test_polyfun_finemap.py`:**
- **Uniform-prior reduction.** With `--non-funct True` (PolyFun's `--non-funct` flag), assert PolyFun + SuSiE produces identical PIPs to vanilla SuSiE on the same fixture.
- **Extreme-prior bias.** Set `prior_pi_per_snp` to all-zero except SNP $j$ = 0.99; assert `BayesianVS.fit()` returns PIP ~ 0.99 for SNP $j$.
- **Backward-compat scalar.** Call `BayesianVS.fit(prior_pi=0.01)` (no `prior_pi_per_snp`); assert byte-equal output to pre-refactor.

**`tests/test_polyfun_per_snp_prior.py`:**
- **D3 shim correctness.** Direct test of `bayesian_vs.py` per-SNP prior injection. Includes the validation that raises `ValueError` if both `prior_pi != 0.01` AND `prior_pi_per_snp is not None` are passed.

**`tests/test_baseline_lf_streaming.py`:**
- **Memory ceiling.** Iterate full baseline-LF for chr 22 (smallest); assert peak RSS < 1 GB.
- **Chunk completeness.** Concatenate yielded chunks; assert variant count matches `parquet.read_metadata().num_rows`.

### 5.2 Tier 2 — smoke parity vs PolyFun upstream Python

Required test in `tests/test_polyfun_parity_upstream.py`:

- **Fixture**: PolyFun's bundled `example_data/` directory per [C18] — small smoke set: bundled annotations.{1..22}.parquet, weights.*.l2.*, reference.{1..14}.{bed,bim,fam}, RBC.sumstats.small.parquet, chr1.finemap_sumstats.txt.gz, posterior_betas.{gz,parquet}. ~100 MB.
- **Pre-flight install check**: PolyFun installable via `git clone https://github.com/omerwe/polyfun` + `conda env create -f polyfun.yml`. **Footprint check**: ≥ 35 GB free for code + baseline-LF v2.2.UKB.polyfun bundle (~30 GB) + smoke fixture. **Memory headroom check** per `feedback_preflight`.
- **Upstream pin**: `master` branch as of 2026-05-11 (PolyFun has no tagged releases per [C1]). Pin to specific commit SHA in `tests/conftest.py::POLYFUN_PINNED_SHA`.
- **Force `--skip-Ckmedian`** for byte-deterministic comparison (per brief §11 R59-7 — Ckmeans vs sklearn KMeans produces different bin assignments for borderline SNPs).
- **Three-step parity test** (per brief §8):
  1. **L2 S-LDSC parity:** run upstream `polyfun.py --compute-h2-L2 --no-partitions ...` and our `torchgwas polyfun-h2-l2 ...` on the same sumstats. Compare per-SNP SNPVAR.
  2. **Binning parity:** run upstream `polyfun.py --compute-h2-bins ...` and our `torchgwas polyfun-h2-bins ...`. Compare bin assignments + per-bin h².
  3. **Fine-mapping parity:** run upstream `finemapper.py --method susie ...` and our `torchgwas polyfun-finemap-locus --method susie ...`. Compare PIPs + credible-set membership.
- **Tolerance** (per master spec §5; FULL PARITY scope): `floor + observed × 1.5`. Floor: 1e-6 absolute on SNPVAR; 1e-3 absolute on PIP.
- **Expected sources of divergence** to verify against tolerance:
  - L2 penalty $\lambda$ choice — must recover exact upstream value at implementation time per §2.2.
  - Pure-Python SuSiE vs R-susier (deliberate divergence per §2.7) — expect 1e-3 to 1e-2 absolute PIP differences. Within tolerance.
  - Numerical solver differences (torch LAPACK vs R LAPACK) — within tolerance.
- **Failure mode**: any divergence outside tolerance → log to `docs/validation_findings.md`. Phase does NOT block on documented divergences.

### 5.3 Tier 2 — uniform-prior reduction parity (existing SuSiE check)

Run our `polyfun-finemap-locus --non-funct` (PolyFun's `--non-funct` flag) AND existing `BayesianVS.fit(prior_pi=0.01)` on the same fixture; assert per-SNP PIP agrees within 1e-4 absolute. Validates the per-SNP prior injection nests the existing scalar-prior case.

### 5.4 Tier 3 — full-scale empirical (deferred per NA3)

Documented assertion (not run):
- Reproduce Weissbrod 2020 [C3] published result on 49 UKB traits: ≥ 32% more PIP > 0.95 fine-mapped variant–trait pairs vs uniform-prior SuSiE.
- Per-SNP SNPVAR agreement with published baseline-LF v2.2.UKB pre-computed file: ≤ `floor + observed × 1.5` Pearson R > 0.999.

Deferred to NA3.

### 5.5 Per-SNP-prior injection regression net

Required test in `tests/test_polyfun_per_snp_prior.py`:
- All existing `tests/test_bayesian_vs.py` and `tests/test_susie*.py` tests MUST continue to pass with no modifications. Verifies the D3 shim is non-disruptive.

## 6. CLI integration

### 6.1 New subcommands

```bash
# Combined end-to-end (most common case)
torchgwas polyfun-finemap \
    --sumstats data/trait.sumstats.parquet \
    --baseline-lf-dir data/baselineLF_v2.2.UKB \
    --weights-ld-chr data/weights. \
    --bfile-chr data/reference. \
    --geno data/target \           # for in-sample LD
    --regions-file data/regions.txt \
    --snpvar-source non-parametric \  # or l2-ldsc / precomputed
    --num-bins 20 \
    --max-num-causal 10 \
    --method susie \
    --skip-Ckmedian \              # use sklearn instead of R Ckmeans
    --out out/finemap

# Individual scripts (for users who want to compose pipeline manually)
torchgwas polyfun-h2-l2 ...
torchgwas polyfun-h2-bins ...
torchgwas polyfun-finemap-locus ...
torchgwas polyfun-aggregate ...
torchgwas polyfun-munge-sumstats ...
torchgwas polyfun-extract-snpvar ...
torchgwas polyfun-compute-ldscores-from-ld ...
torchgwas polyfun-extract-annotations ...
```

### 6.2 Flag mapping (ours → PolyFun upstream)

For brevity, only the non-trivial mappings are listed here. The full 88-flag table (every upstream flag mapped 1:1 to a kebab-case equivalent in our CLI) is **a Tier A deliverable** (see §10.1 step A8.5) and will be appended to this spec at implementation time. Until then, the contract is: any flag in upstream PolyFun's argparse must have a mapped equivalent in our CLI; an UNMAPPED upstream flag is a Tier A blocker.

| Ours (combined `polyfun-finemap`) | Upstream | Notes |
|---|---|---|
| `--snpvar-source {precomputed, l2-ldsc, non-parametric}` | implicit (combination of `--compute-h2-L2`, `--no-partitions`, `--compute-h2-bins`) | We expose explicitly to disambiguate |
| `--baseline-lf-dir` | `--ref-ld-chr` (with baseline-LF prefix) | We use a clearer flag name |
| `--weights-ld-chr` | `--w-ld-chr` | identical |
| `--bfile-chr` | `--bfile-chr` | identical |
| `--geno` | `--geno` | identical |
| `--ld` | `--ld` | identical (pre-computed npz / BCOR) |
| `--regions-file` | `--regions-file` (in `aggregate_finemapper_results.py`) | identical; default is bundled UKB 2,763-region file |
| `--method {susie, finemap}` | `--method` | identical (`finemap` deferred to Tier C) |
| `--max-num-causal` | `--max-num-causal` | identical |
| `--non-funct` | `--non-funct` | identical (forces uniform prior) |
| `--skip-Ckmedian` | `--skip-Ckmedian` | identical |
| `--num-bins` | `--num-bins` | identical (default 20 in paper) |
| `--allow-missing` | `--allow-missing` | identical |
| `--memory` | `--memory` | identical (LDstore memory; Tier C) |
| `--threads` | `--threads` | identical |
| `--out` | `--out` | identical |

### 6.3 Output format

Per [C7] and [C8]:

**Per-locus output:**
```
SNP   CHR  BP   A1  A2  SNPVAR    Z      N    P    PIP    BETA_MEAN  BETA_SD  CREDIBLE_SET
```

**Genome-wide aggregated output (after `polyfun-aggregate`):**
Same columns, concatenated across all regions and chromosomes.

Tab-separated `.gz` (default) or `.txt` (with `--no-gz`). Compatible with downstream consumers expecting PolyFun output.

### 6.4 Streaming compliance

Per `feedback_streaming` and brief R59-5:
- `polyfun-finemap` is summary-stats based; no genotype `iter_chunks` integration needed.
- Baseline-LF loading uses chunked PyArrow streaming per §2.5; verified in `tests/test_streaming_memory.py`.
- S-LDSC chi² regression already chunked.

## 7. Native acceleration scope (deferred)

Per `Python-as-spec, native-as-shortcut` convention. Pure-torch path is the spec.

Candidate hot loops for a future Phase-41-style sub-phase:

- **L2 S-LDSC linear solve.** Currently `torch.linalg.solve` on ~$M \times A$ system; vectorizable; native unlikely to outperform existing torch.
- **20-bin partition (sklearn KMeans).** `--skip-Ckmedian` path is already fast (1-D problem); no native needed.
- **Per-locus LD matrix construction from BED.** Already covered by existing `torchgwas.linalg` paths.
- **SuSiE IBSS inner loop.** Existing optimization in `models/bayesian_vs.py`; native dispatch already considered there.

Speedup targets not committed; deferred to native sub-phase if profiling identifies a hot loop.

## 8. F3 severity application

Per master spec §7 — note that **PolyFun is FULL PARITY scope** so the tolerance is tighter than the MVPs:

| Event | Action |
|---|---|
| Tier 1 test failure | Halt; fix the bug |
| Tier 2 SNPVAR divergence within `floor + observed × 1.5` | Pass; no findings entry |
| Tier 2 SNPVAR divergence outside tolerance | Document in `docs/validation_findings.md`; investigate (likely L2 $\lambda$ value mismatch or LD computation difference); does not block |
| Tier 2 PIP divergence outside tolerance | Document; investigate (likely SuSiE convergence criteria difference per §2.7); does not block |
| Tier 2 uniform-prior reduction (Tier 2 §5.3) divergence > 1e-4 | **Halt** — this means the per-SNP prior injection is broken |
| Per-SNP-prior injection regression net (§5.5) failure | **Halt** — D3 shim is breaking existing API |
| Baseline-LF streaming memory ceiling > 1 GB | **Halt**; investigate |
| ≥ 3 unrelated divergences during Tier B | C4 emergency stop |

## 9. Risks & open questions

### 9.1 Risks (from brief §11)

| ID | Risk | Mitigation |
|---|---|---|
| R-P59-1 | `bayesian_vs.py` D3 shim breaks existing tests | Tier A test §5.5 — full existing test suite runs unchanged; if any regressed, halt |
| R-P59-2 | L2 $\lambda$ value (UNVERIFIED §2.2) recovered incorrectly from upstream source | Read `polyfun.py` source carefully; if value depends on data (e.g., CV-selected), implement that; if hardcoded, document the line |
| R-P59-3 | Pure-Python SuSiE vs R-susier divergence exceeds tolerance per §2.7 | Tolerance set to `floor + observed × 1.5` accounts for this; if exceeded, port R-susier convergence criteria into our SuSiE |
| R-P59-4 | Baseline-LF S3 download fails (per brief §10 #4 — primary alkesgroup URL is dead) | Use AWS S3 mirror exclusively; document in spec; pre-flight check via `aws s3 ls s3://broad-alkesgroup-ukbb-ld/` |
| R-P59-5 | Ckmeans vs sklearn bin assignment divergence on borderline SNPs (R59-7) | Force `--skip-Ckmedian` in parity tests |
| R-P59-6 | UKB LD matrix download is multi-TB (UNVERIFIED ~1.5–2 TB) | Default to in-sample LD via `--geno`; UKB LD opt-in only with pre-flight disk check per `feedback_preflight` |
| R-P59-7 | Single-chromosome sumstats break the even-odd cross-fit (UNVERIFIED #7) | Detect at runtime; raise `ValueError` if sumstats has < 2 chromosomes; suggest pre-computed SNPVAR mode |
| R-P59-8 | Pandas version sensitivity (per brief §10 #2) | Pin pandas ≥ 0.25.0 in our requirements; document |
| R-P59-9 | Rpy2 / R-susier dependency in upstream complicates parity environment setup | Mitigated by §2.7 — we use pure-Python SuSiE; upstream still requires R for parity tests; document in `tests/conftest.py` |
| R-P59-10 | Strand-ambiguous SNPs handled differently than upstream (per brief §10 #8) | Default to removing strand-ambig per modern best practice; expose `--keep-strand-ambig` flag for parity testing |
| R-P59-11 | HLA region included by default upstream is opt-in (`--keep-hla`) | Mirror upstream default (HLA removed); document in user guide |
| R-P59-12 | SuSiE convergence in dense LD blocks differs (`--susie-max-iter` 100 upstream vs 1000 in our `bayesian_vs.py`) | Preserve our higher default; expose flag for parity |

### 9.2 Open questions (deferred to implementation kickoff)

- **OQ-P59-1**: For the pre-computed SNPVAR mode, do we mirror the published 30 GB bundle or expose a per-trait alternative? Default lean: mirror the published bundle (it's trait-meta-analysis-derived for general use); per-trait alternative is the L2-only or non-parametric mode.
- **OQ-P59-2**: Should we provide an opt-in CLI to download the baseline-LF bundle, or require user manual setup? Default lean: opt-in CLI `torchgwas polyfun-download-baseline-lf --version v2.2 --target-dir <path>` with pre-flight disk check.
- **OQ-P59-3**: FINEMAP backend (vs SuSiE) — implement in Tier C or skip entirely? Default lean: Tier C, requires shelling out to FINEMAP v1.4(.1) external binary; lower priority because SuSiE is the more popular default.
- **OQ-P59-4**: BGEN input + on-the-fly LD via LDstore 2.0 — Tier C? Default lean: yes, Tier C. PLINK BED + pre-computed npz LD cover the MVP.
- **OQ-P59-5**: Custom user annotations layered on baseline-LF (per brief §10 #3 / Wiki FAQ comma-separated `--ref-ld-chr`) — explicitly support? Default lean: yes; the existing comma-separated upstream flag is honored verbatim.
- **OQ-P59-6**: Should the L2 $\lambda$ be exposed as a user flag with the upstream-recovered value as default, or hidden? Default lean: expose as `--lambda-l2 FLOAT` with default = recovered upstream value; documented in flag help.

## 10. Implementation phasing

### 10.1 Tier A — minimum viable shipping unit

Shippable when: `polyfun-finemap` runs end-to-end on the bundled `example_data/` fixture for the L2-only approach, produces output matching upstream within tolerance, per-SNP prior injection works in `bayesian_vs.py`.

| Step | Files | Test |
|---|---|---|
| **A0** Pre-emptively log §2.7 deliberate divergence (pure-Python SuSiE vs upstream R-susier) in `docs/validation_findings.md` BEFORE Tier 2 runs, per `feedback_scientific_rigor` | `docs/validation_findings.md` | n/a (documentation entry) |
| **A1** D3 shim — `prior_pi_per_snp` keyword in `bayesian_vs.py` | `models/bayesian_vs.py` | `tests/test_polyfun_per_snp_prior.py` |
| **A2** L2 regularization in `_sldsc.py` | `postgwas/_sldsc.py` | `tests/test_polyfun_h2.py` L2 closed-form |
| **A3** `predict_per_snp_h2` accessor | `postgwas/_sldsc.py` | unit test |
| **A4** Baseline-LF chunked PyArrow ingestion | `postgwas/_baseline_lf.py` | `tests/test_baseline_lf_streaming.py` |
| **A5** L2-only `compute_h2_l2` (Step 1+2) | `postgwas/_polyfun_h2.py` | `tests/test_polyfun_h2.py` |
| **A6** Per-SNP SNPVAR computation (Step 5 normalization) | `postgwas/_polyfun_finemap.py` | `tests/test_polyfun_finemap.py` |
| **A7** `polyfun_finemap_locus` driver (SuSiE backend only) | `postgwas/_polyfun_finemap.py` | `tests/test_polyfun_finemap.py` |
| **A8** CLI subcommand `polyfun-finemap` (combined) + `polyfun-h2-l2` + `polyfun-finemap-locus` + `polyfun-munge-sumstats` | `cli.py` | `tests/test_cli.py` smoke |
| **A8.5** Full 88-flag mapping table appended to spec §6.2 (per E2 reviewer finding) — every upstream flag from `polyfun.py` (24), `finemapper.py` (32), `aggregate_finemapper_results.py` (7), `munge_polyfun_sumstats.py` (8), `extract_snpvar.py` (5), `compute_ldscores_from_ld.py` (8), `extract_annotations.py` (7) mapped 1:1 to kebab-case equivalents | `docs/superpowers/specs/2026-05-11-phase-59-polyfun-design.md` (this spec, §6.2 appendix) + `cli.py` exhaustive flag wiring | `tests/test_cli.py::test_polyfun_full_flag_surface` — programmatically diff our argparse against upstream's argparse to assert zero unmapped flags |
| **A9** Tier 2 parity vs upstream (L2-only approach) on bundled fixture | `tests/test_polyfun_parity_upstream.py` | this test |
| **A10** Uniform-prior reduction parity (`--non-funct` matches existing SuSiE) | `tests/test_polyfun_finemap.py` | this test |
| **A11** Backward-compat regression net (existing `bayesian_vs.py` tests pass) | full test suite | implicit |
| **A12** Memory regression test for baseline-LF streaming | `tests/test_streaming_memory.py` extension | this test |
| **A13** Wall-time regression net | `bench/native_speedups.py` extension | this test |

### 10.2 Tier B — parity-tightening + scaling

Shippable when: full non-parametric (binned) approach works, pre-computed SNPVAR mode works, full Tier 2 parity passes across all three approaches, custom annotation layering supported.

| Step | Files | Test |
|---|---|---|
| **B1** `compute_h2_bins` (Step 3+4) with `--skip-Ckmedian` | `postgwas/_polyfun_h2.py` | `tests/test_polyfun_h2.py` binning test |
| **B2** Pre-computed SNPVAR loader | `postgwas/_baseline_lf.py` | `tests/test_polyfun_finemap.py` |
| **B3** Tier 2 parity for all three approaches | `tests/test_polyfun_parity_upstream.py` extension | this test |
| **B4** `aggregate_polyfun_results` | `postgwas/_polyfun_finemap.py` | `tests/test_polyfun_finemap.py` aggregate test |
| **B5** Custom annotation layering on top of baseline-LF | `postgwas/_polyfun_h2.py` | `tests/test_polyfun_h2.py` extended |
| **B6** All auxiliary script CLI subcommands | `cli.py` extension | `tests/test_cli.py` |
| **B7** Genome-wide regions-file driver (1000G regions or bundled UKB 2763-region file) | `postgwas/_polyfun_finemap.py` extension | regression test on full chr22 |
| **B8** Optional `--ld-ukb` + AWS S3 download utility | new `postgwas/_polyfun_ld_ukb.py` | smoke test |
| **B9** Ckmeans.1d.dp R-package backend (rpy2) | `postgwas/_polyfun_h2.py` extension | parity test with `--use-Ckmedian` |

### 10.3 Tier C — nice-to-haves & follow-on

| Step | Files | Test |
|---|---|---|
| **C1** FINEMAP v1.4(.1) external-binary backend | `postgwas/_polyfun_finemap.py` extension | parity vs FINEMAP backend |
| **C2** LDstore 2.0 + BGEN input integration | `postgwas/_polyfun_ldstore.py` | smoke test |
| **C3** PolyPred (cross-ancestry PRS) integration with `pgs/prscsx.py` | new `pgs/polypred.py` | parity vs upstream |
| **C4** PolyLoc (polygenic localization) | new `postgwas/_polyloc.py` | parity vs upstream |
| **C5** Native acceleration of L2 S-LDSC solver | `csrc/sldsc_l2.cpp` + dispatcher | parity + bench |
| **C6** Multi-trait fine-mapping with shared functional priors | `postgwas/_polyfun_multi_trait.py` | unit test |

## 11. Citations

- **[C1] omerwe/polyfun GitHub repository** (root README + license + script list). <https://github.com/omerwe/polyfun>. MIT LICENSE.
- **[C2] Weissbrod, O. et al. (2020).** *Functionally informed fine-mapping and polygenic localization of complex trait heritability.* *Nature Genetics* 52:1355–1363. <https://www.nature.com/articles/s41588-020-00735-5>. DOI: 10.1038/s41588-020-00735-5.
- **[C3] Same paper, PMC mirror (open access).** <https://pmc.ncbi.nlm.nih.gov/articles/PMC7710571/>. Methods §"Computing per-SNP heritabilities", Results, Figures 2–3.
- **[C4] Weissbrod*, Kanai*, Shi* et al. (2022).** *Leveraging fine-mapping and multipopulation training data to improve cross-population polygenic risk scores* (PolyPred). *Nature Genetics* 54:450–458. <https://www.nature.com/articles/s41588-022-01036-9>. DOI: 10.1038/s41588-022-01036-9. **Cited only for context; PolyPred deferred per scope §1.**
- **[C5] PolyFun Wiki §1 — Computing prior causal probabilities with PolyFun.** <https://github.com/omerwe/polyfun/wiki/1.-Computing-prior-causal-probabilities-with-PolyFun>. Three-approach descriptions, polyfun.py examples, output column table.
- **[C6] `polyfun.py` source (argparse).** <https://github.com/omerwe/polyfun/blob/master/polyfun.py>. 24 flags.
- **[C7] PolyFun Wiki §3 — Functionally informed fine mapping with finemapper.** <https://github.com/omerwe/polyfun/wiki/3.-Functionally-informed-fine-mapping-with-finemapper>. 32 flags + examples.
- **[C8] `aggregate_finemapper_results.py` source (argparse).** <https://github.com/omerwe/polyfun/blob/master/aggregate_finemapper_results.py>.
- **[C9] PolyFun Wiki §2 — Using and creating functional annotations.** <https://github.com/omerwe/polyfun/wiki/2.-Using-and-creating-functional-annotations>. baseline-LF download URL & 30 GB warning.
- **[C10] UK Biobank Linkage Disequilibrium Matrices on AWS Open Data.** <https://registry.opendata.aws/ukbb-ld/>. S3 ARN, `--no-sign-request` access.
- **[C11] PolyFun Wiki §5 — Estimating functional enrichment using S-LDSC.** <https://github.com/omerwe/polyfun/wiki/5.-Estimating-functional-enrichment-using-S-LDSC>.
- **[C12] `munge_polyfun_sumstats.py` source.** <https://github.com/omerwe/polyfun/blob/master/munge_polyfun_sumstats.py>.
- **[C13] `extract_snpvar.py` source.** <https://github.com/omerwe/polyfun/blob/master/extract_snpvar.py>.
- **[C14] `compute_ldscores_from_ld.py` source.** <https://github.com/omerwe/polyfun/blob/master/compute_ldscores_from_ld.py>.
- **[C15] `extract_annotations.py` source.** <https://github.com/omerwe/polyfun/blob/master/extract_annotations.py>.
- **[C16] omerwe/polyfun Issue #144** — Pre-computed LD matrix link dead. <https://github.com/omerwe/polyfun/issues/144>.
- **[C17] PolyFun Wiki home.** <https://github.com/omerwe/polyfun/wiki>.
- **[C18] omerwe/polyfun `example_data/` directory listing.** <https://github.com/omerwe/polyfun/tree/master/example_data>.
- **[C19] `polyfun.yml` conda environment.** <https://github.com/omerwe/polyfun/blob/master/polyfun.yml>. Exact version pins (Python 3.8, r-susier 0.11.92, pyarrow ≥ 3.0).
- **[C20] PolyFun Wiki §7 — FAQ.** <https://github.com/omerwe/polyfun/wiki/7.-FAQ>. `--allow-missing`, pandas version, comma-sep `--ref-ld-chr`.
- **[C21] Wang et al. (2025).** *Funmap: integrating high-dimensional functional annotations to improve fine-mapping.* *Bioinformatics* 41:btaf017. <https://academic.oup.com/bioinformatics/article/41/1/btaf017/7952015>. Replication improvement 15.5%–26.2%.
- **[C22] omerwe/polyfun GitHub Issues page.** <https://github.com/omerwe/polyfun/issues>. Issues #213, #209, #187, #144.
- **[C23] omerwe/polyfun Issue #17** — LD directly from alkesgroup server. <https://github.com/omerwe/polyfun/issues/17>.
- **[C24] TorchGWAS `models/bayesian_vs.py`** — current `BayesianVS.fit` signature with scalar `prior_pi`. Repo file.
- **[C25] TorchGWAS `postgwas/_sldsc.py`** — current OLS S-LDSC. Repo file.
- **[C26] Wang, G. et al. (2020).** *A simple new approach to variable selection in regression, with application to genetic fine mapping.* *Journal of the Royal Statistical Society Series B* 82:1273–1300. <https://doi.org/10.1111/rssb.12388>. SuSiE / IBSS algorithm.
- **[C27] Benner, C. et al. (2016).** *FINEMAP: efficient variable selection using summary data from genome-wide association studies.* *Bioinformatics* 32:1493–1501. <https://doi.org/10.1093/bioinformatics/btw018>. FINEMAP backend reference.
- **[C28] Bulik-Sullivan, B. K. et al. (2015).** *LD Score regression distinguishes confounding from polygenicity in genome-wide association studies.* *Nature Genetics* 47:291–295. <https://doi.org/10.1038/ng.3211>. Original LDSC weighting derivation.
- **[C29] Finucane, H. K. et al. (2015).** *Partitioning heritability by functional annotation using genome-wide association summary statistics.* *Nature Genetics* 47:1228–1235. <https://doi.org/10.1038/ng.3404>. S-LDSC partitioning + heteroskedasticity weight formula used in §2.1 step 1.

## 12. Approval & next step

This sub-spec is committed once approved alongside the master and the other three sub-specs. Implementation is DEFERRED until validation campaign Pillars A–D and the three NA tasks (NA1, NA2, NA3) close per master spec §3. Independent of X-chrom refactor sequencing (PolyFun is summary-stats based). Can be implemented in parallel with Phase 58 PRS-CSx per master spec §3, subject to `feedback_parallel_agents_no_shortcuts` constraint.

At implementation kickoff, this spec is fed to `superpowers:writing-plans` to produce the per-Tier implementation plan.
