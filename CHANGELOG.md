# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

Surface audit + literature-gap closure (2026-06-17). Empirical re-verification
of every claim in the project documentation, plus three feature additions
motivated by the Pandit et al. 2026 *Theoretical and Applied Genetics* paper
on barley leaf rust resistance (breeding-program multi-environment haplotype
workflow). 8 commits on `fix/v040-ci-cleanup` (origin PR #19 / pulsesmartlab PR #18).

### Added

- **`torchgenomics.api.lgebv()` + `LGEBVResult`** — Local GEBV per haplo-block via
  Endelman 2011 rrBLUP. Consumes pre-computed BLUEs (the function does NOT
  compute them); pairs naturally with externally produced phenotypic-analysis
  output (ASReml-R, `sommer`, etc.). Returns per-block effect, variance, and
  favorable/unfavorable sign. Composes with both `LDBlock` (from `detect_blocks`)
  and `HaplotypeBlock` (from `HaplotypeGWAS.construct_haplotypes`) inputs via
  duck-typing on `.variant_indices`. Also registered as the 14th MCP tool
  `tg_lgebv` (category `pgs`). 10 regression tests.
- **`torchgenomics.postgwas.iclass()` + `IClassResult`** — G×E classification on
  factor-analytic loadings (Smith et al. 2015 / 2021). Takes the rotated FA
  loading matrix from `met-scan --vg-structure "fa(k)"` (via
  `MultiEnvLMM.fa_loadings(null_fit)`) and clusters environments by polarity
  pattern ("PNN", "PNP", "PPN", etc.). Optional `within_cluster_correlation()`
  helper. 8 regression tests.
- **`detect_blocks_r2(..., tolerance=N)` + CLI `--tolerance` flag** —
  SelectionTools-style tolerance parameter. With `tolerance > 0`, the adjacent
  r²-pair walker now allows up to `N` consecutive below-threshold pairs before
  closing a block (the convention used in plant-breeding software). Default
  `tolerance=0` keeps legacy behaviour bit-identical. 6 regression tests.

### Fixed

- **`ld.compute_pairwise_ld` polyploid MAF filter** — hardcoded `af = G.mean / 2.0`
  assumed diploid. For tetraploid+ data with dosage in `[0, ploidy]`, common
  SNPs landed with AF > 1 and got silently dropped by the MAF filter,
  collapsing block detection (`r2`, `gabriel`, `spine`) to zero blocks on
  28 of 50 random seeds. Fix plumbs `ploidy` through `detect_blocks →
  compute_pairwise_ld`. 4 regression tests (`TestPloidyAwareMAFFilter`).
- **`ld._blocks_literature.detect_blocks_cc_graph` polyploid tag-SNP MAF** — same
  hardcode for the tag-SNP selection inside `cc_graph`. Polyploid users got
  nonsensical reported `metadata["tag_snp_maf"]` (often > 0.5 or negative).
  Fix at the same shape — `ploidy` argument, threaded through dispatcher.
  3 regression tests.
- **`api.pgs_fit` device-mismatch crash on CUDA** — `_harmonize` built `idx` on CPU
  via `torch.tensor(...)`, but `_ld_reference_subset` used it to index
  `ld_ref.block_index` after `load_ld_reference(device="cuda")` moved that
  tensor onto the GPU. PyTorch indexing is asymmetric (CUDA-indexed-by-CPU
  works; CPU-indexed-by-CUDA raises `RuntimeError: indices should be either
  on cpu or on the same device`). The crash affected every PGS method
  (LDpred2-Inf/Grid/Auto, PRS-CS, C+T) through `.fit() → _harmonize →
  _ld_reference_subset`. Fix normalizes `idx` to `ld_ref.af.device` at
  function entry. 6 regression tests in
  `tests/test_pgs_score_device_routing.py`.

### Audit notes (not changed in this release)

- 8 hardcoded `/ 2.0` divisors in scan-model modules (`farmcpu.py`, `blink.py`,
  `single_trait_lmm.py`, `multi_trait_lmm.py`, `multi_kernel_lmm.py`,
  `rr_lmm.py`, `threshold_linear.py`) are annotated `# diploid convention`
  and intentional — polyploid input goes through the dedicated `poly-scan`
  CLI (GWASpoly-equivalent gene-action models), not these diploid LMM scans.
- CI hosted-runner billing has been paused on both `origin` (sikiru-atanda
  user account) and `pulsesmartlab-innovations` (org) since 2026-06-05.
  This release's fixes are verified locally and via the self-hosted GPU
  runner; full hosted-runner validation will happen when billing is resolved.

## [0.4.0] — 2026-06-02

Rebrand and surface-expansion release. The package is renamed from
`torchgwas` to `torchgenomics` to reflect a scope that now covers far
more than GWAS (post-GWAS, PGS, MR, TWAS, multi-omics, LD, imputation,
annotation, visualization). On top of the rename, two new surfaces ship:
a curated one-call API for novice / notebook users, and an MCP server
exposing 13 tier-1 tools to LLM clients (Claude Desktop, Claude Code).

Existing `torchgwas` users keep working without code changes — the
legacy package, CLI binary, and environment variables stay live as a
deprecation shim through the v0.x series and are removed in v1.0.0.

### Added — high-level API facade (`torchgenomics.api`)

A new `torchgenomics.api` module (also re-exported at the top level)
provides 12 one-call tier-1 functions designed for three audiences
sharing a single code path:

- **Data tier (3)**: `validate`, `convert`, `impute`
- **Scan tier (2)**: `lmm_scan`, `glm_scan`
- **LD / post-GWAS tier (3)**: `ld_blocks`, `clump`, `meta`
- **PGS tier (2)**: `pgs_fit`, `pgs_score`
- **Utility tier (2)**: `annotate_hits`, `manhattan` (+ bonus `qq`)

Each returns a typed result dataclass with `.summary()`, `.to_dict()`
(JSON-safe), `.runtime_s`, `.output_files`, and per-kind extras
(`.top_hits` DataFrame, `.manhattan()` / `.qq()` matplotlib helpers,
`.lambda_gc`, etc.). Smart defaults: format auto-detection, sample
auto-alignment, auto-named output directories, first-trait auto-pick.

### Added — MCP server (`torchgenomics-mcp`)

A new `torchgenomics.mcp` subpackage publishes the 13 tier-1 api
functions as Model Context Protocol tools over stdio:

```bash
pip install "torchgenomics[mcp]"
```

```json
{
  "mcpServers": {
    "torchgenomics": {"command": "torchgenomics-mcp"}
  }
}
```

Wrapper layer strips non-JSON-serializable params (`progress_callback`,
`Path` → `str`), preserves `Literal` enums as JSON Schema enums, and
auto-`.to_dict()`'s api result objects. The `torchgenomics-mcp
--list-tools` flag emits the tool registry as JSON without booting the
server.

### Changed — package rename

- `torchgwas` → `torchgenomics` across the Python package, CLI binary,
  PyPI distribution, GitHub repo, charter, README, CLAUDE.md, all docs,
  manuscript text, and 10 GitHub Actions workflows.
- 24 native C++ extension module names rebuilt as `torchgenomics._native._*`.
- All `TORCHGWAS_*` env vars renamed to `TORCHGENOMICS_*` (DISABLE_NATIVE,
  DISABLE_GPU, DISABLE_OPENMP, JULIA, BENCH_CI/QUICK/REALISTIC,
  ALLOW_AUTO_INSTALL, NCBI_LIVE).
- Charter file: `TorchGWAS_AI_Agent_Handoff_Charter.md` →
  `TorchGenomics_AI_Agent_Handoff_Charter.md`.

### Added — backwards-compat shim (removed at v1.0.0)

- `torchgwas` Python package — meta-path finder redirects every
  `torchgwas[.X]` import to `torchgenomics[.X]`. Same module objects.
  Emits one-shot `DeprecationWarning`.
- `torchgwas` console script — prints stderr deprecation banner once,
  then forwards argv unchanged to `torchgenomics.cli:main`.
- `TORCHGWAS_DISABLE_NATIVE` / `_GPU` / `_OPENMP` env vars — accepted
  with one-shot `DeprecationWarning`; new `TORCHGENOMICS_*` names take
  precedence when both are set.
- Centralized via `torchgenomics._dispatch.env_var(name)` helper; 9
  direct `os.environ.get` sites under `ld/` and `pgs/` migrated to
  `native_disabled()` so the legacy alias is honored uniformly.

### Refactored — CLI ↔ api dependency direction

For 7 of the 12 tier-1 functions (clump, meta, pgs_fit, pgs_score,
ld_blocks, lmm_scan, glm_scan), the api facade now calls thin
primitive-args helpers in `cli.py` (`_run_clump`, `_run_meta`, etc.)
rather than constructing synthetic `argparse.Namespace` objects. No
api module imports argparse anymore. Side-fix: `api.clump` was
silently passing wrong attribute names (`r2_threshold`/`bp_window_kb`
instead of `r2`/`window_kb`) — corrected. No CLI test was exercising
the api.clump path.

### Tests

- `tests/test_compat_shim.py` (13 tests) — import shim, env-var compat,
  CLI proxy banner + dispatch.
- `tests/test_api_facade.py` (43 tests) — registry, top-level
  re-exports, result-class invariants, end-to-end smoke for validate /
  lmm_scan / glm_scan / ld_blocks / manhattan / qq.
- `tests/test_mcp_server.py` (12 tests) — wrapper hygiene
  (progress_callback stripped, Path → str, Literal preserved), server
  boot, schema validity, end-to-end in-process MCP tool call.

**Verification**: 3,071 passed (3,003 baseline + 68 new), 417 skipped,
0 failures (excluding opt-in external / cli_matrix / reproducibility /
gpu markers). 2m 22s wall time.

### Migration

```python
# Before
import torchgwas
from torchgwas.models import SingleTraitLMM

# After
import torchgenomics
from torchgenomics.models import SingleTraitLMM
```

CLI: `torchgwas <cmd>` → `torchgenomics <cmd>` (old form proxies with a
banner). Env vars: `TORCHGWAS_DISABLE_NATIVE=1` → `TORCHGENOMICS_DISABLE_NATIVE=1`.
Python API surface is unchanged.

## [0.3.10] — 2026-06-01

Performance + infrastructure release. Seven new native C++ accelerators
on top of the existing 24, an ELBO Cholesky cache that unblocks
SuSiE-RSS at biobank-locus size, and a self-hosted GPU CI pipeline.
The public API is unchanged; this is purely faster and safer.

### Added — native C++ accelerators (7 new; 24 → 31 total)

Each is opt-in via `pip install torchgenomics[native]`, falls through to
the pure-Python algorithmic spec when the build is unavailable or
when `TORCHGENOMICS_DISABLE_NATIVE=1` is set.

- **SuSiE-RSS IBSS inner sweep** (`csrc/models/susie_rss_ibss.cpp`).
  One full IBSS iteration — per-layer SER + residual update + Brent
  V optimiser + softmax + EM M-step — runs under a single GIL
  release. Brent ported to C++ so each layer's prior-variance update
  no longer round-trips through `scipy.optimize`. Observed 4.6×–5.6×
  at typical fine-mapping sizes (p ∈ [128, 500]).
- **Mediation σ_u / σ_v per-pair WLS**
  (`csrc/multiomics/mediate_sigma_blocks.cpp`). Per-pair Cholesky on
  precomputed cross-products with OpenMP across SNPs. Observed
  **133× – 3,982×** across realistic block sizes; unblocks
  `mediate-scan` at UKB-size mediator sets.
- **PCHT compat-pair + posterior covariance**
  (`csrc/models/pcht_compat.cpp`). One pass for compat-pair
  enumeration + (H × H) posterior-covariance accumulation, OpenMP
  per sample. Observed **735× – 20,183×**. Shared kernel for all
  five novel haplotype tests (PCHT / HHCT / HSKAT / HapGxE / BayesHap).
- **MAGMA `snp_to_gene` window scan**
  (`csrc/postgwas/snp_to_gene.cpp`). Binary-search + linear scan
  over a globally-sorted SNP array, OpenMP across genes. At biobank
  scale (m = 10⁶ × n_genes = 20K), Python 31s → native 188 ms (167×).
- **TWAS expression normalisation**
  (`csrc/preprocess/expression_norm.cpp`). Per-column average-tie
  rank scan for `inverse_normal_transform` and
  `quantile_normalize`. At GTEx scale (n = 200 × m = 20K):
  rank-INT 18.7s → 11.2 ms (1,666×), quantile-norm 35.0s → 16.2 ms
  (2,164×).
- **hyprcoloc 2^K subset enumeration**
  (`csrc/postgwas/hyprcoloc_subsets.cpp`). Bitmask enumeration with
  per-subset `logsumexp` inline. 1.9× – 18.7× across K ∈ [6, 15];
  peak at K = 12. Only matters at K ≥ 8.
- **OrdinalGLMM threshold Newton-Raphson**
  (`csrc/models/ordinal_threshold_nr.cpp`). Score vector + Hessian
  assembly in one pass over samples, OpenMP-parallel. 3.7× – 48×
  across J ∈ {3, 5, 10}.

### Added — SuSiE-RSS ELBO Cholesky cache

`BayesianVSRss.fit_rss` now factorises R once before the IBSS loop
and reuses the Cholesky factor via `torch.cholesky_solve` inside
`_compute_elbo`. Replaces the per-iteration O(p³) `linalg.solve`
with O(p²) substitution. At p = 2,000 the end-to-end `fit_rss` drops
from 5.8 s to 0.53 s (11×). On a non-PD R the cache falls back
silently to the un-factored path.

### Added — GPU CI pipeline

- `.github/workflows/gpu.yml` triggers on `pull_request_target`
  (label-gated by `gpu-tested`), nightly cron, and
  `workflow_dispatch`. Runs the 71 CUDA-aware tests on a self-hosted
  runner labelled `[self-hosted, gpu, rtx-2000-ada]`.
- `scripts/setup_gpu_runner.sh` walks the operator through runner
  registration with SHA256 verification of the upstream tarball
  against the GitHub release manifest.
- `docs/operations/gpu-ci.md` documents the
  `pull_request_target` + label gate security model with citations
  to GitHub Security Lab's Pwn-Requests writeup.

### Fixed

- `fix(ci): GPU workflow — union (not intersection) of marker +
  file globs`. `pytest -m gpu file1.py file2.py …` is pytest's
  *intersection* semantic, not union; the workflow now runs two
  pytest passes so unmarked CUDA-skipif tests still run.
- `fix(ci): GPU workflow smoke step uses --help not --version`.
  The `torchgenomics` CLI is subcommand-based and has no `--version`
  flag.

### Internal

- Version synchronisation: `torchgenomics.__version__` was stuck at
  `"0.2.0"` while `pyproject.toml` advanced to `"0.3.8"`. Both now
  agree at `"0.3.10"`.
- `MANIFEST.in` now explicitly bundles `CHANGELOG.md` and adds
  belt-and-braces prunes for `paper/`, `validation/`, `examples/`,
  `tools/`, `.github/`, `.claude/`.
- README links rewritten from relative paths to absolute GitHub
  URLs so they resolve on PyPI / TestPyPI.

### Test suite

3,323 passed / 0 failed (was 3,276 pre-Tier-1). 47 new regression
tests across the 7 new native extensions + the ELBO Cholesky cache.

## [0.3.9] — 2026-05-26

Post-V1 expansion release: four F3 statistical fixes, the
observed-expression TWAS surface, the GWAS↔TWAS integration entry
point with 14 combination methods (8 classical + 6 novel), and a
suite of external-tool validation harnesses. Five PRs landed on
master (#2–#6); full test suite goes from **2921 passed, 43 failed**
to **3038 passed, 0 failed**.

### Added

- **GWAS↔TWAS integration** — `torchgenomics.postgwas.combine_gwas_twas`
  routes GWAS sumstats through MAGMA-style `snp_to_gene` aggregation
  and combines the gene-level GWAS p with the TWAS p via any of 14
  kernels.
  - 8 two-input kernels: `fisher_combined`, `brown_combined`,
    `empirical_brown_combined`, `harmonic_mean_p`, `truncated_product`,
    `min_p_combined`, `cauchy_combined` (wraps existing ACAT),
    `stouffer_combined` (wraps existing Stouffer's-Z).
  - 6 novel methods (each "to our knowledge"-tagged):
    `stouffer_r2_weighted` (eQTL CV-R² weighting), `brown_ld_aware`
    (Brown + eigenMT effective tests), `fisher_polyploid_gene_action`
    (two-stage Cauchy→Fisher across autopolyploid gene-action models),
    `cauchy_multi_tissue_plus_lead_snp` (S-MultiXcan + lead-SNP via
    ACAT), `gwas_twas_conditional` (ConditionalLMM-mediated; flags
    mediated vs independent), `gwas_twas_hyprcoloc_gated` (PPFC-gated).
  - New CLI: `torchgenomics combine-gwas-twas`. Subcommand count 39 → 40.
- **Observed-expression TWAS** — `twas_observed_expression()` for the
  workflow where the user has measured normalized expression and a
  phenotype, plus the corresponding `torchgenomics twas-scan` CLI. OLS by
  default; LMM (kinship-corrected) when a GRM is provided. New
  preprocessing module `torchgenomics.preprocess.expression` with
  `inverse_normal_transform` (Blom rank-INT), `quantile_normalize`,
  and `peer_residualize` (R-subprocess wrapper around PEER).
- **PrediXcan / FUSION `.db` reader** —
  `torchgenomics.io.read_predixcan_db()` returns a `PredixcanModel`
  dataclass with weights / snp_lists / eff_alleles / ref_alleles /
  r2_models / gene_names dicts ready to drop into `twas_sumstat`. Plus
  `list_genes_in_db()` convenience helper. Tolerates alternate r²
  column names (`pred.perf.R2`, `pred_perf_R2`, `cv_R2`).
- **Multi-tissue stacking + S-MultiXcan-style aggregation** —
  `twas_multi_tissue_stack` (long-format `{tissue: TWASResult}` →
  list of MultiTissueRow), `twas_multi_tissue_aggregate` (per-gene
  χ² + p_multixcan + driver-tissue identification).
- **Gene-level TWAS visualizations** — `manhattan_twas` (chromosome-
  aware when annotation is supplied; top-N gene-name labelling),
  `qq_twas` (with Beta-CI band), `genomic_inflation_factor_twas`
  (λ_TWAS using `chi2.ppf(0.5, df=1)` as the denominator).
- **External-tool harnesses** — `validation/external/fusion/` against
  the FUSION measured-expression mode (FP-precision agreement on
  β/SE/z/-log10 p), and `validation/external/metap/` against the
  CRAN `metap` + Bioconductor `EmpiricalBrownsMethod` packages
  (closed-form on Fisher / Stouffer / min-p at 1e-10).
- **F2 closure** — LD-aware pruning in `_enumerate_haplotypes_unphased`
  (Tier 1 A5 finding); replaces the marginal-AF-product candidate
  scoring that dropped real LD-driven haplotypes under tight LD.

### Fixed

- **F3 #1 — hyprcoloc Foley 2021 conditional prior**
  (`torchgenomics/postgwas/_hyprcoloc.py`). Replaces the product prior
  with the hierarchical conditional prior. Cluster membership now
  matches R hyprcoloc exactly on the 3-trait shared-causal fixture
  ([0, 1, 2] vs [0, 2] pre-patch).
- **F3 #2 — coloc_pairwise H3 formula**
  (`torchgenomics/postgwas/_hyprcoloc.py`). H3 marginal rewritten as
  *outer − diagonal* of per-SNP weights; spurious `-log m`
  per-hypothesis factor removed. PP.H3 on the distinct-signal fixture
  goes from 0.85 (pre-patch) to 0.997, matching R `coloc::coloc.abf`
  to FP precision.
- **F3 #3 — heidi_test LD-weighted variance**
  (`torchgenomics/postgwas/_smr.py`). New optional `ld_matrix` parameter
  applies the Zhu 2016 sup. eq. 18 per-SNP-pair LD-corrected variance,
  matching SMR v1.3.1 to 0.67% on chi²_HEIDI and 1.55% on p_HEIDI on
  the harness fixture.
- **F3 #4 — OCFLMM `nuisance_learner='ridge_quadratic'`**
  (`torchgenomics/models/ocf_lmm.py`). New optional nuisance learner with
  quadratic feature expansion + small ridge; default stays `'linear'`
  for V1 backward compatibility. Empirical 95% coverage moves from
  0.41 (linear, biased under nonlinear confounding) to 0.94 on the
  Chernozhukov 2018 partially-linear DGP.
- **GPU haplotype parity** (`torchgenomics/models/haplotype_gwas.py`).
  Uses `torch.argsort(stable=True)` for tied LD-aware scores, fixing
  the pre-existing CPU/CUDA divergence in `HaplotypeGWAS(method="window")`.
- **PGS parser cleanup** (`torchgenomics/cli.py`). Restored the missing
  `_add_pgs_fit_parser` / `_add_pgs_score_parser` / `_cmd_pgs_fit` /
  `_cmd_pgs_score` function bodies that had been referenced in
  `cli.py` but never defined on master, closing 40 NameError test
  failures.
- **TWAS observed-expression OLS variance** — switched the OLS path
  from delegating to `GLM.score_chunk` (null-only σ²_e, GAPIT P3D=TRUE
  convention) to a full per-gene FWL OLS refit. Brings the FUSION
  harness from 88% relative SE divergence to FP precision (2e-15).
- **Three test/code drift fixes** — `LinkFunction` ABC expectation
  (TypeError on `LinkFunction()`, not NotImplementedError on methods),
  `fit_mvlmm_null_lbfgs` single-trait rejection (`d >= 2` validation),
  `compare_pvalues` NaN-aware with new keys `n_total` / `n_finite` /
  `n_within_tolerance`.

### Tests

- Full suite: **3038 passed, 0 failed, 492 skipped** (from 2921
  passed / 43 failed at the start of the campaign).
- New test files: `test_postgwas_combine.py` (43), `test_postgwas_twas.py`
  (+13), `test_io_predixcan_db.py` (9), `test_viz_twas.py` (10),
  `test_preprocess_expression.py` (12), `test_cli_combine_gwas_twas.py`
  (2), `test_cli_twas_scan.py` (2). Plus per-F3-patch regression gates
  added to existing test files.

### Validation

- FUSION measured-expression harness: 4/4 PASS at FP precision
  (max |Δβ|/|β| = 2e-15, max |Δz| = 1e-14).
- MetaXcan / S-PrediXcan harness re-verified: 4/4 PASS, Pearson r(z) = 1.0.
- SMR v1.3.1 harness: 0.67% chi²_HEIDI agreement post-patch (was 38%).
- R hyprcoloc / R coloc harnesses: post-patch agreement at the
  observed-then-floored tolerances.
- Five validation findings closed (F2 + F3 #1-#4) in
  `docs/validation_findings.md`.

## [0.3.8] — 2026-05-05

Multi-kernel streaming release (F3). The last tractable materialized
scan path — `mklmm-scan` — now streams kernel construction
chunk-by-chunk. After this release **only `bayes-scan` remains
materialized** (joint posterior over all m SNPs is genuinely
streaming-incompatible). **Audit counts: 35 → 36 streaming
subcommands; 2 → 1 materialized.**

### Added

- **`torchgenomics.linalg.multi_kernel_streaming.build_multi_kernels_streaming`**
  (commit `c075a8b`) — stream-builds the additive / dominance /
  epistatic kernel set from an `iter_chunks` reader. Each base kernel
  (additive, dominance) is accumulated chunk-by-chunk in float64;
  epistatic kernels are Hadamard products of the now-built `(n × n)`
  base kernels (so they cost nothing extra in G residency). Behavioral
  parity with the in-memory `build_multi_kernels(G, ...)` to FP-32-GEMM
  streaming-accumulator slack (~1e-7).

### Changed (F3 streaming rewrite)

- **`cli._cmd_mklmm_scan`** — replaces `_load_scan_data + build_multi_kernels(G, ...)`
  with `_align_samples + build_multi_kernels_streaming(...)`. Peak
  memory drops from `O(n × m × 8 B)` (~40 TB at UKB scale) to
  `O(n² × n_kernels × 8 B)` for the kernels themselves — the same
  ceiling every LMM has — plus `O(n × chunk_size × 8 B)` per
  accumulator pass.

- **`cli._cmd_pipeline`'s mklmm branch** — mechanical follow-on. Now
  dispatches to `build_multi_kernels_streaming`. Removes the prior
  warning log about pipeline mklmm being memory-prohibitive at
  biobank scale.

### Audit (post-F3)

- **36 streaming** subcommands (was 13 baseline → +23 across E1/E3/E4/F1/F2/F3).
- **3 partial** (unchanged: opt-in `--grm-method zhang` for lmm-scan /
  mvlmm-scan; mediate-scan user-side `.npy`/`.tsv` input floor).
- **1 materialized** — `bayes-scan` only. SuSiE / CAVI joint posterior
  over all m SNPs requires random access; streaming would change the
  algorithm semantically. Documented as O3 (algorithmically tight).

The streaming campaign is now algorithmically saturated. The single
remaining materialized path cannot be streamed without changing the
inference algorithm. Pipeline materialized branches are gone.

## [0.3.7] — 2026-05-05

Cached-column / per-block streaming release (F2). Three of the six
remaining materialized paths flagged in F1 are now streaming:
`farmcpu-scan`, `blink-scan`, `mediate-scan`. The `pipeline`
materialized branch for FarmCPU / BLINK is updated as a mechanical
follow-on. **Audit counts: 32 → 35 streaming subcommands; 6 → 2
materialized (excluding the user-side input fixed for `mediate-scan`,
counted as partial).**

### Added

- **`torchgenomics.models.farmcpu.FarmCPU.score_streaming(reader, null_fit,
  chunk_size, test)`** — orchestrates the FEM / REM iteration
  externally. Per iteration: stream the genome once for
  `_glm_scan_streaming` (per-SNP gtg / gty / beta / se / p
  accumulated chunk-by-chunk) and (when prior QTNs exist) once more
  for `_substitute_qtn_pvalues_streaming`. The QTN selection logic
  (`_specify_bins`, p-value filter, `_remove_correlated_with_cols`)
  operates only on the small (≤ ~20) cached QTN columns. Module-level
  helpers `_read_columns_from_reader`, `_glm_scan_streaming`,
  `_substitute_qtn_pvalues_streaming`,
  `_remove_correlated_with_cols`. Behavioral parity to float64
  tolerance vs. eager `score_chunk` (max p diff ~1.8e-15 on a
  100 × 200 fixture).

- **`torchgenomics.models.blink.BLINK.score_streaming(reader, null_fit,
  chunk_size, test)`** — same shape as FarmCPU. LD-removal
  (`_ld_remove_block_with_cols`) and BIC selection
  (`_bic_forward_select_with_cols`) operate on candidate column
  matrices read from the reader on demand (`_read_columns_from_reader`).
  Behavioral parity (max p diff 0.0 on a 100 × 200 fixture).

- **`torchgenomics.multiomics._scan_batched.batched_scan_pairs_streaming`** —
  per-SNP-block lazy rotation. Groups pairs first by SNP-block, then
  by feature-block; for each SNP-block lazy-rotates only that
  block's columns, dispatches all paired feature-blocks, frees. The
  mediator matrix `M_r` is rotated once and held. `scan_mediation`
  gains a `streaming: bool | None` kwarg (default None → on for
  batched scans with ≥1000 pairs). Bit-for-bit zero diff on a
  80 × 200 × 50 fixture vs. eager `batched_scan_pairs` when the same
  SE seed is used.

### Changed (F2 streaming rewrites)

- **`cli._cmd_farmcpu_scan_single`** — replaces `_load_scan_data` +
  `UnifiedScanner` with `_align_samples` + `model.score_streaming`.
  Peak memory drops from O(n × m × 8 B) (~40 TB at UKB) to
  O(n × |QTN| + chunk_size × n × 8 B) (~GB-scale per chunk).

- **`cli._cmd_blink_scan_single`** — same shape. Peak: O(n × |QTN|
  + n × |candidates| + chunk_size × n × 8 B). Both QTN and candidate
  caches are bounded by sqrt(n) / log(n)-ish.

- **`cli._cmd_pipeline`** (FarmCPU / BLINK branch) — mechanical
  follow-on. The previous `logger.warning` block ("genuinely not
  streamable; consider lmm/mvlmm/glm") is removed. The MKLMM branch
  retains the warning (a future F3 candidate).

### Fixed

- **Audit table mediate-scan row.** The F1 / E0 audit had labelled
  `mediate-scan` as materializing via `_load_full_genotype` (via
  `_load_scan_data`). The actual CLI uses
  `_load_array(args.genotype)` on a `.npy` / `.pt` / `.tsv` flat
  file — the user-side input cost is irreducible without a CLI
  input-format extension. Reclassified as "partial" (input-fixed).

### Tests

- **`tests/test_streaming_memory.py`** — 8 new memory regression
  tests across 3 new test classes (FarmCpu, Blink, MediateScan).
  Total streaming-memory tests: 50.
- Suite: 2801 passed (+8 from v0.3.6's 2793), 0 failed.

### Audit (post-F2)

- 35 streaming subcommands (was 13 after E1, 20 after E3, 26 after
  E4, 32 after F1).
- 3 partial (`lmm-scan --grm-method zhang`, `mvlmm-scan --grm-method
  zhang`, `mediate-scan` user-side input).
- 2 materialized: `bayes-scan` (joint posterior over all variants),
  `mklmm-scan` (dominance / epistatic kernels). MKLMM is the F3
  candidate per the task ledger; bayes-scan is genuinely impossible
  per audit observation O2.

## [0.3.6] — 2026-05-05

LD-window streaming release (F1). Rewrites the six remaining
materialized scan paths whose common shape is "needs LD between SNPs
in a bounded window": `ldsc`, `ldsc-rg`, `ld-blocks`, `clump`,
`knockoff-scan`, `lro-scan`. Each path genuinely needs LD inside a
window — but the window is bounded (1 cM ≈ 1000 SNPs for LD scores;
per-chromosome for the rest). The legacy paths all materialized the
full ``(n × m)`` G via either ``_load_scan_data`` or
``torch.cat([chunks])``; at biobank scale ~40 TB float64. **Audit
counts: 26 → 32 streaming subcommands; 12 → 6 materialized.**

### Added

- **`torchgenomics.postgwas.compute_ld_scores_streaming`** — sliding-
  window LD-score helper. Per-chromosome buffer holds only SNPs whose
  right edge has not yet been crossed by the latest streamed position.
  Running r² partial sums accumulate chunk-by-chunk so each pair
  contributes exactly once. Peak memory: O(n × window_size × 8 B),
  independent of total m. Behavioral parity to float64 tolerance vs.
  the in-memory `compute_ld_scores`.

- **`torchgenomics.models.lro_lmm.LROLMM.run` — `K_full` / `normalizer`
  kwargs.** When supplied, `LROLMM.run` skips its internal
  `grm_vanraden(G)` call and uses the caller's genome-wide GRM. The
  per-block `K_b = G_block @ G_block^T / normalizer` semantics are
  preserved exactly — the chromosome-local G slice contributes only
  `K_b`; `K_minus_b = K_full − K_b` correctly removes that
  contribution from the genome-wide GRM. Enables the
  per-chromosome streaming variant of `lro-scan`.

### Changed (F1 streaming rewrites)

- **`cli._cmd_ldsc`, `cli._cmd_ldsc_rg`** — true sliding-window
  streaming via `compute_ld_scores_streaming`. Replaces
  `compute_ld_scores(G_full, …)`. Single shared rewrite (the helper).
  Peak memory drops from ~40 TB to typically ~4 GB at UKB scale.

- **`cli._cmd_ld_blocks`, `cli._cmd_clump`, `cli._cmd_knockoff_scan`,
  `cli._cmd_lro_scan`** — per-chromosome streaming buffer. LD blocks
  never span chromosomes by physical-LD definition, so the chromosome
  is the natural memory unit for block detection / clumping /
  knockoff construction / leave-region-out scans. Each rewrite
  accumulates one chromosome's slice at a time, runs the algorithm,
  and frees the slice before reading the next. Per-chromosome results
  are merged into a genome-wide result; the knockoff+ FDR filter is
  applied ONCE over the merged W-statistic vector for genome-wide
  control. Peak memory: O(n × max_per_chromosome_m × 8 B) plus, for
  knockoff / LRO, the streaming GRM (n²).

### Fixed (F1 latent bug)

- **`cli._cmd_ld_blocks` multi-chromosome mis-attribution.** The
  legacy single-call `detect_blocks(G_full, all_chrs, …)` mis-
  attributed blocks on chr 2+ to chr 1 because
  `compute_pairwise_ld` emits chromosome-LOCAL indices, but
  `_make_block` indexes into the genome-wide `variant_chr` /
  `variant_pos` arrays. The streaming rewrite calls `detect_blocks`
  once per chromosome with that chromosome's slice — fixing the bug
  as a side effect. Test
  `test_streaming_per_chromosome_correctly_attributes_blocks` guards
  the corrected behavior.

### Tests

- **`tests/test_streaming_memory.py`** — 11 new memory regression
  tests across 6 new test classes (LdScores, LdBlocks, Clump,
  Knockoff, Lro). Total streaming-memory tests: 42. Each guards
  parity vs the materialized reference (where well-defined) and a
  per-chromosome scaling assertion (peak roughly invariant in
  chromosome count for fixed per-chromosome size).
- Suite: 2793 passed (+11 from v0.3.5's 2782), 0 failed.

### Audit (post-F1)

- 32 streaming subcommands (was 13 after E1, 20 after E3, 26 after E4).
- 2 partial (unchanged; opt-in `--grm-method zhang`).
- 6 materialized — `farmcpu-scan`, `blink-scan`, `bayes-scan`,
  `mklmm-scan`, `mediate-scan`, `pipeline` materialized branches
  (farmcpu / blink / mklmm). All algorithmically tied to full G —
  documented as O2 (fundamentally not-streamable) in
  `docs/efficiency/streaming_audit.md`.

## [0.3.5] — 2026-05-05

Performance regression CI release (E5). The wall-time analog to the
streaming-memory regression nets shipped in v0.3.2–v0.3.4. PRs that
slow any monitored native kernel by > 10% now fail CI; > 5% gets a
warning. Closes the regression-net loop on TorchGenomics' 24 native C++
accelerators.

### Added

- **`bench/native_speedups.py --output json`** — JSON output mode
  with documented schema (`{kernel, group, size_label, mode,
  n_repeats, wall_seconds_p50, wall_seconds_p95, speedup_vs_python}`).
- **`bench/native_speedups.py --kernel-subset ci`** — fast 12-kernel
  subset (Gabriel / PELT / CC-graph blocks, KNN / mode imputation,
  LDpred2 / PRS-CS Gibbs, HWE diploid, SPA, LDSC jackknife, GRM
  streaming, VanRaden GRM). Plus `TORCHGENOMICS_BENCH_CI=1` env var that
  shrinks input sizes for kernels with O(m⁴) / O(n²) Python reference
  paths so the full CI cycle stays under the 8-min budget.
- **`bench/diff_perf.py`** — reads two benchmark JSON files (master
  + PR), emits a markdown table per kernel, exits 0 (clean) / 1
  (regression > 10% on native mode) / 2 (warning > 5%). Python rows
  are always reported as OK — algorithmic reference, not gated.
- **`.github/workflows/perf.yml`** — runs on PR + push to master.
  Builds native extensions, benchmarks both checkouts (master
  baseline + PR head), runs `diff_perf.py`, posts a markdown diff
  comment on the PR via `actions/github-script` (hidden-tag
  identifier so re-runs replace rather than stack), uploads
  `perf-baseline-master` artifact on master push.
- **`docs/efficiency/perf_regression.md`** — explainer: how to read
  the perf table, how to add a new kernel to the CI subset, how to
  update the gate threshold, regression-vs-warning-vs-noise
  interpretation.
- **`tests/test_perf_regression_ci.py`** — 4 new tests: JSON schema
  shape, regression-flagged exit code, warning-only exit code,
  no-regression pass-through. Plus 2 edge-case tests: Python row
  ungated, missing master baseline returns SKIPPED.

### Tests

- 7 new tests in `test_perf_regression_ci.py`. Total suite: 2782
  passed (+7 from v0.3.4's 2775), 0 failed.

### CI workflow inventory after E5

- `ci.yml` (existing, every PR): default suite + golden + native +
  no-openmp + type-check.
- `cli-matrix.yml` (Pillar C, nightly + manual): Pillar C smoke matrix.
- `external.yml` (Pillar B, weekly + manual): Pillar B reference-tool
  comparisons.
- `reproducibility.yml` (Pillar D, monthly + manual): fixture-drift audit.
- **`perf.yml` (E5, every PR): native-kernel wall-time regression gate.**
  All four campaign + post-campaign regression nets are now wired.

## [0.3.4] — 2026-04-30

End-user runtime efficiency release (E4). Continues the streaming
audit campaign from v0.3.2 / v0.3.3: rewrites four more scan
subcommands and four imputation methods to stream chunks via
`iter_chunks`, plus updates the `pipeline` orchestrator to dispatch
to the new streaming variants. **Audit counts: 13 → 26 streaming
subcommands; 25 → 12 materialized.**

### Added

- **`torchgenomics.preprocess.impute`** — four streaming-friendly
  building blocks alongside the in-memory reference functions:
  `compute_column_means_streaming`, `compute_column_modes_streaming`,
  `impute_chunk_with_means`, `impute_chunk_with_modes`,
  `impute_chunk_with_knn`, `impute_chunk_with_ld_window`. Each
  reduces to the same math as its in-memory equivalent
  chunk-by-chunk.

- **`cli._open_impute_output_sink`** — supports `.zarr` per-chunk
  output sinks (peak memory bounded by chunk size) plus legacy
  `.pt` materialization fallback.

### Changed (E4 streaming rewrites)

- **`cli._cmd_rr_scan`, `cli._cmd_rr_met_scan`, `cli._cmd_met_scan`**
  — Group A "GRM-only" fixes. Replace `_load_full_genotype +
  grm_vanraden(G)` with `grm_vanraden_streaming(...)`. The scan
  loop already streamed; only the kinship build was materialized.

- **`cli._cmd_poly_scan`** — Group B full streaming via
  `UnifiedScanner` + on-the-fly gene-action recoding wrapper
  (`_RecodingReader`). The `--joint-qtl` post-analysis branch
  retains an opt-in materialization since joint QTL needs random
  column access. Surfaces a latent quirk in the materialized path
  (G_scan was computed but never reached UnifiedScanner); streaming
  rewrite preserves that behavior for `additive` / `general`.

- **`cli._cmd_impute`** — Group C streaming for all four built-in
  methods (mean / mode / knn / ld). Mean/mode use two-pass
  per-column statistics; knn uses streaming GRM + per-chunk fill
  (K still held once at n×n×8 B); ld uses sliding-window flank
  buffer (documented buffer cost).

- **`cli._cmd_pipeline`** — Group D mechanical follow-on. `gxe`
  and `met` branches now stream (mirroring gxe-scan / met-scan).
  `farmcpu` / `blink` / `mklmm` branches retain the materialized
  path (genuinely not streamable) but now log a warning that
  explains the algorithmic constraint.

### Tests

- **`tests/test_streaming_memory.py`** — 16 new memory regression
  tests across 8 new test classes (rr-scan, rr-met-scan, met-scan,
  poly-scan, impute mean/mode/knn/ld). Total streaming-memory
  tests: 31. Each guards parity vs the materialized reference and
  an absolute peak budget under tracemalloc.

### Audit (post-E4)

- 26 streaming subcommands (was 13 after E1, 20 after E3).
- 2 partial (unchanged; opt-in `--grm-method zhang`).
- 12 materialized — all algorithmically tied to full G or
  windowed pairwise r² across all variants. See
  `docs/efficiency/streaming_audit.md` observation O2.

## [0.3.3] — 2026-04-30

End-user runtime efficiency release (E3). Streams five more
materialized scan subcommands using the glmm-scan template
established in v0.3.2. Tagged retroactively at commit `418b0cf`
to mark the E3 ledger entry; pyproject version was bumped in the
v0.3.4 release.

### Changed (E3 streaming rewrites)

- **`cli._cmd_me_glmm_scan`** (commit `4fafb74`) — `_align_samples`
  + `grm_vanraden_streaming` + per-chunk `MultiEnvGLMM.score_chunk`
  loop with `_merge_env_results`.
- **`cli._cmd_survival_scan`** (commit `507bf5d`) — `UnifiedScanner`
  default merger (SurvivalGLMM emits `ScanResult`).
- **`cli._cmd_threshold_scan`** (commit `e8a0d3c`) — `UnifiedScanner`
  default merger (ThresholdLinearModel emits `ScanResult`).
- **`cli._cmd_gxe_scan`** (commit `6fc64f7`) — new
  `_merge_gxe_results` helper for `GxEScanResult`. Also fixes a
  latent crash for any multi-chunk gxe scan (the legacy single-chunk
  path bypassed the merger entirely).
- **`cli._cmd_gu_scan`** (commit `34c1e28`) — paired-stream variant:
  G chunks via `iter_chunks`; user-supplied `dosage_var` held once
  and sliced per chunk by running col_offset.

### Tests

- 10 new memory regression tests in
  `tests/test_streaming_memory.py` — one class per E3 rewrite,
  each with parity + budget guards. Total streaming-memory tests:
  15 (3 set-scan + 2 glmm-scan + 10 E3).

### Audit (post-E3)

- 20 streaming subcommands (was 13 after E1).
- 2 partial (unchanged; opt-in `--grm-method zhang`).
- 18 materialized.

## [0.3.2] — 2026-05-05

End-user runtime efficiency release. Audited all 40 CLI scan
subcommands for memory behavior; rewrote two materializing paths
(`set-scan`, `glmm-scan`) to stream chunks via `iter_chunks` /
`UnifiedScanner` for biobank-scale tractability. No API breakage —
legacy materialized entry points are preserved alongside the new
streaming variants.

### Added

- **`models.set_based.SetBasedScanner.scan_regions_streaming(chunk_iter, ...)`**
  — accumulates per-region buffers chunk-by-chunk; peak memory is
  `Σ region_size × n_samples × 8 B` rather than `n_variants × n_samples × 8 B`.
  At biobank scale (n=500K samples × m=10M variants × 20K gene-set regions
  × 50 SNPs/region average): **40 TB → ~4 GB**. Legacy
  `scan_regions(G_full, ...)` retained for callers that already have
  the matrix in hand.

- **`docs/efficiency/streaming_audit.md`** — 40-row table classifying
  every CLI scan subcommand as streaming / partial / materialized,
  with peak-memory estimates at biobank scale. Identifies 5
  tractable next-target rewrites (me-glmm-scan, survival-scan,
  threshold-scan, gxe-scan, gu-scan) and 5 algorithms genuinely
  unsuitable for streaming (FarmCPU, BLINK, BayesianVS, MultiKernelLMM,
  KnockoffLMM).

- **`tests/test_streaming_memory.py`** — 5 memory-regression tests via
  `tracemalloc` that assert streaming peak memory stays under documented
  budgets (8 MiB for set-scan, 16 MiB for glmm-scan at n=200 / m=500
  fixture scale). Includes the load-bearing biobank-relevant assertion
  `peak ∝ Σ region_size`, not `peak ∝ m` — catches future refactors that
  would silently re-materialize.

### Changed

- **`cli._cmd_set_scan`** — replaces `_load_scan_data` (full G
  materialization) with `_align_samples + grm_vanraden_streaming +
  SetBasedScanner.scan_regions_streaming`. Output identical to the
  prior path; verified via `tests/test_streaming_memory.py::TestSetScanStreamingMemory::test_streaming_matches_materialized`
  (float64 tolerance on q-stat / p).

- **`cli._cmd_glmm_scan`** — replaces full-chunk `_load_genotype_matrix`
  with `UnifiedScanner` driving per-chunk `model.score_chunk`. PQL null
  fit only needs Y / X0 / K. GRM remains streaming via
  `grm_vanraden_streaming`. Output identical to the prior path; verified
  via `tests/test_streaming_memory.py::TestGlmmScanStreamingMemory::test_streaming_glmm_matches_full_chunk`.

### Audit findings (40 CLI subcommands)

- **13 streaming** — already do the right thing (lmm-scan, mvlmm-scan,
  glm-scan, etc.).
- **2 partial** — opt-in `--grm-method zhang` materializes for kinship,
  then releases (lmm-scan, mvlmm-scan).
- **25 materialized** before this release. After this release:
  set-scan + glmm-scan removed → **23 remaining materialized**.

The choke point is `cli._load_full_genotype` (used by 23 of 25
materialized paths). It calls `iter_chunks(chunk_size = n_variants)` —
i.e., one chunk = whole matrix — then upcasts to STAT_DTYPE (float64).
The 5 next-target rewrites would each take ~15-line diffs following the
`glmm-scan` template; deferred to subsequent releases.

## [0.3.1] — 2026-05-05

Three deferred post-V1 follow-ups completed shortly after the v0.3.0
campaign close. Resolves outstanding Phase-33 / Phase-44 polish items
identified in the campaign reviewer notes.

### Changed (one behavioral default change)

- **`postgwas.mr.mr_presso`** — default null distribution switched from
  `"permutation"` to `"parametric"` (Verbanck 2018 Eq. 2 / MRPRESSO 1.0
  reference). This brings TG into bit-for-bit agreement with the
  TwoSampleMR R package: post-fix `|Δ corrected β|` = 3.1e-5 (was
  4.9e-2), `|Δ global p|` = 1e-3 (was 9.99e-1), outlier-set Jaccard
  1.0 (was 0.5). Callers depending on the old null can opt back in
  with `mr_presso(..., null="permutation")`.

### Fixed

- **`models.binary_glmm.BinaryGLMM` (Phase 33)** — `converged` flag
  was previously inferred from "did we hit max_iter?", which falsely
  flagged converged=False on small-N PQL fits where β / SE / p all
  agreed with SAIGE to within tolerance. Now uses a per-iteration
  log-likelihood + variance-component delta-check alongside the
  existing β-stability criterion. Used by 5 GLMM models (binary,
  ordinal, multinomial, multi-env, survival) — all 81 PQL-based tests
  pass.

- **`models.lmm_multi_fit.fit_mvlmm_null_{ai_reml,lbfgs}`** —
  `converged` flag was previously a heuristic on trace-tail relative
  log-likelihood change. Now plumbed directly from the underlying
  optimizer's reported converged flag (`pxem_nr_mvreml` and
  `lbfgs_reml` both expose it via `trace[-1]["converged"]`), with
  legacy heuristic as fallback.

### Migration note

Callers of `mr_presso` who relied on the prior permutation-null
behavior must explicitly pass `null="permutation"`:

```python
# Old (v0.3.0 and earlier — permutation null):
mr_presso(sumstats, n_perms=1000)

# New default (v0.3.1+ — parametric LOO bootstrap, matches TwoSampleMR):
mr_presso(sumstats, n_perms=1000)

# Opt back into legacy permutation null:
mr_presso(sumstats, n_perms=1000, null="permutation")
```

The parametric default produces materially different global-p and
outlier-set values for any pleiotropic-signal input — review downstream
analyses if you upgrade.

## [0.3.0] — 2026-05-05

Validation campaign release. A function-by-function validation campaign
ran across four pillars: coverage audit, external reference-tool
comparisons, end-to-end CLI smoke matrix, and fixture-reproducibility
audit. The campaign surfaced 14 V1-platform / V1-core production fixes
and added ~700 new tests. All three committed reference fixtures
(GEMMA / GAPIT / GWASpoly) re-verified as authoritative.

### Fixed (campaign findings)

**Pillar A — coverage audit + tiered fill (8 fix-now findings):**
- `stats.calibrate.compare_pvalues` was a Phase-4 `NotImplementedError`
  stub — implemented with `n / mean_abs_diff / max_abs_diff /
  frac_within_tolerance / corr_neglog10 / mean_abs_diff_neglog10 /
  tolerance` keys; NaN-safe.
- `models.lmm_single`, `models.lmm_multi`, `models.lmm_multi_fit`
  carried Phase-4/5 `NotImplementedError` stubs that diverged from the
  actual implementations in `single_trait_lmm` / `multi_trait_lmm` /
  `optim.{pxem_nr_mvreml,lbfgs_reml}` — converted to thin re-exports
  + populated NullFit wrappers.
- `optim.fisher_scoring.fisher_scoring_reml` was a Phase-4 stub —
  implemented Fisher-scoring REML for single-trait LMM in rotated
  eigenspace (expected-info Newton update on lambda).
- `io.convert._write_zarr` broke on zarr 3.1.5 (`AsyncGroup.create_dataset`
  signature change) — switched to v3-aware `create_array` path with v2
  fallback.
- `postgwas._ld_scores.compute_ld_scores` silently truncated when
  `chr_labels` / `pos` were shorter than `G.shape[1]` — added explicit
  ValueError shape guards before the per-SNP loop.
- `postgwas._clump.ld_clump` silently used the wrong subset of `G`
  when `len(p) != G.shape[1]` — same shape-validation pattern.

**Pillar B — external reference-tool comparisons (1 fix-now):**
- `postgwas._mr.mr_egger` had three deviations from canonical
  Bowden-2015 / TwoSampleMR: missing orientation flip, wrong
  overdispersion direction, normal vs Student's t for p-value.
  Rewrote per Bowden 2015 §3; agreement with TwoSampleMR R package
  now to 5 sig figs (was |Δβ|=4.76e-2; post-fix 4.12e-5).

**Pillar C — CLI smoke matrix (4 fix-now + 1 follow-up audit):**
- `cli.glmm-scan` / `me-glmm-scan` / `survival-scan` crashed on a
  non-existent `linalg.grm` import — fixed to use canonical
  `linalg.kinship.grm_vanraden`.
- `cli._apply_correction_and_save` couldn't handle multi-output result
  schemas (gxe / mvlmm / me-glmm) — restructured to detect and flatten
  2-D tensors + GxE schema variants.
- `cli.impute` crashed on `_load_genotype_matrix` mis-unpacking reader
  chunks — rewrote to handle both `(G, vmeta)` tuple and legacy
  `.dosage` attribute shapes.
- `cli.lmm-scan --device cuda` crashed: `_cmd_lmm_scan_single` left
  Y/X0 on CPU when K was on CUDA — added explicit `.to(device)` calls.
- Device-alignment audit follow-up: `cli.glm-scan` silently fell back
  to CPU under `--device cuda` — same fix.

### Added

**Validation infrastructure:**
- `scripts/audit_public_coverage.py` — AST-based public-symbol coverage
  auditor.
- `scripts/build_tier_worklist.py` — per-package, per-tier untested
  worklist generator (deduplicates across re-export paths).
- `validation/external/<tool>/` — install/fetch/run/compare harnesses
  for 11 reference tools (GEMMA, GAPIT, GWASpoly, PLINK 2.0, LDSC,
  TwoSampleMR, regenie, SAIGE, BOLT-LMM, SoyNAM, SoyMD).
- `validation/external/_lib/preflight.sh` — shared memory + disk
  pre-flight gate (mandatory before every external tool download / run).
- `validation/reproducibility/` — fixture-drift audit harness.
- ~700 new tests across `tests/test_coverage_*.py`,
  `tests/test_external_*.py`, `tests/test_cli_matrix.py`,
  `tests/test_reproducibility.py`.
- New pytest markers: `external` (Pillar B, opt-in via `-m external`),
  `cli_matrix` (Pillar C, opt-in via `-m cli_matrix`),
  `reproducibility` (Pillar D, opt-in).
- New CI workflows: `.github/workflows/cli-matrix.yml` (nightly +
  manual), `external.yml` (weekly + manual), `reproducibility.yml`
  (monthly + manual).

**Phase 37 follow-up (LDSC IRWLS port):**
- `postgwas._ldsc.ldsc_h2` / `ldsc_intercept` / `ldsc_rg` /
  `ldsc_rg_from_z` extended with iteratively reweighted least squares
  (IRWLS). The optional `w_ld` argument accepts the regression-weight
  LD scores; default behavior matches LDSC's 2-iteration loop. Resolves
  the Pillar B B2 intercept divergence: agreement with LDSC now to
  ~5 significant figures (was |Δ intercept|=0.1–0.3; post-port 2.6e-5).
- Single-pass behavior remains available via `n_iter=0`.

**Documentation:**
- Validation campaign spec at
  `docs/superpowers/specs/2026-04-30-validation-campaign-design.md`.
- Per-pillar plans under `docs/superpowers/plans/`.
- Findings ledger at `docs/validation_findings.md` (every divergence
  with F3 classification + resolution).
- Session handoff at `docs/superpowers/SESSION_HANDOFF.md`.

### Changed

- `pyproject.toml` — registered new pytest markers (`external`,
  `timeout`, `cli_matrix`, `reproducibility`).
- `tests/conftest.py` — collection hook auto-skips `external` and
  `cli_matrix` markers unless explicitly requested.
- `CLAUDE.md` — new validation-campaign section with re-run paths.

### Removed (post-campaign cleanup)

- `pgs.validation.pi` (stale `math.pi` re-export, no callers).
- `cli.TYPE_CHECKING` removed from public-symbol enumeration (renamed
  to `_TYPE_CHECKING` so `if _TYPE_CHECKING:` typing-only guards still
  work).

## [0.2.0] — 2026-04-21

Post-v0.1.1 audit release. Four improvement bundles — A (correctness),
B (adoption), C (platform), D (coverage) — land together as a minor
bump because the user-facing surface (docs site, PyPI wheels,
examples) is materially larger.

### Added

**Adoption (Bundle B)**
- MkDocs Material site under `docs/` with auto-deploy to GitHub Pages
  on master push (`.github/workflows/docs.yml`). Covers install,
  quickstart, CLI reference, API auto-gen via `mkdocstrings`, and the
  Section 16 validation protocol.
- `examples/python/01..10_*.py` — ten runnable scripts covering
  single-trait LMM, multi-trait LMM, multi-environment, threshold-
  linear, PGS construction, fine-mapping (SuSiE), mediation, NCBI
  annotation, polyploid GWAS, and haplotype GWAS. Each runs in < 60s
  on CPU.
- `examples/notebooks/quickstart.ipynb` — rendered Manhattan walkthrough.
- README badges (version, license, Python, test-count), MDP maize EarHT
  Manhattan screenshot, quickstart pointing at hosted docs.

**Platform (Bundle C)**
- `tests/test_gpu_model_parity.py` — CPU↔GPU FP64 `allclose` tests for
  every public model family (SingleTraitLMM, MultiTraitLMM,
  MultiKernelLMM, MultiEnvLMM, GxELMM, FarmCPU, BLINK, HaplotypeGWAS,
  RandomRegressionLMM). CUDA-gated; auto-skips on CPU-only CI.
- `.github/workflows/ci.yml` — new `test-no-openmp` matrix entry that
  sets `TORCHGENOMICS_DISABLE_OPENMP=1` at build + test time.
- `.github/workflows/gpu.yml` — scheduled + manual-dispatch GPU job
  on GitHub cloud GPU runners; runs the parity tests plus a smoke CLI
  test.
- `.github/workflows/wheels.yml` — cibuildwheel for `manylinux2014_x86_64`,
  `win_amd64`, `macos_arm64` on CPython 3.10/3.11/3.12. Publishes to
  PyPI on release tag with OIDC trusted publishing. Coexists with the
  existing sdist publisher via `skip-existing: true` — Windows users
  without MSVC can now `pip install torchgenomics` and get the native-path
  speedups.
- `pyproject.toml` — new `docs` optional-dependency group and
  `[tool.cibuildwheel]` block.

**Coverage (Bundle D)**
- `tests/test_scan.py` (15 tests) — UnifiedScanner chunk-boundary
  sweep at N ∈ {1023, 1024, 1025, 2047, 2048, 2049}; CPU/CUDA device
  handoff; `merge_scan_results` preserves the dynamic `_conditional`
  attribute ConditionalLMM attaches (regression gate for the R-wrapper
  bug).
- `tests/test_results.py` (16 tests) — `ScanResult` serialisation
  round-trips: `to_dict` / `to_dataframe` / `to_tsv` / `to_parquet`,
  multi-trait `beta` expansion to `beta_1..beta_d`, CUDA device safety,
  pyarrow-absent `ImportError` clarity.
- `tests/test_multiomics_e2e.py` (8 tests) — planted-mediator recovery
  across 5 seeds (`a=0.8, b=0.7, n=300`); null-pair empirical FDR
  bounded at 0.15 over 10 seeds; coloc-prefilter canaries.
- `tests/test_annotate_cassette.py` (6 tests) + `tests/fixtures/
  ncbi_cassette/maize_chr1_one_hit.json` — replays a recorded end-to-
  end NCBI exchange against the **live** Datasets v2 schema
  (`annotations[].genomic_locations[]` + `gene_ontology` buckets), so
  the `_ensure_legacy_*` shape adapters are actually exercised. Two
  schema-rename canaries prove the test fails loudly if upstream drifts.

**Correctness (Bundle A)**
- `__version__` now tracks `pyproject.toml`.
- `tests/test_golden_gemma.py` — 9 tests against MDP EarHT + (EarHT,
  dpoll) using GEMMA 0.98.5's own kinship + per-SNP Wald/LRT/Score.
  Tolerances calibrated from measured reality.
- `scripts/generate_golden_data.py` + `.R` companion — reproducible
  regeneration harness for GEMMA (Python) and GAPIT3 / GWASpoly (R).
- `.github/workflows/ci.yml` — `pytest -m golden` job gates the
  "4th-decimal agreement with GEMMA" claim on every CI run.

**Roadmap**
- `docs/ROADMAP.md` — deferred internal improvements (CLI decomposition
  of the 4382-line `cli.py`, `csrc/common/` pybind consolidation across
  24 extensions, `pytest-mpl` visual-regression harness) plus Phase 50+
  feature candidates (imputation-scan, sex-chromosome, admixture LMM,
  rare-variant tier, CRAN submission).

### Fixed

- `torchgenomics.scan.unified.merge_scan_results` — now propagates the
  dynamic `_conditional` attribute that `ConditionalLMM` attaches to
  each per-chunk `ScanResult`. Closes the R-wrapper bug where
  `gwas_conditional()` silently lost LD-block metadata for scans
  straddling a chunk boundary (> chunk_size SNPs).

### Changed

- `torchgenomics.models.base.ScanResult` — new canonical `to_dict`,
  `to_dataframe`, `to_tsv`, `to_parquet` methods. Replaces the
  per-CLI-subcommand hand-rolled DataFrame construction pattern;
  multi-trait `beta` automatically expands into `beta_1..beta_d` /
  `se_1..se_d` columns.

### Tests

- Suite goes from 2192 pass / 45 skip (v0.1.1) to **2246 pass / ~42
  skip** with the new Bundle D coverage, GEMMA-golden harness, and
  CUDA-parity scaffolding.

## [0.1.1] — 2026-04-15

### Added
- `torchgenomics.annotate` — NCBI gene annotation for GWAS hit SNPs (Phase 48). Resolves
  crop/taxid/assembly, fetches overlapping genes per hit in a configurable window,
  attaches descriptions, GO terms, and optional orthologs. `torchgenomics annotate` CLI.
- `torchgenomics.multiomics` — GRM-corrected causal mediation + multi-kernel heritability
  (Phase 49). `mediate_lmm` (single-triple, Sobel / Monte-Carlo / bootstrap SE, Imai
  ρ-sensitivity), `scan_mediation` (cis-window filter + BH/BY/Storey FDR),
  `mkernel_h2` (genotype GRM + regulatory-state kernel partition),
  `build_expression_kernel`. Two new CLI subcommands (`mediate`, `mediate-scan`)
  bring the CLI surface from 32 to 35 subcommands.

### Fixed
- `torchgenomics.annotate` region queries now use E-utilities `esearch` with
  `[Base Position]`. The Datasets v2 `annotation_report` endpoint silently ignored
  `chromosomes`/`start`/`stop` filters and returned the whole assembly.
- `torchgenomics.annotate` gene details now route through `POST /gene` (the
  `POST /gene/id` path was removed). Live response shape changes (coordinates moved
  to `annotations[0].genomic_locations[].genomic_range`; GO terms to
  `gene_ontology.{processes,functions,components}`) are translated back to the
  legacy shape via `_ensure_legacy_genomic_ranges` / `_ensure_legacy_ontology_terms`.
- `torchgenomics.io.numeric` — marker-axis orientation now uses the companion `.map` /
  `.bim` file's SNP IDs as a ground-truth disambiguator when both row and column
  IDs are non-numeric. Fixes misclassification of GAPIT-style `mdp_numeric.txt`.

### Dependencies
- Added `requests>=2.28` for NCBI HTTP I/O.

## [0.1.0] — 2026-04-09

Initial alpha release. Phases 0-47 complete (2102 tests, 44 skipped).
