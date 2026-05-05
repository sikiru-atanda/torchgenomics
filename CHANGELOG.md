# Changelog

All notable changes to this project will be documented in this file.

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
  sets `TORCHGWAS_DISABLE_OPENMP=1` at build + test time.
- `.github/workflows/gpu.yml` — scheduled + manual-dispatch GPU job
  on GitHub cloud GPU runners; runs the parity tests plus a smoke CLI
  test.
- `.github/workflows/wheels.yml` — cibuildwheel for `manylinux2014_x86_64`,
  `win_amd64`, `macos_arm64` on CPython 3.10/3.11/3.12. Publishes to
  PyPI on release tag with OIDC trusted publishing. Coexists with the
  existing sdist publisher via `skip-existing: true` — Windows users
  without MSVC can now `pip install torchgwas` and get the native-path
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

- `torchgwas.scan.unified.merge_scan_results` — now propagates the
  dynamic `_conditional` attribute that `ConditionalLMM` attaches to
  each per-chunk `ScanResult`. Closes the R-wrapper bug where
  `gwas_conditional()` silently lost LD-block metadata for scans
  straddling a chunk boundary (> chunk_size SNPs).

### Changed

- `torchgwas.models.base.ScanResult` — new canonical `to_dict`,
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
- `torchgwas.annotate` — NCBI gene annotation for GWAS hit SNPs (Phase 48). Resolves
  crop/taxid/assembly, fetches overlapping genes per hit in a configurable window,
  attaches descriptions, GO terms, and optional orthologs. `torchgwas annotate` CLI.
- `torchgwas.multiomics` — GRM-corrected causal mediation + multi-kernel heritability
  (Phase 49). `mediate_lmm` (single-triple, Sobel / Monte-Carlo / bootstrap SE, Imai
  ρ-sensitivity), `scan_mediation` (cis-window filter + BH/BY/Storey FDR),
  `mkernel_h2` (genotype GRM + regulatory-state kernel partition),
  `build_expression_kernel`. Two new CLI subcommands (`mediate`, `mediate-scan`)
  bring the CLI surface from 32 to 35 subcommands.

### Fixed
- `torchgwas.annotate` region queries now use E-utilities `esearch` with
  `[Base Position]`. The Datasets v2 `annotation_report` endpoint silently ignored
  `chromosomes`/`start`/`stop` filters and returned the whole assembly.
- `torchgwas.annotate` gene details now route through `POST /gene` (the
  `POST /gene/id` path was removed). Live response shape changes (coordinates moved
  to `annotations[0].genomic_locations[].genomic_range`; GO terms to
  `gene_ontology.{processes,functions,components}`) are translated back to the
  legacy shape via `_ensure_legacy_genomic_ranges` / `_ensure_legacy_ontology_terms`.
- `torchgwas.io.numeric` — marker-axis orientation now uses the companion `.map` /
  `.bim` file's SNP IDs as a ground-truth disambiguator when both row and column
  IDs are non-numeric. Fixes misclassification of GAPIT-style `mdp_numeric.txt`.

### Dependencies
- Added `requests>=2.28` for NCBI HTTP I/O.

## [0.1.0] — 2026-04-09

Initial alpha release. Phases 0-47 complete (2102 tests, 44 skipped).
