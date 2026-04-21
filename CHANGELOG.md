# Changelog

All notable changes to this project will be documented in this file.

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
