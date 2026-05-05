# Streaming-vs-Materialized Scan Audit (Efficiency E0)

**Date:** 2026-04-30
**Branch:** `efficiency/streaming-scan-audit`
**Scope:** every CLI subcommand registered in `torchgwas/cli.py` that ingests
genotype data, classified by whether it streams chunks through the scan loop
or materializes the full `(n_samples × n_variants)` genotype matrix at any
point.

## Method

For each subcommand, we traced the call path from `_cmd_<name>` → adapter →
underlying model. Three classifications are used:

- **streaming** — never holds the full G in memory. Uses
  `aligned_reader.iter_chunks(...)` directly (often via `UnifiedScanner`,
  which is the canonical streaming consumer). GRM, when needed, is built by
  `grm_vanraden_streaming(_impute_chunk_iter(...))` — this still iterates
  chunks.
- **partial** — streams the genome scan loop, but materializes the full G
  exactly once (typically for kinship construction or knockoff/LD-block
  generation), then can free it before the per-chunk scan.
- **materialized** — calls `_load_full_genotype` / `_load_genotype_matrix`
  / `torch.cat([chunk for chunk in iter_chunks])` to produce a single
  `(n, m)` tensor that lives for the duration of the scan.

Peak memory at biobank scale (UKB-like: `n=500_000`, `m=10_000_000`) is
estimated as `n × m × 4 B = 20 TB` for `float32` dosage; `× 8 B = 40 TB` for
`float64`. The internal `STAT_DTYPE` is `float64`, so all paths that hold G
in `STAT_DTYPE` see the full 40 TB unless they keep the input dtype on the
materialized tensor (PLINK BED reader yields `float32` by default; `_load_*`
helpers always upcast to `STAT_DTYPE = float64`).

A more realistic ag-panel scale (`n=10_000`, `m=2_000_000`) puts the same
cost at ~160 GB float64 — already above commodity-server RAM.

## Helper inventory

| Helper | Defined at | Materializes? | Used by |
|---|---|---|---|
| `_load_full_genotype` | `cli.py:3007` | yes — calls `iter_chunks(chunk_size=n_variants)` then upcasts to `STAT_DTYPE` (float64) and mean-imputes | `_load_scan_data` (a wrapper) and several scan commands directly |
| `_load_scan_data` | `cli.py:3023` | yes (delegates to `_load_full_genotype`) | `set-scan`, `bayes-scan`, `met-scan`, `farmcpu-scan`, `blink-scan`, `threshold-scan`, `mklmm-scan`, `gxe-scan`, `knockoff-scan` (via `_cmd_knockoff_scan`), `gu-scan` (via `_cmd_gu_scan_inner`), `lro-scan`, `glmm-scan`, `me-glmm-scan`, `survival-scan`, `poly-scan`, `pipeline` (FarmCPU/BLINK/MKLMM/GxE/MET branches) |
| `_load_genotype_matrix` | `cli.py:2828` | yes (`torch.cat(chunks)`) | `impute` (mean / mode / knn / ld / li-stephens / deep-learning) |
| `_align_samples` | `cli.py` | NO — returns aligned reader only | streaming-friendly callers |
| `grm_vanraden_streaming` | `linalg/kinship.py` | NO — accumulator over `iter_chunks` | `lmm-scan` (default), `mvlmm-scan`, `glm-scan` (PC route), `conditional-scan`, `mtmet-scan`, `ocf-scan`, `family-scan`, `pipeline` (LMM/mvLMM branches) |

`_load_full_genotype` is structurally a streaming-killer for any call site
that uses it: it requests `chunk_size = n_variants`, so the BED/VCF/Zarr
reader produces a single tensor of the entire matrix, then the helper
unconditionally upcasts to `STAT_DTYPE` (float64) and mean-imputes. The
`del G` in some callers comes too late — peak memory has already happened.

## CLI subcommand audit

| CLI subcommand | Adapter / model | Path | Peak mem at 500K × 10M (float64) | Notes |
|---|---|---|---|---|
| **validate** | (none — only reads metadata) | streaming | < 1 GB | Inspects header only |
| **convert** | `io.convert.convert` | format-dependent | varies | Most format conversions stream; zarr/bed paths chunk |
| **impute** (mean/mode/knn/ld) | `_load_genotype_matrix` | **materialized** | 40 TB | `torch.cat` of all chunks; needed to compute global stats. KNN also builds GRM after — additional `n²` cost. |
| **impute** (li-stephens / deep-learning) | `_load_genotype_matrix` | **materialized** | 40 TB | GPU model needs full G; algorithmic |
| **impute** (beagle / impute5 / minimac4) | external wrapper | streaming (subprocess) | bounded by external tool | TG only forwards files |
| **dosage-call** | `preprocess.dosage_call` | streaming (per-variant updog) | < 1 GB | updog is per-variant; the pipeline reads VCF lazily |
| **phase-poly** | `preprocess.polyploid_phase` | streaming (PolyOrigin Julia) | bounded by external tool | TG only orchestrates probs.pt + map |
| **glm-scan** | `GLM` / `BinaryGLM` / `OrdinalGLM` / `MultinomialGLM` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` for PCs (when `--n-pcs > 0`) |
| **lmm-scan** (default vanraden) | `SingleTraitLMM` via `UnifiedScanner` | streaming | < 1 GB | streaming GRM, streaming scan |
| **lmm-scan** (`--grm-method zhang`) | `SingleTraitLMM` via `UnifiedScanner` | **partial** | 40 TB once (then `del G`) | `grm_zhang` requires full G; scan after GRM is streaming. Only used when user explicitly opts in. |
| **mvlmm-scan** (default vanraden) | `MultiTraitLMM` via `UnifiedScanner` | streaming | < 1 GB | mirrors lmm-scan |
| **mvlmm-scan** (`--grm-method zhang`) | `MultiTraitLMM` via `UnifiedScanner` | **partial** | 40 TB once | same pattern as lmm-scan |
| **poly-scan** | `SingleTraitLMM` (recoded G) via `UnifiedScanner` | **materialized** | 40 TB | `_load_scan_data` materializes G; gene-action recoding is per-model in-memory; scan loop itself is streaming via UnifiedScanner but reads original G into memory first via the held reader. |
| **mklmm-scan** | `MultiKernelLMM` via `UnifiedScanner` | **materialized** | 40 TB | `build_multi_kernels` needs full G to build dominance/epistatic kernels (algorithmic) |
| **gxe-scan** | `HetLMM` via per-chunk loop + `_merge_gxe_results` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` + per-chunk score_chunk loop. `_merge_gxe_results` handles GxEScanResult concatenation (UnifiedScanner's default merger only knows ScanResult). **Streamed in E3 (commit 6fc64f7).** |
| **set-scan** | `SetBasedScanner` (custom; not UnifiedScanner) | **materialized** | 40 TB | calls `_load_scan_data` then `scanner.scan_regions(G_full, ...)`. Algorithmically per-region; can stream chunks and accumulate into per-region tensors. **REWRITE TARGET (E1).** |
| **bayes-scan** | `BayesianVS.fit` | **materialized** | 40 TB | SuSiE / CAVI need full G for joint variable selection (algorithmically) |
| **met-scan** | `MultiEnvLMM.score_chunk` (manual loop) | **materialized** | 40 TB | `_load_scan_data` materializes; subsequent scan does loop `iter_chunks` but G is already held; GRM uses `grm_vanraden(G)` not streaming. **DEFERRABLE — GRM-only fix would push to partial.** |
| **farmcpu-scan** | `FarmCPU` via `UnifiedScanner` | **materialized** | 40 TB | FarmCPU iteratively re-uses G as covariates; documented as needing full G |
| **blink-scan** | `BLINK` via `UnifiedScanner` | **materialized** | 40 TB | BLINK LD-clusters across all SNPs; documented as needing full G |
| **threshold-scan** | `ThresholdLinearModel` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples`; ThresholdLinearModel null fit only needs Y/X0/R/G_cov (no kinship). **Streamed in E3 (commit e8a0d3c).** |
| **conditional-scan** | `ConditionalLMM` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming`; per-peak random access into reader is via index-based reload (acceptable cost) |
| **mtmet-scan** | `MultiTraitMultiEnvLMM` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` |
| **ocf-scan** | `OCFLMM` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` |
| **knockoff-scan** | `KnockoffLMM.run` | **materialized** | 40 TB | Knockoff construction is per-LD-block; `KnockoffLMM.run` operates on full G. Could stream chunks within natural LD-block boundaries (two-pass: (1) detect blocks from streaming r² windows, (2) stream chunks and build knockoffs per-block on the fly), but the algorithm fundamentally needs all SNPs visible to detect blocks first. **DEFERRABLE — block-based streaming non-trivial.** |
| **gu-scan** | `GULM.score_chunk` (manual chunk loop with paired dvar slice) | streaming (G) / held (dvar) | < 1 GB for G; ~ n×m×8 B for user-supplied dvar | `_align_samples` + `grm_vanraden_streaming`; G streamed via iter_chunks, dvar held once at user dtype and sliced per chunk by running col_offset. **Streamed in E3 (commit 34c1e28).** |
| **lro-scan** | `LROLMM.run` | **materialized** | 40 TB | LRO needs full G + LD blocks + per-block GRM updates; algorithmically tight |
| **family-scan** | `WithinFamilyLMM` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` |
| **glmm-scan** | `BinaryGLMM` / `OrdinalGLMM` / `MultinomialGLMM`.score_chunk | **materialized** | 40 TB | `_load_scan_data` + single `score_chunk(G_full, ...)`. PQL null fit only needs K + Y + X0; scan loop is per-variant. **REWRITE TARGET (E1).** |
| **me-glmm-scan** | `MultiEnvGLMM.score_chunk` via per-chunk loop + `_merge_env_results` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` + per-chunk score_chunk loop; `_merge_env_results` (the same merger met-scan uses) handles EnvScanResult concatenation. **Streamed in E3 (commit 4fafb74).** |
| **survival-scan** | `SurvivalGLMM.score_chunk` via `UnifiedScanner` | streaming | < 1 GB | `_align_samples` + `grm_vanraden_streaming` + UnifiedScanner; SurvivalGLMM emits standard ScanResult. **Streamed in E3 (commit 507bf5d).** |
| **ld-blocks** | `ld.detect_blocks` | **materialized** | 40 TB | LD detection inherently needs all variants; `torch.cat` from `iter_chunks` |
| **ldsc** | `compute_ld_scores` + `ldsc_h2` | **materialized** | 40 TB | LD score computation needs all variants for windowed r² |
| **ldsc-rg** | `compute_ld_scores` + `ldsc_rg_from_z` | **materialized** | 40 TB | same pattern as ldsc |
| **meta** | post-GWAS (uses sumstats files only) | N/A | < 1 GB | no genotype I/O |
| **clump** | `ld_clump` | **materialized** | 40 TB | needs joint r² between top SNP and rest |
| **pgs-fit** | `pgs.PRSCS` / `LDpred2*` / `ClumpingThresholding` | depends on ld-ref input | varies | sumstats-driven; LD ref usually pre-computed |
| **pgs-score** | `pgs.score_individuals` | streaming (per-SNP weight × dosage) | < 1 GB | scores accumulate while iter_chunks |
| **pipeline** (glm/lmm/mvlmm) | streaming branch | streaming | < 1 GB | uses `_align_samples` + `grm_vanraden_streaming` |
| **pipeline** (farmcpu/blink/mklmm/gxe/met) | materialized branch | **materialized** | 40 TB | calls `_load_scan_data`; commented in code as "models that need full G" |
| **rr-scan** | `RandomRegressionLMM` / `SpatioTemporalRR` | **materialized** | 40 TB | `_load_full_genotype` then `grm_vanraden(G)`; scan loop is streaming via `iter_chunks`, but G_full is held |
| **rr-met-scan** | `RandomRegressionMultiEnvLMM` | **materialized** | 40 TB | same shape as rr-scan |
| **annotate** | NCBI Datasets v2 | N/A (network) | < 1 GB | no genotype I/O |
| **mediate** | `multiomics.mediate_lmm` | **input-shape dependent** | < 1 GB | takes ndarrays directly; not a scan |
| **mediate-scan** | `multiomics.scan_mediation` | **materialized** | 40 TB | `_load_full_genotype` (via `_load_scan_data`); GPU-batched scan |

**Total subcommands surveyed:** 40.

**Streaming (post-E1+E3):** 20 — `validate`, `glm-scan`, `lmm-scan`
(default), `mvlmm-scan` (default), `conditional-scan`, `mtmet-scan`,
`ocf-scan`, `family-scan`, `dosage-call`, `phase-poly`, `pgs-score`,
`pipeline` (lmm/mvlmm/glm), `set-scan`, `glmm-scan` (E1), plus
`me-glmm-scan`, `survival-scan`, `threshold-scan`, `gxe-scan`,
`gu-scan` (E3), plus `meta` / `annotate` which take no genotype I/O.

**Partial (one-shot full G then freed for kinship):** 2 — `lmm-scan
--grm-method zhang`, `mvlmm-scan --grm-method zhang`. Both opt-in.

**Materialized:** the rest — 18 subcommands that call `_load_scan_data` /
`_load_full_genotype` / `_load_genotype_matrix` / `torch.cat([chunks])`.
Most are algorithmically tied to full G (FarmCPU / BLINK / BayesianVS /
MultiKernelLMM / LROLMM / KnockoffLMM / LDSC / ld-clump / ld-blocks /
rr-scan / rr-met-scan / mediate-scan / met-scan / poly-scan / impute /
pipeline-materialized-branch).

## Observations and follow-ups

### O1 — `_load_full_genotype` is the single biggest leverage point

23 of 25 materialized scans funnel through `_load_scan_data` →
`_load_full_genotype`. Reworking the helper to return a chunked iterator
instead of a flat tensor would cascade through every call site, but every
call site assumes the eager-tensor semantics. Targeted per-command
refactors are safer than a big-bang `_load_full_genotype` rewrite.

### O2 — Fundamentally not-streamable models

These genuinely need full G:

- **FarmCPU / BLINK** — iterative QTN selection, all variants needed each iter.
- **BayesianVS / SuSiE** — joint posterior over all variants.
- **MultiKernelLMM** — dominance / epistatic kernels are functions of full G.
- **LROLMM / KnockoffLMM** — LD-block detection touches all pairs.
- **LDSC, ld-clump, ld-blocks** — windowed pairwise r² across the whole genome.
- **rr-scan / rr-met-scan** — current `grm_vanraden(G_full)` is mandatory because the kinship is computed from the *aligned* G across long-format observations; a streaming GRM API could be added but it's a non-trivial sample-alignment refactor.

### O3 — Streamable, with concrete rewrite targets

These are V1-platform commonly used and have an obvious streaming
refactor that doesn't break the algorithm:

| Subcommand | Why streamable | Rewrite shape |
|---|---|---|
| **set-scan** | SKAT / Burden / SKAT-O are per-region; `Q = z^T W z` for `z = G_rot^T @ Py` accumulates chunk-by-chunk per region | Two-pass: (1) build region → variant-index map (already does this), (2) stream chunks and append `G_chunk[:, region_local_idx]` columns into per-region buffers, then run skat/burden/skat_o per buffer. Buffer size = sum of per-region variant counts (NOT full m). |
| **glmm-scan** | `BinaryGLMM.score_chunk` is per-variant once null fit is built; current code does one giant `score_chunk(G_full)` | Replace `_load_scan_data` with `_align_samples` + `grm_vanraden_streaming`, then drive scan with `UnifiedScanner` exactly like lmm-scan |
| **gu-scan** | `GULM.score_chunk` accepts dosage_var alongside G; both can be sliced per chunk | Stream G + paired dosage_var slice per chunk; merge results |
| **threshold-scan** | already uses `UnifiedScanner` for the scan loop, only the GRM/G load is eager (and threshold-scan doesn't even use a kinship — only needs Y, X0, R, G_cov for null fit). The G is only needed for `iter_chunks`. | Replace `_load_scan_data` with `_align_samples`. |

### O4 — Documentation correction

The current `CLAUDE.md` claims `lmm-scan` correctly streams; this is true.
But the same project documentation does not warn that **set-scan**,
**glmm-scan**, **gu-scan**, **threshold-scan**, **rr-scan**, etc. all
materialize G. We add a note after the rewrites of E1 are complete.

### O5 — Tier ordering for E1 rewrites

In priority of biobank impact and tractability:

1. **set-scan** — high impact (rare-variant association testing is
   biobank-flagship use case), surgical fix in `SetBasedScanner.scan_regions`.
2. **glmm-scan** — already has streaming infrastructure (UnifiedScanner +
   grm_vanraden_streaming + `BinaryGLMM.score_chunk` is per-chunk safe);
   only the CLI helper choice needs to change.
3. **threshold-scan** — same shape as glmm-scan; trivial after #2 is
   established.
4. **gu-scan** — slightly more involved because dosage_var must travel
   alongside G.

This audit selects #1 and #2 for E1 (set-scan and glmm-scan), with #3 and
#4 deferred.

### Deferred rewrites (concrete next steps)

- **threshold-scan** — replace `_load_scan_data` with `_align_samples` and
  let `UnifiedScanner` drive. ~5-line change once the pattern is set.
- **gu-scan** — wrap the genotype reader with a "paired-stream" helper that
  yields `(G_chunk, dvar_chunk, vmeta)` so per-chunk score_chunk has both.
  ~30-line change in `_cmd_gu_scan_inner`.
- **me-glmm-scan, survival-scan** — same shape as glmm-scan, mechanical.
- **gxe-scan** — needs streaming GRM (currently `grm_vanraden(G)` not
  streaming) — change line 657 in `cli.py` from `grm_vanraden` to
  `grm_vanraden_streaming` over `_impute_chunk_iter(...)`. Then full-G
  load can drop and `UnifiedScanner` drives the scan — 1-day refactor.
- **mklmm-scan, bayes-scan, knockoff-scan, lro-scan, farmcpu-scan,
  blink-scan, met-scan, mtmet-scan-met-only, poly-scan,
  rr-scan, rr-met-scan** — each needs a model-specific streaming API
  redesign. Defer indefinitely; the algorithms genuinely benefit from
  random access.

### F3 verdict

No silent-materialization bugs found: every materialized path explicitly
calls `_load_scan_data` / `_load_full_genotype`. The CLI `lmm-scan` and
`mvlmm-scan` documentation is accurate. **Not F3 fix-now**; the rewrites
in E1 are pure efficiency improvements.

## E3 ledger — 5 deferred streaming rewrites complete

Date: 2026-04-30. Rewrites the deferred set from E1's "concrete next
steps" list. All five mirror the glmm-scan template (commit `e553604`):
`_load_scan_data` swapped for `_align_samples` + (where a kinship is
needed) `grm_vanraden_streaming`, with the scan driven by
`UnifiedScanner` or a dedicated per-chunk merger when the result type
isn't `ScanResult`.

| Subcommand | Commit | Result type | Per-chunk merger |
|---|---|---|---|
| `me-glmm-scan` | `4fafb74` | `EnvScanResult` | `_merge_env_results` (shared with met-scan) |
| `survival-scan` | `507bf5d` | `ScanResult` | `UnifiedScanner` default |
| `threshold-scan` | `e8a0d3c` | `ScanResult` | `UnifiedScanner` default |
| `gxe-scan` | `6fc64f7` | `GxEScanResult` | new `_merge_gxe_results` (also fixes a latent multi-chunk crash in pre-rewrite gxe-scan) |
| `gu-scan` | `34c1e28` | `ScanResult` | `merge_scan_results`; G streamed and dvar held + sliced per chunk by running col_offset |

Memory regression tests in `tests/test_streaming_memory.py` add one
test class per rewrite (10 new tests), each guarding behavioral parity
vs. the legacy materialized path and an absolute peak budget (<16 MiB
at n=200/m=500). Total streaming-memory tests: 15 (3 set-scan + 2
glmm-scan + 10 E3).

Biobank-scale impact: 40 TB → ~ n × chunk_size × 8 B per chunk
(~4 GB per chunk at UKB scale, default chunk_size=1024).

### F3 verdict (E3)

The gxe-scan rewrite revealed a latent crash for any multi-chunk scan
(GxEScanResult ≠ ScanResult, so UnifiedScanner's default merger
crashes when m > chunk_size). The pre-rewrite implementation only
worked for tiny fixtures because the single-chunk path bypasses the
merger entirely. The streaming rewrite fixes both the materialization
regression and this latent crash with the new `_merge_gxe_results`
helper. Remaining four rewrites are pure efficiency improvements (no
behavioral regressions exposed).
