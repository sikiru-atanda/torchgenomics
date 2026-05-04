# TorchGWAS Validation Campaign — Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, with two-track agent review)
- **Date:** 2026-04-30
- **Living spec.** Pillar A is detailed; Pillars B/C/D are outlined and will be fleshed out at each C2 checkpoint.

---

## 1. Goal

Drive every public function in `torchgwas/` through tier-appropriate validation — math correctness, behavior, or smoke — and benchmark every claim of reference equivalence against a fresh install of the actual reference tool on a public dataset. Findings get fixed (V1 core) or ledgered (post-V1) per a strict severity tier.

## 2. Pillars (sequenced)

| Order | Pillar | Scope | Bar |
|---|---|---|---|
| 1 | **A — Coverage audit + fill** | Every public symbol in `torchgwas/` | Tiered: math / behavior / smoke (§4) |
| 2 | **B — Reference-tool comparisons** | All nine reference tools (§5) on public data — three already-wired (GEMMA, GAPIT, GWASpoly) re-housed in the new harness layout, plus six newly added (PLINK 2.0, LDSC, regenie, SAIGE, TwoSampleMR, BOLT-LMM) | 4th-decimal where contracted; documented otherwise |
| 3 | **C — End-to-end CLI smoke matrix** | Every CLI subcommand × every supported genotype format × CPU (and GPU where available) | Exit 0 + output sanity invariants |
| 4 | **D — Reproducibility audit** | Re-run committed golden fixtures against fresh upstream installs | Drift between fixture and fresh tool ≤ tolerance |

## 3. Working model

- **Approach 2 — Living-spec iterative.** Pillar A executes in detail per this spec; pillars B/C/D get just-in-time detail at each C2 checkpoint via `superpowers:writing-plans`.
- **One branch per pillar:** `validation/pillar-A-coverage`, `validation/pillar-B-references`, `validation/pillar-C-cli-smoke`, `validation/pillar-D-reproducibility`. Each branched off latest `master` at start of pillar.
- **Per-pillar PR at the C2 checkpoint** — opened against `master`. Pillar branch is not merged until you approve and the next pillar starts.
- **Master findings ledger:** `docs/validation_findings.md` — append-only.
- **Artifacts:** per-pillar plans live at `docs/superpowers/plans/2026-04-30-pillar-{A,B,C,D}-plan.md`.

## 4. Pillar A — Coverage audit + fill (A4: tiered bar)

### 4.1 Module tiers

| Tier | Modules | Bar |
|---|---|---|
| 1 (math correctness) | `models/`, `linalg/`, `stats/` | Closed-form / Monte-Carlo / `scipy`-comparison test per public function |
| 2 (behavioral) | `io/`, `preprocess/`, `ld/`, `pgs/`, `postgwas/`, `multiomics/` | Golden path + 1 edge case + 1 error path per public function |
| 3 (smoke) | `viz/`, `annotate/`, `cli/` | Single happy-path import + invocation that asserts non-trivial output |

### 4.2 Coverage auditor

`scripts/audit_public_coverage.py`:
- Walks every module under `torchgwas/`.
- Determines public symbols via `__all__` if present, else top-level non-underscore-prefixed `def`/`class`.
- For each symbol, greps `tests/` for `from torchgwas...import <symbol>` or `torchgwas...<symbol>(` patterns.
- Emits `docs/validation_findings/coverage_audit.json` keyed by `module.symbol`, with fields `{tier, kind, has_direct_test, test_files, lineno}`.
- The "has_direct_test" flag is conservative: indirect / transitive use does not count.
- Audit is run once at pillar A start; output committed; revisited at end of pillar A to verify zero remaining `has_direct_test=false` rows for in-tier symbols.

### 4.3 Test-writing rules per tier

**Tier 1 — math correctness.**
- For closed-form-tractable functions (e.g., `linalg.standardize_genotype` mean=0, var=1; `stats.benjamini_hochberg` step-up rule on a hand-computed example), assert the closed form to ≤ 1e-10 absolute.
- For functions whose ground truth comes from another standard library (`scipy.stats.chi2.sf`, `scipy.linalg.eigh`, `numpy.linalg.cholesky`), call both and assert agreement to documented tolerance.
- For functions whose ground truth is a Monte-Carlo expectation (e.g., permutation null distributions, FDR control under a true null), simulate ≥ 10 000 reps and assert empirical convergence (e.g., `lambda_GC` ≈ 1.0 ± 0.05 under no signal).
- Every Tier-1 test exercises *one* function in isolation. Compositional tests (`UnifiedScanner` end-to-end) live in higher pillars.

**Tier 2 — behavioral.**
- Three cases per public function:
  - **Golden path** — typical valid input, assertion on the documented contract (return shape, key invariants).
  - **Edge case** — at least one of: empty input, single-sample input, all-missing input, single-allele input, ploidy=1, near-singular matrix, etc., chosen per function semantics.
  - **Error path** — at least one invalid input asserted to raise the documented exception type with a message substring match.

**Tier 3 — smoke.**
- One test per public function: import + invoke with the simplest valid input; assert (a) doesn't raise, (b) returns the documented type, (c) for `viz/` functions the matplotlib `Axes` object has at least one artist; for `cli/` subcommands the exit code is 0 and at least one output file exists.

### 4.4 Pillar A exit criteria

1. `coverage_audit.json` shows `has_direct_test=true` for every in-tier symbol (Tiers 1, 2, 3).
2. New tests added live under `tests/test_coverage_<module>.py` and are picked up by the default `pytest tests/` run.
3. F3 logic was applied to every divergence found during writing; findings ledger is current.
4. Code-reviewer + re-run reviewer signed off on each tier.
5. C2 summary posted; user approves before Pillar B starts.

## 5. Pillar B — Reference-tool comparisons

Six tools, all wired up. Heavy installs (SAIGE, BOLT-LMM, regenie) get a memory pre-flight check before each run.

| Tool | Version target | TorchGWAS counterparts | Public dataset(s) |
|---|---|---|---|
| **GEMMA** | 0.98.5 (already pinned) | `SingleTraitLMM`, `MultiTraitLMM`, `linalg.kinship` | MDP maize (committed) + 1000G chr22 |
| **GAPIT3** | 3.4 (already pinned) | `FarmCPU`, `BLINK`, `GLM` | MDP maize + 1000G chr22 |
| **GWASpoly** | 2.12 (already pinned) | `poly-scan` (5 gene-action models) | tetraploid potato (committed) |
| **PLINK 2.0** | latest stable | `BinaryGLM`, `linalg.kinship` (`--make-rel`), `ld.r2_pairwise` (`--r2`), `ld_blocks` (`--blocks`) | 1000G chr22 |
| **LDSC** | 1.0.1 | `postgwas.ldsc_h2`, `postgwas.ldsc_rg`, `postgwas.s_ldsc` | GIANT 2018 height + UK Biobank LD scores |
| **regenie** | 3.x | `lmm-scan`, `BinaryGLMM` (Step 1 + Step 2 SPA) | UK Biobank-format synthetic + 1000G chr22 |
| **SAIGE** | 1.x | `BinaryGLMM`, `OrdinalGLMM`, `survival-scan` | 1000G chr22 + simulated case-control |
| **TwoSampleMR** (R) | 0.5+ | `postgwas.mr.{ivw,egger,weighted_median,mr_presso}` | OpenGWAS public sumstats |
| **BOLT-LMM** | 2.4+ | `lmm-scan` (LOCO), `linalg.kinship` | 1000G chr22 |

### 5.1 Per-tool harness layout

```
validation/external/<tool>/
├── install.sh           # idempotent; pinned version; checksum-verifies binaries
├── preflight.sh         # asserts free RAM ≥ documented requirement
├── fetch_data.sh        # downloads + checksum-verifies public dataset(s)
├── run_<tool>.sh        # produces reference output
├── compare.py           # parses outputs, asserts tolerance
├── README.md            # exact reproduction recipe
└── data/                # downloaded; .gitignored
```

### 5.2 Pillar B exit criteria

1. All nine tool harnesses (counting GEMMA/GAPIT/GWASpoly which already exist — they get folded into the new layout) have working `install.sh` + `run_<tool>.sh` + `compare.py`.
2. Each comparison runs and reports either pass-within-tolerance or a ledgered post-V1 finding.
3. `tests/test_external_<tool>.py` registered with new `external` pytest marker (registered in `pyproject.toml`'s `[tool.pytest.ini_options].markers`); skipped by default; runs with `pytest -m external`.
4. Findings ledger entries for every divergence beyond tolerance.
5. C2 summary posted; user approves before Pillar C starts.

### 5.3 Memory & disk pre-flight (mandatory for every download / run)

Before each `fetch_data.sh` and each `run_<tool>.sh`:
- **Disk:** assert `df --output=avail /home` ≥ (downloaded size × 2 + working-set size × 2).
- **RAM:** assert `free` available ≥ (estimated peak in-RAM working set × 1.5 + 4 GB headroom). Each harness records its peak-memory estimate in its `README.md` (e.g., regenie Step 1 on 1000G chr22 ≈ 6 GB; SAIGE Step 1 ≈ 8 GB; LDSC ≈ 2 GB) — the estimates are revised after the first successful run.
- If either check fails, abort with explicit message — do not download / run partially.
- The check is a shell function in `validation/external/_lib/preflight.sh` and is sourced by every per-tool harness.

This is the user's hard rule — encoded as a contract, not a comment.

### 5.4 Pillar B execution order (post Pillar A close)

Pillar A surfaced 8 V1-core / V1-platform fix-now findings (5 unimplemented stubs + 3 silent-truncation guards). With those production paths now correct, Pillar B comparisons are unblocked for the model classes that depended on them.

**Pillar B status: COMPLETE (2026-05-04).** All 11 tools wired, both R4 reviewer tracks approved. 1 V1-core fix-now production finding (mr_egger Bowden-2015 alignment, commit `9d9c114`).

| # | Tool | Commit(s) | Headline result | F3 |
|---|---|---|---|---|
| 1 | GEMMA / GAPIT / GWASpoly | `8ac3181`, `b067911`, `03c1571` | Re-housed; zero regression on goldens after Pillar A's 9 fixes | 0 |
| 2 | PLINK 2.0 | `74878f9` | β corr=1.000, GRM corr=0.99995, r² corr=1.000 | 0 |
| 3 | LDSC | `28a6340` | h² Δ<5e-3 ✓; rg Δ<5e-3 ✓; **intercept needs IRWLS port** (Phase 37 follow-up) | 0 |
| 4 | regenie | `d8bc172` | β corr 0.82–0.88 (LOCO + Y-standardize structural difference) | 0 |
| 5 | SAIGE | `d352aff` | docker (podman); β corr 0.973, -log10p 0.943 | 0 |
| 6 | TwoSampleMR (R) | `9d9c114` (F3) + `0c75070` | IVW/Egger/median agree to 5 sig figs **post-F3 fix** | **1** |
| 7 | BOLT-LMM | `4c9c372` | β corr 0.984, -log10p 0.982 (tightest LMM Pillar-B agreement) | 0 |
| 8 | SoyNAM | `dc1df1d` | STLMM vs rrBLUP corr 0.99991, kinship bit-equal, WFLMM dual-scan validates Young 2022 | 0 |
| 9 | SoyMD | `a7c0ffa` | TG `mediate_lmm` ≈ R `mediation::mediate` near-bit-equal point estimates | 0 |

**Cumulative campaign F3 fix-now findings: 9** (8 from Pillar A + 1 from Pillar B B3 mr_egger).

**Documented post-V1 divergences (NOT F3, ledgered):**
- LDSC intercept — TG single-pass WLS vs LDSC IRWLS (Phase 37 follow-up).
- regenie β corr 0.82 — regenie internally standardizes Y; TG doesn't (post-V1 design choice).
- BOLT-LMM β max |Δ|=3.79 single-SNP outlier on small N — BOLT's documented "ratio of medians" calibration weakness.
- MR-PRESSO null distribution — TwoSampleMR uses parametric LOO bootstrap; TG uses permutation null.
- MR weighted-median SE — parametric bootstrap (TG) vs non-parametric (TwoSampleMR).
- SAIGE β corr 0.973 / -log10p 0.943 — PCG (SAIGE) vs PQL (TG) for null model fit.

## 6. Pillar C — End-to-end CLI smoke matrix

**Status: COMPLETE (2026-05-04).** All 40 CLI subcommands wired into a parametric smoke matrix. Both R4 reviewer tracks approved.

**Implementation:**
- `tests/cli_matrix_spec.py` — `CELL_SPEC` dict with one entry per CLI subcommand (40 entries; coverage parity test asserts no drift vs live `--help`).
- `tests/test_cli_matrix.py` — parametrized executor over `(subcommand, format, device)`; per-kind output assertions (scan TSV p-columns in [0,1], JSON schema for validate, magic bytes for converted files, dosage tensor for impute outputs, etc.).
- New `cli_matrix` pytest marker registered in `pyproject.toml`; default suite (`pytest tests/`) skips the matrix unless opted-in via `pytest -m cli_matrix`.
- `tests/conftest.py` extended `pytest_collection_modifyitems` hook to gate both `external` and `cli_matrix` markers.

**Matrix shape:** 122 tests (108 CPU cells + 13 GPU cells gated on `torch.cuda.is_available()` + 1 coverage self-check). 114 passed, 8 skipped (documented `skip_reason`: `dosage-call` needs updog R + AD-format VCF; `phase-poly` needs PolyOrigin Julia binary; `ldsc` and `ldsc-rg` underdetermined on m=20 / n=10 tiny fixture), 0 failed.

**4 V1-platform F3 fix-now findings surfaced:**
| # | Finding | Fix commit |
|---|---|---|
| 10 | `glmm-scan` / `me-glmm-scan` / `survival-scan` crashed on `from .linalg.grm import grm` (non-existent module) | `ccf1132` |
| 11 | `gxe-scan` / `mvlmm-scan` / `me-glmm-scan` multi-output result schema bug in `_apply_correction_and_save` | `ccf1132` |
| 12 | `impute` crashed on `_load_genotype_matrix` mis-unpacking reader chunks (`chunk.dosage` vs tuple) | `adc7b04` |
| 13 | `lmm-scan --device cuda` crashed: `_cmd_lmm_scan_single` left Y/X0 on CPU when K was on CUDA | `adc7b04` |

**Cumulative campaign F3 fix-now findings: 13** (Pillar A: 8, Pillar B: 1, Pillar C: 4).

## 7. Pillar D — Reproducibility audit (outline)

Detailed at the Pillar C → D checkpoint. Sketch:
- `validation/reproducibility/rerun_goldens.py` — for each committed reference fixture, invoke the matching Pillar B harness and diff the *fresh* tool output against the *committed* fixture (not against TorchGWAS).
- A drift between committed fixture and fresh tool output is logged as a `fixture-drift` finding (distinct from a `torchgwas-divergence` finding).
- Outcome: either the fixtures are still authoritative, or we have a list of fixtures that need regeneration.

## 8. Reviewer agent loop (R4: two-track per tier)

### 8.1 Track 1 — Code quality

**Agent:** `superpowers:code-reviewer`.
**Trigger:** after I commit a tier's tests + any F3-fix commits.
**Prompt template:** `docs/superpowers/reviewer_prompts/code_review.md`. Filled in per-tier with: diff range (`git diff <tier-start>..HEAD`), spec section reference (this file §4 / §5 / §6 / §7), tolerance contract excerpt from `docs/validation.md`, F3 severity guide.
**Output:** blockers / nits / out-of-scope. I address blockers, commit, re-run if structural changes were made.

### 8.2 Track 2 — Re-run verification

**Agent:** `general-purpose` (because it actually installs software and downloads data).
**Trigger:** after Track 1 passes.
**Prompt template:** `docs/superpowers/reviewer_prompts/rerun_verification.md`. Self-contained: install tool X version Y, download dataset Z (URL, checksum), run comparison Q, report whether the diff is within tolerance T. Includes the memory pre-flight contract from §5.3.
**Output:** pass / fail / infra-blocker classification + a short report. Findings appended to `docs/validation_findings.md` by me, not the agent.
**Cost control:** Track 2 only runs once per tier. It does NOT re-run for every test in the tier. Sample = the most representative test in the tier.

### 8.3 Templates committed up front

The two prompt templates at `docs/superpowers/reviewer_prompts/` are committed before Pillar A starts. They are the contract that ensures every review across the campaign is consistent.

## 9. Failure handling (F3: severity-tiered)

| Finding | Action |
|---|---|
| V1-core (Phases 0–13) math error | Fix in same tier; commit fix + regression test; re-run reviewer to confirm |
| V1-core regression on existing golden | **Halt pillar (C4)**; flag in findings; wait for guidance |
| Post-V1 divergence beyond tolerance | Document; `pytest.mark.xfail(strict=True)` with reason; continue |
| New test exposes a missing-feature charter claim | Document; create follow-up issue; do not implement; continue |
| Tooling failure (install / dataset / env) | Retry once; document as `infra-blocker`; skip; continue pillar |
| Three+ unrelated post-V1 divergences in one tier | **Halt pillar (C4)**; smells systemic |

Hard rules:
- **No skipping V1-core.** A V1 finding is fixed within the tier or the pillar halts.
- **xfail is not the default for post-V1.** Try to fix first; fall back to xfail only if the fix is larger than a tier's worth of work.

## 10. Checkpoints (C2 + C4)

**C2 — Per-pillar (4 scheduled stops).** At the end of each pillar:
- Open the per-pillar PR against `master`.
- Post a conversation summary: tests added, divergences classified, fixes shipped, what's deferred, time elapsed, costs.
- Update this spec's pillar-N+1 section from outline → detailed.
- Wait for user approval before opening the next pillar's branch.

**C4 — Emergency stop (unscheduled).** Triggered by:
- Any V1-core fix-immediately divergence.
- Finding that suggests architectural rework (e.g., GRM definition mismatch with GEMMA).
- Three+ unrelated post-V1 divergences in a single tier.

Behavior: post finding, do NOT start the next tier, wait for guidance.

## 11. Findings ledger schema

`docs/validation_findings.md`:

```
| Date       | Pillar | Tier | Module / Function           | Reference   | Dataset         | Δ observed | Tolerance | F3 class            | Resolution               |
|------------|--------|------|-----------------------------|-------------|-----------------|------------|-----------|---------------------|--------------------------|
| 2026-05-01 | A      | 1    | linalg.kinship_polyploid    | closed-form | synthetic-tetra | 1.4e-3     | 1e-6      | V1-core / fix-now   | commit abc123 fixes …    |
| 2026-05-02 | B      | -    | postgwas.ldsc_h2            | LDSC 1.0.1  | UKB chr22 LD    | 0.018      | 0.05 abs  | post-V1 / documented | issue #N opened          |
```

Every entry has a populated Resolution column at C2 checkpoint time.

## 12. Public datasets

| Dataset | Size on disk | Purpose | License |
|---|---|---|---|
| MDP maize | committed (~1 MB) | LMM / mvLMM goldens (already in repo) | public (Goodman lab) |
| Tetraploid potato | committed (~5 MB) | GWASpoly goldens (already in repo) | public (USDA) |
| 1000 Genomes Phase 3 chr22 | ~1.2 GB | GEMMA / PLINK 2 / regenie / BOLT / SAIGE comparisons | public domain |
| 1000 Genomes Phase 3 (all chr) | ~16 GB | optional full-genome regression | public domain |
| GIANT 2018 height sumstats | ~50 MB | LDSC h² / TwoSampleMR | open access (CC0) |
| UK Biobank LD scores (precomputed) | ~5 GB | LDSC | open (Bulik-Sullivan) |
| OpenGWAS sumstats subset | varies | TwoSampleMR | open (MR-Base) |
| SoyNAM (soybean NAM) | ~500 MB–1 GB | Within-family LMM (Phase 23, Young et al. 2022), conditional LMM, multi-trait LMM, BLINK / FarmCPU on real ag data with explicit family structure | public; install via R `install.packages("SoyNAM")` or upstream website. Diploid (2n=40), ~5000 RILs across multiple families crossed to a common parent, ~5000 SNPs, multiple phenotypes (yield, plant height, days to flower, etc.) |
| SoyMD (Soybean Multi-omics DB) | varies (web platform; per-cohort downloads ~hundreds of MB–GB) | `multiomics.{mediate_lmm, scan_mediation, mkernel_h2, build_expression_kernel, eigenmt_adjust, coloc_prefilter_pairs}` (Phase 49) — causal mediation and TWAS-style validation against published soybean expression + metabolomics + phenotype layers. Also: gene-expression-kernel multi-kernel heritability (`mkernel_h2`) and eigenMT FDR (`eigenmt_adjust`) on real eQTL-scale data. | public; web platform plus per-publication download links. Multi-omics dimensions (genome / transcriptome / proteome / metabolome) make it the natural Pillar B fixture for the multiomics module which has no current external comparison. |

Memory budget verified at spec time: 62 GB RAM (54 free), 1.6 TB free on /home, 16 GB VRAM. Headroom for all of the above resident simultaneously, but the §5.3 pre-flight check is run before each download/run regardless.

**SoyNAM access plan (Pillar B execution-time detail):** the R package is the simplest entry point — extract genotype + phenotype to TSV/CSV via a small `validation/external/soynam/fetch_data.R` script, then load into the per-tool harnesses. Memory: phenotype ~5 MB, genotypes ~500 MB in the dense torch.float64 case but well under the 8 GB pre-flight floor. Family ID column is the natural fixture for `family-scan` (within-family LMM) regression equivalence vs. the SoyNAM authors' published QTL hits.

## 13. Exit criteria for the whole campaign

1. Pillars A / B / C / D complete per their per-pillar exit criteria.
2. `docs/validation_findings.md` current; every entry has a populated Resolution column; F3-classification consistent.
3. `CLAUDE.md` updated to point at the validation campaign and explain how to re-run any pillar.
4. `docs/validation.md` updated with the new external-tool comparison results.
5. New pytest marker `external` registered.
6. Per-pillar PRs all merged.

## 14. Non-goals

- No refactor of TorchGWAS production code beyond what F3 requires.
- No new feature scope. (No "while I'm here, let me add BSLMM.")
- No public API changes.
- No CI runtime increase from default `pytest tests/` — the `external` and `golden` markers stay opt-in.

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pillar A balloons (audit reveals hundreds of untested public functions) | Tiered bar (A4) caps Tier 3 work; can split A across multiple sub-checkpoints if it overruns |
| External tool installs are flaky on RHEL 9 (host OS) | One retry per F3; documented as infra-blocker; harness still committed |
| Re-run reviewer agent's environment differs from main session | That's the point — it's the test of independence. Findings flag the divergence regardless. |
| Validation campaign discovers a fundamental math error in Phase 0–13 | C4 emergency-stop; fix is high priority; campaign pauses until resolved |
| Living spec drifts from implementation | At each C2 checkpoint, the next-pillar section is rewritten from outline → detailed before work starts |

---

**Approval gate.** This is a brainstorming-phase spec. After user review, the next step is `superpowers:writing-plans` to produce `docs/superpowers/plans/2026-04-30-pillar-A-plan.md` for Pillar A only. Pillars B/C/D plans are written at their respective C2 checkpoints, not now.
