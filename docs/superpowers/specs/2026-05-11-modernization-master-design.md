# TorchGWAS Modernization Track — Master Design Spec

- **Status:** approved (brainstorming complete; awaiting user spec review)
- **Author:** Claude Code (autonomous, brainstorming-skill-driven, with parallel-research dispatch and serial sub-spec drafting)
- **Date:** 2026-05-11
- **Branch:** `modernization/specs`
- **Living spec.** Master is fixed; the four sub-specs are detailed individually and may be revisited at each implementation kickoff.

---

## 1. Goal

Close four high-impact gaps that distinguish a modern, biobank-grade, multi-ancestry GWAS toolkit from a single-ancestry / autosomal-only / quantitative-only one. The four gaps were identified during a strategic review on 2026-05-11 by surveying the Phase Index in `CLAUDE.md` against current standard-of-practice in human and crop GWAS:

1. **REGENIE step-1 + step-2 parity** — the modern biobank-default LMM workflow is absent from the toolkit.
2. **Cross-ancestry PRS / fine-mapping** — `postgwas/_multi_ancestry.py` covers MR-MEGA / MANTRA meta-analysis only; PRS-CSx and SuSiEx (the modern multi-population PRS / fine-mapping standards) are absent.
3. **Functional-annotation–informed fine-mapping** — SuSiE exists in `models/bayesian_vs.py` but lacks PolyFun-style functional priors, leaving causal-variant precision on the table.
4. **Sex-chromosome (X / Y / PAR / MT) handling** — zero hits across the codebase; a credibility gap for any human-GWAS use case.

**Scope of this master spec:** decision record + index + cross-cutting policies. Detailed designs live in the four sub-specs.

**Out of scope of this master:** algorithm equations, file formats, install procedures, fixture choices — all of those live in the per-item sub-specs.

## 2. Sub-spec index

| # | Spec | Scope | Phase number |
|---|---|---|---|
| 1 | [`2026-05-11-xchromosome-refactor-design.md`](2026-05-11-xchromosome-refactor-design.md) | Sex-chromosome handling. Full parity (additive + X-inactivation + chrY + MT + PAR). Horizontal refactor across `linalg/`, `preprocess/`, every model in `models/`, `scan/`, `io/`. | None — refactor track, not a phase |
| 2 | [`2026-05-11-phase-57-regenie-design.md`](2026-05-11-phase-57-regenie-design.md) | REGENIE MVP. Quantitative + binary-with-SPA only. No Firth, no time-to-event, no gene burden. New `models/regenie_*.py` files + new CLI subcommand. | 57 |
| 3 | [`2026-05-11-phase-58-prscsx-design.md`](2026-05-11-phase-58-prscsx-design.md) | PRS-CSx MVP. Multi-population horseshoe MCMC core only. No auto-tuning, no meta-PRS variants. Extends `pgs/prscs.py`. | 58 |
| 4 | [`2026-05-11-phase-59-polyfun-design.md`](2026-05-11-phase-59-polyfun-design.md) | PolyFun **full parity**. Functional priors + baseline-LF S-LDSC integration + SuSiE layering. Extends `models/bayesian_vs.py`, `postgwas/_finemapping.py`, `postgwas/_sldsc.py`. | 59 |

Per-spec scope was set during the 2026-05-11 brainstorm using the **C: Hybrid** scoping decision — MVPs for the two largest items (REGENIE, PRS-CSx) where edge-case parity is enormously expensive, full parity for the two smaller items (PolyFun, X-chrom) where the surface area is manageable.

## 3. Sequencing

**Spec authoring order** (this brainstorming session, all on branch `modernization/specs`):

1. Master design (this file) — written first to lock cross-cutting decisions.
2. Parallel research dispatch (four `general-purpose` agents, one per item) → research briefs land at `docs/superpowers/research/<topic>-research.md`.
3. Brief audit + re-dispatch any briefs with filler / missing citations.
4. **X-chromosome refactor spec** — written next, because it changes the substrate that the other three plug into.
5. **Phase 57 REGENIE spec** — references the X-chrom decisions.
6. **Phase 58 PRS-CSx spec** — extends `pgs/prscs.py`; mostly independent of X-chrom.
7. **Phase 59 PolyFun spec** — extends `models/bayesian_vs.py` + `postgwas/_finemapping.py` + `postgwas/_sldsc.py`; mostly independent of X-chrom.

**Implementation order** (DEFERRED per user choice **D** during brainstorming — implementation does not start until the validation campaign Pillars A–D and the three NA tasks (NA1 SuSiE streaming, NA2 GPU CI, NA3 UKB-scale validation) are closed):

1. **First: X-chromosome refactor track.** Done first so REGENIE/PRS-CSx/PolyFun inherit X-aware behavior natively rather than needing retrofit.
2. **Then: Phase 57 REGENIE.** Largest of the three feature phases; opens "biobank standard" door.
3. **Then: Phases 58 + 59 in parallel.** Both extend separate module families (PGS vs fine-mapping); no real conflict; can be split between two execution sessions per the user's standing constraint that *parallel agents must not cut corners* (see `feedback_parallel_agents_no_shortcuts`).

**Rationale for X-chrom first:** retrofitting X-chromosome handling into Phase 57+ implementations after the fact is strictly more work than implementing the substrate first and inheriting it. The cost is that the X-chrom refactor has the largest blast radius (touches existing models that already pass the V1-equivalence parity bar against GEMMA / GAPIT / GWASpoly), and any regression there halts the refactor track per F3 severity policy.

## 4. Cross-spec technical dependencies

| Spec | Depends on existing | Affects existing |
|---|---|---|
| X-chrom refactor | `linalg/`, `preprocess/`, every model in `models/`, `scan/`, `io/` | Cross-cutting; conservatively touches 30+ files (per pre-research estimate; final count fixed in the X-chrom sub-spec after research brief lands) |
| Phase 57 REGENIE | `models/single_trait_lmm.py`, `models/lro_lmm.py` (LOCO precedent), `stats/spa.py`, `optim/pql.py`, `scan/` | New `models/regenie_*.py`; new CLI subcommand `regenie-scan` |
| Phase 58 PRS-CSx | `pgs/prscs.py`, `pgs/ld_ref.py`, `pgs/sumstats_io.py`, `postgwas/_multi_ancestry.py` | Extends `pgs/`; no new top-level CLI required (extends `pgs-fit` flags) |
| Phase 59 PolyFun | `models/bayesian_vs.py`, `postgwas/_finemapping.py`, `postgwas/_sldsc.py`, `annotate/` | Extends `models/` + `postgwas/`; new CLI subcommand candidate (decision deferred to PolyFun sub-spec) |

**Validation campaign awareness:** Pillar B of the validation campaign already commits to wiring REGENIE as a reference comparator (per `2026-04-30-validation-campaign-design.md` §5). The Phase 57 sub-spec must reference and reuse that install / fixture work rather than duplicate it.

## 5. Validation gate philosophy (uniform across all 4 sub-specs)

Every sub-spec uses the same three-tier gate, with tolerances **observed-then-floored** per `feedback_validation_spec` memory:

**Tier 1 — math correctness (synthetic, deterministic, fast).**
- Closed-form / scipy-comparison tests on hand-crafted fixtures.
- ≤ 1e-10 absolute on closed-form expectations.
- Runs in default `pytest tests/`, no install gates, no marker.
- Each Tier-1 test exercises one function in isolation.

**Tier 2 — smoke parity vs upstream (small fixture, short runtime).**
- Public mini-fixture (HapMap3 chr22 sample, 1000G phase-3 subset, baseline-LF v2.2 annotation slice — sized so each smoke run completes in < 5 min on CPU).
- Run upstream tool on fixture → capture observed divergence → set tolerance per the rule below. Never aspirational.
- Marker: `@pytest.mark.golden` (matches existing convention).
- Findings logged to `docs/validation_findings.md` per F3 severity policy.
- Pre-flight memory + disk check before each install (per `feedback_preflight`).

**Tier 2 tolerance rule (per spec):**
- **MVP scope** (Phase 57 REGENIE, Phase 58 PRS-CSx, X-chrom additive): `floor + observed × 2`.
- **Full-parity scope** (Phase 59 PolyFun, X-chrom non-additive variants): `floor + observed × 1.5` — tighter, because "full parity" implies tighter agreement.

**Tier 3 — full-scale empirical (deferred).**
- UK Biobank–scale runs deferred to NA3 task per existing `project_next_agent_tasks` memory.
- Tier 3 results, when produced, append to validation campaign Pillar B / D ledger.
- Each sub-spec *describes* what Tier 3 would assert but does not run it.

## 6. Cross-cutting house conventions (per existing memories)

These conventions are inherited verbatim by every sub-spec; sub-specs do not re-derive or override them.

- **Streaming-first** (`feedback_streaming`): every new CLI scan command MUST stream chunks via `iter_chunks`. Two regression nets (`tests/test_streaming_memory.py` + `bench/native_speedups.py` wall-time gate) MUST be extended to cover the new command. Re-materializing the genotype matrix is a regression.
- **Regression nets are paired** (`feedback_regression_nets`): memory regression test + wall-time regression CI must both exist for any new perf-sensitive path.
- **Python-as-spec, native-as-shortcut** (per `CLAUDE.md` §"Repo Conventions"): pure-torch path is the algorithmic spec. Native acceleration is wired *at the top* of the function via `_dispatch.select_path` and falls through to the torch body under `TORCHGWAS_DISABLE_NATIVE=1`. Each sub-spec lists candidate hot loops with rough speedup targets but does NOT commit to native code in the initial implementation; native acceleration is deferred to a Phase-41-style sub-phase per item.
- **Pre-flight gate** (`feedback_preflight`): assert disk + RAM headroom before any install / download / run that exceeds local fixtures. Abort on failure rather than partially execute.
- **GPU dispatch discipline** (per `CLAUDE.md` §"Repo Conventions"): never proactively move CUDA tensors to host for a C++ path; if no dedicated GPU kernel exists, the torch body runs on-device instead.

## 7. F3 severity policy application

All four items are post-V1 (V1 covers Phases 0–13, GEMMA/GAPIT-equivalent quantitative-trait Gaussian GWAS). Therefore, per `feedback_f3`:

| Event | Action |
|---|---|
| Divergence from upstream within tolerance | Pass; no entry in findings ledger |
| Divergence from upstream outside tolerance | **Document** in `docs/validation_findings.md` with root-cause investigation; does NOT block the phase |
| Regression in any *existing* code path during X-chrom refactor | **Halt** the refactor track; revert and re-approach |
| ≥ 3 unrelated divergences within one sub-spec implementation | **C4 emergency stop;** regroup with user before continuing |

## 8. Combined risk register

| ID | Risk | Mitigation |
|---|---|---|
| R-MOD-1 | X-chrom refactor breaks GEMMA/GAPIT-equivalence parity tests on autosomes | Run full V1 parity suite before merging any X-chrom commit; revert on any divergence |
| R-MOD-2 | REGENIE binary's predictions file format does not stream cleanly into our scanner | Sub-spec must include a streaming adapter (or document the memory cost upfront with a numerical bound) |
| R-MOD-3 | PRS-CSx multi-population MCMC has correlated shrinkage that breaks `pgs/prscs.py`'s single-population loop assumptions | Sub-spec must declare whether to extend `prscs.py` or fork to `prscsx.py`; default lean: separate file to avoid V1-equivalence regressions in `prscs.py` |
| R-MOD-4 | PolyFun functional-annotation matrix sizes (millions of variants × hundreds of annotations) exceed the default chunk-streaming assumption | Sub-spec must specify the annotation chunk strategy and a memory bound |
| R-MOD-5 | Reference tool installs (REGENIE binary, PRS-CSx + LD panels, PolyFun + baseline-LF) collectively exceed available disk / RAM | Pre-flight check at install time per `feedback_preflight`; combined estimated footprint ~25–30 GB to be confirmed when each sub-spec's research brief lands |
| R-MOD-6 | Parallel implementation of Phases 58 + 59 produces inconsistent style or skipped corners (per user concern raised 2026-05-11) | Per `feedback_parallel_agents_no_shortcuts` — explicit acceptance criteria + no-shortcuts clause + post-dispatch audit before synthesis |
| R-MOD-7 | These four phases extend the parity-test surface materially; future maintenance burden grows | Documented tradeoff in §1 of each sub-spec; F3 severity policy keeps post-V1 divergences from blocking |

## 9. Open questions deferred to sub-specs

These were intentionally not resolved in the master and have now been answered in each sub-spec (status as of 2026-05-11 sub-spec drafting):

- **REGENIE:** does step-1's BLUP predictions file streamably integrate with `UnifiedScanner`, or do we need an offline → on-demand re-load strategy? **RESOLVED** in Phase 57 sub-spec §4.2 + R-REG-1: LOCO file is row-streamable per chromosome (~4 MB at UKB scale); a `RegenieLOCOReader` adapter loads one row per chromosome change. Resolves R-MOD-2 in §8.
- **PRS-CSx:** extend `pgs/prscs.py` or new file `pgs/prscsx.py`? **RESOLVED** in Phase 58 sub-spec §3.3 + brief §11.1: NEW FILE `pgs/prscsx.py`. Resolves R-MOD-3 in §8.
- **PolyFun:** does the spec ship a CLI subcommand or extend an existing one? **RESOLVED** in Phase 59 sub-spec §6.1: new `polyfun-finemap` (combined) plus 8 individual subcommands for the auxiliary scripts.
- **X-chrom:** which X-inactivation models to support — uniform skewing only, gene-level skewing, or random? **RESOLVED** in X-chrom sub-spec §2.1 + §10: all four PLINK `--xchr-model` values (0/1/2/3) plus sex-stratified meta-analysis for Tier A/B; gene-level XCI (Tukiainen 2017) deferred to Tier C.
- **All four:** native-acceleration sub-phases — assigned phase numbers (Phase 41ag, ah, …) or deferred until profiling evidence justifies? **RESOLVED uniformly:** deferred until profiling evidence justifies; each sub-spec §7 lists candidate hot loops without committing.

## 9b. Brainstorming decisions (D1 / D2 / D3) ledger

Three decisions were made during the 2026-05-11 brainstorming session that affect multiple sub-specs. Where each landed:

| Decision | Resolution | Where operationalized |
|---|---|---|
| **D1 — X-chrom ploidy strategy** | Per-sample-per-variant ploidy tensor (option a). Honors the polyploid-first charter; wider blast radius than splitting chrX into a separate code branch but cleaner long-term. | `xchromosome-refactor-design.md` §3.3 (preprocess/standardize.py — per-sample-per-variant ploidy support), §10.1 step A2, R-XC-1 mitigation. |
| **D2 — X-chrom GRM strategy** (revised under scientific-rigor mandate) | Support BOTH: default to GCTA two-kernel via `MultiKernelLMM` for LMM scanners (statistically rigorous published approach); fall through to weighted single-kernel for FarmCPU/BLINK (which can't use multi-kernel). Exposed via `--xchr-grm-kernel {separate,joint,none}`. | `xchromosome-refactor-design.md` §2.7, §6 (CLI flag), §10.2 step B5 (two-kernel via `MultiKernelLMM`). |
| **D3 — PolyFun `bayesian_vs.py` API change** | Backward-compat shim: add new `prior_pi_per_snp: Optional[Tensor] = None` keyword preserving scalar `prior_pi: float = 0.01`. No breaking change to existing callers. | `phase-59-polyfun-design.md` §2.3 (D3 explanation), §3.2 (modified files table — bayesian_vs.py change), §10.1 step A1 (Tier A first item). |

## 10. Companion artifacts

- **Research briefs** (`docs/superpowers/research/`): four files, one per sub-spec, produced by parallel agents per the 12-section template. Working notes; commit decision per file as briefs land.
- **Implementation plans** (future, `docs/superpowers/plans/`): produced by `superpowers:writing-plans` skill at each implementation kickoff. Not part of this brainstorming output.
- **Findings ledger** (`docs/validation_findings.md`): existing canonical record; receives entries during Tier 2 / Tier 3 runs.

## 11. Approval & next step

This master spec is committed once approved. Sub-specs are committed individually after each is written and self-reviewed. Once all five files are in `docs/superpowers/specs/`, user reviews the full set and either approves or requests changes. Approved → transition to `superpowers:writing-plans` per brainstorming-skill terminal state (deferred until post–validation-campaign per choice D).
