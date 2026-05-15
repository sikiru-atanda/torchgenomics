# Genome Biology Methods paper — design spec

**Date**: 2026-05-15
**Status**: Brainstorming complete; awaiting user review of this spec before transitioning to writing-plans.
**Author**: Sikiru Atanda (corresponding); additional authors TBD.
**Working title**: *TorchGWAS: a GPU-accelerated, polyploid-first toolkit unifying variant, haplotype, and multi-omics GWAS at biobank scale, with end-to-end numerical equivalence to sixteen established tools.*

## 1. Origin

The pre-existing Bioinformatics Application Note scaffold (`paper/PLAN.md`, `paper/drafts/abstract.md` in the `na1-susie-streaming` worktree) under-represented the package's capability surface. A capability audit against `torchgwas/` (modules + CLI registry) showed the manuscript named ~10 capabilities while the package implements ~50 across 56 phases. The unmentioned clusters included haplotype GWAS (Phase 46–47), LD-block design (Phases 20–21), TWAS / SMR / HEIDI / coloc / hyprcoloc (Phases 42, 44, 45), GRM-corrected causal mediation (Phase 49), and the entire specialty model family (Phases 22–35).

Per the brainstorming session (2026-05-15), we are switching venue to **Genome Biology — Methods (Software)** to give every cluster a defensible home, and adding five external reference-tool harnesses to validate the new clusters before submission.

## 2. Brainstorming decisions log

| # | Decision | Choice |
|---|---|---|
| D1 | Manuscript path | Switch venue to longer paper |
| D2 | Target venue | Genome Biology — Methods (Software) |
| D3 | Lead thesis | Synthesis of all four claims (unification + multi-omics + polyploid-first + biobank/reproducibility) |
| D4 | Figure plan | 7 main figures (added F7 specialty-models figure to the original 6-figure plan) |
| D5 | Validation scope | Add the missing external references (FUSION/MetaXcan, SMR-tool, coloc R, hyprcoloc R, haplotype reference) before submission |
| D6 | Multi-omics data sources | All three: simulated + GTEx v8 / UKB + plant multi-omics |
| D7 | Execution mode | Dispatch independent work in parallel via subagents (see §10) |

## 3. Lead thesis

> TorchGWAS is the first single open-source toolkit to (i) **unify** variant-level, haplotype-level, and multi-omics GWAS in one runtime; (ii) treat **polyploidy** as first-class (arbitrary ploidy k; diploid is k=2); (iii) **stream** from biobank-scale data on commodity GPUs; (iv) ship with end-to-end **numerical equivalence** to sixteen established reference tools (eleven existing + five new harnesses), under observed-then-floored tolerances reproducible from a single shell command.

To our knowledge, no prior toolkit combines all four properties.

## 4. Manuscript structure

Genome Biology Methods (Software) format: structured Abstract / Background / Results / Methods / Discussion / Availability. Budget ~7000–9000 words.

| § | Section | Words | Anchor |
|---|---|---|---|
| 1 | Background | ~700 | Biobank-scale GWAS landscape; gap = no toolkit unifies variant + haplotype + multi-omics + polyploid + streaming on GPU with reference equivalence. |
| 2 | Results: Architecture & capability surface | ~600 | F1 — capability map (all ~50 features grouped by cluster). |
| 3 | Results: Reference-tool equivalence (16 tools) | ~900 | F2 — 16-tool grid; observed-then-floored tolerances; all PASSING. |
| 4 | Results: Haplotype layer | ~1100 | F3 — 13 LD-block-design methods + 9 haplotype GWAS methods (HTR/window/block/SKAT + PCHT/HHCT/HSKAT/HapGxE/BayesHap) + multi-env/multi-trait extensions. |
| 5 | Results: Multi-omics integration | ~1200 | F4 — TWAS + SMR/HEIDI + coloc + GRM-corrected mediation; three-panel (sim + GTEx/UKB + plant). |
| 6 | Results: Polyploid pipeline | ~900 | F5 — dosage call → F1 phasing → polyploid GWAS → polyploid LD blocks → polyploid haplotype GWAS on potato F1 + autotetraploid + sim hexaploid/octoploid. |
| 7 | Results: Specialty models | ~700 | F7 — survival, random regression, within-family, threshold-linear, knockoff, OCF, GU, LRO. |
| 8 | Results: GPU + biobank streaming | ~800 | F6 — 24 native kernel speedup distribution + streaming memory slope (19×) + sparse-GRM PCG-REML. |
| 9 | Methods (compressed) | ~800 | Streaming I/O; 6-mode optimizer; native dispatch; observed-then-floored tolerance protocol. |
| 10 | Discussion + limitations + roadmap | ~500 | Limitations: no Windows GPU; phase-coherence check pending for polyploid haplotypes. |
| 11 | Availability | ~200 | Repo, install, license, docs site, reproducibility script. |
| — | Total | ~8400 | Within GB Methods budget. |

## 5. Figure plan (7 main + supplement)

| F | Title | Panels | Data | Source path |
|---|---|---|---|---|
| 1 | Capability map | single sunburst / treemap grouping all ~50 capabilities by cluster | n/a | hand-rendered from CLI registry + module surface |
| 2 | 16-tool numerical equivalence grid | one column per tool × rows for β / SE / p / PIP / etc. | 11 existing + 5 new harnesses | `validation/external/<tool>/compare.py` for each of 16 tools |
| 3 | Haplotype layer | A: 13 LD-block methods on MDP vs PLINK 1.9 `--blocks` (Gabriel); B: 9 haplotype GWAS methods on SoyMD; C: novel methods (PCHT / HHCT / BayesHap) on a real-data hit | MDP, SoyMD, simulated multi-locus | `validation/external/plink2`, new `validation/external/hapref` |
| 4 | Multi-omics integration | A: simulated ground-truth (TWAS Z / SMR β / coloc PIP / mediation β recovery target ≥ 0.95); B: GTEx v8 + UKB worked example (LDL ↔ SORT1 in liver); C: plant multi-omics (maize WiDiv or Arabidopsis 1001G) | three datasets to stage | new `validation/multiomics/{sim,gtex_ukb,plant}/` |
| 5 | Polyploid pipeline | A: dosage-call accuracy on simulated GBS reads; B: PolyOrigin F1 phasing recovery on potato F1; C: polyploid GWAS on tetraploid panel vs GWASpoly; D: arbitrary-ploidy demo (k=4,6,8) | GWASpoly potato F1 + autotetraploid + sim hexaploid/octoploid | reuse Phase 55/56 fixtures + new hexaploid sim |
| 6 | GPU + biobank streaming | A: 24 native kernel speedup distribution (violin/box); B: streaming-vs-materialized memory slope p ∈ [10⁴, 10⁶]; C: sparse-GRM PCG-REML 1.8× speedup; D: CPU↔GPU parity scatter | existing benches | `bench/native_speedups.py` + `validation/streaming_memory/p_sweep.py` |
| 7 | Specialty models | grid: survival (sim Cox); RR longitudinal (SoyNAM time-series); within-family (UKB siblings or sim); threshold-linear (ordinal sim); knockoff (FDR control sim); OCF (DML coverage sim) | mostly simulated; SoyNAM for RR | new `validation/specialty/{survival,rr,family,threshold,knockoff,ocf}/` |

Supplementary figures: per-haplotype-method internals (S3.1–S3.9); extended multi-omics panels; per-ploidy parity scans; OpenMP-off / native-off / GPU-off parity scans.

## 6. Supplement scope

- **S1** Full per-tool equivalence ledger (every metric, every fixture, 16 tools)
- **S2** Complete software-capability inventory (~50 capabilities × CLI × module × phase × test file × validation harness)
- **S3** Validation findings ledger (copy of `docs/validation_findings.md`)
- **S4** Reproducibility manifest (pinned versions, install scripts, run order)
- **S5** Observed-then-floored tolerance protocol + per-tool numerical drift tables
- **S6** Extended figures (per-haplotype-method internals; extended multi-omics; per-ploidy parity)
- **S7** Sensitivity analyses: pre-flight gate behavior; OpenMP off / native off / GPU off parity scans
- **S8** Demo notebooks: one per main figure, each reproduces its figure end-to-end

## 7. Reproducibility-repo architecture

Separate repo: `github.com/sikiru-atanda/torchgwas-paper-reproducibility`.

```
torchgwas-paper-reproducibility/
├── reproduce_paper.sh                # one-command driver
├── preflight.sh                      # disk + RAM + GPU + network gate (hard rule)
├── stages/
│   ├── 01_install_references.sh      # 16 reference tools via per-tool install.sh
│   ├── 02_stage_fixtures.sh          # download + verify (SHA256 manifest)
│   ├── 03_run_references.sh
│   ├── 04_run_torchgwas.sh
│   ├── 05_run_streaming_bench.sh
│   ├── 06_run_native_bench.sh
│   ├── 07_run_multiomics.sh
│   └── 08_render_figures.py
└── manifest.json                     # every numeric result + commit SHA
```

Acceptance: `bash reproduce_paper.sh` from a clean checkout regenerates every number in every main figure and supplementary table. Pre-flight aborts on insufficient disk / RAM / GPU before any download or run (no partial execution; user's hard rule from spec D7 of the validation campaign).

## 8. Numerical-claim policy

Every numerical claim in the paper must be backed by a head-to-head run against the installed upstream reference tool (user's standing rule from `memory/feedback_benchmark_against_installed_tools.md`):

- No extrapolation from paper numbers.
- No partial-tool comparisons.
- Tolerances are observed-then-floored (measure once; floor at the rounded order-of-magnitude; never aspirational; user's rule from `memory/feedback_validation_spec.md`).
- F3 severity classification on any divergence: V1-core → fix-now; post-V1 → document; regression → halt pillar; 3+ unrelated divergences → C4 emergency stop (user's rule from `memory/feedback_f3.md`).

## 9. New work scope

| Work item | Effort | Required for | Independent? |
|---|---|---|---|
| FUSION / MetaXcan harness | 2-3 d | F2, F4 | Yes |
| SMR-tool harness | 2 d | F2, F4 | Yes |
| coloc R harness | 2 d | F2, F4 | Yes |
| hyprcoloc R harness | 1-2 d | F2 | Yes |
| Haplotype reference harness | 3-4 d | F2, F3 | Yes |
| GTEx v8 + UKB sumstats fixture (SORT1 / LDL) | 2 d | F4 | Yes |
| Plant multi-omics fixture (maize WiDiv or Arabidopsis 1001G) | 3-5 d | F4 | Yes |
| Simulated multi-omics ground-truth fixture | 2 d | F4 | Yes |
| Specialty fixtures (survival, RR, family, threshold, knockoff, OCF) | 5-7 d | F7 | Yes (6 sub-tasks) |
| Reproducibility script + figure-rendering code | ~1 w | all figures | No (waits on harnesses) |
| Spec self-review + figure mock-ups | ~2 d | spec | n/a |
| Full draft (~8400 words) | ~2 w | all | No (waits on figures) |
| **Total serial wall-clock** | **~6-8 weeks** | full submission | — |
| **Total with parallel agents** | **~3-4 weeks** | full submission | — |

## 10. Parallel-agent dispatch plan

The work scope in §9 contains 14 independent sub-tasks that share no state and have no sequential dependency on each other. They are the canonical candidate set for parallel agent dispatch (`superpowers:dispatching-parallel-agents`).

### 10.1 Dispatch tiers

The work splits into four tiers. Tiers 1–3 are mutually independent (no shared state, no cross-tier dependency) and run concurrently. Tier 4 is serial; it waits on the audit pass over Tiers 1–3.

**Quick count**: Tier 1 = 5 agents · Tier 2 = 3 agents · Tier 3 = 6 agents · Tier 4 = serial (main session). Maximum concurrent agents at peak: 14. Plus two internal-only items (GU, LRO) handled inside Tier 4 — see §10.6.

**Tier 1 — Reference-tool harnesses (5 agents in parallel)**

| Agent | Output | Acceptance |
|---|---|---|
| A1 | `validation/external/metaxcan/{install.sh,fetch_data.sh,run.sh,compare.py}` | FUSION + S-PrediXcan installed under pinned version; pre-flight gate present; ≥1 fixture run head-to-head; agreement target documented |
| A2 | `validation/external/smr/{install.sh,fetch_data.sh,run.sh,compare.py}` | Yang-lab SMR tool installed at pinned version; cis-eQTL fixture; HEIDI χ² agreement |
| A3 | `validation/external/coloc/{install.sh,fetch_data.sh,run.R,compare.py}` | R `coloc` installed; H0–H4 posterior agreement |
| A4 | `validation/external/hyprcoloc/{install.sh,fetch_data.sh,run.R,compare.py}` | R `hyprcoloc` installed; cluster-level PP agreement |
| A5 | `validation/external/hapref/{install.sh,fetch_data.sh,run.sh,compare.py}` | Reference tool selected (HaploREML or PLINK 1.9 `--hap-*`) + rationale documented; block / haplotype-frequency agreement |

**Tier 2 — Fixture staging (3 agents in parallel; tier 1 result not required)**

| Agent | Output | Acceptance |
|---|---|---|
| B1 | `validation/multiomics/sim/` | Simulated G ↔ expression ↔ trait fixture under known causal model; ground-truth PIP / β / Z / mediation effect persisted; recovery target ≥ 0.95 documented |
| B2 | `validation/multiomics/gtex_ukb/` | GTEx v8 liver + UKB LDL sumstats (or equivalently strong locus, e.g. SORT1) staged; license + access checked; SHA256 manifest |
| B3 | `validation/multiomics/plant/` | Maize Wisconsin Diversity panel + B73 RNA-seq (or Arabidopsis 1001G + 1001 Epigenomes) staged; license checked; SHA256 manifest |

**Tier 3 — Specialty fixtures (6 agents in parallel; tier 1 and tier 2 not required)**

| Agent | Output | Acceptance |
|---|---|---|
| C1 | `validation/specialty/survival/` | Simulated Cox PH frailty fixture; comparison against `coxKM` or SAIGE-COX |
| C2 | `validation/specialty/rr/` | SoyNAM longitudinal fixture; comparison against `asreml` or `gibbs1f90` if installable |
| C3 | `validation/specialty/family/` | Sib-pair simulated fixture; comparison against Young et al. 2022 reference impl |
| C4 | `validation/specialty/threshold/` | Ordinal multi-trait simulated fixture; comparison against THRGIBBSF90 or paper sim |
| C5 | `validation/specialty/knockoff/` | FDR-control simulated fixture; comparison against `knockoff` R package |
| C6 | `validation/specialty/ocf/` | DML coverage simulated fixture; comparison against Chernozhukov reference impl |

**Tier 4 — Reproducibility + manuscript (serial; waits on tiers 1–3)**

| Step | Output |
|---|---|
| D1 | Reproducibility repo skeleton + `reproduce_paper.sh` + `preflight.sh` |
| D2 | Figure-rendering scripts (one per F1–F7) reading from harness outputs |
| D3 | Manuscript draft (Background → Availability) |
| D4 | Internal review + spec-aligned response |
| D5 | bioRxiv submission, then Genome Biology submission |

### 10.2 Briefing template (every agent gets this)

Each agent is dispatched with a self-contained brief carrying:

1. **Goal** — the single output deliverable (one harness or one fixture, scoped tight).
2. **Acceptance criteria** — explicit numerical + structural targets (file paths, agreement tolerance, pre-flight gate, SHA256 manifest, license check).
3. **Project context** — pointers to `CLAUDE.md`, `memory/feedback_*.md` rules, and prior validation harnesses to mirror (e.g., `validation/external/twosamplemr/` as a template).
4. **"No shortcuts" clause** — per user's standing rule in `memory/feedback_parallel_agents_no_shortcuts.md`: "every agent must meet the same quality bar a serial author would; no skipping pre-flight, no extrapolating numbers, no synthetic-only when a real reference exists, no fabricated tolerances."
5. **Output gate** — agent must persist outputs to its own subdirectory; the main session audits before synthesis.

### 10.3 Audit-before-synthesis

After each tier completes, the main session (not a subagent) performs an audit pass before launching the next tier:

- All acceptance criteria met for every agent in the tier?
- Pre-flight gate present and exits non-zero on insufficient resources?
- SHA256 manifests present and verified?
- Tolerances *observed* (measured against an actual reference run), not aspirational?
- Any F3-classified divergence routed through the findings ledger (`docs/validation_findings.md`)?
- 3+ unrelated divergences detected → C4 emergency stop, escalate to user before proceeding.

The audit is non-negotiable. The user's standing rule (`memory/feedback_parallel_agents_no_shortcuts.md`) is that an agent's summary describes intent, not necessarily what it did; the main session verifies the actual file outputs.

### 10.4 Dispatch ordering and rate

- Tiers 1, 2, and 3 launch at session start in a single message (the dispatching-parallel-agents pattern). All 14 agents run concurrently.
- Tier 4 is serial; the main session runs it after every Tier 1–3 agent has been audited.

Where rate or token budget makes 14-way concurrency impractical, the main session may batch in waves (e.g. wave 1: Tier 1 + 2; wave 2: Tier 3). Sequencing is a budget decision, not a correctness one — every Tier 1–3 task is genuinely independent.

### 10.5 No-autonomous-push gate

Per `memory/feedback_no_autonomous_push.md`: no agent is permitted to push to any git remote. Local commits OK; pushes are user-gated per event. This is included in every agent brief.

### 10.6 Internal-only specialty models (GU and LRO)

F7 lists eight specialty models. Six (survival, RR, within-family, threshold-linear, knockoff, OCF) have a credible external reference and are validated through Tier 3 agents C1–C6. The remaining two have no clean external reference:

- **GU (Genotype-Uncertainty LMM, Phase 28)** — dosage-variance score test specific to TorchGWAS; no upstream equivalent.
- **LRO (Leave-Region-Out LMM, Phase 29)** — block-level LOCO specific to TorchGWAS; no upstream equivalent.

Both are validated against internal-consistency tests (recovery of known dosage variance and known block-resolved h² respectively) under the same observed-then-floored protocol. The paper Discussion discloses this asymmetry explicitly. No parallel agent is dispatched; the main session adds these two simulated fixtures during Tier 4 D2 (figure rendering).

## 11. Limitations to disclose in the paper

- No Windows GPU build; CPU + Linux/macOS GPU only.
- Phase-coherence validation for polyploid haplotypes (Phase 56 output → Phase 46 input) is a known limitation flagged in Tier-A spec `2026-04-24-haplotypegwas-shape-adapter-design.md`; disclose in Discussion.
- UKB-scale n = 500 K LMM-from-scratch requires the sparse-GRM PCG-REML path (validated; see F6 panel C).
- Multi-ancestry meta (MR-MEGA / MANTRA) ships with internal-consistency tests; external reference is deferred.

## 12. Acceptance criteria for this spec

This spec is "complete" when:

- [ ] Title, venue, thesis, section budget, figure plan, supplement plan, reproducibility-repo architecture, scope estimate, and parallel-agent plan all written.
- [ ] Spec self-review pass: no TBDs in load-bearing sections, no internal contradictions, scope decomposes cleanly into the parallel agents in §10.
- [ ] User reviews and approves.
- [ ] Transition to `superpowers:writing-plans` to break each tier into the implementation plan.

## 13. Open items deferred to the implementation plan

- Final license choice (Apache 2.0 vs MIT vs BSD-3-Clause vs GPL-3); affects bioconda packaging.
- Author list and affiliations beyond corresponding author.
- Final selection of the haplotype reference tool (HaploREML vs PLINK 1.9 `--hap-*` vs both); resolved by Agent A5 with rationale.
- Final selection of plant multi-omics dataset (maize WiDiv vs Arabidopsis 1001G); resolved by Agent B3 with rationale.
- Bioconda recipe state — confirm before claiming `conda install -c bioconda torchgwas` in Availability.
- Docker image inclusion (yes / no; impact on reviewer experience).
