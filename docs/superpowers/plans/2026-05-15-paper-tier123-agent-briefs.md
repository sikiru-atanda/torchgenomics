# Paper Tiers 1-3 — Parallel agent dispatch implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (this plan is designed for parallel subagent dispatch). Each `## Agent` section is one subagent task; dispatch them concurrently and audit each before synthesis. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Land the 14 parallel-track artifacts (5 reference-tool harnesses, 3 multi-omics fixtures, 6 specialty validation fixtures) on the `paper/genome-biology-methods` working branch with verified head-to-head agreement against installed reference tools, so Tier 4 can render every figure from real data.

**Architecture:** All 14 agents share a common briefing template (§3) and standing-rules block (§2). Each agent owns one subdirectory under `validation/external/` or `validation/{multiomics,specialty}/` and writes its outputs there. The main session audits each agent's deliverable before synthesizing into Tier 4. Pre-flight gate and observed-then-floored tolerance protocol are non-negotiable for every agent.

**Tech Stack:** bash + python + (per agent) R / Julia / external CLI tools at pinned versions.

---

## 1. Pre-conditions

- [ ] Tier 0 plan (`docs/superpowers/plans/2026-05-15-paper-tier0-working-branch.md`) executed to completion
- [ ] On branch `paper/genome-biology-methods`
- [ ] `paper/README.md` present (decision marker)
- [ ] Spec readable at `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
- [ ] `validation/external/_lib/preflight.sh` present and executable

Verify with:

```bash
git rev-parse --abbrev-ref HEAD
test -f docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md && echo "spec OK"
test -x validation/external/_lib/preflight.sh && echo "preflight OK"
```

Expected: `paper/genome-biology-methods`, `spec OK`, `preflight OK`.

---

## 2. Standing rules for every agent

Every agent brief includes the following block verbatim. The main session enforces these during the audit pass.

> ### Standing rules (must be honored by every agent)
>
> 1. **Pre-flight gate is mandatory.** Every install / fetch / run script must source `validation/external/_lib/preflight.sh` at the top and abort non-zero on insufficient disk / RAM / GPU / network. Reference: `memory/feedback_preflight.md`. No partial executions.
> 2. **Benchmark against the *installed* reference tool, not paper numbers or extrapolations.** Reference: `memory/feedback_benchmark_against_installed_tools.md`.
> 3. **Tolerances are observed-then-floored.** Measure actual divergence once on a real run; floor at the rounded order-of-magnitude. Never aspirational. Reference: `memory/feedback_validation_spec.md`.
> 4. **F3 divergence policy.** Any divergence: V1-core → fix-now; post-V1 → document in `docs/validation_findings.md`; regression → halt and escalate; 3+ unrelated divergences → C4 emergency stop. Reference: `memory/feedback_f3.md`.
> 5. **No shortcuts.** Same quality bar as a serial author. No skipping pre-flight. No synthetic-only when a real reference exists. No fabricated tolerances. Reference: `memory/feedback_parallel_agents_no_shortcuts.md`.
> 6. **Persist a SHA256 manifest** for every downloaded fixture, written to `<agent-dir>/results/manifest.sha256`.
> 7. **No autonomous push.** Local commits OK; pushes are user-gated per event. Reference: `memory/feedback_no_autonomous_push.md`.
> 8. **Scientific rigor.** Cite every formula, default, and tolerance to a primary source. Conservative default when literature is split. Reference: `memory/feedback_scientific_rigor.md`.
> 9. **Output gate.** All deliverables under the agent's own subdirectory, with structure: `install.sh`, `fetch_data.sh`, `run.sh` (or `run.R` / `run.jl`), `compare.py`, `README.md`, and `results/{*.tsv, *.json, manifest.sha256}`.
> 10. **Stop on ambiguity.** If acceptance criteria are unclear or a reference tool cannot be installed under license, halt and report — do not fabricate a comparison.

---

## 3. Common briefing template (filled in per-agent in §§4-6)

Every dispatch carries this template, with `<...>` fields filled in:

```
You are dispatched to author <AGENT_ID>: <DELIVERABLE>.

Goal: <ONE_SENTENCE_GOAL>

Output directory: <AGENT_DIR>

Acceptance criteria (all must be met before reporting completion):
<NUMBERED_LIST_OF_ACCEPTANCE_CRITERIA>

Project context:
- Repo root: /home/sikiru.atanda/Documents/GWAS_Expert
- Branch: paper/genome-biology-methods (must not be changed)
- Authoritative spec: docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md (read §10 dispatch tiers and §10.2 briefing template before starting)
- TorchGWAS module the harness compares against: <MODULE_PATH>
- TorchGWAS CLI subcommand (if any): <CLI_SUBCOMMAND>
- Reference harness to mirror in structure: <TEMPLATE_DIR>  (e.g., validation/external/twosamplemr/ for an R-tool harness; validation/external/regenie/ for a CLI-tool harness; validation/external/ukb/ for a real-data fixture)

Standing rules (§2 of plan B — read in full):
<STANDING_RULES_BLOCK_VERBATIM>

Tool-specific guidance:
<TOOL_SPECIFIC_NOTES>

Output gate (reported in your completion message):
- File tree of <AGENT_DIR>
- Pinned version of the reference tool
- Observed agreement metric (concrete number)
- Floored tolerance (concrete number)
- Any F3 divergences encountered + classification

Do not push to any remote. Stop and report if you cannot meet any acceptance criterion.
```

---

## 4. Tier 1 — Reference-tool harnesses (5 agents in parallel)

### Agent A1: MetaXcan / S-PrediXcan harness

**Files:**
- Create: `validation/external/metaxcan/install.sh`
- Create: `validation/external/metaxcan/fetch_data.sh`
- Create: `validation/external/metaxcan/run.sh`
- Create: `validation/external/metaxcan/compare.py`
- Create: `validation/external/metaxcan/README.md`
- Create: `validation/external/metaxcan/results/{summary.tsv,agreement.json,manifest.sha256}`

**TorchGWAS comparison target:** `torchgwas/postgwas/_twas.py` — `twas_sumstat` (sumstats path) and `twas_individual` (individual-level path); CLI integration via `torchgwas mediate-scan` is unrelated (do not confuse).

**Tool-specific guidance:**
- Default choice: **MetaXcan / S-PrediXcan** (https://github.com/hakyimlab/MetaXcan, the sumstats path). FUSION (Gusev) is a sibling method that uses different SNP weights and is *optional*. If A1 decides to also benchmark FUSION, place it under a *separate* sibling directory `validation/external/fusion/` and document the rationale in `metaxcan/README.md` § "FUSION sibling decision". Do not silently combine FUSION and MetaXcan into one tool dir.
- Pinned version: pick MetaXcan release ≥ v0.7.5 (or the latest stable tag at install time) and record the commit SHA in `README.md`.
- Pre-built weights: use a GTEx v8 PrediXcan model (e.g., liver) downloaded via `fetch_data.sh`.
- Fixture: GTEx v8 sumstats × UKB-style lipid trait, or whichever public sumstats `validation/external/ukb/` already stages.

**Acceptance gate (verify before reporting completion):**

```bash
test -f validation/external/metaxcan/install.sh && \
test -f validation/external/metaxcan/fetch_data.sh && \
test -f validation/external/metaxcan/run.sh && \
test -f validation/external/metaxcan/compare.py && \
test -f validation/external/metaxcan/README.md && \
test -f validation/external/metaxcan/results/manifest.sha256 && \
test -f validation/external/metaxcan/results/agreement.json && \
head -3 validation/external/metaxcan/install.sh | grep -q "preflight.sh"
```

- [ ] Dispatch A1 with the filled-in §3 template
- [ ] Audit A1 output against the acceptance gate above
- [ ] If A1 produced any F3 divergence, route through `docs/validation_findings.md` and re-audit
- [ ] Commit A1's outputs with message `Tier 1 A1: MetaXcan/S-PrediXcan harness`

---

### Agent A2: SMR + HEIDI harness

**Files:**
- Create: `validation/external/smr/{install.sh,fetch_data.sh,run.sh,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/postgwas/_smr.py` — `smr_test`, `heidi_test`, `smr_heidi` (combined).

**Tool-specific guidance:**
- Reference: Yang lab SMR tool (https://yanglab.westlake.edu.cn/software/smr/) at pinned version (e.g., 1.3.1). Record SHA256 of the downloaded binary in `manifest.sha256`.
- Fixture: a cis-eQTL panel from GTEx v8 (any tissue) + a GWAS summary set covering an overlapping SNP set. Use the same SORT1/LDL pair as A1 if A1 has staged it; otherwise stage afresh and document.
- Compare: SMR β + SE + p; HEIDI χ² + p; agreement target ≥ 4 sig-figs on β, ≥ 3 sig-figs on p (observed-then-floored).

**Acceptance gate:** same shape as A1, substituting `smr` for `metaxcan`.

- [ ] Dispatch A2
- [ ] Audit
- [ ] Commit `Tier 1 A2: SMR + HEIDI harness`

---

### Agent A3: coloc R harness

**Files:**
- Create: `validation/external/coloc/{install.sh,fetch_data.sh,run.R,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/postgwas/` — the two-trait classical coloc path (Giambartolomei 2014). Search the repo for `def coloc` or `_coloc_classic` to find the exact entry point.

**Tool-specific guidance:**
- Reference: R `coloc` package (CRAN), pinned version ≥ 5.2 (record `packageVersion('coloc')` output in README.md).
- Use the same SNP × eQTL × GWAS triple as A2 to enable a coherent F4 worked example downstream.
- Compare: posterior probabilities PP.H0–PP.H4 across the same locus set.
- Agreement target: max |ΔPP| ≤ 1e-3 (observed-then-floored).

**Acceptance gate:** structural same as A1; additionally `run.R` must reproduce on a clean R 4.x environment.

- [ ] Dispatch A3
- [ ] Audit
- [ ] Commit `Tier 1 A3: coloc R harness`

---

### Agent A4: hyprcoloc R harness

**Files:**
- Create: `validation/external/hyprcoloc/{install.sh,fetch_data.sh,run.R,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/postgwas/_hyprcoloc.py` — multi-trait coloc (Foley 2021).

**Tool-specific guidance:**
- Reference: R `hyprcoloc` package (GitHub: `jrs95/hyprcoloc`), pinned at a release SHA.
- Use ≥ 3 traits to exercise the multi-trait clustering; e.g., 3 lipid traits from UKB + GTEx eQTL.
- Compare: cluster-level posterior probabilities + assigned cluster membership.
- Agreement target: max |ΔPP| ≤ 1e-3 AND cluster-assignment agreement = 100% on the test locus (observed-then-floored).

**Acceptance gate:** structural same as A1.

- [ ] Dispatch A4
- [ ] Audit
- [ ] Commit `Tier 1 A4: hyprcoloc R harness`

---

### Agent A5: Haplotype-reference harness

**Files:**
- Create: `validation/external/hapref/{install.sh,fetch_data.sh,run.sh,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/haplotype_gwas.py` (HaplotypeGWAS class; HTR/window/block/SKAT methods).

**Tool-specific guidance:**
- Reference tool selection: pick exactly one, with rationale recorded in `README.md`:
  - **Option α:** PLINK 1.9 `--hap-*` family — pros: already partially exercised via `validation/external/plink2/`, deterministic, easy to install. Cons: limited to fixed-window haplotype tests, no SKAT.
  - **Option β:** HaploREML (https://github.com/ggentile/HaploREML) — pros: covers HTR + block-based tests with REML. Cons: less widely cited; install path more complex.
  - **Option γ:** Both, with HaploREML as primary and PLINK 1.9 as a sanity check.
- Default selection: **Option α (PLINK 1.9)** for speed-to-completion; agent may upgrade to γ if the install is straightforward.
- Fixture: MDP (Mouse Diversity Panel sample) if present under `validation/external/`; otherwise stage from SoyMD (`validation/external/soymd/`) — pick fixed-window haplotype assignment over an LD-block region.
- Compare: per-haplotype frequency, β, p-value across the same window.
- Agreement target: |Δfreq| ≤ 1e-4; β within 3 sig-figs; p within 2 sig-figs (observed-then-floored).

**Acceptance gate:** structural same as A1.

- [ ] Dispatch A5
- [ ] Audit
- [ ] Commit `Tier 1 A5: haplotype reference harness`

---

## 5. Tier 2 — Multi-omics fixture staging (3 agents in parallel)

### Agent B1: Simulated multi-omics ground-truth fixture

**Files:**
- Create: `validation/multiomics/sim/{generate.py,README.md,fixtures/{G.npy,expr.npy,trait.npy,truth.json,manifest.sha256}}`

**Goal:** Generate a simulated fixture with known causal architecture: G (genotypes) → expr (mediator) → trait (outcome), under a defined β_GE, β_ET, and noise structure, plus a non-causal coloc/SMR negative control.

**Tool-specific guidance:**
- Sample size n=500–2000, marker count p=1000–5000, mediator count m=50.
- Causal architecture: known β vector, fraction of variance explained recorded in `truth.json`.
- Targets for downstream comparison: TWAS Z (against truth), SMR β (against truth), coloc PP.H4 (against truth), mediation β (against truth). Recovery target ≥ 0.95 across all four (observed-then-floored, but the agent must report all four observed values).
- No reference tool needed — this is a *fixture*, not a harness. Truth file is the reference.

**Acceptance gate:**

```bash
test -f validation/multiomics/sim/generate.py && \
test -f validation/multiomics/sim/fixtures/truth.json && \
test -f validation/multiomics/sim/fixtures/manifest.sha256 && \
python -c "import json; t = json.load(open('validation/multiomics/sim/fixtures/truth.json')); assert 'beta_GE' in t and 'beta_ET' in t"
```

- [ ] Dispatch B1
- [ ] Audit
- [ ] Commit `Tier 2 B1: simulated multi-omics ground-truth fixture`

---

### Agent B2: GTEx v8 + UKB sumstats fixture (SORT1 / LDL)

**Files:**
- Create: `validation/multiomics/gtex_ukb/{fetch.sh,prepare.py,README.md,fixtures/{gtex_liver_sort1.tsv.gz,ukb_ldl_sort1.tsv.gz,aligned.parquet,manifest.sha256}}`

**Goal:** Stage a real-data multi-omics worked example: GTEx v8 liver cis-eQTL for SORT1 × UKB LDL-cholesterol GWAS, harmonized to a common SNP set, with provenance.

**Tool-specific guidance:**
- GTEx v8 sumstats: https://www.gtexportal.org/home/datasets — record exact file URL + SHA256.
- UKB lipids sumstats: prefer the Pan-UKB or Neale-lab v3 release (publicly available without UKB application); record exact file URL + SHA256.
- Alignment: harmonize to GRCh38, drop ambiguous strand, write to `aligned.parquet`.
- License check: GTEx v8 sumstats are open; UKB-derived sumstats vary — verify the chosen release is public-domain or open license, document in README.md.
- Pre-flight: this fixture is multi-GB; preflight.sh must assert ≥ 5 GB free disk and ≥ 8 GB free RAM.

**Acceptance gate:**

```bash
test -f validation/multiomics/gtex_ukb/fixtures/aligned.parquet && \
test -f validation/multiomics/gtex_ukb/fixtures/manifest.sha256 && \
test -f validation/multiomics/gtex_ukb/README.md && \
grep -q "License" validation/multiomics/gtex_ukb/README.md
```

- [ ] Dispatch B2
- [ ] Audit
- [ ] Commit `Tier 2 B2: GTEx + UKB SORT1/LDL fixture`

---

### Agent B3: Plant multi-omics fixture

**Files:**
- Create: `validation/multiomics/plant/{fetch.sh,prepare.py,README.md,fixtures/{geno.parquet,expr.parquet,pheno.tsv,aligned.parquet,manifest.sha256}}`

**Goal:** Stage a plant multi-omics worked example. Agent selects exactly one of:
- **Option α:** Maize Wisconsin Diversity Panel (WiDiv) + B73 RNA-seq + 25 agronomic traits — pros: large sample, well-cited; cons: large data.
- **Option β:** Arabidopsis 1001 Genomes + 1001 Epigenomes / 1001 Transcriptomes + a published trait — pros: smaller download; cons: epigenome and transcriptome may need a separate download.
- **Option γ:** A SoyMD / SoyNAM multi-omics extension if available.

Decision must be recorded in `README.md` with rationale tied to (a) data availability + (b) downstream demonstration in F4 panel C.

**Tool-specific guidance:**
- Pre-flight: assert ≥ 10 GB free disk for whichever option is selected.
- Alignment: harmonize geno + expr + pheno onto a common sample ID set; write the joined view to `aligned.parquet`.
- License: must be public / CC-BY / equivalent open license; document.

**Acceptance gate:** structural same as B2; additionally README.md contains "Selected dataset:" line with the chosen option + rationale.

- [ ] Dispatch B3
- [ ] Audit
- [ ] Commit `Tier 2 B3: plant multi-omics fixture`

---

## 6. Tier 3 — Specialty model fixtures (6 agents in parallel)

### Agent C1: Survival GWAS fixture (Cox PH frailty)

**Files:**
- Create: `validation/specialty/survival/{generate.py,run_torchgwas.py,run_reference.R,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/survival_glmm.py` (SurvivalGLMM; Cox PH frailty with SPACox SPA path).

**Tool-specific guidance:**
- Reference: prefer `coxKM` (R) for a small-n SPA-validated comparison; if `coxKM` install fails, fall back to SAIGE-COX (already installed for `validation/external/saige/`) — document choice.
- Simulate n=500 with 10–20% censoring, hazards driven by a known frailty + 1 causal SNP.
- Compare: β, SE, score-test p, SPA-adjusted p where applicable.
- Agreement target: β within 3 sig-figs; SPA-p within 2 sig-figs (observed-then-floored).

**Acceptance gate:**

```bash
test -d validation/specialty/survival/results && \
test -f validation/specialty/survival/README.md && \
test -f validation/specialty/survival/results/manifest.sha256
```

- [ ] Dispatch C1
- [ ] Audit
- [ ] Commit `Tier 3 C1: survival GWAS fixture`

---

### Agent C2: Random regression (longitudinal) fixture

**Files:**
- Create: `validation/specialty/rr/{generate.py,run_torchgwas.py,run_reference.{R,sh},compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/rr_lmm.py` (RandomRegressionLMM; Legendre + B-spline bases) + `torchgwas/models/rr_spatial.py` (SpatioTemporalRR).

**Tool-specific guidance:**
- Reference: prefer `asreml` (R) if license available; else `gibbs1f90` (BLUPF90 family; free); else a published reference simulator (document choice).
- Fixture: SoyNAM-style longitudinal trait (≥3 time points) or simulated; n=200–500.
- Compare: per-time-point β + p; random-effect variance ratios across the longitudinal basis.
- Agreement target: |Δβ| ≤ 3 sig-figs; |Δvar ratio| ≤ 1e-2 (observed-then-floored).

**Acceptance gate:** structural same as C1.

- [ ] Dispatch C2
- [ ] Audit
- [ ] Commit `Tier 3 C2: random regression fixture`

---

### Agent C3: Within-family LMM fixture

**Files:**
- Create: `validation/specialty/family/{generate.py,run_torchgwas.py,run_reference.py,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/within_family_lmm.py` (WithinFamilyLMM; Young et al. 2022 attenuation diagnostics).

**Tool-specific guidance:**
- Reference: `snipar` (Young et al. 2022; GitHub: `AlexTISYoung/snipar`) at pinned release. If unavailable, fall back to published-paper simulator with documented citation.
- Fixture: simulated sib-pair design n=500 families × 2 sibs, with a known direct-effect + indirect-effect ratio.
- Compare: direct-effect β, indirect-effect β, attenuation factor, sandwich SE.
- Agreement target: |Δβ_direct| ≤ 3 sig-figs (observed-then-floored).

**Acceptance gate:** structural same as C1.

- [ ] Dispatch C3
- [ ] Audit
- [ ] Commit `Tier 3 C3: within-family fixture`

---

### Agent C4: Threshold-linear (categorical) fixture

**Files:**
- Create: `validation/specialty/threshold/{generate.py,run_torchgwas.py,run_reference.{f,sh},compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/threshold_linear.py` (ThresholdLinearModel; Bermann et al. 2026; T-EM / T-NR / SQUAREM / ssGWAS solvers).

**Tool-specific guidance:**
- Reference: `THRGIBBSF90` (BLUPF90 family) for multi-trait ordinal; install via the standard BLUPF90 distribution.
- Fixture: simulated 3-trait design (2 ordinal + 1 continuous) with known R + G matrices.
- Compare: posterior modes of liability variance components; per-trait β for a few markers.
- Agreement target: |Δvar comp| ≤ 1e-2; |Δβ| ≤ 3 sig-figs (observed-then-floored).

**Acceptance gate:** structural same as C1.

- [ ] Dispatch C4
- [ ] Audit
- [ ] Commit `Tier 3 C4: threshold-linear fixture`

---

### Agent C5: Knockoff FDR-control fixture

**Files:**
- Create: `validation/specialty/knockoff/{generate.py,run_torchgwas.py,run_reference.R,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/knockoff_lmm.py` (KnockoffLMM; Sesia et al. 2020; group knockoffs + knockoff+ filter).

**Tool-specific guidance:**
- Reference: R `knockoff` package (CRAN) at pinned version.
- Fixture: simulated G with known causal SNPs at a target FDR (e.g., 0.05 nominal). Sample size large enough to demonstrate FDR control empirically (≥ 100 replicates of n=500 simulations recommended; document budget).
- Compare: empirical FDR at the nominal level; power at the same level.
- Agreement target: |ΔFDR| ≤ 0.02 across replicates (observed-then-floored).

**Acceptance gate:** structural same as C1.

- [ ] Dispatch C5
- [ ] Audit
- [ ] Commit `Tier 3 C5: knockoff fixture`

---

### Agent C6: OCF (Orthogonal Cross-Fit) coverage fixture

**Files:**
- Create: `validation/specialty/ocf/{generate.py,run_torchgwas.py,run_reference.py,compare.py,README.md,results/}`

**TorchGWAS comparison target:** `torchgwas/models/ocf_lmm.py` (OCFLMM; Chernozhukov et al. 2018; DML cross-fit with K-fold sample splitting).

**Tool-specific guidance:**
- Reference: the Chernozhukov-published reference implementation in R / Python (e.g., `DoubleML` package) at pinned version. If unavailable, use a hand-coded DML pipeline matching the paper's eq. (3.1) — document explicitly.
- Fixture: simulated G + nuisance covariates with known SNP effect under confounding. Run ≥ 100 replicates to demonstrate coverage.
- Compare: empirical coverage of the 95% CI; debiased β estimator.
- Agreement target: empirical coverage 0.92–0.98 (observed-then-floored at the empirical extreme).

**Acceptance gate:** structural same as C1.

- [ ] Dispatch C6
- [ ] Audit
- [ ] Commit `Tier 3 C6: OCF fixture`

---

## 7. Dispatch execution (the main session does this)

This is the actual command flow the main session runs once Tier 0 has landed.

- [ ] **Step 1: Verify Tier 0 acceptance**

Run: `git rev-parse --abbrev-ref HEAD && test -f docs/superpowers/plans/2026-05-15-paper-tier0-RESULT.md && echo "tier 0 OK"`
Expected: prints `paper/genome-biology-methods` + `tier 0 OK`

- [ ] **Step 2: Build the 14 agent briefs from the template in §3**

For each agent, fill `<...>` placeholders against the per-agent section (§4–§6).

- [ ] **Step 3: Dispatch all 14 in a single message with 14 Agent tool calls**

Use `subagent_type: general-purpose` for all. Set `isolation: worktree` for each so they cannot interfere with the main working tree. Run in foreground (the main session needs each result before audit).

Token-budget fallback: if 14-way concurrency is impractical, dispatch in waves: wave 1 = Tier 1 (5 agents) + Tier 2 (3 agents); wave 2 = Tier 3 (6 agents). Sequencing is a budget decision, not a correctness one.

- [ ] **Step 4: Audit each agent's output against its acceptance gate (§4–§6)**

For each completed agent, run the agent's specific acceptance-gate command block. Failure means re-dispatch or escalate per §8 audit policy.

- [ ] **Step 5: Commit successful agents one tier at a time**

```bash
git add validation/external/{metaxcan,smr,coloc,hyprcoloc,hapref}
git commit -m "Tier 1: 5 reference-tool harnesses (A1-A5)"

git add validation/multiomics/
git commit -m "Tier 2: 3 multi-omics fixtures (B1-B3)"

git add validation/specialty/
git commit -m "Tier 3: 6 specialty validation fixtures (C1-C6)"
```

---

## 8. Audit policy

After each agent reports completion, the main session (not a subagent) runs:

- [ ] All acceptance criteria met for the agent?
- [ ] Pre-flight gate present in every shell script for that agent? (`grep -l preflight.sh <agent-dir>/*.sh`)
- [ ] SHA256 manifest present and validates? (`cd <agent-dir>/results && sha256sum -c manifest.sha256`)
- [ ] Tolerances *measured* (not aspirational)? Look for an actual number in `results/agreement.json` or equivalent.
- [ ] Any F3 divergence routed through `docs/validation_findings.md`?
- [ ] Cumulative count of unrelated divergences across Tiers 1-3 < 3? (If ≥ 3, halt all dispatch and escalate per `memory/feedback_f3.md` C4 emergency-stop policy.)

If any check fails: re-dispatch the agent with explicit pointer to the failed criterion, or escalate to the user if structural.

---

## 9. Acceptance for Tiers 1-3 (gate before transitioning to Plan C / Tier 4)

- [ ] All 5 Tier 1 harnesses present under `validation/external/{metaxcan,smr,coloc,hyprcoloc,hapref}/` with `results/agreement.json` populated
- [ ] All 3 Tier 2 fixtures present under `validation/multiomics/{sim,gtex_ukb,plant}/` with `fixtures/aligned.parquet` (or equivalent) + `manifest.sha256`
- [ ] All 6 Tier 3 specialty fixtures present under `validation/specialty/{survival,rr,family,threshold,knockoff,ocf}/` with `results/` populated
- [ ] All 14 agents' commits applied to `paper/genome-biology-methods`
- [ ] Cumulative F3 divergences logged in `docs/validation_findings.md` (or zero if none)
- [ ] No autonomous push
- [ ] User has been shown the audit summary + green-light for Tier 4

When all checkboxes pass, proceed to Plan C: `docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md`.
