# Paper Tier 4 — Reproducibility + manuscript draft implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for D2 (figure scripts, parallelizable per figure) and superpowers:executing-plans for D1 / D3 / D4 / D5 (serial). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Convert the validated artifacts from Plans A + B into (i) a one-command-reproducible figure pipeline, (ii) a manuscript draft (~8400 words, 7 main figures), (iii) bioRxiv preprint, (iv) Genome Biology Methods submission.

**Architecture:** Tier 4 is serial. D1 stands up the reproducibility-repo skeleton + driver. D2 authors one figure script per main figure plus three new artifacts (GU + LRO simulators + streaming p-sweep). D3 writes the manuscript section-by-section against the §4 word budget in the design spec. D4 is internal review. D5 submits.

**Tech Stack:** bash + python (matplotlib + pandas + numpy + scipy) + git + bioRxiv / GB submission portal.

---

## 1. Pre-conditions

- [ ] Plan A (Tier 0) complete: working branch `paper/genome-biology-methods` carries verified infrastructure
- [ ] Plan B (Tiers 1-3) complete: all 14 agent outputs landed, all acceptance gates passed, divergence count < 3
- [ ] Design spec readable at `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`

Verify:

```bash
git rev-parse --abbrev-ref HEAD                # paper/genome-biology-methods
ls validation/external/{metaxcan,smr,coloc,hyprcoloc,hapref}/results/agreement.json
ls validation/multiomics/{sim,gtex_ukb,plant}/fixtures/manifest.sha256
ls validation/specialty/{survival,rr,family,threshold,knockoff,ocf}/results/manifest.sha256
```

Expected: every command exits 0.

---

## 2. D1 — Reproducibility repo skeleton

The reproducibility repo is a *separate* git repo at `github.com/sikiru-atanda/torchgwas-paper-reproducibility` (to be created at D5; for now, develop locally under `paper/reproducibility/`).

### Task D1.1: Create reproducibility-repo directory structure

**Files:**
- Create: `paper/reproducibility/reproduce_paper.sh`
- Create: `paper/reproducibility/preflight.sh`
- Create: `paper/reproducibility/stages/01_install_references.sh`
- Create: `paper/reproducibility/stages/02_stage_fixtures.sh`
- Create: `paper/reproducibility/stages/03_run_references.sh`
- Create: `paper/reproducibility/stages/04_run_torchgwas.sh`
- Create: `paper/reproducibility/stages/05_run_streaming_bench.sh`
- Create: `paper/reproducibility/stages/06_run_native_bench.sh`
- Create: `paper/reproducibility/stages/07_run_multiomics.sh`
- Create: `paper/reproducibility/stages/08_render_figures.py`
- Create: `paper/reproducibility/manifest.json` (empty placeholder, populated by stages)
- Create: `paper/reproducibility/README.md`

- [ ] **Step 1: Create directories**

Run: `mkdir -p paper/reproducibility/stages paper/reproducibility/output`
Expected: directories exist.

- [ ] **Step 2: Author the top-level driver `reproduce_paper.sh`**

```bash
cat > paper/reproducibility/reproduce_paper.sh <<'EOF'
#!/usr/bin/env bash
# One-command driver: regenerate every number in every figure/table.
set -euo pipefail
cd "$(dirname "$0")"

bash preflight.sh
bash stages/01_install_references.sh
bash stages/02_stage_fixtures.sh
bash stages/03_run_references.sh
bash stages/04_run_torchgwas.sh
bash stages/05_run_streaming_bench.sh
bash stages/06_run_native_bench.sh
bash stages/07_run_multiomics.sh
python stages/08_render_figures.py

echo "Reproducibility complete. See output/ for figures + manifest.json for every numeric result."
EOF
chmod +x paper/reproducibility/reproduce_paper.sh
```

- [ ] **Step 3: Author `preflight.sh` (sources from `validation/external/_lib/preflight.sh`)**

```bash
cat > paper/reproducibility/preflight.sh <<'EOF'
#!/usr/bin/env bash
# Pre-flight gate: assert disk + RAM + GPU + network before any download.
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
source "$REPO_ROOT/validation/external/_lib/preflight.sh"

require_disk_gb 50    # cumulative across all stages
require_ram_gb 16     # baseline workstation
require_network       # for fetches
echo "Pre-flight OK."
EOF
chmod +x paper/reproducibility/preflight.sh
```

- [ ] **Step 4: Author each stage as a thin orchestrator that calls into the per-tool harnesses**

For each stage script, the body is a sequence of `bash validation/external/<tool>/{install,fetch,run}.sh` calls. The full text is mechanical; the agent should write each in the same style. Example for stage 01:

```bash
cat > paper/reproducibility/stages/01_install_references.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
for tool in bolt_lmm gapit gemma gwaspoly ldsc plink2 regenie saige susieR twosamplemr metaxcan smr coloc hyprcoloc hapref; do
  echo "==> Installing $tool"
  bash "$REPO_ROOT/validation/external/$tool/install.sh"
done
EOF
chmod +x paper/reproducibility/stages/01_install_references.sh
```

Write stages 02–07 by analogy (fetch / run / etc.).

- [ ] **Step 5: Author `08_render_figures.py` skeleton**

```python
cat > paper/reproducibility/stages/08_render_figures.py <<'EOF'
#!/usr/bin/env python3
"""Render F1-F7 from the harness outputs into paper/reproducibility/output/."""
import json
import pathlib

OUTPUT = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT.mkdir(exist_ok=True)
MANIFEST = OUTPUT.parent / "manifest.json"

from paper.reproducibility import render_figures  # filled by D2 tasks

manifest = {}
for fig_id, render_fn in render_figures.REGISTRY.items():
    out_path, numeric_result = render_fn(OUTPUT)
    manifest[fig_id] = {"file": str(out_path), "numbers": numeric_result}

MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
print(f"Wrote {MANIFEST}")
EOF
chmod +x paper/reproducibility/stages/08_render_figures.py
```

- [ ] **Step 6: Author `README.md` and commit**

```bash
cat > paper/reproducibility/README.md <<'EOF'
# TorchGWAS paper reproducibility

One-command pipeline: `bash reproduce_paper.sh`. Regenerates every number in every figure/table in the Genome Biology Methods manuscript.

Output: `output/F{1..7}.pdf` and `manifest.json` (every numeric result + commit SHA).

See `paper/reproducibility/stages/` for individual stage scripts. See `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md` for the authoritative spec.
EOF

git add paper/reproducibility/
git commit -m "Tier 4 D1: reproducibility repo skeleton + driver"
```

---

## 3. D2 — Figure-rendering scripts + 3 new artifacts (parallelizable)

These 10 tasks can be dispatched as 10 parallel subagents (one per figure script + 3 helper scripts). Each subagent reads from harness outputs and writes a `.pdf` or `.png` to `paper/reproducibility/output/`.

### Common scaffolding (do once before dispatching)

- [ ] **Step 1: Create the render-functions module**

```bash
mkdir -p paper/reproducibility/render_figures
touch paper/reproducibility/render_figures/__init__.py
cat > paper/reproducibility/render_figures/__init__.py <<'EOF'
"""Registry of figure-rendering functions. Each render_fn takes an output dir
and returns (output_path, numeric_dict)."""
REGISTRY = {}

def register(fig_id):
    def _decorator(fn):
        REGISTRY[fig_id] = fn
        return fn
    return _decorator
EOF
```

- [ ] **Step 2: Commit**

```bash
git add paper/reproducibility/render_figures/
git commit -m "Tier 4 D2: figure-render registry scaffold"
```

### Task D2.1: Render F1 — capability map

**Files:**
- Create: `paper/reproducibility/render_figures/f1_capability_map.py`
- Output: `output/F1.pdf`

**Agent brief outline:** Render a sunburst / treemap of all ~50 capabilities grouped by cluster (variant / haplotype / multi-omics / polyploid / specialty / GLM/GLMM family / post-GWAS / multiple-testing / viz / annotate / pre-process / native). Source data: hand-curated JSON of the 7 clusters × their constituent CLI subcommands + module functions. The agent must extract the data from `torchgwas/cli.py` (40 dispatch dict entries) + the module-level public surface.

- [ ] Dispatch subagent for F1
- [ ] Audit: `python -c "from paper.reproducibility.render_figures import f1_capability_map; f1_capability_map.render('/tmp/F1.pdf')"` produces F1.pdf > 10 KB
- [ ] Commit `Tier 4 D2.1: F1 capability map`

### Task D2.2: Render F2 — 15-tool equivalence grid (+ 3 fixture columns)

**Files:**
- Create: `paper/reproducibility/render_figures/f2_equivalence_grid.py`
- Reads: `validation/external/<tool>/results/agreement.json` for each of 15 tools + 3 fixtures
- Output: `output/F2.pdf`

**Agent brief outline:** Render a heatmap grid: 15 algorithmic tool columns + 3 fixture columns × rows for β / SE / p / PIP / etc. Color = observed sig-fig agreement; annotation = floored tolerance.

- [ ] Dispatch
- [ ] Audit (F2 produces a heatmap, every cell colored or NA-marked)
- [ ] Commit `Tier 4 D2.2: F2 equivalence grid`

### Task D2.3: Render F3 — haplotype layer (3 panels)

**Files:**
- Create: `paper/reproducibility/render_figures/f3_haplotype.py`
- Reads: `validation/external/plink2/`, `validation/external/hapref/`, `validation/external/soymd/`
- Output: `output/F3.pdf`

**Panels:** A (13 LD-block methods vs PLINK 1.9 Gabriel on MDP / SoyMD); B (9 haplotype GWAS methods on SoyMD); C (novel methods PCHT/HHCT/BayesHap on a real hit).

- [ ] Dispatch
- [ ] Audit (3 panels present)
- [ ] Commit `Tier 4 D2.3: F3 haplotype`

### Task D2.4: Render F4 — multi-omics integration (3 panels)

**Files:**
- Create: `paper/reproducibility/render_figures/f4_multiomics.py`
- Reads: `validation/multiomics/{sim,gtex_ukb,plant}/`, `validation/external/{metaxcan,smr,coloc,hyprcoloc}/results/`
- Output: `output/F4.pdf`

**Panels:** A (simulated truth recovery: TWAS Z, SMR β, coloc PP.H4, mediation β — recovery ≥ 0.95 target); B (GTEx + UKB SORT1/LDL worked example); C (plant multi-omics worked example).

- [ ] Dispatch
- [ ] Audit (3 panels, all 4 metrics in panel A annotated)
- [ ] Commit `Tier 4 D2.4: F4 multi-omics`

### Task D2.5: Render F5 — polyploid pipeline (4 panels)

**Files:**
- Create: `paper/reproducibility/render_figures/f5_polyploid.py`
- Reads: `validation/external/gwaspoly/results/`, Phase 55/56 fixture outputs
- Output: `output/F5.pdf`

**Panels:** A (dosage-call accuracy); B (PolyOrigin F1 phasing recovery); C (polyploid GWAS vs GWASpoly); D (arbitrary-ploidy demo k=4,6,8).

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D2.5: F5 polyploid`

### Task D2.6: Render F6 — GPU + biobank streaming (4 panels)

**Files:**
- Create: `paper/reproducibility/render_figures/f6_gpu_streaming.py`
- Reads: `bench/native_speedups.py` outputs, new `bench/streaming_p_sweep.py` (Task D2.9 below), `validation/external/ukb/`, GPU parity test outputs
- Output: `output/F6.pdf`

**Panels:** A (25 native kernel speedup violin); B (streaming-vs-materialized memory slope across p ∈ [10⁴, 10⁶] + a UKB-scale data point from `validation/external/ukb/`); C (sparse-GRM PCG-REML 1.8× speedup); D (CPU↔GPU parity scatter from `.github/workflows/gpu.yml` artifacts).

- [ ] Dispatch (waits on D2.9 for panel B)
- [ ] Audit
- [ ] Commit `Tier 4 D2.6: F6 GPU + streaming`

### Task D2.7: Render F7 — specialty models (6+2 panels)

**Files:**
- Create: `paper/reproducibility/render_figures/f7_specialty.py`
- Reads: `validation/specialty/{survival,rr,family,threshold,knockoff,ocf}/results/`, GU + LRO simulator outputs (Tasks D2.10 below)
- Output: `output/F7.pdf`

**Panels:** 6 from Tier 3 (C1-C6) + 2 internal-only (GU + LRO). The internal-only panels are visually distinguished (e.g., with an "internal validation" border).

- [ ] Dispatch (waits on D2.10 for GU+LRO)
- [ ] Audit
- [ ] Commit `Tier 4 D2.7: F7 specialty`

### Task D2.8: New `bench/streaming_p_sweep.py` (canonical F6 panel B source)

**Files:**
- Create: `bench/streaming_p_sweep.py`

**Agent brief outline:** Author a script that runs torchgwas lmm-scan in both streaming and materialized mode at p ∈ {1e4, 3e4, 1e5, 3e5, 1e6} on a synthetic fixture (n=2000), measures peak RSS, and emits `bench/streaming_p_sweep_results.json`. Use the existing `tests/test_streaming_memory.py` patterns as the structural reference. Pre-flight gate from `validation/external/_lib/preflight.sh` required.

- [ ] Dispatch
- [ ] Audit (`python bench/streaming_p_sweep.py --quick` produces a JSON with 5 entries)
- [ ] Commit `Tier 4 D2.8: streaming p-sweep canonical bench`

### Task D2.9: GU + LRO internal-consistency simulators

**Files:**
- Create: `validation/specialty/gu/{generate.py,run_torchgwas.py,compare.py,README.md,results/}`
- Create: `validation/specialty/lro/{generate.py,run_torchgwas.py,compare.py,README.md,results/}`

**Agent brief outline:** GU validates recovery of known dosage variance; LRO validates block-resolved h² recovery. Both are internal-consistency only (no external reference exists per the spec §10.6). Document this asymmetry in each README.md and in the F7 panel caption.

- [ ] Dispatch (one combined subagent, two outputs)
- [ ] Audit
- [ ] Commit `Tier 4 D2.9: GU + LRO internal-consistency simulators`

### Task D2.10: Commit all D2 outputs and verify the manifest

- [ ] **Step 1: Run the full figure-rendering pipeline**

Run: `python paper/reproducibility/stages/08_render_figures.py`
Expected: 7 PDFs in `paper/reproducibility/output/` + `paper/reproducibility/manifest.json` populated.

- [ ] **Step 2: Commit**

```bash
git add paper/reproducibility/output/ paper/reproducibility/manifest.json
git commit -m "Tier 4 D2.10: full figure pipeline output + manifest"
```

---

## 4. D3 — Manuscript draft (11 sections, ~8400 words)

Each manuscript section is one bite-sized task. The agent for each section reads the corresponding figure + supplement material + the §4 word budget and produces the section text in markdown under `paper/manuscript/`. Cross-references use the figure / section labels from the spec.

### Common scaffolding (do once)

- [ ] **Step 1: Create the manuscript directory and metadata**

```bash
mkdir -p paper/manuscript/sections paper/manuscript/supplement
cat > paper/manuscript/metadata.yaml <<'EOF'
title: "TorchGWAS: a GPU-accelerated, polyploid-first toolkit unifying variant, haplotype, and multi-omics GWAS at biobank scale, with end-to-end numerical equivalence to fifteen reference tools"
venue: "Genome Biology — Methods (Software)"
authors:
  - name: "Sikiru Atanda"
    email: "sikiruandfriends@gmail.com"
    affiliation: "PulseSmartLab Innovations"
    corresponding: true
license: "Apache 2.0 (subject to confirmation)"
repo: "github.com/sikiru-atanda/torchgwas"
reproducibility_repo: "github.com/sikiru-atanda/torchgwas-paper-reproducibility"
spec: "docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md"
EOF

cp paper/_archive_app_note/references.bib paper/manuscript/references.bib 2>/dev/null || \
  touch paper/manuscript/references.bib
git add paper/manuscript/
git commit -m "Tier 4 D3.0: manuscript scaffold + metadata"
```

### Section D3.1: Background (~700 words)

**File:** `paper/manuscript/sections/01_background.md`

**Agent brief:** Write the Background section. Frame the biobank-scale GWAS landscape: regenie / SAIGE / BOLT-LMM dominate diploid LMM; GWASpoly is the only polyploid LMM; SuSiE / SuSiE-RSS for fine-mapping; multi-omics integration (TWAS / SMR / coloc / mediation) typically requires chaining 5+ single-purpose tools. Articulate the gap: no single toolkit unifies variant + haplotype + multi-omics + polyploid + biobank-streaming on GPU with cross-tool reproducibility. Cite primary sources via `references.bib`.

- [ ] Dispatch
- [ ] Audit (~700 words ±10%; ≥10 citations; no novelty claim unqualified)
- [ ] Commit `Tier 4 D3.1: Background`

### Section D3.2: Architecture & capability surface (~600 words; anchors F1)

**File:** `paper/manuscript/sections/02_architecture.md`

**Agent brief:** Describe the data flow (format detection → imputation/phasing → QC → encoding → GRM → null fit → scan → multiple testing → reporting). List the ~50 capabilities grouped by cluster, matching F1 visually. Note: 40 CLI subcommands (re-verify on working branch — may be higher with NA1 `bayes-scan-rss`).

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.2: Architecture`

### Section D3.3: Reference-tool equivalence (15 tools) (~900 words; anchors F2)

**File:** `paper/manuscript/sections/03_equivalence.md`

**Agent brief:** Describe the 15 algorithmic reference tools + 3 real-data fixtures. State the observed-then-floored tolerance protocol. Walk through F2 — every tool PASSING — and call out the 13 V1 fix-now production fixes from the validation campaign (per `docs/validation_findings.md`).

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.3: Equivalence`

### Section D3.4: Haplotype layer (~1100 words; anchors F3)

**File:** `paper/manuscript/sections/04_haplotype.md`

**Agent brief:** Describe the 13 LD-block-design methods (4 classical + 5 novel + 3 literature + 1 diagnostic). Describe the 9 haplotype GWAS methods (HTR / window / block / SKAT + PCHT + HHCT + HSKAT + HapGxE + BayesHap). Cite primary sources. Describe the multi-env / multi-trait / MT-MET haplotype extensions (Phase 47).

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.4: Haplotype`

### Section D3.5: Multi-omics integration (~1200 words; anchors F4)

**File:** `paper/manuscript/sections/05_multiomics.md`

**Agent brief:** Describe TWAS (S-PrediXcan + PrediXcan) + SMR/HEIDI + coloc + hyprcoloc + GRM-corrected mediation (`torchgwas.multiomics`). Walk through F4's three panels (simulated truth recovery + GTEx/UKB SORT1/LDL + plant multi-omics). Emphasize this is the SNP ↔ transcriptomics comparative analysis cluster that justified the manuscript-rewrite scope.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.5: Multi-omics`

### Section D3.6: Polyploid pipeline (~900 words; anchors F5)

**File:** `paper/manuscript/sections/06_polyploid.md`

**Agent brief:** Describe the polyploid-first design (arbitrary ploidy k; diploid is k=2). Walk through F5 — dosage call (Phase 55, updog) → F1 phasing (Phase 56, PolyOrigin) → polyploid GWAS → polyploid LD blocks → polyploid haplotype GWAS. End-to-end on potato F1 + autotetraploid + simulated hexaploid/octoploid.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.6: Polyploid`

### Section D3.7: Specialty models (~700 words; anchors F7)

**File:** `paper/manuscript/sections/07_specialty.md`

**Agent brief:** Describe the specialty cluster: survival (Cox PH frailty + SPACox SPA, Phase 35), random regression (Legendre / B-spline, Phase 38) + spatiotemporal RR, within-family (Young 2022, Phase 23), threshold-linear (Bermann 2026, Phase 22), knockoff (Sesia 2020, Phase 27), OCF (Chernozhukov 2018, Phase 26). Disclose that GU + LRO are internal-only (no external reference).

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.7: Specialty`

### Section D3.8: GPU + biobank streaming (~800 words; anchors F6)

**File:** `paper/manuscript/sections/08_gpu_streaming.md`

**Agent brief:** Describe the device-aware dispatcher (`torchgwas._dispatch.select_path`); the 25 native C++ accelerators; the streaming I/O surface (`iter_chunks` on every reader; `grm_vanraden_streaming`; sparse-GRM + PCG-REML). Walk through F6 — native speedup distribution + memory slope + sparse-GRM 1.8× + GPU parity. UKB-scale data point from NA3 harness.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.8: GPU + streaming`

### Section D3.9: Methods (compressed) (~800 words)

**File:** `paper/manuscript/sections/09_methods.md`

**Agent brief:** Methods section: streaming I/O surface (formal); 6-mode optimizer (PX-EM → AI-REML → LBFGS-autograd → MM → PCG → derivative-free); native-dispatch protocol; observed-then-floored tolerance protocol. Cross-ref the supplement for full mathematical detail.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.9: Methods`

### Section D3.10: Discussion + limitations + roadmap (~500 words)

**File:** `paper/manuscript/sections/10_discussion.md`

**Agent brief:** Limitations: no Windows GPU; phase-coherence check pending for polyploid haplotypes (per Tier-A spec `2026-04-24-haplotypegwas-shape-adapter-design.md`); UKB-scale n=500K LMM-from-scratch requires sparse-GRM PCG-REML path. Roadmap: Phases 57/58 (designed, not shipped); future cross-ancestry haplotype methods.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.10: Discussion`

### Section D3.11: Availability (~200 words)

**File:** `paper/manuscript/sections/11_availability.md`

**Agent brief:** License (Apache 2.0, subject to user confirmation); installation (`pip install torchgwas`, bioconda recipe state TBC); docs site URL; reproducibility script `paper/reproducibility/reproduce_paper.sh`; bench data accession.

- [ ] Dispatch
- [ ] Audit
- [ ] Commit `Tier 4 D3.11: Availability`

### Section D3.12: Abstract (~250 words)

**File:** `paper/manuscript/sections/00_abstract.md`

**Agent brief:** Write the structured abstract last (after all body sections exist). Motivation → Results → Availability → Contact. ≤ 250 words. Targets the 4-claim synthesized thesis from §3 of the spec.

- [ ] Dispatch
- [ ] Audit (`wc -w` ≤ 250)
- [ ] Commit `Tier 4 D3.12: Abstract`

### Section D3.13: Supplement (S1–S8)

The spec §6 mandates 8 supplementary deliverables. Each can be dispatched as its own subagent.

**File tree:**
- `paper/manuscript/supplement/S1_equivalence_ledger.md`
- `paper/manuscript/supplement/S2_capability_inventory.md`
- `paper/manuscript/supplement/S3_validation_findings.md` (verbatim copy of `docs/validation_findings.md` with a header pointing back to the canonical location)
- `paper/manuscript/supplement/S4_reproducibility_manifest.md` (pinned versions + install scripts + run order; rendered from harness install.sh files)
- `paper/manuscript/supplement/S5_tolerance_protocol.md` (observed-then-floored methodology + per-tool numerical drift tables)
- `paper/manuscript/supplement/S6_extended_figures.md` (per-haplotype-method internals; extended F4 panels; per-ploidy F5 parity)
- `paper/manuscript/supplement/S7_sensitivity_analyses.md` (pre-flight gate behavior; OpenMP off / native off / GPU off parity scans)
- `paper/manuscript/supplement/notebooks/F{1..7}_demo.ipynb` (one demo notebook per main figure, each reproduces its figure end-to-end)

**Agent briefs (one per S-item; 8 subagents):**

- **S1**: Walk `validation/external/<tool>/results/agreement.json` for all 15 tools + 3 fixtures; emit a single table with rows = tool, columns = metric (β / SE / p / PIP / coverage / FDR), cells = observed agreement value + floored tolerance.
- **S2**: Walk `torchgwas.cli` dispatch dict + every public function under `torchgwas/{io,preprocess,linalg,models,scan,stats,ld,optim,pgs,postgwas,multiomics,viz,annotate}`; emit a table with columns = (capability, CLI subcommand, module path, phase number, test file, validation harness). Target: every one of the ~50 capabilities has a row.
- **S3**: Copy `docs/validation_findings.md` verbatim into the supplement with a header pointing to the canonical path.
- **S4**: Crawl every `validation/external/*/install.sh` for pinned versions; assemble into a single table (tool, version, source URL, SHA256, install command summary).
- **S5**: Describe the observed-then-floored protocol (cite `memory/feedback_validation_spec.md` semantics in third-person). Per-tool numerical drift tables: pull from S1.
- **S6**: For each main figure, render 2–3 supplementary panels with the same renderer code (e.g., F3.1–F3.9 = per-haplotype-method internals; F4.1–F4.3 = per-multi-omics-method internals at extended n; F5.1–F5.5 = per-ploidy parity at k=4,6,8 + simulated k=10).
- **S7**: Run the existing test suite under each of: TORCHGWAS_DISABLE_NATIVE=1, TORCHGWAS_DISABLE_GPU=1, TORCHGWAS_DISABLE_OPENMP=1 (build re-required). Emit a parity table showing the equivalence-tolerance under each toggle. Demonstrates the spec's "GPU + CPU agree at FP64 tolerance" claim.
- **S8**: For each F1–F7, author a Jupyter notebook that reads from the harness outputs and renders the figure inline, with prose explaining each step. Notebooks should run end-to-end on a fresh checkout after `bash paper/reproducibility/reproduce_paper.sh`.

**Acceptance gate per S-item:**

```bash
test -f paper/manuscript/supplement/S{1..7}_*.md && \
ls paper/manuscript/supplement/notebooks/F{1..7}_demo.ipynb && \
wc -l paper/manuscript/supplement/S{1..7}_*.md
```

Expected: all 7 markdown files exist; all 7 notebooks exist; line counts are non-zero.

- [ ] **Step 1: Dispatch the 8 supplement subagents in parallel**
- [ ] **Step 2: Audit each S-item against its acceptance gate above**
- [ ] **Step 3: Commit**

```bash
git add paper/manuscript/supplement/
git commit -m "Tier 4 D3.13: supplement S1-S8 (full equivalence ledger + capability inventory + findings + reproducibility manifest + tolerance protocol + extended figures + sensitivity + demo notebooks)"
```

---

### Section D3.14: Assemble + final word-count check

- [ ] **Step 1: Concatenate sections in order**

```bash
cat paper/manuscript/sections/00_abstract.md \
    paper/manuscript/sections/01_background.md \
    paper/manuscript/sections/02_architecture.md \
    paper/manuscript/sections/03_equivalence.md \
    paper/manuscript/sections/04_haplotype.md \
    paper/manuscript/sections/05_multiomics.md \
    paper/manuscript/sections/06_polyploid.md \
    paper/manuscript/sections/07_specialty.md \
    paper/manuscript/sections/08_gpu_streaming.md \
    paper/manuscript/sections/09_methods.md \
    paper/manuscript/sections/10_discussion.md \
    paper/manuscript/sections/11_availability.md \
    > paper/manuscript/full_draft.md
```

- [ ] **Step 2: Word-count check**

Run: `wc -w paper/manuscript/full_draft.md`
Expected: roughly 8400 words (target 8000–9000 for GB Methods).

- [ ] **Step 3: Commit**

```bash
git add paper/manuscript/full_draft.md
git commit -m "Tier 4 D3.13: assembled full draft"
```

---

## 5. D4 — Internal review

### Task D4.1: Spec-alignment review

- [ ] **Step 1: For each claim in the manuscript, find its anchor**

For every numerical claim, trace to:
- Figure / supplement table that contains the number
- `manifest.json` entry produced by `reproduce_paper.sh`
- Reference-tool harness output that backs it

Any claim without an anchor → fix the claim or fix the anchor.

- [ ] **Step 2: Capability-coverage review**

For every cluster in the spec's lead thesis (variant + haplotype + multi-omics + polyploid + biobank + reproducibility), confirm the manuscript surfaces it explicitly in the abstract AND in a dedicated Results section AND in at least one figure.

- [ ] **Step 3: Tolerance-policy review**

For every reported agreement number, confirm it is observed-then-floored (matches `results/agreement.json` from the relevant harness; not aspirational).

- [ ] **Step 4: Novelty-claim review**

For every "first", "novel", "to our knowledge" claim, confirm it carries the "to our knowledge" qualifier (per repo convention).

- [ ] **Step 5: Commit the review log**

```bash
cat > paper/manuscript/D4_review.md <<EOF
# D4 internal review log

Date: $(date -I)
Manuscript: paper/manuscript/full_draft.md
Spec: docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md

## Spec-alignment
$(# fill from steps 1-2)

## Tolerance-policy
$(# fill from step 3)

## Novelty-claim audit
$(# fill from step 4)
EOF

git add paper/manuscript/D4_review.md
git commit -m "Tier 4 D4: internal review log"
```

### Task D4.2: User review gate

- [ ] **Step 1: Tag the draft**

Run: `git tag -a paper-draft-v1 -m "Tier 4 D4: draft ready for user review"`

- [ ] **Step 2: Report to user**

Print:
```
Tier 4 D3 + D4 complete.
Manuscript: paper/manuscript/full_draft.md (target ~8400 words)
Figures: paper/reproducibility/output/F1-F7.pdf
Reproducibility: paper/reproducibility/reproduce_paper.sh
Review log: paper/manuscript/D4_review.md

Tag: paper-draft-v1
Awaiting user review before D5 submission.
```

---

## 6. D5 — Submission (user-gated; main session does not auto-submit)

### Task D5.1: bioRxiv preprint

- [ ] **Step 1: User reviews the draft + figures + reproducibility output**
- [ ] **Step 2: User confirms author list + affiliations + license**
- [ ] **Step 3: User confirms bioconda recipe state (or revises availability text)**
- [ ] **Step 4: Push the working branch with explicit user approval** (`git push -u origin paper/genome-biology-methods`)
- [ ] **Step 5: Create the public reproducibility repo on GitHub** (user-driven; this plan does not auto-create org/personal repos)
- [ ] **Step 6: bioRxiv submission via web portal** (user-driven)
- [ ] **Step 7: Commit the preprint DOI to `paper/manuscript/metadata.yaml`**

### Task D5.2: Genome Biology Methods submission

- [ ] **Step 1: Convert manuscript to GB submission format** (LaTeX or .docx per current journal requirements)
- [ ] **Step 2: Cover letter** (highlights the unification thesis, the 15-tool reproducibility, the polyploid-first claim)
- [ ] **Step 3: User submits via portal**
- [ ] **Step 4: Commit submission metadata**

---

## 7. Acceptance for Tier 4

- [ ] Reproducibility repo skeleton runs end-to-end: `bash paper/reproducibility/reproduce_paper.sh` exits 0 and produces 7 PDFs + populated `manifest.json`
- [ ] All 7 figures rendered from real harness outputs (no placeholder or synthetic-only figures)
- [ ] Manuscript word count within 8000–9000 (`wc -w paper/manuscript/full_draft.md`)
- [ ] Every numerical claim in the manuscript has an anchor in `manifest.json` or in a harness `results/agreement.json`
- [ ] D4 review log committed; no unresolved findings
- [ ] User has approved the draft + figures
- [ ] No autonomous push or submission

When all checkboxes pass, the paper is ready for bioRxiv + Genome Biology submission per D5.
