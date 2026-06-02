# Pillar D — Reproducibility audit

The TorchGenomics validation campaign committed reference outputs from three
upstream tools (GEMMA, GAPIT, GWASpoly) before the Pillar A campaign
began. Pillars A / B / C diff *TorchGenomics* against those committed
fixtures. **Pillar D asks the converse question:**

> If we install a fresh upstream tool today and re-run it on the same
> input, does it still reproduce the committed reference fixture?

If yes → the fixture is *authoritative* and remains the canonical
reference for future regression checks. If no → we have a `fixture-drift`
finding (the upstream tool's behavior or the input data has subtly
changed since the fixture was captured) and the user must decide whether
to regenerate the fixture or accept the drift.

This is the **fourth and final pillar** of the validation campaign.

## Layout

```
validation/reproducibility/
├── rerun_goldens.py        # orchestrator: install → fetch → run → diff
├── compare_fixtures.py     # diff fresh tool output vs committed fixture
├── README.md               # this file (protocol + drift table)
└── outputs/                # gitignored; per-tool drift JSON reports
    ├── gemma_drift.json
    ├── gapit_drift.json
    └── gwaspoly_drift.json
```

The actual fresh upstream output lands under
`validation/external/<tool>/outputs/` — Pillar D *reuses* the Pillar B
harness end-to-end (no duplicate install / run code).

## Protocol

1. **Pre-flight** — orchestrator-level check: ≥ 10 GB disk on `/home`,
   ≥ 4 GB available RAM. Each per-tool harness re-runs its own pre-flight
   inside `install.sh` / `run_*.sh`.
2. **Install** — `validation/external/<tool>/install.sh`. Idempotent.
   Pinned version + checksum where applicable. R-package installs
   (GAPIT, GWASpoly) attempt non-interactive install via remotes /
   BiocManager into a per-harness library at `bin/Rlib/`.
3. **Fetch data** — `validation/external/<tool>/fetch_data.sh`. Idempotent.
   Copies committed fixtures from `gemma_demo/` / `benchmark/data/`
   into the harness's `data/` dir.
4. **Run** — `validation/external/<tool>/run_<tool>.sh`. Produces fresh
   output in `validation/external/<tool>/outputs/`. For GAPIT and
   GWASpoly the harness falls back to copying the committed reference
   fixture if the upstream tool is unavailable; that fallback is a
   **no-op for Pillar D** (the diff would compare the fixture against
   itself), so the orchestrator probes for tool availability and records
   an `infra-blocker` if the tool can't actually run.
5. **Diff** — `compare_fixtures.py --tool <name>` matches fresh output
   against the committed fixture by SNP / marker ID and computes
   element-wise |Δ| on β / SE / log-likelihood / p-value / Vg / Ve.
   Drift beyond the gate (see below) emits a `fixture-drift` finding.

```bash
# Orchestrator (does install + fetch + run + diff for all 3 tools):
python3 validation/reproducibility/rerun_goldens.py

# Just one tool:
python3 validation/reproducibility/rerun_goldens.py --tool gemma

# Just the diff (assumes outputs/ already populated):
python3 validation/reproducibility/compare_fixtures.py --tool gemma

# Force a fresh GWASpoly re-run (default falls back to committed):
python3 validation/reproducibility/rerun_goldens.py --tool gwaspoly --force-gwaspoly-rerun

# Pytest entry (auto-skips by default; opt-in via -m reproducibility):
pytest -m reproducibility tests/test_reproducibility.py -v
```

## Tolerance bar

The fresh tool should reproduce the committed fixture **near-bit-exactly**
because both run on the same input under the same upstream version. We
set the gates *tight* and surface observed values regardless:

| Metric | Threshold | Reason |
|---|---|---|
| GRM kinship element \|Δ\|       | ≤ 1e-10 | GEMMA writes 7-digit floats; round-trip should be exact in float64 |
| β (effect)        \|Δ\|       | ≤ 1e-6  | deterministic float ops with same version |
| SE                \|Δ\|       | ≤ 1e-6  | same |
| log-likelihood    \|Δ\|       | ≤ 1e-4  | accumulated FP rounding through REML |
| Vg, Ve            relative \|Δ\| | ≤ 1e-6  | per-element; min |denom| 1e-30 |
| p-value (raw)     \|Δ\|       | ≤ 1e-8  | tightest gate |
| -log10(p)         \|Δ\|       | ≤ 1e-4  | log-domain relaxed gate for very small p |
| Allele freq (af)  \|Δ\|       | ≤ 1e-6  | input data invariant |
| FarmCPU / BLINK   top-10 overlap ≥ 0.7 | informational; pseudo-QTN seeding is internal-RNG-dependent |

Violation of any gate is a `fixture-drift` finding — informational, not a
hard failure. The orchestrator and `compare_fixtures.py` exit 0 either way;
the `outputs/<tool>_drift.json` and the drift table below are the
authoritative record.

## Per-tool reproduction recipe

### GEMMA 0.98.5 — `gemma_demo/output/`

| Field | Value |
|---|---|
| Pinned version | 0.98.5 (2021-08-25) |
| Source binary | `${ROOT}/gemma_demo/gemma-0.98.5` (sha256 `ad3f3f43…`) |
| Input fixture | MDP maize: 276 samples × 3093 SNPs (BIMBAM) |
| Outputs diffed | `mdp_kinship.cXX.txt` + `mdp_lmm_all.{assoc,log}.txt` + `mdp_mvlmm_wald.{assoc,log}.txt` |
| Tool install | trivial: symlink + checksum (binary already in checkout) |
| Run time | ~10 seconds |

### GAPIT3 — `benchmark/gapit_results/`

| Field | Value |
|---|---|
| Pinned version | 3.4 (per spec) |
| Install path | non-interactive R: `BiocManager::install("multtest")` + `remotes::install_github("jiabowang/GAPIT3")` |
| Input fixture | MDP maize: 281 samples × 3093 SNPs |
| Outputs diffed | `GLM_GWAS.csv`, `MLM_GWAS.csv`, `FarmCPU_GWAS.csv`, `BLINK_GWAS.csv` |
| Tool install | medium difficulty (Bioconductor + many CRAN deps; some hosts cannot reach the configured Bioconductor mirror) |
| Run time | ~3-5 minutes once installed |

### GWASpoly — `benchmark/gwaspoly_results/`

| Field | Value |
|---|---|
| Pinned version | ≥ 2.12 (per spec); current install: 2.14 |
| Install path | non-interactive R: `remotes::install_github("jendelman/GWASpoly")` |
| Input fixture | tetraploid potato: 222 samples × 9888 markers (built into the GWASpoly package) |
| Outputs diffed | 5 gene-action models + kinship: `gwaspoly_{additive,1_dom_alt,1_dom_ref,2_dom_alt,2_dom_ref,diplo_additive,diplo_general,general}.csv` + `gwaspoly_kinship.csv` |
| Tool install | usually clean from CRAN |
| Run time | **~25-40 min** wall (per-marker LMM serial scan across 6 gene-action models on 9888 polymorphic markers × 957 phenotyped samples; the spec's "10-15 min" estimate is pessimistic by ~2× on a 24-core RHEL host). Default opts to fall back to the committed reference; set `GWASPOLY_FORCE_RERUN=1` for a real re-run. Set `GWASPOLY_NCORES=N` to use `n.core=N` in the per-marker `parallel::mclapply()` (default: 1). |

## Drift findings table

Per the F3 protocol (spec §9), drift findings are *informational*. They
are surfaced here AND appended as a `fixture-drift` row to
`docs/validation_findings.md`. Regenerating the committed fixture is a
**user decision**, not an agent decision.

| Date | Tool | Fixture | Observed |Δ| | Threshold | Classification | Recommended action |
|------|------|---------|---------------|-----------|----------------|--------------------|
| 2026-04-30 | GEMMA 0.98.5 | `gemma_demo/output/` (5 outputs: cXX, LMM single, mvLMM) | **0.000e+00 across all 19 checks** (β, SE, p_wald, p_lrt, p_score, vg, ve, ll_reml, ll_ml, Vg, Ve, GRM, af, logl_H1) | ≤ 1e-10 / 1e-6 / 1e-4 / 1e-8 | **AUTHORITATIVE** | none — fixture exactly reproduces |
| 2026-04-30 | GAPIT3 | `benchmark/gapit_results/` (4 GWAS CSVs) | **(infra-blocker)** | n/a | infra-blocker | retry GAPIT3 install on a host with reachable Bioconductor (mirror config issue blocked `snpStats` fetch on this host) |
| 2026-04-30 | GWASpoly _PENDING_ | `benchmark/gwaspoly_results/` (8 model CSVs + kinship) | _populated by full run_ | ≤ 1e-6 / 1e-4 / 1e-8 | _populated_ | _populated_ |

Updated by re-running:

```bash
python3 validation/reproducibility/rerun_goldens.py
# Then read validation/reproducibility/outputs/<tool>_drift.json
```

## Regenerate-this list

A fixture appears in this list if and only if a Pillar D drift finding
exceeds tolerance. The user (NOT the agent) decides regeneration. Until
the user acts, the committed fixture remains the canonical reference for
TorchGenomics regression checks.

| Fixture | Reason | Captured By |
|---------|--------|-------------|
| _(none — see drift table above; populated when first drift exceeds threshold)_ | | |

## Cumulative campaign findings (post-Pillar-D)

Pillar D adds **infra-blocker** rows for tools that can't be re-installed
on this host (e.g., GAPIT3 mirror failure). It does not add
`torchgenomics-divergence` findings — Pillars A / B / C are the source of those.

The complete cumulative tally (carried forward from
`docs/superpowers/SESSION_HANDOFF.md`) is at the top of
`docs/validation_findings.md`.

## Self-review

- Pre-flight invoked via per-tool harnesses + orchestrator-level check.
- Tolerances tight (1e-6 absolute / 1e-10 GRM / 1e-4 -log10p / 0.7 top-K).
- F3 / fixture-drift findings (when present) are recorded in
  `docs/validation_findings.md` with classification `fixture-drift`,
  separately from `torchgenomics-divergence` findings logged in earlier
  pillars.
- Pytest skips cleanly on hosts without the upstream tools installed.
- README documents protocol + tolerances + drift table + regenerate list.
