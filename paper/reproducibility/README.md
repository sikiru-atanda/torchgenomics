# TorchGWAS — Genome Biology Methods paper reproducibility

One-command pipeline that regenerates every number in every figure and
table of the Genome Biology Methods manuscript on TorchGWAS.

## Running

From the repo root or from this directory:

```bash
bash paper/reproducibility/reproduce_paper.sh
```

Outputs:
- `paper/reproducibility/output/F{1..7}.pdf` — the seven main figures.
- `paper/reproducibility/manifest.json` — every numeric result + the commit SHA the run was based on.

Environment overrides:
- `REPRODUCE_SKIP_INSTALL=1` — skip stage 01 (use installed tools as-is).
- `REPRODUCE_SKIP_FETCH=1` — skip stage 02 (use staged fixtures as-is).
- `REPRODUCE_QUICK=1` — pass `--quick` to bench scripts when they support it.
- `TORCHGWAS_DISABLE_NATIVE=1` — force pure-torch reference paths (set by default in stages 04 / 07 so figure renders are reproducible across hosts with and without the C++ extensions installed).

## Pipeline stages

| Stage | Script | Purpose |
|---|---|---|
| 0 | `preflight.sh` | Assert 50 GB disk + 16 GB RAM headroom (sourced from `validation/external/_lib/preflight.sh`). |
| 01 | `stages/01_install_references.sh` | Idempotently install the 15 reference-tool harnesses under `validation/external/`. |
| 02 | `stages/02_stage_fixtures.sh` | Stage every reference-tool fixture + the 3 multi-omics fixtures + the 6 specialty fixtures. |
| 03 | `stages/03_run_references.sh` | Run each reference tool against its staged fixture. |
| 04 | `stages/04_run_torchgwas.sh` | Run each harness's `compare.py` (TorchGWAS head-to-head vs reference). |
| 05 | `stages/05_run_streaming_bench.sh` | Streaming memory slope sweep (F6 panel B). |
| 06 | `stages/06_run_native_bench.sh` | 25 native-kernel speedup distribution (F6 panel A). |
| 07 | `stages/07_run_multiomics.sh` | Multi-omics integration (F4 panels A/B/C). |
| 08 | `stages/08_render_figures.py` | Render F1-F7 PDFs + populate `manifest.json`. |

Stages 01 and 02 are idempotent (each harness's `install.sh` writes a
`.install_marker` and skips on subsequent runs; each `fetch_data.sh`
is similarly idempotent). On a warm checkout, only stages 03-08 need
re-running.

## Per-figure data sources

| Figure | Source |
|---|---|
| F1 capability map | hand-curated JSON from CLI registry + module surface |
| F2 15-tool equivalence grid | `validation/external/<tool>/results/agreement.json` for each of 15 tools (+ 3 fixture columns) |
| F3 haplotype layer | `validation/external/{plink2, hapref, soymd}/` + LD-block-method outputs |
| F4 multi-omics integration | `validation/multiomics/{sim, gtex_ukb, plant}/` + `validation/external/{metaxcan, smr, coloc, hyprcoloc}/` |
| F5 polyploid pipeline | `validation/external/gwaspoly/` + Phase 55/56 fixture outputs |
| F6 GPU + streaming | `bench/native_speedups.py` outputs + `bench/streaming_p_sweep.py` outputs (Tier 4 D2.8) + GPU parity test artifacts |
| F7 specialty models | `validation/specialty/{survival, rr, family, threshold, knockoff, ocf}/` + GU + LRO simulators (Tier 4 D2.9) |

## Status

The Tier 4 D1 commit (this directory) ships the orchestration scaffold.
The per-figure renderer modules under `render_figures/` are authored in
Tier 4 D2 (one renderer per figure, parallelizable). Until D2 lands,
`08_render_figures.py` will report "No figure renderers registered" and
exit non-zero — this is expected during the build-up phase.

## Provenance

- Authoritative spec: `docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md`
- Authoritative plan: `docs/superpowers/plans/2026-05-15-paper-tier4-reproducibility-and-draft.md` (D1)
- Pre-flight contract: `validation/external/_lib/preflight.sh` (mandatory across every install / fetch / run script).
- Findings ledger: `docs/validation_findings.md` (every divergence ever found and how it was resolved or deferred).
