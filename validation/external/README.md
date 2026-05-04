# Pillar B — External Reference Tool Harnesses

This directory holds the harness for each external reference tool against which TorchGWAS is validated. Each tool gets its own subdirectory:

```
validation/external/
├── _lib/
│   └── preflight.sh        # Shared memory + disk pre-flight (spec §5.3)
├── gemma/                   # GEMMA 0.98.5 — single-trait LMM, mvLMM, GRM
├── gapit/                   # GAPIT3 3.4 — FarmCPU, BLINK
├── gwaspoly/                # GWASpoly 2.12 — polyploid models
├── plink2/                  # PLINK 2.0 — GLM, kinship, r2, blocks
├── ldsc/                    # LDSC 1.0.1 — h², rg, S-LDSC
├── regenie/                 # regenie 3.x — LMM scan + Step 1/2 binary
├── saige/                   # SAIGE 1.x — binary / ordinal GLMM, survival
├── bolt_lmm/                # BOLT-LMM 2.4+ — LMM with LOCO
├── twosamplemr/             # TwoSampleMR (R) — IVW / Egger / weighted-median / MR-PRESSO
├── soynam/                  # SoyNAM (R) — within-family LMM (Phase 23)
└── soymd/                   # SoyMD multi-omics platform — Phase 49 mediation/eQTL/h²
```

## Per-tool layout

Every `validation/external/<tool>/` directory follows this skeleton:

| File | Purpose |
|---|---|
| `install.sh` | Idempotent install from upstream (apt / conda / pip / source). Pinned version. Records install log + binary checksum. |
| `fetch_data.sh` | Downloads + checksum-verifies the public dataset(s) into `data/`. |
| `run_<tool>.sh` | Runs the tool on the public dataset to produce a reference output file. |
| `compare.py` | Parses tool output, parses TorchGWAS output, computes element-wise diffs, asserts tolerances. |
| `README.md` | Exact reproduction recipe + peak-memory estimate (revised after first successful run). |
| `data/` | `.gitignored`. Downloaded by `fetch_data.sh` from a fixed URL with checksum verification. |

## Memory pre-flight contract (the user's hard rule)

Before each `fetch_data.sh` and each `run_<tool>.sh`:
- **Disk:** `df --output=avail /home` ≥ (download_size × 2 + working_set × 2).
- **RAM:** `free -g` available ≥ (estimated peak in-RAM working set × 1.5 + 4 GB headroom).
- If either fails, abort with explicit message — do not download / run partially.

Implementation: `_lib/preflight.sh` exports `preflight_check`, `preflight_check_with_data_size`, and `verify_checksum`. Source it at the top of every harness shell script.

## Running

```bash
# Run a single tool's full harness (install + fetch + run + compare):
bash validation/external/plink2/install.sh
bash validation/external/plink2/fetch_data.sh
bash validation/external/plink2/run_plink2.sh
python3 validation/external/plink2/compare.py

# Or via pytest with the `external` marker (registered in pyproject.toml):
pytest -m external

# Or just one tool:
pytest tests/test_external_plink2.py -v -m external
```

## Status

| Tool | Harness | Wired in pytest | Findings |
|---|---|---|---|
| GEMMA | **done (B7)** — install/fetch/run/compare scripts; symlinks gemma_demo/gemma-0.98.5; produces 3 reference outputs (centered K, single-trait LMM Wald+LRT+score, mvLMM joint Wald) | yes (`tests/test_external_gemma.py`, marker=`external`) | All 14 §16 tolerance gates pass. β corr = 0.99993, vg/ve rel err < 4e-6, REML logL \|Δ\| < 1e-3, mvLMM −log10(p) corr = 0.99810 (≥ 0.998). **Zero regression** from Pillar A's 9 fixes confirmed end-to-end. |
| GAPIT | **done (B7)** — install/fetch/run/compare; tries non-interactive GAPIT3 install, falls back to canonical pre-Pillar-A reference outputs at `benchmark/gapit_results/` | yes (`tests/test_external_gapit.py`, marker=`external`) | GLM/MLM/FarmCPU/BLINK all pass. GLM −log10(p) corr = 1.000000 (element-wise \|Δ\| < 1e-13 to GAPIT). **Zero regression** from Pillar A's 9 fixes confirmed. |
| GWASpoly | **done (B7)** — install/fetch/run/compare; installs GWASpoly 2.14 + rrBLUP into bin/Rlib/; uses canonical pre-Pillar-A reference by default (12-15 min re-run available via `GWASPOLY_FORCE_RERUN=1`) | yes (`tests/test_external_gwaspoly.py`, marker=`external`) | All 5 gene-action models pass §16's r ≥ 0.999 floor: additive 0.99989, 1-dom 0.99999, 2-dom 0.99991, 3-dom 0.99998. Diplo-additive in expected encoding-mismatch bracket [0.3, 0.7] = 0.488. **Zero regression**. |
| PLINK 2.0 | **done (B1)** — install/fetch/run/compare scripts + 3-comparison harness on MDP | yes (`tests/test_external_plink2.py`, marker=`external`) | β corr = 1.000, GRM off-diag corr = 0.99995, r² corr = 1.000; all pass observed-then-floored tolerances. Documented PLINK-vs-TG SE parameterization mismatch (n−c−1 vs n−c) and the cov-vs-VanRaden normalizer scalar. No F3 fixes required. |
| LDSC | **done (B2)** — install/fetch/run/compare scripts + h²(×2) + rg comparison on simulated chr22 sumstats | yes (`tests/test_external_ldsc.py`, marker=`external`) | h² agreement 2–4e-3, rg agreement 3.5e-3 (within spec §16). Documented IRWLS-vs-single-pass WLS parameterization mismatch on intercept (\|Δ\| ~0.1–0.3); h²/rg unaffected. F3 deferred to a Phase 37 follow-up — IRWLS implementation in `ldsc_h2` would close the intercept gap. |
| TwoSampleMR | done (B3) | yes | — (see B3 commit) |
| regenie | done (B4) | yes | — (see B4 commit) |
| SAIGE | pending | pending | — |
| BOLT-LMM | pending | pending | — |
| SoyNAM | pending | pending | — |
| SoyMD | pending | pending | — |
