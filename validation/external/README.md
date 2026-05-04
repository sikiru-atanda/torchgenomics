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
| GEMMA | already wired (pre-Pillar-A); rehouse pending | yes (golden marker) | — |
| GAPIT | already wired; rehouse pending | yes (golden marker) | — |
| GWASpoly | already wired; rehouse pending | yes (golden marker) | — |
| PLINK 2.0 | pending | pending | — |
| LDSC | pending | pending | — |
| TwoSampleMR | pending | pending | — |
| regenie | pending | pending | — |
| SAIGE | pending | pending | — |
| BOLT-LMM | pending | pending | — |
| SoyNAM | pending | pending | — |
| SoyMD | pending | pending | — |
