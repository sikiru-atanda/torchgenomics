# GEMMA reference harness (Pillar B / B7)

GEMMA is the canonical reference for diploid linear mixed-model GWAS — single-trait LMM (Wald + LRT + score) and multi-trait mvLMM (joint Wald). This harness re-houses the comparison that was wired BEFORE the Pillar A campaign (`gemma_demo/` + `tests/test_golden_gemma.py`) into the `validation/external/<tool>/` layout, and re-runs it end-to-end to confirm zero regression from the 9 Pillar-A V1-core / V1-platform fixes.

## Pinned reference

| Field | Value |
|---|---|
| Tool version | `GEMMA 0.98.5 (2021-08-25)` |
| Source | `gemma_demo/gemma-0.98.5` (committed Linux x86_64 static build) |
| SHA-256 (binary) | `ad3f3f43a2f8a1c00e71fae8f43676614170e06c99fc766a947acfe45605969b` |
| Platform | Linux x86_64 (static AMD64 build; runs on any glibc ≥ 2.6.32) |

GEMMA does not need to be installed by this harness — the binary is already
part of the canonical TorchGenomics checkout. `install.sh` symlinks it into
`bin/` and verifies the SHA-256.

## Reproduction recipe

```bash
# From repo root:
bash validation/external/gemma/install.sh        # idempotent; symlinks bin/gemma → gemma_demo/gemma-0.98.5
bash validation/external/gemma/fetch_data.sh     # copies BIMBAM fixtures + MDP map into data/
bash validation/external/gemma/run_gemma.sh      # 3 reference outputs into outputs/
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/external/gemma/compare.py

# Or via pytest (requires -m external; skipped by default):
pytest -m external tests/test_external_gemma.py -v
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts disk + RAM headroom before doing any work, per the Pillar B contract.

## Reference outputs (in `outputs/`)

| File | Source command | TorchGenomics counterpart |
|---|---|---|
| `mdp_kinship.cXX.txt` | `gemma -gk 1` (centered kinship) | `torchgenomics.linalg.kinship` |
| `mdp_lmm_all.assoc.txt` + `.log.txt` | `gemma -lmm 4 -k <K>` (Wald + LRT + score) | `torchgenomics.models.SingleTraitLMM` |
| `mdp_mvlmm_wald.assoc.txt` + `.log.txt` | `gemma -lmm 1 -n 1 2 -k <K>` (mvLMM joint Wald) | `torchgenomics.models.MultiTraitLMM` |

## §16 tolerance gates (DO NOT loosen)

These are the spec §16 numbers, also pinned in `tests/test_golden_gemma.py`. Observed values from this harness's first re-run on master at `validation/pillar-B-references` (2026-05-04):

| Comparison | Metric | §16 floor | Observed |
|---|---|---|---|
| GRM | Element-wise max \|Δ\| (round-trip) | ≤ 1e-10 | **0.000e+00** |
| GRM | Element-wise max relative err | ≤ 1e-6 | **0.000e+00** |
| LMM single | vg relative error | ≤ 1e-3 | 7.49e-07 |
| LMM single | ve relative error | ≤ 1e-3 | 3.07e-06 |
| LMM single | REML log-likelihood \|Δ\| | ≤ 1e-2 | 7.96e-04 |
| LMM single | β correlation (full set) | ≥ 0.9999 | 0.99993 |
| LMM single | β median \|Δ\| (AF [0.03, 0.97]) | ≤ 0.01 | 2.56e-03 |
| LMM single | SE median \|Δ\| (AF [0.03, 0.97]) | ≤ 0.02 | 2.77e-03 |
| LMM single | −log10(p) Wald correlation | ≥ 0.999 | 0.99943 |
| LMM single | −log10(p) LRT correlation | ≥ 0.998 | 0.99869 |
| LMM single | −log10(p) score correlation | ≥ 0.999 | 0.99980 |
| mvLMM | Vg max relative error | ≤ 1e-2 | 1.25e-06 |
| mvLMM | Ve max relative error | ≤ 1e-2 | 3.38e-06 |
| mvLMM | −log10(p) joint-Wald correlation | ≥ 0.998 | 0.99810 |

All 14 checks pass at §16's floors → **zero regression** from the 9 Pillar A fixes.

## Pillar A regression check

The 9 Pillar A fixes are:

1. `stats.calibrate.compare_pvalues` (was unimplemented stub)
2-4. `models.lmm_single` / `lmm_multi` / `lmm_multi_fit` re-exports (were stubs)
5. `optim.fisher_scoring.fisher_scoring_reml` (was stub)
6. `io.convert._write_zarr` (zarr v3 compat)
7-8. `postgwas._ld_scores` / `_clump` shape validation guards
9. `mr_egger` Bowden 2015 alignment

None of these touch the SingleTraitLMM / MultiTraitLMM canonical math paths exercised by GEMMA goldens. The harness confirms this empirically — every check passes at §16's floor, identically to the pre-Pillar-A behavior.

## Peak memory

| Stage | RAM pre-flight | Observed peak |
|---|---|---|
| install (symlink + checksum) | 6 GB | < 50 MB |
| fetch (copy + checksum) | 6 GB | < 50 MB |
| run (-gk + -lmm 4 + -lmm 1 -n 1 2) | 7 GB | ~150 MB |
| compare.py (TG single + multi LMM) | n/a | ~600 MB |

The fixture (276 samples × 3093 SNPs) is small; the 6 GB pre-flight is conservative headroom for safety.

## CI / test integration

```bash
# Skipped by default (the conftest auto-skips `external` markers):
pytest tests/test_external_gemma.py            # 3 skipped

# Opt-in (after install.sh + fetch_data.sh + run_gemma.sh):
pytest -m external tests/test_external_gemma.py -v
```

## Layout

```
validation/external/gemma/
├── install.sh             # symlinks bin/gemma → gemma_demo/gemma-0.98.5
├── fetch_data.sh          # copies BIMBAM fixtures + MDP map into data/
├── run_gemma.sh           # produces 3 reference outputs in outputs/
├── compare.py             # asserts §16 tolerances vs TorchGenomics
├── README.md              # this file
├── bin/                   # gemma binary symlink (ignored)
├── data/                  # MDP fixtures (ignored)
└── outputs/               # GEMMA reference outputs (ignored)
```
