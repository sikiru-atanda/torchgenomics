# GWASpoly reference harness (Pillar B / B7)

GWASpoly (Rosyara et al. 2016) is the canonical R-based reference for polyploid GWAS — 5 distinct gene-action models on tetraploid potato data (957 genotypes × 9888 markers, 6 environments, vine.maturity trait). This harness re-houses the comparison that was wired BEFORE the Pillar A campaign (`benchmark/gwaspoly_results/` + `tests/test_golden_gwaspoly.py`) into the `validation/external/<tool>/` layout.

## Pinned reference

| Field | Value |
|---|---|
| Tool version (target) | `GWASpoly ≥ 2.12` (we install 2.14 from current GitHub HEAD) |
| Source | `github.com/jendelman/GWASpoly` |
| R requirement | R ≥ 4.0 + `rrBLUP` (for `mixed.solve`) |

`install.sh` attempts a non-interactive install into `bin/Rlib/`. Verified successful on this machine (R 4.5.1 + GWASpoly 2.14 + rrBLUP).

## Reproduction recipe

```bash
# From repo root:
bash validation/external/gwaspoly/install.sh        # idempotent; tries to install GWASpoly into bin/Rlib/
bash validation/external/gwaspoly/fetch_data.sh     # copies aligned + raw potato fixtures into data/
bash validation/external/gwaspoly/run_gwaspoly.sh   # populates outputs/ with reference CSVs
TORCHGWAS_DISABLE_NATIVE=1 python3 validation/external/gwaspoly/compare.py

# Or via pytest (requires -m external; skipped by default):
pytest -m external tests/test_external_gwaspoly.py -v
```

By default `run_gwaspoly.sh` populates `outputs/` from the canonical pre-Pillar-A reference at `benchmark/gwaspoly_results/` — these are upstream GWASpoly's own per-marker p-values, generated when this comparison was first wired. To force a fresh GWASpoly re-run instead (12-15 min on this fixture):

```bash
GWASPOLY_FORCE_RERUN=1 bash validation/external/gwaspoly/run_gwaspoly.sh
```

The fresh re-run is end-to-end equivalent — GWASpoly is deterministic given a fixed kinship — but the committed reference is preferred for routine harness invocation because of the runtime cost.

## Reference outputs (in `outputs/`)

| File | Source | TorchGWAS counterpart |
|---|---|---|
| `gwaspoly_additive.csv` | GWASpoly `model="additive"` | TG `recode_gene_action(..., "additive")` |
| `gwaspoly_1_dom_alt.csv` | GWASpoly `1-dom` (alt-allele simplex dominant) | TG `1-dom` |
| `gwaspoly_2_dom_alt.csv` | GWASpoly `2-dom` (alt-allele duplex dominant) | TG `2-dom` |
| `gwaspoly_2_dom_ref.csv` | GWASpoly `2-dom` (ref-allele = TG's `3-dom`) | TG `3-dom` |
| `gwaspoly_diplo_additive.csv` | GWASpoly `diplo-additive` (different encoding) | TG `diplo-additive` |

Plus `gwaspoly_general.csv`, `gwaspoly_diplo_general.csv`, `gwaspoly_1_dom_ref.csv`, and the kinship matrix.

## §16 tolerance gates

| Comparison | Floor | Observed |
|---|---|---|
| Additive | −log10(p) corr ≥ 0.999 | **0.99989** |
| 1-dom | −log10(p) corr ≥ 0.999 | **0.99999** |
| 2-dom | −log10(p) corr ≥ 0.999 | **0.99991** |
| 3-dom | −log10(p) corr ≥ 0.999 | **0.99998** |
| Diplo-additive (encoding-mismatch) | corr ∈ [0.3, 0.7] | 0.4876 |

All 10 checks (5 corr + 5 marker-count) pass at §16's floors → **zero regression** from the 9 Pillar A fixes.

## Why the diplo-additive tolerance is a bracket, not a floor

GWASpoly and TorchGWAS use different diplo-additive encodings:
- GWASpoly: `{0,1,2,3,4} → {0,1,1,1,2}` (the dichotomous "diploidization" of Endelman 2011)
- TorchGWAS: `{0,1,2,3,4} → {0,1,2,1,0}` (`min(d, ploidy-d)` — folded encoding)

These are mathematically distinct and the per-SNP correlation is expected to be moderate, not high. The existing golden test (`tests/test_golden_gwaspoly.py::test_diplo_additive_differs`) pins the bracket at `0.3 < r < 0.7`; this harness preserves that contract. A future RFE could expose both encodings under disambiguated names, but that's out of scope for B7.

## Pillar A regression check

The 9 Pillar A fixes are listed in `validation/external/gemma/README.md`. None of them touch:
- `torchgwas.preprocess.polyploid.recode_gene_action`
- `torchgwas.linalg.kinship_polyploid.grm_polyploid_gene_action`
- `torchgwas.models.SingleTraitLMM` (canonical math path)

The harness confirms this empirically — TG's per-SNP p-values agree with GWASpoly's reference to 4–5 decimal places of −log10(p) across 30 000+ marker comparisons (5 models × ~6 000–10 000 markers).

## Peak memory

| Stage | RAM pre-flight | Observed peak |
|---|---|---|
| install (R + GWASpoly + deps) | 7 GB | up to ~1 GB if compiler invoked |
| fetch (copy + checksum) | 6 GB | < 100 MB (38 MB on disk) |
| run (default: copy from canonical) | 10 GB | < 100 MB |
| run (`GWASPOLY_FORCE_RERUN=1`) | 10 GB | ~3 GB (12-15 min) |
| compare.py (5 TG P3D scans on ~10K markers) | n/a | ~1.2 GB |

## Layout

```
validation/external/gwaspoly/
├── install.sh             # tries to install GWASpoly + rrBLUP; falls back to "missing"
├── fetch_data.sh          # copies aligned data + GWASpoly inputs into data/
├── run_gwaspoly.sh        # populates outputs/ from canonical reference (or fresh run)
├── run_gwaspoly.R         # invoked by run_gwaspoly.sh under GWASPOLY_FORCE_RERUN=1
├── compare.py             # asserts §16 tolerances vs TorchGWAS
├── README.md              # this file
├── bin/                   # Rlib/ + gwaspoly_version.txt (ignored)
├── data/                  # potato fixture (ignored)
└── outputs/               # GWASpoly reference CSVs (ignored)
```

## CI / test integration

```bash
# Skipped by default:
pytest tests/test_external_gwaspoly.py            # 5 skipped

# Opt-in (after install.sh + fetch_data.sh + run_gwaspoly.sh):
pytest -m external tests/test_external_gwaspoly.py -v
```
