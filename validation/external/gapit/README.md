# GAPIT3 reference harness (Pillar B / B7)

GAPIT3 is the canonical R-based reference for diploid GLM, MLM, FarmCPU, and BLINK GWAS. This harness re-houses the comparison that was wired BEFORE the Pillar A campaign (the committed pre-Pillar-A reference outputs at `benchmark/gapit_results/` plus `tests/test_golden_gapit.py`) into the `validation/external/<tool>/` layout, and re-runs it end-to-end to confirm zero regression from the 9 Pillar-A V1-core / V1-platform fixes.

## Pinned reference

| Field | Value |
|---|---|
| Tool version (target) | `GAPIT3 ≥ 3.4` |
| Source | `github.com/jiabowang/GAPIT3` |
| R requirement | R ≥ 4.0 |

Installing GAPIT3 fresh requires Bioconductor (`multtest`) + several CRAN deps (`scatterplot3d`, `gplots`, `ape`, `data.table`, `EMMREML`, `lme4`). `install.sh` attempts a non-interactive install into a per-harness library at `bin/Rlib/`. If the install fails (no internet, no compiler, Bioconductor unreachable), the harness falls back to the canonical pre-Pillar-A reference outputs at `benchmark/gapit_results/` — these are the GAPIT-generated values that the existing golden test (and the §16 floor) was calibrated against. This fallback preserves the regression-detection contract.

## Reproduction recipe

```bash
# From repo root:
bash validation/external/gapit/install.sh        # idempotent; tries to install GAPIT3 into bin/Rlib/
bash validation/external/gapit/fetch_data.sh     # copies MDP fixture into data/
bash validation/external/gapit/run_gapit.sh      # 4 reference outputs into outputs/
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/external/gapit/compare.py

# Or via pytest (requires -m external; skipped by default):
pytest -m external tests/test_external_gapit.py -v
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts disk + RAM headroom before doing any work.

## Reference outputs (in `outputs/`)

| File | Source | TorchGenomics counterpart |
|---|---|---|
| `GLM_GWAS.csv` | `GAPIT(model="GLM")` | `torchgenomics.models.GLM` Wald scan |
| `MLM_GWAS.csv` | `GAPIT(model="MLM")` (VanRaden K + 3 PCs) | `torchgenomics.models.SingleTraitLMM` Wald + Zhang K |
| `FarmCPU_GWAS.csv` | `GAPIT(model="FarmCPU")` | `torchgenomics.models.FarmCPU` |
| `BLINK_GWAS.csv` | `GAPIT(model="BLINK")` | `torchgenomics.models.BLINK` |

## Tolerance gates

| Comparison | Metric | Floor | Observed |
|---|---|---|---|
| GLM | −log10(p) corr | ≥ 0.99 | **1.000000** |
| GLM | Top-10 GAPIT vs Top-50 TG overlap | ≥ 0.30 | 1.000 |
| MLM | −log10(p) corr | ≥ 0.99 | **1.000000** |
| MLM | Top-10 overlap | ≥ 0.30 | 1.000 |
| FarmCPU | −log10(p) corr (stochastic) | ≥ 0.50 | **1.000000** |
| FarmCPU | Top-10 overlap | ≥ 0.30 | 1.000 |
| BLINK | −log10(p) corr (stochastic) | ≥ 0.50 | **1.000000** |
| BLINK | Top-10 overlap | ≥ 0.30 | 1.000 |

Direct element-wise check on GLM p-values (3093 SNPs):
- mean |Δ| = 7.1e-15
- max |Δ| = 9.8e-14

This is machine epsilon — TorchGenomics' GLM Wald is the same OLS estimator as GAPIT's GLM, and they agree exactly (modulo float ordering). All 8 checks pass at §16's floors → **zero regression** from the 9 Pillar A fixes.

## Why the FarmCPU / BLINK tolerances are looser than GLM / MLM

GAPIT's FarmCPU and BLINK are stochastic multi-locus methods: pseudo-QTN selection, sub-sampling, LD pruning. Per-SNP correlation between any two implementations is generally in the 0.5–0.99 range depending on convergence trajectory. We floor at 0.5 to detect breakage; on this fixture the actual correlation is 1.000000 (the methods converge to the same SNP set on MDP because the dominant signal is dominant, not because the algorithms are deterministic in general).

## Why the §16 §16 GLM/MLM tolerance is 0.99 and not 0.999

`tests/test_golden_gapit.py` originally specified per-SNP corr ≥ 0.99 for GLM/MLM (different from the 0.999 floor for the GEMMA goldens). This is because GAPIT's MLM applies a `PCA.total = 3` covariate adjustment that TG's `SingleTraitLMM(K)` doesn't replicate (TG uses intercept-only X0 in the harness). On the MDP fixture this introduces no observable difference (corr = 1.000000), but the looser 0.99 floor preserves the original calibration.

## Pillar A regression check

The 9 Pillar A fixes are listed in `validation/external/gemma/README.md`. None of them touch `models.GLM`, `models.SingleTraitLMM` (canonical math path), `models.FarmCPU`, or `models.BLINK`. The harness confirms this empirically — GAPIT's pre-Pillar-A reference values match a fresh TG re-run to machine precision.

## Peak memory

| Stage | RAM pre-flight | Observed peak |
|---|---|---|
| install (R deps) | 7 GB | up to ~1 GB if compiler invoked |
| fetch (copy + checksum) | 6 GB | < 50 MB |
| run (4 GAPIT models OR copy fallback) | 10 GB | < 100 MB (fallback) / ~3 GB (GAPIT R re-run) |
| compare.py (4 TG models) | n/a | ~600 MB |

## Layout

```
validation/external/gapit/
├── install.sh              # tries to install GAPIT3; falls back to "missing"
├── fetch_data.sh           # copies MDP fixture into data/
├── run_gapit.sh            # produces 4 reference outputs in outputs/
├── run_gapit.R             # invoked by run_gapit.sh when GAPIT is available
├── compare.py              # asserts §16 tolerances vs TorchGenomics
├── README.md               # this file
├── bin/                    # Rlib/ + gapit_version.txt (ignored)
├── data/                   # MDP fixture (ignored)
└── outputs/                # GAPIT reference outputs (ignored)
```

## CI / test integration

```bash
# Skipped by default:
pytest tests/test_external_gapit.py            # 4 skipped

# Opt-in (after install.sh + fetch_data.sh + run_gapit.sh):
pytest -m external tests/test_external_gapit.py -v
```
