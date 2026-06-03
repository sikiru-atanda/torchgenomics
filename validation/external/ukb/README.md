# UKB-scale validation harness (Task NA3)

Empirical biobank-scale validation of TorchGenomics' streaming + numerical
claims, run end-to-end on a single UKB chromosome (chr22) by the user.

This harness is the empirical answer to the open task surfaced in
`docs/superpowers/SESSION_HANDOFF.md` (lines 91-104) on the validation
branch:

> all of the campaign's biobank-scale memory math (40 TB → 4 GB) is
> *projected* from chunk-size arithmetic. No actual run on UKB-scale
> (n ≈ 500K samples, m ≈ 10M variants) data has been done.

A single chr22 LMM + LDSC h² run takes a few CPU-hours and ratifies the
campaign's memory + correctness story.

## Scope (MVP)

| Aspect | Decision |
|---|---|
| Chromosome | chr22 only (smallest autosome; ~1.6M variants in UKB v3 imputed) |
| Trait type | Quantitative (Gaussian) — Phase 57 modernization MVP scope |
| Reference tool 1 | REGENIE v4.x — `install.sh` tries v4.2 first, falls back to v4.1 (both UNVERIFIED at scaffold time; Anthropic knowledge cutoff Jan 2026). The pinned version actually installed is recorded in `.env_marker` as `REGENIE_VERSION_USED` and reported by `compare.py`. |
| Reference tool 2 | LDSC 1.0.1 (`--h2` on REGENIE-munged sumstats) |
| GPU | Not required; CPU-only baseline first |
| Re-runs | Reference outputs cached in `reference_outputs/` |
| Fixture data | User-supplied UKB chr22 + phenotype (DUA-controlled) |
| Comparison metrics | β Pearson r > 0.999; \|Δ h²\| < 5e-3; peak RAM ≤ 10% of `n × m × 8 B` |

## What the user must do BEFORE running

1. **Have UKB data access.** UKB is application-controlled; this harness
   does NOT download UKB.
2. **Stage UKB chr22 genotype** as either PLINK 1 BED/BIM/FAM or PLINK 2
   PGEN/PVAR/PSAM, on a path the host can read.
3. **Stage a phenotype TSV** with columns `FID  IID  <trait>  [covariates ...]`.
   The trait must be quantitative for the MVP. Default expected covariate
   columns: `AGE,SEX,PC1,PC2,...,PC10` (override via `UKB_COVAR_COLS`).
   **Important:** the SAME TSV is passed to REGENIE as both `--phenoFile`
   and `--covarFile` (run_reference.sh lines 94-97 / 122-125). It therefore
   MUST contain the trait column AND every column listed in
   `UKB_COVAR_COLS` alongside `FID`/`IID`, all in one file. A separate
   covariate file is not currently wired; if you need one, edit
   `STAGED_PHENO` resolution in `fetch_data.sh` (or hand-edit
   `run_reference.sh` to point `--covarFile` at a different path).
4. **Export env vars** before running install.sh:
   ```bash
   export UKB_CHR22_PATH=/path/to/ukb_imp_chr22_v3      # no .bed/.pgen suffix
   export UKB_PHENO_PATH=/path/to/ukb_pheno.tsv
   export UKB_TRAIT_COL=BMI                              # default: PHENO
   export UKB_COVAR_COLS=AGE,SEX,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10
   export UKB_THREADS=4                                  # default: 4
   # Optional REGENIE pin override (default tries v4.2 then v4.1):
   export REGENIE_VERSION=v4.2
   ```
5. **Confirm host headroom.** ≥ 100 GB free disk on `/home`, ≥ 32 GB
   available RAM. The pre-flight check in `install.sh` aborts hard if
   either is short.

## Run order

```bash
# All scripts source ../_lib/preflight.sh and abort on insufficient
# disk/RAM (per the user's hard rule, feedback_preflight memory).

bash validation/external/ukb/install.sh         # ~5-15 min (REGENIE + LDSC env)
bash validation/external/ukb/fetch_data.sh      # ~5-30 min (LD-score archive download)
bash validation/external/ukb/run_torchgenomics.sh   # 30-90 min (LMM scan + h²)
bash validation/external/ukb/run_reference.sh   # 60-180 min (REGENIE Step 1 + Step 2 + LDSC)

python3 validation/external/ukb/compare.py \
    --n-samples <YOUR_POST_QC_N>     \
    --m-variants <YOUR_POST_QC_M_CHR22> \
    --trait-col "${UKB_TRAIT_COL:-PHENO}" \
    --json validation/external/ukb/comparison.json \
    --findings-row /tmp/ukb_findings_row.md
```

The compare step prints a markdown row formatted for direct append to
`docs/validation_findings.md`. The user (or future-Claude with user
approval) pastes that row into the ledger and commits.

## Updating UKB inputs between runs

The harness records `UKB_CHR22_PATH` and `UKB_PHENO_PATH` in `.env_marker`
during `install.sh` (which overwrites the file) and stages them as
symlinks under `data/` during `fetch_data.sh`. If you change either env
var to point at a new file, **you must re-run `install.sh` to refresh
`.env_marker`**, otherwise the run scripts will continue to source the
old paths from the marker. `fetch_data.sh` updates the symlinks
correctly on its own (via `ln -sfn`), but it appends — rather than
overwrites — staged-path entries to `.env_marker`, and `install.sh` is
the canonical source for the user-supplied paths.

A second wrinkle: `run_torchgenomics.sh` computes the sample size N for the
LDSC h² driver as `N = $(wc -l < ${STAGED_PHENO}) - 1`. This assumes the
phenotype TSV has **exactly one** header line and **no comment lines or
trailing blank lines**. If your TSV has comments (`#...`) or extra
blanks, edit the `N_SAMPLES=` line in `run_torchgenomics.sh` (or strip
them before staging).

## Expected runtimes (per Task NA3 spec — "few CPU-hours per run")

| Step | n=50K (typical subset) | n=500K (full UKB) |
|---|---|---|
| `install.sh` | 5-15 min (REGENIE download + conda solve) | same |
| `fetch_data.sh` | 5-30 min (LD-score archive ~600 MB) | same |
| `run_torchgenomics.sh` | 30-60 min | 1-3 hours |
| `run_reference.sh` (REGENIE Step 1 + Step 2) | 30-60 min | 2-4 hours |
| `run_reference.sh` (LDSC --h2) | < 5 min | < 5 min |
| `compare.py` | < 1 min | < 1 min |

Total: a few CPU-hours at typical UKB-subset scale; up to one CPU-day
at full n=500K scale.

## Hardware requirements

| Resource | Required | Why |
|---|---|---|
| Disk | ≥ 100 GB free on `/home` | UKB chr22 staging + LD-score archive + REGENIE outputs + working space |
| RAM | ≥ 32 GB available | REGENIE Step 1 ridge predictor on UKB scale (Mbatchou 2021 reports ~24-28 GB peak) |
| CPU | 4+ cores recommended | REGENIE `--threads ${UKB_THREADS:-4}` |
| GPU | Not required | CPU-baseline first; GPU validation deferred to NA2 (GPU CI infra) |

The pre-flight checks abort with explicit messages if any of disk / RAM
falls short. They never partial-execute.

### Known scale ceiling on the K matrix

TorchGenomics' SingleTraitLMM holds the kinship matrix `K` (n × n,
float64) in RAM throughout the scan. At full UKB scale (n=500K),
`K` ≈ 1 TB — out of scope for any 32 GB host. The MVP harness
expects the user to subsample to n ≤ 50K (≈ 20 GB K) or supply a
pre-computed K via the `--kinship` flag (out of scope for the
scaffold; would need a separate UKB-K-builder script).

This is documented as a known limitation; the streaming claim
TorchGenomics makes is about per-chunk peak (n × chunk_size × 8 B), which
this harness does verify. The full n × n K is a model-architectural
constant, not a streaming-axis cost.

## F3 severity application — what halts the run

This harness applies the `feedback_f3` policy:

| Failure | F3 class | What halts |
|---|---|---|
| Pre-flight: disk or RAM insufficient | infra-blocker | `install.sh` / `fetch_data.sh` / `run_*.sh` exit 1 immediately, no partial execution |
| REGENIE Step 1 OOM at user-specified n | infra-blocker | `run_reference.sh` exits 1; user subsamples and re-runs |
| TG `lmm-scan` raises (e.g., K OOM) | post-V1 / open issue (architectural; documented above) | `run_torchgenomics.sh` exits 1; ledger row records the limit |
| β Pearson r < 0.999 | V1-core / fix-now if regression vs prior; post-V1 / documented otherwise | `compare.py` exits 1; investigate dosage convention, covariate alignment, GRM divergence |
| \|Δ h²\| > 5e-3 | post-V1 / documented (LDSC IRWLS already validated at fixture scale) | `compare.py` exits 1; investigate LD-score panel mismatch or chi² inflation |
| Peak RAM > 10% of full materialization | V1-core / regression (streaming claim is the point of the campaign) | `compare.py` exits 1; investigate which path materialized; halts NA3 |
| 3+ unrelated divergences | C4 emergency stop (per `feedback_f3`) | Stop the harness; surface to user; no auto-resolve |

## Where the result lands

1. **`docs/validation_findings.md`** — `compare.py` emits a single
   markdown row in the canonical schema. The user appends it.
2. **Commit message** — short summary of:
   - β Pearson r observed
   - \|Δ h²\| observed
   - TG peak RSS (GB) vs projected full-materialization (TB)
   - REGENIE peak RSS (GB) for context
3. **`comparison.json`** — full per-check JSON for downstream tooling.

## Pillar B mirror — what we adapt, what we keep

We mirror exactly the layout used in the eleven Pillar B harnesses on
the `validation/pillar-A-coverage` branch
(`validation/external/{regenie,ldsc,gemma,...}/`):

| File | Mirrors | Adaptations |
|---|---|---|
| `install.sh` | `validation/external/regenie/install.sh` + `validation/external/ldsc/install.sh` | Combined; REGENIE pin tries v4.2 then v4.1 instead of v3.3 |
| `fetch_data.sh` | `validation/external/regenie/fetch_data.sh` + `validation/external/ldsc/fetch_data.sh` | UKB symlinking instead of MDP staging; same LD-score archive |
| `run_torchgenomics.sh` | (new — not present in Pillar B; harnesses there only ran the reference tool) | Driven by the streaming-first contract from `feedback_streaming` |
| `run_reference.sh` | `validation/external/regenie/run_regenie.sh` + `validation/external/ldsc/run_ldsc.sh` | Combined; quant-only (no Firth path) per Phase 57 MVP |
| `compare.py` | `validation/external/regenie/compare.py` + `validation/external/ldsc/compare.py` | Three-comparison structure (β corr, h² Δ, peak RAM); thin parser/check pattern |
| `_lib/preflight.sh` | byte-identical from `validation/external/_lib/preflight.sh` | none — copied verbatim |

The detailed mirror analysis lives in
`docs/superpowers/research/na3-ukb-harness-research.md`.

## Layout

```
validation/external/ukb/
├── install.sh              # REGENIE + LDSC install + pre-flight + UKB-path verify
├── fetch_data.sh           # symlink UKB + fetch 1000G LD scores + cache slot
├── run_torchgenomics.sh        # TG lmm-scan + TG h² (peak RSS via /usr/bin/time -v)
├── run_reference.sh        # REGENIE Step 1+2 + LDSC --h2 (peak RSS likewise)
├── compare.py              # parses both sides; emits findings-ledger row
├── README.md               # this file
├── .env_marker             # records REGENIE/LDSC versions + staged paths (gitignored)
├── .cache/                 # downloaded REGENIE zip + LD-score archives (gitignored)
├── bin/                    # local REGENIE binary if downloaded (gitignored)
├── data/                   # symlinks to user UKB + extracted LD scores (gitignored)
├── reference_outputs/      # REGENIE + LDSC outputs (gitignored)
└── torchgenomics_outputs/      # TG outputs (gitignored)
```

## Pre-flight numbers in detail

```
ukb-install:
  required: 100 GB disk on /home, 32 GB RAM
  observed (typical): 8 GB used by REGENIE+LDSC install

ukb-fetch:
  required: 20 GB disk, 8 GB RAM
  observed: ~600 MB LD-score archive + extraction; UKB stays where it is

ukb-tg-run:
  required: 100 GB disk, 32 GB RAM
  observed at n=50K: ~20 GB K + ~4 GB per chunk + ~2 GB Python overhead

ukb-ref-run:
  required: 100 GB disk, 32 GB RAM
  observed at n=500K: ~24 GB REGENIE Step 1 (Mbatchou 2021)
```
