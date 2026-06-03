# LDSC reference harness (Pillar B / B2)

LDSC (LD Score Regression, Bulik-Sullivan et al. 2015) is the canonical reference for SNP heritability (h²) and genetic correlation (rg) from GWAS summary statistics. This harness installs a pinned LDSC build via bioconda, simulates two correlated GWAS sumstats from real 1000G Phase 3 LD scores under known truth (h² = 0.4, rg = 0.5), runs LDSC's `--h2` + `--rg` modes, and asserts that TorchGenomics' `ldsc_h2` + `ldsc_rg_from_z` agree within calibrated tolerances.

## Pinned reference

| Field | Value |
|---|---|
| Tool version | LDSC 1.0.1 (Bulik-Sullivan & Finucane, 2014–2019) |
| Install path | bioconda channel: `ldsc=1.0.1=pyhdfd78af_2` |
| Python runtime | 2.7.15 (LDSC's pinned dependency stack) |
| LD scores | `1000G_Phase3_ldscores.tgz` from Zenodo DOI 10.5281/zenodo.7768714 |
| Weights      | `1000G_Phase3_weights_hm3_no_MHC.tgz` from same Zenodo record |
| SHA-256 (LD scores) | `470fe954080ba164f6b18ca2be8f8a6e2585d983b4a248556b5ffff2c3d0145f` |
| SHA-256 (weights)   | `d3e29112c64766dcee0ad9ec69ebda3d24cbe29ddb559d75443b10bcf8756c91` |

## Why bioconda + simulated sumstats?

- **Bioconda LDSC 1.0.1** packages the canonical Bulik-Sullivan release with its full Python 2.7 dependency stack pre-frozen. Pip-installing upstream directly is intractable on a modern host (numpy<1.17 / scipy<0.19 / pandas<0.21 do not build under Python 3). The maintained Python-3 fork (`belowlab/ldsc`) drifts from the published method numerically. Bioconda gives us the algorithmic reference cleanly.
- **Simulated sumstats over published GWAS**: LDSC's reference (`eur_w_ld_chr.tar.bz2` from the alkesgroup S3) is ~5 GB, and the GIANT 2018 Height archive is ~50 MB. We simulate two correlated chr22 sumstats (under known truth h² = 0.4, rg = 0.5) from the real 1000G LD scores, which exercises the full LDSC estimation path in <2 s per run while keeping the test reproducible by seed.

## Reproduction recipe

```bash
# From repo root:
bash validation/external/ldsc/install.sh        # creates conda env `ldsc_env`
bash validation/external/ldsc/fetch_data.sh     # downloads ~26 MB, simulates sumstats
bash validation/external/ldsc/run_ldsc.sh       # runs LDSC --h2 + --rg
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/external/ldsc/compare.py

# Or via pytest (requires -m external; skipped by default):
pytest -m external tests/test_external_ldsc.py -v
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts disk + RAM headroom before doing any work, per the Pillar B contract.

## Reference outputs (in `outputs/`)

| File | LDSC mode | TorchGenomics counterpart |
|---|---|---|
| `h2_trait1.log` | `--h2 sim_trait1.sumstats.gz --two-step 99999` (single-pass WLS) | `torchgenomics.postgwas.ldsc_h2` |
| `h2_trait2.log` | `--h2 sim_trait2.sumstats.gz --two-step 99999` | `torchgenomics.postgwas.ldsc_h2` |
| `rg.log`        | `--rg sim_trait1.sumstats.gz,sim_trait2.sumstats.gz --two-step 99999` | `torchgenomics.postgwas.ldsc_rg_from_z` |

## Calibrated tolerances

Spec §16 anchors: `|Δ h²| < 0.01`, `|Δ intercept| < 0.005`, `|Δ rg| < 0.02`.

Observed values from the IRWLS-port run (2026-04-30, 17 489 chr22 SNPs;
TorchGenomics now matches LDSC's IRWLS with both `--ref-ld` and `--w-ld`
LD-score inputs):

| Comparison | Metric | Observed | Floor (asserted) | Status |
|---|---|---|---|---|
| h² (trait 1) | \|Δ h²\| | 1.02e-5 | 1e-2 | bit-equal |
| h² (trait 1) | \|Δ intercept\| | **2.60e-5** | 5e-3 | bit-equal (post-IRWLS) |
| h² (trait 1) | \|Δ mean χ²\| | 4.64e-5 | 1e-3 | bit-equal sum |
| h² (trait 1) | \|Δ h² SE\| | 1.55e-4 | 5e-3 | jackknife block boundaries differ |
| h² (trait 2) | \|Δ h²\| | 1.58e-5 | 1e-2 | bit-equal |
| h² (trait 2) | \|Δ intercept\| | **3.91e-5** | 5e-3 | bit-equal (post-IRWLS) |
| h² (trait 2) | \|Δ mean χ²\| | 3.64e-5 | 1e-3 | bit-equal sum |
| h² (trait 2) | \|Δ h² SE\| | 3.34e-4 | 5e-3 | as above |
| rg | \|Δ rg\| | 1.15e-5 | 2e-2 | bit-equal |
| rg | \|Δ h² (T1)\| | 1.02e-5 | 1e-2 | bit-equal |
| rg | \|Δ h² (T2)\| | 1.58e-5 | 1e-2 | bit-equal |
| rg | \|Δ intercept (h² T1)\| | 2.60e-5 | 5e-3 | bit-equal |
| rg | \|Δ intercept (h² T2)\| | 3.91e-5 | 5e-3 | bit-equal |

### RESOLVED: LDSC IRWLS port (Phase 37 follow-up)

LDSC's regression uses **iteratively reweighted least squares** with heteroscedastic weights:

```
w_j = 1 / ( 2 · (intercept + (h²·N/M)·l_j)² · w_ld_j )
```

where:
  - `l_j` is the **reference** LD score (passed via `--ref-ld`)
  - `w_ld_j` is the **regression** LD score (passed via `--w-ld`; computed only over SNPs in the regression set — distinct file from the reference scores)
  - `intercept`, `h²` are the *current iterate's* parameters; LDSC iterates exactly 2 times (`for i in xrange(2)` in `ldsc/irwls.py`).

TorchGenomics' `ldsc_h2` previously used a single-pass WLS with the simpler heuristic
``w = 1.0 / torch.clamp(ld_scores ** 2, min=1.0)`` — no `w_ld` input, no
iteration. Empirically this preserved h² (slope-driven) but mis-estimated
the intercept by 0.10–0.28 absolute on inflated mean-χ² regimes.

**Resolution.** The Phase 37 IRWLS port in
`torchgenomics/postgwas/_ldsc.py` adds:

1. The full LDSC `Hsq.weights` heteroscedastic weight formula
   (`_hsq_weights`).
2. A 2-iteration IRWLS loop (matching LDSC's fixed `for i in xrange(2)`
   loop in `ldsc/irwls.py`).
3. An optional `w_ld=` keyword on `ldsc_h2` / `ldsc_intercept` /
   `ldsc_rg` / `ldsc_rg_from_z` for the regression-weight LD scores.
   When `None`, defaults to `ld_scores` (LDSC's standard fallback when
   only one set of LD scores is available).
4. An optional `n_iter=2` (default) and `two_step=False` (default) for
   compatibility with LDSC's two-step parameterization (currently only
   exposed; `two_step=True` runs LDSC's free→constrained intercept fit).
5. Support for a per-SNP `n` tensor (LDSC's per-SNP N column).

The legacy single-pass behavior is reachable via `n_iter=0`. With proper
`w_ld=...` + `n_iter=2`, TorchGenomics reproduces LDSC's intercept to
~3e-5 absolute (well below the spec §16 anchor of 5e-3) on the
simulated chr22 fixture.

**Tolerance choice (post-fix)**: we floor `|Δ intercept|` at 5e-3
(spec §16 anchor); observed maximum is 3.91e-5, leaving generous
headroom for future stat-numeric drift while still gating the IRWLS
contract.

### Why LDSC's two-step is disabled (`--two-step 99999`)

LDSC's default `--h2` mode runs a two-step estimator: step 1 fits both slope and intercept on chi² < 30 SNPs to estimate the intercept; step 2 fixes the intercept and refits the slope on **all** SNPs. TorchGenomics' `ldsc_h2` is structured as: step 1 fits all SNPs (warm-up), step 2 fits both slope and intercept on chi² < 30 SNPs. Different two-step parameterizations diverge meaningfully when many SNPs have chi² > 30 (which is our case: simulated mean chi² ≈ 45).

To get an apples-to-apples comparison we run LDSC with `--two-step 99999` (no SNPs filtered out) and TorchGenomics with `two_step_cutoff=99999.0` — both compute single-pass WLS. The TG↔LDSC two-step parameterization mismatch is a separate documented divergence, not exercised by this harness, and would also be a Phase 37 follow-up.

## Allele-flip handling

LDSC's `--rg` checks A1/A2 alignment between the two sumstats files unless `--no-check-alleles` is passed. We simulate trait 1 and trait 2 using identical (A1, A2) = (A, G) per SNP, so the alleles are pre-aligned by construction and `--no-check-alleles` is safe. This avoids an unrelated source of divergence (allele-flip semantics).

## Peak memory

LDSC 1.0.1 on 17 K SNPs (chr22) with 200 jackknife blocks completes in <2 s and uses <500 MB RAM. We pre-flight 1 GB working-set + 4 GB peak RAM as a generous safety margin (the Python 2.7 numpy stack carries its own footprint).

| Stage | Disk pre-flight | RAM pre-flight | Observed peak |
|---|---|---|---|
| install (~200 MB conda env download) | 4 GB | 6 GB | ~300 MB during conda solve |
| fetch (~26 MB downloaded, ~150 MB extracted) | 4 GB | 6 GB | <200 MB |
| run (3 LDSC invocations) | 4 GB | 10 GB | <500 MB total |
| compare.py (Python 3 + torchgenomics) | n/a | n/a | ~600 MB |

## Layout

```
validation/external/ldsc/
├── install.sh              # bioconda install of pinned LDSC 1.0.1
├── fetch_data.sh           # download 1000G LD scores + weights, simulate sumstats
├── simulate_sumstats.py    # helper invoked by fetch_data.sh
├── run_ldsc.sh             # produces 3 reference logs into outputs/
├── compare.py              # asserts tolerances vs TorchGenomics
├── README.md               # this file
├── .env_marker             # records conda env name + LDSC version (gitignored)
├── .cache/                 # downloaded archives (gitignored)
├── data/                   # extracted LD scores, weights, simulated sumstats (gitignored)
└── outputs/                # LDSC reference output logs (gitignored)
```

## CI / test integration

```bash
# Skipped by default (the conftest auto-skips `external` markers):
pytest tests/test_external_ldsc.py            # 3 skipped

# Opt-in:
pytest -m external tests/test_external_ldsc.py -v
```

The pytest module dynamically imports `compare.py` from outside the package tree, so no modification of `torchgenomics/` is required to wire this harness.

## Next steps (post-Pillar B)

- ~~IRWLS in `torchgenomics.postgwas.ldsc_h2` to close the intercept gap (Phase 37 follow-up; would tighten `TOL_INTERCEPT_ABSDIFF` from 3.5e-1 to ~5e-3).~~ — **DONE (2026-04-30)**: IRWLS port resolved B2; intercept agreement now ~3e-5 absolute, gated at 5e-3.
- Promote `sldsc_h2_partitioned` from single-pass WLS to full IRWLS (mirrors the fix applied to `ldsc_h2`); currently calls `_hsq_weights` once on the aggregate-derived initial weights.
- Wire `sldsc_h2_partitioned` against an S-LDSC `--h2 --overlap-annot` reference (deferred — needs the partitioned annotation files, ~1 GB, which we judged out-of-scope for B2).
