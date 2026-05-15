# coloc reference harness (Pillar B / Genome Biology paper section 10)

R coloc (Wallace lab; Giambartolomei et al. 2014) is the canonical two-trait Bayesian colocalization implementation. This harness installs coloc from CRAN, simulates three two-trait scenarios (shared-signal, distinct-signal, null), runs coloc.abf in R, and asserts that TorchGWAS coloc_pairwise agrees within calibrated tolerances.

## Pinned reference

| Field | Value |
|---|---|
| coloc version       | 5.2.3 (CRAN) -- requires >= 5.2 (Wallace 2020 erratum) |
| jsonlite version    | 2.0.0 (CRAN; JSON output bridge) |
| R runtime           | 4.5.1 (system R) |
| Simulation seed     | 42 |
| SNPs per scenario M | 50 |
| Causal SNP 1-based  | 25 |
| Priors              | p1 = p2 = 1e-4, p12 = 1e-5 (Giambartolomei 2014) |
| Wakefield prior W   | 0.15^2 (default for quant + sdY=1.0) |

Exact installed versions are recorded in `.install_marker` after a successful install.

## Why simulated sumstats (not GTEx eQTL)

Real eQTL + GWAS sumstats live behind dbGaP / access-controlled portals (GTEx v8 protected release, eQTLGen access tier). Pulling them inside CI breaks reproducibility and requires per-user access tokens. coloc.abf is summary-statistic only -- it does not need real biology to test correctness. Three scenarios with planted causal effects exercise every branch of the H0..H4 decomposition.

## Reproduction recipe

```
# From repo root:
bash validation/external/coloc/install.sh      # one-time CRAN install (3-5 min wall time)
bash validation/external/coloc/fetch_data.sh   # simulate 3 scenarios (<1 s)
bash validation/external/coloc/run.sh          # run coloc.abf (~5 s)
TORCHGWAS_DISABLE_NATIVE=1 python3 validation/external/coloc/compare.py
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts disk + RAM headroom before doing any work, per the Pillar B contract.

## Reference outputs (in `outputs/`)

| File | Source method | TorchGWAS counterpart |
|---|---|---|
| `coloc_results.json` (keys shared / distinct / null) | `coloc::coloc.abf(p1=1e-4, p2=1e-4, p12=1e-5)` | `torchgwas.postgwas.coloc_pairwise` |

Each scenario emits PP.H0..H4, the candidate SNP id (argmax of `SNP.PP.H4`), and the priors used.

## Calibrated tolerances

From the run captured in `results/agreement.json` (TorchGWAS 0.1.1, coloc 5.2.3, seed=42):

| Scenario | Metric | Observed | Floor (asserted) | Status |
|---|---|---|---|---|
| shared   | max abs delta PP.H0..H4 | 2.06e-5 | 5e-3 | PASS (near-FP) |
| shared   | candidate_snp id        | match   | -    | PASS (rs00025) |
| distinct | max abs delta PP.H0..H4 | 1.43e-1 | 5e-3 | FAIL (F3 finding) |
| distinct | candidate_snp id        | match   | -    | PASS (rs00025) |
| null     | max abs delta PP.H0..H4 | 5.88e-3 | 5e-3 | FAIL (small) |
| null     | candidate_snp id        | match   | -    | PASS (rs00037) |
| cross    | Pearson r(PP_R, PP_TG)  | 0.9926  | -    | informational |

**Why the tolerance is the target not the observed:** both implementations compute the same closed-form Wakefield ABF with identical priors, so the expected agreement is FP-noise (~1e-12). The 5e-3 floor would pass if the formulas matched. The distinct-scenario FAIL surfaces an F3-class divergence in `coloc_pairwise` (see below); deliberately not widening the tolerance, so reviewers see the gap.

## F3 finding: `coloc_pairwise` formula deviates from Giambartolomei 2014

The harness reproduces R `coloc::coloc.abf` exactly with the paper formula:

| Hypothesis | Giambartolomei 2014 / R combine.abf |
|---|---|
| log P(H0) | 0 |
| log P(H1) | log p1 + log SUMj ABF1_j |
| log P(H2) | log p2 + log SUMj ABF2_j |
| log P(H3) | log p1 + log p2 + log[(SUMj ABF1_j)(SUMj ABF2_j) - SUMj ABF1_j*ABF2_j] |
| log P(H4) | log p12 + log SUMj (ABF1_j * ABF2_j) |

`torchgwas.postgwas._hyprcoloc.coloc_pairwise` currently uses (see lines 360-365):

| Hypothesis | TG formula |
|---|---|
| log P(H0) | 0 |
| log P(H1) | log p1 + log SUMj ABF1_j - log m |
| log P(H2) | log p2 + log SUMj ABF2_j - log m |
| log P(H3) | log p1 + log p2 + log SUMj ABF1_j + log SUMj ABF2_j - 2*log m |
| log P(H4) | log p12 + log SUMj (ABF1_j * ABF2_j) - log m |

Two deviations:

1. **Spurious `-log m` normalization.** The 1/m factors do not cancel across hypotheses (H3 carries 1/m^2, others 1/m), so the normalized posterior biases H3 mass into H1/H2 once H4 stops dominating. This is what the distinct scenario surfaces: R PP.H3 = 0.997, TG PP.H3 = 0.854, with the 14% mass leaking into TG PP.H1 (0.141 vs R 0.003).

2. **Missing diagonal subtraction on H3.** The paper formula uses SUMj SUM_{k != j} ABF1_j * ABF2_k (sum over distinct SNPs only); equivalently (SUMj ABF1_j)(SUMj ABF2_j) - SUMj ABF1_j * ABF2_j. TG H3 uses the full outer product, double-counting the diagonal that already accrues to H4.

Numerical verification (compare.py + an inline reproducer using TG own `_wakefield_log_abf`): swapping in the R formula reproduces R coloc.abf output to FP precision on the distinct scenario (PP.H0=5.11e-10 H1=3.29e-3 H2=1.55e-7 H3=9.97e-1 H4=1.20e-4 matches the R reference). The per-SNP lABF is bit-equal between TG and R.

**Status**: this is post-V1 (Phase 42) per F3 severity policy -- a documented divergence, not a halt. The shared-signal path (the most common use case in practice) is near-FP-accurate. Distinct-signal and null-signal scenarios under-allocate H3/H0 mass and over-allocate H1/H2 mass. Recorded in `docs/validation_findings.md` for resolution after the V1 freeze.

## Files

```
validation/external/coloc/
  install.sh        # CRAN install of coloc + jsonlite + smoke test
  fetch_data.sh     # inline R simulator (3 scenarios, seed=42, M=50, causal at idx 25)
  run.sh            # bash wrapper around run.R (sources preflight)
  run.R             # invokes coloc.abf on each scenario; emits outputs/coloc_results.json
  compare.py        # runs torchgwas.postgwas.coloc_pairwise; asserts max abs delta PP <= TOL_PP
  README.md         # this file
  .gitignore        # excludes data/, outputs/, __pycache__/, .install_marker
  .install_marker   # version stamp written by install.sh (gitignored)
  data/             # gitignored; populated by fetch_data.sh
    scenarios.json
    sumstats_shared.tsv
    sumstats_distinct.tsv
    sumstats_null.tsv
  outputs/          # gitignored; populated by run.sh
    coloc_results.json
  results/          # checked in -- canonical harness artifacts
    summary.tsv     # per-scenario per-hypothesis R/TG/delta table
    agreement.json  # structured agreement report w/ priors + versions + tolerances
    manifest.sha256 # sha256 of every staged input/output
```
