# Knockoff FDR-control harness (Plan B / Tier 3 C5)

The R `knockoff` package (Patterson & Sesia, CRAN 0.3.6) is the canonical
implementation of the model-X / fixed-design knockoff filter from Candes,
Fan, Janson & Lv (2018) JRSSB and the group-knockoff GWAS extension of
Sesia, Sabatti & Candes (2020) JASA.  This harness installs `knockoff` +
`glmnet`, simulates block-LD genotype/phenotype replicates with a known
causal SNP set, runs both R `knockoff::knockoff.filter` (per-SNP fixed-design
filter) and `torchgenomics.models.knockoff_lmm.KnockoffLMM` (per-block LMM-aware
filter) on the same fixtures, and compares empirical FDR + power at the
matched nominal level.

## Pinned reference

| Field | Value |
|---|---|
| R package          | `knockoff` 0.3.6 (CRAN) |
| glmnet             | 4.1-10 (CRAN; lasso-coef-diff statistic) |
| jsonlite           | as resolved by CRAN at install time |
| Rdsdp              | as resolved (SDP solver dependency) |
| R runtime          | 4.5.1 (system R) |
| Knockoff method (R)        | `create.fixed` (fixed-design Gaussian; n>p) |
| Importance statistic (R)   | `stat.glmnet_coefdiff` (lasso CV coef-diff) |
| Knockoff+ offset (R)       | 1 (paper eq. 3.10 - knockoff+) |
| Knockoff method (TorchGenomics)| equicorrelated, per LD-block group knockoffs |
| LD method (TorchGenomics)      | r2-clumping (window 20, r2 >= 0.3) |
| Importance aggregation     | max_stat (max |Z| within block) |
| Target FDR                 | 0.20 |
| Simulation seed            | 42 |
| Replicates                 | 100 |
| n / p / block_size         | 500 / 200 / 10 |
| Within-block rho           | 0.7 (latent + noise convex combination) |
| k_causal                   | 8 (one per block, evenly spaced across blocks 0..18) |
| Causal effect size         | 1.0 (on standardized G) |
| Phenotype noise sd         | 0.3 |

Exact installed versions are recorded in `.install_marker` after `install.sh`.

## Why simulated, not real biology

The knockoff filter is a generic FDR-control procedure - it does not need real
biology to test correctness.  A block-LD simulation with a known causal set
is the natural minimal informative fixture; it tests every code path in both
`knockoff::knockoff.filter` and `KnockoffLMM` without confounders (population
structure, family relatedness, allele-flip orientation) that would otherwise
bias the empirical FDR estimate.  This follows the Candes/Fan/Janson/Lv 2018
sec. 5 simulation design.

## What is compared

Both tools see the same `data/replicates.npz` (100 replicates seeded at 42,
each n=500 x p=200 with 8 known-causal SNPs).  We measure on each replicate:

- R `knockoff.filter`  -> a SET of selected variant indices (per-SNP FDR).
  Empirical FDR_r = #{selected_r NOT in causal} / max(1, #selected_r).
  Empirical power_r = #{selected_r IN causal} / k_causal.

- TorchGenomics `KnockoffLMM` -> a SET of selected LD blocks (block-FDR a la
  Sesia 2020 sec. 2.2).  A block is a true positive iff it contains any
  causal SNP.  Empirical block-FDR_r = #{sel_blocks NOT containing any
  causal} / max(1, #sel_blocks).  Empirical block-power_r = #{sel_blocks
  containing >=1 causal} / #{true_blocks}.

The comparison is **mean empirical FDR + mean empirical power across all 100
replicates** at the matched target FDR = 0.20.  Both tools should control
their respective FDR at the nominal level (the JRSSB/JASA paper guarantee)
plus a finite-sample Monte-Carlo slack.

## Why two different filters (not the same algorithm twice)

The brief intentionally pairs **R `knockoff` per-SNP filter** with **TorchGenomics
`KnockoffLMM` per-block group filter** because that is the operational
comparison readers care about:

- R `knockoff` is the textbook fixed-design / model-X filter (Candes 2018).
- `KnockoffLMM` extends it with LMM-correlated nulls + LD-block group
  selection (Sesia 2020) - the GWAS-appropriate variant.

They do not share an importance statistic by design (Sesia 2020 sec. 2.3
explains why a per-SNP fixed-design statistic cannot be used directly on LD-
correlated GWAS data without inflated false positives - hence the group
extension).  Comparing them on the SAME fixture answers the operational
question: do both tools achieve FDR control at the nominal level, and how
do their power profiles differ?

## Reproduction recipe

```bash
# From repo root (worktree-aware paths):
bash validation/specialty/knockoff/install.sh    # ~1-3 min, idempotent
bash validation/specialty/knockoff/fetch_data.sh # ~30 s; simulate 100 replicates
bash validation/specialty/knockoff/run.sh        # ~10-30 min full sweep

# Outputs (gitignored except results/):
#   data/replicates.npz, data/sim_truth.json, data/replicates.rds (R cache)
#   outputs/torchgenomics_per_replicate.tsv  outputs/torchgenomics_summary.json
#   outputs/reference_per_replicate.tsv  outputs/reference_summary.json
#   results/summary.tsv  results/agreement.json  results/manifest.sha256
```

## Acceptance tolerances (observed-then-floored)

Per the Pillar B contract (see `memory/feedback_validation_spec.md`), the
acceptance gates floor the brief targets at the observed divergence from a
successful run.  Until the first run lands, these are the **brief targets**:

| Gate | Floor | Source |
|---|---|---|
| `|Delta mean FDR|`   | <= 0.02 | Plan B agent brief (C5) |
| `|Delta mean power|` | <= 5e-2 | Plan B agent brief (C5) |
| FDR overshoot above target (per tool) | <= 0.05 abs | JRSSB sec. 5 finite-n slack |

After the first successful run, the observed values are persisted in
`results/agreement.json` under `tolerances` (current floor) and `extras`
(observed values).  Tightening is allowed; loosening requires an F3 entry
in `docs/validation_findings.md`.

## Files

| File | Role |
|---|---|
| `install.sh`         | Install `knockoff` + `glmnet` + `jsonlite` from CRAN; record versions to `.install_marker` |
| `fetch_data.sh`      | Call `generate.py` to simulate replicate fixtures |
| `generate.py`        | Block-LD genotype + phenotype simulator (seed 42) |
| `run_torchgenomics.py`   | Run `KnockoffLMM` per replicate; write per-rep TSV + summary JSON |
| `run_reference.R`    | Run `knockoff::knockoff.filter` per replicate via the R session |
| `_npz_to_rds.py`     | One-shot NPZ -> RDS converter (so R does not need `reticulate`) |
| `run.sh`             | Orchestrator: pre-flight, run both tools, run `compare.py`, write manifest |
| `compare.py`         | Head-to-head agreement on mean FDR, mean power, FDR overshoot |
| `results/`           | Canonical comparison output (committed; overrides global gitignore) |
| `outputs/`           | Per-tool intermediate output (gitignored) |

## References

- Candes E., Fan Y., Janson L. & Lv J. (2018). "Panning for gold: model-X
  knockoffs for high-dimensional controlled variable selection."
  *Journal of the Royal Statistical Society B*, 80:551-577.  Sec. 2 (the
  knockoff filter); sec. 5 (block-correlated Gaussian simulations).
- Sesia M., Sabatti C. & Candes E. (2020). "Multi-resolution localization of
  causal variants across the genome."  *JASA* 115:1-15.  Sec. 2.2
  (group / block knockoffs) and Theorem 2.1 (block-FDR control).
- R `knockoff` package documentation:
  <https://cran.r-project.org/package=knockoff>

