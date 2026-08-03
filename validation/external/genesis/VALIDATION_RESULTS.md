# GENESIS / SNPRelate validation results — Phase 57 Unit A, Task 6

Definitive external reference-tool comparison of
`torchgenomics.linalg.kinship_admixed`'s three admixture-aware estimators
against the gold-standard R packages **SNPRelate 1.x** and **GENESIS**, run
on synthetic admixed+related fixtures with known ground truth (labeled
population of origin, injected parent-offspring pairs). See
`README.md` in this directory for the reproduction recipe and what each
script does; this file records the actual numbers observed and the verdict
for each estimator.

## Summary table

| torchgenomics function | Reference | Fixture | Agreement | Verdict |
|---|---|---|---|---|
| `king_robust_kinship` | SNPRelate `snpgdsIBDKING(type="KING-robust")` | 2-pop, n=120, m=2000 | max-abs-diff = **4.996e-16**, Pearson r = **1.0** | **REFERENCE-EQUIVALENT** (machine-exact) |
| `pc_air` | GENESIS `pcair()` | 2-pop, n=120, m=2000 (1 real ancestry axis) | PC1 \|r\| = **0.9998** | **REFERENCE-CONCORDANT** (~0.999) |
| `pc_air` | GENESIS `pcair()` | 3-pop, n=135, m=2500 (2 real ancestry axes) | PC1 \|r\| = **0.9992**, PC2 \|r\| = **0.9990** | **REFERENCE-CONCORDANT** (~0.999) |
| `pc_relate` | GENESIS `pcrelate()` | 2-pop, n=120, m=2000 | r = 0.849 (AF on all) → **0.913** (training-set AF, shipped) → **0.937** (GENESIS's exact unrelated set) | **STRONGLY CONCORDANT** (r≈0.94), **NOT** reference-equivalent |

## KING-robust — machine-exact

`king_robust_kinship(G)` vs. `snpgdsIBDKING(g, type="KING-robust")` on the
full, unpruned 2000-marker fixture: max-abs-diff over all off-diagonal pairs
is **4.996e-16** (FP64 machine-epsilon floor) with Pearson r = 1.0.

This was reached after a formula correction (commit `3739a9b`): a prior
version used `(N_AaAa - 2*N_AAaa) / (Nhet_i + Nhet_j)` (SUM denominator,
and a numerator that silently dropped every `|g_i - g_j| == 1` marker). The
correct Manichaikul et al. 2010 KING-robust formula is

```
phi_ij = 0.5 - Sd_ij / (4 * min(Nhet_i, Nhet_j))
```

i.e. an elementwise **MIN**, not sum, of the pairwise-complete
heterozygote counts, with `Sd_ij` counting every diff-category (0/1/4)
marker. Reproduction: `run_king_snprelate.R` → `compare.py`.

**Verdict: REFERENCE-EQUIVALENT.**

## PC-AiR — real ancestry axes match to ~0.999

Two fixtures were used because a single 2-population fixture only has one
real ancestry axis (PC1), which cannot demonstrate that a *second* PC also
recovers ancestry correctly.

- **2-population fixture** (`export_fixture.py`, n=120, m=2000, pruned to
  the LD-independent subset before PCA): PC1 `|r| = 0.9998` vs. GENESIS
  `pcair` PC1. (`run_pcair_pcrelate.R` / `run_pcair_only` path.)
- **3-population fixture** (`export_fixture_3pop.py`, n=135, m=2500, two
  real ancestry axes by construction): PC1 `|r| = 0.9992`, PC2
  `|r| = 0.9990` vs. the corresponding GENESIS `pcair` axes.
  Reproduction: `export_fixture_3pop.py` → `run_pcair_3pop.R` → `compare.py`
  (or manual comparison against `out3/genesis_pcair.tsv`).

Every REAL ancestry axis (i.e. up to the number of source populations minus
one) agrees with GENESIS to `|r| ~ 0.999`. Sub-dominant PCs beyond the
number of real ancestry dimensions are pure sampling/LD noise eigenvectors
with no population-structure signal to anchor them, so two independent
eigensolvers (torch's vs. GENESIS/LAPACK's) are expected to diverge there —
this is expected and correct, not a defect. PC signs are arbitrary
throughout and were compared by absolute correlation.

**Verdict: PC-AiR ancestry axes are REFERENCE-CONCORDANT (~0.999).**

## PC-Relate — strongly concordant, not reference-equivalent

`pc_relate` implements the published Conomos et al. 2016 moment-estimator
formula (eq. 5) directly. Three successive experiments were run to find the
ceiling of agreement with GENESIS `pcrelate`, all on the same 2-population
fixture and using GENESIS's own `pcair` PCs as input to isolate the
PC-Relate step itself:

| Experiment | r vs. GENESIS `pcrelate` | Script |
|---|---|---|
| AF regression fit on **all** individuals (`training_set=None`) | 0.849 | `run_pcair_pcrelate.R` |
| AF regression restricted to **this module's own** unrelated-partition mask (`training_set=<pcair_partition mask>`) — the shipped default recommendation | **0.913** (max-abs-diff 0.045) | `run_pcrelate_only.R` |
| AF regression restricted to **GENESIS's own exact** unrelated training set (`pca$unrels`, exported by `export_unrels.R`) | **0.937** | `export_unrels.R` + rerun with the exported set as `training_set` |

The jump from 0.849 → 0.913 confirms that training-set restriction (per the
Conomos et al. 2016 / GENESIS convention) is a real, verified correctness
improvement, not just a formula match. The further jump to 0.937 when using
GENESIS's *exact* training set (rather than this module's independently
computed partition) isolates the AF-regression step cleanly and shows most
of the partition-choice sensitivity is already resolved.

**r = 0.937 is the empirically-determined ceiling of this moment-estimator
implementation.** A follow-up experiment matched GENESIS's per-pair SNP
filtering (per-pair MAF/missingness exclusions) on top of the exact
training set and **verified it does NOT close the remaining gap** — ruling
out marker-panel mismatch as the explanation. The residual ~6% divergence
is therefore attributed to GENESIS `pcrelate`'s internal small-sample
bias-correction / normalization machinery (e.g. iterative re-weighting of
the AF regression, and the exact self-kinship/inbreeding estimator of
Conomos et al. 2016 eq. 6), which is not implemented here and goes beyond
the published moment-estimator formula this function implements.

**Verdict: STRONGLY CONCORDANT with GENESIS `pcrelate` (r ≈ 0.94), NOT
reference-equivalent.** Callers who require exact numerical equivalence
with GENESIS `pcrelate` (e.g. for a publication claiming tool-parity)
should use GENESIS `pcrelate` directly. `torchgenomics.pc_relate` is
appropriate where a fast, GPU-portable, moment-consistent ancestry-adjusted
kinship estimate is sufficient.

## Build / infrastructure notes

- **conda / RSQLite build fix**: on the validation host, building the R
  package dependency chain (SNPRelate → GWASTools → GENESIS, transitively
  depending on `RSQLite`/`DBI`) under a conda R environment failed until
  `R_MAKEVARS_USER` was cleared (a stale user-level `Makevars` was leaking
  incompatible compiler flags — e.g. a non-conda `CC`/toolchain path — into
  the package build). Fix: ensure `R_MAKEVARS_USER` is unset (or points at a
  clean file with no conflicting flags) before running
  `BiocManager::install()`; see `install.sh`.
- **GENESIS `pcrelate` BiocParallel deadlock on many-core hosts**: on a
  26-core machine, `pcrelate()`'s default `BPPARAM` (multicore
  `BiocParallel`) deadlocked / hung indefinitely inside the
  `GenotypeBlockIterator` C-level read path. Fix: pass
  `BPPARAM=SerialParam()` explicitly (see `run_pcrelate_only.R`,
  `run_pcair_3pop.R` line calling `pcrelate`). This is a known class of
  issue with BiocParallel's multicore backend and file-descriptor-based
  GDS readers under high core counts; serial execution is slower but
  reliable and sufficient for a one-off validation run.

## Reproduction

```bash
bash validation/external/genesis/install.sh
bash validation/external/genesis/fetch_data.sh
Rscript validation/external/genesis/run_king_snprelate.R
Rscript validation/external/genesis/run_pcair_pcrelate.R
Rscript validation/external/genesis/export_unrels.R
Rscript validation/external/genesis/run_pcrelate_only.R
python3 validation/external/genesis/export_fixture_3pop.py
Rscript validation/external/genesis/run_pcair_3pop.R
python3 validation/external/genesis/compare.py
```

## Scope note

This file records the actual observed validation numbers for Phase 57 Unit
A Task 6. The tolerance gates enforced by `compare.py` (and the opt-in
pytest gate `tests/test_kinship_admixed.py::test_genesis_equivalence_if_present`)
are set from these numbers, floored (not rounded up) per the repo's
observed-then-floored tolerance convention — see the comments beside each
`TOL_*` constant in `compare.py`.
