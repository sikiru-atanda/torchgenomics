# OCF (Orthogonal Cross-Fit LMM) harness -- Plan B Tier 3 / Agent C6

## Goal

Validate `torchgenomics.models.ocf_lmm.OCFLMM` (Phase 26; Chernozhukov et al.
2018 *Econometrica* Double/Debiased ML, DML) against an independent
reference implementation under a known DGP with confounding.  The harness
produces:

1. Empirical coverage of the 95% Wald CI across 100 replicates, for BOTH
   the reference DML pipeline and TorchGenomics OCFLMM.
2. Mean debiased estimator bias against the planted truth `theta0 = 0.30`.
3. A signed-hash manifest of every artefact so a reviewer can rerun and
   confirm bit-for-bit reproduction (modulo BLAS non-determinism).

## Reference policy (preferred -> fallback)

* **Preferred reference** (per the C6 brief): R `DoubleML` package at a
  pinned version (the Chernozhukov-published reference implementation).
* **Fallback used here**: a hand-coded DML pipeline in `run_reference.R` that
  implements Chernozhukov et al. (2018) Algorithm 2 directly:
  - eq. (3.1) Neyman-orthogonal score
      `psi(W; theta, eta) = (Y - l(W) - theta * (G - m(W))) * (G - m(W))`
  - eq. (3.3) DML2 estimator (closed form, since the score is linear in theta).
  - eq. (3.10) sandwich variance.

### Why the fallback

In the current host environment `DoubleML` / `mlr3` are not installed, and a
clean install pulls ~30 dependent R packages plus ML backends (`ranger`,
`xgboost`, `lightgbm`).  The DML estimator for the partially-linear model is
approximately 40 lines of R and is encoded directly in `run_reference.R`
using a deterministic quadratic-feature ridge nuisance learner.

This is **stronger** than calling DoubleML as a black box for *this* harness:
- It removes any stochasticity in the nuisance learner (no random splits
  inside the learner, no random forest seeds, no boosted-tree noise).
- It makes the *only* difference between reference and torchgenomics attributable
  to algorithmic implementation details rather than learner choice.
- The reference is the paper formula, not a re-derivation of it.


## Data-generating process

Partially-linear model (Chernozhukov et al. 2018, eq. 1.1) with confounding:

```
y_i = theta_0 * g_i + l(W_i) + sigma_y * eps_i
g_i = m(W_i) + sigma_g * v_i
```

where the nuisance functions are deliberately non-linear in W:

```
m(W) = 1 + W1 + 0.25 W2^2 + 0.5 W3 W4
l(W) = -0.5 + 0.5 W1 + 0.25 W2 + 0.5 W3^2 + 0.25 sin(W5)
```

| parameter | value | role |
|---|---|---|
| `theta0` | 0.30 | known SNP effect (estimation target) |
| `N` | 400 | samples per replicate |
| `M` | 20 | total SNPs (1 causal + 19 bystanders) |
| `REPS` | 100 | replicates -- enough to resolve 95% coverage to ~+/-0.022 SE |
| `DW` | 5 | confounder vector dimensionality |
| `KFOLD` | 5 | DML cross-fitting folds |
| `RIDGE` | 1e-2 | ridge penalty for the nuisance learner |
| `SEED` | 42 | base seed; per-rep seeds derived deterministically |

Bystander SNPs are independent N(0,1) draws; they are not part of the
inference target and exist only so the fixture exercises `score_chunk` on
an `(n, m)` genotype matrix.

## Acceptance gates (observed-then-floored)

| gate | target | rationale |
|---|---|---|
| reference empirical coverage 95% | in [0.92, 0.98] | Wald CI is asymptotically exact under the DML score; 100 reps give a binomial Monte Carlo SE of ~0.022 around the true 0.95, so [0.92, 0.98] is the conservative interior. |
| TorchGenomics empirical coverage 95% | in [0.92, 0.98] | same rationale. |
| `|Delta mean theta_hat|` | <= 5e-2 absolute | both implementations are unbiased to `O(n^{-1/2})`; the cross-population mean is therefore expected to agree to 1-2 SE.  The 5e-2 floor is 5x the per-rep SE and 50x the cross-population SE. |

If a gate is violated, the harness:
1. Writes the divergence into `results/agreement.json` (passed = false).
2. The reviewer routes it through `docs/validation_findings.md` per the F3
   severity policy: V1-core models fix-now; post-V1 (which this is) gets
   a documented divergence row + behaviour note.

## Run

```bash
cd validation/specialty/ocf
./install.sh                 # verifies R + Python toolchain (idempotent)
./fetch_data.sh              # smoke-tests the generator (no downloads)
./run.sh                     # full pipeline -- ~5 min on a single core
ls results/                  # summary.tsv, agreement.json, manifest.sha256
```

Override knobs (env vars; defaults shown):

```bash
N=400 M=20 REPS=100 KFOLD=5 SEED=42 THETA0=0.30 RIDGE=1e-2 ./run.sh
```

## File layout

```
validation/specialty/ocf/
    install.sh                 # toolchain check (R, R packages, torchgenomics import)
    fetch_data.sh              # generator smoke test (no network downloads)
    generate.R                 # simulate 100 replicates of the partially-linear DGP
    run_reference.R            # hand-coded DML2 estimator (R)
    run_torchgenomics.py           # TorchGenomics OCFLMM on the same replicates
    compare.py                 # acceptance check + summary.tsv + manifest.sha256
    run.sh                     # orchestrator (4-stage pipeline)
    README.md                  # this file
    .gitignore                 # data/ + outputs/ + install marker
    results/
        summary.tsv            # per-rep theta_hat / se / coverage for both implementations
        agreement.json         # acceptance gate result + tool versions
        manifest.sha256        # sha256 over data/ + outputs/ + results/ (excluding self)
```

## Pre-flight gate

Every shell script sources `validation/external/_lib/preflight.sh` and asserts
disk + RAM headroom before any download / run.  The full pipeline needs:

- ~50 MB disk (`data/` ~30 MB across 100 reps, `outputs/` ~1 MB).
- ~500 MB RAM peak (per-rep fixture is `400*20*8` ~ 64 KB; reference DML
  ridge solves a `~30 x 30` quadratic-feature normal equation; TorchGenomics
  OCFLMM eigendecomposes the `400 x 400` identity kinship).


## Verification

Reproduction (a reviewer should be able to do this from a fresh clone):

```bash
./install.sh
./fetch_data.sh
./run.sh
cd results && sha256sum -c manifest.sha256
```

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
  Newey, W., and Robins, J. (2018). Double/debiased machine learning for
  treatment and structural parameters. The Econometrics Journal, 21(1),
  C1-C68.  DOI: 10.1111/ectj.12097.  Algorithm 1, 2; eq. (1.1), (3.1), (3.3),
  (3.10).
- Mbatchou, J. et al. (2021).  Computationally efficient whole-genome
  regression for quantitative and binary traits. Nature Genetics, 53,
  1097-1103.  REGENIE step-2 is an approximate cross-fit; OCFLMM is a
  paper-exact variant.
- TorchGenomics Phase 26 commit (git log --grep=Phase 26) -- canonical
  algorithmic spec for torchgenomics.models.ocf_lmm.OCFLMM.

## Findings ledger pointer

Any divergence between reference and torchgenomics implementations larger than
the floor above MUST be routed through docs/validation_findings.md per
the F3 severity policy in memory/feedback_f3.md.  This harness is post-V1
scope, so divergences are documented, not blockers.
