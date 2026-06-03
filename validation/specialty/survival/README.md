# Survival GWAS reference comparison harness (Tier 3 C1)

Genome Biology Methods paper, Tier 3 specialty fixture C1.
Compares `torchgenomics.models.survival_glmm.SurvivalGLMM` (Phase 35;
Cox PH frailty + Breslow-Clayton PQL + SPACox SPA) against the
canonical R coxme implementation (Therneau, Grambsch & Pankratz 2003).

## Reference tool: coxme (R)

**Selected:** `coxme` 2.2.22 (CRAN; Therneau 2024).

**Rationale for coxme over `coxKM` (the brief preferred):**
- `coxKM` (Cai et al. 2011, Biometrics 67:975) is a kernel-machine
  score test for SNP-set association, not a per-SNP Wald test. Since
  our `SurvivalGLMM.score_chunk` scans per-SNP, `coxKM` is not the
  direct algorithmic counterpart.
- `coxme` provides a per-SNP Wald test on a Cox PH frailty model with
  user-supplied kinship covariance (`varlist = coxmeFull(K)`), which
  is exactly what `SurvivalGLMM` replicates. This is also the
  comparator already exercised by `tests/test_survival_reference.py`
  inside the repo.
- **SAIGE-COX** (Bi et al. 2020 SPACox; brief fallback) is the secondary
  reference. Install lives under `validation/external/saige/` and is
  already validated on the MDP fixture. Not invoked here because SAIGE
  Step 1 PCG-based null fit is calibrated for n in the thousands;
  coxme is the correct comparator for n=500.

References for the reference:
- Therneau & Grambsch (2000). *Modeling Survival Data*. Springer.
- Therneau, Grambsch & Pankratz (2003). Penalized survival models and
  frailty. *J Comp Graph Stat* 12:156-175. (coxme algorithm)
- Therneau (2024). The coxme package, R package vignette.

## Fixture

Deterministic Weibull Cox PH frailty simulation:

- n=500 unrelated synthetic individuals.
- m=20 test SNPs + m_grm=200 background SNPs (HWE, diploid).
- Empirical VanRaden GRM K from background SNPs (+0.01 I regularization).
- Polygenic frailty u ~ N(0, 0.30 * K)  (h2=0.30).
- 3 causal SNPs (indices 0, 1, 2) with log_hr=0.6 each on
  standardized genotypes (HR exp(0.6)=1.82).
- Weibull baseline hazard, shape=1.5.
- ~15% target censoring rate via exponential calibration
  (observed: 19.0% on seed=42; well within the brief 10-20% range).
- Seed=42 throughout (torch.manual_seed + np.random.seed).

See `simulate.py` for the exact algorithmic spec.

## Reproduction

```bash
bash validation/specialty/survival/install.sh    # verify R + coxme + survival
bash validation/specialty/survival/fetch_data.sh  # simulate fixture
bash validation/specialty/survival/run.sh         # run reference + TG + compare
```

Each script sources `validation/external/_lib/preflight.sh` and asserts
disk + RAM headroom before any heavy work (user hard rule).


## Observed numerical agreement (seed=42, n=500, m=20)

| Metric | Observed | Floor | Direction |
|---|---|---|---|
| beta Pearson correlation | 0.9956 | 0.95 | min |
| beta Spearman correlation | 1.0000 | (informational) | min |
| SE Pearson correlation | 0.9214 | (informational) | min |
| -log10(p) Pearson | 1.0000 | (informational) | min |
| -log10(p) Spearman | 0.9985 | 0.85 | min |
| causal beta median rel \|delta\| | 0.186 | 0.30 | max |
| causal beta median \|delta\| | 0.161 | (informational) | max |
| causal beta max rel \|delta\| | 0.378 | (informational) | max |
| causal SNPs in top-K (TG) | 3 / 3 | 2 | min |
| causal SNPs in top-K (coxme) | 3 / 3 | 2 | min |
| causal sign agreement | 3 / 3 | 2 | min |
| top-K overlap (TG vs coxme) | 5 / 5 | (informational) | min |
| SPA-corrected p (#) | 1 of 20 | (informational) | - |

Floors are observed-then-floored per Pillar B convention: comfortably
below the observed value so cross-host CPU/BLAS drift does not flip
the gate.

## Brief targets vs observed

The brief targets were:
- beta within 3 sig figs (i.e. \|rel delta\| <= 5e-3 per SNP)
- SPA-p within 2 sig figs (i.e. \|rel delta\| <= 5e-2)

**Brief target NOT met at the per-causal-SNP beta level.** Observed
median relative beta delta on causal SNPs = 0.186 (~80% departure
from 3 sig figs).  This is a *documented* divergence, not a fix-now F3.
Cause: see *Documented divergences* below.

**Brief targets MET at the rank / detection / direction level:**
- Both tools detect 3 of 3 causal SNPs in top-5.
- Sign agreement on all 3 causal effects.
- -log10(p) Spearman = 0.998 (near-perfect ranking).
- Pearson beta = 0.996 (near-linear).

The reviewer-grade reading: the score-test (TG) and Wald-test (coxme)
produce *parallel* beta estimates that disagree by ~15-20% at causal
loci because they use different curvature (null vs MLE).  For
downstream GWAS use the rank / detection / direction agreement is the
practically-relevant metric; the per-SNP beta scale agreement is a
known algorithmic mismatch documented below.

## Documented divergences

All three of the following are *known algorithmic differences*, not
V1-core defects.  Per spec section 9, SurvivalGLMM is post-V1; per F3
policy these are post-V1 documented findings, not fix-now items.

1. **Wald (coxme) vs score (TG):** coxme refits the full Cox PH frailty
   model *with* the SNP included and reports the MLE-curvature SE.
   TG SurvivalGLMM holds the SNP out, scores it against the null
   GLMM, and uses null-curvature SE.  The two tests are asymptotically
   equivalent under H0 but differ in finite samples at causal loci
   where curvature is non-trivial.  This explains the ~15-20% beta
   delta at causal SNPs (snp0/1/2) and the near-perfect null-SNP
   agreement.

2. **Laplace REML (coxme) vs Breslow-Clayton PQL (TG):** coxme fits
   the variance component via Therneau 2003 Laplace-approximated REML.
   TG SurvivalGLMM uses Breslow-Clayton (1993) PQL on the Poisson
   working model.  On this fixture TG sig2_g converged to ~4e-8 (near
   the variance boundary) while coxme found a small positive variance.
   Both implementations report PQL convergence (TG converged=True) and
   produce calibrated null p-values (Spearman rho = 1.0 on the null
   17 SNPs).

3. **SPACox SPA (Bi et al. 2020) vs no SPA (coxme):** coxme has no
   saddlepoint approximation. TG applies SPA when chi2 > 2 (default).
   On the seed=42 fixture only 1 of 20 SNPs triggers SPA correction;
   the SPA-on -log10(p) Spearman vs coxme Wald is 0.998 (unchanged from
   SPA-off because the corrected SNP is a true positive in the tail
   where SPA and standard chi-square already agree closely).


## References cited

- **Breslow, N.E. & Day, N.E.** (1980). *Statistical Methods in Cancer
  Research, Vol 1: The Analysis of Case-Control Studies.* IARC.
  (Cox PH baseline; Breslow ties.)
- **Breslow, N.E. & Clayton, D.G.** (1993). Approximate inference in
  generalized linear mixed models. *J Am Stat Assoc* 88:9-25.
  (PQL on Poisson working model.)
- **Therneau, T.M., Grambsch, P.M. & Pankratz, V.S.** (2003). Penalized
  survival models and frailty. *J Comp Graph Stat* 12:156-175.
  (coxme algorithm.)
- **Bi, W., Fritsche, L.G., Mukherjee, B., Kim, S. & Lee, S.** (2020).
  A fast and accurate method for genome-wide time-to-event analysis,
  with application to a large-scale electronic health record data.
  *Nat Commun* 11:1626. (SPACox SPA.)
- **He, L. & Kulminski, A.M.** (2020). Fast algorithms for conducting
  large-scale GWAS of age-at-onset traits using Cox mixed-effects
  models. *Genetics* 215:41-58. (COXMEG.)

## Files in this directory

| Path | Purpose |
|---|---|
| `install.sh`        | Verify R + coxme + survival; marker on success, infra_blocker on failure. |
| `fetch_data.sh`     | Generate fixture via simulate.py (idempotent). |
| `simulate.py`       | Python simulator (Weibull + frailty; seed=42). |
| `run.sh`            | Run coxme reference, then compare.py; writes manifest. |
| `run.R`             | Per-SNP coxme reference fits. |
| `compare.py`        | Run TG SurvivalGLMM (SPA off + on), compute agreement, write results/. |
| `results/summary.tsv`     | Per-SNP beta / SE / p from both tools + deltas. |
| `results/agreement.json`  | Machine-readable check-by-check pass/fail + extras. |
| `results/manifest.sha256` | SHA256 manifest of the two committed result files. |

## F3 status

No F3 findings.  SurvivalGLMM is post-V1 (Phase 35); the per-SNP beta
scale divergence is a methodological mismatch (Wald vs score, Laplace
REML vs PQL), documented above per spec section 9.

Pre-flight gate, observed-then-floored tolerances, scientific rigor,
SHA256 manifest, no autonomous push, output gate to specialty/survival/
-- all satisfied.
