# S5 — Observed-then-floored tolerance protocol

## Definition

Every numerical tolerance gating a TorchGWAS validation harness is calibrated by the observed-then-floored protocol described in the user standing rule recorded under `memory/feedback_validation_spec.md`. The rule, in third person: tolerances are not aspirational targets copied from spec section 16; they are set by running the harness once on real fixture data, recording the actual observed delta, and then floored to that observed value times a small safety margin chosen by the harness author (typically 2x-10x, never less than the next half-order-of-magnitude). Every gate carries an inline comment naming the observed value and why the floor was chosen (FP roundoff, parameterisation slack, Monte-Carlo noise at the fixture replicate count, reference-tool calibration limit, etc.).

## Why

Spec section 16 sets aspirational tolerances (e.g. `|Delta p| < 5e-8` between TorchGWAS and GEMMA). A failing test in the calibrated regime should signal a real degradation in the algorithm, not a spec-vs-numerical-reality mismatch. The observed-then-floored convention ensures the regression nets remain useful gates: if the implementation drifts, the observed delta exceeds the floor and the test fails for a reason that is actionable. If the implementation improves, the gate can be tightened in a future commit (re-floor against the new observed value).

This convention is paired with the F3 severity policy (`memory/feedback_f3.md`): a fix-now V1-core finding ships in three artefacts (production fix, regression test with the observed-then-floored gate, ledger row). A post-V1 documented finding records the observed delta with its threshold in the agreement.json and the divergence is xfail-ed or carried forward with a documented root cause.

## Per-harness floored tolerances

| Harness | Metric | Observed | Floored tolerance | Cushion | compare.py reference |
|---|---|---|---|---|---|
| MetaXcan | |Delta z| max | 3.07e-08 | 1.0e-06 | ~ 33x | `validation/external/metaxcan/compare.py` |
| MetaXcan | Pearson r (z) | 1.0 | >= 0.9999 | floor-at-noise | `validation/external/metaxcan/compare.py` |
| SMR + HEIDI | |Delta beta_SMR| / |beta_SMR| | 1.14e-06 | 5.0e-05 | ~ 44x | `validation/external/smr/compare.py` |
| SMR + HEIDI | |Delta chi2_HEIDI| / chi2_HEIDI | 0.381 | 1.0 | 2.6x | `validation/external/smr/compare.py` (loose: TG uses 5 SNPs vs SMR 6) |
| coloc | max |Delta PP| (shared scenario) | 2.06e-05 | 5.0e-03 | ~ 240x | `validation/external/coloc/compare.py` |
| hyprcoloc | |Delta candidate_snp_pp| | 3.69e-06 | 1.0e-03 | ~ 270x | `validation/external/hyprcoloc/compare.py` |
| haplo.stats | max |Delta freq| (per-hap EM) | 1.14e-04 | 2.0e-04 | 1.75x | `validation/external/hapref/compare.py` (3 sig figs target) |
| haplo.stats | max relative |Delta beta| | 4.38e-03 | 5.0e-03 | 1.14x | `validation/external/hapref/compare.py` (3 sig figs) |
| haplo.stats | relative |Delta global F-test p| | 7.57e-02 | 1.0e-01 | 1.32x | `validation/external/hapref/compare.py` |
| Cox PH (coxme) | Pearson r (beta) | 0.9956 | >= 0.95 | next half-order | `validation/specialty/survival/compare.py` |
| Random regression (lme4) | |Delta var_ratio_basis| max | 0.0255 | 0.05 | 2.0x | `validation/specialty/rr/compare.py` |
| Random regression (lme4) | |Delta -log10 p_g_P1| | 1.155 | 2.0 | 1.73x | `validation/specialty/rr/compare.py` |
| Within-family | median |Delta beta_indirect| (causal) | 0.0456 | 0.1 | 2.2x | `validation/specialty/family/compare.py` |
| Within-family | median |Delta attenuation| (finite) | 0.396 | 1.0 | 2.5x | `validation/specialty/family/compare.py` |
| Threshold-linear | |Delta R| max | 7.25 | 10.0 | 1.38x | `validation/specialty/threshold/compare.py` (gibbs under-mixed; documented F3-C4-1) |
| Threshold-linear | |Delta beta_SNP| mean abs | 0.0381 | 0.15 | ~ 4x | `validation/specialty/threshold/compare.py` |
| Knockoff FDR | |Delta mean FDR| | 0.1548 | 0.2 | 1.29x | `validation/specialty/knockoff/compare.py` |
| OCF DML | |Delta mean theta_hat| | 0.180 | 0.05 | (FAIL: documented post-V1) | `validation/specialty/ocf/compare.py` |
| LDSC (post-fix) | |Delta h2| | 1.0e-05 | 1.0e-02 | ~ 1000x | `validation/external/ldsc/compare.py` (tightened from 3.5e-01 to 5e-03 per spec sect.16) |
| MR egger (post-fix) | |Delta beta| | 4.1e-05 | 5.0e-03 | ~ 120x | `validation/external/twosamplemr/compare.py` (Bowden 2015 alignment) |
| MR-PRESSO (post-fix) | |Delta global p| | 1.0e-03 | 5.0e-02 | 50x (MC-bounded at NbDistribution=1000) | `validation/external/twosamplemr/compare.py` |
| SuSiE-RSS (post-V-update) | beta_sd Pearson | 0.99933 | >= 0.995 | observed at noise-floor V-update gap | `validation/external/susieR/compare.py` |

## Cross-link

The compare.py scripts under `validation/external/<tool>/` and `validation/specialty/<model>/` are the canonical source for each floor; updates to a floor go in the same commit as the corresponding agreement.json regeneration and a ledger row in `docs/validation_findings.md` (per F3 policy). Every floor above is reachable from the manifest by walking `paper/reproducibility/manifest.json -> F2.numbers.per_harness.<name>.checks[*].threshold` and `F7.numbers.panels.C<n>.checks[*].threshold`.
