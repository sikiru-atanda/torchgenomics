## Results: Specialty models

The V1-core Gaussian LMM does not cleanly cover regimes that GWAS
practice routinely encounters: censored time-to-event traits, irregularly-
sampled longitudinal trajectories, family confounding, ordinal/discrete
liabilities, FDR-controlled discovery under LD, and debiased inference
with high-dimensional nuisance. TorchGWAS treats these as a *specialty
cluster*: eight models, each anchored to a primary methodological paper,
validated through F7 with six external head-to-head panels (C1-C6) and
two internal-consistency panels (GU, LRO). Per-panel pass/fail aggregates
into the F2 equivalence grid alongside the fifteen core reference tools.

**C1 -- Cox PH frailty survival GWAS (Phase 35, `SurvivalGLMM`).** A simulated
PH fixture (n = 500, m = 20, 3 causal SNPs, 19 % censoring) was scanned by
`survival-scan` (Breslow-Clayton PQL null + score test, with SPACox saddlepoint
correction at the tail) and the R `coxme` package Wald test. Across all 20
SNPs the beta-coefficient Pearson correlation was 0.9956 (Spearman 1.0); the
-log10(p) Spearman correlation was 0.998; and TorchGWAS recovered all 3/3
causal SNPs in its top-5 -- matching `coxme` -- with full sign agreement
(`validation/specialty/survival/results/agreement.json`, 6/6 checks PASS).

**C2 -- Random regression LMM (Phase 38, `RandomRegressionLMM`).** Longitudinal
trajectories were fit with a Legendre-polynomial basis (order 2) and benchmarked
against `lme4::lmer` on a five-time-point fixture. Permanent-environment
variance ratios agreed within absolute tolerance 0.05 across all three basis
coefficients; intercept- and slope-effect beta estimates agreed to ten
significant figures (relative delta 0.0); per-time-point beta(t) tracked the
reference to within 13.4 % at the worst sampling point
(`validation/specialty/rr/results/agreement.json`, 7/7 PASS).

**C3 -- Within-family LMM (Phase 23, Young et al. 2022).** A 1000-sibling /
500-family fixture with planted direct (beta_d = 0.20) and indirect
(beta_i = 0.10) effects was scanned by `family-scan`. Because the `snipar`
installer fails on Python 3.13, the comparator is the paper-OLS reference
of Young et al. (Algorithm 1). Median |delta beta_direct| across 50 causal
SNPs is 1.7e-10 -- bit-equal to ten decimal places -- and median
|delta beta_indirect| is 0.046
(`validation/specialty/family/results/agreement.json`, 3/3 PASS).

**C4 -- Threshold-linear multi-trait (Phase 22, Bermann et al. 2026).** A
three-trait ordinal-plus-continuous fixture (planted G, R, beta_sex) was fit by
`threshold-scan` (T-NR solver) and BLUPF90+ `gibbsf90+`. Residual and
genetic-covariance components passed the observed-then-floored gate of 10
absolute units (max |delta| 7.2 / 8.5); fixed-effect betas passed the floored
gate; mean |delta beta_SNP| was 0.038 (gate 0.15). The reference 5000-sample
chain is below production length; tighter gates require ~100 k samples
(~30 min wall-time) and are deferred
(`validation/specialty/threshold/results/agreement.json`, 4/4 PASS).

**C5 -- Knockoff FDR (Phase 27, Sesia et al. 2020).** Across 100 replicates at
target FDR 0.20, `knockoff-scan` (KnockoffLMM, knockoff+ filter, group
knockoffs) achieved empirical FDR 0.000 / power 0.844; the R `knockoff` package
achieved 0.155 / 0.953. Both control FDR at the target; |delta mean FDR| 0.155
and |delta mean power| 0.109 pass the agreement gates
(`validation/specialty/knockoff/results/agreement.json`, 4/4 PASS). The lower
TG FDR reflects a structural design difference -- TorchGWAS operates on LD
blocks and reports block-level FDR while R `knockoff` reports per-SNP FDR --
documented in the harness README as a design choice, not a bug.

**C6 -- Orthogonal Cross-Fit LMM (Phase 26, Chernozhukov et al. 2018).** A
100-replicate partially-linear DGP with non-linear confounders compared TG
`OCFLMM` against a hand-coded DML2 reference implementing Algorithm 2. The
reference attained empirical 95 % coverage of 0.91 and mean theta_hat 0.318
(bias +0.018 against theta_0 = 0.30); TorchGWAS attained 0.41 and 0.498 (bias
+0.198). This is an F3 post-V1 divergence (finding #4 in
`docs/validation_findings.md`): `OCFLMM` `project_genotype=True` path
uses a linear regression of G on W, leaving systematic bias when E[G|W] is
non-linear -- exactly the DGP here. The proposed fix (a `nuisance_learner`
parameter accepting non-linear learners, mirroring `DoubleML` `ml_g` /
`ml_m`) is specified in the findings ledger; the V1 release ships with the
divergence disclosed rather than papered over.

**Internal-only panels (GU, LRO).** Two specialty models have no clean external
reference and are validated against model-specific invariants under the same
observed-then-floored protocol (spec section 10.6). The Genotype-Uncertainty
LMM (Phase 28) passes 4/4 invariants: zero-uncertainty input matches the None
path exactly (max |delta stat| 0.0); non-zero uncertainty is monotone-
conservative for 100 % of SNPs; median causal p (0.0034) is two orders of
magnitude below median null p (0.46); and 5/5 causal SNPs rank above the
5th-percentile null p. The Leave-Region-Out LMM (Phase 29) passes 5/5: at
the causal SNP, beta_LRO = 0.87 versus beta_full-K = 0.78 (truth 1.0) and
p_LRO = 1.8e-27 versus p_full-K = 3.0e-10 -- a seventeen-order-of-magnitude
improvement from removing proximal LD contamination
(`validation/specialty/{gu,lro}/results/agreement.json`).

Of 27 external check rows aggregated into F7, 24 PASS and 3 FAIL (all
three on C6, traceable to one documented root cause). The aggregated
specialty-cluster column joins F2 alongside the fifteen reference tools.
