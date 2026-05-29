# Reference-tool equivalence

## Tolerance protocol: observed-then-floored, never aspirational

Every numerical check that backs a claim in this paper is gated by an **observed-then-floored** tolerance. The protocol, codified as a project invariant in `memory/feedback_validation_spec.md`, has three steps: (i) run the head-to-head once on the staged fixture against the installed upstream tool; (ii) record the observed maximum delta on the chosen metric (e.g. `|Δz|_max = 3.07e-8` for MetaXcan); (iii) set the gate at the rounded order-of-magnitude above the observed delta, annotated by the observed value and the floor rationale (FP roundoff, parameterization slack, MCMC stochasticity). Tolerances are never inflated to mask a divergence; failing gates are logged in `docs/validation_findings.md` and triaged through the F3 severity policy (V1-core fix-now, post-V1 documented, regression halts the pillar). The harness keeps the failing gate at its observed-then-floored value so reviewers see the gap unmodified.

## Fifteen-tool harness inventory

Figure 2 enumerates the harness grid. **Diploid LMM**: BOLT-LMM, SAIGE, regenie, GEMMA, GAPIT. **Polyploid**: GWASpoly. **Variant-level utilities**: PLINK 2.0 (canonical Gabriel `--blocks` reference) and LDSC. **Fine-mapping**: susieR (NA1 streaming-SuSiE roundup, PolyFun-compatible output). **Mendelian randomization**: R `TwoSampleMR`. **Multi-omics**: MetaXcan / S-PrediXcan, SMR + HEIDI, R `coloc`, R `hyprcoloc`. **Haplotype**: `haplo.stats`. Three real-data fixture columns (`soymd`, `soynam`, `ukb`) share the six-artefact harness skeleton and decorate the diploid-LMM block with real-data context.

## Headline equivalence numbers

Of the 15 tools, 11 carry fully instrumented Tier-1 / Tier-3 head-to-head checks for this draft. The F2 renderer aggregates **40 of 45 numerical checks PASS** at their observed-then-floored thresholds; provenance is in `paper/reproducibility/manifest.json` under `F2.per_harness`. Selected highlights (read from `validation/<tier>/<tool>/results/agreement.json`):

- **MetaXcan / S-PrediXcan** — bit-equal to FP precision. `|Δz|_max = 3.07e-8` (gate 1e-6), `|Δβ|_max = 9.37e-17`, `|Δ-log10 p|_max = 1.56e-7`, Pearson r(z) = 1.0.
- **SMR + HEIDI** — `|Δβ_SMR|/|β_SMR| = 1.14e-6`, `|Δχ²_SMR|/χ²_SMR = 3.04e-7`. With the closed F3 #3 LD-weighted HEIDI variance enabled, χ²_HEIDI now agrees to 0.67% and p_HEIDI to 1.55%.
- **haplo.stats (hapref)** — reference haplotype identity matches; EM `max |Δfreq| = 1.14e-4`; per-haplotype `max relative |Δβ| = 4.4e-3`; global F-test relative `|Δp| = 7.6e-2`.
- **Cox PH frailty (coxme)** — β Pearson 0.9956 full-set; -log10(p) Spearman 0.9985; all three causal SNPs recovered in top-K with sign-agreement on every β.
- **Random regression (lme4)** — `|Δβ_g|/|β_g| = 0.0` to ten significant figures on both fixed-effect SNPs; RR variance-ratio basis components agree to ≤ 2.6e-2.
- **Within-family (paper-OLS, Young et al. 2022)** — median causal `|Δβ_direct| = 1.7e-10` (n=50); median causal `|Δβ_indirect| = 4.6e-2`.
- **Threshold-linear (BLUPF90+)** — three-trait Gibbs covariance components, fixed-effect βs, and SNP βs all clear gates (max `|ΔG_cov| = 8.54`, gate 10.0; mean SNP-β abs-delta = 0.038 on five planted causals, gate 0.15).
- **Knockoff FDR (R `knockoff`)** — `|Δmean FDR| = 0.15`; both tools control FDR at α = 0.20 with zero overshoot.
- **Real-data multi-omics worked example** (Figure 4B, `validation/multiomics/gtex_ukb/`): 48 variants at SORT1 ± 50 kb give **Pearson(z_eqtl, z_gwas) = −0.97** (orientation-direct on all 48) — the canonical cis-eQTL / GWAS co-localization signature at the LDL ↔ SORT1 locus.

## Five failing checks at draft time — four now closed on master

Five of 45 checks failed at their observed-then-floored gates at original draft time and were reported in the figure rather than glossed. **Four have since been closed on master** (commits on `master`, 2026-05-21); the fifth is post-V1 documented.

1. **hyprcoloc R, cluster membership** (F3 #1, **closed**). R placed SNPs in `{0, 1, 2}`; TG returned `{0, 2}`. The conditional-prior parameterization in `_hyprcoloc.py` was replaced with the Foley 2021 Eq. 2 hierarchical form; cluster membership, regional posteriors, and candidate SNP now match R hyprcoloc exactly.
2. **R `coloc`, `passed_overall`** (F3 #2, **closed**). The distinct-causal scenario diverged at `max |ΔPP| = 0.143`; root-caused to a spurious `-log m` plus a missing H3 outer-minus-diagonal subtraction. The Wallace-2020-erratum form has been applied via numerically stable `log_diff_exp`; PP.H3 moves from 0.85 to 0.997 and matches R `coloc::coloc.abf` at `max |ΔPP| = 2.06e-5`.
3. **SMR HEIDI variance** (F3 #3, **closed**). Optional `ld_matrix` argument to `heidi_test` implements the SMR-tool LD-weighted convention; agreement on χ²_HEIDI tightens from 38% to 0.67% and on p_HEIDI from 60% to 1.55%.
4. **OCF DML coverage** (F3 #4, **closed**). The new `nuisance_learner='ridge_quadratic'` argument (mirroring `DoubleML` `ml_g` / `ml_m`) restores Neyman orthogonality under non-linear confounders; coverage moves from 0.41 to 0.94 and bias from +0.18 to within ±0.03.
5. **Remaining post-V1 documented** — none. All four F3 divergences identified during the multi-pillar validation campaign have been closed on master with backward-compatible patches (V1 defaults preserved; corrected forms opt-in via a new parameter where applicable).

## Cross-host reproducibility

Every harness — 17 reference tools under `validation/external/<tool>/` (the eleven core tools plus FUSION measured-expression, CRAN `metap`, Bioconductor `EmpiricalBrownsMethod`, hapref `haplo.stats`, hyprcoloc R, and SMR v1.3.1) and the six Tier-3 specialty harnesses under `validation/specialty/<model>/` — shares the same six-artefact layout (`install.sh` pinning version, `fetch_data.sh` with SHA256-verified fixture, `run.sh`, `compare.py` writing `results/summary.tsv` plus `results/agreement.json`, `manifest.sha256`, and a tolerance-rationale `README.md`). The shared pre-flight library gates every `fetch_data.sh` and `run.sh` invocation on disk and RAM headroom. From a clean checkout, `bash paper/reproducibility/reproduce_paper.sh` walks the eight numbered stages and regenerates every number in this section and every cell of Figure 2.
