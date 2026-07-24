# Phase 57 — Tractor-Mix: GWAS for admixed cohorts with relatedness

**Status:** Design approved 2026-07-24. Not yet implemented.
**Branch:** `feat/phase57-tractor-mix` (off `master` `f8986e3`).
**Author flow:** brainstormed with user 2026-07-24; this spec is the approved design.

## 1. Motivation

Admixed populations are the fastest-growing human demographic yet are
under-served by GWAS tooling, largely because handling **admixture and
relatedness simultaneously** is hard. Tan et al. (*Nature Genetics* 2026,
doi:10.1038/s41588-026-02689-6) introduce **Tractor-Mix**, which extends the
local-ancestry-informed Tractor model into a generalized linear mixed model
(GMMAT-style) so that admixed cohorts *with* relatedness get well-calibrated,
ancestry-specific effect sizes and p-values.

TorchGenomics has **no** admixed / local-ancestry GWAS capability today
(verified 2026-07-24: no PC-AiR/PC-Relate/local-ancestry code in `linalg` or
`models`). However, ~70% of the required machinery already exists — sparse GRM
+ REML (`optim/sparse_reml.py`), PQL GLMM null + score test + SPA
(`models/binary_glmm.py`, `stats/spa.py`), COJO-style conditional analysis
(`models/conditional_lmm.py`), and streaming GRM (`linalg/kinship.py`). This
phase fills the gap by reusing that machinery.

## 2. Goal, scope, and the provable claim

**Primary goal (user-selected):** a GPU-accelerated, **reference-equivalent**
re-implementation of Tractor-Mix, with a benchmarked speedup vs. the R original.
This is an **engineering + platform-integration** contribution (faithful port),
not a new-statistics contribution. Novelty (opt-in SPA for low-count ancestry
dosages, which the paper defers) is exposed but is *not* part of the equivalence
claim, and is the seed for a later novelty phase.

**The headline claim, scoped so it is scientifically provable:**

> A GPU-accelerated, reference-equivalent implementation of the Tractor-Mix
> association + inference step, with validated calibration and a benchmarked
> speedup — proven **primarily on synthetic data** (the paper's own simulation
> recipe, where we own the ground truth), with real-data top-hit concordance as
> a secondary check.

Equivalence is claimed **conditional on identical inputs** (ancestry dosages,
covariates, GRM, phenotype); it is *not* end-to-end bitwise equivalence, because
upstream phasing/LAI are stochastic.

**In scope:** ancestry-specific mixed-model association (continuous + binary),
2-df joint Rao score test, ancestry-specific score + Wald effect sizes,
allele-count threshold, opt-in SPA, local-ancestry conditional analysis; native
PC-AiR/PC-Relate admixture-aware GRM/PCs; external-GRM/PCs input path; streaming
scan; CLI + R + MCP surface; validation harness + benchmark.

**Out of scope (upstream / external):** phasing (SHAPEIT5), local-ancestry
inference (RFMix2), and Tractor `ExtractTracts` dosage extraction. We *consume*
ancestry-partitioned dosages; we ship a reader for Tractor's output format but
not LAI itself. **Biobank-scale empirical validation is deferred to NA3**
(needs UKB access); this phase proves equivalence + calibration on synthetic +
moderate scale (n ≤ ~10K) and the paper's public summary statistics.

## 3. Architecture — three independently testable units

| Unit | Location | Responsibility | Reuses |
|---|---|---|---|
| **A. Admixture-aware GRM/PCs** | `linalg/kinship_admixed.py` | KING-robust → LD-prune → PC-AiR → PC-Relate; returns relatedness-robust PCs + sparse GRM | `sparse_grm.py`, `eigh.py`, `ld/` pruning, `kinship.py` streaming |
| **B. Tractor-Mix model** | `models/tractor_lmm.py` | `fit_null` (LMM/GLMM null w/ GRM) + `score_chunk` (K-way ancestry dosages → 2-df joint Rao score + ancestry-specific score/Wald + SE); allele-count threshold; opt-in SPA; local-ancestry conditional | `single_trait_lmm`, `binary_glmm`, `stats/spa.py`, `conditional_lmm`, `sparse_reml.py` |
| **C. Scan / CLI / R / MCP surface** | `scan` adapter, `cli.py`, `api/`, `rTorchGenomics/R/` | streaming scan; `tractor-scan` CLI; `api.tractor_scan`; `tg_tractor_scan` R wrapper; MCP tool | `UnifiedScanner`, existing CLI / api / bridge patterns |

**Interfaces / boundaries:**
- Unit A is standalone and reusable by any future admixed model; validated
  independently against GENESIS.
- Unit B consumes `{g_a}` ancestry dosages, X, Φ, PCs, y; never performs LAI.
- Unit B accepts Φ/PCs either externally supplied or computed by Unit A.

## 4. Unit B — the statistical model (`models/tractor_lmm.py`)

**Model.** For individual `i` with K-way local ancestry:

```
h(mu_i) = X_i alpha + sum_{a=1..K} g_{ia} beta_a + b_i,   b_i ~ MVN(0, tau^2 * Phi)
```

`g_{ia}` = ancestry-a genotype dosage; X = covariates (intercept, age, sex, PCs);
Phi = GRM. Identity link → continuous (LMM); logit → binary (GLMM).

**Null fit (`fit_null`)** — reuse existing backends, no new optimizer:
- Continuous: `single_trait_lmm` REML null → `mu0_hat, tau2_hat, Sigma_hat`.
- Binary: `binary_glmm` PQL null (already GMMAT/SAIGE-style) → `mu0_hat, W_hat, Sigma_hat`.

**Joint K-df Rao score test (`score_chunk`, default).** With `G = [g_A, g_B, ...]` (n×K):

```
T        = G^T (y - mu0_hat)                                    # (K,)
P_hat    = Sigma^-1 - Sigma^-1 X (X^T Sigma^-1 X)^-1 X^T Sigma^-1
Var(T)   = G^T P_hat G                                          # (K,K)
stat     = T^T Var(T)^-1 T   ~  chi^2_K   under H0: beta_1=...=beta_K=0
```

(For 2-way admixture, K=2 → the paper's 2-d.o.f. joint test.)

**Ancestry-specific effects (score approximation — scalable, no full-model fit):**

```
beta_hat = Var(T)^-1 T
SE       = sqrt(diag(Var(T)^-1))          → per-ancestry p-values
```

**Wald option (`test="wald"`):** fit the full model per variant (adapting the
GMMAT approach), `beta_a^2 / Var(beta_a) ~ chi^2_1`. More robust effect sizes,
slower; the paper reports both and notes Score is more significant than Wald for
dichotomous traits and less for continuous, with a small GMMAT-like downward
bias — behavior we must reproduce.

**Low allele-count handling (faithful default).** Per-ancestry allele-count
threshold (default **50**, matching the paper). Ancestry terms below it are
**dropped**; the joint p is computed only if all modeled ancestries pass,
otherwise marginal p-values for the passing ancestries.

**SPA — opt-in, default OFF** (`use_spa=False`). The paper defers saddlepoint
correction, so the default path stays reference-faithful. When enabled, reuse
`stats/spa.py` on the per-ancestry score component below `spa_threshold`.
Documented as an enhancement for rare-ancestry low counts; not part of the
equivalence claim. This is the hook for the later novelty phase.

**Local-ancestry conditional analysis** (SAIGE-GENE-style, no per-segment null
refit). With local-ancestry score `T_L`:

```
E[T_G | T_L]   = G^T P_hat L (L^T P_hat L)^-1 T_L
Var[T_G | T_L] = G^T P_hat G - G^T P_hat L (L^T P_hat L)^-1 L^T P_hat G
cond_stat      = (T_G - E[T_G|T_L])^T Var[T_G|T_L]^-1 (T_G - E[T_G|T_L]) ~ chi^2_K
```

Uses K−1 local-ancestry terms (one is redundant with the total).

## 5. Unit A — admixture-aware GRM/PCs (`linalg/kinship_admixed.py`)

The paper's exact chain (all GPU torch, each a pure function, validated vs GENESIS):

1. **`king_robust_kinship(G)`** — Manichaikul et al. 2010 robust pairwise
   kinship from IBS/heterozygosity counts; ancestry-robust, no allele-freq
   assumption. Streams over marker chunks. Output: sparse pairwise kinship +
   relatedness graph.
2. **`ld_prune(G, r2=0.1)`** — reuse `torchgenomics.ld` pairwise-r² pruning to an
   independent SNP set (paper's r² < 0.1). Feeds steps 3–4.
3. **`pc_air(G_pruned, king_kinship)`** — Conomos et al. 2015. Greedy partition
   into an ancestry-representative **unrelated** set + a **related** set (KING
   kinship + ancestry divergence), eigendecompose the unrelated-set GRM (reuse
   `eigh.py`), project the related set → relatedness-robust PCs.
4. **`pc_relate(G_pruned, pcs)`** — Conomos et al. 2016. Regress genotypes on
   PC-AiR PCs → individual-specific allele frequencies → moment-based recent-
   kinship estimator → GRM. Post-process per paper: rescale ×2, diagonal ≈ 1,
   zero elements < 0.05 → sparse GRM (feeds `sparse_grm.py` / `sparse_reml.py`).

Returns `(pcs, sparse_grm)`. Standard `grm_vanraden` stays available as a
documented non-admixed alternative. **When markers are given but no GRM, the
default is the PC-Relate (faithful) GRM**; `grm_vanraden` is explicit opt-in.

## 6. Data flow & I/O contract

```
              +- external GRM+PCs supplied ------------------------------+
markers (BED/ |                                                          v
VCF, pruned) -+- Unit A: KING -> LD-prune -> PC-AiR -> PC-Relate -> PCs + sparse GRM -+
              +------------------------------------------------------------------------+
ancestry dosages {g_A,g_B,...} (Tractor ExtractTracts: anc0/anc1.dosage.txt)          |
local ancestry L  (optional, for conditional)                                          v
phenotype y, covariates X --------------------------> fit_null (LMM/GLMM) -> score_chunk (streamed)
                                                                                       |
                                                            joint p, beta_a, SE_a, Wald/cond -> ScanResult
```

**Inputs:**
- **Ancestry dosages:** one dense matrix per ancestry, variants × samples;
  native reader for Tractor's `anc{0,1}.dosage.txt`, plus `.npy` / `.pt`. Thin
  adapter documents expected shape/order. **Always required** (LAI is upstream).
- **Local ancestry `L`:** optional, same shape, for conditional analysis.
- **GRM/PCs:** supplied (`--grm`, `--pcs`) or computed by Unit A (`--compute-grm`).

**Streaming & device (hard-rule compliance):**
- Sample-space objects (GRM n×n sparse, PCs n×p) computed once; **marker-space
  is streamed** — KING, LD-prune, PC-Relate, and `score_chunk` all stream marker
  chunks (mirroring `grm_vanraden_streaming`). Never materialize full G / dosages.
- Paired regression nets: `tests/test_streaming_memory.py` entry +
  `bench/native_speedups.py` wall-time gate for the new paths.
- All-torch, on-device; obey `_dispatch.select_path`; no proactive CUDA→host
  copies (the class of bug the sanity scan + NA3 caught).

## 7. Unit C — user-facing surface

**Python API (primary):** `torchgenomics.api.tractor_scan(...)` →
`TractorScanResult`, following the `tg.lmm_scan` one-call pattern (format
auto-detect, sample alignment, auto-named output). Registered as the **15th MCP
tool** (`tg_tractor_scan`).

**CLI — new `tractor-scan` subcommand (bumps 43 → 44):**

```bash
torchgenomics tractor-scan \
  --dosage-prefix out/anc            # reads anc0.dosage.txt, anc1.dosage.txt (K-way: anc0..ancK-1)
  --phenotype pheno.txt --covariates covs.tsv \
  --family gaussian|binary \
  --grm K.npy --pcs pcs.tsv   |   --compute-grm --markers data.bed \
  --test score|wald            # default score \
  --min-allele-count 50 \
  --use-spa/--no-use-spa       # default OFF (faithful) \
  --local-ancestry L.npy --conditional   # optional \
  --correction bh --device cuda --chunk-size N --output results
```

Streams via `UnifiedScanner`.

**R wrapper (secondary, reticulate bridge):** `tg_tractor_scan(dosage_prefix=,
phenotype=, family=, test=, grm=, pcs=, ...)` → S4 `TractorScanResult` with
`top_hits`, `to_dataframe()`, `manhattan()`, `qq()`, `output_files` — same
`bridge_call` / `tg_py` pattern. Added to `NAMESPACE` + `_pkgdown.yml`.

**Result / output schema.** `TractorScanResult` (extends `ScanResult`): per
variant → `joint_p`, and per ancestry `beta_a, se_a, p_a, allele_count_a`;
optional `conditional_joint_p`; `n`. Manhattan panels for joint + each ancestry
(mirroring the paper's Fig 3/4). TSV + Parquet writers reused.

## 8. Validation strategy

Three tiers, **observed-then-floored tolerances** (set after seeing the numbers,
never aspirational), every divergence → `docs/validation_findings.md` with F3
classification.

| # | Claim | Method | Provability |
|---|---|---|---|
| 1 | **Association-step equivalence** (headline) | Paper's PSD-admixture + gene-dropping-pedigree sim (primary-source recipe) → known ancestry dosages. Identical inputs (G, X, Phi, y) to R Tractor-Mix and our port → beta_a, SE_a, joint p match to N sig figs. | Strong (same class as GEMMA gate) |
| 2 | **Calibration** | Null sims: lambda_GC + empirical type-I at alpha = 5e-2, 5e-4 (5e-6 with true GRM), continuous + dichotomous, admixture 50/70/90% AFR. Reproduce their Fig 1 controlled lambda_GC. | Strong |
| 3 | **Unit A vs GENESIS** (separate gate) | Shared dataset (1000G AFR-EUR admixed subset or sim): PC-AiR PCs match GENESIS (Procrustes / abs-corr); PC-Relate kinship matches to tolerance. | Strong |
| 4 | **Real-data concordance** (secondary) | Public Zenodo sumstats — recover known hits (APOE rs7412 cholesterol, HBB rs334 sickle cell, ZNF646P1 BMI) + ancestry-specific sign/magnitude. Not bitwise (stochastic LAI) → rank/sign/correlation concordance. | Moderate (honest caveat) |
| 5 | **Speed/memory benchmark** | Wall-time + peak RAM head-to-head vs R Tractor-Mix, same data & hardware: R-on-CPU vs our-CPU AND our-GPU (no cherry-picking). | Strong, with fair-comparison discipline |

**External harness** `validation/external/tractor_mix/` (existing 11-tool pattern):
`install.sh` (R + GENESIS + Tractor-Mix from GitHub), `fetch_data.sh` (Zenodo
`10.5281/zenodo.19600120` + sim generator), `run_tractormix.sh`, `compare.py`.
**Sources `validation/external/_lib/preflight.sh`** — asserts disk + RAM
headroom before any download/run (hard rule; abort, never partial-execute).
Agents author files via heredoc / `Rscript writeLines` (subagent Write is
unreliable). Head-to-head must be an **actual run** of installed R Tractor-Mix —
no paper-number extrapolation.

## 9. Conventions

- Reviewer-grade docstrings with primary-source citations (Tan 2026; Conomos
  2015, 2016; Manichaikul 2010; Chen/GMMAT 2016). Novelty (opt-in SPA) qualified
  "to our knowledge."
- Update `CLAUDE.md` (Phase Index, CLI section, module list); add a `tractor-scan`
  vignette to `rTorchGenomics`.
- Ships as **Phase 57** — labeled commit series ("Phase 57 …"), non-destructive
  on this feature branch, **no autonomous push** (user-gated each push).
- Paired memory + wall-time regression nets for the new streaming paths.
- Python-as-spec / native-as-shortcut preserved for any future C++ accelerator.

## 10. Risks & open items

- **PC-AiR/PC-Relate fidelity** is the main cost/risk. It is a genuine
  reimplementation of a specific greedy partition (PC-AiR) and moment estimator
  (PC-Relate); its own GENESIS-equivalence gate (Claim 3) de-risks it. If
  equivalence proves hard within budget, fall back to the external-GRM-input
  path for the headline claim while iterating on the native chain.
- **Getting matching inputs.** The Zenodo release is final sumstats + code, not
  intermediate dosages/GRM — hence the synthetic-first strategy (we own the
  inputs). Real data is secondary concordance only.
- **Wald full-model fit** cost per variant; keep it opt-in, score is the default
  scan path.
- **K > 2 ancestries.** Model generalizes; validate on 2-way and 3-way. Paper
  advises limiting to ancestries with >= ~10% global contribution.

## 11. References

- Tan, T. et al. Extending genome-wide association studies to admixed cohorts
  with high degrees of relatedness. *Nat. Genet.* (2026).
  doi:10.1038/s41588-026-02689-6. Code: github.com/Atkinson-Lab/Tractor-Mix.
  Data: Zenodo 10.5281/zenodo.19600120 (sumstats), 19576963 (analysis code).
- Atkinson, E. G. et al. Tractor. *Nat. Genet.* 53, 195–204 (2021).
- Chen, H. et al. GMMAT. *Am. J. Hum. Genet.* 98, 653–666 (2016).
- Conomos, M. P. et al. PC-AiR. *Genet. Epidemiol.* 39, 276–293 (2015).
- Conomos, M. P. et al. PC-Relate. *Am. J. Hum. Genet.* 98, 127–148 (2016).
- Manichaikul, A. et al. KING-robust. *Bioinformatics* 26, 2867–2873 (2010).
- Gogarten, S. M. et al. GENESIS. *Bioinformatics* 35, 5346–5348 (2019).
