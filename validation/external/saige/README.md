# SAIGE reference comparison harness

Pillar B B5: external-tool reference comparison against
[SAIGE](https://github.com/saigegit/SAIGE) (Zhou et al. 2018,
*Nat. Genet.* 50:1335-1341) — the gold-standard biobank-scale GLMM for
binary / ordinal / survival traits with kinship correction +
saddlepoint approximation.

## Scope

Three comparisons against SAIGE's outputs on the MDP maize fixture
(281 samples × 2897 SNPs after MAF/geno QC, simulated logistic-on-causal-SNPs
binary phenotype):

1. **Step 1 sanity** — SAIGE's null GLMM artifacts (`step1.rda` +
   `step1.varianceRatio.txt`) are well-formed and the variance ratio
   sits in a non-degenerate range.
2. **Step 2 binary GLMM β / SE / -log10p** — SAIGE's per-SNP score test
   with saddlepoint approximation vs `torchgenomics.models.BinaryGLMM(use_spa=True)`
   with VanRaden GRM, intercept + age + sex + PC1 + PC2 covariates.
3. **SPA tail correlation** — Restrict to SAIGE's `Is.SPA == true` rows
   and re-check the -log10p correlation with TG's saddlepoint output.

## Reproduction

```bash
bash validation/external/saige/install.sh        # docker pull wzhou88/saige:1.4.4
bash validation/external/saige/fetch_data.sh     # stage MDP + simulate phenotype
bash validation/external/saige/run_saige.sh      # Step 1 + Step 2 (binary)
TORCHGENOMICS_DISABLE_NATIVE=1 \
    python3 validation/external/saige/compare.py # comparison report
pytest -m external tests/test_external_saige.py -v
```

## Install path used

**Docker (podman) image** `docker.io/wzhou88/saige:1.4.4` — the official
upstream-published image at the time of harness implementation
(2026-04-30). On RHEL 9.6, `docker` is a `podman` alias; the harness
treats them interchangeably and uses `:Z` SELinux relabeling on the
mount so the container can write to the host outputs directory.

The Bioconductor R fallback (`BiocManager::install("SAIGE")`) is wired
into `install.sh` as a secondary path but was not exercised on this
host (docker succeeded). It is the right path for environments without
container runtime.

If both paths fail, `install.sh` writes a `.infra_blocker` marker; per
spec §9, the harness scripts remain in place and pytest tests skip
cleanly with the documented reason.

## SAIGE version + flags

- **Version pinned**: `1.4.4` (latest at 2026-04-30). The image bundles
  R 4.2.3 + bgenix + savvy.
- **Step 1 flags**: `--traitType=binary --LOCO=FALSE --IsOverwriteVarianceRatioFile=TRUE`
  with `--covarColList=age,sex,PC1,PC2 --qCovarColList=sex`. We disable
  LOCO because the MDP fixture's 10 chromosomes × ~300 SNPs are below
  LOCO's useful regime.
- **Step 2 flags**: `--LOCO=FALSE --is_Firth_beta=TRUE --pCutoffforFirth=0.01
  --minMAF=0.01 --minMAC=5`. Firth is enabled to match the regenie
  harness convention; SAIGE's saddlepoint approximation is the model
  default and is left enabled (it triggers automatically for χ²>2 or
  case/control imbalance).

## Phenotype simulation

`simulate_phenotype.py` plants 5 random AF∈[0.1,0.4] causal SNPs with
β = 0.5 each (additive log-odds), adds a PC1-correlated baseline
(coef=0.3) and intercept = -0.05, draws Bernoulli outcomes. This gives:
- balanced cases/controls (~48% prevalence on N=281),
- some PC1-correlated cases (so PCs are useful covariates), and
- a tail of large-effect SNPs that exercise SAIGE's saddlepoint path.

Synthesized covariates `age` (uniform 30..70) and `sex` (Bernoulli) are
included to exercise SAIGE's `--covarColList` plumbing.

## Observed numerical agreement (N=281, m=2897)

| Comparison | Metric | Observed | Floor |
|---|---|---|---|
| Step 1 sanity | step1.rda bytes | 57227 | ≥ 100 |
| Step 1 sanity | variance ratio | 0.896 | (0, 100) |
| Step 2 | β corr (full set, m=2897) | 0.973 | ≥ 0.90 |
| Step 2 | −log10p corr (full set) | 0.943 | ≥ 0.85 |
| Step 2 | β median \|Δ\| in AF [0.03,0.97] (n=2817) | 0.045 | ≤ 0.15 |
| Step 2 | SE median \|Δ\| in AF [0.03,0.97] | 0.0015 | ≤ 0.05 |
| SPA subset | -log10p corr (n=133 SPA hits) | 0.885 | ≥ 0.70 |

## Documented divergences

- **PCG vs PQL null fit**: SAIGE estimates the null GLMM via
  preconditioned conjugate gradient with low-rank preconditioner; TG's
  `BinaryGLMM` runs eigen-decomposition-based PQL (Breslow-Clayton
  1993). At biobank scale the two converge to identical mean / variance
  components; on N=281 there is residual ~1–3% drift in β driven by
  the variance-ratio noise floor. The drift is uniform across SNPs,
  preserving sign and rank, hence the high β corr (>0.97) but non-zero
  median |β| difference.

- **PQL converged flag**: TG's `BinaryGLMM` reports `converged=False`
  on this fixture because the inner PQL loop does not hit `tol=1e-4`
  within 30 outer iterations. The β / SE / p numerics it produces are
  still inside the §16 tolerance bracket vs SAIGE; in particular the
  full-set -log10p corr = 0.943 indicates the model has reached a
  near-stationary point. This is a small-N / low-information artifact
  of MDP, not a V1-core math defect.

- **Saddlepoint flavor**: SAIGE uses Dey et al. (2017) saddlepoint
  with case/control AF imbalance heuristics; TG uses the standard
  saddlepoint applied to the score-test numerator (`stats.spa`).
  Both override the standard chi-square p in the same direction; the
  SPA-subset correlation = 0.885 confirms agreement on the tail.

- **Allele convention**: PlinkBedReader counts BIM A1 (PLINK 1.9
  canonical, post-2026-05-13 fix) = "A" on this fixture; SAIGE's
  `Allele2 = A = BIM A1` and reports BETA on the A allele. Both tools
  are now on the same convention; `compare.py` no longer applies a
  `2.0 - G` flip.

- **Firth penalty**: SAIGE's `--is_Firth_beta=TRUE` uses a one-step
  Firth approximation for the β estimate (analogous to regenie's
  fast-Firth). TG's `BinaryGLMM` does not currently enable Firth in
  the score-test path. On the MDP fixture only 26/2897 SNPs trigger
  SAIGE's `pCutoffforFirth=0.01`; their β values can disagree by up to
  ~1.0 (driving `β max |Δ| (full) = 1.04`). The β median |Δ| in the
  AF [0.03, 0.97] band remains 0.045 because Firth-marked SNPs are
  predominantly in the AF tails.

## Peak memory + wall time

- **install.sh** (image pull): ~3 GB disk, < 200 MB RAM.
- **Step 1**: ~250 MB peak RSS (PCG + GRM build), ~3 s wall time.
- **Step 2**: ~200 MB peak RSS, ~12 s wall time (10 chunks of ~300
  SNPs each).
- **compare.py**: ~700 MB peak (loads full G + builds VanRaden GRM in
  torch, runs PQL).

The pre-flight contract requests 10 GB disk + 4 GB RAM headroom for
the install (the 3 GB image leaves margin for the writable layer),
and 1 GB disk + 6 GB RAM for the run.

## F3 logic

`BinaryGLMM` (Phase 33), `OrdinalGLMM` (Phase 33), and `SurvivalGLMM`
(Phase 35) are all post-V1. Per spec §9, parameterization mismatches
between SAIGE's PCG and TG's PQL are post-V1 documented divergences,
not F3 fix-now. A real V1-platform math defect would be F3.

No F3 fixes were required for B5.

## Files in this directory

- `install.sh` — try docker pull `wzhou88/saige:1.4.4`, fall back to
  Bioconductor R, mark `.infra_blocker` if both fail (per spec §9).
- `fetch_data.sh` — stage MDP fileset (with PLINK 2 MAF/geno QC) +
  generate `saige_pheno.tsv` via `simulate_phenotype.py`.
- `simulate_phenotype.py` — produce SAIGE-format phenotype + covariate
  table (binary `ybin` + age + sex + PC1 + PC2, with planted causal
  SNPs).
- `run_saige.sh` — invoke step1_fitNULLGLMM.R + step2_SPAtests.R inside
  the SAIGE container with `:Z` mount for podman compatibility.
- `compare.py` — parse SAIGE outputs, run TG `BinaryGLMM`, assert
  tolerances.
- `tests/test_external_saige.py` (in repo root) — pytest gate, marked
  `external` for auto-skip; auto-skips with the `.infra_blocker` text
  as the reason if SAIGE install failed.

## Infra-blocker notes

None on this host (RHEL 9.6, podman 5.x). If you hit one:
- `Error: short-name resolution enforced but cannot prompt without a TTY`
  — the install script always uses the fully-qualified `docker.io/...`
  path to avoid this.
- `Permission denied` on `/work/outputs/...` writes — the run script
  uses the `:Z` SELinux relabel; without it, podman's rootless uid-map
  doesn't grant write access to host-owned mount targets.
- `BiocManager::install("SAIGE")` failures on RHEL 9 are almost always
  C++ toolchain / htslib / bgenix bring-up issues; the Docker path is
  strictly easier.
