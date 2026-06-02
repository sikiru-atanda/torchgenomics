# BOLT-LMM reference comparison harness

Pillar B B6: external-tool reference comparison against
[BOLT-LMM](https://alkesgroup.broadinstitute.org/BOLT-LMM/) (Loh et al.
2015, *Nat. Genet.* 47:284-290; Loh et al. 2018, *Nat. Genet.* 50:906-908)
— a fast LMM with whole-genome ridge-regression-based predictor + LOCO,
historically the dominant biobank-scale LMM until regenie / SAIGE.

## Scope

One comparison against BOLT's ``--lmmInfOnly`` output on the MDP maize
fixture (281 samples × 2897 SNPs after MAF/geno QC):

1. **Per-SNP β / SE / -log10p (LMM-Inf, internal LOCO)** —
   ``bolt_stats.tsv`` (`P_BOLT_LMM_INF`) vs
   ``torchgenomics.models.SingleTraitLMM`` + ``torchgenomics.linalg.grm_loco``,
   covariates = intercept + PC1 + PC2.

We only compare against ``--lmmInfOnly`` (the textbook infinitesimal
LMM). The full ``--lmm`` / ``--lmmForceNonInf`` "BoltLMM-mixed" mode
applies a Gaussian-mixture spread approximation specific to BOLT and
requires LD-scores tail calibration that ships only for hg19 / hg38
human assemblies; comparing TG to that would be a methods comparison,
not an equivalence check.

## Reproduction

```bash
bash validation/external/bolt_lmm/install.sh        # download v2.5 + smoke-test
bash validation/external/bolt_lmm/fetch_data.sh     # stage MDP + simulate phenotypes
bash validation/external/bolt_lmm/run_bolt.sh       # --lmmInfOnly run
TORCHGENOMICS_DISABLE_NATIVE=1 \
    python3 validation/external/bolt_lmm/compare.py # comparison report
pytest -m external tests/test_external_bolt_lmm.py -v
```

## Install path used

- **BOLT-LMM v2.5** (Aug 2025), pinned by URL +
  ``BOLT-LMM_v2.5.tar.gz`` SHA-256 verification.
- Closed precompiled Linux x86_64 binary; **no source-build path**. The
  upstream is the only redistribution point.
- Tarball size: ~211 MB. Expanded distribution: ~1 GB (bundles
  ``tables/LDSCORE.*`` and ``example/`` data we don't use).

### Infra-blocker fallback

The v2.5 binary is linked against the libstdc++ / glibc shipped with
Ubuntu 22+ (glibc ≥ 2.35). RHEL 9.6 ships glibc 2.34, and on some hosts
the binary fails to load with ``GLIBCXX_*`` or ``GLIBC_*`` errors.

The install script handles this gracefully:

1. Tries to load the binary with ``bolt --help``.
2. On failure, captures the loader's error message into
   ``.infra_blocker`` and exits 0.
3. ``run_bolt.sh`` and ``compare.py`` short-circuit when the blocker
   exists.
4. ``tests/test_external_bolt_lmm.py`` skips cleanly with the file's
   text as the skip reason (per the SAIGE pattern).

To retry after a host upgrade or a libstdc++ install:

```bash
rm validation/external/bolt_lmm/.infra_blocker
bash validation/external/bolt_lmm/install.sh
```

## Observed numerical agreement (N=279, m=2897, BOLT-LMM v2.5)

Calibrated on first successful BOLT-LMM run (2026-04-30, MDP fixture,
N=279 after pheno-NaN drop, m=2897 post-MAF/geno QC). All four checks
pass; tolerances are observed-then-floored with documented headroom for
CPU/BLAS-precision drift + BOLT's stochastic variance-component init.

| Metric | Observed | Floor (gate) | Headroom |
|---|---|---|---|
| β corr (full) | 0.9843 | ≥ 0.95 | ~0.034 |
| -log10p corr (full) | 0.9823 | ≥ 0.95 | ~0.032 |
| β median \|Δ\| in AF [0.03,0.97], n=2814 | 0.148 | ≤ 0.20 | 35% |
| SE median \|Δ\| in AF [0.03,0.97], n=2814 | 0.0426 | ≤ 0.07 | 60% |

Extras:
- Mean -log10(P): BOLT = 0.478, TG = 0.462 (mean-aligned to 4%).
- β max |Δ| (full): 3.79 — single tail outlier on the strongest-effect
  SNP, explained by BOLT's per-block calibration ratio (1.35±0.022 on
  this fixture, which BOLT itself flags as high-stderr at N=279 and
  reverts to "ratio of medians"); TG uses a per-chrom REML refit instead.

If the host is infra-blocked on BOLT install, this section reflects an
"N/A — infra-blocked" outcome and the harness scaffold is exercised by
``test_external_bolt_lmm.py`` skipping with the captured loader error.

## Documented divergences

- **BOLT-LMM is human-genetics-focused**. The shipped LD-scores file
  (``bin/tables/LDSCORE.1000G_EUR.tab.gz``) is hg19-only; chromosome
  IDs and LD patterns don't match MDP maize. ``--lmmInfOnly`` does
  not need LD-scores, so we skip them. The full ``--lmm`` mode is
  out of scope.

- **--lmmInfOnly vs --lmm**. ``--lmmInfOnly`` is the textbook
  infinitesimal LMM equivalent to GEMMA + LOCO — what TG's
  ``SingleTraitLMM`` + ``grm_loco`` implements. The full ``--lmm``
  applies a Gaussian-mixture spread approximation (Loh 2015 §Methods)
  that's BOLT-specific.

- **LOCO algorithm**. BOLT performs LOCO internally inside ``--lmmInf``
  by residualizing the test SNP's chromosome out of the GRM-based
  predictor; TG performs LOCO by building a per-chrom GRM with
  ``grm_loco`` and refitting the null per chromosome. Both are
  textbook-correct; a small numerical gap is expected from the
  per-block variance-component refit.

- **Variance-component estimation**. BOLT estimates the LMM
  variance-component ratio via stochastic Monte-Carlo + AI-REML; TG
  uses GAPIT-EMMA-REML by default. On N=281 these converge to similar
  σ²_g / σ²_e but not bit-for-bit.

- **Allele convention**. PlinkBedReader counts BIM A1 (PLINK 1.9
  canonical, post-2026-05-13 fix); BOLT counts ALLELE1 = BIM A1. Both
  tools are now on the same convention; `compare.py` no longer needs
  a manual `2.0 - G` flip.

## Peak memory + wall time

Observed on first run (RHEL 9.6, glibc 2.34, BOLT-LMM v2.5 binary loaded
successfully — no infra-blocker triggered):

- **install.sh**: ~211 MB tarball; expanded ~1 GB (bundles unused
  ``tables/LDSCORE.*`` + ``example/`` data); ~30 s wall (download-bound).
- **fetch_data.sh**: < 50 MB peak RSS, < 2 s.
- **run_bolt.sh** (--lmmInfOnly, N=279, m=2897): ~150 MB peak RSS,
  ~6 s wall (BOLT-LMM end-to-end including REML + LOCO + per-SNP scan).
- **compare.py**: ~120 MB peak (loads full G + LOCO scan over 10
  chromosomes), ~3 s wall.

The pre-flight contract (validation/external/_lib/preflight.sh) requests
5 GB disk + 1 GB RAM headroom before download, 1 GB disk + 1 GB RAM
before fetch, and 1 GB disk + 4 GB RAM before run — far above what's
actually consumed, but consistent with the harness convention.

## Files in this directory

- ``install.sh`` — download + verify BOLT-LMM v2.5 tarball; smoke-test
  the binary; mark ``.infra_blocker`` on loader failure.
- ``fetch_data.sh`` — stage MDP fileset (with PLINK 2 MAF/geno QC) +
  generate ``bolt_pheno.tsv`` + ``bolt_covar.tsv``.
- ``simulate_phenotype.py`` — produce BOLT-format phenotype + covariate
  tables from MDP (continuous EarHT, 2 PCs from genotype SVD).
- ``run_bolt.sh`` — run BOLT ``--lmmInfOnly``.
- ``compare.py`` — parse BOLT output, run TG SingleTraitLMM + LOCO,
  assert tolerances.
- ``tests/test_external_bolt_lmm.py`` (in repo root) — pytest gate,
  marked ``external`` for auto-skip; skips with ``.infra_blocker`` text
  as reason if install was blocked.
