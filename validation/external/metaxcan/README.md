# MetaXcan / S-PrediXcan harness (Tier 1 A1)

Reference-tool comparison between **MetaXcan** (`software/SPrediXcan.py`,
the canonical S-PrediXcan implementation) and **TorchGenomics**
`torchgenomics.postgwas.twas_sumstat`. Part of the Genome Biology Methods
paper's Tier 1 Pillar B harness suite.

## Pinned tool version

| Field | Value |
|-------|-------|
| Repo  | https://github.com/hakyimlab/MetaXcan |
| Tag   | `v0.8.1` |
| Commit SHA | `964f1fdb5bf9585585690e85bb0eca7b67663ddb` |
| Release date | 2025-06-23 |
| Resolved via | GitHub Releases API on 2026-05-15 |

The tag `v0.8.1` is the latest stable MetaXcan release as of harness
authorship. The spec minimum is `v0.7.5`; v0.8.1 implements the same
S-PrediXcan estimator and adds "minor updates to the var-control method"
(upstream release notes) which do not affect the sumstats-path math.

Internal `__version__` in `metax/__init__.py` still reports `0.7.5` —
this is an upstream-known issue (not bumped at every tag). The commit
SHA is the authoritative identifier.

## TorchGenomics comparison target

`torchgenomics/postgwas/_twas.py` — `twas_sumstat` (S-PrediXcan-equivalent
sumstats path). The harness does NOT exercise `twas_individual` because
that is the PrediXcan (individual-level) entry, validated separately.

## Fixture

A deterministic simulated PrediXcan-compatible triple (model + covariance
+ GWAS sumstats), seed=42:

- `data/model.db` — SQLite with the canonical PrediXcan schema
  (`weights(rsid, gene, weight, ref_allele, eff_allele)` +
  `extra(gene, genename, "n.snps.in.model", "pred.perf.R2", "pred.perf.pval", "pred.perf.qval")`).
- `data/model.txt.gz` — covariance file in MetaXcan format
  (header `GENE RSID1 RSID2 VALUE`, whitespace-delimited).
- `data/gwas_sumstats.txt.gz` — GWAS sumstats (SNP / effect_allele /
  non_effect_allele / zscore / beta / se / pvalue).
- `data/sim_truth.json` — seed, per-gene true beta, per-gene weights,
  LD matrices, and SHA256 of every fixture file.

Fixture parameters:

| Param | Value |
|-------|-------|
| Seed | 42 |
| Genes | 5 |
| SNPs per gene | 8 (cis) |
| Background SNPs | 50 (in GWAS only, exercise SNP-intersection) |
| GWAS sample size | 10000 |
| LD structure | `Sigma[i,j] = rho^|i-j|`, rho=0.3 (diag = 1; correlation form) |
| Planted theta | 0 (genes 0-2, 4) and 0.25 (gene 3) |

### Fixture SHA256

```
c80780a513d915bd2813383ea959a837d6ad62e55a3776c25d6305bf387cff96  data/model.db
80726f6eecefaf536524450c2139593e06c2c48a1c5a44fe891beaa89aa9b7f8  data/model.txt.gz
de41898bb16bbbe22b7527edd0c97528b034439bc99e593cc4c4f7c934e8e2b7  data/gwas_sumstats.txt.gz
a521453eb7615c380e73a56be7823477d4b09c62808fd0cef8a48efd23dbe02e  data/sim_truth.json
```

### Why simulate (not download a GTEx PrediXcan model)

GTEx v8 PrediXcan model bundles (mashr or elastic-net) are 1-2 GB
tarballs whose hosting URL has shifted across PredictDB/Zenodo/Box
across releases. Network-staging them at every harness rerun adds
brittleness without information value: the S-PrediXcan estimator
(Barbeira 2018, Eq. 4) is purely summary-statistic, so the numerical
agreement check needs only the (weights, Sigma, z-scores) triple, not
the underlying biology. This is the same "simulated fixture"
strategy used by `validation/external/twosamplemr/` (the MR harness
plants a known causal theta into IV sumstats; the comparison still
tests every code path of TwoSampleMR's `mr()` and MRPRESSO's
`mr_presso()`).

If a future iteration requires a real GTEx model (e.g., for a
sensitivity check against a held-out real-world panel), wire it as an
*alternative* fixture in `fetch_data.sh` behind a `METAXCAN_USE_REAL_DATA=1`
env-var gate. The simulated path remains the default.

## Observed agreement (first successful run, 2026-05-15)

5 genes compared, MetaXcan v0.8.1 (commit `964f1fdb…`) vs TG
`twas_sumstat` on a Linux x86_64 host (Python 3.13.5).

| Metric | Observed | Floored tolerance | Status |
|--------|----------|-------------------|--------|
| max `|Δ z|` | 3.07e-08 | 1e-6 | PASS |
| max `|Δ effect_size|` | 9.37e-17 | 1e-6 | PASS |
| max `|Δ -log10 p|` | 1.56e-07 | 1e-3 | PASS |
| Pearson r (z, per-gene) | 1.0000 | 0.9999 | PASS |
| Pearson r (effect_size) | 1.0000 | (informational) | PASS |
| Pearson r (-log10 p) | 1.0000 | (informational) | PASS |

The observed numbers reflect FP-precision agreement: MetaXcan's
`numpy.sum(weights * zscore * sigma_l) / sqrt(sigma_g_2)` and TG's
`(w^T z) / sqrt(w^T Σ w)` collapse to the same closed-form sum when
the model covariance is in correlation form (`diag(Σ) = 1`), which is
the case for the simulated fixture. The 3e-8 z-score gap reflects only
the small difference in float64 summation order between numpy.sum and
torch.sum.

These are the canonical observed values; floors are set conservatively
to 1e-6 (z, effect_size) and 1e-3 (-log10 p) to absorb cross-host BLAS
precision drift. If a future run exceeds these floors, treat as an F3
divergence and route through `docs/validation_findings.md`.

## FUSION sibling decision

The plan optionally permits a sibling FUSION harness. **We did not
add one** for this iteration. Rationale:

- FUSION (Gusev et al. 2016, Nat Genet 48:245) and S-PrediXcan
  (Barbeira et al. 2018) use SNP-level weights derived by different
  algorithms (BSLMM/LASSO/top-1 elastic net for FUSION; elastic net /
  mashr for PrediXcan). They are not numerically interchangeable on a
  shared model.
- Adding FUSION would require a *separate* weight-fitting pipeline and
  a *separate* covariance reference panel. Per the plan brief, this
  belongs in a sibling `validation/external/fusion/` directory rather
  than this one.
- The TorchGenomics `twas_sumstat` implementation is FUSION-agnostic: it
  consumes any (weights, LD, GWAS z) triple regardless of how the
  weights were derived. A FUSION harness would not exercise additional
  TG code paths.
- A future iteration can add `validation/external/fusion/` if a paper
  reviewer specifically requests cross-method consistency.

Decision: ship MetaXcan alone; document the rationale here. If a future
agent adds FUSION, place it under `validation/external/fusion/` with
its own `install.sh`/`fetch_data.sh`/`run.sh`/`compare.py` and document
the algorithmic difference (BSLMM vs elastic net) in that sibling's
README.

## Re-run paths

```bash
# From the repo root:
bash validation/external/metaxcan/install.sh        # clone MetaXcan v0.8.1
bash validation/external/metaxcan/fetch_data.sh     # generate simulated fixture
bash validation/external/metaxcan/run.sh            # run SPrediXcan on fixture
python validation/external/metaxcan/compare.py      # compare vs TG twas_sumstat
```

All four scripts source `validation/external/_lib/preflight.sh` and
assert disk + RAM headroom before any work, per the Pillar B contract.
Outputs:

- `outputs/sprediXcan_results.csv` — MetaXcan gene-level CSV.
- `results/summary.tsv` — per-gene MetaXcan vs TG numbers + deltas.
- `results/agreement.json` — pass/fail report + tolerance gates +
  observed numerics.
- `results/manifest.sha256` — SHA256 of every fixture + output file.

## Files

| File | Purpose |
|------|---------|
| `install.sh` | Clone MetaXcan v0.8.1, verify Python deps, write `.install_marker` |
| `fetch_data.sh` | Invoke `simulate_fixture.py` to stage `data/` |
| `simulate_fixture.py` | Deterministically generate model.db + covariance + GWAS sumstats |
| `run.sh` | Invoke `MetaXcan/software/SPrediXcan.py` on the staged fixture |
| `compare.py` | Run `torchgenomics.postgwas.twas_sumstat`, compare gene-level results |
| `results/agreement.json` | Per-check pass/fail + observed numerics |
| `results/summary.tsv` | Per-gene table (MetaXcan z, TG z, Δz, …) |
| `results/manifest.sha256` | SHA256 of every artefact (fixture + outputs) |

## Findings

- **No F3 divergence**: agreement is FP-precision (3.07e-08 z-score max
  delta) on the first successful run after a fixture-side allele
  orientation fix.
- **Fixture iteration log**:
  - Initial run showed a sign flip on z (Δz ~23, Pearson r = -1.0).
    Root cause: `simulate_fixture.py` had inconsistent
    (effect_allele, ref_allele) tuple ordering between the model-db
    insert (`ref, eff = alleles[j]`) and the GWAS-sumstats writer
    (`eff, ref = per_snp_alleles[snp]`). MetaXcan correctly detected
    the model.eff_allele = GWAS.non_effect_allele mismatch and flipped
    the z to align with model orientation; TG twas_sumstat does not
    flip (the caller is expected to align). Both are correct given
    their inputs; the harness-side fix was to make the model and GWAS
    use the same effective-allele convention. This is documented to
    prevent the same mismatch from being reintroduced.
  - Post-fix: bit-precision agreement on all four gates.

## References

- Barbeira AN, ..., Im HK (2018). "Exploring the phenotypic consequences
  of tissue specific gene expression variation inferred from GWAS summary
  statistics." *Nature Communications* 9:1825.
  DOI [10.1038/s41467-018-03621-1](https://doi.org/10.1038/s41467-018-03621-1).
  (S-PrediXcan estimator, Eq. 4 of paper.)
- Gamazon ER, ..., Im HK (2015). "A gene-based association method for
  mapping traits using reference transcriptome data." *Nature Genetics*
  47:1091-1098. (PrediXcan individual-level.)
