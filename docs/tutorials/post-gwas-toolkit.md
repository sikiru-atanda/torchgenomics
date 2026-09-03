# Tutorial: The post-GWAS toolkit

You've run a GWAS and have summary statistics (or you're about to). Now what?
TorchGenomics ships a full post-GWAS facade — LD clumping, fine-mapping,
heritability, LDSC, power analysis, winner's-curse correction, meta-analysis,
multi-ancestry meta-regression, colocalization, Mendelian randomization,
SMR/HEIDI, gene-set enrichment, polygenic scores, and gene annotation — as
one consistent set of functions, reachable identically from Python, the CLI,
and R. This tutorial walks through every one of them on tiny synthetic
fixtures so each call actually runs end to end.

The whole tour is a single runnable script — clone the repo and run it
yourself:

```bash
python examples/python/post_gwas_toolkit.py
```

Every code block below is lifted verbatim from that script
(`examples/python/post_gwas_toolkit.py`), and every "Expected output" excerpt
is quoted from a real run of it — nothing here is aspirational or invented.

## Setup

The script builds one small synthetic fixture per section (a `pandas`
sumstats `DataFrame` written to TSV, plus a genotype CSV or LD-reference file
where a section needs one) using the lowercase sumstats schema
`torchgenomics.postgwas.load_sumstats` expects: `chr, pos, snp, a1, a2, beta,
se, p, n, af`. The one exception is `finemap`, which runs the SuSiE-RSS
engine and uses the uppercase `SNP/CHR/BP/A1/A2/BETA/SE/N` schema instead.

All fixtures are tiny (<= 60 variants) and live under a temporary directory
that's cleaned up on exit. The two imports every section below needs:

```python
import torchgenomics as tg
import torchgenomics.api as tg_api  # only needed for ldsc / ldsc_rg (see the LDSC section)
```

Representative fixture builders (see the script for the rest): a 30-SNP,
3-LD-block panel with one planted causal SNP per block feeds both `clump` and
`ldsc`/`ldsc_rg`:

```python
def build_clump_ldsc_fixture(tmp: Path, seed: int = SEED):
    ...
    df = pd.DataFrame({
        "chr": chrom, "pos": pos, "snp": snp, "a1": a1, "a2": a2,
        "beta": beta, "se": np.full(m, 0.05), "p": p,
        "n": np.full(m, 5000), "af": np.full(m, 0.3),
    })
    gwas_path = tmp / "gwas_clump_ldsc.tsv"
    df.to_csv(gwas_path, sep="\t", index=False)

    geno_path = tmp / "geno_clump_ldsc.csv"
    _write_geno_csv(geno_path, snp, chrom, pos, a1, a2, G)
    return gwas_path, geno_path
```

Each section below shows the actual analysis call plus the printed result
object; the corresponding fixture builder is in the script if you want the
full construction.

## Clump — LD clumping to independent index variants

Collapses correlated significant SNPs down to independent index (lead)
variants, using an LD reference genotype panel. Use it to turn a long list of
correlated hits into a clean set of loci before downstream analysis
(fine-mapping, gene-set enrichment, reporting).

```python
r = tg.clump(
    sumstats=str(gwas_clump_ldsc), genotype=str(geno_clump_ldsc),
    output=str(tmp / "clump_run"), p_threshold=1e-3, r2=0.1, window_kb=250.0,
)
print(r.summary())
print(r.clumps.to_string(index=False))
```

Expected output:

```
Clumped 30 variants → 3 independent loci (p<1.00e-03, r²<0.1) in 0.0s
index_snp  chr  pos            p  n_clumped
      rs5    1 1500 1.000000e-12          9
     rs15    1 2500 1.000000e-12          9
     rs25    1 3500 1.000000e-12          9
```

## Finemap — SuSiE-RSS fine-mapping

Runs the SuSiE-RSS engine (the same core as the `bayes-scan-rss` CLI
subcommand) on summary statistics + an LD reference to produce credible sets
of likely-causal variants. Use it once you have a locus of interest and want
to narrow down which SNP(s) within it are plausibly causal, without
individual-level genotypes.

```python
r = tg.finemap(
    str(finemap_ss), str(finemap_ld), max_num_causal=2, threads=1,
    output=str(tmp / "finemap_out.tsv"),
)
print(r.summary())
print(r.credible_sets.to_string(index=False))
```

Expected output:

```
Fine-mapping (SuSiE-RSS via bayes-scan-rss) — 5 variants, 1 credible sets, 1 variants in a credible set
 credible_set  n_snps lead_snp  lead_pip  sum_pip
            1       1      rs2       1.0      1.0
```

## HESS — local heritability and local genetic correlation

HESS (Heritability Estimation from Summary Statistics) estimates
region-level SNP heritability, or — passing a second GWAS — local genetic
correlation between two traits over the same regions. Use it to see where
heritability (or shared genetic signal between two traits) concentrates
across the genome rather than only a single genome-wide number.

```python
r = tg.hess(str(gwas_hess), str(ld_hess), str(regions_hess),
            output=str(tmp / "hess_h2.tsv"))
print(r.summary())
print(r.results.to_string(index=False))
```

Expected output (h² mode):

```
HESS local h² — 4 regions, 40 SNPs, total=0.0080 (se 0.0009)
region_id chrom  start  end  h2_local  h2_local_se  n_snps  n_eigenvalues_kept
   1:1-10     1      1   10  0.009065     0.000447      10                  10
  1:11-20     1     11   20 -0.000042     0.000447      10                  10
```

Passing `gwas2=` runs the same regions in local-rg mode instead:

```python
r = tg.hess(str(gwas_hess), str(ld_hess), str(regions_hess), gwas2=str(gwas_hess))
print(r.summary())
```

Expected output (rg mode, `gwas2` = same file as a sanity check):

```
HESS local rg — 4 regions, 40 SNPs, total=0.0120 (se 0.0009)
```

## LDSC / LDSC-rg — SNP heritability and genetic correlation via LD score regression

**Caveat on import path:** `ldsc` and `ldsc_rg` are "tier-2" CLI-bridge
functions and are only reachable through `torchgenomics.api` directly — they
are *not* re-exported at the top-level `torchgenomics` package (see
`torchgenomics/__init__.py` vs. `torchgenomics/api/__init__.py`'s
`__getattr__`). That's why the script imports `torchgenomics.api as tg_api`
and calls `tg_api.ldsc(...)` / `tg_api.ldsc_rg(...)` here, unlike every other
section which uses the top-level `tg.*`.

```python
r = tg_api.ldsc(
    sumstats=str(gwas_clump_ldsc), genotype=str(geno_clump_ldsc),
    output=str(tmp / "ldsc_run"),
)
print(r.summary())
print(open(str(tmp / "ldsc_run.ldsc.txt")).read())
```

```python
r = tg_api.ldsc_rg(
    sumstats1=str(gwas_clump_ldsc), sumstats2=str(gwas_clump_ldsc),
    genotype=str(geno_clump_ldsc), output=str(tmp / "ldsc_rg_run"),
)
print(r.summary())
print(open(str(tmp / "ldsc_rg_run.ldsc_rg.txt")).read())
```

**Toy-data caveat:** with only 30 SNPs, the block-jackknife falls back to a
single block (200 requested blocks > 30 SNPs available), so the printed SE is
`NaN`. This demonstrates the call mechanics, not a statistically meaningful
h²/rg estimate — real runs use thousands of SNPs across hundreds of
jackknife blocks, which is what makes the SE well-defined.

Expected output:

```
h2 = -0.0927 (SE = nan)
CLI 'ldsc' — exit 0
Runtime: 0.0s
h2:            -0.0927 (nan)
Intercept:     123.2794 (nan)
Mean chi2:     14.4855
Lambda GC:     0.1506
N SNPs:        30
```

```
rg = -0.9871 (SE = nan)
CLI 'ldsc-rg' — exit 0
rg:          -0.9871 (nan)
cov_g:       -0.091457 (0.142541)
h2_1:        -0.0927 (nan)
h2_2:        -0.0927 (nan)
```

## Power — GWAS detection power

Computes per-variant statistical power (and, optionally, a power curve) for
detecting an effect of a given size at a given sample size and significance
threshold. Use it at study-design time (how big a sample do I need?) or
post-hoc, to see which of your hits were adequately powered.

```python
r = tg.power(str(gwas_power_wc), power_curve=True, output=str(tmp / "power_out.tsv"))
print(r.summary())
print(r.results.to_string(index=False))
```

Expected output:

```
Power — 6 variants, 3 at power>=0.8 (alpha=5.0e-08, N=10000)
snp  af   beta        power      ncp  min_detectable_beta    required_n
rs0 0.3  0.500 1.000000e+00 1050.000             0.071129    377.152276
rs1 0.3  0.010 7.811394e-07    0.420             0.071129 942880.691067
rs2 0.3  0.285 1.000000e+00  341.145             0.071129   1160.825720
```

## Winner's curse — effect-size de-biasing (all 3 methods)

Significant SNPs from a GWAS have systematically inflated effect-size
estimates ("winner's curse") — this corrects for it. All three supported
methods are shown: `conditional_likelihood`, `fiqt`, and `bootstrap`. Use
this before feeding discovery-cohort effect sizes into downstream tools
(e.g. PGS weights, power calculations for a replication cohort) that assume
unbiased estimates.

```python
for method in ("conditional_likelihood", "fiqt", "bootstrap"):
    kw = {"n_boot": 200, "seed": 1} if method == "bootstrap" else {}
    r = tg.winners_curse(str(gwas_power_wc), method=method, **kw)
    print(f"  [{method}] {r.summary()}")
```

Expected output:

```
  [conditional_likelihood] Winner's-curse (conditional_likelihood) — 3 of 6 variants corrected
  [fiqt] Winner's-curse (fiqt) — 3 of 6 variants corrected
  [bootstrap] Winner's-curse (bootstrap) — 3 of 6 variants corrected
```

A representative row (`conditional_likelihood`, showing shrinkage on the
`rs2` hit):

```
snp  beta_original  beta_adjusted  se_adjusted  shrinkage_factor
rs2          0.285       0.252670          NaN          0.886562
```

## Meta — fixed-effect meta-analysis

Combines summary statistics for the same variants across multiple studies
into a single pooled estimate (fixed-effect here; DerSimonian-Laird random
effects, Stouffer, and RE2 are also available via the `method=` argument).
Use it to pool multiple cohorts/studies of the same trait.

```python
r = tg.meta([str(sumstats_a), str(sumstats_b)], method="fixed",
            output=str(tmp / "meta_run"))
print(r.summary())
```

Expected output:

```
Meta-analysis (fixed) across 2 studies, 50 variants, 1 significant, median I²=0.00 in 0.0s
```

## MR-MEGA — multi-ancestry meta-regression

Meta-regresses effect sizes across multiple ancestries/populations against
axes of genetic ancestry, rather than assuming one shared effect — better
suited than standard fixed-effect meta-analysis when effect sizes vary
systematically by ancestry. Needs at least 3 populations for `n_axes=1`.

```python
r = tg.mr_mega([str(p) for p in mrmega_paths], n_axes=1, output=str(tmp / "mrmega_out.tsv"))
print(r.summary())
```

Expected output:

```
MR-MEGA — 4 populations, 1 axes, 30 SNPs min p_meta=1.178e-157
```

## Coloc — pairwise colocalization and hyprcoloc

Tests whether two (pairwise, Giambartolomei 2014) or more (`hyprcoloc`, Foley
2021) traits share a single causal variant at a locus — the standard way to
ask "is this GWAS hit the same signal as this eQTL / this other trait?"

```python
r = tg.coloc(str(sumstats_a), str(sumstats_b), method="pairwise",
            output=str(tmp / "coloc_pairwise.tsv"))
print(r.summary())
```

Expected output:

```
Colocalization (pairwise) — PP.H4=1.000 (strong evidence of a shared causal variant)
  PP.H0=0.000  PP.H1=0.000  PP.H2=0.000  PP.H3=0.000  PP.H4=1.000
```

```python
r = tg.coloc([str(sumstats_a), str(sumstats_b), str(sumstats_c)], method="hyprcoloc")
print(r.summary())
```

Expected output:

```
Colocalization (hyprcoloc) — PP(all colocalize)=1.000
```

## MR — two-sample Mendelian randomization

Estimates a causal effect of an exposure on an outcome from genetic
instruments, using `method="all"` to run IVW, MR-Egger, weighted-median, and
MR-PRESSO together for comparison / sensitivity checking.

```python
r = tg.mr(str(mr_exposure), str(mr_outcome), method="all", n_boot=200, n_perm=200,
          seed=42, output=str(tmp / "mr_out.tsv"))
print(r.summary())
print(r.results.to_string(index=False))
print(f"  (simulated causal effect = {mr_true_effect})")
```

Expected output (simulated causal effect = 0.5):

```
MR (all) — 40 instruments
Causal estimate: beta=0.5090 p=0.000e+00
         method     beta       se          pval  n_instruments  egger_intercept  egger_intercept_p  n_outliers  beta_corrected  pval_corrected
            ivw 0.509009 0.010392  0.000000e+00             40              NaN                NaN         NaN             NaN             NaN
          egger 0.491789 0.040175  9.316286e-15             40         0.005425           0.659746         NaN             NaN             NaN
weighted_median 0.507475 0.015639 5.364814e-231             40              NaN                NaN         NaN             NaN             NaN
         presso 0.509009 0.010392  0.000000e+00             40              NaN                NaN         0.0        0.509009             0.0
  (simulated causal effect = 0.5)
```

## SMR — SMR + HEIDI (expression-mediation test)

Summary-data-based Mendelian Randomization tests whether a GWAS signal at a
gene is mediated through expression (using eQTL summary statistics), and
HEIDI tests whether that's a single shared causal variant rather than
linkage between two distinct signals.

```python
r = tg.smr(str(smr_gwas), str(smr_eqtl), str(smr_gene_map), output=str(tmp / "smr_out.tsv"))
print(r.summary())
print(r.results.to_string(index=False))
```

Expected output:

```
SMR — 3 genes tested, 3 SMR-significant, 3 pass HEIDI
gene_id probe_snp  beta_smr   se_smr    p_smr  chi2_smr  beta_gwas  beta_eqtl  p_heidi  n_heidi_snps  heidi_stat
  gene0       rs1  0.833333 0.216951 0.000122 14.754098       0.25        0.3 0.941518             9    3.492939
```

## Gene-set enrichment — MAGMA-style competitive enrichment

Tests whether a curated gene set (e.g. a pathway) is enriched for GWAS
signal relative to the genomic background — MAGMA-style competitive testing
on gene-level p-values aggregated from SNP sumstats plus a gene annotation
and a gene-set file.

```python
r = tg.gene_set_enrichment(str(enrich_gwas), str(enrich_genes), str(enrich_sets),
                           output=str(tmp / "enrich_out.tsv"))
print(r.summary())
print(r.results.to_string(index=False))
```

Expected output:

```
Gene-set enrichment — 2 sets over 6 genes, 1 significant (p<0.05)
gene_set_name  n_genes_in_set  beta_enrichment      se            p
          hot               2         8.302953 1.43673 3.755909e-09
         null               4        -8.302953 1.43673 1.000000e+00
```

## PGS fit / PGS score — polygenic scores

`pgs_fit` builds polygenic-score weights from GWAS summary statistics plus
an LD reference (methods: C+T, LDpred2-Inf/Grid/Auto, PRS-CS); `pgs_score`
then applies those weights to a target genotype panel to produce
per-individual scores. Use these together to go from summary statistics to
individual-level prediction scores.

```python
r = tg.pgs_fit(str(pgs_sumstats), str(pgs_ld_ref), str(tmp / "pgs_weights.tsv"),
               method="ct", clump_p=0.5, clump_r2=0.9, device="cpu")
print(r.summary())
```

Expected output:

```
PGS fit (ct): 12 weighted variants from 12 input in 0.0s
```

```python
r = tg.pgs_score(str(pgs_target_geno), str(tmp / "pgs_weights.tsv"),
                 str(tmp / "pgs_scores.tsv"), device="cpu")
print(r.summary())
```

Expected output:

```
PGS scored: 50 individuals, 12 variants. Mean=-0.0680, SD=0.4305, range=[-1.1543, 0.7546]. Runtime: 0.0s
```

## Annotate hits — NCBI gene annotation (reference-only, network required)

`annotate_hits` maps significant GWAS hits to nearby genes (via NCBI
Datasets v2 + E-utilities), within a configurable up/downstream window, and
optionally pulls GO terms and orthologs. It needs live network access, so
this section is **shown but never executed** in the verified example:

```python
r = tg.annotate_hits(str(gwas_power_wc), crop="maize", p_threshold=5e-8,
                      window_up=50_000, window_down=50_000,
                      output=str(tmp / "annotate_out"))
```

The script prints this call shape without running it:

```
=== 15. tg.annotate_hits — network-required, NOT executed ===
  Call shape (reference only; needs live NCBI Datasets/E-utilities access):
    r = tg.annotate_hits(str(gwas_power_wc), crop="maize", p_threshold=5e-8,
                          window_up=50_000, window_down=50_000,
                          output=str(tmp / "annotate_out"))
```

## Where to go next

- **CLI**: every function above has a thin CLI wrapper — run
  `torchgenomics <subcommand> --help` (e.g. `torchgenomics clump --help`,
  `torchgenomics finemap --help`, `torchgenomics ldsc --help`) for the full
  flag reference.
- **R**: the same tour, with `tg_*` wrappers and R-built fixtures, is at
  `vignette("post-gwas-toolkit", package = "rTorchGenomics")`.
