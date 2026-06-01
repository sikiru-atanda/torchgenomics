# SMR + HEIDI harness (Tier 1 A2)

Reference-tool comparison between **Yang lab SMR** (`smr` v1.3.1) and
**TorchGWAS** `torchgwas.postgwas.smr_test` / `heidi_test`. Part of the
Genome Biology Methods paper Tier 1 Pillar B harness suite.

## Pinned tool version

| Field | Value |
|-------|-------|
| Distribution URL | https://yanglab.westlake.edu.cn/software/smr/download/smr-1.3.1-linux-x86_64.zip |
| Version | `1.3.1` |
| Build | `Mar 7 2024 17:08:47, GCC 8.3` |
| Released | 2024-03-08 (zip Last-Modified) |
| Zip SHA256 | `4d779197a0b3399db36c9cdf7b4b4190ea40fa33a47253f5419ad27c3bce251e` |
| License | MIT (per banner) |

The Yang lab does not publish point-release git tags; v1.3.1 is the most
recent stable Linux binary release as of harness authorship (2026-05-15).
The zip SHA256 is the authoritative identifier.

## TorchGWAS comparison targets

`torchgwas/postgwas/_smr.py`:

- `smr_test(gwas, eqtl, gene_id, probe_snp, eqtl_p_threshold)` --- single-probe
  SMR (chi-squared 1 d.f.). Formula:
  `chi2_SMR = (z_g^2 * z_e^2) / (z_g^2 + z_e^2)` (Zhu et al. 2016, eq. 1).
  Wald-ratio causal estimate `beta_SMR = beta_GWAS / beta_eQTL` with
  delta-method standard error.
- `heidi_test(gwas, eqtl, probe_snp, nearby_snps, ld_r2_threshold, max_snps)`
  --- heterogeneity in dependent instruments. TorchGWAS uses a
  **delta-method diagonal** variance estimator.

## Fixture

Deterministically simulated single-locus SMR fixture (seed = 42):

| File | Contents |
|------|----------|
| `data/ref.{bed,bim,fam}` | PLINK 1 binary LD reference (n=500, m=15 cis-SNPs, chr 1). |
| `data/gwas.ma` | SMR `--gwas-summary` input (SNP A1 A2 freq b se p N). 15 cis + 50 background. |
| `data/probe.esd` | per-probe eQTL summary (Chr SNP Bp A1 A2 Freq Beta se p), 15 cis-SNPs. |
| `data/probe.flist` | SMR `--eqtl-flist` index (single probe). |
| `data/sim_truth.json` | seed, planted causal SNP, ratios, SHA256 of every fixture file. |

Fixture parameters:

| Param | Value |
|-------|-------|
| Seed | 42 |
| Reference-panel samples (n) | 500 |
| Cis-SNPs per probe (m) | 15 |
| Background GWAS-only SNPs | 50 |
| GWAS sample size | 50000 |
| eQTL sample size | 1000 |
| Top SNP planted b_GWAS | 0.15 |
| Top SNP planted b_eQTL | 0.60 |
| Helper indices (same ratio b_g/b_e) | (5, 7, 9, 11, 13) |
| Helper b_eQTL | +/- 0.35 (alternating sign) |
| Haplotype latent correlation (rho) | 0.6 |
| Realised pairwise SNP r^2 | ~0.05 - 0.20 (compound-symmetric) |

### Why simulate (not download a real GTEx eQTL panel)

GTEx v8 eQTL BESD bundles are gigabyte-scale and the hosting URL has
shifted multiple times. The SMR statistic (chi-squared, Wald ratio) is
purely summary-statistic, so numerical-agreement testing requires only
a well-formed (GWAS, eQTL, BED) triple. Same strategy as
`validation/external/metaxcan/` (A1, S-PrediXcan).

### Why mild LD (rho ~ 0.6, realised r^2 ~ 0.12)

The harness brief originally proposed K i.i.d. SNPs so the SMR LD-weighted
HEIDI covariance reduces to the diagonal (delta-method) variance that
TorchGWAS computes. In practice SMR HEIDI applies an inclusion filter
`0.05 <= r^2 <= 0.9` to the cis-SNPs --- under perfect i.i.d. genotypes
every helper drops below `r^2 = 0.05` (max observed r^2 = 0.012 at n=500),
HEIDI sees `nsnp_HEIDI = 0`, and SMR crashes inside `routine gammp`
(`Invalid arguments` on `chi^2.sf(NaN, 0)`).

To satisfy the filter without inducing strong LD, the fixture draws
haplotypes from a compound-symmetric latent normal copula with
`rho_haplotype = 0.6`. Realised pairwise r^2 lands in [0.045, 0.20] with
mean ~0.12. Off-diagonals are small enough that the LD-weighted HEIDI
variance is dominated by the diagonal term, but not zero --- this is the
source of the documented HEIDI divergence below.

### Fixture SHA256 (first successful run, 2026-05-15)

```
d08a5f6c85edb813610b1aa53ec3e1d259dd3f81a105f348642317f5628ddd91  data/ref.bed
e1c271a3e7341ecbd9e9a7b9fa3ff14d5e600df5077ee33ca6d6ecb78814a6ca  data/ref.bim
e184f73fd3c03d2ccaf74a0d36e28776fd9c2f90b06f5c606b77a70a87a91aaa  data/ref.fam
2f7ec82f9103ed7f1d37c8a1f6bb08333594a76a5c27de2a5bef3048b7d1be8c  data/gwas.ma
874bf6d2c9b78187edb0b47604553a47345ac7c8ec9b7c7cd6cb09ab77867615  data/probe.esd
33a63dea8ddbe8b6a705b42b8d7ff6f5acc3c2137520058320356e82f3076ac2  data/probe.flist
```

## Observed agreement (first successful run, 2026-05-15)

Linux x86_64 host, Python 3.13.5, SMR v1.3.1 (Mar 7 2024 build).
Single probe, n_SNPs in HEIDI = 5 (TG) / 6 (SMR).

| Metric | SMR | TG | Observed agreement | Floored tolerance | Status |
|--------|------|------|--------------------|-------------------|--------|
| beta_SMR | 0.298609 | 0.298609 | rel 1.14e-6 | 5e-5 | PASS |
| chi2_SMR | 110.4453 | 110.4452 | rel 3.04e-7 | 5e-5 | PASS |
| p_SMR | 7.827e-26 | 7.828e-26 | abs |Delta -log10 p| 4.4e-5 | 5e-4 | PASS |
| chi2_HEIDI | 6.86 | 9.47 | rel 3.81e-1 | 1.0 | PASS (F3 post-V1) |
| p_HEIDI | 0.232 | 0.0919 | rel 6.03e-1 | 2.0 | PASS (F3 post-V1) |

SMR beta / p / chi-squared agree at floating-point precision (4-7 sig figs).
HEIDI shows a documented systematic divergence; see below.

### F3 post-V1 divergence: HEIDI variance estimator

**Status: documented, no fix planned for V1.**

The SMR (Yang lab) HEIDI implementation computes the variance of
`d_i = b_g_i / b_e_i - b_g_top / b_e_top` from a full LD-weighted
covariance matrix derived from the PLINK BED reference panel (Zhu 2016
supplementary, HEIDI test, computation of variance of d). TorchGWAS
`heidi_test` uses a delta-method diagonal variance that assumes the
SNPs being tested are mutually uncorrelated.

For perfectly independent SNPs the two estimators agree (LD off-diagonals
= 0), but the SMR inclusion filter `0.05 <= r^2 <= 0.9` rules out the
perfectly-independent regime. With the fixtures mild LD (rho ~ 0.6,
mean r^2 ~ 0.12) the LD-weighted variance is systematically larger than
the diagonal-only delta-method variance, so SMRs chi^2 is smaller (and
p larger) than TorchGWASs on the same input. Observed: SMR chi^2 =
6.86, TG chi^2 = 9.47 (rel diff 38%); SMR p = 0.232, TG p = 0.092.

Classification per the F3 policy:
  - SMR / HEIDI is Phase 45 (post-V1).
  - The TorchGWAS implementation is conservative (smaller HEIDI variance
    -> larger chi^2 -> smaller p -> more likely to reject single-causal),
    so it does not silently inflate false positives in SMRs
    pleiotropy-vs-linkage verdict.
  - Documented in `docs/validation_findings.md`.
  - The HEIDI tolerance floors (`TOL_REL_CHI2_HEIDI = 1.0`,
    `TOL_REL_P_HEIDI = 2.0`) gate that the test runs to completion and
    produces a same-order-of-magnitude answer, NOT bit-precision agreement.

To remove this divergence in a future release, port the LD-weighted
variance from Zhu 2016 supplementary into `torchgwas.postgwas._smr`
(takes a `ld_matrix` argument symmetric to `torchgwas.postgwas._twas`).

## Re-run paths

```bash
# From the repo root:
bash validation/external/smr/install.sh        # unpack SMR v1.3.1 into bin/
bash validation/external/smr/fetch_data.sh     # simulate fixture
bash validation/external/smr/run.sh            # make-besd + SMR + HEIDI
python validation/external/smr/compare.py      # compare vs TG smr_test/heidi_test

# Verify SHA256 manifest:
cd validation/external/smr && sha256sum -c results/manifest.sha256
```

All four scripts source `validation/external/_lib/preflight.sh` and
assert disk + RAM headroom before any work, per the Pillar B contract.

Outputs:
- `outputs/smr_results.smr` --- SMR gene-level table.
- `outputs/sim_eqtl.{besd,esi,epi,summary}` --- BESD eQTL summary.
- `outputs/smr_run.log` / `outputs/make_besd.log` --- SMR stdout/stderr.
- `results/summary.tsv` --- per-metric SMR vs TG numbers + deltas.
- `results/agreement.json` --- pass/fail report + tolerance gates +
  observed numerics + extras.
- `results/manifest.sha256` --- SHA256 of every fixture + output file.

## Files

| File | Purpose |
|------|---------|
| `install.sh` | Verify cached / fetch SMR v1.3.1 zip, SHA256 check, unpack into bin/ |
| `fetch_data.sh` | Invoke `simulate_smr_fixture.py` to stage `data/` |
| `simulate_smr_fixture.py` | Generate BED + .ma + .esd + .flist + sim_truth.json |
| `run.sh` | Invoke SMR --make-besd then SMR + HEIDI |
| `compare.py` | Run TG smr_test + heidi_test, compute deltas, write results/ |
| `results/agreement.json` | Per-check pass/fail + observed numerics |
| `results/summary.tsv` | Per-metric SMR / TG / delta table |
| `results/manifest.sha256` | SHA256 of every artefact (fixture + outputs) |

## References

- Zhu Z, Zhang F, Hu H, et al. (2016). "Integration of summary data from
  GWAS and eQTL studies predicts complex trait gene targets."
  *Nature Genetics* 48:481-487. DOI
  [10.1038/ng.3538](https://doi.org/10.1038/ng.3538). (SMR + HEIDI
  primary methods paper; supplementary contains the LD-weighted HEIDI
  variance formula.)
- Yang lab SMR distribution page:
  https://yanglab.westlake.edu.cn/software/smr/

