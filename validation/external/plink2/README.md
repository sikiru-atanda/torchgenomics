# PLINK 2.0 reference harness (Pillar B / B1)

PLINK 2.0 is the closest ground truth available for diploid linear-regression GWAS, GRMs, and pairwise LD r². This harness installs a pinned PLINK 2 build, runs three reference computations against the MDP maize fixture (281 samples × 3093 SNPs), and asserts that TorchGenomics agrees within calibrated tolerances.

## Pinned reference

| Field | Value |
|---|---|
| Tool version | `v2.0.0-a.7.0LM 64-bit Intel (25 Apr 2026)` |
| Upstream URL | `https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_x86_64_20260425.zip` |
| SHA-256 (zip) | `e70a283aefe004122fca3e632ae0b24023a24635f98a8e768ea8d542bbc659a9` |
| Platform | Linux x86_64 only (other platforms abort install with a clear message) |

## Reproduction recipe

```bash
# From repo root:
bash validation/external/plink2/install.sh        # idempotent; prints binary version on completion
bash validation/external/plink2/fetch_data.sh     # converts MDP numeric → PLINK BED into data/
bash validation/external/plink2/run_plink2.sh     # 3 reference outputs into outputs/
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/external/plink2/compare.py

# Or via pytest (requires -m external; skipped by default):
pytest -m external tests/test_external_plink2.py -v
```

Each shell script sources `validation/external/_lib/preflight.sh` and asserts disk + RAM headroom before doing any work, per the Pillar B contract.

## Reference outputs (in `outputs/`)

| File | Source | TorchGenomics counterpart |
|---|---|---|
| `glm.EarHT.glm.linear` | `--glm hide-covar` on EarHT | `torchgenomics.models.GLM` Wald scan |
| `kinship.rel` + `kinship.rel.id` | `--make-rel triangle cov` (centered VanRaden-style) | `torchgenomics.linalg.kinship.grm_vanraden` |
| `r2.unphased.vcor2` + `.vars` | `--r2-unphased square` | `torchgenomics.ld._pairwise.compute_r2_matrix` |

## Calibrated tolerances

The spec's §16 numbers are aspirational; the actual regression gates are observed-then-floored. From the first successful run (2026-05-04):

| Comparison | Metric | Observed | Floor (asserted) |
|---|---|---|---|
| GLM | β correlation (full set, 2953 SNPs) | **1.000000** | 0.9999 |
| GLM | β median \|Δ\| (AF [0.03, 0.97], n=2814) | 1.48e-06 | 1e-3 |
| GLM | SE median \|Δ\| (AF [0.03, 0.97]) | 2.89e-03 | 1e-2 |
| GLM | −log₁₀(P) correlation | 0.99972 | 0.999 |
| GRM | Off-diag Pearson corr (281×281) | 0.999949 | 0.9999 |
| GRM | Off-diag Spearman rank corr | 0.999868 | 0.999 |
| GRM | Median rel-err on \|K\|>0.05 (after trace rescale) | 4.77e-03 | 1e-2 |
| r² | Off-diag Pearson corr (3.9 M pairs) | 1.000000 | 0.999 |
| r² | Off-diag max \|Δ\| | 5.00e-07 | 1e-3 |

### Why the SE floor is 1e-2 instead of spec §16's 2e-2

Both PLINK 2 (`--glm`) and TorchGenomics' `GLM` produce a Wald β with the exact same OLS estimator. The SE comes from the residual mean-square: PLINK uses `MSE = RSS_full / (n − c − 1)`, while TorchGenomics uses `sig2_e = RSS_null / (n − c)` (no per-SNP DF correction). On 281 samples that introduces a uniform scale factor of ~`(n − c) / (n − c − 1) ≈ 1.004`, so SE values differ by ~0.2% across the matrix, hence the observed median |Δ|≈3e-3 on SE values around 1.4. This is a documented parameterization difference, not a bug — see comments in `compare.py`.

### Why the GRM is `--make-rel cov` not `--make-rel`

PLINK 2's default `--make-rel` is GCTA-style per-SNP variance standardization with an implementation-specific diagonal correction that doesn't match a clean closed-form. The `cov` modifier instead computes `K = (G − 2p)(G − 2p)^T / m`, which differs from VanRaden only by a global scalar (the per-SNP variance is summed instead of using `m` as the divisor). The two correlate at >0.9999 element-wise after rescaling, giving a clean GRM reference.

A future extension can wire `--make-rel` (default GCTA mode) against `grm_yang_gcta` once we reverse-engineer PLINK 2's exact diagonal correction; for now, `cov` ↔ VanRaden is the comparison that gives a clean tolerance gate.

### Why we filter to MAF > 1e-6 for the GRM

PLINK 2 refuses `--make-rel` on truly monomorphic variants (it cannot standardize a zero-variance column). On the MDP fixture, 158 SNPs are monomorphic, so the GRM uses 2935 of 3093 SNPs. The filtered set is dumped to `outputs/kinship_snps.snplist` and TorchGenomics scores against the same set.

## Allele-flip handling

PLINK 2's `--glm` reports β per copy of the **minor** allele (it auto-picks A1). Our `convert_mdp_to_bed.py` writes the .bim with `A1 = "A"` and `A2 = "G"`, and TorchGenomics' `PlinkBedReader` decodes raw 2-bit code `0b00` to dosage 2 — i.e. TG dosage counts the .bim A1 ("A") per PLINK 1.9 canonical convention (post-2026-05-13 fix). For SNPs where PLINK chose `A1 = "G"` (the opposite of our BIM A1), the reported β has opposite sign to TorchGenomics', so `compare.py` flips PLINK's β before comparing. This affects ~half of MDP variants.

## Peak memory

PLINK 2 on the MDP fixture (281 × 3093, ~6 MB) uses < 100 MB peak — far below the headroom budgeted by `preflight_check_with_data_size "plink2-run" 1 1`. The pre-flight is configured for 1 GB working-set + 1 GB peak RAM as a conservative safety margin.

| Stage | Disk pre-flight | RAM pre-flight | Observed peak |
|---|---|---|---|
| install (~8 MB zip) | 4 GB | 6 GB | < 50 MB |
| fetch (HapMap → BED) | 4 GB | 6 GB | < 200 MB |
| run (--glm + --make-rel + --r2) | 4 GB | 6 GB | < 100 MB |
| compare.py | n/a | n/a | ~500 MB (loads 3093×3093 r² matrix) |

## Layout

```
validation/external/plink2/
├── install.sh              # pinned-version download + checksum + extract
├── fetch_data.sh           # MDP numeric → PLINK BED (no internet download)
├── convert_mdp_to_bed.py   # helper invoked by fetch_data.sh
├── run_plink2.sh           # produces 3 reference outputs into outputs/
├── compare.py              # asserts tolerances vs TorchGenomics
├── README.md               # this file
├── bin/                    # plink2 binary (ignored)
├── data/                   # converted MDP fileset (ignored)
└── outputs/                # PLINK 2 reference outputs (ignored)
```

## CI / test integration

```bash
# Skipped by default (the conftest auto-skips `external` markers):
pytest tests/test_external_plink2.py            # 3 skipped

# Opt-in:
pytest -m external tests/test_external_plink2.py -v
```

The pytest module dynamically imports `compare.py` from outside the package tree, so no modification of `torchgenomics/` is required to wire this harness.
