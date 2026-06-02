# Validation

TorchGenomics targets **4th-decimal-place p-value agreement** with established
GWAS tools on canonical datasets. Reference equivalence is enforced on every
CI run by the `golden` test marker.

## Validation tiers

The current release exposes three supported pytest markers: `golden`, `slow`,
and `gpu`. Optional integrations with GEMMA, GAPIT3, GWASpoly, PolyOrigin,
Julia, and CUDA are skip-gated when the required external toolchain is not
available.

```bash
# Fast pure-Python package check
LC_ALL=C.UTF-8 LANG=C.UTF-8 TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/ -v --tb=short -x -q --timeout=300

# Reference-output equivalence
LC_ALL=C.UTF-8 LANG=C.UTF-8 TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/ -m golden -v --tb=short --timeout=600

# CUDA-only checks, when a CUDA PyTorch build and GPU are available
pytest tests/ -m gpu -v --tb=short

# Longer checks excluded from the fast tier
pytest tests/ -m slow -v --tb=short
```

## Reference tools and versions

| Tool            | Version  | Role                                               |
|-----------------|----------|----------------------------------------------------|
| GEMMA           | 0.98.5   | Single-trait LMM, mvLMM, GRM                       |
| GAPIT3          | 3.4      | FarmCPU, BLINK                                     |
| GWASpoly        | 2.12     | Polyploid GWAS (5 gene-action models)              |
| PLINK           | 1.9      | LD block reference (Gabriel)                       |

## Canonical datasets

- **MDP maize** (`benchmark/data/mdp_numeric.txt`) — 276 samples, 3093 SNPs,
  EarHT and pollen-shed traits. Diploid.
- **GWASpoly tetraploid potato** — 957 genotypes, 9888 markers. Tetraploid.

Both datasets are small enough to commit in the repo and run in under a
minute per scan on CI.

## Section-16 tolerances

The charter's Section 16 specifies the numerical tolerances between TorchGenomics
and reference output:

| Statistic                | Tolerance                                    |
|--------------------------|----------------------------------------------|
| Variance components      | 1e-3 relative                                |
| β (effect size)          | Corr > 0.9999 full set; median \|Δ\| < 0.01 on AF ∈ [0.03, 0.97] |
| SE (standard error)      | Corr > 0.9999 full set; median \|Δ\| < 0.02 on AF ∈ [0.03, 0.97] |
| −log₁₀ p (Wald)          | Corr > 0.999                                 |
| −log₁₀ p (LRT)           | Corr > 0.998                                 |
| −log₁₀ p (score)         | Corr > 0.999                                 |
| GRM elements             | Exact (element-wise equality)                |
| Joint Wald (mvLMM)       | Corr > 0.998                                 |

Tolerances are **calibrated to observed measured reality**, not aspirational —
they serve as a regression gate, so future changes that degrade below current
levels fail CI.

## Reference fixtures

Fixtures live under:

- `gemma_demo/output/` — GEMMA LMM + mvLMM + GRM outputs
- `gapit_demo/output/` — GAPIT3 FarmCPU + BLINK outputs
- `benchmark/gwaspoly_results/` — GWASpoly tetraploid outputs

These are committed to the repo (~500 KB total) because the tools are not
trivially installable in every CI environment and freezing a specific
upstream version is part of the regression contract.

## Regenerating fixtures

One-time orchestrator for maintainers:

```bash
# GEMMA (Linux / WSL with gemma-0.98.5 on PATH)
python scripts/generate_golden_data.py --gemma

# GAPIT3 (R)
Rscript scripts/generate_golden_data.R --gapit

# GWASpoly (R)
Rscript scripts/generate_golden_data.R --gwaspoly

# Verify fixture presence only
python scripts/generate_golden_data.py --check
```

## Running the golden tests

```bash
# Locally
pytest tests/ -m golden -v

# Subset: GEMMA only
pytest tests/test_golden_gemma.py -v

# CI runs this on every push:
TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/ -m golden -v --tb=short --timeout=600
```

## Current status

- **GEMMA**: 9 golden tests pass (variance components, log-likelihood, GRM,
  p-values [Wald, LRT, score], β/SE, mvLMM joint Wald)
- **GWASpoly**: 6 golden tests pass across 5 gene-action models
- **GAPIT3**: 3 tests skipped pending fixture regeneration

The golden CI job is gated by `TORCHGENOMICS_DISABLE_NATIVE=1` to ensure the
pure-Python reference path matches the external tool. Separate `test-native`
and `test-no-openmp` jobs exercise the native accelerators.

## NCBI annotation

The NCBI annotation module (`torchgenomics.annotate`) is network-required. It is
exercised in CI by a recorded-cassette test that replays a captured NCBI
exchange offline. To refresh the cassette when NCBI's response schema changes:

```bash
export TORCHGENOMICS_NCBI_LIVE=1
pytest tests/test_annotate_cassette.py --record
```
