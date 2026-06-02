# TorchGenomics

[![PyPI version](https://img.shields.io/pypi/v/torchgenomics.svg)](https://pypi.org/project/torchgenomics/)
[![Python](https://img.shields.io/pypi/pyversions/torchgenomics.svg)](https://pypi.org/project/torchgenomics/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sikiru-atanda/torchgenomics/blob/master/LICENSE)
[![Tests](https://img.shields.io/badge/tests-3071%20passing-brightgreen.svg)](https://github.com/sikiru-atanda/torchgenomics/tree/master/tests)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#status)

**GPU-accelerated statistical and quantitative genomics on PyTorch — a single
engine for GWAS, post-GWAS, polygenic scoring, LD analysis, imputation,
multi-omics integration, and visualization.**

Originally released as `torchgwas`; renamed in v0.4.0 to reflect a scope
that now extends well beyond Genome-Wide Association Studies. Existing
`import torchgwas` users continue working through a deprecation shim — see
[Migrating from torchgwas](#migrating-from-torchgwas).

<p align="center">
  <img src="https://raw.githubusercontent.com/sikiru-atanda/torchgenomics/master/docs/assets/mdp_earht_manhattan.png" alt="Manhattan plot — MDP maize EarHT" width="720"/>
  <br/>
  <em>Example output from <code>examples/python/01_single_trait_lmm.py</code>
  (MDP maize Ear Height, SingleTraitLMM, 276 samples × 3093 SNPs).</em>
</p>

## Three audiences, one engine

| Audience | Entry point | Example |
|---|---|---|
| **Novice / notebook** | One-call `torchgenomics.api` facade with smart defaults | `result = tg.lmm_scan(genotype="data.bed", phenotype="pheno.tsv"); result.manhattan()` |
| **Advanced / research** | Composable low-level modules: `models`, `scan`, `linalg`, `ld`, `pgs`, `postgwas`, `multiomics`, ... | `SingleTraitLMM().fit_null(Y, X0, K)` → `UnifiedScanner(reader, model).scan(null_fit)` |
| **LLM / MCP tools** | 13 tier-1 functions published as MCP tools over stdio (`pip install torchgenomics[mcp]`) | Add `torchgenomics-mcp` to your Claude Desktop / Claude Code MCP config |

See [`docs/api/quickstart.md`](docs/api/quickstart.md) for the novice walkthrough
and [`docs/mcp/index.md`](docs/mcp/index.md) for the MCP server setup.

## Status

**v0.4.0 · Alpha · 3071 tests passing · Python 3.10 – 3.13 · Linux + macOS + Windows**

V1 core (Phases 0 – 13) delivers GEMMA / GAPIT reference equivalence for
Gaussian single- and multi-trait GWAS. Post-V1 extensions implemented through
Phase 49b cover GLM / GLMM, survival, multi-environment, haplotype, threshold-
linear, polygenic scoring, fine-mapping, post-GWAS, and causal mediation.

**Validation campaign (v0.3.0–v0.3.1):** function-by-function audit across four
pillars (coverage, external reference tools, CLI smoke matrix, fixture
reproducibility). 14 V1 production fixes shipped from validation work + 700
new tests. Reference equivalence validated against GEMMA 0.98.5, GAPIT3,
GWASpoly 2.14, PLINK 2.0, LDSC, regenie 3.3, SAIGE 1.4.4, BOLT-LMM 2.5,
TwoSampleMR 0.7.5, SoyNAM, and SoyMD-style mediation references. All three
committed reference fixtures (GEMMA, GAPIT, GWASpoly) re-verified bit-exactly
authoritative.

**Streaming + efficiency campaign (v0.3.2–v0.3.8):** 36 of 40 CLI scan
subcommands now stream chunks via `iter_chunks` — peak memory
`O(n × chunk_size)` instead of `O(n × m)`. Median UKB-scale (n=500K × m=10M)
GWAS goes from "needs 40 TB cluster" to "runs on a 4 GB workstation." Three
opt-in materializing paths preserved for callers that already have G in
hand. One algorithmically-tied path (`bayes-scan` / SuSiE joint posterior)
retained — streaming would change the inference algorithm.

**Regression nets:** memory (in-test `tracemalloc` guards on streaming peak)
and wall-time (`perf.yml` workflow with > 10% native-kernel regression =
hard fail, > 5% = warning). Plus golden-data CI (GEMMA / GAPIT / GWASpoly),
external-tool CI (Pillar B, weekly), CLI smoke matrix (Pillar C, nightly),
and fixture-drift audit (Pillar D, monthly).

**Post-V1 expansion campaign (v0.3.9, 2026-05-21 → 2026-05-26):** four F3
post-V1 statistical fixes landed on master (hyprcoloc Foley-2021 conditional
prior, coloc_pairwise H3 outer-minus-diagonal, heidi_test LD-weighted
variance, OCFLMM `nuisance_learner='ridge_quadratic'`) with closed-form
agreement against the upstream R hyprcoloc / coloc / SMR v1.3.1 / DoubleML
references. Plus the **observed-expression TWAS surface** (`twas_observed_expression`,
`torchgenomics twas-scan` CLI, rank-INT / quantile-norm / PEER residualization
helpers), the **PrediXcan / FUSION `.db` reader**, **multi-tissue stacking**
+ S-MultiXcan-style aggregation, **gene-level TWAS plots** (Manhattan / QQ /
λ_TWAS), and the **GWAS↔TWAS integration entry point** with 14 combination
methods — 8 classical kernels (Fisher, Brown, Empirical Brown, HMP,
truncated product, min-p, Cauchy, Stouffer) plus 6 novel methods (R²-weighted
Stouffer, LD-aware Brown via eigenMT, polyploid gene-action Fisher,
multi-tissue ACAT + lead-SNP, conditional GWAS+TWAS via COJO, hyprcoloc-PPFC-
gated combination). External-tool head-to-head harnesses for FUSION
measured-expression mode and `metap`/`EmpiricalBrownsMethod` round out the
validation surface. CLI subcommand count: 38 → 40.

## Install

```bash
pip install torchgenomics             # wheel where available; sdist fallback
pip install "torchgenomics[all]"      # + zarr, h5py, pyarrow, seaborn
pip install "torchgenomics[mcp]"      # + MCP server (Claude Desktop / Claude Code)
pip install "torchgenomics[dev]"      # + pytest, ruff, mypy (editable development)
pip install "torchgenomics[docs]"     # + mkdocs + mkdocs-material + mkdocstrings
```

From source:

```bash
git clone https://github.com/sikiru-atanda/torchgenomics.git
cd torchgenomics
pip install -e ".[dev]"
pytest tests/ -v
```

GPU support: install the matching PyTorch CUDA wheel first (see
[docs/getting-started/installation.md](https://github.com/sikiru-atanda/torchgenomics/blob/master/docs/getting-started/installation.md)),
then `pip install torchgenomics`. The same Python code runs on CPU or CUDA by
changing one tensor device.

## Migrating from torchgwas

`torchgwas` v0.3.x users keep working without code changes — the legacy
package, CLI binary, and environment variables stay live as a deprecation
shim through the v0.x series and are removed in v1.0.0:

```python
import torchgwas                      # still works; emits DeprecationWarning once
from torchgwas.models import SingleTraitLMM   # → resolves to torchgenomics.models
```

```bash
torchgwas lmm-scan ...                # still works; prints stderr banner once
TORCHGWAS_DISABLE_NATIVE=1 ...        # accepted (warns); use TORCHGENOMICS_DISABLE_NATIVE
```

To migrate: change `torchgwas` to `torchgenomics` everywhere (imports, CLI
invocations, env vars). The Python API surface is identical.

## Quickstart

### Novice — one line

```python
import torchgenomics as tg
result = tg.lmm_scan(genotype="data.bed", phenotype="pheno.tsv")
print(result.summary())
result.manhattan()        # matplotlib figure
result.top_hits           # pandas DataFrame
result.output_files["tsv"]  # path to full results
```

`tg.lmm_scan` returns a `ScanRun` with summary stats, top hits, plotting
helpers, and explicit output file paths. Same idea for `tg.glm_scan`,
`tg.pgs_fit`, `tg.ld_blocks`, `tg.clump`, `tg.annotate_hits`, etc.
See [`docs/api/quickstart.md`](docs/api/quickstart.md) for the full list.

### CLI

```bash
torchgenomics lmm-scan \
  --genotype benchmark/data/mdp_numeric.txt \
  --map benchmark/data/mdp_SNP_information.txt \
  --phenotype benchmark/data/mdp_traits.txt \
  --trait EarHT --test wald --correction bh \
  --output results/mdp_earht
```

### LLM / MCP

```bash
pip install "torchgenomics[mcp]"
```

Then add to your Claude Desktop / Claude Code MCP config:

```json
{
  "mcpServers": {
    "torchgenomics": {
      "command": "torchgenomics-mcp"
    }
  }
}
```

The LLM can now invoke `tg_validate`, `tg_lmm_scan`, `tg_pgs_fit`,
`tg_annotate`, `tg_manhattan`, and 8 more tier-1 tools as MCP calls.
See [`docs/mcp/index.md`](docs/mcp/index.md) for the full walkthrough.

### Advanced — low-level API (trimmed; full script at
[`examples/python/01_single_trait_lmm.py`](https://github.com/sikiru-atanda/torchgenomics/blob/master/examples/python/01_single_trait_lmm.py)):

```python
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.stats import benjamini_hochberg

geno = pd.read_csv("benchmark/data/mdp_numeric.txt", sep="\t")
pheno = pd.read_csv("benchmark/data/mdp_traits.txt", sep="\t")

# ... sample alignment / missing imputation (see examples/python/01) ...

G = torch.tensor(geno.values, dtype=STAT_DTYPE)
Y = torch.tensor(pheno["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
X0 = torch.ones(G.shape[0], 1, dtype=STAT_DTYPE)
K = torch.tensor(pd.read_csv("gemma_demo/output/mdp_kinship.cXX.txt",
                             sep="\t", header=None).values, dtype=STAT_DTYPE)
vmeta = VariantMeta(snp=list(geno.columns), chr=chrs, pos=pos,
                    a1=["A"] * G.shape[1], a2=["G"] * G.shape[1])

model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
nf = model.fit_null(Y, X0, K=K)
result = model.score_chunk(G, nf, vmeta, test="wald")

fdr = benjamini_hochberg(result.p.cpu())
```

## Learn more

- [**Docs site**](https://sikiru-atanda.github.io/torchgenomics/) — installation,
  tutorials, full API reference, and validation protocol. Built with MkDocs
  Material; deployed on pushes to `master` / `main`.
- [**examples/python/**](https://github.com/sikiru-atanda/torchgenomics/tree/master/examples/python) — 10 working end-to-end scripts
  (single/multi-trait LMM, MET, threshold-linear, PGS, SuSiE fine-mapping,
  mediation, NCBI annotation, polyploid, haplotype).
- [**examples/notebooks/quickstart.ipynb**](https://github.com/sikiru-atanda/torchgenomics/blob/master/examples/notebooks/quickstart.ipynb)
  — same as example 01 but renders the Manhattan inline.
- [**docs/cli.md**](https://github.com/sikiru-atanda/torchgenomics/blob/master/docs/cli.md) — every CLI subcommand with expected inputs.
- [**docs/validation.md**](https://github.com/sikiru-atanda/torchgenomics/blob/master/docs/validation.md) — GEMMA / GAPIT agreement tables.
- [**docs/ROADMAP.md**](https://github.com/sikiru-atanda/torchgenomics/blob/master/docs/ROADMAP.md) — deferred internal improvements and
  candidate Phase 50+ features.

## Features

### GWAS models
- **Classical**: GLM, single-trait LMM, multi-trait LMM (MVLMM), FarmCPU, BLINK
- **Multi-environment**: MET (reaction-norm, FA(k), multi-kernel), MT-MET (Kronecker separable)
- **Novel LMMs**: Multi-kernel, GxE/heteroscedastic, orthogonal cross-fit (DML), knockoff FDR-controlled, genotype-uncertainty, leave-region-out LOCO
- **Categorical traits**: Binary/ordinal/multinomial GLM and GLMM (PQL, SPA), multi-environment GLMM
- **Survival**: Cox PH frailty with Breslow-Clayton PQL and SPACox SPA
- **Longitudinal**: Random regression LMM with Legendre / B-spline bases, spatio-temporal 2D P-spline, RR-MET
- **Haplotype-based**: HTR, block, window, SKAT, plus novel methods (PCHT, HHCT, HSKAT, HapGxE, BayesHap); multi-env / multi-trait haplotype GWAS
- **Fine-mapping**: SuSiE (IBSS) and CAVI Bayesian variable selection
- **LD-conditional**: COJO-style stepwise conditioning with persistence metrics
- **Within-family**: Dual-scan attenuation diagnostics for confounding detection
- **Threshold-linear**: Bermann et al. 2026 for mixed ordinal + continuous traits

### Polyploid support
- Arbitrary ploidy (diploid through hexaploid+)
- Gene-action models: additive, 1-dom through (k−1)-dom, diplo-additive, overdominant, general
- HWE testing with double-reduction correction for autopolyploids

### LD block detection
- 13 methods across 4 categories (classical / literature / novel / diagnostic)
- PLINK 1.9 `--blocks` validated (Jaccard ≥ 0.78 for Gabriel)
- Diploid and polyploid compatible

### Post-GWAS
- **Heritability**: LDSC h² / rg, S-LDSC partitioned, HESS regional
- **Meta-analysis**: IVW, DerSimonian-Laird, Stouffer, RE2, MR-MEGA, MANTRA
- **Colocalization**: Giambartolomei coloc, HyPrColoc multi-trait
- **Mendelian randomization**: IVW, MR-Egger, weighted median, MR-PRESSO
- **SMR / HEIDI** and **TWAS** (S-PrediXcan / PrediXcan)
- **Gene-set enrichment**: MAGMA-style SNP-to-gene + competitive enrichment
- **LD clumping**, **fine-mapping utilities**, **power analysis**, **winner's curse correction**

### Polygenic scores
- C+T (clumping and thresholding), LDpred2 (Inf / Grid / Auto), PRS-CS
- Allele harmonization, individual scoring, PGS validation metrics

### Visualization
- Manhattan (linear and Circos-style polar), Miami, QQ with λ_GC
- Haploview-style LD triangle heatmap, trumpet plot

### Multi-omics
- GRM-corrected causal mediation (single triple + GPU-batched scan + gene-set Wald)
- Multi-kernel heritability; eigenMT FDR; colocalisation prefilter

### Performance
- 24 native C++ extension modules via pybind11 (up to ~12 000× speedup on hot kernels)
- OpenMP parallelization where measured beneficial
- GPU kernels for imputation (mode, KNN, LD-based)
- Three-tier dispatch: GPU > native C++ > Python (automatic fallback)
- AMP (FP16 / BF16) for I/O and GRM; FP64 mandatory for statistical inference

### Multiple testing
- Bonferroni, Holm, BH, BY, Storey q-value, simpleM / M_eff
- GPU-accelerated permutation, Cauchy combination, local FDR
- IHW (covariate-adaptive weighting), AdaPT (covariate-adaptive thresholding)

## Architecture

```
torchgenomics/
  io/          Format detection & readers (PLINK, VCF, BGEN, HapMap, Zarr, CSV)
  preprocess/  Imputation (built-in + BEAGLE/IMPUTE5/Minimac4), QC, standardization
  linalg/      GRM (diploid + polyploid), eigendecomposition, batched Cholesky
  models/      All GWAS models (BaseModel protocol: fit_null + score_chunk)
  scan/        UnifiedScanner streaming chunks through any model
  stats/       Multiple testing, SPA, PVE
  ld/          LD block detection (13 methods), pairwise LD (r², D', CI)
  optim/       6-mode optimizer stack (PX-EM → AI-REML → ... → derivative-free rescue)
  pgs/         PGS construction & validation
  postgwas/    LDSC, meta-analysis, coloc, MR, TWAS, enrichment
  multiomics/  GRM-corrected mediation + multi-kernel heritability
  annotate/    NCBI Datasets v2 + E-utilities gene annotation
  viz/         Manhattan, QQ, Miami, Circos, Haploview, trumpet plots
  _native/     24 C++ pybind11 extension modules + GPU kernels + select_path dispatcher
  cli.py       40 CLI subcommands
```

Data flow: **Format detection → Imputation / phasing → QC → Genotype encoding
→ GRM → Eigendecomposition → Null fitting → Scan loop → Multiple testing →
Reporting.**

## Supported genotype formats

| Format | Extension | Read | Write |
|---|---|---|---|
| PLINK BED | `.bed` / `.bim` / `.fam` | ✓ | ✓ |
| PLINK2 PGEN | `.pgen` / `.pvar` / `.psam` | ✓ | — |
| VCF / BCF | `.vcf` / `.vcf.gz` / `.bcf` | ✓ | — |
| BGEN | `.bgen` | ✓ | — |
| HapMap | `.hmp.txt` | ✓ | — |
| Zarr | `.zarr` | ✓ | ✓ |
| CSV dosage | `.csv` | ✓ | — |

## Testing

```bash
pytest tests/ -v                    # default suite (2801 passing, 544 skipped)
pytest tests/ -v -m golden          # GEMMA / GAPIT / GWASpoly golden-data gate
pytest tests/ -v -m external        # Pillar B external reference-tool comparisons
pytest tests/ -v -m cli_matrix      # Pillar C end-to-end CLI smoke matrix
pytest tests/ -v -m reproducibility # Pillar D fixture-drift audit
pytest tests/ -v -m "not slow"      # skip slow tests
```

The `external`, `cli_matrix`, and `reproducibility` markers are opt-in (the
default suite skips them) and run on cron schedules in CI:

| Workflow | Trigger | What |
|---|---|---|
| `.github/workflows/ci.yml` | every PR + master push | default suite + golden + native + no-openmp + type-check |
| `.github/workflows/perf.yml` | every PR + master push | native-kernel wall-time regression gate (> 10% slower = fail) |
| `.github/workflows/cli-matrix.yml` | nightly | 122-cell CLI smoke (40 subcommands × formats × CPU + GPU) |
| `.github/workflows/external.yml` | weekly (Sunday) | Pillar B reference-tool comparisons (PLINK / LDSC / TwoSampleMR / regenie / SAIGE / BOLT / GEMMA / GAPIT / GWASpoly / SoyNAM) |
| `.github/workflows/reproducibility.yml` | monthly (1st) | fixture-drift audit |

Golden-data fixtures live under `tests/golden/` and are regenerated with
`scripts/generate_golden_data.py` (see `docs/validation.md`). The validation
campaign findings ledger lives at `docs/validation_findings.md`.

### Memory regression nets

Streaming-friendly scans are protected from accidental re-materialization
by `tracemalloc`-based peak-memory assertions in
`tests/test_streaming_memory.py` (42 tests across the streamed
subcommands). The biobank-relevant invariant is that peak memory scales
with `chunk_size` / `window_size`, not with total `m`.

## Requirements

Python ≥ 3.10 and < 3.13, PyTorch ≥ 2.0 and < 2.5, NumPy ≥ 1.24 and < 2.0,
pandas ≥ 2.0, SciPy ≥ 1.10, matplotlib ≥ 3.7, requests ≥ 2.28.

Optional: `zarr`, `h5py`, `pyarrow`, `seaborn`. A C++ compiler with pybind11 is
optional; if unavailable, every native path falls back to the pure-torch
reference implementation (`TORCHGENOMICS_DISABLE_NATIVE=1` forces this).

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{torchgenomics2026,
  title  = {TorchGenomics: GPU-accelerated Genome-Wide Association Studies with PyTorch},
  author = {Atanda, Sikiru A.},
  year   = {2026},
  url    = {https://github.com/sikiru-atanda/torchgenomics},
  note   = {Version 0.3.8},
}
```
