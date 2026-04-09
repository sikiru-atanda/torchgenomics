# TorchGWAS

**GPU-accelerated Genome-Wide Association Studies with PyTorch**

TorchGWAS is a modular Python library that brings GPU acceleration to GWAS pipelines via PyTorch. It replicates and extends established tools like GEMMA and GAPIT, achieving 4th-decimal-place p-value agreement while adding novel statistical models, native C++ accelerators, and support for both diploid and polyploid organisms.

## Features

### GWAS Models
- **Classical**: GLM, single-trait LMM, multi-trait LMM (MVLMM), FarmCPU, BLINK
- **Multi-environment**: MET (reaction-norm, FA(k), multi-kernel), MT-MET (Kronecker separable)
- **Novel LMMs**: Multi-kernel, GxE/heteroscedastic, orthogonal cross-fit (DML), knockoff FDR-controlled, genotype-uncertainty, leave-region-out LOCO
- **Categorical traits**: Binary/ordinal/multinomial GLM and GLMM (PQL, SPA), multi-environment GLMM
- **Survival**: Cox PH frailty model with Breslow-Clayton PQL and SPACox SPA
- **Longitudinal**: Random regression LMM with Legendre/B-spline bases, spatio-temporal 2D P-spline, RR-MET
- **Haplotype-based**: HTR, block, window, SKAT, plus novel methods (PCHT, HHCT, HSKAT, HapGxE, BayesHap), multi-env/multi-trait haplotype GWAS
- **Fine-mapping**: SuSiE (IBSS) and CAVI Bayesian variable selection
- **LD-conditional**: COJO-style stepwise conditioning with persistence metrics
- **Within-family**: Dual-scan attenuation diagnostics for confounding detection
- **Threshold-linear**: Bermann et al. 2026 for mixed ordinal + continuous traits

### Polyploid Support
- Arbitrary ploidy (diploid through hexaploid+)
- Gene-action models: additive, dominance (all orders), diplo-additive, overdominant
- HWE testing with double-reduction correction for autopolyploids

### LD Block Detection
- 13 methods across 4 categories (classical, literature, novel, diagnostic)
- PLINK 1.9 `--blocks` validated (Jaccard = 0.78 for Gabriel)
- Diploid and polyploid compatible

### Post-GWAS
- **Heritability**: LDSC h2/rg, S-LDSC partitioned, HESS regional
- **Meta-analysis**: IVW, DerSimonian-Laird, Stouffer, RE2, MR-MEGA, MANTRA
- **Colocalization**: Giambartolomei coloc, HyPrColoc multi-trait
- **Mendelian randomization**: IVW, MR-Egger, weighted median, MR-PRESSO
- **SMR/HEIDI** and **TWAS** (S-PrediXcan / PrediXcan)
- **Gene-set enrichment**: MAGMA-style SNP-to-gene + competitive enrichment
- **LD clumping**, **fine-mapping utilities**, **power analysis**, **winner's curse correction**

### Polygenic Score Construction
- C+T (clumping and thresholding)
- LDpred2 (infinitesimal, grid, auto)
- PRS-CS (continuous shrinkage)
- Allele harmonization, individual scoring, PGS validation metrics

### Visualization
- Manhattan plot (linear and Circos-style polar)
- Miami plot (two-trait comparison)
- QQ plot with genomic inflation factor
- Haploview-style LD triangle heatmap
- Trumpet plot (AF vs effect size with power curves)

### Performance
- 26 native C++ accelerators via pybind11 (up to 12,000x speedup)
- OpenMP parallelization for hot-loop kernels
- GPU kernels for imputation (mode, KNN, LD-based)
- Three-tier dispatch: GPU > native C++ > Python (automatic fallback)
- AMP (FP16/BF16) for I/O and GRM; FP64 for statistical inference

### Multiple Testing
- Bonferroni, Holm, BH, BY, Storey q-value
- simpleM / M_eff, GPU-accelerated permutation
- IHW (covariate-adaptive weighting), AdaPT (covariate-adaptive thresholding)
- Cauchy combination, local FDR, weighted FDR

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sikiru-atanda/torchgwas.git
cd torchgwas
pip install -e ".[dev]"
```

### With optional dependencies

```bash
pip install -e ".[all]"     # zarr, h5py, pyarrow, seaborn
pip install -e ".[dev]"     # all + pytest, ruff, mypy
```

### Docker

```bash
# CPU only
docker build -t torchgwas .

# With CUDA support
docker build -t torchgwas:gpu --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime .
```

## Quick Start

```python
import torch
from torchgwas.io import read_plink
from torchgwas.preprocess import preprocess_genotypes
from torchgwas.linalg import compute_grm, eigendecompose
from torchgwas.models import SingleTraitLMM
from torchgwas.scan import UnifiedScanner

# Load data
geno, pheno, covar = read_plink("data.bed", "pheno.txt")

# Preprocess
G, variant_qc = preprocess_genotypes(geno, maf_threshold=0.05)

# Compute kinship and eigendecompose
K = compute_grm(G)
eig = eigendecompose(K)

# Fit null model and scan
model = SingleTraitLMM()
model.fit_null(pheno, covar, eigendecomposition=eig)

scanner = UnifiedScanner(model)
results = scanner.scan(G)

# Multiple testing correction
from torchgwas.stats import apply_correction
results = apply_correction(results, method="bh")
```

## CLI Usage

TorchGWAS provides 32 CLI subcommands covering the full GWAS pipeline:

```bash
# Validate input files
torchgwas validate --genotype data.bed --phenotype pheno.txt

# Standard LMM scan
torchgwas lmm-scan --genotype data.bed --phenotype pheno.txt --test wald --correction bh

# Multi-trait scan
torchgwas mvlmm-scan --genotype data.bed --phenotype pheno.txt --traits Y1,Y2,Y3

# Polyploid scan
torchgwas poly-scan --genotype data.csv --phenotype pheno.txt --ploidy 4 --gene-action all

# Multi-environment GWAS
torchgwas met-scan --genotype data.bed --phenotype pheno_env.txt --vg-structure "fa(2)"

# GLMM (binary trait, SAIGE-style)
torchgwas glmm-scan --genotype data.bed --phenotype pheno.txt --family binary

# Survival GWAS
torchgwas survival-scan --genotype data.bed --phenotype pheno_surv.txt

# LD block detection
torchgwas ld-blocks --genotype data.bed --method gabriel

# Polygenic scores
torchgwas pgs-fit --genotype data.bed --sumstats gwas_results.tsv --method ldpred2-auto
torchgwas pgs-score --weights weights.pt --genotype target.bed --output scores.tsv

# Full pipeline
torchgwas pipeline --genotype data.vcf.gz --impute beagle --model lmm --test wald
```

## Architecture

```
torchgwas/
  io/          # Format detection & reading (PLINK, VCF, BGEN, HapMap, Zarr, CSV)
  preprocess/  # Imputation, QC, standardization, polyploid encoding
  linalg/      # GRM, eigendecomposition, batched Cholesky
  models/      # All GWAS models (BaseModel protocol)
  scan/        # UnifiedScanner streaming chunks through any model
  stats/       # Multiple testing, SPA, PVE
  ld/          # LD block detection (13 methods), pairwise LD
  optim/       # 6-mode optimizer stack (PX-EM -> AI-REML -> ... -> rescue)
  pgs/         # Polygenic score construction & validation
  postgwas/    # LDSC, meta-analysis, coloc, MR, TWAS, enrichment
  viz/         # Manhattan, QQ, Miami, Circos, Haploview, trumpet plots
  _native/     # 26 C++ pybind11 accelerators + GPU kernels
  cli.py       # 32 CLI subcommands
```

## Data Flow

```
Format detection -> Imputation/phasing -> QC/preprocessing -> Genotype encoding
-> GRM -> Eigendecomposition -> Null fitting -> Scan loop -> Multiple testing -> Reporting
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/ -v -m golden    # Reference validation tests
pytest tests/ -v -m "not slow" # Skip slow tests
```

2102 tests pass across all 47 development phases.

## Supported Genotype Formats

| Format | Extension | Read | Write |
|--------|-----------|------|-------|
| PLINK BED | .bed/.bim/.fam | Yes | Yes |
| PLINK2 PGEN | .pgen/.pvar/.psam | Yes | - |
| VCF/BCF | .vcf/.vcf.gz/.bcf | Yes | - |
| BGEN | .bgen | Yes | - |
| HapMap | .hmp.txt | Yes | - |
| Zarr | .zarr | Yes | Yes |
| CSV Dosage | .csv | Yes | - |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.24
- pandas >= 2.0
- SciPy >= 1.10
- matplotlib >= 3.7

Optional: zarr, h5py, pyarrow, seaborn

A C++ compiler (with pybind11) is optional but recommended for native accelerators.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use TorchGWAS in your research, please cite:

```bibtex
@software{torchgwas2026,
  title={TorchGWAS: GPU-accelerated Genome-Wide Association Studies with PyTorch},
  year={2026},
  url={https://github.com/sikiru-atanda/torchgwas}
}
```
