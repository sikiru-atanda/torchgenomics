# CLI Reference

TorchGWAS ships a single entry point, `torchgwas`, with 37 subcommands. Every
subcommand accepts `--help` for full argument listings.

```bash
torchgwas --help
torchgwas lmm-scan --help
```

## Data management

```bash
torchgwas validate --genotype data.bed --phenotype pheno.txt
torchgwas convert  --input data.vcf.gz --output data --format bed
torchgwas impute   --genotype data.bed --method mean --output imp.pt
torchgwas impute   --genotype data.vcf.gz --method beagle \
                   --ref-panel 1000G.vcf.gz --output imp.vcf.gz
torchgwas impute   --genotype data.vcf.gz --method li-stephens \
                   --ploidy 4 --output imp.pt
torchgwas impute   --genotype data.vcf.gz --method deep-learning \
                   --output imp.pt
```

## Polyploid allele dosage calling (Phase 55)

Convert polyploid VCF read counts (AD field) into posterior
`P(dosage=0..k)` tensors via the R package
[`updog`](https://cran.r-project.org/package=updog). See
`docs/getting-started/polyploid_dosage_call.md` for the full recipe.

```bash
torchgwas dosage-call --vcf calls.vcf.gz --output out/dcall \
                      --ploidy 4 --model norm --n-cores 4
```

```text
usage: torchgwas dosage-call [-h] --vcf VCF --output OUTPUT --ploidy PLOIDY
                             [--model MODEL] [--rscript RSCRIPT]
                             [--bias | --no-bias] [--od | --no-od]
                             [--seq-error SEQ_ERROR] [--n-cores N_CORES]
                             [--keep-tmpdir]

options:
  -h, --help            show this help message and exit
  --vcf VCF             Input VCF with AD format field (biallelic)
  --output OUTPUT       Output prefix for .probs.pt / .meta.json /
                        .snp_diag.tsv
  --ploidy PLOIDY       Organism ploidy (2..8)
  --model MODEL         updog flexdog model name (default: norm)
  --rscript RSCRIPT     Path to Rscript (default: PATH lookup)
  --bias                Estimate allele bias (default)
  --no-bias             Fix allele bias to 1
  --od                  Estimate overdispersion (default)
  --no-od               Fix overdispersion to 0
  --seq-error SEQ_ERROR
                        Fix sequencing error rate (default: estimate)
  --n-cores N_CORES     Parallelism passed to updog::multidog (default: 1)
  --keep-tmpdir         Skip tempdir cleanup (debug aid)
```

## Polyploid F1 phasing (Phase 56)

Phase a polyploid F1 population using
[PolyOrigin](https://github.com/chaozhi/PolyOrigin.jl) (Zheng et al. 2021).
Reads posterior dosage probabilities from `dosage-call` and a pedigree TSV,
runs PolyOrigin via Julia (auto-installed on first use if absent), and writes
a haplotype tensor plus per-marker recombination estimates.

```bash
torchgwas phase-poly --probs out/dcall.probs.pt --pedigree ped.tsv \
                     --map markers.tsv --output out/phased --ploidy 4
```

```text
usage: torchgwas phase-poly [-h] --probs PROBS --pedigree PEDIGREE --map MAP
                            --output OUTPUT --ploidy {2,4,6}
                            [--parent-phased PARENT_PHASED] [--no-refinemap]
                            [--recomrate RECOMRATE] [--julia-path JULIA_PATH]
                            [--auto-install | --no-auto-install]
                            [--keep-workdir]

options:
  -h, --help            show this help message and exit
  --probs PROBS         Path to .probs.pt from 'dosage-call' (raw (n,m,k+1)
                        tensor or dict). If a raw tensor, the sibling
                        <prefix>.meta.json is read for sample/variant IDs.
  --pedigree PEDIGREE   TSV with columns: offspring, parent1, parent2
                        (optional: ploidy)
  --map MAP             TSV with columns: marker, chrom, pos_bp (optional: cm)
  --output OUTPUT       Output prefix; writes <prefix>.haplotypes.pt etc.
  --ploidy {2,4,6}      Organism ploidy (2, 4, or 6)
  --parent-phased PARENT_PHASED
                        Optional CSV of pre-phased parent genotypes (escape
                        hatch)
  --no-refinemap        Disable PolyOrigin's map refinement (default: enabled)
  --recomrate RECOMRATE
                        cM/Mb to synthesize genetic positions when --map lacks
                        a cm column (default: 1.0)
  --julia-path JULIA_PATH
                        Path to an existing Julia binary; else discovered or
                        auto-installed
  --auto-install        Auto-install Julia if not found (non-interactive; sets
                        consent=True)
  --no-auto-install     Refuse to auto-install Julia (non-interactive; sets
                        consent=False)
  --keep-workdir        Do not delete the temp work directory after success
                        (debug aid)
```

## GWAS scans

```bash
torchgwas glm-scan      --genotype data.bed --phenotype pheno.txt
torchgwas lmm-scan      --genotype data.bed --phenotype pheno.txt \
                        --test wald --correction bh
torchgwas mvlmm-scan    --genotype data.bed --phenotype pheno.txt \
                        --traits Y1,Y2,Y3
torchgwas poly-scan     --genotype data.csv --phenotype pheno.txt \
                        --ploidy 4 --gene-action all
torchgwas mklmm-scan    --genotype data.bed --phenotype pheno.txt \
                        --kernels additive,dominance
torchgwas gxe-scan      --genotype data.bed --phenotype pheno.txt \
                        --env env.tsv --gxe-model het
torchgwas set-scan      --genotype data.bed --phenotype pheno.txt \
                        --regions genes.bed --set-test skat
torchgwas bayes-scan    --genotype data.bed --phenotype pheno.txt \
                        --method susie --n-signals 10
torchgwas farmcpu-scan  --genotype data.bed --phenotype pheno.txt
torchgwas blink-scan    --genotype data.bed --phenotype pheno.txt
```

## `bayes-scan-rss`

SuSiE-RSS fine-mapping on summary statistics + LD reference. Per Zou et al.
2022 PLOS Genet 18(7):e1010299. Memory bound is per-locus `O(p_block_max^2)`,
typically ~200 MB per LD block at p_block ≤ 5000; vs `bayes-scan` which is
`O(n × p)` and materializes the full genotype matrix.

```bash
torchgwas bayes-scan-rss \
    --sumstats hits.tsv \
    --ld-ref ld_chr22.pt \
    --max-num-causal 10 \
    --coverage 0.95 \
    --purity 0.5 \
    --output out/finemap.tsv
```

### Required arguments

- `--sumstats PATH` — TSV with columns SNP, CHR, BP, A1, A2 + (BETA + SE + N) or Z + N.
- One of:
  - `--ld-ref PATH` — pre-built LD reference file (`.pt` or `.npz`).
  - `--geno PATH` — genotype panel (BED/PGEN); LD computed in-sample per locus. *Tier B — not yet wired.*
- `--output PATH` — output TSV path.

### Optional arguments

- `--regions PATH` — TSV with start, stop columns for explicit block boundaries.
  Default: auto-detect (fixed-size tiling at the block-size threshold; full
  ldetect-based detection requires --geno mode which is Tier B).
- `--max-num-causal INT` — Number of single-effect layers (L). Default: 10.
- `--coverage FLOAT` — Credible-set coverage threshold. Default: 0.95.
- `--purity FLOAT` — Minimum |R_jk| within a credible set. Default: 0.5
  (matches susieR `min_abs_corr`).
- `--prior-pi VALUE` — Scalar (e.g., `0.01`) OR path to a per-SNP prior
  TSV file (column `PRIOR_PI`). D3 shim from Phase 59 PolyFun spec.
- `--block-size-threshold INT` — Block-decomposition trigger. Default: 5000.
- `--threads INT` — Number of CPU threads. Default: 4.

### Output schema (PolyFun-compatible)

```
SNP   CHR   BP   A1   A2   Z   N   PIP   BETA_MEAN   BETA_SD   CREDIBLE_SET
```

`CREDIBLE_SET` = integer index of the credible set the variant belongs to
(0 = not in any credible set).

## Multi-environment & MT-MET

```bash
torchgwas met-scan   --genotype data.bed --phenotype pheno_env.txt \
                     --parameterization reaction_norm
torchgwas met-scan   --genotype data.bed --phenotype pheno_env.txt \
                     --vg-structure "fa(2)"
torchgwas mtmet-scan --genotype data.bed --phenotype pheno.txt \
                     --traits Y1,Y2 --env-cols E1,E2
```

## Threshold-linear / within-family / conditional

```bash
torchgwas threshold-scan   --genotype data.bed --phenotype pheno.txt \
                           --trait-types ordinal,continuous \
                           --n-categories 3,0
torchgwas family-scan      --genotype data.bed --phenotype pheno.txt \
                           --family-col FID
torchgwas conditional-scan --genotype data.bed --phenotype pheno.txt \
                           --ld-method r2
```

## Novel LMMs

```bash
torchgwas ocf-scan       --genotype data.bed --phenotype pheno.txt --n-folds 5
torchgwas knockoff-scan  --genotype data.bed --phenotype pheno.txt --fdr-level 0.05
torchgwas gu-scan        --genotype data.bed --phenotype pheno.txt \
                         --dosage-var dosage_var.pt
torchgwas lro-scan       --genotype data.bed --phenotype pheno.txt --ld-method r2
```

## Categorical traits (GLM / GLMM)

```bash
torchgwas glm-scan  --family binary      --firth
torchgwas glm-scan  --family ordinal     --n-categories 3
torchgwas glm-scan  --family multinomial --n-categories 4
torchgwas glmm-scan --family binary
torchgwas glmm-scan --family ordinal     --n-categories 3
torchgwas glmm-scan --family multinomial --n-categories 4
```

## Survival

```bash
torchgwas survival-scan --genotype data.bed --phenotype pheno_surv.txt
```

## LD block detection

```bash
torchgwas ld-blocks --genotype data.bed --method gabriel --output blocks
torchgwas ld-blocks --genotype data.bed --method big_ld --r2-threshold 0.5
torchgwas ld-blocks --genotype data.bed --method cc_graph
torchgwas ld-blocks --genotype data.bed --method changepoint
torchgwas ld-blocks --genotype data.bed --method graphical --l1-penalty 0.1
```

## Random regression (longitudinal)

```bash
torchgwas rr-scan     --genotype data.bed --phenotype pheno_long.txt \
                      --basis legendre --order 2
torchgwas rr-met-scan --genotype data.bed --phenotype pheno_long.txt \
                      --env-col ENV --vg-structure separable
```

## Polygenic scores

```bash
torchgwas pgs-fit   --sumstats ss.tsv --ld-ref ld.pt \
                    --method ldpred2-auto --output weights.tsv
torchgwas pgs-score --weights weights.tsv --genotype target.bed \
                    --output scores.tsv
```

## Multi-omics (mediation)

```bash
torchgwas mediate      --y pheno.npy --snp snp.npy \
                       --mediator mediator.npy --kinship K.npy
torchgwas mediate-scan --y pheno.npy --genotype data.bed \
                       --mediator-matrix mediators.npy \
                       --kinship K.npy --fdr bh
```

## NCBI annotation

```bash
torchgwas annotate --sumstats hits.tsv --crop maize \
                   --p-threshold 5e-8 --window-up 50000 --window-down 50000
```

## Full pipeline

```bash
torchgwas pipeline --genotype data.vcf.gz --impute beagle \
                   --model lmm --test wald
torchgwas pipeline --genotype data.bed --phenotype pheno.txt \
                   --model met --env-cols E1,E2,E3
```
