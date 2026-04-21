# CLI Reference

TorchGWAS ships a single entry point, `torchgwas`, with 35 subcommands. Every
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
