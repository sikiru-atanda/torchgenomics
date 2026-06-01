## Background

Biobank-scale GWAS now interrogates 10⁵–10⁷ variants across 10⁵–10⁶ samples, yet the analytic stack remains fragmented across single-purpose tools — variant mixed models, haplotype tests, multi-omics colocalization, polyploid scans, and biobank-streaming I/O each live in a different code base, with incompatible formats, hard-coded ploidies, and CPU-only kernels.

## Results

TorchGWAS is a Python/PyTorch library unifying 121 GWAS capabilities across 11 functional clusters — variant scans, haplotype tests, multi-omics integration, polyploid pipelines, specialty mixed models, GLM/GLMM families, multi-environment/multi-trait extensions, post-GWAS (PGS, MR, LDSC, fine-mapping), multiple-testing, visualization, and I/O — on a GPU-portable tensor backbone. The post-GWAS layer ships three TWAS workflows (S-PrediXcan sumstat, PrediXcan individual-level, FUSION measured-expression with OLS / kinship-corrected LMM) plus a top-level `combine_gwas_twas` entry point with **14 gene-level combination methods**: eight classical kernels (Fisher, Brown, Empirical Brown, harmonic-mean p, truncated product, min-p, Cauchy/ACAT, Stouffer) and six novel methods (R²-weighted Stouffer, LD-aware Brown via eigenMT, polyploid gene-action Fisher, multi-tissue ACAT + GWAS lead-SNP, conditional GWAS+TWAS via COJO, hyprcoloc-PPFC-gated). Equivalence is exercised against **17 reference tools** (GEMMA, GAPIT, GWASpoly, PLINK 2.0, LDSC, regenie, SAIGE, BOLT-LMM, TwoSampleMR, MetaXcan, SMR v1.3.1, coloc/hyprcoloc, FUSION measured-expression, CRAN `metap`, Bioconductor `EmpiricalBrownsMethod`, BLUPF90+, coxme), with every check passing at observed-then-floored tolerances after the four documented F3 statistical fixes landed on master. At p = 10⁶ variants the streaming scan loop reduces peak materialized memory 8.7-fold (924 MB vs 8.05 GB), with log-log scaling slopes 0.020 (streaming) versus 0.933 (materialized). Thirty-one pybind11 C++ kernels deliver per-kernel speedups of 1.0×–20,183×; a Cholesky cache in the SuSiE-RSS ELBO drops end-to-end fit_rss at p=2,000 from 5.8s to 0.53s. A SORT1/LDL multi-omics worked example reproduces Pearson z_eQTL–z_GWAS = −0.97 across 48 aligned variants.

## Conclusions

TorchGWAS is open-source, version-pinned, and end-to-end reproducible from one shell command; every numerical claim above is regenerable through the validation-findings ledger and an eight-stage orchestrator.

## Availability

github.com/sikiru-atanda/torchgwas; pip install torchgwas.

## Contact

sikiruandfriends@gmail.com
