## Background

Biobank-scale GWAS now interrogates 10^5-10^7 variants across 10^5-10^6 samples, yet the analytic stack remains fragmented across single-purpose tools - variant mixed models, haplotype tests, multi-omics colocalization, polyploid scans, and biobank-streaming I/O each live in a different code base, with incompatible formats, hard-coded ploidies, and CPU-only kernels.

## Results

TorchGWAS is a Python/PyTorch library unifying 121 GWAS capabilities across 11 functional clusters - variant scans, haplotype tests, multi-omics integration, polyploid pipelines, specialty mixed models (survival, random regression, threshold-linear, within-family, knockoff, OCF, GU, LRO), GLM/GLMM families, multi-environment/multi-trait extensions, post-GWAS (PGS, MR, LDSC, fine-mapping), multiple-testing, visualization, and I/O - on a GPU-portable tensor backbone. Equivalence is exercised by 11 fully-instrumented harnesses against 15 reference tools (GEMMA, GAPIT, GWASpoly, PLINK 2.0, LDSC, regenie, SAIGE, BOLT-LMM, TwoSampleMR, SoyNAM, MetaXcan, SMR, coloc/hyprcoloc, BLUPF90+, coxme), passing 40 of 45 numerical checks (5 deviations are post-V1, ledgered, and bounded). At p = 10^6 variants the streaming scan loop reduces peak materialized memory 8.7-fold (924 MB vs 8.05 GB), with log-log scaling slopes 0.020 (streaming) versus 0.933 (materialized). Twenty-four pybind11 C++ kernels with device-aware dispatch deliver per-kernel speedups from 1.0x to 9,210x (median 93.5x). A SORT1/LDL multi-omics worked example reproduces a Pearson z_eQTL-z_GWAS correlation of -0.97 across 48 aligned variants.

## Conclusions

TorchGWAS is open-source, version-pinned, and end-to-end reproducible from one shell command; every numerical claim above is regenerable through the validation-findings ledger and an eight-stage orchestrator.

## Availability

github.com/sikiru-atanda/torchgwas; pip install torchgwas.

## Contact

sikiruandfriends@gmail.com
