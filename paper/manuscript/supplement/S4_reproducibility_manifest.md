# S4 — Reproducibility manifest (pinned versions)

Every reference-tool comparison in the TorchGWAS validation campaign is gated by a version-pinned installer at `validation/external/<tool>/install.sh` (or `validation/specialty/<model>/` for the internal harnesses). The pins below were extracted from the `<TOOL>_VERSION` / `<TOOL>_COMMIT` / `<TOOL>_TAG` / `<TOOL>_SHA256` constants in each installer at HEAD of the paper branch. Where an upstream tool is distributed only as a binary (BOLT-LMM, PLINK 2.0, REGENIE, SMR, GEMMA), the SHA-256 of the downloaded archive is recorded so future runs abort cleanly on bit-drift. Where an upstream tool is distributed as an R package or python source (coloc, hyprcoloc, TwoSampleMR, susieR, MetaXcan), the version is captured post-install into `<tool>/.install_marker`.

## A. Tool installer pins (15 external + 3 fixture)

| Tool | Version | Source | Checksum / Commit | Install entry-point |
|---|---|---|---|---|
| GEMMA | 0.98.5 (2021-08-25) | static AMD64 binary at `gemma_demo/gemma-0.98.5` | SHA256 `ad3f3f43a2f8a1c00e71fae8f43676614170e06c99fc766a947acfe45605969b` | `validation/external/gemma/install.sh` |
| GAPIT | 4.1.0 (resolves to 4.x via `jiabowang/GAPIT` GitHub) | `remotes::install_github("jiabowang/GAPIT")` | (per-build resolved; recorded in `bin/gapit_version.txt`) | `validation/external/gapit/install.sh` |
| GWASpoly | 2.14 (min pin 2.12) | `remotes::install_github("jendelman/GWASpoly")` | (per-build resolved; recorded in `bin/gwaspoly_version.txt`) | `validation/external/gwaspoly/install.sh` |
| PLINK 2.0 | v2.0.0-a.7.0LM (build 20260425) | `https://s3.amazonaws.com/plink2-assets/alpha7/plink2_linux_x86_64_20260425.zip` | SHA256 `e70a283aefe004122fca3e632ae0b24023a24635f98a8e768ea8d542bbc659a9` | `validation/external/plink2/install.sh` |
| LDSC | 1.0.1 (bioconda build `pyhdfd78af_2`) | `conda install -c bioconda ldsc=1.0.1=pyhdfd78af_2` | (bioconda build hash pinned in installer) | `validation/external/ldsc/install.sh` |
| REGENIE | v3.3 (Aug 2023) | `https://github.com/rgcgithub/regenie/releases/download/v3.3/regenie_v3.3.gz_x86_64_Linux.zip` | SHA256 `88fa48a93779321d1e1b585c006c6bb863de15380d61431ab8d20cfe5b31900d` | `validation/external/regenie/install.sh` |
| SAIGE | 1.4.4 (docker image `docker.io/wzhou88/saige:1.4.4`) | docker pull (or Bioconductor R fallback) | (docker tag is the pin) | `validation/external/saige/install.sh` |
| BOLT-LMM | v2.5 (Aug 2025) | `https://alkesgroup.broadinstitute.org/BOLT-LMM/downloads/BOLT-LMM_v2.5.tar.gz` | SHA256 `a5fdc49f79f42aa007282d9080e5b97c6ef2aab25ace2af306f432c44eedebca` | `validation/external/bolt_lmm/install.sh` (infra-blocker on RHEL 9.6 glibc 2.34) |
| TwoSampleMR | (HEAD via `remotes::install_github("MRCIEU/TwoSampleMR")`) | GitHub `MRCIEU/TwoSampleMR` | (per-build resolved; recorded in `.install_marker`) | `validation/external/twosamplemr/install.sh` |
| MR-PRESSO | (HEAD via `remotes::install_github("rondolab/MR-PRESSO")`) | GitHub `rondolab/MR-PRESSO` | (per-build resolved; recorded in `.install_marker`) | bundled with TwoSampleMR installer |
| coloc | 5.2.3 (CRAN; min pin 5.2 for Wallace 2020 H4 erratum) | CRAN | (per-build resolved; recorded in `.install_marker`) | `validation/external/coloc/install.sh` |
| hyprcoloc | 0.0.2 (jrs95/hyprcoloc @ commit `0348bbd`) | GitHub | commit `0348bbd`; RcppEigen pinned to `0.3.3.9.4` (CRAN archive) for Eigen-3.3 ABI | `validation/external/hyprcoloc/install.sh` |
| haplo.stats | 1.9.8.7 (CRAN, R 4.5.1) | CRAN | (per-build resolved; recorded in `.install_marker`) | `validation/external/hapref/install.sh` |
| SMR | 1.3.1 (released 2024-03-08, GCC 8.3 build) | `https://yanglab.westlake.edu.cn/software/smr/download/smr-1.3.1-linux-x86_64.zip` | SHA256 `4d779197a0b3399db36c9cdf7b4b4190ea40fa33a47253f5419ad27c3bce251e` | `validation/external/smr/install.sh` |
| MetaXcan / S-PrediXcan | tag v0.8.1, commit `964f1fdb5bf9585585690e85bb0eca7b67663ddb` | GitHub `hakyimlab/MetaXcan` | git commit pinned in installer | `validation/external/metaxcan/install.sh` |
| susieR | (HEAD via CRAN `install.packages("susieR")`) | CRAN | (per-build resolved) | `validation/external/susieR/install.sh` |
| SoyNAM + rrBLUP | (HEAD via CRAN) | CRAN | (per-build resolved; recorded in `.install_marker`) | `validation/external/soynam/install.sh` |
| SoyMD (`mediation` R + jsonlite) | (HEAD via CRAN; Tingley 2014 JSS) | CRAN | (per-build resolved; recorded in `.install_marker`) | `validation/external/soymd/install.sh` |

## B. Harness taxonomy

The 26 harnesses split into four classes by their reference and intent:

### B.1 — External tool harnesses (15)

Direct head-to-head comparison against an installed upstream tool that implements the same statistical test:

1. `gemma` (V1-core GWAS LMM; fixture-authoritative D0)
2. `gapit` (V1-core GLM/MLM/FarmCPU/BLINK; fixture-authoritative D3)
3. `gwaspoly` (polyploid LMM; fixture-authoritative D3)
4. `plink2` (LD-block `--blocks`; LD clumping)
5. `ldsc` (h2 / rg / S-LDSC)
6. `regenie` (LMM step1 + step2)
7. `saige` (GLMM at biobank scale; docker; infra-blocker fallback)
8. `bolt_lmm` (LMM at biobank scale; closed-source; documented infra-blocker on RHEL 9 glibc 2.34)
9. `twosamplemr` (MR-IVW / Egger / weighted-median / PRESSO)
10. `coloc` (Giambartolomei 2014 two-trait)
11. `hyprcoloc` (Foley 2021 multi-trait)
12. `hapref` (haplo.stats EM + GLM; haplotype GWAS reference)
13. `smr` (Yang lab SMR + HEIDI)
14. `metaxcan` (S-PrediXcan TWAS)
15. `susieR` (SuSiE-RSS fine-mapping)

### B.2 — Fixture harnesses (3)

The upstream is a data source (not a tool that re-runs the same analysis); reference outputs come from a published / curated panel:

1. `soynam` — Soybean Nested Association Mapping (`SoyNAM` R package; data + LMM reference for biobank-relevant ag-scale fixture)
2. `soymd` — Soybean Multi-omics Database fixture for `torchgwas.multiomics` mediation (reference: `mediation` R package, Tingley 2014 JSS)
3. `ukb` — UK Biobank biobank-scale validation (DUA-controlled; user supplies path via env vars; harness verifies but never downloads)

### B.3 — Specialty model harnesses (6 external + 2 internal-only)

Reference is either a small R / Python implementation (specialty) or an internal-consistency check (no external reference exists). All eight back F7 panels.

External:

1. `validation/specialty/survival/` — Cox PH frailty vs R `coxme`
2. `validation/specialty/rr/` — Random Regression LMM vs R `lme4`
3. `validation/specialty/family/` — Within-family LMM vs Young 2022 paper-OLS
4. `validation/specialty/threshold/` — Threshold-linear vs BLUPF90+ `gibbsf90+`
5. `validation/specialty/knockoff/` — Knockoff FDR vs R `knockoff` + `glmnet`
6. `validation/specialty/ocf/` — OCFLMM vs hand-coded DML2 (Chernozhukov 2018)

Internal-only:

7. `validation/specialty/gu/` — Genotype-Uncertainty LMM (Phase 28 internal consistency)
8. `validation/specialty/lro/` — Leave-Region-Out LMM (Phase 29 internal consistency)

## C. Pre-flight contract

Every installer + fetch + run script sources `validation/external/_lib/preflight.sh` and calls `preflight_check_with_data_size <tool> <data_gb> <peak_ram_gb>` BEFORE any download, install, or run. Per `memory/feedback_preflight.md` this is a hard rule: the script aborts with an explicit shortfall message on insufficient disk or RAM rather than partially execute. The library exposes:

- `preflight_check(<tool>, <disk_gb>, <ram_gb>)` — direct gate
- `preflight_check_with_data_size(<tool>, <data_gb>, <peak_ram_gb>)` — auto-applies multipliers per spec section 5.3 (disk = data_gb * 2 + working_set * 2; ram = peak * 1.5 + 4 GB headroom)
- `verify_checksum(<file>, <sha256>)` — aborts the run if the downloaded artefact mismatches the pinned SHA-256

## D. One-command reproducibility driver

The entire paper (every JSON, every figure, every supplement table) is reproducible from:

```bash
bash paper/reproducibility/reproduce_paper.sh
```

which orchestrates the eight stages in `paper/reproducibility/stages/`:

1. `01_install_references.sh` — runs every `install.sh` in section A
2. `02_stage_fixtures.sh` — stages every committed + computed fixture (SHA256-pinned)
3. `03_run_references.sh` — runs every reference tool against its fixture
4. `04_run_torchgwas.sh` — runs every TG-side counterpart
5. `05_run_streaming_bench.sh` — populates `bench/streaming_p_sweep.json` for F6 panel B
6. `06_run_native_bench.sh` — populates `bench/native_speedups_realistic.md` for F6 panel A
7. `07_run_multiomics.sh` — runs the multi-omics fixtures (sim + GTEx/UKB + plant) for F4
8. `08_render_figures.py` — re-renders F1-F7 from `manifest.json` + the raw `agreement.json` artefacts

The driver is idempotent: each stage short-circuits when its outputs are already present (governed by `.install_marker` / `.fetch_marker` / output file timestamps). Re-runs are typically minutes; cold runs (including all installs from scratch) are 4-6 hours of wall time, dominated by R-package compilation for hyprcoloc + TwoSampleMR + GAPIT and the SAIGE docker image pull.
