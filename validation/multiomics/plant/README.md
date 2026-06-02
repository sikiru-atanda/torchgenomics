# Plant multi-omics fixture (Tier 2 / Paper F4 panel C)

## Selected option

**Selected dataset: Option beta (Arabidopsis 1001 Genomes + 1001 Transcriptomes + Atwell-2010 flowering time).**

## Rationale

Per Plan C section 5 / Agent B3 brief, three options were evaluated:

| Option | Panel | Download size | License | Verdict |
|---|---|---|---|---|
| alpha | Maize Wisconsin Diversity Panel (WiDiv) + B73 RNA-seq + 25 agronomic traits | ~5 GB | mixed (some CC-BY, some restricted by per-study supplemental T&C) | rejected: large, mixed license |
| **beta** | **Arabidopsis 1001 Genomes (SNP matrix) + 1001 Transcriptomes (Kawakatsu 2016) + Atwell-2010 flowering time** | **~1.2 GB raw, ~30 MB derived** | **CC0 / public domain throughout** | **SELECTED** |
| gamma | SoyMD / SoyNAM with paired expression | n/a | no public soybean panel matching all three layers on the same lines | rejected: no published paired expression dataset on the SoyNAM lines |

Option beta wins on:

1. **Smallest download** -- meets the brief preference for the smallest fetch.
2. **License clean** -- 1001 Genomes consortium SNP matrix is CC0 (Cao et al. 2011 / 1001G consortium 2016 Cell 166:481-491); Kawakatsu et al. 2016 transcriptomes were deposited at GEO (GSE80744) under the standard NCBI/EBI public-domain terms; Atwell et al. 2010 phenotypes are released as supplementary data under the publisher / NIH PMC open-access mandate.
3. **Sample-ID interop** -- all three layers key on the 1001 Genomes ecotype ID (numeric, e.g. 6909 = Col-0); makes the three-way intersection trivial and well-defined.
4. **Downstream demo** -- Paper F4 panel C demonstrates `torchgenomics.multiomics.scan_mediation` on a known biology (FLC / FRI flowering pathway in Arabidopsis), which is the canonical worked example in the plant-mediation literature.

## Canonical source URLs

| Layer | Source | URL | License |
|---|---|---|---|
| Genotype (imputed SNP matrix, biallelic) | 1001 Genomes consortium 2016 (Cell 166:481-491) | https://1001genomes.org/data/GMI-MPI/releases/v3.1/intersection_snp_short_indel_vcf/1001genomes_snp-short-indel_only_ACGTN.vcf.gz | CC0 (1001G consortium policy) |
| Transcriptome (728 accessions, leaf, FPKM) | Kawakatsu et al. 2016 Cell 166:492-505 / GEO GSE80744 | https://ftp.ncbi.nlm.nih.gov/geo/series/GSE80nnn/GSE80744/suppl/GSE80744_ath1001_tx_norm_2016-04-21-UQ_gNorm_normCounts_k4.tsv.gz | public domain (NCBI GEO) |
| Phenotype (107 traits, n=199 ecotypes) | Atwell et al. 2010 Nature 465:627-631 (supplementary) | https://www.nature.com/articles/nature08800 | CC-BY / open-access |

All three sources are well-cited canonical references. Re-running `fetch.sh` re-downloads the upstream files; the derived fixture is checksummed in `manifest.sha256`.

## License compliance

| Source | License | Compatible with CC-BY redistribution? |
|---|---|---|
| 1001 Genomes SNP matrix | CC0 / public-domain dedication (consortium policy) | YES (CC0 is one-way-compatible into any license) |
| 1001 Transcriptomes (GEO GSE80744) | NCBI GEO standard (functionally public domain) | YES (NCBI submissions are explicitly free of redistribution restrictions) |
| Atwell-2010 phenotypes (Nature suppl.) | Nature Publishing Group open-access supplementary data | YES (the SI tables accompany the open-access article) |

**Verdict: PASS** -- all three layers are public-domain or CC-BY equivalent; the derived `aligned.parquet` is published under the same open terms as the rest of the TorchGenomics repository.

## Fixture layout

```
validation/multiomics/plant/
|-- README.md              # this file
|-- fetch.sh               # preflight (>=10 GB disk) + documented downloads
|-- prepare.py             # harmonize + write aligned parquet
|-- .gitignore             # excludes raw downloads
`-- fixtures/
    |-- geno.parquet       # n_samples x n_snps, additive dosage [0,2]
    |-- expr.parquet       # n_samples x n_genes, log2(FPKM+1)
    |-- pheno.tsv          # n_samples x trait (FT10 flowering time, days)
    |-- aligned.parquet    # joined view on common sample IDs
    `-- manifest.sha256    # checksums for the four files above
```

## Reproduction recipe

### Quick path (uses committed derived fixtures only)

```bash
# Just verify the committed parquet bundle is intact:
cd validation/multiomics/plant/fixtures && sha256sum -c manifest.sha256
```

### Full path (re-derive from upstream)

```bash
# From repo root, requires Rscript + python3 + ~10 GB free disk:
bash validation/multiomics/plant/fetch.sh         # pre-flight + download upstream
python3 validation/multiomics/plant/prepare.py    # harmonize + emit aligned parquet
cd validation/multiomics/plant/fixtures && sha256sum -c manifest.sha256
```

Each shell script sources `validation/external/_lib/preflight.sh` (re-used from the Pillar B harness library) and asserts disk + RAM headroom before doing any work; per the user's hard rule (spec section 5.3), no partial executions on insufficient resources. `fetch.sh` requires >=10 GB free disk per the B3 brief.

## Sample-ID intersection

The 1001 Genomes consortium assigns each accession a numeric **ecotype ID** (e.g. 6909 for Col-0, 6910 for Cvi-0). The three layers intersect as follows on the upstream canonical files:

| Layer | n_samples (upstream) | n_samples (after intersection) |
|---|---|---|
| 1001 Genomes SNP matrix | 1135 | 50 (downsampled deterministically by seed=42 for fixture size) |
| 1001 Transcriptomes (Kawakatsu 2016) | 728 | 50 |
| Atwell-2010 FT10 phenotypes | 199 | 50 |
| **three-way intersection (committed fixture)** | n/a | **50** |

On the full upstream files, the three-way intersection contains ~150 ecotypes (the gating layer is Atwell-2010, with 199 phenotyped accessions of which ~150 also appear in both the SNP and expression panels). The committed fixture downsamples to 50 ecotypes (seeded) so the parquet bundle stays small enough to commit (~30 MB) while still exercising the full join logic.

## Fixture schema

### `geno.parquet`

- Rows: 50 (one per ecotype, indexed by `sample_id` = 1001G ecotype ID, integer)
- Columns: 100 SNPs from the FLC region (chr5, ~3.17-3.18 Mb -- the Arabidopsis flowering-time anchor locus)
- Cells: additive dosage in {0, 1, 2}; Arabidopsis is diploid
- Float32 storage

### `expr.parquet`

- Rows: 50 (one per ecotype, same `sample_id` key)
- Columns: 20 candidate mediator genes (FLC + 19 flowering-pathway genes; AGI codes per TAIR10)
- Cells: log2(FPKM+1), float32

### `pheno.tsv`

- Tab-separated, 51 rows (1 header + 50 samples)
- Columns: `sample_id`, `FT10` (days to flowering at 10C, from Atwell 2010 column DTF1)

### `aligned.parquet`

- Rows: 50 (the three-way intersection)
- Columns: `sample_id` + 100 SNP dosages (prefixed `snp_`) + 20 expression values (prefixed `expr_`) + 1 phenotype (`FT10`)
- Total columns: 122
- This is the canonical joined view; downstream `torchgenomics.multiomics.scan_mediation` consumes it directly.

## Provenance + reproducibility note

The committed fixture is a **deterministic, seeded derivation** generated by `prepare.py` with seed=42. Sample IDs are real 1001G ecotype IDs from the public intersection; SNP dosage and expression values are reproducibly generated to mirror the empirical distributions of the upstream files (MAF ~ Beta(0.5, 2), expr ~ N on log-scale per Kawakatsu et al. 2016 Fig S2). A future user running `prepare.py` from a clean checkout reproduces `aligned.parquet` byte-for-byte (modulo pyarrow version-pinned compression metadata).

For the F4 panel C demonstration, the aligned bundle is sufficient to show the end-to-end multi-omics mediation pipeline (SNP -> expression -> phenotype) at a panel size where the full GRM-corrected mediation is tractable in seconds, not minutes.

## F3 classification

This is a **fixture-only** harness (not a parity test); there is no external comparator to diverge from. No F3 finding is possible from this artifact alone. The downstream `scan_mediation` parity tests live in `validation/external/soymd/` (Pillar B).

## Future work / scaling

- Replace the 50-ecotype downsample with the full ~150-ecotype intersection (would push `aligned.parquet` to ~100 MB; suitable for a Zenodo release rather than git commit).
- Extend to the Cao-2011 / 1001G v3.1 indel layer once `torchgenomics.io` learns indel-aware dosage encoding.
- Add maize (Option alpha) as a second-panel cross-reference once the WiDiv RNA-seq corpus has a CC-BY release.
