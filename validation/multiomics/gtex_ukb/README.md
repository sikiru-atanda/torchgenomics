# Tier 2 B2: GTEx v8 Liver eQTL x UKB-derived LDL GWAS (SORT1 locus)

Real-data multi-omics worked example for **F4 panel B** of the
TorchGenomics Genome Biology methods paper. Stages two public summary
statistics resources, slices them at the **SORT1 / chr1p13** cis
window, harmonizes them onto a common GRCh38 SNP set, and writes a
reproducible aligned.parquet fixture with full provenance.

## Selected locus

**SORT1 (sortilin 1)** on chr1p13.3 -- Ensembl ID ENSG00000134243,
GRCh38 gene body chr1:109,309,568-109,397,951. Cis window used:
chr1:108,700,000-110,000,000 (gene body +/- ~500 kb, which contains
the full set of GTEx v8 Liver signif variants for SORT1).


### Rationale

SORT1 is the canonical strong-signal locus on chr1p13 for the
cis-eQTL x LDL-cholesterol axis. Three reasons to pick it for the
F4 panel B worked example over the brief alternatives (LIPA, APOE):

1. **Effect size.** In GTEx v8 Liver the SORT1 lead variant
   (rs12740374, chr1:109,274,968 G>T b38) has the strongest cis-eQTL
   in the entire liver transcriptome among LDL/CAD-implicated genes:
   slope = 1.273, p_nominal = 3.09e-54, qval = 3.22e-45
   (from Liver.v8.egenes.txt.gz).

   Probed alternatives in the same Liver egenes file (this work,
   2026-05-15):


   | Gene | p_nominal | qval | slope |
   | --- | --- | --- | --- |
   | SORT1 | 3.09e-54 | 3.22e-45 | +1.273 |
   | CELSR2 | 9.43e-34 | 2.21e-27 | +1.039 |
   | PCSK9 | 6.02e-07 | 4.38e-03 | +0.359 |
   | LIPA  | 4.51e-05 | 0.104   | +0.175 |
   | APOE  | 4.65e-05 | 0.126   | +0.217 |
   | APOB  | 1.90e-04 | 0.240   | -0.600 |
   | HMGCR | 5.98e-03 | 0.465   | +0.284 |

2. **Causal biology established in literature.** Musunuru et al.,
   *Nature* 2010 (PMID 20686566) established SORT1 as the causal gene
   at the chr1p13 LDL locus via reporter assays and mouse work;
   rs12740374 is the canonical regulatory variant. Textbook
   cis-eQTL -> GWAS coloc example.

3. **Paper alignment.** The dispatch brief lists SORT1 as the
   first-choice locus ("SORT1 / LDL") in
   docs/superpowers/specs/2026-05-15-genome-biology-paper-design.md
   line 184: "GTEx v8 liver + UKB LDL sumstats (or equivalently
   strong locus, e.g. SORT1)". SORT1 is feasible, so we honor the
   first choice rather than fall back.

## Data sources

### Source 1 -- GTEx v8 single-tissue cis-eQTL (Liver)

- **Project**: GTEx Consortium, v8 release
- **Reference**: GTEx Consortium, *Science* 2020 (PMID 32913098)
- **URL** (raw tar, 1.56 GB):
  https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar
- **File used** (extracted from tar):
  GTEx_Analysis_v8_eQTL/Liver.v8.signif_variant_gene_pairs.txt.gz
  GTEx_Analysis_v8_eQTL/Liver.v8.egenes.txt.gz
- **Coordinate frame**: GRCh38 (b38)
- **Effect direction**: slope is the effect of the alt allele on
  normalized expression (TPM, inverse-normal transformed per the GTEx
  v8 pipeline; see GTEx v8 methods).
- **License**: Open access (CC0); GTEx Portal data use policy, public
  domain summary statistics. See
  https://www.gtexportal.org/home/protectedDataAccess and
  https://www.gtexportal.org/home/datasets

### Source 2 -- UKB-derived LDL-cholesterol GWAS (GLGC 2021)

- **Project**: Global Lipids Genetics Consortium (GLGC), 2021 release
- **Reference**: Graham et al., *Nature* 2021 (PMID 34887591); also
  GWAS Catalog accession GCST90239658
- **URL** (harmonised TSV, 1.42 GB):
  https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90239001-GCST90240000/GCST90239658/harmonised/GCST90239658.h.tsv.gz
- **Coordinate frame**: GRCh38, EBI-harmonized (matches GTEx natively;
  no liftover required).
- **UKB inclusion**: This GLGC 2021 LDL-C release is the canonical
  trans-ancestry meta-analysis of LDL-C in n = 1,654,960 participants
  across 201 cohorts. **UK Biobank Biomarker Biochemistry**
  contributes the largest single-study sample (n_UKB ~ 357K of n_total
  ~ 1.65M) and is explicitly listed in Supplementary Table 1 of Graham
  2021. The brief at
  docs/superpowers/plans/2026-05-15-paper-tier123-agent-briefs.md
  §5 B2 allows "UKB-style lipid trait, or whichever public sumstats";
  GLGC 2021 is the strongest UKB-inclusive LDL release available
  publicly without UKB application access.
- **Effect direction**: beta is in normalised LDL-C units
  (inverse-rank-transformed per GLGC analysis plan; see Graham 2021
  Methods).
- **License**: CC0 / open (GWAS Catalog harmonised summary statistics
  policy). https://www.ebi.ac.uk/gwas/docs/methods/summary-statistics

### Why GLGC 2021 rather than Pan-UKB or Neale lab v3?

Both Pan-UKB and Neale-lab v3 (the brief preferred sources) ship LDL
sumstats in **GRCh37** coordinates, while GTEx v8 is **GRCh38**.
Harmonising the two requires a liftover step. The B2 dispatch
environment has neither liftOver, pyliftover, nor the Bioconductor
rtracklayer package installed, and downloading + applying a chain
file inside a fixture-staging script is fragile (cross-build allele
flips at indel boundaries are a well-known source of silent errors).

Two ways to resolve:

1. Use a GRCh38-native UKB-inclusive release. The GLGC 2021 LDL meta is
   the largest published LDL GWAS, the GWAS Catalog re-harmonises it
   to GRCh38, UKB Biomarker biochemistry is its largest constituent
   cohort, and the brief explicitly permits this substitution
   ("UKB-style lipid trait, or whichever public sumstats"). **Chosen.**
2. Use Pan-UKB and add a chain-file liftover. Reserved as a follow-up
   extension if a future agent wants single-cohort UKB-only sumstats.

## License compliance

| Source | License | Verdict |
| --- | --- | --- |
| GTEx v8 summary statistics | Open access; GTEx Portal data use policy (public-domain sumstats) | OK to redistribute slices |
| GLGC 2021 LDL harmonised TSV | GWAS Catalog harmonised summary statistics policy (CC0 equivalent) | OK to redistribute slices |

Both sources are summary-statistics-only; no individual-level data is
staged or redistributed. Both source policies permit redistribution
of derived slices with attribution. The Graham 2021 paper and the
GTEx Consortium 2020 paper are cited in the manuscript draft.

## Strand-alignment handling

Both source files use REF/ALT (or other_allele / effect_allele) on
the GRCh38 reference strand. The harmonization rule applied in
prepare.py `_harmonize`:

1. Inner-join on (chr, pos_b38).
2. Drop strand-ambiguous palindromic SNPs where the allele pair is
   either {A,T} or {C,G}, following the field-standard convention
   used by Pan-UKB and PRS-CS (impossible to resolve flips by
   matching alleles only; would need allele-frequency-based
   resolution which the brief does not require for a worked example).
3. For each remaining variant, classify as:
   - **direct**: gtex.ref == glgc.other_allele AND gtex.alt == glgc.effect_allele
   - **flipped**: gtex.ref == glgc.effect_allele AND gtex.alt == glgc.other_allele
   - **unknown**: neither (drop conservatively; covers indel forms
     mismatched between sources)
4. For flipped variants invert the sign of beta_gwas + z_gwas and
   replace eaf_gwas with (1 - eaf_gwas), so the final aligned.parquet
   reports every effect on the GTEx alt allele as the effect allele.

The `orientation` column in aligned.parquet records `direct` /
`flipped` for every row, so the harmonization is fully auditable.

## Reproduction recipe

```bash
# (1) Pre-flight gate: asserts >=5 GB disk on /home and >=8 GB RAM.
#     Aborts non-zero on insufficient resources (memory/feedback_preflight.md).
bash validation/multiomics/gtex_ukb/fetch.sh

# (2) Harmonize. Writes fixtures/{gtex_liver_sort1.tsv.gz,
#     ukb_ldl_sort1.tsv.gz, aligned.parquet, manifest.sha256}.
#     Requires pyarrow (any conda env that has pandas + pyarrow >= 12);
#     this dispatch used /home/sikiru.atanda/miniconda3/envs/bayesalpha-cpu
#     (pyarrow 24.0.0, pandas 2.3.3).
python3 validation/multiomics/gtex_ukb/prepare.py

# (3) Manifest verification.
cd validation/multiomics/gtex_ukb/fixtures && sha256sum -c manifest.sha256
```

## Outputs

Committed under fixtures/:

| File | Description | Size (approx) |
| --- | --- | --- |
| gtex_liver_sort1.tsv.gz | Raw GTEx Liver signif rows for SORT1 (input to harmonisation, retained as provenance) | ~2 KB |
| ukb_ldl_sort1.tsv.gz | Raw GLGC LDL slice at chr1:108.7-110 Mb (input to harmonisation) | ~1-2 MB |
| aligned.parquet | Inner-joined beta/SE/p/z for both traits on a common GRCh38 SNP set | ~few KB |
| manifest.sha256 | SHA256 of every committed fixture | ~300 B |

## aligned.parquet schema

| Column | Type | Source | Meaning |
| --- | --- | --- | --- |
| variant_id | string | GTEx | chr_pos_ref_alt_b38 |
| chr | string | join | chrN (GRCh38) |
| pos_b38 | int64 | join | 1-based bp position (GRCh38) |
| ref | string | GTEx | reference allele |
| alt | string | GTEx | alternative allele (= effect allele after harmonisation) |
| rsid | string | GLGC | dbSNP rsid (preferred; falls back to GTEx) |
| gene_id | string | GTEx | ENSG with version suffix (ENSG00000134243.x for SORT1) |
| gene_name | string | GTEx | HGNC symbol (SORT1) |
| beta_eqtl | float64 | GTEx | slope = effect of alt allele on normalised expression |
| se_eqtl | float64 | GTEx | slope_se |
| p_eqtl | float64 | GTEx | pval_nominal (cis-eQTL) |
| z_eqtl | float64 | derived | beta_eqtl / se_eqtl |
| maf_eqtl | float64 | GTEx | MAF within the GTEx Liver cohort |
| beta_gwas | float64 | GLGC | effect of alt allele on LDL-C (post-harmonisation) |
| se_gwas | float64 | GLGC | standard_error |
| p_gwas | float64 | GLGC | p_value |
| z_gwas | float64 | derived | beta_gwas / se_gwas (post-harmonisation) |
| eaf_gwas | float64 | GLGC | effect-allele frequency, on the alt allele (post-harmonisation) |
| n_gwas | int64 | GLGC | effective sample size per row |
| orientation | string | derived | direct vs flipped (audit trail for sign change) |

## Assumptions

- GTEx and GLGC are both on GRCh38; no liftover is performed.
- `slope` in GTEx and `beta` in GLGC use the alt / effect allele as the
  effect direction within their respective files; we align both onto
  the GTEx alt allele in aligned.parquet.
- The cis window of 1.3 Mb (chr1:108.7-110 Mb) is wider than the 1 Mb
  GTEx-v8-default cis window so that every SORT1 signif eQTL is
  contained by the join key (the signif file alone has 58 SORT1 rows
  spanning chr1:109.05 - 109.59 Mb).
- Strand-ambiguous palindromic SNPs (allele pair = {A,T} or {C,G}) are
  dropped per the field convention; no allele-frequency-based
  resolution is attempted at this stage of the worked example.

## F3 divergence policy

No reference-tool head-to-head is performed at this Tier 2 stage --
this is a fixture-staging step. The Tier 1 A1 / A2 / A3 agents
(MetaXcan / SMR / coloc) will consume `aligned.parquet` as their
input fixture and run the head-to-head numerical equivalence tests
that produce the F2 / F4 figure entries. Any F3 divergences will be
attributed to the respective Tier 1 harness, not to this fixture.

## Pre-flight contract

fetch.sh sources `validation/external/_lib/preflight.sh` and asserts:

- Disk: >= 5 GB free on /home
- RAM:  >= 8 GB available

Verified empirically on the dispatch host (2026-05-15): 1.5 TB free,
49 GB RAM available -- comfortably above the thresholds. Both
downloads succeeded on first attempt.

## Layout

```
validation/multiomics/gtex_ukb/
|-- fetch.sh                  # downloads GTEx tar + GLGC TSV, asserts preflight
|-- prepare.py                # locus slice + harmonisation + parquet writer
|-- README.md                 # this file
|-- .gitignore                # excludes raw .cache/ multi-GB downloads
|-- .cache/                   # raw downloads (gitignored)
|   |-- GTEx_Analysis_v8_eQTL.tar
|   |-- GTEx_Analysis_v8_eQTL/{Liver.v8.signif_variant_gene_pairs.txt.gz, Liver.v8.egenes.txt.gz}
|   `-- GCST90239658.h.tsv.gz
`-- fixtures/                 # committed
    |-- gtex_liver_sort1.tsv.gz
    |-- ukb_ldl_sort1.tsv.gz
    |-- aligned.parquet
    `-- manifest.sha256
```

