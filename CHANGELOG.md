# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] — 2026-04-15

### Added
- `torchgwas.annotate` — NCBI gene annotation for GWAS hit SNPs (Phase 48). Resolves
  crop/taxid/assembly, fetches overlapping genes per hit in a configurable window,
  attaches descriptions, GO terms, and optional orthologs. `torchgwas annotate` CLI.
- `torchgwas.multiomics` — GRM-corrected causal mediation + multi-kernel heritability
  (Phase 49). `mediate_lmm` (single-triple, Sobel / Monte-Carlo / bootstrap SE, Imai
  ρ-sensitivity), `scan_mediation` (cis-window filter + BH/BY/Storey FDR),
  `mkernel_h2` (genotype GRM + regulatory-state kernel partition),
  `build_expression_kernel`. Two new CLI subcommands (`mediate`, `mediate-scan`)
  bring the CLI surface from 32 to 35 subcommands.

### Fixed
- `torchgwas.annotate` region queries now use E-utilities `esearch` with
  `[Base Position]`. The Datasets v2 `annotation_report` endpoint silently ignored
  `chromosomes`/`start`/`stop` filters and returned the whole assembly.
- `torchgwas.annotate` gene details now route through `POST /gene` (the
  `POST /gene/id` path was removed). Live response shape changes (coordinates moved
  to `annotations[0].genomic_locations[].genomic_range`; GO terms to
  `gene_ontology.{processes,functions,components}`) are translated back to the
  legacy shape via `_ensure_legacy_genomic_ranges` / `_ensure_legacy_ontology_terms`.
- `torchgwas.io.numeric` — marker-axis orientation now uses the companion `.map` /
  `.bim` file's SNP IDs as a ground-truth disambiguator when both row and column
  IDs are non-numeric. Fixes misclassification of GAPIT-style `mdp_numeric.txt`.

### Dependencies
- Added `requests>=2.28` for NCBI HTTP I/O.

## [0.1.0] — 2026-04-09

Initial alpha release. Phases 0-47 complete (2102 tests, 44 skipped).
