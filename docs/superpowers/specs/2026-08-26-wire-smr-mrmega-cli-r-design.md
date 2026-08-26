# Design — Wire SMR + MR-MEGA through the api / CLI / R

**Date:** 2026-08-26
**Status:** APPROVED
**Base branch:** `feat/postgwas-cli-r` off `origin/master` (the merged friendly-API stack; tree `00b148c`).

## Goal

Continue closing the Python-only postgwas parity gap: expose **SMR + HEIDI** (`smr_heidi`) and **MR-MEGA** (`mr_mega`) — currently reachable only from `torchgenomics.postgwas` — as first-class **api functions + CLI subcommands + R wrappers**. Thin orchestration over the existing, validated `postgwas` functions; **no new statistics**. Mirrors the proven mr/coloc pattern shipped in the friendly-API stack.

Scope this iteration (user-approved): **`smr` (SMR+HEIDI integration)** and **`mr_mega` (multi-ancestry MR)** only. HESS, power, winner's-curse, and gene-set enrichment (tensor / structured-object inputs) remain Python-only — a later follow-up.

## Background — verified facts (cross-checked against current code)

The proven exposure pattern (used by `clump`/`meta`/`mr`/`coloc`): a real `torchgenomics.api.<fn>` in `api/postgwas.py` (in `api.__all__`) → a CLI subcommand thin-wrapping it → a hand-crafted R wrapper in `rTorchGenomics/R/api.R` (`bridge_call("<fn>", ...)` → `<Class>$new_from_dict`) + an S4 result class. Sumstats load from files via `postgwas.load_sumstats(path, ..., sep='\t') -> SumStats` (columns chr/pos/snp/a1/a2/beta/se/p/n/af).

Verified postgwas signatures + result fields:
- `smr_heidi(gwas: SumStats, eqtl: SumStats, gene_map: dict[str, list[str]], eqtl_p_threshold=5e-8, smr_p_threshold=0.05, heidi_p_threshold=0.05, heidi_max_snps=20, ld_matrix: Tensor | None = None) -> SMRSummary`.
  - `SMRSummary` fields: `results` (list of `SMRResult`), `n_genes_tested`, `n_significant_smr`, `n_pass_heidi`.
  - `SMRResult` fields: `gene_id, probe_snp, beta_smr, se_smr, p_smr, chi2_smr, beta_gwas, beta_eqtl, p_heidi, n_heidi_snps, heidi_stat`.
  - `gene_map` maps a gene id → its list of cis-SNP ids. `ld_matrix` is optional (defaults None).
- `mr_mega(sumstats_list: list[SumStats], n_axes=4, random_effects=False) -> MultiAncestryResult`.
  - `MultiAncestryResult` fields: `beta_meta, se_meta, p_meta, p_heterogeneity, method, p_ancestry, p_residual, n_axes, log10_bf, posterior_effect, n_populations, n_snps` (the per-SNP arrays are tensors).
- MR-MEGA fits `n_axes` ancestry axes; it needs at least `n_axes + 2` populations (a regression of effect sizes on ancestry PCs). So `len(sumstats_list) >= n_axes + 2` — validate.

## Design

### Component 1 — `api.smr()`

In `torchgenomics/api/postgwas.py`:

```python
def smr(gwas, eqtl, gene_map, *, output=None, eqtl_p_threshold=5e-8,
        smr_p_threshold=0.05, heidi_p_threshold=0.05, heidi_max_snps=20,
        sep="\t") -> SMRRun: ...
```

- Load `gwas`, `eqtl` via `load_sumstats`.
- Load `gene_map` from a **long-format TSV** with header `gene\tsnp` (one row per gene–cis-SNP pair) into `dict[gene -> [snp, ...]]` (group by gene, preserve order, dedupe). Accept a path (str/Path) OR an already-built dict (Python callers). Missing/empty/wrong columns → friendly `ValueError`.
- Run `smr_heidi(gwas, eqtl, gene_map, eqtl_p_threshold=..., smr_p_threshold=..., heidi_p_threshold=..., heidi_max_snps=...)` (ld_matrix left None — a full LD matrix isn't a natural file input; documented).
- Build a per-gene results DataFrame from `SMRSummary.results` (columns = the `SMRResult` fields). Write to `output` TSV when given.
- Return `SMRRun(results=df, n_genes_tested, n_significant_smr, n_pass_heidi)`.

### Component 2 — `api.mr_mega()`

```python
def mr_mega(sumstats, *, output=None, n_axes=4, random_effects=False,
            sep="\t") -> MRMegaRun: ...
```

- `sumstats` = a list of ≥ (`n_axes` + 2) sumstats paths (one per ancestry). A non-list or too-few paths → friendly `ValueError` naming the requirement.
- Load each via `load_sumstats`; run `mr_mega(ss_list, n_axes=n_axes, random_effects=random_effects)`.
- Build a per-SNP results DataFrame (columns: `beta_meta, se_meta, p_meta, p_heterogeneity, p_ancestry, p_residual, log10_bf, posterior_effect` — the per-SNP tensors converted to columns; scalar fields `method, n_axes, n_populations, n_snps` reported on the run object). Write to `output` TSV when given.
- Return `MRMegaRun(results=df, method, n_axes, n_populations, n_snps, min_p_meta)`.

### Component 3 — result classes `SMRRun` / `MRMegaRun`

In `torchgenomics/api/_results.py`, `@dataclass`es subclassing `_BaseRun` (mirror `MetaRun`/`MRRun`):
- `SMRRun`: `results: pd.DataFrame`, `n_genes_tested: int`, `n_significant_smr: int`, `n_pass_heidi: int`. `.summary()`: "SMR — N genes tested, M SMR-significant, K pass HEIDI".
- `MRMegaRun`: `results: pd.DataFrame`, `method: str`, `n_axes: int`, `n_populations: int`, `n_snps: int`, `min_p_meta: float | None`. `.summary()`: "MR-MEGA — n_populations pops, n_axes axes, n_snps SNPs, min p_meta=…".
- Export both from `_results.py` + `api/__init__.py` (`__all__`), and add `smr`/`mr_mega` to `api.__all__`.

### Component 4 — CLI subcommands

In `torchgenomics/cli.py` (mirror the `mr`/`coloc` subcommand pattern):
- `torchgenomics smr --gwas g.tsv --eqtl e.tsv --gene-map map.tsv [--eqtl-p-threshold 5e-8 --smr-p-threshold 0.05 --heidi-p-threshold 0.05 --heidi-max-snps 20] --output smr.tsv` → `_cmd_smr` calls `api.smr(...)`, prints `.summary()`, returns 0.
- `torchgenomics mr-mega --sumstats pop1.tsv pop2.tsv pop3.tsv [--n-axes 1 --random-effects] --output mrmega.tsv` (`--sumstats nargs="+"`) → `_cmd_mr_mega` calls `api.mr_mega(...)`.
- Register both in `build_parser` + the dispatch dict; add `"smr"`, `"mr-mega"` to `_TIER_2_CLI_SUBCOMMANDS`. Bump the CLI count note in `CLAUDE.md` (48 → 50) + add example lines. Because they're in `_TIER_2_CLI_SUBCOMMANDS`, the api CLI-runner bridge (`api.smr`/`api.mr_mega` — real functions here) takes precedence over `__getattr__`; verify no shadowing.

### Component 5 — R wrappers `tg_smr` / `tg_mr_mega`

Hand-crafted in `rTorchGenomics/R/api.R` (mirror `tg_mr`/`tg_coloc`):
```r
tg_smr(gwas, eqtl, gene_map, output = NULL, eqtl_p_threshold = 5e-8,
       smr_p_threshold = 0.05, heidi_p_threshold = 0.05, heidi_max_snps = 20)
  -> SMRRun (bridge_call("smr", ...))
tg_mr_mega(sumstats, output = NULL, n_axes = 4, random_effects = FALSE)
  -> MRMegaRun (bridge_call("mr_mega", ...))    # sumstats = character vector of paths
```
Add S4 `SMRRun`/`MRMegaRun` in `rTorchGenomics/R/result_classes.R` (mirror `MRRun`, reuse `%||%`/`.as_*`/`.tibble_from_list_of_dicts`); `NAMESPACE` exports; `_pkgdown.yml` "Post-GWAS: MR + coloc" section (rename to "Post-GWAS: MR / coloc / SMR"); regenerate `man/*.Rd`. Author R via heredoc.

## Error handling

- `smr`: gene-map file missing the `gene`/`snp` columns, or empty → friendly `ValueError`. Missing gwas/eqtl file/column → the actionable error from `load_sumstats`.
- `mr_mega`: `sumstats` not a list / fewer than `n_axes + 2` paths → friendly `ValueError` stating the population requirement.

## Testing

Python (`tests/test_api_smr_mrmega.py`, new):
- Simulate gwas + eqtl sumstats sharing a causal cis-SNP for a couple of genes + a gene-map TSV → `api.smr(...)` returns an `SMRRun` with `n_genes_tested == #genes` and at least one SMR-significant gene; a non-shared setup → fewer/no significant.
- Simulate ≥3 ancestry sumstats over shared SNPs → `api.mr_mega(sumstats=[...], n_axes=1)` returns an `MRMegaRun` with `n_populations == #pops`, `n_snps > 0`, and a per-SNP results DataFrame.
- Friendly errors: gene-map missing columns; `mr_mega` with too few populations.
- Assert on the reused postgwas math (already validated); the api layer only orchestrates.

CLI (`tests/test_cli_smr_mrmega.py`): `smr --help` / `mr-mega --help` show the flags; end-to-end `smr`/`mr-mega` on the fixtures write the TSV.

R (`rTorchGenomics/tests/testthat/test-smr-mrmega.R`): mocked `bridge_call` — `tg_smr`/`tg_mr_mega` forward args + wrap into the S4; integration (guarded) runs against the fixtures.

Full Python suite must stay green.

## Scope / non-goals

- Only `smr` + `mr_mega`. HESS / power / winner's-curse / gene-set enrichment (tensor / GeneResult / GMT inputs) are a later follow-up.
- No new statistics; no change to the `postgwas` math or existing api/CLI/R surfaces.
- SMR `ld_matrix` is not exposed via CLI/R (optional; a full LD matrix isn't a natural file input) — documented; Python callers can still pass it to `postgwas.smr_heidi` directly.
- MCP: not registered as `@tool` (consistent with mr/coloc) — out of scope.

## Backward compatibility

Purely additive: two api functions + two result classes, two CLI subcommands, two R wrappers + S4 classes, two `_TIER_2_CLI_SUBCOMMANDS` entries. No existing signature/default changes.
