# Wire HESS / Power / Winner's-Curse / Gene-Set Enrichment through api / CLI / R — Design

**Date:** 2026-08-27
**Status:** Approved (design)
**Author:** subagent-driven session

## Goal

Expose four post-GWAS functions that are currently reachable only via the deep
`torchgenomics.postgwas` library API as first-class functions on all three user
surfaces — the Python `torchgenomics.api` facade (`tg.*`), the CLI, and the
`rTorchGenomics` R package — matching the shipped `mr` / `coloc` / `smr` /
`mr_mega` pattern:

- **HESS** — local (regional) heritability + local genetic correlation
- **Power** — per-variant GWAS detection power + required-N + power curve
- **Winner's-curse** — effect-size de-biasing (three methods)
- **Gene-set enrichment** — MAGMA-style competitive gene-set test

## Binding global constraints

- **No new statistics.** Every wrapper is thin orchestration over the existing,
  validated `postgwas` functions (`hess_local_h2`, `hess_local_rg`,
  `gwas_power`, `required_n`, `power_curve`, `correct_winners_curse`,
  `snp_to_gene`, `gene_set_enrichment`). The `postgwas/` math is **not** changed.
  The only added code is I/O glue: loading sumstats / LD matrices / gene files,
  the bp→SNP-index region mapping, and a GMT parser.
- **Backward compatible / additive only.** New `api` functions + result classes,
  four CLI subcommands, four R wrappers + S4 classes, four
  `_TIER_2_CLI_SUBCOMMANDS` entries. No existing signature or default changes.
- **Not MCP tools.** No `@tool` decorator — consistent with `mr`/`coloc`/`smr`/
  `mr_mega`. MCP tool count stays 14.
- **Precedence.** The four names are added to `_TIER_2_CLI_SUBCOMMANDS` (for
  manifest parity) AND have real `api.*` functions; the real module attributes
  must win over the `api.__getattr__` CLI-runner.
- **Verification always under `TORCHGENOMICS_DISABLE_NATIVE=1`.**
- Reference-tool fidelity: the underlying postgwas functions are the spec; the
  wrappers must not alter their numerical output. Tests assert real, non-trivial
  behavior (not just "runs").

## Upstream signatures (the spec — verbatim, do not change)

```python
# _hess.py
hess_local_h2(z: Tensor, ld_matrix: Tensor, n: int|float,
              region_bounds: list[tuple[int,int]],
              region_labels: list[str]|None = None,
              eigenvalue_threshold: float = 1.0) -> HESSResult
hess_local_rg(z1, z2, ld_matrix, n1, n2, region_bounds,
              region_labels=None, eigenvalue_threshold=1.0) -> HESSResult
# HESSResult(regions: list[HESSRegionResult], h2_total, h2_total_se, n_regions, n_snps_total)
# HESSRegionResult(region_id, chrom, start, end, h2_local, h2_local_se, n_snps, n_eigenvalues_kept)

# _power.py
gwas_power(n, af: Tensor, beta: Tensor, alpha=5e-8, target_power=0.8) -> PowerResult
power_curve(n, af_grid: Tensor, alpha=5e-8, target_power=0.8) -> Tensor  # (g,)
required_n(af: Tensor, beta: Tensor, alpha=5e-8, target_power=0.8) -> Tensor  # (m,)
# PowerResult(power, ncp, alpha, n, min_detectable_beta)

# _winners_curse.py
correct_winners_curse(beta, se, method="conditional_likelihood",
                      alpha=5e-8, **kwargs) -> WinnersCurseResult
# method in {"conditional_likelihood","fiqt","bootstrap"};
# bootstrap kwargs: n_boot=10000, seed=None
# WinnersCurseResult(method, beta_adjusted, se_adjusted, shrinkage_factor, n_corrected)

# _enrichment.py
snp_to_gene(ss: SumStats, gene_id: list[str], gene_chr: list[str],
            gene_start: list[int], gene_end: list[int], window_kb=0.0) -> GeneResult
gene_set_enrichment(gene_result: GeneResult, gene_sets: dict[str, list[str]],
                    covariate_gene_size=True, covariate_log_size=True) -> EnrichmentResult
# GeneResult(gene_id, gene_chr, gene_start, gene_end, n_snps, stat, p)
# EnrichmentResult(gene_set_name, n_genes_in_set, n_genes_total, beta_enrichment, se, p)

# _sumstats.py
load_sumstats(path, ..., sep="\t") -> SumStats
# SumStats(chr, pos, snp, a1, a2, beta, se, p, n, af); .z=beta/se, .chi2=(beta/se)**2, .m
```

## api layer (`torchgenomics/api/postgwas.py`)

All four mirror `api.smr`/`api.mr_mega`: lazy import from `..postgwas`, load inputs,
`with timed() as elapsed:` around the core call, flatten to a `pd.DataFrame`,
optional TSV write, return the `_results` dataclass. No `@tool`.

### `power`
```python
power(gwas, *, n=None, alpha=5e-8, target_power=0.8,
      power_curve=False, af_grid=None, output=None, sep="\t") -> PowerRun
```
- Load sumstats → `af`, `beta`. `af` must be present (raise friendly `ValueError`
  if the sumstats has no `af` column). `n`: use `--n` if given, else the median
  of the finite `n` column; raise if neither is available.
- Call `gwas_power(n, af, beta, alpha, target_power)` → `power`, `ncp`,
  `min_detectable_beta`. Also call `required_n(af, beta, alpha, target_power)`
  and fold the result in as a `required_n` column (reuses the same inputs).
- If `power_curve=True`: build the AF grid (`af_grid` = explicit comma list or a
  default 50-point linspace over `[0.01, 0.5]`), call the upstream curve function,
  store as a second DataFrame. **Name-collision guard:** the boolean parameter is
  named `power_curve`, which shadows the imported `power_curve` postgwas function —
  import it under an alias (`from ..postgwas import power_curve as _power_curve`)
  and call `_power_curve(...)` inside the wrapper.
- **PowerRun**(`alpha, n, target_power, n_variants, n_powered`, `results` df:
  `snp, af, beta, power, ncp, min_detectable_beta, required_n`; `curve` df|None:
  `af, min_detectable_beta`).

### `winners_curse`
```python
winners_curse(gwas, *, method="conditional_likelihood", alpha=5e-8,
              n_boot=10000, seed=None, output=None, sep="\t") -> WinnersCurseRun
```
- Load sumstats → `beta`, `se`. Validate `method` ∈ the three; friendly error otherwise.
- **Kwargs forwarding (mandatory conditional).** `correct_winners_curse` dispatches
  to `conditional_likelihood(beta, se, alpha)` and `fiqt(z, se, ...)` — **neither
  accepts `n_boot`/`seed`, so passing them raises `TypeError`.** Only
  `bootstrap_correction` accepts them. Therefore build kwargs conditionally:
  ```python
  extra = {"n_boot": n_boot, "seed": seed} if method == "bootstrap" else {}
  res = correct_winners_curse(beta, se, method=method, alpha=alpha, **extra)
  ```
  Do NOT call `correct_winners_curse(..., n_boot=n_boot, seed=seed)` unconditionally.
  (For `method="fiqt"`, `correct_winners_curse` itself computes `z = beta/se`
  internally, so the wrapper passes `beta`/`se` uniformly for all three methods.)
- **WinnersCurseRun**(`method, n_corrected, n_variants`, `results` df:
  `snp, beta_original, beta_adjusted, se_adjusted, shrinkage_factor`).
  Upstream returns `se_adjusted=None` for **all three** methods (the docstring's
  "available for conditional likelihood" is stale), so the `se_adjusted` column is
  always all-NaN — build it like MR-MEGA's `_col(None)`, and do NOT assert a
  non-NaN `se_adjusted` in tests.

### `gene_set_enrichment`
```python
gene_set_enrichment(gwas, gene_annotation, gene_sets, *, window_kb=0.0,
                    covariate_gene_size=True, covariate_log_size=True,
                    gene_sets_format="auto", output=None, sep="\t") -> EnrichmentRun
```
- Load sumstats. Load gene annotation TSV via a new `_load_gene_annotation(path, sep)`
  → four parallel lists (`gene_id, gene_chr, gene_start, gene_end`); require columns
  `gene, chr, start, end`; friendly error on missing columns.
- Load gene sets via a new `_load_gene_sets(path, fmt, sep)`:
  - `fmt="auto"`: `.gmt`/`.GMT` extension → GMT; else long TSV.
  - `fmt="tsv"`: long TSV with columns `set, gene` (one row per pair) → `dict[set,[genes]]`.
  - `fmt="gmt"`: MSigDB `.gmt` — each line `set_name<TAB>description<TAB>gene1<TAB>gene2...`
    → `dict[set,[genes]]` (description column skipped).
  - Friendly error on unreadable / empty / wrong-column input.
- Stage 1 `snp_to_gene(ss, *lists, window_kb=window_kb)`; Stage 2
  `gene_set_enrichment(gene_result, gene_sets, covariate_gene_size,
  covariate_log_size)`.
- **EnrichmentRun**(`n_genes_total, n_gene_sets, n_significant` (count of set p<0.05),
  `results` df: `gene_set_name, n_genes_in_set, beta_enrichment, se, p`;
  `genes` df: `gene_id, gene_chr, gene_start, gene_end, n_snps, stat, p`).

### `hess`
```python
hess(gwas, ld_matrix, regions, *, n=None, gwas2=None,
     eigenvalue_threshold=1.0, output=None, sep="\t") -> HessRun
```
- Load sumstats → `z` (`ss.z`). If `gwas2` given, load it → `z2`; `mode="rg"`, else `mode="h2"`.
- Load LD matrix via a new `_load_matrix(path)`: `.npy` → `np.load`; `.pt`/`.pth` →
  `torch.load`; convert to a float64 `torch.Tensor`. Must be square `(m, m)` with
  `m == ss.m`; friendly error otherwise.
- Load regions TSV via a new `_load_regions(path, sep)`: columns `chrom, start, end`
  (bp); friendly error on missing columns.
- **Region → SNP-index mapping** (the guarded glue, NOT a statistic): for each
  region, find the sumstats row indices `i` where `str(ss.chr[i]) == str(chrom)`
  and `start <= ss.pos[i] <= end`. If none, skip the region with a `log`/warning.
  Build `(first, last+1)`. **Explicit guard:** the matched indices must be a
  contiguous block (`sorted(idx) == list(range(first, last+1))`); otherwise raise
  a friendly `ValueError` telling the user the sumstats + LD matrix must be sorted
  by genomic position. `region_labels = f"{chrom}:{start}-{end}"`.
- **All-regions-skipped guard:** if, after mapping, NO region matched any SNP (the
  bounds list is empty), raise a friendly `ValueError` ("none of the N regions in
  `<regions>` matched any SNP in the sumstats — check chrom naming / coordinates")
  BEFORE calling `hess_local_*`. Otherwise upstream raises the misleading
  "At least one region is required." (`_hess.py:141`).
- `n`: `--n` if given, else the median of the finite `n` column; for `rg`, `n1`/`n2`
  both derived (a single `--n` applies to both unless a separate `--n2` is given —
  add `--n2`). If `--n`/`--n2` is omitted AND the finite-`n` median is empty/NaN,
  raise a friendly `ValueError` (parity with the `power` N guard) rather than
  silently producing NaN heritabilities.
- Call `hess_local_h2(z, ld, n, bounds, labels, eigenvalue_threshold)` or
  `hess_local_rg(z, z2, ld, n1, n2, bounds, labels, eigenvalue_threshold)`.
- **HessRun**(`mode, h2_total, h2_total_se, n_regions, n_snps_total`, `results` df:
  `region_id, chrom, start, end, h2_local, h2_local_se, n_snps, n_eigenvalues_kept`).
  Note: overwrite the per-region `chrom/start/end` with the *bp* region values from
  the regions file (upstream sets `chrom=""` and start/end = SNP indices).

## Result classes (`torchgenomics/api/_results.py`)

Add `PowerRun`, `WinnersCurseRun`, `EnrichmentRun`, `HessRun` — `@dataclass`
subclassing `_BaseRun`, each with scalar summary fields (defaults), one or two
`pd.DataFrame` fields (`field(default_factory=pd.DataFrame)`), a
`_kind: ClassVar[str]`, and a `summary()` method. Mirror `SMRRun`/`MRMegaRun`.
Import them into `api/__init__.py` `__all__` and `api/postgwas.py`. Add the four
`api` functions + result classes to `torchgenomics/__init__.py` top-level exports.

## CLI (`torchgenomics/cli.py`)

Four `_add_*_parser` + `_cmd_*` pairs, thin wrappers over the api functions,
registered in `_build_parser` and the dispatch dict:

- `power`: `--gwas` (req), `--n`, `--alpha` (5e-8), `--target-power` (0.8),
  `--power-curve` (flag), `--af-grid` (comma list), `--output`.
- `winners-curse`: `--gwas` (req), `--method` (choices, default
  `conditional_likelihood`), `--alpha`, `--n-boot`, `--seed`, `--output`.
- `gene-set-enrichment`: `--gwas` (req), `--gene-annotation` (req),
  `--gene-sets` (req), `--window-kb` (0.0), `--no-covariate-gene-size`,
  `--no-covariate-log-size`, `--gene-sets-format` (auto/tsv/gmt), `--output`.
- `hess`: `--gwas` (req), `--ld-matrix` (req), `--regions` (req), `--n`, `--n2`,
  `--gwas2`, `--eigenvalue-threshold` (1.0), `--output`.

Add `"power"`, `"winners-curse"`, `"gene-set-enrichment"`, `"hess"` to
`_manifest/__init__.py` `_TIER_2_CLI_SUBCOMMANDS`. CLAUDE.md CLI count 50 → 54 +
four example lines.

## R (`rTorchGenomics/`)

`tg_power`, `tg_winners_curse`, `tg_gene_set_enrichment`, `tg_hess` in `R/api.R`
(hand-crafted `bridge_call(...)` → `<Class>$new_from_dict`), + `PowerRun`,
`WinnersCurseRun`, `EnrichmentRun`, `HessRun` S4 classes in `R/result_classes.R`
mirroring `SMRRun`/`MRMegaRun` (reuse `%||%`, `.as_int/.as_num/.as_chr/.as_list`,
`.tibble_from_list_of_dicts`). Update `NAMESPACE`, `_pkgdown.yml`, add `man/*.Rd`
(one per function + one per S4 class), add mocked `tests/testthat/test-hess-power-wc-enrichment.R`.
The two-DataFrame results (`PowerRun.curve`, `EnrichmentRun.genes`) serialize as a
second records list and rebuild as a second tibble.

**Codegen exclusion (REQUIRED — prevents auto-wrapper clobber).** Every command
with a hand-crafted R wrapper must be listed in `.HANDCRAFTED_COMMANDS` in
`rTorchGenomics/R/codegen.R:14`, or a future `generate_api_auto()` regenerates a
colliding auto-wrapper in `api_auto.R`. Add the four new manifest names
(underscored): `"power"`, `"winners_curse"`, `"gene_set_enrichment"`, `"hess"`.
**Also back-fill the two names the SMR/MR-MEGA pass missed:** `"smr"`, `"mr_mega"`
(latent gap, currently masked only because `api_auto.R` is stale). After editing,
regenerate the bundled manifest + `api_auto.R` and confirm no duplicate wrapper for
any hand-crafted command.

**Single-trait scope note.** `load_sumstats` reads a single `beta_col`, so `beta`
is always 1-D `(m,)` and the wrappers are single-trait by construction; no
multi-trait branch is added. (Optional defensive `beta.ndim == 1` assertion in
power/winners-curse/hess is fine but not required.)

## Testing

- `tests/test_api_hess_power_wc_enrichment.py` — for each: result-class
  shape/summary; a real end-to-end run on a constructed fixture asserting
  *non-trivial* behavior:
  - **power**: high-power variant (large β, moderate AF) has `power` near 1 and a
    *smaller* `required_n` than a low-power variant; `--power-curve` returns a
    monotone-ish envelope.
  - **winners-curse**: for genome-wide-significant hits, `|beta_adjusted| <
    |beta_original|` and `0 < shrinkage_factor <= 1` for at least the corrected
    variants; all three methods run.
  - **gene-set enrichment**: a gene set enriched for high-χ² genes gets a lower
    `p` than a null set; both TSV and GMT gene-set formats parse to the same dict.
  - **hess**: total h² recovered within tolerance on a fixture with known
    per-region signal; the contiguity guard raises `ValueError` on shuffled input;
    `--gwas2` path produces an `rg` HessRun.
- `tests/test_cli_hess_power_wc_enrichment.py` — `--help` for all four + one real
  end-to-end CLI invocation (at least power, the cheapest).
- R: mocked `test-hess-power-wc-enrichment.R` (reticulate patched), asserting the
  S4 classes build from a dict and expose the right slots.
- Precedence test: `api.hess.__name__ == "hess"` etc. (real fn wins over
  `__getattr__`).

## Out of scope

- No change to any `postgwas/` statistics (including the tracked
  `smr_heidi` global-argmin probe follow-up and any HESS/enrichment internals).
- No MCP tools for these four.
- No new LD-matrix computation — the LD matrix is a user-supplied file.
