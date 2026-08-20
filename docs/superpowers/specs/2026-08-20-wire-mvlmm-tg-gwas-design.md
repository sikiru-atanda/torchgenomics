# Design — Wire `mvlmm` (multi-trait) through `tg.gwas`

**Date:** 2026-08-20
**Status:** APPROVED
**Base branch:** `feat/gwas-mvlmm` off `feat/gwas-bayes-gxe-set` (which stacks on `feat/gwas-glmm-mklmm` → `feat/friendly-api`). Reuses all the low-level-route plumbing (read-back selection, `_lowlevel_extra_argv`, `model_options`, materialization).

## Goal

Wire the last deferred registry model — **`mvlmm`** (multi-trait mixed-model GWAS, `MultiTraitLMM`) — through the friendly `tg.gwas` API (Python/CLI/R). This is the one model that needs a **multi-trait input extension** the single-trait friendly layer does not yet carry. After this, **every registry model is wired** through `tg.gwas`. Thin orchestration over the existing `_cmd_mvlmm_scan` CLI runner; **no new statistics**.

## Background — verified facts (cross-checked against current code)

- `torchgenomics/cli.py` `_cmd_mvlmm_scan`: requires `--traits` (comma-separated column names, ≥2) and `--ploidy` (`required=True`). Loads the multi-trait `Y` via `_align_samples(args, config, trait_columns=trait_names)` (which calls `io.phenotype.load_phenotype` with the named columns) and errors if `Y.shape[1] < 2`. Writes `<prefix>.assoc.tsv` with a `P_JOINT` column via `_apply_correction_and_save`. So the **existing default read-back (`_read_scan_output`, which already handles `P_JOINT`) works unchanged**.
- `torchgenomics/api/_inputs.py` `load_inputs` is single-trait: `_resolve_phenotype(phenotype, trait)` → a `pd.Series`, then `_align_by_ids(pheno_series, covar_df, reader)` → `GwasInputs.phenotype` (Series), `.n_samples`, `.trait_name`.
- `_write_phenotype_tsv` (`_dispatch.py`) writes a two-column `sample_id\t<trait>` TSV. mvlmm needs a **multi-column** phenotype TSV (`sample_id` + each trait column).
- The friendly dispatch already materializes a phenotype TSV + (for in-memory genotype) a numeric-dosage CSV in `_materialize_inputs`.

## Design

### Component 1 — multi-trait `load_inputs` / `GwasInputs`

- `load_inputs(...)` gains `traits: list[str] | None = None`.
- **Single-trait path (unchanged):** `traits is None` → current behavior exactly.
- **Multi-trait path:** `traits` is a non-empty list →
  - `phenotype` must be a `pd.DataFrame` (indexed by sample id) or a path to a table containing those columns; a bare `Series` or missing columns → friendly `ValueError` naming the missing columns.
  - Resolve to a DataFrame of exactly the `traits` columns (in the given order), coerced numeric, indexed by sample id.
  - Align samples by id intersection across the multi-trait DataFrame ∩ genotype ∩ covariates (reuse the same intersection/sort logic as `_align_by_ids`; a small `_align_by_ids_multi` or a generalization that accepts a DataFrame).
- `GwasInputs` gains two fields (both default `None`, so single-trait callers are unaffected):
  - `phenotypes: pd.DataFrame | None` — the aligned `(n_samples, d)` multi-trait table (columns = `trait_names`).
  - `trait_names: list[str] | None` — the ordered trait column names.
  - `phenotype` (Series) stays populated = the **first** trait column, so `n_samples`, `.index` (used by env alignment), and single-trait models keep working unchanged.
  - `trait_type = "continuous"` (mvlmm is a Gaussian multi-trait model).

### Component 2 — wire `mvlmm` in the dispatch

- Register `"mvlmm": ("mvlmm-scan", "_cmd_mvlmm_scan", "MultiTraitLMM", None)` (default `P_JOINT` read-back).
- `_mvlmm_extra_argv(inputs, opts, user_opts)`:
  - Require `inputs.trait_names` with `len >= 2` → else friendly `ValueError`: "mvlmm needs ≥2 traits; pass traits=['Y1','Y2'] naming columns of a multi-column phenotype."
  - Emit `["--traits", ",".join(inputs.trait_names), "--ploidy", str(ploidy)]` where `ploidy` defaults to `2` and is overridable via `model_options["mvlmm"]["ploidy"]` (pop it from user_opts before the generic `_model_options_to_argv` pass so it isn't double-emitted).
  - Append `_model_options_to_argv(remaining user_opts)` (e.g. `--gene-action`, `--grm-method`).
- Route `alias == "mvlmm"` in `_lowlevel_extra_argv`.
- **Multi-trait materialization:** in `_run_lowlevel_cli` (or a helper), when `inputs.phenotypes is not None`, write a multi-column phenotype TSV (`sample_id` + each `trait_names` column) instead of the single-trait TSV, so `_align_samples(trait_columns=...)` in the runner finds the named columns. Add a `_write_multitrait_phenotype_tsv(inputs, path)`; `_materialize_inputs` chooses it when `inputs.phenotypes is not None`.

### Component 3 — `gwas()` surface + validation

- `gwas(...)` gains `traits: list[str] | None = None` (keyword-only), threaded into `load_inputs(traits=traits)`.
- If `models` includes `mvlmm` but `traits` is None/`<2` → the friendly `ValueError` from `_mvlmm_extra_argv` (raised inside `run_model`'s try, workdir cleaned).
- If `traits` is given but a single-trait model is selected, the single-trait model uses `inputs.phenotype` (first trait); document this. (No error — traits simply enables multi-trait models.)
- `run_model`'s `NotImplementedError` fallthrough now covers **no registry model** — every alias resolves. Keep the generic "unrecognized runner" guard for safety, but there is no longer a "not yet wired" model.

### Component 4 — CLI + R parity + docs

- **CLI**: add `--traits` (comma-separated) to the `gwas` subparser; `_cmd_gwas` parses it (`.split(",")`) and passes `traits=` to `api.gwas`. (`--models mvlmm --traits Y1,Y2`.)
- **R**: `tg_gwas(traits = NULL)` — a character vector forwarded (paths/columns); reticulate converts to a Python list. `.compact` drops `NULL`.
- **CLAUDE.md**: wired list → **all registry models** (`lmm/glm/blink/farmcpu/glmm/mklmm/gxe/set/bayes/mvlmm`); remove the "only mvlmm unwired" caveat; note `models="mvlmm"` needs `traits=[≥2 columns]` and returns a joint multi-trait result (`P_JOINT`).

## Error handling

- mvlmm without `traits` / with `<2` traits → friendly `ValueError`.
- `traits` naming columns absent from the phenotype → friendly `ValueError` naming the missing columns.
- `phenotype` a bare Series when `traits` is requested → friendly `ValueError` ("multi-trait needs a DataFrame/path with the named columns").

## Testing

Python (`tests/test_api_gwas.py`):
- `load_inputs(phenotype=<DataFrame with Y1,Y2,Y3>, genotype=G, traits=["Y1","Y2"])` → `GwasInputs` with `phenotypes` shape `(n,2)`, `trait_names==["Y1","Y2"]`, `phenotype` = the `Y1` Series, `trait_type=="continuous"`.
- `traits` naming a missing column → friendly `ValueError`.
- `traits` with a bare Series phenotype → friendly `ValueError`.
- `_mvlmm_extra_argv` emits `--traits Y1,Y2 --ploidy 2`; `model_options={"mvlmm":{"ploidy":4}}` → `--ploidy 4` (not duplicated).
- `tg.gwas(pheno_df, G, models="mvlmm", traits=["Y1","Y2"], kinship="auto", pcs=0)` → `GwasResult` with `model=="MultiTraitLMM"`, `n_variants>0` (P_JOINT read-back).
- `tg.gwas(..., models="mvlmm")` without `traits` → friendly `ValueError`.
- Multi-trait phenotype TSV materialization writes the named columns (unit-test `_write_multitrait_phenotype_tsv`).

CLI (`tests/test_cli_gwas.py`): `gwas --help` shows `--traits`.

R (`rTorchGenomics/tests/testthat/test-gwas.R`): mocked test that a `traits` vector is forwarded to the bridge.

Use an HWE-consistent Binomial genotype fixture and a phenotype **DataFrame** with ≥2 correlated continuous columns + enough samples/variants for mvLMM REML to converge on the tiny fixture.

## Scope / non-goals

- Only `mvlmm`. Multi-trait **multi-environment** (`mtmet`) is not in the friendly registry and is out of scope.
- No new statistics; no change to model math or to `lmm_scan`/`glm_scan`.
- R accepts `traits` as column names (character vector) against a phenotype **path/file**; an in-memory R data.frame phenotype is out of scope for the R wrapper (path-based), documented.

## Backward compatibility

Purely additive: two new `GwasInputs` fields (default `None`), a new optional `load_inputs(traits=)` / `gwas(traits=)` / CLI `--traits` / R `tg_gwas(traits=)`, one new `_LOWLEVEL_CLI` entry, one new argv builder, one multi-trait TSV writer. The single-trait path is byte-for-byte unchanged (all new behavior gated on `traits`/`phenotypes` being non-None). Every previously-wired model is unaffected.
