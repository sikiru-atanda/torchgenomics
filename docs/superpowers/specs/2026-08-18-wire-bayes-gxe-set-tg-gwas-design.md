# Design — Wire `bayes` + `gxe` + `set` through `tg.gwas`

**Date:** 2026-08-18
**Status:** APPROVED
**Base branch:** `feat/gwas-bayes-gxe-set` off `feat/gwas-glmm-mklmm` (which is off `feat/friendly-api`). Reuses the `model_options`/`extra_argv`/`_run_lowlevel_cli` plumbing added by the glmm+mklmm work.

## Goal

Wire three more registry models through the friendly `tg.gwas` API:
- **`gxe`** — genotype×environment interaction LMM (needs an environment variable).
- **`set`** — region/gene-based test (SKAT/burden; needs a regions file).
- **`bayes`** — SuSiE/CAVI Bayesian fine-mapping (no extra input; PIP/credible-set output).

Thin orchestration over the existing CLI runners (`_cmd_gxe_scan`, `_cmd_set_scan`, `_cmd_bayes_scan`); **no new statistics**. Additive and backward-compatible. `mvlmm` (which needs a multi-trait input extension) is a separate later plan and remains `NotImplementedError`.

## Background — verified facts (cross-checked against current code)

`torchgenomics/api/_dispatch.py` already routes low-level models through their CLI single-trait runner via `_run_lowlevel_cli`, which builds a defaulted `argparse.Namespace`, runs the runner, and reads `<prefix>.assoc.tsv` back via `_read_scan_output`. From glmm+mklmm it also carries `_lowlevel_extra_argv(alias, inputs, opts)` (model-specific flags, incl. `model_options`) and `RunOptions.model_options`.

Confirmed in `torchgenomics/cli.py`:
- **`_cmd_gxe_scan`** — requires `--env` (a TSV read with `pd.read_csv(args.env, sep="\t")["ENV"]`, consumed in **file order**, assumed aligned to sample order). Writes `<prefix>.assoc.tsv` with a `P_JOINT` column via `_apply_correction_and_save`. `_read_scan_output` **already handles `P_JOINT`** (`p_col = "P" if "P" in df else "P_JOINT" if "P_JOINT" in df else None`) — so gxe reuses the existing read-back unchanged.
- **`_cmd_set_scan`** — requires `--regions` (a BED/region file via `io.regions.load_regions`). Writes a **custom** `<prefix>_set_based.tsv` with columns `REGION, START, END, …, P` (region-level; NOT `.assoc.tsv`). Needs a dedicated read-back.
- **`_cmd_bayes_scan`** — no extra required input (SuSiE defaults: `--method susie`, `--n-signals 10`). Writes a **custom** `<prefix>_bayesian_vs.tsv` with columns `SNP, CHR, POS, A1, A2, AF, PIP, BETA_MEAN, BETA_SD` (no P column), plus a `<prefix>_credible_sets.tsv`. Needs a dedicated PIP read-back.
- The `gxe-scan`, `set-scan`, `bayes-scan` subparsers all accept the base `--genotype/--phenotype/--output/--correction/--device` args (verified by parsing).

## Design

### Component 1 — per-model read-back selection

`_run_lowlevel_cli` currently hardcodes `_read_scan_output`. Generalize so each model chooses its read-back. Add a 4th element to each `_LOWLEVEL_CLI` tuple: a read-back callable (or `None` → the default `_read_scan_output`). `_run_lowlevel_cli` looks it up and calls it with the same signature (`output_prefix`, `model_label`, `test`, `correction`, `trait_type`, `n_samples`, `significance_threshold`, `top_k`, `runtime_s`).

- `blink/farmcpu/glmm/mklmm/gxe` → `_read_scan_output` (default).
- `bayes` → `_read_bayes_output`.
- `set` → `_read_set_output`.

### Component 2 — `_read_bayes_output`

Reads `<prefix>_bayesian_vs.tsv`. Builds a `GwasResult` with:
- `top_hits` = the DataFrame sorted by `PIP` descending, truncated to `top_k` (columns kept as-is: SNP/CHR/POS/PIP/BETA_MEAN/…).
- `model = "BayesianVS"`, `test = "susie"` (or the method), `lambda_gc = None` (no p-values), `n_variants = len(df)`.
- `n_significant` = number of variants in the credible sets, read from `<prefix>_credible_sets.tsv` if present (else 0). (Per user decision: credible-set membership, not a PIP cutoff.)
- `summary()` reflects fine-mapping: since `lambda_gc is None` and there is no P column, the enriched `GwasResult.summary()` already omits λ_GC gracefully; the model label + n_significant convey "N variants in credible sets". (If needed, set a short `warnings`/note; do not fabricate p-values.)
- Wrapped so a missing/empty file raises a clear error, not a bare `FileNotFoundError`.

### Component 3 — `_read_set_output`

Reads `<prefix>_set_based.tsv`. Builds a `GwasResult` with:
- `top_hits` = regions sorted by `P` ascending, truncated to `top_k`.
- `model = "SetBasedScanner"`, `test` = the set-test (`skat`/`burden`/`skat_o`), `n_variants = len(df)` (region count), `lambda_gc = None` (region-level, not per-SNP), `n_significant` = number of regions with `P < significance_threshold`.
- Reuses `api._helpers.top_hits` with `p_column="P"` where possible (the file has a `P` column).

### Component 4 — new inputs `env=` / `regions=`

- `RunOptions` gains `env: Any = None`, `regions: Any = None`.
- `gwas()` gains `env=None`, `regions=None` (keyword-only), threaded into `RunOptions`.
- Materialization (in `_run_lowlevel_cli`'s per-model argv step, or a helper), both accept in-memory **or** path:
  - **env** (for gxe): a `pd.Series`/1-D array aligned to samples → write a temp TSV with a single `ENV` column in the aligned sample order (`inputs.phenotype.index` order), and add `--env <tmp>`. A path (`str`) → passed through as `--env <path>` unchanged (user is responsible for order, matching the CLI). A Series with a sample-id index is reindexed to the aligned order before writing.
  - **regions** (for set): a path (`str`) → `--regions <path>`. A `pd.DataFrame` (chrom/start/end[/id]) → write a temp regions file in the format `io.regions.load_regions` reads, then `--regions <tmp>`.
- **Required-input validation:** selecting `gxe` without `env` → friendly `ValueError` ("gxe needs an environment variable; pass env=<Series|path>"). Selecting `set` without `regions` → friendly `ValueError` ("set needs regions; pass regions=<path|DataFrame>"). Raised while building the argv inside `run_model`'s try (temp workdir still cleaned).

### Component 5 — per-model argv (`_lowlevel_extra_argv`)

Add branches:
- `gxe` → `["--env", <materialized_or_path>]` + `_model_options_to_argv(model_options["gxe"])` (e.g. `--gxe-model`).
- `set` → `["--regions", <materialized_or_path>]` + `_model_options_to_argv(model_options["set"])` (e.g. `--set-test`).
- `bayes` → `_model_options_to_argv(model_options["bayes"])` (e.g. `--method`, `--n-signals`); no required extra input.

### Component 6 — registry / dispatch / docs

- `_LOWLEVEL_CLI` += `gxe`, `set`, `bayes` (with their read-back selector).
- `run_model`'s `NotImplementedError` set shrinks to just `mvlmm`.
- `_auto_model` unchanged (gxe/set/bayes are explicit-only).
- **CLI**: add `--env` and `--regions` to the `gwas` subparser; `_cmd_gwas` forwards them (path form on the CLI). `--model-options` already exists.
- **R**: `tg_gwas(env=NULL, regions=NULL)` passthrough (paths; a Series/DataFrame from R is out of scope for the R wrapper — path-only there, documented).
- **CLAUDE.md**: wired list → `lmm/glm/blink/farmcpu/glmm/mklmm/gxe/set/bayes`; not-yet-wired → `mvlmm` only.

## Error handling

- gxe without env / set without regions → friendly `ValueError` (above).
- bayes/set custom output file missing → clear error naming the file.
- An unrecognized `model_options` key → the friendly `ValueError` from the existing `parse_known_args` guard (glmm+mklmm work).

## Testing

Python (`tests/test_api_gwas.py`):
- `gxe` with an in-memory `env` Series → `GwasResult` (uses `P_JOINT`); `gxe` without env → friendly `ValueError`.
- `set` with a small regions file/DataFrame on a tiny fixture → `GwasResult` (region-level, `model="SetBasedScanner"`); `set` without regions → friendly `ValueError`.
- `bayes` → `GwasResult` with PIP-sorted `top_hits`, `lambda_gc is None`, `n_significant` = credible-set size; `model="BayesianVS"`.
- `_read_bayes_output` / `_read_set_output` unit-tested on a tiny hand-written TSV (deterministic; no scan needed).
- `model_options` override reaches the runner (e.g. set `--set-test burden`, bayes `--n-signals 5`).
- Multi-model `GwasComparison` including one new model.
- `mvlmm` still raises a clear `NotImplementedError`.

CLI (`tests/test_cli_gwas.py`): `gwas --help` shows `--env`/`--regions`; `gwas --models set --regions <file>` smoke on the tiny fixture.

R (`rTorchGenomics/tests/testthat/test-gwas.R`): mocked test that `env`/`regions` paths are forwarded to the bridge.

Use HWE-consistent Binomial fixtures; a quantitative trait for gxe/set/bayes with enough samples/variants for the models to run on the tiny fixture. For set, define ≥1 region covering some of the fixture's variants.

## Scope / non-goals

- Only `gxe` + `set` + `bayes`. `mvlmm` (multi-trait) is a separate plan.
- No new statistics; no change to model math or to `lmm_scan`/`glm_scan`.
- R wrapper accepts `env`/`regions` as **paths** only (in-memory materialization is Python-side); documented.
- `_auto_model` unchanged.

## Backward compatibility

Purely additive: a 4th optional element on `_LOWLEVEL_CLI` tuples (default read-back when absent/None), two new `RunOptions` fields (`env`/`regions`, default `None`), two new `gwas()`/CLI/R params (default `None`), three new `_LOWLEVEL_CLI` entries, two new read-back functions. No existing default or signature changes; blink/farmcpu/glm/lmm/glmm/mklmm behavior unchanged.
