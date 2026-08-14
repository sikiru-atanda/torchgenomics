# Design — Wire `glmm` + `mklmm` through `tg.gwas`

**Date:** 2026-08-14
**Status:** APPROVED
**Base branch:** `feat/gwas-glmm-mklmm` off `feat/friendly-api` (the friendly API `tg.gwas` lives on `feat/friendly-api`, PRs open, not yet merged to master).

## Goal

Extend the friendly one-call API `tg.gwas(phenotype, genotype, models=…)` to run two more registry models — `glmm` (SAIGE-style PQL GLMM for binary/categorical traits) and `mklmm` (multi-kernel LMM) — through the same shared dispatch core. This is **thin orchestration over the existing, already-tested CLI runners**; **no new statistics** are introduced. The low-level API and every other surface stay backward-compatible.

This is the first slice of the deferred "wire the remaining registry models through `tg.gwas`" follow-up. Scope for this iteration is **glmm + mklmm only**. `bayes`, `gxe`, `set`, and `mvlmm` remain `NotImplementedError` (with clear messages) and are a later PR — they each need either extra required inputs (`env` / `regions` / ≥2 `traits`) or a divergent result schema (`bayes` writes PIP / credible sets, not an association table).

## Background — verified facts (cross-checked against current code)

The friendly dispatch (`torchgenomics/api/_dispatch.py`) already routes low-level models (`blink`, `farmcpu`) through their CLI single-trait runner via `_run_lowlevel_cli`: it builds a fully-defaulted `argparse.Namespace` by parsing a minimal arg vector against the real subparser, overrides genotype/phenotype/output/correction/device, runs the runner (which writes `<prefix>.assoc.tsv` via `_apply_correction_and_save`), then reads it back into a uniform `GwasResult` via `_read_scan_output`.

Confirmed by inspection of `torchgenomics/cli.py`:

- **`_cmd_glmm_scan`** — single-trait-direct (no `_run_per_trait` wrapper). Reads `--family` (default `binary`) and `--n-categories` (default `3`). Writes `<prefix>.assoc.tsv` via `_apply_correction_and_save` (standard `P`-column schema).
- **`_cmd_mklmm_scan`** — single-trait-direct. Reads `--kernels` (CLI default `additive,dominance,epistatic`). Writes `<prefix>.assoc.tsv` via `_apply_correction_and_save`.
- Both `glmm-scan` and `mklmm-scan` subparsers accept the base arg set the dispatch already passes (`--genotype/--phenotype/--output/--correction/--device`) — verified by parsing.
- Neither has a `_cmd_*_scan_single` variant; the top-level `_cmd_*_scan` is the single-trait entry.

Therefore both models reuse the **existing** `_read_scan_output` (`.assoc.tsv`) unchanged. The only new plumbing is passing model-specific flags (`--family` for glmm; `--kernels` for mklmm) into `_run_lowlevel_cli`.

## Design

### Component 1 — generalize `_run_lowlevel_cli`

Add an `extra_argv: list[str] | None = None` parameter. When present, its tokens are appended to the base argv before `parser.parse_args(...)`. Existing callers (blink/farmcpu) pass nothing and are unaffected.

### Component 2 — register the two models

`_LOWLEVEL_CLI` gains:
- `"glmm": ("glmm-scan", "_cmd_glmm_scan", "GLMM")`
- `"mklmm": ("mklmm-scan", "_cmd_mklmm_scan", "MultiKernelLMM")`

### Component 3 — per-model argv builder

A small helper builds each model's `extra_argv` from `inputs` + `model_options.get(alias, {})`:

- **glmm** — family inferred from `inputs.trait_type`:
  - `binary` → `["--family", "binary"]`
  - `categorical` → `["--family", "multinomial", "--n-categories", str(n_distinct)]` where `n_distinct` is computed from the aligned phenotype. Multinomial is the default (no ordering assumption); a user can override to `ordinal` via `model_options`.
  - `continuous` → raise a friendly `ValueError`: "glmm is for binary/categorical traits; use models='lmm' for a continuous trait." (No silent wrong result.)
  - `model_options["glmm"]` may override/add any glmm flag (`family`, `n_categories`, `firth`, `no_spa`, `pql_max_iter`, …).
- **mklmm** — default `["--kernels", "additive,dominance"]` (overrides the heavier `epistatic`-including CLI default). `model_options["mklmm"]["kernels"]` overrides.

`model_options` values are converted to CLI tokens: key `k` → `--k-with-dashes`; a bool `True` → a bare `--flag` (store_true), `False` → omitted; other scalars → `--flag value`. This is generic and reused by later models.

### Component 4 — surface `model_options`

- `RunOptions` gains `model_options: dict[str, dict] | None = None`.
- `gwas()` gains `model_options: dict[str, dict] | None = None`, threaded into `RunOptions`.
- `run_model` passes `opts.model_options` to the argv builder for `lowlevel:` models.

### Component 5 — shrink the NotImplementedError set

`run_model` now wires `glmm`/`mklmm`; the explicit `NotImplementedError` remains for `bayes`, `gxe`, `set`, `mvlmm` with messages naming the missing capability and the dedicated CLI (e.g. "bayes is not yet wired through tg.gwas — use `torchgenomics bayes-scan`"). `_auto_model` is unchanged: it only ever returns `lmm`/`glm`, so glmm/mklmm are reached only when the user names them explicitly.

### Component 6 — cross-surface parity

- **CLI**: `torchgenomics gwas --models glmm,mklmm` already flows through `api.gwas`. Add an optional `--model-options '<json>'` flag to the `gwas` subparser, parsed to a dict and passed through.
- **R**: `tg_gwas(models = "glmm", model_options = list(glmm = list(family = "ordinal")))` — reticulate converts the named list to a dict; passed straight through the bridge.
- **Docs**: `CLAUDE.md` "Getting started" wired-model list becomes `lmm / glm / blink / farmcpu + glmm / mklmm`; the not-yet-wired list becomes `bayes / gxe / set / mvlmm`.

## Error handling

- glmm on a continuous trait → friendly `ValueError` (above).
- An unknown key in `model_options` for a model → surfaces argparse's clear "unrecognized arguments" error from the subparser (actionable), wrapped with the model name.
- Everything else inherits the existing friendly-error and NotImplementedError behavior of `run_model`.

## Testing

Python (`tests/test_api_gwas.py` additions):
- `tg.gwas(binary_y, G, models="glmm", …)` → `GwasResult`, `n_variants` correct, standard summary.
- `tg.gwas(categorical_y, G, models="glmm", …)` → runs with `family=multinomial` inferred (assert via a spy/records or that it completes and returns the expected variant count).
- `tg.gwas(continuous_y, G, models="glmm")` → friendly `ValueError` mentioning binary/categorical and lmm.
- `tg.gwas(y, G, models="mklmm", …)` → `GwasResult`.
- `model_options` override reaches the runner (e.g. `model_options={"mklmm": {"kernels": "additive,dominance"}}` or glmm `family="ordinal"`) — assert the built argv contains the override (unit-test the argv builder directly).
- Multi-model `GwasComparison` including `glmm`.
- Regression: `bayes`/`gxe`/`set`/`mvlmm` still raise a clear `NotImplementedError`.

CLI (`tests/test_cli_gwas.py`): `gwas --models mklmm` smoke on the tiny fixture; `--model-options '{"mklmm":{"kernels":"additive,dominance"}}'` parses and runs.

R (`rTorchGenomics/tests/testthat/test-gwas.R`): mocked test that `model_options` is forwarded; integration (guarded) that `tg_gwas(models="glmm")` on a binary fixture returns a result.

Use HWE-consistent Binomial fixtures (per the existing test convention) and binary/categorical phenotypes with enough samples for PQL to converge on the tiny fixtures.

## Scope / non-goals

- Only `glmm` + `mklmm` this iteration. `bayes` (PIP schema), `gxe` (`env`), `set` (`regions`), `mvlmm` (multi-trait) are a subsequent PR.
- No new statistics, no change to model math, no change to existing `lmm_scan`/`glm_scan` signatures.
- `_auto_model` unchanged — no auto-selection of glmm/mklmm.

## Backward compatibility

Purely additive: a new optional `extra_argv` param (default off), a new optional `model_options` (default `None`), two new `_LOWLEVEL_CLI` entries, a shrunk NotImplementedError set. No existing signature or default changes.
