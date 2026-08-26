# Design — Fix the broken auto-generated tier-2 R wrappers

**Date:** 2026-08-26
**Status:** APPROVED
**Base branch:** `fix/tier2-r-wrappers` off `feat/mr-coloc-cli-r` (top of the friendly-API stack).

## Problem (root cause — from systematic-debugging investigation)

`rTorchGenomics/R/api_auto.R` contains 32 auto-generated wrappers (`tg_bayes_scan`, `tg_met_scan`, `tg_mediate`, `tg_ldsc`, …). The codegen (`codegen.R:183`) emits, for every tier-2 command:

```r
d <- bridge_call("<name>", args)        # e.g. bridge_call("bayes_scan", args)
GwasResult$new_from_dict(d, command = "<name>")
```

But `bridge_call(name)` resolves `torchgenomics.api.<name>` — and **those api functions do not exist** (only `lmm_scan`/`glm_scan` and the hand-crafted `clump`/`meta`/`mr`/`coloc`/`gwas`/… are real `api` functions). So **calling any of the 32 wrappers raises** `tg_runtime_error: "torchgenomics.api has no function 'bayes_scan'"`. Reproduced directly.

Two defects in the same codegen:
1. **Wrong bridge target** — references nonexistent `api.<name>` functions.
2. **Wrong result class** — all 32 wrap into `GwasResult`, but many tier-2 commands (`ldsc`, `mediate`, `pipeline`, `dosage-call`, `phase-poly`, `combine-gwas-twas`) do not produce a scan/GwasResult shape.

Additional finding: `generate_api_auto()` iterates **all** manifest commands with **no exclusion filter**, so the committed `api_auto.R` is stale vs the current manifest (37 tier-2 commands now, incl. `gwas`/`recommend`/`models`/`mr`/`coloc`, which already have **hand-crafted** `tg_*` wrappers). A naive regenerate would clobber those.

## Chosen approach (user-approved): CLI-runner bridge + `CliRun`

Route each tier-2 auto wrapper to what it actually represents — a **CLI subcommand** — via a real Python `api` callable, returning an honest `CliRun` result (not a faked `GwasResult`). This also gives Python `api.<name>` parity. `torchgenomics.cli.main(argv)` is the proven programmatic entry point (`main` builds the parser, dispatches `handlers[command](args) -> int`); the friendly `_run_lowlevel_cli` already invokes CLI handlers programmatically.

## Design

### Component 1 — `CliRun` result class (Python)

In `torchgenomics/api/_results.py`, a `@dataclass CliRun(_BaseRun)`:
- Fields: `command: str`, `exit_code: int = 0`, `args: dict = {}` (the kwargs passed, JSON-safe), plus inherited `output_files`, `runtime_s`.
- `.summary()`: `f"CLI '{command}' — exit {exit_code}" + output-file lines + runtime`.
- Inherits `.to_dict()`. Export from `_results.py` (no `__all__` there) + `api/__init__.py`.

### Component 2 — the CLI-runner (`api/_cli_bridge.py`)

```python
def run_cli_subcommand(subcommand: str, **kwargs) -> CliRun: ...
```
- Convert kwargs → argv: key `k` → `--k-with-dashes`; `None`/`False` → omitted; `True` → bare flag; list/tuple → the flag followed by each `str(v)` (for `nargs="+"`); else `--flag str(v)`.
- Run `from ..cli import main; rc = main([subcommand, *argv])` inside `timed()`.
- On `rc != 0` → raise a friendly `RuntimeError(f"torchgenomics {subcommand} failed (exit {rc}). Check inputs/paths.")` (never a bare `SystemExit`; if `main`/argparse raises `SystemExit`, catch it and convert to a friendly `RuntimeError`).
- Collect `output_files`: if an `output` / `output_dir` / `output_prefix` kwarg was given and the path (or common scan suffixes `<output>.assoc.tsv`/`.tsv`) exists, add it to `output_files`. Best-effort — a command with no `output` yields an empty `output_files`.
- Return `CliRun(command=subcommand, exit_code=rc, args=<json-safe kwargs>, output_files=..., runtime_s=elapsed())`.

### Component 3 — `api.__getattr__` dynamic tier-2 dispatch

In `torchgenomics/api/__init__.py`, add a module-level `__getattr__(name)`:
- If `name` maps to a tier-2 CLI subcommand (`name.replace("_","-") in _manifest._TIER_2_CLI_SUBCOMMANDS`) **and** `name` is not already a real attribute (a hand-crafted api function), return `functools.partial(run_cli_subcommand, name.replace("_","-"))`.
- Otherwise `raise AttributeError(name)` (normal behavior).
- This makes `api.bayes_scan(genotype=..., phenotype=..., output=...)` callable → so `bridge_call("bayes_scan", args)` resolves. Hand-crafted functions (`gwas`/`mr`/`clump`/…) are real attributes and take precedence — `__getattr__` only fires for missing names.

### Component 4 — fix the R codegen

In `rTorchGenomics/R/codegen.R`:
- Add a **hand-crafted exclusion set** `.HANDCRAFTED_COMMANDS <- c("gwas","recommend","models","mr","coloc","clump","meta","lmm_scan","glm_scan","pgs_fit","pgs_score","ld_blocks","lgebv","annotate_hits","convert","impute","validate")` (every command that has a dedicated hand-crafted `tg_*` wrapper). `generate_api_auto()` skips any manifest command whose `name` is in this set — so regeneration never clobbers hand-crafted wrappers.
- Change the emitted result wrapping from `GwasResult$new_from_dict(d, command=...)` to `CliRun$new_from_dict(d, command=...)`, and the `@return` doc to `CliRun`.

### Component 5 — regenerate the manifest + `api_auto.R`

- Regenerate `rTorchGenomics/inst/rbridge_manifest.json` from the current CLI: `python -m torchgenomics._manifest > .../rbridge_manifest.json` (or the existing emit command — verify how it's produced).
- Run `generate_api_auto()` to rewrite `R/api_auto.R` (now excluding hand-crafted commands, wrapping into `CliRun`). The auto wrapper set should be the tier-2 commands minus the hand-crafted set (~32).

### Component 6 — R `CliRun` S4 + exports

- Add `CliRun` S4 class + `new_from_dict` (accepting a `command=` arg, like the other `new_from_dict`s) in `rTorchGenomics/R/result_classes.R` (mirror `MetaRun`; fields: command, exit_code, args, output_files; plus a `show()`), reusing `%||%`/`.as_*` helpers.
- Add `export(CliRun)` + `exportClasses(CliRun)` to NAMESPACE (mirror MetaRun's form). Add to `_pkgdown.yml`.
- Regenerate `man/*.Rd`.

## Error handling

- Unknown subcommand → argparse in `main()` errors; `run_cli_subcommand` converts the resulting `SystemExit`/nonzero to a friendly `RuntimeError`.
- A failing command (bad path, nonzero exit) → friendly `RuntimeError` naming the subcommand.
- `api.<unknown>` (not a tier-2 command, no real attr) → normal `AttributeError`.

## Testing

Python (`tests/test_api_cli_bridge.py`, new):
- `run_cli_subcommand` on a light real subcommand (e.g. `validate` on the tiny fixture, or `bayes-scan` on a small synthesized input) → returns `CliRun`, `exit_code == 0`, and `output_files` populated when `output` given.
- `api.bayes_scan` is callable (reachable via `__getattr__`) and returns a `CliRun`; `hasattr(api, 'bayes_scan')` is True; a hand-crafted name like `api.gwas` still resolves to the real function (not the CLI runner).
- A failing invocation (bad path) → friendly `RuntimeError` (not `SystemExit`), naming the subcommand.
- `api.definitely_not_a_command` → `AttributeError`.

R (`rTorchGenomics/tests/testthat/test-tier2-wrappers.R`):
- Mocked `bridge_call` test that an auto wrapper (e.g. `tg_bayes_scan`) forwards args and wraps into a `CliRun` S4.
- A regenerated-file sanity check: `api_auto.R` contains no `GwasResult$new_from_dict` (all `CliRun`), and contains no hand-crafted names (`tg_gwas`/`tg_mr`/`tg_coloc`/`tg_recommend`/`tg_models`).
- Integration (guarded): run one auto wrapper (`tg_validate`-equivalent tier-2 or `tg_bayes_scan`) against a fixture and assert a `CliRun` with `exit_code == 0`.

Full Python suite must stay green (the `api.__getattr__` addition must not break existing `from torchgenomics.api import X` or `hasattr` checks — verify manifest/facade/MCP tests).

## Scope / non-goals

- This restores the 32 auto wrappers to a **working, honest** state (they run the CLI subcommand and return a `CliRun` reporting command/exit/output files). It does NOT give each a rich typed result (that would be per-command api facades — a much larger, separate effort).
- No change to any statistics or CLI behavior. `api.__getattr__` is additive; hand-crafted api functions are unaffected.
- MCP: `CliRun`/`run_cli_subcommand`/the tier-2 dynamic callables are NOT registered as MCP tools (no `@tool`) — out of scope, consistent with mr/coloc.

## Backward compatibility

Additive: a new `CliRun` class, a new `_cli_bridge` module, a module-level `api.__getattr__` (fires only for otherwise-missing names), a codegen exclusion set + result-class change, and regenerated `api_auto.R`/manifest/`CliRun` S4. No existing api function, CLI behavior, or hand-crafted R wrapper changes. The 32 auto wrappers change their return type from a (never-working) `GwasResult` to a working `CliRun` — a fix, not a break, since they never worked before.
