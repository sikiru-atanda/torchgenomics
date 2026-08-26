# Fix broken auto-generated tier-2 R wrappers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The 32 auto-generated tier-2 R wrappers (`tg_bayes_scan`, `tg_ldsc`, `tg_mediate`, …) currently raise `tg_runtime_error` because they `bridge_call("<name>")` → a nonexistent `api.<name>`. Make them work by (1) adding a Python `api` CLI-runner so `api.<name>` runs the real CLI subcommand and returns a `CliRun`, and (2) regenerating the codegen to wrap into `CliRun` (not a faked `GwasResult`) and skip hand-crafted commands.

**Architecture:** `torchgenomics.cli.main(argv)` is the programmatic entry point. New `api._cli_bridge.run_cli_subcommand(subcommand, **kwargs) -> CliRun` builds argv, calls `main`, returns a `CliRun`. `api.__getattr__` exposes tier-2 subcommand names as callables (so `bridge_call("bayes_scan")` resolves). R codegen gets a hand-crafted-exclusion set + `CliRun` wrapping; `api_auto.R` + manifest regenerated.

**Tech Stack:** Python 3.12; `torchgenomics.cli`/`_manifest`; R reticulate bridge + codegen.

## Global Constraints

- **No new statistics** — the CLI-runner just invokes existing CLI subcommands. No change to any CLI/stat behavior.
- **Backward compatible / additive** — new `CliRun` class, new `_cli_bridge` module, additive `api.__getattr__` (fires only for otherwise-missing names), codegen exclusion + result-class change, regenerated `api_auto.R`/manifest/S4. No existing api function, CLI behavior, or hand-crafted R wrapper changes.
- **Friendly errors** — `run_cli_subcommand` converts nonzero exit / `SystemExit` into a friendly `RuntimeError` naming the subcommand (never a bare `SystemExit`/traceback).
- **`api.__getattr__` must not shadow real attributes** — hand-crafted api functions (`gwas`/`mr`/`clump`/…) take precedence; `__getattr__` only fires for missing names.
- Reviewer-grade docstrings. Verify commands under `TORCHGENOMICS_DISABLE_NATIVE=1`.
- R files authored via heredoc/`writeLines` (Write unreliable for R); parse-check each.

## File Structure

- Modify: `torchgenomics/api/_results.py` — add `CliRun` dataclass (+ export).
- Create: `torchgenomics/api/_cli_bridge.py` — `run_cli_subcommand`.
- Modify: `torchgenomics/api/__init__.py` — `__getattr__` tier-2 dispatch; export `CliRun`, `run_cli_subcommand`.
- Modify: `rTorchGenomics/R/codegen.R` — exclusion set + `CliRun` wrapping.
- Regenerate: `rTorchGenomics/inst/rbridge_manifest.json`, `rTorchGenomics/R/api_auto.R`.
- Modify: `rTorchGenomics/R/result_classes.R` — `CliRun` S4; `NAMESPACE`; `_pkgdown.yml`; `man/*.Rd`.
- Test: `tests/test_api_cli_bridge.py`, `rTorchGenomics/tests/testthat/test-tier2-wrappers.R`.

**Interface contract:**

```python
# _cli_bridge.py
def run_cli_subcommand(subcommand: str, **kwargs) -> "CliRun": ...
# _results.py — CliRun(_BaseRun): command:str, exit_code:int=0, args:dict={} (+ inherited output_files, runtime_s)
# api/__init__.py
def __getattr__(name: str): ...   # tier-2 subcommand name -> partial(run_cli_subcommand, dashed); else AttributeError
```

---

### Task 1: `CliRun` result class + `run_cli_subcommand` + `api.__getattr__`

**Files:** Modify `torchgenomics/api/_results.py`, `torchgenomics/api/__init__.py`; Create `torchgenomics/api/_cli_bridge.py`; Test `tests/test_api_cli_bridge.py`.

**Interfaces:** Produces `CliRun`, `run_cli_subcommand`, `api.__getattr__` dispatch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_cli_bridge.py
import numpy as np
import pandas as pd
import pytest


def _tiny_bed_fixture():
    """Reuse the repo's tiny PLINK fixture for a real CLI subcommand run."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    bed = root / "tests" / "fixtures" / "tiny.bed"
    pheno = root / "tests" / "fixtures" / "tiny_pheno.txt"
    return str(bed), str(pheno)


def test_run_cli_subcommand_validate(tmp_path):
    from torchgenomics.api import run_cli_subcommand, CliRun
    bed, pheno = _tiny_bed_fixture()
    r = run_cli_subcommand("validate", genotype=bed, phenotype=pheno)
    assert isinstance(r, CliRun) and r.command == "validate" and r.exit_code == 0


def test_api_getattr_exposes_tier2_and_preserves_handcrafted():
    import torchgenomics.api as a
    # a tier-2 CLI subcommand with no hand-crafted api fn is now reachable
    assert callable(a.bayes_scan)
    assert hasattr(a, "bayes_scan")
    # hand-crafted api fns still resolve to the real function, not the CLI runner
    from torchgenomics.api._cli_bridge import run_cli_subcommand
    assert a.gwas is not run_cli_subcommand
    assert getattr(a.gwas, "__name__", "") == "gwas"
    # a genuinely unknown name still raises AttributeError
    with pytest.raises(AttributeError):
        _ = a.definitely_not_a_command


def test_run_cli_subcommand_friendly_error_on_failure():
    from torchgenomics.api import run_cli_subcommand
    with pytest.raises(RuntimeError) as e:
        run_cli_subcommand("lmm-scan", genotype="/no/such.bed", phenotype="/no/such.txt")
    assert "lmm-scan" in str(e.value)   # friendly, names the subcommand; not a SystemExit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_cli_bridge.py -v`
Expected: FAIL (`run_cli_subcommand`/`CliRun`/`api.bayes_scan` missing).

- [ ] **Step 3: Write minimal implementation**

1. `torchgenomics/api/_results.py` — add after `MetaRun`:

```python
@dataclass
class CliRun(_BaseRun):
    """Result of running a tier-2 CLI subcommand via the api CLI-runner bridge.

    These commands (e.g. ``bayes-scan``, ``ldsc``, ``mediate``) are exposed for
    Python/R parity by invoking the real CLI subcommand programmatically. The
    command does its own file I/O; this object reports what ran, its exit code,
    and any output files it wrote — not a rich per-command typed result.
    """

    command: str = ""
    exit_code: int = 0
    args: dict = field(default_factory=dict)

    _kind: ClassVar[str] = "cli"

    def summary(self) -> str:
        lines = [f"CLI '{self.command}' — exit {self.exit_code}"]
        for label, path in self.output_files.items():
            lines.append(f"  {label}: {path}")
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)
```

2. Create `torchgenomics/api/_cli_bridge.py`:

```python
"""Run tier-2 CLI subcommands programmatically for Python/R parity.

The friendly api exposes rich, typed functions for tier-1 workflows. The many
tier-2 CLI subcommands (bayes-scan, ldsc, mediate, ...) don't each have a typed
api facade; this bridge runs them via ``torchgenomics.cli.main`` and returns a
uniform :class:`CliRun` so R (and Python) callers can reach them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ._helpers import timed
from ._results import CliRun


def _kwargs_to_argv(kwargs: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, val in kwargs.items():
        flag = "--" + str(key).replace("_", "-")
        if val is None or val is False:
            continue
        if val is True:
            argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv += [str(v) for v in val]
        else:
            argv += [flag, str(val)]
    return argv


def _collect_output_files(kwargs: dict[str, Any]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for key in ("output", "output_dir", "output_prefix"):
        base = kwargs.get(key)
        if not base:
            continue
        p = Path(str(base))
        if p.exists():
            out[key] = p
        for suffix in (".assoc.tsv", ".tsv"):
            cand = Path(str(base) + suffix)
            if cand.exists():
                out["tsv"] = cand
    return out


def run_cli_subcommand(subcommand: str, **kwargs) -> CliRun:
    """Run a tier-2 CLI ``subcommand`` programmatically and return a ``CliRun``.

    ``kwargs`` are converted to CLI flags (``key`` -> ``--key-with-dashes``;
    ``True`` -> bare flag; ``False``/``None`` -> omitted; list -> repeated
    values for ``nargs``). Raises a friendly :class:`RuntimeError` (never a bare
    ``SystemExit``) when the command exits non-zero or argparse rejects the args.
    """
    from ..cli import main

    argv = [subcommand, *_kwargs_to_argv(kwargs)]
    with timed() as elapsed:
        try:
            rc = main(argv)
        except SystemExit as e:  # argparse error / explicit sys.exit
            code = e.code if isinstance(e.code, int) else 2
            raise RuntimeError(
                f"torchgenomics {subcommand} failed (exit {code}); check the "
                f"arguments and input paths."
            ) from None
        if rc != 0:
            raise RuntimeError(
                f"torchgenomics {subcommand} failed (exit {rc}); check the "
                f"arguments and input paths."
            )
        return CliRun(
            runtime_s=elapsed(),
            output_files=_collect_output_files(kwargs),
            command=subcommand,
            exit_code=int(rc),
            args={str(k): (str(v) if isinstance(v, Path) else v) for k, v in kwargs.items()},
        )
```

3. `torchgenomics/api/__init__.py` — import `CliRun` + `run_cli_subcommand`, add to `__all__`, and add a module `__getattr__`:

```python
def __getattr__(name: str):
    """Expose tier-2 CLI subcommands as api callables (Python/R parity).

    Fires only for names not already defined on the module, so hand-crafted
    api functions (gwas/mr/clump/...) always take precedence. A name matching a
    tier-2 CLI subcommand returns a callable that runs that subcommand via
    :func:`run_cli_subcommand`; anything else raises ``AttributeError``.
    """
    from functools import partial
    from .._manifest import _TIER_2_CLI_SUBCOMMANDS
    dashed = name.replace("_", "-")
    if dashed in _TIER_2_CLI_SUBCOMMANDS:
        return partial(run_cli_subcommand, dashed)
    raise AttributeError(f"module 'torchgenomics.api' has no attribute {name!r}")
```

(Verify `_TIER_2_CLI_SUBCOMMANDS` is importable from `torchgenomics._manifest`; it is defined there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_cli_bridge.py -v`
Expected: PASS.
Regression: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/ -k "api and (facade or manifest or registry or mcp)" -q` → no regressions (the `__getattr__` must not break `from torchgenomics.api import X`, `hasattr` checks, or the MCP/registry introspection).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_results.py torchgenomics/api/_cli_bridge.py torchgenomics/api/__init__.py tests/test_api_cli_bridge.py
git commit -m "api: CliRun + run_cli_subcommand + __getattr__ tier-2 dispatch (fixes bridge target for auto R wrappers)"
```

---

### Task 2: Fix the R codegen (exclusion set + CliRun wrapping) + regenerate

**Files:** Modify `rTorchGenomics/R/codegen.R`; Regenerate `rTorchGenomics/inst/rbridge_manifest.json` + `rTorchGenomics/R/api_auto.R`; Test `rTorchGenomics/tests/testthat/test-tier2-wrappers.R`.

**IMPORTANT:** author/modify R via heredoc/`sed`; parse-check with `Rscript -e 'invisible(parse("<file>"))'`.

- [ ] **Step 1: Write the failing test**

```r
# rTorchGenomics/tests/testthat/test-tier2-wrappers.R
test_that("regenerated api_auto.R wraps into CliRun and excludes hand-crafted commands", {
  src <- readLines(system.file("..", "R", "api_auto.R", package = "rTorchGenomics",
                               mustWork = FALSE))
  if (length(src) == 0) src <- readLines("R/api_auto.R")
  joined <- paste(src, collapse = "\n")
  # no auto wrapper fakes a GwasResult anymore
  expect_false(grepl("GwasResult\\$new_from_dict", joined))
  # every auto wrapper wraps into CliRun
  expect_true(grepl("CliRun\\$new_from_dict", joined))
  # hand-crafted commands are NOT auto-generated (no clash)
  for (fn in c("tg_gwas <- function", "tg_mr <- function", "tg_coloc <- function",
               "tg_recommend <- function", "tg_models <- function")) {
    expect_false(grepl(fn, joined, fixed = TRUE))
  }
})

test_that("tg_bayes_scan forwards to the bridge and wraps into CliRun", {
  captured <- NULL
  mockery::stub(tg_bayes_scan, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(command = "bayes-scan", exit_code = 0L, output_files = list(), args = list())
  })
  r <- tg_bayes_scan(genotype = "g.bed", phenotype = "p.txt")
  expect_equal(captured$fn, "bayes_scan")
  expect_s4_class(r, "CliRun")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-tier2-wrappers.R")'`
Expected: FAIL (`api_auto.R` still uses `GwasResult`; `CliRun` S4 missing — Task 3 adds it, so this task's second test may also need Task 3; keep both tasks' pieces coherent — see note).

- [ ] **Step 3: Write minimal implementation**

1. `rTorchGenomics/R/codegen.R`:
   - Add near the top: `.HANDCRAFTED_COMMANDS <- c("gwas","recommend","models","mr","coloc","clump","meta","lmm_scan","glm_scan","pgs_fit","pgs_score","ld_blocks","lgebv","annotate_hits","convert","impute","validate")`.
   - In `generate_api_auto()`, filter the commands before rendering: `manifest$commands <- Filter(function(cmd) !(cmd$name %in% .HANDCRAFTED_COMMANDS), manifest$commands)`.
   - In `.render_wrapper` (line ~179-184), change `#' @return A \`GwasResult\`` → `#' @return A \`CliRun\`` and `GwasResult$new_from_dict(d, command = "..."` → `CliRun$new_from_dict(d, command = "..."`.

2. Regenerate the manifest + api_auto.R:
```bash
TORCHGENOMICS_DISABLE_NATIVE=1 python -m torchgenomics._manifest > rTorchGenomics/inst/rbridge_manifest.json
cd rTorchGenomics && Rscript -e 'source("R/codegen.R"); generate_api_auto("inst/rbridge_manifest.json", "R/api_auto.R")' && cd ..
```
   Verify: `grep -c "CliRun\$new_from_dict" rTorchGenomics/R/api_auto.R` == the auto-wrapper count; `grep -c "GwasResult" rTorchGenomics/R/api_auto.R` == 0; no `tg_gwas`/`tg_mr`/`tg_coloc`/`tg_recommend`/`tg_models` in api_auto.R.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-tier2-wrappers.R")'`
Expected: the first test (file-content) PASSES; the second (S4 wrap) passes once Task 3's `CliRun` S4 exists. If running Task 2 before Task 3, the second `test_that` may error on missing `CliRun` — that's expected; it goes green after Task 3. (Alternatively implement Task 3's `CliRun` S4 first — see note below.)

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/codegen.R rTorchGenomics/inst/rbridge_manifest.json rTorchGenomics/R/api_auto.R rTorchGenomics/tests/testthat/test-tier2-wrappers.R
git commit -m "r(codegen): exclude hand-crafted commands + wrap auto wrappers into CliRun; regenerate api_auto.R + manifest"
```

**NOTE for the implementer:** Tasks 2 and 3 are coupled (regenerated `api_auto.R` references `CliRun` S4). Implement Task 3's `CliRun` S4 class FIRST (or in the same sitting) so `devtools::load_all()` succeeds and both tests pass. The controller may merge Tasks 2+3 into one dispatch if cleaner.

---

### Task 3: R `CliRun` S4 class + NAMESPACE + pkgdown + man

**Files:** Modify `rTorchGenomics/R/result_classes.R`, `rTorchGenomics/NAMESPACE`, `rTorchGenomics/_pkgdown.yml`, `rTorchGenomics/man/*.Rd`.

- [ ] **Step 1: Write the failing test** — covered by the second `test_that` in `test-tier2-wrappers.R` (Task 2): `expect_s4_class(r, "CliRun")`.

- [ ] **Step 2: Run test to verify it fails** — `CliRun` S4 undefined → error.

- [ ] **Step 3: Write minimal implementation**

Add to `rTorchGenomics/R/result_classes.R` (mirror `MetaRun`; reuse `%||%`/`.as_int`/`.as_chr`/`.as_list`):

```r
setClass("CliRun", contains = "_BaseRun",
  representation(command = "character", exit_code = "integer",
                 args = "list", raw = "list"))
CliRun <- list(new_from_dict = function(d, command = NULL) {
  new("CliRun",
      command = .as_chr(d$command %||% command %||% ""),
      exit_code = .as_int(d$exit_code %||% 0L),
      args = .as_list(d$args %||% list()),
      output_files = .as_list(d$output_files %||% list()),
      raw = d)
})
```

(Match the exact base-class slot names — read how `MetaRun`'s `new_from_dict` sets inherited `_BaseRun` slots like `output_files`/`runtime_s`; reuse the same helpers/idiom. Add a `setMethod("show", "CliRun", ...)` printing command + exit + output files, consistent with siblings.)

Add to `NAMESPACE`: `export(CliRun)` + `exportClasses(CliRun)` (mirror MetaRun). Add `CliRun` to `_pkgdown.yml` result-classes section. Regenerate man: `Rscript -e 'roxygen2::roxygenise()'`.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-tier2-wrappers.R")'` → both tests PASS. Then the FULL R mocked suite: `Rscript -e 'devtools::load_all("."); testthat::test_dir("tests/testthat")'` → no failures (pre-existing integration skips OK).

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/result_classes.R rTorchGenomics/NAMESPACE rTorchGenomics/_pkgdown.yml rTorchGenomics/man
git commit -m "r: CliRun S4 result class + NAMESPACE/pkgdown/man for tier-2 auto wrappers"
```

---

### Task 4: End-to-end verification (Python + optional R integration)

**Files:** Modify `tests/test_api_cli_bridge.py` (add an end-to-end auto-command test).

- [ ] **Step 1: Write the failing/confirming test**

```python
# tests/test_api_cli_bridge.py (append)
def test_api_bayes_scan_runs_end_to_end(tmp_path):
    """The api CLI-runner reachable as api.bayes_scan actually runs the scan."""
    import torchgenomics.api as a
    from torchgenomics.api import CliRun
    bed, pheno = _tiny_bed_fixture()
    out = tmp_path / "bayes_out"
    r = a.bayes_scan(genotype=bed, phenotype=pheno, output=str(out))
    assert isinstance(r, CliRun) and r.command == "bayes-scan" and r.exit_code == 0
```

- [ ] **Step 2: Run to verify** — RED if the bridge path has any gap; else GREEN.

- [ ] **Step 3: Implement** — none expected (Tasks 1-3 cover it); if `bayes-scan` needs a flag the fixture lacks, either supply it or pick a lighter tier-2 command (e.g. `validate`-adjacent) that runs on the tiny fixture. If a real gap surfaces, fix it minimally in `_cli_bridge.py`.

- [ ] **Step 4: Run** — `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_cli_bridge.py -q` → all green. Then the FULL suite: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/ -q` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add tests/test_api_cli_bridge.py
git commit -m "test: end-to-end tier-2 api CLI-runner (api.bayes_scan runs the scan -> CliRun)"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (CliRun Python) → Task 1. ✅
- §Component 2 (run_cli_subcommand) → Task 1. ✅
- §Component 3 (api.__getattr__) → Task 1. ✅
- §Component 4 (codegen exclusion + CliRun wrapping) → Task 2. ✅
- §Component 5 (regenerate manifest + api_auto.R) → Task 2. ✅
- §Component 6 (R CliRun S4 + exports) → Task 3. ✅
- §Testing (run_cli_subcommand; __getattr__ precedence + AttributeError; friendly error; regenerated-file content; mocked S4 wrap; e2e) → Tasks 1-4. ✅

**Placeholder scan:** No TBD/TODO; code steps carry real code. The regenerate step is exact commands.

**Type consistency:** `CliRun` fields (command/exit_code/args/output_files) set in Task 1 (Python) match the R `CliRun` S4 (Task 3) and the codegen's `CliRun$new_from_dict(d, command=...)` (Task 2). `run_cli_subcommand`/`api.__getattr__` names consistent across tasks.

**Known verification points for the implementer:**
- (a) Confirm `main()`'s return contract: it returns the handler's int (0 on success). If any handler returns None on success, treat None as 0 in `run_cli_subcommand` (guard `rc = rc or 0`).
- (b) `api.__getattr__` interaction with `hasattr`/`from ... import`: ensure existing star-imports and MCP registry introspection (`registered_tools`) don't accidentally trigger it for real names — real attributes bypass `__getattr__`, so only genuinely-missing names hit it. Run the facade/MCP tests.
- (c) Tasks 2+3 are coupled (regenerated api_auto.R needs the CliRun S4); implement the S4 (Task 3) before/with the regenerate (Task 2) so `load_all()` succeeds.
- (d) The regenerated auto-wrapper count = tier-2 commands (37) minus the hand-crafted set — confirm no hand-crafted `tg_*` clash and that `tg_bayes_scan` et al. now wrap `CliRun`.
- (e) `_collect_output_files` is best-effort; a command whose output flag isn't named output/output_dir/output_prefix just yields empty output_files — acceptable (CliRun still reports command+exit_code).
