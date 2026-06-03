# rTorchGenomics — R-side wrapper for TorchGenomics

**Status:** approved design, ready for implementation plan
**Author:** brainstormed 2026-06-03 with the project owner
**Target release:** rTorchGenomics v0.4.0 (version-locked to torchgenomics v0.4.0)

## Context

TorchGenomics (renamed from `torchgwas` in v0.4.0, 2026-06-03) is a
GPU-accelerated PyTorch engine for statistical and quantitative genomics —
GWAS + post-GWAS + PGS + LD + imputation + multi-omics + annotation +
visualization. Python is the canonical interface; v0.4.0 added a curated
12-function `torchgenomics.api` facade for one-call workflows and a
`torchgenomics-mcp` server for LLM clients.

The R community is a large, scientifically-active audience for GWAS work
(GenomicRanges, plinkR, SNPRelate, lme4, sommer, BGLR). They have not been
served by TorchGenomics so far. The goal of `rTorchGenomics` is to give R
users a clean, idiomatic R interface to the same engine — not a Python
shim, an R-first experience that happens to call Python underneath. R-user
adoption hinges on three things:

1. **Install just works.** One R command sets up everything.
2. **Calls are responsive.** No multi-second wait per `tg_*()` call.
3. **Returns look like R.** Tibbles, S4 classes, ggplot2 figures — not
   `<list of dicts>` text dumps.

This spec covers the design that meets those three bars. Implementation
follows in a separate plan once the spec is approved.

## Goals

- **G1.** Ship an R package `rTorchGenomics` covering the full TorchGenomics
  CLI surface (12 polished tier-1 wrappers + 31 auto-generated tier-2
  wrappers = 43 total) plus an escape hatch for the long-tail API.
- **G2.** Per-call latency in the millisecond range after the first call
  in an R session. First call pays one-time ~1-3 sec for Python init.
- **G3.** R-idiomatic returns: S4 result objects with tibble payloads;
  `ggplot2` plots; `summary()` / `show()` methods.
- **G4.** One-command install via `tg_install()`. R user never touches
  pip, venv, conda manually.
- **G5.** Available on GitHub + r-universe immediately; CRAN-ready API
  surface (skip-on-CRAN guards for Python-dependent tests). Bioconductor
  submission deferred until post-publication.
- **G6.** Version-locked to the parent Python package: `rTorchGenomics`
  v0.4.0 strictly requires `torchgenomics==0.4.0` in its managed venv.

## Non-goals

- **NG1.** No re-implementation of TorchGenomics in R. The Python engine
  is the source of truth; R is a façade.
- **NG2.** No MCP-protocol involvement. The MCP server stays Python-side
  for LLM clients. R uses reticulate directly.
- **NG3.** No replacement for native R packages (sommer, BGLR, lme4).
  `rTorchGenomics` is complementary — for GPU acceleration, biobank-scale
  streaming, post-GWAS integration that those packages lack.
- **NG4.** Not in v0.4.0: tidymodels integration, parsnip recipes,
  workflowsets bindings. Add later if the audience asks.

## Architecture

```
                  R session
                  ├── api.R (12 hand-crafted + 31 auto-generated = 43 funcs)
                  ├── result_classes.R (S4 result classes)
                  ├── plots.R (ggplot2 manhattan / qq)
                  ├── bridge.R (reticulate ↔ torchgenomics.api)
                  ├── install.R (tg_install)
                  └── codegen.R (build-time auto-gen)
                            │
                            │ reticulate (in-process Python via Python C API)
                            ▼
                  Python interpreter (one per R session)
                  └── torchgenomics.api.* (the existing facade)

                  Independent: torchgenomics-mcp server, untouched.
                  Used by Claude Desktop / Claude Code, not by R.
```

### Bridge mechanism — reticulate

Reticulate hosts a Python interpreter inside the R process via the
Python C API. This means:

- First `tg_*()` call in an R session: ~1-3 sec to spin up Python and
  import `torch` + `torchgenomics`.
- Subsequent calls: microseconds for arg conversion + the actual compute
  time.

R never speaks JSON-RPC, never spawns subprocesses, never marshals
through stdio. R passes paths + primitives to `torchgenomics.api.lmm_scan`
directly; the api function returns a `ScanRun` dataclass; we call
`.to_dict()` to get a JSON-safe dict; reticulate converts that dict to
a nested R list; we construct an S4 `ScanRun` from the list.

**Why not MCP**: MCP is for LLM tool clients that need protocol-level
tool discovery, schema negotiation, capability listings. R doesn't need
any of that — we hand-write the wrappers. Forcing MCP between R and
Python would couple the R package to the MCP protocol's evolution.

**Why not per-call subprocess**: each fresh Python process pays 1-3 sec
to import torch. Hurts interactive R workflows (`tg_validate`,
`tg_convert`, `tg_manhattan` in quick succession).

**Why not a hand-rolled daemon**: reticulate already solves persistent
in-process Python. Re-implementing it is reinventing the wheel.

### Three audiences served

| Audience | Entry point | Example |
|---|---|---|
| **Python users** | `import torchgenomics as tg; tg.lmm_scan(...)` | Untouched by this spec |
| **LLM tool clients** | `torchgenomics-mcp` stdio MCP server | Untouched by this spec |
| **R users** | `library(rTorchGenomics); tg_lmm_scan(...)` | This spec |

All three call the same `torchgenomics.api.*` functions under the hood.
A bug fix in the Python facade benefits all three audiences immediately.

## Components

### `rTorchGenomics/` package layout

```
rTorchGenomics/
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
├── README.md
├── R/
│   ├── api.R              # 12 hand-crafted polished wrappers
│   ├── api_auto.R         # 31 auto-generated tier-2 wrappers (built from manifest)
│   ├── result_classes.R   # S4 ValidateRun, ScanRun, LDBlocksRun, ClumpRun,
│   │                      # MetaRun, PgsFitRun, PgsScoreRun, AnnotateRun,
│   │                      # PlotResult, ConvertRun, ImputeRun, GwasResult (generic)
│   ├── plots.R            # tg_manhattan, tg_qq — native ggplot2
│   ├── bridge.R           # tg_py() singleton; py_api$<fn>() + py_to_r + error mapping
│   ├── install.R          # tg_install() — reticulate venv + pip install
│   ├── codegen.R          # build-time: reads manifest, writes api_auto.R
│   ├── utils.R
│   └── zzz.R              # .onLoad: configure reticulate to use managed venv
├── inst/
│   ├── extdata/           # tiny PLINK + phenotype fixture
│   └── rbridge_manifest.json  # 31 tier-2 funcs + argparse schemas
├── tests/
│   ├── testthat.R
│   └── testthat/
│       ├── test-bridge.R         # mocks reticulate; assert dict→S4 round-trip
│       ├── test-result_classes.R
│       ├── test-api.R            # all 43 wrappers exist, validate args
│       ├── test-plots.R          # ggplot structure assertions
│       ├── test-codegen.R
│       └── test-integration.R    # opt-in: real reticulate + tiny fixture
├── vignettes/
│   ├── quickstart.Rmd
│   ├── lmm-walkthrough.Rmd
│   └── reticulate-bridge.Rmd
├── man/                   # roxygen2-generated
└── pkgdown/
    └── _pkgdown.yml
```

### Module ownership

| File | Purpose | R deps |
|---|---|---|
| `api.R` | 12 hand-crafted wrappers (validate, lmm_scan, glm_scan, ld_blocks, clump, meta, pgs_fit, pgs_score, annotate_hits, manhattan, qq, impute, convert). Each ~15-30 lines: arg validation → `tg_py()$api$<fn>(!!!args)$to_dict()` → `py_to_r` → S4. Full roxygen2 docs with `@examples` blocks. | `methods`, `tibble` |
| `api_auto.R` | 31 stub wrappers generated at build time from `inst/rbridge_manifest.json`. Each ~5-10 lines: build args, call bridge, return generic `GwasResult` S4 wrapping summary + output files + top hits (if available). Auto-generated comment at top. | `methods`, `tibble` |
| `result_classes.R` | S4 class definitions, one per Python result type, plus generic `GwasResult` for tier-2. `setClass`, `setMethod("show", ...)`, `setMethod("summary", ...)`, accessor functions (`top_hits`, `output_files`, etc.). | `methods`, `tibble` |
| `bridge.R` | `tg_py()` returns the cached `reticulate::import("torchgenomics")` reference. `bridge_call(fn_name, args)` invokes it with error mapping. ~80 lines total. | `reticulate`, `methods` |
| `python_runtime.R` | Locate the managed venv; configure reticulate to use it (`reticulate::use_virtualenv("r-rtorchgenomics", required = TRUE)`). | `reticulate` |
| `install.R` | `tg_install(python_version = "3.12", method = c("virtualenv", "conda"))` — bootstraps Python if needed, creates venv `r-rtorchgenomics`, pip installs `torchgenomics[mcp]==<R_pkg_version>`. Records install state to `tools::R_user_dir("rTorchGenomics", "config")/runtime.json`. | `reticulate` |
| `codegen.R` | Standalone R script run at package build (and in tests). Reads `inst/rbridge_manifest.json`, writes `R/api_auto.R`. Idempotent. Manifest is regenerated by `python -m torchgenomics._manifest > inst/rbridge_manifest.json` — a tiny helper in the Python package. | (none — base R) |
| `plots.R` | `tg_manhattan(scan_run)` reads `scan_run@output_files$tsv` into a tibble, builds the ggplot2 figure. Same for `tg_qq`. ~50 lines each. | `ggplot2`, `tibble`, `readr` |
| `utils.R` | Path coercion, tempdir guards, `jsonlite::fromJSON`-safe wrappers. | `jsonlite` |
| `zzz.R` | `.onLoad` — sets reticulate's preferred venv; no-op if `tg_install()` hasn't been run. `.onAttach` — friendly hint with package version. | (none) |

### Python-side companion files

One small Python helper added to the torchgenomics repo:

```
torchgenomics/_manifest/
├── __init__.py     # builds the tier-2 manifest from cli.py argparse subparsers
└── __main__.py     # entry point: writes JSON to stdout
```

Invoked at rTorchGenomics build time:
```bash
python -m torchgenomics._manifest > rTorchGenomics/inst/rbridge_manifest.json
```

Output schema:
```json
{
  "torchgenomics_version": "0.4.0",
  "commands": [
    {
      "name": "mvlmm_scan",
      "cli_subcommand": "mvlmm-scan",
      "description": "Multi-trait mvLMM association scan",
      "args": [
        {"name": "genotype", "type": "str", "required": true, "default": null},
        {"name": "phenotype", "type": "str", "required": true, "default": null},
        {"name": "traits", "type": "str", "required": false, "default": null}
        ...
      ]
    }
    ...
  ]
}
```

The R-side `codegen.R` reads this and emits R function stubs.

## Data flow — one operation end to end

Tracing `r <- tg_lmm_scan(genotype = "data.bed", phenotype = "pheno.tsv")`:

1. **R-side arg validation** (`api.R`):
   - Coerce `genotype`, `phenotype` to character strings.
   - Apply defaults: `test = "wald"`, `correction = "bh"`, `chunk_size = 10000`, etc.
   - Build a named list of all args.

2. **Bridge invocation** (`bridge.R`):
   ```r
   py_api <- tg_py()$api
   py_result <- do.call(py_api$lmm_scan, args)
   dict_result <- py_api$lmm_scan(!!!args)$to_dict()   # API-style; preferred
   list_result <- reticulate::py_to_r(dict_result)
   ```

3. **Python compute** (in-process via reticulate):
   - `torchgenomics.api.lmm_scan(**args)` runs through the existing facade.
   - GRM streaming, REML fit, scan loop, multiple-testing correction.
   - Writes TSV + Parquet to the run dir.
   - Returns a `ScanRun` dataclass.

4. **Result serialization** (Python side):
   - `result.to_dict()` returns a JSON-safe dict — primitives + lists +
     `list[dict]` for tibbles, paths as strings.
   - Reticulate auto-converts to a nested R list.

5. **S4 construction** (R side):
   ```r
   new("ScanRun",
     n_variants = list_result$n_variants,
     n_significant = list_result$n_significant,
     lambda_gc = list_result$lambda_gc,
     top_hits = tibble::as_tibble(list_result$top_hits),
     output_files = list_result$output_files,
     ...
   )
   ```

6. **Return**: R user gets a `ScanRun` S4 object.

7. **Subsequent access**: `r@top_hits`, `top_hits(r)`, `summary(r)`,
   `tg_manhattan(r)`. Each is a pure R operation; no Python re-entry
   unless the user calls another `tg_*()` function.

## Lifecycle

| Phase | What happens | Cost |
|---|---|---|
| Package install (CRAN/r-universe) | Standard R package install. No Python touched. | ~5 sec |
| `library(rTorchGenomics)` | `.onLoad` configures reticulate to prefer `r-rtorchgenomics` venv if it exists. Lazy. No Python imports. `.onAttach` prints a one-liner. | <100 ms |
| `tg_install()` (first time) | `reticulate::install_python("3.12")` if no Python on system; `reticulate::virtualenv_create("r-rtorchgenomics", python = "3.12")`; `reticulate::py_install("torchgenomics[mcp]==<R_pkg_version>", envname = "r-rtorchgenomics")`. Writes install state to user config dir. | 60-180 sec, one-time |
| First `tg_*()` call | `tg_py()` calls `reticulate::import("torchgenomics")`. Python interpreter starts; loads torch, numpy, torchgenomics. | one-time ~1-3 sec |
| Subsequent `tg_*()` calls | Direct in-process call; reticulate marshals args. | per-call ~ms + compute |
| Session end | Reticulate cleans up Python. Nothing for us to do. | ~0 |

## Error handling

Three R condition classes, mapped from Python exception types:

| R class | Python source | Trigger | R message format |
|---|---|---|---|
| `tg_no_python` | (none — R-side check) | `tg_install()` not run; no managed venv at expected path | "TorchGenomics Python runtime not configured. Run tg_install() to set it up." |
| `tg_input_error` | `python.builtin.ValueError`, `python.builtin.TypeError`, `python.builtin.FileNotFoundError` | Bad args, missing input files | "<fn>(): <python message>" |
| `tg_runtime_error` | Everything else (`python.builtin.RuntimeError`, `python.builtin.RecursionError`, REML divergence, GPU OOM, ...) | Computation failed | "<fn>(): <python exception type>: <message>\n<traceback excerpt>" |

`bridge.R::bridge_call` wraps every reticulate call in:
```r
tryCatch(
  py_api[[fn_name]](!!!args)$to_dict(),
  python.builtin.ValueError = function(e) {
    stop(structure(
      class = c("tg_input_error", "tg_python_error", "error", "condition"),
      list(message = conditionMessage(e), call = sys.call(-2L))
    ))
  },
  python.builtin.RuntimeError = function(e) {
    stop(structure(
      class = c("tg_runtime_error", "tg_python_error", "error", "condition"),
      list(message = paste(conditionMessage(e), reticulate::py_last_error()$traceback %||% "", sep = "\n"))
    ))
  }
  # ... other categories ...
)
```

Reticulate exposes Python exception class hierarchies as R conditions, so
`tryCatch` matches by Python class name. Users can write idiomatic R:
```r
result <- tryCatch(
  tg_lmm_scan("missing.bed", "pheno.tsv"),
  tg_input_error = function(e) {
    message("Input file missing — falling back to default")
    NULL
  }
)
```

## Testing

| Test file | What it covers | Needs Python? |
|---|---|---|
| `test-bridge.R` | Mock `reticulate::import` and its returned module. Assert: arg passthrough, `to_dict()` invocation, `py_to_r` conversion, error condition mapping for each of 4 Python exception types. | No |
| `test-result_classes.R` | Construct each S4 class from fixture JSON. Assert: `show()` output non-empty, `summary()` returns string, `top_hits()` accessor returns tibble, slot types correct. | No |
| `test-api.R` | For each of the 43 wrappers: signature matches docs; missing-required-arg raises; optional-arg defaults apply; bridge is called with correct payload (mocked). | No |
| `test-plots.R` | `tg_manhattan(r)` and `tg_qq(r)` return `gg` objects. Layer count + aesthetic mappings + axis labels correct. | No |
| `test-codegen.R` | Run `codegen()` against bundled manifest. Assert: 31 generated functions exist, names match, argument lists match the manifest. | No |
| `test-install.R` | Mock `reticulate::virtualenv_create` and `py_install`. Assert: `tg_install()` writes the state file with the right version pin. | No |
| `test-integration.R` | **Opt-in only.** Gated by `Sys.getenv("RTORCHGENOMICS_INTEGRATION") == "1"`. Real `tg_install()` into a temp venv (or assume one exists). Real `tg_validate()` + `tg_lmm_scan()` on `inst/extdata/tiny.bed`. Assert resulting S4 matches expectations. | Yes |

Unit tests run on every CRAN/Bioc CI machine without Python. Integration
tests run on a self-hosted runner with the venv pre-installed.

## Installation UX

The R user's first session:

```r
install.packages("rTorchGenomics",
                 repos = "https://sikiru-atanda.r-universe.dev")
library(rTorchGenomics)

# One-time Python setup. Idempotent — safe to re-run.
tg_install()
# 🐍 Setting up Python runtime for rTorchGenomics...
# ✓ Python 3.12.7 installed via reticulate.
# ✓ Created venv 'r-rtorchgenomics'.
# ✓ Installed torchgenomics[mcp]==0.4.0 (and 41 dependencies).
# Setup complete. tg_*() calls are now ready.

# Run a scan.
r <- tg_lmm_scan(genotype = "data.bed", phenotype = "pheno.tsv")
print(r)
# ScanRun (SingleTraitLMM, test=wald, correction=bh)
#   Samples: 2500   Variants tested: 489102
#   Significant @ α=5.00e-08: 12
#   λ_GC: 1.012
#   ...
plot_obj <- tg_manhattan(r)
ggplot2::ggsave("manhattan.png", plot_obj)
```

Subsequent R sessions: `library(rTorchGenomics)` is enough; the venv
persists across sessions.

## Distribution

| Channel | Cadence | Audience | When |
|---|---|---|---|
| GitHub | Tagged with each `v0.x.y` of torchgenomics | Devs, early adopters | v0.4.0 (this release) |
| r-universe | Auto-built from GitHub pushes to `master`. Install via `install.packages("rTorchGenomics", repos = "https://sikiru-atanda.r-universe.dev")` | Day-1 R adopters | v0.4.0 |
| CRAN | After API freeze; SystemRequirements declared; skip-on-CRAN guards for Python-dependent tests | Mainstream R community | Target: post-v0.5.0 of torchgenomics |
| Bioconductor | Biannual cycle; BiocCheck pass; aligned with manuscript publication | Genomics research community | Target: post-v0.5.0 + manuscript acceptance |

## Versioning

The R package version mirrors `torchgenomics`. v0.4.0 of `rTorchGenomics`
strictly requires Python `torchgenomics==0.4.0` in its managed venv;
`tg_install()` enforces the pin. This guarantees that R wrappers can't
desync from the Python facade they wrap.

`DESCRIPTION`:
```
Version: 0.4.0
SystemRequirements: Python (>= 3.10), torchgenomics (== 0.4.0)
```

`install.R`:
```r
TARGET_PYTHON_PACKAGE <- "torchgenomics[mcp]==0.4.0"
```

Both literals are updated together when the parent version bumps.

## Critical files to add / modify

**New, R-side (rTorchGenomics package):**
- `rTorchGenomics/DESCRIPTION`, `NAMESPACE`, `LICENSE`, `README.md`
- `rTorchGenomics/R/{api,api_auto,result_classes,plots,bridge,install,codegen,utils,zzz,python_runtime}.R`
- `rTorchGenomics/inst/extdata/` (PLINK fixture)
- `rTorchGenomics/inst/rbridge_manifest.json` (generated)
- `rTorchGenomics/tests/testthat/test-*.R` (7 files)
- `rTorchGenomics/vignettes/{quickstart,lmm-walkthrough,reticulate-bridge}.Rmd`
- `rTorchGenomics/man/` (roxygen-generated)
- `rTorchGenomics/pkgdown/_pkgdown.yml`

**New, Python-side (torchgenomics package):**
- `torchgenomics/_manifest/__init__.py`, `__main__.py` — emits the
  tier-2 manifest as JSON. Reads `torchgenomics.cli`'s argparse
  subparsers to extract command schemas.

**New CI:**
- `.github/workflows/rcheck.yml` — R CMD check on every push;
  pkgdown::build_site on master; r-universe builds from master.

**Reuse from existing code:**
- `torchgenomics.api.*` — all 12 tier-1 functions called directly.
- `torchgenomics.cli` argparse subparsers — read by `_manifest` to
  enumerate tier-2 commands.
- `torchgenomics.api._results._BaseRun.to_dict()` — already JSON-safe;
  no changes needed.

## Rollout phases (executed during writing-plans, summarized here)

1. **Python manifest emitter**: add `torchgenomics/_manifest/` + tests
   that the JSON shape matches the spec. Commit: `feat(manifest): add
   rbridge manifest emitter for rTorchGenomics codegen`.
2. **R package scaffold**: create `rTorchGenomics/` with DESCRIPTION,
   NAMESPACE, empty R/, tests/, vignettes/. Commit: `feat(r): scaffold
   rTorchGenomics package`.
3. **Bridge + install**: `R/{bridge,install,zzz,python_runtime}.R` +
   matching unit tests. Verify `tg_install()` works in a clean Linux
   container. Commit: `feat(r): reticulate bridge + tg_install()`.
4. **Result classes**: `R/result_classes.R` with all 12 S4 classes +
   `GwasResult` generic + show/summary methods. Commit: `feat(r): S4
   result classes`.
5. **Hand-crafted api wrappers**: `R/api.R` with all 12 polished
   functions + roxygen docs + examples. Commit: `feat(r): tier-1 api
   wrappers (12 hand-crafted)`.
6. **Codegen + auto wrappers**: `R/codegen.R` + bundled manifest +
   generated `R/api_auto.R`. Commit: `feat(r): codegen for tier-2 (31
   auto-generated wrappers)`.
7. **Plots**: `R/plots.R` with ggplot2 manhattan + qq. Commit:
   `feat(r): native ggplot2 plotting`.
8. **Vignettes + docs site**: three Rmd vignettes; pkgdown config;
   README.md with quickstart. Commit: `docs(r): vignettes + pkgdown
   site`.
9. **Integration tests + CI**: `tests/testthat/test-integration.R` +
   `.github/workflows/rcheck.yml`. Commit: `ci(r): R CMD check + opt-in
   integration suite`.
10. **r-universe submission**: register the repo with r-universe; verify
    the build succeeds; document in README. (No code commit; external
    action.)

## Verification

End-to-end checks before declaring v0.4.0 of `rTorchGenomics` ready:

**Build:**
- [ ] `R CMD check --as-cran rTorchGenomics_0.4.0.tar.gz` passes with
      0 errors, 0 warnings, ≤1 note (Python SystemRequirements is OK).
- [ ] `pkgdown::build_site()` produces a complete docs site.
- [ ] r-universe build green on the linked repo.

**Install UX:**
- [ ] In a clean Linux container with no Python: `install.packages` +
      `library` + `tg_install()` completes without manual intervention.
- [ ] `tg_install()` is idempotent — re-running doesn't break the venv.

**API surface:**
- [ ] All 43 R wrappers exist; `?tg_lmm_scan` etc. produce help pages.
- [ ] Each of the 12 hand-crafted wrappers has an `@examples` block
      that runs cleanly (`R CMD check --run-donttest`).

**Bridge correctness:**
- [ ] Mock-bridge unit tests: 100% pass.
- [ ] Integration test: `tg_validate()` + `tg_lmm_scan()` on the maize
      tiny PLINK fixture produces a `ScanRun` matching the Python facade's
      output byte-for-byte (modulo float precision).

**Result classes:**
- [ ] All 12 + 1 generic S4 classes round-trip through `to_dict` →
      reticulate → S4 → `as.list()` → JSON; semantic content preserved.

**Plots:**
- [ ] `tg_manhattan(r)` and `tg_qq(r)` return valid `gg` objects;
      manual eyeball comparison vs Python's matplotlib output for sanity.

**Versioning:**
- [ ] `rTorchGenomics::tg_version()` returns "0.4.0"; matches
      `tg_install()`'s installed `torchgenomics==0.4.0`.

**No push without approval**: all rTorchGenomics commits stay on the
`feat/torchgenomics-v0.4.0` branch (or a sibling `feat/rtorchgenomics`)
locally. Pushes to origin / pulsesmartlab / r-universe are
user-gated per the project's push-rate-conserving rule.
