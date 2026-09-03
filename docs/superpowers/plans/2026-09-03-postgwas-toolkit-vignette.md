# Post-GWAS Toolkit Vignette (R + Python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a comprehensive, runnable "Post-GWAS Analysis Toolkit" tutorial to BOTH packages — a Python tutorial + runnable example, and an R vignette — showcasing the full post-GWAS facade (clump, finemap, hess, ldsc/ldsc_rg, power, winners_curse, meta, mr_mega, coloc, mr, smr, gene_set_enrichment, pgs_fit/pgs_score; annotate_hits reference-only), with every example executed and verified.

**Architecture:** A verified runnable Python harness already exists at `examples/python/post_gwas_toolkit.py` (built + run to exit 0, covering 19 executed call-sites + 1 reference-only). It is the SOURCE OF TRUTH for every call signature and fixture. The Python tutorial embeds its verified snippets; the R vignette mirrors each section with the `tg_*` equivalents and R-built fixtures, and is knit-verified.

**Tech Stack:** Python (torchgenomics.api), R (rTorchGenomics S4 + reticulate), R Markdown vignette.

## Global Constraints

- **Every executed example must actually run** (no aspirational code). Python verified via `PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 python …`; R vignette verified by knitting via `PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 Rscript -e 'rmarkdown::render(...)'`.
- **Shipped files use CLEAN imports** — no `sys.path` hacks in `examples/python/post_gwas_toolkit.py` (strip the scout's env guard; it was only needed when running from `examples/python/` with a stale editable install). Verify via `PYTHONPATH` / `python -m` instead.
- **Additive only** — new files: `examples/python/post_gwas_toolkit.py` (finalize), `docs/tutorials/post-gwas-toolkit.md`, `rTorchGenomics/vignettes/post-gwas-toolkit.Rmd`, plus `_pkgdown.yml` article entry. Do NOT edit existing vignettes, `torchgenomics/` source, `rTorchGenomics/R/` source, or tests.
- **`annotate_hits` is network-required** → reference-only (shown, not executed) in all three deliverables, clearly labeled.
- **`ldsc`/`ldsc_rg` in Python are reachable only via `torchgenomics.api`** (not the top-level `torchgenomics` package); the example imports `torchgenomics.api as tg_api` for those two. In R they ARE first-class `tg_ldsc`/`tg_ldsc_rg`.
- **R vignette must degrade gracefully** where the Python backend is unavailable (CRAN) — guard with a backend-availability check so chunks skip rather than error; but HERE it runs live.
- **Environment note (this machine):** the pip editable `torchgenomics` install points at a different worktree; always prepend this worktree via `PYTHONPATH=$(pwd)` (or run `python -m` from repo root) when verifying, so imports resolve to THIS checkout (which has `finemap`/`smr`/etc.).

## File Structure

- `examples/python/post_gwas_toolkit.py` — the runnable tour (already drafted + verified; finalize: strip path hack, keep clean).
- `docs/tutorials/post-gwas-toolkit.md` — narrative Python tutorial with verified snippets + representative output.
- `rTorchGenomics/vignettes/post-gwas-toolkit.Rmd` — R vignette mirroring each section with `tg_*` calls + R fixtures.
- `rTorchGenomics/_pkgdown.yml` — add the vignette under `articles`.

---

## Task 1: Finalize the runnable Python example

**Files:**
- Modify: `examples/python/post_gwas_toolkit.py` (strip the `sys.path`/`_REPO_ROOT` guard block; keep clean `import torchgenomics as tg` + `import torchgenomics.api as tg_api`)

**Interfaces:**
- Produces: the canonical verified example; every later task lifts snippets/signatures from it.

- [ ] **Step 1: Read the current example** `examples/python/post_gwas_toolkit.py` (it is complete and verified). Confirm it runs: `PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 python examples/python/post_gwas_toolkit.py` → expect exit 0 and the final "EXECUTED (19): …  REFERENCE-ONLY (1): …" banner.

- [ ] **Step 2: Remove the machine-specific path guard.** Delete lines that manipulate `sys.path`/`_REPO_ROOT` (the block between the module docstring and the `import numpy` line that inserts the repo root). Keep the `from __future__ import annotations` and the standard imports. Update the module docstring's "Run::" line to:
  ```
  Run from the repository root::

      PYTHONPATH=. TORCHGENOMICS_DISABLE_NATIVE=1 python examples/python/post_gwas_toolkit.py

  (or ``python -m examples.python.post_gwas_toolkit``). With ``torchgenomics``
  properly pip-installed, plain ``python examples/python/post_gwas_toolkit.py`` works too.
  ```

- [ ] **Step 3: Re-verify after the edit** — `PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 python examples/python/post_gwas_toolkit.py` → MUST exit 0 with the same 19-executed banner. If it fails, STOP (do not re-add hacks unless the failure is genuinely unrelated).

- [ ] **Step 4: Capture the full stdout to a scratch file** for Task 2 to quote representative output: `PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 python examples/python/post_gwas_toolkit.py > .superpowers/sdd/vig-python-output.txt 2>&1` (scratch, gitignored).

- [ ] **Step 5: Commit**

```bash
git add examples/python/post_gwas_toolkit.py
git commit -m "docs(examples): runnable post-GWAS toolkit tour (clean imports, 19 executed)"
```

---

## Task 2: Python tutorial `docs/tutorials/post-gwas-toolkit.md`

**Files:**
- Create: `docs/tutorials/post-gwas-toolkit.md`
- Reference: `examples/python/post_gwas_toolkit.py` (verified snippets) + `.superpowers/sdd/vig-python-output.txt` (real output)

**Interfaces:**
- Consumes: the verified example (Task 1). Every code block MUST be copied verbatim (or faithfully trimmed) from the example so it stays runnable.

- [ ] **Step 1: Write the tutorial.** Structure: a short intro ("You've run a GWAS and have summary statistics; here's the post-GWAS toolkit — one facade across Python/CLI/R"), then one section per facade area, each with: (a) 1-2 sentences of what it does + when to use it; (b) a fenced ```python block lifted from the example (the fixture setup can be shown once up front or referenced); (c) a short "Expected output" excerpt quoted from `vig-python-output.txt` (trim to a few representative lines). Sections, in the example's order: Setup · clump · finemap · hess (h2 + rg) · ldsc/ldsc_rg (note the `torchgenomics.api` import + the toy-data SE caveat) · power · winners_curse (3 methods) · meta · mr_mega · coloc (pairwise + hyprcoloc) · mr · smr · gene_set_enrichment · pgs_fit/pgs_score · annotate_hits (reference-only, network). End with a one-line pointer to the CLI (`torchgenomics <subcommand> --help`) and R (`vignette("post-gwas-toolkit", package = "rTorchGenomics")`).
  - State clearly at top: run the full script with `python examples/python/post_gwas_toolkit.py`.
  - Do NOT invent numbers — every "Expected output" line comes from the captured run.
  - Keep the ldsc caveat verbatim in spirit (toy data → single jackknife block → NaN SE; real runs use thousands of SNPs).

- [ ] **Step 2: Link it from the docs index if one exists.** Check `docs/` for an index/TOC (e.g. `docs/README.md`, `mkdocs.yml`, `docs/tutorials/` listing). If a tutorials index exists, add a one-line link to `post-gwas-toolkit.md`. If none exists, skip (don't invent one).

- [ ] **Step 3: Verify snippet fidelity.** For each ```python block, confirm the code appears (verbatim or as a faithful subset) in `examples/python/post_gwas_toolkit.py`. A quick programmatic check: extract the function-call lines from the md and grep them in the example. Any block that isn't backed by the verified example is a failure — fix it to match.

- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/post-gwas-toolkit.md
git commit -m "docs: post-GWAS toolkit Python tutorial (verified snippets)"
```

---

## Task 3: R vignette `rTorchGenomics/vignettes/post-gwas-toolkit.Rmd`

**Files:**
- Create: `rTorchGenomics/vignettes/post-gwas-toolkit.Rmd`
- Reference: `examples/python/post_gwas_toolkit.py` (the fixture shapes + call semantics to mirror); `rTorchGenomics/vignettes/reticulate-bridge.Rmd` (the backend-availability guard pattern); `rTorchGenomics/R/api.R` (the exact `tg_*` signatures).

**IMPORTANT tooling note:** subagent Write/Edit have been unreliable in this harness — author the `.Rmd` via Bash heredoc and VERIFY with `cat`/`grep`. Knit failures are the real gate.

**Interfaces:**
- Consumes: the `tg_*` R wrappers (tg_clump, tg_finemap, tg_hess, tg_ldsc, tg_ldsc_rg, tg_power, tg_winners_curse, tg_meta, tg_mr_mega, tg_coloc, tg_mr, tg_smr, tg_gene_set_enrichment, tg_pgs_fit, tg_pgs_score, tg_annotate_hits). Confirm each name + its args in `rTorchGenomics/R/api.R` before use.

- [ ] **Step 1: Write the vignette header + setup guard.** YAML front-matter with `title`, `output: rmarkdown::html_vignette`, and the required `vignette:` block (`%\VignetteIndexEntry{Post-GWAS Analysis Toolkit}`, `%\VignetteEngine{knitr::rmarkdown}`, `%\VignetteEncoding{UTF-8}`). A first `setup` chunk that: loads `library(rTorchGenomics)`; checks the Python backend is available AND has the new functions (e.g. `have_backend <- requireNamespace("reticulate", quietly=TRUE) && reticulate::py_module_available("torchgenomics")`); sets `knitr::opts_chunk$set(eval = have_backend, ...)` so chunks skip gracefully on CRAN / no-Python. Mirror the guard style already used in `reticulate-bridge.Rmd`.

- [ ] **Step 2: Write a fixtures chunk (R).** Build the same tiny fixtures the Python example uses, in R (`data.frame` + `write.table(sep="\t")` to `tempfile()`s), mirroring the shapes in `examples/python/post_gwas_toolkit.py`'s `build_*` functions: the clump/ldsc GWAS + genotype CSV; the finemap uppercase sumstats + a `.pt` LD reference (build it by calling the Python `save_ld_reference` through reticulate, since it's a Python artifact — `reticulate::import("torchgenomics.postgwas._ld_ref_loader")`); the hess GWAS + `.npy` LD + regions; the coloc/meta sumstats a/b/c; the mr exposure/outcome; the mr_mega populations; the smr gwas/eqtl/gene-map; the enrichment gwas/genes/sets; the pgs sumstats + LD ref + target genotype. Keep every fixture tiny. (For artifacts that are Python-format `.pt`/`.npy`, generating them via reticulate or `np`/`torch` through reticulate is acceptable and keeps them valid.)

- [ ] **Step 3: Write one section per facade area** mirroring the Python example, each an ```{r} chunk calling the `tg_*` wrapper and printing the result object (its `show` method) + the key tibble (e.g. `r@results`, `r@credible_sets`, `r@clumps`). Same order/sections as the Python tutorial. For `ldsc`/`ldsc_rg` use `tg_ldsc`/`tg_ldsc_rg` (first-class in R) — note the same toy-data SE caveat. For `annotate_hits`, show the `tg_annotate_hits(...)` call in a chunk marked `eval=FALSE` with a note (network-required).

- [ ] **Step 4: KNIT-VERIFY (the gate).** Run:
  ```bash
  PYTHONPATH=$(pwd) TORCHGENOMICS_DISABLE_NATIVE=1 Rscript -e 'setwd("rTorchGenomics"); rmarkdown::render("vignettes/post-gwas-toolkit.Rmd", output_dir=tempdir(), quiet=TRUE); cat("RENDER OK\n")'
  ```
  Expect `RENDER OK` and NO chunk errors. If a chunk errors, fix the fixture/call to satisfy the real `tg_*` contract (read `R/api.R`); do NOT set `eval=FALSE` to hide a real failure (only `annotate_hits` is legitimately non-executed). If the backend guard evaluates FALSE here (so chunks skip), that means reticulate isn't importing this worktree — fix the invocation (PYTHONPATH) until chunks actually execute, and confirm by grepping the rendered HTML for real result output (e.g. a credible-set table), not skipped placeholders.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/vignettes/post-gwas-toolkit.Rmd
git commit -m "docs(r): post-GWAS toolkit vignette (knit-verified, all tg_* sections)"
```

---

## Task 4: Vignette index wiring + build check

**Files:**
- Modify: `rTorchGenomics/_pkgdown.yml` (add the vignette under `articles`)

**Interfaces:**
- Consumes: the vignette from Task 3.

- [ ] **Step 1: Add the vignette to `_pkgdown.yml`.** Read the current `articles:` section; add an entry for `post-gwas-toolkit` (matching the existing article entry style — the `.Rmd` basename). If there is no `articles:` section, add one following pkgdown conventions (contents: the vignette basenames including the existing ones). Do not remove existing entries.

- [ ] **Step 2: Confirm R CMD build discovers the vignette.** Run:
  ```bash
  Rscript -e 'setwd("rTorchGenomics"); tools::checkVignettes(dir=".", tangle=FALSE, weave=FALSE); cat(basename(tools::pkgVignettes(dir=".")$docs), sep="\n")'
  ```
  (or `Rscript -e 'setwd("rTorchGenomics"); print(tools::pkgVignettes(dir="."))'`) — confirm `post-gwas-toolkit.Rmd` is listed among the package vignettes. Also confirm `devtools::build_vignettes()` is NOT required to pass here (heavy); the knit-verify in Task 3 is the real proof it renders.

- [ ] **Step 3: Confirm no regression to the existing vignettes/reference** — `Rscript -e 'setwd("rTorchGenomics"); pkgdown::check_pkgdown()'` if available (validates `_pkgdown.yml` topics), else `Rscript -e 'yaml::read_yaml("rTorchGenomics/_pkgdown.yml")'` to confirm the YAML still parses. Report the outcome.

- [ ] **Step 4: Commit**

```bash
git add rTorchGenomics/_pkgdown.yml
git commit -m "docs(r): index post-GWAS toolkit vignette in _pkgdown articles"
```

---

## Self-Review notes (addressed)

- Coverage: the vignette/tutorial cover every facade function the packages expose (19 executed + annotate_hits reference-only), mirroring the verified example 1:1.
- Runnable/verified: Task 1 (Python exit 0), Task 3 (R knit RENDER OK with real chunk output) are the hard gates; no example is aspirational.
- Clean shipped files: Task 1 strips the machine-specific path hack; verification uses `PYTHONPATH`.
- Additive: only new files + one `_pkgdown.yml` articles entry; existing vignettes and package source untouched.
- Known caveats surfaced honestly: ldsc/ldsc_rg toy-data SE; annotate_hits network-only; ldsc Python-only via `torchgenomics.api`.
