# TorchGenomics Roadmap

This page tracks improvements that are **intentionally deferred** out of the
active execution stream, plus the set of candidate feature phases on the
horizon. Contributors are welcome to pick any item up — each entry has
enough context for a future author to scope the work without re-deriving
the motivation.

Last refreshed: **2026-04-21**, coming out of the v0.1.1 audit that
produced Bundles A (correctness) / B (adoption) / C (platform) /
D (coverage). See `CHANGELOG.md` for what shipped; this document is for
what did not.

---

## Deferred internal improvements

These were surfaced in the v0.1.1 audit but are not on the immediate
critical path. None of them change user-visible behaviour.

### 1. CLI decomposition

`torchgenomics/cli.py` is a single ~4700-line module with 40 subcommands inline.
Every new subcommand grows the file; every refactor requires reading the
full surface. The suggested split:

```
torchgenomics/cli/
├── __init__.py          # thin dispatcher; keeps `torchgenomics.cli:main` stable
├── scans.py             # glm/lmm/mvlmm/mklmm/gxe/bayes/farmcpu/blink/poly
├── pgs.py               # pgs-fit, pgs-score
├── postgwas.py          # ldsc, meta, clump, coloc, mr, enrichment, twas
├── multiomics.py        # mediate, mediate-scan
├── annotate.py          # annotate
├── ld.py                # ld-blocks
├── qc.py                # validate, convert, impute
└── common.py            # argument parsers, I/O helpers, shared decorators
```

**Target**: each file ≤ 800 lines. The public entry point stays at
`torchgenomics.cli:main` (re-exported from `__init__.py`), so neither the
`console_scripts` entry in `pyproject.toml` nor any user script changes.
No behaviour change.

**Effort estimate**: 1–2 days. The work is mechanical (move, update
imports, re-run `pytest tests/test_cli_*.py`) but needs care to preserve
the argparse grouping so `torchgenomics --help` output is unchanged.

### 2. `csrc/common/` consolidation

The 24 pybind11 extensions under `csrc/` each re-declare their own
`py::array` → `torch::Tensor` buffer-descriptor conversion helpers and
re-import OpenMP pragma wrappers. Suggested: extract these into
`csrc/common/pybind_utils.hpp` with:

- `tensor_from_numpy(py::array, ScalarType)` — buffer-protocol check +
  `torch::from_blob` with ownership management.
- `OMP_PARALLEL_FOR(n)` macro that collapses to a no-op when
  `TORCHGENOMICS_DISABLE_OPENMP` is set at build time.
- Shared `check_shape_2d` / `check_contiguous` assertions.

**Target**: each of the 24 `.cpp` files loses 15–40 lines of boilerplate
and the common header owns the invariants. No behaviour change; existing
benchmarks and tests must stay green to within numerical tolerance.

**Effort estimate**: 2–3 days, most of it in verifying the `-fopenmp`
toolchain matrix (Linux/macOS clang, Linux gcc, Windows MSVC) produces
bit-identical outputs before/after.

### 3. Visual-regression harness for `torchgenomics.viz`

`tests/test_viz_*.py` exercise the Manhattan / Miami / QQ / Haploview
entry points as **headless smoke tests** — they assert the call doesn't
raise and the returned `matplotlib.Figure` has the expected number of
axes. A silent regression in axis label placement, tick formatting, or
colorbar range would pass.

Suggested fix: add `pytest-mpl` (`pip install pytest-mpl`) and pin a
small catalogue of reference PNGs under `tests/baseline_images/` for the
canonical plots (one per public `plot_*` function). The comparator uses a
perceptual-difference threshold, so antialiasing noise doesn't flake.

**Target**: 8–10 canonical plots covered; `pytest --mpl` added as a
nightly CI stage (not every PR — image diffs can be slow).

**Effort estimate**: 1–2 days, most of which is generating and curating
the baseline images and tuning the tolerance.

---

## Platform / release hardening

### 4. Documentation deployment

MkDocs site (Bundle B.1) builds on every push; a GitHub Pages deploy
workflow is wired under `.github/workflows/docs.yml`. The deferred piece:
**a Read the Docs mirror** so readers on ReadTheDocs get the same content
with RtD's built-in versioning. Add `.readthedocs.yaml` when the canonical
domain story stabilises — we currently point at GitHub Pages.

### 5. CUDA CI runtime

`.github/workflows/gpu.yml` (Bundle C.3) runs on scheduled + manual
dispatch only to control cost. Once GitHub-hosted GPU runners are broadly
available and cheap (or a self-hosted runner is provisioned), drop the
schedule gate and run the GPU job on every PR. Until then, maintainers
should run the matrix manually before tagging a release.

---

## Candidate feature phases (Phase 50+)

Ordered by how much the recent audit and user conversations have pointed
at them, not by scheduled priority. The user has **not yet committed**
to any of these — each needs a brainstorming pass before execution.

- **Phase 50 — GWAS-by-imputation of untyped variants.** Lift the scan
  from a genotyped panel to the imputed-dosage panel with proper
  treatment of imputation uncertainty (ties into the existing
  `GULM` / `GU-LMM` work). Delivered sensibly, this would let users run
  `torchgenomics lmm-scan --genotype raw.bed --impute-ref 1000G.vcf.gz` in
  one step.
- **Phase 51 — Sex-chromosome GWAS.** The current LMM scanners assume
  autosomal variants. X-chromosome support requires ploidy-aware dosage
  handling (XY vs. XX) and separate null fits by inferred sex; Y and
  mitochondrial would be follow-ons.
- **Phase 52 — Admixture-aware LMM.** Add local-ancestry covariates to
  the null model so association tests don't confound ancestry with
  association. References: Tractor (2021), GAUDI.
- **Phase 53 — Rare-variant burden + dispersion tests beyond SKAT.** The
  `SetBasedScanner` already ships SKAT / SKAT-O / burden; the next tier
  is ACAT-O, STAAR, and regenie-style step-2 rare-variant scores.
- **Phase 54 — Full R wrapper release.** Memory item
  `project_rtorchgenomics_windows_install.md` catalogues the Windows install
  pitfalls. The R wrapper is installable but not on CRAN; a CRAN
  submission requires tests that don't touch the internet and a
  vignette that walks through a published dataset.

Ongoing maintenance items (not phases, but worth listing so they don't
get forgotten):

- Keep `bench/native_speedups.md` refreshed whenever a `csrc/` kernel
  changes — the numbers are cited in the README and the paper draft.
- Refresh the golden-data fixtures (`tests/golden/`) whenever the
  upstream reference tools (GEMMA 0.98.x, GAPIT 3.x, GWASpoly) cut a new
  release and our tolerances need re-setting.

---

## How to claim an item

1. Open a tracking issue on GitHub titled `[Roadmap] <item>`.
2. Post a one-paragraph design sketch in the issue and tag the
   maintainer for a thumbs-up before starting a PR.
3. For any item touching `csrc/`, run the native benchmark suite on
   your branch and paste the before/after numbers in the PR description.
