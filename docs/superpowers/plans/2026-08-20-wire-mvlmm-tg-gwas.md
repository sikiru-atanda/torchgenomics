# Wire mvlmm (multi-trait) through tg.gwas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tg.gwas(phenotype_df, genotype, models="mvlmm", traits=["Y1","Y2"])` runs the multi-trait `MultiTraitLMM` scan through the friendly dispatch — the last unwired registry model. After this, every registry model is wired.

**Architecture:** Extend `load_inputs`/`GwasInputs` with an optional multi-trait mode (`traits=` → a `phenotypes` DataFrame), materialize a multi-column phenotype TSV, and wire `mvlmm` through the existing low-level CLI-runner route (it writes `.assoc.tsv` with `P_JOINT`, so the default read-back is reused). No new statistics.

**Tech Stack:** Python 3.12, pandas; `torchgenomics.api._inputs`/`._dispatch`/`.gwas`/`cli.py`; R reticulate bridge.

## Global Constraints

- **No new statistics** — orchestrate the existing `_cmd_mvlmm_scan`. No change to model math or `lmm_scan`/`glm_scan`.
- **Backward compatible / additive** — two new `GwasInputs` fields (`phenotypes`, `trait_names`, default `None`); new optional `load_inputs(traits=)`/`gwas(traits=)`/CLI `--traits`/R `tg_gwas(traits=)`; one new `_LOWLEVEL_CLI` entry; one argv builder; one multi-trait TSV writer. The single-trait path is byte-for-byte unchanged (all new behavior gated on `traits`/`phenotypes` being non-None).
- **Reviewer-grade docstrings**; **friendly errors** (mvlmm without ≥2 traits; missing trait columns; bare-Series phenotype with traits).
- `mvlmm` uses `MultiTraitLMM`, `trait_type="continuous"`; default `--ploidy 2`, overridable via `model_options={"mvlmm":{"ploidy":k}}`.
- `_auto_model` unchanged (mvlmm explicit-only). Verify commands under `TORCHGENOMICS_DISABLE_NATIVE=1`.

## File Structure

- Modify: `torchgenomics/api/_inputs.py` — `traits=` on `load_inputs`; `_resolve_phenotypes`; multi-trait alignment; `GwasInputs.phenotypes`/`.trait_names`.
- Modify: `torchgenomics/api/_dispatch.py` — `_mvlmm_extra_argv`; route in `_lowlevel_extra_argv`; `_write_multitrait_phenotype_tsv` + `_materialize_inputs` branch; `_LOWLEVEL_CLI["mvlmm"]`.
- Modify: `torchgenomics/api/gwas.py` — `gwas(traits=None)` → `load_inputs(traits=)`.
- Modify: `torchgenomics/cli.py` — `--traits` on the `gwas` subparser; `_cmd_gwas` forwards it.
- Modify: `CLAUDE.md` — all registry models wired.
- Modify: `rTorchGenomics/R/gwas.R` (+ `man/tg_gwas.Rd`) — `tg_gwas(traits=NULL)`.
- Test: `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`, `rTorchGenomics/tests/testthat/test-gwas.R`.

**Interface contract:**

```python
# _inputs.py
# GwasInputs gains:  phenotypes: pd.DataFrame | None = None ; trait_names: list[str] | None = None
def load_inputs(phenotype, genotype, covariates=None, trait=None, trait_type=None,
                traits: list[str] | None = None) -> GwasInputs: ...
# _dispatch.py
def _mvlmm_extra_argv(inputs, opts, user_opts: dict) -> list[str]: ...
def _write_multitrait_phenotype_tsv(inputs, path) -> None: ...
# gwas.py
def gwas(..., traits: list[str] | None = None) -> "GwasResult | GwasComparison": ...
```

---

### Task 1: Multi-trait `load_inputs` / `GwasInputs`

**Files:** Modify `torchgenomics/api/_inputs.py`; Test `tests/test_api_gwas.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def _multitrait_fixture(n=140, m=160, seed=3):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    base = rng.normal(size=n)
    pheno = pd.DataFrame(
        {"Y1": base + rng.normal(scale=0.5, size=n),
         "Y2": base + rng.normal(scale=0.5, size=n),
         "Y3": rng.normal(size=n)},
        index=ids,
    )
    return pheno, G

def test_load_inputs_multitrait():
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    assert inp.trait_names == ["Y1", "Y2"]
    assert inp.phenotypes is not None and inp.phenotypes.shape[1] == 2
    assert list(inp.phenotypes.columns) == ["Y1", "Y2"]
    assert inp.phenotype.name == "Y1"          # first trait as the Series
    assert inp.n_samples == inp.phenotypes.shape[0]
    assert inp.trait_type == "continuous"

def test_load_inputs_multitrait_missing_column_errors():
    from torchgenomics.api._inputs import load_inputs
    import pytest
    pheno, G = _multitrait_fixture()
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "NOPE"])
    assert "NOPE" in str(e.value)

def test_load_inputs_multitrait_bare_series_errors():
    from torchgenomics.api._inputs import load_inputs
    import pandas as pd, numpy as np, pytest
    ids = [f"s{i}" for i in range(20)]
    y = pd.Series(np.arange(20.0), index=ids, name="Y1")
    G = pd.DataFrame(np.zeros((20, 10)), index=ids)
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=y, genotype=G, traits=["Y1", "Y2"])
    assert "multi-trait" in str(e.value).lower() or "dataframe" in str(e.value).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "multitrait" -v`
Expected: FAIL (`load_inputs` has no `traits`; `GwasInputs` has no `phenotypes`).

- [ ] **Step 3: Write minimal implementation**

1. Add fields to the `GwasInputs` dataclass (after `trait_name`), with docstring entries:

```python
    phenotypes: "pd.DataFrame | None" = None
    trait_names: "list[str] | None" = None
```

2. Add `_resolve_phenotypes` (near `_resolve_phenotype`):

```python
def _resolve_phenotypes(phenotype: Any, traits: list[str]) -> pd.DataFrame:
    """Resolve a multi-trait phenotype to a DataFrame of the named columns.

    ``phenotype`` must be a :class:`pandas.DataFrame` (indexed by sample id) or
    a path to a table containing every column in ``traits``. A bare Series (a
    single trait) is rejected with a friendly error. Columns are returned in
    the order given in ``traits`` and coerced to float.
    """
    if isinstance(phenotype, pd.Series):
        raise ValueError(
            "multi-trait mode (traits=[...]) needs a phenotype DataFrame or a "
            "path/file with the named columns, not a single Series. Provide a "
            f"DataFrame with columns {traits}."
        )
    if isinstance(phenotype, (str, Path)):
        df = _load_tabular(phenotype)  # existing helper used by _resolve_phenotype
        df = _index_by_sample_id(df)   # reuse whatever _resolve_phenotype uses to set the id index
    elif isinstance(phenotype, pd.DataFrame):
        df = phenotype.copy()
    else:
        raise ValueError(
            f"Unsupported phenotype type for multi-trait mode: {type(phenotype).__name__}. "
            "Pass a DataFrame indexed by sample id or a path to a phenotype table."
        )
    missing = [t for t in traits if t not in df.columns]
    if missing:
        raise ValueError(
            f"traits {missing} not found in the phenotype columns "
            f"({list(df.columns)[:10]}...). Check the column names."
        )
    out = df[list(traits)].apply(pd.to_numeric, errors="coerce")
    return out
```

Note: reuse the SAME tabular-load + id-index helpers `_resolve_phenotype` already uses (read that function and call the same internals; if it inlines them, factor a tiny shared helper rather than duplicating). Do not duplicate parsing logic.

3. Add a multi-trait alignment. Generalize `_align_by_ids` to accept a DataFrame, or add `_align_by_ids_multi(pheno_df, covariates, reader)` mirroring `_align_by_ids` (intersect string ids, sort lexicographically, `.loc[shared]` on the DataFrame, reindex covariates + reader, friendly error on empty/tiny intersection). Return `(aligned_df, aligned_covariates, aligned_reader)`.

4. In `load_inputs`, add `traits: list[str] | None = None` and branch:

```python
    if traits is not None:
        if len(traits) < 1:
            raise ValueError("traits=[] is empty; pass the trait column names.")
        pheno_df = _resolve_phenotypes(phenotype, traits)
        covar_df = _resolve_covariates(covariates, pheno_df.index) if covariates is not None else None
        # in-memory genotype -> ArrayReader keyed on the multi-trait index; else align by ids
        if _is_inmemory_genotype(genotype):   # same predicate load_inputs already uses
            reader = _make_array_reader(genotype, pheno_df.index)
            aligned_df, aligned_cov, aligned_reader = pheno_df, covar_df, reader
        else:
            reader = genotype
            aligned_df, aligned_cov, aligned_reader = _align_by_ids_multi(pheno_df, covar_df, reader)
        first = aligned_df.iloc[:, 0]
        return GwasInputs(
            genotype_path_or_reader=aligned_reader,
            phenotype=first,                      # first trait as the Series
            covariates=aligned_cov,
            trait_type="continuous",
            n_samples=len(aligned_df),
            trait_name=str(aligned_df.columns[0]),
            phenotypes=aligned_df,
            trait_names=list(aligned_df.columns),
        )
    # ...existing single-trait path unchanged...
```

Match the exact predicate/branch the current single-trait `load_inputs` uses to distinguish in-memory vs path genotype (read the function; reuse its structure). Keep the single-trait path untouched.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "multitrait" -v`
Expected: PASS. Then `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py tests/test_api_inputs.py -q` → no regressions (single-trait path unchanged).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_inputs.py tests/test_api_gwas.py
git commit -m "friendly-api: multi-trait load_inputs (traits= -> phenotypes DataFrame + trait_names)"
```

---

### Task 2: Wire `mvlmm` dispatch + multi-trait materialization

**Files:** Modify `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** Consumes Task 1's `GwasInputs.phenotypes`/`.trait_names`. Produces `_mvlmm_extra_argv`, `_write_multitrait_phenotype_tsv`, `_LOWLEVEL_CLI["mvlmm"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_mvlmm_extra_argv_traits_and_ploidy():
    from torchgenomics.api._dispatch import _mvlmm_extra_argv, RunOptions
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    argv = _mvlmm_extra_argv(inp, RunOptions(), {})
    assert "--traits" in argv and "Y1,Y2" in argv
    assert "--ploidy" in argv and argv[argv.index("--ploidy") + 1] == "2"
    # ploidy override via model_options, not duplicated
    argv4 = _mvlmm_extra_argv(inp, RunOptions(), {"ploidy": 4})
    assert argv4[argv4.index("--ploidy") + 1] == "4" and argv4.count("--ploidy") == 1

def test_mvlmm_requires_two_traits_friendly_error():
    from torchgenomics.api._dispatch import _mvlmm_extra_argv, RunOptions
    from torchgenomics.api._inputs import load_inputs
    import pytest
    pheno, G = _multitrait_fixture()
    inp1 = load_inputs(phenotype=pheno, genotype=G, traits=["Y1"])   # only 1 trait
    with pytest.raises(ValueError) as e:
        _mvlmm_extra_argv(inp1, RunOptions(), {})
    assert "mvlmm" in str(e.value).lower() and "trait" in str(e.value).lower()

def test_write_multitrait_phenotype_tsv(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _write_multitrait_phenotype_tsv
    from torchgenomics.api._inputs import load_inputs
    pheno, G = _multitrait_fixture()
    inp = load_inputs(phenotype=pheno, genotype=G, traits=["Y1", "Y2"])
    p = tmp_path / "ph.tsv"
    _write_multitrait_phenotype_tsv(inp, p)
    back = pd.read_csv(p, sep="\t")
    assert "sample_id" in back.columns and "Y1" in back.columns and "Y2" in back.columns
    assert len(back) == inp.n_samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "mvlmm_extra or multitrait_phenotype_tsv" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def _mvlmm_extra_argv(inputs: GwasInputs, opts: RunOptions, user_opts: dict) -> list[str]:
    """Build mvlmm CLI flags: the required ``--traits`` (>=2) and ``--ploidy``.

    Requires a multi-trait ``inputs`` (``trait_names`` with >=2 entries) — else a
    friendly ``ValueError``. Ploidy defaults to 2 and is overridable via
    ``model_options={"mvlmm": {"ploidy": k}}``.
    """
    names = inputs.trait_names or []
    if len(names) < 2:
        raise ValueError(
            "Model 'mvlmm' needs >=2 traits. Pass traits=['Y1','Y2'] naming columns "
            "of a multi-column phenotype (DataFrame or file)."
        )
    opts_map = dict(user_opts)
    ploidy = opts_map.pop("ploidy", 2)
    return (["--traits", ",".join(names), "--ploidy", str(ploidy)]
            + _model_options_to_argv(opts_map))


def _write_multitrait_phenotype_tsv(inputs: GwasInputs, path) -> None:
    """Write ``sample_id`` + each trait column as a TSV the mvlmm runner reads."""
    df = inputs.phenotypes.copy()
    df.insert(0, "sample_id", [str(s) for s in df.index])
    df.to_csv(path, sep="\t", index=False)
```

Route in `_lowlevel_extra_argv`:

```python
    if alias == "mvlmm":
        return _mvlmm_extra_argv(inputs, opts, user_opts)
```

In `_materialize_inputs` (and/or `_run_lowlevel_cli`'s phenotype write), choose the multi-trait writer when multi-trait:

```python
    if inputs.phenotypes is not None:
        _write_multitrait_phenotype_tsv(inputs, pheno_path)
    else:
        _write_phenotype_tsv(inputs, pheno_path)
```

Register:

```python
    "mvlmm": ("mvlmm-scan", "_cmd_mvlmm_scan", "MultiTraitLMM", None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "mvlmm_extra or multitrait_phenotype_tsv or mvlmm_requires" -v`
Expected: PASS. Then `-q` full file → no regressions.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire mvlmm dispatch (traits/ploidy argv + multi-trait pheno TSV)"
```

---

### Task 3: `gwas(traits=)` + CLI `--traits` + e2e + docs

**Files:** Modify `torchgenomics/api/gwas.py`, `torchgenomics/cli.py`, `CLAUDE.md`; Test `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_mvlmm_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    pheno, G = _multitrait_fixture()
    r = tg.gwas(pheno, G, models="mvlmm", traits=["Y1", "Y2"],
                kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "MultiTraitLMM" and r.n_variants > 0

def test_mvlmm_without_traits_friendly_error():
    import torchgenomics as tg, pytest
    pheno, G = _multitrait_fixture()
    with pytest.raises(ValueError) as e:
        tg.gwas(pheno, G, models="mvlmm", kinship="auto", pcs=0, verbose=False)
    assert "mvlmm" in str(e.value).lower() and "trait" in str(e.value).lower()
```

```python
# tests/test_cli_gwas.py (append)
def test_cli_gwas_help_has_traits():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--traits" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_mvlmm_runs_through_tg_gwas tests/test_api_gwas.py::test_mvlmm_without_traits_friendly_error tests/test_cli_gwas.py::test_cli_gwas_help_has_traits -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

1. `gwas.py` — add `traits: list[str] | None = None` (keyword-only) to `gwas(...)` with a docstring paragraph; pass `traits=traits` into the `load_inputs(...)` call inside `gwas()`.
2. `cli.py` — `_add_gwas_parser`: `p.add_argument("--traits", default=None, help="Comma-separated trait columns for --models mvlmm (multi-trait).")`. `_cmd_gwas`: `traits = args.traits.split(",") if getattr(args, "traits", None) else None`, and pass `traits=traits` to `api.gwas(...)`.
3. `CLAUDE.md` — update the wired-list sentence: **all** registry models now run through `tg.gwas` (`lmm/glm/blink/farmcpu/glmm/mklmm/gxe/set/bayes/mvlmm`); remove the "only mvlmm unwired" caveat. Add: `models="mvlmm"` needs `traits=[≥2 columns]` of a multi-column phenotype and returns a joint multi-trait result (`P_JOINT`); ploidy via `model_options={"mvlmm":{"ploidy":k}}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py tests/test_cli_gwas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/gwas.py torchgenomics/cli.py CLAUDE.md tests/test_api_gwas.py tests/test_cli_gwas.py
git commit -m "friendly-api: gwas(traits=) + CLI --traits; mvlmm wired; all registry models wired; docs"
```

---

### Task 4: R `tg_gwas(traits=)` passthrough

**Files:** Modify `rTorchGenomics/R/gwas.R`, `rTorchGenomics/man/tg_gwas.Rd`; Test `rTorchGenomics/tests/testthat/test-gwas.R`.

**IMPORTANT:** author/modify R files via Bash heredoc / `sed` / `Rscript writeLines` (subagent Write/Edit is unreliable for this repo's R). Parse-check with `Rscript -e 'invisible(parse("rTorchGenomics/R/gwas.R"))'`. Run the R suite with `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-gwas.R")'` (load_all first — the file needs the package loaded).

- [ ] **Step 1: Write the failing test** (append; use the file's `mockery::stub` idiom)

```r
test_that("tg_gwas forwards traits to the bridge", {
  captured <- NULL
  mockery::stub(tg_gwas, "bridge_call", function(fn, args = list()) {
    captured <<- args
    .fake_scan_dict()
  })
  tg_gwas("pheno.tsv", "data.bed", models = "mvlmm", traits = c("Y1", "Y2"))
  expect_equal(captured$traits, list("Y1", "Y2"))   # reticulate list; adjust to c("Y1","Y2") if the bridge keeps a vector
})
```

(If the mock captures an R character vector rather than a list, assert `expect_equal(as.character(captured$traits), c("Y1","Y2"))` — pick the form that matches how `.compact`/`bridge_call` passes it; verify by reading the existing `model_options` test's captured shape.)

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: FAIL (`unused argument (traits = ...)`).

- [ ] **Step 3: Write minimal implementation**

Edit `rTorchGenomics/R/gwas.R`: add `traits = NULL` to `tg_gwas`'s signature; add `traits = traits` inside the `.compact(list(...))` feeding `bridge_call("gwas", ...)` (a character vector is forwarded; `.compact` drops `NULL`). Add a roxygen `@param traits` line (character vector of ≥2 column names for `models="mvlmm"`; paths/columns only in R). Regenerate: `Rscript -e 'roxygen2::roxygenise()'`.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: PASS. Parse-check gwas.R.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/gwas.R rTorchGenomics/man/tg_gwas.Rd rTorchGenomics/tests/testthat/test-gwas.R
git commit -m "friendly-api: R tg_gwas traits passthrough"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (multi-trait load_inputs/GwasInputs) → Task 1. ✅
- §Component 2 (wire mvlmm dispatch + multi-trait materialization) → Task 2. ✅
- §Component 3 (gwas(traits=) + shrink NotImplementedError) → Task 3 (mvlmm now registered, so nothing falls through). ✅
- §Component 4 (CLI + R + docs) → Tasks 3 (CLI+docs), 4 (R). ✅
- §Testing → Tasks 1-4 (multi-trait load + errors; argv + materialization; e2e + no-traits error; CLI help; R). ✅

**Placeholder scan:** No TBD/TODO; code steps carry real code. The `_resolve_phenotypes`/alignment steps say "reuse the same helper `_resolve_phenotype` uses" — this is a verification step (match the existing tabular-load + id-index internals), not a placeholder; the surrounding logic is fully specified.

**Type consistency:** `GwasInputs.phenotypes`/`.trait_names` set in Task 1 and consumed by `_mvlmm_extra_argv`/`_write_multitrait_phenotype_tsv` (Task 2) and by `_materialize_inputs`'s branch (Task 2). `gwas(traits=)`→`load_inputs(traits=)` (Task 3) matches Task 1's new param. `_LOWLEVEL_CLI["mvlmm"]` is a 4-tuple (matches the shape established earlier). ✅

**Known verification points for the implementer:**
- (a) Read the current single-trait `load_inputs` to reuse its exact in-memory-vs-path genotype predicate and its tabular-load/id-index helpers — do not duplicate parsing.
- (b) `_align_by_ids_multi` must mirror `_align_by_ids`'s intersection/sort/friendly-empty-error behavior exactly, on a DataFrame instead of a Series.
- (c) mvLMM REML convergence on the tiny fixture: the fixture's two traits are correlated (shared latent) so Vg is estimable; if it doesn't converge, bump samples/variants but keep assertions meaningful (report DONE_WITH_CONCERNS rather than weaken).
- (d) Confirm `_align_samples(trait_columns=...)` in the runner reads the multi-trait TSV's `sample_id` column as ids and the named trait columns as `Y` — the materialized TSV header must match what `io.phenotype.load_phenotype` expects (sample-id column + trait columns).
