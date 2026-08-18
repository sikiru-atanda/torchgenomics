# Wire bayes + gxe + set through tg.gwas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `tg.gwas(models="gxe", env=…)`, `models="set", regions=…`, and `models="bayes"` through the friendly dispatch core, with a per-model read-back so bayes (PIP) and set (per-region) return a uniform `GwasResult`.

**Architecture:** Reuse the low-level CLI-runner route from the glmm+mklmm work. Generalize `_run_lowlevel_cli` so each `_LOWLEVEL_CLI` entry selects its read-back. gxe reuses the existing `_read_scan_output` (it writes `.assoc.tsv` with `P_JOINT`); bayes/set get dedicated read-backs for their custom output files. Add `env=`/`regions=` inputs (in-memory or path). No new statistics.

**Tech Stack:** Python 3.12, pandas, argparse; `torchgenomics.api._dispatch`/`.gwas`/`cli.py`; R reticulate bridge.

## Global Constraints

- **No new statistics** — orchestrate the existing `_cmd_gxe_scan`/`_cmd_set_scan`/`_cmd_bayes_scan`. No model-math change; no change to `lmm_scan`/`glm_scan`.
- **Backward compatible / additive** — a 4th optional element on `_LOWLEVEL_CLI` tuples (absent/`None` → default `_read_scan_output`); new optional `RunOptions.env`/`.regions`; new optional `gwas(env=, regions=)` + CLI/R params; three new `_LOWLEVEL_CLI` entries; two new read-back functions. blink/farmcpu/glmm/mklmm/glm/lmm behavior byte-for-byte unchanged.
- **Reviewer-grade docstrings** on every new/changed public symbol.
- **Friendly errors** — gxe without `env` and set without `regions` raise actionable `ValueError`s (no raw tracebacks); missing bayes/set output file → clear error.
- **`_auto_model` unchanged** — gxe/set/bayes are explicit-only, never auto-selected.
- bayes significance = **credible-set membership** (from `<prefix>_credible_sets.tsv`), not a PIP cutoff.
- `env` accepts a `pd.Series`/array (aligned to sample order) or a path; `regions` accepts a path or a `pd.DataFrame`.
- Verify commands run under `TORCHGENOMICS_DISABLE_NATIVE=1`.

## File Structure

- Modify: `torchgenomics/api/_dispatch.py` — read-back selector on `_run_lowlevel_cli` + `_LOWLEVEL_CLI` 4-tuples; `RunOptions.env`/`.regions`; `_read_bayes_output`; `_read_set_output`; `_materialize_env`/`_materialize_regions`; `_gxe_extra_argv`/`_set_extra_argv`/`_bayes_extra_argv`; three `_LOWLEVEL_CLI` entries; shrink `NotImplementedError` set.
- Modify: `torchgenomics/api/gwas.py` — `gwas(env=None, regions=None)` threaded into `RunOptions`.
- Modify: `torchgenomics/cli.py` — `--env`/`--regions` on the `gwas` subparser; `_cmd_gwas` forwards them.
- Modify: `CLAUDE.md` — wired-list (`+ gxe/set/bayes`; not-yet-wired = `mvlmm`).
- Modify: `rTorchGenomics/R/gwas.R` (+ `man/tg_gwas.Rd`) — `tg_gwas(env=NULL, regions=NULL)` passthrough.
- Test: `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`, `rTorchGenomics/tests/testthat/test-gwas.R`.

**Interface contract:**

```python
# _dispatch.py — read-backs share _read_scan_output's keyword signature exactly:
def _read_bayes_output(output_prefix, *, model_label, test, correction, trait_type,
                       n_samples, significance_threshold, top_k, runtime_s) -> GwasResult: ...
def _read_set_output(output_prefix, *, model_label, test, correction, trait_type,
                     n_samples, significance_threshold, top_k, runtime_s) -> GwasResult: ...
def _materialize_env(env, inputs, workdir) -> str: ...       # Series/array->tmp ENV tsv (sample order) | path->itself
def _materialize_regions(regions, workdir) -> str: ...       # DataFrame->tmp regions file | path->itself
# RunOptions gains:  env: Any = None ; regions: Any = None
# _LOWLEVEL_CLI values become 4-tuples: (subcommand, runner_name, model_label, readback_or_None)
# gwas.py
def gwas(..., env=None, regions=None) -> "GwasResult | GwasComparison": ...
```

---

### Task 1: Per-model read-back selection (mechanism; no new models)

**Files:** Modify `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** Produces the 4-tuple `_LOWLEVEL_CLI` shape + read-back selection in `_run_lowlevel_cli`. Consumes existing `_read_scan_output`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_lowlevel_cli_entries_are_4_tuples_with_default_readback():
    from torchgenomics.api._dispatch import _LOWLEVEL_CLI, _read_scan_output
    for alias, entry in _LOWLEVEL_CLI.items():
        assert len(entry) == 4, f"{alias} entry must be a 4-tuple (subcommand, runner, label, readback)"
        # existing models use the default read-back (None) or _read_scan_output explicitly
        assert entry[3] is None or callable(entry[3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_lowlevel_cli_entries_are_4_tuples_with_default_readback -v`
Expected: FAIL (entries are 3-tuples).

- [ ] **Step 3: Write minimal implementation**

1. Extend each `_LOWLEVEL_CLI` entry to a 4-tuple with `None` (default read-back):

```python
_LOWLEVEL_CLI = {
    "blink": ("blink-scan", "_cmd_blink_scan_single", "BLINK", None),
    "farmcpu": ("farmcpu-scan", "_cmd_farmcpu_scan_single", "FarmCPU", None),
    "glmm": ("glmm-scan", "_cmd_glmm_scan", "GLMM", None),
    "mklmm": ("mklmm-scan", "_cmd_mklmm_scan", "MultiKernelLMM", None),
}
```

2. `_run_lowlevel_cli` — accept the read-back. Change its signature to add `readback=None` (keyword), and replace the final `return _read_scan_output(...)` with:

```python
    reader = readback or _read_scan_output
    return reader(
        output_prefix,
        model_label=model_label,
        test=("score" if alias == "glmm" else getattr(args, "test", "wald")),
        correction=opts.correction,
        trait_type=inputs.trait_type,
        n_samples=inputs.n_samples,
        significance_threshold=5e-8,
        top_k=opts.top_k,
        runtime_s=runtime_s,
    )
```

3. In `run_model`, unpack the 4-tuple and pass the read-back:

```python
            subcommand, runner_name, model_label, readback = wiring
            runner = getattr(cli, runner_name)
            result = _run_lowlevel_cli(
                inputs, opts,
                subcommand=subcommand, runner=runner, model_label=model_label,
                workdir=workdir, alias=low, readback=readback,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_lowlevel_cli_entries_are_4_tuples_with_default_readback -v`
Expected: PASS.
Regression: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "glmm or mklmm or blink or farmcpu or comparison" -q` → PASS (existing models still read back via the default).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: per-model read-back selection on the low-level route"
```

---

### Task 2: `env=` input + wire `gxe`

**Files:** Modify `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** Consumes Task 1. Produces `RunOptions.env`, `_materialize_env`, `_gxe_extra_argv`, `_LOWLEVEL_CLI["gxe"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_gxe_requires_env_friendly_error():
    import torchgenomics as tg
    y, G = _quant_fixture()          # helper already in this file
    import pytest
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="gxe", kinship="auto", pcs=0, verbose=False)
    assert "gxe" in str(e.value).lower() and "env" in str(e.value).lower()

def test_gxe_runs_with_inmemory_env():
    import numpy as np, pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    ids = list(y.index)
    env = pd.Series(np.random.default_rng(7).normal(size=len(ids)), index=ids, name="ENV")
    r = tg.gwas(y, G, models="gxe", env=env, kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k gxe -v`
Expected: FAIL (gxe unwired / no `env`).

- [ ] **Step 3: Write minimal implementation**

1. Add `env` + `regions` fields to `RunOptions` (both `Any = None`, with docstring paragraphs):

```python
    env: Any = None
    regions: Any = None
```

2. Add `_materialize_env` (place near the other helpers):

```python
def _materialize_env(env: Any, inputs: GwasInputs, workdir: Path) -> str:
    """Return a path to a TSV with a single ``ENV`` column for gxe-scan.

    A ``str``/``Path`` is treated as an existing env file and returned as-is
    (the caller is responsible for its sample order, matching the CLI). A
    ``pandas.Series``/1-D array is written to a temp TSV with one ``ENV``
    column, reindexed to the aligned sample order (``inputs.phenotype.index``)
    when the Series carries a sample-id index — so the env lines up with the
    genotype/phenotype the scan will align to.
    """
    import numpy as np
    import pandas as pd

    if isinstance(env, (str, Path)):
        return str(env)
    if isinstance(env, pd.Series):
        aligned = env.reindex(inputs.phenotype.index) if env.index.equals(inputs.phenotype.index) or set(inputs.phenotype.index).issubset(set(env.index)) else env
        vals = pd.Series(np.asarray(aligned, dtype=float), name="ENV")
    else:
        arr = np.asarray(env, dtype=float).reshape(-1)
        if arr.shape[0] != inputs.n_samples:
            raise ValueError(
                f"env has {arr.shape[0]} values but there are {inputs.n_samples} "
                f"aligned samples; pass a pandas Series indexed by sample id, or an "
                f"array in sample order."
            )
        vals = pd.Series(arr, name="ENV")
    path = str(workdir / "env.tsv")
    vals.to_frame().to_csv(path, sep="\t", index=False)
    return path
```

3. Add `_gxe_extra_argv` + route it:

```python
def _gxe_extra_argv(inputs: GwasInputs, opts: RunOptions, user_opts: dict, workdir: Path) -> list[str]:
    """Build gxe CLI flags: the required ``--env`` plus any user model_options.

    Raises a friendly ``ValueError`` if no ``env`` was supplied.
    """
    if opts.env is None:
        raise ValueError(
            "Model 'gxe' needs an environment variable. Pass env=<pandas Series "
            "indexed by sample id | array in sample order | path to a TSV with an "
            "ENV column>."
        )
    env_path = _materialize_env(opts.env, inputs, workdir)
    return ["--env", env_path] + _model_options_to_argv(user_opts)
```

In `_lowlevel_extra_argv`, add the branch (note: `_lowlevel_extra_argv` needs `workdir` to materialize — thread it through; `_run_lowlevel_cli` already creates/knows `workdir` and calls `_lowlevel_extra_argv` — pass `workdir` to it):

```python
    if alias == "gxe":
        return _gxe_extra_argv(inputs, opts, user_opts, workdir)
```

Update `_lowlevel_extra_argv`'s signature to `(alias, inputs, opts, workdir)` and its one call site in `_run_lowlevel_cli` to pass `workdir`. (blink/farmcpu/glmm/mklmm branches ignore `workdir`.)

4. Register gxe (reuses default read-back — it writes `.assoc.tsv` with `P_JOINT`):

```python
    "gxe": ("gxe-scan", "_cmd_gxe_scan", "GxELMM", None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k gxe -v`
Expected: PASS. Then `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -q` → no regressions.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire gxe through tg.gwas (env= input, in-memory or path)"
```

---

### Task 3: `_read_bayes_output` + wire `bayes`

**Files:** Modify `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** Consumes Task 1. Produces `_read_bayes_output`, `_bayes_extra_argv`, `_LOWLEVEL_CLI["bayes"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_read_bayes_output_pip_sorted(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _read_bayes_output
    prefix = str(tmp_path / "run")
    pd.DataFrame({
        "SNP": ["s1", "s2", "s3"], "CHR": [1, 1, 1], "POS": [10, 20, 30],
        "A1": ["A", "A", "A"], "A2": ["B", "B", "B"], "AF": [0.3, 0.4, 0.5],
        "PIP": [0.1, 0.9, 0.4], "BETA_MEAN": [0.0, 1.0, 0.2], "BETA_SD": [0.1, 0.1, 0.1],
    }).to_csv(prefix + "_bayesian_vs.tsv", sep="\t", index=False)
    pd.DataFrame({"SNP": ["s2"], "CS": [1]}).to_csv(prefix + "_credible_sets.tsv", sep="\t", index=False)
    r = _read_bayes_output(prefix, model_label="BayesianVS", test="susie", correction="none",
                           trait_type="continuous", n_samples=100, significance_threshold=5e-8,
                           top_k=50, runtime_s=0.0)
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult)
    assert r.lambda_gc is None and r.model == "BayesianVS"
    assert list(r.top_hits["SNP"])[0] == "s2"        # highest PIP first
    assert r.n_significant == 1                        # one variant in a credible set

def test_bayes_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    r = tg.gwas(y, G, models="bayes", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "BayesianVS"
    assert r.lambda_gc is None and "PIP" in r.top_hits.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k bayes -v`
Expected: FAIL (`_read_bayes_output` missing; bayes unwired).

- [ ] **Step 3: Write minimal implementation**

```python
def _read_bayes_output(output_prefix, *, model_label, test, correction, trait_type,
                       n_samples, significance_threshold, top_k, runtime_s) -> GwasResult:
    """Read a BayesianVS fine-mapping run into a ``GwasResult``.

    Reads ``<prefix>_bayesian_vs.tsv`` (columns incl. ``PIP``), sorts by ``PIP``
    descending for ``top_hits``, and counts credible-set membership from
    ``<prefix>_credible_sets.tsv`` as ``n_significant``. There are no p-values,
    so ``lambda_gc`` is ``None`` and ``summary()`` presents fine-mapping output.
    """
    prefix = str(output_prefix)
    vs_path = Path(prefix + "_bayesian_vs.tsv")
    if not vs_path.exists():
        raise RuntimeError(f"bayes scan produced no fine-mapping table at {vs_path}")
    df = pd.read_csv(vs_path, sep="\t")
    top = df.sort_values("PIP", ascending=False).head(top_k).reset_index(drop=True)
    cs_path = Path(prefix + "_credible_sets.tsv")
    n_sig = 0
    if cs_path.exists():
        cs = pd.read_csv(cs_path, sep="\t")
        n_sig = int(len(cs))
    return GwasResult(
        runtime_s=runtime_s, output_files={"bayesian_vs": vs_path},
        model=model_label, test=test, correction=correction,
        n_variants=int(len(df)), n_significant=n_sig,
        significance_threshold=significance_threshold, lambda_gc=None,
        n_samples=n_samples, top_hits=top, trait_type=trait_type,
    )


def _bayes_extra_argv(user_opts: dict) -> list[str]:
    """Build bayes CLI flags from model_options (method/n-signals/priors); no
    required extra input (SuSiE defaults apply)."""
    return _model_options_to_argv(user_opts)
```

Route in `_lowlevel_extra_argv`:

```python
    if alias == "bayes":
        return _bayes_extra_argv(user_opts)
```

Register (with the dedicated read-back):

```python
    "bayes": ("bayes-scan", "_cmd_bayes_scan", "BayesianVS", _read_bayes_output),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k bayes -v`
Expected: PASS. Then `-q` full file → no regressions.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire bayes through tg.gwas (PIP/credible-set read-back)"
```

---

### Task 4: `regions=` input + `_read_set_output` + wire `set`

**Files:** Modify `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** Consumes Task 1. Produces `_materialize_regions`, `_read_set_output`, `_set_extra_argv`, `_LOWLEVEL_CLI["set"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_read_set_output_region_level(tmp_path):
    import pandas as pd
    from torchgenomics.api._dispatch import _read_set_output
    prefix = str(tmp_path / "run")
    pd.DataFrame({
        "REGION": ["geneA", "geneB"], "CHR": [1, 1], "START": [1, 100], "END": [50, 150],
        "N_VARIANTS": [10, 8], "P": [1e-9, 0.2],
    }).to_csv(prefix + "_set_based.tsv", sep="\t", index=False)
    r = _read_set_output(prefix, model_label="SetBasedScanner", test="skat", correction="bh",
                         trait_type="continuous", n_samples=100, significance_threshold=5e-8,
                         top_k=50, runtime_s=0.0)
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult) and r.model == "SetBasedScanner"
    assert r.n_variants == 2 and r.n_significant == 1        # geneA below threshold
    assert list(r.top_hits["REGION"])[0] == "geneA"          # smallest P first

def test_set_requires_regions_friendly_error():
    import torchgenomics as tg, pytest
    y, G = _quant_fixture()
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="set", kinship="auto", pcs=0, verbose=False)
    assert "set" in str(e.value).lower() and "region" in str(e.value).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "set_output or set_requires" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def _read_set_output(output_prefix, *, model_label, test, correction, trait_type,
                     n_samples, significance_threshold, top_k, runtime_s) -> GwasResult:
    """Read a set-based (region) scan into a ``GwasResult``.

    Reads ``<prefix>_set_based.tsv`` (region rows with a ``P`` column), sorts by
    ``P`` ascending for ``top_hits``, and counts regions below
    ``significance_threshold`` as ``n_significant``. ``lambda_gc`` is ``None``
    (region-level, not per-SNP).
    """
    prefix = str(output_prefix)
    path = Path(prefix + "_set_based.tsv")
    if not path.exists():
        raise RuntimeError(f"set-based scan produced no region table at {path}")
    df = pd.read_csv(path, sep="\t")
    top = df.sort_values("P", ascending=True).head(top_k).reset_index(drop=True)
    n_sig = int((df["P"] < significance_threshold).sum()) if "P" in df.columns else 0
    return GwasResult(
        runtime_s=runtime_s, output_files={"set_based": path},
        model=model_label, test=test, correction=correction,
        n_variants=int(len(df)), n_significant=n_sig,
        significance_threshold=significance_threshold, lambda_gc=None,
        n_samples=n_samples, top_hits=top, trait_type=trait_type,
    )


def _materialize_regions(regions: Any, workdir: Path) -> str:
    """Return a path to a regions file for set-scan.

    A ``str``/``Path`` is returned as-is. A ``pandas.DataFrame`` (with
    chrom/start/end[/id] columns) is written to a temp tab-separated regions
    file that :func:`torchgenomics.io.regions.load_regions` can read.
    """
    import pandas as pd

    if isinstance(regions, (str, Path)):
        return str(regions)
    if isinstance(regions, pd.DataFrame):
        path = str(workdir / "regions.tsv")
        regions.to_csv(path, sep="\t", index=False)
        return path
    raise ValueError(
        f"regions must be a path or a pandas DataFrame (chrom/start/end[/id]); "
        f"got {type(regions).__name__}."
    )


def _set_extra_argv(opts: RunOptions, user_opts: dict, workdir: Path) -> list[str]:
    """Build set CLI flags: the required ``--regions`` plus any user model_options."""
    if opts.regions is None:
        raise ValueError(
            "Model 'set' needs regions. Pass regions=<path to a BED/region file | "
            "pandas DataFrame with chrom/start/end columns>."
        )
    regions_path = _materialize_regions(opts.regions, workdir)
    return ["--regions", regions_path] + _model_options_to_argv(user_opts)
```

Route in `_lowlevel_extra_argv`:

```python
    if alias == "set":
        return _set_extra_argv(opts, user_opts, workdir)
```

Register (dedicated read-back):

```python
    "set": ("set-scan", "_cmd_set_scan", "SetBasedScanner", _read_set_output),
```

Note: `_read_set_output`/`_read_bayes_output` are referenced in `_LOWLEVEL_CLI`, so define them ABOVE the `_LOWLEVEL_CLI` assignment (or assign `_LOWLEVEL_CLI` after the function defs — keep the module order valid).

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "set_output or set_requires or (set and gwas)" -v`
Expected: PASS. Then `-q` full file → no regressions.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire set through tg.gwas (regions= input + per-region read-back)"
```

---

### Task 5: `gwas(env=, regions=)` + CLI `--env`/`--regions` + shrink NotImplementedError + docs

**Files:** Modify `torchgenomics/api/gwas.py`, `torchgenomics/cli.py`, `CLAUDE.md`; Test `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py (append)
def test_set_runs_end_to_end_with_regions_df():
    import pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture(n=140, m=160)
    # one region spanning the first 40 synthesized variants (POS 0..39; see ArrayReader meta)
    regions = pd.DataFrame({"CHR": ["1"], "START": [0], "END": [39], "REGION": ["blockA"]})
    r = tg.gwas(y, G, models="set", regions=regions, kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.model == "SetBasedScanner" and r.n_variants >= 1

def test_only_mvlmm_remains_unwired():
    import torchgenomics as tg, pytest
    y, G = _quant_fixture()
    with pytest.raises(NotImplementedError):
        tg.gwas(y, G, models="mvlmm", kinship="auto", pcs=0, verbose=False)
```

```python
# tests/test_cli_gwas.py (append)
def test_cli_gwas_help_has_env_and_regions():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--env" in out.stdout and "--regions" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_set_runs_end_to_end_with_regions_df tests/test_cli_gwas.py::test_cli_gwas_help_has_env_and_regions -v`
Expected: FAIL (`gwas()` has no `env`/`regions`; CLI flags missing).

- [ ] **Step 3: Write minimal implementation**

1. `gwas.py` — add `env=None, regions=None` (keyword-only) to `gwas(...)` with docstring paragraphs; pass `env=env, regions=regions` into the `RunOptions(...)` built inside `gwas()` (the ~line 713 construction, not `recommend()`'s).

2. `cli.py` — in `_add_gwas_parser`:

```python
    p.add_argument("--env", default=None,
                   help="Environment variable file (TSV with an ENV column) for --models gxe.")
    p.add_argument("--regions", default=None,
                   help="Regions/BED file for --models set.")
```

In `_cmd_gwas`, pass `env=args.env, regions=args.regions` to the `api.gwas(...)` call.

3. Confirm `run_model`'s `NotImplementedError` set is now only `mvlmm` (gxe/set/bayes are registered in `_LOWLEVEL_CLI` from Tasks 2-4, so they no longer fall through; verify the fallthrough message still names the dedicated CLI for mvlmm).

4. `CLAUDE.md` — update the "Getting started" wired-list sentence to: `lmm/glm/blink/farmcpu/glmm/mklmm/gxe/set/bayes` run through `tg.gwas`; only `mvlmm` is not yet wired (needs a multi-trait extension). Mention `env=`/`regions=` and that `models="bayes"` returns fine-mapping (PIP/credible sets).

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py tests/test_cli_gwas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/gwas.py torchgenomics/cli.py CLAUDE.md tests/test_api_gwas.py tests/test_cli_gwas.py
git commit -m "friendly-api: gwas(env=, regions=) + CLI flags; only mvlmm remains unwired; docs"
```

---

### Task 6: R `tg_gwas(env=, regions=)` passthrough (paths)

**Files:** Modify `rTorchGenomics/R/gwas.R`, `rTorchGenomics/man/tg_gwas.Rd`; Test `rTorchGenomics/tests/testthat/test-gwas.R`.

**IMPORTANT:** author/modify R files via Bash heredoc / `sed` / `Rscript writeLines` (subagent Write/Edit is unreliable for this repo's R files). Parse-check with `Rscript -e 'invisible(parse("rTorchGenomics/R/gwas.R"))'`.

- [ ] **Step 1: Write the failing test** (append to `test-gwas.R`, mocking `bridge_call` via `mockery::stub` like the other tests in that file)

```r
test_that("tg_gwas forwards env and regions paths to the bridge", {
  captured <- NULL
  mockery::stub(tg_gwas, "bridge_call", function(fn, args = list()) {
    captured <<- args
    .fake_scan_dict()          # existing helper in this file
  })
  tg_gwas("pheno.tsv", "data.bed", models = "gxe", env = "env.tsv")
  expect_equal(captured$env, "env.tsv")
  tg_gwas("pheno.tsv", "data.bed", models = "set", regions = "regions.bed")
  expect_equal(captured$regions, "regions.bed")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: FAIL (`tg_gwas` has no `env`/`regions` args).

- [ ] **Step 3: Write minimal implementation**

Edit `rTorchGenomics/R/gwas.R`: add `env = NULL, regions = NULL` to `tg_gwas`'s signature; add `env = .as_path(env)` and `regions = .as_path(regions)` inside the `.compact(list(...))` that feeds `bridge_call("gwas", ...)`. Add roxygen `@param env`/`@param regions` lines (state: paths only in R; gxe needs env, set needs regions). Regenerate: `Rscript -e 'roxygen2::roxygenise()'`.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: PASS. Parse-check gwas.R.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/gwas.R rTorchGenomics/man/tg_gwas.Rd rTorchGenomics/tests/testthat/test-gwas.R
git commit -m "friendly-api: R tg_gwas env/regions passthrough"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (read-back selection) → Task 1. ✅
- §Component 2 (`_read_bayes_output`) → Task 3. ✅
- §Component 3 (`_read_set_output`) → Task 4. ✅
- §Component 4 (`env=`/`regions=` inputs + materialization + validation) → Tasks 2 (env), 4 (regions), 5 (gwas/CLI surface). ✅
- §Component 5 (per-model argv gxe/set/bayes) → Tasks 2, 3, 4. ✅
- §Component 6 (registry/dispatch/docs/CLI/R) → Tasks 2-6; NotImplementedError shrink + docs in Task 5; R in Task 6. ✅
- §Testing → Tasks 1-6 (unit read-back tests in 3/4; e2e in 2/3/5; friendly errors in 2/4; CLI in 5; R in 6). ✅

**Placeholder scan:** No TBD/TODO; every code step has real code. ✅

**Type consistency:** `_read_bayes_output`/`_read_set_output` share `_read_scan_output`'s exact keyword signature (so they slot into the read-back selector). `_lowlevel_extra_argv` gains a `workdir` param in Task 2 and is used by gxe (Task 2) and set (Task 4); its call site in `_run_lowlevel_cli` is updated in Task 2. `_LOWLEVEL_CLI` 4-tuple shape is set in Task 1 and every later entry matches. ✅

**Known verification points for the implementer:**
- (a) Order the module so `_read_bayes_output`/`_read_set_output` are defined before the `_LOWLEVEL_CLI` dict that references them (or reference by name and assign the dict after the defs). Task 3 adds bayes's read-back; Task 4 adds set's — ensure the dict edits keep valid module ordering.
- (b) `_lowlevel_extra_argv` gains `workdir`; confirm the single call site in `_run_lowlevel_cli` passes the `workdir` it already created, and that `_materialize_env`/`_materialize_regions` write into it (so temp files are GC-cleaned with the result).
- (c) The set end-to-end test (Task 5) depends on the region coordinates overlapping the ArrayReader's synthesized `POS` values (0..m-1 per the in-memory `ArrayReader` metadata). If the region yields zero variants, widen it; keep the assertion `n_variants >= 1`.
- (d) gxe/bayes/set convergence on the tiny fixture: if a model needs more samples/variants, bump the fixture but keep assertions meaningful; if genuinely non-converging, report DONE_WITH_CONCERNS rather than weaken.
