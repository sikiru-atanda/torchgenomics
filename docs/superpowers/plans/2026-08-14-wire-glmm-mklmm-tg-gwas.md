# Wire glmm + mklmm through tg.gwas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tg.gwas(phenotype, genotype, models="glmm")` and `models="mklmm"` run through the existing friendly dispatch core, plus a `model_options` knob to pass per-model settings — across Python, CLI, and R.

**Architecture:** Reuse the existing `_run_lowlevel_cli` route (blink/farmcpu pattern). Both new models write `<prefix>.assoc.tsv` via `_apply_correction_and_save`, so the existing `_read_scan_output` returns a uniform `GwasResult` unchanged. The only new plumbing is passing model-specific CLI flags (`--family` for glmm, `--kernels` for mklmm) into the runner via a new `extra_argv`, sourced from inferred defaults + a user `model_options` dict. No new statistics.

**Tech Stack:** Python 3.12, pandas, argparse; existing `torchgenomics.api._dispatch` / `.gwas` / `cli.py`; R reticulate bridge in `rTorchGenomics/`.

## Global Constraints

- **No new statistics** — orchestrate the existing CLI runners (`_cmd_glmm_scan`, `_cmd_mklmm_scan`). No change to model math or to `lmm_scan`/`glm_scan` signatures.
- **Backward compatible / additive** — new optional `extra_argv` (default `None`), new optional `RunOptions.model_options` (default `None`), new optional `gwas(model_options=None)`, two new `_LOWLEVEL_CLI` entries. No existing default changes. blink/farmcpu behavior byte-for-byte unchanged.
- **Reviewer-grade docstrings** on every new/changed public symbol (binding project standard).
- **Friendly errors** — glmm on a continuous trait raises an actionable `ValueError` (never a raw traceback); no silent wrong results.
- **`_auto_model` unchanged** — it only ever returns `lmm`/`glm`; glmm/mklmm are reached only when the user names them.
- **mklmm default kernels = `additive,dominance`** (overrides the heavier `additive,dominance,epistatic` CLI default).
- **glmm categorical default family = `multinomial`** (no ordering assumption); overridable via `model_options`.
- Verify commands run under `TORCHGENOMICS_DISABLE_NATIVE=1`.

## File Structure

- Modify: `torchgenomics/api/_dispatch.py` — `RunOptions.model_options`; `_model_options_to_argv`; `_lowlevel_extra_argv` (+ `_glmm_extra_argv`, `_mklmm_extra_argv`); `extra_argv` on `_run_lowlevel_cli`; two `_LOWLEVEL_CLI` entries; pass `alias` through in `run_model`.
- Modify: `torchgenomics/api/gwas.py` — `gwas(model_options=None)` threaded into `RunOptions`.
- Modify: `torchgenomics/cli.py` — `--model-options` JSON flag on the `gwas` subparser; `_cmd_gwas` parses + forwards it.
- Modify: `CLAUDE.md` — "Getting started" wired-model list (`+ glmm/mklmm`; not-yet-wired `bayes/gxe/set/mvlmm`).
- Modify: `rTorchGenomics/R/gwas.R` — `tg_gwas(model_options = NULL)` passthrough; regenerate `man/tg_gwas.Rd`.
- Test: `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`, `rTorchGenomics/tests/testthat/test-gwas.R`.

**Interface contract (types later tasks rely on):**

```python
# _dispatch.py
def _model_options_to_argv(opts_map: dict) -> list[str]: ...
def _lowlevel_extra_argv(alias: str, inputs: "GwasInputs", opts: "RunOptions") -> list[str]: ...
# RunOptions gains:  model_options: dict[str, dict] | None = None
def _run_lowlevel_cli(inputs, opts, *, subcommand, runner, model_label, workdir,
                      alias: str) -> "GwasResult": ...   # now also takes alias; builds extra_argv internally
# gwas.py
def gwas(..., model_options: dict[str, dict] | None = None) -> "GwasResult | GwasComparison": ...
```

---

### Task 1: `model_options` → argv plumbing (generic, models still unwired)

**Files:**
- Modify: `torchgenomics/api/_dispatch.py`
- Test: `tests/test_api_gwas.py`

**Interfaces:**
- Produces: `_model_options_to_argv`, `_lowlevel_extra_argv`, `RunOptions.model_options`, `extra_argv`/`alias` on `_run_lowlevel_cli`.
- Consumes: existing `_run_lowlevel_cli`, `_LOWLEVEL_CLI`, `run_model`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py  (append)
def test_model_options_to_argv_conversion():
    from torchgenomics.api._dispatch import _model_options_to_argv
    assert _model_options_to_argv({}) == []
    assert _model_options_to_argv({"kernels": "additive,dominance"}) == ["--kernels", "additive,dominance"]
    assert _model_options_to_argv({"n_categories": 3}) == ["--n-categories", "3"]
    # bool True -> bare store_true flag; False -> omitted
    assert _model_options_to_argv({"firth": True}) == ["--firth"]
    assert _model_options_to_argv({"firth": False}) == []

def test_runoptions_has_model_options_default_none():
    from torchgenomics.api._dispatch import RunOptions
    assert RunOptions().model_options is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_model_options_to_argv_conversion tests/test_api_gwas.py::test_runoptions_has_model_options_default_none -v`
Expected: FAIL (`_model_options_to_argv` missing / `model_options` attribute missing).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/_dispatch.py`:

1. Add the field to `RunOptions` (alongside the other fields, after `verbose`), with a docstring paragraph in the class docstring:

```python
    model_options: dict[str, dict] | None = None
```

Docstring paragraph to add to `RunOptions` (under Parameters):
```
    model_options : dict[str, dict] | None, default None
        Per-model advanced settings for low-level (CLI-backed) models, keyed
        by model alias — e.g. ``{"glmm": {"family": "ordinal"}, "mklmm":
        {"kernels": "additive,dominance,epistatic"}}``. Each inner dict is
        translated into the model's CLI flags (key ``k`` -> ``--k-with-dashes``;
        a bool ``True`` -> a bare store-true flag, ``False`` -> omitted; any
        other scalar -> ``--flag value``). Ignored by api-backed models
        (``lmm``/``glm``). Absent keys fall back to each model's friendly
        defaults (see :func:`_lowlevel_extra_argv`).
```

2. Add the helpers (place them just above `_LOWLEVEL_CLI`):

```python
def _model_options_to_argv(opts_map: dict) -> list[str]:
    """Translate a per-model options dict into CLI argument tokens.

    Key ``k`` becomes ``--k-with-dashes``. A bool ``True`` becomes a bare
    store-true flag; ``False`` is omitted entirely. Any other scalar ``v``
    becomes the pair ``["--flag", str(v)]``. Insertion order is preserved so
    the resulting argv is deterministic.
    """
    argv: list[str] = []
    for key, val in (opts_map or {}).items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    return argv


def _lowlevel_extra_argv(alias: str, inputs: GwasInputs, opts: RunOptions) -> list[str]:
    """Build the model-specific CLI flags for a low-level (CLI-backed) model.

    Merges each model's friendly defaults with any user overrides in
    ``opts.model_options[alias]`` and returns them as argv tokens appended to
    the base ``_run_lowlevel_cli`` argv. Models with no special handling
    (e.g. ``blink``/``farmcpu``) simply forward any user options verbatim.
    Per-model default/inference logic is added by later tasks.
    """
    user_opts = (opts.model_options or {}).get(alias, {})
    return _model_options_to_argv(user_opts)
```

3. Thread `extra_argv` + `alias` through `_run_lowlevel_cli`. Change its signature to accept `alias: str` (keyword) and, at the very top of the body (before `_materialize_inputs`), compute:

```python
    extra_argv = _lowlevel_extra_argv(alias, inputs, opts)
```

Then in the `argv = [...]` construction, append `extra_argv` after the base args (after the `--correction` entry, before/with the device block):

```python
    argv = [
        subcommand,
        "--genotype", geno_path,
        "--phenotype", pheno_path,
        "--output", output_prefix,
        "--correction", opts.correction,
    ]
    if opts.device is not None:
        argv += ["--device", str(opts.device)]
    argv += extra_argv
```

4. In `run_model`, pass `alias=low` in the `_run_lowlevel_cli(...)` call:

```python
            result = _run_lowlevel_cli(
                inputs,
                opts,
                subcommand=subcommand,
                runner=runner,
                model_label=model_label,
                workdir=workdir,
                alias=low,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_model_options_to_argv_conversion tests/test_api_gwas.py::test_runoptions_has_model_options_default_none -v`
Expected: PASS.

Also run the blink/farmcpu regression (they must still work with the new `extra_argv`/`alias`): `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "run_model or blink or farmcpu or comparison" -q`
Expected: PASS (blink/farmcpu unaffected; their `extra_argv` is `[]`).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: model_options->argv plumbing (extra_argv on low-level route)"
```

---

### Task 2: Wire `glmm` (family inferred from trait type)

**Files:**
- Modify: `torchgenomics/api/_dispatch.py`
- Test: `tests/test_api_gwas.py`

**Interfaces:**
- Consumes: Task 1's `_model_options_to_argv`, `_lowlevel_extra_argv`, `_LOWLEVEL_CLI`.
- Produces: `_glmm_extra_argv`; `_LOWLEVEL_CLI["glmm"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py  (append)
def _binary_fixture(n=160, m=200, seed=0):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    y = pd.Series(rng.integers(0, 2, size=n), index=ids, name="disease")
    return y, G

def test_glmm_extra_argv_infers_family_from_trait_type():
    import pandas as pd
    from torchgenomics.api._dispatch import _glmm_extra_argv
    from torchgenomics.api._inputs import GwasInputs
    # binary trait -> --family binary
    inp_b = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([0, 1, 1, 0]),
                       covariates=None, trait_type="binary", n_samples=4, trait_name="t")
    assert _glmm_extra_argv(inp_b, {}) == ["--family", "binary"]
    # categorical -> --family multinomial --n-categories <#distinct>
    inp_c = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([0, 1, 2, 1, 2]),
                       covariates=None, trait_type="categorical", n_samples=5, trait_name="t")
    argv = _glmm_extra_argv(inp_c, {})
    assert argv[:2] == ["--family", "multinomial"]
    assert "--n-categories" in argv and "3" in argv
    # continuous -> friendly ValueError
    inp_q = GwasInputs(genotype_path_or_reader="x", phenotype=pd.Series([1.1, 2.2, 3.3]),
                       covariates=None, trait_type="continuous", n_samples=3, trait_name="height")
    import pytest
    with pytest.raises(ValueError) as e:
        _glmm_extra_argv(inp_q, {})
    assert "glmm" in str(e.value).lower() and "lmm" in str(e.value).lower()
    # user override wins (family explicitly set)
    assert _glmm_extra_argv(inp_b, {"family": "ordinal", "n_categories": 3}) == [
        "--family", "ordinal", "--n-categories", "3"]

def test_glmm_runs_through_tg_gwas_on_binary_trait():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _binary_fixture()
    r = tg.gwas(y, G, models="glmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0

def test_glmm_on_continuous_trait_friendly_error():
    import numpy as np, pandas as pd, pytest, torchgenomics as tg
    rng = np.random.default_rng(1)
    ids = [f"s{i}" for i in range(120)]
    p = rng.uniform(0.2, 0.8, size=150)
    G = rng.binomial(2, p, size=(120, 150)).astype(float)
    y = pd.Series(rng.normal(size=120), index=ids, name="height")
    with pytest.raises(ValueError) as e:
        tg.gwas(y, G, models="glmm", kinship="auto", pcs=0, verbose=False)
    assert "glmm" in str(e.value).lower() and "lmm" in str(e.value).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k glmm -v`
Expected: FAIL (`_glmm_extra_argv` missing; `glmm` still raises `NotImplementedError`).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/_dispatch.py`:

1. Add `_glmm_extra_argv` (place near `_lowlevel_extra_argv`):

```python
def _glmm_extra_argv(inputs: GwasInputs, user_opts: dict) -> list[str]:
    """Build glmm CLI flags, inferring ``--family`` from the trait type.

    binary -> ``--family binary``; categorical -> ``--family multinomial``
    with ``--n-categories`` set to the number of distinct phenotype values
    (multinomial is the default because it assumes no category ordering — a
    user with ordered categories can pass ``model_options={"glmm":
    {"family": "ordinal"}}``). A continuous trait raises a friendly
    ``ValueError``. Explicit user options always win over the inference.
    """
    import pandas as pd

    opts_map = dict(user_opts)
    if "family" not in opts_map:
        tt = inputs.trait_type
        if tt == "binary":
            opts_map["family"] = "binary"
        elif tt == "categorical":
            opts_map["family"] = "multinomial"
            opts_map.setdefault("n_categories", int(pd.Series(inputs.phenotype).nunique()))
        else:
            raise ValueError(
                f"Model 'glmm' is for binary/categorical traits, but trait "
                f"'{inputs.trait_name}' is {tt}. Use models='lmm' for a "
                f"continuous trait (or models='glm' for fixed-effects)."
            )
    return _model_options_to_argv(opts_map)
```

2. Route glmm inside `_lowlevel_extra_argv`:

```python
def _lowlevel_extra_argv(alias: str, inputs: GwasInputs, opts: RunOptions) -> list[str]:
    user_opts = (opts.model_options or {}).get(alias, {})
    if alias == "glmm":
        return _glmm_extra_argv(inputs, user_opts)
    return _model_options_to_argv(user_opts)
```

3. Register glmm in `_LOWLEVEL_CLI`:

```python
    "glmm": ("glmm-scan", "_cmd_glmm_scan", "GLMM"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k glmm -v`
Expected: PASS. (The continuous-trait `ValueError` is raised while building `extra_argv` inside `run_model`'s `try`, so the temp workdir is still cleaned by the existing `except` handler.)

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire glmm through tg.gwas (family inferred from trait type)"
```

---

### Task 3: Wire `mklmm` (lighter default kernels) + comparison

**Files:**
- Modify: `torchgenomics/api/_dispatch.py`
- Test: `tests/test_api_gwas.py`

**Interfaces:**
- Consumes: Task 1/2 helpers, `_LOWLEVEL_CLI`.
- Produces: `_mklmm_extra_argv`; `_LOWLEVEL_CLI["mklmm"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py  (append)
def _quant_fixture(n=140, m=180, seed=2):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    p = rng.uniform(0.2, 0.8, size=m)
    G = rng.binomial(2, p, size=(n, m)).astype(float)
    y = pd.Series(rng.normal(size=n), index=ids, name="yield")
    return y, G

def test_mklmm_extra_argv_default_and_override():
    from torchgenomics.api._dispatch import _mklmm_extra_argv
    assert _mklmm_extra_argv({}) == ["--kernels", "additive,dominance"]
    assert _mklmm_extra_argv({"kernels": "additive,dominance,epistatic"}) == [
        "--kernels", "additive,dominance,epistatic"]

def test_mklmm_runs_through_tg_gwas():
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture()
    r = tg.gwas(y, G, models="mklmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult) and r.n_variants > 0

def test_multimodel_comparison_includes_glmm():
    import torchgenomics as tg
    from torchgenomics.api import GwasComparison
    y, G = _binary_fixture()
    cmp = tg.gwas(y, G, models=["glm", "glmm"], kinship=False, pcs=0, verbose=False)
    assert isinstance(cmp, GwasComparison)
    assert set(cmp.results) == {"glm", "glmm"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "mklmm or comparison_includes_glmm" -v`
Expected: FAIL (`_mklmm_extra_argv` missing; `mklmm` still `NotImplementedError`).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/_dispatch.py`:

1. Add `_mklmm_extra_argv`:

```python
def _mklmm_extra_argv(user_opts: dict) -> list[str]:
    """Build mklmm CLI flags with a friendly default kernel set.

    Defaults to ``--kernels additive,dominance`` (the mklmm-scan CLI default
    also includes the expensive ``epistatic`` kernel; tg.gwas omits it for a
    fast, friendly default). A user can opt back in via
    ``model_options={"mklmm": {"kernels": "additive,dominance,epistatic"}}``.
    """
    opts_map = dict(user_opts)
    opts_map.setdefault("kernels", "additive,dominance")
    return _model_options_to_argv(opts_map)
```

2. Route mklmm inside `_lowlevel_extra_argv` (add the branch):

```python
    if alias == "mklmm":
        return _mklmm_extra_argv(user_opts)
```

3. Register mklmm in `_LOWLEVEL_CLI`:

```python
    "mklmm": ("mklmm-scan", "_cmd_mklmm_scan", "MultiKernelLMM"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py -k "mklmm or comparison_includes_glmm" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_dispatch.py tests/test_api_gwas.py
git commit -m "friendly-api: wire mklmm through tg.gwas (default kernels additive,dominance)"
```

---

### Task 4: Surface `model_options` in `gwas()` + CLI `--model-options` + docs

**Files:**
- Modify: `torchgenomics/api/gwas.py`, `torchgenomics/cli.py`, `CLAUDE.md`
- Test: `tests/test_api_gwas.py`, `tests/test_cli_gwas.py`

**Interfaces:**
- Consumes: `RunOptions.model_options` (Task 1), `run_model` glmm/mklmm wiring (Tasks 2-3).
- Produces: `gwas(model_options=...)`; CLI `gwas --model-options '<json>'`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py  (append)
def test_gwas_model_options_reaches_runner():
    # mklmm with an explicit kernels override runs end-to-end (proves the
    # option threads gwas() -> RunOptions -> run_model -> argv).
    import torchgenomics as tg
    from torchgenomics.api import GwasResult
    y, G = _quant_fixture()
    r = tg.gwas(y, G, models="mklmm", kinship="auto", pcs=0, verbose=False,
                model_options={"mklmm": {"kernels": "additive,dominance"}})
    assert isinstance(r, GwasResult) and r.n_variants > 0
```

```python
# tests/test_cli_gwas.py  (append)
def test_cli_gwas_help_has_model_options_flag():
    import subprocess, sys
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--model-options" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_gwas_model_options_reaches_runner tests/test_cli_gwas.py::test_cli_gwas_help_has_model_options_flag -v`
Expected: FAIL (`gwas()` has no `model_options` kwarg; CLI flag missing).

- [ ] **Step 3: Write minimal implementation**

1. `torchgenomics/api/gwas.py` — add `model_options: dict[str, dict] | None = None` as a keyword-only parameter of `gwas(...)` (in the signature and its docstring), and pass it into the `RunOptions(...)` construction that `gwas` builds (add `model_options=model_options` to that call). Reviewer-grade docstring paragraph:

```
    model_options : dict[str, dict] | None, default None
        Per-model advanced settings for CLI-backed models, keyed by alias —
        e.g. ``{"glmm": {"family": "ordinal"}, "mklmm": {"kernels":
        "additive,dominance,epistatic"}}``. Ignored by ``lmm``/``glm``.
        See :class:`torchgenomics.api._dispatch.RunOptions`.
```

2. `torchgenomics/cli.py` — in `_add_gwas_parser`, add the flag (mirror the existing string flags there):

```python
    p.add_argument("--model-options", default=None,
                   help="JSON dict of per-model advanced settings, keyed by "
                        "model alias, e.g. '{\"glmm\": {\"family\": \"ordinal\"}}'.")
```

In `_cmd_gwas`, parse and forward it (near where the other args are read; use the stdlib `json`):

```python
    import json
    model_options = json.loads(args.model_options) if getattr(args, "model_options", None) else None
```

and add `model_options=model_options` to the `api.gwas(...)` call. If `json.loads` raises, let it surface as a clear error (or wrap with a friendly message naming the flag).

3. `CLAUDE.md` — in the "Getting started" section, update the wired/not-yet-wired sentence to:

> **`lmm`, `glm`, `blink`, `farmcpu`, `glmm`, and `mklmm` run through `tg.gwas` today.** Registry models `bayes`, `gxe`, `set`, and `mvlmm` are listed by `tg.list_models()` but not yet wired (they raise a clear `NotImplementedError` pointing at the dedicated CLI). Pass per-model settings via `model_options={"glmm": {"family": "ordinal"}}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_gwas.py::test_gwas_model_options_reaches_runner tests/test_cli_gwas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/gwas.py torchgenomics/cli.py CLAUDE.md tests/test_api_gwas.py tests/test_cli_gwas.py
git commit -m "friendly-api: model_options on gwas() + CLI --model-options; docs wired-list update"
```

---

### Task 5: R `tg_gwas` `model_options` passthrough

**Files:**
- Modify: `rTorchGenomics/R/gwas.R`, `rTorchGenomics/man/tg_gwas.Rd`
- Test: `rTorchGenomics/tests/testthat/test-gwas.R`

**Interfaces:**
- Consumes: Python `torchgenomics.api.gwas(model_options=...)` (Task 4).
- Produces: `tg_gwas(model_options = NULL)`.

**IMPORTANT:** author/modify R files via Bash heredoc / `Rscript writeLines` (subagent Write/Edit is unreliable for this repo's R files). Parse-check every edit with `Rscript -e 'parse("rTorchGenomics/R/gwas.R")'`.

- [ ] **Step 1: Write the failing test** (append to `rTorchGenomics/tests/testthat/test-gwas.R`, mocking the bridge like the existing mocked tests in that file)

```r
test_that("tg_gwas forwards model_options to the bridge", {
  captured <- NULL
  testthat::local_mocked_bindings(
    bridge_call = function(fn_name, args = list()) {
      captured <<- args
      # minimal ScanRun-shaped dict so new_from_dict succeeds
      list(model = "GLMM", test = "score", correction = "bh",
           n_variants = 0L, n_significant = 0L, top_hits = list())
    }
  )
  tg_gwas("pheno.tsv", "data.bed", models = "glmm",
          model_options = list(glmm = list(family = "ordinal")))
  expect_equal(captured$model_options$glmm$family, "ordinal")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: FAIL (`tg_gwas` has no `model_options` arg, so it is not forwarded).

- [ ] **Step 3: Write minimal implementation**

Edit `rTorchGenomics/R/gwas.R` (via heredoc/sed): add `model_options = NULL` to the `tg_gwas` signature and include it in the `.compact(list(...))` passed to `bridge_call("gwas", ...)`:

```r
    model_options = model_options
```

Add a roxygen `@param model_options` line:

```r
#' @param model_options Named list of per-model advanced settings, keyed by
#'   model alias (e.g. `list(glmm = list(family = "ordinal"))`). Forwarded to
#'   the Python friendly API; ignored by lmm/glm. Default `NULL`.
```

Regenerate the man page: `Rscript -e 'roxygen2::roxygenise()'` (or hand-edit `man/tg_gwas.Rd` to add the `\item{model_options}` entry if roxygen is unavailable).

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'testthat::test_file("tests/testthat/test-gwas.R")'`
Expected: PASS (mocked suite green). Parse-check: `Rscript -e 'invisible(parse("R/gwas.R"))'` → no error.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/gwas.R rTorchGenomics/man/tg_gwas.Rd rTorchGenomics/tests/testthat/test-gwas.R
git commit -m "friendly-api: R tg_gwas model_options passthrough"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (`extra_argv` on `_run_lowlevel_cli`) → Task 1. ✅
- §Component 2 (register glmm/mklmm) → Tasks 2, 3. ✅
- §Component 3 (per-model argv builder: glmm family inference + continuous error; mklmm default kernels) → Tasks 2, 3. ✅
- §Component 4 (`model_options` on RunOptions + gwas) → Tasks 1, 4. ✅
- §Component 5 (shrink NotImplementedError set) → happens implicitly when glmm/mklmm join `_LOWLEVEL_CLI` (Tasks 2-3); the not-yet-wired list `bayes/gxe/set/mvlmm` is documented in Task 4 (CLAUDE.md). ✅
- §Component 6 (CLI + R parity + docs) → Tasks 4, 5. ✅
- §Testing (glmm binary/categorical/continuous; mklmm; model_options override; comparison; CLI; R) → Tasks 2-5. ✅

**Placeholder scan:** No TBD/TODO; every code step shows the actual code. ✅

**Type consistency:** `_model_options_to_argv` / `_lowlevel_extra_argv` / `_glmm_extra_argv` / `_mklmm_extra_argv` names and signatures are consistent across Tasks 1-3; `RunOptions.model_options` and `gwas(model_options=...)` match; `_run_lowlevel_cli(..., alias=...)` is added in Task 1 and used by `run_model` there. ✅

**Known verification points for the implementer:**
- (a) Confirm the exact construction site of `RunOptions(...)` inside `gwas()` (gwas.py) to thread `model_options` — reuse it, don't add a second construction.
- (b) The glmm-continuous `ValueError` is raised while building `extra_argv` inside `run_model`'s `try`; confirm the existing `except BaseException: _cleanup_workdir(workdir); raise` still cleans the temp dir (it does — the workdir is created before the `try`).
- (c) Categorical fixtures for glmm: PQL on a tiny fixture must converge; if a categorical end-to-end test is flaky, keep the categorical assertion at the `_glmm_extra_argv` unit level (Task 2 Step 1 already does) and only run binary glmm end-to-end.
