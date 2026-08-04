# User-Friendly API (`tg.gwas`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single friendly entry point `tg.gwas(phenotype, genotype, models=…)` that auto-runs QC → kinship → PCA → the **user's chosen model(s)** (one or several) and returns a self-explaining result — plus a model registry, guidance, and Python/R/CLI parity. Thin orchestration over existing code; no new statistics.

**Architecture:** New modules under `torchgenomics/api/`: `_registry.py` (model aliases), `_inputs.py` (load/align/trait-type + friendly errors), `_dispatch.py` (`_run_model` — the shared core), `gwas.py` (`gwas`/`recommend`/`models`). Enrich the existing `ScanRun` result (aliased as `GwasResult`) with `.hits`/`.diagnostics`/plain-language `.summary()`/`.report()`. Add a `GwasComparison`. CLI `gwas`/`recommend`/`models` + R wrappers call the same core.

**Tech Stack:** Python 3.12, pandas, torch; existing `torchgenomics.api/models/scan/linalg/viz`. Spec: `docs/superpowers/specs/2026-08-03-user-friendly-api-design.md`.

## Global Constraints

- **No new statistics** — orchestrate existing `api`/`models`/`scan`/`linalg`/`viz`. Low-level API unchanged; fully backward-compatible.
- **`models=` is the user's choice** — accepts one alias, a **list** (→ `GwasComparison`), or `"auto"` (opt-in fallback, printed + overridable). Never force a model on the user.
- **Friendly errors:** every entry validates inputs early and raises **actionable** messages (never a raw torch/pandas traceback). Unknown model → list valid names; unmatched samples → counts + hint; wrong trait/model → suggest the right one.
- **Simple by default, powerful on demand:** auto choices are made AND printed (`verbose=True` default); every choice overridable.
- **Consistency:** every `api` scan returns the same enriched result shape; uniform arg names (`phenotype, genotype, covariates, kinship, pcs, correction, output, device, verbose`).
- **Cross-surface parity:** Python `tg.gwas`, CLI `torchgenomics gwas`, R `tg_gwas` all call one shared core (`_dispatch` + `gwas` logic); the existing CLI `pipeline` becomes an alias.
- Reviewer-grade docstrings; commits reference "friendly-api".

## File Structure

- Create: `torchgenomics/api/_registry.py` — `MODEL_REGISTRY`, `resolve_model`, `list_models`, GAPIT synonyms.
- Create: `torchgenomics/api/_inputs.py` — `load_inputs`, `detect_trait_type`, friendly-error helpers.
- Create: `torchgenomics/api/_dispatch.py` — `run_model(alias, inputs, opts) -> GwasResult` (shared core).
- Create: `torchgenomics/api/gwas.py` — `gwas`, `recommend`, `models`, `GwasComparison`.
- Modify: `torchgenomics/api/_results.py` — enrich `ScanRun` (alias `GwasResult`): `.hits`, `.diagnostics`, plain-language `.summary()`, `.report()`, `trait_type`.
- Modify: `torchgenomics/api/__init__.py` + `torchgenomics/__init__.py` — export `gwas`, `recommend`, `models`, `GwasResult`, `GwasComparison`.
- Modify: `torchgenomics/cli.py` — `gwas`/`recommend`/`models` subcommands; `pipeline` → alias.
- Create: `rTorchGenomics/R/gwas.R` — `tg_gwas`/`tg_recommend`/`tg_models`; + `NAMESPACE`, `_pkgdown.yml`.
- Create: `tests/test_api_gwas.py`, `tests/test_api_registry.py`, `tests/test_api_inputs.py`.

**Interface contract (types later tasks rely on):**

```python
# _registry.py
@dataclass(frozen=True)
class ModelSpec:
    alias: str; label: str; trait_types: tuple[str, ...]; description: str
    runner: str   # "api_lmm" | "api_glm" | "lowlevel:FarmCPU" | ...
MODEL_REGISTRY: dict[str, ModelSpec]
def resolve_model(name: str) -> ModelSpec: ...          # alias/synonym -> spec; friendly error on unknown
def list_models() -> pd.DataFrame: ...                  # alias | label | trait types | description

# _inputs.py
@dataclass
class GwasInputs:
    genotype_path_or_reader: object
    phenotype: "pd.Series"       # aligned, indexed by sample id
    covariates: "pd.DataFrame | None"
    trait_type: str              # "continuous"|"binary"|"categorical"
    n_samples: int; trait_name: str
def detect_trait_type(y: "pd.Series") -> str: ...
def load_inputs(phenotype, genotype, covariates=None, trait=None,
                trait_type=None) -> GwasInputs: ...     # friendly errors on mismatch

# _dispatch.py
@dataclass
class RunOptions:
    kinship="auto"; pcs="auto"; qc=True; correction="bh"
    device=None; output=None; top_k=50; verbose=True
def run_model(alias: str, inputs: "GwasInputs", opts: "RunOptions") -> "GwasResult": ...

# gwas.py
def gwas(phenotype, genotype, *, covariates=None, kinship="auto", pcs="auto",
         trait=None, trait_type=None, models="auto", qc=True, correction="bh",
         output=None, device=None, verbose=True) -> "GwasResult | GwasComparison": ...
def recommend(phenotype, genotype, *, covariates=None) -> "Recommendation": ...
def models() -> "pd.DataFrame": ...
```

---

### Task 1: Model registry (`_registry.py`)

**Files:** Create `torchgenomics/api/_registry.py`; Test `tests/test_api_registry.py`.

**Interfaces:** Produces `MODEL_REGISTRY`, `ModelSpec`, `resolve_model`, `list_models` (see contract).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_registry.py
import pytest
from torchgenomics.api._registry import resolve_model, list_models, MODEL_REGISTRY

def test_registry_core_aliases_and_synonyms():
    assert resolve_model("lmm").runner == "api_lmm"
    assert resolve_model("glm").runner == "api_glm"
    # GAPIT-name synonyms map in
    assert resolve_model("MLM").alias == "lmm"
    assert resolve_model("Blink").alias == "blink"
    assert resolve_model("FarmCPU").alias == "farmcpu"
    # every spec advertises trait types + a description
    for spec in MODEL_REGISTRY.values():
        assert spec.trait_types and spec.description
    # listing is a tidy table
    df = list_models()
    assert {"alias", "label", "trait_types", "description"}.issubset(df.columns)
    assert "lmm" in set(df["alias"])

def test_unknown_model_raises_friendly_error():
    with pytest.raises(ValueError) as e:
        resolve_model("mlmx")
    msg = str(e.value)
    assert "mlmx" in msg and "lmm" in msg  # names the bad input + lists valid options
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_api_registry.py -v` → FAIL (module missing).

- [ ] **Step 3: Write minimal implementation**

```python
# torchgenomics/api/_registry.py
"""Model registry: short aliases over the models/api layer for tg.gwas(models=…)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    label: str
    trait_types: tuple[str, ...]
    description: str
    runner: str  # "api_lmm" | "api_glm" | "lowlevel:<ClassName>"


_SPECS = [
    ModelSpec("lmm", "SingleTraitLMM", ("continuous",), "GRM mixed model (GEMMA-equivalent)", "api_lmm"),
    ModelSpec("glm", "GLM", ("continuous", "binary", "categorical"), "Fixed-effects; PCs as covariates", "api_glm"),
    ModelSpec("farmcpu", "FarmCPU", ("continuous",), "Iterative multi-locus", "lowlevel:FarmCPU"),
    ModelSpec("blink", "BLINK", ("continuous",), "Fast multi-locus", "lowlevel:BLINK"),
    ModelSpec("mvlmm", "MultiTraitLMM", ("continuous",), "Multi-trait LMM", "lowlevel:MultiTraitLMM"),
    ModelSpec("glmm", "BinaryGLMM", ("binary", "categorical"), "PQL GLMM (SAIGE-style)", "lowlevel:BinaryGLMM"),
    ModelSpec("mklmm", "MultiKernelLMM", ("continuous",), "Additive+dominance kernels", "lowlevel:MultiKernelLMM"),
    ModelSpec("gxe", "GxELMM", ("continuous",), "Genotype × environment", "lowlevel:GxELMM"),
    ModelSpec("set", "SetBasedScanner", ("continuous", "binary"), "Region/gene-based (SKAT)", "lowlevel:SetBasedScanner"),
    ModelSpec("bayes", "BayesianVS", ("continuous",), "SuSiE fine-mapping scan", "lowlevel:BayesianVS"),
]
MODEL_REGISTRY: dict[str, ModelSpec] = {s.alias: s for s in _SPECS}

# GAPIT-name synonyms (case-insensitive) for migrants
_SYNONYMS = {"mlm": "lmm", "cmlm": "lmm", "glm": "glm", "blink": "blink",
             "farmcpu": "farmcpu", "mlmm": "farmcpu"}


def resolve_model(name: str) -> ModelSpec:
    key = str(name).strip().lower()
    key = _SYNONYMS.get(key, key)
    if key not in MODEL_REGISTRY:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{name}'. Valid models: {valid}. "
            f"(GAPIT names like MLM/Blink/FarmCPU are accepted too.) See tg.models()."
        )
    return MODEL_REGISTRY[key]


def list_models() -> pd.DataFrame:
    return pd.DataFrame(
        [{"alias": s.alias, "label": s.label,
          "trait_types": ",".join(s.trait_types), "description": s.description}
         for s in _SPECS]
    )
```

- [ ] **Step 4: Run test** — `pytest tests/test_api_registry.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: model registry (aliases + GAPIT synonyms + tg.models listing)"`

---

### Task 2: Input loading + trait-type detection + friendly errors (`_inputs.py`)

**Files:** Create `torchgenomics/api/_inputs.py`; Test `tests/test_api_inputs.py`.

**Interfaces:** Produces `load_inputs`, `detect_trait_type`, `GwasInputs` (see contract).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_inputs.py
import numpy as np, pandas as pd, pytest
from torchgenomics.api._inputs import detect_trait_type, load_inputs

def test_trait_type_detection():
    assert detect_trait_type(pd.Series([0,1,1,0,1])) == "binary"
    assert detect_trait_type(pd.Series([1.2,3.4,0.1,9.9,2.2])) == "continuous"
    assert detect_trait_type(pd.Series([0,1,2,1,2,0,1])) == "categorical"

def test_load_inputs_in_memory_and_alignment(tmp_path):
    # in-memory phenotype Series + genotype array with matching sample ids
    ids = [f"s{i}" for i in range(20)]
    y = pd.Series(np.random.default_rng(0).normal(size=20), index=ids, name="yield")
    G = np.random.default_rng(1).integers(0,3,(20,50)).astype(float)  # (n, m)
    inp = load_inputs(phenotype=y, genotype=G)
    assert inp.n_samples == 20 and inp.trait_name == "yield"
    assert inp.trait_type == "continuous"

def test_friendly_error_on_no_shared_samples():
    y = pd.Series([1.0,2.0,3.0], index=["a","b","c"], name="t")
    G = pd.DataFrame(np.zeros((2,10)), index=["x","y"])   # disjoint ids
    with pytest.raises(ValueError) as e:
        load_inputs(phenotype=y, genotype=G)
    assert "shared" in str(e.value).lower()   # actionable, not a KeyError/traceback
```

- [ ] **Step 2: Run test** → FAIL (module missing).

- [ ] **Step 3: Write minimal implementation** — implement `detect_trait_type` (binary if 2 distinct non-NA; categorical if all-integer with ≤ ~10 distinct; else continuous) and `load_inputs`:
  - Accept phenotype as path (delegate to existing readers) | `pd.Series`/`pd.DataFrame` (pick `trait` col or first) | array (needs ids or positional).
  - Accept genotype as path (return a reader/path for the scan layer) | `np.ndarray`/`pd.DataFrame` (wrap via a tiny in-memory reader adapter — reuse the existing reader interface; if none trivially exists, write a minimal `ArrayReader` exposing the same `iter_chunks`/metadata the scan layer needs).
  - Align samples by id intersection (reuse the existing alignment path if importable; else intersect indices). If the intersection is empty or tiny, raise `ValueError` naming both counts + hint. Wrong-shape/length → friendly `ValueError`.
  - Return `GwasInputs`.

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: input loading (in-memory + path), trait-type detection, friendly errors"`

---

### Task 3: Enrich the result object → `GwasResult` (`_results.py`)

**Files:** Modify `torchgenomics/api/_results.py`; Test `tests/test_api_gwas.py` (result-shape tests).

**Interfaces:** `ScanRun` gains `trait_type`, `.hits`, `.diagnostics`, richer `.summary()`, `.report(dir)`. Export alias `GwasResult = ScanRun`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_gwas.py
import pandas as pd
from torchgenomics.api import GwasResult

def test_gwasresult_shape_and_summary_and_report(tmp_path):
    r = GwasResult(model="SingleTraitLMM", test="wald", correction="bh",
                   n_variants=1000, n_significant=2, lambda_gc=0.99, n_samples=300,
                   top_hits=pd.DataFrame({"SNP":["s1"],"CHR":[1],"BP":[10],"P":[1e-9]}))
    r.trait_type = "continuous"
    assert isinstance(r.hits, pd.DataFrame) and not r.hits.empty     # .hits alias
    s = r.summary()
    assert "continuous" in s and "λ_GC" in s and "0.99" in s
    assert "well-calibrated" in s.lower() or "calibrat" in s.lower() # plain-language λ interpretation
    assert "Next" in s                                              # next-step suggestions
    d = r.diagnostics
    assert d["lambda_gc"] == 0.99 and d["n_significant"] == 2
    # report writes a folder (summary.txt at minimum; plots best-effort)
    out = r.report(tmp_path / "rep")
    assert (tmp_path / "rep" / "summary.txt").exists()
```

- [ ] **Step 2: Run test** → FAIL (`GwasResult` export / `.hits`/`.diagnostics`/`.report` missing).

- [ ] **Step 3: Write minimal implementation** — add to `ScanRun`: a `trait_type: str = ""` field; `@property hits` returning `self.top_hits`; `@property diagnostics` returning a dict (`lambda_gc`, `n_significant`, `n_variants`, `n_samples`, `model`, `trait_type`, `warnings` if present); rewrite `summary()` to the plain-language block (trait+type, n, model, correction, λ_GC + interpretation `≤1.05 ✓ well-calibrated` / `>1.10 ⚠ inflated — consider more PCs`, #significant + top locus, warnings, a `Next:` line); add `report(dir)` writing `summary.txt` + (best-effort, wrapped in try/except so a headless env can't crash it) `manhattan.png`/`qq.png` via the existing `.manhattan()`/`.qq()`, plus `results.tsv` if `output_files` has one (copy/reference). Export `GwasResult = ScanRun` from `_results.py` and `api/__init__.py`.

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: enrich ScanRun -> GwasResult (.hits/.diagnostics/plain summary/.report)"`

---

### Task 4: Model dispatch — the shared core (`_dispatch.py`)

**Files:** Create `torchgenomics/api/_dispatch.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** `run_model(alias, inputs, opts) -> GwasResult` (see contract). Runs the resolved model: `api_lmm`/`api_glm` delegate to the existing `lmm_scan`/`glm_scan`; `lowlevel:*` build the model + `UnifiedScanner`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_model_lmm_and_a_lowlevel_model(tmp_path):
    # reuse an existing tiny GWAS fixture if the repo has one; else synth a small BED
    from torchgenomics.api._inputs import load_inputs
    from torchgenomics.api._dispatch import run_model, RunOptions
    import numpy as np, pandas as pd
    ids=[f"s{i}" for i in range(150)]
    G=np.random.default_rng(0).integers(0,3,(150,300)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=150), index=ids, name="y")
    inp=load_inputs(phenotype=y, genotype=G)
    r=run_model("lmm", inp, RunOptions(kinship="auto", pcs=0, correction="bh", verbose=False))
    from torchgenomics.api import GwasResult
    assert isinstance(r, GwasResult) and r.n_variants==300
    # a low-level model also returns the same GwasResult shape
    r2=run_model("blink", inp, RunOptions(kinship=False, pcs=0, verbose=False))
    assert isinstance(r2, GwasResult) and r2.n_variants>0
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Write minimal implementation** — `run_model`:
  - `spec = resolve_model(alias)`.
  - Build kinship (auto → `linalg.kinship.grm_vanraden` unless supplied/`False`) and PCs (auto → eigendecompose the GRM top-N unless supplied/0) ONCE, reused across models in a comparison (dispatch receives them via `opts` or computes+caches).
  - `runner == "api_lmm"` → call `lmm_scan(...)` with the loaded/aligned data written to a temp or passed as arrays (if `lmm_scan` requires paths, materialize a temp BED/npy via the existing writers — or, cleaner, add an internal `_scan_arrays()` the api scans already use; verify and reuse). Return its `ScanRun`, tagging `trait_type`.
  - `runner == "api_glm"` → `glm_scan(...)`.
  - `runner.startswith("lowlevel:")` → import the class from `torchgenomics.models`, `fit_null(Y, X0, K)`, run `scan.UnifiedScanner(reader, model).scan(null)`, and **wrap the result into a `GwasResult`** (map n_variants/top_hits/lambda_gc/correction) via a small `_scanresult_to_gwasresult` helper. Apply `correction` via `stats`.
  - Everything returns a `GwasResult` with consistent fields.
  - (If wiring a given low-level model is non-trivial, the test only exercises lmm + blink; other registry models can be wired incrementally but MUST at least dispatch + raise a clear "not yet wired" for unimplemented ones — no silent wrong result.)

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: _dispatch.run_model shared core (api + low-level models -> GwasResult)"`

---

### Task 5: `tg.gwas` orchestrator — single + multi-model + `GwasComparison` (`gwas.py`)

**Files:** Create `torchgenomics/api/gwas.py`; Modify `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py`; Test `tests/test_api_gwas.py`.

**Interfaces:** `gwas(...)`, `models()`, `GwasComparison`. Exported at top level (`tg.gwas`, `tg.models`).

- [ ] **Step 1: Write the failing test**

```python
def test_tg_gwas_single_auto_and_multimodel(capsys):
    import numpy as np, pandas as pd
    import torchgenomics as tg
    ids=[f"s{i}" for i in range(150)]
    G=np.random.default_rng(0).integers(0,3,(150,300)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=150), index=ids, name="yield")
    # bare call: auto model, prints decisions, returns one GwasResult
    r=tg.gwas(y, G, kinship="auto", pcs=0)
    assert isinstance(r, tg.GwasResult)
    out=capsys.readouterr().out
    assert "yield" in out and ("lmm" in out.lower() or "SingleTraitLMM" in out)  # decision log
    # user picks multiple models -> GwasComparison
    cmp=tg.gwas(y, G, models=["lmm","blink"], kinship="auto", pcs=0, verbose=False)
    assert isinstance(cmp, tg.GwasComparison)
    assert set(cmp.results) == {"lmm","blink"}
    assert "lmm" in cmp.summary() and "blink" in cmp.summary()
    # tg.models() lists the registry
    assert "lmm" in set(tg.models()["alias"])
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Write minimal implementation** — `gwas()`:
  1. `inputs = load_inputs(phenotype, genotype, covariates, trait, trait_type)`.
  2. Resolve `models`: `"auto"` → the §4b tree over `inputs.trait_type`+kinship (a small `_auto_model(inputs, kinship)` helper); a string → `[that]`; a list → itself. Validate each via `resolve_model`.
  3. Build `RunOptions` (kinship/pcs/qc/correction/device/output/verbose).
  4. If `verbose`: print the decision log (trait+type, n, chosen model(s)+rationale, #PCs, correction).
  5. For each model alias: `run_model(alias, inputs, opts)`.
  6. Single → return the `GwasResult` (and `.report(output)` if `output`); list → build `GwasComparison(results={alias: res})` with `.summary()` (per-model λ_GC + #hits table) and `.report(dir)` (per-model subfolders + overlap table).
  Friendly errors bubble from `load_inputs`/`resolve_model`. Export `gwas`, `models`, `GwasResult`, `GwasComparison` from `api/__init__.py` and re-export at `torchgenomics/__init__.py`.

- [ ] **Step 4: Run test** → PASS. Also run `pytest tests/test_api_gwas.py tests/test_api_registry.py tests/test_api_inputs.py -v`.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: tg.gwas orchestrator (auto+user models, multi-model GwasComparison, tg.models)"`

---

### Task 6: `tg.recommend` (dry-run) + warnings

**Files:** Modify `torchgenomics/api/gwas.py` (+ `_inputs.py`/`_dispatch.py` for warnings); Test `tests/test_api_gwas.py`.

**Interfaces:** `recommend(...) -> Recommendation`; warnings surfaced in `GwasResult.diagnostics["warnings"]` + `.summary()`.

- [ ] **Step 1: Write the failing test**

```python
def test_recommend_dry_run_and_warnings():
    import numpy as np, pandas as pd, torchgenomics as tg
    ids=[f"s{i}" for i in range(120)]
    yb=pd.Series(np.r_[np.ones(5), np.zeros(115)], index=ids, name="disease")  # imbalanced binary
    G=np.random.default_rng(0).integers(0,3,(120,200)).astype(float)
    rec=tg.recommend(yb, G)
    assert rec.trait_type=="binary"
    assert rec.suggested_model in {"glmm","glm"}
    assert "binary" in rec.explain().lower()          # human-readable plan, nothing run
    # a run on the imbalanced binary emits a case/control-imbalance warning
    r=tg.gwas(yb, G, models="glm", kinship=False, pcs=0, verbose=False)
    warns=" ".join(r.diagnostics.get("warnings", []))
    assert "imbalance" in warns.lower() or "case" in warns.lower()
```

- [ ] **Step 2: Run test** → FAIL.

- [ ] **Step 3: Write minimal implementation** — `recommend()` runs `load_inputs` + `_auto_model` + the QC/PC plan but STOPS before scanning; returns a `Recommendation` dataclass (`trait_type`, `suggested_model`, `n_pcs`, `qc_summary`, `.explain()` string). `_warnings(inputs, result)` computes a small set — genomic inflation (λ_GC>1.10), low post-QC variant count, case/control imbalance (min class < ~5%), high mean relatedness, many-missing phenotype — each a one-sentence action; attach to `result.diagnostics["warnings"]` (and shown in `.summary()`).

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: tg.recommend dry-run + high-value warnings"`

---

### Task 7: Consistency audit — every `api` scan returns `GwasResult`

**Files:** Modify `torchgenomics/api/scans.py` (+ any other `api` scan modules); Test `tests/test_api_gwas.py`.

**Interfaces:** existing `lmm_scan`/`glm_scan` already return `ScanRun` (= `GwasResult`); ensure they expose `trait_type` and the enriched result. Add thin `api` wrappers for the registry models that lack one (or confirm `run_model` is the single public path and document that per-model `api.*_scan` remain for lmm/glm only).

- [ ] **Step 1: Write the failing test**

```python
def test_existing_scans_return_gwasresult():
    import numpy as np, pandas as pd, torchgenomics as tg
    from torchgenomics.api import GwasResult
    # lmm_scan / glm_scan return the enriched GwasResult with .hits/.summary/.diagnostics
    ids=[f"s{i}" for i in range(120)]
    G=np.random.default_rng(0).integers(0,3,(120,200)).astype(float)
    y=pd.Series(np.random.default_rng(1).normal(size=120), index=ids, name="y")
    r=tg.gwas(y, G, models="lmm", kinship="auto", pcs=0, verbose=False)
    assert isinstance(r, GwasResult)
    assert hasattr(r,"hits") and hasattr(r,"diagnostics") and callable(r.summary)
```

- [ ] **Step 2: Run test** → FAIL if any gap; else confirm.

- [ ] **Step 3: Write minimal implementation** — ensure `run_model`'s api-delegating paths tag `trait_type` and return the enriched `GwasResult`; if `lmm_scan`/`glm_scan` need a small tweak to set `trait_type`, do it (additive). Document that `run_model` is the unified scan path and the registry is the model surface. Fix any scan that returns a divergent object.

- [ ] **Step 4: Run test** → PASS; run the full `tests/test_api_gwas.py`.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: consistency — enriched GwasResult from every api scan path"`

---

### Task 8: CLI `gwas` / `recommend` / `models` (share the core; `pipeline` alias)

**Files:** Modify `torchgenomics/cli.py`; Test `tests/test_cli_gwas.py` (or the existing cli-smoke pattern).

**Interfaces:** `torchgenomics gwas --phenotype … --genotype … --models lmm,farmcpu --kinship auto --correction bh --output DIR`; `recommend`; `models`. All call `api.gwas`/`recommend`/`models`. Existing `pipeline` routes to the same core.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_gwas.py
import subprocess, sys
def test_cli_models_lists_registry():
    out = subprocess.run([sys.executable,"-m","torchgenomics","models"],
                         capture_output=True, text=True)
    assert out.returncode==0 and "lmm" in out.stdout and "farmcpu" in out.stdout
def test_cli_gwas_help_has_models_flag():
    out = subprocess.run([sys.executable,"-m","torchgenomics","gwas","--help"],
                         capture_output=True, text=True)
    assert out.returncode==0 and "--models" in out.stdout and "--phenotype" in out.stdout
```

- [ ] **Step 2: Run test** → FAIL (subcommands missing).

- [ ] **Step 3: Write minimal implementation** — add `_add_gwas_parser`/`_cmd_gwas`, `_add_recommend_parser`/`_cmd_recommend`, `_add_models_parser`/`_cmd_models` mirroring the existing subcommand pattern (see `_add_pipeline_parser`/`_cmd_pipeline`). `--models` takes a comma-list. `_cmd_gwas` parses flags → calls `api.gwas(...)` with `output` defaulted → prints the summary. Point the existing `pipeline` command's core at the same `api.gwas` (keep its flags working; it becomes a thin alias). Register in the subparser + dispatch dicts. Bump CLI count note in `CLAUDE.md`.

- [ ] **Step 4: Run test** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: CLI gwas/recommend/models (shared core; pipeline alias)"`

---

### Task 9: R wrappers `tg_gwas` / `tg_recommend` / `tg_models`

**Files:** Create `rTorchGenomics/R/gwas.R`; Modify `rTorchGenomics/NAMESPACE`, `rTorchGenomics/_pkgdown.yml`; Test `rTorchGenomics/tests/testthat/test-gwas.R`.

**Interfaces:** `tg_gwas(phenotype, genotype, models=…, …)` → S4 `GwasResult` mirroring Python attrs (`top_hits`, `summary()`, `manhattan()`, `report()`); `tg_recommend`, `tg_models`. Follows the existing `bridge_call`/`tg_*` pattern.

- [ ] **Step 1: Write the failing test** (testthat) — `tg_models()` returns a data.frame containing `lmm`; `tg_gwas(...)` on a tiny in-memory example returns an object whose `top_hits` is a data.frame and `summary()` prints trait + model. (Mirror an existing `test-*.R` in the R pkg for structure.)

- [ ] **Step 2: Run test** — `R CMD ... testthat` → FAIL (functions missing).

- [ ] **Step 3: Write minimal implementation** — author `gwas.R` via the `bridge_call`/`tg_py` pattern used by `tg_lmm_scan` (read `rTorchGenomics/R/` for the exact idiom): `tg_gwas` calls Python `torchgenomics.gwas` through the reticulate bridge and wraps the result into the existing S4 result class (or a new `GwasResult` S4 mirroring `ScanRun`). Add `tg_gwas`/`tg_recommend`/`tg_models` to `NAMESPACE` (`export(...)`) and `_pkgdown.yml` reference index (the "GWAS scans" section). Author files via bash heredoc / `Rscript writeLines` (subagent Write is unreliable for this repo).

- [ ] **Step 4: Run test** → PASS (or documented skip if the R reticulate env isn't available in CI; at minimum `R CMD check`-clean parsing).
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: R wrappers tg_gwas/tg_recommend/tg_models + NAMESPACE/pkgdown"`

---

### Task 10: Docs + top-level exports + quickstart

**Files:** Modify `torchgenomics/__init__.py` (`__all__`), `CLAUDE.md` (new API section + CLI count), add a `docs/` quickstart snippet; Test: `tests/test_api_gwas.py::test_toplevel_exports`.

- [ ] **Step 1: Write the failing test**

```python
def test_toplevel_exports():
    import torchgenomics as tg
    for name in ("gwas","recommend","models","GwasResult","GwasComparison"):
        assert hasattr(tg, name), name
```

- [ ] **Step 2: Run test** → FAIL if any missing.
- [ ] **Step 3: Write minimal implementation** — add the five names to `torchgenomics/__init__.py` imports + `__all__`; add a "Getting started" block to `CLAUDE.md` (one `tg.gwas` example + `tg.models()`); bump the CLI subcommand count note.
- [ ] **Step 4: Run test** → PASS; run `TORCHGENOMICS_DISABLE_NATIVE=1 pytest tests/test_api_gwas.py tests/test_api_registry.py tests/test_api_inputs.py tests/test_cli_gwas.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "friendly-api: top-level exports + CLAUDE.md getting-started + CLI count"`

---

## Self-Review

**Spec coverage:**
- §3a `tg.gwas` (auto trait-type/QC/kinship/PCA, user model(s), friendly errors, decision log) → Tasks 2,4,5. ✅
- §3b enriched `GwasResult` (.hits/.diagnostics/summary/report) + `GwasComparison` → Tasks 3,5. ✅
- §3c `recommend` + warnings → Task 6. ✅
- §4 registry (aliases, GAPIT synonyms, `tg.models`) + auto fallback → Tasks 1,5. ✅
- §4 wire the WHOLE registry through the friendly surface → Tasks 1,4 (`run_model` low-level path). ✅
- §5 cross-surface parity (Python/R/CLI shared core) → Tasks 5,8,9. ✅ consistency audit → Task 7. ✅
- §6 backward compat (additive; low-level untouched) → held by construction (new modules; scans only additively tagged). ✅
- Reconcile with existing CLI `pipeline` → Task 8. ✅

**Known verification points for the implementer:** (a) whether `lmm_scan`/`glm_scan` can accept in-memory arrays or need a temp-file/`_scan_arrays` path — confirm and reuse, don't duplicate; (b) the existing sample-alignment utility to reuse in `load_inputs`; (c) the low-level `UnifiedScanner` + reader interface an `ArrayReader` must satisfy; (d) the exact `bridge_call` idiom in `rTorchGenomics/R`; (e) the existing `pipeline` model-selection logic to reuse for `_auto_model`.

**Placeholder scan:** the low-level dispatch (Task 4 Step 3) and R bridge (Task 9 Step 3) reference existing idioms to reuse rather than pasting them, because they must match current signatures the implementer verifies at those file:lines — these are verification steps, not placeholders. All Tier-1 tests carry runnable code + concrete assertions.

**Deferred (per spec §7):** `preset=` bundles, HTML reports, ML model suggestion, LLM/MCP, biobank-scale orchestrator loads.
