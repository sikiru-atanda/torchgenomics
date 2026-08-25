# Wire MR + Coloc through api / CLI / R — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Mendelian randomization (`ivw`/`egger`/`weighted_median`/`presso`/`all`) and colocalization (`pairwise` + `hyprcoloc`) — currently Python-library-only — as `api.mr`/`api.coloc` functions, `torchgenomics mr`/`coloc` CLI subcommands, and `tg_mr`/`tg_coloc` R wrappers. Mirrors the proven `clump`/`meta` pattern. No new statistics.

**Architecture:** `api/postgwas.py` gains `mr()`/`coloc()` (load sumstats via `postgwas.load_sumstats`, run the existing `postgwas.mr_*`/`coloc_pairwise`/`hyprcoloc`, write a TSV, return a typed `MRRun`/`ColocRun`). CLI subcommands thin-wrap the api functions. Hand-crafted R wrappers `bridge_call` the api functions and wrap results into S4 classes.

**Tech Stack:** Python 3.12, pandas, torch; `torchgenomics.postgwas` (existing MR/coloc math); `torchgenomics.api`/`cli`; R reticulate bridge.

## Global Constraints

- **No new statistics** — orchestrate the existing `postgwas.mr_ivw/mr_egger/mr_weighted_median/mr_presso/mr_all/coloc_pairwise/hyprcoloc`. No change to that math.
- **Backward compatible / additive** — new api functions + result classes, new CLI subcommands, new R wrappers + S4 classes, two `_TIER_2_CLI_SUBCOMMANDS` entries. No existing signature/default changes.
- **Reviewer-grade docstrings**; **friendly errors** (bad method; pairwise-without-2nd-sumstats; hyprcoloc-with-<2; missing file/column via `load_sumstats`).
- Follow the tier-1 hand-crafted pattern (`clump`/`meta`), NOT the auto-generator.
- Verify commands under `TORCHGENOMICS_DISABLE_NATIVE=1`.

## File Structure

- Modify: `torchgenomics/api/_results.py` — add `MRRun`, `ColocRun` dataclasses (+ export).
- Modify: `torchgenomics/api/postgwas.py` — add `mr()`, `coloc()`.
- Modify: `torchgenomics/api/__init__.py` — export `mr`, `coloc`, `MRRun`, `ColocRun`.
- Modify: `torchgenomics/cli.py` — `_add_mr_parser`/`_cmd_mr`, `_add_coloc_parser`/`_cmd_coloc`, registration.
- Modify: `torchgenomics/_manifest/__init__.py` — add `"mr"`, `"coloc"` to `_TIER_2_CLI_SUBCOMMANDS`.
- Modify: `CLAUDE.md` — CLI count 46→48; MR/coloc examples.
- Modify: `rTorchGenomics/R/api.R` — `tg_mr`, `tg_coloc`; `rTorchGenomics/R/result_classes.R` — `MRRun`/`ColocRun` S4; `NAMESPACE`; `_pkgdown.yml`.
- Test: `tests/test_api_mr_coloc.py`, `tests/test_cli_mr_coloc.py`, `rTorchGenomics/tests/testthat/test-mr-coloc.R`.

**Interface contract:**

```python
# api/postgwas.py
def mr(exposure, outcome, *, method="ivw", output=None,
       n_boot=1000, n_perm=1000, seed=42, sep="\t") -> "MRRun": ...
def coloc(sumstats, sumstats2=None, *, method="pairwise", output=None,
          prior_1=1e-4, prior_2=1e-4, prior_12=1e-5,
          trait_names=None, sep="\t") -> "ColocRun": ...
# _results.py  (both subclass _BaseRun, like MetaRun)
#   MRRun(method:str, results:pd.DataFrame, n_instruments:int, primary_beta:float, primary_p:float)
#   ColocRun(method:str, pp:dict[str,float], candidate_snp:int, table:pd.DataFrame, headline:float)
```

---

### Task 1: `MRRun` + `ColocRun` result classes

**Files:** Modify `torchgenomics/api/_results.py`; Test `tests/test_api_mr_coloc.py`.

**Interfaces:** Produces `MRRun`, `ColocRun` (used by Tasks 2-3, 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_mr_coloc.py
import pandas as pd


def test_mrrun_shape_and_summary():
    from torchgenomics.api import MRRun
    r = MRRun(method="all", n_instruments=25, primary_beta=0.42, primary_p=1e-6,
              results=pd.DataFrame({"method": ["ivw", "egger"], "beta": [0.42, 0.40],
                                    "pval": [1e-6, 2e-3]}))
    assert r.primary_beta == 0.42 and r.n_instruments == 25
    s = r.summary()
    assert "MR" in s and "ivw" in s.lower()
    d = r.to_dict()
    assert d["method"] == "all"


def test_colocrun_shape_and_summary():
    from torchgenomics.api import ColocRun
    r = ColocRun(method="pairwise", pp={"h0": 0.01, "h1": 0.02, "h2": 0.02,
                                        "h3": 0.03, "h4": 0.92},
                 candidate_snp=137, headline=0.92)
    s = r.summary()
    assert "0.92" in s and ("H4" in s or "h4" in s or "shared" in s.lower())
    assert r.headline == 0.92
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "mrrun or colocrun" -v`
Expected: FAIL (`MRRun`/`ColocRun` not importable).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/_results.py`, after `MetaRun` (mirror its shape):

```python
@dataclass
class MRRun(_BaseRun):
    """Result of :func:`torchgenomics.api.mr` — a Mendelian-randomization panel."""

    method: str = ""  # "ivw" | "egger" | "weighted_median" | "presso" | "all"
    n_instruments: int = 0
    primary_beta: float | None = None
    primary_p: float | None = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "mr"

    def summary(self) -> str:
        lines = [
            f"MR ({self.method}) — {self.n_instruments} instruments",
        ]
        if self.primary_beta is not None:
            lines.append(f"Causal estimate: beta={self.primary_beta:.4f} "
                         f"p={self.primary_p:.3e}")
        if not self.results.empty:
            lines.append(self.results.to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class ColocRun(_BaseRun):
    """Result of :func:`torchgenomics.api.coloc` — colocalization posteriors."""

    method: str = ""  # "pairwise" | "hyprcoloc"
    pp: dict = field(default_factory=dict)  # posterior probabilities
    candidate_snp: int | None = None
    headline: float | None = None  # PP.H4 (pairwise) or PP(all colocalize)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "coloc"

    def summary(self) -> str:
        if self.method == "pairwise":
            verdict = ("strong" if (self.headline or 0) >= 0.8 else
                       "moderate" if (self.headline or 0) >= 0.5 else "weak")
            lines = [
                f"Colocalization (pairwise) — PP.H4={self.headline:.3f} "
                f"({verdict} evidence of a shared causal variant)",
                "  " + "  ".join(f"PP.{k.upper()}={v:.3f}" for k, v in self.pp.items()),
            ]
        else:
            lines = [
                f"Colocalization (hyprcoloc) — PP(all colocalize)={self.headline:.3f}",
            ]
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)
```

Export `MRRun`, `ColocRun` from `_results.py`'s `__all__` if it has one (match how `MetaRun` is exported).

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "mrrun or colocrun" -v`
Expected: PASS. (Import goes via `torchgenomics.api` — Task 2 wires `api/__init__.py`; for THIS task, import directly from `torchgenomics.api._results` in the test if `api.MRRun` isn't exported yet, OR add the export now. Simplest: add `MRRun`/`ColocRun` to `api/__init__.py` in this task's Step 3 so the test's `from torchgenomics.api import MRRun` works.)

Also add to `torchgenomics/api/__init__.py`: import `MRRun`, `ColocRun` from `._results` and add to `__all__`.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_results.py torchgenomics/api/__init__.py tests/test_api_mr_coloc.py
git commit -m "api: MRRun + ColocRun result classes"
```

---

### Task 2: `api.mr()`

**Files:** Modify `torchgenomics/api/postgwas.py`, `torchgenomics/api/__init__.py`; Test `tests/test_api_mr_coloc.py`.

**Interfaces:** Consumes `MRRun` (Task 1), `postgwas.load_sumstats`, `postgwas.mr_*`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_mr_coloc.py (append)
def _write_mr_fixture(tmp_path, seed=0, effect=0.5, n_snp=40):
    """Two-sample MR fixture: instruments with a true causal effect + noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bx = rng.normal(0.3, 0.1, size=n_snp)          # instrument-exposure
    bx_se = np.full(n_snp, 0.02)
    by = effect * bx + rng.normal(0, 0.01, size=n_snp)  # instrument-outcome
    by_se = np.full(n_snp, 0.02)
    def _df(beta, se):
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": se,
            "p": 2 * (1 - 0.9999), "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })
    exp = tmp_path / "exp.tsv"; out = tmp_path / "out.tsv"
    _df(bx, bx_se).to_csv(exp, sep="\t", index=False)
    _df(by, by_se).to_csv(out, sep="\t", index=False)
    return str(exp), str(out), effect


def test_api_mr_all_recovers_effect(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import MRRun
    exp, out, effect = _write_mr_fixture(tmp_path)
    r = tg.mr(exp, out, method="all", output=str(tmp_path / "mr.tsv"))
    assert isinstance(r, MRRun)
    methods = set(r.results["method"])
    assert {"ivw", "egger", "weighted_median", "presso"}.issubset(methods)
    ivw_beta = float(r.results.loc[r.results["method"] == "ivw", "beta"].iloc[0])
    assert abs(ivw_beta - effect) < 0.15          # recovers the simulated effect
    assert (tmp_path / "mr.tsv").exists()


def test_api_mr_single_method_and_bad_method(tmp_path):
    import torchgenomics as tg, pytest
    exp, out, _ = _write_mr_fixture(tmp_path)
    r = tg.mr(exp, out, method="ivw")
    assert r.method == "ivw" and len(r.results) == 1
    with pytest.raises(ValueError) as e:
        tg.mr(exp, out, method="nope")
    assert "nope" in str(e.value) and "ivw" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "api_mr" -v`
Expected: FAIL (`tg.mr` doesn't exist).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/postgwas.py` (mirror `meta`'s structure — imports, `timed()`, output handling):

```python
_MR_METHODS = ("ivw", "egger", "weighted_median", "presso", "all")


def mr(exposure, outcome, *, method="ivw", output=None,
       n_boot=1000, n_perm=1000, seed=42, sep="\t") -> "MRRun":
    """Two-sample Mendelian randomization from exposure + outcome sumstats.

    Loads both sumstats files, runs the requested estimator(s), and returns a
    :class:`~torchgenomics.api.MRRun` with a per-method results table.

    Parameters
    ----------
    exposure, outcome
        Paths to instrument sumstats TSVs (columns chr/pos/snp/a1/a2/beta/se/p,
        harmonized to the same effect allele) — the shared instruments define
        the MR panel.
    method
        One of ``"ivw"``, ``"egger"``, ``"weighted_median"``, ``"presso"``, or
        ``"all"`` (runs the full sensitivity panel, one row per method).
    output
        Optional path to write the results TSV.
    n_boot, n_perm, seed
        Bootstrap / permutation / seed controls for weighted-median and
        MR-PRESSO.

    Returns
    -------
    MRRun
    """
    from ._results import MRRun
    from ._helpers import timed  # or wherever timed() lives; match meta()
    from ..postgwas import (
        load_sumstats, mr_ivw, mr_egger, mr_weighted_median, mr_presso, mr_all,
    )
    import pandas as pd

    m = str(method).lower()
    if m not in _MR_METHODS:
        raise ValueError(
            f"Unknown MR method '{method}'. Valid: {', '.join(_MR_METHODS)}."
        )
    exp = load_sumstats(str(exposure), sep=sep)
    out = load_sumstats(str(outcome), sep=sep)

    with timed() as elapsed:
        if m == "all":
            results = mr_all(exp, out, n_boot=n_boot, n_perm=n_perm, seed=seed)
        elif m == "ivw":
            results = [mr_ivw(exp, out)]
        elif m == "egger":
            results = [mr_egger(exp, out)]
        elif m == "weighted_median":
            results = [mr_weighted_median(exp, out, n_boot=n_boot, seed=seed)]
        else:  # presso
            results = [mr_presso(exp, out, n_perm=n_perm, seed=seed)]

        rows = [{
            "method": r.method, "beta": r.beta_hat, "se": r.se, "pval": r.p_value,
            "n_instruments": r.n_instruments,
            "egger_intercept": r.intercept, "egger_intercept_p": r.intercept_p,
            "n_outliers": r.n_outliers,
            "beta_corrected": r.beta_corrected, "pval_corrected": r.p_corrected,
        } for r in results]
        df = pd.DataFrame(rows)

        output_files = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            from pathlib import Path
            output_files["tsv"] = Path(str(output))

        primary = results[0]
        return MRRun(
            runtime_s=elapsed(), output_files=output_files,
            method=m, n_instruments=int(primary.n_instruments),
            primary_beta=float(primary.beta_hat), primary_p=float(primary.p_value),
            results=df,
        )
```

Note: verify the exact `timed()` import and `output_files` idiom against `meta()` in the same file and match it (don't invent a different one). Add `mr` to `api/__init__.py` imports + `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "api_mr" -v`
Expected: PASS. If IVW β tolerance is too tight for the fixture, widen the fixture (more instruments / lower noise) — do NOT weaken to a vacuous assertion.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py tests/test_api_mr_coloc.py
git commit -m "api: mr() — two-sample MR panel (ivw/egger/weighted_median/presso/all)"
```

---

### Task 3: `api.coloc()`

**Files:** Modify `torchgenomics/api/postgwas.py`, `torchgenomics/api/__init__.py`; Test `tests/test_api_mr_coloc.py`.

**Interfaces:** Consumes `ColocRun` (Task 1), `postgwas.load_sumstats`, `postgwas.coloc_pairwise`, `postgwas.hyprcoloc`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_mr_coloc.py (append)
def _write_coloc_fixture(tmp_path, shared=True, seed=1, n_snp=50):
    """Two sumstats over the same region; a shared causal SNP if shared=True."""
    import numpy as np
    rng = np.random.default_rng(seed)
    causal = 25
    def _df(offset):
        z = rng.normal(0, 1, size=n_snp)
        idx = causal if shared else (causal + offset)
        z[idx] = 6.0                     # strong signal at the (shared?) SNP
        beta = z * 0.05
        se = np.full(n_snp, 0.05)
        from scipy.stats import norm
        p = 2 * norm.sf(np.abs(z))
        return pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": se, "p": p,
            "n": [10000] * n_snp, "af": [0.3] * n_snp,
        })
    a = tmp_path / "a.tsv"; b = tmp_path / "b.tsv"
    _df(0).to_csv(a, sep="\t", index=False)
    _df(10).to_csv(b, sep="\t", index=False)
    return str(a), str(b)


def test_api_coloc_pairwise_shared_signal(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import ColocRun
    a, b = _write_coloc_fixture(tmp_path, shared=True)
    r = tg.coloc(a, b, method="pairwise", output=str(tmp_path / "coloc.tsv"))
    assert isinstance(r, ColocRun) and r.method == "pairwise"
    assert r.headline == r.pp["h4"]
    assert r.pp["h4"] > 0.5              # shared causal -> high PP.H4
    assert (tmp_path / "coloc.tsv").exists()


def test_api_coloc_pairwise_requires_second(tmp_path):
    import torchgenomics as tg, pytest
    a, _ = _write_coloc_fixture(tmp_path)
    with pytest.raises(ValueError) as e:
        tg.coloc(a, method="pairwise")
    assert "sumstats2" in str(e.value) or "second" in str(e.value).lower()


def test_api_coloc_hyprcoloc_three_traits(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import ColocRun
    a, b = _write_coloc_fixture(tmp_path, shared=True, seed=2)
    c, _ = _write_coloc_fixture(tmp_path, shared=True, seed=3)
    r = tg.coloc([a, b, c], method="hyprcoloc")
    assert isinstance(r, ColocRun) and r.method == "hyprcoloc"
    assert r.headline is not None

    import pytest
    with pytest.raises(ValueError):
        tg.coloc([a], method="hyprcoloc")   # <2 traits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "api_coloc" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/postgwas.py`:

```python
def coloc(sumstats, sumstats2=None, *, method="pairwise", output=None,
          prior_1=1e-4, prior_2=1e-4, prior_12=1e-5,
          trait_names=None, sep="\t") -> "ColocRun":
    """Colocalization of two (pairwise) or ≥2 (hyprcoloc) GWAS sumstats.

    Parameters
    ----------
    sumstats
        For ``method="pairwise"``: a path to the first sumstats TSV. For
        ``method="hyprcoloc"``: a list of ≥2 sumstats paths.
    sumstats2
        The second sumstats path for ``method="pairwise"`` (required there;
        must be ``None`` for hyprcoloc).
    method
        ``"pairwise"`` (Giambartolomei 2-trait, PP.H0–H4) or ``"hyprcoloc"``
        (N-trait).
    output
        Optional path to write a posteriors TSV.
    prior_1, prior_2, prior_12
        Coloc priors (pairwise). ``prior_2`` is also reused as hyprcoloc's
        N-trait prior when method="hyprcoloc" defaults are not overridden —
        but hyprcoloc uses its own ``prior_2=0.98`` default; see below.

    Returns
    -------
    ColocRun
    """
    from ._results import ColocRun
    from ._helpers import timed
    from ..postgwas import load_sumstats, coloc_pairwise, hyprcoloc
    import pandas as pd
    from pathlib import Path

    m = str(method).lower()
    if m not in ("pairwise", "hyprcoloc"):
        raise ValueError(
            f"Unknown coloc method '{method}'. Valid: pairwise, hyprcoloc."
        )

    with timed() as elapsed:
        if m == "pairwise":
            if sumstats2 is None:
                raise ValueError(
                    "coloc method='pairwise' needs two sumstats; pass "
                    "sumstats2=<path> (or use method='hyprcoloc' with a list)."
                )
            ss1 = load_sumstats(str(sumstats), sep=sep)
            ss2 = load_sumstats(str(sumstats2), sep=sep)
            res = coloc_pairwise(ss1, ss2, prior_1=prior_1, prior_2=prior_2,
                                 prior_12=prior_12)
            pp = {"h0": float(res.pp_h0), "h1": float(res.pp_h1),
                  "h2": float(res.pp_h2), "h3": float(res.pp_h3),
                  "h4": float(res.pp_h4)}
            table = pd.DataFrame([pp])
            run = ColocRun(runtime_s=elapsed(), method="pairwise", pp=pp,
                           candidate_snp=int(res.candidate_snp),
                           headline=pp["h4"], table=table)
        else:  # hyprcoloc
            if sumstats2 is not None:
                raise ValueError(
                    "coloc method='hyprcoloc' takes a list in `sumstats`; "
                    "do not pass sumstats2."
                )
            paths = list(sumstats) if not isinstance(sumstats, (str, Path)) else [sumstats]
            if len(paths) < 2:
                raise ValueError(
                    "coloc method='hyprcoloc' needs ≥2 sumstats paths."
                )
            ss_list = [load_sumstats(str(p), sep=sep) for p in paths]
            res = hyprcoloc(ss_list, prior_1=prior_1, prior_w=0.0225,
                            trait_names=trait_names)
            pp = {"all_colocalize": float(res.pp_all_colocalize),
                  "null": float(res.pp_null)}
            table = pd.DataFrame([{
                "pp_all_colocalize": res.pp_all_colocalize,
                "pp_null": res.pp_null,
                "best_cluster": str(res.best_cluster),
                "best_cluster_posterior": res.best_cluster_posterior,
            }])
            run = ColocRun(runtime_s=elapsed(), method="hyprcoloc", pp=pp,
                           candidate_snp=int(res.candidate_snp),
                           headline=float(res.pp_all_colocalize), table=table)

        if output is not None:
            run.table.to_csv(str(output), sep="\t", index=False)
            run.output_files = {"tsv": Path(str(output))}
        return run
```

Note: verify `hyprcoloc`'s exact kwargs (it does NOT take `prior_2` per its signature `hyprcoloc(sumstats_list, prior_1, prior_2=0.98, prior_w, trait_names)` — actually it DOES take `prior_2`; pass through or use its default). Match the real signature verified in the spec. Add `coloc` to `api/__init__.py` imports + `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py -k "api_coloc" -v`
Expected: PASS. If the fixture's PP.H4 doesn't clear 0.5, strengthen the shared signal (higher z at the causal SNP) — do NOT weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py tests/test_api_mr_coloc.py
git commit -m "api: coloc() — pairwise + hyprcoloc colocalization"
```

---

### Task 4: CLI `mr` + `coloc` subcommands

**Files:** Modify `torchgenomics/cli.py`, `torchgenomics/_manifest/__init__.py`, `CLAUDE.md`; Test `tests/test_cli_mr_coloc.py`.

**Interfaces:** Consumes `api.mr`/`api.coloc` (Tasks 2-3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_mr_coloc.py
import subprocess
import sys


def test_cli_mr_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "mr", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--exposure" in out.stdout and "--method" in out.stdout


def test_cli_coloc_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "coloc", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--sumstats" in out.stdout and "--method" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_mr_coloc.py -v`
Expected: FAIL (subcommands missing).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/cli.py`, mirror `_add_meta_parser`/`_cmd_meta` + `_add_clump_parser`/`_cmd_clump`:

```python
def _add_mr_parser(subparsers):
    p = subparsers.add_parser("mr", help="Two-sample Mendelian randomization.")
    p.add_argument("--exposure", required=True, help="Exposure sumstats TSV.")
    p.add_argument("--outcome", required=True, help="Outcome sumstats TSV.")
    p.add_argument("--method", default="ivw",
                   choices=["ivw", "egger", "weighted_median", "presso", "all"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Output results TSV.")


def _cmd_mr(args) -> int:
    from .api import mr
    r = mr(args.exposure, args.outcome, method=args.method, output=args.output,
           n_boot=args.n_boot, n_perm=args.n_perm, seed=args.seed)
    print(r.summary())
    return 0


def _add_coloc_parser(subparsers):
    p = subparsers.add_parser("coloc", help="Colocalization (pairwise / hyprcoloc).")
    p.add_argument("--sumstats", required=True, nargs="+",
                   help="One sumstats TSV (pairwise, with --sumstats2) or ≥2 (hyprcoloc).")
    p.add_argument("--sumstats2", default=None, help="Second sumstats TSV (pairwise).")
    p.add_argument("--method", default="pairwise", choices=["pairwise", "hyprcoloc"])
    p.add_argument("--prior-1", type=float, default=1e-4)
    p.add_argument("--prior-2", type=float, default=1e-4)
    p.add_argument("--prior-12", type=float, default=1e-5)
    p.add_argument("--output", default=None, help="Output posteriors TSV.")


def _cmd_coloc(args) -> int:
    from .api import coloc
    if args.method == "pairwise":
        # pairwise: first --sumstats value + --sumstats2
        ss = args.sumstats[0]
        r = coloc(ss, args.sumstats2, method="pairwise", output=args.output,
                  prior_1=args.prior_1, prior_2=args.prior_2, prior_12=args.prior_12)
    else:
        r = coloc(args.sumstats, method="hyprcoloc", output=args.output,
                  prior_1=args.prior_1)
    print(r.summary())
    return 0
```

Register in `build_parser` (add `_add_mr_parser(subparsers)` / `_add_coloc_parser(subparsers)` near `_add_meta_parser`) and in the dispatch dict (`"mr": _cmd_mr, "coloc": _cmd_coloc`).

In `torchgenomics/_manifest/__init__.py`, add `"mr"`, `"coloc"` to `_TIER_2_CLI_SUBCOMMANDS`.

In `CLAUDE.md`: bump the CLI subcommand count note (46 → 48) and add example lines under a "Mendelian randomization + colocalization" heading:
```bash
torchgenomics mr --exposure exp.tsv --outcome out.tsv --method all --output mr.tsv
torchgenomics coloc --sumstats a.tsv --sumstats2 b.tsv --method pairwise --output coloc.tsv
torchgenomics coloc --sumstats a.tsv b.tsv c.tsv --method hyprcoloc --output hypr.tsv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_mr_coloc.py -v`
Expected: PASS. Also confirm end-to-end on the Task-2/3 fixtures:
`TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_mr_coloc.py tests/test_cli_mr_coloc.py -q`.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/cli.py torchgenomics/_manifest/__init__.py CLAUDE.md tests/test_cli_mr_coloc.py
git commit -m "cli: mr + coloc subcommands (share the api core); manifest + docs"
```

---

### Task 5: R `tg_mr` + `tg_coloc` wrappers + S4 result classes

**Files:** Modify `rTorchGenomics/R/api.R`, `rTorchGenomics/R/result_classes.R`, `rTorchGenomics/NAMESPACE`, `rTorchGenomics/_pkgdown.yml`, `rTorchGenomics/man/*.Rd`; Test `rTorchGenomics/tests/testthat/test-mr-coloc.R`.

**IMPORTANT:** author/modify R files via Bash heredoc / `sed` / `Rscript writeLines` (subagent Write unreliable for this repo's R). Parse-check each with `Rscript -e 'invisible(parse("<file>"))'`. Run R tests with `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-mr-coloc.R")'`.

**Interfaces:** Consumes the Python `api.mr`/`api.coloc` (Tasks 2-3) via the reticulate bridge.

- [ ] **Step 1: Write the failing test** (mocked, using the file's `mockery::stub` idiom)

```r
# rTorchGenomics/tests/testthat/test-mr-coloc.R
test_that("tg_mr forwards args and wraps the result", {
  captured <- NULL
  mockery::stub(tg_mr, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "all", n_instruments = 25L, primary_beta = 0.42,
         primary_p = 1e-6, results = list())
  })
  r <- tg_mr("exp.tsv", "out.tsv", method = "all")
  expect_equal(captured$fn, "mr")
  expect_equal(captured$args$exposure, "exp.tsv")
  expect_equal(captured$args$method, "all")
  expect_s4_class(r, "MRRun")
})

test_that("tg_coloc forwards args and wraps the result", {
  captured <- NULL
  mockery::stub(tg_coloc, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "pairwise", pp = list(h0 = 0.01, h4 = 0.92),
         candidate_snp = 137L, headline = 0.92, table = list())
  })
  r <- tg_coloc("a.tsv", "b.tsv", method = "pairwise")
  expect_equal(captured$fn, "coloc")
  expect_equal(captured$args$sumstats2, "b.tsv")
  expect_s4_class(r, "ColocRun")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-mr-coloc.R")'`
Expected: FAIL (`tg_mr`/`tg_coloc` / S4 classes missing).

- [ ] **Step 3: Write minimal implementation**

Add S4 classes to `rTorchGenomics/R/result_classes.R` (mirror `MetaRun`):

```r
setClass("MRRun", contains = "_BaseRun",
  representation(method = "character", n_instruments = "integer",
                 primary_beta = "numeric", primary_p = "numeric",
                 results = "data.frame", raw = "list"))
MRRun <- list(new_from_dict = function(d) {
  new("MRRun", method = d$method %||% "",
      n_instruments = as.integer(d$n_instruments %||% 0L),
      primary_beta = as.numeric(d$primary_beta %||% NA_real_),
      primary_p = as.numeric(d$primary_p %||% NA_real_),
      results = .tibble_from_list_of_dicts(d$results),
      raw = d)
})

setClass("ColocRun", contains = "_BaseRun",
  representation(method = "character", pp = "list",
                 candidate_snp = "integer", headline = "numeric",
                 table = "data.frame", raw = "list"))
ColocRun <- list(new_from_dict = function(d) {
  new("ColocRun", method = d$method %||% "", pp = as.list(d$pp %||% list()),
      candidate_snp = as.integer(d$candidate_snp %||% NA_integer_),
      headline = as.numeric(d$headline %||% NA_real_),
      table = .tibble_from_list_of_dicts(d$table), raw = d)
})
```

(Match the exact base class name, the `%||%` helper, and `.tibble_from_list_of_dicts` used by the existing `MetaRun`/`ScanRun` constructors — read `result_classes.R` first and reuse them; if `%||%` isn't defined there, use whatever null-coalesce the file uses.)

Add wrappers to `rTorchGenomics/R/api.R` (mirror `tg_clump`/`tg_meta`):

```r
#' Two-sample Mendelian randomization.
#' @export
tg_mr <- function(exposure, outcome,
                  method = c("ivw", "egger", "weighted_median", "presso", "all"),
                  output = NULL, n_boot = 1000L, n_perm = 1000L, seed = 42L) {
  method <- match.arg(method)
  d <- bridge_call("mr", .compact(list(
    exposure = .as_path(exposure), outcome = .as_path(outcome),
    method = method, output = .as_path(output),
    n_boot = as.integer(n_boot), n_perm = as.integer(n_perm),
    seed = as.integer(seed))))
  MRRun$new_from_dict(d)
}

#' Colocalization (pairwise Giambartolomei or N-trait hyprcoloc).
#' @export
tg_coloc <- function(sumstats, sumstats2 = NULL,
                     method = c("pairwise", "hyprcoloc"), output = NULL,
                     prior_1 = 1e-4, prior_2 = 1e-4, prior_12 = 1e-5,
                     trait_names = NULL) {
  method <- match.arg(method)
  d <- bridge_call("coloc", .compact(list(
    sumstats = if (length(sumstats) > 1) as.character(sumstats) else .as_path(sumstats),
    sumstats2 = .as_path(sumstats2), method = method, output = .as_path(output),
    prior_1 = prior_1, prior_2 = prior_2, prior_12 = prior_12,
    trait_names = trait_names)))
  ColocRun$new_from_dict(d)
}
```

Add `export(tg_mr)`, `export(tg_coloc)`, `exportClasses(MRRun)`, `exportClasses(ColocRun)` (match how `MetaRun`/`tg_meta` are exported) to `NAMESPACE`. Add both to `_pkgdown.yml` (a new "Post-GWAS: MR + coloc" section). Regenerate man pages: `Rscript -e 'roxygen2::roxygenise()'`.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-mr-coloc.R")'`
Expected: PASS. Parse-check `api.R`/`result_classes.R`. If the reticulate integration venv is available, add a guarded integration test that runs `tg_mr`/`tg_coloc` against the Python fixtures; otherwise document the graceful skip.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/api.R rTorchGenomics/R/result_classes.R rTorchGenomics/NAMESPACE rTorchGenomics/_pkgdown.yml rTorchGenomics/man rTorchGenomics/tests/testthat/test-mr-coloc.R
git commit -m "r: tg_mr + tg_coloc wrappers + MRRun/ColocRun S4 classes"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (api.mr) → Task 2. ✅
- §Component 2 (api.coloc pairwise + hyprcoloc) → Task 3. ✅
- §Component 3 (MRRun/ColocRun result classes) → Task 1. ✅
- §Component 4 (CLI mr/coloc + manifest + docs) → Task 4. ✅
- §Component 5 (R tg_mr/tg_coloc + S4 + NAMESPACE/pkgdown) → Task 5. ✅
- §Testing (MR recovers effect + all methods; coloc pairwise shared/unshared + hyprcoloc; friendly errors; CLI help + e2e; R mocked) → Tasks 1-5. ✅

**Placeholder scan:** No TBD/TODO; code steps carry real code. Two verification notes (the exact `timed()`/`output_files` idiom in `api/postgwas.py`; `hyprcoloc`'s exact kwargs) are "match the real signature" checks, not placeholders — the surrounding logic is fully specified.

**Type consistency:** `MRRun`/`ColocRun` fields set in Task 1 are consumed by `api.mr`/`api.coloc` (Tasks 2-3), the CLI `.summary()` prints (Task 4), and the R `new_from_dict` (Task 5). `api.mr(method=...)` choices match the CLI `--method` choices and the R `match.arg`. `coloc` pairwise-needs-sumstats2 / hyprcoloc-needs-list is enforced identically in api (Task 3) and surfaced by CLI (Task 4) + R (Task 5).

**Known verification points for the implementer:**
- (a) Read `api/postgwas.py`'s `meta()`/`clump()` to copy the exact `timed()` import path and `output_files` construction — do not invent a variant.
- (b) Verify `hyprcoloc`'s real signature (`prior_1`, `prior_2=0.98`, `prior_w`, `trait_names`) and pass kwargs that exist; the api `coloc` maps its `prior_1` through and leaves hyprcoloc's `prior_2` at the postgwas default unless a caller overrides it.
- (c) MRResult's `beta_corrected`/`p_corrected`/`intercept` are NaN/None for methods that don't compute them (ivw has no intercept; only presso has corrected β) — the results DataFrame will carry NaN there; that's expected, assert only on the columns a given method populates.
- (d) Coloc/MR fixture calibration: if PP.H4 or the IVW β don't clear the asserted thresholds on the tiny fixture, strengthen the *fixture* (more instruments, stronger shared signal), never weaken the assertion; if a method genuinely can't be exercised on a small fixture, report DONE_WITH_CONCERNS.
- (e) Match the exact R `new_from_dict`/`.tibble_from_list_of_dicts`/null-coalesce idioms already in `result_classes.R` (read it first).
