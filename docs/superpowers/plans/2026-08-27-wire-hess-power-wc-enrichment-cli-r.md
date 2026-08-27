# Wire HESS / Power / Winner's-Curse / Gene-Set Enrichment through api/CLI/R — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose four post-GWAS functions (HESS local h²/rg, GWAS power, winner's-curse correction, MAGMA-style gene-set enrichment) as first-class `tg.*` / CLI / R functions, thin-wrapping the existing validated `torchgenomics.postgwas` functions.

**Architecture:** Mirror the shipped `api.smr` / `api.mr_mega` pattern exactly — a real `api.<fn>` in `torchgenomics/api/postgwas.py` (added to `api.__all__` and top-level `torchgenomics`), a thin CLI subcommand, and a hand-crafted R wrapper + S4 result class. Load inputs via `postgwas.load_sumstats` + small file loaders, call the postgwas function inside `with timed() as elapsed:`, flatten to a `pd.DataFrame`, optional TSV write, return a `_results` dataclass.

**Tech Stack:** Python (torch, pandas, numpy), argparse CLI, R (S4 + reticulate bridge, testthat/mockery).

## Global Constraints

- **No new statistics.** Delegate all math to `postgwas` (`hess_local_h2`, `hess_local_rg`, `gwas_power`, `required_n`, `power_curve`, `correct_winners_curse`, `snp_to_gene`, `gene_set_enrichment`). Only I/O glue is new: file loaders, bp→SNP-index region mapping, GMT parser. Do NOT edit any file under `torchgenomics/postgwas/`.
- **Additive / backward-compatible only.** No existing signature or default changes.
- **Not MCP tools.** No `@tool` decorator on the four new api functions (consistent with `smr`/`mr_mega`). MCP tool count stays 14.
- **Precedence.** Add the four names to `_TIER_2_CLI_SUBCOMMANDS` AND define real `api.*` functions; the real module attributes win over `api.__getattr__`.
- **Winner's-curse kwargs (mandatory conditional):** `conditional_likelihood`/`fiqt` take NO `**kwargs` and raise `TypeError` on `n_boot`/`seed`. Forward those ONLY when `method=="bootstrap"`. `se_adjusted` is `None` for all three methods → that column is always NaN.
- **HESS `region_bounds` are end-EXCLUSIVE contiguous `(start,end)` slices into `z`;** the per-region LD sub-block is `ld[start:end, start:end]`. Guard: matched indices must be contiguous, else friendly `ValueError`. If NO region matches any SNP, raise a friendly `ValueError` before calling upstream. If N is unavailable (no `--n` and all-NaN `n` column), raise.
- **R codegen exclusion:** every hand-crafted R wrapper's manifest name (underscored) must be in `.HANDCRAFTED_COMMANDS` (`rTorchGenomics/R/codegen.R:14`) or `generate_api_auto()` clobbers it. Add `power, winners_curse, gene_set_enrichment, hess` AND back-fill `smr, mr_mega`.
- **Verify under `TORCHGENOMICS_DISABLE_NATIVE=1`.** Single-trait only (`load_sumstats` yields 1-D `beta`).

---

## File Structure

- `torchgenomics/api/_results.py` — add `PowerRun`, `WinnersCurseRun`, `EnrichmentRun`, `HessRun` (Task 1).
- `torchgenomics/api/postgwas.py` — add `power`, `winners_curse` (Task 2), `gene_set_enrichment` + loaders (Task 3), `hess` + loaders (Task 4).
- `torchgenomics/api/__init__.py` — import + `__all__` the 4 classes and 4 functions.
- `torchgenomics/__init__.py` — top-level export the 4 functions + 4 classes.
- `torchgenomics/cli.py` — 4 `_add_*_parser` + 4 `_cmd_*` + registrations (Task 5).
- `torchgenomics/_manifest/__init__.py` — 4 `_TIER_2_CLI_SUBCOMMANDS` entries (Task 5).
- `CLAUDE.md` — CLI count 50→54 + 4 example lines (Task 5).
- `rTorchGenomics/R/api.R`, `R/result_classes.R`, `R/codegen.R`, `NAMESPACE`, `_pkgdown.yml`, `man/*.Rd`, `inst/rbridge_manifest.json`, `R/api_auto.R`, `tests/testthat/test-hess-power-wc-enrichment.R` (Task 6).
- Tests: `tests/test_api_hess_power_wc_enrichment.py`, `tests/test_cli_hess_power_wc_enrichment.py`.

---

## Task 1: Result classes

**Files:**
- Modify: `torchgenomics/api/_results.py` (after `MRMegaRun`, ~line 586)
- Modify: `torchgenomics/api/__init__.py` (imports + `__all__`)
- Test: `tests/test_api_hess_power_wc_enrichment.py` (new)

**Interfaces:**
- Produces: `PowerRun`, `WinnersCurseRun`, `EnrichmentRun`, `HessRun` (dataclasses subclassing `_BaseRun`), each with `results: pd.DataFrame`, scalar summary fields, `_kind`, `summary()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_hess_power_wc_enrichment.py
import pandas as pd


def test_result_classes_shape_and_summary():
    from torchgenomics.api import PowerRun, WinnersCurseRun, EnrichmentRun, HessRun
    p = PowerRun(alpha=5e-8, n=5000, target_power=0.8, n_variants=3, n_powered=1,
                 results=pd.DataFrame({"snp": ["a"], "power": [0.9]}))
    assert "Power" in p.summary() and p.to_dict()["n_powered"] == 1
    w = WinnersCurseRun(method="conditional_likelihood", n_corrected=2, n_variants=10,
                        results=pd.DataFrame({"snp": ["a"], "beta_adjusted": [0.1]}))
    assert "curse" in w.summary().lower() and w.to_dict()["n_corrected"] == 2
    e = EnrichmentRun(n_genes_total=100, n_gene_sets=5, n_significant=1,
                      results=pd.DataFrame({"gene_set_name": ["s1"], "p": [1e-3]}),
                      genes=pd.DataFrame({"gene_id": ["g1"], "p": [1e-4]}))
    assert "nrichment" in e.summary() and e.to_dict()["n_gene_sets"] == 5
    h = HessRun(mode="h2", h2_total=0.3, h2_total_se=0.05, n_regions=4, n_snps_total=400,
                results=pd.DataFrame({"region_id": ["chr1:1-2"], "h2_local": [0.1]}))
    assert "HESS" in h.summary() and h.to_dict()["n_regions"] == 4
```

- [ ] **Step 2: Run it, expect ImportError**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py::test_result_classes_shape_and_summary -q`
Expected: FAIL (cannot import PowerRun).

- [ ] **Step 3: Add the four dataclasses** to `_results.py` after `MRMegaRun` (mirror `SMRRun`):

```python
@dataclass
class PowerRun(_BaseRun):
    """Result of :func:`torchgenomics.api.power` — per-variant GWAS detection power."""

    alpha: float = 5e-8
    n: float = 0.0
    target_power: float = 0.8
    n_variants: int = 0
    n_powered: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    curve: pd.DataFrame | None = None

    _kind: ClassVar[str] = "power"

    def summary(self) -> str:
        lines = [
            f"Power — {self.n_variants} variants, {self.n_powered} at power>={self.target_power} "
            f"(alpha={self.alpha:.1e}, N={self.n:g})",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class WinnersCurseRun(_BaseRun):
    """Result of :func:`torchgenomics.api.winners_curse` — effect-size de-biasing."""

    method: str = ""
    n_corrected: int = 0
    n_variants: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "winners_curse"

    def summary(self) -> str:
        lines = [
            f"Winner's-curse ({self.method}) — {self.n_corrected} of {self.n_variants} "
            f"variants corrected",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class EnrichmentRun(_BaseRun):
    """Result of :func:`torchgenomics.api.gene_set_enrichment` — MAGMA-style test."""

    n_genes_total: int = 0
    n_gene_sets: int = 0
    n_significant: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    genes: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "enrichment"

    def summary(self) -> str:
        lines = [
            f"Gene-set enrichment — {self.n_gene_sets} sets over {self.n_genes_total} genes, "
            f"{self.n_significant} significant (p<0.05)",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class HessRun(_BaseRun):
    """Result of :func:`torchgenomics.api.hess` — local heritability / genetic correlation."""

    mode: str = "h2"  # "h2" | "rg"
    h2_total: float = 0.0
    h2_total_se: float = 0.0
    n_regions: int = 0
    n_snps_total: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "hess"

    def summary(self) -> str:
        label = "local h²" if self.mode == "h2" else "local rg"
        lines = [
            f"HESS {label} — {self.n_regions} regions, {self.n_snps_total} SNPs, "
            f"total={self.h2_total:.4f} (se {self.h2_total_se:.4f})",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)
```

- [ ] **Step 4: Export from `api/__init__.py`** — add the four names to the `_results` import block and to `__all__` (find the line importing `SMRRun, MRMegaRun` and the `__all__` list; add `PowerRun, WinnersCurseRun, EnrichmentRun, HessRun` to both).

- [ ] **Step 5: Run the test, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py::test_result_classes_shape_and_summary -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/_results.py torchgenomics/api/__init__.py tests/test_api_hess_power_wc_enrichment.py
git commit -m "api: PowerRun/WinnersCurseRun/EnrichmentRun/HessRun result classes"
```

---

## Task 2: `api.power` + `api.winners_curse`

**Files:**
- Modify: `torchgenomics/api/postgwas.py` (add two functions after `mr_mega`, ~line 556)
- Modify: `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py` (export `power`, `winners_curse`)
- Test: `tests/test_api_hess_power_wc_enrichment.py`

**Interfaces:**
- Consumes: `postgwas.load_sumstats`, `postgwas.gwas_power`, `postgwas.required_n`, `postgwas.power_curve`, `postgwas.correct_winners_curse`; `PowerRun`, `WinnersCurseRun` from Task 1.
- Produces: `api.power(gwas, *, n=None, alpha=5e-8, target_power=0.8, power_curve=False, af_grid=None, output=None, sep="\t") -> PowerRun`; `api.winners_curse(gwas, *, method="conditional_likelihood", alpha=5e-8, n_boot=10000, seed=None, output=None, sep="\t") -> WinnersCurseRun`.

- [ ] **Step 1: Write the failing tests**

```python
def _write_power_fixture(tmp_path):
    import pandas as pd, numpy as np
    # variant 0: large effect + moderate AF (high power); variant 1: tiny effect (low power)
    df = pd.DataFrame({
        "chr": [1, 1], "pos": [10, 20], "snp": ["rsA", "rsB"],
        "a1": ["A", "A"], "a2": ["G", "G"],
        "beta": [0.5, 0.01], "se": [0.05, 0.05], "p": [1e-20, 0.8],
        "n": [10000, 10000], "af": [0.3, 0.3],
    })
    fp = tmp_path / "gwas.tsv"; df.to_csv(fp, sep="\t", index=False)
    return str(fp)


def test_api_power_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import PowerRun
    fp = _write_power_fixture(tmp_path)
    r = tg.power(fp, power_curve=True, output=str(tmp_path / "power.tsv"))
    assert isinstance(r, PowerRun)
    assert r.n_variants == 2
    res = r.results.set_index("snp")
    # high-power variant: power near 1 and SMALLER required_n than the low-power one
    assert res.loc["rsA", "power"] > 0.9
    assert res.loc["rsA", "required_n"] < res.loc["rsB", "required_n"]
    assert r.curve is not None and len(r.curve) >= 2
    assert (tmp_path / "power.tsv").exists()


def test_api_power_needs_af(tmp_path):
    import torchgenomics as tg, pandas as pd, pytest
    df = pd.DataFrame({"chr": [1], "pos": [1], "snp": ["a"], "a1": ["A"], "a2": ["G"],
                       "beta": [0.1], "se": [0.05], "p": [0.1], "n": [1000]})
    fp = tmp_path / "no_af.tsv"; df.to_csv(fp, sep="\t", index=False)
    with pytest.raises(ValueError):
        tg.power(str(fp))


def test_api_winners_curse_all_methods(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import WinnersCurseRun
    import pandas as pd
    # a strong hit (should be shrunk) + noise
    df = pd.DataFrame({
        "chr": [1] * 5, "pos": list(range(5)), "snp": [f"rs{i}" for i in range(5)],
        "a1": ["A"] * 5, "a2": ["G"] * 5,
        "beta": [0.6, 0.02, -0.01, 0.03, 0.55], "se": [0.05] * 5,
        "p": [1e-20, 0.7, 0.8, 0.6, 1e-18], "n": [10000] * 5, "af": [0.3] * 5,
    })
    fp = tmp_path / "gwas.tsv"; df.to_csv(fp, sep="\t", index=False)
    for method in ("conditional_likelihood", "fiqt", "bootstrap"):
        kw = {"n_boot": 200, "seed": 1} if method == "bootstrap" else {}
        r = tg.winners_curse(str(fp), method=method, **kw)
        assert isinstance(r, WinnersCurseRun) and r.n_variants == 5
        res = r.results.set_index("snp")
        # the strong hit is shrunk toward zero
        assert abs(res.loc["rs0", "beta_adjusted"]) <= abs(res.loc["rs0", "beta_original"]) + 1e-9
```

- [ ] **Step 2: Run, expect FAIL** (`tg.power` missing)

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "power or winners" -q`
Expected: FAIL.

- [ ] **Step 3: Implement both functions** in `postgwas.py`. Add `PowerRun, WinnersCurseRun` to the `from ._results import (...)` line first.

```python
def power(gwas, *, n=None, alpha=5e-8, target_power=0.8,
          power_curve=False, af_grid=None, output=None, sep="\t") -> "PowerRun":
    """Per-variant GWAS detection power, NCP, min-detectable-beta and required-N.

    Loads ``af``/``beta`` from a sumstats TSV; ``n`` defaults to the median of
    the finite ``n`` column. With ``power_curve=True`` also returns the
    min-detectable-|beta| envelope over an allele-frequency grid.
    """
    import numpy as np
    import torch
    from ..postgwas import load_sumstats, gwas_power, required_n
    from ..postgwas import power_curve as _power_curve  # avoid shadowing the bool param

    ss = load_sumstats(str(gwas), sep=sep)
    if ss.af is None:
        raise ValueError("power needs an allele-frequency column ('af') in the sumstats.")
    if n is None:
        finite = ss.n[torch.isfinite(ss.n)]
        if finite.numel() == 0:
            raise ValueError("power needs a sample size: pass n=... (no usable 'n' column).")
        n = float(finite.median().item())
    n = float(n)

    with timed() as elapsed:
        pr = gwas_power(n, ss.af, ss.beta, alpha=alpha, target_power=target_power)
        req = required_n(ss.af, ss.beta, alpha=alpha, target_power=target_power)
        df = pd.DataFrame({
            "snp": ss.snp,
            "af": [float(v) for v in ss.af.tolist()],
            "beta": [float(v) for v in ss.beta.tolist()],
            "power": [float(v) for v in pr.power.tolist()],
            "ncp": [float(v) for v in pr.ncp.tolist()],
            "min_detectable_beta": [float(v) for v in pr.min_detectable_beta.tolist()],
            "required_n": [float(v) for v in req.tolist()],
        })
        curve = None
        if power_curve:
            if af_grid is None:
                grid = torch.linspace(0.01, 0.5, 50, dtype=torch.float64)
            elif isinstance(af_grid, str):
                grid = torch.tensor([float(x) for x in af_grid.split(",") if x.strip()],
                                    dtype=torch.float64)
            else:
                grid = torch.as_tensor(list(af_grid), dtype=torch.float64)
            mdb = _power_curve(n, grid, alpha=alpha, target_power=target_power)
            curve = pd.DataFrame({
                "af": [float(v) for v in grid.tolist()],
                "min_detectable_beta": [float(v) for v in mdb.tolist()],
            })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        n_powered = int((df["power"] >= target_power).sum())
        return PowerRun(
            runtime_s=elapsed(), output_files=output_files, results=df, curve=curve,
            alpha=float(alpha), n=n, target_power=float(target_power),
            n_variants=int(len(df)), n_powered=n_powered,
        )


def winners_curse(gwas, *, method="conditional_likelihood", alpha=5e-8,
                  n_boot=10000, seed=None, output=None, sep="\t") -> "WinnersCurseRun":
    """Winner's-curse effect-size de-biasing (conditional_likelihood / fiqt / bootstrap)."""
    from ..postgwas import load_sumstats, correct_winners_curse

    valid = {"conditional_likelihood", "fiqt", "bootstrap"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}; got {method!r}.")
    ss = load_sumstats(str(gwas), sep=sep)

    with timed() as elapsed:
        # CL/fiqt take no **kwargs and raise TypeError on n_boot/seed — forward only for bootstrap
        extra = {"n_boot": int(n_boot), "seed": seed} if method == "bootstrap" else {}
        res = correct_winners_curse(ss.beta, ss.se, method=method, alpha=alpha, **extra)
        beta_orig = [float(v) for v in ss.beta.tolist()]
        beta_adj = [float(v) for v in res.beta_adjusted.tolist()]
        shrink = [float(v) for v in res.shrinkage_factor.tolist()]
        m = len(beta_orig)
        se_adj = ([float("nan")] * m if res.se_adjusted is None
                  else [float(v) for v in res.se_adjusted.tolist()])
        df = pd.DataFrame({
            "snp": ss.snp, "beta_original": beta_orig, "beta_adjusted": beta_adj,
            "se_adjusted": se_adj, "shrinkage_factor": shrink,
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return WinnersCurseRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            method=str(res.method), n_corrected=int(res.n_corrected), n_variants=int(m),
        )
```

- [ ] **Step 4: Export** — add `power, winners_curse` to `api/__init__.py` imports + `__all__`, and to `torchgenomics/__init__.py` top-level exports (mirror how `smr`/`mr_mega` appear there).

- [ ] **Step 5: Run, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "power or winners" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py torchgenomics/__init__.py tests/test_api_hess_power_wc_enrichment.py
git commit -m "api: power() + winners_curse() thin wrappers over postgwas"
```

---

## Task 3: `api.gene_set_enrichment` + loaders

**Files:**
- Modify: `torchgenomics/api/postgwas.py` (loaders `_load_gene_annotation`, `_load_gene_sets` near `_load_gene_map`; function after `winners_curse`)
- Modify: `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py`
- Test: `tests/test_api_hess_power_wc_enrichment.py`

**Interfaces:**
- Consumes: `postgwas.snp_to_gene`, `postgwas.gene_set_enrichment`, `EnrichmentRun`.
- Produces: `api.gene_set_enrichment(gwas, gene_annotation, gene_sets, *, window_kb=0.0, covariate_gene_size=True, covariate_log_size=True, gene_sets_format="auto", output=None, sep="\t") -> EnrichmentRun`; loaders `_load_gene_annotation(path, sep) -> (ids, chrs, starts, ends)` and `_load_gene_sets(path, fmt, sep) -> dict[str, list[str]]`.

- [ ] **Step 1: Write the failing tests**

```python
def _write_enrichment_fixture(tmp_path, gmt=False):
    import pandas as pd, numpy as np
    rng = np.random.default_rng(0)
    # 6 genes x 10 SNPs each; genes g0,g1 carry strong signal, g2..g5 null
    rows, ann = [], []
    from scipy.stats import norm
    for gi in range(6):
        strong = gi < 2
        for j in range(10):
            i = gi * 10 + j
            z = rng.normal(4.0 if strong else 0.0, 1.0)
            rows.append({"chr": 1, "pos": gi * 1000 + j, "snp": f"rs{i}",
                         "a1": "A", "a2": "G", "beta": z * 0.05, "se": 0.05,
                         "p": float(2 * norm.sf(abs(z))), "n": 5000, "af": 0.3})
        ann.append({"gene": f"g{gi}", "chr": 1, "start": gi * 1000, "end": gi * 1000 + 9})
    gwas = tmp_path / "gwas.tsv"; pd.DataFrame(rows).to_csv(gwas, sep="\t", index=False)
    gann = tmp_path / "ann.tsv"; pd.DataFrame(ann).to_csv(gann, sep="\t", index=False)
    if gmt:
        gsets = tmp_path / "sets.gmt"
        gsets.write_text("hot\tdesc\tg0\tg1\nnull\tdesc\tg3\tg4\tg5\n")
    else:
        gsets = tmp_path / "sets.tsv"
        pd.DataFrame({"set": ["hot", "hot", "null", "null", "null"],
                      "gene": ["g0", "g1", "g3", "g4", "g5"]}).to_csv(gsets, sep="\t", index=False)
    return str(gwas), str(gann), str(gsets)


def test_api_gene_set_enrichment_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import EnrichmentRun
    gwas, gann, gsets = _write_enrichment_fixture(tmp_path)
    r = tg.gene_set_enrichment(gwas, gann, gsets, output=str(tmp_path / "enr.tsv"))
    assert isinstance(r, EnrichmentRun)
    res = r.results.set_index("gene_set_name")
    # the enriched set has a smaller p than the null set
    assert res.loc["hot", "p"] < res.loc["null", "p"]
    assert not r.genes.empty
    assert (tmp_path / "enr.tsv").exists()


def test_gene_sets_tsv_and_gmt_agree(tmp_path):
    from torchgenomics.api.postgwas import _load_gene_sets
    _, _, tsv = _write_enrichment_fixture(tmp_path, gmt=False)
    tmp2 = tmp_path / "g"; tmp2.mkdir()
    _, _, gmt = _write_enrichment_fixture(tmp2, gmt=True)
    a = _load_gene_sets(tsv, "auto", "\t")
    b = _load_gene_sets(gmt, "auto", "\t")
    assert {k: sorted(v) for k, v in a.items()} == {k: sorted(v) for k, v in b.items()}


def test_gene_annotation_bad_columns(tmp_path):
    import torchgenomics as tg, pandas as pd, pytest
    gwas, _, gsets = _write_enrichment_fixture(tmp_path)
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError):
        tg.gene_set_enrichment(gwas, str(bad), gsets)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "enrichment or gene_sets or annotation" -q`
Expected: FAIL.

- [ ] **Step 3: Implement loaders + function.** Add `EnrichmentRun` to the `_results` import.

```python
def _load_gene_annotation(gene_annotation, sep="\t"):
    """Load a gene annotation TSV (columns gene, chr, start, end) into four lists."""
    df = pd.read_csv(str(gene_annotation), sep=sep)
    need = {"gene", "chr", "start", "end"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"gene-annotation file must have columns {sorted(need)}; got {list(df.columns)}."
        )
    return (
        [str(x) for x in df["gene"]],
        [str(x) for x in df["chr"]],
        [int(x) for x in df["start"]],
        [int(x) for x in df["end"]],
    )


def _load_gene_sets(gene_sets, fmt="auto", sep="\t") -> dict:
    """Load a gene-set mapping from a long TSV (set,gene) or a .gmt file."""
    if isinstance(gene_sets, dict):
        return gene_sets
    path = str(gene_sets)
    if fmt == "auto":
        fmt = "gmt" if path.lower().endswith(".gmt") else "tsv"
    out: dict[str, list[str]] = {}
    if fmt == "gmt":
        with open(path) as fh:
            for line in fh:
                parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
                if len(parts) < 3:
                    continue  # need set-name, description, >=1 gene
                name, genes = parts[0], parts[2:]
                out.setdefault(name, [])
                for g in genes:
                    if g not in out[name]:
                        out[name].append(g)
    elif fmt == "tsv":
        df = pd.read_csv(path, sep=sep)
        if "set" not in df.columns or "gene" not in df.columns:
            raise ValueError(
                f"gene-sets TSV must have 'set' and 'gene' columns; got {list(df.columns)}."
            )
        for s, g in zip(df["set"].astype(str), df["gene"].astype(str)):
            out.setdefault(s, [])
            if g not in out[s]:
                out[s].append(g)
    else:
        raise ValueError(f"gene_sets_format must be auto/tsv/gmt; got {fmt!r}.")
    if not out:
        raise ValueError(f"no gene sets parsed from {path!r}.")
    return out


def gene_set_enrichment(gwas, gene_annotation, gene_sets, *, window_kb=0.0,
                        covariate_gene_size=True, covariate_log_size=True,
                        gene_sets_format="auto", output=None, sep="\t") -> "EnrichmentRun":
    """MAGMA-style competitive gene-set enrichment (snp_to_gene then set test)."""
    from ..postgwas import load_sumstats, snp_to_gene
    from ..postgwas import gene_set_enrichment as _gse

    ss = load_sumstats(str(gwas), sep=sep)
    gid, gchr, gstart, gend = _load_gene_annotation(gene_annotation, sep=sep)
    sets = _load_gene_sets(gene_sets, fmt=gene_sets_format, sep=sep)

    with timed() as elapsed:
        gene_result = snp_to_gene(ss, gid, gchr, gstart, gend, window_kb=window_kb)
        enr = _gse(gene_result, sets, covariate_gene_size=covariate_gene_size,
                   covariate_log_size=covariate_log_size)
        df = pd.DataFrame({
            "gene_set_name": list(enr.gene_set_name),
            "n_genes_in_set": [int(x) for x in enr.n_genes_in_set],
            "beta_enrichment": [float(v) for v in enr.beta_enrichment.tolist()],
            "se": [float(v) for v in enr.se.tolist()],
            "p": [float(v) for v in enr.p.tolist()],
        })
        genes_df = pd.DataFrame({
            "gene_id": list(gene_result.gene_id),
            "gene_chr": list(gene_result.gene_chr),
            "gene_start": [int(x) for x in gene_result.gene_start],
            "gene_end": [int(x) for x in gene_result.gene_end],
            "n_snps": [int(x) for x in gene_result.n_snps],
            "stat": [float(v) for v in gene_result.stat.tolist()],
            "p": [float(v) for v in gene_result.p.tolist()],
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return EnrichmentRun(
            runtime_s=elapsed(), output_files=output_files, results=df, genes=genes_df,
            n_genes_total=int(enr.n_genes_total), n_gene_sets=int(len(df)),
            n_significant=int((df["p"] < 0.05).sum()),
        )
```

- [ ] **Step 4: Export** `gene_set_enrichment` from `api/__init__.py` + `torchgenomics/__init__.py`. NOTE: the api facade name `gene_set_enrichment` differs from the postgwas function of the same name — inside the wrapper it's imported `as _gse`, so no collision. In `api/__init__.py` the facade `gene_set_enrichment` is the one exported.

- [ ] **Step 5: Run, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "enrichment or gene_sets or annotation" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py torchgenomics/__init__.py tests/test_api_hess_power_wc_enrichment.py
git commit -m "api: gene_set_enrichment() + gene-annotation/gene-set (TSV+GMT) loaders"
```

---

## Task 4: `api.hess` + LD/region loaders + guards

**Files:**
- Modify: `torchgenomics/api/postgwas.py` (loaders `_load_matrix`, `_load_regions`; function after `gene_set_enrichment`)
- Modify: `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py`
- Test: `tests/test_api_hess_power_wc_enrichment.py`

**Interfaces:**
- Consumes: `postgwas.load_sumstats`, `postgwas.hess_local_h2`, `postgwas.hess_local_rg`, `HessRun`.
- Produces: `api.hess(gwas, ld_matrix, regions, *, n=None, n2=None, gwas2=None, eigenvalue_threshold=1.0, output=None, sep="\t") -> HessRun`.

- [ ] **Step 1: Write the failing tests**

```python
def _write_hess_fixture(tmp_path, shuffle=False):
    import numpy as np, pandas as pd, torch
    rng = np.random.default_rng(0)
    m = 40  # 4 regions x 10 SNPs, positions sorted
    pos = list(range(1, m + 1))
    z = rng.normal(0, 1, size=m)
    z[0:10] += 3.0  # region 1 carries signal
    order = list(range(m))
    if shuffle:
        rng.shuffle(order)
    df = pd.DataFrame({
        "chr": [1] * m, "pos": [pos[i] for i in order],
        "snp": [f"rs{order[i]}" for i in range(m)],
        "a1": ["A"] * m, "a2": ["G"] * m,
        "beta": [z[order[i]] * 0.02 for i in range(m)], "se": [0.02] * m,
        "p": [0.01] * m, "n": [10000] * m, "af": [0.3] * m,
    })
    gwas = tmp_path / "gwas.tsv"; df.to_csv(gwas, sep="\t", index=False)
    ld = (torch.eye(m, dtype=torch.float64)).numpy()
    ldp = tmp_path / "ld.npy"; np.save(ldp, ld)
    reg = pd.DataFrame({"chrom": [1, 1, 1, 1],
                        "start": [1, 11, 21, 31], "end": [10, 20, 30, 40]})
    regp = tmp_path / "regions.tsv"; reg.to_csv(regp, sep="\t", index=False)
    return str(gwas), str(ldp), str(regp)


def test_api_hess_h2_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import HessRun
    gwas, ld, reg = _write_hess_fixture(tmp_path)
    r = tg.hess(gwas, ld, reg, output=str(tmp_path / "hess.tsv"))
    assert isinstance(r, HessRun) and r.mode == "h2"
    assert r.n_regions == 4 and r.n_snps_total == 40
    # region label carries the bp coords from the regions file, not indices
    assert r.results["region_id"].iloc[0] == "1:1-10"
    assert (tmp_path / "hess.tsv").exists()


def test_api_hess_contiguity_guard(tmp_path):
    import torchgenomics as tg, pytest
    gwas, ld, reg = _write_hess_fixture(tmp_path, shuffle=True)
    with pytest.raises(ValueError):
        tg.hess(gwas, ld, reg)


def test_api_hess_rg_mode(tmp_path):
    import torchgenomics as tg
    gwas, ld, reg = _write_hess_fixture(tmp_path)
    # reuse the same sumstats as a second trait -> rg path runs
    r = tg.hess(gwas, ld, reg, gwas2=gwas)
    assert r.mode == "rg" and r.n_regions == 4
```

- [ ] **Step 2: Run, expect FAIL**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "hess" -q`
Expected: FAIL.

- [ ] **Step 3: Implement loaders + function.** Add `HessRun` to the `_results` import.

```python
def _load_matrix(path):
    """Load an (m, m) LD matrix from a .npy or .pt/.pth file as a float64 Tensor."""
    import numpy as np
    import torch
    p = str(path)
    if p.lower().endswith(".npy"):
        arr = np.load(p)
        mat = torch.as_tensor(arr, dtype=torch.float64)
    elif p.lower().endswith((".pt", ".pth")):
        obj = torch.load(p, map_location="cpu")
        mat = torch.as_tensor(obj, dtype=torch.float64)
    else:
        raise ValueError(f"ld_matrix must be a .npy or .pt/.pth file; got {p!r}.")
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"ld_matrix must be square (m, m); got shape {tuple(mat.shape)}.")
    return mat


def _load_regions(regions, sep="\t"):
    """Load a genomic regions TSV (columns chrom, start, end) into a list of tuples."""
    df = pd.read_csv(str(regions), sep=sep)
    need = {"chrom", "start", "end"}
    if not need.issubset(df.columns):
        raise ValueError(
            f"regions file must have columns {sorted(need)} (bp); got {list(df.columns)}."
        )
    return [(str(c), int(s), int(e))
            for c, s, e in zip(df["chrom"], df["start"], df["end"])]


def hess(gwas, ld_matrix, regions, *, n=None, n2=None, gwas2=None,
         eigenvalue_threshold=1.0, output=None, sep="\t") -> "HessRun":
    """HESS local heritability (or local genetic correlation with ``gwas2``).

    ``ld_matrix`` (.npy/.pt) must be (m, m) aligned to the sumstats row order.
    ``regions`` is a TSV (chrom,start,end in bp); each region is mapped to a
    CONTIGUOUS SNP-index block, so the sumstats + LD matrix must be sorted by
    genomic position.
    """
    import warnings
    import torch
    from ..postgwas import load_sumstats, hess_local_h2, hess_local_rg

    ss = load_sumstats(str(gwas), sep=sep)
    ld = _load_matrix(ld_matrix)
    if ld.shape[0] != ss.m:
        raise ValueError(f"ld_matrix is {ld.shape[0]}x{ld.shape[0]} but the sumstats "
                         f"has {ss.m} SNPs — they must align 1:1.")
    reg = _load_regions(regions, sep=sep)

    def _resolve_n(nval, label):
        if nval is not None:
            return float(nval)
        finite = ss.n[torch.isfinite(ss.n)]
        if finite.numel() == 0:
            raise ValueError(f"hess needs {label}: pass it explicitly (no usable 'n' column).")
        return float(finite.median().item())

    # map bp regions -> contiguous (start, end) index blocks
    chr_str = [str(c) for c in ss.chr]
    pos = [int(p) for p in ss.pos]
    bounds: list[tuple[int, int]] = []
    labels: list[str] = []
    bp_meta: list[tuple[str, int, int]] = []
    for chrom, start, end in reg:
        idx = [i for i in range(ss.m)
               if chr_str[i] == str(chrom) and start <= pos[i] <= end]
        if not idx:
            warnings.warn(f"hess: region {chrom}:{start}-{end} matched no SNPs — skipping.")
            continue
        first, last = min(idx), max(idx)
        if sorted(idx) != list(range(first, last + 1)):
            raise ValueError(
                f"region {chrom}:{start}-{end} maps to non-contiguous SNP indices — "
                f"the sumstats and LD matrix must be sorted by genomic position."
            )
        bounds.append((first, last + 1))
        labels.append(f"{chrom}:{start}-{end}")
        bp_meta.append((str(chrom), int(start), int(end)))
    if not bounds:
        raise ValueError(
            f"none of the {len(reg)} regions matched any SNP in the sumstats — "
            f"check chromosome naming and coordinates."
        )

    with timed() as elapsed:
        if gwas2 is not None:
            ss2 = load_sumstats(str(gwas2), sep=sep)
            if ss2.m != ss.m:
                raise ValueError("gwas2 must have the same SNPs (same m) as gwas.")
            n1v = _resolve_n(n, "n (trait 1)")
            n2v = _resolve_n(n2 if n2 is not None else n, "n2 (trait 2)")
            hres = hess_local_rg(ss.z, ss2.z, ld, n1v, n2v, bounds,
                                 region_labels=labels,
                                 eigenvalue_threshold=eigenvalue_threshold)
            mode = "rg"
        else:
            nv = _resolve_n(n, "n")
            hres = hess_local_h2(ss.z, ld, nv, bounds, region_labels=labels,
                                 eigenvalue_threshold=eigenvalue_threshold)
            mode = "h2"
        rows = []
        for rr, (chrom, start, end) in zip(hres.regions, bp_meta):
            rows.append({
                "region_id": rr.region_id, "chrom": chrom, "start": start, "end": end,
                "h2_local": rr.h2_local, "h2_local_se": rr.h2_local_se,
                "n_snps": rr.n_snps, "n_eigenvalues_kept": rr.n_eigenvalues_kept,
            })
        df = pd.DataFrame(rows)
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return HessRun(
            runtime_s=elapsed(), output_files=output_files, results=df, mode=mode,
            h2_total=float(hres.h2_total), h2_total_se=float(hres.h2_total_se),
            n_regions=int(hres.n_regions), n_snps_total=int(hres.n_snps_total),
        )
```

Note: `api/postgwas.py` has no `log()` helper (confirmed — it imports only `emit_progress`, which requires a callback). Use `warnings.warn(...)` for the skipped-region message, with `import warnings` at the top of the function body (shown above).

- [ ] **Step 4: Export** `hess` from `api/__init__.py` + `torchgenomics/__init__.py`.

- [ ] **Step 5: Run, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_hess_power_wc_enrichment.py -k "hess" -q`
Expected: PASS. Also run the whole api test file: `... tests/test_api_hess_power_wc_enrichment.py -q`.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py torchgenomics/__init__.py tests/test_api_hess_power_wc_enrichment.py
git commit -m "api: hess() local h2/rg with LD-matrix + bp-region loaders and contiguity guard"
```

---

## Task 5: CLI subcommands + manifest + docs

**Files:**
- Modify: `torchgenomics/cli.py` (4 `_add_*_parser`, 4 `_cmd_*`, register in `_build_parser` ~line 104 and dispatch dict ~line 177)
- Modify: `torchgenomics/_manifest/__init__.py` (add 4 names to `_TIER_2_CLI_SUBCOMMANDS`, line 25)
- Modify: `CLAUDE.md` (CLI count 50→54, add 4 example lines)
- Test: `tests/test_cli_hess_power_wc_enrichment.py` (new)

**Interfaces:**
- Consumes: `api.power`, `api.winners_curse`, `api.gene_set_enrichment`, `api.hess`.
- Produces: CLI subcommands `power`, `winners-curse`, `gene-set-enrichment`, `hess`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_hess_power_wc_enrichment.py
import subprocess, sys


def _help(sub):
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", sub, "--help"],
                       capture_output=True, text=True)
    return r


def test_cli_help_all_four():
    for sub in ("power", "winners-curse", "gene-set-enrichment", "hess"):
        r = _help(sub)
        assert r.returncode == 0, r.stderr
        assert "--gwas" in r.stdout


def test_cli_power_end_to_end(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"chr": [1, 1], "pos": [1, 2], "snp": ["a", "b"], "a1": ["A", "A"],
                       "a2": ["G", "G"], "beta": [0.5, 0.01], "se": [0.05, 0.05],
                       "p": [1e-20, 0.8], "n": [10000, 10000], "af": [0.3, 0.3]})
    fp = tmp_path / "g.tsv"; df.to_csv(fp, sep="\t", index=False)
    out = tmp_path / "power.tsv"
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "power",
                        "--gwas", str(fp), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
```

- [ ] **Step 2: Run, expect FAIL** (unknown subcommands).

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_hess_power_wc_enrichment.py -q`
Expected: FAIL.

- [ ] **Step 3: Add parsers + commands** (mirror `_add_smr_parser`/`_cmd_smr`). Parsers:

```python
def _add_power_parser(subparsers):
    p = subparsers.add_parser("power", help="Per-variant GWAS detection power.")
    p.add_argument("--gwas", required=True, help="GWAS sumstats TSV (needs af, beta).")
    p.add_argument("--n", type=float, default=None, help="Sample size (default: median n col).")
    p.add_argument("--alpha", type=float, default=5e-8)
    p.add_argument("--target-power", type=float, default=0.8)
    p.add_argument("--power-curve", action="store_true", default=False,
                   help="Also compute the min-detectable-|beta| AF envelope.")
    p.add_argument("--af-grid", default=None, help="Comma-separated AF grid for the curve.")
    p.add_argument("--output", default=None)


def _add_winners_curse_parser(subparsers):
    p = subparsers.add_parser("winners-curse", help="Winner's-curse effect-size de-biasing.")
    p.add_argument("--gwas", required=True)
    p.add_argument("--method", choices=["conditional_likelihood", "fiqt", "bootstrap"],
                   default="conditional_likelihood")
    p.add_argument("--alpha", type=float, default=5e-8)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)


def _add_gene_set_enrichment_parser(subparsers):
    p = subparsers.add_parser("gene-set-enrichment", help="MAGMA-style gene-set enrichment.")
    p.add_argument("--gwas", required=True)
    p.add_argument("--gene-annotation", required=True, help="TSV: gene,chr,start,end.")
    p.add_argument("--gene-sets", required=True, help="Long TSV (set,gene) or .gmt.")
    p.add_argument("--window-kb", type=float, default=0.0)
    p.add_argument("--gene-sets-format", choices=["auto", "tsv", "gmt"], default="auto")
    p.add_argument("--no-covariate-gene-size", action="store_true", default=False)
    p.add_argument("--no-covariate-log-size", action="store_true", default=False)
    p.add_argument("--output", default=None)


def _add_hess_parser(subparsers):
    p = subparsers.add_parser("hess", help="HESS local heritability / genetic correlation.")
    p.add_argument("--gwas", required=True)
    p.add_argument("--ld-matrix", required=True, help="(m,m) LD matrix .npy or .pt.")
    p.add_argument("--regions", required=True, help="TSV: chrom,start,end (bp).")
    p.add_argument("--n", type=float, default=None)
    p.add_argument("--n2", type=float, default=None, help="Trait-2 N (rg mode).")
    p.add_argument("--gwas2", default=None, help="Second sumstats -> local rg mode.")
    p.add_argument("--eigenvalue-threshold", type=float, default=1.0)
    p.add_argument("--output", default=None)
```

Commands:

```python
def _cmd_power(args):
    from .api import power
    r = power(args.gwas, n=args.n, alpha=args.alpha, target_power=args.target_power,
              power_curve=args.power_curve, af_grid=args.af_grid, output=args.output)
    print(r.summary()); return 0


def _cmd_winners_curse(args):
    from .api import winners_curse
    r = winners_curse(args.gwas, method=args.method, alpha=args.alpha,
                      n_boot=args.n_boot, seed=args.seed, output=args.output)
    print(r.summary()); return 0


def _cmd_gene_set_enrichment(args):
    from .api import gene_set_enrichment
    r = gene_set_enrichment(args.gwas, args.gene_annotation, args.gene_sets,
                            window_kb=args.window_kb, gene_sets_format=args.gene_sets_format,
                            covariate_gene_size=not args.no_covariate_gene_size,
                            covariate_log_size=not args.no_covariate_log_size,
                            output=args.output)
    print(r.summary()); return 0


def _cmd_hess(args):
    from .api import hess
    r = hess(args.gwas, args.ld_matrix, args.regions, n=args.n, n2=args.n2,
             gwas2=args.gwas2, eigenvalue_threshold=args.eigenvalue_threshold,
             output=args.output)
    print(r.summary()); return 0
```

Register the 4 `_add_*_parser(subparsers)` calls next to `_add_mr_mega_parser` (~line 104) and the 4 dispatch entries next to `"mr-mega": _cmd_mr_mega` (~line 177): `"power": _cmd_power, "winners-curse": _cmd_winners_curse, "gene-set-enrichment": _cmd_gene_set_enrichment, "hess": _cmd_hess`.

- [ ] **Step 4: Manifest + docs.** In `torchgenomics/_manifest/__init__.py` line 25, add `"power", "winners-curse", "gene-set-enrichment", "hess"` to the set. In `CLAUDE.md`, bump the CLI count 50→54 and add four example lines under the post-GWAS section:

```bash
torchgenomics power --gwas gwas.tsv --power-curve --output power.tsv
torchgenomics winners-curse --gwas gwas.tsv --method conditional_likelihood --output wc.tsv
torchgenomics gene-set-enrichment --gwas gwas.tsv --gene-annotation genes.tsv --gene-sets sets.gmt --output enr.tsv
torchgenomics hess --gwas gwas.tsv --ld-matrix ld.npy --regions regions.tsv --output hess.tsv
```

- [ ] **Step 5: Run, expect PASS** + confirm precedence + CLI count.

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_hess_power_wc_enrichment.py -q`
Then: `TORCHGENOMICS_DISABLE_NATIVE=1 python -c "import torchgenomics.api as a; assert a.power.__name__=='power' and a.hess.__name__=='hess'; print('precedence ok')"`
Also run the manifest test: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_manifest.py -q` (expects no CLI drift).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/cli.py torchgenomics/_manifest/__init__.py CLAUDE.md tests/test_cli_hess_power_wc_enrichment.py
git commit -m "cli: power/winners-curse/gene-set-enrichment/hess subcommands + manifest + docs"
```

---

## Task 6: R wrappers + S4 classes + codegen fix

**Files:**
- Modify: `rTorchGenomics/R/api.R` (4 `tg_*` wrappers), `R/result_classes.R` (4 setClass + 4 `new_from_dict` + 4 `show`), `R/codegen.R` (`.HANDCRAFTED_COMMANDS`), `NAMESPACE`, `_pkgdown.yml`
- Create: `man/tg_power.Rd`, `tg_winners_curse.Rd`, `tg_gene_set_enrichment.Rd`, `tg_hess.Rd`, `PowerRun.Rd`, `WinnersCurseRun.Rd`, `EnrichmentRun.Rd`, `HessRun.Rd` (roxygen-generated), `tests/testthat/test-hess-power-wc-enrichment.R`
- Regenerate: `inst/rbridge_manifest.json`, `R/api_auto.R`

**Interfaces:**
- Consumes: the CLI subcommands from Task 5 via `bridge_call("<name>", args)` (the R bridge invokes the api through the CLI-runner).
- Produces: `tg_power`, `tg_winners_curse`, `tg_gene_set_enrichment`, `tg_hess` + S4 classes.

- [ ] **Step 1: Add the four `tg_*` wrappers** to `R/api.R` (mirror `tg_smr`). Use `.as_path`/`.compact`; `bridge_call` name = underscored manifest name.

```r
#' GWAS detection power
#' @export
tg_power <- function(gwas, n = NULL, alpha = 5e-8, target_power = 0.8,
                     power_curve = FALSE, af_grid = NULL, output = NULL) {
  d <- bridge_call("power", .compact(list(
    gwas = .as_path(gwas), n = n, alpha = alpha, target_power = target_power,
    power_curve = power_curve, af_grid = af_grid, output = .as_path(output))))
  PowerRun$new_from_dict(d)
}

#' Winner's-curse correction
#' @export
tg_winners_curse <- function(gwas, method = "conditional_likelihood", alpha = 5e-8,
                             n_boot = 10000L, seed = NULL, output = NULL) {
  d <- bridge_call("winners_curse", .compact(list(
    gwas = .as_path(gwas), method = method, alpha = alpha,
    n_boot = as.integer(n_boot), seed = seed, output = .as_path(output))))
  WinnersCurseRun$new_from_dict(d)
}

#' Gene-set enrichment (MAGMA-style)
#' @export
tg_gene_set_enrichment <- function(gwas, gene_annotation, gene_sets, window_kb = 0.0,
                                   covariate_gene_size = TRUE, covariate_log_size = TRUE,
                                   gene_sets_format = "auto", output = NULL) {
  d <- bridge_call("gene_set_enrichment", .compact(list(
    gwas = .as_path(gwas), gene_annotation = .as_path(gene_annotation),
    gene_sets = .as_path(gene_sets), window_kb = window_kb,
    covariate_gene_size = covariate_gene_size, covariate_log_size = covariate_log_size,
    gene_sets_format = gene_sets_format, output = .as_path(output))))
  EnrichmentRun$new_from_dict(d)
}

#' HESS local heritability / genetic correlation
#' @export
tg_hess <- function(gwas, ld_matrix, regions, n = NULL, n2 = NULL, gwas2 = NULL,
                    eigenvalue_threshold = 1.0, output = NULL) {
  d <- bridge_call("hess", .compact(list(
    gwas = .as_path(gwas), ld_matrix = .as_path(ld_matrix), regions = .as_path(regions),
    n = n, n2 = n2, gwas2 = .as_path(gwas2),
    eigenvalue_threshold = eigenvalue_threshold, output = .as_path(output))))
  HessRun$new_from_dict(d)
}
```

- [ ] **Step 2: Add S4 classes** to `R/result_classes.R` (mirror `SMRRun` setClass + `new_from_dict` list + `show` method). `PowerRun` and `EnrichmentRun` each carry a SECOND data.frame slot (`curve`, `genes`). Example for `PowerRun`:

```r
setClass("PowerRun", contains = "_BaseRun",
  representation(alpha = "numeric", n = "numeric", target_power = "numeric",
                n_variants = "integer", n_powered = "integer",
                results = "data.frame", curve = "data.frame", raw = "list"))
PowerRun <- list(new_from_dict = function(d) {
  new("PowerRun",
    runtime_s = .as_num(d$runtime_s), output_files = .as_list(d$output_files),
    log_excerpt = .as_chr_vec(d$log_excerpt),
    alpha = .as_num(d$alpha), n = .as_num(d$n), target_power = .as_num(d$target_power),
    n_variants = .as_int(d$n_variants), n_powered = .as_int(d$n_powered),
    results = .tibble_from_list_of_dicts(d$results),
    curve = .tibble_from_list_of_dicts(d$curve %||% list()), raw = d)
})
setMethod("show", "PowerRun", function(object) {
  cat(sprintf("<PowerRun> %d variants, %d powered (alpha=%.1e, N=%g)\n",
              object@n_variants, object@n_powered, object@alpha, object@n))
})
```

Do the same for `WinnersCurseRun` (slots: method, n_corrected, n_variants, results), `EnrichmentRun` (slots: n_genes_total, n_gene_sets, n_significant, results, genes), `HessRun` (slots: mode="character", h2_total, h2_total_se, n_regions, n_snps_total, results). NOTE: `curve` may be `null` in the dict → guard with `%||% list()` so `.tibble_from_list_of_dicts` gets an empty list.

- [ ] **Step 3: Fix `.HANDCRAFTED_COMMANDS`** in `R/codegen.R:14` — add the four new + back-fill the two missed:

```r
.HANDCRAFTED_COMMANDS <- c("gwas", "recommend", "models", "mr", "coloc",
                           "clump", "meta", "lmm_scan", "glm_scan",
                           "pgs_fit", "pgs_score", "ld_blocks", "lgebv",
                           "annotate_hits", "convert", "impute", "validate",
                           "smr", "mr_mega",
                           "power", "winners_curse", "gene_set_enrichment", "hess")
```

- [ ] **Step 4: Update NAMESPACE + `_pkgdown.yml`** — add `export(tg_power/tg_winners_curse/tg_gene_set_enrichment/tg_hess)`, `export(PowerRun/...)` list objects are not exported (they're plain lists, like `SMRRun` the *list* is not in NAMESPACE — only the S4 class is), `exportClasses(PowerRun/WinnersCurseRun/EnrichmentRun/HessRun)`. Add the four functions + four classes to `_pkgdown.yml` reference index (post-GWAS section). Generate `man/*.Rd` via `Rscript -e 'devtools::document()'` (or roxygen2::roxygenise).

- [ ] **Step 5: Regenerate manifest + api_auto.** From repo root:

```bash
python -m torchgenomics._manifest > rTorchGenomics/inst/rbridge_manifest.json
Rscript -e 'setwd("rTorchGenomics"); devtools::load_all("."); generate_api_auto()'
grep -E "tg_power|tg_hess|tg_smr|tg_mr_mega|tg_winners_curse|tg_gene_set_enrichment" rTorchGenomics/R/api_auto.R || echo "OK: no handcrafted names in api_auto.R"
```
Expected: the `grep` finds NOTHING (all six are excluded), printing "OK".

- [ ] **Step 6: Write mocked R tests** in `tests/testthat/test-hess-power-wc-enrichment.R` (mirror `test-smr-mrmega.R`): for each `tg_*`, `mockery::stub(<fn>, "bridge_call", ...)` returning a dict, assert `captured$fn` is the underscored name, key args forwarded, and `expect_s4_class(r, "<Run>")`. Run:

```bash
Rscript -e 'setwd("rTorchGenomics"); devtools::load_all("."); testthat::test_file("tests/testthat/test-hess-power-wc-enrichment.R")'
```
Expected: all pass. Also run the full R suite to confirm no regression: `Rscript -e 'setwd("rTorchGenomics"); devtools::test()'`.

- [ ] **Step 7: Commit**

```bash
git add rTorchGenomics/
git commit -m "r: tg_power/winners_curse/gene_set_enrichment/hess wrappers + S4 classes + codegen fix"
```

---

## Self-Review notes (addressed)

- Spec coverage: Task 1 (result classes) → §"Result classes"; Task 2 (power/wc) → §1/§2; Task 3 (enrichment + both file formats) → §3; Task 4 (hess + guards) → §4; Task 5 (CLI + manifest + docs) → §"CLI"; Task 6 (R + codegen back-fill) → §"R". All spec sections mapped.
- Type consistency: result-class field names match between `_results.py` (Task 1), the api builders (Tasks 2-4), and the R `new_from_dict` slots (Task 6). `power_curve` bool param vs `_power_curve` import alias handled in Task 2. Winner's-curse conditional kwargs + always-NaN `se_adjusted` handled in Task 2. HESS end-exclusive bounds + contiguity + all-skipped + N guards in Task 4. `.HANDCRAFTED_COMMANDS` back-fill (smr/mr_mega) in Task 6.
- Resolved: `api/postgwas.py` has no `log()`; Task 4 uses `warnings.warn` (import inside the function). All R helpers used in Task 6 (`.compact`, `.as_path`, `%||%`, `.tibble_from_list_of_dicts`, `.as_int/.as_num/.as_list/.as_chr_vec`) exist in `R/utils.R` + `R/result_classes.R`.
