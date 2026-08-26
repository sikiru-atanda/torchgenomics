# Wire SMR + MR-MEGA through api / CLI / R — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose `smr_heidi` (SMR+HEIDI) and `mr_mega` (multi-ancestry MR) — currently Python-library-only — as `api.smr`/`api.mr_mega`, `torchgenomics smr`/`mr-mega` CLI subcommands, and `tg_smr`/`tg_mr_mega` R wrappers. Mirrors the shipped mr/coloc pattern. No new statistics.

**Architecture:** `api/postgwas.py` gains `smr()`/`mr_mega()` (load sumstats via `postgwas.load_sumstats`, load gene-map TSV, run the existing `postgwas.smr_heidi`/`mr_mega`, write a TSV, return a typed `SMRRun`/`MRMegaRun`). CLI subcommands thin-wrap. Hand-crafted R wrappers `bridge_call` them + wrap into S4.

**Tech Stack:** Python 3.12, pandas, torch; `torchgenomics.postgwas` (existing SMR/MR-MEGA math); `torchgenomics.api`/`cli`; R reticulate bridge.

## Global Constraints

- **No new statistics** — orchestrate `postgwas.smr_heidi`/`postgwas.mr_mega`. No change to that math.
- **Backward compatible / additive** — new api functions + result classes, new CLI subcommands, new R wrappers + S4, two `_TIER_2_CLI_SUBCOMMANDS` entries. No existing signature/default changes.
- **Reviewer-grade docstrings**; **friendly errors** (gene-map missing columns; mr_mega too few populations; missing sumstats file/column via `load_sumstats`).
- Follow the tier-1 hand-crafted pattern (mr/coloc), NOT the auto-generator.
- gene-map = long-format TSV (`gene\tsnp`, one row per cis pair). SMR `ld_matrix` NOT exposed via CLI/R.
- Verify commands under `TORCHGENOMICS_DISABLE_NATIVE=1`.

## File Structure

- Modify: `torchgenomics/api/_results.py` — add `SMRRun`, `MRMegaRun` (+ export).
- Modify: `torchgenomics/api/postgwas.py` — add `smr()`, `mr_mega()`.
- Modify: `torchgenomics/api/__init__.py` — export `smr`, `mr_mega`, `SMRRun`, `MRMegaRun`.
- Modify: `torchgenomics/cli.py` — `_add_smr_parser`/`_cmd_smr`, `_add_mr_mega_parser`/`_cmd_mr_mega`, registration.
- Modify: `torchgenomics/_manifest/__init__.py` — add `"smr"`, `"mr-mega"`.
- Modify: `CLAUDE.md` — CLI count 48→50; SMR/MR-MEGA examples.
- Modify: `rTorchGenomics/R/api.R` — `tg_smr`, `tg_mr_mega`; `R/result_classes.R` — `SMRRun`/`MRMegaRun` S4; `NAMESPACE`; `_pkgdown.yml`.
- Test: `tests/test_api_smr_mrmega.py`, `tests/test_cli_smr_mrmega.py`, `rTorchGenomics/tests/testthat/test-smr-mrmega.R`.

**Interface contract:**

```python
# api/postgwas.py
def smr(gwas, eqtl, gene_map, *, output=None, eqtl_p_threshold=5e-8,
        smr_p_threshold=0.05, heidi_p_threshold=0.05, heidi_max_snps=20,
        sep="\t") -> "SMRRun": ...
def mr_mega(sumstats, *, output=None, n_axes=4, random_effects=False,
            sep="\t") -> "MRMegaRun": ...
# _results.py  (subclass _BaseRun like MetaRun/MRRun)
#   SMRRun(results:pd.DataFrame, n_genes_tested:int, n_significant_smr:int, n_pass_heidi:int)
#   MRMegaRun(results:pd.DataFrame, method:str, n_axes:int, n_populations:int, n_snps:int, min_p_meta:float|None)
```

---

### Task 1: `SMRRun` + `MRMegaRun` result classes

**Files:** Modify `torchgenomics/api/_results.py`, `torchgenomics/api/__init__.py`; Test `tests/test_api_smr_mrmega.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_smr_mrmega.py
import pandas as pd


def test_smrrun_shape_and_summary():
    from torchgenomics.api import SMRRun
    r = SMRRun(n_genes_tested=5, n_significant_smr=2, n_pass_heidi=1,
               results=pd.DataFrame({"gene_id": ["g1", "g2"], "p_smr": [1e-9, 0.2]}))
    assert r.n_genes_tested == 5 and r.n_significant_smr == 2
    s = r.summary()
    assert "SMR" in s and "5" in s
    assert r.to_dict()["n_pass_heidi"] == 1


def test_mrmegarun_shape_and_summary():
    from torchgenomics.api import MRMegaRun
    r = MRMegaRun(method="mr_mega", n_axes=1, n_populations=3, n_snps=40,
                  min_p_meta=1e-8,
                  results=pd.DataFrame({"p_meta": [1e-8, 0.3], "p_ancestry": [0.5, 0.6]}))
    s = r.summary()
    assert "MR-MEGA" in s and "3" in s and "40" in s
    assert r.to_dict()["n_populations"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "smrrun or mrmegarun" -v`
Expected: FAIL (classes not importable).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/_results.py`, after `MRRun`/`ColocRun` (mirror their shape):

```python
@dataclass
class SMRRun(_BaseRun):
    """Result of :func:`torchgenomics.api.smr` — SMR + HEIDI across genes."""

    n_genes_tested: int = 0
    n_significant_smr: int = 0
    n_pass_heidi: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "smr"

    def summary(self) -> str:
        lines = [
            f"SMR — {self.n_genes_tested} genes tested, "
            f"{self.n_significant_smr} SMR-significant, {self.n_pass_heidi} pass HEIDI",
        ]
        if not self.results.empty:
            lines.append(self.results.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)


@dataclass
class MRMegaRun(_BaseRun):
    """Result of :func:`torchgenomics.api.mr_mega` — multi-ancestry MR meta-analysis."""

    method: str = ""
    n_axes: int = 0
    n_populations: int = 0
    n_snps: int = 0
    min_p_meta: float | None = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "mr_mega"

    def summary(self) -> str:
        mp = f"min p_meta={self.min_p_meta:.3e}" if self.min_p_meta is not None else ""
        lines = [
            f"MR-MEGA — {self.n_populations} populations, {self.n_axes} axes, "
            f"{self.n_snps} SNPs {mp}",
        ]
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)
```

Add `SMRRun`, `MRMegaRun` to `api/__init__.py` imports + `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "smrrun or mrmegarun" -v`
Expected: PASS. Regression: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/ -k "api and (result or facade)" -q`.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/_results.py torchgenomics/api/__init__.py tests/test_api_smr_mrmega.py
git commit -m "api: SMRRun + MRMegaRun result classes"
```

---

### Task 2: `api.smr()`

**Files:** Modify `torchgenomics/api/postgwas.py`, `torchgenomics/api/__init__.py`; Test `tests/test_api_smr_mrmega.py`.

**Interfaces:** Consumes `SMRRun` (Task 1), `postgwas.load_sumstats`, `postgwas.smr_heidi`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_smr_mrmega.py (append)
def _write_smr_fixture(tmp_path, seed=0, n_snp=30, n_genes=3):
    """GWAS + eQTL sumstats sharing a cis causal SNP per gene + a gene-map TSV."""
    import numpy as np
    rng = np.random.default_rng(seed)

    def _row(i):
        return {"chr": 1, "pos": i + 1, "snp": f"rs{i}", "a1": "A", "a2": "G", "n": 5000, "af": 0.3}

    gwas_rows, eqtl_rows, map_rows = [], [], []
    per = n_snp // n_genes
    for g in range(n_genes):
        causal = g * per + 1
        for j in range(per):
            i = g * per + j
            z_e = 6.0 if i == causal else rng.normal()
            z_g = 5.0 if i == causal else rng.normal()   # shared signal at the causal cis-SNP
            gwas_rows.append({**_row(i), "beta": z_g * 0.05, "se": 0.05, "p": 2 * (1 - 0.9999)})
            eqtl_rows.append({**_row(i), "beta": z_e * 0.05, "se": 0.05, "p": 2 * (1 - 0.9999)})
            map_rows.append({"gene": f"gene{g}", "snp": f"rs{i}"})
    import pandas as pd
    gwas = tmp_path / "gwas.tsv"; eqtl = tmp_path / "eqtl.tsv"; gmap = tmp_path / "map.tsv"
    pd.DataFrame(gwas_rows).to_csv(gwas, sep="\t", index=False)
    pd.DataFrame(eqtl_rows).to_csv(eqtl, sep="\t", index=False)
    pd.DataFrame(map_rows).to_csv(gmap, sep="\t", index=False)
    return str(gwas), str(eqtl), str(gmap), n_genes


def test_api_smr_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import SMRRun
    gwas, eqtl, gmap, n_genes = _write_smr_fixture(tmp_path)
    r = tg.smr(gwas, eqtl, gmap, output=str(tmp_path / "smr.tsv"))
    assert isinstance(r, SMRRun)
    assert r.n_genes_tested == n_genes
    assert "gene_id" in r.results.columns
    assert (tmp_path / "smr.tsv").exists()


def test_api_smr_bad_gene_map(tmp_path):
    import torchgenomics as tg, pandas as pd, pytest
    gwas, eqtl, _, _ = _write_smr_fixture(tmp_path)
    bad = tmp_path / "bad.tsv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, sep="\t", index=False)
    with pytest.raises(ValueError) as e:
        tg.smr(gwas, eqtl, str(bad))
    assert "gene" in str(e.value) and "snp" in str(e.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "api_smr" -v`
Expected: FAIL (`tg.smr` doesn't exist).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/api/postgwas.py` (mirror `mr()`'s structure — module-level `timed`/`Path`, local postgwas imports):

```python
def _load_gene_map(gene_map, sep="\t") -> dict:
    """Load a long-format gene->cis-SNP map (columns ``gene``, ``snp``) into a dict.

    Accepts a path (str/Path) to a TSV with ``gene`` and ``snp`` columns (one row
    per gene–SNP pair) or an already-built ``dict[str, list[str]]`` (returned
    as-is). Raises a friendly ``ValueError`` if the columns are missing.
    """
    import pandas as pd
    if isinstance(gene_map, dict):
        return gene_map
    df = pd.read_csv(str(gene_map), sep=sep)
    if "gene" not in df.columns or "snp" not in df.columns:
        raise ValueError(
            f"gene-map file must have 'gene' and 'snp' columns (one row per "
            f"gene–cis-SNP pair); got columns {list(df.columns)}."
        )
    out: dict[str, list[str]] = {}
    for gene, snp in zip(df["gene"].astype(str), df["snp"].astype(str)):
        out.setdefault(gene, [])
        if snp not in out[gene]:
            out[gene].append(snp)
    return out


def smr(gwas, eqtl, gene_map, *, output=None, eqtl_p_threshold=5e-8,
        smr_p_threshold=0.05, heidi_p_threshold=0.05, heidi_max_snps=20,
        sep="\t") -> "SMRRun":
    """SMR + HEIDI: test whether a gene's expression mediates the GWAS signal.

    Parameters
    ----------
    gwas, eqtl
        Paths to GWAS and eQTL summary-statistics TSVs (columns chr/pos/snp/
        a1/a2/beta/se/p). The eQTL file holds all cis-SNPs across genes.
    gene_map
        Path to a long-format TSV (``gene``, ``snp`` columns; one row per
        gene–cis-SNP pair) or a ``dict[str, list[str]]``.
    output
        Optional path to write the per-gene results TSV.
    eqtl_p_threshold, smr_p_threshold, heidi_p_threshold, heidi_max_snps
        SMR / HEIDI thresholds (see :func:`torchgenomics.postgwas.smr_heidi`).

    Returns
    -------
    SMRRun
    """
    from ._results import SMRRun
    from ..postgwas import load_sumstats, smr_heidi
    import pandas as pd
    from pathlib import Path

    g = load_sumstats(str(gwas), sep=sep)
    e = load_sumstats(str(eqtl), sep=sep)
    gm = _load_gene_map(gene_map, sep=sep)

    with timed() as elapsed:
        summ = smr_heidi(g, e, gm, eqtl_p_threshold=eqtl_p_threshold,
                         smr_p_threshold=smr_p_threshold,
                         heidi_p_threshold=heidi_p_threshold,
                         heidi_max_snps=heidi_max_snps)
        rows = [{
            "gene_id": r.gene_id, "probe_snp": r.probe_snp,
            "beta_smr": r.beta_smr, "se_smr": r.se_smr, "p_smr": r.p_smr,
            "chi2_smr": r.chi2_smr, "beta_gwas": r.beta_gwas, "beta_eqtl": r.beta_eqtl,
            "p_heidi": r.p_heidi, "n_heidi_snps": r.n_heidi_snps, "heidi_stat": r.heidi_stat,
        } for r in summ.results]
        df = pd.DataFrame(rows)
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        return SMRRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            n_genes_tested=int(summ.n_genes_tested),
            n_significant_smr=int(summ.n_significant_smr),
            n_pass_heidi=int(summ.n_pass_heidi),
        )
```

Note: verify the exact `timed()` import/usage and `output_files` idiom against `mr()`/`meta()` in the same file and match it. Add `smr` to `api/__init__.py` imports + `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "api_smr" -v`
Expected: PASS. If the fixture doesn't produce a significant gene, strengthen the shared cis signal (higher z) — do NOT weaken assertions. If `smr_heidi` needs `n` or `af` columns the fixture already provides them.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py tests/test_api_smr_mrmega.py
git commit -m "api: smr() — SMR + HEIDI integration (gene-map TSV -> SMRRun)"
```

---

### Task 3: `api.mr_mega()`

**Files:** Modify `torchgenomics/api/postgwas.py`, `torchgenomics/api/__init__.py`; Test `tests/test_api_smr_mrmega.py`.

**Interfaces:** Consumes `MRMegaRun` (Task 1), `postgwas.load_sumstats`, `postgwas.mr_mega`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_smr_mrmega.py (append)
def _write_mrmega_fixture(tmp_path, n_pops=4, n_snp=30, seed=1):
    """n_pops ancestry sumstats over the same SNPs with a shared causal effect."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    true = rng.normal(0.2, 0.05, size=n_snp)          # shared causal effects
    paths = []
    for p in range(n_pops):
        beta = true + rng.normal(0, 0.02, size=n_snp)  # ancestry-perturbed
        df = pd.DataFrame({
            "chr": [1] * n_snp, "pos": list(range(1, n_snp + 1)),
            "snp": [f"rs{i}" for i in range(n_snp)], "a1": ["A"] * n_snp,
            "a2": ["G"] * n_snp, "beta": beta, "se": [0.02] * n_snp,
            "p": [0.001] * n_snp, "n": [5000] * n_snp, "af": [0.3] * n_snp,
        })
        fp = tmp_path / f"pop{p}.tsv"; df.to_csv(fp, sep="\t", index=False)
        paths.append(str(fp))
    return paths


def test_api_mr_mega_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import MRMegaRun
    paths = _write_mrmega_fixture(tmp_path, n_pops=4)
    r = tg.mr_mega(paths, n_axes=1, output=str(tmp_path / "mrmega.tsv"))
    assert isinstance(r, MRMegaRun)
    assert r.n_populations == 4 and r.n_snps > 0
    assert "p_meta" in r.results.columns
    assert (tmp_path / "mrmega.tsv").exists()


def test_api_mr_mega_too_few_populations(tmp_path):
    import torchgenomics as tg, pytest
    paths = _write_mrmega_fixture(tmp_path, n_pops=2)
    with pytest.raises(ValueError) as e:
        tg.mr_mega(paths, n_axes=1)   # needs >= n_axes + 2 = 3
    assert "population" in str(e.value).lower() or "sumstats" in str(e.value).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "api_mr_mega" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
def mr_mega(sumstats, *, output=None, n_axes=4, random_effects=False,
            sep="\t") -> "MRMegaRun":
    """MR-MEGA: multi-ancestry meta-regression of SNP effects on ancestry axes.

    Parameters
    ----------
    sumstats
        A list of ≥ ``n_axes + 2`` sumstats TSV paths, one per ancestry.
    output
        Optional path to write the per-SNP results TSV.
    n_axes
        Number of ancestry principal axes to fit.
    random_effects
        Whether to use the random-effects variant.

    Returns
    -------
    MRMegaRun
    """
    from ._results import MRMegaRun
    from ..postgwas import load_sumstats, mr_mega as _mr_mega
    import pandas as pd
    from pathlib import Path

    if isinstance(sumstats, (str, Path)):
        raise ValueError(
            "mr_mega needs a list of sumstats paths (one per ancestry), not a "
            "single path."
        )
    paths = list(sumstats)
    need = int(n_axes) + 2
    if len(paths) < need:
        raise ValueError(
            f"MR-MEGA with n_axes={n_axes} needs at least {need} populations "
            f"(sumstats files); got {len(paths)}."
        )
    ss_list = [load_sumstats(str(p), sep=sep) for p in paths]

    with timed() as elapsed:
        res = _mr_mega(ss_list, n_axes=int(n_axes), random_effects=bool(random_effects))

        def _col(x):
            try:
                return [float(v) for v in x.tolist()]
            except Exception:
                return list(x)

        df = pd.DataFrame({
            "beta_meta": _col(res.beta_meta), "se_meta": _col(res.se_meta),
            "p_meta": _col(res.p_meta), "p_heterogeneity": _col(res.p_heterogeneity),
            "p_ancestry": _col(res.p_ancestry), "p_residual": _col(res.p_residual),
            "log10_bf": _col(res.log10_bf), "posterior_effect": _col(res.posterior_effect),
        })
        output_files: dict[str, Path] = {}
        if output is not None:
            df.to_csv(str(output), sep="\t", index=False)
            output_files["tsv"] = Path(str(output))
        pmeta = df["p_meta"].dropna()
        return MRMegaRun(
            runtime_s=elapsed(), output_files=output_files, results=df,
            method=str(res.method), n_axes=int(res.n_axes),
            n_populations=int(res.n_populations), n_snps=int(res.n_snps),
            min_p_meta=float(pmeta.min()) if len(pmeta) else None,
        )
```

Note: verify `MultiAncestryResult`'s per-SNP fields are array-like (tensors) — the `_col` helper coerces them; if any field is a scalar, adjust. Add `mr_mega` to `api/__init__.py` imports + `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -k "api_mr_mega" -v`
Expected: PASS. Then `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_smr_mrmega.py -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py tests/test_api_smr_mrmega.py
git commit -m "api: mr_mega() — multi-ancestry MR meta-analysis (list of sumstats -> MRMegaRun)"
```

---

### Task 4: CLI `smr` + `mr-mega` subcommands

**Files:** Modify `torchgenomics/cli.py`, `torchgenomics/_manifest/__init__.py`, `CLAUDE.md`; Test `tests/test_cli_smr_mrmega.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_smr_mrmega.py
import subprocess
import sys


def test_cli_smr_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "smr", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--gwas" in out.stdout and "--eqtl" in out.stdout and "--gene-map" in out.stdout


def test_cli_mr_mega_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "mr-mega", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--sumstats" in out.stdout and "--n-axes" in out.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_smr_mrmega.py -v`
Expected: FAIL (subcommands missing).

- [ ] **Step 3: Write minimal implementation**

In `torchgenomics/cli.py`, mirror `_add_mr_parser`/`_cmd_mr`:

```python
def _add_smr_parser(subparsers):
    p = subparsers.add_parser("smr", help="SMR + HEIDI (gene expression -> GWAS mediation).")
    p.add_argument("--gwas", required=True, help="GWAS sumstats TSV.")
    p.add_argument("--eqtl", required=True, help="eQTL sumstats TSV (all cis-SNPs).")
    p.add_argument("--gene-map", required=True,
                   help="Long-format TSV with gene,snp columns (one row per cis pair).")
    p.add_argument("--eqtl-p-threshold", type=float, default=5e-8)
    p.add_argument("--smr-p-threshold", type=float, default=0.05)
    p.add_argument("--heidi-p-threshold", type=float, default=0.05)
    p.add_argument("--heidi-max-snps", type=int, default=20)
    p.add_argument("--output", default=None, help="Output per-gene results TSV.")


def _cmd_smr(args) -> int:
    from .api import smr
    r = smr(args.gwas, args.eqtl, args.gene_map, output=args.output,
            eqtl_p_threshold=args.eqtl_p_threshold, smr_p_threshold=args.smr_p_threshold,
            heidi_p_threshold=args.heidi_p_threshold, heidi_max_snps=args.heidi_max_snps)
    print(r.summary())
    return 0


def _add_mr_mega_parser(subparsers):
    p = subparsers.add_parser("mr-mega", help="Multi-ancestry MR meta-analysis (MR-MEGA).")
    p.add_argument("--sumstats", required=True, nargs="+",
                   help="One sumstats TSV per ancestry (>= n_axes + 2).")
    p.add_argument("--n-axes", type=int, default=4)
    p.add_argument("--random-effects", action="store_true", default=False)
    p.add_argument("--output", default=None, help="Output per-SNP results TSV.")


def _cmd_mr_mega(args) -> int:
    from .api import mr_mega
    r = mr_mega(args.sumstats, output=args.output, n_axes=args.n_axes,
                random_effects=args.random_effects)
    print(r.summary())
    return 0
```

Register in `build_parser` (`_add_smr_parser(subparsers)` / `_add_mr_mega_parser(subparsers)`) + the dispatch dict (`"smr": _cmd_smr, "mr-mega": _cmd_mr_mega`). Add `"smr"`, `"mr-mega"` to `_TIER_2_CLI_SUBCOMMANDS`. In `CLAUDE.md`, bump the CLI count (48 → 50) + add example lines:
```bash
torchgenomics smr --gwas gwas.tsv --eqtl eqtl.tsv --gene-map cis_map.tsv --output smr.tsv
torchgenomics mr-mega --sumstats pop1.tsv pop2.tsv pop3.tsv --n-axes 1 --output mrmega.tsv
```

Note: `smr`/`mr-mega` are now in `_TIER_2_CLI_SUBCOMMANDS` AND have real `api.smr`/`api.mr_mega` functions — the real functions win (module attr beats `__getattr__`). Verify `api.smr` is the real function, not the CLI-runner.

- [ ] **Step 4: Run test to verify it passes**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_smr_mrmega.py -q`
Expected: PASS. Also run a drift check: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/ -k "manifest or drift or cli_bridge" -q` (update any hardcoded count that legitimately changed). Confirm `TORCHGENOMICS_DISABLE_NATIVE=1 python -c "import torchgenomics.api as a; print(a.smr.__name__, a.mr_mega.__name__)"` prints `smr mr_mega` (real functions, not partials).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/cli.py torchgenomics/_manifest/__init__.py CLAUDE.md tests/test_cli_smr_mrmega.py
git commit -m "cli: smr + mr-mega subcommands (share the api core); manifest + docs"
```

---

### Task 5: R `tg_smr` + `tg_mr_mega` wrappers + S4 result classes

**Files:** Modify `rTorchGenomics/R/api.R`, `rTorchGenomics/R/result_classes.R`, `rTorchGenomics/NAMESPACE`, `rTorchGenomics/_pkgdown.yml`, `rTorchGenomics/man/*.Rd`; Test `rTorchGenomics/tests/testthat/test-smr-mrmega.R`.

**IMPORTANT:** author/modify R via heredoc/`sed`; parse-check each; run tests with `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-smr-mrmega.R")'`.

- [ ] **Step 1: Write the failing test** (mocked, using the file's `mockery::stub` idiom)

```r
# rTorchGenomics/tests/testthat/test-smr-mrmega.R
test_that("tg_smr forwards args and wraps into SMRRun", {
  captured <- NULL
  mockery::stub(tg_smr, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(n_genes_tested = 5L, n_significant_smr = 2L, n_pass_heidi = 1L, results = list())
  })
  r <- tg_smr("gwas.tsv", "eqtl.tsv", "map.tsv")
  expect_equal(captured$fn, "smr")
  expect_equal(captured$args$gwas, "gwas.tsv")
  expect_equal(captured$args$gene_map, "map.tsv")
  expect_s4_class(r, "SMRRun")
})

test_that("tg_mr_mega forwards a sumstats vector and wraps into MRMegaRun", {
  captured <- NULL
  mockery::stub(tg_mr_mega, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(method = "mr_mega", n_axes = 1L, n_populations = 3L, n_snps = 40L,
         min_p_meta = 1e-8, results = list())
  })
  r <- tg_mr_mega(c("p1.tsv", "p2.tsv", "p3.tsv"), n_axes = 1L)
  expect_equal(captured$fn, "mr_mega")
  expect_equal(captured$args$sumstats, c("p1.tsv", "p2.tsv", "p3.tsv"))
  expect_s4_class(r, "MRMegaRun")
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-smr-mrmega.R")'`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

Add S4 classes to `rTorchGenomics/R/result_classes.R` (mirror `MRRun`; reuse `%||%`/`.as_int`/`.as_num`/`.as_chr`/`.tibble_from_list_of_dicts`; add `show()`):

```r
setClass("SMRRun", contains = "_BaseRun",
  representation(n_genes_tested = "integer", n_significant_smr = "integer",
                 n_pass_heidi = "integer", results = "data.frame", raw = "list"))
SMRRun <- list(new_from_dict = function(d) {
  new("SMRRun",
      n_genes_tested = .as_int(d$n_genes_tested %||% 0L),
      n_significant_smr = .as_int(d$n_significant_smr %||% 0L),
      n_pass_heidi = .as_int(d$n_pass_heidi %||% 0L),
      results = .tibble_from_list_of_dicts(d$results),
      output_files = .as_list(d$output_files %||% list()), raw = d)
})

setClass("MRMegaRun", contains = "_BaseRun",
  representation(method = "character", n_axes = "integer", n_populations = "integer",
                 n_snps = "integer", min_p_meta = "numeric",
                 results = "data.frame", raw = "list"))
MRMegaRun <- list(new_from_dict = function(d) {
  new("MRMegaRun",
      method = .as_chr(d$method %||% ""), n_axes = .as_int(d$n_axes %||% 0L),
      n_populations = .as_int(d$n_populations %||% 0L), n_snps = .as_int(d$n_snps %||% 0L),
      min_p_meta = .as_num(d$min_p_meta %||% NA_real_),
      results = .tibble_from_list_of_dicts(d$results),
      output_files = .as_list(d$output_files %||% list()), raw = d)
})
```

(Match how `MRRun`/`MetaRun` set inherited `_BaseRun` slots + which helpers exist — read `result_classes.R` first and reuse them.)

Add wrappers to `rTorchGenomics/R/api.R` (mirror `tg_mr`/`tg_coloc`):

```r
#' SMR + HEIDI (gene expression -> GWAS mediation).
#' @export
tg_smr <- function(gwas, eqtl, gene_map, output = NULL,
                   eqtl_p_threshold = 5e-8, smr_p_threshold = 0.05,
                   heidi_p_threshold = 0.05, heidi_max_snps = 20L) {
  d <- bridge_call("smr", .compact(list(
    gwas = .as_path(gwas), eqtl = .as_path(eqtl), gene_map = .as_path(gene_map),
    output = .as_path(output), eqtl_p_threshold = eqtl_p_threshold,
    smr_p_threshold = smr_p_threshold, heidi_p_threshold = heidi_p_threshold,
    heidi_max_snps = as.integer(heidi_max_snps))))
  SMRRun$new_from_dict(d)
}

#' Multi-ancestry MR meta-analysis (MR-MEGA).
#' @export
tg_mr_mega <- function(sumstats, output = NULL, n_axes = 4L, random_effects = FALSE) {
  d <- bridge_call("mr_mega", .compact(list(
    sumstats = as.character(sumstats), output = .as_path(output),
    n_axes = as.integer(n_axes), random_effects = isTRUE(random_effects))))
  MRMegaRun$new_from_dict(d)
}
```

Add `export(tg_smr)`, `export(tg_mr_mega)`, `export(SMRRun)`, `export(MRMegaRun)`, `exportClasses(SMRRun)`, `exportClasses(MRMegaRun)` to NAMESPACE (mirror MRRun form). Add all four to `_pkgdown.yml`. Regenerate man: `Rscript -e 'roxygen2::roxygenise()'`.

- [ ] **Step 4: Run test to verify it passes**

Run (from `rTorchGenomics/`): `Rscript -e 'devtools::load_all("."); testthat::test_file("tests/testthat/test-smr-mrmega.R")'` → both pass. Parse-check api.R/result_classes.R. Full R mocked suite: `Rscript -e 'devtools::load_all("."); testthat::test_dir("tests/testthat")'` → no failures.

- [ ] **Step 5: Commit**

```bash
git add rTorchGenomics/R/api.R rTorchGenomics/R/result_classes.R rTorchGenomics/NAMESPACE rTorchGenomics/_pkgdown.yml rTorchGenomics/man rTorchGenomics/tests/testthat/test-smr-mrmega.R
git commit -m "r: tg_smr + tg_mr_mega wrappers + SMRRun/MRMegaRun S4 classes"
```

---

## Self-Review

**Spec coverage:**
- §Component 1 (api.smr) → Task 2. ✅
- §Component 2 (api.mr_mega) → Task 3. ✅
- §Component 3 (SMRRun/MRMegaRun) → Task 1. ✅
- §Component 4 (CLI smr/mr-mega + manifest + docs) → Task 4. ✅
- §Component 5 (R tg_smr/tg_mr_mega + S4 + NAMESPACE/pkgdown) → Task 5. ✅
- §Testing (smr runs + n_genes; bad gene-map; mr_mega runs + too-few-pops; CLI help; R mocked) → Tasks 1-5. ✅

**Placeholder scan:** No TBD/TODO; code steps carry real code. Two verification notes (`timed()`/`output_files` idiom; `MultiAncestryResult` per-SNP fields array-like) are "match the real signature" checks.

**Type consistency:** `SMRRun`/`MRMegaRun` fields set in Task 1 are consumed by `api.smr`/`api.mr_mega` (Tasks 2-3), CLI `.summary()` (Task 4), R `new_from_dict` (Task 5). `smr`/`mr_mega` names consistent across api/CLI/R. `--gene-map` (CLI) → `gene_map` (api) mapping is exact.

**Known verification points for the implementer:**
- (a) Read `mr()`/`coloc()` in `api/postgwas.py` to copy the exact `timed()` import + `output_files` construction — don't invent a variant.
- (b) Confirm `MultiAncestryResult`'s per-SNP fields (`beta_meta` etc.) are tensors/arrays (the `_col` helper coerces via `.tolist()`); if any is scalar, put it on the run object instead of the DataFrame.
- (c) SMR/MR-MEGA fixture calibration: if `smr` yields no significant gene or `mr_mega` errors on the tiny fixture, strengthen the fixture (stronger shared signal / more SNPs / more pops), never weaken the assertion; report DONE_WITH_CONCERNS if a method genuinely can't run on a small fixture.
- (d) `smr`/`mr-mega` in `_TIER_2_CLI_SUBCOMMANDS` + real `api.smr`/`api.mr_mega` — confirm the real functions win over the `__getattr__` CLI-runner (module attribute precedence); the Task-4 check verifies `a.smr.__name__ == "smr"`.
- (e) Match the exact R `new_from_dict`/`.tibble_from_list_of_dicts`/`.as_*` idioms in `result_classes.R`.
