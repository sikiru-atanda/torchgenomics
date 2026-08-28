# Wire SuSiE-RSS `finemap` (api/CLI/R) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a friendly `finemap` surface — `api.finemap` (Python), a `finemap` CLI subcommand, and `tg_finemap` (R) — that returns a rich fine-mapping result object by thinly wrapping the validated `bayes-scan-rss` SuSiE-RSS pipeline.

**Architecture:** `api.finemap` runs `bayes-scan-rss` via the existing in-process CLI-runner bridge (`run_cli_subcommand`) and parses its output TSV into a rich `FineMapRun` (per-variant PIP table + per-credible-set summary). No re-implementation, no new statistics; `bayes-scan-rss` and the engine are untouched. Mirrors the `smr` hand-crafted pattern (api + CLI + R + `.HANDCRAFTED_COMMANDS`).

**Tech Stack:** Python (pandas, tempfile), argparse CLI, R (S4 + reticulate bridge, testthat/mockery).

## Global Constraints

- **No new statistics / no re-implementation.** Delegate everything to `bayes-scan-rss` via `run_cli_subcommand`; do NOT edit `torchgenomics/models/bayesian_vs_rss.py`, `bayes-scan-rss`'s handler/parser, or `write_results_tsv`.
- **NO MISMATCH (hard).** `finemap`'s output TSV MUST be byte-identical to a direct `bayes-scan-rss` TSV on identical inputs, and `FineMapRun.results` MUST equal a re-read of that TSV. Locked by a byte-identity parity test with `threads=1` (multi-thread FP order is the only non-determinism source and could flip a 6th sig-fig under `:.6g`). No numeric-tolerance compare.
- **Clarity/transparency.** Help text, docstring, and `summary()` state `finemap` is the friendly front-end over the same engine as `bayes-scan-rss`. Per-CS columns named `lead_pip`/`sum_pip` (what the TSV provides), NOT a claimed exact SuSiE coverage.
- **Additive.** `bayes-scan-rss` and `tg_bayes_scan_rss` stay exactly as-is. Not an MCP tool (no `@tool`). Verify under `TORCHGENOMICS_DISABLE_NATIVE=1`.
- `finemap` gets the `smr` treatment: in `_TIER_2_CLI_SUBCOMMANDS` AND a real hand-crafted `api.finemap` (wins over `__getattr__`) AND a hand-crafted `tg_finemap` AND `"finemap"` in `.HANDCRAFTED_COMMANDS` (so codegen won't emit a competing wrapper).

---

## File Structure

- `torchgenomics/api/_results.py` — add `FineMapRun` (Task 1).
- `torchgenomics/api/postgwas.py` — add `finemap` + `_summarize_credible_sets` (Task 2).
- `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py` — export `finemap` + `FineMapRun`.
- `torchgenomics/cli.py` — `_add_finemap_parser` + `_cmd_finemap` + registration (Task 3).
- `torchgenomics/_manifest/__init__.py` — add `"finemap"` to `_TIER_2_CLI_SUBCOMMANDS` (Task 3).
- `CLAUDE.md` — CLI count 54→55 + one example (Task 3).
- `rTorchGenomics/R/api.R`, `R/result_classes.R`, `R/codegen.R`, `NAMESPACE`, `_pkgdown.yml`, `man/*.Rd`, `inst/rbridge_manifest.json`, `R/api_auto.R`, `tests/testthat/test-finemap.R` (Task 4).
- Tests: `tests/test_api_finemap.py`, `tests/test_cli_finemap.py`.

---

## Task 1: `FineMapRun` result class

**Files:**
- Modify: `torchgenomics/api/_results.py` (after `HessRun`, or after the last postgwas Run)
- Modify: `torchgenomics/api/__init__.py` (imports + `__all__`)
- Test: `tests/test_api_finemap.py` (new)

**Interfaces:**
- Produces: `FineMapRun(_BaseRun)` with `n_variants:int, n_credible_sets:int, n_variants_in_credible_sets:int, results:pd.DataFrame, credible_sets:pd.DataFrame`, `_kind="finemap"`, `summary()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api_finemap.py
import pandas as pd


def test_finemaprun_shape_and_summary():
    from torchgenomics.api import FineMapRun
    r = FineMapRun(
        n_variants=5, n_credible_sets=1, n_variants_in_credible_sets=2,
        results=pd.DataFrame({"SNP": ["rs0"], "PIP": [0.9], "CREDIBLE_SET": [1]}),
        credible_sets=pd.DataFrame({"credible_set": [1], "n_snps": [2],
                                    "lead_snp": ["rs0"], "lead_pip": [0.9],
                                    "sum_pip": [1.1]}),
    )
    s = r.summary()
    assert "Fine-mapping" in s and "SuSiE-RSS" in s
    d = r.to_dict()
    assert d["n_credible_sets"] == 1 and d["n_variants_in_credible_sets"] == 2
```

- [ ] **Step 2: Run it, expect ImportError**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_finemap.py::test_finemaprun_shape_and_summary -q`
Expected: FAIL (cannot import FineMapRun).

- [ ] **Step 3: Add the dataclass** to `_results.py` (mirror `SMRRun`/`EnrichmentRun` two-frame pattern):

```python
@dataclass
class FineMapRun(_BaseRun):
    """Result of :func:`torchgenomics.api.finemap` — SuSiE-RSS credible sets.

    Produced by wrapping the validated ``bayes-scan-rss`` pipeline; ``results`` is
    the PolyFun-format per-variant table (byte-identical to that command's TSV) and
    ``credible_sets`` is a per-set summary derived from it.
    """
    n_variants: int = 0
    n_credible_sets: int = 0
    n_variants_in_credible_sets: int = 0
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    credible_sets: pd.DataFrame = field(default_factory=pd.DataFrame)

    _kind: ClassVar[str] = "finemap"

    def summary(self) -> str:
        lines = [
            f"Fine-mapping (SuSiE-RSS via bayes-scan-rss) — {self.n_variants} variants, "
            f"{self.n_credible_sets} credible sets, "
            f"{self.n_variants_in_credible_sets} variants in a credible set",
        ]
        if not self.credible_sets.empty:
            lines.append(self.credible_sets.head(10).to_string(index=False))
        lines.append(f"Runtime: {self.runtime_s:.1f}s")
        return "\n".join(lines)
```

- [ ] **Step 4: Export from `api/__init__.py`** — add `FineMapRun` to the `from ._results import (...)` block and to `__all__`.

- [ ] **Step 5: Run the test, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_finemap.py::test_finemaprun_shape_and_summary -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/_results.py torchgenomics/api/__init__.py tests/test_api_finemap.py
git commit -m "api: FineMapRun result class (SuSiE-RSS credible sets)"
```

---

## Task 2: `api.finemap` + `_summarize_credible_sets`

**Files:**
- Modify: `torchgenomics/api/postgwas.py` (add `_summarize_credible_sets` near the other helpers; `finemap` after the last postgwas api fn)
- Modify: `torchgenomics/api/__init__.py`, `torchgenomics/__init__.py` (export `finemap`)
- Test: `tests/test_api_finemap.py`

**Interfaces:**
- Consumes: `api._cli_bridge.run_cli_subcommand`, `timed`, `FineMapRun`.
- Produces: `api.finemap(sumstats, ld_ref=None, *, geno=None, regions=None, prior_pi=None, max_num_causal=10, coverage=0.95, purity=0.5, block_size_threshold=5000, output=None, threads=4) -> FineMapRun`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_api_finemap.py`). This includes the fixture builder (reused from `tests/test_cli.py:197`'s proven pattern), the end-to-end run, the friendly-error guards, and the no-output temp path. The PARITY test lives in Task 5 wording but write the fixture helper here.

```python
def _write_finemap_fixture(tmp_path, p=5):
    """Sumstats TSV (uppercase schema) + a .pt LD reference carrying SNP ids."""
    import torch
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata
    sumstats_path = tmp_path / "sumstats.tsv"
    sumstats = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(p)], "CHR": [22] * p,
        "BP": [1000 + i * 10 for i in range(p)], "A1": ["A"] * p, "A2": ["G"] * p,
        "BETA": [0.05, 0.03, 0.6, 0.02, 0.04][:p], "SE": [0.05, 0.04, 0.05, 0.04, 0.05][:p],
        "N": [1000] * p,
    })
    sumstats.to_csv(sumstats_path, sep="\t", index=False)
    ld_path = tmp_path / "ld.pt"
    R = torch.eye(p, dtype=torch.float64)
    meta = LDReferenceMetadata(cohort_id="test", n=1000, build="GRCh38",
                               panel_provenance="synthetic")
    save_ld_reference(ld_path, R, sumstats["SNP"].tolist(), meta)
    return str(sumstats_path), str(ld_path)


def test_api_finemap_runs(tmp_path):
    import torchgenomics as tg
    from torchgenomics.api import FineMapRun
    ss, ld = _write_finemap_fixture(tmp_path)
    out = tmp_path / "fm.tsv"
    r = tg.finemap(ss, ld, max_num_causal=2, threads=1, output=str(out))
    assert isinstance(r, FineMapRun)
    assert r.n_variants == 5
    assert list(r.results.columns) == [
        "SNP", "CHR", "BP", "A1", "A2", "Z", "N", "PIP", "BETA_MEAN", "BETA_SD",
        "CREDIBLE_SET"]
    assert list(r.credible_sets.columns) == [
        "credible_set", "n_snps", "lead_snp", "lead_pip", "sum_pip"]
    assert r.n_variants_in_credible_sets == int((r.results["CREDIBLE_SET"] > 0).sum())
    assert out.exists()


def test_api_finemap_no_output_temp(tmp_path):
    import torchgenomics as tg
    ss, ld = _write_finemap_fixture(tmp_path)
    r = tg.finemap(ss, ld, max_num_causal=2, threads=1)  # output=None
    assert r.n_variants == 5 and not r.results.empty
    assert r.output_files == {}  # temp dir cleaned; nothing persisted


def test_api_finemap_needs_ld(tmp_path):
    import torchgenomics as tg, pytest
    ss, _ = _write_finemap_fixture(tmp_path)
    with pytest.raises(ValueError):
        tg.finemap(ss)  # no ld_ref, no geno
```

- [ ] **Step 2: Run, expect FAIL** (`tg.finemap` missing)

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_finemap.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `_summarize_credible_sets` + `finemap`** in `postgwas.py`. Add `FineMapRun` to the `from ._results import (...)` line.

```python
def _summarize_credible_sets(df) -> "pd.DataFrame":
    """Per-credible-set summary derived from the finemap output TSV.

    Columns: credible_set, n_snps, lead_snp, lead_pip, sum_pip. Named for exactly
    what the TSV provides — lead_pip is the max per-variant PIP in the set and
    sum_pip is the PIP sum; neither is a claimed SuSiE per-layer coverage (the
    per-layer alpha is not written to the TSV).
    """
    rows = []
    for cs in sorted(int(c) for c in df["CREDIBLE_SET"].unique() if int(c) > 0):
        sub = df[df["CREDIBLE_SET"] == cs]
        lead = sub.loc[sub["PIP"].idxmax()]
        rows.append({
            "credible_set": int(cs), "n_snps": int(len(sub)),
            "lead_snp": str(lead["SNP"]), "lead_pip": float(lead["PIP"]),
            "sum_pip": float(sub["PIP"].sum()),
        })
    return pd.DataFrame(rows, columns=["credible_set", "n_snps", "lead_snp",
                                       "lead_pip", "sum_pip"])


def finemap(sumstats, ld_ref=None, *, geno=None, regions=None, prior_pi=None,
            max_num_causal=10, coverage=0.95, purity=0.5,
            block_size_threshold=5000, output=None, threads=4) -> "FineMapRun":
    """SuSiE-RSS fine-mapping — friendly rich-result front-end over ``bayes-scan-rss``.

    Runs the validated ``bayes-scan-rss`` SuSiE-RSS pipeline unchanged (identical
    inputs, identical output) and returns a rich :class:`FineMapRun` with a
    per-variant PIP table and a per-credible-set summary. See ``bayes-scan-rss`` for
    input formats: uppercase ``SNP/CHR/BP/A1/A2`` + ``Z`` (or ``BETA/SE/N``) sumstats;
    a ``.pt``/``.npz`` LD reference carrying SNP ids; index-based ``start,stop`` regions.
    """
    import tempfile
    from ._cli_bridge import run_cli_subcommand

    if ld_ref is None and geno is None:
        raise ValueError(
            "finemap needs an LD reference: pass ld_ref=<.pt/.npz> (carrying SNP "
            "ids). (geno= in-sample LD is not yet wired upstream.)"
        )

    def _run(out_path):
        return run_cli_subcommand(
            "bayes-scan-rss",
            sumstats=str(sumstats),
            ld_ref=(str(ld_ref) if ld_ref is not None else None),
            geno=(str(geno) if geno is not None else None),
            regions=(str(regions) if regions is not None else None),
            prior_pi=(str(prior_pi) if prior_pi is not None else None),
            max_num_causal=int(max_num_causal),
            coverage=float(coverage),
            purity=float(purity),
            block_size_threshold=int(block_size_threshold),
            output=out_path,
            threads=int(threads),
        )

    with timed() as elapsed:
        output_files: dict[str, Path] = {}
        if output is not None:
            _run(str(output))
            df = pd.read_csv(str(output), sep="\t")
            output_files["tsv"] = Path(str(output))
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp = str(Path(td) / "finemap.tsv")
                _run(tmp)
                df = pd.read_csv(tmp, sep="\t")
        cs_df = _summarize_credible_sets(df)
        return FineMapRun(
            runtime_s=elapsed(), output_files=output_files,
            results=df, credible_sets=cs_df,
            n_variants=int(len(df)),
            n_credible_sets=int(cs_df.shape[0]),
            n_variants_in_credible_sets=int((df["CREDIBLE_SET"] > 0).sum()),
        )
```

- [ ] **Step 4: Export** `finemap` from `api/__init__.py` (imports + `__all__`) and `torchgenomics/__init__.py` top-level (mirror how `smr` appears).

- [ ] **Step 5: Run, expect PASS**

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_api_finemap.py -q`
Expected: PASS (result-class test + 3 finemap tests).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/api/postgwas.py torchgenomics/api/__init__.py torchgenomics/__init__.py tests/test_api_finemap.py
git commit -m "api: finemap() — rich SuSiE-RSS front-end over bayes-scan-rss"
```

---

## Task 3: CLI `finemap` + manifest + no-mismatch parity test

**Files:**
- Modify: `torchgenomics/cli.py` (`_add_finemap_parser`, `_cmd_finemap`, register in `_build_parser` + dispatch dict)
- Modify: `torchgenomics/_manifest/__init__.py` (add `"finemap"` to `_TIER_2_CLI_SUBCOMMANDS`)
- Modify: `CLAUDE.md` (count 54→55 + example)
- Test: `tests/test_cli_finemap.py` (new), and the PARITY test appended to `tests/test_api_finemap.py`

**Interfaces:**
- Consumes: `api.finemap`.
- Produces: CLI subcommand `finemap`.

- [ ] **Step 1: Write the failing tests**

Parity test (append to `tests/test_api_finemap.py` — reuses `_write_finemap_fixture`):

```python
def test_finemap_byte_identical_to_bayes_scan_rss(tmp_path):
    """NO-MISMATCH: finemap's TSV is byte-identical to a direct bayes-scan-rss run,
    and the rich table equals a re-read of that TSV. threads=1 for determinism."""
    import filecmp
    import torchgenomics as tg
    from torchgenomics.api._cli_bridge import run_cli_subcommand
    ss, ld = _write_finemap_fixture(tmp_path)
    a = tmp_path / "direct.tsv"
    b = tmp_path / "finemap.tsv"
    run_cli_subcommand("bayes-scan-rss", sumstats=ss, ld_ref=ld,
                       max_num_causal=2, threads=1, output=str(a))
    r = tg.finemap(ss, ld, max_num_causal=2, threads=1, output=str(b))
    assert filecmp.cmp(str(a), str(b), shallow=False), "finemap TSV != bayes-scan-rss TSV"
    pd.testing.assert_frame_equal(r.results, pd.read_csv(str(b), sep="\t"))
```

CLI tests (`tests/test_cli_finemap.py`):

```python
import subprocess, sys


def test_cli_finemap_help():
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "finemap", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--sumstats" in r.stdout and "--ld-ref" in r.stdout


def test_cli_finemap_end_to_end(tmp_path):
    import pandas as pd, torch
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata
    p = 5
    ss = tmp_path / "s.tsv"
    pd.DataFrame({"SNP": [f"rs{i}" for i in range(p)], "CHR": [22]*p,
                  "BP": [1000+i*10 for i in range(p)], "A1": ["A"]*p, "A2": ["G"]*p,
                  "BETA": [0.05,0.03,0.6,0.02,0.04], "SE": [0.05,0.04,0.05,0.04,0.05],
                  "N": [1000]*p}).to_csv(ss, sep="\t", index=False)
    ld = tmp_path / "ld.pt"
    save_ld_reference(ld, torch.eye(p, dtype=torch.float64), [f"rs{i}" for i in range(p)],
                      LDReferenceMetadata(cohort_id="t", n=1000, build="GRCh38",
                                          panel_provenance="syn"))
    out = tmp_path / "o.tsv"
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "finemap",
                        "--sumstats", str(ss), "--ld-ref", str(ld),
                        "--max-num-causal", "2", "--threads", "1", "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
```

- [ ] **Step 2: Run, expect FAIL** (unknown subcommand `finemap` / parity depends on Task 2 which exists — the CLI tests fail on unknown subcommand)

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_finemap.py tests/test_api_finemap.py::test_finemap_byte_identical_to_bayes_scan_rss -q`
Expected: CLI tests FAIL (unknown subcommand); parity test should already PASS (api.finemap exists from Task 2) — that's fine, it locks the invariant.

- [ ] **Step 3: Add parser + command** (mirror `_add_smr_parser`/`_cmd_smr`). Parser:

```python
def _add_finemap_parser(subparsers):
    p = subparsers.add_parser(
        "finemap",
        help=("SuSiE-RSS fine-mapping (friendly front-end over the same engine as "
              "bayes-scan-rss); prints a credible-set summary."))
    p.add_argument("--sumstats", required=True,
                   help="Sumstats TSV (SNP,CHR,BP,A1,A2 + Z or BETA/SE/N).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ld-ref", help="Pre-built LD reference (.pt or .npz, carrying SNP ids).")
    g.add_argument("--geno", help="Genotype panel for in-sample LD (not yet wired upstream).")
    p.add_argument("--regions", default=None,
                   help="Block regions TSV with start,stop INDEX columns.")
    p.add_argument("--prior-pi", default=None,
                   help="Scalar prior inclusion prob OR path to a PRIOR_PI file.")
    p.add_argument("--max-num-causal", type=int, default=10)
    p.add_argument("--coverage", type=float, default=0.95)
    p.add_argument("--purity", type=float, default=0.5)
    p.add_argument("--block-size-threshold", type=int, default=5000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", default=None, help="Optional output TSV path.")
```

Command:

```python
def _cmd_finemap(args):
    """SuSiE-RSS fine-mapping (thin friendly front-end over api.finemap/bayes-scan-rss)."""
    from .api import finemap
    r = finemap(args.sumstats, ld_ref=args.ld_ref, geno=args.geno,
                regions=args.regions, prior_pi=args.prior_pi,
                max_num_causal=args.max_num_causal, coverage=args.coverage,
                purity=args.purity, block_size_threshold=args.block_size_threshold,
                output=args.output, threads=args.threads)
    print(r.summary())
    return 0
```

Register `_add_finemap_parser(subparsers)` in `_build_parser` (next to `_add_smr_parser`) and `"finemap": _cmd_finemap` in the dispatch dict.

- [ ] **Step 4: Manifest + docs.** Add `"finemap"` to `_TIER_2_CLI_SUBCOMMANDS` in `_manifest/__init__.py`. In `CLAUDE.md` bump the CLI count 54→55 and add an example line:

```bash
torchgenomics finemap --sumstats ss.tsv --ld-ref ld.pt --max-num-causal 10 --output finemap.tsv
```

- [ ] **Step 5: Run, expect PASS** + precedence + no-drift + MCP count.

Run: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_cli_finemap.py tests/test_api_finemap.py -q`
Then: `TORCHGENOMICS_DISABLE_NATIVE=1 python -c "import torchgenomics.api as a; assert a.finemap.__name__=='finemap'; print('precedence ok')"`
Then: `TORCHGENOMICS_DISABLE_NATIVE=1 python -m pytest tests/test_manifest.py tests/test_mcp_server.py -q` (no CLI drift; MCP count stays 14 — `finemap` is not `@tool`).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/cli.py torchgenomics/_manifest/__init__.py CLAUDE.md tests/test_cli_finemap.py tests/test_api_finemap.py
git commit -m "cli: finemap subcommand + manifest + no-mismatch parity test"
```

---

## Task 4: R `tg_finemap` + S4 + codegen exclusion

**Files:**
- Modify: `rTorchGenomics/R/api.R` (`tg_finemap`), `R/result_classes.R` (`FineMapRun` setClass + new_from_dict + show), `R/codegen.R` (`.HANDCRAFTED_COMMANDS`), `NAMESPACE`, `_pkgdown.yml`
- Create: `man/tg_finemap.Rd`, `man/FineMapRun.Rd` (roxygen), `tests/testthat/test-finemap.R`
- Regenerate: `inst/rbridge_manifest.json`, `R/api_auto.R`

**IMPORTANT tooling note:** subagent Write/Edit have been unreliable in this harness — author R changes via Bash heredoc or Edit, and VERIFY each landed by `cat`/`grep` afterward.

**Interfaces:**
- Consumes: `api.finemap` via `bridge_call("finemap", ...)` (reticulate direct).
- Produces: `tg_finemap` + `FineMapRun` S4.

- [ ] **Step 1: Add `tg_finemap`** to `R/api.R` (mirror `tg_smr`):

```r
#' SuSiE-RSS fine-mapping (friendly front-end over bayes-scan-rss)
#' @export
tg_finemap <- function(sumstats, ld_ref = NULL, geno = NULL, regions = NULL,
                       prior_pi = NULL, max_num_causal = 10L, coverage = 0.95,
                       purity = 0.5, block_size_threshold = 5000L,
                       output = NULL, threads = 4L) {
  d <- bridge_call("finemap", .compact(list(
    sumstats = .as_path(sumstats), ld_ref = .as_path(ld_ref), geno = .as_path(geno),
    regions = .as_path(regions), prior_pi = prior_pi,
    max_num_causal = as.integer(max_num_causal), coverage = as.numeric(coverage),
    purity = as.numeric(purity),
    block_size_threshold = as.integer(block_size_threshold),
    output = .as_path(output), threads = as.integer(threads))))
  FineMapRun$new_from_dict(d)
}
```

- [ ] **Step 2: Add `FineMapRun` S4** to `R/result_classes.R` (mirror `EnrichmentRun` two-tibble pattern):

```r
setClass("FineMapRun", contains = "_BaseRun",
  representation(n_variants = "integer", n_credible_sets = "integer",
                n_variants_in_credible_sets = "integer",
                results = "data.frame", credible_sets = "data.frame", raw = "list"))
FineMapRun <- list(new_from_dict = function(d) {
  new("FineMapRun",
    runtime_s = .as_num(d$runtime_s), output_files = .as_list(d$output_files),
    log_excerpt = .as_chr_vec(d$log_excerpt),
    n_variants = .as_int(d$n_variants), n_credible_sets = .as_int(d$n_credible_sets),
    n_variants_in_credible_sets = .as_int(d$n_variants_in_credible_sets),
    results = .tibble_from_list_of_dicts(d$results),
    credible_sets = .tibble_from_list_of_dicts(d$credible_sets %||% list()), raw = d)
})
setMethod("show", "FineMapRun", function(object) {
  cat(sprintf("FineMapRun (n_variants=%d, n_credible_sets=%d, runtime=%.1fs)\n",
              object@n_variants, object@n_credible_sets, object@runtime_s))
  invisible(object)
})
```

- [ ] **Step 3: Fix `.HANDCRAFTED_COMMANDS`** in `R/codegen.R` — add `"finemap"` to the vector (so codegen skips it; `bayes_scan_rss` stays OUT of the list because it IS auto-generated).

- [ ] **Step 4: Update NAMESPACE + `_pkgdown.yml`.** Add `export(tg_finemap)`, `export(FineMapRun)`, `exportClasses(FineMapRun)` — mirror EXACTLY how `SMRRun` appears (verify: the repo exports BOTH the list object `export(SMRRun)` AND `exportClasses(SMRRun)`). Add `tg_finemap` + `FineMapRun` to `_pkgdown.yml` reference (post-GWAS / fine-mapping section). Generate `man/*.Rd` via `Rscript -e 'setwd("rTorchGenomics"); devtools::document()'`.

- [ ] **Step 5: Regenerate manifest + api_auto.** From repo root:

```bash
python -m torchgenomics._manifest > rTorchGenomics/inst/rbridge_manifest.json
Rscript -e 'setwd("rTorchGenomics"); devtools::load_all("."); generate_api_auto()'
grep -E "tg_finemap" rTorchGenomics/R/api_auto.R && echo "FAIL: tg_finemap in api_auto.R" || echo "OK: no tg_finemap in api_auto.R"
grep -c "tg_bayes_scan_rss" rTorchGenomics/R/api_auto.R   # must still be 1 (bayes-scan-rss stays auto)
```
Expected: "OK: no tg_finemap in api_auto.R" AND `tg_bayes_scan_rss` still present (count 1).

- [ ] **Step 6: Write mocked R test** `tests/testthat/test-finemap.R` (mirror `test-smr-mrmega.R`):

```r
test_that("tg_finemap forwards args and wraps into FineMapRun", {
  captured <- NULL
  mockery::stub(tg_finemap, "bridge_call", function(fn, args = list()) {
    captured <<- list(fn = fn, args = args)
    list(n_variants = 5L, n_credible_sets = 1L, n_variants_in_credible_sets = 2L,
         results = list(), credible_sets = list())
  })
  r <- tg_finemap("ss.tsv", ld_ref = "ld.pt", max_num_causal = 2L)
  expect_equal(captured$fn, "finemap")
  expect_equal(captured$args$sumstats, "ss.tsv")
  expect_equal(captured$args$ld_ref, "ld.pt")
  expect_s4_class(r, "FineMapRun")
})
```

Run:
```bash
Rscript -e 'setwd("rTorchGenomics"); devtools::load_all("."); testthat::test_file("tests/testthat/test-finemap.R")'
Rscript -e 'setwd("rTorchGenomics"); devtools::test()'   # full suite, no new failures vs baseline (~275 pass)
```

- [ ] **Step 7: Commit**

```bash
git add rTorchGenomics/
git commit -m "r: tg_finemap wrapper + FineMapRun S4 + codegen exclusion"
```

---

## Self-Review notes (addressed)

- Spec coverage: Task 1 (FineMapRun) → "Result class"; Task 2 (api.finemap + summary helper) → "api layer"; Task 3 (CLI + manifest + docs + PARITY test) → "CLI" + "Testing/no-mismatch"; Task 4 (R + codegen) → "R". All spec sections mapped.
- No-mismatch: the byte-identity parity test (Task 3 Step 1) with `threads=1` locks the hard requirement; do NOT weaken to tolerance.
- Type consistency: `FineMapRun` fields match across `_results.py` (Task 1), the api builder (Task 2), and the R `new_from_dict` slots (Task 4). Column lists (`SNP,CHR,BP,A1,A2,Z,N,PIP,BETA_MEAN,BETA_SD,CREDIBLE_SET` and `credible_set,n_snps,lead_snp,lead_pip,sum_pip`) are identical in the api builder and the tests.
- Codegen: `"finemap"` in `.HANDCRAFTED_COMMANDS` (Task 4) prevents a competing auto-wrapper; `bayes-scan-rss` stays auto (not added to the list).
- Verified upstream: `save_ld_reference(path, R, snp_ids, meta)` + `LDReferenceMetadata(cohort_id, n, build, panel_provenance)` (fixture builder); `run_cli_subcommand` kwargs→flags mapping; the output TSV's 11 columns + `:.6g` formatting (so the rich table is 6-sig-fig, matching the TSV).
