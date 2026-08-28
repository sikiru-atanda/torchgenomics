# Wire SuSiE-RSS Fine-Mapping as a friendly `finemap` (api / CLI / R) — Design

**Date:** 2026-08-28
**Status:** Approved (design)

## Goal

Add a friendly, first-class **`finemap`** surface — `api.finemap` (Python), a `finemap`
CLI subcommand, and `tg_finemap` (R) — that returns a **rich fine-mapping result
object** (per-variant PIP table + per-credible-set summary), by thinly wrapping the
**already-validated `bayes-scan-rss`** SuSiE-RSS pipeline. Today fine-mapping is only
reachable via `bayes-scan-rss` (CLI) / `tg_bayes_scan_rss` (R, returns a bare
`CliRun` = exit code + file path). This adds the friendly name + the rich object.

## Binding global constraints

- **No new statistics, no re-implementation.** `api.finemap` runs `bayes-scan-rss`
  via the existing in-process CLI-runner bridge (`run_cli_subcommand`) and parses its
  output TSV. The SuSiE-RSS engine, `bayes-scan-rss`, and all its input handling
  (uppercase/`Z` sumstats schema, `n=max`, `.pt/.npz` LD-reference carrying SNP IDs,
  index-based regions, `BlockSpec` blocks, prior handling) are **untouched** — so all
  seven scientific-correctness requirements established in review are handled by the
  validated tool itself, not by new code.
- **NO MISMATCH (hard requirement).** `finemap`'s written output TSV MUST be
  byte-identical to `bayes-scan-rss`'s TSV on the same inputs, and the in-memory
  `FineMapRun.results` MUST equal a re-read of that TSV. Both are locked by an
  explicit parity test. Because the TSV formats floats at `:.6g` (6 significant
  figures — `write_results_tsv`, `bayesian_vs_rss.py:919-925`), the rich object is
  6-sig-fig by construction; this is stated transparently (no hidden precision claim).
- **Clarity & transparency.** The `finemap` CLI help, the `FineMapRun.summary()`, and
  the docstrings state explicitly that `finemap` is the friendly front-end over the
  same validated SuSiE-RSS engine as `bayes-scan-rss`. Per-credible-set columns are
  named honestly for what the TSV provides (`lead_pip`, `sum_pip` — NOT a claimed
  exact SuSiE "coverage", since the per-layer alpha is not in the TSV).
- **Additive / backward-compatible.** `bayes-scan-rss` and `tg_bayes_scan_rss` remain
  exactly as they are. `finemap` is a new hand-crafted surface added alongside.
- Not an MCP tool (consistent with the other postgwas wirings). Verify under
  `TORCHGENOMICS_DISABLE_NATIVE=1`.

## Upstream facts the design relies on (verbatim, verified — do not re-derive)

- `run_cli_subcommand(subcommand, **kwargs) -> CliRun` (`api/_cli_bridge.py:65`).
  kwargs → flags: underscores→hyphens; `None`/`False` omitted; `True` → bare flag;
  else `--flag str(val)` (`_cli_bridge.py:17-38`). Raises a friendly `RuntimeError`
  on argparse `SystemExit` or a non-zero exit or a handler exception. `CliRun` has NO
  stdout capture.
- `bayes-scan-rss` args (`cli.py:6797-6854`): `--sumstats` (required), exactly one of
  `--ld-ref`/`--geno` (mutually-exclusive, required; `--geno` currently raises
  `NotImplementedError`), `--regions` (TSV with `start,stop` INDEX columns),
  `--max-num-causal` (int, 10), `--coverage` (float, 0.95), `--purity` (float, 0.5),
  `--prior-pi` (scalar OR path to a `PRIOR_PI` file), `--block-size-threshold` (int,
  5000), `--output` (**required**, one TSV, no sidecars), `--threads` (int, 4).
- Output TSV (`write_results_tsv`, `bayesian_vs_rss.py:891-939`): tab-separated, header,
  columns exactly `SNP, CHR, BP, A1, A2, Z, N, PIP, BETA_MEAN, BETA_SD, CREDIBLE_SET`.
  `Z/PIP/BETA_MEAN/BETA_SD` formatted `:.6g`; `CREDIBLE_SET` = 1-based CS index in
  returned order, `0` = none, first-CS-wins for multi-membership.
- `_BaseRun` (`_results.py:47-76`): `runtime_s`, `output_files`, `log_excerpt`;
  `to_dict()` serializes DataFrames → `list[dict]`. `timed()` (`_helpers.py:61-83`).
- No tempfile helper exists in `api/`; use `tempfile.TemporaryDirectory` directly.
- `.HANDCRAFTED_COMMANDS` in `rTorchGenomics/R/codegen.R` must list a hand-crafted
  command or codegen emits a competing auto-wrapper. `smr`/`mr_mega`/etc. are the
  precedent (in `_TIER_2` AND hand-crafted AND `.HANDCRAFTED_COMMANDS`).

## api layer — `torchgenomics/api/postgwas.py`

```python
def finemap(sumstats, ld_ref=None, *, geno=None, regions=None, prior_pi=None,
            max_num_causal=10, coverage=0.95, purity=0.5,
            block_size_threshold=5000, output=None, threads=4) -> "FineMapRun":
    """SuSiE-RSS fine-mapping — friendly rich-result front-end over ``bayes-scan-rss``.

    Runs the validated ``bayes-scan-rss`` SuSiE-RSS pipeline unchanged (identical
    inputs, identical output) and returns a rich :class:`FineMapRun` with a
    per-variant PIP table and a per-credible-set summary. For byte-for-byte parity
    with ``bayes-scan-rss`` see its documentation for input formats (uppercase
    ``SNP/CHR/BP/A1/A2`` + ``Z`` or ``BETA/SE/N`` sumstats; a ``.pt``/``.npz`` LD
    reference carrying SNP ids; index-based ``start,stop`` regions).
    """
    import tempfile
    from pathlib import Path
    from ._cli_bridge import run_cli_subcommand

    if ld_ref is None and geno is None:
        raise ValueError(
            "finemap needs an LD reference: pass ld_ref=<.pt/.npz> "
            "(carrying SNP ids). (geno= in-sample LD is not yet wired upstream.)"
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
                df = pd.read_csv(tmp, sep="\t")   # parsed into memory before cleanup
        cs_df = _summarize_credible_sets(df)
        return FineMapRun(
            runtime_s=elapsed(), output_files=output_files,
            results=df, credible_sets=cs_df,
            n_variants=int(len(df)),
            n_credible_sets=int(cs_df.shape[0]),
            n_variants_in_credible_sets=int((df["CREDIBLE_SET"] > 0).sum()),
        )


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
```

Notes:
- `prior_pi` passes through as the CLI expects (a scalar string or a path); the
  wrapper does not interpret it.
- A `geno=` value is forwarded and will surface the upstream `NotImplementedError`
  as a friendly `RuntimeError` (from `run_cli_subcommand`) — acceptable and honest.
- Import `FineMapRun` into the `from ._results import (...)` block; export `finemap`
  + `FineMapRun` from `api/__init__.py` (imports + `__all__`) and top-level
  `torchgenomics/__init__.py`.

## Result class — `torchgenomics/api/_results.py`

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

## CLI — `torchgenomics/cli.py`

New `finemap` subcommand, thin front-end over `api.finemap` (the transparency layer:
prints the rich credible-set summary; writes the TSV only if `--output` given).

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

Register `_add_finemap_parser(subparsers)` in `_build_parser` and `"finemap": _cmd_finemap`
in the dispatch dict (next to the other postgwas commands). Add `"finemap"` to
`_TIER_2_CLI_SUBCOMMANDS` (`_manifest/__init__.py`). CLAUDE.md CLI count 54 → 55 + one
example line.

## R — `rTorchGenomics/`

`tg_finemap` (hand-crafted, mirrors `tg_smr`) calling `bridge_call("finemap", ...)` →
`api.finemap` directly (reticulate), + `FineMapRun` S4 with the two-tibble pattern
(`results` + `credible_sets`, the `EnrichmentRun`/`PowerRun` `%||% list()` guard):

```r
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
- `FineMapRun` setClass (contains `_BaseRun`): slots `n_variants="integer"`,
  `n_credible_sets="integer"`, `n_variants_in_credible_sets="integer"`,
  `results="data.frame"`, `credible_sets="data.frame"`, `raw="list"`; `new_from_dict`
  with `credible_sets = .tibble_from_list_of_dicts(d$credible_sets %||% list())`; a
  `show` method. NAMESPACE `export(tg_finemap)` + `exportClasses(FineMapRun)` +
  `export(FineMapRun)` (mirror how `SMRRun` is exported).
- `codegen.R` `.HANDCRAFTED_COMMANDS`: add `"finemap"` so the auto-generator skips it
  (a competing `CliRun` `tg_finemap` would otherwise be emitted once `finemap` is in
  `_TIER_2`). Regenerate `inst/rbridge_manifest.json` + `api_auto.R`; confirm no
  `tg_finemap` in `api_auto.R`. `tg_bayes_scan_rss` stays (its command stays in
  `_TIER_2`). Add man/*.Rd + `_pkgdown.yml` entries + mocked test.

## Testing — the NO-MISMATCH core

`tests/test_api_finemap.py`:
- **Fixture:** a small sumstats TSV (uppercase `SNP,CHR,BP,A1,A2,Z,N`) + a small
  `.pt` LD reference built via `save_ld_reference` (from
  `torchgenomics.postgwas._ld_ref_loader`) so it carries `snp_ids` + valid metadata,
  aligned to the sumstats. Reuse any existing `bayes-scan-rss` test fixture/helper if
  present. Keep it tiny (≤ ~30 SNPs, few IBSS iterations) for speed.
- **Parity (the hard requirement):** run `bayes-scan-rss` directly (via
  `run_cli_subcommand("bayes-scan-rss", ..., output=A)` or `cli.main([...])`) → TSV
  `A`; run `finemap(..., output=B)`; assert `A` and `B` are **byte-identical**
  (`filecmp.cmp(A, B, shallow=False)`). This is the no-mismatch guarantee.
  **Determinism:** both runs MUST use identical parameters INCLUDING `threads=1` —
  multi-thread floating-point summation order is the only non-determinism source and
  could flip a 6th significant figure (`:.6g`) and break byte-identity. Pin
  `threads=1` in the parity test so the test is reliable; this proves `finemap`'s
  argv construction and file handling introduce zero divergence from a direct
  `bayes-scan-rss` call. (Do not weaken to a numeric-tolerance compare — the mandate
  is byte-identity.)
- **In-memory faithfulness:** `pd.testing.assert_frame_equal(r.results, pd.read_csv(B, sep="\t"))`.
- **Rich result behavior:** `results` has the 11 PolyFun columns; `credible_sets`
  columns `[credible_set,n_snps,lead_snp,lead_pip,sum_pip]`; `n_variants==len(results)`;
  `n_credible_sets==credible_sets.shape[0]==results.CREDIBLE_SET.gt(0).nunique-ish`
  (distinct positive CS labels); `n_variants_in_credible_sets==(CREDIBLE_SET>0).sum()`.
- **No-output temp path:** `finemap(...)` with `output=None` returns a populated
  `results` and leaves no file behind (temp dir cleaned); `output_files` empty.
- **Friendly errors:** missing `ld_ref` and `geno` → `ValueError`; `geno=` set →
  friendly `RuntimeError` (upstream NotImplementedError surfaced).
- **Precedence:** `api.finemap.__name__ == "finemap"` (real fn wins over the
  `__getattr__` CLI-runner facade despite `"finemap"` being in `_TIER_2`).

`tests/test_cli_finemap.py`: `finemap --help` (flags present) + one real end-to-end
`finemap` CLI invocation on the fixture (exit 0; `--output` writes the TSV).

R mocked `tests/testthat/test-finemap.R`: `mockery::stub(tg_finemap, "bridge_call", ...)`
returns a dict with the FineMapRun keys; assert `captured$fn=="finemap"`, args
forwarded, `expect_s4_class(r,"FineMapRun")`, and both tibbles rebuilt.

## Out of scope

- No change to `bayes-scan-rss`, the SuSiE-RSS engine, or `write_results_tsv`.
- The `_finemapping.py` post-processing utilities (`extract_credible_sets`,
  `annotate_sumstats`, `locus_summary`, `to_coloc_sumstats`) remain Python-library-only
  (user-deprioritized).
- No MCP tool for `finemap`.
