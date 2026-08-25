# Design — Wire MR + Coloc through the api / CLI / R

**Date:** 2026-08-25
**Status:** APPROVED
**Base branch:** `feat/mr-coloc-cli-r` off `feat/friendly-api-polish` (top of the friendly-API stack).

## Goal

Close the biggest Python-only parity gap surfaced in the R/CLI audit: expose
**Mendelian randomization** and **colocalization** — currently reachable only
from `torchgenomics.postgwas` in Python — as first-class **api functions + CLI
subcommands + R wrappers** (which also makes them MCP tools). Thin orchestration
over the existing, validated `postgwas` functions; **no new statistics**.

Scope this iteration (user-approved):
- **MR:** all four estimators — `ivw`, `egger`, `weighted_median`, `presso` — plus
  `method="all"` (runs the full sensitivity panel, one row per method).
- **Coloc:** `pairwise` (Giambartolomei 2-trait, PP.H0–H4) **and** `hyprcoloc`
  (N-trait, takes a list of sumstats).

## Background — verified facts (cross-checked against current code)

- The proven, working exposure pattern is the tier-1 hand-crafted one used by
  `clump` / `meta`: a real `torchgenomics.api.<fn>` (in `api/postgwas.py`, listed
  in `api.__all__`) → a CLI subcommand → a **hand-crafted** R wrapper in
  `rTorchGenomics/R/api.R` (`bridge_call("<fn>", ...)` → `<Class>$new_from_dict`).
  Confirmed: `hasattr(api,'clump')` / `hasattr(api,'meta')` are True and their R
  wrappers work.
- The ~34 **auto-generated** tier-2 R wrappers (`tg_bayes_scan`, …) call
  `bridge_call("<name>")` → `api.<name>`, but those `api.<name>` functions do NOT
  exist (`hasattr(api,'bayes_scan')` is False) — so those auto wrappers would fail
  at runtime. This is a **pre-existing, separate** issue; MR/coloc will use the
  hand-crafted path, not the auto-generator.
- The postgwas functions (verified signatures):
  - `mr_ivw(exposure, outcome) -> MRResult`
  - `mr_egger(exposure, outcome) -> MRResult`
  - `mr_weighted_median(exposure, outcome, n_boot=1000, seed=42) -> MRResult`
  - `mr_presso(exposure, outcome, n_perm=1000, outlier_threshold=0.05, seed=42, null='parametric') -> MRResult`
  - `mr_all(exposure, outcome, n_boot=1000, n_perm=1000, seed=42) -> list[MRResult]`
  - `coloc_pairwise(ss1, ss2, prior_1=1e-4, prior_2=1e-4, prior_12=1e-5, prior_w=0.0225) -> ColocPairwiseResult`
  - `hyprcoloc(sumstats_list, prior_1=1e-4, prior_2=0.98, prior_w=0.0225, trait_names=None) -> HyprcolocResult`
- `MRResult` fields: `method, beta_hat, se, p_value, n_instruments, intercept,
  intercept_se, intercept_p, egger_i2, global_rss, global_p, outlier_indices,
  n_outliers, beta_corrected, se_corrected, p_corrected`.
- `ColocPairwiseResult` fields: `pp_h0, pp_h1, pp_h2, pp_h3, pp_h4, candidate_snp`.
- `HyprcolocResult` fields: `subset_posteriors, pp_all_colocalize, pp_null,
  best_cluster, best_cluster_posterior, candidate_snp, candidate_snp_posterior,
  log_bf_marginal, trait_names`.
- Sumstats are loaded from a file via `postgwas.load_sumstats(path, chr_col=...,
  ..., sep='\t') -> SumStats` (the same reader `clump` uses). `SumStats` carries
  `chr, pos, snp, a1, a2, beta, se, p, n, af`.

## Design

### Component 1 — `api.mr()`

In `torchgenomics/api/postgwas.py`:

```python
def mr(exposure, outcome, *, method="ivw", output=None,
       n_boot=1000, n_perm=1000, seed=42, sep="\t") -> MRRun: ...
```

- `method ∈ {"ivw","egger","weighted_median","presso","all"}` (validated;
  friendly `ValueError` listing valid options otherwise).
- Loads `exposure`/`outcome` via `load_sumstats`.
- Runs the corresponding `postgwas.mr_*` (or `mr_all` for `"all"`).
- Builds a results DataFrame (one row per method) with columns:
  `method, beta, se, pval, n_instruments, egger_intercept, egger_intercept_p,
  n_outliers, beta_corrected, pval_corrected` (NaN where a column doesn't apply
  to a method). Writes it to `output` TSV when given.
- Returns `MRRun` (see Component 3).

### Component 2 — `api.coloc()`

```python
def coloc(sumstats, sumstats2=None, *, method="pairwise", output=None,
          prior_1=1e-4, prior_2=1e-4, prior_12=1e-5,
          trait_names=None, sep="\t") -> ColocRun: ...
```

- `method ∈ {"pairwise","hyprcoloc"}`.
- **pairwise:** requires `sumstats` + `sumstats2` (two file paths); runs
  `coloc_pairwise`; result carries `pp_h0..pp_h4`, `candidate_snp`. Missing
  `sumstats2` → friendly `ValueError`.
- **hyprcoloc:** `sumstats` is a **list** of ≥2 file paths (and `sumstats2`
  must be None); runs `hyprcoloc` with `prior_2=0.98` (its own default for the
  N-trait prior); result carries `pp_all_colocalize, pp_null, best_cluster,
  best_cluster_posterior, candidate_snp`. A list of <2 → friendly `ValueError`.
- Loads each file via `load_sumstats`; writes a one-/few-row TSV when `output`.
- Returns `ColocRun` (see Component 3).

### Component 3 — result classes `MRRun` / `ColocRun`

In `torchgenomics/api/_results.py`, dataclasses subclassing `_BaseRun`
(mirroring `MetaRun`), each with `.summary()` and inheriting `.to_dict()`:

- `MRRun`: `method: str`, `results: pd.DataFrame` (the per-method table),
  `n_instruments: int`, `primary_beta: float`, `primary_p: float`. `.summary()`
  prints the panel and flags Egger-intercept pleiotropy / PRESSO outliers when
  present.
- `ColocRun`: `method: str`, `pp_h4: float` (pairwise) or `pp_all: float`
  (hyprcoloc), plus the full posterior set as fields + a small `table`
  DataFrame. `.summary()` prints the posteriors and a plain-language verdict
  ("PP.H4=0.92 → strong evidence of a shared causal variant").

Export both from `_results.py` and `api/__init__.py` (`__all__`), and add
`mr`/`coloc` to `api.__all__`.

### Component 4 — CLI subcommands

In `torchgenomics/cli.py`, following the `clump`/`meta` subcommand pattern
(`_add_<name>_parser` + `_cmd_<name>`, registered in `build_parser` + the
dispatch dict), each thin-wrapping the api function:

- `torchgenomics mr --exposure exp.tsv --outcome out.tsv --method all
  [--n-boot 1000 --n-perm 1000 --seed 42] --output mr.tsv`
- `torchgenomics coloc --sumstats a.tsv --sumstats2 b.tsv --method pairwise
  --output coloc.tsv`
- `torchgenomics coloc --sumstats a.tsv b.tsv c.tsv --method hyprcoloc
  --output hyprcoloc.tsv` (`--sumstats nargs="+"`; pairwise uses the first two
  or requires `--sumstats2` — spec: pairwise takes `--sumstats` + `--sumstats2`,
  hyprcoloc takes `--sumstats` with ≥2 values).

Add both to `_TIER_2_CLI_SUBCOMMANDS` (`torchgenomics/_manifest/__init__.py`) so
the manifest + MCP see them, and bump the CLI subcommand count note in CLAUDE.md
(46 → 48).

### Component 5 — R wrappers `tg_mr` / `tg_coloc`

Hand-crafted in `rTorchGenomics/R/api.R` (mirroring `tg_clump`/`tg_meta`):

```r
tg_mr(exposure, outcome, method = c("ivw","egger","weighted_median","presso","all"),
      output = NULL, n_boot = 1000, n_perm = 1000, seed = 42)
  -> MRRun (bridge_call("mr", ...))
tg_coloc(sumstats, sumstats2 = NULL, method = c("pairwise","hyprcoloc"),
         output = NULL, prior_1 = 1e-4, prior_2 = 1e-4, prior_12 = 1e-5,
         trait_names = NULL) -> ColocRun (bridge_call("coloc", ...))
```

Add S4 `MRRun`/`ColocRun` result classes + `new_from_dict` in
`rTorchGenomics/R/result_classes.R`; add exports to `NAMESPACE`; add both to the
`_pkgdown.yml` reference index (a new "Post-GWAS: MR + coloc" section). Author R
files via heredoc/`writeLines` (subagent Write unreliable for R).

## Error handling

- Unknown `method` → friendly `ValueError` (Python) / `match.arg` (R) listing
  valid options.
- pairwise coloc without a second sumstats / hyprcoloc with <2 → friendly error.
- Missing file / missing required column → the actionable error already raised
  by `load_sumstats` (wrapped with the offending path).

## Testing

Python (`tests/test_api_mr_coloc.py`, new):
- Simulate a small exposure/outcome pair with a true causal effect (shared
  instruments) → `api.mr(method="all")` returns a `MRRun` whose IVW β is close
  to the simulated effect; Egger intercept ≈ 0; all four method rows present.
- Simulate two coloc sumstats sharing one causal variant → `api.coloc(method=
  "pairwise")` returns `ColocRun` with `pp_h4` high; a non-shared pair → low
  `pp_h4`. Three-trait shared signal → `api.coloc(method="hyprcoloc")` →
  high `pp_all_colocalize`.
- Friendly errors: bad method; pairwise without `sumstats2`; hyprcoloc with 1 file.
- These assert on the reused postgwas math (already validated against
  TwoSampleMR / coloc); the api layer only orchestrates.

CLI (`tests/test_cli_mr_coloc.py`): `mr --help` / `coloc --help` show the flags;
`mr --exposure … --outcome … --method ivw --output …` and `coloc --sumstats …
--sumstats2 … --output …` run end-to-end on the fixtures and write the TSV.

R (`rTorchGenomics/tests/testthat/test-mr-coloc.R`): mocked `bridge_call` tests
that `tg_mr`/`tg_coloc` forward the right args and wrap the dict into the S4
result; integration (guarded) that they run against the fixtures.

## Scope / non-goals

- Only MR (4 estimators + all) and coloc (pairwise + hyprcoloc). SMR/HEIDI,
  HESS, MR-MEGA, power, winner's curse, fine-mapping utils remain Python-only
  (future follow-ups).
- No new statistics; no change to the `postgwas` math or to existing api/CLI/R
  surfaces.
- Fixing the pre-existing broken auto-generated tier-2 R wrappers is out of
  scope (separate issue).

## Backward compatibility

Purely additive: two new api functions (+ their result classes), two new CLI
subcommands, two new R wrappers + S4 classes, two `_TIER_2_CLI_SUBCOMMANDS`
entries. No existing signature/default changes.
