# Performance Regression CI (Efficiency E5)

**Workflow:** `.github/workflows/perf.yml`
**Diff script:** `bench/diff_perf.py`
**Bench runner:** `bench/native_speedups.py`
**Tests:** `tests/test_perf_regression_ci.py`

The wall-time analog to the streaming memory regression nets in
`tests/test_streaming_memory.py`. Where the memory tests catch a PR
that re-materializes the full `(n × m)` genotype matrix, this
workflow catches a PR that silently makes a hot native kernel slower.

## What the PR comment looks like

The workflow posts a markdown table to every PR (replacing any prior
comment from this bot). Verdict column legend:

| Verdict | Meaning | Effect on the workflow |
|---|---|---|
| `OK` | Within 5% of master | None |
| `WARNING` | 5–10% slower than master | `::warning::` in the job log; PR comment posted; workflow still passes (exit 2) |
| `REGRESSION` | > 10% slower than master | `::error::` and the job **fails** (exit 1) |
| `NEW` | Kernel exists in PR but not master | Reported, not gated |
| `REMOVED` | Kernel exists in master but not PR | Reported, not gated |

Only the `native` mode is gated. The `python` mode is the algorithmic
reference and its wall time on the runner reflects baseline noise
rather than something the PR is responsible for; it appears in the
table for context but its verdict is always `OK`.

## How to read the table

Each row is one (kernel, mode) pair. Columns:

- **Kernel** — human-readable kernel name with size envelope.
- **Mode** — `native` (gated) or `python` (informational).
- **master p50** — 50th-percentile wall time on master.
- **PR p50** — 50th-percentile wall time on PR head.
- **%Δ p50** — `(pr - master) / master * 100`. Positive = PR slower.
- **%Δ p95** — same, on the 95th percentile.
- **Verdict** — see legend above.

## Adding a new kernel to the CI subset

1. Add the bench definition to `BENCHES` in `bench/native_speedups.py`
   (mirroring the existing `_build_*` / `_run_*` pair pattern).
   Give it a unique `slug=` argument — that's the identifier
   `--kernel-subset` filters on.
2. If the python reference path takes more than ~5 s at the default
   sizes, add a `ci=` arg to each `_pick(small, realistic, ci=...)`
   call inside the builder so the CI run uses smaller inputs.
3. Append the slug to `CI_SUBSET_SLUGS` (same file).
4. Update the wall-time budget comment at the top of
   `.github/workflows/perf.yml` if needed.
5. Run `python bench/native_speedups.py --kernel-subset <slug> --n-repeats 5 --output json --output-path /tmp/test.json` locally to confirm timing fits.

The 12-kernel default subset covers:

- 3 LD kernels (Gabriel, PELT, CC-graph)
- 2 imputation kernels (KNN, mode)
- 2 PGS kernels (LDpred2 Gibbs, PRS-CS Gibbs)
- 1 HWE kernel (diploid)
- 1 SPA kernel
- 1 LDSC h² jackknife (post-GWAS)
- 2 streaming-GRM linalg kernels (the post-Efficiency-E1/E4 hot path)

## Updating the gate threshold

The thresholds are CLI arguments to `bench/diff_perf.py`:

```bash
python bench/diff_perf.py master.json pr.json \
    --regression-pct 10.0 \
    --warn-pct 5.0
```

The defaults (10% / 5%) match the Efficiency E5 spec. To change them
for the workflow, edit the `Diff master vs PR` step in
`.github/workflows/perf.yml` and pass the new flags. The defaults
also propagate to `tests/test_perf_regression_ci.py` (the diff-script
exit-code tests bake in 7% slower as a warning trip-wire and 20% as
a regression trip-wire — both inside the default thresholds, so a
threshold change won't break the tests unless you tighten things
below the test's fixture deltas).

## Interpretation: regression vs warning vs noise

GitHub-hosted runners are not isolated and not pinned to a specific
machine class — the same workflow can run on a machine 1.5× faster
or slower than the previous run. **Do not interpret a single +12 %
regression as a definitive bug**; re-run the workflow first
(workflow_dispatch is wired). Persistent regressions across re-runs
are real.

Rules of thumb:

- **< 5 %** is below the noise floor on a shared runner. The CI
  ignores it.
- **5–10 %** is in the noisy band. The workflow posts a `WARNING`
  comment so reviewers can eyeball trends across PRs without
  individual PRs being blocked.
- **> 10 %** is the hard-fail threshold. A real regression at this
  scale is reproducible and well above the noise floor on every
  hosted runner class we've measured.

The `--n-repeats 5` minimum in the workflow gives the p50 / p95
percentiles enough samples to be stable; lower than that and the
median collapses to a single point and noise becomes unfilterable.

## When the master baseline is missing

The first PR after the workflow lands will not have a master baseline
JSON to compare against (the corresponding master push hasn't run
yet). In that case:

- The diff script writes a `SKIPPED` markdown body to the PR comment.
- It exits 0 (workflow passes).
- The next push to master will produce the baseline; subsequent PRs
  will diff normally.

This is also the fallback if the master artifact has expired
(`retention-days: 90`).

## Local reproduction

```bash
# Run the same CI subset locally:
TORCHGENOMICS_BENCH_CI=1 python bench/native_speedups.py \
    --output json \
    --output-path /tmp/local.json \
    --kernel-subset ci \
    --n-repeats 5

# Compare two locally-captured JSONs:
python bench/diff_perf.py /tmp/before.json /tmp/after.json \
    --markdown /tmp/diff.md
```

For the full informational sweep (25 kernels, no CI shrinking):

```bash
TORCHGENOMICS_BENCH_REALISTIC=1 python bench/native_speedups.py
# writes bench/native_speedups_realistic.md
```
