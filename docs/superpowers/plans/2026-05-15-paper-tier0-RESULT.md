# Tier 0 result log

Working branch: `paper/genome-biology-methods`
Base: `consolidated/na-roundup@480cc99fe13fb7089be69110992d174315ff7cfa`
Tier-0 commits applied on top of the base:
- `e9e692c` — Tier 0: bring paper spec + 3 plan files from modernization/specs
- `3a02165` — Tier 0: paper/ scaffold marker

## Smoke pass

Command:

```
pytest -x --ignore=tests/test_streaming_memory.py \
       -m "not slow and not external and not cli_matrix and not reproducibility and not gpu" \
       --tb=no -q
```

Result: **2879 passed, 328 skipped, 165 deselected, 32 warnings** in ~113s.

## Verified infrastructure

- `validation/external/` — 14 directories (`bolt_lmm`, `gapit`, `gemma`, `gwaspoly`, `ldsc`, `_lib`, `plink2`, `regenie`, `saige`, `soymd`, `soynam`, `susieR`, `twosamplemr`, `ukb`)
- `csrc/` — 25 `.cpp` files
- `bench/native_speedups.py` — present
- `tests/test_streaming_memory.py` — present
- `docs/efficiency/streaming_audit.md` — present
- `docs/validation_findings.md` — present
- `validation/external/_lib/preflight.sh` — present
- `.github/workflows/gpu.yml` — present

## Fresh-checkout step — fixture regeneration

On the first checkout of this branch (and any future fresh checkout), the gitignored test fixtures under `tests/fixtures/` must be regenerated before the smoke pass:

```
python tests/fixtures/create_fixtures.py
```

This is required because:
- `tests/fixtures/{tiny.bed, tiny.bim, tiny.fam, tiny_dosage_truth.npy, tiny.hmp.txt, tiny_dosage.csv, tiny_dosage.map, tiny.zarr, tiny.h5, tiny_pheno.txt, tiny_covar.txt}` are all gitignored.
- `create_fixtures.py` was updated when the PLINK BED reader's allele convention was corrected to PLINK 1.9's A1-counting convention (commit `3f9cccc`).
- A stale on-disk `tiny.bed` (generated under the OLD A2 convention before that fix) causes `tests/test_io_plink.py::TestPlinkBedReader::test_dosage_matches_truth` to fail with a 69% mismatch and a clean 0↔2 swap, because the truth array and the reader disagree on which allele is being counted.
- Running `create_fixtures.py` against the current generator (which is correct) regenerates the BED + truth pair together under the A1 convention. After regeneration the test passes.

This is a fresh-checkout setup step, not a code bug. The repo is correct; the on-disk fixtures may be stale. Tier 0 of this plan handled it locally. If `paper/genome-biology-methods` is pushed to remote and later cloned fresh, the cloner must run `create_fixtures.py` before the smoke pass — consider adding this to a project bootstrap script in a future task.

## Next

Proceed to Plan B (`docs/superpowers/plans/2026-05-15-paper-tier123-agent-briefs.md`).
