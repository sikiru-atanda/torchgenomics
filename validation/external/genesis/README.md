# GENESIS reference harness (Phase 57 Unit A, Task 6)

**This is the definitive external reference-equivalence gate for
`torchgenomics.linalg.kinship_admixed`'s three admixture-aware estimators:**

| torchgenomics | GENESIS / SNPRelate reference | Original paper |
|---|---|---|
| `king_robust_kinship` | `SNPRelate::snpgdsIBDKING(type="KING-robust")` | Manichaikul et al. 2010, *Bioinformatics* 26:2867, eq. 11 |
| `pc_air` | `GENESIS::pcair()` | Conomos et al. 2015, *Genet. Epidemiol.* 39:276 |
| `pc_relate` | `GENESIS::pcrelate()` | Conomos et al. 2016, *AJHG* 98:127 |

Every internal test in `tests/test_kinship_admixed.py` (Tasks 2, 4, 5)
explicitly disclaims itself as reference-equivalence evidence — those tests
check internal consistency (hand-computed KING values, ancestry-separation
sanity, pedigree recovery) on synthetic data only. **This harness is what
actually closes the reference-equivalence gap** (spec §8 Claim 3).

## Why a synthetic fixture, not real 1000G data

The synthetic 2-population Balding-Nichols admixed cohort with injected
parent-offspring pairs (`tests/fixtures/admixed/make_admixed.py`) has a
*known* ground truth (labeled `true_kinship`, labeled ancestry), so
disagreement between torchgenomics and GENESIS reflects the algorithm, not
uncertain real-world pedigree/ancestry labels. It also needs no network
access or external download, keeping this harness reproducible offline.

## Reproduction recipe

```bash
# From repo root:
bash validation/external/genesis/install.sh        # idempotent; Bioconductor R packages
bash validation/external/genesis/fetch_data.sh      # exports the fixture -> out/
Rscript validation/external/genesis/run_genesis.R   # snpgdsIBDKING -> pcair -> pcrelate -> out/*.tsv
python3 validation/external/genesis/compare.py      # torchgenomics vs GENESIS; PASS/FAIL

# Or via pytest (opt-in; skipped by default):
pytest -m external tests/test_kinship_admixed.py::test_genesis_equivalence_if_present -v
```

Each shell script sources `validation/external/_lib/preflight.sh` and
asserts disk + RAM headroom before doing any work — this is the user's hard
rule (spec §5.3): no partial executions on insufficient resources.

## What each script does

- **`export_fixture.py`** — generates the deterministic admixed+related
  fixture (`n_per_pop=60, n_related_pairs=15, m=2000, fst=0.15, seed=42`)
  and writes it to **two byte-identical representations**: a PLINK 1
  BED/BIM/FAM triple (`out/fixture.{bed,bim,fam}`, for SNPRelate) and a
  plain `out/G.npy` array (for torchgenomics) — so both tools consume
  exactly the same numbers. Writes its own minimal PLINK BED encoder/decoder
  (see the module docstring for the exact 2-bit-code convention) and
  immediately round-trips its own output to verify the encoding is
  self-consistent before declaring success. Also computes the LD-pruned
  marker subset (`torchgenomics.linalg.kinship_admixed.ld_prune_independent`,
  `r2_threshold=0.1`) and writes it to `out/pruned_snp_ids.txt` so GENESIS's
  PC-AiR/PC-Relate step can be restricted to the identical marker panel.
- **`run_genesis.R`** — `snpgdsBED2GDS` → `snpgdsIBDKING(type="KING-robust")`
  (full unpruned panel) → `GENESIS::pcair()` → `GENESIS::pcrelate()` (both
  restricted to `pruned_snp_ids.txt`). Writes three golden TSVs to `out/`:
  `genesis_king_kinship.tsv`, `genesis_pcair_pcs.tsv`,
  `genesis_pcrelate_kinship.tsv`, each row/column-ordered by `sample_id`
  (`IND0000`, `IND0001`, ... — identical order to `G.npy`'s rows).
- **`compare.py`** — loads `out/G.npy`, runs torchgenomics'
  `king_robust_kinship` / `ld_prune_independent` + `pc_air` / `pc_relate` on
  it, loads the GENESIS TSVs (if present), and prints a PASS/FAIL summary
  table: KING kinship max|diff| + correlation, PC-AiR per-axis |Pearson
  correlation| (signs arbitrary), PC-Relate kinship max|diff| + correlation
  + explicit agreement on the fixture's known parent-offspring pairs. Exits
  nonzero on any FAIL. **Gracefully skips the GENESIS comparison (exit 0)**
  if the golden TSVs are absent, so `export_fixture.py` + `compare.py` can
  be smoke-tested without R/GENESIS installed.
- **`install.sh`** — idempotent Bioconductor install of `SNPRelate` +
  `GWASTools` + `GENESIS` via `BiocManager::install()`.
- **`fetch_data.sh`** — thin wrapper that runs `export_fixture.py` (nothing
  to download; synthetic fixture).

## Tolerances

`compare.py`'s `TOL_*` constants at the top of the file are **placeholders**
marked `# PLACEHOLDER` / `# TODO(Task 6 rerun)` — per repo convention
(observed-then-floored, never aspirational), they must be replaced with the
actual values observed on the first successful `run_genesis.R` execution,
floored (not rounded up), before this gate is trusted as a real regression
check.

## Opt-in pytest gate

`tests/test_kinship_admixed.py::test_genesis_equivalence_if_present` is
marked `@pytest.mark.external` (skipped by default; run with `pytest -m
external`). It skips with a clear message if `out/genesis_*.tsv` are not
present, and otherwise re-runs the same comparison logic as `compare.py` and
asserts the (eventually non-placeholder) tolerances.
