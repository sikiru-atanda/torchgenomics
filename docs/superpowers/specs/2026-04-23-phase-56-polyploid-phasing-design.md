# Phase 56 — Polyploid Phasing (PolyOrigin wrapper)

**Status**: Design approved 2026-04-23. Awaiting implementation plan.
**Scope**: One phase. Single external tool (PolyOrigin). Follow-up phasers (WhatsHap-polyphase, hapCON, TetraOrigin) are deferred to separately-scoped later phases.
**Authors**: Sikiru Atanda + Claude Code (brainstorming session 2026-04-23).

---

## 1. Motivation and scope boundary

TorchGWAS already contains the downstream half of a polyploid haplotype-GWAS pipeline:

- `torchgwas.preprocess.phase.load_haplotypes` — ploidy-generic, reads a phased polyploid VCF into `(n, ploidy, m)` tensors.
- `torchgwas.models.haplotype_gwas` — Phase 46/47 haplotype scans, ploidy-generic on the input side (consumes `(n, ploidy, m)` tensors directly).
- `torchgwas.preprocess.dosage_call.run_updog` (Phase 55) — produces posterior dosage probabilities `(n, m, k+1)` from VCF read counts.

What is *missing* is the **producer** side of phased haplotypes for polyploids. `torchgwas/preprocess/phase.py::phase_beagle` is explicitly diploid-only (BEAGLE 5.x does not phase polyploids; the docstring was corrected alongside the Phase 55 commit). There is no tool inside TorchGWAS that converts posterior dosages on a connected F1 polyploid population into phased haplotypes.

Phase 56 delivers **exactly one** piece of that producer side: a thin external-wrapper module `torchgwas.preprocess.phase_polyorigin` that invokes the Julia package [`PolyOrigin.jl`](https://github.com/chaozhi/PolyOrigin.jl) (Zheng et al. 2021, *Genetics* 219(2):iyab106) to phase connected tetraploid/hexaploid F1 populations. Output is tensor-native, consumed by Phase 46/47 without modification.

### In scope

- One wrapper function `run_polyorigin(...)` around `PolyOrigin.polyOrigin()`.
- One dataclass `PhasingResult` describing the phasing output.
- One private runtime module `_polyorigin_runtime.py` holding the discover-first Julia bootstrap (via `juliacall`).
- Two pure private converters `_build_polyorigin_pedfile`, `_build_polyorigin_genofile` (independently unit-testable without Julia).
- One CLI subcommand `torchgwas phase-poly`.
- Tests: ~25 always-on Python tests (juliacall-stubbed) plus ~6 release-time end-to-end tests that run real PolyOrigin.jl.
- One getting-started recipe; one memory file.

### Out of scope (explicitly deferred)

- **Unrelated-panel phasing** (WhatsHap-polyphase, hapCON) — separate future phase.
- **TetraOrigin (MATLAB) and other F1 phasers** — same wrapper pattern applies; queued as Phase 56 follow-ups.
- **Multi-generation pedigrees** — upstream PolyOrigin hard constraint (F1 crosses + founder selfings only).
- **Ploidies other than {2, 4, 6}** — upstream hard constraint.
- **A `haplotypes.pt → phased VCF` export utility** — no current consumer in the pipeline; queue if asked.
- **Julia sysimage build** — juliacall's own caching handles same-process warm-up; a sysimage recipe may be a later documentation follow-up.
- **Integration with the `torchgwas pipeline` monolithic subcommand** — same cleanup concern as Phase 55.
- **A pure-Python reimplementation of PolyOrigin** — producer step, matches `impute_external.py` / `dosage_call.py` precedent.

### Integration point

```
probs.pt (n,m,k+1)   pedigree.tsv   map.tsv   [parent_phased.csv]
       │                  │             │              │
       └──────────┬───────┴──────┬──────┴──────┬───────┘
                  ▼              ▼             ▼
             run_polyorigin(..., ploidy ∈ {2,4,6})
                  │
                  ├── haplotypes       (n_off, ploidy, m)           → HaplotypeGWAS
                  ├── origin_probs     (n_off, ploidy, m, n_ph)     → optional downstream
                  ├── parent_phased    (n_par, max_ploidy, m)
                  ├── map_refined, valent_diag, postdose_probs      → diagnostics
                  └── per_individual_ploidy dict                    → mixed-ploidy support
```

No code changes to the downstream path. Scope boundary enforced by an explicit `ValueError` on any multi-generation pedigree (a parent appearing as another row's offspring).

---

## 2. Components and interfaces

Four components in one new file `torchgwas/preprocess/phase_polyorigin.py`, plus a shared runtime helper at `torchgwas/preprocess/_polyorigin_runtime.py`, plus one shipped dep-manifest `torchgwas/preprocess/juliapkg.json`.

### 2.1 `PhasingResult` dataclass

Mirrors `DosageCallResult` in `torchgwas/preprocess/dosage_call.py` for conceptual symmetry.

```python
@dataclass
class PhasingResult:
    haplotypes: Tensor                     # (n_offspring, ploidy, m) int8 — argmax of origin_probs
    origin_probs: Tensor                   # (n_offspring, ploidy, m, n_parent_haps) float64
    parent_phased: Tensor                  # (n_parents, max_ploidy, m) int8
    offspring_ids: list[str]               # length n_offspring
    parent_ids: list[str]                  # length n_parents
    variant_ids: list[str]                 # length m (refined order if refinemap=True)
    chrom: list[str]
    pos_bp: Tensor                         # (m,) int64
    pos_cm: Tensor                         # (m,) float64 — refined if refinemap=True
    per_individual_ploidy: dict[str, int]  # sample_id -> ploidy; supports mixed ploidy
    map_refined: bool                      # True iff refinemap=True and output order != input order
    valent_diag: pd.DataFrame              # per-marker valent configs from *_polyancestry.csv
    postdose_probs: Tensor                 # (n_offspring, m, max_ploidy+1) float64
    tool: str                              # "polyorigin"
    tool_version: str                      # e.g. "1.0.5"
    input_hash: str                        # sha256 of merged genofile+pedfile+map CSVs
    cmd: str                               # exact Julia call for reproducibility
    workdir: str | None                    # None unless keep_workdir=True
```

### 2.2 `run_polyorigin(...)` function

```python
def run_polyorigin(
    probs: Tensor | str,                      # (n, m, k+1) tensor OR path to .probs.pt
    pedigree_tsv: str,                        # user 3-col TSV (offspring, p1, p2[, ploidy])
    map_tsv: str,                             # marker, chrom, pos_bp[, cm]
    output_path: str,                         # prefix for persistent artifacts
    *,
    ploidy: int,                              # default ploidy for rows without per-row override
    sample_ids: list[str] | None = None,      # required if probs is a tensor
    variant_ids: list[str] | None = None,     # required if probs is a tensor
    parent_phased_csv: str | None = None,     # optional escape hatch for pre-phased parents
    julia_path: str | None = None,            # override discovery; else TORCHGWAS_JULIA; else discover
    auto_install_julia: bool = False,         # consent to download Julia if discovery fails
    refinemap: bool = True,                   # PolyOrigin default
    recomrate: float = 1.0,                   # cM/Mb; used to synthesize cm if map lacks it
    nworkers: int = 1,                        # Julia -p; PolyOrigin's own parallelism knob
    seed: int | None = 1234,
    keep_workdir: bool = False,
) -> PhasingResult: ...
```

**Preconditions** (all raised before Julia is touched):

- `ploidy in {2, 4, 6}` → else `ValueError`.
- Pedigree TSV schema valid (3 required cols, optional `ploidy`; offspring IDs unique; parent IDs present as founder rows) → else `ValueError` naming the offending row.
- F1-only check: every offspring's parent must itself be a founder, not an offspring of another row → else `ValueError`.
- Map TSV has `marker`, `chrom`, `pos_bp`; optional `cm` used as-is; missing → warn and synthesize `cm = pos_bp * recomrate / 1e6`.
- Map positions monotonic within chromosome → else `ValueError` naming first offender.
- Probs shape `(n, m, k+1)` where `k = ploidy` → else `ValueError`.
- Probs rows summing to less than `1 - 1e-3` → warn + renormalize.
- Sample IDs in probs must cover the pedigree's parent+offspring set (three-way intersection) → else `ValueError` listing first 10 missing + total count.
- If `parent_phased_csv` is given: every pedigree parent must appear in it; parents in both probs and the CSV → warn + prefer pre-phased.

**Postconditions**: persistent artifacts written atomically *after* Julia returns successfully and all output CSVs parse:

- `<output>.haplotypes.pt`, `<output>.origin_probs.pt`, `<output>.parent_phased.pt`, `<output>.postdose_probs.pt` — `torch.save`
- `<output>.map_refined.tsv`, `<output>.valent_diag.tsv` — `DataFrame.to_csv`
- `<output>.meta.json` — IDs, chrom/pos, `per_individual_ploidy`, `map_refined` flag, tool version, input hash, cmd, timestamp

### 2.3 Julia runtime (`_polyorigin_runtime.py`)

Discover-first, lazy, thread-safe, cached. No subprocess — in-process via `juliacall`.

```python
# _polyorigin_runtime.py
import os, shutil, sys, threading
from typing import Any

_lock = threading.Lock()
_jl: Any = None
_polyorigin: Any = None
_version: str | None = None

_COMMON_JULIA_PATHS = [
    os.path.expanduser("~/.juliaup/bin/julia"),
    os.path.expanduser("~/.juliaup/bin/julia.exe"),
    "/opt/julia/bin/julia",
    "/usr/local/bin/julia",
    r"C:\Program Files\Julia\bin\julia.exe",  # globbed at call time
]

def _find_existing_julia(override: str | None) -> str | None: ...
def _version_ok(julia_path: str) -> tuple[bool, str]: ...  # >= 1.10
def _consent_to_install(auto_install: bool) -> bool: ...   # tty prompt or kwarg

def get_runtime(
    julia_path: str | None = None,
    auto_install_julia: bool = False,
) -> tuple[Any, Any, str]:
    """Bootstrap PolyOrigin on demand. Steps:
    1. Resolve julia_path: arg > TORCHGWAS_JULIA env > discover common locations.
    2. If found, probe `julia --version`; reject if < 1.10.
    3. If not found or rejected: interactive consent (tty) or auto_install_julia=True,
       else RuntimeError naming both escape hatches.
    4. If a user-Julia was accepted, pin it via juliacall's env var (PYTHON_JULIAPKG_EXE
       per current juliacall docs; verified at implementation time) BEFORE importing juliacall.
    5. from juliacall import Main as jl; jl.seval("using Pkg; Pkg.activate(<torchgwas-project>); Pkg.instantiate()")
       — juliapkg.json drives PolyOrigin.jl install from pinned URL+rev if not already present.
    6. jl.seval("using PolyOrigin"); capture pkgversion(PolyOrigin).
    7. Cache (jl, PolyOrigin module, version) module-wide; return.
    """
```

**Shipped**: `torchgwas/preprocess/juliapkg.json`

```json
{
  "julia": "1.10",
  "packages": {
    "PolyOrigin": {
      "uuid": "<verified-from-upstream-Project.toml-at-implementation>",
      "url": "https://github.com/chaozhi/PolyOrigin.jl",
      "rev": "v1.0.5"
    }
  }
}
```

Rationale for in-process via `juliacall` over subprocess + embedded driver string: (1) auto-install of Julia across Linux/macOS/Windows is `juliacall`'s core feature, avoiding a per-OS installer in TorchGWAS; (2) same-process JIT amortization matters for users phasing multiple populations in one session; (3) library-call failure modes are cleaner (Julia exceptions become Python exceptions via `juliacall`).

### 2.4 Pure helper converters

Two module-private pure functions in `phase_polyorigin.py`, unit-testable without Julia:

- `_build_polyorigin_pedfile(user_tsv, ploidy_default, sample_ids_in_probs) -> Path` — writes `pedfile.csv` at PolyOrigin's native schema (`individual, population, motherid, fatherid, ploidy`). Assigns integer `population` IDs by grouping offspring on unique `(parent1, parent2)` pairs; founders get `motherid=fatherid=0, population=0`. Per-individual ploidy taken from the TSV's optional column, else the `ploidy_default`.

- `_build_polyorigin_genofile(probs, sample_ids, variant_ids, map_df, parent_phased_df | None) -> Path` — writes one CSV with header `marker, chromosome, pos, ind1, ind2, ..., indN`. Cells:
  - probability-encoded offspring: `f"{p0:.4f}|{p1:.4f}|...|{pk:.4f}"`
  - raw-probs parents (default): same probability encoding
  - pre-phased parents (escape hatch): pipe-joined allele strings from `parent_phased_df`
  - map cols 1–3: `marker`, `chromosome`, `cm` (synthesized from bp if map TSV lacked `cm`)

### 2.5 Directory layout

```
torchgwas/preprocess/
├── phase.py                    # unchanged (legacy BEAGLE diploid wrapper)
├── phase_polyorigin.py         # NEW — run_polyorigin, PhasingResult, helper converters
├── _polyorigin_runtime.py      # NEW — juliacall bootstrap, discover-first, cached
├── juliapkg.json               # NEW — Julia + PolyOrigin.jl dep manifest
└── dosage_call.py              # unchanged (Phase 55)
```

Rationale for a standalone `phase_polyorigin.py` rather than extending `phase.py`: `phase_beagle` returns a VCF path (I/O-heavy), runs Java, is diploid-only. `run_polyorigin` returns tensors, runs in-process Julia, supports {2,4,6}. Different result types and different dep tiers argue for separate modules. Mirrors Phase 55's decision to keep `dosage_call.py` separate from `impute_external.py`.

---

## 3. Data flow

```
USER INPUT                                                    WRITTEN BY
──────────────────────────────────────────────────────────────────────────
probs.pt (n, m, k+1)                ← Phase 55 dosage-call     user
pedigree.tsv (offspring, p1, p2[, ploidy])                     user
map.tsv     (marker, chrom, pos_bp[, cm])                      user
parents_phased.csv (optional)                                  user
│
▼ validate + three-way ID intersection (probs ∩ pedigree ∩ [parents_phased])
│   F1-only check, ploidy ∈ {2,4,6}, map schema, monotonic bp
│
▼ _build_polyorigin_pedfile  →  workdir/pedfile.csv              run_polyorigin
│   (population IDs per (p1,p2) pair; founders pop=0; per-ind ploidy)
│
▼ _build_polyorigin_genofile →  workdir/genofile.csv             run_polyorigin
│   (merged map cols + probability-encoded individual cells;
│    parents as phasedgeno strings iff parent_phased_csv supplied)
│
▼ _polyorigin_runtime.get_runtime(julia_path, auto_install_julia)
│   discover → version-check → [consent → juliacall install] → import → Pkg.instantiate
│
▼ jl.PolyOrigin.polyOrigin(genofile, pedfile;
│     workdir, isphysmap=false, refinemap, nworkers, seed, outstem="out")
│
workdir/out_genoprob.csv         (offspring × marker posterior) PolyOrigin
workdir/out_postdoseprob.csv     (offspring dosage probs)       PolyOrigin
workdir/out_parentphased.csv     (phased parent genotypes)      PolyOrigin
workdir/out_maprefined.csv       (refined genetic map)          PolyOrigin
workdir/out_polyancestry.csv     (valent configs, diagnostics)  PolyOrigin
workdir/out.log                                                 PolyOrigin
workdir/done.marker              (Python-written in try/finally after the jl call)
│
▼ _parse_polyorigin_outputs (pandas; ID-keyed; no positional assumption)
│   origin_probs   ← out_genoprob.csv
│   haplotypes     ← origin_probs argmax (int8)
│   parent_phased  ← out_parentphased.csv
│   postdose_probs ← out_postdoseprob.csv
│   map_refined    ← out_maprefined.csv
│   valent_diag    ← out_polyancestry.csv valent-freq rows
│
▼ hash input CSVs; build PhasingResult in RAM
│
▼ persist (atomic — only after all parsing succeeds):
│   <output>.haplotypes.pt         torch.save
│   <output>.origin_probs.pt       torch.save
│   <output>.parent_phased.pt      torch.save
│   <output>.postdose_probs.pt     torch.save
│   <output>.map_refined.tsv       DataFrame.to_csv
│   <output>.valent_diag.tsv       DataFrame.to_csv
│   <output>.meta.json             json.dump
│
▼ downstream (existing code, unchanged)
│
haplotypes  → HaplotypeGWAS (Phase 46/47)
origin_probs → optional future haplotype-origin GWAS consumer (captured for free)
```

**Workdir lifecycle**: `tempfile.TemporaryDirectory()` scoped to the call. Auto-cleanup on exit (incl. exceptions). `keep_workdir=True` suppresses cleanup and records the path on `PhasingResult.workdir` for debugging.

**Atomicity**: persistent `<output>.*` artifacts are written only after `done.marker` is present and every expected output CSV parses. Partial Julia failure → workdir briefly exists (unless `keep_workdir=True`) but nothing user-visible appears at `<output>.*`.

**Ordering**: the merged genofile writes rows in map-TSV order and columns in (map cols 1–3, parents alphabetically, offspring alphabetically) order. PolyOrigin preserves marker and individual IDs in its output CSVs; Python parses with `index_col=0` — never positional.

**Map refinement**: `refinemap=True` can reorder markers within a chromosome. The refined order is written to `map_refined.tsv`, all returned tensors follow the refined order, and `map_refined: bool` on `PhasingResult` flags that the returned variant order is *not* input order. Callers who need to realign to a reference map must rely on `variant_ids`, not positional indexing.

**Mixed-ploidy**: when pedigree rows have different ploidies (e.g., a tetraploid × diploid cross), `haplotypes.shape[1]` uses `max_ploidy`, with `-1` fill for rows shorter than max. `per_individual_ploidy` on the result maps sample_id → its actual ploidy. `HaplotypeGWAS` handling of `-1` sentinel must be verified at implementation (flagged in Section 5 tests).

---

## 4. Error handling and environmental dependencies

Four failure classes, each with a deterministic response. Mirrors Phase 55's taxonomy.

### 4.1 Environment missing

| Failure | Response |
| --- | --- |
| `juliacall` not importable (user pip-installed without the `polyploid-phase` extra) | `RuntimeError("Install the polyploid-phase extra: pip install torchgwas[polyploid-phase]")` |
| Julia not discovered, `auto_install_julia=False`, no tty | `RuntimeError` naming both escape hatches (https://julialang.org/downloads/ and `auto_install_julia=True`) |
| Julia found but `julia --version` < 1.10 | `RuntimeError("Julia at {path} is v{ver}; PolyOrigin requires >= 1.10.")` |
| `julia_path=` or `TORCHGWAS_JULIA` points to a nonexistent/non-executable file | `ValueError` |
| Managed install network failure | Propagate juliacall's exception; point to `~/.julia/logs/juliapkg.log` |
| PolyOrigin.jl install fails (transient network, Git host down) | `RuntimeError` with `Pkg.resolve()` recovery pointer |
| First-call JIT cost (30–60 s) | `logger.info(...)` single line; not an error |
| Runtime cached | `get_runtime` is memoized on `(_jl, _polyorigin, _version)`; subsequent calls skip all of steps 1–6 |

No silent fallback to a Python reimplementation. Always-external-or-error.

### 4.2 Input validation (raised before Julia is touched)

| Failure | Response |
| --- | --- |
| `ploidy not in {2, 4, 6}` | `ValueError` |
| Pedigree TSV missing required cols (`offspring`, `parent1`, `parent2`) | `ValueError` naming missing col(s) |
| Duplicate offspring ID in pedigree | `ValueError` listing duplicate |
| Parent referenced but absent from founder rows | `ValueError` |
| Multi-generation pedigree (a parent is another row's offspring) | `ValueError("PolyOrigin models F1 + founder selfings only; multi-generation pedigrees not supported. Offending parent: {pid}.")` |
| Map TSV missing `marker`/`chrom`/`pos_bp` | `ValueError` |
| Non-monotonic `pos_bp` within a chromosome | `ValueError` naming first offender |
| Map lacks `cm` | `logger.warning` + synthesize via `recomrate`; don't fail |
| `probs` shape ≠ `(n, m, k+1)` for ploidy `k` | `ValueError` |
| `probs` rows < 0.999 sum | `logger.warning` + renormalize |
| `sample_ids` / `variant_ids` missing when `probs` is a tensor | `ValueError` |
| probs sample IDs ⊋ pedigree IDs | `ValueError` listing first 10 missing + count |
| `parent_phased_csv` given but missing a pedigree parent | `ValueError` |
| Parent in both probs and `parent_phased_csv` | `logger.warning` + prefer pre-phased |

### 4.3 Julia exception mapping

- `juliacall.JuliaError` raised inside `polyOrigin()` → `RuntimeError(f"PolyOrigin failed: {exc}")`, preserving the Julia stacktrace via `__cause__`.
- Call completed but `done.marker` missing → `RuntimeError("PolyOrigin call returned but produced no done.marker — check the Julia log in the workdir.")`.
- Call completed, `done.marker` present, but an expected output CSV missing → `RuntimeError` naming the file.
- No subprocess timeout path (no subprocess); long hangs surface as Python hangs, interruptible via Ctrl-C.

### 4.4 Output validation

After parsing:

- `origin_probs.shape == (n_off, ploidy, m, n_parent_haps)` — else `RuntimeError`.
- `origin_probs.sum(dim=-1)` within `atol=1e-3` of 1.0 (looser than Phase 55 — PolyOrigin's CSV writer rounds to 4 decimal places); below tolerance → renormalize + `logger.warning`.
- `haplotypes` values in `{0, 1, ..., n_parent_haps-1, -1}` (`-1` = mixed-ploidy padding); anything else → `RuntimeError`.
- `parent_phased` allele values in `{0, 1, -1}` — else `RuntimeError`.
- `map_refined` row count matches `m` and IDs are a permutation of input — else `RuntimeError`.

### 4.5 Dependency matrix summary

| Dependency | Install trigger | Disk cost |
| --- | --- | --- |
| `juliacall` (Python) | `pip install torchgwas[polyploid-phase]` | ~15 MB |
| Julia ≥ 1.10 | first `run_polyorigin` call **iff** no existing Julia found **iff** user consents | ~300 MB (managed) or 0 (reused) |
| `PolyOrigin.jl` v1.0.5 | first `run_polyorigin` call | ~5 MB |
| `pandas`, `torch` | already hard deps | n/a |

Users who never invoke `run_polyorigin` pay only the 15 MB for `juliacall` — and only if they opted into the extra.

`pyproject.toml` addition:

```toml
[project.optional-dependencies]
polyploid-phase = ["juliacall>=0.9"]
```

---

## 5. Testing and validation gate

Two tiers. Tier 1 always runs. Tier 2 runs locally and in a release-time job where Julia + PolyOrigin.jl are provisioned.

### 5.1 Tier 1 — always-on Python tests (CI on every push)

File: `tests/test_phase_polyorigin.py`. No Julia involved. `_polyorigin_runtime.get_runtime` and its helpers are monkeypatched. Ships canned PolyOrigin-output CSVs under `tests/fixtures/phase_polyorigin/` (toy tetraploid F1: 2 parents × 6 offspring × 10 markers).

| Category | Test |
| --- | --- |
| Pedigree conversion | Single biparental → pedfile.csv with `population=1`, parents as founders |
| | Multi-family connected (3 parents, 2 sub-families) → correct integer populations |
| | Offspring-as-parent → `ValueError` (multi-generation) |
| | Duplicate offspring ID → `ValueError` |
| | Parent in offspring col but not founder col → `ValueError` |
| | Mixed-ploidy column respected; default falls back to `--ploidy` kwarg |
| Genofile conversion | `(n, m, k+1)` probs + IDs + map → correct header, rowcount=m, probability-encoded cells (4 decimals) |
| | Parent-phased escape hatch → parents as `phasedgeno`, offspring probs unchanged |
| | Parent in both sources → warn + pre-phased wins |
| | Probs rows summing to 0.999 → silent renormalize |
| | Probs rows summing to 0.9 → renormalize + warning |
| Map handling | `cm` column present → used verbatim |
| | `cm` missing → synthesize via `recomrate` with warning |
| | Non-monotonic bp → `ValueError` with offender |
| Input validation | `ploidy not in {2,4,6}` → `ValueError` |
| | `probs` tensor without `sample_ids` → `ValueError` |
| | probs IDs ⊋ pedigree IDs → `ValueError` listing missing |
| Bootstrap discovery | `TORCHGWAS_JULIA` set to stub `julia --version=1.10.0` → picked |
| | Stub `julia --version=1.8.0` → rejected |
| | Nothing on PATH + `auto_install_julia=False` + non-tty → `RuntimeError` naming both escape hatches |
| | `julia_path=` to nonexistent file → `ValueError` |
| Runtime stubbing | Monkeypatched runtime writes canned CSVs → correct `PhasingResult` shape, IDs, tensors parse |
| | Monkeypatched runtime raises `JuliaError` → `RuntimeError` preserving `__cause__` |
| | Runtime returns but no `done.marker` → `RuntimeError` |
| | Missing output CSV post-run → `RuntimeError` naming file |
| Output parsing | `*_genoprob.csv` → correct shape + row sums |
| | argmax → valid haplotype int values |
| | `*_parentphased.csv` → values in `{0,1}` |
| | `refinemap=True` reorder → `variant_ids` reflects new order, `map_refined=True` |
| | Mixed-ploidy → `haplotypes` dim-1 = max_ploidy, `-1` padding correct, dict correct |
| Metadata | `sha256(genofile+pedfile+map)` matches `meta.json.input_hash` |
| | `cmd` captures Julia call for reproducibility |
| Atomicity | Partial runtime output (2 of 5 CSVs) → `RuntimeError`, no `<output>.*` on disk |

~25 tests. Pass on Linux + Windows × 3.10/3.11/3.12 with zero Julia installed.

### 5.2 Tier 2 — end-to-end with real PolyOrigin (release-time + local)

File: `tests/test_phase_polyorigin_e2e.py`. Module-level `pytest.mark.skipif` gates on (i) `juliacall` importable, (ii) a Julia ≥ 1.10 discoverable, (iii) PolyOrigin.jl available or `TORCHGWAS_ALLOW_AUTO_INSTALL=1` set.

| Test | Pass criterion |
| --- | --- |
| `test_parity_tetraploid_f1` | Build genofile+pedfile, call `run_polyorigin(..., keep_workdir=True)`. A separate test-local Julia snippet calls `PolyOrigin.polyOrigin()` directly on the *same* files. Compare: `torch.allclose(origin_probs_ours, origin_probs_ref, atol=1e-4)` and `parent_phased_ours == parent_phased_ref`. Parity = our wrapper vs bare `polyOrigin()` on identical inputs. |
| `test_parity_hexaploid_f1` | Same contract, ploidy=6, simulated data in-test. |
| `test_simulated_f1_haplotype_recovery` | Simulated tetraploid F1 (2 parents, 50 offspring, 200 markers, 30× depth). Full chain: `dosage-call` → `run_polyorigin`. Haplotype-origin accuracy ≥ calibrated rate − 2 %. |
| `test_simulated_f1_recovery_lowdepth` | Same at 8× depth. Same calibrated-minus-2 % pattern. |
| `test_mixed_ploidy_f1` | Simulated 4n × 2n cross → triploid offspring. Confirm `per_individual_ploidy` correct, `-1` padding in `haplotypes`, no crashes. *If PolyOrigin rejects triploid offspring under F1 rules, downgrade to an expected-`ValueError` at our layer.* |
| `test_gwaspoly_potato_roundtrip` | **The roadmap's named integration gate.** Published GWASpoly tetraploid potato F1 dataset. Full chain: `dosage-call` → `run_polyorigin` → `HaplotypeGWAS`. Capture haplotype p-values. Re-run after phasing the same data via bare `polyOrigin()` + manual tensor construction. Compare Phase 46 haplotype p-values: Spearman ρ > 0.999, max absolute `-log10(p)` difference < 0.01. |

### 5.3 Calibration helper (dev-time, not CI)

File: `bench/calibrate_polyorigin_recovery.py`. Not pytest-collected. Run once at phase sign-off to measure haplotype-origin accuracy under the `test_simulated_f1_recovery_*` parameters. Paste observed rate + `# source: bench/calibrate_polyorigin_recovery.py run 2026-XX-XX` into Tier 2 assertions.

**Why 2 % slack (same rationale as Phase 55)**: PolyOrigin's HMM is deterministic given a fixed seed; the slack is for cross-version drift between v1.0.5 and future v1.x releases. Tightens if calibration shows near-100 % recovery.

### 5.4 The gate — Phase 56 is complete when

1. All Tier 1 tests pass on every CI matrix job (Linux + Windows × 3.10/3.11/3.12), with zero Julia installed.
2. All Tier 2 tests pass on a machine with Julia ≥ 1.10 + PolyOrigin.jl v1.0.5.
3. `test_gwaspoly_potato_roundtrip` specifically passes — the roadmap's named integration target.
4. The Phase 56 entry in `docs/ROADMAP.md` is **removed** (shipped).
5. `torchgwas phase-poly --help` captured in `docs/cli.md`.
6. `docs/getting-started/polyploid_phasing.md` has one recipe for the `dosage-call → phase-poly → haplotype-scan` chain.
7. A memory file `project_polyploid_phasing.md` added alongside the other per-phase memories.
8. No `csrc/` changes → no `bench/native_speedups.md` entry.

### 5.5 Test-fixture footprint

Tier 1 canned CSVs (tetraploid F1, 10 markers) are ~4 KB total. The `test_gwaspoly_potato_roundtrip` dataset is public but ~several MB — lazy-downloaded on first Tier 2 run into `tests/fixtures/_cache/` (gitignored), mirroring BEAGLE reference panels in Phase 41+ benchmarks.

---

## 6. CLI

New subcommand `torchgwas phase-poly`, wired into `torchgwas/cli.py`. No `cli.py` decomposition in this phase (same standing cleanup as Phase 55).

```bash
torchgwas phase-poly \
  --probs out/dcall.probs.pt \
  --pedigree pedigree.tsv \
  --map markers.tsv \
  --output out/phased \
  --ploidy 4 \
  [--parent-phased parents_phased.csv] \
  [--no-refinemap] \
  [--recomrate 1.0] \
  [--nworkers N] \
  [--seed 1234] \
  [--julia-path /custom/julia] \
  [--auto-install / --no-auto-install] \
  [--keep-workdir]
```

**Writes**:
- `<output>.haplotypes.pt` — `(n_off, ploidy, m)` int8
- `<output>.origin_probs.pt` — `(n_off, ploidy, m, n_parent_haps)` float64
- `<output>.parent_phased.pt` — `(n_parents, max_ploidy, m)` int8
- `<output>.postdose_probs.pt` — `(n_off, m, max_ploidy+1)` float64
- `<output>.map_refined.tsv`, `<output>.valent_diag.tsv`
- `<output>.meta.json`

**Defaults**:
- `--refinemap` on (PolyOrigin's recommendation).
- `--nworkers 1`.
- `--auto-install` prompts interactively on a tty; `--auto-install` / `--no-auto-install` overrides for non-interactive runs.

**Documented pipeline recipe** (in `docs/getting-started/polyploid_phasing.md`):

```bash
# End-to-end tetraploid F1 pipeline
torchgwas dosage-call  --vcf calls.vcf.gz              --output out/dcall  --ploidy 4
torchgwas phase-poly   --probs out/dcall.probs.pt      --pedigree ped.tsv \
                       --map markers.tsv               --output out/phased --ploidy 4

python -c "
import torch
from torchgwas.models import HaplotypeGWAS
haps = torch.load('out/phased.haplotypes.pt')
scanner = HaplotypeGWAS(haplotypes=haps, ploidy=4)
# ... feed phenotype + run
"
```

### Deferred cleanup (not Phase 56)

- `haplotypes.pt → phased VCF` export utility.
- `phase-poly` wiring into the `torchgwas pipeline` monolithic subcommand.
- WhatsHap-polyphase wrapper for unrelated diversity panels.

---

## 7. Non-goals and explicit decisions

Recorded so the implementation plan doesn't re-litigate:

1. **Single-tool first, queue follow-ups.** WhatsHap-polyphase, hapCON, TetraOrigin are separate phases.
2. **PolyOrigin-only first cut.** `PolyOriginR` and `PolyOriginCmd` are dormant; a Python wrapper is the effective reference.
3. **F1 crosses + founder selfings only.** Multi-generation pedigrees → explicit `ValueError`. Upstream hard constraint.
4. **Ploidies {2, 4, 6} only.** Upstream hard constraint.
5. **Tensor-return output contract, no native VCF.** `load_haplotypes` is bypassed; Phase 46/47 consume tensors anyway. CSV→VCF export deferred.
6. **Input chains off Phase 55**: `.probs.pt` (parents+offspring) + pedigree TSV (3-col) + map TSV. `--parent-phased` escape hatch for pre-phased parents.
7. **In-process Julia via `juliacall`, no subprocess driver.** Motivated by cross-OS auto-install requirement.
8. **Discover-first bootstrap**: check PATH + common install paths + user override; only auto-install Julia if missing *and* the user consents (interactive tty prompt or explicit `auto_install_julia=True` / `--auto-install`).
9. **No Julia sysimage build shipped.** Juliacall's own caching covers same-process warmth; sysimage recipe may be a later documentation follow-up.
10. **Both parity *and* simulated-F1 recovery** in the validation gate, plus the GWASpoly potato round-trip as the named integration target.
11. **`polyploid-phase` pip extra**, not a hard dep.
12. **`juliapkg.json` pins PolyOrigin.jl to URL+rev (v1.0.5).** Upgrades are deliberate spec-revision events.
13. **Recovery thresholds self-calibrated** via `bench/calibrate_polyorigin_recovery.py`, same pattern as Phase 55.
14. **Mixed-ploidy via `-1`-padded tensor + `per_individual_ploidy` dict**, not a list-of-variable-width arrays. Torch-native; dict is small, explicit, inspectable.

---

## 8. Open questions that do not block implementation

- **PolyOrigin UUID**: `juliapkg.json` needs the UUID copied verbatim from upstream `Project.toml` during coding.
- **Mixed-ploidy F1 acceptance**: flagged in `test_mixed_ploidy_f1`. Implementer verifies against a live run; downgrade to expected-`ValueError` if PolyOrigin rejects triploid offspring under its F1 rules.
- **juliacall env var for pinning existing Julia binary**: named generically in Section 2.3; implementer confirms exact name (`PYTHON_JULIAPKG_EXE` per current juliacall docs, but version-dependent).
- **`load_haplotypes` reuse**: the legacy `load_haplotypes(phased_vcf_path, ploidy)` is unused by Phase 56 (we skip VCF). No change needed; it remains for BEAGLE-diploid and user-external-phaser paths.

---

## Appendix: file-level impact summary

**New files**:
- `torchgwas/preprocess/phase_polyorigin.py` — ~500 LOC including helper converters.
- `torchgwas/preprocess/_polyorigin_runtime.py` — ~150 LOC, discover-first bootstrap.
- `torchgwas/preprocess/juliapkg.json` — dep manifest, hand-edited.
- `tests/test_phase_polyorigin.py` — Tier 1, ~25 tests.
- `tests/test_phase_polyorigin_e2e.py` — Tier 2, ~6 tests.
- `tests/fixtures/phase_polyorigin/` — canned CSVs, toy VCF.
- `bench/calibrate_polyorigin_recovery.py` — dev-time calibration helper.
- `docs/getting-started/polyploid_phasing.md` — one recipe.
- `~/.claude/projects/.../memory/project_polyploid_phasing.md` — phase memory.

**Modified files**:
- `torchgwas/cli.py` — one new subcommand (`phase-poly`) + argparse entry.
- `torchgwas/preprocess/__init__.py` — re-export `run_polyorigin`, `PhasingResult`.
- `pyproject.toml` — add `[project.optional-dependencies]` entry for `polyploid-phase = ["juliacall>=0.9"]`; include `juliapkg.json` as package data.
- `docs/ROADMAP.md` — remove the Phase 56 entry on ship.
- `docs/cli.md` — `torchgwas phase-poly --help` capture.
- `CLAUDE.md` — add `phase-poly` to CLI commands; bump subcommand count from 36 to 37.

**Unchanged**:
- `torchgwas/preprocess/phase.py` (legacy BEAGLE diploid wrapper).
- `torchgwas/preprocess/dosage_call.py` (Phase 55 consumer).
- `torchgwas/models/haplotype_gwas.py` (already ploidy-generic on input).
- `torchgwas/preprocess/load_haplotypes` in `phase.py` (unused by Phase 56 but preserved for other paths).
