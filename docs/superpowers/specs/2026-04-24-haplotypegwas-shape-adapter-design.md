# Phase 56 Tier-A #1 — HaplotypeGWAS Shape Adapter

**Status**: Design approved 2026-04-24. Awaiting implementation plan.
**Scope**: Tier-A close-out for Phase 56. Bridges PolyOrigin's joint-origin state output into the per-chromosome-copy tensor that `HaplotypeGWAS.scan()` already consumes.
**Authors**: Sikiru Atanda + Claude Code (brainstorming session 2026-04-24).
**Spec parent**: `docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md`.

---

## 1. Motivation and scope boundary

`PhasingResult.haplotypes` is `(n_off, m) int64` — argmax of PolyOrigin's joint-origin-state posterior, not per-chromosome-copy alleles. `HaplotypeGWAS._enumerate_haplotypes_phased` (Phase 46) consumes `(n, ploidy, m)` integer-valued tensors, treating each per-marker copy-allele sequence as a haplotype identifier. No bridge exists today, so Phase 56 ships rich tensors but cannot feed the haplotype-association scanners that motivated it.

This adapter delivers the missing bridge: a decoder that expands PolyOrigin's joint-origin state index per offspring-marker into per-copy parental-allele assignments. It runs eagerly inside `run_polyorigin`, produces two new persisted artifacts, and lets downstream users hand `result.haplotypes_per_copy` directly to `HaplotypeGWAS.scan()` — no Julia dependency at decode time, no manual reshape, no separate utility call.

### State-table sourcing — Python reimplementation + Julia round-trip validation

Cross-check verified PolyOrigin v1.0.3 has **no documented state-table API**. The canonical state ordering lives inside `src/inferF1.jl` upstream. The honest path uses two layers:

1. **Python enumeration**, deterministic and version-stable. For an F1 pedigree at ploidy `k` with two founder parents, bivalent meiosis only:
   - Each parent gamete is a sorted `(k/2)`-element subset of `{0..k-1}` in lexicographic order — `C(k, k/2)` gametes per parent.
   - State `s = (p1_gamete_idx, p2_gamete_idx)`, flat-indexed as `s = p1_gamete_idx * n_gametes + p2_gamete_idx`.
   - State table row `s`: tuple of `k` parental-copy indices, half from parent 1 and half from parent 2.
   - For tetraploid 2-parent F1 → 36 states; ploidy=2 → 4 states; ploidy=6 → 400 states.

2. **Round-trip validation against PolyOrigin's own output**, run once per `run_polyorigin` call while Julia output is fresh. Compute expected dosage two independent ways — via our state table (using `origin_probs` and `parent_phased`) and via `postdose_probs` directly. Equality (within `atol=1e-3`) confirms our enumeration matches PolyOrigin's. Mismatch raises `RuntimeError` with diagnostic info — this is the right place to catch a future PolyOrigin reorder.

The validation lives in `run_polyorigin` (where Julia is live by definition). Downstream users of `PhasingResult` have **zero Julia dependency**.

### In scope

- One pure helper `_enumerate_state_table(ploidy)` in `phase_polyorigin.py`.
- One pure helper `_decode_haplotypes_per_copy(...)`.
- One validation step `_validate_state_table(...)` invoked inside `run_polyorigin`.
- Two new `PhasingResult` fields: `state_table` and `haplotypes_per_copy`.
- Two new persisted artifacts: `<output>.state_table.pt`, `<output>.haplotypes_per_copy.pt`.
- ~10 new Tier 1 tests (CI-always-on, juliacall-stubbed) and 1 Tier 2 end-to-end test feeding `haplotypes_per_copy` into `HaplotypeGWAS.scan()`.

### Out of scope (explicitly deferred)

- **Mixed-ploidy F1** — `ValueError` if `per_individual_ploidy` values aren't all equal. A separate phase will add per-offspring state tables when needed.
- **Multivalent (quadrivalent) pairing** — PolyOrigin v1.0.3 default is bivalent-only. Multivalent support requires both upstream and adapter changes.
- **Migration tool for older `PhasingResult` artifacts** — re-run `phase-poly` to get the new fields; no backport.
- **Allele-probability floats per copy** — strict integer alleles, matching `HaplotypeGWAS`'s existing contract.
- **Allele values restricted to 0/1** — adapter passes through whatever integer values `parent_phased` carries. Biallelic data lands at 0/1 in practice, but the contract doesn't assume.
- **CLI surface changes** — `torchgwas phase-poly` already runs the full `run_polyorigin` flow; users get the two new files automatically.

### Integration point

```
existing run_polyorigin output (origin_probs, parent_phased,
                                postdose_probs, haplotypes, ...)
                  │
                  ▼
       NEW: same-ploidy guard + state-table enumerate + validate + decode
                  │
                  ▼
PhasingResult now carries:
   • state_table: (n_states, ploidy) int8
   • haplotypes_per_copy: (n_off, ploidy, m) int8   ← drop-in for HaplotypeGWAS.scan
                  │
                  ▼
   torchgwas.models.HaplotypeGWAS.scan(Y=Y, G=G, haplotypes=result.haplotypes_per_copy)
```

---

## 2. Components and interfaces

Three new pure helpers in `torchgwas/preprocess/phase_polyorigin.py` plus two new `PhasingResult` fields. No new files; no changes to `_polyorigin_runtime.py`.

### 2.1 New `PhasingResult` fields

```python
@dataclass
class PhasingResult:
    # ... (existing 18 fields unchanged) ...
    state_table: Tensor              # (n_states, ploidy) int8
    haplotypes_per_copy: Tensor      # (n_off, ploidy, m) int8
```

State-table encoding: each value `v ∈ [0, 2*ploidy)` is `parent_id * ploidy + copy_in_parent`. For ploidy=4: values `0..3` index parent1's copies, `4..7` index parent2's copies. Compact, ploidy-generic, no separate parent-id tensor.

### 2.2 `_enumerate_state_table(ploidy: int) -> Tensor`

```python
def _enumerate_state_table(ploidy: int) -> Tensor:
    """Build the joint-origin state table for a 2-parent F1 at given ploidy.

    Bivalent meiosis only: each parent contributes ploidy/2 copies. Gametes
    are sorted (ploidy/2)-element subsets of {0..ploidy-1} in lexicographic
    order. State s = (p1_gamete_idx, p2_gamete_idx) flat-indexed as
    s = p1_gamete_idx * n_gametes + p2_gamete_idx.

    Returns
    -------
    Tensor, shape (n_states, ploidy), int8
        Each row holds ploidy values v in [0, 2*ploidy):
        v = parent_id * ploidy + copy_in_parent.
    """
```

Pure, deterministic, depends only on `ploidy`. Independent of any pedigree, parent IDs, or marker layout. Cacheable; we don't need module-level caching for the foreseeable workload size (max 400 rows for ploidy=6).

### 2.3 `_validate_state_table(...)`

```python
def _validate_state_table(
    state_table: Tensor,           # (n_states, ploidy) int8
    origin_probs: Tensor,          # (n_off, m, n_states) float64
    parent_phased: Tensor,         # (n_parents, m, max_ploidy) int8
    postdose_probs: Tensor,        # (n_off, m, ploidy+1) float64
    ploidy: int,
    atol: float = 1e-3,
) -> None:
    """Raise on state-ordering mismatch with PolyOrigin.

    For each (offspring, marker) cell, compute expected dosage two ways:

    - Via state_table:
        dose_at(j, s) = sum over copies in row s of state_table of
                        parent_phased's allele at marker j
        E_state[i, j]  = sum_s origin_probs[i, j, s] * dose_at(j, s)

    - Via postdose_probs:
        E_post[i, j]   = sum_d d * postdose_probs[i, j, d]

    Both arrive at the same per-cell expected dosage *only if* our
    state_table rows correspond to PolyOrigin's state indices in the
    same order. A row-permutation in our table changes E_state but
    not E_post, so the comparison is sensitive to ordering bugs.

    Raises
    ------
    RuntimeError
        If max |E_state - E_post| > atol. Message includes max-deviation
        cell coordinates, both computed values, and a hint about the
        likely cause (PolyOrigin reordering between releases).
    """
```

Single round-trip across all `(offspring, marker)` cells. Vectorized in torch — cheap.

### 2.4 `_decode_haplotypes_per_copy(...)`

```python
def _decode_haplotypes_per_copy(
    haplotypes: Tensor,            # (n_off, m) int64 — argmax joint-origin state
    parent_phased: Tensor,         # (n_parents, m, max_ploidy) int8
    state_table: Tensor,           # (n_states, ploidy) int8
    ploidy: int,
) -> Tensor:
    """Expand joint-origin state indices into per-copy parental alleles.

    For each (offspring i, marker j) with state s = haplotypes[i, j]:
        for k in 0..ploidy-1:
            v = state_table[s, k]
            parent_id, copy_in_parent = divmod(v, ploidy)
            output[i, k, j] = parent_phased[parent_id, j, copy_in_parent]

    Vectorized via torch.gather. Returns (n_off, ploidy, m) int8.
    """
```

Independently unit-testable against hand-crafted state_table + parent_phased + haplotypes.

### 2.5 `run_polyorigin` orchestration extension

After the existing output-parse step and before `_persist_result`:

```python
ploidy_values = set(per_individual_ploidy.values())
if len(ploidy_values) > 1:
    raise ValueError(
        f"haplotypes_per_copy decoding requires uniform ploidy across all "
        f"individuals; got {sorted(ploidy_values)}. Mixed-ploidy F1 is "
        f"deferred."
    )
ploidy = next(iter(ploidy_values))

state_table = _enumerate_state_table(ploidy)
_validate_state_table(state_table, origin_probs, parent_phased, postdose_probs, ploidy)
haplotypes_per_copy = _decode_haplotypes_per_copy(haplotypes, parent_phased, state_table, ploidy)
```

Both new fields are added to the `PhasingResult` constructor at the end. Workdir cleanup and persistence proceed as before.

### 2.6 Persistence

`_persist_result` gains two atomic writes alongside the existing seven:

```python
_persist("state_table.pt",         lambda p: torch.save(result.state_table, p))
_persist("haplotypes_per_copy.pt", lambda p: torch.save(result.haplotypes_per_copy, p))
```

Atomicity guarantee unchanged: failure at any stage rolls back all already-renamed siblings.

### 2.7 Directory layout

```
torchgwas/preprocess/
├── phase_polyorigin.py   # MODIFIED — adds 3 helpers + 2 PhasingResult fields + orchestration extension
├── _polyorigin_runtime.py # unchanged
└── juliapkg.json          # unchanged
```

No new modules. Total file growth: ~150 LOC of pure logic + tests.

---

## 3. Data flow

```
existing run_polyorigin output (parsed from PolyOrigin CSVs)
   origin_probs       (n_off, m, n_states) float64
   parent_phased      (n_parents, m, max_ploidy) int8
   postdose_probs     (n_off, m, max_ploidy+1) float64
   haplotypes         (n_off, m) int64        = origin_probs.argmax(-1)
   per_individual_ploidy: dict[str, int]
   │
   ▼ NEW: same-ploidy guard
   if len(set(per_individual_ploidy.values())) > 1: ValueError
   ploidy = unique value
   │
   ▼ NEW: enumerate state table (pure, deterministic; depends only on ploidy)
   state_table = _enumerate_state_table(ploidy)
   #  shape (n_states, ploidy) int8
   │
   ▼ NEW: validate against PolyOrigin's emitted output
   _validate_state_table(state_table, origin_probs, parent_phased,
                         postdose_probs, ploidy)
   # raises RuntimeError if max |E_state - E_post| > 1e-3
   │
   ▼ NEW: decode per-copy alleles
   haplotypes_per_copy = _decode_haplotypes_per_copy(
       haplotypes, parent_phased, state_table, ploidy)
   # shape (n_off, ploidy, m) int8
   │
   ▼ assemble PhasingResult — existing 18 fields + state_table + haplotypes_per_copy
   │
   ▼ persist atomically — existing 7 artifacts + 2 new
   <output>.state_table.pt
   <output>.haplotypes_per_copy.pt
   │
   ▼ downstream (no code changes; new capability)
   from torchgwas.models import HaplotypeGWAS
   scanner = HaplotypeGWAS(method="block", test="f_test", ploidy=ploidy)
   scanner.scan(Y=Y, G=G, haplotypes=result.haplotypes_per_copy)
```

**Validation timing**: the only new step that *can* fail is the state-table round-trip. It runs after PolyOrigin emits its output but *before* persistence — so a mismatch leaves the workdir intact (with `keep_workdir=True` for debug) and produces no `<output>.*` artifacts. Phase 56's existing rollback semantics carry over unchanged.

**Determinism**: `_enumerate_state_table(ploidy)` is a pure function of one integer. `_decode_haplotypes_per_copy` is pure tensor indexing. Same inputs → bit-identical outputs.

**Mixed-ploidy detection**: gated *before* state-table enumeration so the failure surfaces with a clear, scope-specific message rather than midway through the validate step.

---

## 4. Error handling

Three new failure classes, all inside `run_polyorigin`. No new environment-dependency errors (decoder is pure Python).

### 4.1 Mixed-ploidy guard

```
ValueError(
    "haplotypes_per_copy decoding requires uniform ploidy across all "
    "individuals; got {sorted_ploidy_set}. Mixed-ploidy F1 is deferred."
)
```

Raised before any state-table work. Pedigree TSV with mixed-ploidy column entries triggers this; user response is to split the pedigree into same-ploidy sub-runs or wait for the deferred mixed-ploidy phase.

### 4.2 State-table round-trip failure

```
RuntimeError(
    "State-table round-trip mismatch — our enumeration disagrees with "
    "PolyOrigin v{tool_version}. Max deviation {max_dev:.4f} at "
    "(offspring={off_id}, marker={var_id}); ours={ours:.4f}, "
    "PolyOrigin={theirs:.4f}. Likely cause: PolyOrigin reordered states "
    "between releases. Open an issue with the offending phase-poly inputs."
)
```

Version-drift safety net. `_enumerate_state_table` is the single point of truth; any reorder is fixed there.

### 4.3 Decoder shape sanity

```
RuntimeError(
    "_decode_haplotypes_per_copy produced unexpected shape "
    "{actual_shape}; expected (n_off={n_off}, ploidy={ploidy}, m={m})."
)
```

Defensive check after the decoder; cheap; catches indexing bugs during refactors.

### 4.4 No new dependency-environment errors

Decoder runs in pure Python. juliacall / Julia bootstrap conditions are unchanged from Phase 56's existing flow (Section 4.1 of `2026-04-23-phase-56-polyploid-phasing-design.md`).

---

## 5. Testing and validation gate

Two tiers. Tier 1 always-on (CI). Tier 2 gated on Julia + PolyOrigin.jl.

### 5.1 Tier 1 — pure-helper unit tests

File: append to `tests/test_phase_polyorigin.py`. Adds the three helpers to the existing import block.

| Category | Test |
| --- | --- |
| `_enumerate_state_table` | `test_enumerate_state_table_count` — ploidy=2 → 4 states; ploidy=4 → 36; ploidy=6 → 400. |
| | `test_enumerate_state_table_first_state_canonical` — ploidy=4 state 0 is `[0, 1, 4, 5]` (parent1 gamete (0,1), parent2 gamete (0,1)) under our `parent_id*ploidy + copy_in_parent` encoding. |
| | `test_enumerate_state_table_lex_ordering` — gametes within each parent appear in lex order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3). |
| `_decode_haplotypes_per_copy` | `test_decode_known_inputs` — hand-crafted small state_table; parent_phased = `[[1,0,1,0], [0,1,0,1]]`; haplotypes = `[[0]]` (1 offspring, 1 marker, state 0) → expected output `[[1, 0, 0, 1]]` (parent1 copies 0,1 = alleles 1,0; parent2 copies 0,1 = alleles 0,1). |
| `_validate_state_table` | `test_validate_passes_on_self_consistent_input` — synthesize origin_probs + matching postdose_probs from our own state_table → validation passes silently. |
| | `test_validate_fails_on_reorder` — permute state_table rows after constructing the consistent inputs → `RuntimeError` matching the round-trip-mismatch diagnostic. |

### 5.2 Tier 1 — orchestration tests

| Test | Pass criterion |
| --- | --- |
| `test_run_polyorigin_haplotypes_per_copy_field_populated` | Stubbed runtime → `result.haplotypes_per_copy.shape == (n_off, ploidy, m)`, dtype int8, values in `{0, 1}`. |
| `test_run_polyorigin_state_table_field_populated` | Stubbed runtime → `result.state_table.shape == (n_states, ploidy)`, all values in `[0, 2*ploidy)`. |
| `test_run_polyorigin_mixed_ploidy_rejected` | Pedigree TSV with mixed ploidy → `ValueError` matching `uniform ploidy`. No persistent artifacts. |
| `test_persist_per_copy_artifacts` | Happy path → `<output>.state_table.pt` and `<output>.haplotypes_per_copy.pt` exist, round-trip via `torch.load`. |

Total Tier 1 after this adapter: existing 43 + ~10 new = ~53.

### 5.3 Tier 2 — end-to-end HaplotypeGWAS integration

File: append to `tests/test_phase_polyorigin_e2e.py`.

| Test | Pass criterion |
| --- | --- |
| `test_haplotypegwas_scan_consumes_per_copy` | Build the same toy 6-offspring × 10-marker tetraploid F1 used in `test_parity_tetraploid_f1`. Run `run_polyorigin` (real Julia). Construct synthetic phenotype `Y` (e.g., expected-dosage-derived plus noise) and `G = expected_dosage(result.postdose_probs, ploidy=4)`. Call `HaplotypeGWAS(method="block", test="f_test", ploidy=4).scan(Y=Y, G=G, haplotypes=result.haplotypes_per_copy)`. Assert: returns a `HaplotypeGWASResult`; `len(result.start) > 0`; all per-block `p_global` are valid floats in `[0, 1]`. **Smoke test for end-to-end correctness — not a calibrated power gate.** |

### 5.4 The gate — adapter is complete when

1. All Tier 1 tests pass on every CI matrix job (Linux + Windows × Python 3.10/3.11/3.12) with zero Julia installed.
2. `test_haplotypegwas_scan_consumes_per_copy` passes locally on a machine with Julia + PolyOrigin.jl available.
3. `project_post_phase56_backlog.md` Tier A #1 entry is marked closed; `project_polyploid_phasing.md` is updated to note the adapter shipped.
4. No new external dependencies (no edits to `pyproject.toml`).

---

## 6. Non-goals and explicit decisions

Locked during this brainstorm so the implementation plan doesn't re-litigate:

1. **Python reimplementation + Julia round-trip validation, not API queries.** No documented PolyOrigin state-table API. Internal-function calls would be brittle. Combinatorial enumeration in Python + dosage round-trip is honest and version-stable.
2. **Eager materialization at `run_polyorigin` time, not lazy.** New fields are computed, validated, and persisted in the same call.
3. **Same-ploidy F1 only.** Mixed-ploidy raises `ValueError`. Per-individual state tables are a follow-up.
4. **Bivalent meiosis only.** Matches PolyOrigin v1.0.3 default.
5. **Integer allele passthrough, not strict 0/1.** `parent_phased` values pass through unchanged.
6. **State encoding: `parent_id * ploidy + copy_in_parent`.** Single int8 tensor, ploidy-generic.
7. **Validation: marginal expected-dosage match within `atol=1e-3`.** A reorder of states changes per-marker expected-dosage curves; one all-cells comparison catches it.
8. **Vectorized decoder.** `gather`-based torch indexing, not an explicit Python loop.
9. **No CLI changes.** Users get the new persisted files automatically.
10. **No backport for older `PhasingResult` artifacts.** Re-run `phase-poly` to upgrade.

---

## 7. Open questions that do not block implementation

- **Caching `_enumerate_state_table` results** — depends only on `ploidy ∈ {2, 4, 6}`. Trivial cache via `functools.lru_cache`; can be added if profiling shows it's worth the complexity. Not blocking.
- **Per-state diagnostic in `_validate_state_table`** — the current diagnostic names the worst-deviation cell and both computed values. A future improvement could enumerate all states with non-zero posterior contributions and compare per-state — useful if a reorder affects only some states. Not blocking.
- **PolyOrigin internal symbol vendoring** — if a future PolyOrigin release changes the state ordering and our enumeration needs updating, the fix is a one-time edit to `_enumerate_state_table`. The validation step catches the discrepancy on the first failing run. Not blocking.

---

## Appendix: file-level impact summary

**Modified files**:
- `torchgwas/preprocess/phase_polyorigin.py` — add 3 helpers, extend `PhasingResult`, extend `run_polyorigin`, extend `_persist_result`. ~150 LOC of pure logic.
- `tests/test_phase_polyorigin.py` — append ~9 Tier 1 tests.
- `tests/test_phase_polyorigin_e2e.py` — append 1 Tier 2 end-to-end test.

**Unchanged**:
- `torchgwas/preprocess/_polyorigin_runtime.py`, `juliapkg.json`, `pyproject.toml`, `torchgwas/cli.py`, `torchgwas/__main__.py`, `torchgwas/models/haplotype_gwas.py`, `docs/getting-started/polyploid_phasing.md`, all other Phase 56 artifacts.

**New** (none — all edits land in existing files).
