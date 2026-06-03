# HaplotypeGWAS Shape Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a per-chromosome-copy allele decoder that converts Phase 56's PolyOrigin joint-origin-state output into the `(n_off, ploidy, m)` integer tensor that `HaplotypeGWAS.scan()` already consumes — so downstream users can drop `result.haplotypes_per_copy` straight into Phase 46/47.

**Architecture:** Three new pure helpers in `torchgenomics/preprocess/phase_polyorigin.py` (`_enumerate_state_table`, `_decode_haplotypes_per_copy`, `_validate_state_table`), two new `PhasingResult` fields (`state_table`, `haplotypes_per_copy`) materialized eagerly inside `run_polyorigin` and persisted as `.pt` artifacts. State table is built deterministically in Python from bivalent-gamete combinatorics and validated via a round-trip expected-dosage match against PolyOrigin's emitted `postdose_probs`.

**Tech Stack:** Python 3.10+, PyTorch. No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-04-24-haplotypegwas-shape-adapter-design.md`

---

## File Structure

**Modified files:**
- `torchgenomics/preprocess/phase_polyorigin.py` — add 3 helpers, extend `PhasingResult` (2 new fields), extend `run_polyorigin` orchestration, extend `_persist_result` (2 new atomic writes). ~150 LOC of new logic.
- `tests/test_phase_polyorigin.py` — append ~10 Tier 1 tests (6 pure-helper + 4 orchestration).
- `tests/test_phase_polyorigin_e2e.py` — append 1 Tier 2 end-to-end test.

**Unchanged files:**
- `torchgenomics/preprocess/_polyorigin_runtime.py`
- `torchgenomics/preprocess/juliapkg.json`
- `pyproject.toml`
- `torchgenomics/cli.py`, `torchgenomics/__main__.py`
- `torchgenomics/models/haplotype_gwas.py`
- `docs/getting-started/polyploid_phasing.md` (recipe stays the "inspect tensors" pattern until user decides to expand)

**New files:** none.

---

## Task 1 — Extend `PhasingResult` with `state_table` and `haplotypes_per_copy`

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py` (the `PhasingResult` dataclass, around line 32)
- Modify: `tests/test_phase_polyorigin.py` (existing `test_phasing_result_dataclass_fields` test)

- [ ] **Step 1: Update the existing `test_phasing_result_dataclass_fields` test to include the new fields**

Find the existing test (it's in `tests/test_phase_polyorigin.py`, already passing). Amend it so the `PhasingResult(...)` construction includes the two new fields:

```python
def test_phasing_result_dataclass_fields():
    r = PhasingResult(
        haplotypes=torch.zeros(3, 2, dtype=torch.int64),
        origin_probs=torch.zeros(3, 2, 36, dtype=torch.float64),
        parent_phased=torch.zeros(2, 2, 4, dtype=torch.int8),
        offspring_ids=["o1", "o2", "o3"],
        parent_ids=["p1", "p2"],
        variant_ids=["v1", "v2"],
        chrom=["1", "1"],
        pos_bp=torch.tensor([100, 200], dtype=torch.int64),
        pos_cm=torch.tensor([0.0001, 0.0002], dtype=torch.float64),
        per_individual_ploidy={"p1": 4, "p2": 4, "o1": 4, "o2": 4, "o3": 4},
        map_refined=False,
        valent_diag=pd.DataFrame({"marker": ["v1"], "valent": ["4x"]}),
        postdose_probs=torch.zeros(3, 2, 5, dtype=torch.float64),
        tool="polyorigin",
        tool_version="1.0.3",
        input_hash="abc",
        cmd="polyOrigin(...)",
        workdir=None,
        state_table=torch.zeros(36, 4, dtype=torch.int8),
        haplotypes_per_copy=torch.zeros(3, 4, 2, dtype=torch.int8),
    )
    assert r.haplotypes.shape == (3, 2)
    assert r.state_table.shape == (36, 4)
    assert r.haplotypes_per_copy.shape == (3, 4, 2)
    assert r.tool == "polyorigin"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_phase_polyorigin.py::test_phasing_result_dataclass_fields -v
```

Expected: FAIL — `PhasingResult.__init__() got an unexpected keyword argument 'state_table'`.

- [ ] **Step 3: Add the two new fields to `PhasingResult`**

Open `torchgenomics/preprocess/phase_polyorigin.py`. Find the `@dataclass class PhasingResult:` block. Append these two fields AFTER the existing `workdir: str | None` field (making them the last two fields):

```python
    state_table: Tensor              # (n_states, ploidy) int8 — joint-origin state → per-copy parental source; v = parent_id*ploidy + copy_in_parent
    haplotypes_per_copy: Tensor      # (n_off, ploidy, m) int8 — decoded per-copy parental alleles, drop-in for HaplotypeGWAS.scan
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_phase_polyorigin.py -v
```

Expected: all pre-existing tests still pass; `test_phasing_result_dataclass_fields` now passes with the two new assertions.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: extend PhasingResult with state_table + haplotypes_per_copy"
```

---

## Task 2 — Implement `_enumerate_state_table(ploidy)`

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write 3 failing tests**

Append to `tests/test_phase_polyorigin.py`. Add `_enumerate_state_table` to the existing `from torchgenomics.preprocess.phase_polyorigin import (...)` block.

```python
import math


def test_enumerate_state_table_count():
    # ploidy=2 → C(2,1)² = 4 states
    st2 = _enumerate_state_table(2)
    assert st2.shape == (4, 2)
    assert st2.dtype == torch.int8
    # ploidy=4 → C(4,2)² = 36 states
    st4 = _enumerate_state_table(4)
    assert st4.shape == (36, 4)
    assert st4.dtype == torch.int8
    # ploidy=6 → C(6,3)² = 400 states
    st6 = _enumerate_state_table(6)
    assert st6.shape == (400, 6)
    assert st6.dtype == torch.int8


def test_enumerate_state_table_first_state_canonical():
    # For ploidy=4, state 0 = (parent1 gamete (0,1), parent2 gamete (0,1))
    # Encoding: v = parent_id * ploidy + copy_in_parent
    # So state 0 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] = [0, 1, 4, 5]
    st4 = _enumerate_state_table(4)
    assert st4[0].tolist() == [0, 1, 4, 5]


def test_enumerate_state_table_lex_ordering():
    # For ploidy=4, each parent's gametes are 2-subsets of {0,1,2,3} in lex
    # order: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    # State s = p1_gamete_idx * 6 + p2_gamete_idx.
    # State 1 = (p1 gamete 0 = (0,1), p2 gamete 1 = (0,2))
    #         → [0*4+0, 0*4+1, 1*4+0, 1*4+2] = [0, 1, 4, 6]
    # State 6 = (p1 gamete 1 = (0,2), p2 gamete 0 = (0,1))
    #         → [0*4+0, 0*4+2, 1*4+0, 1*4+1] = [0, 2, 4, 5]
    # State 35 = (p1 gamete 5 = (2,3), p2 gamete 5 = (2,3))
    #         → [0*4+2, 0*4+3, 1*4+2, 1*4+3] = [2, 3, 6, 7]
    st4 = _enumerate_state_table(4)
    assert st4[1].tolist() == [0, 1, 4, 6]
    assert st4[6].tolist() == [0, 2, 4, 5]
    assert st4[35].tolist() == [2, 3, 6, 7]
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k enumerate_state_table
```

Expected: 3 errors — `ImportError: cannot import name '_enumerate_state_table'`.

- [ ] **Step 3: Implement `_enumerate_state_table`**

Open `torchgenomics/preprocess/phase_polyorigin.py`. Add `from itertools import combinations` to the stdlib imports if not already present. Append this function (placement: after the existing pure helpers like `_load_map_tsv`, before `_validate_inputs`):

```python
def _enumerate_state_table(ploidy: int) -> Tensor:
    """Build the joint-origin state table for a 2-parent F1 at given ploidy.

    Bivalent meiosis only: each parent contributes ploidy/2 copies per
    gamete. Gametes are sorted (ploidy/2)-subsets of ``{0..ploidy-1}`` in
    lexicographic order. States are (parent1 gamete, parent2 gamete)
    Cartesian-product, flat-indexed as
    ``s = p1_gamete_idx * n_gametes + p2_gamete_idx``.

    Parameters
    ----------
    ploidy : int
        Must be even and in ``{2, 4, 6}``.

    Returns
    -------
    Tensor, shape (n_states, ploidy), int8
        Each row holds ``ploidy`` values ``v`` in ``[0, 2*ploidy)``:
        ``v = parent_id * ploidy + copy_in_parent``. Parent1 copies
        occupy the first ``ploidy/2`` slots, parent2 copies the last
        ``ploidy/2`` slots.
    """
    if ploidy not in _VALID_PLOIDIES:
        raise ValueError(
            f"_enumerate_state_table: ploidy must be in {{2, 4, 6}}; got {ploidy}."
        )
    half = ploidy // 2
    gametes = list(combinations(range(ploidy), half))
    n_gametes = len(gametes)
    n_states = n_gametes * n_gametes

    rows: list[list[int]] = []
    for p1_gamete in gametes:
        for p2_gamete in gametes:
            row = [0 * ploidy + c for c in p1_gamete] + [1 * ploidy + c for c in p2_gamete]
            rows.append(row)

    return torch.tensor(rows, dtype=torch.int8)
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k enumerate_state_table
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: _enumerate_state_table pure helper"
```

---

## Task 3 — Implement `_decode_haplotypes_per_copy`

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase_polyorigin.py` (add `_decode_haplotypes_per_copy` to the existing import block):

```python
def test_decode_known_inputs():
    # Setup: ploidy=4, 2 parents, 1 offspring, 1 marker.
    # parent_phased shape (2 parents, 1 marker, 4 copies):
    #   parent1 alleles = [1, 0, 1, 0]
    #   parent2 alleles = [0, 1, 0, 1]
    parent_phased = torch.tensor([
        [[1, 0, 1, 0]],  # parent1 at marker 0
        [[0, 1, 0, 1]],  # parent2 at marker 0
    ], dtype=torch.int8)

    # haplotypes shape (1 offspring, 1 marker): offspring inherits state 0
    haplotypes = torch.tensor([[0]], dtype=torch.int64)

    # State table from our enumeration — state 0 row = [0, 1, 4, 5]
    # Decoded: parent1[0]=1, parent1[1]=0, parent2[0]=0, parent2[1]=1
    state_table = _enumerate_state_table(4)

    out = _decode_haplotypes_per_copy(
        haplotypes=haplotypes,
        parent_phased=parent_phased,
        state_table=state_table,
        ploidy=4,
    )
    # Expected shape (1 offspring, 4 copies, 1 marker)
    assert out.shape == (1, 4, 1)
    assert out.dtype == torch.int8
    # Copy 0 = parent1 copy 0 = 1
    # Copy 1 = parent1 copy 1 = 0
    # Copy 2 = parent2 copy 0 = 0
    # Copy 3 = parent2 copy 1 = 1
    assert out[0, :, 0].tolist() == [1, 0, 0, 1]
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_phase_polyorigin.py::test_decode_known_inputs -v
```

Expected: FAIL — `ImportError: cannot import name '_decode_haplotypes_per_copy'`.

- [ ] **Step 3: Implement `_decode_haplotypes_per_copy`**

Append to `torchgenomics/preprocess/phase_polyorigin.py` after `_enumerate_state_table`:

```python
def _decode_haplotypes_per_copy(
    haplotypes: Tensor,
    parent_phased: Tensor,
    state_table: Tensor,
    ploidy: int,
) -> Tensor:
    """Expand joint-origin state indices into per-chromosome-copy alleles.

    For each (offspring i, marker j) with state s = ``haplotypes[i, j]``:
    for each copy slot k in ``[0, ploidy)``::

        v = state_table[s, k]
        parent_id, copy_in_parent = divmod(int(v), ploidy)
        output[i, k, j] = parent_phased[parent_id, j, copy_in_parent]

    Vectorized via torch.gather.

    Parameters
    ----------
    haplotypes : Tensor, shape (n_off, m), int64
        Joint-origin state index per offspring per marker.
    parent_phased : Tensor, shape (n_parents, m, max_ploidy), int8
        Per-copy parental alleles.
    state_table : Tensor, shape (n_states, ploidy), int8
        As produced by ``_enumerate_state_table``.
    ploidy : int
        Must match ``state_table.shape[1]``.

    Returns
    -------
    Tensor, shape (n_off, ploidy, m), int8
        Per-copy parental alleles. Drop-in for ``HaplotypeGWAS.scan(haplotypes=...)``.
    """
    n_off, m = haplotypes.shape

    # Look up state table rows for each offspring-marker cell.
    # state_rows shape: (n_off, m, ploidy), values v = parent_id*ploidy + copy_in_parent
    state_rows = state_table[haplotypes]  # advanced indexing

    # Decode (parent_id, copy_in_parent) per cell
    parent_id = (state_rows.to(torch.int64) // ploidy)       # (n_off, m, ploidy)
    copy_in_parent = (state_rows.to(torch.int64) % ploidy)   # (n_off, m, ploidy)

    # Gather alleles from parent_phased[parent_id, marker_j, copy_in_parent].
    # parent_phased shape: (n_parents, m, max_ploidy)
    # Broadcast marker index j:
    m_idx = torch.arange(m, dtype=torch.int64, device=haplotypes.device)
    m_idx_b = m_idx.view(1, m, 1).expand(n_off, m, ploidy)  # (n_off, m, ploidy)

    alleles = parent_phased[parent_id, m_idx_b, copy_in_parent]  # (n_off, m, ploidy) int8

    # Reorder to (n_off, ploidy, m)
    return alleles.permute(0, 2, 1).contiguous().to(torch.int8)
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_phase_polyorigin.py::test_decode_known_inputs -v
```

Expected: PASS.

- [ ] **Step 5: Ruff check**

```bash
ruff check torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
```

Expected: `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: _decode_haplotypes_per_copy vectorized decoder"
```

---

## Task 4 — Implement `_validate_state_table`

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write 2 failing tests**

Append to `tests/test_phase_polyorigin.py` (add `_validate_state_table` to the import block):

```python
def _self_consistent_validate_inputs():
    """Build (state_table, origin_probs, parent_phased, postdose_probs) that
    are self-consistent under ploidy=4."""
    ploidy = 4
    state_table = _enumerate_state_table(ploidy)

    parent_phased = torch.tensor([
        [[1, 0, 1, 0]],  # parent1 at marker 0
        [[0, 1, 0, 1]],  # parent2 at marker 0
    ], dtype=torch.int8)  # (2, 1, 4)

    # All probability mass on state 0 → dose 2 (per the decoded alleles 1,0,0,1)
    n_states = state_table.shape[0]
    origin_probs = torch.zeros(1, 1, n_states, dtype=torch.float64)
    origin_probs[0, 0, 0] = 1.0

    # Consistent postdose_probs: dose 2 with probability 1
    postdose_probs = torch.zeros(1, 1, ploidy + 1, dtype=torch.float64)
    postdose_probs[0, 0, 2] = 1.0

    return state_table, origin_probs, parent_phased, postdose_probs, ploidy


def test_validate_passes_on_self_consistent_input():
    state_table, origin_probs, parent_phased, postdose_probs, ploidy = \
        _self_consistent_validate_inputs()
    # Should return None (pass silently)
    _validate_state_table(
        state_table, origin_probs, parent_phased, postdose_probs,
        ploidy=ploidy,
    )


def test_validate_fails_on_reorder():
    # Use parent_phased = [[1,0,1,0], [0,1,0,1]], ploidy=4.
    # State 0 has decoded alleles [1,0,0,1] → dose 2.
    # State 1 = (p1 gamete (0,1), p2 gamete (0,2)) → copies [0, 1, 4, 6] →
    #   parent1[0]=1, parent1[1]=0, parent2[0]=0, parent2[2]=0 → dose 1.
    # Swapping rows 0 and 1 changes per-state dose (2 ↔ 1), so the
    # round-trip expected-dosage check must fail.
    state_table, origin_probs, parent_phased, postdose_probs, ploidy = \
        _self_consistent_validate_inputs()

    swapped = state_table.clone()
    swapped[[0, 1]] = swapped[[1, 0]]

    with pytest.raises(RuntimeError, match="State-table round-trip mismatch"):
        _validate_state_table(
            swapped, origin_probs, parent_phased, postdose_probs,
            ploidy=ploidy,
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k validate_state_table
```

Expected: 2 errors — `ImportError: cannot import name '_validate_state_table'`.

Wait — the two tests are named `test_validate_passes_*` and `test_validate_fails_*`. Use `-k "validate_passes or validate_fails"` instead:

```bash
pytest tests/test_phase_polyorigin.py -v -k "validate_passes or validate_fails"
```

- [ ] **Step 3: Implement `_validate_state_table`**

Append to `torchgenomics/preprocess/phase_polyorigin.py` after `_decode_haplotypes_per_copy`:

```python
def _validate_state_table(
    state_table: Tensor,
    origin_probs: Tensor,
    parent_phased: Tensor,
    postdose_probs: Tensor,
    ploidy: int,
    atol: float = 1e-3,
    tool_version: str = "unknown",
) -> None:
    """Round-trip check: state_table-derived expected dosage must match
    PolyOrigin's emitted ``postdose_probs`` expected dosage.

    For each state s and marker j, compute dose_at(j, s) = sum over the
    copies listed in ``state_table[s]`` of the corresponding
    ``parent_phased`` allele. Then::

        E_state[i, j] = sum_s origin_probs[i, j, s] * dose_at(j, s)
        E_post[i, j]  = sum_d d * postdose_probs[i, j, d]

    Both should be equal (within ``atol``) iff our state enumeration's
    per-state copy-SETS match PolyOrigin's. The check catches
    dose-changing reorderings but is blind to within-parent gamete copy
    permutations (which are downstream-equivalent; see spec Section 2.3).

    Raises
    ------
    RuntimeError
        If max |E_state - E_post| > ``atol``. Diagnostic names the
        offending (offspring_idx, marker_idx) cell plus both computed
        values.
    """
    n_off, m, n_states = origin_probs.shape
    n_parents = parent_phased.shape[0]

    # Compute per-state per-marker dose: dose_per_state[j, s] = sum over
    # copies in state_table[s] of parent_phased allele at (parent, j, copy_in_parent).
    # state_table shape (n_states, ploidy)
    st64 = state_table.to(torch.int64)
    parent_id = st64 // ploidy           # (n_states, ploidy)
    copy_in_parent = st64 % ploidy       # (n_states, ploidy)

    # Gather alleles: for each (state_s, slot_k, marker_j), look up
    # parent_phased[parent_id[s,k], j, copy_in_parent[s,k]].
    # Broadcast to (n_states, ploidy, m):
    p_id_b = parent_id.unsqueeze(-1).expand(n_states, ploidy, m)              # (n_states, ploidy, m)
    c_in_p_b = copy_in_parent.unsqueeze(-1).expand(n_states, ploidy, m)       # (n_states, ploidy, m)
    m_idx = torch.arange(m, dtype=torch.int64, device=state_table.device)
    m_idx_b = m_idx.view(1, 1, m).expand(n_states, ploidy, m)                 # (n_states, ploidy, m)
    alleles = parent_phased[p_id_b, m_idx_b, c_in_p_b].to(torch.float64)      # (n_states, ploidy, m)
    dose_per_state = alleles.sum(dim=1)                                       # (n_states, m)

    # E_state[i, j] = sum_s origin_probs[i, j, s] * dose_per_state[j, s]
    # Reshape to align axes for einsum.
    # origin_probs: (n_off, m, n_states)  →  (n_off, m, n_states)
    # dose_per_state: (n_states, m)        →  (m, n_states) for the sum
    e_state = torch.einsum("ijs,sj->ij", origin_probs, dose_per_state.to(torch.float64))

    # E_post[i, j] = sum_d d * postdose_probs[i, j, d]
    kp1 = postdose_probs.shape[-1]
    d_vals = torch.arange(kp1, dtype=torch.float64, device=postdose_probs.device)
    e_post = (postdose_probs * d_vals).sum(dim=-1)

    # Compare
    abs_dev = (e_state - e_post).abs()
    max_dev = float(abs_dev.max())
    if max_dev > atol:
        # Locate worst cell
        flat_idx = int(abs_dev.argmax())
        off_idx, mkr_idx = divmod(flat_idx, m)
        raise RuntimeError(
            f"State-table round-trip mismatch — our enumeration disagrees "
            f"with PolyOrigin v{tool_version}. Max deviation {max_dev:.4f} at "
            f"(offspring_idx={off_idx}, marker_idx={mkr_idx}); "
            f"ours={float(e_state[off_idx, mkr_idx]):.4f}, "
            f"PolyOrigin={float(e_post[off_idx, mkr_idx]):.4f}. "
            f"Likely cause: PolyOrigin reordered states between releases. "
            f"Open an issue with the offending phase-poly inputs."
        )
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k "validate_passes or validate_fails"
```

Expected: 2 passed.

- [ ] **Step 5: Ruff check**

```bash
ruff check torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
```

Expected: `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: _validate_state_table round-trip dosage check"
```

---

## Task 5 — Wire state-table + decode + validate into `run_polyorigin`

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py` (inside `run_polyorigin`)
- Modify: `tests/test_phase_polyorigin.py`

Order-of-operations: after the output-parse block (where `origin_probs`, `parent_phased`, `postdose_probs`, `haplotypes`, `per_individual_ploidy` are computed) and BEFORE `_persist_result` is called / the `PhasingResult` is constructed. Also add the two new fields to the `PhasingResult(...)` constructor call.

- [ ] **Step 1: Write 3 failing orchestration tests**

Append to `tests/test_phase_polyorigin.py`. These tests reuse the existing `_stub_runtime` helper that's already defined earlier in the file.

```python
def test_run_polyorigin_haplotypes_per_copy_field_populated(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )
    # 1 offspring, 4 copies, 2 markers
    assert result.haplotypes_per_copy.shape == (1, 4, 2)
    assert result.haplotypes_per_copy.dtype == torch.int8
    assert set(result.haplotypes_per_copy.unique().tolist()) <= {0, 1}


def test_run_polyorigin_state_table_field_populated(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )
    # ploidy=4 → 36 states, 4 copies per state
    assert result.state_table.shape == (36, 4)
    assert result.state_table.dtype == torch.int8
    # Values in [0, 2*ploidy) = [0, 8)
    vmin = int(result.state_table.min())
    vmax = int(result.state_table.max())
    assert vmin >= 0
    assert vmax < 8


def test_run_polyorigin_mixed_ploidy_rejected(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    # Two offspring with explicit mixed ploidies
    ped.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t4\n"
        "o2\tp1\tp2\t2\n"
    )
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # probs shape must include 2 offspring + 2 parents = 4 samples
    probs4 = torch.zeros(4, 2, 5, dtype=torch.float64)
    probs4[:, :, 2] = 1.0

    with pytest.raises(ValueError, match="uniform ploidy"):
        run_polyorigin(
            probs=probs4,
            pedigree_tsv=str(ped), map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1", "o2"], variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )

    # Mixed-ploidy rejection happens before persistence — no artifacts
    assert not (out_prefix.parent / "phased.haplotypes_per_copy.pt").exists()
    assert not (out_prefix.parent / "phased.meta.json").exists()
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k "haplotypes_per_copy_field or state_table_field or mixed_ploidy_rejected"
```

Expected: all 3 fail with `AttributeError: 'PhasingResult' object has no attribute 'haplotypes_per_copy'` (or similar — the fields aren't being populated by `run_polyorigin` yet).

- [ ] **Step 3: Extend `run_polyorigin`**

Open `torchgenomics/preprocess/phase_polyorigin.py`. Find `run_polyorigin(...)`. Locate the block AFTER output parsing (where `haplotypes`, `origin_probs`, `parent_phased`, `postdose_probs`, `per_individual_ploidy`, and `valent_diag` have all been computed) and BEFORE the `PhasingResult(...)` constructor call.

Insert this block:

```python
        # --- Tier A #1: same-ploidy guard + state-table + decoder ---
        ploidy_values = set(per_individual_ploidy.values())
        if len(ploidy_values) > 1:
            raise ValueError(
                f"haplotypes_per_copy decoding requires uniform ploidy across "
                f"all individuals; got {sorted(ploidy_values)}. Mixed-ploidy "
                f"F1 is deferred."
            )
        decode_ploidy = next(iter(ploidy_values))

        state_table = _enumerate_state_table(decode_ploidy)
        _validate_state_table(
            state_table, origin_probs, parent_phased, postdose_probs,
            ploidy=decode_ploidy, tool_version=tool_version,
        )
        haplotypes_per_copy = _decode_haplotypes_per_copy(
            haplotypes, parent_phased, state_table, ploidy=decode_ploidy,
        )
```

Then update the `PhasingResult(...)` constructor call to include the two new fields. The existing call has kwargs like `tool="polyorigin", tool_version=tool_version, ...`. Add:

```python
            state_table=state_table,
            haplotypes_per_copy=haplotypes_per_copy,
```

- [ ] **Step 4: Run the orchestration tests**

```bash
pytest tests/test_phase_polyorigin.py -v -k "haplotypes_per_copy_field or state_table_field or mixed_ploidy_rejected"
```

Expected: 3 passed.

- [ ] **Step 5: Run the full Tier 1 suite to catch regressions**

```bash
pytest tests/test_phase_polyorigin.py -v
```

Expected: all previously-passing tests still pass. Total count should be 43 (pre-adapter) + 3 (Task 2) + 1 (Task 3) + 2 (Task 4) + 3 (Task 5) = 52.

- [ ] **Step 6: Ruff check**

```bash
ruff check torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
```

Expected: `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: wire state_table + decoder into run_polyorigin"
```

---

## Task 6 — Persist the two new artifacts

**Files:**
- Modify: `torchgenomics/preprocess/phase_polyorigin.py` (inside `_persist_result`)
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_phase_polyorigin.py`:

```python
def test_persist_per_copy_artifacts(tmp_path, monkeypatch):
    from torchgenomics.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped), map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"], variant_ids=["v1", "v2"],
        auto_install_julia=False,
    )

    st_path = Path(f"{out_prefix}.state_table.pt")
    hpc_path = Path(f"{out_prefix}.haplotypes_per_copy.pt")
    assert st_path.is_file()
    assert hpc_path.is_file()

    st_loaded = torch.load(st_path)
    hpc_loaded = torch.load(hpc_path)
    assert torch.equal(st_loaded, result.state_table)
    assert torch.equal(hpc_loaded, result.haplotypes_per_copy)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_phase_polyorigin.py::test_persist_per_copy_artifacts -v
```

Expected: FAIL — `AssertionError: assert st_path.is_file()` (artifacts not persisted yet).

- [ ] **Step 3: Extend `_persist_result`**

Open `torchgenomics/preprocess/phase_polyorigin.py`. Find `_persist_result`. The existing body has a sequence of `_persist("...", lambda p: ...)` calls. Add two more, placed right after the `postdose_probs.pt` write (to keep per-result tensors grouped):

```python
        _persist("state_table.pt",         lambda p: torch.save(result.state_table, p))
        _persist("haplotypes_per_copy.pt", lambda p: torch.save(result.haplotypes_per_copy, p))
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_phase_polyorigin.py::test_persist_per_copy_artifacts -v
```

Expected: PASS.

- [ ] **Step 5: Run the full Tier 1 suite**

```bash
pytest tests/test_phase_polyorigin.py -v
```

Expected: 53 passed (52 + 1 new).

- [ ] **Step 6: Ruff check**

```bash
ruff check torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
```

Expected: `All checks passed!`.

- [ ] **Step 7: Commit**

```bash
git add torchgenomics/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Tier A #1: persist state_table + haplotypes_per_copy artifacts"
```

---

## Task 7 — Tier 2 end-to-end HaplotypeGWAS integration

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py`

This test is `skipif`-gated on real Julia + PolyOrigin.jl availability (module-level pytestmark already exists). Add one test at the bottom of the file.

- [ ] **Step 1: Write the test**

Append to `tests/test_phase_polyorigin_e2e.py`:

```python
def test_haplotypegwas_scan_consumes_per_copy(tmp_path):
    """Tier 2 smoke test: Phase 56 → HaplotypeGWAS end-to-end.

    Runs the full chain dosage-call (simulated) → run_polyorigin (real
    Julia) → HaplotypeGWAS.scan() with haplotypes=result.haplotypes_per_copy.
    Asserts that scan returns a valid result — not a calibrated power
    gate, just verifies the per-copy tensor is shape-compatible with
    Phase 46's scanner.
    """
    import numpy as np
    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin
    from torchgenomics.preprocess.dosage_uncertainty import expected_dosage
    from torchgenomics.models import HaplotypeGWAS

    rng = np.random.default_rng(42)
    n_off, m, ploidy = 6, 10, 4
    probs, sample_ids, variant_ids = _make_toy_tetraploid_f1_probs(rng, n_off, m)

    ped = tmp_path / "ped.tsv"
    ped.write_text(
        "offspring\tparent1\tparent2\n"
        + "\n".join(f"o{k+1}\tp1\tp2" for k in range(n_off))
        + "\n"
    )
    mp = tmp_path / "map.tsv"
    mp.write_text(
        "marker\tchrom\tpos_bp\n"
        + "\n".join(f"v{j+1}\t1\t{(j+1)*1000}" for j in range(m))
        + "\n"
    )
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    auto = bool(os.environ.get("TORCHGENOMICS_ALLOW_AUTO_INSTALL"))
    result = run_polyorigin(
        probs=probs,
        pedigree_tsv=str(ped),
        map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=ploidy,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        auto_install_julia=auto,
    )

    # haplotypes_per_copy must be shape (n_off, ploidy, m)
    assert result.haplotypes_per_copy.shape == (n_off, ploidy, m)

    # Build G from the offspring postdose_probs (refined-map order)
    G = expected_dosage(result.postdose_probs, ploidy=ploidy)  # (n_off, m)

    # Synthetic phenotype: mean dosage per offspring + small noise
    rng_t = torch.Generator()
    rng_t.manual_seed(42)
    Y = G.mean(dim=1) + 0.1 * torch.randn(n_off, generator=rng_t, dtype=torch.float64)

    scanner = HaplotypeGWAS(method="block", test="f_test", ploidy=ploidy)
    scan_result = scanner.scan(
        Y=Y,
        G=G.to(torch.float64),
        haplotypes=result.haplotypes_per_copy,
    )

    # Sanity checks on the scan output
    assert len(scan_result.start) > 0, "expected at least one block"
    # All global p-values are valid floats in [0, 1]
    for pv in scan_result.p_global.tolist():
        assert isinstance(pv, float)
        assert not math.isnan(pv)
        assert 0.0 <= pv <= 1.0
```

Note: `math` may not already be imported at the top of the file. Check and add `import math` alongside the existing stdlib imports.

- [ ] **Step 2: Run the test locally (requires Julia + PolyOrigin.jl on this machine)**

```bash
pytest tests/test_phase_polyorigin_e2e.py::test_haplotypegwas_scan_consumes_per_copy -v
```

Expected on a Julia-equipped machine: PASS. On a CI job with no Julia: SKIPPED via the module-level pytestmark.

- [ ] **Step 3: Run the full suite to confirm no regressions**

```bash
pytest tests/test_phase_polyorigin.py tests/test_phase_polyorigin_e2e.py -v --deselect tests/test_phase_polyorigin_e2e.py::test_parity_tetraploid_f1
```

Expected: 53 Tier 1 passed + 3 Tier 2 skipped (parity deselected; the other 3 scaffold-skip) + the new test either passes or skips.

- [ ] **Step 4: Ruff check**

```bash
ruff check tests/test_phase_polyorigin_e2e.py
```

Expected: `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Tier A #1: Tier 2 HaplotypeGWAS end-to-end integration test"
```

---

## Task 8 — Close-out: update backlog + per-phase memory

**Files:**
- Modify: `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\project_post_phase56_backlog.md`
- Modify: `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\project_polyploid_phasing.md`
- Modify: `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\MEMORY.md`

These memory files live outside the repo (user's Claude config dir) and are NOT committed. This task is documentation-only.

- [ ] **Step 1: Mark Tier A #1 closed in the backlog**

Open `project_post_phase56_backlog.md`. Edit the "Tier A #1" bullet (the "HaplotypeGWAS shape adapter" entry) to:
- Strike through the original text, OR
- Replace it with a "CLOSED (2026-04-24, commits <range>)" marker.

Suggested replacement:

```markdown
1. **~~HaplotypeGWAS shape adapter~~** — **CLOSED 2026-04-24**. Shipped as commits from spec `2026-04-24-haplotypegwas-shape-adapter-design.md`. `PhasingResult.haplotypes_per_copy` now available as drop-in for `HaplotypeGWAS.scan()`.
```

Update the "Recommended next action" footer to point at **Tier A #2** (calibration fill) or **Tier A #3** (GWASpoly potato URL pin).

- [ ] **Step 2: Update the per-phase memory**

Open `project_polyploid_phasing.md`. Append a new section under "**Gotchas to flag to future sessions**":

```markdown
- Per-copy haplotype adapter shipped 2026-04-24. `PhasingResult` now carries
  `state_table: (n_states, ploidy) int8` and `haplotypes_per_copy: (n_off, ploidy, m) int8`
  (both persisted). Decoder is pure-Python with a Julia round-trip validation
  at `run_polyorigin` time. Same-ploidy F1 only — mixed-ploidy raises
  `ValueError`. Phase coherence across markers is best-effort (marginal
  argmax, not Viterbi) — see spec Section 7 for details.
```

- [ ] **Step 3: Update `MEMORY.md`**

Open `MEMORY.md`. Find the line pointing to `project_post_phase56_backlog.md`. Update its description to reflect that Tier A #1 is closed:

```markdown
- [Post-Phase-56 Action Backlog](project_post_phase56_backlog.md) — Tier A #1 CLOSED (shape adapter shipped). Remaining: Tier A (calibration fill, potato URL pin), Tier B (CI billing + push), Tier C (Phases 50–54 candidates), Tier D (maintenance).
```

- [ ] **Step 4: No git commit needed**

Memory files live under `~/.claude/projects/`. They are NOT part of the repo and NOT committed. Just save the edits.

---

## Self-review notes

**Spec coverage check**:

- **§1 Motivation / scope boundary** — Tasks 1–7 deliver everything in "In scope"; Task 8 closes out "Section 5.4 #3" (memory update). ✓
- **§2.1 `PhasingResult` new fields** — Task 1. ✓
- **§2.2 `_enumerate_state_table`** — Task 2. ✓
- **§2.3 `_validate_state_table`** — Task 4. ✓
- **§2.4 `_decode_haplotypes_per_copy`** — Task 3. ✓
- **§2.5 orchestration extension** — Task 5. ✓
- **§2.6 persistence** — Task 6. ✓
- **§3 data flow** — Tasks 5 + 6 implement the new data flow nodes. ✓
- **§4.1 mixed-ploidy guard** — Task 5's `test_run_polyorigin_mixed_ploidy_rejected`. ✓
- **§4.2 state-table round-trip RuntimeError** — Task 4's `test_validate_fails_on_reorder`. ✓
- **§4.3 decoder shape sanity** — NOT explicitly tested by the plan. The decoder output shape is implicitly checked by `test_run_polyorigin_haplotypes_per_copy_field_populated`. The spec's §4.3 is a defensive check IN the decoder; the tests verify correct shape output, which catches shape errors indirectly. Acceptable — no standalone test of the RuntimeError needed at plan level.
- **§5.1 pure-helper tests** — Tasks 2, 3, 4. ✓ (6 tests)
- **§5.2 orchestration tests** — Task 5 (3 tests) + Task 6 (1 test) = 4. ✓
- **§5.3 Tier 2 end-to-end** — Task 7. ✓
- **§5.4 the gate** — Tasks 5, 6, 7 cover points 1–2; Task 8 covers point 3. Point 4 ("no new deps") is invariant-by-construction. ✓
- **§6 non-goals** — all honored (no CLI change, no juliacall change, no pyproject.toml change). ✓
- **§7 known limitations** — documentation-only; no task needed. ✓
- **§8 open questions** — non-blocking; no tasks needed. ✓

**Placeholder scan**: no TBD / TODO / FIXME / "implement later" / vague "handle edge cases" anywhere. Every code step shows actual code.

**Type consistency**: `_enumerate_state_table(ploidy) -> Tensor`, `_decode_haplotypes_per_copy(haplotypes, parent_phased, state_table, ploidy) -> Tensor`, `_validate_state_table(state_table, origin_probs, parent_phased, postdose_probs, ploidy, atol, tool_version) -> None` — signatures match across Tasks 2–6. `PhasingResult` field names (`state_table`, `haplotypes_per_copy`) are consistent across Tasks 1, 5, 6, 7, 8.

One clarification: Task 5 introduces the `decode_ploidy` local variable (extracted from `per_individual_ploidy`). This is intentional — the function's `ploidy=` parameter to `_enumerate_state_table` and `_decode_haplotypes_per_copy` comes from the pedigree's actual values, not the `run_polyorigin` caller's `ploidy` argument (which is the *default* when the pedigree TSV lacks a per-row ploidy column). The same-ploidy guard ensures they're equal in practice, but decoupling them keeps the code honest about its inputs.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-24-haplotypegwas-shape-adapter.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
