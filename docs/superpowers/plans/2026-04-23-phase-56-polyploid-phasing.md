# Phase 56 — Polyploid Phasing (PolyOrigin Wrapper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a thin Python wrapper around the Julia package `PolyOrigin.jl` for phasing connected tetraploid / hexaploid F1 polyploid populations, chaining off Phase 55's `.probs.pt` output, plus a `torchgwas phase-poly` CLI subcommand, without touching any downstream consumer.

**Architecture:** Two new modules: `torchgwas/preprocess/phase_polyorigin.py` (public surface — `run_polyorigin`, `PhasingResult`, pure converters) and `torchgwas/preprocess/_polyorigin_runtime.py` (discover-first Julia bootstrap via `juliacall`). Julia + PolyOrigin.jl are provisioned on demand on first `run_polyorigin` call: search PATH + known install paths first, fall back to juliacall's managed install only if the user consents. Julia runs in-process via `juliacall`, not a subprocess. Output is tensor-native (not VCF) — feeds `HaplotypeGWAS` directly. Tier 1 tests monkeypatch the runtime module and never touch Julia. Tier 2 tests `pytest.mark.skipif` on missing Julia / PolyOrigin.jl.

**Tech Stack:** Python 3.10+, PyTorch, pandas, tempfile. External: Julia ≥ 1.10 + `PolyOrigin.jl` v1.0.3 (Tier 2 only, auto-provisioned via `juliacall`).

**Spec:** `docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md`

---

## File Structure

**New files:**
- `torchgwas/preprocess/phase_polyorigin.py` — wrapper + pure converters (~500 LOC).
- `torchgwas/preprocess/_polyorigin_runtime.py` — discover-first Julia bootstrap (~150 LOC).
- `torchgwas/preprocess/juliapkg.json` — shipped dep manifest.
- `tests/test_phase_polyorigin.py` — Tier 1, always-on; runtime stubbed.
- `tests/test_phase_polyorigin_e2e.py` — Tier 2, skipif-gated; runs real PolyOrigin.
- `tests/fixtures/phase_polyorigin/` — canned PolyOrigin-output CSVs, toy pedigree + map.
- `bench/calibrate_polyorigin_recovery.py` — dev-time calibration helper.
- `docs/getting-started/polyploid_phasing.md` — user recipe.

**Modified files:**
- `torchgwas/preprocess/__init__.py` — re-export `run_polyorigin`, `PhasingResult`.
- `torchgwas/cli.py` — new `phase-poly` subcommand + dispatch entry.
- `pyproject.toml` — (a) `[project.optional-dependencies]` gains `polyploid-phase = ["juliacall>=0.9"]`; (b) `[tool.setuptools.package-data]` gains `"torchgwas.preprocess" = ["juliapkg.json"]`.
- `docs/ROADMAP.md` — remove Phase 56 entry on ship.
- `docs/cli.md` — append `phase-poly --help` capture.
- `CLAUDE.md` — bump subcommand count +1; add `phase-poly` example.

**Unchanged:**
- `torchgwas/preprocess/dosage_call.py`, `torchgwas/preprocess/dosage_uncertainty.py`, `torchgwas/preprocess/phase.py`, `torchgwas/models/haplotype_gwas.py` — already correct upstream/downstream code.

---

## Task 1 — Dependencies + project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `torchgwas/preprocess/juliapkg.json`
- Create: `torchgwas/preprocess/phase_polyorigin.py` (module skeleton)
- Create: `torchgwas/preprocess/_polyorigin_runtime.py` (module skeleton)
- Create: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Add `polyploid-phase` extra + juliapkg.json package-data to pyproject.toml**

In `pyproject.toml`, extend `[project.optional-dependencies]` (around line 35):

```toml
[project.optional-dependencies]
zarr = ["zarr>=2.14"]
hdf5 = ["h5py>=3.8"]
parquet = ["pyarrow>=12.0"]
plot = ["seaborn>=0.12"]
polyploid-phase = ["juliacall>=0.9"]
all = [
    "zarr>=2.14",
    "h5py>=3.8",
    "pyarrow>=12.0",
    "seaborn>=0.12",
]
```

And extend `[tool.setuptools.package-data]` (around line 77):

```toml
[tool.setuptools.package-data]
"torchgwas._native" = ["*.pyi"]
"torchgwas.preprocess" = ["juliapkg.json"]
```

- [ ] **Step 2: Create `torchgwas/preprocess/juliapkg.json` with pinned PolyOrigin.jl**

```json
{
  "julia": "1.10",
  "packages": {
    "PolyOrigin": {
      "uuid": "IMPLEMENTER-VERIFY-FROM-UPSTREAM-PROJECT-TOML",
      "url": "https://github.com/chaozhi/PolyOrigin.jl",
      "rev": "v1.0.3"
    }
  }
}
```

Note: the UUID must be copied verbatim from `https://github.com/chaozhi/PolyOrigin.jl/blob/v1.0.3/Project.toml` during implementation. Replace the placeholder string before committing.

- [ ] **Step 3: Create `torchgwas/preprocess/phase_polyorigin.py` skeleton**

```python
"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20
```

- [ ] **Step 4: Create `torchgwas/preprocess/_polyorigin_runtime.py` skeleton**

```python
"""Discover-first Julia bootstrap for PolyOrigin.jl (Phase 56 internal).

Not public API. ``run_polyorigin`` uses ``get_runtime`` to obtain a
``juliacall`` handle. Search order: explicit kwarg → TORCHGWAS_JULIA env
var → known install paths → juliacall's own managed install (only if the
caller consents). See the Phase 56 design spec Section 2.3 for rationale.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jl: Any = None
_polyorigin: Any = None
_version: str | None = None
```

- [ ] **Step 5: Create `tests/test_phase_polyorigin.py` with one smoke test**

```python
"""Phase 56: polyploid phasing via PolyOrigin (Tier 1, always-on)."""
from __future__ import annotations

import torchgwas.preprocess.phase_polyorigin as mod


def test_module_imports_without_juliacall():
    # juliacall must NOT be imported at module load — only inside
    # get_runtime(). Verify the import surface is inert.
    assert hasattr(mod, "_VALID_PLOIDIES")
    assert mod._VALID_PLOIDIES == frozenset({2, 4, 6})
```

- [ ] **Step 6: Install the extra and run the test**

```bash
pip install -e ".[dev,polyploid-phase]"
pytest tests/test_phase_polyorigin.py -v
```

Expected: 1 passed. `juliacall` installed (~15 MB) but Julia is NOT downloaded (no `import juliacall` yet).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml \
        torchgwas/preprocess/juliapkg.json \
        torchgwas/preprocess/phase_polyorigin.py \
        torchgwas/preprocess/_polyorigin_runtime.py \
        tests/test_phase_polyorigin.py
git commit -m "Phase 56: scaffold phase_polyorigin module + juliapkg manifest + polyploid-phase extra"
```

---

## Task 2 — `PhasingResult` dataclass

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_phase_polyorigin.py`:

```python
import torch
import pandas as pd
from torchgwas.preprocess.phase_polyorigin import PhasingResult


def test_phasing_result_dataclass_fields():
    r = PhasingResult(
        haplotypes=torch.zeros(3, 4, 2, dtype=torch.int8),
        origin_probs=torch.zeros(3, 4, 2, 4, dtype=torch.float64),
        parent_phased=torch.zeros(2, 4, 2, dtype=torch.int8),
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
    )
    assert r.haplotypes.shape == (3, 4, 2)
    assert r.origin_probs.shape == (3, 4, 2, 4)
    assert r.tool == "polyorigin"
    assert r.map_refined is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_phase_polyorigin.py::test_phasing_result_dataclass_fields -v
```

Expected: FAIL with `ImportError: cannot import name 'PhasingResult'`.

- [ ] **Step 3: Implement `PhasingResult`**

Append to `torchgwas/preprocess/phase_polyorigin.py` after the constants block:

```python
@dataclass
class PhasingResult:
    haplotypes: Tensor                     # (n_off, ploidy, m) int8
    origin_probs: Tensor                   # (n_off, ploidy, m, n_parent_haps) float64
    parent_phased: Tensor                  # (n_parents, max_ploidy, m) int8
    offspring_ids: list[str]
    parent_ids: list[str]
    variant_ids: list[str]
    chrom: list[str]
    pos_bp: Tensor                         # (m,) int64
    pos_cm: Tensor                         # (m,) float64
    per_individual_ploidy: dict[str, int]
    map_refined: bool
    valent_diag: "pd.DataFrame"
    postdose_probs: Tensor                 # (n_off, m, max_ploidy+1) float64
    tool: str                              # "polyorigin"
    tool_version: str
    input_hash: str
    cmd: str
    workdir: str | None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_phase_polyorigin.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: add PhasingResult dataclass"
```

---

## Task 3 — Pedigree converter `_build_polyorigin_pedfile`

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

Pure helper. Converts a simple 3-col user TSV (`offspring, parent1, parent2`, optional `ploidy`) into PolyOrigin's native pedfile CSV with columns `individual, population, motherid, fatherid, ploidy`. Founders get `motherid=fatherid=0, population=0`. Offspring get integer `population` IDs grouped by unique `(parent1, parent2)` pair.

- [ ] **Step 1: Write failing tests for single biparental cross**

Append to `tests/test_phase_polyorigin.py`:

```python
import tempfile
from pathlib import Path

import pytest
from torchgwas.preprocess.phase_polyorigin import _build_polyorigin_pedfile


def _write_ped(tmp: Path, rows: list[str]) -> Path:
    p = tmp / "ped.tsv"
    p.write_text("offspring\tparent1\tparent2\n" + "\n".join(rows) + "\n")
    return p


def test_pedfile_single_biparental(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o2\tp1\tp2", "o3\tp1\tp2"])
    out = _build_polyorigin_pedfile(
        str(ped_in), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "o1", "o2", "o3"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert list(df.columns) == ["individual", "population", "motherid", "fatherid", "ploidy"]
    founders = df[df["population"] == 0]
    assert set(founders["individual"]) == {"p1", "p2"}
    assert (founders["motherid"] == 0).all()
    assert (founders["fatherid"] == 0).all()
    offspring = df[df["population"] != 0]
    assert set(offspring["individual"]) == {"o1", "o2", "o3"}
    assert (offspring["population"] == 1).all()
    assert (offspring["ploidy"] == 4).all()


def test_pedfile_multi_family_connected(tmp_path):
    # 3 parents (p1, p2, p3) → two sub-families: (p1,p2) and (p1,p3)
    ped_in = _write_ped(tmp_path, [
        "o1\tp1\tp2", "o2\tp1\tp2",
        "o3\tp1\tp3", "o4\tp1\tp3",
    ])
    out = _build_polyorigin_pedfile(
        str(ped_in), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "p3", "o1", "o2", "o3", "o4"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    founders = df[df["population"] == 0]
    assert set(founders["individual"]) == {"p1", "p2", "p3"}
    offspring = df[df["population"] != 0]
    # Two distinct non-zero populations assigned
    assert offspring["population"].nunique() == 2
    # Each sub-family has two offspring
    assert (offspring.groupby("population").size() == 2).all()


def test_pedfile_multi_generation_rejected(tmp_path):
    # o1 is a parent of o2 — multi-generation, not allowed
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o2\to1\tp2"])
    with pytest.raises(ValueError, match="multi-generation|F1.*only"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1", "o2"},
            workdir=str(tmp_path),
        )


def test_pedfile_duplicate_offspring_rejected(tmp_path):
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp2", "o1\tp1\tp2"])
    with pytest.raises(ValueError, match="duplicate"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1"},
            workdir=str(tmp_path),
        )


def test_pedfile_missing_parent_rejected(tmp_path):
    # p3 referenced but not present in the probs sample set
    ped_in = _write_ped(tmp_path, ["o1\tp1\tp3"])
    with pytest.raises(ValueError, match="p3"):
        _build_polyorigin_pedfile(
            str(ped_in), ploidy_default=4,
            sample_ids_in_probs={"p1", "p2", "o1"},
            workdir=str(tmp_path),
        )


def test_pedfile_per_row_ploidy_column(tmp_path):
    # Optional 4th column overrides ploidy_default
    p = tmp_path / "ped.tsv"
    p.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t6\n"
        "o2\tp1\tp2\t6\n"
    )
    out = _build_polyorigin_pedfile(
        str(p), ploidy_default=4,
        sample_ids_in_probs={"p1", "p2", "o1", "o2"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    # offspring ploidy from the column, founders default from ploidy_default
    assert df.loc[df["individual"] == "o1", "ploidy"].iloc[0] == 6
    assert df.loc[df["individual"] == "p1", "ploidy"].iloc[0] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k pedfile
```

Expected: FAIL with `ImportError: cannot import name '_build_polyorigin_pedfile'`.

- [ ] **Step 3: Implement `_build_polyorigin_pedfile`**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
def _build_polyorigin_pedfile(
    user_tsv: str,
    ploidy_default: int,
    sample_ids_in_probs: set[str],
    workdir: str,
) -> Path:
    """Convert user 3-col pedigree TSV to PolyOrigin's native pedfile.

    User TSV schema: ``offspring\tparent1\tparent2[\tploidy]``.
    Output CSV schema: ``individual,population,motherid,fatherid,ploidy``.

    Founders: ``motherid=fatherid=0, population=0``. Offspring: integer
    ``population`` grouped by unique ``(parent1, parent2)`` pair, starting
    from 1. Per-individual ploidy from optional TSV column, else the
    ``ploidy_default`` fallback.
    """
    df = pd.read_csv(user_tsv, sep="\t")
    required = {"offspring", "parent1", "parent2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"pedigree TSV missing required column(s): {sorted(missing)}. "
            "Expected header: offspring, parent1, parent2 (optional: ploidy)."
        )
    has_ploidy_col = "ploidy" in df.columns

    # Duplicate offspring check
    dups = df[df.duplicated("offspring", keep=False)]["offspring"].unique().tolist()
    if dups:
        raise ValueError(f"pedigree TSV has duplicate offspring IDs: {dups}")

    offspring_set = set(df["offspring"])
    parents_set = set(df["parent1"]) | set(df["parent2"])

    # Multi-generation check: any parent also appears as an offspring
    multigen = parents_set & offspring_set
    if multigen:
        raise ValueError(
            f"PolyOrigin models F1 + founder selfings only; multi-generation "
            f"pedigrees not supported. Offending parent(s) also present as "
            f"offspring: {sorted(multigen)}."
        )

    # Coverage: every pedigree individual must appear in probs sample set
    all_ped = offspring_set | parents_set
    missing_in_probs = all_ped - sample_ids_in_probs
    if missing_in_probs:
        missing_list = sorted(missing_in_probs)[:10]
        raise ValueError(
            f"pedigree references {len(missing_in_probs)} individual(s) missing "
            f"from the probs sample set (first 10: {missing_list})."
        )

    # Assign integer populations per unique (parent1, parent2) pair
    fam_keys = list(dict.fromkeys(zip(df["parent1"], df["parent2"])))
    fam_pop = {pair: i + 1 for i, pair in enumerate(fam_keys)}

    rows: list[dict] = []
    for parent_id in sorted(parents_set):
        rows.append({
            "individual": parent_id,
            "population": 0,
            "motherid": 0,
            "fatherid": 0,
            "ploidy": ploidy_default,
        })
    for _, r in df.iterrows():
        ploidy = int(r["ploidy"]) if has_ploidy_col and pd.notna(r.get("ploidy")) else ploidy_default
        rows.append({
            "individual": str(r["offspring"]),
            "population": fam_pop[(r["parent1"], r["parent2"])],
            "motherid": str(r["parent1"]),
            "fatherid": str(r["parent2"]),
            "ploidy": ploidy,
        })

    out = Path(workdir) / "pedfile.csv"
    pd.DataFrame(rows, columns=["individual", "population", "motherid", "fatherid", "ploidy"]).to_csv(out, index=False)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k pedfile
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: pure pedigree converter (_build_polyorigin_pedfile)"
```

---

## Task 4 — Map TSV loader with cM synthesis

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_phase_polyorigin.py`:

```python
from torchgwas.preprocess.phase_polyorigin import _load_map_tsv


def _write_map(tmp_path: Path, rows: list[str], header="marker\tchrom\tpos_bp") -> Path:
    p = tmp_path / "map.tsv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    return p


def test_load_map_with_cm_column_used_verbatim(tmp_path):
    m = _write_map(
        tmp_path,
        ["v1\t1\t1000000\t0.1", "v2\t1\t2000000\t0.2"],
        header="marker\tchrom\tpos_bp\tcm",
    )
    df = _load_map_tsv(str(m), recomrate=1.0)
    assert list(df.columns) == ["marker", "chrom", "pos_bp", "cm"]
    assert df["cm"].tolist() == [0.1, 0.2]


def test_load_map_without_cm_synthesized_with_warning(tmp_path, caplog):
    m = _write_map(tmp_path, ["v1\t1\t1000000", "v2\t1\t2000000"])
    with caplog.at_level("WARNING"):
        df = _load_map_tsv(str(m), recomrate=1.0)
    assert df["cm"].tolist() == [1.0, 2.0]
    assert any("synthesiz" in rec.message.lower() for rec in caplog.records)


def test_load_map_non_monotonic_bp_rejected(tmp_path):
    m = _write_map(tmp_path, ["v1\t1\t2000000", "v2\t1\t1000000"])
    with pytest.raises(ValueError, match="monotonic|v2"):
        _load_map_tsv(str(m), recomrate=1.0)


def test_load_map_missing_required_col_rejected(tmp_path):
    p = tmp_path / "map.tsv"
    p.write_text("marker\tpos_bp\nv1\t1000\n")  # no chrom
    with pytest.raises(ValueError, match="chrom"):
        _load_map_tsv(str(p), recomrate=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k load_map
```

Expected: FAIL, `_load_map_tsv` not importable.

- [ ] **Step 3: Implement `_load_map_tsv`**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
def _load_map_tsv(path: str, recomrate: float) -> pd.DataFrame:
    """Load marker map TSV. Synthesize cm from pos_bp if missing.

    Required columns: marker, chrom, pos_bp. Optional: cm.
    Missing cm → synthesize cm = pos_bp * recomrate / 1e6 with a warning.
    Non-monotonic pos_bp within a chromosome → ValueError.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"marker", "chrom", "pos_bp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"map TSV missing required column(s): {sorted(missing)}. "
            "Expected: marker, chrom, pos_bp (optional: cm)."
        )

    # Monotonic bp within each chromosome
    for chrom, sub in df.groupby("chrom", sort=False):
        diffs = sub["pos_bp"].diff().dropna()
        bad = diffs[diffs < 0]
        if not bad.empty:
            first_bad_idx = bad.index[0]
            first_bad_marker = df.loc[first_bad_idx, "marker"]
            raise ValueError(
                f"map TSV has non-monotonic pos_bp within chromosome {chrom!r}. "
                f"First offender: {first_bad_marker}."
            )

    if "cm" not in df.columns:
        logger.warning(
            "Map TSV lacks 'cm' column; synthesizing genetic positions via "
            "%.4f cM/Mb. Pass a linkage-map-derived 'cm' column for production runs.",
            recomrate,
        )
        df = df.copy()
        df["cm"] = df["pos_bp"].astype(float) * recomrate / 1e6

    return df[["marker", "chrom", "pos_bp", "cm"]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k load_map
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: _load_map_tsv with cM synthesis + monotonic-bp check"
```

---

## Task 5 — Genofile converter `_build_polyorigin_genofile`

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_phase_polyorigin.py`:

```python
from torchgwas.preprocess.phase_polyorigin import _build_polyorigin_genofile


def _minimal_map_df():
    return pd.DataFrame({
        "marker": ["v1", "v2"],
        "chrom": ["1", "1"],
        "pos_bp": [1000, 2000],
        "cm": [0.001, 0.002],
    })


def test_genofile_probability_encoded_cells(tmp_path):
    # 3 samples (1 parent + 2 offspring), 2 markers, tetraploid (k+1 = 5)
    probs = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0  # dosage 2 for everything
    out = _build_polyorigin_genofile(
        probs=probs,
        sample_ids=["p1", "o1", "o2"],
        variant_ids=["v1", "v2"],
        map_df=_minimal_map_df(),
        parent_phased_df=None,
        parent_ids={"p1"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    assert list(df.columns[:3]) == ["marker", "chromosome", "pos"]
    # Map columns preserved
    assert df["marker"].tolist() == ["v1", "v2"]
    # Expected ordering: parents alphabetical, then offspring alphabetical
    assert list(df.columns[3:]) == ["p1", "o1", "o2"]
    # Cell format: "p0|p1|p2|p3|p4" at 4 decimals; dosage=2 probability=1
    assert df.loc[0, "p1"] == "0.0000|0.0000|1.0000|0.0000|0.0000"


def test_genofile_parent_phased_escape_hatch(tmp_path):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    # Pre-phased parent: 4 chromosomes per marker encoded like "1|0|1|0"
    parent_phased = pd.DataFrame(
        {"v1": ["1|0|1|0"], "v2": ["0|1|0|1"]}, index=["p1"]
    )
    parent_phased.index.name = "individual"
    out = _build_polyorigin_genofile(
        probs=probs,
        sample_ids=["p1", "o1"],
        variant_ids=["v1", "v2"],
        map_df=_minimal_map_df(),
        parent_phased_df=parent_phased,
        parent_ids={"p1"},
        workdir=str(tmp_path),
    )
    df = pd.read_csv(out)
    # Parent cell is the pre-phased string, offspring is probability-encoded
    assert df.loc[0, "p1"] == "1|0|1|0"
    assert df.loc[0, "o1"].count("|") == 4  # 5 probability slots


def test_genofile_parent_in_both_sources_prefers_phased(tmp_path, caplog):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    parent_phased = pd.DataFrame(
        {"v1": ["1|0|1|0"], "v2": ["0|1|0|1"]}, index=["p1"]
    )
    parent_phased.index.name = "individual"
    with caplog.at_level("WARNING"):
        out = _build_polyorigin_genofile(
            probs=probs,
            sample_ids=["p1", "o1"],
            variant_ids=["v1", "v2"],
            map_df=_minimal_map_df(),
            parent_phased_df=parent_phased,
            parent_ids={"p1"},
            workdir=str(tmp_path),
        )
    df = pd.read_csv(out)
    # Pre-phased source wins
    assert df.loc[0, "p1"] == "1|0|1|0"
    assert any("pre-phased" in rec.message.lower() for rec in caplog.records)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k genofile
```

Expected: FAIL, importerror.

- [ ] **Step 3: Implement `_build_polyorigin_genofile`**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
def _build_polyorigin_genofile(
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    map_df: pd.DataFrame,
    parent_phased_df: Optional[pd.DataFrame],
    parent_ids: set[str],
    workdir: str,
) -> Path:
    """Write the merged PolyOrigin genofile CSV.

    Schema: ``marker, chromosome, pos, ind1, ind2, ..., indN``.
    Offspring cells: probability strings ``p0|p1|...|pk`` (4 decimals).
    Parents default to the same encoding; if ``parent_phased_df`` is
    given, those parents use their ``phasedgeno`` strings instead and a
    warning is logged for any parent present in both sources.

    Variant order: ``map_df`` row order. Sample columns: parents
    alphabetical, then offspring alphabetical.
    """
    if probs.dtype != torch.float64:
        probs = probs.to(torch.float64)

    n, m, kp1 = probs.shape
    if len(sample_ids) != n:
        raise ValueError(f"sample_ids length {len(sample_ids)} != probs.shape[0] {n}")
    if len(variant_ids) != m:
        raise ValueError(f"variant_ids length {len(variant_ids)} != probs.shape[1] {m}")

    parents_sorted = sorted(parent_ids & set(sample_ids))
    offspring_sorted = sorted(set(sample_ids) - parent_ids)
    col_order = parents_sorted + offspring_sorted

    # Warn about parents in both sources
    if parent_phased_df is not None:
        both = parent_ids & set(parent_phased_df.index)
        both_in_probs = both & set(sample_ids)
        for pid in both_in_probs:
            logger.warning(
                "Parent %s has both probs-encoded and pre-phased entries; "
                "using pre-phased.",
                pid,
            )

    # Index probs by sample_id for easy lookup
    sample_idx = {sid: i for i, sid in enumerate(sample_ids)}

    # Build per-variant rows
    map_sub = map_df.set_index("marker").loc[variant_ids]  # preserve variant_ids order
    rows: list[dict] = []
    for j, vid in enumerate(variant_ids):
        row = {
            "marker": vid,
            "chromosome": map_sub.loc[vid, "chrom"],
            "pos": map_sub.loc[vid, "cm"],
        }
        for sid in col_order:
            if sid in parent_ids and parent_phased_df is not None and sid in parent_phased_df.index:
                row[sid] = str(parent_phased_df.loc[sid, vid])
            else:
                p = probs[sample_idx[sid], j, :].tolist()
                row[sid] = "|".join(f"{v:.4f}" for v in p)
        rows.append(row)

    out = Path(workdir) / "genofile.csv"
    cols = ["marker", "chromosome", "pos"] + col_order
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    return out
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k genofile
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: pure genofile converter with phasedgeno escape hatch"
```

---

## Task 6 — Input validation layer

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

Central `_validate_inputs` that enforces ploidy, probs shape, sample-ID coverage, and probs-row renormalization, to be called from `run_polyorigin` before any Julia work.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_phase_polyorigin.py`:

```python
from torchgwas.preprocess.phase_polyorigin import _validate_inputs


def test_validate_rejects_bad_ploidy():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match=r"ploidy.*2.*4.*6"):
        _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=3)


def test_validate_rejects_probs_shape_mismatch():
    probs = torch.zeros(2, 2, 4, dtype=torch.float64)  # k+1=4 but ploidy=4 wants 5
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="probs.*shape"):
        _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)


def test_validate_rejects_length_mismatch_sample_ids():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 1.0
    with pytest.raises(ValueError, match="sample_ids"):
        _validate_inputs(probs, ["s1"], ["v1", "v2"], ploidy=4)


def test_validate_renormalizes_slightly_off_rows():
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 0.9995  # sums to 0.9995 — within tolerance
    out = _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)
    # Returned tensor rows sum to ~1
    assert torch.allclose(out.sum(dim=-1), torch.ones_like(out.sum(dim=-1)), atol=1e-6)


def test_validate_warns_on_far_off_rows(caplog):
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 0] = 0.9  # sums to 0.9 — beyond 1e-3 tolerance
    with caplog.at_level("WARNING"):
        _ = _validate_inputs(probs, ["s1", "s2"], ["v1", "v2"], ploidy=4)
    assert any("renormaliz" in rec.message.lower() for rec in caplog.records)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k validate
```

Expected: FAIL on import.

- [ ] **Step 3: Implement `_validate_inputs`**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
def _validate_inputs(
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
) -> Tensor:
    """Raise on invalid input; return probs (possibly renormalized)."""
    if ploidy not in _VALID_PLOIDIES:
        raise ValueError(
            f"PolyOrigin supports ploidy 2, 4, or 6 only; got {ploidy}."
        )
    if probs.ndim != 3:
        raise ValueError(f"probs must be 3D (n, m, k+1); got shape {tuple(probs.shape)}.")
    n, m, kp1 = probs.shape
    if kp1 != ploidy + 1:
        raise ValueError(
            f"probs.shape[-1]={kp1} inconsistent with ploidy={ploidy} "
            f"(expected {ploidy + 1})."
        )
    if len(sample_ids) != n:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} != probs.shape[0] {n}."
        )
    if len(variant_ids) != m:
        raise ValueError(
            f"variant_ids length {len(variant_ids)} != probs.shape[1] {m}."
        )

    # Row-sum check + renormalize
    if probs.dtype != torch.float64:
        probs = probs.to(torch.float64)
    row_sums = probs.sum(dim=-1)
    max_dev = float((row_sums - 1.0).abs().max())
    if max_dev > 1e-3:
        logger.warning(
            "probs rows deviate from 1.0 by up to %.4f (tol=1e-3); renormalizing.",
            max_dev,
        )
    # Always renormalize to be safe (cheap op); upstream calibration may
    # round to 4 decimal places.
    probs = probs / row_sums.unsqueeze(-1).clamp(min=1e-12)
    return probs
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k validate
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: _validate_inputs guard + row-sum renormalization"
```

---

## Task 7 — Runtime discovery logic in `_polyorigin_runtime`

**Files:**
- Modify: `torchgwas/preprocess/_polyorigin_runtime.py`
- Modify: `tests/test_phase_polyorigin.py`

Discovery-only path: find an existing Julia binary (PATH / TORCHGWAS_JULIA / known locations), probe `--version`, reject < 1.10. juliacall install path deferred to Task 8.

- [ ] **Step 1: Write failing tests — use stub julia binaries**

Append to `tests/test_phase_polyorigin.py`:

```python
import platform
import stat
import textwrap

from torchgwas.preprocess._polyorigin_runtime import _find_existing_julia, _probe_version


def _make_fake_julia(tmp_path: Path, version: str) -> Path:
    """Create a cross-platform stub 'julia' that responds to --version."""
    if platform.system() == "Windows":
        p = tmp_path / "julia.bat"
        p.write_text(f"@echo off\necho julia version {version}\n")
    else:
        p = tmp_path / "julia"
        p.write_text(f'#!/bin/sh\necho "julia version {version}"\n')
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_find_existing_julia_via_explicit_path(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    found = _find_existing_julia(override=str(fj))
    assert found == str(fj)


def test_find_existing_julia_via_env_var(tmp_path, monkeypatch):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    monkeypatch.setenv("TORCHGWAS_JULIA", str(fj))
    found = _find_existing_julia(override=None)
    assert found == str(fj)


def test_find_existing_julia_nothing_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("TORCHGWAS_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # no julia here
    found = _find_existing_julia(override=None)
    assert found is None


def test_probe_version_ok(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.10.2")
    ok, ver = _probe_version(str(fj))
    assert ok is True
    assert ver == "1.10.2"


def test_probe_version_too_low(tmp_path):
    fj = _make_fake_julia(tmp_path, "1.8.5")
    ok, ver = _probe_version(str(fj))
    assert ok is False
    assert ver == "1.8.5"


def test_override_to_nonexistent_file_raises(tmp_path):
    bogus = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="not.*exist|not a file"):
        _find_existing_julia(override=str(bogus))
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k "find_existing or probe"
```

Expected: FAIL, not importable.

- [ ] **Step 3: Implement discovery logic in `_polyorigin_runtime.py`**

Append to `torchgwas/preprocess/_polyorigin_runtime.py`:

```python
import glob
import re

_MIN_JULIA = (1, 10)


def _common_julia_paths() -> list[str]:
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, ".juliaup", "bin", "julia"),
        os.path.join(home, ".juliaup", "bin", "julia.exe"),
        "/opt/julia/bin/julia",
        "/usr/local/bin/julia",
    ]
    # Windows Program Files globs
    for pf in (os.environ.get("ProgramFiles", r"C:\Program Files"),
               os.environ.get("LOCALAPPDATA", "")):
        if pf:
            cands.extend(glob.glob(os.path.join(pf, "Julia-*", "bin", "julia.exe")))
    return cands


def _find_existing_julia(override: str | None) -> str | None:
    """Return an absolute path to a Julia binary, or None if not found.

    Order: explicit override → TORCHGWAS_JULIA env var → shutil.which
    ("julia") → common install paths. An override that points to a
    nonexistent/non-executable file raises ValueError.
    """
    if override is not None:
        p = os.path.abspath(override)
        if not os.path.isfile(p):
            raise ValueError(
                f"julia_path={override!r}: does not exist or is not a file."
            )
        if not os.access(p, os.X_OK):
            raise ValueError(
                f"julia_path={override!r}: exists but is not executable."
            )
        return p

    env = os.environ.get("TORCHGWAS_JULIA")
    if env:
        if not os.path.isfile(env):
            raise ValueError(
                f"TORCHGWAS_JULIA={env!r}: does not exist or is not a file."
            )
        return os.path.abspath(env)

    which = shutil.which("julia")
    if which:
        return which

    for cand in _common_julia_paths():
        if os.path.isfile(cand):
            return cand

    return None


def _probe_version(julia_path: str) -> tuple[bool, str]:
    """Return (ok, version_string). ok iff version >= 1.10."""
    try:
        res = subprocess.run(
            [julia_path, "--version"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("julia --version probe failed for %s: %s", julia_path, e)
        return False, "unknown"
    text = (res.stdout or "") + (res.stderr or "")
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return False, text.strip() or "unknown"
    major, minor = int(m.group(1)), int(m.group(2))
    ver = f"{major}.{minor}.{m.group(3)}"
    return (major, minor) >= _MIN_JULIA, ver
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k "find_existing or probe or override"
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/_polyorigin_runtime.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: runtime discover-first Julia search + version probe"
```

---

## Task 8 — Runtime `get_runtime` — juliacall bootstrap + consent gate

**Files:**
- Modify: `torchgwas/preprocess/_polyorigin_runtime.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_phase_polyorigin.py`:

```python
from torchgwas.preprocess import _polyorigin_runtime as rt


def test_get_runtime_raises_when_not_found_and_auto_install_false(tmp_path, monkeypatch):
    # Ensure nothing is discoverable
    monkeypatch.delenv("TORCHGWAS_JULIA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(rt, "_common_julia_paths", lambda: [])
    # Non-tty
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    # Clear any cached state
    rt._jl = None
    with pytest.raises(RuntimeError, match="Julia not detected|auto_install_julia"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_raises_when_found_julia_too_old(tmp_path, monkeypatch):
    fj = _make_fake_julia(tmp_path, "1.8.5")
    monkeypatch.setenv("TORCHGWAS_JULIA", str(fj))
    rt._jl = None
    with pytest.raises(RuntimeError, match=r"1\.8\.5|>= 1\.10"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_raises_without_juliacall_installed(tmp_path, monkeypatch):
    """If juliacall isn't importable, surface a clear install pointer.

    We stub a valid-looking julia and then force an ImportError on juliacall
    to isolate the error path.
    """
    fj = _make_fake_julia(tmp_path, "1.10.2")
    monkeypatch.setenv("TORCHGWAS_JULIA", str(fj))
    rt._jl = None

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "juliacall":
            raise ImportError("stubbed absence of juliacall")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="polyploid-phase|pip install"):
        rt.get_runtime(julia_path=None, auto_install_julia=False)


def test_get_runtime_cached_after_first_success(monkeypatch):
    """When already cached, get_runtime returns without touching the discovery logic."""
    sentinel_jl = object()
    sentinel_po = object()
    rt._jl = sentinel_jl
    rt._polyorigin = sentinel_po
    rt._version = "9.9.9"

    jl, po, ver = rt.get_runtime(julia_path=None, auto_install_julia=False)
    assert jl is sentinel_jl
    assert po is sentinel_po
    assert ver == "9.9.9"
```

Note: add `import sys, types` to the test file header if not already present.

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k get_runtime
```

Expected: FAIL, `get_runtime` not defined.

- [ ] **Step 3: Implement `get_runtime` + `_consent_to_install`**

Append to `torchgwas/preprocess/_polyorigin_runtime.py`:

```python
def _consent_to_install(auto_install: bool) -> bool:
    """Return True iff the caller has authorized a Julia download."""
    if auto_install:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        reply = input(
            "Julia not detected. Install Julia 1.10 (~300 MB) + "
            "PolyOrigin.jl now? [y/N]: "
        )
    except EOFError:
        return False
    return reply.strip().lower() in {"y", "yes"}


def get_runtime(
    julia_path: str | None = None,
    auto_install_julia: bool = False,
) -> tuple[Any, Any, str]:
    """Bootstrap PolyOrigin on demand. Returns (Main, PolyOrigin, version)."""
    global _jl, _polyorigin, _version
    if _jl is not None:
        return _jl, _polyorigin, _version

    with _lock:
        if _jl is not None:
            return _jl, _polyorigin, _version

        # Step 1-2: discover + version-probe
        found = _find_existing_julia(julia_path)
        chosen: str | None = None
        if found is not None:
            ok, ver = _probe_version(found)
            if ok:
                chosen = found
                logger.info("Using existing Julia at %s (v%s)", chosen, ver)
            else:
                if julia_path is not None or os.environ.get("TORCHGWAS_JULIA"):
                    # User explicitly pinned an old Julia — hard fail
                    raise RuntimeError(
                        f"Julia at {found!r} is v{ver}; PolyOrigin requires >= 1.10. "
                        "Install a newer Julia or unset the override."
                    )
                logger.info("Found Julia at %s (v%s) but version < 1.10; ignoring.", found, ver)

        # Step 3: if still no chosen binary, require consent
        if chosen is None:
            if not _consent_to_install(auto_install_julia):
                raise RuntimeError(
                    "Julia not detected and auto_install_julia=False. "
                    "Either: (a) install Julia yourself from "
                    "https://julialang.org/downloads/ (ensure `julia` is on "
                    "PATH or set TORCHGWAS_JULIA), or (b) retry with "
                    "auto_install_julia=True (library) / --auto-install (CLI) "
                    "to let juliacall provision Julia 1.10 automatically "
                    "(~300 MB download)."
                )
            logger.info("Installing Julia 1.10 via juliacall (~300 MB, one-time)")

        # Step 4: pin the existing Julia via juliacall's env var if we found one
        if chosen is not None:
            os.environ["PYTHON_JULIAPKG_EXE"] = chosen

        # Step 5: import juliacall (this triggers juliacall's own download
        # only if chosen is None and juliapkg.json can't be satisfied from
        # an existing managed install). The env var name is per current
        # juliacall docs; if upstream renames it, adjust here.
        try:
            from juliacall import Main as jl_local  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "juliacall not installed. Install the polyploid-phase extra: "
                "pip install torchgwas[polyploid-phase]"
            ) from e

        # Step 6: Pkg.instantiate reads juliapkg.json next to this module.
        # PolyOrigin.jl installs from the pinned URL+rev on first call.
        try:
            jl_local.seval("using PolyOrigin")
            version_str = str(jl_local.seval("string(pkgversion(PolyOrigin))"))
        except Exception as e:
            raise RuntimeError(
                f"PolyOrigin.jl unavailable after juliacall bootstrap: {e}. "
                "If this is a transient network issue, retry; otherwise run "
                "'julia -e \"using Pkg; Pkg.resolve()\"' to recover."
            ) from e

        _jl = jl_local
        _polyorigin = jl_local.PolyOrigin
        _version = version_str
        return _jl, _polyorigin, _version
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k get_runtime
```

Expected: 4 passed. (The "cached" test passes trivially; the "not found" / "too old" / "no juliacall" tests exercise the error paths without actually launching Julia.)

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/_polyorigin_runtime.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: get_runtime — consent gate, version guard, juliacall bootstrap"
```

---

## Task 9 — Output CSV parsers + canned fixtures

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`
- Create: `tests/fixtures/phase_polyorigin/out_genoprob.csv`
- Create: `tests/fixtures/phase_polyorigin/out_postdoseprob.csv`
- Create: `tests/fixtures/phase_polyorigin/out_parentphased.csv`
- Create: `tests/fixtures/phase_polyorigin/out_maprefined.csv`
- Create: `tests/fixtures/phase_polyorigin/out_polyancestry.csv`

**Important**: the exact CSV column layouts are verified against a live PolyOrigin run during implementation (Section 8 of the spec flags this). Below uses the **best-guess layouts derived from upstream docs**; if the first Tier 2 parity run disagrees, regenerate these fixtures from a real Julia run and re-run Task 9.

- [ ] **Step 1: Write best-guess canned fixtures**

Create `tests/fixtures/phase_polyorigin/out_genoprob.csv` (ID-keyed; `marker` as first column, then one column per offspring-chromosome-origin slot — tetraploid with 2 parents ⇒ 4 parental haplotypes; rows are markers; values sum per (marker, offspring, chromosome) to 1):

```csv
marker,chromosome,pos,o1_c1_h1,o1_c1_h2,o1_c1_h3,o1_c1_h4,o1_c2_h1,o1_c2_h2,o1_c2_h3,o1_c2_h4,o1_c3_h1,o1_c3_h2,o1_c3_h3,o1_c3_h4,o1_c4_h1,o1_c4_h2,o1_c4_h3,o1_c4_h4
v1,1,0.0010,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0
v2,1,0.0020,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
```

> If the upstream layout turns out to be long-format (one row per `(marker, ind, chrom, origin)`) rather than wide, adjust the parser in Step 3 and re-seed this fixture from a real run.

Create `tests/fixtures/phase_polyorigin/out_postdoseprob.csv`:

```csv
marker,chromosome,pos,o1_d0,o1_d1,o1_d2,o1_d3,o1_d4
v1,1,0.0010,0.0000,0.0000,1.0000,0.0000,0.0000
v2,1,0.0020,0.0000,0.1000,0.8000,0.1000,0.0000
```

Create `tests/fixtures/phase_polyorigin/out_parentphased.csv`:

```csv
marker,chromosome,pos,p1,p2
v1,1,0.0010,1|0|1|0,0|1|0|1
v2,1,0.0020,1|0|0|1,1|0|1|0
```

Create `tests/fixtures/phase_polyorigin/out_maprefined.csv`:

```csv
marker,chromosome,pos
v1,1,0.0010
v2,1,0.0020
```

Create `tests/fixtures/phase_polyorigin/out_polyancestry.csv`:

```csv
marker,chromosome,valent,freq
v1,1,bivalent,1.0
v2,1,bivalent,1.0
```

- [ ] **Step 2: Write failing parser tests**

Append to `tests/test_phase_polyorigin.py`:

```python
FIXTURES = Path(__file__).parent / "fixtures" / "phase_polyorigin"

from torchgwas.preprocess.phase_polyorigin import (
    _parse_genoprob, _parse_parentphased, _parse_postdose, _parse_maprefined,
    _parse_polyancestry,
)


def test_parse_genoprob_shape(tmp_path):
    origin_probs, offspring_ids = _parse_genoprob(
        str(FIXTURES / "out_genoprob.csv"),
        expected_offspring=["o1"],
        ploidy=4,
    )
    # (n_off, ploidy, m, n_parent_haps) — 1 offspring, tetraploid, 2 markers, 4 parent haps
    assert origin_probs.shape == (1, 4, 2, 4)
    # Row sums to ~1 along the origin axis
    assert torch.allclose(origin_probs.sum(dim=-1),
                          torch.ones((1, 4, 2), dtype=torch.float64), atol=1e-3)
    assert offspring_ids == ["o1"]


def test_parse_parentphased_values():
    parent_phased, parent_ids = _parse_parentphased(
        str(FIXTURES / "out_parentphased.csv"),
        expected_parents=["p1", "p2"],
        max_ploidy=4,
    )
    # (n_parents, max_ploidy, m) — 2 parents, tetraploid, 2 markers
    assert parent_phased.shape == (2, 4, 2)
    # Only {0, 1, -1}
    unique = set(parent_phased.unique().tolist())
    assert unique <= {0, 1, -1}


def test_parse_postdose_shape():
    pd_t, offspring_ids = _parse_postdose(
        str(FIXTURES / "out_postdoseprob.csv"),
        expected_offspring=["o1"],
        max_ploidy=4,
    )
    # (n_off, m, max_ploidy+1) — 1 offspring, 2 markers, 5 dosage slots
    assert pd_t.shape == (1, 2, 5)
    assert torch.allclose(pd_t.sum(dim=-1),
                          torch.ones((1, 2), dtype=torch.float64), atol=1e-3)


def test_parse_maprefined_preserves_input_order():
    chrom, variant_ids, pos_cm = _parse_maprefined(
        str(FIXTURES / "out_maprefined.csv"),
        expected_markers=["v1", "v2"],
    )
    assert variant_ids == ["v1", "v2"]
    assert chrom == ["1", "1"]
    assert pos_cm.shape == (2,)


def test_parse_polyancestry_returns_dataframe():
    df = _parse_polyancestry(str(FIXTURES / "out_polyancestry.csv"))
    assert "marker" in df.columns
    assert "valent" in df.columns
```

- [ ] **Step 3: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k parse_
```

Expected: FAIL, parsers not defined.

- [ ] **Step 4: Implement the parsers (keyed by name, no positional assumptions)**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
def _parse_genoprob(
    path: str,
    expected_offspring: list[str],
    ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse PolyOrigin's *_genoprob.csv.

    Expected column scheme (best-guess from upstream docs; implementer
    verifies against a live run): ``marker, chromosome, pos`` followed by
    offspring × chromosome-copy × parent-haplotype probability columns
    named like ``<offspring>_c<chromosome_copy>_h<parent_haplotype>``.

    Returns (origin_probs (n_off, ploidy, m, n_parent_haps), offspring_ids).
    """
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    prob_cols = [c for c in df.columns if c not in meta_cols]

    # Parse column names into (offspring, c_idx, h_idx)
    import re as _re
    pat = _re.compile(r"^(?P<off>.+)_c(?P<c>\d+)_h(?P<h>\d+)$")
    parsed: list[tuple[str, int, int, str]] = []
    for c in prob_cols:
        m = pat.match(c)
        if m is None:
            raise RuntimeError(
                f"Unexpected column in genoprob file: {c!r}. "
                "Parser expects '<offspring>_c<idx>_h<idx>' naming."
            )
        parsed.append((m["off"], int(m["c"]), int(m["h"]), c))

    offsprings = sorted({x[0] for x in parsed})
    c_vals = sorted({x[1] for x in parsed})
    h_vals = sorted({x[2] for x in parsed})
    if len(c_vals) != ploidy:
        raise RuntimeError(
            f"genoprob column copy-count {len(c_vals)} != ploidy {ploidy}."
        )
    n_parent_haps = len(h_vals)

    # Order offspring per the caller's expected order
    if set(offsprings) != set(expected_offspring):
        raise RuntimeError(
            f"genoprob offspring set {sorted(offsprings)} != expected "
            f"{sorted(expected_offspring)}."
        )
    off_order = list(expected_offspring)

    n_off = len(off_order)
    m = len(df)
    out = torch.zeros(n_off, ploidy, m, n_parent_haps, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    c_idx = {c: i for i, c in enumerate(c_vals)}
    h_idx = {h: i for i, h in enumerate(h_vals)}
    for off, c, h, col in parsed:
        out[off_idx[off], c_idx[c], :, h_idx[h]] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float64
        )
    return out, off_order


def _parse_parentphased(
    path: str,
    expected_parents: list[str],
    max_ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse *_parentphased.csv — one column per parent, cell = ``a1|a2|...|ak``."""
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    parent_cols = [c for c in df.columns if c not in meta_cols]
    if set(parent_cols) != set(expected_parents):
        raise RuntimeError(
            f"parentphased column set {sorted(parent_cols)} != expected "
            f"{sorted(expected_parents)}."
        )
    m = len(df)
    out = torch.full((len(expected_parents), max_ploidy, m), -1, dtype=torch.int8)
    for i, p in enumerate(expected_parents):
        for j, cell in enumerate(df[p]):
            alleles = str(cell).split("|")
            for k, a in enumerate(alleles[:max_ploidy]):
                try:
                    out[i, k, j] = int(a)
                except ValueError:
                    raise RuntimeError(
                        f"parentphased cell at parent={p} marker_idx={j}: "
                        f"non-integer allele {a!r}."
                    )
    return out, list(expected_parents)


def _parse_postdose(
    path: str,
    expected_offspring: list[str],
    max_ploidy: int,
) -> tuple[Tensor, list[str]]:
    """Parse *_postdoseprob.csv — columns like ``<offspring>_d<dosage>``."""
    df = pd.read_csv(path)
    meta_cols = {"marker", "chromosome", "pos"}
    import re as _re
    pat = _re.compile(r"^(?P<off>.+)_d(?P<d>\d+)$")
    parsed: list[tuple[str, int, str]] = []
    for c in df.columns:
        if c in meta_cols:
            continue
        m = pat.match(c)
        if m is None:
            raise RuntimeError(
                f"Unexpected column in postdose file: {c!r}. "
                "Parser expects '<offspring>_d<dosage>' naming."
            )
        parsed.append((m["off"], int(m["d"]), c))

    offsprings = sorted({x[0] for x in parsed})
    if set(offsprings) != set(expected_offspring):
        raise RuntimeError(
            f"postdose offspring set {sorted(offsprings)} != expected."
        )
    kmax = max(x[1] for x in parsed)
    if kmax + 1 > max_ploidy + 1:
        raise RuntimeError(
            f"postdose dosage slots {kmax + 1} > max_ploidy+1 {max_ploidy + 1}."
        )

    off_order = list(expected_offspring)
    n_off = len(off_order)
    m = len(df)
    out = torch.zeros(n_off, m, max_ploidy + 1, dtype=torch.float64)
    off_idx = {o: i for i, o in enumerate(off_order)}
    for off, d, col in parsed:
        out[off_idx[off], :, d] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float64
        )
    return out, off_order


def _parse_maprefined(
    path: str,
    expected_markers: list[str],
) -> tuple[list[str], list[str], Tensor]:
    """Return (chrom, variant_ids, pos_cm_tensor)."""
    df = pd.read_csv(path)
    got = set(df["marker"])
    if got != set(expected_markers):
        raise RuntimeError(
            f"map_refined marker set differs from input "
            f"(missing {set(expected_markers) - got}, extra {got - set(expected_markers)})."
        )
    return (
        [str(c) for c in df["chromosome"]],
        [str(v) for v in df["marker"]],
        torch.tensor(df["pos"].to_numpy(), dtype=torch.float64),
    )


def _parse_polyancestry(path: str) -> pd.DataFrame:
    """Return raw DataFrame of the *_polyancestry.csv — diagnostic output."""
    return pd.read_csv(path)
```

- [ ] **Step 5: Run to verify they pass**

```bash
pytest tests/test_phase_polyorigin.py -v -k parse_
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py \
        tests/test_phase_polyorigin.py \
        tests/fixtures/phase_polyorigin/
git commit -m "Phase 56: output parsers + canned best-guess PolyOrigin CSV fixtures"
```

---

## Task 10 — `run_polyorigin` orchestration + atomicity

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing orchestration tests with monkeypatched runtime**

Append to `tests/test_phase_polyorigin.py`:

```python
import shutil as _shutil

from torchgwas.preprocess.phase_polyorigin import run_polyorigin


def _stub_runtime(workdir_spy: dict):
    """Return a fake PolyOrigin module that copies the canned fixtures
    into the workdir so the parser path can run end-to-end.
    """
    class FakePO:
        @staticmethod
        def polyOrigin(genofile, pedfile, **kwargs):
            workdir_spy["genofile"] = genofile
            workdir_spy["pedfile"] = pedfile
            wd = Path(kwargs["workdir"])
            outstem = kwargs.get("outstem", "out")
            for name in ("genoprob", "postdoseprob", "parentphased",
                         "maprefined", "polyancestry"):
                _shutil.copy(
                    FIXTURES / f"out_{name}.csv",
                    wd / f"{outstem}_{name}.csv",
                )
            return None

    class FakeMain:  # juliacall Main proxy
        PolyOrigin = FakePO

    return FakeMain, FakePO, "1.0.3-fake"


def test_run_polyorigin_happy_path(tmp_path, monkeypatch):
    from torchgwas.preprocess import _polyorigin_runtime as rt

    # Stub the runtime — no Julia touched
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    # Minimal inputs: 1 parent, 1 offspring (edge case — but shape is what we're testing)
    probs = torch.zeros(2, 2, 5, dtype=torch.float64)
    probs[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp1\n")  # selfing
    # NOTE: the parentphased fixture ships p1 + p2; for this test we align
    # on a 2-parent setup — regenerate ped/map to match the fixture.
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0

    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")

    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    result = run_polyorigin(
        probs=probs3,
        pedigree_tsv=str(ped),
        map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=["p1", "p2", "o1"],
        variant_ids=["v1", "v2"],
        auto_install_julia=False,  # stubbed away
    )

    assert isinstance(result, PhasingResult)
    assert result.tool == "polyorigin"
    assert result.ploidy == 4 or result.per_individual_ploidy["o1"] == 4
    # Output artifacts exist
    assert Path(f"{out_prefix}.haplotypes.pt").is_file()
    assert Path(f"{out_prefix}.origin_probs.pt").is_file()
    assert Path(f"{out_prefix}.parent_phased.pt").is_file()
    assert Path(f"{out_prefix}.postdose_probs.pt").is_file()
    assert Path(f"{out_prefix}.meta.json").is_file()
    meta = json.loads(Path(f"{out_prefix}.meta.json").read_text())
    assert meta["tool_version"] == "1.0.3-fake"
    assert "input_hash" in meta


def test_run_polyorigin_partial_failure_no_persistent_output(tmp_path, monkeypatch):
    """If the runtime 'succeeds' but emits fewer CSVs than expected, we
    must raise AND leave no <output>.* artifacts."""
    from torchgwas.preprocess import _polyorigin_runtime as rt

    def _bad_runtime():
        class FakePO:
            @staticmethod
            def polyOrigin(genofile, pedfile, **kwargs):
                wd = Path(kwargs["workdir"])
                # Only write 2 of the 5 expected CSVs
                _shutil.copy(FIXTURES / "out_genoprob.csv", wd / "out_genoprob.csv")
                _shutil.copy(FIXTURES / "out_parentphased.csv", wd / "out_parentphased.csv")
                return None

        class FakeMain: PolyOrigin = FakePO
        return FakeMain, FakePO, "1.0.3-fake"

    monkeypatch.setattr(rt, "get_runtime", lambda **_: _bad_runtime())

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="missing|parse"):
        run_polyorigin(
            probs=probs3,
            pedigree_tsv=str(ped),
            map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1"],
            variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )

    # No persistent artifacts
    assert not Path(f"{out_prefix}.haplotypes.pt").exists()
    assert not Path(f"{out_prefix}.meta.json").exists()


def test_run_polyorigin_julia_error_bubbles(tmp_path, monkeypatch):
    """juliacall.JuliaError raised by polyOrigin() → RuntimeError with __cause__."""
    from torchgwas.preprocess import _polyorigin_runtime as rt

    class FakeJuliaError(Exception):
        pass

    class FakePO:
        @staticmethod
        def polyOrigin(*a, **kw):
            raise FakeJuliaError("convergence failed")

    class FakeMain: PolyOrigin = FakePO

    monkeypatch.setattr(rt, "get_runtime", lambda **_: (FakeMain, FakePO, "1.0.3-fake"))
    # Also patch juliacall.JuliaError to our fake class so our except clause matches
    monkeypatch.setattr(
        "torchgwas.preprocess.phase_polyorigin._JULIA_ERROR",
        FakeJuliaError,
    )

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text("offspring\tparent1\tparent2\no1\tp1\tp2\n")
    mp = tmp_path / "map.tsv"
    mp.write_text("marker\tchrom\tpos_bp\nv1\t1\t1000\nv2\t1\t2000\n")
    out_prefix = tmp_path / "out" / "phased"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="PolyOrigin failed") as ei:
        run_polyorigin(
            probs=probs3,
            pedigree_tsv=str(ped),
            map_tsv=str(mp),
            output_path=str(out_prefix),
            ploidy=4,
            sample_ids=["p1", "p2", "o1"],
            variant_ids=["v1", "v2"],
            auto_install_julia=False,
        )
    assert isinstance(ei.value.__cause__, FakeJuliaError)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_phase_polyorigin.py -v -k run_polyorigin
```

Expected: FAIL, not implemented.

- [ ] **Step 3: Implement `run_polyorigin` + helpers**

Append to `torchgwas/preprocess/phase_polyorigin.py`:

```python
import tempfile
from datetime import datetime, timezone

# Module-level alias so tests can monkeypatch without touching juliacall
try:
    from juliacall import JuliaError as _JULIA_ERROR  # type: ignore[import-not-found]
except ImportError:
    class _JULIA_ERROR(Exception):  # fallback placeholder
        pass


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_polyorigin(
    probs: Tensor | str,
    pedigree_tsv: str,
    map_tsv: str,
    output_path: str,
    *,
    ploidy: int,
    sample_ids: Optional[list[str]] = None,
    variant_ids: Optional[list[str]] = None,
    parent_phased_csv: Optional[str] = None,
    julia_path: Optional[str] = None,
    auto_install_julia: bool = False,
    refinemap: bool = True,
    recomrate: float = 1.0,
    nworkers: int = 1,
    seed: Optional[int] = 1234,
    keep_workdir: bool = False,
) -> PhasingResult:
    """Run PolyOrigin phasing end-to-end. See design spec Section 2.2."""
    # Resolve probs input — tensor directly, or path to a .probs.pt
    if isinstance(probs, str):
        loaded = torch.load(probs)
        if isinstance(loaded, dict):
            probs_t = loaded["probs"]
            sample_ids = sample_ids or loaded.get("sample_ids")
            variant_ids = variant_ids or loaded.get("variant_ids")
        else:
            probs_t = loaded
    else:
        probs_t = probs

    if sample_ids is None or variant_ids is None:
        raise ValueError(
            "sample_ids and variant_ids are required when probs is a "
            "tensor (or a .pt that doesn't carry them as keys)."
        )

    probs_t = _validate_inputs(probs_t, sample_ids, variant_ids, ploidy)

    # Parse pedigree first so we know parent_ids for the genofile builder
    map_df = _load_map_tsv(map_tsv, recomrate=recomrate)

    # Parent-phased escape hatch
    parent_phased_df: Optional[pd.DataFrame] = None
    if parent_phased_csv is not None:
        parent_phased_df = pd.read_csv(parent_phased_csv).set_index("individual")

    # Import the runtime late to avoid heavy import at module load
    from torchgwas.preprocess import _polyorigin_runtime as _rt

    # Build tempdir and converted files
    with tempfile.TemporaryDirectory(prefix="polyorigin_") as _tmp:
        workdir = Path(_tmp) if not keep_workdir else Path(tempfile.mkdtemp(prefix="polyorigin_keep_"))

        # We need parent_ids from the pedigree TSV; parse once and derive.
        user_ped = pd.read_csv(pedigree_tsv, sep="\t")
        parent_ids = set(user_ped["parent1"]) | set(user_ped["parent2"])

        pedfile = _build_polyorigin_pedfile(
            pedigree_tsv, ploidy_default=ploidy,
            sample_ids_in_probs=set(sample_ids),
            workdir=str(workdir),
        )
        genofile = _build_polyorigin_genofile(
            probs=probs_t, sample_ids=sample_ids, variant_ids=variant_ids,
            map_df=map_df, parent_phased_df=parent_phased_df,
            parent_ids=parent_ids, workdir=str(workdir),
        )

        # Compute input hash (genofile + pedfile + original map file)
        combined = (
            _sha256_file(str(genofile)) + _sha256_file(str(pedfile)) + _sha256_file(map_tsv)
        )
        input_hash = _sha256_str(combined)

        # Bootstrap Julia + invoke PolyOrigin
        jl, po, tool_version = _rt.get_runtime(
            julia_path=julia_path, auto_install_julia=auto_install_julia,
        )
        cmd_str = (
            f'polyOrigin("{genofile.name}", "{pedfile.name}", '
            f'workdir="{workdir}", isphysmap=false, refinemap={str(refinemap).lower()}, '
            f'nworkers={nworkers}, seed={seed}, outstem="out")'
        )
        try:
            po.polyOrigin(
                str(genofile), str(pedfile),
                workdir=str(workdir),
                isphysmap=False,
                refinemap=refinemap,
                nworkers=nworkers,
                seed=seed,
                outstem="out",
            )
        except _JULIA_ERROR as e:
            raise RuntimeError(f"PolyOrigin failed: {e}") from e

        # Validate the full set of output CSVs exist
        expected = ["out_genoprob.csv", "out_postdoseprob.csv",
                    "out_parentphased.csv", "out_maprefined.csv",
                    "out_polyancestry.csv"]
        for fname in expected:
            p = workdir / fname
            if not p.is_file():
                raise RuntimeError(
                    f"PolyOrigin returned but expected output file missing: {fname}. "
                    f"Check the Julia log at {workdir / 'out.log'}."
                )

        # Identify offspring / parents from the constructed pedfile
        ped_df = pd.read_csv(pedfile)
        parents = ped_df[ped_df["population"] == 0]["individual"].astype(str).tolist()
        offspring = ped_df[ped_df["population"] != 0]["individual"].astype(str).tolist()
        max_ploidy = int(ped_df["ploidy"].max())
        per_ind_ploidy = dict(zip(ped_df["individual"].astype(str), ped_df["ploidy"].astype(int)))

        # Parse outputs
        chrom_ref, var_ids_ref, pos_cm_ref = _parse_maprefined(
            str(workdir / "out_maprefined.csv"), expected_markers=variant_ids,
        )
        map_refined_flag = var_ids_ref != variant_ids
        origin_probs, _ = _parse_genoprob(
            str(workdir / "out_genoprob.csv"),
            expected_offspring=offspring, ploidy=max_ploidy,
        )
        postdose_probs, _ = _parse_postdose(
            str(workdir / "out_postdoseprob.csv"),
            expected_offspring=offspring, max_ploidy=max_ploidy,
        )
        parent_phased, _ = _parse_parentphased(
            str(workdir / "out_parentphased.csv"),
            expected_parents=parents, max_ploidy=max_ploidy,
        )
        valent_diag = _parse_polyancestry(str(workdir / "out_polyancestry.csv"))

        # Derive haplotypes as argmax of origin_probs along last axis → int8
        haplotypes = origin_probs.argmax(dim=-1).to(torch.int8)

        # Build map-based tensors on the refined order
        map_df_ref = map_df.set_index("marker").loc[var_ids_ref]
        pos_bp = torch.tensor(map_df_ref["pos_bp"].to_numpy(), dtype=torch.int64)

        result = PhasingResult(
            haplotypes=haplotypes,
            origin_probs=origin_probs,
            parent_phased=parent_phased,
            offspring_ids=offspring,
            parent_ids=parents,
            variant_ids=var_ids_ref,
            chrom=chrom_ref,
            pos_bp=pos_bp,
            pos_cm=pos_cm_ref,
            per_individual_ploidy=per_ind_ploidy,
            map_refined=map_refined_flag,
            valent_diag=valent_diag,
            postdose_probs=postdose_probs,
            tool="polyorigin",
            tool_version=tool_version,
            input_hash=input_hash,
            cmd=cmd_str,
            workdir=str(workdir) if keep_workdir else None,
        )

        # Atomic persistence — only after every parse succeeds
        _persist_result(result, output_path)
        return result
```

Note the `_persist_result` call — implemented in Task 11.

- [ ] **Step 4: Temporarily stub `_persist_result` to let tests compile**

Add a placeholder at the bottom of `phase_polyorigin.py`:

```python
def _persist_result(result: PhasingResult, output_path: str) -> None:
    """Placeholder — implemented in Task 11."""
    pass
```

- [ ] **Step 5: Run to verify orchestration tests reach `_persist_result`**

```bash
pytest tests/test_phase_polyorigin.py -v -k run_polyorigin
```

Expected: `test_run_polyorigin_happy_path` will FAIL on `<output>.haplotypes.pt` assertion — persistence not implemented. `test_run_polyorigin_partial_failure_no_persistent_output` and `test_run_polyorigin_julia_error_bubbles` should PASS.

- [ ] **Step 6: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: run_polyorigin orchestration + atomicity (persistence stubbed)"
```

---

## Task 11 — Persistence (`_persist_result`)

**Files:**
- Modify: `torchgwas/preprocess/phase_polyorigin.py`

- [ ] **Step 1: Implement `_persist_result` — replace the Task 10 stub**

Replace the placeholder:

```python
def _persist_result(result: PhasingResult, output_path: str) -> None:
    """Atomically persist a PhasingResult to ``<output_path>.*``.

    Uses a temp-sibling-then-rename scheme: each artifact is written to
    ``<output_path>.<name>.<tmpid>`` then renamed. If any write fails,
    previously-renamed siblings are removed so the filesystem ends up
    with either all new artifacts or none.
    """
    prefix = Path(output_path)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    tmpid = os.getpid()
    renamed: list[Path] = []

    def _persist(name: str, writer) -> None:
        tmp = prefix.with_suffix(prefix.suffix + f".{name}.tmp.{tmpid}")
        writer(str(tmp))
        final = prefix.with_suffix(prefix.suffix + f".{name}")
        os.replace(tmp, final)
        renamed.append(final)

    try:
        _persist("haplotypes.pt", lambda p: torch.save(result.haplotypes, p))
        _persist("origin_probs.pt", lambda p: torch.save(result.origin_probs, p))
        _persist("parent_phased.pt", lambda p: torch.save(result.parent_phased, p))
        _persist("postdose_probs.pt", lambda p: torch.save(result.postdose_probs, p))
        _persist("map_refined.tsv", lambda p: result.valent_diag.to_csv(p, index=False, sep="\t"))  # see note below
        _persist("valent_diag.tsv", lambda p: result.valent_diag.to_csv(p, index=False, sep="\t"))

        meta = {
            "tool": result.tool,
            "tool_version": result.tool_version,
            "ploidy_per_individual": result.per_individual_ploidy,
            "offspring_ids": result.offspring_ids,
            "parent_ids": result.parent_ids,
            "variant_ids": result.variant_ids,
            "chrom": result.chrom,
            "pos_bp": result.pos_bp.tolist(),
            "pos_cm": result.pos_cm.tolist(),
            "map_refined": result.map_refined,
            "input_hash": result.input_hash,
            "cmd": result.cmd,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _persist("meta.json", lambda p: Path(p).write_text(json.dumps(meta, indent=2)))
    except Exception:
        # Roll back any renamed artifacts so we leave no partial <output>.*
        for p in renamed:
            try:
                p.unlink()
            except OSError:
                pass
        raise
```

Note: `_persist("map_refined.tsv", ...)` above should write the refined-map DataFrame, not `valent_diag`. Correct that before continuing:

```python
        _persist("map_refined.tsv", lambda p: pd.DataFrame({
            "marker": result.variant_ids,
            "chrom": result.chrom,
            "pos_bp": result.pos_bp.tolist(),
            "pos_cm": result.pos_cm.tolist(),
        }).to_csv(p, index=False, sep="\t"))
```

- [ ] **Step 2: Run the happy-path test**

```bash
pytest tests/test_phase_polyorigin.py::test_run_polyorigin_happy_path -v
```

Expected: PASS. All `<output>.*` artifacts exist, `meta.json` parses.

- [ ] **Step 3: Write a persistence-specific atomicity test**

Append to `tests/test_phase_polyorigin.py`:

```python
def test_persist_rolls_back_on_partial_write(tmp_path, monkeypatch):
    from torchgwas.preprocess.phase_polyorigin import _persist_result

    prefix = tmp_path / "out" / "phased"
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Build a minimal PhasingResult
    r = PhasingResult(
        haplotypes=torch.zeros(1, 4, 2, dtype=torch.int8),
        origin_probs=torch.zeros(1, 4, 2, 4, dtype=torch.float64),
        parent_phased=torch.zeros(2, 4, 2, dtype=torch.int8),
        offspring_ids=["o1"],
        parent_ids=["p1", "p2"],
        variant_ids=["v1", "v2"],
        chrom=["1", "1"],
        pos_bp=torch.tensor([100, 200], dtype=torch.int64),
        pos_cm=torch.tensor([0.0001, 0.0002], dtype=torch.float64),
        per_individual_ploidy={"p1": 4, "p2": 4, "o1": 4},
        map_refined=False,
        valent_diag=pd.DataFrame({"marker": ["v1"], "valent": ["bivalent"]}),
        postdose_probs=torch.zeros(1, 2, 5, dtype=torch.float64),
        tool="polyorigin",
        tool_version="1.0.3-fake",
        input_hash="abc",
        cmd="...",
        workdir=None,
    )

    # Force meta.json write to fail
    real_write_text = Path.write_text
    call_count = {"n": 0}

    def fake_write_text(self, content, *a, **kw):
        call_count["n"] += 1
        if self.name.startswith("phased.meta.json"):
            raise OSError("simulated disk full")
        return real_write_text(self, content, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fake_write_text)

    with pytest.raises(OSError):
        _persist_result(r, str(prefix))

    # No persistent artifacts survived
    leftovers = list(prefix.parent.glob("phased.*"))
    assert leftovers == [], f"Unexpected leftover artifacts: {leftovers}"
```

- [ ] **Step 4: Run**

```bash
pytest tests/test_phase_polyorigin.py::test_persist_rolls_back_on_partial_write -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/phase_polyorigin.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: atomic _persist_result with rollback on partial write"
```

---

## Task 12 — CLI `phase-poly` subcommand

**Files:**
- Modify: `torchgwas/cli.py`
- Modify: `torchgwas/preprocess/__init__.py`
- Modify: `tests/test_phase_polyorigin.py`

- [ ] **Step 1: Write failing CLI test**

Append to `tests/test_phase_polyorigin.py`:

```python
import subprocess as _sp


def test_cli_phase_poly_help():
    res = _sp.run(
        [sys.executable, "-m", "torchgwas", "phase-poly", "--help"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    # Core flags present
    for flag in ["--probs", "--pedigree", "--map", "--output", "--ploidy",
                 "--parent-phased", "--auto-install", "--julia-path",
                 "--keep-workdir"]:
        assert flag in res.stdout
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_phase_polyorigin.py::test_cli_phase_poly_help -v
```

Expected: FAIL — subcommand unknown.

- [ ] **Step 3: Add the subparser in `torchgwas/cli.py`**

Insert near the other phase-related subparsers (e.g. after `dosage-call`):

```python
    p = subparsers.add_parser(
        "phase-poly",
        help="Polyploid F1 phasing via PolyOrigin (Phase 56)",
    )
    p.add_argument("--probs", required=True,
                   help="Path to .probs.pt from 'dosage-call' (or any (n,m,k+1) torch tensor)")
    p.add_argument("--pedigree", required=True,
                   help="TSV with columns: offspring, parent1, parent2 (optional: ploidy)")
    p.add_argument("--map", required=True,
                   help="TSV with columns: marker, chrom, pos_bp (optional: cm)")
    p.add_argument("--output", required=True,
                   help="Output prefix; writes <prefix>.haplotypes.pt etc.")
    p.add_argument("--ploidy", type=int, required=True, choices=[2, 4, 6])
    p.add_argument("--parent-phased", default=None,
                   help="Optional CSV of pre-phased parent genotypes")
    p.add_argument("--no-refinemap", dest="refinemap", action="store_false",
                   help="Disable PolyOrigin's map refinement (default: enabled)")
    p.set_defaults(refinemap=True)
    p.add_argument("--recomrate", type=float, default=1.0,
                   help="cM/Mb to synthesize genetic positions when --map lacks a cm column")
    p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--julia-path", default=None,
                   help="Path to an existing Julia binary; else discovered or auto-installed")
    install_group = p.add_mutually_exclusive_group()
    install_group.add_argument("--auto-install", dest="auto_install", action="store_true",
                               help="Auto-install Julia if not found (non-interactive)")
    install_group.add_argument("--no-auto-install", dest="auto_install", action="store_false",
                               help="Refuse to auto-install Julia (non-interactive)")
    p.set_defaults(auto_install=None)  # None → interactive prompt on tty
    p.add_argument("--keep-workdir", action="store_true",
                   help="Do not delete the temp work directory after success")
    p.set_defaults(func=_cmd_phase_poly)
```

- [ ] **Step 4: Add the dispatch function `_cmd_phase_poly` in `torchgwas/cli.py`**

Near the other `_cmd_*` functions:

```python
def _cmd_phase_poly(args):
    from torchgwas.preprocess.phase_polyorigin import run_polyorigin

    # Map CLI's None sentinel for --auto-install to False (non-interactive default)
    auto = args.auto_install if args.auto_install is not None else False

    # Load probs — either a raw (n,m,k+1) tensor or a dict with sample/variant IDs
    probs_obj = torch.load(args.probs)
    if isinstance(probs_obj, dict):
        probs = probs_obj["probs"]
        sample_ids = probs_obj.get("sample_ids")
        variant_ids = probs_obj.get("variant_ids")
    else:
        probs = probs_obj
        sample_ids = None
        variant_ids = None
    if sample_ids is None or variant_ids is None:
        raise SystemExit(
            "--probs must be a dict with 'sample_ids' and 'variant_ids' keys "
            "(as produced by 'torchgwas dosage-call'), or you must pass them "
            "explicitly via the Python API."
        )

    run_polyorigin(
        probs=probs,
        pedigree_tsv=args.pedigree,
        map_tsv=args.map,
        output_path=args.output,
        ploidy=args.ploidy,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        parent_phased_csv=args.parent_phased,
        julia_path=args.julia_path,
        auto_install_julia=auto,
        refinemap=args.refinemap,
        recomrate=args.recomrate,
        nworkers=args.nworkers,
        seed=args.seed,
        keep_workdir=args.keep_workdir,
    )
```

- [ ] **Step 5: Re-export from `torchgwas/preprocess/__init__.py`**

Add to `torchgwas/preprocess/__init__.py` (or the existing re-export block if one exists):

```python
from torchgwas.preprocess.phase_polyorigin import PhasingResult, run_polyorigin

__all__ = [*globals().get("__all__", []), "PhasingResult", "run_polyorigin"]
```

- [ ] **Step 6: Run the CLI test**

```bash
pytest tests/test_phase_polyorigin.py::test_cli_phase_poly_help -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add torchgwas/cli.py torchgwas/preprocess/__init__.py tests/test_phase_polyorigin.py
git commit -m "Phase 56: phase-poly CLI subcommand + preprocess re-exports"
```

---

## Task 13 — Additional Tier 1 guard tests (top-up to ~25)

**Files:**
- Modify: `tests/test_phase_polyorigin.py`

Earlier tasks produced ~19 tests. Top up to cover the remaining spec-required scenarios: metadata hash matching, renormalization warning at the output layer, mixed-ploidy per-row handling, and env-var-override interaction.

- [ ] **Step 1: Add the remaining Tier 1 tests**

```python
def test_meta_json_hash_matches_inputs(tmp_path, monkeypatch):
    from torchgwas.preprocess import _polyorigin_runtime as rt

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

    meta = json.loads(Path(f"{out_prefix}.meta.json").read_text())
    assert meta["input_hash"] == result.input_hash
    # Julia cmd captured for reproducibility
    assert "polyOrigin" in meta["cmd"]


def test_mixed_ploidy_per_row_honored(tmp_path, monkeypatch):
    from torchgwas.preprocess import _polyorigin_runtime as rt
    spy: dict = {}
    monkeypatch.setattr(rt, "get_runtime", lambda **_: _stub_runtime(spy))

    probs3 = torch.zeros(3, 2, 5, dtype=torch.float64)
    probs3[:, :, 2] = 1.0
    ped = tmp_path / "ped.tsv"
    ped.write_text(
        "offspring\tparent1\tparent2\tploidy\n"
        "o1\tp1\tp2\t4\n"
    )
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
    assert result.per_individual_ploidy["o1"] == 4
    assert result.per_individual_ploidy["p1"] == 4


def test_env_override_wins_over_path(tmp_path, monkeypatch):
    from torchgwas.preprocess._polyorigin_runtime import _find_existing_julia

    a = _make_fake_julia(tmp_path / "a", "1.10.0")  # never mind — make_fake is flat; use tmp_path
    # Put a stub on PATH and a different one in the env var
    tmp_path.joinpath("path_bin").mkdir(exist_ok=True)
    tmp_path.joinpath("env_bin").mkdir(exist_ok=True)
    path_julia = _make_fake_julia(tmp_path / "path_bin", "1.10.0")
    env_julia = _make_fake_julia(tmp_path / "env_bin", "1.10.2")
    monkeypatch.setenv("PATH", str(tmp_path / "path_bin"))
    monkeypatch.setenv("TORCHGWAS_JULIA", str(env_julia))
    found = _find_existing_julia(override=None)
    assert found == str(env_julia)
```

Note: `_make_fake_julia` writes a single file named `julia`/`julia.bat` — ensure the per-subdir calls work on both POSIX and Windows. If the fake creates a file at `tmp_path/env_bin/julia`, `TORCHGWAS_JULIA=<that path>` and the discovery logic will pick it up by path. Adjust `_make_fake_julia` to accept an explicit parent dir if needed.

- [ ] **Step 2: Run all Tier 1 tests**

```bash
pytest tests/test_phase_polyorigin.py -v
```

Expected: ~22–25 tests pass on Linux + Windows.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase_polyorigin.py
git commit -m "Phase 56: Tier 1 top-up — meta hash, mixed ploidy, env-var precedence"
```

---

## Task 14 — Tier 2 e2e scaffold + parity test

**Files:**
- Create: `tests/test_phase_polyorigin_e2e.py`

Gated on real Julia + PolyOrigin.jl.

- [ ] **Step 1: Create the Tier 2 test file with gating**

```python
"""Phase 56 Tier 2 tests — require real Julia + PolyOrigin.jl.

Gated: (i) juliacall importable, (ii) Julia >= 1.10 discoverable,
(iii) PolyOrigin available (or TORCHGWAS_ALLOW_AUTO_INSTALL=1).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import pandas as pd


def _tier2_skipif_reason() -> str | None:
    try:
        import juliacall  # noqa: F401
    except ImportError:
        return "juliacall not installed (pip install torchgwas[polyploid-phase])"
    from torchgwas.preprocess._polyorigin_runtime import _find_existing_julia, _probe_version
    jl = _find_existing_julia(override=None)
    if jl is None and not os.environ.get("TORCHGWAS_ALLOW_AUTO_INSTALL"):
        return "No Julia found and TORCHGWAS_ALLOW_AUTO_INSTALL unset"
    if jl is not None:
        ok, ver = _probe_version(jl)
        if not ok:
            return f"Julia at {jl} is v{ver} (< 1.10)"
    return None


pytestmark = pytest.mark.skipif(
    _tier2_skipif_reason() is not None,
    reason=_tier2_skipif_reason() or "",
)


def _make_toy_tetraploid_f1_probs(rng, n_off: int, m: int) -> tuple[torch.Tensor, list[str], list[str]]:
    """Simulate posterior dosage probabilities for 2 parents + N offspring."""
    n = n_off + 2
    probs = torch.zeros(n, m, 5, dtype=torch.float64)
    for i in range(n):
        for j in range(m):
            d = int(rng.integers(0, 5))
            probs[i, j, d] = 1.0
    sample_ids = ["p1", "p2"] + [f"o{k+1}" for k in range(n_off)]
    variant_ids = [f"v{j+1}" for j in range(m)]
    return probs, sample_ids, variant_ids


def test_parity_tetraploid_f1(tmp_path):
    """Our wrapper vs bare PolyOrigin.polyOrigin() on the SAME files.

    Runs run_polyorigin with keep_workdir=True, captures the genofile +
    pedfile it wrote, re-invokes PolyOrigin directly in Julia against those
    files, and compares origin_probs and parent_phased.
    """
    import numpy as np
    rng = np.random.default_rng(0)

    n_off, m = 6, 10
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

    from torchgwas.preprocess.phase_polyorigin import run_polyorigin
    auto = bool(os.environ.get("TORCHGWAS_ALLOW_AUTO_INSTALL"))
    result_ours = run_polyorigin(
        probs=probs,
        pedigree_tsv=str(ped),
        map_tsv=str(mp),
        output_path=str(out_prefix),
        ploidy=4,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        keep_workdir=True,
        auto_install_julia=auto,
    )

    # Re-run PolyOrigin directly on the same files
    workdir = Path(result_ours.workdir)
    from juliacall import Main as jl
    jl.seval("using PolyOrigin")
    ref_dir = workdir / "ref"
    ref_dir.mkdir()
    jl.PolyOrigin.polyOrigin(
        str(workdir / "genofile.csv"),
        str(workdir / "pedfile.csv"),
        workdir=str(ref_dir),
        isphysmap=False, refinemap=True,
        nworkers=1, seed=1234, outstem="ref",
    )

    # Parse the reference output with the same parsers (ID-keyed)
    from torchgwas.preprocess.phase_polyorigin import (
        _parse_genoprob, _parse_parentphased,
    )
    ref_origin, _ = _parse_genoprob(
        str(ref_dir / "ref_genoprob.csv"),
        expected_offspring=result_ours.offspring_ids,
        ploidy=4,
    )
    ref_parent, _ = _parse_parentphased(
        str(ref_dir / "ref_parentphased.csv"),
        expected_parents=result_ours.parent_ids,
        max_ploidy=4,
    )
    assert torch.allclose(result_ours.origin_probs, ref_origin, atol=1e-4), \
        "origin_probs diverge from bare polyOrigin() output"
    assert torch.equal(result_ours.parent_phased, ref_parent), \
        "parent_phased diverges from bare polyOrigin() output"
```

- [ ] **Step 2: Run locally (requires Julia + PolyOrigin.jl)**

```bash
pytest tests/test_phase_polyorigin_e2e.py::test_parity_tetraploid_f1 -v
```

If Julia / PolyOrigin are not installed the test skips cleanly with a reason.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Phase 56: Tier 2 parity test (wrapper vs bare polyOrigin)"
```

---

## Task 15 — Tier 2 recovery + calibration

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py`
- Create: `bench/calibrate_polyorigin_recovery.py`

- [ ] **Step 1: Create the calibration helper**

```python
"""Calibrate haplotype-origin recovery rate for Phase 56 Tier 2 tests.

Not picked up by pytest (no 'test_' prefix). Run once at phase sign-off:

    python bench/calibrate_polyorigin_recovery.py

Prints the observed recovery rate at two depths (30x, 8x). Paste the
numbers into tests/test_phase_polyorigin_e2e.py:
    RECOVERY_30X = <value observed>
    RECOVERY_8X  = <value observed>
"""
from __future__ import annotations

import numpy as np
import torch

N_REPS = 10
N_OFFSPRING = 50
N_MARKERS = 200


def _simulate_f1_reads(depth: int, seed: int):
    """Simulate AD counts for a tetraploid F1 + compute posterior probs.

    Not attempting full realism — just enough to get a calibration number.
    """
    # Placeholder for implementer to fill in using updog's own simulator
    # (via torchgwas.preprocess.dosage_call) once Phase 55 and a local
    # updog install are available.
    raise NotImplementedError(
        "Implementer: plug in updog's rgeno + rflexdog-simulated AD, then "
        "run dosage_call.run_updog to derive posterior probs, then feed "
        "them through run_polyorigin with a known ground-truth origin "
        "assignment, then compute haplotype-origin argmax accuracy."
    )


def calibrate(depth: int, seed: int) -> float:
    accuracies = []
    for rep in range(N_REPS):
        # ... run pipeline, compare haplotypes.argmax vs truth ...
        acc = 0.0
        accuracies.append(acc)
    return float(np.mean(accuracies))


if __name__ == "__main__":
    for depth in (30, 8):
        rate = calibrate(depth=depth, seed=depth)
        print(f"depth={depth}x  recovery={rate:.4f}")
```

- [ ] **Step 2: Add recovery tests with hard-coded thresholds (placeholders → replace after calibration run)**

Append to `tests/test_phase_polyorigin_e2e.py`:

```python
# Calibrated by bench/calibrate_polyorigin_recovery.py on <DATE>; see
# docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md
# Section 5.3 for why 2% slack.
RECOVERY_30X = 0.95  # placeholder — replace after calibration
RECOVERY_8X = 0.80   # placeholder — replace after calibration
SAFETY_FLOOR = 0.02


def test_simulated_f1_haplotype_recovery_30x(tmp_path):
    pytest.skip("Calibration numbers placeholder — fill after running bench/calibrate_polyorigin_recovery.py")


def test_simulated_f1_haplotype_recovery_8x(tmp_path):
    pytest.skip("Calibration numbers placeholder — fill after running bench/calibrate_polyorigin_recovery.py")
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase_polyorigin_e2e.py bench/calibrate_polyorigin_recovery.py
git commit -m "Phase 56: Tier 2 recovery test skeletons + calibration helper"
```

- [ ] **Step 4: At phase sign-off — run calibrate, fill numbers, replace skips with real tests**

Run:

```bash
python bench/calibrate_polyorigin_recovery.py
```

Copy the printed numbers into `RECOVERY_30X` / `RECOVERY_8X`, replace the `pytest.skip` bodies with real recovery-rate asserts:

```python
def test_simulated_f1_haplotype_recovery_30x(tmp_path):
    acc = _run_recovery(depth=30, tmp_path=tmp_path)
    assert acc >= RECOVERY_30X - SAFETY_FLOOR


def test_simulated_f1_haplotype_recovery_8x(tmp_path):
    acc = _run_recovery(depth=8, tmp_path=tmp_path)
    assert acc >= RECOVERY_8X - SAFETY_FLOOR
```

(`_run_recovery` should mirror the simulator in the calibration helper.)

Commit:

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Phase 56: Tier 2 — fill calibrated recovery thresholds"
```

---

## Task 16 — Tier 2 GWASpoly potato round-trip

**Files:**
- Modify: `tests/test_phase_polyorigin_e2e.py`

- [ ] **Step 1: Add the round-trip test skeleton**

Append to `tests/test_phase_polyorigin_e2e.py`:

```python
import hashlib
import urllib.request

# Lazy-download cache (gitignored)
_CACHE = Path(__file__).parent / "fixtures" / "_cache"
_CACHE.mkdir(parents=True, exist_ok=True)

# The GWASpoly tetraploid potato F1 dataset URL and SHA-256 checksum
# are resolved at implementation time — the implementer picks a stable
# public mirror and captures the checksum on first download. Once pinned,
# CI jobs rely on this checksum to detect dataset drift.
_POTATO_URL = "PLACEHOLDER — implementer pins before first commit of this test"
_POTATO_SHA256 = "PLACEHOLDER — implementer pins before first commit of this test"


def _ensure_potato_dataset() -> Path:
    local = _CACHE / "gwaspoly_potato_f1.tar.gz"
    if local.exists():
        h = hashlib.sha256(local.read_bytes()).hexdigest()
        if h == _POTATO_SHA256:
            return local
    if _POTATO_URL.startswith("PLACEHOLDER"):
        pytest.skip("GWASpoly potato dataset URL not pinned yet (see test source)")
    urllib.request.urlretrieve(_POTATO_URL, local)
    h = hashlib.sha256(local.read_bytes()).hexdigest()
    if h != _POTATO_SHA256:
        raise AssertionError(
            f"Downloaded potato dataset sha256 mismatch: got {h}, expected {_POTATO_SHA256}."
        )
    return local


def test_gwaspoly_potato_roundtrip(tmp_path):
    """The roadmap's integration gate.

    dosage-call → phase-poly → HaplotypeGWAS (Phase 46). Compare p-values
    against a bare-polyOrigin reference run to verify wrapper fidelity
    end-to-end.
    """
    dataset = _ensure_potato_dataset()
    # Unpack and pipe through the full chain
    # ... implementer fills in based on dataset structure ...
    pytest.skip("Implementer: fill once potato dataset URL/SHA are pinned")
```

- [ ] **Step 2: Commit the skeleton**

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Phase 56: Tier 2 — GWASpoly potato round-trip skeleton"
```

- [ ] **Step 3: At phase sign-off — pin URL + SHA and fill the test body**

Pick a stable public mirror for the GWASpoly potato F1 dataset (likely the published Bourke 2021 potato-flesh-color study companion data, or the synthetic dataset shipped with GWASpoly's R package). Download it once, capture the SHA-256, and substitute into `_POTATO_URL` / `_POTATO_SHA256`. Then replace the `pytest.skip` body with the real pipeline invocation:

```python
def test_gwaspoly_potato_roundtrip(tmp_path):
    dataset = _ensure_potato_dataset()
    # extract ...
    # Run ours:
    #   dosage-call → run_polyorigin → HaplotypeGWAS.scan(Y, G, haplotypes)
    # Run reference:
    #   bare polyOrigin() in Julia on the same genofile/pedfile →
    #   identical HaplotypeGWAS.scan(Y, G, haplotypes_ref)
    # Compare Phase 46 p-values:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(neglog10_pvals_ours, neglog10_pvals_ref)
    assert rho > 0.999
    max_diff = float(np.abs(np.array(neglog10_pvals_ours) - np.array(neglog10_pvals_ref)).max())
    assert max_diff < 0.01
```

Commit:

```bash
git add tests/test_phase_polyorigin_e2e.py
git commit -m "Phase 56: Tier 2 — GWASpoly potato round-trip pinned + enabled"
```

---

## Task 17 — Documentation: getting-started recipe + memory file

**Files:**
- Create: `docs/getting-started/polyploid_phasing.md`
- Create: `~/.claude/projects/C--Users-Sikiru-Documents-GWAS-Expert/memory/project_polyploid_phasing.md`
- Modify: `~/.claude/projects/.../memory/MEMORY.md`

- [ ] **Step 1: Write the getting-started recipe**

Create `docs/getting-started/polyploid_phasing.md`:

```markdown
# Polyploid Phasing (Phase 56)

End-to-end recipe for phasing a connected tetraploid F1 population with
`torchgwas phase-poly`, chaining off `dosage-call`.

## Prerequisites

- A multi-sample VCF with the `AD` (allele-depth) field for parents + offspring.
- A pedigree TSV (`offspring\tparent1\tparent2`, optional `ploidy` column).
- A marker map TSV (`marker\tchrom\tpos_bp`, optional `cm` column).

Install the polyploid-phase extra:

    pip install torchgwas[polyploid-phase]

On first `phase-poly` run you will be prompted to install Julia 1.10
(~300 MB) if it is not already on your machine. Add `--auto-install` to
accept the prompt non-interactively, or pre-install Julia from
https://julialang.org/downloads/ and ensure `julia` is on `PATH`
(alternatively set `TORCHGWAS_JULIA=/abs/path/to/julia`).

## Pipeline

    # 1. Posterior dosage calling from VCF read counts (Phase 55)
    torchgwas dosage-call \
      --vcf       calls.vcf.gz \
      --output    out/dcall \
      --ploidy    4

    # 2. Polyploid phasing of the F1 progeny
    torchgwas phase-poly \
      --probs     out/dcall.probs.pt \
      --pedigree  pedigree.tsv \
      --map       markers.tsv \
      --output    out/phased \
      --ploidy    4

    # 3. Haplotype GWAS (Phase 46)
    python <<'PY'
    import torch
    from torchgwas.models import HaplotypeGWAS
    from torchgwas.preprocess.dosage_uncertainty import expected_dosage

    haps = torch.load('out/phased.haplotypes.pt')          # (n_off, 4, m)
    postdose = torch.load('out/phased.postdose_probs.pt')  # (n_off, m, 5)
    G = expected_dosage(postdose, ploidy=4)                # (n_off, m)
    Y = torch.load('pheno.pt')                             # user-supplied

    scanner = HaplotypeGWAS(method="block", test="f_test", ploidy=4)
    result = scanner.scan(Y=Y, G=G, haplotypes=haps)
    PY

`out/phased.meta.json` lists the exact offspring / parent ordering so
you can align `pheno.pt` to the haplotype tensor. The included
`out/phased.valent_diag.tsv` reports PolyOrigin's per-marker valent
configuration frequencies — useful for screening markers.

## Troubleshooting

| Symptom | Remedy |
| --- | --- |
| `PolyOrigin failed: ...` | Inspect the Julia log in the work directory. Re-run with `--keep-workdir` to preserve inputs for manual re-invocation. |
| `Julia at ... is v1.8.x; requires >= 1.10` | Install a newer Julia via `juliaup` (`juliaup install 1.10 && juliaup default 1.10`) or unset `TORCHGWAS_JULIA`. |
| First run hangs for 1–2 minutes | Normal — Julia JIT + PolyOrigin.jl precompile. Subsequent calls in the same process are fast. |
```

- [ ] **Step 2: Write the per-phase memory file**

Create the memory file at the per-project memory directory (substituting the user's actual path):

```markdown
---
name: Phase 56 — Polyploid Phasing
description: Thin wrapper around PolyOrigin.jl for connected tetraploid / hexaploid F1 polyploid phasing; in-process Julia via juliacall with discover-first bootstrap
type: project
---

Phase 56 ships `torchgwas.preprocess.phase_polyorigin.run_polyorigin`
plus the `torchgwas phase-poly` CLI. Phases connected F1 polyploid
populations (ploidy ∈ {2, 4, 6}) by calling PolyOrigin.jl v1.0.3 through
`juliacall` in-process. Julia is auto-provisioned only if absent and the
user consents. Output is tensor-native (`haplotypes: (n_off, ploidy,
m)`, `origin_probs`, `parent_phased`, `postdose_probs`, `valent_diag`) —
fed directly to `HaplotypeGWAS.scan()`.

**Scope boundaries** (hard constraints from upstream):
- F1 crosses + founder selfings only; multi-generation pedigrees →
  explicit ValueError.
- Ploidies {2, 4, 6}. Unrelated-sample phasing is a separate future
  phase (WhatsHap-polyphase / hapCON queued as follow-ups).

**Chain**: `dosage-call → phase-poly → HaplotypeGWAS` — no VCF round-trip.

**Install extra**: `pip install torchgwas[polyploid-phase]` adds
`juliacall`. Julia itself is discovered first (`TORCHGWAS_JULIA`, PATH,
common install paths) and only downloaded by juliacall (~300 MB) if
missing AND the user consents (CLI: interactive prompt or
`--auto-install`; library: `auto_install_julia=True`).

**Tests**: ~25 Tier 1 (juliacall stubbed) + 6 Tier 2 (real Julia +
PolyOrigin.jl gated). Parity contract: our wrapper vs. bare
`PolyOrigin.polyOrigin()` on identical genofile/pedfile input.
Integration gate: GWASpoly potato F1 round-trip must match reference
Phase 46 p-values at Spearman ρ > 0.999 and max |Δ-log10p| < 0.01.

**Gotchas to flag to future sessions**:
- `PhasingResult.haplotypes` dim 1 is `max_ploidy` with `-1` padding for
  mixed-ploidy rows. Same-ploidy pedigrees (the typical case) have no
  `-1` sentinels.
- `refinemap=True` can reorder markers within a chromosome; the
  returned `variant_ids` reflects the refined order and
  `result.map_refined: bool` flags that.
- The exact PolyOrigin output CSV column schemas were verified against
  a live v1.0.3 run during implementation; if upstream changes the
  schema in a future release, regenerate the canned fixtures under
  `tests/fixtures/phase_polyorigin/`.
```

- [ ] **Step 3: Add pointer to MEMORY.md**

Append to the top-level index file `MEMORY.md`:

```markdown
- [Polyploid Phasing Phase 56](project_polyploid_phasing.md) — PolyOrigin.jl wrapper for connected F1 {2,4,6}; juliacall in-process; tensor-native output to HaplotypeGWAS
```

- [ ] **Step 4: Commit**

```bash
git add docs/getting-started/polyploid_phasing.md
# Memory files live outside the repo in the user's ~/.claude — skip those from git
git commit -m "Phase 56: getting-started recipe"
```

Then separately create/update the memory files (they are outside the repo and are not committed).

---

## Task 18 — Roadmap removal + CLI docs + CLAUDE.md bump

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/cli.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove the Phase 56 entry from `docs/ROADMAP.md`**

Delete lines 138–150 (the `Phase 56 — Polyploid phasing` block), preserving surrounding list structure.

- [ ] **Step 2: Add `phase-poly --help` capture to `docs/cli.md`**

Append:

```
### `torchgwas phase-poly`

```
$ torchgwas phase-poly --help
<paste the actual --help output captured from the CLI>
```
```

- [ ] **Step 3: Bump subcommand count + add example in `CLAUDE.md`**

Add a `phase-poly` example under the "CLI Commands" section alongside the other `phase*` examples. Adjust the subcommand count line — the exact number follows the existing CLAUDE.md convention (user-facing commands, not raw `add_parser` calls). If the current line reads "CLI subcommand count: 36 as of Phase 55", update to "CLI subcommand count: 37 as of Phase 56" (or whatever +1 yields under the current convention).

Example to insert:

```bash
# --- Polyploid F1 phasing (Phase 56) ---
torchgwas phase-poly --probs out/dcall.probs.pt --pedigree ped.tsv \
    --map markers.tsv --output out/phased --ploidy 4
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v --no-header
```

Expected: all Tier 1 tests pass; Tier 2 tests skip unless Julia + PolyOrigin are locally available.

- [ ] **Step 5: Final commit**

```bash
git add docs/ROADMAP.md docs/cli.md CLAUDE.md
git commit -m "Phase 56: ship — roadmap removal + CLI docs + CLAUDE.md update"
```

---

## Self-review notes

**Spec coverage check**:
- §2.1 PhasingResult → Task 2 ✓
- §2.2 run_polyorigin signature → Task 10 ✓
- §2.3 runtime bootstrap → Tasks 7, 8 ✓
- §2.4 pure converters (pedfile + genofile) → Tasks 3, 5 ✓
- §3 data flow (validation, converters, Julia invocation, parsing, persistence) → Tasks 6, 7–8, 9, 10, 11 ✓
- §4.1 env missing → Tasks 7, 8 ✓
- §4.2 input validation → Tasks 3, 4, 5, 6 ✓
- §4.3 JuliaError mapping → Task 10 ✓
- §4.4 output validation → Tasks 9, 10 ✓
- §4.5 dep matrix → Task 1 ✓
- §5.1 Tier 1 tests → Tasks 2–13 ✓ (~25 tests total)
- §5.2 Tier 2 → Tasks 14, 15, 16 ✓
- §5.3 calibration helper → Task 15 ✓
- §5.4 the gate (roadmap removal, docs, memory) → Tasks 17, 18 ✓
- §6 CLI → Task 12 ✓
- §7 non-goals locked in — no action required
- §8 open questions — implementer-flagged in Task 1 (UUID), Task 8 (juliacall env var), Task 9 (CSV layouts), Tasks 14–16 (mixed-ploidy triploid accept test) ✓

**Placeholder scan**: the canned fixture CSVs in Task 9 and the calibration numbers in Task 15 are documented as best-guess-regenerate-at-implementation-time, with explicit instructions for the implementer — not TBD blanks. The GWASpoly potato URL/SHA in Task 16 is explicitly left as a pinning step at phase sign-off, not a vague "add later."

**Type consistency** (signatures repeated across tasks stay consistent): `PhasingResult` field set in Task 2 matches Task 10's `run_polyorigin` construction and Task 11's `_persist_result` read path; `_build_polyorigin_pedfile` signature in Task 3 matches the call site in Task 10; parser signatures in Task 9 match invocations in Task 10.

---

Plan complete and saved to `docs/superpowers/plans/2026-04-23-phase-56-polyploid-phasing.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
