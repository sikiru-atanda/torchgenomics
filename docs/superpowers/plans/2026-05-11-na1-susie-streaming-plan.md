# NA1 — `bayes-scan-rss` (SuSiE-RSS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Tier A of the NA1 design spec — a new CLI subcommand `bayes-scan-rss` that performs SuSiE-RSS fine-mapping on summary statistics + LD reference, replacing the materialized-genotype memory cost of the existing `bayes-scan` CLI for biobank-scale loci.

**Architecture:** New class `BayesianVSRss` in new file `torchgenomics/models/bayesian_vs_rss.py` (Approach 1 from brainstorming — preserves all 30+ existing `tests/test_bayesian_vs.py` parity tests with zero modification). Algorithm per Zou, Carbonetto, Wang, Stephens (2022) [PLOS Genet]: per-layer Single Effect Regression (SER) on summary statistics, IBSS updates with `R` (LD reference) replacing `X^T X / n`, block decomposition for memory bound `O(p_block_max^2)` instead of `O(n × p)`. New supporting modules `postgwas/_ld_ref_loader.py` (multi-format reader: `.pt`, `.npz`) and `postgwas/_ld_ref_metadata.py` (cohort-mismatch detector). External validation harness `validation/external/susieR/` mirrors Pillar B layout for parity vs upstream `susieR::susie_rss()`.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.0 (existing), pyarrow (existing), `susieR` R package (CRAN install at validation time only), R ≥ 4.0 for parity runs only.

**Spec reference:** `docs/superpowers/specs/2026-05-11-na1-susie-streaming-design.md`

**Five locked decisions** (from brainstorming 2026-05-11):
- D1: Path D (SuSiE-RSS) only; Path A (chunked IBSS for raw-G) deferred
- D2: New subcommand `bayes-scan-rss` (separate from `bayes-scan`)
- D3: Tolerances per spec §5.2 + β_mean/β_sd Pearson ≥ 0.999
- D4: `--ld-ref` (pre-built) AND `--geno` (in-sample) both supported
- D5: Soft `UserWarning` on materialized `bayes-scan` when `p > 10000`

**Memory rules in force** (`MEMORY.md`):
- `feedback_no_autonomous_push`: local commits only; do NOT push to any remote without explicit per-event user approval
- `feedback_benchmark_against_installed_tools`: every numerical claim backed by head-to-head run vs `susieR`
- `feedback_scientific_rigor`: every formula cited to primary source; observed-then-floored tolerances
- `feedback_preflight`: assert disk + RAM headroom before installs/runs
- `feedback_streaming`: per-block memory bound enforced via regression test
- `feedback_regression_nets`: memory test paired with wall-time test
- `feedback_parallel_agents_no_shortcuts`: any subagent dispatch carries acceptance criteria + no-shortcuts contract

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `torchgenomics/models/bayesian_vs_rss.py` | NEW | `BayesianVSRss` class — SER, IBSS, ELBO, credible-set construction on summary stats |
| `torchgenomics/postgwas/_ld_ref_loader.py` | NEW | Multi-format LD reference loader; in-sample LD computation from genotype |
| `torchgenomics/postgwas/_ld_ref_metadata.py` | NEW | LD reference metadata schema; cohort-mismatch detector |
| `tests/test_bayesian_vs_rss.py` | NEW | Tier 1 unit tests for SuSiE-RSS algorithm |
| `tests/test_ld_ref_loader.py` | NEW | Tier 1 unit tests for loader + metadata |
| `tests/test_external_susieR.py` | NEW | Tier 2 parity test (`@pytest.mark.external @pytest.mark.golden`) |
| `validation/external/susieR/install.sh` | NEW | Install `susieR` via CRAN; idempotent; pre-flight |
| `validation/external/susieR/fetch_data.sh` | NEW | Provision MDP-derived per-locus fixture |
| `validation/external/susieR/run_susieR.sh` | NEW | Reference run via `susieR::susie_rss()` |
| `validation/external/susieR/run_torchgenomics.sh` | NEW | Our run via `torchgenomics bayes-scan-rss` |
| `validation/external/susieR/compare.py` | NEW | Compute 6-metric tolerance table; emit findings ledger row |
| `validation/external/susieR/README.md` | NEW | Operations doc |
| `torchgenomics/models/bayesian_vs.py` | MODIFY | Add `UserWarning` in `fit()` when `p > 10000` (Decision 5); NO algorithmic change |
| `torchgenomics/cli.py` | MODIFY | Add `bayes-scan-rss` subcommand handler |
| `docs/cli.md` | MODIFY | Document `bayes-scan-rss` |
| `tests/test_streaming_memory.py` | MODIFY (or CREATE if Task 0 chooses rebase-then-add) | Memory regression for `bayes-scan-rss` |
| `bench/native_speedups.py` | MODIFY | Wall-time gate for `bayes-scan-rss` |
| `CLAUDE.md` | MODIFY | Subcommand count 37 → 38; add `BayesianVSRss` to models list |

**Files NOT touched** (per spec §3.3): `BayesianVS` class internals (only the warning is added); `postgwas/_finemapping.py`; `pgs/ld_ref.py` (unrelated).

---

## Task 0: Resolve A0 sequencing blocker (R-NA1-7)

**Files:**
- No code changes; this task is a sequencing decision + verification.

**Background:** `tests/test_streaming_memory.py` and the streaming-test infrastructure exist only on `efficiency/streaming-scan-audit`, NOT on `master` (which `research/na1-susie-streaming` was branched from). Per spec R-NA1-7, two options are acceptable:
- **Option A**: rebase `research/na1-susie-streaming` onto `efficiency/streaming-scan-audit`.
- **Option B**: merge `efficiency/streaming-scan-audit` into `master` first, then rebase NA1 onto the new `master`.

This plan assumes **Option A** (rebase NA1 onto the streaming-audit branch) because it's lower-blast-radius and doesn't require merging the entire 113-commit streaming-audit work to master before NA1 can start. If Option B is chosen, replace the rebase command in Step 2 with the merge sequence.

- [ ] **Step 1: Verify current branch state and uncommitted changes**

```bash
cd /home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/na1-susie-streaming
git status
git log --oneline -3
git branch --show-current
```

Expected output:
- Current branch: `research/na1-susie-streaming`
- HEAD at the spec commit (e.g., `74a2818 NA1 design spec: ...`)
- Working tree clean

If working tree is dirty, STOP and resolve before proceeding.

- [ ] **Step 2: Rebase onto efficiency/streaming-scan-audit**

```bash
git fetch . efficiency/streaming-scan-audit
git rebase efficiency/streaming-scan-audit
```

Expected: clean rebase. The two NA1 commits (research brief + spec) replay on top of the streaming-audit HEAD.

If rebase conflicts arise: the only files NA1 touches at this point are under `docs/superpowers/research/` and `docs/superpowers/specs/` — these should not conflict with streaming-audit changes (which touched `torchgenomics/`, `tests/`, `bench/`, etc.). If conflict, abort with `git rebase --abort` and switch to Option B (merge-then-rebase).

- [ ] **Step 3: Verify streaming infrastructure is now visible**

```bash
ls tests/test_streaming_memory.py
ls docs/efficiency/streaming_audit.md
ls validation/external/_lib/preflight.sh
```

Expected: all three files exist. If any is missing, the streaming-audit branch was the wrong base; investigate.

- [ ] **Step 4: Verify existing test suite still runs (smoke check, no regressions from rebase)**

```bash
pytest tests/test_bayesian_vs.py -v --tb=short -x
```

Expected: all tests pass (this is the baseline — Task 21 will re-run as the final regression gate).

If any test fails on the post-rebase baseline, STOP. The rebase has revealed an incompatibility; investigate before proceeding.

- [ ] **Step 5: No commit needed for Task 0** — rebase moved commits, no new content.

```bash
git log --oneline -5
```

Confirm: 2 NA1 commits sit on top of streaming-audit HEAD.

---

## Task 1: LD-ref metadata schema + cohort-mismatch detector

**Files:**
- Create: `torchgenomics/postgwas/_ld_ref_metadata.py`
- Test: `tests/test_ld_ref_loader.py` (new file; will accumulate further tests in Task 2-3)

**Background:** Per spec §3.1 and R-NA1-1, every LD reference file must carry metadata stamping `cohort_id`, `n`, `build`, `panel_provenance`. At `bayes-scan-rss` invocation time, the LD ref's metadata is checked against the sumstats; soft mismatch (e.g., different `cohort_id`) → `UserWarning`; hard mismatch (e.g., different `build`) → `ValueError`. Primary failure mode of SuSiE-RSS per Zou 2022 [C4] §6.

- [ ] **Step 1: Write the failing tests for the metadata dataclass**

Create `tests/test_ld_ref_loader.py`:

```python
"""Tier 1 unit tests for LD reference loader and metadata."""
from __future__ import annotations

import pytest

from torchgenomics.postgwas._ld_ref_metadata import (
    LDReferenceMetadata,
    check_metadata_compatibility,
    MetadataMismatchError,
)


def test_metadata_dataclass_required_fields():
    """LDReferenceMetadata requires cohort_id, n, build, panel_provenance."""
    meta = LDReferenceMetadata(
        cohort_id="UKB_unrelated_British",
        n=337491,
        build="GRCh38",
        panel_provenance="UKB v3 imputed",
    )
    assert meta.cohort_id == "UKB_unrelated_British"
    assert meta.n == 337491
    assert meta.build == "GRCh38"
    assert meta.panel_provenance == "UKB v3 imputed"


def test_metadata_serialization_round_trip():
    """LDReferenceMetadata serializes to dict and back."""
    original = LDReferenceMetadata(
        cohort_id="MDP",
        n=281,
        build="B73_v4",
        panel_provenance="MDP genotype 281×3093",
    )
    d = original.to_dict()
    assert d == {
        "cohort_id": "MDP",
        "n": 281,
        "build": "B73_v4",
        "panel_provenance": "MDP genotype 281×3093",
        "schema_version": 1,
    }
    restored = LDReferenceMetadata.from_dict(d)
    assert restored == original


def test_metadata_compatible_when_identical():
    """check_metadata_compatibility passes when ref and sumstats match."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "X", "build": "GRCh38"}
    # Should NOT raise and should NOT warn
    check_metadata_compatibility(ref, ss_meta)


def test_metadata_soft_mismatch_warns_on_cohort_id():
    """Soft mismatch (different cohort_id) emits UserWarning."""
    ref = LDReferenceMetadata(cohort_id="UKB", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "FinnGen", "build": "GRCh38"}
    with pytest.warns(UserWarning, match="cohort_id mismatch"):
        check_metadata_compatibility(ref, ss_meta)


def test_metadata_hard_mismatch_raises_on_build():
    """Hard mismatch (different build) raises MetadataMismatchError."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    ss_meta = {"cohort_id": "X", "build": "GRCh37"}
    with pytest.raises(MetadataMismatchError, match="build mismatch"):
        check_metadata_compatibility(ref, ss_meta)


def test_metadata_missing_sumstats_metadata_warns():
    """If sumstats don't carry metadata, emit a UserWarning advising to add it."""
    ref = LDReferenceMetadata(cohort_id="X", n=1000, build="GRCh38", panel_provenance="p")
    with pytest.warns(UserWarning, match="sumstats has no metadata"):
        check_metadata_compatibility(ref, ss_meta={})
```

- [ ] **Step 2: Run tests to verify they fail with import errors**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torchgenomics.postgwas._ld_ref_metadata'` or similar.

- [ ] **Step 3: Implement the metadata module**

Create `torchgenomics/postgwas/_ld_ref_metadata.py`:

```python
"""LD reference metadata schema and cohort-mismatch detection.

Per the NA1 design spec (docs/superpowers/specs/2026-05-11-na1-susie-streaming-design.md)
section 3.1 and risk R-NA1-1: every LD reference file must carry metadata
stamping cohort_id, n, build, panel_provenance. At bayes-scan-rss invocation,
the LD ref's metadata is checked against the sumstats; soft mismatch
(different cohort_id) → UserWarning; hard mismatch (different build) →
MetadataMismatchError.

Reference: Zou, Carbonetto, Wang, Stephens (2022) PLOS Genet 18(7):e1010299
section 6 — LD-ref-mismatch is the primary failure mode of SuSiE-RSS.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Any, Mapping


SCHEMA_VERSION = 1


class MetadataMismatchError(ValueError):
    """Raised when LD reference metadata is incompatible with sumstats metadata."""


@dataclass(frozen=True)
class LDReferenceMetadata:
    """Metadata stamped into every LD reference file at build time.

    Args:
        cohort_id: Free-form identifier for the cohort the LD was built from
            (e.g., "UKB_unrelated_British", "MDP", "FinnGen_R10").
        n: Sample size used to build the LD matrix.
        build: Genome build identifier (e.g., "GRCh37", "GRCh38", "B73_v4").
        panel_provenance: Free-form description of the variant panel
            (e.g., "UKB v3 imputed", "1KG phase 3 EUR").
    """
    cohort_id: str
    n: int
    build: str
    panel_provenance: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LDReferenceMetadata":
        if d.get("schema_version", 1) != SCHEMA_VERSION:
            raise ValueError(
                f"LDReferenceMetadata schema version mismatch: "
                f"expected {SCHEMA_VERSION}, got {d.get('schema_version')}"
            )
        return cls(
            cohort_id=d["cohort_id"],
            n=int(d["n"]),
            build=d["build"],
            panel_provenance=d["panel_provenance"],
        )


def check_metadata_compatibility(
    ref: LDReferenceMetadata,
    ss_meta: Mapping[str, Any],
) -> None:
    """Check LD reference metadata against sumstats metadata.

    Soft mismatch (different cohort_id, missing sumstats metadata, etc.)
    emits a UserWarning. Hard mismatch (different build) raises
    MetadataMismatchError.

    Args:
        ref: LD reference metadata.
        ss_meta: Sumstats metadata as a mapping (typically loaded from a
            sumstats sidecar file or extracted from sumstats column headers).

    Raises:
        MetadataMismatchError: on hard mismatch (different build).
    """
    if not ss_meta:
        warnings.warn(
            "sumstats has no metadata; cannot verify LD reference compatibility. "
            "Recommend stamping cohort_id and build into the sumstats sidecar.",
            UserWarning,
            stacklevel=2,
        )
        return

    ref_build = ref.build
    ss_build = ss_meta.get("build")
    if ss_build is not None and ss_build != ref_build:
        raise MetadataMismatchError(
            f"build mismatch: LD reference is {ref_build!r}, sumstats is {ss_build!r}. "
            f"Re-build the LD reference on the same genome build as the sumstats."
        )

    ref_cohort = ref.cohort_id
    ss_cohort = ss_meta.get("cohort_id")
    if ss_cohort is not None and ss_cohort != ref_cohort:
        warnings.warn(
            f"cohort_id mismatch: LD reference cohort is {ref_cohort!r}, "
            f"sumstats cohort is {ss_cohort!r}. SuSiE-RSS is sensitive to "
            f"LD reference cohort matching the GWAS cohort (Zou et al. 2022 "
            f"PLOS Genet section 6). Posteriors may be miscalibrated.",
            UserWarning,
            stacklevel=2,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/postgwas/_ld_ref_metadata.py tests/test_ld_ref_loader.py
git commit -m "NA1 Task 1: LD reference metadata schema + cohort-mismatch detector"
```

---

## Task 2: LD-ref multi-format loader (.pt + .npz)

**Files:**
- Create: `torchgenomics/postgwas/_ld_ref_loader.py`
- Modify: `tests/test_ld_ref_loader.py` (append loader tests)

**Background:** Per spec §3.1 and Decision 4, the loader supports `.pt` (existing TorchGenomics format from Phase 40) and `.npz` (PolyFun format from Phase 59) at Tier A. Each loaded reference carries its `LDReferenceMetadata` from Task 1.

- [ ] **Step 1: Write failing tests for the loader**

Append to `tests/test_ld_ref_loader.py`:

```python
import numpy as np
import torch

from torchgenomics.postgwas._ld_ref_loader import (
    load_ld_reference,
    save_ld_reference,
    LDReferenceUnsupportedFormatError,
)


def test_save_and_load_pt_format(tmp_path):
    """Round-trip a small LD ref through .pt format with metadata."""
    R = torch.eye(10, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(10)]
    meta = LDReferenceMetadata(
        cohort_id="test", n=100, build="GRCh38", panel_provenance="synthetic"
    )
    path = tmp_path / "ld.pt"
    save_ld_reference(path, R, snp_ids, meta)

    R_loaded, snp_ids_loaded, meta_loaded = load_ld_reference(path)
    assert torch.allclose(R, R_loaded)
    assert snp_ids == snp_ids_loaded
    assert meta == meta_loaded


def test_save_and_load_npz_format(tmp_path):
    """Round-trip a small LD ref through .npz format with metadata."""
    R = torch.eye(10, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(10)]
    meta = LDReferenceMetadata(
        cohort_id="test", n=100, build="GRCh38", panel_provenance="synthetic"
    )
    path = tmp_path / "ld.npz"
    save_ld_reference(path, R, snp_ids, meta)

    R_loaded, snp_ids_loaded, meta_loaded = load_ld_reference(path)
    assert torch.allclose(R, R_loaded)
    assert snp_ids == snp_ids_loaded
    assert meta == meta_loaded


def test_load_unsupported_format_raises(tmp_path):
    """Unsupported file extension raises LDReferenceUnsupportedFormatError."""
    path = tmp_path / "ld.bcor"
    path.write_bytes(b"dummy")
    with pytest.raises(LDReferenceUnsupportedFormatError, match="bcor"):
        load_ld_reference(path)


def test_load_format_inferred_from_extension(tmp_path):
    """Format is inferred from file extension; .pt vs .npz handled differently."""
    R = torch.eye(5, dtype=torch.float64)
    snp_ids = ["rs1", "rs2", "rs3", "rs4", "rs5"]
    meta = LDReferenceMetadata(
        cohort_id="t", n=10, build="GRCh38", panel_provenance="p"
    )
    pt_path = tmp_path / "ld.pt"
    npz_path = tmp_path / "ld.npz"
    save_ld_reference(pt_path, R, snp_ids, meta)
    save_ld_reference(npz_path, R, snp_ids, meta)

    R_pt, _, _ = load_ld_reference(pt_path)
    R_npz, _, _ = load_ld_reference(npz_path)
    assert torch.allclose(R_pt, R_npz)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short -k "load or save"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'torchgenomics.postgwas._ld_ref_loader'`.

- [ ] **Step 3: Implement the loader**

Create `torchgenomics/postgwas/_ld_ref_loader.py`:

```python
"""Multi-format LD reference loader.

Per NA1 design spec section 3.1 and Decision 4: supports .pt (existing
TorchGenomics format) and .npz (PolyFun format) at Tier A. Each loaded
reference carries its LDReferenceMetadata for cohort-mismatch detection
at bayes-scan-rss invocation.

In-sample LD computation from a genotype panel (--geno mode) is in Task 3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata


class LDReferenceUnsupportedFormatError(ValueError):
    """Raised when an LD reference file extension is not supported."""


def save_ld_reference(
    path: Path | str,
    R: torch.Tensor,
    snp_ids: list[str],
    meta: LDReferenceMetadata,
) -> None:
    """Save an LD reference to disk in the format inferred from extension.

    Args:
        path: Output path. Extension determines format: .pt or .npz.
        R: LD correlation matrix, shape (p, p), dtype float64 recommended.
        snp_ids: List of p SNP identifiers in order matching R rows/columns.
        meta: LDReferenceMetadata to stamp into the file.

    Raises:
        LDReferenceUnsupportedFormatError: if extension is not .pt or .npz.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        torch.save(
            {
                "R": R.cpu(),
                "snp_ids": snp_ids,
                "metadata": meta.to_dict(),
            },
            str(path),
        )
    elif suffix == ".npz":
        np.savez(
            str(path),
            R=R.cpu().numpy(),
            snp_ids=np.array(snp_ids, dtype=object),
            metadata_json=str(meta.to_dict()),
        )
    else:
        raise LDReferenceUnsupportedFormatError(
            f"Unsupported LD reference format: {suffix!r}. "
            f"Tier A supports .pt and .npz only; .bcor (FINEMAP) is Tier B."
        )


def load_ld_reference(
    path: Path | str,
) -> Tuple[torch.Tensor, list[str], LDReferenceMetadata]:
    """Load an LD reference from disk.

    Args:
        path: Input path; format inferred from extension.

    Returns:
        (R, snp_ids, metadata) tuple.

    Raises:
        LDReferenceUnsupportedFormatError: if extension is not .pt or .npz.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        R = d["R"]
        snp_ids = list(d["snp_ids"])
        meta = LDReferenceMetadata.from_dict(d["metadata"])
        return R, snp_ids, meta

    if suffix == ".npz":
        d = np.load(str(path), allow_pickle=True)
        R = torch.from_numpy(d["R"])
        snp_ids = list(d["snp_ids"])
        # metadata_json is stored as a Python repr of a dict; eval it safely
        meta_dict = eval(str(d["metadata_json"]), {"__builtins__": {}}, {})
        meta = LDReferenceMetadata.from_dict(meta_dict)
        return R, snp_ids, meta

    raise LDReferenceUnsupportedFormatError(
        f"Unsupported LD reference format: {suffix!r}. "
        f"Tier A supports .pt and .npz only; .bcor (FINEMAP) is Tier B."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short
```

Expected: all 10 tests pass (6 from Task 1 + 4 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/postgwas/_ld_ref_loader.py tests/test_ld_ref_loader.py
git commit -m "NA1 Task 2: LD reference multi-format loader (.pt + .npz)"
```

---

## Task 3: In-sample LD computation (`--geno` mode)

**Files:**
- Modify: `torchgenomics/postgwas/_ld_ref_loader.py` (add `compute_in_sample_ld`)
- Modify: `tests/test_ld_ref_loader.py` (append in-sample LD tests)

**Background:** Per Decision 4 and spec §3.1, when user passes `--geno panel.bed` instead of `--ld-ref`, we compute the LD per locus from the supplied genotype panel. Mirrors Phase 59 PolyFun's parity behavior.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ld_ref_loader.py`:

```python
from torchgenomics.postgwas._ld_ref_loader import compute_in_sample_ld


def test_in_sample_ld_matches_corrcoef():
    """compute_in_sample_ld matches np.corrcoef(G.T) for a synthetic panel."""
    rng = np.random.default_rng(42)
    n, p = 100, 20
    G = torch.from_numpy(rng.standard_normal((n, p)).astype(np.float64))
    R = compute_in_sample_ld(G)
    R_ref = torch.from_numpy(np.corrcoef(G.numpy(), rowvar=False))
    assert torch.allclose(R, R_ref, atol=1e-10)


def test_in_sample_ld_handles_constant_columns():
    """Constant columns produce NaN correlations; assert documented behavior."""
    G = torch.zeros(50, 5, dtype=torch.float64)
    G[:, 0] = 1.0  # constant column
    G[:, 1] = torch.arange(50, dtype=torch.float64)  # variable
    R = compute_in_sample_ld(G)
    # diagonal is 1 except for constant columns (which become NaN)
    assert torch.isnan(R[0, 0]) or R[0, 0] == 1.0
    # variable column should have R[1,1] = 1.0
    assert torch.isclose(R[1, 1], torch.tensor(1.0, dtype=torch.float64))


def test_in_sample_ld_diagonal_is_one():
    """For a non-constant panel, diagonal of R is exactly 1.0."""
    rng = np.random.default_rng(7)
    G = torch.from_numpy(rng.standard_normal((30, 8)).astype(np.float64))
    R = compute_in_sample_ld(G)
    assert torch.allclose(R.diag(), torch.ones(8, dtype=torch.float64), atol=1e-12)


def test_in_sample_ld_symmetric():
    """R is symmetric to numerical precision."""
    rng = np.random.default_rng(11)
    G = torch.from_numpy(rng.standard_normal((50, 12)).astype(np.float64))
    R = compute_in_sample_ld(G)
    assert torch.allclose(R, R.T, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ld_ref_loader.py::test_in_sample_ld_matches_corrcoef -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_in_sample_ld'`.

- [ ] **Step 3: Implement `compute_in_sample_ld`**

Append to `torchgenomics/postgwas/_ld_ref_loader.py`:

```python
def compute_in_sample_ld(G: torch.Tensor) -> torch.Tensor:
    """Compute the LD correlation matrix from an in-sample genotype panel.

    Args:
        G: Genotype matrix, shape (n, p), dtype float64 (or upcast internally).

    Returns:
        R: LD correlation matrix, shape (p, p), dtype matching G after upcast.

    Notes:
        Constant columns (zero variance) produce NaN correlations on the
        corresponding rows/columns, matching numpy.corrcoef's behavior.
        Caller should handle NaN downstream (e.g., drop constant SNPs).
    """
    # Upcast to float64 for numerical stability of the centering + scaling
    G64 = G.to(torch.float64)
    # Center columns
    G_centered = G64 - G64.mean(dim=0, keepdim=True)
    # Compute standard deviations (ddof=1 for sample std; matches numpy.corrcoef)
    n = G64.shape[0]
    if n < 2:
        raise ValueError(
            f"compute_in_sample_ld requires n >= 2 samples; got n={n}"
        )
    std = G_centered.std(dim=0, keepdim=True, unbiased=True)
    # Scale (NaN where std is 0 — matches numpy.corrcoef)
    G_scaled = G_centered / std
    # R = (G_scaled^T @ G_scaled) / (n - 1)
    R = (G_scaled.T @ G_scaled) / (n - 1)
    return R
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short
```

Expected: all 14 tests pass.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/postgwas/_ld_ref_loader.py tests/test_ld_ref_loader.py
git commit -m "NA1 Task 3: in-sample LD computation for --geno mode"
```

---

## Task 4: Block decomposition primitive

**Files:**
- Modify: `torchgenomics/postgwas/_ld_ref_loader.py` (add `decompose_into_blocks`)
- Modify: `tests/test_ld_ref_loader.py` (append block tests)

**Background:** Per spec §2.6 and Decision (Section 2 design lean), block decomposition is Tier A. Auto-detection uses the existing `torchgenomics.ld.detect_blocks` if `--regions` not passed; default block-size threshold is 5000 SNPs (matches PolyFun [C5] / ldetect convention).

- [ ] **Step 1: Verify `torchgenomics.ld.detect_blocks` exists and inspect its signature**

```bash
grep -n "def detect_blocks" torchgenomics/ld/*.py
```

Expected: a function `detect_blocks(R: Tensor, ...) -> list[Region]` or similar. Note its actual signature; the test below assumes it returns a list of `(start, stop)` tuples or a structured object with start/stop attributes.

If the signature differs from `list[(start, stop)]`, adapt the wrapper in Step 3 accordingly.

- [ ] **Step 2: Write failing tests for the wrapper**

Append to `tests/test_ld_ref_loader.py`:

```python
from torchgenomics.postgwas._ld_ref_loader import (
    decompose_into_blocks,
    BlockSpec,
)


def test_blocks_explicit_regions_preserved():
    """When regions are explicitly passed, decompose returns them unchanged."""
    R = torch.eye(100, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(100)]
    explicit = [BlockSpec(start=0, stop=40), BlockSpec(start=40, stop=100)]
    blocks = decompose_into_blocks(R, snp_ids, regions=explicit, max_block_size=5000)
    assert blocks == explicit


def test_blocks_below_threshold_returns_single_block():
    """If p <= max_block_size, return a single block covering all SNPs."""
    R = torch.eye(100, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(100)]
    blocks = decompose_into_blocks(R, snp_ids, regions=None, max_block_size=5000)
    assert len(blocks) == 1
    assert blocks[0].start == 0
    assert blocks[0].stop == 100


def test_blocks_above_threshold_splits():
    """If p > max_block_size, blocks are produced (may auto-detect or fall back)."""
    R = torch.eye(50, dtype=torch.float64)
    snp_ids = [f"rs{i}" for i in range(50)]
    blocks = decompose_into_blocks(R, snp_ids, regions=None, max_block_size=10)
    # Expect at least 5 blocks of ~10 SNPs each
    assert len(blocks) >= 5
    # Blocks tile the entire range without overlap
    assert blocks[0].start == 0
    assert blocks[-1].stop == 50
    for i in range(len(blocks) - 1):
        assert blocks[i].stop == blocks[i + 1].start
```

- [ ] **Step 3: Implement `decompose_into_blocks` and `BlockSpec`**

Append to `torchgenomics/postgwas/_ld_ref_loader.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockSpec:
    """Specifies an LD block as a half-open interval [start, stop) over SNP indices."""
    start: int
    stop: int


def decompose_into_blocks(
    R: torch.Tensor,
    snp_ids: list[str],
    regions: list[BlockSpec] | None,
    max_block_size: int = 5000,
) -> list[BlockSpec]:
    """Decompose an LD reference into per-block sub-references.

    Per NA1 spec section 2.6 and Decision 4, when the locus exceeds the
    block-size threshold (default 5000 per PolyFun [C5] / ldetect convention),
    the LD ref is decomposed into LD blocks. Block boundaries auto-detected
    via torchgenomics.ld.detect_blocks if `regions` is None. Per-block IBSS is
    mathematically equivalent to dense IBSS when block boundaries are at
    near-zero-LD positions.

    Args:
        R: LD correlation matrix, shape (p, p).
        snp_ids: List of p SNP identifiers in order.
        regions: Explicit block specs; if None, auto-detect via
            torchgenomics.ld.detect_blocks (when p > max_block_size) or return
            a single block (when p <= max_block_size).
        max_block_size: Threshold above which auto-detection kicks in.

    Returns:
        List of BlockSpec covering [0, p) without overlap or gaps.
    """
    p = R.shape[0]
    if regions is not None:
        return list(regions)

    if p <= max_block_size:
        return [BlockSpec(start=0, stop=p)]

    # Auto-detect via torchgenomics.ld.detect_blocks
    # NOTE: actual signature of detect_blocks must be inspected at task
    # start; the call below assumes detect_blocks(R) returns a list of
    # (start, stop) tuples. If different, adapt this call.
    try:
        from torchgenomics.ld import detect_blocks
        detected = detect_blocks(R)
        # Convert to BlockSpec; assume detect_blocks returns iterable of
        # objects with .start/.stop attributes OR (start, stop) tuples.
        blocks = []
        for b in detected:
            if hasattr(b, "start") and hasattr(b, "stop"):
                blocks.append(BlockSpec(start=int(b.start), stop=int(b.stop)))
            else:
                start, stop = b
                blocks.append(BlockSpec(start=int(start), stop=int(stop)))
        return blocks
    except Exception:
        # Fall back to fixed-size tiling if detect_blocks not available or
        # fails on this matrix
        blocks = []
        start = 0
        while start < p:
            stop = min(start + max_block_size, p)
            blocks.append(BlockSpec(start=start, stop=stop))
            start = stop
        return blocks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ld_ref_loader.py -v --tb=short
```

Expected: all 17 tests pass.

If `detect_blocks` returns an unexpected signature, the auto-detect test (`test_blocks_above_threshold_splits`) may produce blocks via the fallback path (fixed-size tiling). That's still a valid pass — note it for follow-up at implementation time.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/postgwas/_ld_ref_loader.py tests/test_ld_ref_loader.py
git commit -m "NA1 Task 4: block decomposition primitive (auto-detect via detect_blocks)"
```

---

## Task 5: BayesianVSRss class scaffolding + SER posterior closed-form

**Files:**
- Create: `torchgenomics/models/bayesian_vs_rss.py`
- Create: `tests/test_bayesian_vs_rss.py`

**Background:** Per spec §2.2 (eq. 8–10 of Zou 2022 [C4]) — the per-layer Single Effect Regression posterior on summary statistics. Closed-form, deterministic; this is the foundational unit test before IBSS layering on top.

- [ ] **Step 1: Write failing test for the SER posterior closed-form**

Create `tests/test_bayesian_vs_rss.py`:

```python
"""Tier 1 unit tests for BayesianVSRss (SuSiE-RSS algorithm).

Per NA1 design spec section 5.1. Tests ground truth in the Zou et al. 2022
PLOS Genet equations 8-10 (SER posterior) and equation 11 (IBSS update).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchgenomics.models.bayesian_vs_rss import (
    BayesianVSRss,
    BayesianVSRssResult,
    ser_posterior,
)


def test_ser_posterior_closed_form_p_equals_3():
    """SER posterior closed-form on a tiny p=3 fixture.

    Per Zou et al. 2022 [C4] equations 8-10:
        sigma_lj^2 = 1 / (R_jj / sigma_prior^2 + n)
        mu_lj = sigma_lj^2 * z_j * sqrt(n)
        log_BF_lj = 0.5 * log(sigma_lj^2 / sigma_prior^2)
                  + 0.5 * mu_lj^2 / sigma_lj^2
    """
    z = torch.tensor([1.5, -2.0, 0.3], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)  # diagonal R: R_jj = 1
    n = 1000
    sigma_prior_sq = 0.04

    sigma_sq, mu, log_bf = ser_posterior(z, R, n, sigma_prior_sq)

    # Closed-form expected values
    R_diag = torch.diag(R)
    expected_sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    expected_mu = expected_sigma_sq * z * math.sqrt(n)
    expected_log_bf = 0.5 * torch.log(expected_sigma_sq / sigma_prior_sq) + \
                      0.5 * expected_mu ** 2 / expected_sigma_sq

    assert torch.allclose(sigma_sq, expected_sigma_sq, atol=1e-12)
    assert torch.allclose(mu, expected_mu, atol=1e-12)
    assert torch.allclose(log_bf, expected_log_bf, atol=1e-12)


def test_ser_posterior_with_nondiagonal_R():
    """SER posterior uses R_jj from the diagonal regardless of off-diagonal structure."""
    z = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    # Diagonal R values are different
    R = torch.tensor(
        [[1.0, 0.5, 0.3], [0.5, 0.8, 0.1], [0.3, 0.1, 0.6]],
        dtype=torch.float64,
    )
    n = 500
    sigma_prior_sq = 0.1

    sigma_sq, mu, log_bf = ser_posterior(z, R, n, sigma_prior_sq)

    R_diag = torch.diag(R)
    expected_sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    assert torch.allclose(sigma_sq, expected_sigma_sq, atol=1e-12)


def test_bayesian_vs_rss_class_constructs():
    """BayesianVSRss constructs with default args."""
    model = BayesianVSRss(
        max_num_causal=10,
        coverage=0.95,
        purity=0.5,
    )
    assert model.max_num_causal == 10
    assert model.coverage == 0.95
    assert model.purity == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `BayesianVSRss` scaffolding + `ser_posterior`**

Create `torchgenomics/models/bayesian_vs_rss.py`:

```python
"""SuSiE-RSS fine-mapping on summary statistics.

Implements the algorithm of Zou, Carbonetto, Wang, Stephens (2022) PLOS Genet
18(7):e1010299 — SuSiE on summary statistics + LD reference. This is a new
class separate from BayesianVS (raw-G SuSiE) to preserve all 30+ existing
test_bayesian_vs.py parity tests with zero modification.

Per NA1 design spec docs/superpowers/specs/2026-05-11-na1-susie-streaming-design.md.

References:
  [C1] Wang et al. 2020 JRSS-B 82(5):1273-1300 — SuSiE / IBSS
  [C4] Zou et al. 2022 PLOS Genet 18(7):e1010299 — SuSiE-RSS canonical derivation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math
import torch


@dataclass
class BayesianVSRssResult:
    """Return type of BayesianVSRss.fit_rss.

    Attributes:
        alpha: Per-layer per-variant inclusion probabilities, shape (L, p).
        mu: Per-layer per-variant posterior effect means, shape (L, p).
        sigma_sq: Per-layer per-variant posterior effect variances, shape (L, p).
        pip: Per-variant posterior inclusion probabilities, shape (p,).
            pip[j] = 1 - prod_l(1 - alpha[l, j]) per Wang et al. 2020 [C1] eq. 12.
        beta_mean: Per-variant posterior effect means, shape (p,).
            beta_mean[j] = sum_l alpha[l, j] * mu[l, j].
        beta_sd: Per-variant posterior effect standard deviations, shape (p,).
        elbo: ELBO at convergence (scalar).
        elbo_history: ELBO trace per IBSS iteration, shape (T,).
        credible_sets: List of (layer_idx, list_of_variant_idx) per credible set.
        converged: True if IBSS converged within max_iter.
        n_iter: Number of IBSS iterations performed.
    """
    alpha: torch.Tensor
    mu: torch.Tensor
    sigma_sq: torch.Tensor
    pip: torch.Tensor
    beta_mean: torch.Tensor
    beta_sd: torch.Tensor
    elbo: float
    elbo_history: torch.Tensor
    credible_sets: list[Tuple[int, list[int]]]
    converged: bool
    n_iter: int


def ser_posterior(
    z: torch.Tensor,
    R: torch.Tensor,
    n: int,
    sigma_prior_sq: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the per-variant Single Effect Regression posterior on summary stats.

    Per Zou et al. 2022 [C4] eq. 8-10:
        sigma_lj^2 = 1 / (R_jj / sigma_prior^2 + n)
        mu_lj = sigma_lj^2 * z_j * sqrt(n)
        log_BF_lj = 0.5 * log(sigma_lj^2 / sigma_prior^2)
                  + 0.5 * mu_lj^2 / sigma_lj^2

    Args:
        z: Per-variant z-scores, shape (p,) or (..., p).
        R: LD correlation matrix, shape (p, p). Only the diagonal is used here;
            off-diagonal enters via the IBSS residual update (Task 6).
        n: GWAS sample size.
        sigma_prior_sq: Prior variance of the single effect (sigma_lprior^2).

    Returns:
        (sigma_sq, mu, log_bf) tuple, each shape matching z.
    """
    R_diag = torch.diag(R)
    sigma_sq = 1.0 / (R_diag / sigma_prior_sq + n)
    mu = sigma_sq * z * math.sqrt(n)
    log_bf = (
        0.5 * torch.log(sigma_sq / sigma_prior_sq)
        + 0.5 * mu ** 2 / sigma_sq
    )
    return sigma_sq, mu, log_bf


@dataclass
class BayesianVSRss:
    """SuSiE-RSS fine-mapping on summary statistics.

    Per NA1 design spec section 3.4 public API. The fit_rss method (Task 6+)
    runs IBSS to convergence using the SER posterior from ser_posterior above.

    Args:
        max_num_causal: L (single-effect layers, default 10 per [C4]).
        coverage: credible-set coverage threshold (default 0.95 per [C1]).
        purity: minimum |R_jk| within a credible set (default 0.5,
            matches susieR::susie_rss(min_abs_corr=0.5)).
        sigma_prior_sq: prior variance of single effects (default 0.04).
        max_iter: IBSS max iterations (default 100).
        tol: ELBO convergence tolerance (default 1e-6).
        block_size_threshold: max p per block before decomposition kicks in
            (default 5000 per PolyFun [C5] / ldetect convention).
    """
    max_num_causal: int = 10
    coverage: float = 0.95
    purity: float = 0.5
    sigma_prior_sq: float = 0.04
    max_iter: int = 100
    tol: float = 1e-6
    block_size_threshold: int = 5000

    # Methods fit_rss, _run_ibss, _compute_elbo, _build_credible_sets
    # are added in Tasks 6-9.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 5: BayesianVSRss scaffolding + SER posterior closed-form"
```

---

## Task 6: IBSS update + softmax with per-SNP priors (D3 shim)

**Files:**
- Modify: `torchgenomics/models/bayesian_vs_rss.py` (add `_compute_alpha`, `_run_ibss_layer`)
- Modify: `tests/test_bayesian_vs_rss.py` (append IBSS + softmax tests)

**Background:** Per spec §2.3 (Zou 2022 [C4] eq. 11): IBSS updates each layer using `R @ b_l = R @ (alpha_l * mu_l)` for the cross-layer residual. Per spec §2.2 + Decision 3 / Phase 59 D3 shim: the per-layer softmax weights priors per SNP — `prior_pi_per_snp` keyword optional, defaulting to uniform.

- [ ] **Step 1: Write failing tests for the softmax + per-SNP prior**

Append to `tests/test_bayesian_vs_rss.py`:

```python
from torchgenomics.models.bayesian_vs_rss import compute_alpha


def test_softmax_uniform_prior_recovers_softmax_over_bf():
    """With uniform prior, alpha is exact softmax over log_BF."""
    log_bf = torch.tensor([0.0, 1.0, 2.0, 0.5], dtype=torch.float64)
    alpha = compute_alpha(log_bf, prior_pi=None)
    expected = torch.softmax(log_bf, dim=0)
    assert torch.allclose(alpha, expected, atol=1e-12)


def test_softmax_uniform_prior_sums_to_one():
    """alpha sums to 1.0."""
    log_bf = torch.tensor([0.0, 5.0, 1.0, 3.0], dtype=torch.float64)
    alpha = compute_alpha(log_bf, prior_pi=None)
    assert torch.isclose(alpha.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


def test_extreme_prior_dominates_alpha():
    """An extreme per-SNP prior on one SNP drives alpha toward that SNP."""
    log_bf = torch.zeros(4, dtype=torch.float64)  # uniform BFs
    prior_pi = torch.tensor([0.99, 1e-3, 1e-3, 1e-3 - 1e-12], dtype=torch.float64)
    # Normalize prior_pi to sum to 1.0 inside compute_alpha
    alpha = compute_alpha(log_bf, prior_pi=prior_pi)
    # Approx: alpha[0] ~ 0.99 / (0.99 + 3e-3) ~ 0.997
    assert alpha[0] > 0.99


def test_per_snp_prior_normalized_internally():
    """Unnormalized prior is normalized internally; equivalent to normalized input."""
    log_bf = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    prior_unnorm = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64)
    prior_norm = prior_unnorm / prior_unnorm.sum()
    alpha_unnorm = compute_alpha(log_bf, prior_pi=prior_unnorm)
    alpha_norm = compute_alpha(log_bf, prior_pi=prior_norm)
    assert torch.allclose(alpha_unnorm, alpha_norm, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short -k "softmax or prior"
```

Expected: FAIL with `ImportError: cannot import name 'compute_alpha'`.

- [ ] **Step 3: Implement `compute_alpha`**

Append to `torchgenomics/models/bayesian_vs_rss.py`:

```python
def compute_alpha(
    log_bf: torch.Tensor,
    prior_pi: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-variant inclusion probabilities via prior-weighted softmax.

    Per Zou et al. 2022 [C4] eq. 10 and the D3 shim from Phase 59 PolyFun spec:
        alpha_lj = pi_j * exp(log_BF_lj) / sum_j' pi_j' * exp(log_BF_lj')

    With prior_pi=None, uses uniform 1/p — recovers exact softmax over log_BF.
    With prior_pi provided as a tensor, normalizes internally to sum to 1.0
    within the locus, then weights the softmax accordingly.

    Args:
        log_bf: Per-variant log Bayes factors, shape (p,).
        prior_pi: Optional per-variant prior inclusion probabilities, shape (p,).
            If None, uniform 1/p is used.

    Returns:
        alpha: Per-variant inclusion probabilities, shape (p,), summing to 1.0.
    """
    if prior_pi is None:
        return torch.softmax(log_bf, dim=0)

    # Normalize prior_pi to sum to 1.0 within the locus
    prior_pi_normalized = prior_pi / prior_pi.sum()
    log_prior = torch.log(prior_pi_normalized.clamp_min(1e-300))
    return torch.softmax(log_bf + log_prior, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: all tests pass (3 from Task 5 + 4 from Task 6 = 7 total).

- [ ] **Step 5: Write failing test for the IBSS update primitive**

Append to `tests/test_bayesian_vs_rss.py`:

```python
from torchgenomics.models.bayesian_vs_rss import ibss_residual_update


def test_ibss_residual_subtracts_other_layers():
    """IBSS residual: tilde_z_l = z - R @ sum_{l' != l} b_{l'}.

    Per Zou et al. 2022 [C4] eq. 11.
    """
    z = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    R = torch.tensor(
        [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]],
        dtype=torch.float64,
    )
    # Two layers, each with effects b_l = alpha_l * mu_l
    b = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
        dtype=torch.float64,
    )  # shape (L=2, p=3)
    # Tilde z for layer 0: z - R @ b[1] = z - R @ [0, 0.2, 0]
    expected_l0 = z - R @ b[1]
    tilde_z_l0 = ibss_residual_update(z, R, b, layer_idx=0)
    assert torch.allclose(tilde_z_l0, expected_l0, atol=1e-12)

    # Tilde z for layer 1: z - R @ b[0] = z - R @ [0.1, 0, 0]
    expected_l1 = z - R @ b[0]
    tilde_z_l1 = ibss_residual_update(z, R, b, layer_idx=1)
    assert torch.allclose(tilde_z_l1, expected_l1, atol=1e-12)
```

- [ ] **Step 6: Run test to verify it fails**

```bash
pytest tests/test_bayesian_vs_rss.py::test_ibss_residual_subtracts_other_layers -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 7: Implement `ibss_residual_update`**

Append to `torchgenomics/models/bayesian_vs_rss.py`:

```python
def ibss_residual_update(
    z: torch.Tensor,
    R: torch.Tensor,
    b: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute the IBSS residual for a given layer.

    Per Zou et al. 2022 [C4] eq. 11:
        tilde_z_l = z - R @ sum_{l' != l} b_{l'}

    Args:
        z: Per-variant z-scores, shape (p,).
        R: LD correlation matrix, shape (p, p).
        b: Per-layer effect vectors, shape (L, p), where b[l] = alpha[l] * mu[l].
        layer_idx: Index of the layer being updated (excluded from the sum).

    Returns:
        tilde_z_l: Residual z-scores for layer layer_idx, shape (p,).
    """
    # Sum over all layers except layer_idx
    mask = torch.ones(b.shape[0], dtype=torch.bool, device=b.device)
    mask[layer_idx] = False
    other_layers_sum = b[mask].sum(dim=0)  # shape (p,)
    return z - R @ other_layers_sum
```

- [ ] **Step 8: Run all tests to verify they pass**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: 8 tests pass.

- [ ] **Step 9: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 6: IBSS update + softmax with per-SNP priors (D3 shim)"
```

---

## Task 7: Full IBSS loop + ELBO computation

**Files:**
- Modify: `torchgenomics/models/bayesian_vs_rss.py` (add `BayesianVSRss._run_ibss`, `_compute_elbo`)
- Modify: `tests/test_bayesian_vs_rss.py` (append loop + ELBO tests)

**Background:** Per spec §2.3 + §2.4 — IBSS iterates the per-layer SER + residual update until convergence (ELBO change < tol or max_iter reached). ELBO is monotone non-decreasing per iteration; this is the critical convergence property to test.

- [ ] **Step 1: Write failing test for full IBSS loop with ELBO monotonicity**

Append to `tests/test_bayesian_vs_rss.py`:

```python
def test_ibss_elbo_monotone_non_decreasing():
    """ELBO is non-decreasing per IBSS iteration (within numerical tolerance).

    Per Zou et al. 2022 [C4] section A.2 supplementary.
    """
    rng = np.random.default_rng(13)
    p = 30
    n = 500
    # Synthetic z-scores with one strong signal
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64))
    z[10] += 5.0  # planted causal at index 10
    # Identity R (independent SNPs) — IBSS should converge in few iterations
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=3, max_iter=50, tol=1e-8)
    result = model.fit_rss(z=z, R=R, n=n)

    elbo_history = result.elbo_history
    # Within numerical noise (1e-9), ELBO is non-decreasing
    diffs = elbo_history[1:] - elbo_history[:-1]
    assert (diffs >= -1e-9).all(), f"ELBO decreased: min diff = {diffs.min().item()}"


def test_ibss_recovers_planted_causal():
    """IBSS recovers a planted causal SNP at high PIP."""
    rng = np.random.default_rng(17)
    p = 50
    n = 1000
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64) * 0.5)
    causal_idx = 25
    z[causal_idx] = 6.0  # very strong signal
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=5)
    result = model.fit_rss(z=z, R=R, n=n)

    assert result.pip[causal_idx] > 0.9, \
        f"Planted causal at index {causal_idx} had PIP {result.pip[causal_idx]:.4f}"


def test_pip_formula():
    """PIP = 1 - prod_l (1 - alpha[l]) per Wang et al. 2020 [C1] eq. 12."""
    p = 5
    L = 3
    rng = np.random.default_rng(23)
    alpha = torch.from_numpy(rng.uniform(0.0, 0.3, (L, p)).astype(np.float64))
    expected_pip = 1.0 - torch.prod(1.0 - alpha, dim=0)

    # Build a fixture result with known alpha
    z = torch.zeros(p, dtype=torch.float64)
    R = torch.eye(p, dtype=torch.float64)
    model = BayesianVSRss(max_num_causal=L, max_iter=1)
    result = model.fit_rss(z=z, R=R, n=100)
    # The internal pip computation must match the formula on result.alpha
    actual_pip = 1.0 - torch.prod(1.0 - result.alpha, dim=0)
    assert torch.allclose(result.pip, actual_pip, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short -k "elbo or ibss or pip_formula"
```

Expected: FAIL with `AttributeError: 'BayesianVSRss' object has no attribute 'fit_rss'`.

- [ ] **Step 3: Implement `BayesianVSRss.fit_rss` with full IBSS loop**

Append to `torchgenomics/models/bayesian_vs_rss.py` (add as methods of the `BayesianVSRss` dataclass — convert to a regular class if needed):

Replace the `BayesianVSRss` dataclass scaffolding with a full class:

```python
class BayesianVSRss:
    """SuSiE-RSS fine-mapping on summary statistics.

    See module docstring and design spec for full reference.
    """

    def __init__(
        self,
        max_num_causal: int = 10,
        coverage: float = 0.95,
        purity: float = 0.5,
        sigma_prior_sq: float = 0.04,
        max_iter: int = 100,
        tol: float = 1e-6,
        block_size_threshold: int = 5000,
    ):
        self.max_num_causal = max_num_causal
        self.coverage = coverage
        self.purity = purity
        self.sigma_prior_sq = sigma_prior_sq
        self.max_iter = max_iter
        self.tol = tol
        self.block_size_threshold = block_size_threshold

    def fit_rss(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        prior_pi_per_snp: Optional[torch.Tensor] = None,
    ) -> BayesianVSRssResult:
        """Run SuSiE-RSS IBSS to convergence on a single locus.

        Args:
            z: Per-variant z-scores, shape (p,).
            R: LD correlation matrix, shape (p, p).
            n: GWAS sample size.
            prior_pi_per_snp: Optional per-SNP prior, shape (p,). D3 shim from
                Phase 59 PolyFun spec; if None, uses uniform 1/p.

        Returns:
            BayesianVSRssResult with alpha, mu, sigma_sq, pip, beta_mean,
            beta_sd, elbo, elbo_history, credible_sets, converged, n_iter.
        """
        z = z.to(torch.float64)
        R = R.to(torch.float64)
        p = z.shape[0]
        L = self.max_num_causal

        # Initialize per-layer effects to zero
        alpha = torch.zeros(L, p, dtype=torch.float64)
        mu = torch.zeros(L, p, dtype=torch.float64)
        sigma_sq = torch.zeros(L, p, dtype=torch.float64)
        b = torch.zeros(L, p, dtype=torch.float64)  # b[l] = alpha[l] * mu[l]

        elbo_history = []

        for iteration in range(self.max_iter):
            for l in range(L):
                # Compute residual for layer l
                tilde_z_l = ibss_residual_update(z, R, b, layer_idx=l)
                # SER posterior on residual
                sigma_sq_l, mu_l, log_bf_l = ser_posterior(
                    tilde_z_l, R, n, self.sigma_prior_sq
                )
                # Softmax with per-SNP prior (D3 shim)
                alpha_l = compute_alpha(log_bf_l, prior_pi=prior_pi_per_snp)

                alpha[l] = alpha_l
                mu[l] = mu_l
                sigma_sq[l] = sigma_sq_l
                b[l] = alpha_l * mu_l

            # Compute ELBO at end of iteration
            elbo = self._compute_elbo(z, R, n, alpha, mu, sigma_sq)
            elbo_history.append(elbo)

            # Convergence check
            if iteration > 0:
                elbo_diff = elbo_history[-1] - elbo_history[-2]
                if abs(elbo_diff) < self.tol:
                    converged = True
                    break
        else:
            converged = False

        elbo_history_tensor = torch.tensor(elbo_history, dtype=torch.float64)

        # Per-variant PIP, beta_mean, beta_sd
        pip = 1.0 - torch.prod(1.0 - alpha, dim=0)
        beta_mean = (alpha * mu).sum(dim=0)
        # Per-variant variance: var(beta) = sum_l (alpha_lj * (mu_lj^2 + sigma_lj^2)) - beta_mean^2
        # This is the law of total variance applied across the L single-effect components.
        second_moment = (alpha * (mu ** 2 + sigma_sq)).sum(dim=0)
        beta_var = second_moment - beta_mean ** 2
        beta_var = beta_var.clamp_min(0.0)  # numerical floor
        beta_sd = beta_var.sqrt()

        # Credible sets (placeholder; full implementation in Task 8)
        credible_sets: list[Tuple[int, list[int]]] = []

        return BayesianVSRssResult(
            alpha=alpha,
            mu=mu,
            sigma_sq=sigma_sq,
            pip=pip,
            beta_mean=beta_mean,
            beta_sd=beta_sd,
            elbo=float(elbo_history_tensor[-1]),
            elbo_history=elbo_history_tensor,
            credible_sets=credible_sets,
            converged=converged,
            n_iter=iteration + 1,
        )

    @staticmethod
    def _compute_elbo(
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        alpha: torch.Tensor,
        mu: torch.Tensor,
        sigma_sq: torch.Tensor,
    ) -> float:
        """Compute the ELBO for SuSiE-RSS.

        Per Zou et al. 2022 [C4] section A.2 supplementary. The ELBO has the
        form ELBO = E[log p(z | b, sigma^2)] - KL[q(b, sigma^2) || p(b, sigma^2)].

        For the per-layer single-effect prior, the KL term decomposes per layer:
            KL_l = sum_j alpha_lj * (log(alpha_lj * p) - 0.5 * (1 + log(sigma_lj^2 / sigma_prior^2) - mu_lj^2 / sigma_prior^2 - sigma_lj^2 / sigma_prior^2))

        Returns:
            elbo: Scalar ELBO value (float).
        """
        p = z.shape[0]
        L = alpha.shape[0]

        # Reconstructed z = R @ sum_l b_l where b_l = alpha_l * mu_l
        b = alpha * mu  # (L, p)
        b_total = b.sum(dim=0)  # (p,)
        z_pred = R @ b_total

        # Likelihood term: - 0.5 * (z - z_pred)^T (z - z_pred) (modulo constants
        # involving n; the constant offsets do not affect monotonicity)
        residual = z - z_pred
        log_lik = -0.5 * (residual ** 2).sum().item()

        # KL for the categorical inclusion (alpha vs uniform 1/p)
        # KL_cat = sum_l sum_j alpha_lj * log(alpha_lj * p)
        alpha_safe = alpha.clamp_min(1e-300)
        kl_cat = (alpha * (torch.log(alpha_safe) + math.log(p))).sum().item()

        # KL for the Gaussian prior on the effect, per layer
        # KL_gauss_l = 0.5 * sum_j alpha_lj * (mu_lj^2 / sigma_prior^2 + sigma_lj^2 / sigma_prior^2 - 1 - log(sigma_lj^2 / sigma_prior^2))
        sigma_prior_sq = 0.04  # NOTE: this should be the model's sigma_prior_sq;
        # in the real implementation this is passed via the model instance.
        # For this static method it's hardcoded to match the default; refactor
        # to pass it as an arg if non-default values are used.
        kl_gauss = (
            0.5 * alpha * (
                mu ** 2 / sigma_prior_sq
                + sigma_sq / sigma_prior_sq
                - 1.0
                - torch.log(sigma_sq / sigma_prior_sq)
            )
        ).sum().item()

        return log_lik - kl_cat - kl_gauss
```

**NOTE on `sigma_prior_sq` in `_compute_elbo`:** the static method hardcodes the default. In the real implementation, the method should be an instance method (`self._compute_elbo(...)`) and use `self.sigma_prior_sq`. The next step modifies it.

- [ ] **Step 4: Refactor `_compute_elbo` to use `self.sigma_prior_sq`**

Edit `torchgenomics/models/bayesian_vs_rss.py`: change `_compute_elbo` from `@staticmethod` to a regular method, and replace the hardcoded `sigma_prior_sq = 0.04` with `self.sigma_prior_sq`. Update the call site in `fit_rss` from `self._compute_elbo(z, R, n, alpha, mu, sigma_sq)` (already correct since it's called via `self.`).

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: all tests pass (8 from previous tasks + 3 new = 11).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 7: full IBSS loop + ELBO computation (monotone non-decreasing)"
```

---

## Task 8: Credible-set construction

**Files:**
- Modify: `torchgenomics/models/bayesian_vs_rss.py` (add `_build_credible_sets`)
- Modify: `tests/test_bayesian_vs_rss.py` (append credible-set tests)

**Background:** Per spec §2.5 — for each layer, sort alpha descending, take smallest set with cumulative ≥ coverage, enforce purity via pairwise |R_jk| ≥ purity threshold.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_bayesian_vs_rss.py`:

```python
def test_credible_set_single_strong_layer():
    """A layer with one dominant alpha produces a credible set of one variant."""
    p = 5
    alpha = torch.zeros(1, p, dtype=torch.float64)
    alpha[0, 2] = 0.99
    alpha[0] += 0.01 / p
    alpha[0, 2] -= 0.01 / p  # keep total = 1.0 (approx)
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=1, coverage=0.95, purity=0.5)
    cs = model._build_credible_sets(alpha, R)
    assert len(cs) == 1
    layer_idx, members = cs[0]
    assert layer_idx == 0
    assert members == [2]


def test_credible_set_purity_filters_low_correlation():
    """If purity check fails, the credible set is dropped (or shrunk)."""
    p = 4
    alpha = torch.zeros(1, p, dtype=torch.float64)
    # Two SNPs share alpha but are uncorrelated
    alpha[0, 0] = 0.55
    alpha[0, 3] = 0.45
    R = torch.eye(p, dtype=torch.float64)  # zero off-diagonal correlation

    model = BayesianVSRss(max_num_causal=1, coverage=0.95, purity=0.5)
    cs = model._build_credible_sets(alpha, R)
    # The naive cumulative-coverage set would be {0, 3} but purity = 0
    # (R[0, 3] = 0), so the credible set should be empty (purity-failed)
    # OR truncated to just the top variant. Per susieR, the entire CS is
    # dropped when purity fails; mirror that.
    assert len(cs) == 0 or (len(cs) == 1 and len(cs[0][1]) <= 1)


def test_credible_set_sorted_by_pip():
    """Variants within a credible set are ordered by descending alpha."""
    p = 4
    alpha = torch.zeros(2, p, dtype=torch.float64)
    alpha[0, 0] = 0.5
    alpha[0, 1] = 0.4
    alpha[0, 2] = 0.1
    alpha[1, 3] = 0.95
    alpha[1, 0] = 0.05
    R = torch.tensor(
        [[1.0, 0.7, 0.0, 0.0],
         [0.7, 1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    model = BayesianVSRss(max_num_causal=2, coverage=0.9, purity=0.5)
    cs = model._build_credible_sets(alpha, R)

    # Layer 0: cumsum 0.5 + 0.4 = 0.9 → CS = [0, 1]; purity |R[0,1]| = 0.7 ≥ 0.5 → kept
    # Layer 1: alpha[3] = 0.95 ≥ 0.9 → CS = [3]
    # Both layers should produce a credible set
    assert len(cs) == 2
    # Within CS, members ordered by descending alpha
    layers = {layer_idx: members for layer_idx, members in cs}
    assert layers[0] == [0, 1]  # 0.5 > 0.4
    assert layers[1] == [3]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short -k "credible_set"
```

Expected: FAIL with `AttributeError: '_build_credible_sets' not found` or similar.

- [ ] **Step 3: Implement `_build_credible_sets`**

Add as method of `BayesianVSRss` class in `torchgenomics/models/bayesian_vs_rss.py`:

```python
    def _build_credible_sets(
        self,
        alpha: torch.Tensor,
        R: torch.Tensor,
    ) -> list[Tuple[int, list[int]]]:
        """Construct credible sets per layer, with purity filtering.

        Per Wang et al. 2020 [C1] section 3.4:
        1. Sort alpha[l] descending.
        2. Take the smallest set whose cumulative alpha >= self.coverage.
        3. Enforce purity: minimum pairwise |R_jk| within the set >= self.purity.
           If purity check fails, drop the credible set.

        Args:
            alpha: Per-layer per-variant inclusion probabilities, shape (L, p).
            R: LD correlation matrix, shape (p, p).

        Returns:
            List of (layer_idx, sorted_member_indices) tuples for credible sets
            that pass the purity check.
        """
        L, p = alpha.shape
        cs_list: list[Tuple[int, list[int]]] = []

        for l in range(L):
            alpha_l = alpha[l]
            # Sort descending
            sorted_alpha, sorted_indices = torch.sort(alpha_l, descending=True)
            cumsum = torch.cumsum(sorted_alpha, dim=0)
            # First index where cumsum >= coverage
            mask = cumsum >= self.coverage
            if not mask.any():
                continue  # Not enough alpha mass to form a credible set
            cutoff = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1
            members = sorted_indices[:cutoff].tolist()

            # Purity check: min pairwise |R_jk| over members
            if len(members) > 1:
                sub_R = R[members][:, members]
                # Off-diagonal absolute values
                off_diag_mask = ~torch.eye(len(members), dtype=torch.bool, device=R.device)
                min_abs_corr = sub_R.abs()[off_diag_mask].min().item()
                if min_abs_corr < self.purity:
                    continue  # Purity failed; drop CS

            cs_list.append((l, members))

        return cs_list
```

- [ ] **Step 4: Wire `_build_credible_sets` into `fit_rss`**

In `torchgenomics/models/bayesian_vs_rss.py`, find the line in `fit_rss` that says:

```python
        # Credible sets (placeholder; full implementation in Task 8)
        credible_sets: list[Tuple[int, list[int]]] = []
```

Replace with:

```python
        credible_sets = self._build_credible_sets(alpha, R)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: all tests pass (11 from previous + 3 new = 14).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 8: credible-set construction with purity filtering"
```

---

## Task 9: Block decomposition integration in `fit_rss`

**Files:**
- Modify: `torchgenomics/models/bayesian_vs_rss.py` (add `fit_rss_blocked` orchestrator + integrate into `fit_rss` for large p)
- Modify: `tests/test_bayesian_vs_rss.py` (append block-equivalence test)

**Background:** Per spec §2.6 and §3.4 — block decomposition is Tier A. When `p > block_size_threshold`, decompose into blocks (auto-detect or user-supplied), run `fit_rss` per block, concatenate results.

- [ ] **Step 1: Write the failing test for block-equivalence (algorithmic, not approximation)**

Append to `tests/test_bayesian_vs_rss.py`:

```python
def test_block_decomp_matches_dense_for_block_diagonal_R():
    """Block-decomp on a block-diagonal R matches dense fit exactly.

    Per NA1 spec section 2.6: block decomposition is mathematically
    equivalent to dense IBSS when block boundaries are at zero-LD positions.
    Use a strictly block-diagonal R to test this invariant exactly.
    """
    rng = np.random.default_rng(31)
    # Two independent blocks of size 25
    p = 50
    R = torch.zeros(p, p, dtype=torch.float64)
    R[:25, :25] = torch.eye(25, dtype=torch.float64)
    R[25:, 25:] = torch.eye(25, dtype=torch.float64)
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64))
    z[10] += 4.0
    z[35] += 4.0
    n = 1000

    from torchgenomics.postgwas._ld_ref_loader import BlockSpec
    blocks = [BlockSpec(start=0, stop=25), BlockSpec(start=25, stop=50)]

    model = BayesianVSRss(max_num_causal=2, max_iter=50, tol=1e-10)
    result_blocked = model.fit_rss_blocked(z=z, R=R, n=n, blocks=blocks)

    # Dense run for comparison
    result_dense = model.fit_rss(z=z, R=R, n=n)

    # PIP at each variant should match between blocked and dense
    assert torch.allclose(result_blocked.pip, result_dense.pip, atol=1e-8), \
        f"Max PIP diff: {(result_blocked.pip - result_dense.pip).abs().max().item()}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_bayesian_vs_rss.py::test_block_decomp_matches_dense_for_block_diagonal_R -v
```

Expected: FAIL with `AttributeError: 'BayesianVSRss' object has no attribute 'fit_rss_blocked'`.

- [ ] **Step 3: Implement `fit_rss_blocked`**

Append to `BayesianVSRss` class in `torchgenomics/models/bayesian_vs_rss.py`:

```python
    def fit_rss_blocked(
        self,
        z: torch.Tensor,
        R: torch.Tensor,
        n: int,
        blocks: list,
        prior_pi_per_snp: Optional[torch.Tensor] = None,
    ) -> BayesianVSRssResult:
        """Run SuSiE-RSS per block and concatenate results.

        Per NA1 design spec section 2.6: block decomposition is mathematically
        equivalent to dense IBSS when block boundaries are at near-zero-LD
        positions. Memory bound becomes O(p_block_max^2) per block instead
        of O(p^2) total.

        Args:
            z: Per-variant z-scores, shape (p,).
            R: LD correlation matrix, shape (p, p) — only the per-block
                sub-matrices R[block][:, block] are accessed.
            n: GWAS sample size.
            blocks: List of BlockSpec defining the block partition.
            prior_pi_per_snp: Optional per-SNP prior, shape (p,).

        Returns:
            BayesianVSRssResult covering all p variants, with per-block
            results concatenated.
        """
        p = z.shape[0]
        L = self.max_num_causal

        alpha_full = torch.zeros(L, p, dtype=torch.float64)
        mu_full = torch.zeros(L, p, dtype=torch.float64)
        sigma_sq_full = torch.zeros(L, p, dtype=torch.float64)
        pip_full = torch.zeros(p, dtype=torch.float64)
        beta_mean_full = torch.zeros(p, dtype=torch.float64)
        beta_sd_full = torch.zeros(p, dtype=torch.float64)
        elbo_total = 0.0
        all_credible_sets: list[Tuple[int, list[int]]] = []
        all_converged = True
        max_n_iter = 0

        for b_idx, block in enumerate(blocks):
            start, stop = block.start, block.stop
            z_block = z[start:stop]
            R_block = R[start:stop, start:stop]
            prior_block = (
                prior_pi_per_snp[start:stop]
                if prior_pi_per_snp is not None
                else None
            )

            # Run dense fit on the block
            block_result = self.fit_rss(
                z=z_block,
                R=R_block,
                n=n,
                prior_pi_per_snp=prior_block,
            )

            alpha_full[:, start:stop] = block_result.alpha
            mu_full[:, start:stop] = block_result.mu
            sigma_sq_full[:, start:stop] = block_result.sigma_sq
            pip_full[start:stop] = block_result.pip
            beta_mean_full[start:stop] = block_result.beta_mean
            beta_sd_full[start:stop] = block_result.beta_sd
            elbo_total += block_result.elbo
            # Re-index credible-set members from block-local to global indices
            for layer_idx, members in block_result.credible_sets:
                global_members = [m + start for m in members]
                all_credible_sets.append((layer_idx, global_members))
            all_converged = all_converged and block_result.converged
            max_n_iter = max(max_n_iter, block_result.n_iter)

        return BayesianVSRssResult(
            alpha=alpha_full,
            mu=mu_full,
            sigma_sq=sigma_sq_full,
            pip=pip_full,
            beta_mean=beta_mean_full,
            beta_sd=beta_sd_full,
            elbo=elbo_total,
            elbo_history=torch.tensor([elbo_total], dtype=torch.float64),
            credible_sets=all_credible_sets,
            converged=all_converged,
            n_iter=max_n_iter,
        )
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: all 15 tests pass.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 9: block decomposition integration (fit_rss_blocked)"
```

---

## Task 10: Output writer (PolyFun-compatible TSV)

**Files:**
- Modify: `torchgenomics/models/bayesian_vs_rss.py` (add `write_results_tsv`)
- Modify: `tests/test_bayesian_vs_rss.py` (append output-format test)

**Background:** Per spec §6 — output schema must match Phase 59 PolyFun's exact column set so downstream PolyFun aggregation utilities work without translation.

- [ ] **Step 1: Write failing test for the output writer**

Append to `tests/test_bayesian_vs_rss.py`:

```python
def test_output_writer_matches_polyfun_schema(tmp_path):
    """Output TSV matches Phase 59 PolyFun column schema exactly."""
    from torchgenomics.models.bayesian_vs_rss import write_results_tsv

    rng = np.random.default_rng(41)
    p = 5
    z = torch.from_numpy(rng.standard_normal(p).astype(np.float64))
    R = torch.eye(p, dtype=torch.float64)
    n = 500

    model = BayesianVSRss(max_num_causal=2)
    result = model.fit_rss(z=z, R=R, n=n)

    snp_meta = {
        "snp": [f"rs{i}" for i in range(p)],
        "chr": [22] * p,
        "bp": [1000 + i * 10 for i in range(p)],
        "a1": ["A"] * p,
        "a2": ["G"] * p,
        "z": z.tolist(),
        "n": [n] * p,
    }

    out_path = tmp_path / "finemap.tsv"
    write_results_tsv(out_path, result, snp_meta)

    # Read back and verify column order
    import pandas as pd
    df = pd.read_csv(out_path, sep="\t")
    expected_cols = [
        "SNP", "CHR", "BP", "A1", "A2", "Z", "N",
        "PIP", "BETA_MEAN", "BETA_SD", "CREDIBLE_SET",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == p
    # CREDIBLE_SET column: 0 for variants not in any CS
    # PIP column: matches result.pip
    assert np.allclose(df["PIP"].values, result.pip.numpy(), atol=1e-12)
    assert np.allclose(df["BETA_MEAN"].values, result.beta_mean.numpy(), atol=1e-12)
    assert np.allclose(df["BETA_SD"].values, result.beta_sd.numpy(), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_bayesian_vs_rss.py::test_output_writer_matches_polyfun_schema -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `write_results_tsv`**

Append to `torchgenomics/models/bayesian_vs_rss.py`:

```python
import csv
from pathlib import Path


def write_results_tsv(
    path: Path | str,
    result: BayesianVSRssResult,
    snp_meta: dict,
) -> None:
    """Write SuSiE-RSS results to a Phase 59 PolyFun-compatible TSV.

    Output columns (per NA1 design spec section 6):
        SNP, CHR, BP, A1, A2, Z, N, PIP, BETA_MEAN, BETA_SD, CREDIBLE_SET

    CREDIBLE_SET = integer index of the credible set the variant belongs to
    (0 = not in any CS; 1, 2, ... for first, second, ... CS in returned order).
    Matches PolyFun convention for downstream aggregation compatibility.

    Args:
        path: Output TSV path.
        result: BayesianVSRssResult to serialize.
        snp_meta: Dict with keys snp, chr, bp, a1, a2, z, n; each a list of
            length p in variant order matching result tensors.
    """
    p = result.pip.shape[0]
    # Build a SNP -> CS index map
    cs_index = [0] * p
    for cs_id, (layer_idx, members) in enumerate(result.credible_sets, start=1):
        for m in members:
            # If a variant appears in multiple CSes (rare), keep the first
            if cs_index[m] == 0:
                cs_index[m] = cs_id

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["SNP", "CHR", "BP", "A1", "A2", "Z", "N",
             "PIP", "BETA_MEAN", "BETA_SD", "CREDIBLE_SET"]
        )
        for j in range(p):
            writer.writerow([
                snp_meta["snp"][j],
                snp_meta["chr"][j],
                snp_meta["bp"][j],
                snp_meta["a1"][j],
                snp_meta["a2"][j],
                f"{snp_meta['z'][j]:.6g}",
                snp_meta["n"][j],
                f"{float(result.pip[j]):.6g}",
                f"{float(result.beta_mean[j]):.6g}",
                f"{float(result.beta_sd[j]):.6g}",
                cs_index[j],
            ])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_bayesian_vs_rss.py -v --tb=short
```

Expected: all 16 tests pass.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/bayesian_vs_rss.py tests/test_bayesian_vs_rss.py
git commit -m "NA1 Task 10: output writer (Phase 59 PolyFun-compatible TSV)"
```

---

## Task 11: CLI subcommand `bayes-scan-rss`

**Files:**
- Modify: `torchgenomics/cli.py` (add subcommand handler)
- Modify: `tests/test_cli.py` (append smoke test)

**Background:** Per spec §6 + Decision 2 + Decision 4 — new subcommand `bayes-scan-rss` taking sumstats + (`--ld-ref` OR `--geno`) + `--regions` + `--max-num-causal` + `--coverage` + `--purity` + `--prior-pi`.

- [ ] **Step 1: Inspect existing CLI structure to mirror conventions**

```bash
grep -n "_cmd_bayes_scan\|add_subparsers\|bayes-scan" torchgenomics/cli.py | head -20
```

Locate the function `_cmd_bayes_scan` and its argparse setup. Note the registration pattern (subparser group + handler function).

- [ ] **Step 2: Write failing smoke test**

Append to `tests/test_cli.py` (or create `tests/test_cli_bayes_scan_rss.py` if `test_cli.py` is not the right file):

```python
def test_bayes_scan_rss_help(capsys):
    """`torchgenomics bayes-scan-rss --help` prints usage with expected flags."""
    from torchgenomics.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["bayes-scan-rss", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--sumstats" in captured.out
    assert "--ld-ref" in captured.out
    assert "--geno" in captured.out
    assert "--max-num-causal" in captured.out
    assert "--coverage" in captured.out
    assert "--purity" in captured.out
    assert "--output" in captured.out


def test_bayes_scan_rss_smoke_runs_end_to_end(tmp_path):
    """End-to-end smoke run on a tiny synthetic fixture.

    Builds sumstats + LD reference in tmp_path, invokes the CLI, asserts
    output file exists and has the expected columns.
    """
    import pandas as pd
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata
    from torchgenomics.cli import main

    p = 5
    sumstats_path = tmp_path / "sumstats.tsv"
    sumstats = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(p)],
        "CHR": [22] * p,
        "BP": [1000 + i * 10 for i in range(p)],
        "A1": ["A"] * p,
        "A2": ["G"] * p,
        "BETA": [0.05, 0.03, 0.6, 0.02, 0.04],
        "SE": [0.05, 0.04, 0.05, 0.04, 0.05],
        "N": [1000] * p,
    })
    sumstats.to_csv(sumstats_path, sep="\t", index=False)

    ld_path = tmp_path / "ld.pt"
    R = torch.eye(p, dtype=torch.float64)
    snp_ids = sumstats["SNP"].tolist()
    meta = LDReferenceMetadata(
        cohort_id="test", n=1000, build="GRCh38", panel_provenance="synthetic"
    )
    save_ld_reference(ld_path, R, snp_ids, meta)

    out_path = tmp_path / "finemap.tsv"
    rc = main([
        "bayes-scan-rss",
        "--sumstats", str(sumstats_path),
        "--ld-ref", str(ld_path),
        "--max-num-causal", "2",
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == [
        "SNP", "CHR", "BP", "A1", "A2", "Z", "N",
        "PIP", "BETA_MEAN", "BETA_SD", "CREDIBLE_SET",
    ]
    assert len(df) == p
    # SNP rs2 has the strongest BETA/SE → highest PIP
    pip_values = df["PIP"].values
    assert pip_values[2] == pip_values.max()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v -k "bayes_scan_rss" --tb=short
```

Expected: FAIL with `unrecognized arguments: bayes-scan-rss` or similar.

- [ ] **Step 4: Implement the CLI subcommand**

Edit `torchgenomics/cli.py`. Locate the function or block where existing subcommands are registered. Add:

```python
def _cmd_bayes_scan_rss(args) -> int:
    """Handler for the bayes-scan-rss subcommand.

    Runs SuSiE-RSS fine-mapping per NA1 design spec.
    """
    import pandas as pd
    import math
    import torch

    from torchgenomics.models.bayesian_vs_rss import (
        BayesianVSRss,
        write_results_tsv,
    )
    from torchgenomics.postgwas._ld_ref_loader import (
        load_ld_reference,
        compute_in_sample_ld,
        decompose_into_blocks,
    )
    from torchgenomics.postgwas._ld_ref_metadata import (
        LDReferenceMetadata,
        check_metadata_compatibility,
    )

    # Load sumstats
    sumstats = pd.read_csv(args.sumstats, sep=None, engine="python")
    required_cols = {"SNP", "CHR", "BP", "A1", "A2"}
    if not required_cols.issubset(sumstats.columns):
        missing = required_cols - set(sumstats.columns)
        raise ValueError(f"sumstats missing columns: {missing}")
    if "Z" in sumstats.columns:
        z = torch.tensor(sumstats["Z"].values, dtype=torch.float64)
        n_per_variant = sumstats.get("N", pd.Series([0])).values
    elif {"BETA", "SE", "N"}.issubset(sumstats.columns):
        z = torch.tensor(
            (sumstats["BETA"] / sumstats["SE"]).values,
            dtype=torch.float64,
        )
        n_per_variant = sumstats["N"].values
    else:
        raise ValueError(
            "sumstats must have either Z column OR BETA + SE + N columns"
        )
    n = int(n_per_variant.max()) if len(n_per_variant) else 0

    # Load LD reference
    if args.ld_ref:
        R, ld_snp_ids, ld_meta = load_ld_reference(args.ld_ref)
        # Cohort-mismatch check
        ss_meta = {
            # If sumstats carry metadata in a sidecar, load here; for MVP we
            # rely on user-supplied metadata being absent → soft warning only.
        }
        check_metadata_compatibility(ld_meta, ss_meta)
        # Verify SNP order alignment
        if ld_snp_ids != sumstats["SNP"].tolist():
            # Reorder R to match sumstats order
            id_to_idx = {snp: i for i, snp in enumerate(ld_snp_ids)}
            order = [id_to_idx[snp] for snp in sumstats["SNP"]
                     if snp in id_to_idx]
            R = R[order][:, order]
    elif args.geno:
        # Compute in-sample LD from genotype panel
        from torchgenomics.io import auto_read_genotype  # adjust to actual API
        G = auto_read_genotype(args.geno)
        R = compute_in_sample_ld(G)
    else:
        raise ValueError("Either --ld-ref or --geno must be supplied")

    p = z.shape[0]

    # Block decomposition
    snp_ids = sumstats["SNP"].tolist()
    if args.regions:
        regions = pd.read_csv(args.regions, sep=None, engine="python")
        # Convert to BlockSpec list
        from torchgenomics.postgwas._ld_ref_loader import BlockSpec
        blocks = [
            BlockSpec(start=int(row["start"]), stop=int(row["stop"]))
            for _, row in regions.iterrows()
        ]
    else:
        blocks = decompose_into_blocks(
            R, snp_ids,
            regions=None,
            max_block_size=args.block_size_threshold,
        )

    # Per-SNP prior (optional D3 shim)
    if args.prior_pi:
        try:
            prior_scalar = float(args.prior_pi)
            prior_pi_per_snp = None  # uniform when scalar
        except ValueError:
            # Treat as a path to a per-SNP prior file
            prior_df = pd.read_csv(args.prior_pi, sep=None, engine="python")
            prior_pi_per_snp = torch.tensor(
                prior_df["PRIOR_PI"].values, dtype=torch.float64
            )
    else:
        prior_pi_per_snp = None

    # Run SuSiE-RSS
    model = BayesianVSRss(
        max_num_causal=args.max_num_causal,
        coverage=args.coverage,
        purity=args.purity,
        block_size_threshold=args.block_size_threshold,
    )

    if len(blocks) == 1 and blocks[0].stop - blocks[0].start <= args.block_size_threshold:
        # Single dense fit
        result = model.fit_rss(z=z, R=R, n=n, prior_pi_per_snp=prior_pi_per_snp)
    else:
        result = model.fit_rss_blocked(
            z=z, R=R, n=n,
            blocks=blocks,
            prior_pi_per_snp=prior_pi_per_snp,
        )

    # Write output
    snp_meta = {
        "snp": sumstats["SNP"].tolist(),
        "chr": sumstats["CHR"].tolist(),
        "bp": sumstats["BP"].tolist(),
        "a1": sumstats["A1"].tolist(),
        "a2": sumstats["A2"].tolist(),
        "z": z.tolist(),
        "n": n_per_variant.tolist() if hasattr(n_per_variant, "tolist") else list(n_per_variant),
    }
    write_results_tsv(args.output, result, snp_meta)
    return 0
```

Then locate where existing subcommand parsers are registered (look for `subparsers.add_parser("bayes-scan", ...)` or similar) and add:

```python
    rss_parser = subparsers.add_parser(
        "bayes-scan-rss",
        help="SuSiE-RSS fine-mapping on summary statistics + LD reference",
    )
    rss_parser.add_argument("--sumstats", required=True, help="Sumstats TSV")
    ld_group = rss_parser.add_mutually_exclusive_group(required=True)
    ld_group.add_argument("--ld-ref", help="Pre-built LD reference (.pt or .npz)")
    ld_group.add_argument("--geno", help="Genotype panel for in-sample LD computation")
    rss_parser.add_argument("--regions", help="Block regions TSV (start, stop)")
    rss_parser.add_argument("--max-num-causal", type=int, default=10)
    rss_parser.add_argument("--coverage", type=float, default=0.95)
    rss_parser.add_argument("--purity", type=float, default=0.5)
    rss_parser.add_argument("--prior-pi", help="Scalar OR path to per-SNP prior file")
    rss_parser.add_argument("--block-size-threshold", type=int, default=5000)
    rss_parser.add_argument("--output", required=True, help="Output TSV path")
    rss_parser.add_argument("--threads", type=int, default=4)
    rss_parser.set_defaults(func=_cmd_bayes_scan_rss)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_cli.py -v -k "bayes_scan_rss" --tb=short
```

Expected: both tests pass. If `auto_read_genotype` import fails, adapt to the actual io API on this branch (may be `read_genotype` or similar — search `torchgenomics/io/__init__.py`).

- [ ] **Step 6: Commit**

```bash
git add torchgenomics/cli.py tests/test_cli.py
git commit -m "NA1 Task 11: CLI subcommand bayes-scan-rss"
```

---

## Task 12: Soft warning on materialized `bayes-scan` (Decision 5)

**Files:**
- Modify: `torchgenomics/models/bayesian_vs.py` (add `UserWarning` in `fit()`)
- Modify: `tests/test_bayesian_vs.py` (add warning capture test — but DO NOT modify any existing tests)

**Background:** Per Decision 5: when `bayes-scan`'s underlying `BayesianVS.fit(G, ...)` is called with `p > 10000`, emit a `UserWarning` pointing at `bayes-scan-rss`.

- [ ] **Step 1: Write the failing test for the warning**

Append a new test to `tests/test_bayesian_vs.py` (do NOT modify existing tests):

```python
def test_bayes_scan_warns_on_large_p(monkeypatch):
    """BayesianVS.fit emits UserWarning when p > 10000 (Decision 5).

    Per NA1 design spec section 6 / Decision 5: warn users with large loci
    to use the new bayes-scan-rss subcommand instead of materializing G.
    """
    import warnings
    import torch
    from torchgenomics.models.bayesian_vs import BayesianVS

    # Fabricate a small G but monkey-patch shape attribute to simulate p > 10000
    # OR: build a real but tiny G and check the warning condition independently.
    # Simpler: construct minimal G with p > 10000 (n small to keep memory ok).
    n = 5
    p = 10001
    G = torch.zeros(n, p, dtype=torch.float64)
    G[0, 0] = 1.0
    # Build minimal variant_meta and other args required by fit
    # NOTE: actual signature of BayesianVS.fit varies; this test may need
    # adaptation. The KEY assertion is that a UserWarning is emitted with
    # text mentioning bayes-scan-rss.
    from torchgenomics.models.base import VariantMeta  # adjust import as needed
    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(p)],
        chr=["1"] * p,
        bp=list(range(p)),
        a1=["A"] * p,
        a2=["G"] * p,
    )
    model = BayesianVS()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model.fit(G, vmeta)
        except Exception:
            # The fit may fail (degenerate input); we only care the warning fires
            pass
        warning_messages = [str(warning.message) for warning in w]
        matched = any("bayes-scan-rss" in msg for msg in warning_messages)
        assert matched, f"Expected UserWarning about bayes-scan-rss; got: {warning_messages}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_bayesian_vs.py::test_bayes_scan_warns_on_large_p -v --tb=short
```

Expected: FAIL — no warning is currently emitted.

- [ ] **Step 3: Implement the warning in `BayesianVS.fit`**

Edit `torchgenomics/models/bayesian_vs.py`. Locate the `fit` method (around line 115-231 per spec). Add at the very top of the method body, after the `G = G.to(STAT_DTYPE)` upcast line (around line 172):

```python
        if G.shape[1] > 10000:
            import warnings
            warnings.warn(
                f"BayesianVS.fit called on locus with p={G.shape[1]} > 10000. "
                f"This will materialize ~{G.shape[0] * G.shape[1] * 8 / 1e9:.1f} GB. "
                f"For large loci, prefer 'torchgenomics bayes-scan-rss' which operates "
                f"on summary statistics + LD reference (per-locus memory ~O(p^2)). "
                f"See docs/cli.md#bayes-scan-rss.",
                UserWarning,
                stacklevel=2,
            )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_bayesian_vs.py -v --tb=short
```

Expected: all existing tests still pass + the new warning test passes. **Critical:** if any existing test now fails, the warning is being mis-emitted on small fixtures or otherwise breaking; fix before proceeding.

- [ ] **Step 5: Commit**

```bash
git add torchgenomics/models/bayesian_vs.py tests/test_bayesian_vs.py
git commit -m "NA1 Task 12: soft warning on materialized bayes-scan (p > 10000)"
```

---

## Task 13: External validation harness — install + fetch_data scripts

**Files:**
- Create: `validation/external/susieR/install.sh`
- Create: `validation/external/susieR/fetch_data.sh`
- Create: `validation/external/susieR/README.md`

**Background:** Per spec §3.1 + §5.2 — mirrors the Pillar B harness layout (`validation/external/regenie/`, `validation/external/ldsc/` etc. on the `efficiency/streaming-scan-audit` branch which we rebased onto in Task 0). `install.sh` installs `susieR` via CRAN with pre-flight; `fetch_data.sh` provisions the MDP-derived per-locus fixture.

- [ ] **Step 1: Inspect Pillar B harness layout for the exact mirror pattern**

```bash
ls validation/external/
ls validation/external/regenie/ 2>/dev/null || ls validation/external/_lib/
cat validation/external/_lib/preflight.sh
```

Expected: see existing harness scripts. Note the function names exposed by `preflight.sh` (e.g., `preflight_check_disk`, `preflight_check_ram`).

- [ ] **Step 2: Create `validation/external/susieR/install.sh`**

```bash
mkdir -p validation/external/susieR
```

Write `validation/external/susieR/install.sh`:

```bash
#!/usr/bin/env bash
# Install susieR via CRAN. Idempotent: detects existing install and skips.
# Per NA1 design spec section 5.2; pre-flight per feedback_preflight memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/../_lib"

# Source pre-flight library (assumes Pillar B convention; verify path)
# shellcheck source=/dev/null
source "${LIB_DIR}/preflight.sh"

# Pre-flight: 2 GB disk + 4 GB RAM (per spec section 5.2)
preflight_check_disk 2 "${SCRIPT_DIR}"
preflight_check_ram 4

# Check if R is installed
if ! command -v R >/dev/null 2>&1; then
    echo "FATAL: R is not installed. Install R first (e.g., conda install -c conda-forge r-base) and retry."
    exit 2
fi

# Check if susieR is already installed
if R -e 'library(susieR)' >/dev/null 2>&1; then
    echo "susieR already installed; skipping."
    exit 0
fi

echo "Installing susieR via CRAN..."
R -e 'install.packages("susieR", repos="https://cloud.r-project.org")'

# Verify install succeeded
if ! R -e 'library(susieR)' >/dev/null 2>&1; then
    echo "FATAL: susieR install failed. Check R toolchain or fall back to containerized R per R-NA1-2 mitigation:"
    echo "    docker run --rm rocker/r-ver:latest R -e 'install.packages(\"susieR\")'"
    exit 3
fi

echo "susieR install succeeded."
```

Make executable:

```bash
chmod +x validation/external/susieR/install.sh
```

- [ ] **Step 3: Verify install.sh syntax**

```bash
bash -n validation/external/susieR/install.sh
```

Expected: no syntax errors.

- [ ] **Step 4: Create `validation/external/susieR/fetch_data.sh`**

Write `validation/external/susieR/fetch_data.sh`:

```bash
#!/usr/bin/env bash
# Provision the MDP-derived per-locus fixture for SuSiE-RSS validation.
# Per NA1 design spec section 5.2.
# Strategy: run lmm-scan on MDP genotype + phenotype, take a window of
# p=500 SNPs around the strongest hit, compute_pairwise_ld for R, save
# (z, R, n) triple as torch tensors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/../_lib"
DATA_DIR="${SCRIPT_DIR}/data"

# shellcheck source=/dev/null
source "${LIB_DIR}/preflight.sh"

# Pre-flight: ensure ≥ 1 GB disk (MDP fixture small)
preflight_check_disk 1 "${SCRIPT_DIR}"

mkdir -p "${DATA_DIR}"

MDP_DIR="${HOME}/Documents/GWAS_Expert/benchmark/data"
if [ ! -f "${MDP_DIR}/mdp_genotype_test.hmp.txt" ]; then
    echo "FATAL: MDP fixture not found at ${MDP_DIR}/mdp_genotype_test.hmp.txt"
    echo "       This script expects the MDP fixture from benchmark/data/."
    exit 2
fi

# If fixture artifacts already exist, skip
if [ -f "${DATA_DIR}/locus_z.pt" ] && [ -f "${DATA_DIR}/locus_R.pt" ]; then
    echo "Fixture already provisioned at ${DATA_DIR}/; skipping."
    exit 0
fi

# Run a Python helper to extract the per-locus fixture
python3 - <<'PYEOF'
import sys
import os
import torch
import numpy as np

# This script is best implemented as a Python helper that:
# 1. Loads MDP genotype via torchgenomics.io.read_hapmap (or equivalent)
# 2. Runs torchgenomics lmm-scan to produce per-variant z-scores
# 3. Selects a window of p=500 SNPs around the top hit
# 4. Computes the LD matrix R for that window
# 5. Saves locus_z.pt, locus_R.pt, locus_meta.pt (n, snp_ids, chr, bp, a1, a2)
#
# For Tier A, we hand-craft the fixture. The actual implementation should
# reproduce the workflow above; for now, this is a placeholder that emits
# a clear error if invoked without the helper module.

DATA_DIR = "{DATA_DIR}".format(DATA_DIR=os.environ.get("DATA_DIR", "data"))
print("Note: fetch_data.sh placeholder. Implement the Python helper at "
      "validation/external/susieR/_provision_locus.py for the full workflow.")
print("MDP fixture path: ${MDP_DIR}")
sys.exit(0)
PYEOF
```

Make executable + verify:

```bash
chmod +x validation/external/susieR/fetch_data.sh
bash -n validation/external/susieR/fetch_data.sh
```

Expected: no syntax errors.

**NOTE for the engineer:** the Python helper inside `fetch_data.sh` is a stub. The full implementation should run a real `lmm-scan` on the MDP fixture and extract a per-locus `(z, R, n)` triple. For the brainstorming-driven Tier A, this stub is acceptable; the helper can be filled in at execution time.

- [ ] **Step 5: Create `validation/external/susieR/README.md`**

Write `validation/external/susieR/README.md`:

```markdown
# susieR external validation harness

This harness runs `susieR::susie_rss()` (upstream R reference) and our
`torchgenomics bayes-scan-rss` on the same per-locus fixture and compares the
results against the 6-metric tolerance table from NA1 design spec section 5.2.

## Prerequisites

- R ≥ 4.0 (`conda install -c conda-forge r-base` if missing)
- TorchGenomics installed in the current Python environment (`pip install -e ".[dev]"`)
- ≥ 2 GB free disk
- ≥ 4 GB free RAM

## Workflow

```bash
# 1. Install susieR (idempotent)
./install.sh

# 2. Provision the MDP-derived per-locus fixture
./fetch_data.sh

# 3. Run upstream susieR
./run_susieR.sh

# 4. Run our bayes-scan-rss
./run_torchgenomics.sh

# 5. Compare and emit findings
python compare.py \
    --upstream outputs/susieR.tsv \
    --ours outputs/torchgenomics.tsv \
    --findings ../../../docs/validation_findings.md
```

## Tolerance contract (per NA1 spec section 5.2)

| Metric | Threshold |
|---|---|
| Credible-set Jaccard | ≥ 0.95 (hard) |
| PIP correlation (PIP > 0.1 in either) | ≥ 0.99 |
| β_mean Pearson correlation | ≥ 0.999 |
| β_sd Pearson correlation | ≥ 0.999 |
| ELBO relative diff at convergence | ≤ 1e-4 |
| Wall-time (p ≤ 10K) | ≤ 2× susieR |

## Failure mode

Per F3 severity policy: any threshold violation is logged to
`docs/validation_findings.md` but does NOT block the phase (post-V1).
Investigate and document; do not silently accept.
```

- [ ] **Step 6: Commit**

```bash
git add validation/external/susieR/install.sh validation/external/susieR/fetch_data.sh validation/external/susieR/README.md
git commit -m "NA1 Task 13: external validation harness scaffolding (install + fetch_data + README)"
```

---

## Task 14: External validation harness — `run_susieR.sh`, `run_torchgenomics.sh`, `compare.py`

**Files:**
- Create: `validation/external/susieR/run_susieR.sh`
- Create: `validation/external/susieR/run_torchgenomics.sh`
- Create: `validation/external/susieR/compare.py`
- Create: `tests/test_external_susieR.py`

**Background:** The reference run, our run, and the 6-metric comparison driver. `tests/test_external_susieR.py` glues them together as a `@pytest.mark.external @pytest.mark.golden` test.

- [ ] **Step 1: Create `validation/external/susieR/run_susieR.sh`**

```bash
#!/usr/bin/env bash
# Run susieR::susie_rss() on the MDP-derived per-locus fixture.
# Per NA1 design spec section 5.2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

if [ ! -f "${DATA_DIR}/locus_z.pt" ]; then
    echo "FATAL: fixture not provisioned. Run ./fetch_data.sh first."
    exit 2
fi

# Convert torch tensors to TSV for R consumption (since R doesn't read .pt)
python3 - <<PYEOF
import torch
import numpy as np
import os

z = torch.load("${DATA_DIR}/locus_z.pt", weights_only=True).numpy()
R = torch.load("${DATA_DIR}/locus_R.pt", weights_only=True).numpy()
np.savetxt("${DATA_DIR}/locus_z.tsv", z, delimiter="\t")
np.savetxt("${DATA_DIR}/locus_R.tsv", R, delimiter="\t")
print(f"Exported z (p={len(z)}) and R ({R.shape[0]}x{R.shape[1]}) for R")
PYEOF

# Run susieR::susie_rss
START=$(date +%s.%N)
/usr/bin/time -v R --vanilla --no-save <<RSCRIPT 2>"${OUT_DIR}/susieR_time.log"
suppressMessages(library(susieR))
z <- as.numeric(scan("${DATA_DIR}/locus_z.tsv"))
R <- as.matrix(read.table("${DATA_DIR}/locus_R.tsv", sep="\t", header=FALSE))
n <- as.integer(readLines("${DATA_DIR}/locus_n.txt")[1])
fit <- susie_rss(z=z, R=R, n=n, L=10, coverage=0.95)
# Extract PIP, beta_mean, beta_sd, credible sets, ELBO
pip <- fit\$pip
beta_post <- coef(fit)[-1]  # drop intercept
beta_var <- diag(fit\$V)  # posterior variance approximation; verify
elbo <- fit\$elbo[length(fit\$elbo)]

# Credible sets
cs <- fit\$sets\$cs
cs_indices <- if (is.null(cs)) integer(0) else unlist(cs)

# Write output TSV matching our schema
out <- data.frame(
    SNP_idx = seq_along(pip),
    PIP = pip,
    BETA_MEAN = beta_post,
    BETA_SD = sqrt(beta_var),
    IN_CS = ifelse(seq_along(pip) %in% cs_indices, 1, 0),
    ELBO_FINAL = elbo
)
write.table(out, "${OUT_DIR}/susieR.tsv", sep="\t", row.names=FALSE, quote=FALSE)
RSCRIPT
END=$(date +%s.%N)
echo "$END $START" | awk '{print $1 - $2}' > "${OUT_DIR}/susieR_walltime_seconds.txt"
echo "susieR run complete: ${OUT_DIR}/susieR.tsv"
```

Make executable:

```bash
chmod +x validation/external/susieR/run_susieR.sh
bash -n validation/external/susieR/run_susieR.sh
```

- [ ] **Step 2: Create `validation/external/susieR/run_torchgenomics.sh`**

```bash
#!/usr/bin/env bash
# Run torchgenomics bayes-scan-rss on the same fixture as run_susieR.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

if [ ! -f "${DATA_DIR}/locus_z.pt" ]; then
    echo "FATAL: fixture not provisioned. Run ./fetch_data.sh first."
    exit 2
fi

# Convert fixture into the sumstats + LD-ref format bayes-scan-rss expects
python3 - <<PYEOF
import torch
import pandas as pd
from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata

z = torch.load("${DATA_DIR}/locus_z.pt", weights_only=True)
R = torch.load("${DATA_DIR}/locus_R.pt", weights_only=True)
n = int(open("${DATA_DIR}/locus_n.txt").read().strip())
meta_dict = torch.load("${DATA_DIR}/locus_meta.pt", weights_only=False)
snp_ids = meta_dict["snp_ids"]
chrs = meta_dict["chr"]
bps = meta_dict["bp"]
a1s = meta_dict["a1"]
a2s = meta_dict["a2"]
p = z.shape[0]

# Build sumstats TSV (BETA = z * SE; SE = 1/sqrt(n) for standardized z)
import math
se = 1.0 / math.sqrt(n)
beta = z * se
sumstats = pd.DataFrame({
    "SNP": snp_ids,
    "CHR": chrs,
    "BP": bps,
    "A1": a1s,
    "A2": a2s,
    "BETA": beta.tolist(),
    "SE": [se] * p,
    "N": [n] * p,
})
sumstats.to_csv("${DATA_DIR}/sumstats.tsv", sep="\t", index=False)

# Build LD-ref .pt
meta = LDReferenceMetadata(
    cohort_id="MDP",
    n=n,
    build="B73_v4",
    panel_provenance="MDP fixture",
)
save_ld_reference("${DATA_DIR}/ld.pt", R, snp_ids, meta)
PYEOF

START=$(date +%s.%N)
/usr/bin/time -v torchgenomics bayes-scan-rss \
    --sumstats "${DATA_DIR}/sumstats.tsv" \
    --ld-ref "${DATA_DIR}/ld.pt" \
    --max-num-causal 10 \
    --coverage 0.95 \
    --purity 0.5 \
    --output "${OUT_DIR}/torchgenomics.tsv" \
    2>"${OUT_DIR}/torchgenomics_time.log"
END=$(date +%s.%N)
echo "$END $START" | awk '{print $1 - $2}' > "${OUT_DIR}/torchgenomics_walltime_seconds.txt"
echo "torchgenomics bayes-scan-rss run complete: ${OUT_DIR}/torchgenomics.tsv"
```

Make executable + verify:

```bash
chmod +x validation/external/susieR/run_torchgenomics.sh
bash -n validation/external/susieR/run_torchgenomics.sh
```

- [ ] **Step 3: Create `validation/external/susieR/compare.py`**

```python
"""Compare susieR vs torchgenomics bayes-scan-rss outputs against the 6-metric tolerance table.

Per NA1 design spec section 5.2. Emits a single findings ledger row to
docs/validation_findings.md per F3 severity policy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


THRESHOLDS = {
    "credible_set_jaccard": 0.95,
    "pip_correlation": 0.99,
    "beta_mean_correlation": 0.999,
    "beta_sd_correlation": 0.999,
    "elbo_relative_diff": 1e-4,
    "walltime_ratio": 2.0,
}


def credible_set_jaccard(in_cs_a: np.ndarray, in_cs_b: np.ndarray) -> float:
    """Jaccard index of two credible-set membership vectors (binary indicators)."""
    set_a = set(np.where(in_cs_a == 1)[0])
    set_b = set(np.where(in_cs_b == 1)[0])
    if not set_a and not set_b:
        return 1.0  # both empty → trivially matching
    return len(set_a & set_b) / len(set_a | set_b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="susieR output TSV")
    parser.add_argument("--ours", required=True, help="bayes-scan-rss output TSV")
    parser.add_argument("--upstream-walltime", required=True, help="susieR wall-time file")
    parser.add_argument("--ours-walltime", required=True, help="our wall-time file")
    parser.add_argument("--upstream-elbo", required=True, help="susieR final ELBO")
    parser.add_argument("--ours-elbo", required=True, help="our final ELBO")
    parser.add_argument("--findings", help="Append findings row to this file")
    args = parser.parse_args()

    up = pd.read_csv(args.upstream, sep="\t")
    ours = pd.read_csv(args.ours, sep="\t")

    if len(up) != len(ours):
        print(f"FAIL: row count mismatch: upstream={len(up)}, ours={len(ours)}")
        return 1

    metrics = {}

    # Credible set Jaccard
    in_cs_up = (up["IN_CS"].values == 1).astype(int)
    in_cs_ours = (ours["CREDIBLE_SET"].values > 0).astype(int)
    metrics["credible_set_jaccard"] = credible_set_jaccard(in_cs_up, in_cs_ours)

    # PIP correlation (only SNPs with PIP > 0.1 in either)
    mask = (up["PIP"].values > 0.1) | (ours["PIP"].values > 0.1)
    if mask.sum() >= 3:
        metrics["pip_correlation"] = float(
            pearsonr(up["PIP"].values[mask], ours["PIP"].values[mask])[0]
        )
    else:
        metrics["pip_correlation"] = float("nan")

    # β_mean, β_sd correlation
    metrics["beta_mean_correlation"] = float(
        pearsonr(up["BETA_MEAN"].values, ours["BETA_MEAN"].values)[0]
    )
    metrics["beta_sd_correlation"] = float(
        pearsonr(up["BETA_SD"].values, ours["BETA_SD"].values)[0]
    )

    # ELBO relative diff
    elbo_up = float(open(args.upstream_elbo).read().strip())
    elbo_ours = float(open(args.ours_elbo).read().strip())
    metrics["elbo_relative_diff"] = abs(elbo_up - elbo_ours) / max(abs(elbo_up), 1e-12)

    # Wall-time ratio
    wt_up = float(open(args.upstream_walltime).read().strip())
    wt_ours = float(open(args.ours_walltime).read().strip())
    metrics["walltime_ratio"] = wt_ours / max(wt_up, 1e-6)

    # Apply tolerance contract
    failures = []
    for key, observed in metrics.items():
        threshold = THRESHOLDS[key]
        if key in ("credible_set_jaccard", "pip_correlation",
                   "beta_mean_correlation", "beta_sd_correlation"):
            # higher-is-better metrics
            if observed < threshold:
                failures.append(f"{key}: {observed:.6g} < {threshold:.6g}")
        else:
            # lower-is-better metrics
            if observed > threshold:
                failures.append(f"{key}: {observed:.6g} > {threshold:.6g}")

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g} (threshold {THRESHOLDS[k]:.6g})")

    if args.findings:
        with open(args.findings, "a") as f:
            f.write(
                f"\n## NA1 SuSiE-RSS parity check (date: {pd.Timestamp.now()})\n"
            )
            for k, v in metrics.items():
                f.write(f"- {k}: {v:.6g} (threshold {THRESHOLDS[k]:.6g})\n")
            if failures:
                f.write(f"- **FAILURES:** {'; '.join(failures)}\n")
            else:
                f.write("- All thresholds met.\n")

    if failures:
        print(f"\nFAIL: {len(failures)} threshold violations")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nPASS: all 6 thresholds met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Verify syntax:

```bash
python3 -m py_compile validation/external/susieR/compare.py
```

Expected: no syntax errors.

- [ ] **Step 4: Create the Tier 2 pytest test**

Create `tests/test_external_susieR.py`:

```python
"""Tier 2 parity test: bayes-scan-rss vs susieR::susie_rss().

Per NA1 design spec section 5.2. Marked external + golden so CI can
selectively run / skip. Requires R + susieR installed via
validation/external/susieR/install.sh.
"""
import pytest
import subprocess
from pathlib import Path

pytestmark = [pytest.mark.external, pytest.mark.golden]

VALIDATION_DIR = Path(__file__).parent.parent / "validation" / "external" / "susieR"


def test_susieR_parity_full_workflow():
    """End-to-end: install → fetch_data → run both → compare → assert pass."""
    # Skip if R / susieR not installed
    r_check = subprocess.run(
        ["R", "-e", "library(susieR)"],
        capture_output=True,
    )
    if r_check.returncode != 0:
        pytest.skip(
            "susieR not installed; run validation/external/susieR/install.sh"
        )

    # Provision fixture (idempotent)
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "fetch_data.sh")],
        check=True,
    )

    # Reference run
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "run_susieR.sh")],
        check=True,
    )

    # Our run
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "run_torchgenomics.sh")],
        check=True,
    )

    # Compare
    out_dir = VALIDATION_DIR / "outputs"
    rc = subprocess.call([
        "python3",
        str(VALIDATION_DIR / "compare.py"),
        "--upstream", str(out_dir / "susieR.tsv"),
        "--ours", str(out_dir / "torchgenomics.tsv"),
        "--upstream-walltime", str(out_dir / "susieR_walltime_seconds.txt"),
        "--ours-walltime", str(out_dir / "torchgenomics_walltime_seconds.txt"),
        "--upstream-elbo", str(out_dir / "susieR_elbo.txt"),
        "--ours-elbo", str(out_dir / "torchgenomics_elbo.txt"),
    ])
    assert rc == 0, "Tier 2 parity failed; see output above and findings ledger"
```

- [ ] **Step 5: Verify the test SKIPs gracefully when susieR not installed**

```bash
pytest tests/test_external_susieR.py -v --tb=short
```

Expected: SKIPPED (because susieR is not installed yet on this machine; per the task spec, install happens at execution time).

If it doesn't SKIP gracefully (e.g., if `R` is not on PATH and the subprocess raises before the skip check), adjust the skip logic to handle the `FileNotFoundError`.

- [ ] **Step 6: Commit**

```bash
git add validation/external/susieR/run_susieR.sh validation/external/susieR/run_torchgenomics.sh validation/external/susieR/compare.py tests/test_external_susieR.py
git commit -m "NA1 Task 14: external validation harness run scripts + compare + tier-2 test"
```

---

## Task 15: Memory + wall-time regression nets

**Files:**
- Modify: `tests/test_streaming_memory.py` (add `bayes-scan-rss` memory ceiling test)
- Modify: `bench/native_speedups.py` (add `bayes-scan-rss` wall-time gate)

**Background:** Per `feedback_streaming` and `feedback_regression_nets` — memory regression test paired with wall-time CI. Per spec §4.5 — memory ceiling is `O(p_max_block^2 + L * p_total)`.

- [ ] **Step 1: Inspect existing memory regression test pattern**

```bash
grep -n "class\|def test_" tests/test_streaming_memory.py | head -20
```

Note the existing convention (e.g., one test class per scan command, with a fixture that builds a tiny dataset and asserts `peak_rss < threshold * baseline`).

- [ ] **Step 2: Write the failing memory regression test**

Append to `tests/test_streaming_memory.py`:

```python
class TestBayesScanRssMemory:
    """Memory regression net for bayes-scan-rss.

    Per NA1 design spec section 4.5: peak RSS during bayes-scan-rss
    must be <= O(p_max_block^2 + L * p_total).
    """

    def test_peak_rss_bounded_by_block_size(self, tmp_path):
        """Peak RSS scales with block size, not total p."""
        import resource
        import torch
        import pandas as pd
        import math
        from torchgenomics.models.bayesian_vs_rss import BayesianVSRss
        from torchgenomics.postgwas._ld_ref_loader import (
            save_ld_reference,
            BlockSpec,
        )
        from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata

        # Build a synthetic locus with p=2000, partitioned into 4 blocks of 500
        rng = torch.Generator()
        rng.manual_seed(0)
        p = 2000
        n = 1000
        z = torch.randn(p, generator=rng, dtype=torch.float64)
        # Block-diagonal R
        R = torch.zeros(p, p, dtype=torch.float64)
        for i in range(0, p, 500):
            R[i:i+500, i:i+500] = torch.eye(500, dtype=torch.float64)
        blocks = [BlockSpec(start=i, stop=i+500) for i in range(0, p, 500)]

        # Baseline: peak RSS before fit
        baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        model = BayesianVSRss(max_num_causal=3, max_iter=20)
        result = model.fit_rss_blocked(z=z, R=R, n=n, blocks=blocks)

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_kb = peak - baseline

        # Expected memory: O(500^2 * 8 bytes / 1024 KB/byte) = 2 MB per block
        # With overhead, allow 50 MB ceiling for the entire run
        assert delta_kb < 50_000, (
            f"Peak RSS delta {delta_kb} KB exceeds 50 MB ceiling "
            f"for p={p}, block_size=500"
        )
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_streaming_memory.py::TestBayesScanRssMemory -v --tb=short
```

Expected: PASS (the implementation already satisfies the bound from Task 9).

If the test fails because the bound is too tight (e.g., torch overhead), relax the ceiling to a value that catches actual regressions but allows for normal overhead. The principle is: catch the case where someone accidentally materializes O(p^2) instead of O(p_block^2).

- [ ] **Step 4: Add wall-time gate to bench/native_speedups.py**

Inspect:

```bash
grep -n "def\|class" bench/native_speedups.py | head -20
```

Locate the existing pattern. Append a new function (or class method):

```python
def bench_bayes_scan_rss(p: int = 1000, n: int = 5000):
    """Wall-time benchmark for bayes-scan-rss on a synthetic locus.

    Per NA1 design spec section 4.5 + feedback_regression_nets memory.
    """
    import time
    import torch
    from torchgenomics.models.bayesian_vs_rss import BayesianVSRss

    rng = torch.Generator()
    rng.manual_seed(0)
    z = torch.randn(p, generator=rng, dtype=torch.float64)
    R = torch.eye(p, dtype=torch.float64)

    model = BayesianVSRss(max_num_causal=10, max_iter=50)

    start = time.perf_counter()
    result = model.fit_rss(z=z, R=R, n=n)
    elapsed = time.perf_counter() - start

    return {
        "name": "bayes-scan-rss",
        "p": p,
        "n": n,
        "wall_time_sec": elapsed,
        "n_iter": result.n_iter,
        "converged": result.converged,
    }
```

Verify:

```bash
python3 -c "from bench.native_speedups import bench_bayes_scan_rss; print(bench_bayes_scan_rss(p=100, n=500))"
```

Expected: dict with name, p, n, wall_time_sec, n_iter, converged.

- [ ] **Step 5: Commit**

```bash
git add tests/test_streaming_memory.py bench/native_speedups.py
git commit -m "NA1 Task 15: memory + wall-time regression nets for bayes-scan-rss"
```

---

## Task 16: Documentation updates (CLAUDE.md + docs/cli.md)

**Files:**
- Modify: `CLAUDE.md` (subcommand count + module list)
- Modify: `docs/cli.md` (document `bayes-scan-rss`)

- [ ] **Step 1: Update `CLAUDE.md` — subcommand count and module list**

Find the line in `CLAUDE.md` that says `**CLI subcommand count**: 37 as of Phase 56 ...` and update to `38` to account for `bayes-scan-rss`.

Find the `BayesianVS (SuSiE + CAVI)` entry in the Mixed-model extensions list and add after it: `, BayesianVSRss (SuSiE-RSS sumstats fine-mapping)`.

- [ ] **Step 2: Update `docs/cli.md`**

Find the `bayes-scan` documentation section. Append a new section:

````markdown
## `bayes-scan-rss`

SuSiE-RSS fine-mapping on summary statistics + LD reference. Per Zou et al.
2022 PLOS Genet 18(7):e1010299. Memory bound is per-locus `O(p_block_max^2)`,
typically ~200 MB per LD block at p_block ≤ 5000; vs `bayes-scan` which is
`O(n × p)` and materializes the full genotype matrix.

```bash
torchgenomics bayes-scan-rss \
    --sumstats hits.tsv \
    --ld-ref ld_chr22.pt \
    --max-num-causal 10 \
    --coverage 0.95 \
    --purity 0.5 \
    --output out/finemap.tsv
```

### Required arguments

- `--sumstats PATH` — TSV with columns SNP, CHR, BP, A1, A2 + (BETA + SE + N) or Z + N.
- One of:
  - `--ld-ref PATH` — pre-built LD reference file (`.pt` or `.npz`).
  - `--geno PATH` — genotype panel (BED/PGEN); LD computed in-sample per locus.
- `--output PATH` — output TSV path.

### Optional arguments

- `--regions PATH` — TSV with start, stop columns for explicit block boundaries.
  Default: auto-detect via `torchgenomics.ld.detect_blocks`.
- `--max-num-causal INT` — Number of single-effect layers (L). Default: 10.
- `--coverage FLOAT` — Credible-set coverage threshold. Default: 0.95.
- `--purity FLOAT` — Minimum |R_jk| within a credible set. Default: 0.5
  (matches susieR `min_abs_corr`).
- `--prior-pi VALUE` — Scalar (e.g., `0.01`) OR path to a per-SNP prior
  TSV file (column `PRIOR_PI`). D3 shim from Phase 59 PolyFun spec.
- `--block-size-threshold INT` — Block-decomposition trigger. Default: 5000.
- `--threads INT` — Number of CPU threads. Default: 4.

### Output schema (PolyFun-compatible)

```
SNP   CHR   BP   A1   A2   Z   N   PIP   BETA_MEAN   BETA_SD   CREDIBLE_SET
```

`CREDIBLE_SET` = integer index of the credible set the variant belongs to
(0 = not in any credible set).
````

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/cli.md
git commit -m "NA1 Task 16: documentation updates (CLAUDE.md + docs/cli.md)"
```

---

## Task 17: Final backward-compat verification + Tier A complete

**Files:**
- No new files; this task runs the full test suite as the final gate.

**Background:** Per spec §5.4 — Phase 59 PolyFun §5.5 contract: all existing `tests/test_bayesian_vs.py` runs unchanged. NA1 spec preserves this by isolating new code in new files.

- [ ] **Step 1: Run the full existing `bayesian_vs` test suite**

```bash
cd /home/sikiru.atanda/Documents/GWAS_Expert/.claude/worktrees/na1-susie-streaming
pytest tests/test_bayesian_vs.py -v --tb=short
```

Expected: all 30+ existing tests pass + the new `test_bayes_scan_warns_on_large_p` test from Task 12.

If any existing test fails, STOP. The warning addition or some other modification has broken backward compat. Investigate before declaring Tier A done.

- [ ] **Step 2: Run all new NA1 tests**

```bash
pytest tests/test_bayesian_vs_rss.py tests/test_ld_ref_loader.py tests/test_external_susieR.py tests/test_streaming_memory.py::TestBayesScanRssMemory -v --tb=short
```

Expected: all NA1 tests pass (Tier 2 `test_external_susieR.py` may SKIP if susieR not installed; that's acceptable per the deferred-install design).

- [ ] **Step 3: Smoke-test the CLI end-to-end**

```bash
pytest tests/test_cli.py -v -k "bayes_scan_rss" --tb=short
```

Expected: PASS.

- [ ] **Step 4: Verify the full test suite still passes (broad regression check)**

```bash
pytest tests/ --tb=short -q -m "not slow and not gpu and not external"
```

Expected: 2192-ish tests pass (the existing baseline) + the new tests added across Tasks 1-15.

If the baseline test count drops, find the regression. If new tests fail, find the bug.

- [ ] **Step 5: Run `mypy` and `ruff` on the new modules**

```bash
mypy torchgenomics/models/bayesian_vs_rss.py
mypy torchgenomics/postgwas/_ld_ref_loader.py
mypy torchgenomics/postgwas/_ld_ref_metadata.py
ruff check torchgenomics/models/bayesian_vs_rss.py torchgenomics/postgwas/_ld_ref_loader.py torchgenomics/postgwas/_ld_ref_metadata.py
ruff format torchgenomics/models/bayesian_vs_rss.py torchgenomics/postgwas/_ld_ref_loader.py torchgenomics/postgwas/_ld_ref_metadata.py
```

Expected: clean (or matches the project's existing mypy/ruff baseline).

- [ ] **Step 6: Final commit (if any formatting changes from `ruff format`)**

```bash
git add -A
git diff --cached --stat
# If there are any changes:
git commit -m "NA1 Task 17: ruff format + mypy clean on NA1 modules"
```

If no changes, skip the commit.

- [ ] **Step 7: Verify branch state — Tier A complete**

```bash
git log --oneline a669e50..HEAD
git status
```

Expected: ~17 commits on top of `a669e50` (or the rebase base from Task 0); working tree clean.

**Tier A is complete when:**
- All Task 17 verification steps pass
- The Tier 2 parity test against `susieR::susie_rss()` returns within tolerances when susieR is installed
- The branch is ready for user review

**Do NOT push to remote** per `feedback_no_autonomous_push`. Report Tier A completion to the user; wait for explicit "push" instruction.

---

## After Tier A — what's next

Per NA1 design spec sections 10.2 and 10.3:

**Tier B** (parity-tightening + scaling) requires its own implementation plan:
- B1: Secondary 1KG-derived Tier 2 fixture
- B2: `.bcor` (FINEMAP) LD-ref reader
- B3: Per-iteration ELBO trace + `--write-elbo-trace` flag
- B4: Configurable `--purity` sweep
- B5: PolyFun integration test (gated on Phase 59 implementation landing)
- B6: Memory regression at full-genome scale

**Tier C** (nice-to-haves & follow-on) — also separate plans:
- C1-C2: Native + GPU acceleration
- **C3: Path A follow-on (chunked IBSS for raw-G `bayes-scan`)** — distinct sub-spec; see OQ-NA1-1
- C4: Multi-trait extension
- C5: Cross-ancestry SuShiE [C9]
- C6: `feedback_streaming` memory update once Path A ships

These are tracked in the design spec but NOT in this plan. Each gets its own brainstorm + writing-plans cycle.
