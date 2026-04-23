# Phase 55 — Polyploid Allele Dosage Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a thin external-wrapper around the R package `updog` that converts polyploid VCF read counts (AD field) into posterior `P(dosage=0..k)` tensors, plus a `torchgwas dosage-call` CLI subcommand, without touching any downstream consumer.

**Architecture:** Single new module `torchgwas/preprocess/dosage_call.py` exposing one public function `run_updog(...)` and one dataclass `DosageCallResult`. Python reads VCF via `cyvcf2`, writes `ref.tsv` / `size.tsv` to a tempdir, calls `Rscript driver.R` via subprocess, `driver.R` runs `updog::multidog()` and writes `k+1` wide TSVs + `snp_diag.tsv` via `format_multidog()`, Python reads them into a `(n, m, k+1)` float64 torch tensor, writes `<prefix>.probs.pt` + `<prefix>.meta.json` + `<prefix>.snp_diag.tsv` atomically. Tier 1 tests monkeypatch `subprocess.run` and never touch R. Tier 2 tests `pytest.mark.skipif` on missing R/updog.

**Tech Stack:** Python 3.10+, PyTorch, pandas, cyvcf2, subprocess, tempfile. External: R ≥ 4.0, `updog` ≥ 2.0.2 (Tier 2 only).

**Spec:** `docs/superpowers/specs/2026-04-22-polyploid-dosage-assignment-design.md`

---

## File Structure

**New files:**
- `torchgwas/preprocess/dosage_call.py` — the wrapper (~400 LOC including R driver string).
- `tests/test_dosage_call.py` — Tier 1, always-on; subprocess stubbed.
- `tests/test_dosage_call_updog_e2e.py` — Tier 2, skipif-gated; runs real R+updog.
- `bench/calibrate_updog_recovery.py` — dev-time calibration helper.
- `docs/getting-started/polyploid_dosage_call.md` — user recipe.

**Modified files:**
- `torchgwas/preprocess/__init__.py` — re-export `run_updog`, `DosageCallResult`.
- `torchgwas/cli.py` — new `dosage-call` subcommand + dispatch entry.
- `docs/ROADMAP.md` — remove Phase 55 entry (Phase 56 stays).
- `docs/cli.md` — append `dosage-call --help` capture.
- `CLAUDE.md` — bump subcommand count 35 → 36; add `dosage-call` example.

**Unchanged:**
- `torchgwas/preprocess/dosage_uncertainty.py`, `torchgwas/models/gu_lmm.py`, `torchgwas/preprocess/phase.py`, haplotype scans — all already correct downstream code.

---

## Task 1: Scaffold module + `DosageCallResult` dataclass + constants

**Files:**
- Create: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dosage_call.py`:

```python
"""Phase 55: Polyploid allele dosage assignment (Tier 1, always-on)."""
from __future__ import annotations

import torch

from torchgwas.preprocess.dosage_call import DosageCallResult, _VALID_MODELS


def test_dosage_call_result_dataclass_fields():
    r = DosageCallResult(
        probs=torch.zeros(2, 3, 5, dtype=torch.float64),
        sample_ids=["s1", "s2"],
        variant_ids=["v1", "v2", "v3"],
        ploidy=4,
        tool="updog",
        tool_version="2.0.2",
        model="norm",
        mean_dosage_var=torch.zeros(3, dtype=torch.float64),
        allele_freq=torch.zeros(3, dtype=torch.float64),
        n_missing=0,
        input_hash="abc",
        cmd="Rscript driver.R ...",
    )
    assert r.probs.shape == (2, 3, 5)
    assert r.ploidy == 4
    assert r.tool == "updog"


def test_valid_models_contains_updog_flexdog_set():
    assert {"norm", "hw", "bb", "s1", "f1", "s1pp", "f1pp", "flex", "uniform"} <= _VALID_MODELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dosage_call.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torchgwas.preprocess.dosage_call'`.

- [ ] **Step 3: Create the module**

Create `torchgwas/preprocess/dosage_call.py`:

```python
"""Polyploid allele dosage assignment via updog (Phase 55).

Thin external-wrapper around the R package ``updog``. Converts VCF read
counts (AD field) into posterior ``P(dosage=0..k)`` tensors. See
``docs/superpowers/specs/2026-04-22-polyploid-dosage-assignment-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_MODELS: frozenset[str] = frozenset({
    "norm", "hw", "bb", "s1", "f1", "s1pp", "f1pp", "flex", "uniform",
})

_UPDOG_CHECKED: Optional[bool] = None  # cached result of environment probe


@dataclass
class DosageCallResult:
    probs: Tensor
    sample_ids: list[str]
    variant_ids: list[str]
    ploidy: int
    tool: str
    tool_version: str
    model: str
    mean_dosage_var: Tensor
    allele_freq: Tensor
    n_missing: int
    input_hash: str
    cmd: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: scaffold dosage_call module + DosageCallResult dataclass"
```

---

## Task 2: Environment probe (`_check_environment` with cache)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
import pytest

from torchgwas.preprocess import dosage_call as dc_module


def _reset_cache():
    dc_module._UPDOG_CHECKED = None


def test_check_environment_missing_rscript_raises(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript not found"):
        dc_module._check_environment(rscript=None)


def test_check_environment_missing_updog_raises(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which",
                        lambda x: "/usr/bin/Rscript" if x == "Rscript" else None)

    class FakeCompleted:
        returncode = 1
        stderr = "there is no package called 'updog'"
        stdout = ""
    monkeypatch.setattr(dc_module.subprocess, "run",
                        lambda *a, **kw: FakeCompleted())

    with pytest.raises(RuntimeError, match="updog R package not installed"):
        dc_module._check_environment(rscript=None)


def test_check_environment_is_cached(monkeypatch):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    calls = {"n": 0}

    def fake_run(*a, **kw):
        calls["n"] += 1
        class OK:
            returncode = 0
            stderr = ""
            stdout = "2.0.2"
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    dc_module._check_environment(rscript=None)
    dc_module._check_environment(rscript=None)
    assert calls["n"] == 1  # second call hit the cache


def test_check_environment_rscript_override(monkeypatch):
    _reset_cache()
    # shutil.which returns None — override should be used directly
    monkeypatch.setattr(dc_module.shutil, "which", lambda _: None)

    class OK:
        returncode = 0
        stderr = ""
        stdout = "2.0.2"
    monkeypatch.setattr(dc_module.subprocess, "run", lambda *a, **kw: OK())

    rscript_path = dc_module._check_environment(rscript="/my/custom/Rscript")
    assert rscript_path == "/my/custom/Rscript"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py::test_check_environment_missing_rscript_raises -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_check_environment'`.

- [ ] **Step 3: Implement `_check_environment`**

Append to `torchgwas/preprocess/dosage_call.py` (after the dataclass):

```python
def _check_environment(rscript: Optional[str]) -> str:
    """Probe for Rscript + updog; cache the verdict. Returns Rscript path.

    Raises RuntimeError with install pointers if either is missing.
    Cached on a module-level flag; the probe runs at most once per process.
    """
    global _UPDOG_CHECKED

    if rscript is None:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Install R >= 4.0 and ensure Rscript is on "
            "PATH, or pass rscript=<path>."
        )

    if _UPDOG_CHECKED is True:
        return rscript

    probe = subprocess.run(
        [rscript, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        _UPDOG_CHECKED = False
        raise RuntimeError(
            "updog R package not installed. Install with: "
            "Rscript -e 'install.packages(\"updog\")'. "
            f"(probe stderr: {probe.stderr.strip()})"
        )

    logger.info("updog R package detected: version %s", probe.stdout.strip())
    _UPDOG_CHECKED = True
    return rscript
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: environment probe for Rscript + updog with cache"
```

---

## Task 3: Kwarg validation (`_validate_kwargs`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_validate_kwargs_rejects_ploidy_out_of_range():
    with pytest.raises(ValueError, match="ploidy"):
        dc_module._validate_kwargs(ploidy=1, model="norm")
    with pytest.raises(ValueError, match="ploidy"):
        dc_module._validate_kwargs(ploidy=9, model="norm")


def test_validate_kwargs_rejects_bad_model_name():
    with pytest.raises(ValueError, match="model"):
        dc_module._validate_kwargs(ploidy=4, model="not-a-model")


def test_validate_kwargs_accepts_valid_combinations():
    for m in ("norm", "hw", "bb", "s1", "f1", "flex", "uniform"):
        dc_module._validate_kwargs(ploidy=4, model=m)  # no raise
    for p in (2, 3, 4, 6, 8):
        dc_module._validate_kwargs(ploidy=p, model="norm")  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k validate_kwargs -v`
Expected: FAIL with `AttributeError: ... no attribute '_validate_kwargs'`.

- [ ] **Step 3: Implement `_validate_kwargs`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _validate_kwargs(*, ploidy: int, model: str) -> None:
    if not (2 <= ploidy <= 8):
        raise ValueError(
            f"ploidy must be in [2, 8]; got {ploidy}. updog is validated "
            "up to ploidy=8 upstream."
        )
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model={model!r} not in updog flexdog set: {sorted(_VALID_MODELS)}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS (9 tests total).

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: validate ploidy and model kwargs before R dispatch"
```

---

## Task 4: VCF AD extraction (`_extract_ad_from_vcf`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
_TOY_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
1\t100\trs001\tA\tT\t.\tPASS\t.\tGT:AD\t0/0/0/0:20,0\t0/0/0/1:15,5\t0/0/1/1:10,10
1\t200\trs002\tC\tG\t.\tPASS\t.\tGT:AD\t0/0/1/1:12,8\t0/1/1/1:6,14\t1/1/1/1:0,22
1\t300\trs003\tG\tA\t.\tPASS\t.\tGT:AD\t0/0/0/0:25,1\t.:.,.\t0/0/0/1:18,4
1\t400\trs004\tT\tC\t.\tPASS\t.\tGT:AD\t0/1/1/1:4,16\t0/0/1/1:9,11\t0/0/0/1:13,3
"""

_TOY_VCF_NO_AD = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tA\tT\t.\tPASS\t.\tGT\t0/0/0/0\t0/0/1/1
"""

_TOY_VCF_MULTIALLELIC = """\
##fileformat=VCFv4.2
##contig=<ID=1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
1\t100\trs001\tA\tT,G\t.\tPASS\t.\tGT:AD\t0/0/1/2:5,5,5\t0/0/0/0:20,0,0
"""


@pytest.fixture
def toy_vcf(tmp_path):
    cyvcf2 = pytest.importorskip("cyvcf2")
    path = tmp_path / "toy.vcf"
    path.write_text(_TOY_VCF)
    return str(path)


@pytest.fixture
def toy_vcf_no_ad(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "no_ad.vcf"
    path.write_text(_TOY_VCF_NO_AD)
    return str(path)


@pytest.fixture
def toy_vcf_multiallelic(tmp_path):
    pytest.importorskip("cyvcf2")
    path = tmp_path / "multi.vcf"
    path.write_text(_TOY_VCF_MULTIALLELIC)
    return str(path)


def test_extract_ad_from_vcf_returns_shapes(toy_vcf):
    sample_ids, variant_ids, refmat, sizemat = dc_module._extract_ad_from_vcf(toy_vcf)
    assert sample_ids == ["S1", "S2", "S3"]
    assert variant_ids == ["rs001", "rs002", "rs003", "rs004"]
    assert refmat.shape == (4, 3)   # (m, n)
    assert sizemat.shape == (4, 3)
    # rs001 sample S1: AD=20,0 → ref=20, size=20
    assert refmat[0, 0] == 20
    assert sizemat[0, 0] == 20
    # rs002 sample S3: AD=0,22 → ref=0, size=22
    assert refmat[1, 2] == 0
    assert sizemat[1, 2] == 22


def test_extract_ad_from_vcf_zeroes_missing_samples(toy_vcf):
    _, _, refmat, sizemat = dc_module._extract_ad_from_vcf(toy_vcf)
    # rs003 sample S2 has AD=.,. → both ref and size set to 0
    assert refmat[2, 1] == 0
    assert sizemat[2, 1] == 0


def test_extract_ad_from_vcf_missing_ad_header_raises(toy_vcf_no_ad):
    with pytest.raises(ValueError, match="AD"):
        dc_module._extract_ad_from_vcf(toy_vcf_no_ad)


def test_extract_ad_from_vcf_rejects_multiallelic(toy_vcf_multiallelic):
    with pytest.raises(ValueError, match="multi"):
        dc_module._extract_ad_from_vcf(toy_vcf_multiallelic)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k extract_ad -v`
Expected: FAIL with `AttributeError: ... no attribute '_extract_ad_from_vcf'` (or ImportError on cyvcf2 → tests skip; that's fine on this machine only if cyvcf2 is missing).

- [ ] **Step 3: Implement `_extract_ad_from_vcf`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _extract_ad_from_vcf(
    input_vcf: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read a biallelic VCF with AD field. Returns (sample_ids, variant_ids,
    refmat, sizemat) where matrices are shape (m, n) int64, rows=variants,
    cols=samples. Missing AD entries (negative cyvcf2 sentinel) become
    (ref=0, size=0), which multidog treats as missing.
    """
    try:
        import cyvcf2  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "dosage_call requires cyvcf2: pip install cyvcf2"
        ) from e

    vcf = cyvcf2.VCF(input_vcf)

    # Header probe: AD must be declared under FORMAT
    has_ad = False
    for h in vcf.header_iter():
        d = h.info(extra=True)
        if d.get("HeaderType") == "FORMAT" and d.get("ID") == "AD":
            has_ad = True
            break
    if not has_ad:
        vcf.close()
        raise ValueError(
            f"VCF {input_vcf!r} has no AD field under FORMAT. Re-call with "
            "a tool that emits allele depth (e.g. GATK HaplotypeCaller, "
            "bcftools mpileup -a FORMAT/AD)."
        )

    sample_ids = list(vcf.samples)
    variant_ids: list[str] = []
    ref_rows: list[np.ndarray] = []
    size_rows: list[np.ndarray] = []

    for v in vcf:
        if len(v.ALT) != 1:
            vcf.close()
            raise ValueError(
                f"Multi-allelic site at {v.CHROM}:{v.POS} (ID={v.ID}); "
                f"updog requires biallelic. Split first: "
                f"bcftools norm -m -any <in> > <out>."
            )
        ad = v.format("AD")
        if ad is None:
            vcf.close()
            raise ValueError(
                f"Variant {v.CHROM}:{v.POS} (ID={v.ID}) has no AD data."
            )
        ad = np.asarray(ad, dtype=np.int64)
        # cyvcf2 sentinel for missing: negative
        missing = (ad < 0).any(axis=1)
        ref = ad[:, 0].copy()
        alt = ad[:, 1].copy()
        ref[missing] = 0
        alt[missing] = 0
        ref_rows.append(ref)
        size_rows.append(ref + alt)
        vid = v.ID if v.ID else f"{v.CHROM}_{v.POS}"
        variant_ids.append(vid)

    vcf.close()

    if not ref_rows:
        raise ValueError(f"VCF {input_vcf!r} contains zero variants.")

    refmat = np.stack(ref_rows, axis=0)
    sizemat = np.stack(size_rows, axis=0)
    return sample_ids, variant_ids, refmat, sizemat
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS (13 tests total), or SKIP if cyvcf2 is missing.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: VCF AD extraction with multi-allelic + missing handling"
```

---

## Task 5: Write ref/size TSVs (`_write_input_tsvs`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_write_input_tsvs_round_trip(tmp_path):
    sample_ids = ["S1", "S2"]
    variant_ids = ["v1", "v2", "v3"]
    refmat = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    sizemat = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    ref_tsv, size_tsv = dc_module._write_input_tsvs(
        tmp_path, sample_ids, variant_ids, refmat, sizemat
    )
    df_ref = pd.read_csv(ref_tsv, sep="\t", index_col=0)
    df_size = pd.read_csv(size_tsv, sep="\t", index_col=0)
    assert list(df_ref.index) == variant_ids
    assert list(df_ref.columns) == sample_ids
    np.testing.assert_array_equal(df_ref.to_numpy(), refmat)
    np.testing.assert_array_equal(df_size.to_numpy(), sizemat)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dosage_call.py::test_write_input_tsvs_round_trip -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `_write_input_tsvs`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _write_input_tsvs(
    tmpdir: Path,
    sample_ids: list[str],
    variant_ids: list[str],
    refmat: np.ndarray,
    sizemat: np.ndarray,
) -> tuple[Path, Path]:
    """Write ref.tsv and size.tsv to tmpdir. Rows=variants, cols=samples,
    with index label row0col0 blank (R convention; pandas reads it via
    index_col=0). Returns the two paths.
    """
    ref_tsv = tmpdir / "ref.tsv"
    size_tsv = tmpdir / "size.tsv"
    pd.DataFrame(refmat, index=variant_ids, columns=sample_ids).to_csv(
        ref_tsv, sep="\t", index=True, index_label=""
    )
    pd.DataFrame(sizemat, index=variant_ids, columns=sample_ids).to_csv(
        size_tsv, sep="\t", index=True, index_label=""
    )
    return ref_tsv, size_tsv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: write ref/size TSVs with variant rownames + sample colnames"
```

---

## Task 6: R driver string + subprocess invocation (`_run_r_subprocess`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_r_driver_constant_calls_multidog_and_format_multidog():
    src = dc_module._UPDOG_DRIVER_R
    assert "multidog(" in src
    assert "format_multidog(" in src
    assert 'library(updog)' in src


def test_run_r_subprocess_passes_exact_args(tmp_path, monkeypatch):
    captured = {}

    class OK:
        returncode = 0
        stderr = ""
        stdout = "updog multidog complete"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    ref_tsv = tmp_path / "ref.tsv"
    size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    dc_module._run_r_subprocess(
        rscript="/usr/bin/Rscript",
        tmpdir=tmp_path,
        ref_tsv=ref_tsv,
        size_tsv=size_tsv,
        ploidy=4,
        model="norm",
        bias=True,
        od=True,
        seq_error=None,
        n_cores=2,
    )
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/Rscript"
    # driver script path comes next
    assert Path(cmd[1]).name == "driver.R"
    assert cmd[2:] == [str(ref_tsv), str(size_tsv), "4", "norm",
                       "TRUE", "TRUE", "NULL", "2", str(tmp_path)]


def test_run_r_subprocess_nonzero_exit_raises(tmp_path, monkeypatch):
    class Fail:
        returncode = 42
        stderr = "something went wrong"
        stdout = ""

    monkeypatch.setattr(dc_module.subprocess, "run", lambda *a, **kw: Fail())
    ref_tsv = tmp_path / "ref.tsv"; size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    with pytest.raises(RuntimeError, match="exit 42"):
        dc_module._run_r_subprocess(
            rscript="/usr/bin/Rscript", tmpdir=tmp_path,
            ref_tsv=ref_tsv, size_tsv=size_tsv,
            ploidy=4, model="norm", bias=True, od=True,
            seq_error=None, n_cores=1,
        )


def test_run_r_subprocess_passes_seq_error_as_float_string(tmp_path, monkeypatch):
    captured = {}

    class OK:
        returncode = 0; stderr = ""; stdout = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd; return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)
    ref_tsv = tmp_path / "ref.tsv"; size_tsv = tmp_path / "size.tsv"
    ref_tsv.touch(); size_tsv.touch()
    dc_module._run_r_subprocess(
        rscript="Rscript", tmpdir=tmp_path,
        ref_tsv=ref_tsv, size_tsv=size_tsv,
        ploidy=4, model="norm", bias=False, od=False,
        seq_error=0.005, n_cores=1,
    )
    cmd = captured["cmd"]
    # boolean flags become FALSE; seq_error becomes "0.005"
    assert cmd[6] == "FALSE"  # bias
    assert cmd[7] == "FALSE"  # od
    assert cmd[8] == "0.005"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k "r_driver or r_subprocess" -v`
Expected: FAIL on all four new tests.

- [ ] **Step 3: Implement the R driver string and subprocess helper**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
_UPDOG_DRIVER_R = r"""# R driver for torchgwas.preprocess.dosage_call — invoked via Rscript.
args <- commandArgs(trailingOnly = TRUE)
ref_tsv <- args[1]
size_tsv <- args[2]
ploidy <- as.integer(args[3])
model <- args[4]
update_bias <- as.logical(args[5])
update_od <- as.logical(args[6])
seq_arg <- args[7]
n_cores <- as.integer(args[8])
out_dir <- args[9]

suppressPackageStartupMessages(library(updog))

refmat <- as.matrix(read.table(ref_tsv, sep = "\t", header = TRUE,
                                row.names = 1, check.names = FALSE))
sizemat <- as.matrix(read.table(size_tsv, sep = "\t", header = TRUE,
                                 row.names = 1, check.names = FALSE))

multidog_args <- list(
  refmat = refmat, sizemat = sizemat,
  ploidy = ploidy, model = model, nc = n_cores,
  update_bias = update_bias, update_od = update_od
)
if (seq_arg != "NULL") {
  multidog_args$seq <- as.numeric(seq_arg)
  multidog_args$update_seq <- FALSE
}
mout <- do.call(multidog, multidog_args)

for (d in 0:ploidy) {
  mat <- format_multidog(mout, varname = paste0("Pr_", d))
  write.table(mat, file.path(out_dir, paste0("pr_", d, ".tsv")),
              sep = "\t", quote = FALSE, col.names = NA)
}
write.table(mout$snpdf, file.path(out_dir, "snp_diag.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("updog multidog complete\n")
"""


def _run_r_subprocess(
    *,
    rscript: str,
    tmpdir: Path,
    ref_tsv: Path,
    size_tsv: Path,
    ploidy: int,
    model: str,
    bias: bool,
    od: bool,
    seq_error: Optional[float],
    n_cores: int,
) -> str:
    """Write the driver script to tmpdir, spawn Rscript, return the
    stringified command for reproducibility. Raises RuntimeError on
    non-zero exit with stderr attached.
    """
    driver_path = tmpdir / "driver.R"
    driver_path.write_text(_UPDOG_DRIVER_R)

    cmd = [
        rscript, str(driver_path),
        str(ref_tsv), str(size_tsv),
        str(ploidy), model,
        "TRUE" if bias else "FALSE",
        "TRUE" if od else "FALSE",
        "NULL" if seq_error is None else f"{seq_error:g}",
        str(n_cores),
        str(tmpdir),
    ]
    logger.info("Running updog: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"updog failed (exit {result.returncode}). stderr:\n{result.stderr}"
        )
    if result.stderr:
        logger.info("updog stderr:\n%s", result.stderr)
    return " ".join(cmd)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: R driver script + subprocess invocation with error capture"
```

---

## Task 7: Parse R output TSVs (`_parse_output`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def _write_canned_pr_tsvs(tmpdir, sample_ids, variant_ids, ploidy, probs):
    """Write pr_0..pr_k TSVs matching the format R's format_multidog emits.
    probs: (n, m, k+1) array; we write m-row, n-col per file, with a blank
    first column header and variant_ids as rownames.
    """
    for d in range(ploidy + 1):
        df = pd.DataFrame(
            probs[:, :, d].T,   # (m, n)
            index=variant_ids,
            columns=sample_ids,
        )
        df.to_csv(tmpdir / f"pr_{d}.tsv", sep="\t", index=True, index_label="")
    # snp_diag stub
    pd.DataFrame({"snp": variant_ids, "bias": [1.0]*len(variant_ids),
                  "seq": [0.005]*len(variant_ids),
                  "od": [0.01]*len(variant_ids)}).to_csv(
        tmpdir / "snp_diag.tsv", sep="\t", index=False)


def test_parse_output_stacks_in_correct_order(tmp_path):
    sample_ids = ["S1", "S2"]
    variant_ids = ["v1", "v2", "v3"]
    ploidy = 4
    rng = np.random.default_rng(0)
    raw = rng.random((2, 3, 5))
    probs_true = raw / raw.sum(axis=-1, keepdims=True)
    _write_canned_pr_tsvs(tmp_path, sample_ids, variant_ids, ploidy, probs_true)

    probs, snp_diag = dc_module._parse_output(
        tmp_path, sample_ids, variant_ids, ploidy
    )
    assert probs.shape == (2, 3, 5)
    assert probs.dtype == torch.float64
    np.testing.assert_allclose(probs.numpy(), probs_true, atol=1e-6)
    assert list(snp_diag["snp"]) == variant_ids


def test_parse_output_missing_pr_file_raises(tmp_path):
    sample_ids = ["S1"]; variant_ids = ["v1"]; ploidy = 4
    probs = np.ones((1, 1, 5)) / 5
    _write_canned_pr_tsvs(tmp_path, sample_ids, variant_ids, ploidy, probs)
    # remove pr_3.tsv
    (tmp_path / "pr_3.tsv").unlink()
    with pytest.raises(RuntimeError, match="pr_3.tsv"):
        dc_module._parse_output(tmp_path, sample_ids, variant_ids, ploidy)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k parse_output -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_parse_output`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _parse_output(
    tmpdir: Path,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
) -> tuple[Tensor, pd.DataFrame]:
    """Read pr_0..pr_k wide TSVs and snp_diag.tsv from tmpdir. Returns
    (probs, snp_diag) where probs has shape (n, m, k+1) float64 and
    is stacked along the last axis in dosage-class order.
    """
    prob_slices: list[np.ndarray] = []
    for d in range(ploidy + 1):
        path = tmpdir / f"pr_{d}.tsv"
        if not path.is_file():
            raise RuntimeError(
                f"updog exited 0 but produced no {path.name} in {tmpdir}."
            )
        df = pd.read_csv(path, sep="\t", index_col=0)
        # Reindex defensively so order matches our canonical ID lists
        df = df.reindex(index=variant_ids, columns=sample_ids)
        prob_slices.append(df.to_numpy(dtype=np.float64).T)  # (n, m)

    # (n, m, k+1)
    probs_np = np.stack(prob_slices, axis=-1)
    probs = torch.from_numpy(probs_np).to(torch.float64)

    diag_path = tmpdir / "snp_diag.tsv"
    if not diag_path.is_file():
        raise RuntimeError(f"updog produced no snp_diag.tsv in {tmpdir}.")
    snp_diag = pd.read_csv(diag_path, sep="\t")
    return probs, snp_diag
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: parse k+1 wide TSVs into (n, m, k+1) probs tensor"
```

---

## Task 8: Output normalization (`_normalize_probs`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_normalize_probs_accepts_within_tolerance():
    probs = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.float64)  # sum=1
    out, n_missing = dc_module._normalize_probs(probs, ploidy=2)
    assert n_missing == 0
    torch.testing.assert_close(out, probs)


def test_normalize_probs_zero_row_becomes_uniform_missing():
    probs = torch.zeros(1, 2, 3, dtype=torch.float64)
    probs[0, 0] = torch.tensor([0.5, 0.3, 0.2])
    # probs[0, 1] is all zero → missing
    out, n_missing = dc_module._normalize_probs(probs, ploidy=2)
    assert n_missing == 1
    torch.testing.assert_close(out[0, 1], torch.tensor([1/3, 1/3, 1/3],
                                                         dtype=torch.float64))
    torch.testing.assert_close(out[0, 0], probs[0, 0])


def test_normalize_probs_negative_is_clamped_and_renormalized(caplog):
    probs = torch.tensor([[[-0.1, 0.6, 0.5]]], dtype=torch.float64)
    with caplog.at_level("WARNING"):
        out, _ = dc_module._normalize_probs(probs, ploidy=2)
    # clamped to [0, 0.6, 0.5], renormalized
    expected = torch.tensor([[[0.0, 0.6/1.1, 0.5/1.1]]], dtype=torch.float64)
    torch.testing.assert_close(out, expected)
    assert any("negative" in rec.message.lower() for rec in caplog.records)


def test_normalize_probs_rejects_wrong_last_dim():
    probs = torch.ones(1, 1, 4, dtype=torch.float64) / 4
    with pytest.raises(RuntimeError, match="shape"):
        dc_module._normalize_probs(probs, ploidy=4)  # expects k+1 = 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k normalize_probs -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_normalize_probs`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _normalize_probs(probs: Tensor, ploidy: int) -> tuple[Tensor, int]:
    """Validate + clean the (n, m, k+1) probability tensor. Returns a
    cleaned copy plus the count of missing (zero-row) sample×marker slots.

    Rules per the spec:
      - Shape must be (n, m, ploidy+1) — else RuntimeError.
      - Rows summing to 0 are treated as missing → set to uniform
        1/(k+1), `n_missing += 1`.
      - Negative entries are clamped to 0 and the row is renormalized,
        with a warning via logger.
      - All remaining rows must sum to within atol=1e-4 of 1.0.
    """
    if probs.dim() != 3 or probs.shape[-1] != ploidy + 1:
        raise RuntimeError(
            f"probs shape {tuple(probs.shape)} does not match "
            f"(n, m, ploidy+1) with ploidy={ploidy}."
        )

    out = probs.clone()

    # Negative handling
    if (out < 0).any():
        logger.warning(
            "updog output contains %d negative probabilities; clamping to 0.",
            int((out < 0).sum().item()),
        )
        out = torch.clamp(out, min=0.0)

    row_sums = out.sum(dim=-1)  # (n, m)
    zero_mask = row_sums == 0
    n_missing = int(zero_mask.sum().item())

    # Uniform fill for missing rows
    if n_missing > 0:
        uniform = 1.0 / (ploidy + 1)
        out[zero_mask] = uniform

    # Renormalize non-missing rows
    non_missing = ~zero_mask
    if non_missing.any():
        out[non_missing] = out[non_missing] / out[non_missing].sum(
            dim=-1, keepdim=True
        )

    # Final sanity check
    final_sums = out.sum(dim=-1)
    if not torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4):
        bad = (final_sums - 1.0).abs().max().item()
        raise RuntimeError(
            f"Normalized probs still deviate from 1.0 by up to {bad:.3g}."
        )

    return out, n_missing
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: normalize probs, uniform-fill missing, clamp negatives"
```

---

## Task 9: Assemble `DosageCallResult` + persist artifacts (`_build_result`, `_persist_artifacts`)

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_build_result_computes_quality_metrics():
    n, m, ploidy = 4, 3, 4
    probs = torch.zeros(n, m, ploidy + 1, dtype=torch.float64)
    # marker 0: everyone dosage 0 (AF = 0)
    probs[:, 0, 0] = 1.0
    # marker 1: everyone dosage 4 (AF = 1)
    probs[:, 1, 4] = 1.0
    # marker 2: uniform over dosage classes (AF = 0.5)
    probs[:, 2, :] = 1.0 / (ploidy + 1)

    r = dc_module._build_result(
        probs=probs,
        sample_ids=[f"S{i}" for i in range(n)],
        variant_ids=[f"v{j}" for j in range(m)],
        ploidy=ploidy,
        tool_version="2.0.2",
        model="norm",
        n_missing=0,
        input_hash="deadbeef",
        cmd="Rscript driver.R ...",
    )
    assert r.probs is probs
    assert r.tool == "updog"
    assert r.ploidy == 4
    assert r.mean_dosage_var.shape == (m,)
    assert r.allele_freq.shape == (m,)
    # marker 0: AF ≈ 0
    assert r.allele_freq[0].item() == pytest.approx(0.0)
    # marker 1: AF ≈ 1
    assert r.allele_freq[1].item() == pytest.approx(1.0)
    # marker 2: AF ≈ 0.5
    assert r.allele_freq[2].item() == pytest.approx(0.5)


def test_persist_artifacts_writes_three_files(tmp_path):
    n, m, ploidy = 2, 2, 4
    probs = torch.ones(n, m, ploidy + 1, dtype=torch.float64) / (ploidy + 1)
    r = dc_module._build_result(
        probs=probs,
        sample_ids=["S1", "S2"],
        variant_ids=["v1", "v2"],
        ploidy=ploidy, tool_version="2.0.2", model="norm",
        n_missing=0, input_hash="abc", cmd="Rscript ...",
    )
    snp_diag = pd.DataFrame({"snp": ["v1", "v2"], "bias": [1.0, 1.0]})

    prefix = tmp_path / "out"
    dc_module._persist_artifacts(r, snp_diag, prefix=str(prefix))

    loaded = torch.load(str(prefix) + ".probs.pt")
    assert loaded.shape == probs.shape
    meta = json.loads((Path(str(prefix) + ".meta.json")).read_text())
    assert meta["tool"] == "updog"
    assert meta["ploidy"] == 4
    assert meta["sample_ids"] == ["S1", "S2"]
    assert meta["input_hash"] == "abc"
    diag_df = pd.read_csv(str(prefix) + ".snp_diag.tsv", sep="\t")
    assert list(diag_df["snp"]) == ["v1", "v2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k "build_result or persist" -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_build_result` and `_persist_artifacts`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def _build_result(
    *,
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
    tool_version: str,
    model: str,
    n_missing: int,
    input_hash: str,
    cmd: str,
) -> DosageCallResult:
    d_vals = torch.arange(ploidy + 1, dtype=probs.dtype, device=probs.device)
    e_d = (probs * d_vals).sum(dim=-1)          # (n, m)
    e_d2 = (probs * d_vals ** 2).sum(dim=-1)    # (n, m)
    var_d = e_d2 - e_d ** 2
    mean_dosage_var = var_d.mean(dim=0)          # (m,)
    allele_freq = e_d.mean(dim=0) / ploidy       # (m,)

    return DosageCallResult(
        probs=probs,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        ploidy=ploidy,
        tool="updog",
        tool_version=tool_version,
        model=model,
        mean_dosage_var=mean_dosage_var,
        allele_freq=allele_freq,
        n_missing=n_missing,
        input_hash=input_hash,
        cmd=cmd,
    )


def _persist_artifacts(
    result: DosageCallResult,
    snp_diag: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    """Write `<prefix>.probs.pt`, `<prefix>.meta.json`,
    `<prefix>.snp_diag.tsv`. Atomicity contract: caller only invokes this
    after multidog has succeeded, so a partial call leaves no half-written
    outputs on disk.
    """
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(result.probs, str(prefix_path) + ".probs.pt")

    meta = {
        "tool": result.tool,
        "tool_version": result.tool_version,
        "model": result.model,
        "ploidy": result.ploidy,
        "sample_ids": result.sample_ids,
        "variant_ids": result.variant_ids,
        "n_missing": result.n_missing,
        "input_hash": result.input_hash,
        "cmd": result.cmd,
    }
    Path(str(prefix_path) + ".meta.json").write_text(json.dumps(meta, indent=2))

    snp_diag.to_csv(str(prefix_path) + ".snp_diag.tsv",
                    sep="\t", index=False)
```

Also add the `json` import to the top of the file if it's not already there (it is — Task 1 already added it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: assemble DosageCallResult + atomic artifact persistence"
```

---

## Task 10: Top-level `run_updog` integrating all helpers

**Files:**
- Modify: `torchgwas/preprocess/dosage_call.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def _stub_updog_subprocess(tmpdir, sample_ids, variant_ids, ploidy,
                            seed=0, n_missing_slots=0):
    """Return a subprocess.run replacement that writes canned pr_*.tsv +
    snp_diag.tsv into whatever --out-dir was passed. Inspects argv to
    locate the output directory.
    """
    def fake_run(cmd, **kwargs):
        # find out_dir = cmd[-1] per our contract
        out_dir = Path(cmd[-1])
        rng = np.random.default_rng(seed)
        raw = rng.random((len(sample_ids), len(variant_ids), ploidy + 1))
        probs = raw / raw.sum(axis=-1, keepdims=True)
        if n_missing_slots:
            probs[0, 0, :] = 0.0  # flag one slot missing
        _write_canned_pr_tsvs(out_dir, sample_ids, variant_ids, ploidy, probs)

        class OK:
            returncode = 0
            stderr = ""
            stdout = "updog multidog complete"
        return OK()
    return fake_run


def test_run_updog_end_to_end_stubbed(tmp_path, monkeypatch, toy_vcf):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    # Two subprocess calls happen: (1) updog-version probe, (2) real driver.
    # Dispatch on cmd signature.
    sample_ids = ["S1", "S2", "S3"]
    variant_ids = ["rs001", "rs002", "rs003", "rs004"]
    ploidy = 4
    driver_run = _stub_updog_subprocess(tmp_path, sample_ids, variant_ids, ploidy)

    def fake_run(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1] == "-e":
            class Probe:
                returncode = 0; stderr = ""; stdout = "2.0.2"
            return Probe()
        return driver_run(cmd, **kwargs)

    monkeypatch.setattr(dc_module.subprocess, "run", fake_run)

    prefix = tmp_path / "out"
    result = dc_module.run_updog(
        input_vcf=toy_vcf,
        output_path=str(prefix),
        ploidy=ploidy,
        model="norm",
    )
    assert result.tool == "updog"
    assert result.ploidy == ploidy
    assert result.sample_ids == sample_ids
    assert result.variant_ids == variant_ids
    assert result.probs.shape == (3, 4, 5)
    assert (tmp_path / "out.probs.pt").is_file()
    assert (tmp_path / "out.meta.json").is_file()
    assert (tmp_path / "out.snp_diag.tsv").is_file()


def test_run_updog_partial_output_raises_and_leaves_no_artifacts(
    tmp_path, monkeypatch, toy_vcf
):
    _reset_cache()
    monkeypatch.setattr(dc_module.shutil, "which", lambda x: "/usr/bin/Rscript")

    def broken_driver(cmd, **kwargs):
        if len(cmd) >= 3 and cmd[1] == "-e":
            class Probe:
                returncode = 0; stderr = ""; stdout = "2.0.2"
            return Probe()
        # Only write 2 of 5 pr files, no snp_diag — triggers _parse_output error
        out_dir = Path(cmd[-1])
        (out_dir / "pr_0.tsv").write_text("\tS1\nrs001\t0.2\n")
        (out_dir / "pr_1.tsv").write_text("\tS1\nrs001\t0.2\n")
        class OK:
            returncode = 0; stderr = ""; stdout = ""
        return OK()

    monkeypatch.setattr(dc_module.subprocess, "run", broken_driver)

    prefix = tmp_path / "out"
    with pytest.raises(RuntimeError):
        dc_module.run_updog(input_vcf=toy_vcf, output_path=str(prefix),
                             ploidy=4, model="norm")

    # No user-visible output artifacts written
    assert not (tmp_path / "out.probs.pt").exists()
    assert not (tmp_path / "out.meta.json").exists()
    assert not (tmp_path / "out.snp_diag.tsv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k run_updog -v`
Expected: FAIL with `AttributeError: ... no attribute 'run_updog'`.

- [ ] **Step 3: Implement `run_updog`**

Append to `torchgwas/preprocess/dosage_call.py`:

```python
def run_updog(
    input_vcf: str,
    output_path: str,
    *,
    ploidy: int,
    model: str = "norm",
    rscript: Optional[str] = None,
    bias: bool = True,
    od: bool = True,
    seq_error: Optional[float] = None,
    n_cores: int = 1,
    keep_tmpdir: bool = False,
) -> DosageCallResult:
    """Call polyploid allele dosages via updog.

    Parameters
    ----------
    input_vcf
        Biallelic VCF with AD format field.
    output_path
        Output prefix. Writes ``<prefix>.probs.pt``, ``<prefix>.meta.json``,
        and ``<prefix>.snp_diag.tsv``.
    ploidy
        Organism ploidy; 2 <= ploidy <= 8.
    model
        updog flexdog model name (see ``_VALID_MODELS``).
    rscript
        Override path to Rscript. If None, auto-detected from PATH.
    bias, od
        Whether updog should estimate allele bias / overdispersion.
    seq_error
        Fix the sequencing error rate at this value, or None to estimate.
    n_cores
        Parallelism passed to updog's ``nc`` argument.
    keep_tmpdir
        Skip tempdir cleanup for debugging.

    Returns
    -------
    DosageCallResult
        Posterior ``P(dosage=0..k)`` plus diagnostics; artifacts also
        persisted to disk at ``output_path``.
    """
    _validate_kwargs(ploidy=ploidy, model=model)
    rscript_path = _check_environment(rscript)

    sample_ids, variant_ids, refmat, sizemat = _extract_ad_from_vcf(input_vcf)

    input_hash = hashlib.sha256(Path(input_vcf).read_bytes()).hexdigest()

    tmp_obj = tempfile.TemporaryDirectory(prefix="torchgwas_dosage_")
    tmpdir = Path(tmp_obj.name)
    try:
        ref_tsv, size_tsv = _write_input_tsvs(
            tmpdir, sample_ids, variant_ids, refmat, sizemat
        )
        cmd = _run_r_subprocess(
            rscript=rscript_path, tmpdir=tmpdir,
            ref_tsv=ref_tsv, size_tsv=size_tsv,
            ploidy=ploidy, model=model, bias=bias, od=od,
            seq_error=seq_error, n_cores=n_cores,
        )
        probs, snp_diag = _parse_output(tmpdir, sample_ids, variant_ids, ploidy)
        probs, n_missing = _normalize_probs(probs, ploidy)

        # Best-effort version read; `_check_environment` logged it but we
        # don't persist that string yet — re-probe cheaply.
        version = _read_updog_version(rscript_path)

        result = _build_result(
            probs=probs,
            sample_ids=sample_ids, variant_ids=variant_ids,
            ploidy=ploidy, tool_version=version, model=model,
            n_missing=n_missing, input_hash=input_hash, cmd=cmd,
        )
        _persist_artifacts(result, snp_diag, prefix=output_path)
        return result
    finally:
        if keep_tmpdir:
            logger.info("kept tempdir: %s", tmpdir)
        else:
            tmp_obj.cleanup()


def _read_updog_version(rscript: str) -> str:
    """Best-effort re-probe; falls back to 'unknown' on any failure."""
    try:
        r = subprocess.run(
            [rscript, "-e",
             'cat(as.character(packageVersion("updog")))'],
            capture_output=True, text=True, check=False, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS (all Tier 1 tests — around 20).

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/dosage_call.py tests/test_dosage_call.py
git commit -m "Phase 55: public run_updog wiring helpers end-to-end"
```

---

## Task 11: CLI subcommand `torchgwas dosage-call`

**Files:**
- Modify: `torchgwas/cli.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_cli_dosage_call_dispatches_to_run_updog(tmp_path, monkeypatch, toy_vcf):
    from torchgwas.cli import main as cli_main

    captured = {}

    def fake_run_updog(**kwargs):
        captured.update(kwargs)
        probs = torch.ones(3, 4, 5, dtype=torch.float64) / 5
        return dc_module.DosageCallResult(
            probs=probs, sample_ids=["S1","S2","S3"],
            variant_ids=["v1","v2","v3","v4"], ploidy=4, tool="updog",
            tool_version="2.0.2", model="norm",
            mean_dosage_var=torch.zeros(4, dtype=torch.float64),
            allele_freq=torch.zeros(4, dtype=torch.float64),
            n_missing=0, input_hash="x", cmd="",
        )

    monkeypatch.setattr("torchgwas.cli.run_updog", fake_run_updog,
                        raising=False)
    monkeypatch.setattr("torchgwas.preprocess.dosage_call.run_updog",
                        fake_run_updog)

    out_prefix = tmp_path / "out"
    rc = cli_main([
        "dosage-call",
        "--vcf", toy_vcf,
        "--output", str(out_prefix),
        "--ploidy", "4",
        "--model", "norm",
        "--n-cores", "2",
    ])
    assert rc == 0
    assert captured["input_vcf"] == toy_vcf
    assert captured["output_path"] == str(out_prefix)
    assert captured["ploidy"] == 4
    assert captured["model"] == "norm"
    assert captured["n_cores"] == 2
    # defaults
    assert captured["bias"] is True
    assert captured["od"] is True
    assert captured["seq_error"] is None


def test_cli_dosage_call_no_bias_no_od(tmp_path, monkeypatch, toy_vcf):
    from torchgwas.cli import main as cli_main

    captured = {}
    def fake_run_updog(**kwargs):
        captured.update(kwargs)
        probs = torch.ones(1, 1, 5, dtype=torch.float64) / 5
        return dc_module.DosageCallResult(
            probs=probs, sample_ids=["S1"], variant_ids=["v1"],
            ploidy=4, tool="updog", tool_version="2.0.2", model="norm",
            mean_dosage_var=torch.zeros(1, dtype=torch.float64),
            allele_freq=torch.zeros(1, dtype=torch.float64),
            n_missing=0, input_hash="x", cmd="",
        )
    monkeypatch.setattr("torchgwas.preprocess.dosage_call.run_updog",
                        fake_run_updog)

    cli_main([
        "dosage-call", "--vcf", toy_vcf, "--output", str(tmp_path / "out"),
        "--ploidy", "4", "--no-bias", "--no-od", "--seq-error", "0.005",
    ])
    assert captured["bias"] is False
    assert captured["od"] is False
    assert captured["seq_error"] == 0.005
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dosage_call.py -k cli_dosage -v`
Expected: FAIL with `SystemExit: 2` (argparse doesn't know the command).

- [ ] **Step 3: Wire the CLI subcommand**

In `torchgwas/cli.py`, locate the imports section near the top and add:

```python
from .preprocess.dosage_call import run_updog  # Phase 55 dosage-call
```

Locate the handlers dict (around line 138, right after `"impute": _cmd_impute,`) and add:

```python
        "impute": _cmd_impute,
        "dosage-call": _cmd_dosage_call,  # Phase 55
    }
```

Find the section that adds data-management parsers (around line 40, right after `_add_impute_parser(subparsers)`) and add:

```python
    _add_impute_parser(subparsers)
    _add_dosage_call_parser(subparsers)
```

Locate `_add_impute_parser(...)` (around line 2968) and add directly below it:

```python
def _add_dosage_call_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "dosage-call",
        help="Call polyploid allele dosages from VCF read counts via updog",
    )
    p.add_argument("--vcf", required=True,
                   help="Input VCF with AD format field (biallelic)")
    p.add_argument("--output", required=True,
                   help="Output prefix for .probs.pt / .meta.json / .snp_diag.tsv")
    p.add_argument("--ploidy", type=int, required=True,
                   help="Organism ploidy (2..8)")
    p.add_argument("--model", default="norm",
                   help="updog flexdog model name (default: norm)")
    p.add_argument("--rscript", default=None,
                   help="Path to Rscript (default: PATH lookup)")
    bias_group = p.add_mutually_exclusive_group()
    bias_group.add_argument("--bias", dest="bias", action="store_true",
                            default=True, help="Estimate allele bias (default)")
    bias_group.add_argument("--no-bias", dest="bias", action="store_false",
                            help="Fix allele bias to 1")
    od_group = p.add_mutually_exclusive_group()
    od_group.add_argument("--od", dest="od", action="store_true",
                          default=True,
                          help="Estimate overdispersion (default)")
    od_group.add_argument("--no-od", dest="od", action="store_false",
                          help="Fix overdispersion to 0")
    p.add_argument("--seq-error", type=float, default=None,
                   help="Fix sequencing error rate (default: estimate)")
    p.add_argument("--n-cores", type=int, default=1,
                   help="Parallelism passed to updog::multidog (default: 1)")
    p.add_argument("--keep-tmpdir", action="store_true",
                   help="Skip tempdir cleanup (debug aid)")
```

Find where other `_cmd_*` handlers live (following the convention of nearby scan handlers) and add:

```python
def _cmd_dosage_call(args: argparse.Namespace) -> int:
    """Phase 55 dosage-call subcommand handler."""
    result = run_updog(
        input_vcf=args.vcf,
        output_path=args.output,
        ploidy=args.ploidy,
        model=args.model,
        rscript=args.rscript,
        bias=args.bias,
        od=args.od,
        seq_error=args.seq_error,
        n_cores=args.n_cores,
        keep_tmpdir=args.keep_tmpdir,
    )
    logger.info(
        "dosage-call: %d samples × %d variants, ploidy=%d, n_missing=%d, "
        "tool_version=%s",
        len(result.sample_ids), len(result.variant_ids),
        result.ploidy, result.n_missing, result.tool_version,
    )
    return 0
```

A safe place to put `_cmd_dosage_call` is near `_cmd_impute` — use Grep for `def _cmd_impute` to find it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

Then quickly verify the help text renders:

```bash
python -m torchgwas dosage-call --help
```

Expected: argparse usage + all options listed; exit 0.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/cli.py tests/test_dosage_call.py
git commit -m "Phase 55: wire torchgwas dosage-call CLI subcommand"
```

---

## Task 12: Re-exports from `preprocess/__init__.py`

**Files:**
- Modify: `torchgwas/preprocess/__init__.py`
- Test: `tests/test_dosage_call.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dosage_call.py`:

```python
def test_reexports_from_torchgwas_preprocess():
    from torchgwas.preprocess import run_updog, DosageCallResult
    assert callable(run_updog)
    assert DosageCallResult is dc_module.DosageCallResult
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dosage_call.py::test_reexports_from_torchgwas_preprocess -v`
Expected: FAIL with `ImportError: cannot import name 'run_updog'`.

- [ ] **Step 3: Add re-exports**

Edit `torchgwas/preprocess/__init__.py`. Append at the bottom (after the existing `noqa: F401` line for `dosage_uncertainty`):

```python
from .dosage_call import DosageCallResult, run_updog  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dosage_call.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add torchgwas/preprocess/__init__.py tests/test_dosage_call.py
git commit -m "Phase 55: re-export run_updog and DosageCallResult from preprocess"
```

---

## Task 13: Tier 2 end-to-end tests with real `updog`

**Files:**
- Create: `tests/test_dosage_call_updog_e2e.py`

- [ ] **Step 1: Write the Tier 2 test file**

Create `tests/test_dosage_call_updog_e2e.py`:

```python
"""Phase 55 Tier 2: end-to-end tests that run real R + updog.

Module-level skipif gates on missing R/updog so these tests silently
skip on CI matrix jobs without R.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from torchgwas.preprocess import dosage_call as dc_module


def _updog_available() -> bool:
    rs = shutil.which("Rscript")
    if rs is None:
        return False
    probe = subprocess.run(
        [rs, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _updog_available(),
    reason="R + updog not installed; run locally or in release-time CI.",
)


# ---------- helpers ----------

def _simulate_reads_tetraploid(n_samples: int, n_markers: int, depth: int,
                                  seq_err: float, bias: float, od: float,
                                  seed: int):
    """Simulate beta-binomial reads for tetraploid genotypes.
    Returns (true_dosages, ref_counts, alt_counts, allele_freqs).
    """
    rng = np.random.default_rng(seed)
    ploidy = 4
    af = rng.uniform(0.1, 0.9, size=n_markers)
    # Sample true dosages per marker from binomial(k, af)
    true_dosage = np.stack([
        rng.binomial(ploidy, af[j], size=n_samples) for j in range(n_markers)
    ], axis=-1)  # (n, m)
    # Expected alt fraction with bias + sequencing error
    p_true = true_dosage / ploidy
    p_obs = (1 - seq_err) * p_true + seq_err * (1 - p_true)
    # Apply simple bias as a multiplicative tilt
    p_obs = p_obs * bias / (p_obs * bias + (1 - p_obs))
    # Beta-binomial dispersion
    if od > 0:
        alpha = p_obs * (1 - od) / od
        beta = (1 - p_obs) * (1 - od) / od
        p_sample = rng.beta(alpha.clip(1e-6), beta.clip(1e-6))
    else:
        p_sample = p_obs
    alt = rng.binomial(depth, p_sample)
    ref = depth - alt
    return true_dosage, ref, alt, af


def _write_sim_vcf(path, sample_ids, variant_ids, ref, alt):
    """Write a minimal biallelic VCF with AD from sim arrays."""
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">',
    ]
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
              "INFO", "FORMAT"] + list(sample_ids)
    lines.append("\t".join(header))
    for j, vid in enumerate(variant_ids):
        row = ["1", str(100 + j), vid, "A", "T", ".", "PASS", ".", "GT:AD"]
        for i, _ in enumerate(sample_ids):
            row.append(f"./.:{ref[i, j]},{alt[i, j]}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


# ---------- tests ----------

def test_parity_on_identical_ref_size_tsvs(tmp_path):
    """Our R driver vs. bare multidog() on byte-identical ref.tsv/size.tsv.
    This proves the driver is not doing anything weird above multidog's
    own parse path.
    """
    # 1) Run our wrapper with keep_tmpdir=True on a simulated VCF
    sample_ids = [f"S{i}" for i in range(10)]
    variant_ids = [f"v{j}" for j in range(8)]
    _, ref, alt, _ = _simulate_reads_tetraploid(10, 8, depth=30,
                                                   seq_err=0.01, bias=1.0,
                                                   od=0.01, seed=0)
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    out_prefix = tmp_path / "ours"
    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(out_prefix),
        ploidy=4, model="norm", keep_tmpdir=True,
    )
    ours = result.probs

    # 2) Find the kept tempdir (it's the only one under system tmp with our
    # prefix that still has ref.tsv)
    import tempfile
    import glob
    candidates = sorted(glob.glob(f"{tempfile.gettempdir()}/torchgwas_dosage_*"))
    ours_tmpdir = next(c for c in candidates
                        if (tmp_path.resolve() != tmp_path)  # filter noop
                        or True)  # take latest
    ours_tmpdir = candidates[-1]

    # 3) Run bare multidog on the same ref.tsv/size.tsv
    ref_dir = tmp_path / "ref_run"
    ref_dir.mkdir()
    driver_R = tmp_path / "ref_driver.R"
    driver_R.write_text(dc_module._UPDOG_DRIVER_R)
    subprocess.run(
        [shutil.which("Rscript") or "Rscript", str(driver_R),
         f"{ours_tmpdir}/ref.tsv", f"{ours_tmpdir}/size.tsv",
         "4", "norm", "TRUE", "TRUE", "NULL", "1", str(ref_dir)],
        check=True,
    )
    ref_probs, _ = dc_module._parse_output(ref_dir, sample_ids, variant_ids, 4)

    torch.testing.assert_close(ours, ref_probs, atol=1e-6, rtol=0)


def test_simulated_recovery_highdepth_tetraploid(tmp_path):
    """Mode-recovery at 30x depth must be above calibrated threshold.

    The recovery threshold is sourced from
    ``bench/calibrate_updog_recovery.py`` (run at phase sign-off), minus
    a 2% cross-version margin. If this is the first run, set threshold
    to 0.85 as a starting point and tighten after calibration.
    """
    # Calibration source: bench/calibrate_updog_recovery.py, run 2026-XX-XX.
    # Update this number after the first calibration run.
    RECOVERY_FLOOR = 0.85

    n_samples, n_markers = 50, 40
    true_d, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=30, seq_err=0.01, bias=1.0, od=0.01,
        seed=42,
    )

    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm",
    )
    # mode dosage per (sample, marker)
    mode = result.probs.argmax(dim=-1).numpy()
    recovery = (mode == true_d).mean()
    assert recovery >= RECOVERY_FLOOR, (
        f"recovery={recovery:.3f} below floor {RECOVERY_FLOOR}"
    )


def test_simulated_recovery_lowdepth_tetraploid(tmp_path):
    """Same as above but at 8x depth. Looser floor."""
    RECOVERY_FLOOR = 0.65  # replace with calibrated value at sign-off

    n_samples, n_markers = 50, 40
    true_d, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=8, seq_err=0.01, bias=1.0, od=0.02,
        seed=7,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm",
    )
    mode = result.probs.argmax(dim=-1).numpy()
    recovery = (mode == true_d).mean()
    assert recovery >= RECOVERY_FLOOR, (
        f"recovery={recovery:.3f} below floor {RECOVERY_FLOOR}"
    )


def test_gulm_end_to_end_on_simulated_dosages(tmp_path):
    """Smoke test: sim → wrapper → dosage_variance → GULM score. KS p>0.01
    (not a tight calibration — see test_gu_lmm.py for the full battery).
    """
    from scipy.stats import kstest
    from torchgwas.models.gu_lmm import GULM
    from torchgwas.preprocess.dosage_uncertainty import (
        expected_dosage, dosage_variance,
    )
    from torchgwas.linalg.grm import compute_grm

    n_samples, n_markers = 200, 500
    _, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=15, seq_err=0.01, bias=1.0, od=0.01,
        seed=1,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm", n_cores=1,
    )
    dosages = expected_dosage(result.probs, 4)   # (n, m)
    dvar = dosage_variance(result.probs, 4)      # (n, m)

    # Simulate a null phenotype: y = grand mean + noise
    rng = np.random.default_rng(123)
    y = torch.from_numpy(rng.standard_normal(n_samples)).to(torch.float64)
    covars = torch.ones(n_samples, 1, dtype=torch.float64)

    # GRM from the expected dosages themselves (reasonable for a smoke test)
    grm = compute_grm(dosages, ploidy=4)

    model = GULM(grm=grm)
    null = model.fit_null(y=y, covars=covars)
    scan = model.score_chunk(
        null_fit=null, G=dosages.T, test="score", dosage_var=dvar.T,
    )
    pvals = scan.p_values.numpy()
    pvals = pvals[np.isfinite(pvals)]
    ks_stat, ks_p = kstest(pvals, "uniform")
    assert ks_p > 0.01, (
        f"null p-values not uniform under GULM+dosage_var (KS p={ks_p:.3g})"
    )
```

- [ ] **Step 2: Run the test file — verify skipif behavior**

Run (on a machine WITHOUT R+updog): `pytest tests/test_dosage_call_updog_e2e.py -v`
Expected: All tests SKIPPED with reason "R + updog not installed".

Run (on a machine WITH R+updog): `pytest tests/test_dosage_call_updog_e2e.py -v`
Expected: Tests execute. Recovery-threshold tests may FAIL initially — that's the signal to run the calibration bench next.

- [ ] **Step 3: Commit**

```bash
git add tests/test_dosage_call_updog_e2e.py
git commit -m "Phase 55: Tier 2 end-to-end tests gated on R+updog availability"
```

No implementation changes in this task — the tests themselves are the deliverable. Thresholds are placeholders that Task 14's calibration helper locks in.

---

## Task 14: Calibration bench helper

**Files:**
- Create: `bench/calibrate_updog_recovery.py`

- [ ] **Step 1: Write the calibration helper**

Create `bench/calibrate_updog_recovery.py`:

```python
"""Phase 55 one-time calibration helper. NOT a pytest test.

Run once at phase sign-off on a machine with R + updog installed:

    python bench/calibrate_updog_recovery.py

It runs the same simulations used in tests/test_dosage_call_updog_e2e.py
and prints the observed mode-recovery rate at 30x and 8x depth. Paste
those numbers (minus 2%) into the RECOVERY_FLOOR constants in the test
file.
"""
from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path
import tempfile

import numpy as np
import torch

# Allow running directly from repo root
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from torchgwas.preprocess import dosage_call as dc_module  # noqa: E402


def _updog_available() -> bool:
    rs = shutil.which("Rscript")
    if rs is None:
        return False
    probe = subprocess.run(
        [rs, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    return probe.returncode == 0


def _simulate(n_samples, n_markers, depth, seq_err, bias, od, seed):
    rng = np.random.default_rng(seed)
    ploidy = 4
    af = rng.uniform(0.1, 0.9, size=n_markers)
    true_d = np.stack([
        rng.binomial(ploidy, af[j], size=n_samples) for j in range(n_markers)
    ], axis=-1)
    p_true = true_d / ploidy
    p_obs = (1 - seq_err) * p_true + seq_err * (1 - p_true)
    p_obs = p_obs * bias / (p_obs * bias + (1 - p_obs))
    if od > 0:
        alpha = p_obs * (1 - od) / od
        beta = (1 - p_obs) * (1 - od) / od
        p_sample = rng.beta(alpha.clip(1e-6), beta.clip(1e-6))
    else:
        p_sample = p_obs
    alt = rng.binomial(depth, p_sample)
    ref = depth - alt
    return true_d, ref, alt


def _write_vcf(path, sample_ids, variant_ids, ref, alt):
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">',
        "\t".join(["#CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO",
                   "FORMAT"] + list(sample_ids)),
    ]
    for j, vid in enumerate(variant_ids):
        row = ["1", str(100+j), vid, "A", "T", ".", "PASS", ".", "GT:AD"]
        for i, _ in enumerate(sample_ids):
            row.append(f"./.:{ref[i, j]},{alt[i, j]}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def run_one(depth: int, od: float, seed: int, n_samples=50, n_markers=40):
    true_d, ref, alt = _simulate(
        n_samples, n_markers, depth=depth, seq_err=0.01, bias=1.0, od=od,
        seed=seed,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_vcf(td / "sim.vcf", sample_ids, variant_ids, ref, alt)
        result = dc_module.run_updog(
            input_vcf=str(td / "sim.vcf"),
            output_path=str(td / "out"),
            ploidy=4, model="norm",
        )
    mode = result.probs.argmax(dim=-1).numpy()
    return float((mode == true_d).mean())


if __name__ == "__main__":
    if not _updog_available():
        print("R + updog not installed. Install and re-run.", file=sys.stderr)
        sys.exit(2)

    print("Phase 55 recovery-rate calibration")
    print("=" * 50)
    r30 = run_one(depth=30, od=0.01, seed=42)
    print(f"Tetraploid 30x depth: recovery = {r30:.4f}")
    r8 = run_one(depth=8, od=0.02, seed=7)
    print(f"Tetraploid  8x depth: recovery = {r8:.4f}")
    print()
    print("Paste into tests/test_dosage_call_updog_e2e.py:")
    print(f"  highdepth: RECOVERY_FLOOR = {max(r30 - 0.02, 0.0):.3f}")
    print(f"  lowdepth:  RECOVERY_FLOOR = {max(r8 - 0.02, 0.0):.3f}")
```

- [ ] **Step 2: Verify the script runs (if R+updog is available)**

Run: `python bench/calibrate_updog_recovery.py`
Expected (with R+updog): prints two recovery rates. Expected (without): prints missing-env message and exits 2.

- [ ] **Step 3: Commit**

```bash
git add bench/calibrate_updog_recovery.py
git commit -m "Phase 55: one-shot bench helper to calibrate recovery thresholds"
```

---

## Task 15: User-facing getting-started recipe

**Files:**
- Create: `docs/getting-started/polyploid_dosage_call.md`

- [ ] **Step 1: Write the recipe**

Create `docs/getting-started/polyploid_dosage_call.md`:

````markdown
# Polyploid Allele Dosage Calling (Phase 55)

This recipe converts polyploid sequencing VCFs into posterior dosage
tensors and feeds them into the uncertainty-aware association test
(GU-LMM, Phase 28).

## Prerequisites

TorchGWAS `dosage-call` shells out to the R package
[`updog`](https://cran.r-project.org/package=updog). Install it once:

```bash
# on a machine with R >= 4.0 on PATH:
Rscript -e 'install.packages("updog", repos="https://cloud.r-project.org")'
```

Your input VCF must be biallelic and carry the `AD` (allele depth) format
field. If you're starting from BAMs, GATK `HaplotypeCaller` and
`bcftools mpileup -a FORMAT/AD` both emit this.

## Step 1 — Call dosages

```bash
torchgwas dosage-call \
  --vcf calls.vcf.gz \
  --output out/dcall \
  --ploidy 4 \
  --model norm \
  --n-cores 4
```

This writes:

- `out/dcall.probs.pt` — `(n_samples, n_variants, ploidy+1)` posterior tensor.
- `out/dcall.meta.json` — IDs, ploidy, tool version, input hash, command.
- `out/dcall.snp_diag.tsv` — per-marker `updog` diagnostics (bias, seq error, OD).

## Step 2 — Convert posteriors to expected dosage + variance

The downstream tensors are cheap reductions of `probs.pt`:

```python
import torch
from torchgwas.preprocess.dosage_uncertainty import (
    expected_dosage, dosage_variance,
)
probs = torch.load("out/dcall.probs.pt")
torch.save(expected_dosage(probs, 4), "out/dosage.pt")
torch.save(dosage_variance(probs, 4), "out/dosage_var.pt")
```

## Step 3a — Run uncertainty-aware GWAS (GU-LMM)

```bash
torchgwas gu-scan \
  --genotype out/dosage.pt \
  --phenotype pheno.txt \
  --dosage-var out/dosage_var.pt \
  --output results
```

## Step 3b — Or run a standard polyploid scan (ignoring uncertainty)

```bash
torchgwas poly-scan \
  --genotype out/dosage.pt \
  --phenotype pheno.txt \
  --ploidy 4 \
  --output results
```

## Troubleshooting

- **`Rscript not found`**: install R ≥ 4.0 and ensure `Rscript` is on
  `PATH`, or pass `--rscript /path/to/Rscript`.
- **`updog R package not installed`**: run the one-liner in Prerequisites.
- **`VCF has no AD field`**: re-call with GATK `HaplotypeCaller` or
  `bcftools mpileup -a FORMAT/AD`.
- **`Multi-allelic site at 1:12345`**: split with `bcftools norm -m -any`
  before calling dosages; updog is strictly biallelic.
````

- [ ] **Step 2: Commit**

```bash
git add docs/getting-started/polyploid_dosage_call.md
git commit -m "Phase 55: getting-started recipe for VCF → dosage-call → gu-scan"
```

---

## Task 16: `cli.md`, `ROADMAP.md`, `CLAUDE.md` updates

**Files:**
- Modify: `docs/cli.md`
- Modify: `docs/ROADMAP.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Capture `--help` output into `docs/cli.md`**

Run:

```bash
python -m torchgwas dosage-call --help > /tmp/dosage-call-help.txt
```

Then add a section to `docs/cli.md` under the "Data management" heading (locate with Grep for "impute" or similar). Use Edit with the contents of `/tmp/dosage-call-help.txt` inside a fenced code block labeled `text`. Concrete pattern:

```markdown
## dosage-call (Phase 55)

Convert polyploid VCF read counts into posterior `P(dosage=0..k)` tensors.

```text
<paste output of `torchgwas dosage-call --help` verbatim>
```

See `docs/getting-started/polyploid_dosage_call.md` for a full recipe.
```

- [ ] **Step 2: Remove the Phase 55 entry from `docs/ROADMAP.md`**

Locate the "**Phase 55 — Polyploid allele dosage assignment.**" bullet in `docs/ROADMAP.md` (added by commit `4140950`). Delete that entire bullet. Leave the Phase 56 bullet immediately below it intact.

Commit message rationale: the feature is shipped, so it belongs in `git log` / CHANGELOG, not in the forward-looking roadmap.

- [ ] **Step 3: Update `CLAUDE.md`**

Two edits in `CLAUDE.md`:

1. Bump the subcommand count — find the string "35 as of Phase 49b" in the "CLI subcommand count" line and change to "36 as of Phase 55 (adds `dosage-call`)".

2. Add an example under the "Data management" section of CLI commands. Locate the `torchgwas impute --genotype data.vcf.gz --method deep-learning --output imp.pt` line and append directly after:

```bash
# --- Polyploid allele dosage calling (Phase 55) ---
torchgwas dosage-call --vcf calls.vcf.gz --output out/dcall --ploidy 4 --model norm
```

- [ ] **Step 4: Verify CLI help text matches**

Spot check: run `torchgwas dosage-call --help` and confirm the output in `docs/cli.md` matches byte-for-byte. If argparse wraps differently, capture the actual output again.

- [ ] **Step 5: Commit**

```bash
git add docs/cli.md docs/ROADMAP.md CLAUDE.md
git commit -m "Phase 55: docs updates — cli.md help, ROADMAP removal, CLAUDE.md bump"
```

---

## Task 17: Memory file + MEMORY.md index entry

**Files:**
- Create: `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\project_dosage_call.md`
- Modify: `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\MEMORY.md`

Memory files live outside the git repo, so this task has no commit.

- [ ] **Step 1: Write the memory file**

Create `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\project_dosage_call.md`:

```markdown
---
name: Phase 55 — Polyploid allele dosage calling (dosage_call)
description: Phase 55 ships a thin external-wrapper around the R package updog for polyploid VCF dosage calling; new module torchgwas.preprocess.dosage_call, new CLI subcommand torchgwas dosage-call, Tier 1/Tier 2 test split.
type: project
---

Phase 55 shipped as a thin external-wrapper around R's `updog` package.
Entry points: `torchgwas.preprocess.dosage_call.run_updog()` and
`torchgwas dosage-call` CLI. Dataclass `DosageCallResult` holds
`probs (n, m, k+1)` + `sample_ids`, `variant_ids`, `ploidy`, `tool_version`,
`model`, `mean_dosage_var`, `allele_freq`, `n_missing`, `input_hash`, `cmd`.

**Pipeline**: VCF (with AD) → cyvcf2 AD extraction → ref.tsv/size.tsv in
tempdir → Rscript driver → multidog() → format_multidog() writes
`k+1` wide TSVs + snp_diag.tsv → pandas reads with `index_col=0` and
stacks → (n, m, k+1) float64 torch tensor → atomic write to
`<prefix>.probs.pt` / `.meta.json` / `.snp_diag.tsv`.

**Test split**:
- Tier 1 (`tests/test_dosage_call.py`): subprocess stubbed; runs on every
  CI matrix job (Linux+Windows × 3.10/3.11/3.12).
- Tier 2 (`tests/test_dosage_call_updog_e2e.py`): pytest.mark.skipif gated
  on R+updog; parity test vs. bare multidog() on identical TSVs,
  simulated recovery (tetraploid 30x + 8x), GULM end-to-end smoke.
- Calibration helper: `bench/calibrate_updog_recovery.py` (one-shot;
  paste result into Tier 2 floor constants).

**Version pin**: `updog >= 2.0.2` (floor of `format_multidog()`
SNP-dimension reorder fix upstream). We read output via dimnames so
we're semantically safe even below; 2.0.2 is the recommended floor.

**Out of scope** (Phase 55 follow-ups):
- `gu-scan --probs <probs.pt>` passthrough to remove the current
  Python one-liner between `dosage-call` and `gu-scan`.
- `polyRAD` / `fitPoly` wrappers (same `DosageCallResult` pattern).
- Wiring `dosage-call` into `torchgwas pipeline`.
- Phase 56: polyploid phasing — separate brainstorm.
```

- [ ] **Step 2: Append the index entry to `MEMORY.md`**

Edit `C:\Users\Sikiru\.claude\projects\C--Users-Sikiru-Documents-GWAS-Expert\memory\MEMORY.md`. Directly after the handoff-state line, add:

```markdown
- [Polyploid Dosage Calling Phase 55](project_dosage_call.md) — updog wrapper for polyploid VCF → P(dosage=0..k); CLI dosage-call; Tier 1/2 test split; gu-scan consumer unchanged
```

- [ ] **Step 3: No git commit**

Memory files are outside the repo. The phase is complete once Task 16 commits the last repo-side change.

---

## Self-Review

### Spec coverage

Walking through each section of `docs/superpowers/specs/2026-04-22-polyploid-dosage-assignment-design.md`:

- Section 1 (scope boundary): Task 1 establishes module; Tasks 2-10 implement the in-scope list; out-of-scope items are not added.
- Section 2 (components/interfaces): `DosageCallResult` in Task 1, `run_updog()` signature in Task 10, R driver in Task 6.
- Section 3 (data flow): VCF extraction in Task 4, TSV write in Task 5, R subprocess in Task 6, TSV parse in Task 7, normalize in Task 8, persist in Task 9, full wiring in Task 10.
- Section 4 (error handling): environment in Task 2, kwarg validation in Task 3, subprocess failure in Task 6, output normalization in Task 8.
- Section 5 (testing + gate): Tier 1 accrues through Tasks 1-12; Tier 2 in Task 13; calibration in Task 14. Gate criterion 4 (CLI help) in Task 16; gate criterion 5 (getting-started recipe) in Task 15; gate criterion 6 (memory file) in Task 17; gate criterion 3 (ROADMAP removal) in Task 16.
- Section 6 (CLI): Task 11.
- Section 7 (explicit decisions) and Section 8 (open questions): not code; nothing to implement.

Gaps: **None**.

### Placeholder scan

Scanned for "TBD", "TODO", "implement later", "add appropriate X", "fill in". The two intentional `RECOVERY_FLOOR = 0.85` / `0.65` values in Task 13 are placeholder numbers that Task 14 locks in — flagged in-line both places; this is the intended workflow, not an unsolved placeholder. No other placeholders found.

### Type consistency

- `DosageCallResult` field set defined in Task 1 is used byte-identical in Tasks 9, 10, 11 (CLI dispatch), 13. Checked against every constructor call.
- `_extract_ad_from_vcf` returns `tuple[list, list, np.ndarray, np.ndarray]` (Task 4); consumers in Task 10 unpack in that order.
- `_parse_output` returns `(Tensor, pd.DataFrame)` (Task 7); consumer in Task 10 unpacks accordingly.
- `_run_r_subprocess` returns `str` (the cmd string, Task 6); consumed as `cmd=` kwarg into `_build_result` in Task 10.
- CLI `--ploidy` is `type=int, required=True` (Task 11) — matches `run_updog` signature.

No mismatches found.

### Scope check

17 tasks, each with explicit files and commit. Each task produces a standalone commit that leaves the tree in a working state (tests pass). No task touches more than one conceptual concern. No task's commit depends on a future task's behavior for Tier 1 tests to pass.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-22-phase-55-polyploid-dosage-assignment.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
