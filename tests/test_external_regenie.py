"""External-tool harness: regenie vs TorchGenomics reference comparison.

These tests are skipped by default (they require the regenie binary, the
fetched fixture, and the harness reference outputs). To run:

    bash validation/external/regenie/install.sh
    bash validation/external/regenie/fetch_data.sh
    bash validation/external/regenie/run_regenie.sh
    pytest -m external tests/test_external_regenie.py -v

Each test invokes one comparison function from
``validation/external/regenie/compare.py`` and asserts every check in the
ComparisonReport passed. Tolerances are calibrated to observed values from
the first successful run on the MDP fixture (281 samples, 2897 SNPs after
QC). See compare.py for per-tolerance rationale.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "regenie"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
BIN = HARNESS / "bin" / "regenie"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory."""
    spec = importlib.util.spec_from_file_location("regenie_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["regenie_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "regenie harness artifacts not present (run install.sh + fetch_data.sh + run_regenie.sh): "
            + ", ".join(missing)
        )


def _require_binary() -> None:
    if not BIN.exists() or not os.access(BIN, os.X_OK):
        pytest.skip(f"regenie binary not installed at {BIN}; run install.sh")


@pytest.fixture(scope="module")
def compare_mod():
    # Match the docs/validation.md golden CI gate by exercising the
    # pure-Python reference path (native disabled).
    os.environ.setdefault("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_step1_loco_predictor_finite(compare_mod) -> None:
    """regenie Step 1 LOCO predictor sanity (finite + sample subset)."""
    _require_binary()
    _require_artifacts(
        DATA / "mdp.bed", DATA / "mdp.bim", DATA / "mdp.fam",
        OUT / "step1_qt_1.loco", OUT / "step1_bin_1.loco",
    )
    rep = compare_mod.compare_step1_loco(OUT, DATA)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Step 1 LOCO sanity violated: {failed}"


def test_step2_quantitative_matches_regenie(compare_mod) -> None:
    """regenie Step 2 --qt vs TorchGenomics GLM Wald with LOCO offset on MDP."""
    _require_binary()
    _require_artifacts(
        DATA / "mdp.bed", DATA / "regenie_pheno.tsv", DATA / "regenie_covar.tsv",
        OUT / "step2_qt_EarHT.regenie", OUT / "step1_qt_1.loco",
    )
    rep = compare_mod.compare_step2_qt(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Step 2 quantitative tolerances violated: {failed}"


def test_step2_binary_firth_matches_regenie(compare_mod) -> None:
    """regenie Step 2 --bt --firth vs TorchGenomics BinaryGLM(firth=True) on MDP."""
    _require_binary()
    _require_artifacts(
        DATA / "mdp.bed", DATA / "regenie_pheno.tsv", DATA / "regenie_covar.tsv",
        OUT / "step2_bin_EarHT_bin.regenie",
    )
    rep = compare_mod.compare_step2_binary(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Step 2 binary Firth tolerances violated: {failed}"
