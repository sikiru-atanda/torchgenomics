"""External-tool harness: TwoSampleMR + MRPRESSO vs TorchGenomics reference.

These tests are skipped by default (they require the R packages installed
by the harness install.sh, the simulated sumstats from fetch_data.sh, and
the reference R output from run.sh). To run:

    bash validation/external/twosamplemr/install.sh
    bash validation/external/twosamplemr/fetch_data.sh
    bash validation/external/twosamplemr/run.sh
    pytest -m external tests/test_external_twosamplemr.py -v

Each test invokes one comparison function from
``validation/external/twosamplemr/compare.py`` and asserts every check in
the ComparisonReport passed. Tolerances are calibrated to observed values
from the first successful run, with margins documented in compare.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "twosamplemr"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
INSTALL_MARKER = HARNESS / ".install_marker"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (`validation/` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location(
        "twosamplemr_compare", HARNESS / "compare.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["twosamplemr_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "TwoSampleMR harness artifacts not present "
            "(run install.sh + fetch_data.sh + run.sh): "
            + ", ".join(missing)
        )


def _require_install() -> None:
    if not INSTALL_MARKER.exists():
        pytest.skip(
            f"TwoSampleMR R packages not installed (no marker at {INSTALL_MARKER}); "
            "run validation/external/twosamplemr/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Disable native so we exercise the pure-Python reference path.
    os.environ.setdefault("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_ivw(compare_mod) -> None:
    """TwoSampleMR mr_ivw vs TorchGenomics mr_ivw."""
    _require_install()
    _require_artifacts(
        DATA / "sumstats.tsv",
        DATA / "sim_truth.json",
        OUT / "twosamplemr_results.json",
    )
    rep = compare_mod.compare_ivw(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"IVW tolerances violated: {failed}"


def test_egger(compare_mod) -> None:
    """TwoSampleMR mr_egger_regression vs TorchGenomics mr_egger."""
    _require_install()
    _require_artifacts(
        DATA / "sumstats.tsv",
        DATA / "sim_truth.json",
        OUT / "twosamplemr_results.json",
    )
    rep = compare_mod.compare_egger(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"MR-Egger tolerances violated: {failed}"


def test_weighted_median(compare_mod) -> None:
    """TwoSampleMR mr_weighted_median vs TorchGenomics mr_weighted_median."""
    _require_install()
    _require_artifacts(
        DATA / "sumstats.tsv",
        DATA / "sim_truth.json",
        OUT / "twosamplemr_results.json",
    )
    rep = compare_mod.compare_weighted_median(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Weighted median tolerances violated: {failed}"


def test_mr_presso(compare_mod) -> None:
    """MRPRESSO mr_presso vs TorchGenomics mr_presso."""
    _require_install()
    _require_artifacts(
        DATA / "sumstats.tsv",
        DATA / "sim_truth.json",
        OUT / "twosamplemr_results.json",
    )
    rep = compare_mod.compare_presso(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"MR-PRESSO tolerances violated: {failed}"
