"""External-tool harness: SoyMD reference (mediation R package) vs TorchGWAS
multiomics module.

These tests are skipped by default (they require the R `mediation` package
installed by the harness install.sh, the simulated multi-omics triple from
fetch_data.sh, and the reference R output from run.sh). To run:

    bash validation/external/soymd/install.sh
    bash validation/external/soymd/fetch_data.sh
    bash validation/external/soymd/run.sh
    pytest -m external tests/test_external_soymd.py -v

Each test invokes one comparison function from
``validation/external/soymd/compare.py`` and asserts every check in
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
HARNESS = ROOT / "validation" / "external" / "soymd"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
INSTALL_MARKER = HARNESS / ".install_marker"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (`validation/` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location(
        "soymd_compare", HARNESS / "compare.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["soymd_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "SoyMD harness artifacts not present "
            "(run install.sh + fetch_data.sh + run.sh): "
            + ", ".join(missing)
        )


def _require_install() -> None:
    if not INSTALL_MARKER.exists():
        pytest.skip(
            f"SoyMD R packages not installed (no marker at {INSTALL_MARKER}); "
            "run validation/external/soymd/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Disable native so we exercise the pure-Python reference path.
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_stage_coefficients(compare_mod) -> None:
    """mediation::mediate stage-wise coefs vs TorchGWAS mediate_lmm a, b, c'."""
    _require_install()
    _require_artifacts(
        DATA / "triple.tsv",
        DATA / "K.tsv",
        DATA / "sim_truth.json",
        OUT / "mediation_results.json",
    )
    rep = compare_mod.compare_stage_coefficients(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Stage-wise coefficient tolerances violated: {failed}"


def test_acme(compare_mod) -> None:
    """mediation::mediate ACME vs TorchGWAS indirect; both vs planted truth."""
    _require_install()
    _require_artifacts(
        DATA / "triple.tsv",
        DATA / "K.tsv",
        DATA / "sim_truth.json",
        OUT / "mediation_results.json",
    )
    rep = compare_mod.compare_acme(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"ACME tolerances violated: {failed}"


def test_total_and_ade(compare_mod) -> None:
    """mediation::mediate total/ADE vs TorchGWAS total/c_prime."""
    _require_install()
    _require_artifacts(
        DATA / "triple.tsv",
        DATA / "K.tsv",
        DATA / "sim_truth.json",
        OUT / "mediation_results.json",
    )
    rep = compare_mod.compare_total_and_ade(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Total/ADE tolerances violated: {failed}"


def test_scan_mediation_smoke(compare_mod) -> None:
    """`scan_mediation` runs end-to-end on a 1-SNP × 3-mediator triple."""
    _require_install()
    _require_artifacts(
        DATA / "triple.tsv",
        DATA / "K.tsv",
        DATA / "sim_truth.json",
        OUT / "mediation_results.json",
    )
    rep = compare_mod.smoke_scan_mediation(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"scan_mediation smoke checks violated: {failed}"
