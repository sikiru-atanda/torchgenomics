"""External-tool harness: LDSC 1.0.1 vs TorchGWAS reference comparison.

These tests are skipped by default (they require the LDSC conda env, the
fetched 1000G LD-score / weights archives, and the harness reference
outputs). To run:

    bash validation/external/ldsc/install.sh
    bash validation/external/ldsc/fetch_data.sh
    bash validation/external/ldsc/run_ldsc.sh
    pytest -m external tests/test_external_ldsc.py -v

Each test invokes one comparison function from
``validation/external/ldsc/compare.py`` and asserts every check in the
ComparisonReport passed. Tolerances are calibrated to observed values from
the first successful run, with margins documented in compare.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "ldsc"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
ENV_MARKER = HARNESS / ".env_marker"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (`validation/` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location("ldsc_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ldsc_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "LDSC harness artifacts not present (run install.sh + fetch_data.sh + run_ldsc.sh): "
            + ", ".join(missing)
        )


def _require_env() -> None:
    if not ENV_MARKER.exists():
        pytest.skip(
            f"LDSC conda env not installed (no marker at {ENV_MARKER}); "
            "run validation/external/ldsc/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Ensure native is disabled so we exercise the pure-Python reference path
    # (matches docs/validation.md golden CI gate).
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_h2_trait1(compare_mod) -> None:
    """LDSC --h2 (single-pass IRWLS) vs TorchGWAS ldsc_h2 on simulated trait 1."""
    _require_env()
    _require_artifacts(
        DATA / "sim_trait1.sumstats.gz",
        DATA / "sim_truth.json",
        DATA / "ld_scores" / "LDscore.22.l2.ldscore.gz",
        DATA / "weights" / "weights.hm3_noMHC.22.l2.ldscore.gz",
        OUT / "h2_trait1.log",
    )
    rep = compare_mod.compare_h2_trait1(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"h² trait 1 tolerances violated: {failed}"


def test_h2_trait2(compare_mod) -> None:
    """LDSC --h2 (single-pass IRWLS) vs TorchGWAS ldsc_h2 on simulated trait 2."""
    _require_env()
    _require_artifacts(
        DATA / "sim_trait2.sumstats.gz",
        DATA / "sim_truth.json",
        DATA / "ld_scores" / "LDscore.22.l2.ldscore.gz",
        DATA / "weights" / "weights.hm3_noMHC.22.l2.ldscore.gz",
        OUT / "h2_trait2.log",
    )
    rep = compare_mod.compare_h2_trait2(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"h² trait 2 tolerances violated: {failed}"


def test_rg(compare_mod) -> None:
    """LDSC --rg vs TorchGWAS ldsc_rg_from_z on simulated traits 1,2."""
    _require_env()
    _require_artifacts(
        DATA / "sim_trait1.sumstats.gz",
        DATA / "sim_trait2.sumstats.gz",
        DATA / "sim_truth.json",
        DATA / "weights" / "weights.hm3_noMHC.22.l2.ldscore.gz",
        OUT / "rg.log",
    )
    rep = compare_mod.compare_rg(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"rg tolerances violated: {failed}"
