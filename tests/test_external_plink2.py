"""External-tool harness: PLINK 2.0 vs TorchGWAS reference comparison.

These tests are skipped by default (they require the PLINK 2 binary, the
fetched fixture, and the harness reference outputs). To run:

    bash validation/external/plink2/install.sh
    bash validation/external/plink2/fetch_data.sh
    bash validation/external/plink2/run_plink2.sh
    pytest -m external tests/test_external_plink2.py -v

Each test invokes one comparison function from
``validation/external/plink2/compare.py`` and asserts every check in the
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
HARNESS = ROOT / "validation" / "external" / "plink2"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
BIN = HARNESS / "bin" / "plink2"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (`validation/` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location("plink2_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plink2_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "PLINK 2 harness artifacts not present (run install.sh + fetch_data.sh + run_plink2.sh): "
            + ", ".join(missing)
        )


def _require_binary() -> None:
    if not BIN.exists() or not os.access(BIN, os.X_OK):
        pytest.skip(f"PLINK 2 binary not installed at {BIN}; run install.sh")


@pytest.fixture(scope="module")
def compare_mod():
    # Ensure native is disabled so we exercise the pure-Python reference path
    # (matches docs/validation.md golden CI gate).
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_glm_linear(compare_mod) -> None:
    """PLINK 2 --glm linear vs TorchGWAS GLM Wald scan on MDP fixture."""
    _require_binary()
    _require_artifacts(
        DATA / "mdp.bed", DATA / "mdp.bim", DATA / "mdp.fam", DATA / "mdp_pheno.txt",
        OUT / "glm.EarHT.glm.linear",
    )
    rep = compare_mod.compare_glm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"GLM tolerances violated: {failed}"


def test_kinship_relative(compare_mod) -> None:
    """PLINK 2 --make-rel cov vs TorchGWAS grm_vanraden on MDP fixture."""
    _require_binary()
    _require_artifacts(
        OUT / "kinship.rel", OUT / "kinship.rel.id", OUT / "kinship_snps.snplist",
    )
    rep = compare_mod.compare_grm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"GRM tolerances violated: {failed}"


def test_pairwise_r2(compare_mod) -> None:
    """PLINK 2 --r2-unphased square vs TorchGWAS compute_r2_matrix on MDP."""
    _require_binary()
    _require_artifacts(
        OUT / "r2.unphased.vcor2", OUT / "r2.unphased.vcor2.vars",
    )
    rep = compare_mod.compare_r2(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"r² tolerances violated: {failed}"
