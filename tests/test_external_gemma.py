"""External-tool harness: GEMMA 0.98.5 vs TorchGWAS reference comparison.

These tests are skipped by default (they require the GEMMA binary symlinked
into ``validation/external/gemma/bin/``, the fetched fixture, and the harness
reference outputs). To run::

    bash validation/external/gemma/install.sh
    bash validation/external/gemma/fetch_data.sh
    bash validation/external/gemma/run_gemma.sh
    pytest -m external tests/test_external_gemma.py -v

Each test invokes one comparison function from
``validation/external/gemma/compare.py`` and asserts every check in the
ComparisonReport passed. Tolerances are pinned to ``docs/validation.md`` §16
and mirror the existing ``tests/test_golden_gemma.py`` golden contract — this
harness re-runs GEMMA + TorchGWAS end-to-end to confirm zero regression from
the 9 Pillar A V1-core / V1-platform fixes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "gemma"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
BIN = HARNESS / "bin" / "gemma"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (``validation/`` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location("gemma_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gemma_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "GEMMA harness artifacts not present (run install.sh + fetch_data.sh + run_gemma.sh): "
            + ", ".join(missing)
        )


def _require_binary() -> None:
    if not BIN.exists() or not os.access(BIN, os.X_OK):
        pytest.skip(f"GEMMA binary not symlinked at {BIN}; run install.sh")


@pytest.fixture(scope="module")
def compare_mod():
    # Ensure native is disabled so we exercise the pure-Python reference path
    # (matches docs/validation.md golden CI gate).
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_grm_round_trip(compare_mod) -> None:
    """GEMMA -gk 1 cXX kinship round-trips exactly through TorchGWAS dtype."""
    _require_binary()
    _require_artifacts(OUT / "mdp_kinship.cXX.txt")
    rep = compare_mod.compare_grm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"GRM tolerances violated: {failed}"


def test_lmm_single_trait_matches(compare_mod) -> None:
    """GEMMA -lmm 4 (Wald + LRT + score) vs TG SingleTraitLMM at §16 tolerances."""
    _require_binary()
    _require_artifacts(
        OUT / "mdp_lmm_all.assoc.txt",
        OUT / "mdp_lmm_all.log.txt",
        OUT / "mdp_kinship.cXX.txt",
        DATA / "mdp_traits.txt",
        DATA / "mdp_numeric.txt",
        DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_lmm_single(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"LMM single-trait tolerances violated: {failed}"


def test_mvlmm_joint_wald_matches(compare_mod) -> None:
    """GEMMA -lmm 1 -n 1 2 (mvLMM) vs TG MultiTraitLMM joint Wald at §16 tolerances."""
    _require_binary()
    _require_artifacts(
        OUT / "mdp_mvlmm_wald.assoc.txt",
        OUT / "mdp_mvlmm_wald.log.txt",
        OUT / "mdp_kinship.cXX.txt",
        DATA / "mdp_traits.txt",
        DATA / "mdp_numeric.txt",
        DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_mvlmm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"mvLMM tolerances violated: {failed}"
