"""External-tool harness: GAPIT3 vs TorchGenomics reference comparison.

These tests are skipped by default (they require the GAPIT3 R package OR the
canonical pre-Pillar-A reference outputs at benchmark/gapit_results/, plus
the harness reference outputs in validation/external/gapit/outputs/). To run::

    bash validation/external/gapit/install.sh        # tries to install GAPIT3
    bash validation/external/gapit/fetch_data.sh
    bash validation/external/gapit/run_gapit.sh      # falls back if GAPIT3 missing
    pytest -m external tests/test_external_gapit.py -v

Each test invokes one comparison function from
``validation/external/gapit/compare.py``. Tolerances mirror the existing
``tests/test_golden_gapit.py`` golden contract — this harness re-runs GAPIT
(or uses the committed pre-Pillar-A reference) + TorchGenomics end-to-end to
confirm zero regression from the 9 Pillar A V1-core / V1-platform fixes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "gapit"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("gapit_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gapit_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "GAPIT harness artifacts not present (run install.sh + fetch_data.sh + run_gapit.sh): "
            + ", ".join(missing)
        )


@pytest.fixture(scope="module")
def compare_mod():
    os.environ.setdefault("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_glm_matches(compare_mod) -> None:
    """GAPIT GLM vs TG GLM Wald: -log10(p) corr ≥ 0.99 (observed: 1.000000)."""
    _require_artifacts(
        OUT / "GLM_GWAS.csv",
        DATA / "mdp_traits.txt", DATA / "mdp_numeric.txt", DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_glm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"GLM tolerances violated: {failed}"


def test_mlm_matches(compare_mod) -> None:
    """GAPIT MLM vs TG SingleTraitLMM Wald: -log10(p) corr ≥ 0.99."""
    _require_artifacts(
        OUT / "MLM_GWAS.csv",
        DATA / "mdp_traits.txt", DATA / "mdp_numeric.txt", DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_mlm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"MLM tolerances violated: {failed}"


def test_farmcpu_matches(compare_mod) -> None:
    """GAPIT FarmCPU vs TG FarmCPU: stochastic per-SNP corr ≥ 0.5 + top-k overlap."""
    _require_artifacts(
        OUT / "FarmCPU_GWAS.csv",
        DATA / "mdp_traits.txt", DATA / "mdp_numeric.txt", DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_farmcpu(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"FarmCPU tolerances violated: {failed}"


def test_blink_matches(compare_mod) -> None:
    """GAPIT BLINK vs TG BLINK: stochastic per-SNP corr ≥ 0.5 + top-k overlap."""
    _require_artifacts(
        OUT / "BLINK_GWAS.csv",
        DATA / "mdp_traits.txt", DATA / "mdp_numeric.txt", DATA / "mdp_SNP_information.txt",
    )
    rep = compare_mod.compare_blink(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"BLINK tolerances violated: {failed}"
