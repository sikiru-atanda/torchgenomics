"""External-tool harness: GWASpoly vs TorchGenomics reference comparison.

These tests are skipped by default (they require the harness reference
outputs in validation/external/gwaspoly/outputs/). To run::

    bash validation/external/gwaspoly/install.sh
    bash validation/external/gwaspoly/fetch_data.sh
    bash validation/external/gwaspoly/run_gwaspoly.sh
    pytest -m external tests/test_external_gwaspoly.py -v

Each test invokes one comparison function from
``validation/external/gwaspoly/compare.py``. Tolerances mirror the existing
``tests/test_golden_gwaspoly.py`` golden contract on 5 polyploid gene-action
models (additive, 1-dom, 2-dom, 3-dom, diplo-additive) — this harness re-runs
GWASpoly (or uses the committed pre-Pillar-A reference) + TorchGenomics
end-to-end to confirm zero regression from the 9 Pillar A V1-core /
V1-platform fixes.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "gwaspoly"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("gwaspoly_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gwaspoly_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "GWASpoly harness artifacts not present (run install.sh + fetch_data.sh + run_gwaspoly.sh): "
            + ", ".join(missing)
        )


@pytest.fixture(scope="module")
def compare_mod():
    os.environ.setdefault("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return _load_compare_module()


@pytest.fixture(scope="module")
def cached_data(compare_mod):
    """Load potato data once for all tests in this module."""
    _require_artifacts(
        DATA / "potato_geno_aligned.csv",
        DATA / "potato_pheno_with_genoid.csv",
        DATA / "potato_map.csv",
    )
    return compare_mod._load_data(DATA)


def test_additive_matches(compare_mod, cached_data) -> None:
    """TG additive vs GWASpoly additive: -log10(p) corr ≥ 0.999."""
    _require_artifacts(OUT / "gwaspoly_additive.csv")
    rep = compare_mod.compare_additive(DATA, OUT, cached_data=cached_data)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Additive tolerances violated: {failed}"


def test_1dom_matches(compare_mod, cached_data) -> None:
    """TG 1-dom vs GWASpoly 1-dom-alt: -log10(p) corr ≥ 0.999."""
    _require_artifacts(OUT / "gwaspoly_1_dom_alt.csv")
    rep = compare_mod.compare_1dom(DATA, OUT, cached_data=cached_data)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"1-dom tolerances violated: {failed}"


def test_2dom_matches(compare_mod, cached_data) -> None:
    """TG 2-dom vs GWASpoly 2-dom-alt: -log10(p) corr ≥ 0.999."""
    _require_artifacts(OUT / "gwaspoly_2_dom_alt.csv")
    rep = compare_mod.compare_2dom(DATA, OUT, cached_data=cached_data)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"2-dom tolerances violated: {failed}"


def test_3dom_matches(compare_mod, cached_data) -> None:
    """TG 3-dom vs GWASpoly 2-dom-ref (mirror of 3-dom): corr ≥ 0.999."""
    _require_artifacts(OUT / "gwaspoly_2_dom_ref.csv")
    rep = compare_mod.compare_3dom(DATA, OUT, cached_data=cached_data)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"3-dom tolerances violated: {failed}"


def test_diplo_additive_differs(compare_mod, cached_data) -> None:
    """TG diplo-additive vs GWASpoly diplo_additive: encoding-mismatch corr ∈ [0.3, 0.7]."""
    _require_artifacts(OUT / "gwaspoly_diplo_additive.csv")
    rep = compare_mod.compare_diplo_additive(DATA, OUT, cached_data=cached_data)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"diplo-additive tolerances violated: {failed}"
