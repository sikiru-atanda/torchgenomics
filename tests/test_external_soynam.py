"""External-tool harness: SoyNAM + rrBLUP vs TorchGWAS reference.

These tests are skipped by default (they require the R packages installed
by the harness install.sh, the extracted SoyNAM TSVs from fetch_data.sh,
and the rrBLUP reference output from run.sh). To run:

    bash validation/external/soynam/install.sh
    bash validation/external/soynam/fetch_data.sh
    bash validation/external/soynam/run.sh
    pytest -m external tests/test_external_soynam.py -v

Each test invokes one comparison function from
``validation/external/soynam/compare.py`` and asserts every check in
the ComparisonReport passed. Tolerances are calibrated to observed
values from the first successful run, with margins documented in
compare.py.

The SoyNAM panel is the only external reference for ``WithinFamilyLMM``
(Phase 23, Young et al. 2022). Its multi-family RIL design with a common
parent is exactly the regime where family-mean confounding is large for
SNPs whose allele frequency varies across families — and the within-
family scan produces materially different effect-size estimates than
the standard scan, with the attenuation diagnostic flagging the
divergence per SNP.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.external

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "soynam"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
INSTALL_MARKER = HARNESS / ".install_marker"


def _load_compare_module():
    """Dynamically import compare.py from the harness directory.

    We do this instead of a regular import because the harness lives outside
    the importable package tree (`validation/` is not a Python package).
    """
    spec = importlib.util.spec_from_file_location(
        "soynam_compare", HARNESS / "compare.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["soynam_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "SoyNAM harness artifacts not present "
            "(run install.sh + fetch_data.sh + run.sh): "
            + ", ".join(missing)
        )


def _require_install() -> None:
    if not INSTALL_MARKER.exists():
        pytest.skip(
            f"SoyNAM R packages not installed (no marker at {INSTALL_MARKER}); "
            "run validation/external/soynam/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Disable native so we exercise the pure-Python reference path.
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def _required_artifacts() -> list[Path]:
    return [
        DATA / "geno.tsv",
        DATA / "pheno.tsv",
        DATA / "family.tsv",
        DATA / "marker_map.tsv",
        OUT / "rrblup_results.json",
        OUT / "A_mat.tsv",
    ]


def test_kinship_matches_rrblup_amat(compare_mod) -> None:
    """rrBLUP A.mat vs TorchGWAS grm_vanraden — kinship matrix equivalence."""
    _require_install()
    _require_artifacts(*_required_artifacts())
    rep = compare_mod.compare_kinship(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Kinship tolerances violated: {failed}"


def test_stlmm_matches_rrblup_gwas(compare_mod) -> None:
    """rrBLUP::GWAS vs TorchGWAS SingleTraitLMM — variance components + −log10p."""
    _require_install()
    _require_artifacts(*_required_artifacts())
    rep = compare_mod.compare_stlmm(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"SingleTraitLMM tolerances violated: {failed}"


def test_within_family_dual_scan_runs_to_completion(compare_mod) -> None:
    """WithinFamilyLMM dual-scan runs cleanly across all SoyNAM SNPs."""
    _require_install()
    _require_artifacts(*_required_artifacts())
    rep = compare_mod.compare_wflmm(DATA, OUT)
    rep.print()
    # The "frac finite" checks gate this test.
    finite_checks = [
        c for c in rep.checks
        if c.name.startswith("frac finite")
    ]
    failed = [c for c in finite_checks if not c.passed]
    assert not failed, f"Dual-scan completeness violated: {failed}"


def test_within_family_attenuation_diagnostic_present(compare_mod) -> None:
    """WithinFamilyLMM attenuation diagnostic populated on every SNP, with the
    confounding flag firing on a non-trivial fraction of SNPs (the SoyNAM
    multi-family RIL design is exactly the regime where family-mean
    confounding is detectable on a sizeable fraction of variant SNPs)."""
    _require_install()
    _require_artifacts(*_required_artifacts())
    rep = compare_mod.compare_wflmm(DATA, OUT)
    # Find the attenuation + sanity checks
    atten = [c for c in rep.checks if "attenuation" in c.name or "sanity" in c.name]
    failed = [c for c in atten if not c.passed]
    assert not failed, f"Attenuation diagnostic violated: {failed}"
