"""External-tool harness: SAIGE vs TorchGWAS reference comparison.

These tests are skipped by default (they require:
  - the SAIGE container image already pulled (or Bioconductor R install),
  - the staged data fixture, and
  - the reference outputs from a successful run_saige.sh).

To run:

    bash validation/external/saige/install.sh
    bash validation/external/saige/fetch_data.sh
    bash validation/external/saige/run_saige.sh
    pytest -m external tests/test_external_saige.py -v

Each test invokes one comparison function from
``validation/external/saige/compare.py`` and asserts every check in the
ComparisonReport passed. Tolerances are calibrated to observed values from
the first successful run on the MDP fixture (281 samples, 2897 SNPs after
QC). See compare.py for per-tolerance rationale.

Per spec §9 + the harness prompt, if SAIGE installation failed (both
docker pull and Bioconductor) we mark `.infra_blocker` in the harness
directory and pytest skips with that file's text as the reason. The rest
of the skeleton is in place for a future user with a working install.

F3 logic (per spec §9):
  - BinaryGLMM, OrdinalGLMM, SurvivalGLMM are post-V1 (Phases 31–35).
  - Documented parameterization mismatches (PCG vs PQL, fast-Firth vs
    full-iteration Firth, SPA flavor differences) are NOT F3.
  - A real V1-platform math defect would be F3.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "saige"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
INSTALL_MARKER = HARNESS / ".install_marker"
INSTALL_METHOD = HARNESS / ".install_method"
INFRA_BLOCKER = HARNESS / ".infra_blocker"

_INFRA_REASON = ""
if INFRA_BLOCKER.exists():
    try:
        _INFRA_REASON = INFRA_BLOCKER.read_text().strip()
    except OSError:
        _INFRA_REASON = "SAIGE install marked infra-blocker (unreadable .infra_blocker)"

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        INFRA_BLOCKER.exists(),
        reason=_INFRA_REASON or "SAIGE install marked infra-blocker",
    ),
]


def _load_compare_module():
    """Dynamically import compare.py from the harness directory."""
    spec = importlib.util.spec_from_file_location("saige_compare", HARNESS / "compare.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["saige_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "SAIGE harness artifacts not present (run install.sh + "
            "fetch_data.sh + run_saige.sh): " + ", ".join(missing)
        )


def _require_install() -> None:
    if not INSTALL_MARKER.exists():
        pytest.skip(
            f"SAIGE not installed (no marker at {INSTALL_MARKER}); "
            "run validation/external/saige/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Match the docs/validation.md golden CI gate by exercising the
    # pure-Python reference path (native disabled).
    os.environ.setdefault("TORCHGWAS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_step1_null_model_finite(compare_mod) -> None:
    """SAIGE Step 1 .rda + variance-ratio file are well-formed and non-degenerate."""
    _require_install()
    _require_artifacts(
        OUT / "step1.rda",
        OUT / "step1.varianceRatio.txt",
    )
    rep = compare_mod.compare_step1_sanity(OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Step 1 sanity violated: {failed}"


def test_step2_binary_glmm_matches_saige(compare_mod) -> None:
    """SAIGE Step 2 (PCG + SPA) vs TG BinaryGLMM(use_spa=True) on the MDP fixture."""
    _require_install()
    _require_artifacts(
        DATA / "sample.bed", DATA / "sample.bim", DATA / "sample.fam",
        DATA / "saige_pheno.tsv",
        OUT / "step2.txt",
    )
    rep = compare_mod.compare_step2(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"Step 2 binary GLMM tolerances violated: {failed}"


def test_spa_correction_matches_saige(compare_mod) -> None:
    """The SPA-marked subset of SAIGE rows correlates with TG's saddlepoint p-values.

    We don't hard-gate on the SPA subset cardinality (the cutoff selects
    ~10–200 SNPs depending on PQL init), but we do require the subset
    correlation to be present in the report and to reach a meaningful
    threshold when the subset has at least 5 SNPs.
    """
    _require_install()
    _require_artifacts(
        DATA / "sample.bed", DATA / "saige_pheno.tsv",
        OUT / "step2.txt",
    )
    rep = compare_mod.compare_step2(DATA, OUT)
    spa_corr = rep.extras.get("SPA -log10p corr", None)
    spa_n = rep.extras.get("SPA subset n", 0)
    if spa_n is not None and isinstance(spa_n, int) and spa_n >= 5:
        # SPA tail correlation: SAIGE's saddlepoint and TG's saddlepoint
        # both override the standard score-test p in the same direction.
        # First-run observed = 0.885 on 133 hits; floor at 0.70 to absorb
        # PQL-init drift across hosts.
        assert isinstance(spa_corr, float), (
            f"Expected SPA -log10p corr to be a float when SPA subset n={spa_n}, "
            f"got {spa_corr!r}"
        )
        assert spa_corr >= 0.70, (
            f"SPA -log10p correlation too low: {spa_corr:.3f} < 0.70 "
            f"(SAIGE Is.SPA=true subset, n={spa_n})"
        )
    else:
        pytest.skip(
            f"SPA subset too small (n={spa_n}) for a stable correlation check. "
            "This typically indicates a fixture where SAIGE's saddlepoint cutoff "
            "(default χ²>2) was not triggered; consider widening the simulated "
            "phenotype's effect distribution."
        )
