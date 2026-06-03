"""External-tool harness: BOLT-LMM vs TorchGenomics reference comparison.

These tests are skipped by default (they require:
  - the BOLT-LMM v2.5 prebuilt binary already extracted into
    ``validation/external/bolt_lmm/bin/``,
  - the staged data fixture, and
  - the reference outputs from a successful run_bolt.sh).

To run:

    bash validation/external/bolt_lmm/install.sh
    bash validation/external/bolt_lmm/fetch_data.sh
    bash validation/external/bolt_lmm/run_bolt.sh
    pytest -m external tests/test_external_bolt_lmm.py -v

Per spec §9 + the harness prompt, if BOLT-LMM installation produced an
``.infra_blocker`` file (the documented failure mode is GLIBCXX / glibc
mismatch on RHEL 9.6), pytest skips with that file's text as the reason.
The skeleton remains in place for a future user with a working install.

F3 logic (per spec §9):
  - SingleTraitLMM is V1-core.
  - --lmmInfOnly is the textbook infinitesimal LMM, the closest BOLT
    analog to TG's SingleTraitLMM.
  - Material disagreement on β / -log10p in this mode would be V1-core
    fix-now F3 — diagnose, separate fix commit, ledger row.
  - BOLT's "BoltLMM-mixed" mode (Gaussian-mixture spread approximation)
    is a BOLT-specific algorithm; we do NOT compare against it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "validation" / "external" / "bolt_lmm"
DATA = HARNESS / "data"
OUT = HARNESS / "outputs"
INSTALL_MARKER = HARNESS / ".install_marker"
INFRA_BLOCKER = HARNESS / ".infra_blocker"

_INFRA_REASON = ""
if INFRA_BLOCKER.exists():
    try:
        _INFRA_REASON = INFRA_BLOCKER.read_text().strip()
    except OSError:
        _INFRA_REASON = "BOLT-LMM install marked infra-blocker (unreadable .infra_blocker)"

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(
        INFRA_BLOCKER.exists(),
        reason=_INFRA_REASON or "BOLT-LMM install marked infra-blocker",
    ),
]


def _load_compare_module():
    """Dynamically import compare.py from the harness directory."""
    spec = importlib.util.spec_from_file_location(
        "bolt_compare", HARNESS / "compare.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {HARNESS / 'compare.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bolt_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_artifacts(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "BOLT-LMM harness artifacts not present (run install.sh + "
            "fetch_data.sh + run_bolt.sh): " + ", ".join(missing)
        )


def _require_install() -> None:
    if not INSTALL_MARKER.exists():
        pytest.skip(
            f"BOLT-LMM not installed (no marker at {INSTALL_MARKER}); "
            "run validation/external/bolt_lmm/install.sh"
        )


@pytest.fixture(scope="module")
def compare_mod():
    # Match the docs/validation.md golden CI gate by exercising the
    # pure-Python reference path (native disabled).
    os.environ.setdefault("TORCHGENOMICS_DISABLE_NATIVE", "1")
    return _load_compare_module()


def test_bolt_lmm_inf_only_matches_torchgenomics(compare_mod) -> None:
    """BOLT-LMM --lmmInfOnly per-SNP β / SE / -log10p vs TG SingleTraitLMM+LOCO."""
    _require_install()
    _require_artifacts(
        DATA / "mdp.bed", DATA / "mdp.bim", DATA / "mdp.fam",
        DATA / "bolt_pheno.tsv", DATA / "bolt_covar.tsv",
        OUT / "bolt_stats.tsv",
    )
    rep = compare_mod.compare_lmm_inf(DATA, OUT)
    rep.print()
    failed = [c for c in rep.checks if not c.passed]
    assert not failed, f"BOLT-LMM --lmmInfOnly tolerances violated: {failed}"
