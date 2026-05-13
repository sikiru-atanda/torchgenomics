"""Pillar D — reproducibility audit pytest entry.

These tests are *opt-in*. They install + run upstream tools and diff their
fresh output against the committed reference fixtures (gemma_demo/output,
benchmark/gapit_results, benchmark/gwaspoly_results).

Run them with:

    pytest -m reproducibility tests/test_reproducibility.py -v

The conftest hook auto-skips them on `pytest tests/` (default suite). When
the upstream tool is missing or unbuildable on the host, each test falls
back to ``pytest.skip(...)`` rather than failing — drift findings are
informational by spec §7.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.reproducibility

REPO = Path(__file__).resolve().parent.parent
HARNESS_DIR = REPO / "validation" / "external"
COMPARE_PY = REPO / "validation" / "reproducibility" / "compare_fixtures.py"


def _maybe_install(tool: str) -> tuple[bool, str]:
    """Run install.sh once. Return (ok, message)."""
    install = HARNESS_DIR / tool / "install.sh"
    if not install.exists():
        return False, f"missing {install}"
    try:
        proc = subprocess.run(
            ["bash", str(install)],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "install timed out (>900s)"
    return proc.returncode == 0, (proc.stderr or proc.stdout)[-400:]


def _maybe_fetch(tool: str) -> tuple[bool, str]:
    fetch = HARNESS_DIR / tool / "fetch_data.sh"
    if not fetch.exists():
        return False, f"missing {fetch}"
    try:
        proc = subprocess.run(
            ["bash", str(fetch)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "fetch timed out (>600s)"
    return proc.returncode == 0, (proc.stderr or proc.stdout)[-400:]


def _has_outputs(tool: str) -> bool:
    out_dir = HARNESS_DIR / tool / "outputs"
    return out_dir.exists() and any(out_dir.iterdir())


def _tool_available(tool: str) -> tuple[bool, str]:
    """Cheap probe — does the tool appear loadable in this env?"""
    if tool == "gemma":
        bin_path = HARNESS_DIR / "gemma" / "bin" / "gemma"
        return bin_path.exists() and os.access(bin_path, os.X_OK), str(bin_path)
    elif tool == "gapit":
        # GAPIT3 / GAPIT loadable in R?
        try:
            proc = subprocess.run(
                ["Rscript", "-e", (
                    'rlib <- Sys.getenv("RLIB", unset=NA); '
                    'if (!is.na(rlib) && nzchar(rlib) && dir.exists(rlib)) '
                    '.libPaths(c(rlib, .libPaths())); '
                    'for (p in c("GAPIT3", "GAPIT")) '
                    'if (requireNamespace(p, quietly=TRUE)) { '
                    'cat(p); quit(status=0) }; '
                    'quit(status=1)'
                )],
                env={**os.environ, "RLIB": str(HARNESS_DIR / "gapit" / "bin" / "Rlib")},
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:
            return False, str(e)
        return proc.returncode == 0, proc.stdout.strip()
    elif tool == "gwaspoly":
        try:
            proc = subprocess.run(
                ["Rscript", "-e", (
                    'rlib <- Sys.getenv("RLIB", unset=NA); '
                    'if (!is.na(rlib) && nzchar(rlib) && dir.exists(rlib)) '
                    '.libPaths(c(rlib, .libPaths())); '
                    'if (requireNamespace("GWASpoly", quietly=TRUE)) { '
                    'cat(as.character(packageVersion("GWASpoly"))); quit(status=0) }; '
                    'quit(status=1)'
                )],
                env={**os.environ, "RLIB": str(HARNESS_DIR / "gwaspoly" / "bin" / "Rlib")},
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as e:
            return False, str(e)
        return proc.returncode == 0, proc.stdout.strip()
    return False, f"unknown tool {tool}"


def _compare(tool: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(COMPARE_PY), "--tool", tool],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("tool", ["gemma", "gapit", "gwaspoly"])
def test_fixture_reproducibility(tool: str) -> None:
    """Diff fresh upstream tool output against the committed reference fixture.

    Order of operations:
      1. install.sh  — idempotent.
      2. probe the tool (Rscript / binary).
      3. If unavailable → skip cleanly (infra-blocker).
      4. fetch_data.sh — idempotent.
      5. If outputs/ already populated, do not re-run (the previous
         orchestrator run is the authoritative fresh output for this
         session). This avoids 10-15-minute GWASpoly re-runs from the
         test suite.
      6. compare_fixtures.py — assert rc == 0 (which it is by design,
         since drift is informational, not a hard failure).
    """
    install_ok, install_msg = _maybe_install(tool)
    if not install_ok:
        pytest.skip(f"{tool} install failed: {install_msg}")

    avail, probe = _tool_available(tool)
    if not avail:
        pytest.skip(f"{tool} not available after install: {probe}")

    fetch_ok, fetch_msg = _maybe_fetch(tool)
    if not fetch_ok:
        pytest.skip(f"{tool} fetch failed: {fetch_msg}")

    if not _has_outputs(tool):
        pytest.skip(
            f"{tool} outputs/ is empty; run validation/reproducibility/"
            f"rerun_goldens.py --tool {tool} (or run_{tool}.sh) first to "
            f"produce a fresh upstream output for the diff."
        )

    proc = _compare(tool)
    # compare_fixtures.py exits 0 even on drift (drift is informational).
    # The test asserts the script *ran* end-to-end, not that drift was zero —
    # the actual drift status is recorded in
    # validation/reproducibility/outputs/<tool>_drift.json plus the README
    # drift table for human review.
    assert proc.returncode == 0, (
        f"compare_fixtures.py rc={proc.returncode} for {tool}\n"
        f"stdout:\n{proc.stdout[-1500:]}\n"
        f"stderr:\n{proc.stderr[-1500:]}"
    )
