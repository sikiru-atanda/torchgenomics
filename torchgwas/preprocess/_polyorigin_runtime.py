"""Discover-first Julia bootstrap for PolyOrigin.jl (Phase 56 internal).

Not public API. ``run_polyorigin`` uses ``get_runtime`` to obtain a
``juliacall`` handle. Search order: explicit kwarg → TORCHGWAS_JULIA env
var → known install paths → juliacall's own managed install (only if the
caller consents). See the Phase 56 design spec Section 2.3 for rationale.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jl: Any = None
_polyorigin: Any = None
_version: str | None = None

# ---------------------------------------------------------------------------
# Julia discovery helpers (Task 7)
# ---------------------------------------------------------------------------

_MIN_JULIA = (1, 10)


def _common_julia_paths() -> list[str]:
    home = os.path.expanduser("~")
    cands = [
        os.path.join(home, ".juliaup", "bin", "julia"),
        os.path.join(home, ".juliaup", "bin", "julia.exe"),
        "/opt/julia/bin/julia",
        "/usr/local/bin/julia",
    ]
    for pf in (
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("LOCALAPPDATA", ""),
    ):
        if pf:
            cands.extend(glob.glob(os.path.join(pf, "Julia-*", "bin", "julia.exe")))
    return cands


def _find_existing_julia(override: str | None) -> str | None:
    """Return an absolute path to a Julia binary, or None if not found.

    Order: explicit override → TORCHGWAS_JULIA env var → shutil.which
    ("julia") → common install paths. An override that points to a
    nonexistent/non-executable file raises ValueError.
    """
    if override is not None:
        p = os.path.abspath(override)
        if not os.path.isfile(p):
            raise ValueError(
                f"julia_path={override!r}: does not exist or is not a file."
            )
        if not os.access(p, os.X_OK):
            raise ValueError(
                f"julia_path={override!r}: exists but is not executable."
            )
        return p

    env = os.environ.get("TORCHGWAS_JULIA")
    if env:
        if not os.path.isfile(env):
            raise ValueError(
                f"TORCHGWAS_JULIA={env!r}: does not exist or is not a file."
            )
        return os.path.abspath(env)

    which = shutil.which("julia")
    if which:
        return which

    for cand in _common_julia_paths():
        if os.path.isfile(cand):
            return cand

    return None


def _probe_version(julia_path: str) -> tuple[bool, str]:
    """Return (ok, version_string). ok iff version >= 1.10."""
    try:
        res = subprocess.run(
            [julia_path, "--version"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("julia --version probe failed for %s: %s", julia_path, e)
        return False, "unknown"
    text = (res.stdout or "") + (res.stderr or "")
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    if not m:
        return False, text.strip() or "unknown"
    major, minor = int(m.group(1)), int(m.group(2))
    ver = f"{major}.{minor}.{m.group(3)}"
    return (major, minor) >= _MIN_JULIA, ver
