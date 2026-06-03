"""Discover-first Julia bootstrap for PolyOrigin.jl (Phase 56 internal).

Not public API. ``run_polyorigin`` uses ``get_runtime`` to obtain a
``juliacall`` handle. Search order: explicit kwarg → TORCHGENOMICS_JULIA env
var → known install paths → juliacall's own managed install (only if the
caller consents). See the Phase 56 design spec Section 2.3 for rationale.
"""
from __future__ import annotations

import glob
import logging
import os
from .._dispatch import env_var
import re
import shutil
import subprocess
import sys
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

    Order: explicit override → TORCHGENOMICS_JULIA env var → shutil.which
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

    env = env_var("TORCHGENOMICS_JULIA")
    if env:
        if not os.path.isfile(env):
            raise ValueError(
                f"TORCHGENOMICS_JULIA={env!r}: does not exist or is not a file."
            )
        return os.path.abspath(env)

    which = shutil.which("julia")
    if which:
        return which

    for cand in _common_julia_paths():
        if os.path.isfile(cand):
            return cand

    # juliapkg (juliacall's companion) manages its own Julia install.
    # Try it without importing juliacall itself (which would start Julia).
    try:
        import juliapkg  # lightweight — does NOT start Julia
        jl_exe = juliapkg.executable()
        if jl_exe and os.path.isfile(jl_exe):
            return os.path.abspath(jl_exe)
    except Exception:
        pass

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


# ---------------------------------------------------------------------------
# Consent gate + bootstrap (Task 8)
# ---------------------------------------------------------------------------

def _consent_to_install(auto_install: bool) -> bool:
    """Return True iff the caller authorized a Julia download."""
    if auto_install:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        reply = input(
            "Julia not detected. Install Julia 1.10 (~300 MB) + "
            "PolyOrigin.jl now? [y/N]: "
        )
    except EOFError:
        return False
    return reply.strip().lower() in {"y", "yes"}


def get_runtime(
    julia_path: str | None = None,
    auto_install_julia: bool = False,
) -> tuple[Any, Any, str]:
    """Bootstrap PolyOrigin on demand. Returns (Main, PolyOrigin, version).

    Discover-first: resolve an existing Julia via PATH / TORCHGENOMICS_JULIA /
    common install paths. Only fall back to juliacall's managed install if
    nothing is found AND the caller consents (interactive prompt on a tty
    or explicit ``auto_install_julia=True``).
    """
    global _jl, _polyorigin, _version
    if _jl is not None:
        return _jl, _polyorigin, _version

    with _lock:
        if _jl is not None:
            return _jl, _polyorigin, _version

        found = _find_existing_julia(julia_path)
        chosen: str | None = None
        if found is not None:
            ok, ver = _probe_version(found)
            if ok:
                chosen = found
                logger.info("Using existing Julia at %s (v%s)", chosen, ver)
            else:
                if julia_path is not None or env_var("TORCHGENOMICS_JULIA"):
                    raise RuntimeError(
                        f"Julia at {found!r} is v{ver}; PolyOrigin requires >= 1.10. "
                        "Install a newer Julia or unset the override."
                    )
                logger.info(
                    "Found Julia at %s (v%s) but version < 1.10; ignoring.",
                    found, ver,
                )

        if chosen is None:
            if not _consent_to_install(auto_install_julia):
                raise RuntimeError(
                    "Julia not detected and auto_install_julia=False. "
                    "Either: (a) install Julia yourself from "
                    "https://julialang.org/downloads/ (ensure `julia` is on "
                    "PATH or set TORCHGENOMICS_JULIA), or (b) retry with "
                    "auto_install_julia=True (library) / --auto-install (CLI) "
                    "to let juliacall provision Julia 1.10 automatically "
                    "(~300 MB download)."
                )
            logger.info("Installing Julia 1.10 via juliacall (~300 MB, one-time)")

        if chosen is not None:
            os.environ["PYTHON_JULIAPKG_EXE"] = chosen

        try:
            from juliacall import Main as jl_local  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "juliacall not installed. Install the polyploid-phase extra: "
                "pip install torchgenomics[polyploid-phase]"
            ) from e

        try:
            jl_local.seval("using PolyOrigin")
            version_str = str(jl_local.seval("string(pkgversion(PolyOrigin))"))
        except Exception as e:
            raise RuntimeError(
                f"PolyOrigin.jl unavailable after juliacall bootstrap: {e}. "
                "If this is a transient network issue, retry; otherwise run "
                "'julia -e \"using Pkg; Pkg.resolve()\"' to recover."
            ) from e

        _jl = jl_local
        _polyorigin = jl_local.PolyOrigin
        _version = version_str
        return _jl, _polyorigin, _version
