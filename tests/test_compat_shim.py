"""Tests for the v0.4.0 ``torchgwas`` → ``torchgenomics`` backwards-compat shim.

Covers:
  - ``import torchgwas`` emits a one-shot ``DeprecationWarning`` and resolves
    to the same module objects as ``torchgenomics``.
  - Submodule imports (``from torchgwas.models import X``) work and identify
    with their torchgenomics counterparts.
  - Legacy ``TORCHGWAS_DISABLE_*`` env vars are accepted with a one-shot
    ``DeprecationWarning``; new ``TORCHGENOMICS_DISABLE_*`` names work silently.
  - The ``torchgwas`` CLI proxy prints a stderr deprecation banner and forwards
    argv unchanged to the real CLI.

These tests use ``subprocess`` for the import-shim and CLI-proxy cases so the
``DeprecationWarning`` and one-shot caches are exercised in a clean Python
process (the test runner imports torchgenomics first, which would otherwise
poison the cache).
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
TORCHGWAS_META_ROOT = REPO_ROOT / "packaging" / "torchgwas-meta"


def _run_in_clean_subprocess(code: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a snippet of Python in a fresh interpreter that sees the worktree first.

    Both the main ``torchgenomics`` package (at REPO_ROOT) and the legacy
    ``torchgwas`` meta-package (at REPO_ROOT/packaging/torchgwas-meta/) need
    to be on PYTHONPATH so the shim works without a real ``pip install``.
    """
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(TORCHGWAS_META_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if env_extra:
        env.update(env_extra)
    # Make sure deprecation warnings show up in stderr
    env.setdefault("PYTHONWARNINGS", "always")
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestImportShim:
    """Shim re-exports torchgenomics under the legacy ``torchgwas`` name."""

    def test_top_level_import_warns(self):
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('always')\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import torchgwas\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert len(deps) == 1, f'expected 1 DeprecationWarning, got {len(deps)}'\n"
            "    assert 'torchgwas' in str(deps[0].message).lower()\n"
            "    assert 'torchgenomics' in str(deps[0].message)\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_version_attribute(self):
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('ignore')\n"
            "import torchgwas, torchgenomics\n"
            "assert torchgwas.__version__ == torchgenomics.__version__\n"
            "print(torchgwas.__version__)\n"
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert result.stdout.strip()  # non-empty version

    def test_submodule_import_resolves_to_torchgenomics(self):
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('ignore')\n"
            "from torchgwas.models import SingleTraitLMM as Old\n"
            "from torchgenomics.models import SingleTraitLMM as New\n"
            "assert Old is New, 'classes must be identical'\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_nested_submodule_import(self):
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('ignore')\n"
            "from torchgwas.scan.unified import UnifiedScanner as Old\n"
            "from torchgenomics.scan.unified import UnifiedScanner as New\n"
            "assert Old is New, 'classes must be identical'\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_multi_submodule_import_via_torchgwas(self):
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('ignore')\n"
            "from torchgwas import io, models, scan, stats\n"
            "for m in (io, models, scan, stats):\n"
            "    assert m.__name__.startswith('torchgenomics.'), m.__name__\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_attribute_access_forwards(self):
        """``torchgwas.foo`` should fall through to ``torchgenomics.foo``."""
        result = _run_in_clean_subprocess(
            "import warnings\n"
            "warnings.simplefilter('ignore')\n"
            "import torchgwas, torchgenomics\n"
            "torchgenomics._test_marker = 'sentinel'\n"
            "assert torchgwas._test_marker == 'sentinel'\n"
            "print('OK')\n"
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout


class TestEnvVarCompat:
    """``TORCHGWAS_DISABLE_*`` legacy env vars resolve to their new names."""

    def test_new_disable_native_no_warning(self):
        result = _run_in_clean_subprocess(
            "import warnings, torchgenomics._dispatch as d\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    assert d.native_disabled() is True\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert deps == [], f'new name should not warn: {[str(x.message) for x in deps]}'\n"
            "print('OK')\n",
            env_extra={"TORCHGENOMICS_DISABLE_NATIVE": "1"},
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_legacy_disable_native_warns(self):
        result = _run_in_clean_subprocess(
            "import warnings, torchgenomics._dispatch as d\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    assert d.native_disabled() is True\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert len(deps) == 1, f'expected 1 deprecation, got {len(deps)}'\n"
            "    assert 'TORCHGWAS_DISABLE_NATIVE' in str(deps[0].message)\n"
            "    assert 'TORCHGENOMICS_DISABLE_NATIVE' in str(deps[0].message)\n"
            "print('OK')\n",
            env_extra={"TORCHGWAS_DISABLE_NATIVE": "1"},
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_legacy_env_warns_only_once(self):
        result = _run_in_clean_subprocess(
            "import warnings, torchgenomics._dispatch as d\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    for _ in range(5):\n"
            "        d.native_disabled()\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert len(deps) == 1, f'expected 1 deprecation across 5 reads, got {len(deps)}'\n"
            "print('OK')\n",
            env_extra={"TORCHGWAS_DISABLE_NATIVE": "1"},
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_new_takes_precedence_over_legacy(self):
        result = _run_in_clean_subprocess(
            "import warnings, torchgenomics._dispatch as d\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    val = d.env_var('TORCHGENOMICS_DISABLE_NATIVE')\n"
            "    assert val == 'newval', f'expected newval, got {val!r}'\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert deps == [], 'no warning when new var is set'\n"
            "print('OK')\n",
            env_extra={
                "TORCHGENOMICS_DISABLE_NATIVE": "newval",
                "TORCHGWAS_DISABLE_NATIVE": "oldval",
            },
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout

    def test_disable_gpu_compat(self):
        result = _run_in_clean_subprocess(
            "import warnings, torchgenomics._dispatch as d\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    assert d.gpu_disabled() is True\n"
            "    deps = [x for x in w if issubclass(x.category, DeprecationWarning)]\n"
            "    assert len(deps) == 1\n"
            "    assert 'TORCHGWAS_DISABLE_GPU' in str(deps[0].message)\n"
            "print('OK')\n",
            env_extra={"TORCHGWAS_DISABLE_GPU": "1"},
        )
        assert result.returncode == 0, f"stderr={result.stderr!r}"
        assert "OK" in result.stdout


class TestCliProxy:
    """The ``torchgwas`` console-script wrapper prints a banner and forwards argv."""

    def test_legacy_cli_help_proxies(self):
        result = _run_in_clean_subprocess(
            "from torchgwas._legacy_cli import main\n"
            "try:\n"
            "    rc = main(['--help'])\n"
            "except SystemExit as e:\n"
            "    rc = e.code\n"
            "print('EXIT', rc)\n",
        )
        assert "[deprecation]" in result.stderr
        assert "torchgwas" in result.stderr.lower()
        assert "torchgenomics" in result.stderr.lower()
        assert "usage: torchgenomics" in result.stdout
        # CLI subcommand list should appear in --help output
        assert "lmm-scan" in result.stdout
        assert "ld-blocks" in result.stdout

    def test_legacy_cli_validate_subcommand_dispatches(self):
        """Smoke check: a real subcommand reaches the torchgenomics dispatcher."""
        result = _run_in_clean_subprocess(
            "from torchgwas._legacy_cli import main\n"
            "try:\n"
            "    rc = main(['validate', '--help'])\n"
            "except SystemExit as e:\n"
            "    rc = e.code\n"
            "print('EXIT', rc)\n",
        )
        assert "[deprecation]" in result.stderr
        assert "validate" in result.stdout or "validate" in result.stderr
