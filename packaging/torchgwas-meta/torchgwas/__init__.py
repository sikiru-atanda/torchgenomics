"""Deprecated alias for :mod:`torchgenomics`.

This package was renamed from ``torchgwas`` to ``torchgenomics`` in v0.4.0
to reflect a scope that now covers far more than GWAS (GWAS + post-GWAS
+ PGS + MR + TWAS + multi-omics + LD + imputation + annotation +
visualization). Importing ``torchgwas`` still works as a transparent
shim through the v0.x series and will be removed in v1.0.0.

Migration: change ``import torchgwas`` to ``import torchgenomics``.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

warnings.warn(
    "`torchgwas` has been renamed to `torchgenomics`. "
    "Please update your imports to `import torchgenomics`. "
    "This compatibility shim will be removed in v1.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

import torchgenomics as _torchgenomics


class _AliasLoader(Loader):
    """Loader that returns the real torchgenomics submodule unchanged."""

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return importlib.import_module(self._real_name)

    def exec_module(self, module: ModuleType) -> None:
        return None


class _TorchgwasFinder(MetaPathFinder):
    """Redirect ``torchgwas.X`` imports to ``torchgenomics.X``."""

    _PREFIX_OLD = "torchgwas"
    _PREFIX_NEW = "torchgenomics"

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith(self._PREFIX_OLD + "."):
            return None
        suffix = fullname[len(self._PREFIX_OLD):]
        real_name = self._PREFIX_NEW + suffix
        try:
            spec = importlib.util.find_spec(real_name)
        except (ImportError, ValueError):
            return None
        if spec is None:
            return None
        return ModuleSpec(fullname, _AliasLoader(real_name), is_package=spec.submodule_search_locations is not None)


if not any(isinstance(f, _TorchgwasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _TorchgwasFinder())


def __getattr__(name: str):
    return getattr(_torchgenomics, name)


def __dir__():
    return dir(_torchgenomics)


__version__ = _torchgenomics.__version__
