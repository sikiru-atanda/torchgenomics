"""High-level, one-call API for TorchGenomics.

This module provides curated convenience functions for the 12 most common
workflows — designed for three audiences with a single code path:

  - **Novice users** (notebook one-liners with smart defaults):

      >>> import torchgenomics as tg
      >>> result = tg.lmm_scan(genotype="data.bed", phenotype="pheno.tsv")
      >>> print(result.summary())
      >>> result.manhattan()

  - **Advanced users** keep using the low-level surface
    (:mod:`torchgenomics.models`, :mod:`torchgenomics.scan`, etc.).
    Nothing in :mod:`torchgenomics.api` shadows those modules.

  - **LLM tools via MCP** (:mod:`torchgenomics.mcp` calls these functions
    directly; their typed result objects serialize cleanly via
    :meth:`~torchgenomics.api._results._BaseRun.to_dict`).

All tier-1 functions accept and return Python primitives plus typed
result objects. No torch Tensors or numpy arrays in the call signature.
File-path inputs accept ``str`` or :class:`pathlib.Path`.
"""
from __future__ import annotations

# Result types (re-exported for type hints / isinstance checks)
from ._results import (
    AnnotateRun,
    ClumpRun,
    ConvertRun,
    ImputeRun,
    LDBlocksRun,
    MetaRun,
    PgsFitRun,
    PgsScoreRun,
    PlotResult,
    ScanRun,
    ValidateRun,
)

# Registry hook for MCP layer
from ._decorator import registered_tools, tools_by_category

# Function imports happen here once the per-tier modules are added.
from .data import convert, impute, validate
from .scans import glm_scan, lmm_scan
from .ld import ld_blocks
from .postgwas import clump, meta
from .pgs import pgs_fit, pgs_score
from .annotate import annotate_hits
from .plotting import manhattan, qq

__all__ = [
    # Functions (tier-1: 12)
    "validate",
    "convert",
    "impute",
    "lmm_scan",
    "glm_scan",
    "ld_blocks",
    "clump",
    "meta",
    "pgs_fit",
    "pgs_score",
    "annotate_hits",
    "manhattan",
    "qq",
    # Result types
    "ValidateRun",
    "ConvertRun",
    "ImputeRun",
    "ScanRun",
    "LDBlocksRun",
    "ClumpRun",
    "MetaRun",
    "PgsFitRun",
    "PgsScoreRun",
    "AnnotateRun",
    "PlotResult",
    # MCP registry hooks
    "registered_tools",
    "tools_by_category",
]
