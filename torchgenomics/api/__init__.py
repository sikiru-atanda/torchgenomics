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

# Registry hook for MCP layer
from ._decorator import registered_tools, tools_by_category

# Result types (re-exported for type hints / isinstance checks)
from ._results import (
    AnnotateRun,
    ClumpRun,
    ConvertRun,
    GwasResult,
    ImputeRun,
    LDBlocksRun,
    LGEBVResult,
    MetaRun,
    PgsFitRun,
    PgsScoreRun,
    PlotResult,
    ScanRun,
    ValidateRun,
)
from .annotate import annotate_hits

# Function imports happen here once the per-tier modules are added.
from .data import convert, impute, validate
from .gwas import GwasComparison, gwas, models
from .ld import ld_blocks
from .lgebv import lgebv
from .pgs import pgs_fit, pgs_score
from .plotting import manhattan, qq
from .postgwas import clump, meta
from .scans import glm_scan, lmm_scan

__all__ = [
    # Functions (tier-1: 12 + lgebv)
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
    "lgebv",
    "annotate_hits",
    "manhattan",
    "qq",
    # Friendly-API orchestrator (tg.gwas)
    "gwas",
    "models",
    "GwasComparison",
    # Result types
    "ValidateRun",
    "ConvertRun",
    "ImputeRun",
    "ScanRun",
    "GwasResult",
    "LDBlocksRun",
    "ClumpRun",
    "MetaRun",
    "PgsFitRun",
    "PgsScoreRun",
    "LGEBVResult",
    "AnnotateRun",
    "PlotResult",
    # MCP registry hooks
    "registered_tools",
    "tools_by_category",
]
