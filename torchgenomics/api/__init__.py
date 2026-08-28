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

# CLI bridge: run tier-2 CLI subcommands (bayes-scan, ldsc, mediate, ...)
# programmatically for Python/R parity. See `__getattr__` below for the
# module-level dispatch that exposes each subcommand as `api.<name>(...)`.
from ._cli_bridge import run_cli_subcommand

# Registry hook for MCP layer
from ._decorator import registered_tools, tools_by_category

# Result types (re-exported for type hints / isinstance checks)
from ._results import (
    AnnotateRun,
    CliRun,
    ClumpRun,
    ColocRun,
    ConvertRun,
    GwasResult,
    ImputeRun,
    LDBlocksRun,
    LGEBVResult,
    MetaRun,
    MRRun,
    MRMegaRun,
    SMRRun,
    PowerRun,
    WinnersCurseRun,
    EnrichmentRun,
    HessRun,
    FineMapRun,
    PgsFitRun,
    PgsScoreRun,
    PlotResult,
    ScanRun,
    ValidateRun,
)
from .annotate import annotate_hits

# Function imports happen here once the per-tier modules are added.
from .data import convert, impute, validate
from .gwas import GwasComparison, Recommendation, gwas, list_models, recommend
from .ld import ld_blocks
from .lgebv import lgebv
from .pgs import pgs_fit, pgs_score
from .plotting import manhattan, qq
from .postgwas import (
    clump,
    coloc,
    finemap,
    gene_set_enrichment,
    hess,
    meta,
    mr,
    mr_mega,
    power,
    smr,
    winners_curse,
)
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
    "mr",
    "smr",
    "mr_mega",
    "power",
    "winners_curse",
    "gene_set_enrichment",
    "hess",
    "coloc",
    "finemap",
    "pgs_fit",
    "pgs_score",
    "lgebv",
    "annotate_hits",
    "manhattan",
    "qq",
    # Friendly-API orchestrator (tg.gwas)
    "gwas",
    "list_models",
    "GwasComparison",
    "recommend",
    "Recommendation",
    # Result types
    "ValidateRun",
    "ConvertRun",
    "ImputeRun",
    "ScanRun",
    "GwasResult",
    "LDBlocksRun",
    "ClumpRun",
    "MetaRun",
    "MRRun",
    "MRMegaRun",
    "SMRRun",
    "PowerRun",
    "WinnersCurseRun",
    "EnrichmentRun",
    "HessRun",
    "FineMapRun",
    "ColocRun",
    "PgsFitRun",
    "PgsScoreRun",
    "LGEBVResult",
    "AnnotateRun",
    "PlotResult",
    "CliRun",
    # CLI bridge (tier-2 subcommand parity — see `__getattr__` below)
    "run_cli_subcommand",
    # MCP registry hooks
    "registered_tools",
    "tools_by_category",
]


def __getattr__(name: str):
    """Expose tier-2 CLI subcommands as api callables (Python/R parity).

    Python's module ``__getattr__`` (PEP 562) only fires for names *not*
    already resolvable as a module attribute, so this never shadows the
    hand-crafted tier-1 functions imported above (``gwas``, ``mr``,
    ``clump``, ...) or the result classes — those are found by normal
    attribute lookup first and this function is never called for them.

    For everything else, ``name`` (underscores) is compared against
    :data:`torchgenomics._manifest._TIER_2_CLI_SUBCOMMANDS` (dashes) by
    translating ``_`` -> ``-``. A match returns a callable —
    ``functools.partial(run_cli_subcommand, dashed_name)`` — that runs that
    subcommand's real CLI handler via :func:`run_cli_subcommand` and returns
    a :class:`~torchgenomics.api.CliRun`. Anything that isn't a tier-2
    subcommand name raises ``AttributeError``, exactly as an unrecognized
    module attribute normally would (so ``hasattr``, ``getattr(..., default)``,
    and star-imports keep their ordinary semantics).
    """
    from functools import partial

    from .._manifest import _TIER_2_CLI_SUBCOMMANDS

    dashed = name.replace("_", "-")
    if dashed in _TIER_2_CLI_SUBCOMMANDS:
        return partial(run_cli_subcommand, dashed)
    raise AttributeError(f"module 'torchgenomics.api' has no attribute {name!r}")
