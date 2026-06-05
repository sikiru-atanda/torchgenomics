"""TorchGenomics: GPU-accelerated statistical and quantitative genomics on PyTorch.

A single engine that covers GWAS + post-GWAS + PGS + MR + TWAS + multi-omics
+ LD + imputation + annotation + visualization, with native C++ accelerators
and biobank-scale streaming.

Three audiences share one code path:

  - **Novice / notebook**: one-call workflows with smart defaults::

        import torchgenomics as tg
        r = tg.lmm_scan(genotype="data.bed", phenotype="pheno.tsv")
        print(r.summary())
        r.manhattan()

  - **Advanced / research**: composable low-level building blocks under
    :mod:`torchgenomics.models`, :mod:`torchgenomics.scan`,
    :mod:`torchgenomics.linalg`, etc.

  - **LLM / MCP tools**: the same tier-1 functions registered with
    JSON-safe inputs and outputs under :mod:`torchgenomics.mcp`.
"""
from __future__ import annotations

__version__ = "0.4.0"

# --- Top-level high-level API ------------------------------------------------
# These re-exports give novice users a clean one-line surface. Advanced users
# continue to import from submodules directly (no breakage — these names just
# also exist at the top level for convenience).

from .api import (
    # Result types (for type hints / isinstance)
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
    # 12 tier-1 functions
    annotate_hits,
    clump,
    convert,
    glm_scan,
    impute,
    ld_blocks,
    lmm_scan,
    manhattan,
    meta,
    pgs_fit,
    pgs_score,
    qq,
    validate,
)

__all__ = [
    "__version__",
    # Tier-1 functions
    "annotate_hits",
    "clump",
    "convert",
    "glm_scan",
    "impute",
    "ld_blocks",
    "lmm_scan",
    "manhattan",
    "meta",
    "pgs_fit",
    "pgs_score",
    "qq",
    "validate",
    # Result types
    "AnnotateRun",
    "ClumpRun",
    "ConvertRun",
    "ImputeRun",
    "LDBlocksRun",
    "MetaRun",
    "PgsFitRun",
    "PgsScoreRun",
    "PlotResult",
    "ScanRun",
    "ValidateRun",
]
