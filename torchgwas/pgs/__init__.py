"""Polygenic Score (PGS) construction methods.

Phase 40: GPU-accelerated PGS module providing C+T, LDpred2 family,
and PRS-CS methods over a shared `BasePGSMethod` interface with a
common `LDReference` and `PGSResult` data model.

Public API:
    BasePGSMethod, LDReference, PGSResult,
    ClumpingThresholding, LDpred2Inf, LDpred2Grid, LDpred2Auto, PRSCS,
    score_individuals, build_ld_reference, harmonize_to_reference,
    rhat, ess, check_convergence
"""

from __future__ import annotations

from .base import BasePGSMethod, LDReference, PGSResult
from .ct import ClumpingThresholding
from .diagnostics import check_convergence, ess, rhat
from .ld_ref import build_ld_reference, load_ld_reference, save_ld_reference
from .ldpred2 import LDpred2Auto, LDpred2Grid, LDpred2Inf
from .prscs import PRSCS, sample_gig
from .scoring import ScoringResult, score_individuals
from .sumstats_io import harmonize_to_reference, load_pgs_sumstats
from .validation import PGSValidation, validate_pgs

__all__ = [
    "BasePGSMethod",
    "ClumpingThresholding",
    "LDReference",
    "LDpred2Auto",
    "LDpred2Grid",
    "LDpred2Inf",
    "PGSResult",
    "PRSCS",
    "ScoringResult",
    "build_ld_reference",
    "check_convergence",
    "ess",
    "harmonize_to_reference",
    "load_ld_reference",
    "load_pgs_sumstats",
    "rhat",
    "sample_gig",
    "save_ld_reference",
    "score_individuals",
    "PGSValidation",
    "validate_pgs",
]
