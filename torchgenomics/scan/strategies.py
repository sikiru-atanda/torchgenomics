"""Scan strategies: null-VC-fixed, per-SNP-refit, hybrid heuristics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedNullStrategy:
    """Fit null once, reuse for all SNP chunks (standard GWAS)."""

    refit_per_snp: bool = False


@dataclass(frozen=True)
class PerSNPRefitStrategy:
    """Refit variance components per SNP (expensive, for diagnostics)."""

    refit_per_snp: bool = True
