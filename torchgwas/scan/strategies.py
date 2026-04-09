"""Scan strategies: null-VC-fixed, per-SNP-refit, hybrid heuristics."""

from __future__ import annotations


class FixedNullStrategy:
    """Fit null once, reuse for all SNP chunks (standard GWAS)."""
    pass


class PerSNPRefitStrategy:
    """Refit variance components per SNP (expensive, for diagnostics)."""
    pass
