"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20


@dataclass
class PhasingResult:
    haplotypes: Tensor                     # (n_offspring, ploidy, m) int8 — argmax of origin_probs
    origin_probs: Tensor                   # (n_offspring, ploidy, m, n_parent_haps) float64
    parent_phased: Tensor                  # (n_parents, max_ploidy, m) int8
    offspring_ids: list[str]
    parent_ids: list[str]
    variant_ids: list[str]
    chrom: list[str]
    pos_bp: Tensor                         # (m,) int64
    pos_cm: Tensor                         # (m,) float64
    per_individual_ploidy: dict[str, int]
    map_refined: bool
    valent_diag: pd.DataFrame
    postdose_probs: Tensor                 # (n_offspring, m, max_ploidy+1) float64
    tool: str
    tool_version: str
    input_hash: str
    cmd: str
    workdir: str | None
