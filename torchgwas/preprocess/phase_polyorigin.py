"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib  # noqa: F401 — used by Tasks 5/10
import json  # noqa: F401 — used by Tasks 10/11
import logging
import os  # noqa: F401 — used by Task 13
from dataclasses import dataclass  # noqa: F401 — used by Task 2
from pathlib import Path  # noqa: F401 — used by Tasks 3–11

import pandas as pd  # noqa: F401 — used by Tasks 3–5
import torch  # noqa: F401 — used by Tasks 2/10
from torch import Tensor  # noqa: F401 — used by Task 2

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20
