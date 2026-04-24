"""Polyploid phasing via PolyOrigin.jl (Phase 56).

Thin external wrapper around the Julia package PolyOrigin.jl for
connected tetraploid / hexaploid F1 populations. Chains off Phase 55's
dosage-call output. See
``docs/superpowers/specs/2026-04-23-phase-56-polyploid-phasing-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_PLOIDIES: frozenset[int] = frozenset({2, 4, 6})
_HASH_CHUNK_BYTES = 1 << 20
