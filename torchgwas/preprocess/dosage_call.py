"""Polyploid allele dosage assignment via updog (Phase 55).

Thin external-wrapper around the R package ``updog``. Converts VCF read
counts (AD field) into posterior ``P(dosage=0..k)`` tensors. See
``docs/superpowers/specs/2026-04-22-polyploid-dosage-assignment-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_MODELS: frozenset[str] = frozenset({
    "norm", "hw", "bb", "s1", "f1", "s1pp", "f1pp", "flex", "uniform",
})

_UPDOG_CHECKED: Optional[bool] = None  # cached result of environment probe


@dataclass
class DosageCallResult:
    probs: Tensor
    sample_ids: list[str]
    variant_ids: list[str]
    ploidy: int
    tool: str
    tool_version: str
    model: str
    mean_dosage_var: Tensor
    allele_freq: Tensor
    n_missing: int
    input_hash: str
    cmd: str


def _check_environment(rscript: Optional[str]) -> str:
    """Probe for Rscript + updog; cache the verdict. Returns Rscript path.

    Raises RuntimeError with install pointers if either is missing.
    Cached on a module-level flag; the probe runs at most once per process.
    """
    global _UPDOG_CHECKED

    if rscript is None:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Install R >= 4.0 and ensure Rscript is on "
            "PATH, or pass rscript=<path>."
        )

    if _UPDOG_CHECKED is True:
        return rscript

    probe = subprocess.run(
        [rscript, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        _UPDOG_CHECKED = False
        raise RuntimeError(
            "updog R package not installed. Install with: "
            "Rscript -e 'install.packages(\"updog\")'. "
            f"(probe stderr: {probe.stderr.strip()})"
        )

    logger.info("updog R package detected: version %s", probe.stdout.strip())
    _UPDOG_CHECKED = True
    return rscript


def _validate_kwargs(*, ploidy: int, model: str) -> None:
    """Validate ploidy and model kwargs before R dispatch.

    Raises ValueError if ploidy is out of range [2, 8] or model is not in
    the updog flexdog set.
    """
    if not (2 <= ploidy <= 8):
        raise ValueError(
            f"ploidy must be in [2, 8]; got {ploidy}. updog is validated "
            "up to ploidy=8 upstream."
        )
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model={model!r} not in updog flexdog set: {sorted(_VALID_MODELS)}"
        )
