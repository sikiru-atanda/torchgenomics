"""Discover-first Julia bootstrap for PolyOrigin.jl (Phase 56 internal).

Not public API. ``run_polyorigin`` uses ``get_runtime`` to obtain a
``juliacall`` handle. Search order: explicit kwarg → TORCHGWAS_JULIA env
var → known install paths → juliacall's own managed install (only if the
caller consents). See the Phase 56 design spec Section 2.3 for rationale.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_jl: Any = None
_polyorigin: Any = None
_version: str | None = None
