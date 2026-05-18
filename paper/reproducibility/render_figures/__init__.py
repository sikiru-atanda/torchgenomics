"""Registry of figure-rendering functions.

Each renderer is registered via ``@register("F<N>")`` and takes the output
directory as its argument. It must return ``(output_path, numeric_dict)``.
The numeric_dict aggregates into ``paper/reproducibility/manifest.json``,
which is the authoritative record of every number cited in the paper.

Tier 4 D2 authors one module per figure under this package
(``f1_capability_map.py``, ``f2_equivalence_grid.py``, ..., ``f7_specialty.py``)
plus helper artifacts (``f6_streaming_p_sweep.py``, ``f7_gu_lro_simulators.py``).
Import the module here so the @register decorator runs at package import.
"""
from __future__ import annotations

from typing import Any, Callable

REGISTRY: dict[str, Callable[..., tuple[Any, dict]]] = {}


def register(fig_id: str) -> Callable[[Callable[..., tuple[Any, dict]]], Callable[..., tuple[Any, dict]]]:
    """Decorator: register a renderer under a figure id (e.g. ``"F2"``)."""
    def _decorator(fn: Callable[..., tuple[Any, dict]]) -> Callable[..., tuple[Any, dict]]:
        if fig_id in REGISTRY:
            raise ValueError(f"renderer {fig_id!r} already registered")
        REGISTRY[fig_id] = fn
        return fn
    return _decorator


# Each Tier 4 D2 renderer module imports this `register` and applies it on
# its main function. Adding the import here causes the decorator to fire
# at package-import time.
#
# Tier 4 D2 populates this list as each renderer lands:
from . import f1_capability_map  # noqa: F401  (Tier 4 D2.1)
from . import f2_equivalence_grid   # noqa: F401  (Tier 4 D2.2)
#
# Pending in subsequent D2 sub-tasks:
#   from . import f3_haplotype        # D2.3
#   from . import f4_multiomics       # D2.4
#   from . import f5_polyploid        # D2.5
#   from . import f6_gpu_streaming    # D2.6
#   from . import f7_specialty        # D2.7
