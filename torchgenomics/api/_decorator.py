"""Registration decorator for :mod:`torchgenomics.api` tier-1 functions.

The decorator records function metadata in a module-level registry that
:mod:`torchgenomics.mcp` reads to publish tools. It is a no-op at call
time — calling a decorated function behaves exactly like calling the
undecorated one. This lets the api module stay usable without the MCP
SDK installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ToolEntry:
    """Metadata about a registered api function."""

    name: str
    func: Callable[..., Any]
    title: str
    description: str
    long_running: bool
    category: str  # "data" | "scan" | "ld" | "postgwas" | "pgs" | "annotate" | "plot"
    tags: list[str] = field(default_factory=list)


_REGISTRY: dict[str, ToolEntry] = {}


def tool(
    *,
    name: str,
    title: str,
    description: str,
    long_running: bool = False,
    category: str = "misc",
    tags: list[str] | None = None,
) -> Callable[[F], F]:
    """Mark a function as a tier-1 MCP tool.

    The function itself is unchanged — this decorator only records metadata.

    Parameters
    ----------
    name
        Tool name as it will appear to MCP clients (e.g. ``"tg_lmm_scan"``).
        Must be unique across the registry.
    title
        Short human-readable title (a few words).
    description
        Long-form description shown to the LLM during tool selection. Should
        be specific enough that the LLM knows when to choose this tool.
    long_running
        If True, the MCP wrapper will emit periodic progress notifications
        and surface :attr:`torchgenomics.api._helpers.ProgressCallback` to
        the underlying function.
    category
        Coarse grouping (data / scan / ld / postgwas / pgs / annotate / plot).
    tags
        Optional keywords used for discovery (e.g. ``["gwas", "lmm", "biobank"]``).
    """

    def _decorate(fn: F) -> F:
        entry = ToolEntry(
            name=name,
            func=fn,
            title=title,
            description=description,
            long_running=long_running,
            category=category,
            tags=list(tags or []),
        )
        if name in _REGISTRY:
            existing = _REGISTRY[name].func
            if existing is not fn:
                raise ValueError(
                    f"Duplicate MCP tool name {name!r}: already registered to {existing!r}"
                )
        _REGISTRY[name] = entry
        return fn

    return _decorate


def registered_tools() -> dict[str, ToolEntry]:
    """Return a copy of the current tool registry."""
    return dict(_REGISTRY)


def tools_by_category(category: str) -> list[ToolEntry]:
    return [t for t in _REGISTRY.values() if t.category == category]
