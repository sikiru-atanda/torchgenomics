"""MCP-side wrappers for :mod:`torchgenomics.api` tier-1 functions.

Each :class:`torchgenomics.api._decorator.ToolEntry` registered by the api
layer is converted into an MCP-callable function by :func:`build_mcp_tool`:

  - ``Path`` and ``str | Path`` parameters are coerced to plain ``str`` so
    JSON Schema generation produces clean ``"string"`` fields.
  - ``progress_callback: Callable | None`` arguments are dropped (callables
    cannot travel through JSON / MCP).
  - ``pandas.DataFrame`` union parameters (e.g. ``sumstats: str | Path |
    pd.DataFrame``) are narrowed to ``str`` for the MCP signature; the LLM
    passes file paths.
  - The wrapper calls the api function and, when the result implements
    ``.to_dict()``, returns the JSON-safe dict. Other returns pass through.

The original api function and its full signature remain unchanged — the
wrapper is purely additive at the MCP boundary.
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

from ..api._decorator import ToolEntry


def _is_union(origin: Any) -> bool:
    """True iff ``origin`` represents a Union (typing.Union or PEP 604 ``X | Y``)."""
    return origin is Union or origin is types.UnionType


_DROPPED_PARAMS = {"progress_callback"}
"""Parameters dropped from the MCP-side signature (not JSON-serializable)."""


def _clean_annotation(annotation: Any) -> Any:
    """Return a JSON-Schema-friendly version of a Python type annotation.

    Coerces ``Path`` (or unions containing ``Path``) to ``str``, and drops
    ``pandas.DataFrame`` arms from unions — the MCP-side caller passes
    file paths, not in-memory DataFrames.

    Non-Union origins (Literal, list, dict, ...) pass through unchanged
    so we don't accidentally re-wrap Literal's string values into Union[
    ForwardRef(...)].
    """
    if annotation is Path:
        return str
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if not _is_union(origin):
        # Literal, list, dict, Annotated, etc. → pass through unchanged.
        return annotation

    args = get_args(annotation)
    cleaned_args = []
    for a in args:
        if a is Path:
            cleaned_args.append(str)
            continue
        try:
            from pandas import DataFrame  # type: ignore[import-untyped]
            if isinstance(a, type) and issubclass(a, DataFrame):
                continue  # drop DataFrame arms
        except (ImportError, TypeError):
            pass
        cleaned_args.append(a)

    # Dedupe while preserving order
    seen = []
    for a in cleaned_args:
        if a not in seen:
            seen.append(a)
    if not seen:
        return str
    if len(seen) == 1:
        return seen[0]

    # Reconstruct the Union (or Optional)
    return Union[tuple(seen)]  # type: ignore[return-value]


def _build_mcp_signature(api_fn: Callable[..., Any]) -> inspect.Signature:
    """Derive a JSON-friendly Signature for the MCP wrapper from ``api_fn``.

    The api modules use ``from __future__ import annotations``, so
    ``param.annotation`` is a string at this point. We resolve via
    :func:`typing.get_type_hints` first, then clean each type.
    """
    import typing

    sig = inspect.signature(api_fn)
    try:
        resolved_hints = typing.get_type_hints(api_fn)
    except Exception:  # noqa: BLE001
        resolved_hints = {}

    new_params: list[inspect.Parameter] = []
    for name, param in sig.parameters.items():
        if name in _DROPPED_PARAMS:
            continue
        if name in resolved_hints:
            new_annotation = _clean_annotation(resolved_hints[name])
        elif param.annotation is not inspect.Parameter.empty:
            new_annotation = _clean_annotation(param.annotation)
        else:
            new_annotation = inspect.Parameter.empty
        new_params.append(
            param.replace(
                kind=inspect.Parameter.KEYWORD_ONLY if param.kind is inspect.Parameter.VAR_KEYWORD else param.kind,
                annotation=new_annotation,
            )
        )
    return sig.replace(parameters=new_params, return_annotation=dict)


def build_mcp_tool(entry: ToolEntry) -> Callable[..., Any]:
    """Build an MCP-callable wrapper around ``entry.func``.

    Returned function:
      - Has a cleaned ``__signature__`` (Path → str, progress_callback dropped).
      - Forwards ``**kwargs`` to the api function unchanged.
      - Returns ``result.to_dict()`` if the result has it; otherwise the
        raw return value.
      - Carries ``__name__`` / ``__doc__`` from the registry entry so MCP
        clients see the right tool identity.
    """
    api_fn = entry.func
    mcp_sig = _build_mcp_signature(api_fn)

    def _mcp_wrapper(**kwargs):
        result = api_fn(**kwargs)
        if hasattr(result, "to_dict") and callable(result.to_dict):
            try:
                return result.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return result

    _mcp_wrapper.__name__ = entry.name
    _mcp_wrapper.__qualname__ = entry.name
    _mcp_wrapper.__doc__ = entry.description
    _mcp_wrapper.__signature__ = mcp_sig  # type: ignore[attr-defined]
    _mcp_wrapper.__annotations__ = {
        p.name: p.annotation
        for p in mcp_sig.parameters.values()
        if p.annotation is not inspect.Parameter.empty
    }
    _mcp_wrapper.__annotations__["return"] = dict
    return _mcp_wrapper
