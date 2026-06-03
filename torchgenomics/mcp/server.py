"""TorchGenomics MCP server.

Run via the ``torchgenomics-mcp`` console script (installed by the
``torchgenomics[mcp]`` extra) or programmatically::

    from torchgenomics.mcp.server import build_server
    server = build_server()
    server.run(transport="stdio")

The server publishes the 13 tier-1 tools registered with
:func:`torchgenomics.api.tool` as MCP tools. Stdio transport only in
v0.4.0; HTTP / SSE transports are deferred to a later release.

Clients configure the server via stdio entry, e.g. in Claude Desktop's
``claude_desktop_config.json``::

    {
      "mcpServers": {
        "torchgenomics": {
          "command": "torchgenomics-mcp"
        }
      }
    }
"""
from __future__ import annotations

import logging
from typing import Any

from ..api import registered_tools
from ..api._decorator import ToolEntry
from ._wrappers import build_mcp_tool

logger = logging.getLogger("torchgenomics.mcp")


def _import_fastmcp():
    """Late-import the MCP SDK so the api package stays usable without it."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "torchgenomics.mcp requires the `mcp` SDK. "
            "Install with: pip install 'torchgenomics[mcp]'"
        ) from e
    return FastMCP


def build_server(name: str = "torchgenomics", instructions: str | None = None):
    """Construct a FastMCP server with all tier-1 tools registered."""
    FastMCP = _import_fastmcp()

    if instructions is None:
        instructions = (
            "TorchGenomics: a unified engine for GWAS, post-GWAS, polygenic "
            "scoring, LD analysis, imputation, multi-omics integration, and "
            "genomic visualization. Each tool wraps a one-call workflow with "
            "smart defaults; results carry summary stats inline and write "
            "full tables to disk."
        )

    server = FastMCP(name=name, instructions=instructions)

    entries: dict[str, ToolEntry] = registered_tools()
    for entry in entries.values():
        wrapper = build_mcp_tool(entry)
        server.add_tool(
            wrapper,
            name=entry.name,
            title=entry.title,
            description=entry.description,
        )
        logger.debug("registered MCP tool: %s (%s)", entry.name, entry.category)

    logger.info("torchgenomics MCP server: %d tools registered.", len(entries))
    return server


def list_tools() -> list[dict[str, Any]]:
    """Return tool metadata as plain dicts (no MCP SDK required).

    Useful for ``torchgenomics-mcp --list-tools`` and for in-process tests
    that want to introspect the registry without booting the server.
    """
    out: list[dict[str, Any]] = []
    for entry in registered_tools().values():
        out.append({
            "name": entry.name,
            "title": entry.title,
            "description": entry.description,
            "category": entry.category,
            "long_running": entry.long_running,
            "tags": list(entry.tags),
        })
    return out


def main() -> None:
    """Console-script entry point: boot the stdio MCP server."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="torchgenomics-mcp",
        description="TorchGenomics MCP server (stdio)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List registered tools as JSON and exit (does not start the server).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.list_tools:
        json.dump(list_tools(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    server = build_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
