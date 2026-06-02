"""MCP server for TorchGenomics.

Publishes the 13 tier-1 :mod:`torchgenomics.api` functions as MCP tools.
See :mod:`torchgenomics.mcp.server` for the entry point and details.
"""
from __future__ import annotations

from .server import build_server, list_tools, main

__all__ = ["build_server", "list_tools", "main"]
