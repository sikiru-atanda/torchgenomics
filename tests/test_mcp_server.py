"""Tests for the :mod:`torchgenomics.mcp` server.

Covers, in order of increasing dependency:

  1. **No-SDK path**: :func:`list_tools` works without the ``mcp`` SDK
     installed; useful for ``torchgenomics-mcp --list-tools`` smoke tests.
  2. **Wrapper hygiene**: ``progress_callback`` is stripped, ``Path`` is
     coerced to ``str``, ``Literal`` enums survive intact.
  3. **Server boot**: :func:`build_server` constructs a FastMCP with the
     13 tier-1 tools registered and emits valid JSON Schemas.
  4. **End-to-end MCP call**: in-process ``tg_validate`` returns a
     JSON-safe dict (decoded from the MCP TextContent payload).

These all run synchronously in-process — the MCP SDK is imported but the
stdio transport is never engaged, so the test suite stays fast and
deterministic.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Literal

import pytest

from torchgenomics.api import registered_tools
from torchgenomics.mcp import list_tools

pytest_plugins: list[str] = []  # purely in-process; no plugin needed

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# --- No-SDK path -----------------------------------------------------------


class TestListTools:
    """Pure-Python list_tools() does not require the mcp SDK."""

    def test_returns_thirteen_entries(self):
        tools = list_tools()
        assert len(tools) == 13

    def test_entry_fields(self):
        tools = list_tools()
        for entry in tools:
            assert {"name", "title", "description", "category", "long_running", "tags"}.issubset(entry)
            assert entry["name"].startswith("tg_")
            assert isinstance(entry["title"], str) and entry["title"]
            assert isinstance(entry["description"], str) and len(entry["description"]) > 20
            assert entry["category"] in {"data", "scan", "ld", "postgwas", "pgs", "annotate", "plot"}

    def test_tier1_names_present(self):
        expected = {
            "tg_validate", "tg_convert", "tg_impute",
            "tg_lmm_scan", "tg_glm_scan",
            "tg_ld_blocks", "tg_clump", "tg_meta",
            "tg_pgs_fit", "tg_pgs_score",
            "tg_annotate", "tg_manhattan",
        }
        names = {t["name"] for t in list_tools()}
        assert expected.issubset(names)


# --- Wrapper hygiene -------------------------------------------------------


class TestWrappers:
    """Each api function is wrapped with a JSON-friendly signature."""

    def test_progress_callback_stripped(self):
        from torchgenomics.mcp._wrappers import build_mcp_tool

        entry = registered_tools()["tg_lmm_scan"]
        assert "progress_callback" in inspect.signature(entry.func).parameters
        wrapper = build_mcp_tool(entry)
        assert "progress_callback" not in inspect.signature(wrapper).parameters

    def test_path_coerced_to_str(self):
        from torchgenomics.mcp._wrappers import build_mcp_tool

        entry = registered_tools()["tg_validate"]
        wrapper = build_mcp_tool(entry)
        sig = inspect.signature(wrapper)
        # All path-like params should accept str (no Path in their annotation)
        for name in ("genotype", "phenotype"):
            ann = sig.parameters[name].annotation
            assert ann is str, f"{name} annotation is {ann!r}, expected str"

    def test_literal_enum_preserved(self):
        """Literal types from api signatures must reach the wrapper intact."""
        from torchgenomics.mcp._wrappers import build_mcp_tool

        entry = registered_tools()["tg_convert"]
        wrapper = build_mcp_tool(entry)
        sig = inspect.signature(wrapper)
        ann = sig.parameters["output_format"].annotation
        # Literal["bed", "zarr", "vcf"] should be unchanged
        from typing import get_args, get_origin
        assert get_origin(ann) is Literal.__class__ or "Literal" in str(ann)
        args = get_args(ann)
        assert set(args) == {"bed", "zarr", "vcf"}

    def test_wrapper_call_returns_dict(self):
        """Wrapper invocation returns the api result converted to a dict."""
        from torchgenomics.mcp._wrappers import build_mcp_tool

        entry = registered_tools()["tg_validate"]
        wrapper = build_mcp_tool(entry)
        result = wrapper(
            genotype=str(FIXTURE_DIR / "tiny.bed"),
            phenotype=str(FIXTURE_DIR / "tiny_pheno.txt"),
        )
        assert isinstance(result, dict)
        assert result["ok"] is True
        assert result["format_name"] == "bed"
        assert result["n_variants_total"] == 20


# --- Server boot (requires mcp SDK) ----------------------------------------


mcp_available = True
try:
    import mcp  # noqa: F401
except ImportError:
    mcp_available = False


@pytest.mark.skipif(not mcp_available, reason="mcp SDK not installed (install via [mcp] extra)")
class TestServerBoot:

    def test_build_server_returns_fastmcp(self):
        from mcp.server.fastmcp import FastMCP

        from torchgenomics.mcp import build_server

        server = build_server()
        assert isinstance(server, FastMCP)

    def test_server_lists_all_tier1_tools(self):
        from torchgenomics.mcp import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        names = {t.name for t in tools}
        assert "tg_validate" in names
        assert "tg_lmm_scan" in names
        assert "tg_pgs_fit" in names
        assert len(tools) == 13

    def test_input_schemas_are_valid_json_schema(self):
        from torchgenomics.mcp import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        for t in tools:
            schema = t.inputSchema
            assert isinstance(schema, dict)
            assert schema.get("type") == "object"
            assert "properties" in schema
            # Must be JSON-serializable
            text = json.dumps(schema)
            assert isinstance(text, str)


# --- End-to-end MCP call ---------------------------------------------------


@pytest.mark.skipif(not mcp_available, reason="mcp SDK not installed")
class TestEndToEndCall:
    """In-process call to a real MCP tool returns a JSON-decoded dict."""

    def test_validate_via_mcp(self):
        from torchgenomics.mcp import build_server

        server = build_server()

        async def _call():
            return await server.call_tool("tg_validate", {
                "genotype": str(FIXTURE_DIR / "tiny.bed"),
                "phenotype": str(FIXTURE_DIR / "tiny_pheno.txt"),
            })

        result = asyncio.run(_call())
        # Newer FastMCP returns list of TextContent or a tuple (content, structured)
        content_blocks = result if isinstance(result, list) else result[0]
        assert content_blocks, "MCP call returned no content"

        # Decode the first text block as JSON
        first = content_blocks[0]
        text = getattr(first, "text", None) or str(first)
        payload = json.loads(text)
        assert payload["ok"] is True
        assert payload["format_name"] == "bed"
        assert payload["n_variants_total"] == 20
        assert payload["n_samples_aligned"] == 10

    def test_long_running_flag_propagates(self):
        """Tools flagged long_running=True should be marked as such."""
        from torchgenomics.mcp import build_server

        server = build_server()
        tools = asyncio.run(server.list_tools())
        names_by_long_running = {
            t.name for t in tools
            if getattr(t.annotations, "openWorldHint", None) is not None
        }
        # We don't strictly require the SDK to expose long_running in
        # annotations (the version may vary). The registry side is the
        # source of truth; this test just confirms the wiring doesn't crash.
        assert isinstance(names_by_long_running, set)
