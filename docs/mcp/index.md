# MCP server

The `torchgenomics-mcp` server publishes the 13 tier-1
[`torchgenomics.api`](../api/quickstart.md) functions as
[Model Context Protocol](https://modelcontextprotocol.io) tools. Any MCP-aware
LLM client — Claude Desktop, Claude Code, Cursor, custom — can then invoke
TorchGenomics as a tool: "run an LMM GWAS on this BED file", "annotate these
hits to maize genes", "fit a PRS-CS polygenic score from these sumstats".

## Installation

```bash
pip install "torchgenomics[mcp]"
```

This installs the MCP Python SDK alongside TorchGenomics and wires the
`torchgenomics-mcp` console script.

Confirm the server boots and lists its tools:

```bash
torchgenomics-mcp --list-tools | head -20
```

Output (truncated):

```json
[
  {
    "name": "tg_validate",
    "title": "Validate a TorchGenomics dataset",
    "description": "Run pre-flight validation ...",
    "category": "data",
    "long_running": false,
    "tags": ["validation", "preflight", "qc"]
  },
  ...
]
```

## Configure your MCP client

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "torchgenomics": {
      "command": "torchgenomics-mcp"
    }
  }
}
```

If `torchgenomics-mcp` isn't on `PATH` in the launched Python environment,
give the absolute path: `"command": "/path/to/venv/bin/torchgenomics-mcp"`.

Restart Claude Desktop. The server should appear in the "Tools" panel with
13 tools listed.

### Claude Code

Same config shape; add to `~/.claude/claude_code_config.json` or per the
[Claude Code MCP docs](https://docs.claude.com/en/docs/claude-code/mcp).

### Other clients

Any client that speaks MCP stdio works. See
[modelcontextprotocol.io](https://modelcontextprotocol.io) for the
protocol spec.

## The 13 tools

| Category | Tool | Wraps |
|---|---|---|
| Data | `tg_validate` | `torchgenomics.api.validate` |
| Data | `tg_convert` | `torchgenomics.api.convert` |
| Data | `tg_impute` | `torchgenomics.api.impute` |
| Scan | `tg_lmm_scan` | `torchgenomics.api.lmm_scan` |
| Scan | `tg_glm_scan` | `torchgenomics.api.glm_scan` |
| LD | `tg_ld_blocks` | `torchgenomics.api.ld_blocks` |
| Post-GWAS | `tg_clump` | `torchgenomics.api.clump` |
| Post-GWAS | `tg_meta` | `torchgenomics.api.meta` |
| PGS | `tg_pgs_fit` | `torchgenomics.api.pgs_fit` |
| PGS | `tg_pgs_score` | `torchgenomics.api.pgs_score` |
| Annotate | `tg_annotate` | `torchgenomics.api.annotate_hits` |
| Plot | `tg_manhattan` | `torchgenomics.api.manhattan` |
| Plot | `tg_qq` | `torchgenomics.api.qq` |

Each tool's input schema is auto-derived from the corresponding api
function's typed signature. `Path` parameters become `string`, `Literal`
enums become JSON Schema `enum`, optional parameters become nullable.
The `progress_callback` parameter (a Python callable) is stripped at the
MCP boundary.

## What the LLM sees

When the LLM lists tools, it gets the tool name, title, description,
input schema (parameter names + types + descriptions), and tags. For
example, `tg_lmm_scan` advertises:

> Run a single-trait linear mixed-model GWAS. Streams genotype chunks,
> auto-computes a VanRaden GRM (unless one is provided), fits the null
> with REML, runs the variant-wise score / Wald / LRT test, applies
> multiple-testing correction, and writes a TSV + Parquet summary.
> Returns ScanRun with .top_hits DataFrame, .lambda_gc, .manhattan() /
> .qq() plotting.

Input schema includes `genotype` (string), `phenotype` (string),
`covariate` (string | null), `correction` (enum: "bonferroni" | "bh" |
"by" | ...), `chunk_size` (integer), `maf_min` (number), `device`
(enum: "cpu" | "cuda" | "auto"), etc.

When the LLM calls the tool, it gets back a JSON-encoded dict matching
the [`ScanRun.to_dict()`](../api/quickstart.md#result-objects) shape:
top hits inline (top-50 by p-value), full results file path, summary
stats, runtime, log excerpt.

## Long-running operations

`tg_lmm_scan`, `tg_glm_scan`, `tg_impute`, `tg_pgs_fit`, `tg_ld_blocks`
are flagged `long_running=True` in the tool registry. In v0.4.0 they
run synchronously — the LLM blocks on the call until completion. Async
job-ID semantics with MCP `progress` notifications are deferred to
v0.5.0; transport-level streaming and HTTP / SSE transports likewise.

## Programmatic use

If you want to embed the MCP server in your own service:

```python
from torchgenomics.mcp import build_server, list_tools

# 1. List tools (no SDK required).
tools = list_tools()
for t in tools:
    print(t["name"], t["title"])

# 2. Boot a FastMCP server in-process.
server = build_server()
server.run(transport="stdio")
```

## Troubleshooting

**Server doesn't appear in Claude Desktop's tools panel.**
Check that `torchgenomics-mcp` is on `PATH` in the Python environment
Claude Desktop spawns. Try `"command": "/abs/path/to/python", "args":
["-m", "torchgenomics.mcp.server"]` as an alternative.

**Tool calls hang on long-running scans.**
v0.4.0 calls are synchronous. A 10-million-variant biobank LMM scan
can take 20+ minutes; consider running the analysis via the CLI and
asking the LLM to operate on the produced TSV instead.

**`ModuleNotFoundError: No module named 'mcp'`.**
You installed the base package without the MCP extra. Run
`pip install "torchgenomics[mcp]"`.

**Schema validation errors at tool registration.**
File an issue with the output of `torchgenomics-mcp --list-tools`. This
is a Pydantic-related compatibility issue and can usually be patched in
the wrapper layer.
