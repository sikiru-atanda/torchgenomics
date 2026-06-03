# torchgenomics/_manifest/__init__.py
"""Emit the tier-2 CLI argparse schema as JSON for rTorchGenomics codegen.

The R-side ``codegen.R`` reads this manifest at package-build time to
generate ``R/api_auto.R`` — one R wrapper function per tier-2 CLI
subcommand. The manifest is the contract between the Python CLI and
the R-side auto-generation.
"""
from __future__ import annotations

import argparse
from typing import Any

from .. import __version__
from ..cli import _build_parser


_TIER_2_CLI_SUBCOMMANDS: set[str] = {
    "mvlmm-scan", "poly-scan", "mklmm-scan", "gxe-scan", "set-scan",
    "bayes-scan", "bayes-scan-rss", "met-scan", "farmcpu-scan",
    "blink-scan", "threshold-scan", "family-scan", "conditional-scan",
    "mtmet-scan", "ocf-scan", "knockoff-scan", "gu-scan", "lro-scan",
    "glmm-scan", "me-glmm-scan", "survival-scan", "rr-scan",
    "rr-met-scan", "twas-scan", "combine-gwas-twas", "ldsc", "ldsc-rg",
    "dosage-call", "phase-poly", "mediate", "mediate-scan", "pipeline",
}


def _action_type_name(action: argparse.Action) -> str:
    """Map an argparse action to a JSON-Schema-ish type name."""
    if action.type is None:
        if isinstance(action, argparse._StoreTrueAction):
            return "bool"
        if isinstance(action, argparse._StoreFalseAction):
            return "bool"
        return "str"
    name = getattr(action.type, "__name__", None)
    if name in {"int", "float", "bool", "str"}:
        return name
    return "str"


def _extract_args(subparser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    args: list[dict[str, Any]] = []
    for action in subparser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        # argparse flag names: ["--genotype", "-g"]. We use the first
        # long-form flag stripped of leading dashes as the arg name.
        flag = next((s for s in action.option_strings if s.startswith("--")), None)
        if flag is None:
            # Positional or short-only — skip; tier-2 CLI is long-flag only.
            continue
        name = flag.lstrip("-").replace("-", "_")
        args.append({
            "name": name,
            "type": _action_type_name(action),
            "required": bool(action.required),
            "default": action.default if action.default is not argparse.SUPPRESS else None,
            "help": action.help or "",
        })
    return args


def build_manifest() -> dict[str, Any]:
    """Build the tier-2 manifest by walking the CLI's argparse subparsers."""
    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )

    commands: list[dict[str, Any]] = []
    for cli_name, subparser in subparsers_action.choices.items():
        if cli_name not in _TIER_2_CLI_SUBCOMMANDS:
            continue
        commands.append({
            "name": cli_name.replace("-", "_"),
            "cli_subcommand": cli_name,
            "description": (subparser.description or "").strip(),
            "args": _extract_args(subparser),
        })

    commands.sort(key=lambda c: c["name"])

    return {
        "torchgenomics_version": __version__,
        "schema_version": 1,
        "commands": commands,
    }
