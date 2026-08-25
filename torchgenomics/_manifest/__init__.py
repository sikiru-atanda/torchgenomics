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
    "gwas", "recommend", "models", "mr", "coloc",
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
    seen_dests: set[str] = set()
    for action in subparser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        # Deduplicate by `dest` so paired `--foo` / `--no-foo` flags emit
        # one entry per logical parameter (the affirmative form, which
        # appears first in argparse construction).
        if action.dest in seen_dests:
            continue
        # argparse flag names: ["--genotype", "-g"]. We use the first
        # long-form flag stripped of leading dashes as the arg name.
        flag = next((s for s in action.option_strings if s.startswith("--")), None)
        if flag is None:
            # Positional or short-only — skip; tier-2 CLI is long-flag only.
            continue
        seen_dests.add(action.dest)
        name = flag.lstrip("-").replace("-", "_")
        args.append({
            "name": name,
            "type": _action_type_name(action),
            "required": bool(action.required),
            "default": action.default if action.default is not argparse.SUPPRESS else None,
            "help": action.help or "",
            "choices": list(action.choices) if action.choices is not None else None,
            "nargs": action.nargs if action.nargs is not None else None,
        })
    return args


def build_manifest() -> dict[str, Any]:
    """Build the tier-2 manifest by walking the CLI's argparse subparsers."""
    parser = _build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )

    # Subparsers are usually created via `subparsers.add_parser("name", help=...)`
    # without a `description=` kwarg, so `subparser.description` is None.
    # Fall back to the help text registered on the _SubParsersAction so the
    # R-side codegen has a non-empty function description.
    help_by_name = {
        ca.dest: ca.help for ca in subparsers_action._choices_actions
    }

    commands: list[dict[str, Any]] = []
    for cli_name, subparser in subparsers_action.choices.items():
        if cli_name not in _TIER_2_CLI_SUBCOMMANDS:
            continue
        description = (
            subparser.description or help_by_name.get(cli_name) or ""
        ).strip()
        commands.append({
            "name": cli_name.replace("-", "_"),
            "cli_subcommand": cli_name,
            "description": description,
            "args": _extract_args(subparser),
        })

    commands.sort(key=lambda c: c["name"])

    return {
        "torchgenomics_version": __version__,
        "schema_version": 1,
        "commands": commands,
    }
