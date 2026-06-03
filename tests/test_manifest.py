# tests/test_manifest.py
"""Tests for the rTorchGenomics manifest emitter."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest


TIER_2_NAMES = {
    "mvlmm_scan", "poly_scan", "mklmm_scan", "gxe_scan", "set_scan",
    "bayes_scan", "bayes_scan_rss", "met_scan", "farmcpu_scan", "blink_scan",
    "threshold_scan", "family_scan", "conditional_scan", "mtmet_scan",
    "ocf_scan", "knockoff_scan", "gu_scan", "lro_scan", "glmm_scan",
    "me_glmm_scan", "survival_scan", "rr_scan", "rr_met_scan", "twas_scan",
    "combine_gwas_twas", "ldsc", "ldsc_rg", "dosage_call", "phase_poly",
    "mediate", "mediate_scan", "pipeline",
}


class TestManifestEmitter:
    def test_emit_to_stdout(self):
        """`python -m torchgenomics._manifest` writes valid JSON to stdout."""
        result = subprocess.run(
            [sys.executable, "-m", "torchgenomics._manifest"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        data = json.loads(result.stdout)
        assert "torchgenomics_version" in data
        assert "commands" in data
        assert isinstance(data["commands"], list)

    def test_emits_all_tier2_commands(self):
        from torchgenomics._manifest import build_manifest

        manifest = build_manifest()
        names = {c["name"] for c in manifest["commands"]}
        missing = TIER_2_NAMES - names
        # Tier-2 set is at least the names above; manifest may include more
        # if new CLI subcommands are added later.
        assert not missing, f"missing tier-2 commands: {missing}"

    def test_command_schema_shape(self):
        from torchgenomics._manifest import build_manifest

        manifest = build_manifest()
        for cmd in manifest["commands"]:
            assert "name" in cmd
            assert "cli_subcommand" in cmd
            assert "description" in cmd
            assert "args" in cmd and isinstance(cmd["args"], list)
            for arg in cmd["args"]:
                assert {"name", "type", "required", "default", "choices", "nargs"}.issubset(arg)

    def test_descriptions_are_populated(self):
        """Each command must have a non-empty description for downstream codegen."""
        from torchgenomics._manifest import build_manifest

        manifest = build_manifest()
        empty = [c["name"] for c in manifest["commands"] if not c["description"]]
        assert not empty, f"commands with empty description: {empty}"

    def test_no_cli_drift(self):
        """All non-tier-1 CLI subcommands must be either tier-2 or explicitly excluded."""
        import argparse

        from torchgenomics._manifest import _TIER_2_CLI_SUBCOMMANDS
        from torchgenomics.cli import _build_parser

        parser = _build_parser()
        sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        all_cli = set(sub.choices.keys())
        tier1_cli = {
            "validate", "convert", "impute", "lmm-scan", "glm-scan",
            "ld-blocks", "clump", "meta", "pgs-fit", "pgs-score", "annotate",
        }
        unclassified = all_cli - tier1_cli - _TIER_2_CLI_SUBCOMMANDS
        assert not unclassified, (
            f"CLI subcommand(s) not classified as tier-1 or tier-2: {unclassified}. "
            f"Add to _TIER_2_CLI_SUBCOMMANDS or the test's tier1_cli set."
        )
