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
                assert {"name", "type", "required", "default"}.issubset(arg)
