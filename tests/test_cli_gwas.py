"""CLI tests for the friendly-API `gwas` / `recommend` / `models` subcommands (Task 8).

These three subcommands share the friendly-API core (`torchgenomics.api.gwas`
/ `recommend` / `models`) so the CLI, the Python API, and the MCP tools all
run through exactly one decision path. See `torchgenomics/api/gwas.py` for
the underlying implementation this CLI layer wraps.
"""
import subprocess
import sys


def test_cli_models_lists_registry():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "models"],
                          capture_output=True, text=True)
    assert out.returncode == 0 and "lmm" in out.stdout and "farmcpu" in out.stdout


def test_cli_gwas_help_has_models_flag():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                          capture_output=True, text=True)
    assert out.returncode == 0 and "--models" in out.stdout and "--phenotype" in out.stdout


def test_cli_gwas_help_has_model_options_flag():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                          capture_output=True, text=True)
    assert out.returncode == 0 and "--model-options" in out.stdout


def test_cli_gwas_help_has_env_and_regions():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--env" in out.stdout and "--regions" in out.stdout


def test_cli_gwas_help_has_traits():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "gwas", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--traits" in out.stdout
