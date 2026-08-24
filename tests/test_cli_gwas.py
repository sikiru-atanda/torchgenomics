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


def test_cli_gwas_bad_pcs_fails_at_parse_time():
    # --pcs is validated by an argparse `type=` function, so a bad value must
    # be rejected during parsing (exit code 2, "--pcs" named in the message)
    # rather than surfacing deep inside dispatch.
    out = subprocess.run(
        [sys.executable, "-m", "torchgenomics", "gwas", "--pcs", "foo",
         "--phenotype", "x", "--genotype", "y"],
        capture_output=True, text=True,
    )
    assert out.returncode == 2
    assert "--pcs" in out.stderr


def test_cli_gwas_pcs_int_parses_ok():
    from torchgenomics.cli import _build_parser
    args = _build_parser().parse_args(
        ["gwas", "--pcs", "5", "--phenotype", "x", "--genotype", "y"]
    )
    assert args.pcs == 5 and isinstance(args.pcs, int)


def test_cli_gwas_pcs_auto_parses_ok():
    from torchgenomics.cli import _build_parser
    args = _build_parser().parse_args(
        ["gwas", "--pcs", "auto", "--phenotype", "x", "--genotype", "y"]
    )
    assert args.pcs == "auto"

    args2 = _build_parser().parse_args(
        ["gwas", "--pcs", "AUTO", "--phenotype", "x", "--genotype", "y"]
    )
    assert args2.pcs == "auto"
