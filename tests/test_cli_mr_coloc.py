import subprocess
import sys


def test_cli_mr_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "mr", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--exposure" in out.stdout and "--method" in out.stdout


def test_cli_coloc_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "coloc", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--sumstats" in out.stdout and "--method" in out.stdout
