import subprocess
import sys


def test_cli_smr_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "smr", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--gwas" in out.stdout and "--eqtl" in out.stdout and "--gene-map" in out.stdout


def test_cli_mr_mega_help():
    out = subprocess.run([sys.executable, "-m", "torchgenomics", "mr-mega", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--sumstats" in out.stdout and "--n-axes" in out.stdout
