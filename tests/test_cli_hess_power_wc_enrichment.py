import subprocess, sys


def _help(sub):
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", sub, "--help"],
                       capture_output=True, text=True)
    return r


def test_cli_help_all_four():
    for sub in ("power", "winners-curse", "gene-set-enrichment", "hess"):
        r = _help(sub)
        assert r.returncode == 0, r.stderr
        assert "--gwas" in r.stdout


def test_cli_power_end_to_end(tmp_path):
    import pandas as pd
    df = pd.DataFrame({"chr": [1, 1], "pos": [1, 2], "snp": ["a", "b"], "a1": ["A", "A"],
                       "a2": ["G", "G"], "beta": [0.5, 0.01], "se": [0.05, 0.05],
                       "p": [1e-20, 0.8], "n": [10000, 10000], "af": [0.3, 0.3]})
    fp = tmp_path / "g.tsv"; df.to_csv(fp, sep="\t", index=False)
    out = tmp_path / "power.tsv"
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "power",
                        "--gwas", str(fp), "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
