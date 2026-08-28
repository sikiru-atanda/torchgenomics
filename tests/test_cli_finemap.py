import subprocess, sys


def test_cli_finemap_help():
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "finemap", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "--sumstats" in r.stdout and "--ld-ref" in r.stdout


def test_cli_finemap_end_to_end(tmp_path):
    import pandas as pd, torch
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata
    p = 5
    ss = tmp_path / "s.tsv"
    pd.DataFrame({"SNP": [f"rs{i}" for i in range(p)], "CHR": [22]*p,
                  "BP": [1000+i*10 for i in range(p)], "A1": ["A"]*p, "A2": ["G"]*p,
                  "BETA": [0.05,0.03,0.6,0.02,0.04], "SE": [0.05,0.04,0.05,0.04,0.05],
                  "N": [1000]*p}).to_csv(ss, sep="\t", index=False)
    ld = tmp_path / "ld.pt"
    save_ld_reference(ld, torch.eye(p, dtype=torch.float64), [f"rs{i}" for i in range(p)],
                      LDReferenceMetadata(cohort_id="t", n=1000, build="GRCh38",
                                          panel_provenance="syn"))
    out = tmp_path / "o.tsv"
    r = subprocess.run([sys.executable, "-m", "torchgenomics.cli", "finemap",
                        "--sumstats", str(ss), "--ld-ref", str(ld),
                        "--max-num-causal", "2", "--threads", "1", "--output", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists()
