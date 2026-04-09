"""End-to-end tests for the ``torchgwas rr-scan`` CLI subcommand
(Phase 38, Step 12)."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgwas.cli import main


def _write_plink_files(G, ids, out_prefix):
    """Minimal PLINK BED writer (n samples, m variants, 0/1/2 dosage)."""
    n, m = G.shape
    with open(f"{out_prefix}.fam", "w") as f:
        for i in range(n):
            f.write(f"FAM{i}\t{ids[i]}\t0\t0\t0\t-9\n")
    with open(f"{out_prefix}.bim", "w") as f:
        for j in range(m):
            f.write(f"1\trs{j}\t0\t{j+1}\tA\tG\n")
    bytes_per_snp = (n + 3) // 4
    with open(f"{out_prefix}.bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(m):
            snp_bytes = bytearray(bytes_per_snp)
            for i in range(n):
                g = int(G[i, j].item())
                code = {0: 0b11, 1: 0b10, 2: 0b00}.get(g, 0b01)
                snp_bytes[i // 4] |= (code << ((i % 4) * 2))
            f.write(snp_bytes)


def _make_long_pheno(n=20, T=6, seed=0):
    """Build a tiny balanced longitudinal phenotype + 0/1/2 genotype matrix."""
    torch.manual_seed(seed)
    m = 8
    G = (torch.rand(n, m) > 0.5).long() + (torch.rand(n, m) > 0.5).long()
    times = torch.linspace(0.0, 100.0, T, dtype=torch.float64)

    # Simple phenotype: linear time trend + small genetic effect on SNP 3
    rows = []
    for i in range(n):
        snp_eff = float(G[i, 3].item()) * 0.5
        for t in times:
            y = 1.0 + 0.01 * float(t) + snp_eff + torch.randn(1).item() * 0.5
            rows.append({"IID": f"SAMP{i}", "TIME": float(t), "Y": y})
    pheno = pd.DataFrame(rows)
    ids = [f"SAMP{i}" for i in range(n)]
    return G, pheno, ids


class TestRRScanCLIParser:
    def test_rr_scan_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["rr-scan", "--help"])
        assert exc_info.value.code == 0

    def test_rr_scan_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            main([
                "rr-scan",
                "--genotype", "nonexistent.bed",
                "--phenotype", "nonexistent.tsv",
            ])


class TestRRScanCLIEndToEnd:
    def test_rr_scan_basic_run(self, tmp_path):
        G, pheno, ids = _make_long_pheno(n=20, T=6, seed=0)
        bed_prefix = tmp_path / "rr_test"
        _write_plink_files(G, ids, str(bed_prefix))
        pheno_path = tmp_path / "pheno_long.tsv"
        pheno.to_csv(pheno_path, sep="\t", index=False)

        out_prefix = tmp_path / "rr_out"
        rc = main([
            "rr-scan",
            "--genotype", str(bed_prefix) + ".bed",
            "--phenotype", str(pheno_path),
            "--id-col", "IID",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "2",
            "--output", str(out_prefix),
        ])
        assert rc == 0
        rr_tsv = out_prefix.with_suffix(".rr.tsv")
        rr_null = out_prefix.with_suffix(".rr_null.json")
        assert rr_tsv.exists()
        assert rr_null.exists()

        df = pd.read_csv(rr_tsv, sep="\t")
        for col in (
            "SNP", "P_JOINT", "P_INTERCEPT", "P_SLOPE", "P_TIME_VARYING",
        ):
            assert col in df.columns
        assert len(df) == 8
        assert (df["P_JOINT"] >= 0).all() and (df["P_JOINT"] <= 1).all()

        with open(rr_null) as f:
            meta = json.load(f)
        assert meta["b"] == 3
        assert meta["basis_kind"] == "legendre"
        assert "K_coef" in meta
        assert len(meta["K_coef"]) == 3

    def test_rr_scan_eval_times_writes_extra_tsv(self, tmp_path):
        G, pheno, ids = _make_long_pheno(n=20, T=6, seed=1)
        bed_prefix = tmp_path / "rr_test"
        _write_plink_files(G, ids, str(bed_prefix))
        pheno_path = tmp_path / "pheno_long.tsv"
        pheno.to_csv(pheno_path, sep="\t", index=False)

        out_prefix = tmp_path / "rr_out"
        rc = main([
            "rr-scan",
            "--genotype", str(bed_prefix) + ".bed",
            "--phenotype", str(pheno_path),
            "--id-col", "IID",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "2",
            "--eval-times", "10,40,80",
            "--output", str(out_prefix),
        ])
        assert rc == 0
        at_t_path = out_prefix.with_suffix(".rr_at_t.tsv")
        assert at_t_path.exists()
        df = pd.read_csv(at_t_path, sep="\t")
        # 8 SNPs × 3 times = 24 rows
        assert len(df) == 24
        assert set(df["TIME"].unique()) == {10.0, 40.0, 80.0}
