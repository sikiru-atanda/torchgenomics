"""End-to-end tests for the ``torchgwas rr-met-scan`` CLI subcommand
(Phase 39, Step 6)."""

import json

import pandas as pd
import pytest
import torch

from torchgwas.cli import main


def _write_plink_files(G, ids, out_prefix):
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


def _make_met_long_pheno(n=20, T=6, E=2, seed=0):
    """Balanced MET longitudinal phenotype + 0/1/2 genotype matrix."""
    torch.manual_seed(seed)
    m = 8
    G = (torch.rand(n, m) > 0.5).long() + (torch.rand(n, m) > 0.5).long()
    rows = []
    for i in range(n):
        snp_eff = float(G[i, 3].item()) * 0.4
        for e in range(E):
            times = torch.linspace(0.0 + 0.5 * e, 100.0 + 0.5 * e, T, dtype=torch.float64)
            for t in times:
                y = (
                    1.0
                    + 0.2 * e
                    + 0.01 * float(t)
                    + snp_eff
                    + torch.randn(1).item() * 0.5
                )
                rows.append({
                    "IID": f"SAMP{i}",
                    "ENV": f"E{e}",
                    "TIME": float(t),
                    "Y": y,
                })
    pheno = pd.DataFrame(rows)
    ids = [f"SAMP{i}" for i in range(n)]
    return G, pheno, ids


class TestRRMetScanCLIParser:
    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["rr-met-scan", "--help"])
        assert exc_info.value.code == 0


class TestRRMetScanCLIEndToEnd:
    def test_basic_run(self, tmp_path):
        G, pheno, ids = _make_met_long_pheno(n=20, T=6, E=2, seed=0)
        bed_prefix = tmp_path / "rr_met_test"
        _write_plink_files(G, ids, str(bed_prefix))
        pheno_path = tmp_path / "pheno_met_long.tsv"
        pheno.to_csv(pheno_path, sep="\t", index=False)

        out_prefix = tmp_path / "rr_met_out"
        rc = main([
            "rr-met-scan",
            "--genotype", str(bed_prefix) + ".bed",
            "--phenotype", str(pheno_path),
            "--id-col", "IID",
            "--env-col", "ENV",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "2",
            "--vg-structure", "separable",
            "--output", str(out_prefix),
        ])
        assert rc == 0
        rr_tsv = out_prefix.with_suffix(".rr_met.tsv")
        rr_null = out_prefix.with_suffix(".rr_met_null.json")
        assert rr_tsv.exists()
        assert rr_null.exists()

        df = pd.read_csv(rr_tsv, sep="\t")
        assert len(df) == 8
        for col in ("SNP", "P_JOINT", "P_GXE_JOINT", "P_MEAN_CURVE"):
            assert col in df.columns
        assert (df["P_JOINT"] >= 0).all() and (df["P_JOINT"] <= 1).all()

        with open(rr_null) as f:
            meta = json.load(f)
        assert meta["b"] == 3
        assert meta["E"] == 2
        assert meta["basis_kind"] == "legendre"
        assert meta["env_labels"] == ["E0", "E1"]
        assert len(meta["K_coef"]) == 3
        assert len(meta["Vg_env"]) == 2

    def test_eval_times_writes_per_time_env_tsv(self, tmp_path):
        G, pheno, ids = _make_met_long_pheno(n=20, T=6, E=2, seed=1)
        bed_prefix = tmp_path / "rr_met_test"
        _write_plink_files(G, ids, str(bed_prefix))
        pheno_path = tmp_path / "pheno_met_long.tsv"
        pheno.to_csv(pheno_path, sep="\t", index=False)

        out_prefix = tmp_path / "rr_met_out"
        rc = main([
            "rr-met-scan",
            "--genotype", str(bed_prefix) + ".bed",
            "--phenotype", str(pheno_path),
            "--id-col", "IID",
            "--env-col", "ENV",
            "--time-col", "TIME",
            "--pheno-col", "Y",
            "--basis", "legendre",
            "--order", "2",
            "--eval-times", "10,40,80",
            "--output", str(out_prefix),
        ])
        assert rc == 0
        at_t_path = out_prefix.with_suffix(".rr_met_at_t.tsv")
        assert at_t_path.exists()
        df = pd.read_csv(at_t_path, sep="\t")
        # 8 SNPs × 3 times × 2 envs = 48 rows
        assert len(df) == 48
        assert set(df["TIME"].unique()) == {10.0, 40.0, 80.0}
        assert set(df["ENV"].unique()) == {"E0", "E1"}
