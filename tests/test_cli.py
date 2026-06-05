"""CLI entry point tests."""

from __future__ import annotations

import pytest

from torchgenomics.cli import main


class TestCLIParsing:
    """Tests for CLI argument parsing."""

    def test_no_args_exits(self):
        """No arguments prints help and exits with error."""
        with pytest.raises(SystemExit):
            main([])

    def test_help_exits_zero(self):
        """--help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_glm_scan_parses(self):
        """glm-scan subcommand accepts required args (fails on missing file)."""
        with pytest.raises(FileNotFoundError):
            main(["glm-scan", "--genotype", "data.bed", "--phenotype", "pheno.txt"])

    def test_lmm_scan_parses(self):
        """lmm-scan subcommand accepts required args (fails on missing file)."""
        with pytest.raises(FileNotFoundError):
            main(["lmm-scan", "--genotype", "data.bed", "--phenotype", "pheno.txt"])

    def test_poly_scan_requires_ploidy(self):
        """poly-scan requires --ploidy argument."""
        with pytest.raises(SystemExit):
            main(["poly-scan", "--genotype", "data.bed", "--phenotype", "pheno.txt"])

    def test_poly_scan_parses(self):
        """poly-scan with --ploidy parses (fails on missing file, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main([
                "poly-scan", "--genotype", "data.bed",
                "--phenotype", "pheno.txt", "--ploidy", "4",
            ])

    def test_poly_scan_gene_action_all(self):
        """poly-scan accepts --gene-action all."""
        with pytest.raises(FileNotFoundError):
            main([
                "poly-scan", "--genotype", "data.bed",
                "--phenotype", "pheno.txt", "--ploidy", "4",
                "--gene-action", "all",
            ])

    def test_mvlmm_scan_parses(self):
        """mvlmm-scan parses with --traits (fails on missing file, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main([
                "mvlmm-scan", "--genotype", "data.bed",
                "--phenotype", "pheno.txt", "--traits", "Y1,Y2",
            ])

    def test_pipeline_parses(self):
        """pipeline subcommand parses (fails on missing file, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main([
                "pipeline", "--genotype", "data.bed",
                "--phenotype", "pheno.txt",
            ])

    def test_pipeline_with_model_and_ploidy(self):
        """pipeline accepts --model and --ploidy."""
        with pytest.raises(FileNotFoundError):
            main([
                "pipeline", "--genotype", "data.bed",
                "--phenotype", "pheno.txt",
                "--model", "lmm", "--ploidy", "4",
            ])

    def test_pipeline_farmcpu(self):
        """pipeline accepts --model farmcpu."""
        with pytest.raises(FileNotFoundError):
            main([
                "pipeline", "--genotype", "data.bed",
                "--phenotype", "pheno.txt",
                "--model", "farmcpu",
            ])

    def test_pipeline_gxe_parses(self):
        """pipeline accepts --model gxe and its environment argument."""
        with pytest.raises(FileNotFoundError):
            main([
                "pipeline", "--genotype", "data.bed",
                "--phenotype", "pheno.txt",
                "--model", "gxe", "--env", "env.tsv",
            ])

    def test_convert_wired(self):
        """convert subcommand is wired (fails on missing input, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main(["convert", "--input", "a.vcf", "--output", "b.bed", "--format", "bed"])

    def test_convert_vcf_format_parses(self):
        """convert accepts VCF as an output format."""
        with pytest.raises(FileNotFoundError):
            main(["convert", "--input", "a.bed", "--output", "out.vcf", "--format", "vcf"])

    def test_gxe_env_vector_aligns_by_sample_id(self, tmp_path):
        """GxE env files are aligned by sample ID rather than raw row order."""
        import torch

        from torchgenomics.cli import _load_env_vector

        env = tmp_path / "env.tsv"
        env.write_text("SAMPLE\tENV\ns2\t2.0\ns1\t1.0\n")

        values = _load_env_vector(
            str(env),
            ["s1", "s2"],
            dtype=torch.float64,
            device=torch.device("cpu"),
        )

        assert values.tolist() == [1.0, 2.0]

    def test_impute_wired(self):
        """impute subcommand is wired (fails on missing input, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main(["impute", "--genotype", "a.bed", "--method", "mean", "--output", "out.bed"])

    def test_impute_mean_runs_on_csv_fixture(self, tmp_path):
        """mean imputation loads real reader chunks and writes a dosage tensor."""
        import torch

        out = tmp_path / "imputed.pt"
        rc = main([
            "impute",
            "--genotype", "tests/fixtures/tiny_dosage.csv",
            "--method", "mean",
            "--output", str(out),
        ])

        assert rc == 0
        payload = torch.load(out, weights_only=False)
        assert set(payload) == {"dosage"}
        assert payload["dosage"].shape == (10, 20)
        assert torch.isfinite(payload["dosage"]).all()

    def test_ld_blocks_parses(self):
        """ld-blocks subcommand accepts required args (fails on missing file)."""
        with pytest.raises(FileNotFoundError):
            main(["ld-blocks", "--genotype", "data.bed", "--method", "gabriel"])


class TestDeviceFallback:
    """Test that device resolution works correctly."""

    def test_resolve_device_cpu_explicit(self):
        from torchgenomics.config import resolve_device
        dev = resolve_device("cpu")
        assert dev.type == "cpu"

    def test_resolve_device_auto(self):
        """Auto device should return cpu or cuda without error."""
        from torchgenomics.config import resolve_device
        dev = resolve_device(None)
        assert dev.type in ("cpu", "cuda")

    def test_resolve_device_invalid_raises(self):
        """Invalid device string raises RuntimeError."""
        from torchgenomics.config import resolve_device
        with pytest.raises(RuntimeError):
            resolve_device("nonexistent_device_xyz")


# ---------------------------------------------------------------------------
# NA1 Task 11: bayes-scan-rss subcommand
# ---------------------------------------------------------------------------

def test_bayes_scan_rss_help(capsys):
    """`torchgenomics bayes-scan-rss --help` prints usage with expected flags."""
    from torchgenomics.cli import main
    with pytest.raises(SystemExit) as exc_info:
        main(["bayes-scan-rss", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--sumstats" in captured.out
    assert "--ld-ref" in captured.out
    assert "--geno" in captured.out
    assert "--max-num-causal" in captured.out
    assert "--coverage" in captured.out
    assert "--purity" in captured.out
    assert "--output" in captured.out


def test_bayes_scan_rss_smoke_runs_end_to_end(tmp_path):
    """End-to-end smoke run on a tiny synthetic fixture.

    Builds sumstats + LD reference in tmp_path, invokes the CLI, asserts
    output file exists and has the expected columns.
    """
    import pandas as pd
    import torch

    from torchgenomics.cli import main
    from torchgenomics.postgwas._ld_ref_loader import save_ld_reference
    from torchgenomics.postgwas._ld_ref_metadata import LDReferenceMetadata

    p = 5
    sumstats_path = tmp_path / "sumstats.tsv"
    sumstats = pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(p)],
        "CHR": [22] * p,
        "BP": [1000 + i * 10 for i in range(p)],
        "A1": ["A"] * p,
        "A2": ["G"] * p,
        "BETA": [0.05, 0.03, 0.6, 0.02, 0.04],
        "SE": [0.05, 0.04, 0.05, 0.04, 0.05],
        "N": [1000] * p,
    })
    sumstats.to_csv(sumstats_path, sep="\t", index=False)

    ld_path = tmp_path / "ld.pt"
    R = torch.eye(p, dtype=torch.float64)
    snp_ids = sumstats["SNP"].tolist()
    meta = LDReferenceMetadata(
        cohort_id="test", n=1000, build="GRCh38", panel_provenance="synthetic"
    )
    save_ld_reference(ld_path, R, snp_ids, meta)

    out_path = tmp_path / "finemap.tsv"
    rc = main([
        "bayes-scan-rss",
        "--sumstats", str(sumstats_path),
        "--ld-ref", str(ld_path),
        "--max-num-causal", "2",
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == [
        "SNP", "CHR", "BP", "A1", "A2", "Z", "N",
        "PIP", "BETA_MEAN", "BETA_SD", "CREDIBLE_SET",
    ]
    assert len(df) == p
    # SNP rs2 has the strongest BETA/SE -> highest PIP
    pip_values = df["PIP"].values
    assert pip_values[2] == pip_values.max()
