"""CLI entry point tests."""

from __future__ import annotations

import pytest

from torchgwas.cli import main


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

        from torchgwas.cli import _load_env_vector

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
        from torchgwas.config import resolve_device
        dev = resolve_device("cpu")
        assert dev.type == "cpu"

    def test_resolve_device_auto(self):
        """Auto device should return cpu or cuda without error."""
        from torchgwas.config import resolve_device
        dev = resolve_device(None)
        assert dev.type in ("cpu", "cuda")

    def test_resolve_device_invalid_raises(self):
        """Invalid device string raises RuntimeError."""
        from torchgwas.config import resolve_device
        with pytest.raises(RuntimeError):
            resolve_device("nonexistent_device_xyz")
