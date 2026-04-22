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

    def test_convert_wired(self):
        """convert subcommand is wired (fails on missing input, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main(["convert", "--input", "a.vcf", "--output", "b.bed", "--format", "bed"])

    def test_impute_wired(self):
        """impute subcommand is wired (fails on missing input, not NotImplementedError)."""
        with pytest.raises(FileNotFoundError):
            main(["impute", "--genotype", "a.bed", "--method", "mean", "--output", "out.bed"])

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
