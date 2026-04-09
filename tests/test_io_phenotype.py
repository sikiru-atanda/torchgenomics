"""Phase 1: Phenotype loader and sample alignment tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchgwas.io.phenotype import load_phenotype, write_alignment_manifest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
GENO_IDS = [f"IND{i:03d}" for i in range(10)]


class TestPhenotypeLoader:
    """Tests for torchgwas.io.phenotype.load_phenotype."""

    def test_load_phenotype_basic(self):
        """Loads phenotype file and returns PhenotypeData."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
        )
        assert result.Y.dtype == torch.float64
        assert result.X0.dtype == torch.float64
        assert len(result.sample_ids) > 0
        assert len(result.trait_names) == 2  # Y1, Y2

    def test_three_way_alignment(self):
        """Aligns genotype, phenotype, and covariate samples by intersection."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
            covariate_path=FIXTURE_DIR / "tiny_covar.txt",
        )
        # All 10 genotype samples should be in intersection
        assert len(result.sample_ids) == 10
        assert result.Y.shape[0] == 10
        assert result.X0.shape[0] == 10

    def test_alignment_manifest_written(self, tmp_path):
        """AlignmentManifest is produced with correct counts."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
        )
        manifest = result.manifest
        assert len(manifest.iid) > 0
        assert sum(manifest.included) == len(result.sample_ids)

        # Write it to verify format
        out = tmp_path / "manifest.tsv"
        write_alignment_manifest(manifest, out)
        assert out.is_file()
        lines = out.read_text().splitlines()
        assert lines[0].startswith("IID")  # header

    def test_empty_intersection_raises(self):
        """Raises error when no sample IDs overlap."""
        with pytest.raises(ValueError, match="Empty intersection"):
            load_phenotype(
                FIXTURE_DIR / "tiny_pheno.txt",
                ["NOMATCH_001", "NOMATCH_002"],
            )

    def test_majority_drop_raises(self):
        """Raises error when alignment would drop >50% of samples."""
        # Create a scenario with 20 genotype IDs, only 5 overlap
        many_ids = [f"EXTRA{i:03d}" for i in range(15)] + GENO_IDS[:5]
        with pytest.raises(ValueError, match="drop"):
            load_phenotype(
                FIXTURE_DIR / "tiny_pheno.txt",
                many_ids,
            )

    def test_lexicographic_sort_order(self):
        """Aligned samples are in deterministic lexicographic order."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
        )
        assert result.sample_ids == sorted(result.sample_ids)

    def test_intercept_column(self):
        """Covariate matrix includes intercept as first column."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
        )
        # First column should be all 1s (intercept)
        assert torch.all(result.X0[:, 0] == 1.0)
        assert result.covariate_names[0] == "intercept"

    def test_with_covariates(self):
        """Covariate file is loaded and appended to X0."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
            covariate_path=FIXTURE_DIR / "tiny_covar.txt",
        )
        # intercept + SEX + AGE = 3 columns
        assert result.X0.shape[1] >= 3

    def test_trait_column_selection(self):
        """Specific trait columns can be selected."""
        result = load_phenotype(
            FIXTURE_DIR / "tiny_pheno.txt",
            GENO_IDS,
            trait_columns=["Y1"],
        )
        assert result.Y.shape[1] == 1
        assert result.trait_names == ["Y1"]

    def test_missing_trait_column_raises(self):
        """Raises error for nonexistent trait column."""
        with pytest.raises(ValueError, match="not found"):
            load_phenotype(
                FIXTURE_DIR / "tiny_pheno.txt",
                GENO_IDS,
                trait_columns=["NONEXISTENT"],
            )
