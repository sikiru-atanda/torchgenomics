"""Tests for :mod:`torchgenomics.api` tier-1 facade.

Coverage:

  - Registry: 13 tools registered with correct metadata.
  - Result classes: ``.to_dict()`` is JSON-safe, ``.summary()`` non-empty.
  - End-to-end smoke: :func:`validate`, :func:`lmm_scan`, :func:`glm_scan`,
    :func:`ld_blocks`, :func:`manhattan`, :func:`qq` against the tiny PLINK
    fixture (``tests/fixtures/tiny.{bed,bim,fam}`` + ``tiny_pheno.txt``).
  - Top-level re-exports: ``torchgenomics.lmm_scan is torchgenomics.api.lmm_scan``.

PGS / meta / clump / annotate are exercised by the existing CLI tests
(see test_cli.py + test_cli_matrix.py); the facade functions for those
delegate to the same CLI handlers, so they're covered transitively.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import torchgenomics as tg
from torchgenomics.api import (
    AnnotateRun,
    ClumpRun,
    ConvertRun,
    ImputeRun,
    LDBlocksRun,
    MetaRun,
    PgsFitRun,
    PgsScoreRun,
    PlotResult,
    ScanRun,
    ValidateRun,
    registered_tools,
    tools_by_category,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
TINY_BED = FIXTURE_DIR / "tiny.bed"
TINY_PHENO = FIXTURE_DIR / "tiny_pheno.txt"


# --- Registry --------------------------------------------------------------


class TestRegistry:
    """The decorator records all tier-1 functions in the MCP registry."""

    EXPECTED_TOOLS = {
        "tg_validate", "tg_convert", "tg_impute",
        "tg_lmm_scan", "tg_glm_scan",
        "tg_ld_blocks", "tg_clump", "tg_meta",
        "tg_pgs_fit", "tg_pgs_score",
        "tg_annotate", "tg_manhattan", "tg_qq",
    }

    def test_all_expected_tools_registered(self):
        registry = registered_tools()
        missing = self.EXPECTED_TOOLS - set(registry)
        assert not missing, f"missing tier-1 tools: {missing}"

    def test_no_duplicate_names(self):
        registry = registered_tools()
        names = list(registry.keys())
        assert len(names) == len(set(names)), "duplicate tool names registered"

    def test_metadata_completeness(self):
        for name, entry in registered_tools().items():
            assert entry.title, f"{name} missing title"
            assert entry.description, f"{name} missing description"
            assert entry.category, f"{name} missing category"
            assert callable(entry.func), f"{name} func not callable"

    def test_categories(self):
        cats = {entry.category for entry in registered_tools().values()}
        assert cats.issuperset({"data", "scan", "ld", "postgwas", "pgs", "annotate", "plot"})

    def test_long_running_flag(self):
        """Heavy ops should declare long_running=True."""
        registry = registered_tools()
        for name in ("tg_lmm_scan", "tg_glm_scan", "tg_pgs_fit", "tg_impute"):
            assert registry[name].long_running is True, f"{name} should be long_running"

    def test_tools_by_category(self):
        scans = tools_by_category("scan")
        assert {t.name for t in scans} == {"tg_lmm_scan", "tg_glm_scan"}


# --- Top-level re-exports --------------------------------------------------


class TestTopLevelExports:
    """``torchgenomics.X`` resolves to ``torchgenomics.api.X``."""

    @pytest.mark.parametrize("name", [
        "validate", "convert", "impute",
        "lmm_scan", "glm_scan",
        "ld_blocks", "clump", "meta",
        "pgs_fit", "pgs_score",
        "annotate_hits", "manhattan", "qq",
    ])
    def test_function_reexported(self, name: str):
        from torchgenomics import api as api_mod

        assert getattr(tg, name) is getattr(api_mod, name)


# --- Result-class invariants -----------------------------------------------


class TestResultClasses:
    """Every result class has summary/to_dict/to_json that work on a default instance."""

    @pytest.mark.parametrize("cls", [
        ValidateRun, ConvertRun, ImputeRun, ScanRun, LDBlocksRun,
        ClumpRun, MetaRun, PgsFitRun, PgsScoreRun, AnnotateRun, PlotResult,
    ])
    def test_default_instance_serializes(self, cls):
        r = cls()
        s = r.summary()
        assert isinstance(s, str) and len(s) > 0
        d = r.to_dict()
        assert isinstance(d, dict)
        # Must round-trip through json.dumps without TypeError
        text = json.dumps(d, default=str)
        assert isinstance(text, str)

    def test_scan_run_top_hits_serializes(self):
        df = pd.DataFrame({
            "CHR": ["1", "1", "2"],
            "POS": [100, 200, 300],
            "P": [1e-9, 1e-5, 0.5],
        })
        r = ScanRun(top_hits=df, n_variants=3, n_significant=1)
        d = r.to_dict()
        # top_hits should serialize as list[dict] (orient="records")
        assert isinstance(d["top_hits"], list)
        assert len(d["top_hits"]) == 3
        assert d["top_hits"][0]["CHR"] == "1"

    def test_plot_result_drops_figure_from_dict(self):
        """The matplotlib Figure isn't JSON-serializable; to_dict() must skip it."""
        sentinel = object()  # stand-in for a Figure
        r = PlotResult(figure=sentinel, png_base64="abc")
        d = r.to_dict()
        assert "figure" not in d
        assert d["png_base64"] == "abc"


# --- End-to-end smoke tests against tiny fixtures --------------------------


@pytest.fixture
def tiny_paths(tmp_path):
    """Tiny PLINK + phenotype paths."""
    return {
        "genotype": str(TINY_BED),
        "phenotype": str(TINY_PHENO),
        "output_dir": tmp_path,
    }


class TestValidateE2E:
    def test_validate_tiny_fixture(self, tiny_paths):
        r = tg.validate(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
            output=tiny_paths["output_dir"] / "report.json",
        )
        assert isinstance(r, ValidateRun)
        assert r.format_name == "bed"
        assert r.n_samples_genotype == 10
        assert r.n_variants_total == 20
        assert r.runtime_s >= 0.0
        assert r.output_files.get("json") == tiny_paths["output_dir"] / "report.json"
        assert (tiny_paths["output_dir"] / "report.json").exists()
        # ok is True iff errors are empty
        assert r.ok == (len(r.errors) == 0)

    def test_validate_no_output_file(self, tiny_paths):
        r = tg.validate(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
        )
        assert isinstance(r, ValidateRun)
        assert "json" not in r.output_files


class TestScanE2E:
    """End-to-end LMM and GLM smoke tests on tiny fixtures."""

    def test_lmm_scan_tiny(self, tiny_paths):
        r = tg.lmm_scan(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
            output=tiny_paths["output_dir"] / "lmm",
            chunk_size=10,
            maf_min=0.0,  # don't filter the tiny fixture
            correction="bh",
        )
        assert isinstance(r, ScanRun)
        assert r.model == "SingleTraitLMM"
        assert r.test == "wald"
        assert r.n_variants > 0
        # Output files exist
        assert "tsv" in r.output_files or "parquet" in r.output_files
        # Top hits is a DataFrame
        assert isinstance(r.top_hits, pd.DataFrame)
        # Summary is non-empty
        assert len(r.summary()) > 0

    def test_lmm_scan_to_dict_json_safe(self, tiny_paths):
        r = tg.lmm_scan(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
            output=tiny_paths["output_dir"] / "lmm",
            chunk_size=10,
            maf_min=0.0,
            correction="bh",
            top_k=5,
        )
        d = r.to_dict()
        # Round-trip through JSON
        text = json.dumps(d, default=str)
        assert isinstance(text, str)
        # top_hits is list-of-dicts
        assert isinstance(d["top_hits"], list)
        assert len(d["top_hits"]) <= 5

    def test_glm_scan_tiny_gaussian(self, tiny_paths):
        r = tg.glm_scan(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
            output=tiny_paths["output_dir"] / "glm",
            chunk_size=10,
            maf_min=0.0,
            correction="bh",
            family="gaussian",
        )
        assert isinstance(r, ScanRun)
        assert r.model.startswith("GLM-")
        assert r.n_variants > 0


class TestPlotE2E:
    def test_manhattan_from_dataframe(self, tmp_path):
        df = pd.DataFrame({
            "CHR": ["1"] * 50 + ["2"] * 50,
            "POS": list(range(100)),
            "P": [1.0 / (i + 1) for i in range(100)],
        })
        r = tg.manhattan(
            df,
            output=tmp_path / "manhattan.png",
            return_base64=False,
        )
        assert isinstance(r, PlotResult)
        assert r.png_path == tmp_path / "manhattan.png"
        assert (tmp_path / "manhattan.png").exists()
        assert r.figure is not None

    def test_qq_returns_base64_when_requested(self, tmp_path):
        df = pd.DataFrame({"P": [0.5, 0.1, 0.01, 1e-3, 1e-5, 1e-8]})
        r = tg.qq(df, return_base64=True)
        assert isinstance(r, PlotResult)
        assert isinstance(r.png_base64, str) and len(r.png_base64) > 100

    def test_plot_to_dict_drops_figure(self, tmp_path):
        df = pd.DataFrame({
            "CHR": ["1"] * 10,
            "POS": list(range(10)),
            "P": [0.5] * 10,
        })
        r = tg.manhattan(df, return_base64=False)
        d = r.to_dict()
        assert "figure" not in d


class TestLDBlocksE2E:
    def test_ld_blocks_tiny(self, tiny_paths):
        r = tg.ld_blocks(
            genotype=tiny_paths["genotype"],
            output=tiny_paths["output_dir"] / "blocks",
            method="r2",
            r2_threshold=0.5,
        )
        assert isinstance(r, LDBlocksRun)
        assert r.method == "r2"
        # Output files exist (bed and/or det)
        assert any(p.exists() for p in r.output_files.values()) or r.n_blocks == 0


# --- Smart defaults --------------------------------------------------------


class TestSmartDefaults:
    """Novice usage: omitting `output` auto-creates a unique run directory."""

    def test_validate_auto_output_dir(self, tiny_paths):
        # No `output=` → no JSON file produced, but call still succeeds
        r = tg.validate(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
        )
        assert isinstance(r, ValidateRun)
        assert "json" not in r.output_files

    def test_lmm_scan_auto_output_dir(self, tiny_paths, monkeypatch):
        # Force CWD to a tmp dir so auto-named output doesn't pollute the repo
        monkeypatch.chdir(tiny_paths["output_dir"])
        r = tg.lmm_scan(
            genotype=tiny_paths["genotype"],
            phenotype=tiny_paths["phenotype"],
            chunk_size=10,
            maf_min=0.0,
        )
        assert isinstance(r, ScanRun)
        # Auto-created an output directory
        assert any(p.exists() for p in r.output_files.values())
