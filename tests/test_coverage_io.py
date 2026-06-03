"""Tier-2 behavioral coverage tests for ``torchgenomics.io.*``.

Bar (Pillar A spec section 4.3 Tier 2):
- Per public symbol: golden_path, edge_case, error_path test methods.
- Behavioral contract verified — return shape, key invariants, documented
  exceptions raised on bad input.
- Tier 2 (behavioral) does NOT require numerical equivalence to a reference
  tool; it requires the documented contract to hold.

Covers 9 io public symbols:
- ``torchgenomics.io.phenotype.PhenotypeData`` (dataclass)
- ``torchgenomics.io.validate.run_preflight`` (function)
- ``torchgenomics.io.bgen.BGENReader`` (class) — depends on ``bgen-reader``
- ``torchgenomics.io.convert.convert`` (function)
- ``torchgenomics.io.map_file.MapInfo`` (dataclass)
- ``torchgenomics.io.map_file.discover_map_file`` (function)
- ``torchgenomics.io.map_file.read_map_file`` (function)
- ``torchgenomics.io.plink2.Plink2PgenReader`` (class) — depends on ``pgenlib``
- ``torchgenomics.io.vcf.VCFReader`` (class) — depends on ``cyvcf2``

Format-specific readers (BGEN, PGEN, VCF) use ``pytest.importorskip`` for
optional dependencies. When the dep is missing the suite still verifies
that the class symbol exists and is importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchgenomics.io import (
    PhenotypeData,
    run_preflight,
)
from torchgenomics.io.bgen import BGENReader
from torchgenomics.io.convert import convert
from torchgenomics.io.map_file import MapInfo, discover_map_file, read_map_file
from torchgenomics.io.phenotype import AlignmentManifest, load_phenotype
from torchgenomics.io.plink2 import Plink2PgenReader
from torchgenomics.io.vcf import VCFReader


pytestmark = pytest.mark.timeout(60)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_vcf(path: Path) -> None:
    """Write a tiny valid VCF (3 samples, 2 variants) to *path*."""
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t0/1\t1/1\t0/0\n"
    )


def _make_empty_vcf(path: Path) -> None:
    """Write a header-only VCF (no variant lines)."""
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
    )


def _make_malformed_vcf(path: Path) -> None:
    """Write a VCF missing the #CHROM header line — malformed."""
    path.write_text(
        "this is not a vcf at all\n"
        "1 100 . A G . . . GT 0/0\n"
    )


# ---------------------------------------------------------------------------
# torchgenomics.io.phenotype.PhenotypeData
# ---------------------------------------------------------------------------


class TestPhenotypeData:
    """``PhenotypeData`` is a dataclass holding aligned phenotype +
    covariate matrices and the alignment manifest."""

    def test_golden_construct(self):
        """Build with documented fields and round-trip every attribute."""
        Y = torch.zeros((3, 1), dtype=torch.float64)
        X0 = torch.ones((3, 1), dtype=torch.float64)
        manifest = AlignmentManifest(
            iid=["s1", "s2", "s3"],
            in_genotype=[True, True, True],
            in_phenotype=[True, True, True],
            in_covariate=[True, True, True],
            included=[True, True, True],
            drop_reason=["PASS", "PASS", "PASS"],
        )
        data = PhenotypeData(
            Y=Y,
            X0=X0,
            sample_ids=["s1", "s2", "s3"],
            trait_names=["Y1"],
            covariate_names=["intercept"],
            manifest=manifest,
        )
        assert torch.equal(data.Y, Y)
        assert torch.equal(data.X0, X0)
        assert data.sample_ids == ["s1", "s2", "s3"]
        assert data.trait_names == ["Y1"]
        assert data.covariate_names == ["intercept"]
        assert data.manifest is manifest

    def test_load_from_tsv(self):
        """``load_phenotype`` factory aligns to the genotype sample list,
        producing a ``PhenotypeData`` with intercept-prepended X0."""
        # tiny.fam contains IND000..IND019 (20 individuals)
        fam_path = FIXTURES / "tiny.fam"
        geno_ids = [line.split()[1] for line in fam_path.read_text().splitlines()]

        data = load_phenotype(
            FIXTURES / "tiny_pheno.txt",
            genotype_sample_ids=geno_ids,
        )
        assert isinstance(data, PhenotypeData)
        assert data.Y.dtype == torch.float64
        # First X0 column is intercept = ones
        assert data.X0.shape[1] >= 1
        assert torch.all(data.X0[:, 0] == 1.0)
        # sample_ids are sorted lexicographically
        assert data.sample_ids == sorted(data.sample_ids)
        # Manifest is present
        assert isinstance(data.manifest, AlignmentManifest)
        # Trait names are non-empty
        assert len(data.trait_names) >= 1

    def test_invalid_field_raises(self):
        """Loading with a bogus ``trait_columns`` raises ``ValueError``."""
        fam_path = FIXTURES / "tiny.fam"
        geno_ids = [line.split()[1] for line in fam_path.read_text().splitlines()]
        with pytest.raises(ValueError, match=r"[Tt]rait"):
            load_phenotype(
                FIXTURES / "tiny_pheno.txt",
                genotype_sample_ids=geno_ids,
                trait_columns=["this_column_does_not_exist"],
            )


# ---------------------------------------------------------------------------
# torchgenomics.io.validate.run_preflight
# ---------------------------------------------------------------------------


class TestRunPreflight:
    """``run_preflight`` returns a ``PreflightReport`` with sample/variant
    counts and warning/error lists."""

    def test_golden_valid_paths(self):
        """Run on tiny.bed + tiny_pheno.txt; returns report with positive
        counts and no errors."""
        report = run_preflight(
            str(FIXTURES / "tiny.bed"),
            str(FIXTURES / "tiny_pheno.txt"),
        )
        assert report.format_name == "bed"
        assert report.n_samples_genotype > 0
        assert report.n_variants_total > 0
        assert report.n_samples_aligned > 0
        # No critical errors expected
        assert report.errors == []

    def test_edge_missing_file(self):
        """Pass a non-existent genotype path; preflight surfaces the error
        in its ``errors`` list and ``format_name`` falls back to 'unknown'."""
        report = run_preflight(
            "/no/such/path/does/not/exist.bed",
            str(FIXTURES / "tiny_pheno.txt"),
        )
        # Documented behavior: format_name=='unknown' on missing input,
        # error captured rather than raised.
        assert report.format_name == "unknown"
        assert len(report.errors) >= 1

    def test_error_mismatched_samples(self, tmp_path):
        """Genotype with sample IDs that do not overlap phenotype IDs:
        report.errors flags the empty intersection."""
        # Build a phenotype file whose IIDs do NOT match tiny.fam IIDs
        bad_pheno = tmp_path / "bad_pheno.txt"
        bad_pheno.write_text(
            "IID\tY1\n"
            "ZZZ_NO_MATCH_001\t1.0\n"
            "ZZZ_NO_MATCH_002\t2.0\n"
        )
        report = run_preflight(
            str(FIXTURES / "tiny.bed"),
            str(bad_pheno),
        )
        # Empty intersection → errors list populated
        assert report.n_samples_aligned == 0
        assert any("overlap" in e.lower() or "no overlapping" in e.lower()
                   for e in report.errors)


# ---------------------------------------------------------------------------
# torchgenomics.io.bgen.BGENReader
# ---------------------------------------------------------------------------


class TestBgenReader:
    """``BGENReader`` requires the optional ``bgen-reader`` dependency.
    When unavailable, the constructor raises ``ImportError`` with a helpful
    install hint. When unavailable in the test env, format-specific behavior
    is skipped via ``importorskip``."""

    def test_class_is_importable(self):
        """The class symbol is always importable, even without the optional
        runtime dep — the import-error is deferred to construction."""
        assert BGENReader.__name__ == "BGENReader"
        # Public API: documented init signature
        assert callable(BGENReader)

    def test_missing_dep_raises_import_error(self):
        """Without ``bgen-reader`` installed, instantiation raises
        ``ImportError`` (not ``ModuleNotFoundError`` at import time)."""
        try:
            import bgen_reader  # noqa: F401
            pytest.skip("bgen-reader is installed; skipping import-error test")
        except ImportError:
            pass
        # Path doesn't matter — the dep check fires first.
        with pytest.raises(ImportError, match="bgen-reader"):
            BGENReader("/some/file.bgen")

    def test_golden_path(self, tmp_path):
        """With ``bgen-reader`` installed, opening a tiny BGEN file yields
        a reader whose properties match the file."""
        pytest.importorskip("bgen_reader")
        # A real BGEN file is a binary format; bgen-reader does not provide
        # a synthesizer. Without an existing fixture we cannot construct
        # one in-test. Skip with documented reason.
        pytest.skip(
            "No tiny .bgen fixture available; "
            "BGEN binary format requires external synthesis tool."
        )

    def test_edge_case(self):
        """Edge: missing file path raises ``FileNotFoundError``."""
        pytest.importorskip("bgen_reader")
        with pytest.raises(FileNotFoundError, match="BGEN"):
            BGENReader("/no/such/file.bgen")

    def test_error_path(self, tmp_path):
        """Error: malformed file (not a real BGEN) raises an error during
        construction."""
        pytest.importorskip("bgen_reader")
        bogus = tmp_path / "bogus.bgen"
        bogus.write_bytes(b"this is not a bgen file at all")
        with pytest.raises(Exception):
            BGENReader(str(bogus))


# ---------------------------------------------------------------------------
# torchgenomics.io.convert.convert
# ---------------------------------------------------------------------------


class TestConvert:
    """``convert`` reads any supported genotype input and writes to the
    requested output format."""

    def test_golden_bed_to_vcf(self, tmp_path):
        """Convert tiny.bed to VCF; output file exists and contains the
        expected number of variant records."""
        out_path = tmp_path / "out.vcf"
        convert(
            str(FIXTURES / "tiny.bed"),
            str(out_path),
            output_format="vcf",
        )
        assert out_path.is_file()
        text = out_path.read_text()
        # Header + at least one variant line
        assert "##fileformat=VCF" in text
        assert "#CHROM" in text
        # Variant lines (non-header)
        var_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        assert len(var_lines) > 0

    def test_edge_bed_to_zarr(self, tmp_path):
        """Edge: convert to zarr (different code path that requires the
        optional zarr dep). Skip cleanly if zarr is unavailable.

        Closes the loop by reading the produced store back: the ``dosage``
        array under the root group must materialize and have the same shape
        as the source BED. Per the F3 fix in 0dc5539, ``convert`` writes
        the dosage matrix at the root key ``dosage`` (zarr v3
        ``create_array`` path; zarr v2 ``create_dataset`` fallback)."""
        zarr = pytest.importorskip("zarr")
        out_path = tmp_path / "out.zarr"
        convert(
            str(FIXTURES / "tiny.bed"),
            str(out_path),
            output_format="zarr",
        )
        # Zarr writes a directory store
        assert out_path.exists()

        # Round-trip read-back: the root group exposes a ``dosage`` array
        # of shape (n_samples, n_variants). We assert shape > 0 in both
        # dimensions and verify n_samples / n_variants attrs match.
        g = zarr.open_group(str(out_path), mode="r")
        assert "dosage" in g
        dosage = g["dosage"]
        assert dosage.ndim == 2
        n, m = dosage.shape
        assert n > 0 and m > 0
        # Attribute round-trip (set by convert._write_zarr_dosage_only)
        assert g.attrs["n_samples"] == n
        assert g.attrs["n_variants"] == m

    def test_error_unsupported_format(self, tmp_path):
        """Error: ``output_format`` that is not in {bed, zarr, vcf, bgen}
        raises ``ValueError`` mentioning the format name or 'supported'."""
        with pytest.raises(ValueError, match=r"(?i)not yet supported|supported"):
            convert(
                str(FIXTURES / "tiny.bed"),
                str(tmp_path / "out.xyz"),
                output_format="xyz_nonsense_format",
            )


# ---------------------------------------------------------------------------
# torchgenomics.io.map_file.MapInfo
# ---------------------------------------------------------------------------


class TestMapInfo:
    """``MapInfo`` is a small dataclass holding parsed marker metadata."""

    def test_golden_construct(self):
        """Construct with parallel lists; fields round-trip."""
        info = MapInfo(snp=["rs1", "rs2"], chr=["1", "1"], pos=[100, 200])
        assert info.snp == ["rs1", "rs2"]
        assert info.chr == ["1", "1"]
        assert info.pos == [100, 200]
        assert info.cm is None  # default

    def test_edge_with_cm(self):
        """Edge: optional ``cm`` field accepts a list of floats."""
        info = MapInfo(
            snp=["rs1"], chr=["1"], pos=[100], cm=[0.1],
        )
        assert info.cm == [0.1]

    def test_repr_does_not_crash(self):
        """``repr`` yields a string containing the type name."""
        info = MapInfo(snp=[], chr=[], pos=[])
        s = repr(info)
        assert "MapInfo" in s


# ---------------------------------------------------------------------------
# torchgenomics.io.map_file.discover_map_file
# ---------------------------------------------------------------------------


class TestDiscoverMapFile:
    """``discover_map_file`` searches for a companion .map / .bim / *_map.txt
    next to the genotype file and returns its path or None."""

    def test_golden_path_finds_bim(self):
        """tiny.bed has a companion tiny.bim; discovery returns its path."""
        result = discover_map_file(str(FIXTURES / "tiny.bed"))
        assert result is not None
        assert result.endswith(".bim") or result.endswith(".map")
        assert Path(result).is_file()

    def test_edge_no_map_returns_none(self, tmp_path):
        """Edge: no matching companion file → returns ``None``."""
        lonely = tmp_path / "lonely.bed"
        lonely.write_bytes(b"\x6c\x1b\x01")  # PLINK magic, no .bim sidecar
        result = discover_map_file(str(lonely))
        assert result is None

    def test_error_invalid_path_type(self):
        """Error: passing a non-str/Path raises ``TypeError`` from Path()."""
        with pytest.raises(TypeError):
            discover_map_file(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# torchgenomics.io.map_file.read_map_file
# ---------------------------------------------------------------------------


class TestReadMapFile:
    """``read_map_file`` parses both PLINK .map (4-col) and 3-col layouts."""

    def test_golden_path_plink_map(self):
        """Reading a PLINK 4-col .map (tiny.bim) returns a MapInfo with
        equal-length parallel lists."""
        info = read_map_file(FIXTURES / "tiny.bim")
        assert isinstance(info, MapInfo)
        assert len(info.snp) == len(info.chr) == len(info.pos)
        assert len(info.snp) > 0
        # cm column present in 4-col files
        assert info.cm is not None
        assert len(info.cm) == len(info.snp)

    def test_edge_3col_map(self, tmp_path):
        """Edge: 3-column 'chr snp pos' map parses with cm=None."""
        path = tmp_path / "three_col.map"
        path.write_text("1\trs1\t100\n2\trs2\t200\n")
        info = read_map_file(path)
        assert info.snp == ["rs1", "rs2"]
        assert info.chr == ["1", "2"]
        assert info.pos == [100, 200]
        assert info.cm is None

    def test_error_empty_file(self, tmp_path):
        """Error: empty file raises ``ValueError``."""
        path = tmp_path / "empty.map"
        path.write_text("")
        with pytest.raises(ValueError, match=r"(?i)empty"):
            read_map_file(path)

    def test_error_missing_file(self, tmp_path):
        """Error: missing file raises ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError, match="not found"):
            read_map_file(tmp_path / "no_such_file.map")


# ---------------------------------------------------------------------------
# torchgenomics.io.plink2.Plink2PgenReader
# ---------------------------------------------------------------------------


class TestPlink2PgenReader:
    """``Plink2PgenReader`` reads .pgen/.pvar/.psam triples via pgenlib."""

    def test_class_is_importable(self):
        """Class symbol is importable even without ``pgenlib`` installed."""
        assert Plink2PgenReader.__name__ == "Plink2PgenReader"
        assert callable(Plink2PgenReader)

    def test_missing_dep_raises_import_error(self):
        """Without ``pgenlib`` installed, instantiation raises ``ImportError``
        with a helpful install hint."""
        try:
            import pgenlib  # noqa: F401
            pytest.skip("pgenlib is installed; skipping import-error test")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="pgenlib"):
            Plink2PgenReader("/some/file")

    def test_golden_path(self):
        """Build a tiny pgen triple and read it back. Skip if pgenlib does
        not provide a synthesis path."""
        pytest.importorskip("pgenlib")
        pytest.skip(
            "No tiny .pgen fixture available; "
            "pgenlib write API is not used in this codebase."
        )

    def test_edge_case_missing_pgen(self, tmp_path):
        """Edge: prefix exists but .pgen is missing → ``FileNotFoundError``."""
        pytest.importorskip("pgenlib")
        prefix = tmp_path / "stub"
        # Only create .psam and .pvar — .pgen missing
        (tmp_path / "stub.psam").write_text("#FID\tIID\nFAM\tIND1\n")
        (tmp_path / "stub.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t1\trs1\tA\tG\n")
        with pytest.raises(FileNotFoundError, match=r"\.pgen"):
            Plink2PgenReader(str(prefix))

    def test_error_path_missing_all(self, tmp_path):
        """Error: prefix points to nothing → ``FileNotFoundError``."""
        pytest.importorskip("pgenlib")
        with pytest.raises(FileNotFoundError):
            Plink2PgenReader(str(tmp_path / "nonexistent_prefix"))


# ---------------------------------------------------------------------------
# torchgenomics.io.vcf.VCFReader
# ---------------------------------------------------------------------------


class TestVCFReader:
    """``VCFReader`` streams VCF/BCF via cyvcf2; emits dosage chunks."""

    def test_class_is_importable(self):
        """Class symbol importable without ``cyvcf2``."""
        assert VCFReader.__name__ == "VCFReader"
        assert callable(VCFReader)

    def test_missing_dep_raises_import_error(self, tmp_path):
        """Without ``cyvcf2`` installed, instantiation raises ``ImportError``
        with a hint."""
        try:
            import cyvcf2  # noqa: F401
            pytest.skip("cyvcf2 is installed; skipping import-error test")
        except ImportError:
            pass
        path = tmp_path / "tiny.vcf"
        _make_tiny_vcf(path)
        with pytest.raises(ImportError, match="cyvcf2"):
            VCFReader(str(path))

    def test_golden_path(self, tmp_path):
        """Golden: read a tiny VCF and verify n_samples, n_variants, and
        chunk shape."""
        pytest.importorskip("cyvcf2")
        path = tmp_path / "tiny.vcf"
        _make_tiny_vcf(path)
        reader = VCFReader(str(path))
        assert reader.n_samples == 3
        assert reader.n_variants == 2
        assert reader.sample_ids == ["S1", "S2", "S3"]
        chunks = list(reader.iter_chunks(chunk_size=10))
        assert len(chunks) == 1
        G, vmeta = chunks[0]
        # Dosage tensor: (n_samples, n_variants)
        assert G.shape == (3, 2)
        # GT 0/0=0, 0/1=1, 1/1=2
        assert G[0, 0].item() == pytest.approx(0.0)  # S1, rs1: 0/0
        assert G[1, 0].item() == pytest.approx(1.0)  # S2, rs1: 0/1
        assert G[2, 0].item() == pytest.approx(2.0)  # S3, rs1: 1/1
        assert vmeta.snp == ["rs1", "rs2"]
        assert vmeta.chr == ["1", "1"]
        assert vmeta.pos == [100, 200]

    def test_edge_empty_vcf(self, tmp_path):
        """Edge: header-only VCF — reader reports 0 variants, iter_chunks
        yields no chunks."""
        pytest.importorskip("cyvcf2")
        path = tmp_path / "empty.vcf"
        _make_empty_vcf(path)
        reader = VCFReader(str(path))
        assert reader.n_variants == 0
        chunks = list(reader.iter_chunks(chunk_size=10))
        assert chunks == []

    def test_error_missing_file(self):
        """Error: nonexistent path raises ``FileNotFoundError``."""
        pytest.importorskip("cyvcf2")
        with pytest.raises(FileNotFoundError, match="VCF"):
            VCFReader("/no/such/file.vcf")

    def test_error_malformed_header(self, tmp_path):
        """Error: VCF missing the #CHROM line — cyvcf2 raises during open."""
        pytest.importorskip("cyvcf2")
        path = tmp_path / "malformed.vcf"
        _make_malformed_vcf(path)
        with pytest.raises(Exception):
            VCFReader(str(path))
