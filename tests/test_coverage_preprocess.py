"""Tier-2 behavioral coverage tests for ``torchgwas.preprocess.*``.

Bar (Pillar A spec section 4.3 Tier 2):
- Per public symbol: golden_path, edge_case, error_path test methods.
- Behavioral contract verified — return shape, key invariants, documented
  exceptions raised on bad input.
- Tier 2 (behavioral) does NOT require numerical equivalence to a reference
  tool; it requires the documented contract to hold.

Covers 14 preprocess public symbols across 6 submodules:

- ``torchgwas.preprocess.polyrad_wrapper.DosageProbabilities`` (dataclass)
- ``torchgwas.preprocess.polyrad_wrapper.run_polyrad`` (function)
- ``torchgwas.preprocess.dosage_uncertainty.dosage_rsq`` (function)
- ``torchgwas.preprocess.impute.HAS_NATIVE_IMPUTE_KNN`` (bool constant)
- ``torchgwas.preprocess.impute.HAS_NATIVE_IMPUTE_LD`` (bool constant)
- ``torchgwas.preprocess.impute.HAS_NATIVE_IMPUTE_MODE`` (bool constant)
- ``torchgwas.preprocess._impute_gpu.impute_knn_gpu`` (function)
- ``torchgwas.preprocess._impute_gpu.impute_ld_gpu`` (function)
- ``torchgwas.preprocess._impute_gpu.impute_mode_gpu`` (function)
- ``torchgwas.preprocess.impute_external.run_beagle`` (function)
- ``torchgwas.preprocess.impute_external.run_impute5`` (function)
- ``torchgwas.preprocess.impute_external.run_minimac4`` (function)
- ``torchgwas.preprocess.phase.load_haplotypes`` (function)
- ``torchgwas.preprocess.phase.phase_beagle`` (function)

External-tool wrappers (run_polyrad, run_beagle, run_impute5,
run_minimac4, phase_beagle) and CUDA-only GPU imputation tests use
``pytest.importorskip`` / ``pytest.mark.skipif`` gracefully — these
are documented optional dependencies.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from torchgwas.preprocess.impute import (
    impute_knn_gpu,
    impute_ld_gpu,
    impute_mode_gpu,
)
from torchgwas.preprocess.dosage_uncertainty import dosage_rsq
from torchgwas.preprocess.impute import (
    HAS_NATIVE_IMPUTE_KNN,
    HAS_NATIVE_IMPUTE_LD,
    HAS_NATIVE_IMPUTE_MODE,
    impute_mean,
)
from torchgwas.preprocess.impute_external import (
    ImputationResult,
    run_beagle,
    run_impute5,
    run_minimac4,
)
from torchgwas.preprocess.phase import load_haplotypes, phase_beagle
from torchgwas.preprocess.polyrad_wrapper import DosageProbabilities, run_polyrad


pytestmark = pytest.mark.timeout(60)

CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phased_vcf(path: Path, ploidy: int = 2, phased: bool = True) -> None:
    """Write a tiny VCF with phased (or unphased) GT for a few variants.

    Diploid samples have genotypes like ``0|1`` (phased) / ``0/1`` (unphased).
    """
    sep = "|" if phased else "/"
    samples = ["S1", "S2", "S3"]

    # Construct GT strings per sample; ploidy-aware.
    def gt(allele_pattern: list[int]) -> str:
        return sep.join(str(a) for a in allele_pattern)

    s1_v1 = gt([0] * ploidy)              # all reference
    s2_v1 = gt([0, 1] * (ploidy // 2))    # alternating
    s3_v1 = gt([1] * ploidy)              # all alt

    s1_v2 = gt([0, 1] * (ploidy // 2))
    s2_v2 = gt([1] * ploidy)
    s3_v2 = gt([0] * ploidy)

    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{samples[0]}\t{samples[1]}\t{samples[2]}\n"
        f"1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t{s1_v1}\t{s2_v1}\t{s3_v1}\n"
        f"1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t{s1_v2}\t{s2_v2}\t{s3_v2}\n"
    )


# ---------------------------------------------------------------------------
# torchgwas.preprocess.polyrad_wrapper.DosageProbabilities
# ---------------------------------------------------------------------------


class TestDosageProbabilities:
    """``DosageProbabilities`` is a small dataclass holding posterior
    probabilities, expected dosages, and ID lists from polyRAD/updog calls."""

    def test_golden_construct(self):
        """Build with documented fields and round-trip every attribute."""
        n, m, ploidy = 4, 3, 2
        probs = torch.full((n, m, ploidy + 1), 1.0 / (ploidy + 1), dtype=torch.float64)
        expected = torch.full((n, m), float(ploidy) / 2.0, dtype=torch.float64)
        dp = DosageProbabilities(
            probs=probs,
            expected=expected,
            sample_ids=[f"S{i}" for i in range(n)],
            marker_ids=[f"M{j}" for j in range(m)],
            ploidy=ploidy,
        )
        assert torch.equal(dp.probs, probs)
        assert torch.equal(dp.expected, expected)
        assert dp.sample_ids == ["S0", "S1", "S2", "S3"]
        assert dp.marker_ids == ["M0", "M1", "M2"]
        assert dp.ploidy == ploidy

    def test_edge_polyploid(self):
        """Edge: ploidy=4 → probs last-dim is 5."""
        n, m, ploidy = 2, 2, 4
        probs = torch.zeros((n, m, ploidy + 1), dtype=torch.float64)
        probs[..., 0] = 1.0  # all-zero dosage
        dp = DosageProbabilities(
            probs=probs,
            expected=torch.zeros((n, m), dtype=torch.float64),
            sample_ids=["A", "B"],
            marker_ids=["M1", "M2"],
            ploidy=ploidy,
        )
        assert dp.probs.shape == (n, m, ploidy + 1)
        assert dp.ploidy == 4

    def test_repr_contains_class_name(self):
        """``repr`` yields a string containing the type name."""
        dp = DosageProbabilities(
            probs=torch.zeros((1, 1, 3)),
            expected=torch.zeros((1, 1)),
            sample_ids=["S1"],
            marker_ids=["M1"],
            ploidy=2,
        )
        s = repr(dp)
        assert "DosageProbabilities" in s


# ---------------------------------------------------------------------------
# torchgwas.preprocess.polyrad_wrapper.run_polyrad
# ---------------------------------------------------------------------------


class TestRunPolyrad:
    """``run_polyrad`` shells out to R and the polyRAD package. Without R
    installed, instantiation raises ``RuntimeError`` with an install hint."""

    def test_signature_callable(self):
        """The function symbol is importable and callable."""
        assert callable(run_polyrad)
        assert run_polyrad.__name__ == "run_polyrad"

    def test_missing_R_raises_runtime_error(self):
        """Without ``Rscript`` discoverable, run_polyrad raises ``RuntimeError``."""
        if shutil.which("Rscript") is not None:
            pytest.skip("Rscript is on PATH; skipping the missing-R test")
        with pytest.raises(RuntimeError, match=r"(?i)rscript"):
            run_polyrad("/no/such/file.vcf", ploidy=2)

    def test_explicit_bogus_r_executable(self, tmp_path):
        """Edge: explicit ``r_executable`` pointing to a non-existent path
        causes the subprocess invocation to fail (FileNotFoundError or
        RuntimeError, depending on whether the script reaches subprocess.run).
        Either is acceptable per the documented contract."""
        if shutil.which("Rscript") is None:
            pytest.skip("Rscript not available; missing-R path is the wrong fixture")
        bogus_r = str(tmp_path / "no_rscript_here")
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            run_polyrad(
                "/no/such/file.vcf",
                ploidy=2,
                r_executable=bogus_r,
            )


# ---------------------------------------------------------------------------
# torchgwas.preprocess.dosage_uncertainty.dosage_rsq
# ---------------------------------------------------------------------------


class TestDosageRsq:
    """``dosage_rsq`` computes per-marker imputation-quality R² in [0, 1]
    from a (n_samples, n_markers, ploidy+1) probability tensor."""

    def test_golden_path(self):
        """Build a small probabilities tensor and verify shape + range."""
        n, m, ploidy = 6, 4, 2
        # Mix of certain and uncertain calls so AF is in (0, 1)
        probs = torch.zeros((n, m, ploidy + 1), dtype=torch.float64)
        for i in range(n):
            for j in range(m):
                d = (i + j) % (ploidy + 1)
                probs[i, j, d] = 1.0
        rsq = dosage_rsq(probs, ploidy)
        assert rsq.shape == (m,)
        assert torch.all(rsq >= 0.0)
        assert torch.all(rsq <= 1.0)

    def test_certain_genotypes_high_rsq(self):
        """Sharp probabilities (delta distributions) → R² ≈ 1 across markers."""
        n, m, ploidy = 8, 3, 2
        probs = torch.zeros((n, m, ploidy + 1), dtype=torch.float64)
        # ensure a range of dosages so AF is not 0 or 1
        for i in range(n):
            for j in range(m):
                probs[i, j, (i + j) % (ploidy + 1)] = 1.0
        rsq = dosage_rsq(probs, ploidy)
        assert torch.allclose(rsq, torch.ones(m, dtype=torch.float64), atol=1e-9)

    def test_uniform_probs_low_rsq(self):
        """Uniform probabilities (max entropy) → R² ≈ 0 across markers."""
        n, m, ploidy = 5, 3, 2
        probs = torch.full(
            (n, m, ploidy + 1), 1.0 / (ploidy + 1), dtype=torch.float64
        )
        rsq = dosage_rsq(probs, ploidy)
        # Uniform probs collapse mean dosage to ploidy/2 = 1 (AF = 0.5)
        # var per cell = ploidy * (1 - 1/(ploidy+1)) / ... contract: clamps to >= 0.
        assert torch.all(rsq >= 0.0)
        assert torch.all(rsq <= 1.0)
        # Documented contract: clamp(rsq, min=0, max=1). We expect a low value;
        # exact identity to 0 is not contracted (depends on AF vs uniform).
        assert torch.all(rsq < 0.5)

    def test_invalid_input_wrong_last_dim(self):
        """Error: probs.shape[-1] != ploidy+1 raises ``ValueError``."""
        n, m, ploidy = 3, 2, 2
        bad = torch.zeros((n, m, ploidy), dtype=torch.float64)  # missing one bin
        with pytest.raises(ValueError, match=r"ploidy\+1|last dim"):
            dosage_rsq(bad, ploidy)


# ---------------------------------------------------------------------------
# torchgwas.preprocess.impute.HAS_NATIVE_IMPUTE_*
# ---------------------------------------------------------------------------


class TestHasNativeImputeKnn:
    """``HAS_NATIVE_IMPUTE_KNN`` reflects whether the C++ ``_impute_knn_native``
    extension was built and is importable."""

    def test_is_bool(self):
        """Constant is a Python bool (not bool-y)."""
        assert isinstance(HAS_NATIVE_IMPUTE_KNN, bool)

    def test_consistent_with_native_module(self):
        """Cross-check by attempting to import ``_impute_knn_native`` from
        ``torchgwas._native``: True iff that import succeeds."""
        try:
            from torchgwas._native import _impute_knn_native  # noqa: F401
            native_present = _impute_knn_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_IMPUTE_KNN == native_present


class TestHasNativeImputeLd:
    """``HAS_NATIVE_IMPUTE_LD`` reflects whether the C++ ``_impute_ld_native``
    extension was built."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_IMPUTE_LD, bool)

    def test_consistent_with_native_module(self):
        try:
            from torchgwas._native import _impute_ld_native  # noqa: F401
            native_present = _impute_ld_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_IMPUTE_LD == native_present


class TestHasNativeImputeMode:
    """``HAS_NATIVE_IMPUTE_MODE`` reflects whether the C++
    ``_impute_mode_native`` extension was built."""

    def test_is_bool(self):
        assert isinstance(HAS_NATIVE_IMPUTE_MODE, bool)

    def test_consistent_with_native_module(self):
        try:
            from torchgwas._native import _impute_mode_native  # noqa: F401
            native_present = _impute_mode_native is not None
        except ImportError:
            native_present = False
        assert HAS_NATIVE_IMPUTE_MODE == native_present


# ---------------------------------------------------------------------------
# torchgwas.preprocess._impute_gpu.impute_knn_gpu / impute_ld_gpu / impute_mode_gpu
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestImputeModeGpu:
    """``impute_mode_gpu`` returns per-column mode of an integer dosage matrix
    where -1 marks missing entries."""

    def test_smoke_runs_on_gpu(self):
        """Build small G_int with -1 sentinels; verify shape and mode value."""
        G_int = torch.tensor(
            [[0, 1, 2], [0, -1, 2], [1, 1, 2]],
            dtype=torch.long,
            device="cuda",
        )
        modes = impute_mode_gpu(G_int, max_dosage=2, n_cols=3)
        assert modes.shape == (3,)
        assert modes.dtype == torch.float64
        # col 0 has [0, 0, 1] → mode 0; col 1 has [1, ., 1] → mode 1; col 2 → 2
        assert modes[0].item() == pytest.approx(0.0)
        assert modes[1].item() == pytest.approx(1.0)
        assert modes[2].item() == pytest.approx(2.0)

    def test_all_missing_column_returns_zero(self):
        """Edge: column with all -1 → mode is 0.0 (argmax over zero counts)."""
        G_int = torch.tensor(
            [[-1, 2], [-1, 2], [-1, 2]],
            dtype=torch.long,
            device="cuda",
        )
        modes = impute_mode_gpu(G_int, max_dosage=2, n_cols=2)
        assert modes[0].item() == pytest.approx(0.0)
        assert modes[1].item() == pytest.approx(2.0)

    def test_invalid_input_raises(self):
        """Error: passing a non-integer dtype (scatter_add_ requires int32 /
        int64 indices) raises ``RuntimeError`` synchronously — does not
        corrupt the CUDA context."""
        G_bad = torch.tensor([[0.0, 1.0]], dtype=torch.float64, device="cuda")
        with pytest.raises(RuntimeError, match=r"(?i)int32|int64|index"):
            impute_mode_gpu(G_bad, max_dosage=2, n_cols=2)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestImputeKnnGpu:
    """``impute_knn_gpu`` fills NaN entries via weighted KNN over a GRM."""

    def test_smoke_runs_on_gpu(self):
        """Tiny G with NaN markers; output has matching shape, no NaN."""
        G = torch.tensor(
            [[0.0, 1.0], [float("nan"), 0.0], [2.0, 1.0], [1.0, float("nan")]],
            device="cuda",
            dtype=torch.float64,
        )
        K = torch.eye(4, device="cuda", dtype=torch.float64) + 0.1
        mask = torch.isnan(G)
        K_mod = K.clone()
        K_mod.fill_diagonal_(-float("inf"))
        _, knn_idx = torch.topk(K_mod, k=2, dim=1)
        out = impute_knn_gpu(G, K, knn_idx, mask)
        assert out.shape == G.shape
        assert not torch.isnan(out).any()
        # Observed entries are preserved
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[2, 0].item() == pytest.approx(2.0)

    def test_no_nan_to_impute(self):
        """Edge: G with no NaN → output equals input element-wise."""
        G = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]],
            device="cuda",
            dtype=torch.float64,
        )
        K = torch.eye(3, device="cuda", dtype=torch.float64) + 0.1
        mask = torch.isnan(G)  # all False
        K_mod = K.clone()
        K_mod.fill_diagonal_(-float("inf"))
        _, knn_idx = torch.topk(K_mod, k=2, dim=1)
        out = impute_knn_gpu(G, K, knn_idx, mask)
        assert torch.equal(out, G)

    def test_invalid_input_shape_mismatch(self):
        """Error: mismatched mask vs G shapes → torch raises during indexing."""
        G = torch.zeros((3, 4), device="cuda", dtype=torch.float64)
        G[0, 0] = float("nan")
        K = torch.eye(3, device="cuda", dtype=torch.float64)
        K_mod = K.clone()
        K_mod.fill_diagonal_(-float("inf"))
        _, knn_idx = torch.topk(K_mod, k=2, dim=1)
        bad_mask = torch.zeros((3, 99), dtype=torch.bool, device="cuda")
        with pytest.raises((RuntimeError, IndexError, ValueError)):
            impute_knn_gpu(G, K, knn_idx, bad_mask)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestImputeLdGpu:
    """``impute_ld_gpu`` fills NaN entries via local correlation-weighted
    regression over a flanking window of SNPs."""

    def test_smoke_runs_on_gpu(self):
        """Tiny G with NaN markers; output has matching shape, no NaN."""
        G = torch.tensor(
            [
                [0.0, 1.0, 2.0, 1.0],
                [float("nan"), 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, float("nan"), 1.0],
            ],
            device="cuda",
            dtype=torch.float64,
        )
        mask = torch.isnan(G)
        G_complete = impute_mean(G)
        out = impute_ld_gpu(G, G_complete, mask, window_size=2)
        assert out.shape == G.shape
        assert not torch.isnan(out).any()
        # Observed entries preserved exactly
        assert out[0, 0].item() == pytest.approx(0.0)
        assert out[2, 0].item() == pytest.approx(2.0)

    def test_no_nan_to_impute(self):
        """Edge: G with no NaN → output equals G (every position un-touched)."""
        G = torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]],
            device="cuda",
            dtype=torch.float64,
        )
        mask = torch.isnan(G)
        G_complete = impute_mean(G)
        out = impute_ld_gpu(G, G_complete, mask, window_size=1)
        assert torch.equal(out, G)

    def test_invalid_input_negative_window(self):
        """Error: negative window_size collapses every column to the
        target-only window (right - left <= 1) and falls back to G_complete.
        Documented contract: returns without raising — output equals
        mean-imputed G at NaN positions."""
        G = torch.tensor(
            [[0.0, 1.0], [float("nan"), 1.0], [2.0, 0.0]],
            device="cuda",
            dtype=torch.float64,
        )
        mask = torch.isnan(G)
        G_complete = impute_mean(G)
        # window_size = -1 ⇒ left=j+1, right=j → idxs empty → fallback
        out = impute_ld_gpu(G, G_complete, mask, window_size=-1)
        assert not torch.isnan(out).any()
        # Imputed value matches mean-imputation fallback
        assert out[1, 0].item() == pytest.approx(G_complete[1, 0].item())


# ---------------------------------------------------------------------------
# torchgwas.preprocess.impute_external.run_beagle / run_impute5 / run_minimac4
# ---------------------------------------------------------------------------


class TestRunBeagle:
    """``run_beagle`` shells out to BEAGLE 5.x via java -jar."""

    def test_signature_callable(self):
        """Function importable and callable; returns ``ImputationResult``."""
        assert callable(run_beagle)
        # The dataclass companion is also part of the public surface.
        assert ImputationResult.__name__ == "ImputationResult"

    def test_missing_java_raises_runtime_error(self):
        """Without ``java`` on PATH, the wrapper raises ``RuntimeError``."""
        if shutil.which("java") is not None:
            pytest.skip("java is on PATH; this test exercises the no-Java path")
        with pytest.raises(RuntimeError, match=r"(?i)java"):
            run_beagle("in.vcf", "out")

    def test_missing_jar_fails_subprocess(self, tmp_path):
        """Edge: java exists but BEAGLE jar is missing → subprocess returns
        non-zero, surfaced as ``RuntimeError``."""
        if shutil.which("java") is None:
            pytest.skip("java not on PATH; cannot exercise the jar-missing path")
        if shutil.which("beagle.jar") is not None:
            pytest.skip("beagle.jar found on PATH; this test exercises the missing-jar path")
        # Provide a tmp input path so subprocess.run actually fires
        in_vcf = tmp_path / "in.vcf"
        in_vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        with pytest.raises(RuntimeError, match=r"(?i)beagle"):
            run_beagle(str(in_vcf), str(tmp_path / "out"))


class TestRunImpute5:
    """``run_impute5`` shells out to the impute5 binary."""

    def test_signature_callable(self):
        assert callable(run_impute5)

    def test_missing_binary_raises_runtime_error(self):
        """Without ``impute5`` on PATH, raises ``RuntimeError``."""
        if shutil.which("impute5") is not None:
            pytest.skip("impute5 is on PATH; skipping the missing-binary test")
        with pytest.raises(RuntimeError, match=r"impute5"):
            run_impute5("in.vcf", "out.vcf", ref_panel="ref.vcf")


class TestRunMinimac4:
    """``run_minimac4`` shells out to the minimac4 binary."""

    def test_signature_callable(self):
        assert callable(run_minimac4)

    def test_missing_binary_raises_runtime_error(self):
        """Without ``minimac4`` on PATH, raises ``RuntimeError``."""
        if shutil.which("minimac4") is not None:
            pytest.skip("minimac4 is on PATH; skipping the missing-binary test")
        with pytest.raises(RuntimeError, match=r"minimac4"):
            run_minimac4("in.vcf", "out", ref_panel="ref")


# ---------------------------------------------------------------------------
# torchgwas.preprocess.phase.load_haplotypes
# ---------------------------------------------------------------------------


class TestLoadHaplotypes:
    """``load_haplotypes`` reads phased VCFs into a (n, ploidy, p) tensor
    via ``cyvcf2``. When cyvcf2 is missing, the function raises
    ``ImportError`` with a hint."""

    def test_signature_callable(self):
        assert callable(load_haplotypes)

    def test_missing_cyvcf2_raises_import_error(self, tmp_path):
        """Without ``cyvcf2`` installed, ``load_haplotypes`` raises
        ``ImportError`` with an install hint."""
        try:
            import cyvcf2  # noqa: F401
            pytest.skip("cyvcf2 is installed; skipping the import-error test")
        except ImportError:
            pass
        path = tmp_path / "tiny.vcf"
        _make_phased_vcf(path)
        with pytest.raises(ImportError, match="cyvcf2"):
            load_haplotypes(str(path), ploidy=2)

    def test_golden_path(self, tmp_path):
        """With cyvcf2, load a tiny phased VCF and verify shape + values."""
        pytest.importorskip("cyvcf2")
        path = tmp_path / "tiny.vcf"
        _make_phased_vcf(path, ploidy=2, phased=True)
        haps = load_haplotypes(str(path), ploidy=2)
        # Documented shape: (n_samples, ploidy, n_variants)
        assert haps.shape == (3, 2, 2)
        # S1 v1 = 0|0 → both haplotypes are 0
        assert haps[0, 0, 0].item() == pytest.approx(0.0)
        assert haps[0, 1, 0].item() == pytest.approx(0.0)
        # S3 v1 = 1|1 → both haplotypes are 1
        assert haps[2, 0, 0].item() == pytest.approx(1.0)
        assert haps[2, 1, 0].item() == pytest.approx(1.0)

    def test_error_invalid_path(self):
        """Error: non-existent file → ``cyvcf2`` raises (we surface that)."""
        pytest.importorskip("cyvcf2")
        # cyvcf2 raises Exception on missing path.
        with pytest.raises(Exception):
            load_haplotypes("/no/such/path/tiny.vcf", ploidy=2)


# ---------------------------------------------------------------------------
# torchgwas.preprocess.phase.phase_beagle
# ---------------------------------------------------------------------------


class TestPhaseBeagle:
    """``phase_beagle`` shells out to BEAGLE 5.x for diploid phasing.
    Polyploid is rejected up-front with a ``ValueError``."""

    def test_signature_callable(self):
        assert callable(phase_beagle)

    def test_polyploid_raises_value_error(self):
        """Edge: ploidy != 2 raises ``ValueError`` (BEAGLE is diploid-only)."""
        with pytest.raises(ValueError, match=r"(?i)diploid|ploidy"):
            phase_beagle("in.vcf", "out", ploidy=4)

    def test_missing_java_raises_runtime_error(self):
        """Without ``java`` on PATH, raises ``RuntimeError``."""
        if shutil.which("java") is not None:
            pytest.skip("java is on PATH; cannot exercise the no-Java path")
        with pytest.raises(RuntimeError, match=r"(?i)java"):
            phase_beagle("in.vcf", "out", ploidy=2)

    def test_missing_jar_fails_subprocess(self, tmp_path):
        """Error: java exists but BEAGLE jar is absent → subprocess fails,
        surfaced as ``RuntimeError``."""
        if shutil.which("java") is None:
            pytest.skip("java not on PATH; cannot exercise jar-missing path")
        if shutil.which("beagle.jar") is not None:
            pytest.skip("beagle.jar found; this test targets the missing-jar path")
        in_vcf = tmp_path / "in.vcf"
        in_vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        with pytest.raises(RuntimeError, match=r"(?i)beagle"):
            phase_beagle(str(in_vcf), str(tmp_path / "out"), ploidy=2)
