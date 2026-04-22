"""Phase 4+: UnifiedScanner and merge_scan_results tests."""

from __future__ import annotations

import pytest
import torch

from torchgwas.config import STAT_DTYPE, TorchGWASConfig
from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.preprocess.qc import QCFilterConfig
from torchgwas.scan.unified import UnifiedScanner, merge_scan_results

# --- Fakes for testing ---

class FakeReader:
    """Minimal GenotypeReader for testing."""

    def __init__(self, G: torch.Tensor, vmeta: VariantMeta):
        self._G = G
        self._vmeta = vmeta

    @property
    def n_samples(self) -> int:
        return self._G.shape[0]

    @property
    def n_variants(self) -> int:
        return self._G.shape[1]

    @property
    def sample_ids(self) -> list[str]:
        return [f"S{i}" for i in range(self._G.shape[0])]

    def iter_chunks(self, chunk_size: int = 1024):
        for start in range(0, self.n_variants, chunk_size):
            end = min(start + chunk_size, self.n_variants)
            vmeta_chunk = VariantMeta(
                snp=self._vmeta.snp[start:end],
                chr=self._vmeta.chr[start:end],
                pos=self._vmeta.pos[start:end],
                a1=self._vmeta.a1[start:end],
                a2=self._vmeta.a2[start:end],
            )
            yield self._G[:, start:end], vmeta_chunk


class FakeModel:
    """Minimal BaseModel that returns dummy scan results."""

    def fit_null(self, Y, X0, K=None, **kwargs):
        return NullFit(converged=True)

    def score_chunk(self, G_chunk, null_fit, variant_meta, test="wald"):
        n, m = G_chunk.shape
        return ScanResult(
            chr=variant_meta.chr,
            pos=variant_meta.pos,
            snp=variant_meta.snp,
            a1=variant_meta.a1,
            a2=variant_meta.a2,
            af=G_chunk.mean(dim=0) / 2.0,
            beta=torch.randn(m, dtype=STAT_DTYPE),
            se=torch.abs(torch.randn(m, dtype=STAT_DTYPE)),
            stat=torch.abs(torch.randn(m, dtype=STAT_DTYPE)),
            p=torch.rand(m, dtype=STAT_DTYPE),
            test=test,
        )


@pytest.fixture
def sample_data():
    """50 samples, 100 SNPs."""
    torch.manual_seed(42)
    n, m = 50, 100
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * 50 + ["2"] * 50,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )
    return G, vmeta


class TestMergeScanResults:
    def test_merge_single(self, sample_data):
        G, vmeta = sample_data
        model = FakeModel()
        nf = model.fit_null(G, torch.ones(50, 1))
        r = model.score_chunk(G, nf, vmeta)
        merged = merge_scan_results([r])
        assert len(merged) == 100

    def test_merge_multiple(self, sample_data):
        G, vmeta = sample_data
        model = FakeModel()
        nf = model.fit_null(G, torch.ones(50, 1))
        r1 = model.score_chunk(G[:, :50], nf, VariantMeta(
            snp=vmeta.snp[:50], chr=vmeta.chr[:50], pos=vmeta.pos[:50],
            a1=vmeta.a1[:50], a2=vmeta.a2[:50],
        ))
        r2 = model.score_chunk(G[:, 50:], nf, VariantMeta(
            snp=vmeta.snp[50:], chr=vmeta.chr[50:], pos=vmeta.pos[50:],
            a1=vmeta.a1[50:], a2=vmeta.a2[50:],
        ))
        merged = merge_scan_results([r1, r2])
        assert len(merged) == 100
        assert merged.snp == vmeta.snp

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="No scan results"):
            merge_scan_results([])


class TestUnifiedScanner:
    def test_scan_returns_scanresult(self, sample_data):
        """scan() returns a ScanResult with all required fields."""
        G, vmeta = sample_data
        reader = FakeReader(G, vmeta)
        model = FakeModel()
        config = TorchGWASConfig(chunk_size=30)
        scanner = UnifiedScanner(reader, model, config)
        nf = model.fit_null(G, torch.ones(50, 1))
        result = scanner.scan(nf, test="wald")
        assert isinstance(result, ScanResult)
        assert len(result) == 100
        assert result.test == "wald"

    def test_scan_chunked_matches_full(self, sample_data):
        """Chunked scan matches full-matrix scan on small data."""
        G, vmeta = sample_data
        reader = FakeReader(G, vmeta)
        model = FakeModel()

        # Full scan (one chunk)
        config_full = TorchGWASConfig(chunk_size=100)
        scanner_full = UnifiedScanner(reader, model, config_full)
        nf = model.fit_null(G, torch.ones(50, 1))
        result_full = scanner_full.scan(nf)

        # Chunked scan
        config_chunked = TorchGWASConfig(chunk_size=30)
        reader2 = FakeReader(G, vmeta)
        scanner_chunked = UnifiedScanner(reader2, model, config_chunked)
        result_chunked = scanner_chunked.scan(nf)

        # Same number of variants
        assert len(result_full) == len(result_chunked)
        assert result_full.snp == result_chunked.snp

    def test_scan_respects_maf_filter(self, sample_data):
        """Variants below MAF threshold are excluded from results."""
        G, vmeta = sample_data
        # Make first 10 SNPs monomorphic
        G[:, :10] = 0.0
        reader = FakeReader(G, vmeta)
        model = FakeModel()
        config = TorchGWASConfig(chunk_size=50)
        scanner = UnifiedScanner(reader, model, config)
        nf = model.fit_null(G, torch.ones(50, 1))
        qc = QCFilterConfig(maf_min=0.01, miss_max=1.0)
        result = scanner.scan(nf, qc_config=qc)
        # Monomorphic SNPs should be excluded
        assert len(result) < 100
        for snp in result.snp:
            idx = int(snp.replace("rs", ""))
            assert idx >= 10  # mono SNPs excluded

    def test_scan_respects_missingness_filter(self, sample_data):
        """Variants above miss-max threshold are excluded."""
        G, vmeta = sample_data
        # Make first 5 SNPs 100% missing
        G[:, :5] = float("nan")
        reader = FakeReader(G, vmeta)
        model = FakeModel()
        config = TorchGWASConfig(chunk_size=50)
        scanner = UnifiedScanner(reader, model, config)
        nf = model.fit_null(G, torch.ones(50, 1))
        qc = QCFilterConfig(maf_min=0.0, miss_max=0.5)
        result = scanner.scan(nf, qc_config=qc)
        assert len(result) <= 95


class TestModelAdapter:
    def test_adapter_wraps_any_basemodel(self):
        """ModelAdapter accepts any BaseModel-compatible object."""
        from torchgwas.scan.adapters import ModelAdapter
        model = FakeModel()
        adapter = ModelAdapter(model)
        assert adapter.model is model
