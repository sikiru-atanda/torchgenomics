"""Phase D.1 — direct tests for torchgenomics.scan.

Complements tests/test_scan_unified.py with coverage the plan identified
as missing:

* chunk-boundary arithmetic at N = 1023, 1024, 1025;
* model-adapter protocol: fit_null invoked once, score_chunk invoked
  once per chunk;
* device handoff from CPU-resident genotype chunks to a target device;
* regression test for the ``_conditional`` attribute drop in
  ``merge_scan_results`` that broke the R-wrapper's gwas_conditional()
  on scans straddling a chunk boundary (> chunk_size SNPs).
"""

from __future__ import annotations

import pytest
import torch

from torchgenomics.config import STAT_DTYPE, TorchGenomicsConfig
from torchgenomics.models.base import NullFit, ScanResult, VariantMeta
from torchgenomics.models.conditional_lmm import ConditionalScanResult
from torchgenomics.scan.unified import UnifiedScanner, merge_scan_results

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _CountingModel:
    """Records every call so tests can assert call counts."""

    def __init__(self) -> None:
        self.fit_null_calls = 0
        self.score_chunk_calls = 0
        self.seen_device: torch.device | None = None

    def fit_null(self, Y, X0, K=None, **kwargs):  # type: ignore[no-untyped-def]
        self.fit_null_calls += 1
        return NullFit(converged=True)

    def score_chunk(self, G_chunk, null_fit, variant_meta, test="wald"):  # type: ignore[no-untyped-def]
        self.score_chunk_calls += 1
        self.seen_device = G_chunk.device
        m = G_chunk.shape[1]
        return ScanResult(
            chr=list(variant_meta.chr),
            pos=list(variant_meta.pos),
            snp=list(variant_meta.snp),
            a1=list(variant_meta.a1),
            a2=list(variant_meta.a2),
            af=G_chunk.mean(dim=0) / 2.0,
            beta=torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device),
            se=torch.ones(m, dtype=STAT_DTYPE, device=G_chunk.device),
            stat=torch.zeros(m, dtype=STAT_DTYPE, device=G_chunk.device),
            p=torch.full((m,), 0.5, dtype=STAT_DTYPE, device=G_chunk.device),
            test=test,
        )


class _ListReader:
    """Minimal GenotypeReader backed by an in-memory genotype tensor."""

    def __init__(self, G: torch.Tensor, vmeta: VariantMeta) -> None:
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
        return [f"S{i}" for i in range(self.n_samples)]

    def iter_chunks(self, chunk_size: int = 1024):
        for start in range(0, self.n_variants, chunk_size):
            end = min(start + chunk_size, self.n_variants)
            vm = VariantMeta(
                snp=self._vmeta.snp[start:end],
                chr=self._vmeta.chr[start:end],
                pos=self._vmeta.pos[start:end],
                a1=self._vmeta.a1[start:end],
                a2=self._vmeta.a2[start:end],
            )
            yield self._G[:, start:end], vm


def _make_vmeta(m: int) -> VariantMeta:
    return VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


def _make_fake_conditional_chunk(
    start: int, m: int, snp_ids: list[str], chr_labels: list[str],
    positions: list[int],
) -> ScanResult:
    """Build a ScanResult whose `_conditional` attribute carries unique
    sentinel values per-SNP so concatenation order is verifiable."""
    n = 10
    G_mean = torch.rand(m, dtype=STAT_DTYPE)
    sr = ScanResult(
        chr=chr_labels,
        pos=positions,
        snp=snp_ids,
        a1=["A"] * m,
        a2=["G"] * m,
        af=G_mean,
        beta=torch.arange(start, start + m, dtype=STAT_DTYPE),
        se=torch.ones(m, dtype=STAT_DTYPE),
        stat=torch.arange(start, start + m, dtype=STAT_DTYPE),
        p=torch.full((m,), 0.5, dtype=STAT_DTYPE),
        test="wald",
    )
    sr._conditional = ConditionalScanResult(
        marginal=sr,
        conditional_beta=torch.arange(start, start + m, dtype=STAT_DTYPE),
        conditional_se=torch.ones(m, dtype=STAT_DTYPE),
        conditional_stat=torch.arange(start, start + m, dtype=STAT_DTYPE),
        conditional_p=torch.full((m,), 0.5, dtype=STAT_DTYPE),
        persistence=torch.ones(m, dtype=torch.bool),
        ld_block_id=[f"b{start + i // 4}" for i in range(m)],
        r2_to_lead=torch.linspace(0.0, 1.0, m, dtype=STAT_DTYPE),
        lead_snp=[f"lead_{start}"] * m,
    )
    return sr


# ---------------------------------------------------------------------------
# Chunk boundary arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_variants", [1023, 1024, 1025, 2047, 2048, 2049])
def test_scanner_covers_every_variant_across_chunk_boundary(n_variants):
    """Every variant appears exactly once in the merged output, regardless
    of whether n_variants falls just below, at, or just above a chunk
    boundary. Guards against off-by-one in UnifiedScanner's range slicing.
    """
    torch.manual_seed(0)
    n = 20
    G = torch.randint(0, 3, (n, n_variants), dtype=STAT_DTYPE)
    vmeta = _make_vmeta(n_variants)

    model = _CountingModel()
    config = TorchGenomicsConfig(chunk_size=1024)
    scanner = UnifiedScanner(_ListReader(G, vmeta), model, config)
    nf = model.fit_null(torch.zeros(n, 1), torch.ones(n, 1))
    result = scanner.scan(nf)

    assert len(result) == n_variants
    assert result.snp == vmeta.snp
    expected_chunks = (n_variants + 1024 - 1) // 1024
    # fit_null was called once above (bookkeeping) plus the scanner itself
    # does not call it again — the scanner consumes the pre-fit NullFit.
    assert model.score_chunk_calls == expected_chunks


# ---------------------------------------------------------------------------
# Model-adapter protocol: call-count guarantees
# ---------------------------------------------------------------------------


def test_scan_calls_score_chunk_once_per_chunk():
    """UnifiedScanner.scan() must invoke score_chunk exactly n_chunks
    times — not once, and not per-variant. Guards against accidental
    regression to a per-SNP inner loop."""
    torch.manual_seed(1)
    n, m, chunk_size = 15, 350, 64
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
    vmeta = _make_vmeta(m)

    model = _CountingModel()
    scanner = UnifiedScanner(
        _ListReader(G, vmeta), model,
        TorchGenomicsConfig(chunk_size=chunk_size),
    )
    nf = NullFit(converged=True)
    scanner.scan(nf)

    expected = (m + chunk_size - 1) // chunk_size
    assert model.score_chunk_calls == expected
    assert model.fit_null_calls == 0  # scanner must not refit


def test_scan_does_not_refit_null():
    """The scanner consumes a pre-fit NullFit and must never call fit_null
    itself. Regression: an earlier draft threaded fit_null into scan()."""
    torch.manual_seed(2)
    G = torch.randint(0, 3, (10, 40), dtype=STAT_DTYPE)
    model = _CountingModel()
    scanner = UnifiedScanner(
        _ListReader(G, _make_vmeta(40)), model,
        TorchGenomicsConfig(chunk_size=16),
    )
    scanner.scan(NullFit(converged=True))
    assert model.fit_null_calls == 0


# ---------------------------------------------------------------------------
# Device handoff
# ---------------------------------------------------------------------------


def test_scan_cpu_keeps_chunks_on_cpu():
    """With device=cpu, chunks reach score_chunk on CPU — no stray .cuda()
    conversion. This guards the `if G_chunk.device != device:` branch."""
    G = torch.randint(0, 3, (12, 50), dtype=STAT_DTYPE)
    model = _CountingModel()
    scanner = UnifiedScanner(
        _ListReader(G, _make_vmeta(50)), model,
        TorchGenomicsConfig(chunk_size=25, device=torch.device("cpu")),
    )
    scanner.scan(NullFit(converged=True))
    assert model.seen_device is not None
    assert model.seen_device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
def test_scan_moves_cpu_chunks_to_cuda():
    """CPU genotype chunks must be moved to CUDA before score_chunk runs
    when the config device is CUDA. Tests the PrefetchIterator fast path."""
    G = torch.randint(0, 3, (8, 64), dtype=STAT_DTYPE)
    model = _CountingModel()
    scanner = UnifiedScanner(
        _ListReader(G, _make_vmeta(64)), model,
        TorchGenomicsConfig(chunk_size=32, device=torch.device("cuda")),
    )
    scanner.scan(NullFit(converged=True))
    assert model.seen_device is not None
    assert model.seen_device.type == "cuda"


# ---------------------------------------------------------------------------
# merge_scan_results: _conditional attribute preservation
# ---------------------------------------------------------------------------


class TestMergeConditionalAttribute:
    """Regression: `_conditional` is a *dynamic* attribute attached by
    ConditionalLMM.score_chunk — it is NOT a ScanResult dataclass field.
    Before the fix, merge_scan_results rebuilt ScanResult from dataclass
    fields only and silently dropped `_conditional`, breaking
    gwas_conditional() for any scan with > chunk_size SNPs."""

    def test_attribute_survives_merge(self):
        chunks = [
            _make_fake_conditional_chunk(
                start=0, m=4,
                snp_ids=["rs0", "rs1", "rs2", "rs3"],
                chr_labels=["1"] * 4,
                positions=[0, 1, 2, 3],
            ),
            _make_fake_conditional_chunk(
                start=4, m=3,
                snp_ids=["rs4", "rs5", "rs6"],
                chr_labels=["1"] * 3,
                positions=[4, 5, 6],
            ),
        ]
        merged = merge_scan_results(chunks)
        assert hasattr(merged, "_conditional"), (
            "merge_scan_results dropped the ConditionalScanResult payload — "
            "regression of the R-wrapper gwas_conditional chunk-boundary bug."
        )

    def test_conditional_tensors_concatenate_in_order(self):
        chunks = [
            _make_fake_conditional_chunk(
                start=0, m=4,
                snp_ids=["rs0", "rs1", "rs2", "rs3"],
                chr_labels=["1"] * 4,
                positions=[0, 1, 2, 3],
            ),
            _make_fake_conditional_chunk(
                start=4, m=3,
                snp_ids=["rs4", "rs5", "rs6"],
                chr_labels=["1"] * 3,
                positions=[4, 5, 6],
            ),
        ]
        merged = merge_scan_results(chunks)
        cond = merged._conditional
        assert cond.conditional_beta.shape == (7,)
        assert torch.equal(
            cond.conditional_beta,
            torch.arange(7, dtype=STAT_DTYPE),
        )

    def test_conditional_lists_concatenate_in_order(self):
        chunks = [
            _make_fake_conditional_chunk(
                start=0, m=4,
                snp_ids=["rs0", "rs1", "rs2", "rs3"],
                chr_labels=["1"] * 4,
                positions=[0, 1, 2, 3],
            ),
            _make_fake_conditional_chunk(
                start=4, m=3,
                snp_ids=["rs4", "rs5", "rs6"],
                chr_labels=["1"] * 3,
                positions=[4, 5, 6],
            ),
        ]
        merged = merge_scan_results(chunks)
        cond = merged._conditional
        assert len(cond.ld_block_id) == 7
        assert len(cond.lead_snp) == 7
        # lead_snp sentinels encode chunk provenance; first 4 should say
        # "lead_0", last 3 should say "lead_4".
        assert cond.lead_snp[:4] == ["lead_0"] * 4
        assert cond.lead_snp[4:] == ["lead_4"] * 3

    def test_merged_conditional_points_at_merged_marginal(self):
        """The nested `marginal` handle should refer to the *merged*
        ScanResult, not one of the input chunks — otherwise downstream
        code inspecting cond.marginal.snp would see only the first chunk."""
        chunks = [
            _make_fake_conditional_chunk(
                start=0, m=4,
                snp_ids=["rs0", "rs1", "rs2", "rs3"],
                chr_labels=["1"] * 4,
                positions=[0, 1, 2, 3],
            ),
            _make_fake_conditional_chunk(
                start=4, m=3,
                snp_ids=["rs4", "rs5", "rs6"],
                chr_labels=["1"] * 3,
                positions=[4, 5, 6],
            ),
        ]
        merged = merge_scan_results(chunks)
        assert merged._conditional.marginal is merged
        assert len(merged._conditional.marginal) == 7

    def test_attribute_not_forced_when_absent(self):
        """If no chunk carries `_conditional`, the merged result must not
        grow a spurious one — keeps the fast path clean for plain LMM."""
        plain = [
            ScanResult(
                chr=["1"] * 3, pos=[0, 1, 2], snp=["rs0", "rs1", "rs2"],
                a1=["A"] * 3, a2=["G"] * 3,
                af=torch.zeros(3, dtype=STAT_DTYPE),
                beta=torch.zeros(3, dtype=STAT_DTYPE),
                se=torch.ones(3, dtype=STAT_DTYPE),
                stat=torch.zeros(3, dtype=STAT_DTYPE),
                p=torch.ones(3, dtype=STAT_DTYPE),
                test="wald",
            ),
            ScanResult(
                chr=["1"] * 2, pos=[3, 4], snp=["rs3", "rs4"],
                a1=["A"] * 2, a2=["G"] * 2,
                af=torch.zeros(2, dtype=STAT_DTYPE),
                beta=torch.zeros(2, dtype=STAT_DTYPE),
                se=torch.ones(2, dtype=STAT_DTYPE),
                stat=torch.zeros(2, dtype=STAT_DTYPE),
                p=torch.ones(2, dtype=STAT_DTYPE),
                test="wald",
            ),
        ]
        merged = merge_scan_results(plain)
        assert not hasattr(merged, "_conditional")
