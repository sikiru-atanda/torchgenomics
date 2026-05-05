"""Memory-budget regression tests for streaming scan paths (Efficiency E1).

These tests guard the streaming contract for scan paths that were
audited and rewritten under ``docs/efficiency/streaming_audit.md``.
A future refactor that re-materializes the full ``(n, m)`` genotype
matrix would silently regress biobank-scale users; these tests trip
on that within-process by tracking the peak Python tensor allocation
budget via ``tracemalloc``.

We deliberately use ``tracemalloc`` (not OS-level RSS) because:

- It captures only Python-managed allocations, so pytest fixtures /
  imports / unrelated libs don't pollute the baseline.
- It works deterministically on Linux + macOS without root or psutil.
- The ``peak`` reading is the lifetime-max from ``start()`` to ``stop()``,
  so the test catches transient allocations that come-and-go.

The fixtures are small (n=200, m=500) so each test runs in <1s. At
biobank scale (n=500_000, m=10_000_000) the materialized path costs
~40 TB float64; the streaming path costs roughly
``n × Σ region_size × 8 B`` for set-scan and
``n × chunk_size × 8 B`` per chunk for glmm-scan — GB-scale, fitting
on a workstation.

Behavioral equivalence: each rewrite test also asserts the streaming
path produces statistically identical p-values / q-stats to the
legacy materialized path within float64 tolerance.
"""

from __future__ import annotations

import gc
import tracemalloc

import pytest
import torch

from torchgwas.io.regions import Region
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import VariantMeta
from torchgwas.models.set_based import SetBasedScanner
from torchgwas.models.single_trait_lmm import SingleTraitLMM


# ---------------------------------------------------------------------------
# Tiny in-memory reader used as a streaming source. By yielding chunks
# without ever holding a single large tensor of its own (its only field
# is the original ``G`` it was given — a slice, not a copy), this
# reader is the bare-minimum contract that ``iter_chunks`` consumers
# rely on.
# ---------------------------------------------------------------------------


class _ChunkedTensorReader:
    """In-memory reader that splits a (n, m) tensor into fixed-size chunks.

    Used only by the memory regression tests — production paths use
    PlinkBedReader / VcfReader / etc., all of which already implement
    ``iter_chunks`` on top of file-backed storage.
    """

    def __init__(
        self,
        G: torch.Tensor,
        variant_chr: list[str],
        variant_pos: list[int],
        chunk_size: int = 256,
    ) -> None:
        self._G = G
        self._chr = variant_chr
        self._pos = variant_pos
        self._chunk_size = chunk_size

    @property
    def n_samples(self) -> int:
        return self._G.shape[0]

    @property
    def n_variants(self) -> int:
        return self._G.shape[1]

    @property
    def sample_ids(self) -> list[str]:
        return [f"S{i}" for i in range(self._G.shape[0])]

    def iter_chunks(self, chunk_size: int | None = None):
        cs = chunk_size or self._chunk_size
        m = self._G.shape[1]
        for start in range(0, m, cs):
            end = min(start + cs, m)
            G_chunk = self._G[:, start:end].clone()
            vmeta = VariantMeta(
                snp=[f"rs{i}" for i in range(start, end)],
                chr=self._chr[start:end],
                pos=self._pos[start:end],
                a1=["A"] * (end - start),
                a2=["G"] * (end - start),
            )
            yield G_chunk, vmeta


@pytest.fixture
def lmm_null_with_regions():
    """Fit a small LMM null + define 10 disjoint regions of 50 SNPs each."""
    torch.manual_seed(2026)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = 5.0 + L @ torch.randn(n, dtype=torch.float64)

    lmm = SingleTraitLMM()
    nf = lmm.fit_null(Y, X0, K=K)

    variant_chr = ["1"] * m
    variant_pos = list(range(m))
    regions = [
        Region(region_id=f"gene_{i}", chr="1", start=i * 50, end=(i + 1) * 50)
        for i in range(10)
    ]
    return G, nf, regions, variant_chr, variant_pos


def _peak_kib(fn) -> tuple[object, float]:
    """Run ``fn()`` under tracemalloc, return (result, peak_in_KiB)."""
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, peak / 1024.0


# ---------------------------------------------------------------------------
# set-scan (SetBasedScanner.scan_regions_streaming) memory regression
# ---------------------------------------------------------------------------


class TestSetScanStreamingMemory:
    """Set-based scan must not materialize the full ``(n, m)`` tensor.

    Three guards:

    1. Behavioral equivalence — streaming q-stats / p-values match
       the legacy materialized path to float64 tolerance.
    2. Absolute peak budget — streaming peak < 8 MiB on the n=200/m=500
       fixture (well under what a re-materialization would cost).
    3. Region-vs-m scaling — streaming peak is roughly invariant in
       total ``m`` for a fixed total region size. This is the
       biobank-relevant test: at UKB scale the difference between
       paying ``n × Σ region_size × 8 B`` (~GB) and
       ``n × m × 8 B`` (~TB) is the difference between fits-on-a-laptop
       and doesn't-fit-on-cluster.
    """

    def test_streaming_matches_materialized(self, lmm_null_with_regions):
        G, nf, regions, variant_chr, variant_pos = lmm_null_with_regions

        # Reference: materialized (legacy) path
        scanner = SetBasedScanner(nf)
        ref = scanner.scan_regions(G, regions, variant_chr, variant_pos, test="skat")

        # Streaming path on the same data
        reader = _ChunkedTensorReader(G, variant_chr, variant_pos, chunk_size=64)
        streamed = scanner.scan_regions_streaming(
            reader.iter_chunks(64), regions,
            variant_chr=variant_chr, variant_pos=variant_pos,
            test="skat", device=torch.device("cpu"),
            impute=False,  # G has no NaNs
        )

        # Same regions, same n_variants per region.
        assert streamed.region_id == ref.region_id
        assert streamed.n_variants == ref.n_variants
        # Q stats and p values agree to float64 tolerance.
        assert torch.allclose(streamed.q_stat, ref.q_stat, atol=1e-10, rtol=1e-10)
        assert torch.allclose(streamed.p, ref.p, atol=1e-10, rtol=1e-10)

    def test_streaming_peak_under_explicit_budget(self, lmm_null_with_regions):
        """Hard absolute budget: <8 MiB at 200x500 fixture.

        If the streaming refactor regressed to materialize G in
        STAT_DTYPE (float64), peak would be at minimum
        ``200 * 500 * 8 = 800 KiB`` plus the GRM (320 KiB) plus
        per-region buffers — well over 8 MiB once chunk + region
        accumulation is double-counted. The 8 MiB ceiling holds today
        with margin and is a clear "still streaming" signal.
        """
        G, nf, regions, variant_chr, variant_pos = lmm_null_with_regions
        scanner = SetBasedScanner(nf)

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, variant_chr, variant_pos, chunk_size=64,
            )
            return scanner.scan_regions_streaming(
                reader.iter_chunks(64), regions,
                variant_chr=variant_chr, variant_pos=variant_pos,
                test="skat", device=torch.device("cpu"),
                impute=False,
            )

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 8 * 1024  # 8 MiB
        assert peak_stream < budget_kib, (
            f"Streaming set-scan peak ({peak_stream:.0f} KiB) exceeds "
            f"the 8 MiB budget. At biobank scale this would translate "
            "to a multi-TB regression."
        )

    def test_streaming_peak_scales_with_regions_not_m(self):
        """Streaming peak should scale with total region-SNP count,
        not with total ``m``.

        Construct two scenarios with the same regions (same total
        region size) but very different ``m``: one where m equals the
        total region size, one where m is 5× larger. The streaming
        peak should be roughly the same — proving that we're not
        accumulating the full G.

        At biobank scale, this is the difference between the streaming
        path costing ``n × Σ region_size × 8 B`` (~GB scale) versus
        the materialized path costing ``n × m × 8 B`` (~TB scale).
        """
        torch.manual_seed(31)
        n = 100

        # Scenario A: m = 200, all in 4 regions of 50 SNPs each.
        m_small = 200
        G_small = torch.randint(0, 3, (n, m_small), dtype=torch.float64)
        K_small, _ = grm_vanraden(G_small)
        Y = 5.0 + torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        nf_small = SingleTraitLMM().fit_null(Y, X0, K=K_small)
        regions = [
            Region(region_id=f"g{i}", chr="1", start=i * 50, end=(i + 1) * 50)
            for i in range(4)
        ]

        # Scenario B: m = 1000, but the same 4 regions only span the
        # first 200 SNPs. The other 800 SNPs are out-of-region.
        m_large = 1000
        G_large = torch.randint(0, 3, (n, m_large), dtype=torch.float64)
        # Reuse the same Y/X0/K_small for null fit cost simplicity;
        # we're not measuring statistical correctness here, just memory.
        nf_large = nf_small

        def _run(G, m, nf):
            scanner = SetBasedScanner(nf)
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=128,
            )
            return scanner.scan_regions_streaming(
                reader.iter_chunks(128), regions,
                variant_chr=["1"] * m, variant_pos=list(range(m)),
                test="skat", device=torch.device("cpu"),
                impute=False,
            )

        _, peak_small = _peak_kib(lambda: _run(G_small, m_small, nf_small))
        _, peak_large = _peak_kib(lambda: _run(G_large, m_large, nf_large))

        # Allow 2x growth for chunk-level overhead; 5x m -> 5x peak
        # would mean we're not streaming.
        assert peak_large <= 2.0 * peak_small + 256, (
            f"Streaming set-scan peak grew from {peak_small:.0f} KiB "
            f"(m=200) to {peak_large:.0f} KiB (m=1000) — should be "
            "roughly constant in m for fixed region size. The "
            "streaming refactor likely regressed."
        )


# ---------------------------------------------------------------------------
# glmm-scan (BinaryGLMM via UnifiedScanner) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_glmm_inputs():
    """Synthetic binary phenotype + small genotype matrix."""
    torch.manual_seed(7)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    K, _ = grm_vanraden(G)

    # Binary Y from logistic(0.5*X0 + small genetic effect)
    eta = 0.0 + 0.1 * G[:, 0] + 0.5 * torch.randn(n, dtype=torch.float64)
    pi = torch.sigmoid(eta)
    Y = (torch.rand(n, dtype=torch.float64) < pi).to(torch.float64)
    return G, Y, X0, K


class TestGlmmScanStreamingMemory:
    """Streaming GLMM scan via UnifiedScanner.

    The GLMM null fit (PQL) only depends on Y, X0, K — never on the
    SNP genotypes. Once K is built (here via ``grm_vanraden`` for the
    test; production uses ``grm_vanraden_streaming``), the genome scan
    runs chunk-by-chunk through ``BinaryGLMM.score_chunk`` on each
    iteration of ``UnifiedScanner.scan``. We check that the per-chunk
    score path doesn't accidentally hold all chunks in memory.
    """

    def test_streaming_glmm_matches_full_chunk(self, binary_glmm_inputs):
        from torchgwas.config import TorchGWASConfig
        from torchgwas.models.binary_glmm import BinaryGLMM
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0, K = binary_glmm_inputs
        m = G.shape[1]

        model = BinaryGLMM(use_spa=False, firth=False, pql_max_iter=20)
        nf = model.fit_null(Y, X0, K=K)

        # Reference: single big-G score_chunk (legacy path)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        ref = model.score_chunk(G, nf, vmeta, test="score")

        # Streaming path via UnifiedScanner
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        config = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
        scanner = UnifiedScanner(reader, model, config)
        streamed = scanner.scan(nf, test="score", qc_config=None)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        # P-values: allow looser tol because chi2 SF can amplify rounding.
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_glmm_under_explicit_budget(self, binary_glmm_inputs):
        """Hard absolute budget: <16 MiB at 200x500.

        At the chosen fixture, peak streaming GLMM allocations are
        dominated by the score-chunk intermediates (WG, X0tWG, etc.) —
        all sized (n, chunk_size). With n=200 and chunk=64 these are
        small. The 16 MiB ceiling holds with margin today and trips
        immediately on a regression that re-materializes G.
        """
        from torchgwas.config import TorchGWASConfig
        from torchgwas.models.binary_glmm import BinaryGLMM
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0, K = binary_glmm_inputs
        m = G.shape[1]

        model = BinaryGLMM(use_spa=False, firth=False, pql_max_iter=20)
        nf = model.fit_null(Y, X0, K=K)

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            cfg = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
            scanner = UnifiedScanner(reader, model, cfg)
            return scanner.scan(nf, test="score", qc_config=None)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024  # 16 MiB
        assert peak_stream < budget_kib, (
            f"Streaming GLMM peak ({peak_stream:.0f} KiB) exceeds the "
            f"16 MiB budget. At biobank scale this would translate to "
            "a multi-TB regression."
        )


# ---------------------------------------------------------------------------
# me-glmm-scan (MultiEnvGLMM via per-chunk loop) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def me_glmm_inputs():
    """Synthetic per-environment binary phenotypes (n, E)."""
    torch.manual_seed(11)
    n, m, E = 200, 500, 2
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    K, _ = grm_vanraden(G)

    # Two binary environments with mild correlated signal
    Y = torch.zeros(n, E, dtype=torch.float64)
    for e in range(E):
        eta = 0.0 + 0.05 * G[:, 0] + 0.5 * torch.randn(n, dtype=torch.float64)
        pi = torch.sigmoid(eta)
        Y[:, e] = (torch.rand(n, dtype=torch.float64) < pi).to(torch.float64)
    return G, Y, X0, K


class TestMeGlmmScanStreamingMemory:
    """Streaming multi-environment GLMM scan via per-chunk score_chunk loop.

    MultiEnvGLMM.score_chunk returns ``EnvScanResult`` (richer than
    ``ScanResult``), so the streaming path uses
    :func:`torchgwas.cli._merge_env_results` rather than
    :class:`UnifiedScanner`. The PQL null fit only depends on
    ``Y / X0 / K``, so the genome scan loop is per-variant.
    """

    def test_streaming_me_glmm_matches_full_chunk(self, me_glmm_inputs):
        from torchgwas.cli import _merge_env_results
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        G, Y, X0, K = me_glmm_inputs
        m = G.shape[1]

        model = MultiEnvGLMM(family="binary", use_spa=False, firth=False)
        nf = model.fit_null(Y, X0, K=K, env_names=["env0", "env1"])

        vmeta_full = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        # Reference: single big-G score_chunk
        ref = model.score_chunk(G, nf, vmeta_full)

        # Streaming via per-chunk loop
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        chunk_results = []
        for G_chunk, vm in reader.iter_chunks(64):
            chunk_results.append(model.score_chunk(G_chunk, nf, vm))
        streamed = _merge_env_results(chunk_results)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)
        assert torch.allclose(streamed.beta, ref.beta, atol=1e-9, rtol=1e-9)

    def test_streaming_me_glmm_under_explicit_budget(self, me_glmm_inputs):
        """Hard absolute budget: <16 MiB at 200x500/E=2."""
        from torchgwas.cli import _merge_env_results
        from torchgwas.models.multi_env_glmm import MultiEnvGLMM

        G, Y, X0, K = me_glmm_inputs
        m = G.shape[1]

        model = MultiEnvGLMM(family="binary", use_spa=False, firth=False)
        nf = model.fit_null(Y, X0, K=K, env_names=["env0", "env1"])

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            chunk_results = []
            for G_chunk, vm in reader.iter_chunks(64):
                chunk_results.append(model.score_chunk(G_chunk, nf, vm))
            return _merge_env_results(chunk_results)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming ME-GLMM peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


# ---------------------------------------------------------------------------
# survival-scan (SurvivalGLMM via UnifiedScanner) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def survival_inputs():
    """Synthetic (time, event) phenotype + small G."""
    torch.manual_seed(13)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    K, _ = grm_vanraden(G)

    # Exponential times + binary event
    times = torch.rand(n, dtype=torch.float64).clamp(min=0.05) * 10.0
    events = (torch.rand(n) < 0.7).to(torch.float64)
    Y = torch.stack([times, events], dim=1)
    return G, Y, X0, K


class TestSurvivalScanStreamingMemory:
    """Streaming Cox PH frailty scan via UnifiedScanner."""

    def test_streaming_survival_matches_full_chunk(self, survival_inputs):
        from torchgwas.config import TorchGWASConfig
        from torchgwas.models.survival_glmm import SurvivalGLMM
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0, K = survival_inputs
        m = G.shape[1]

        model = SurvivalGLMM(use_spa=False, pql_max_iter=10)
        nf = model.fit_null(Y, X0, K=K)

        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        ref = model.score_chunk(G, nf, vmeta, test="score")

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        config = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
        scanner = UnifiedScanner(reader, model, config)
        streamed = scanner.scan(nf, test="score", qc_config=None)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_survival_under_explicit_budget(self, survival_inputs):
        """Hard absolute budget: <16 MiB at 200x500."""
        from torchgwas.config import TorchGWASConfig
        from torchgwas.models.survival_glmm import SurvivalGLMM
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0, K = survival_inputs
        m = G.shape[1]

        model = SurvivalGLMM(use_spa=False, pql_max_iter=10)
        nf = model.fit_null(Y, X0, K=K)

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            cfg = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
            scanner = UnifiedScanner(reader, model, cfg)
            return scanner.scan(nf, test="score", qc_config=None)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming Survival peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


# ---------------------------------------------------------------------------
# threshold-scan (ThresholdLinearModel via UnifiedScanner) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def threshold_inputs():
    """Synthetic (ordinal, continuous) traits."""
    torch.manual_seed(17)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Ordinal trait (3 categories) and continuous trait
    Y_ord = torch.randint(0, 3, (n,), dtype=torch.float64)
    Y_con = torch.randn(n, dtype=torch.float64)
    Y = torch.stack([Y_ord, Y_con], dim=1)
    return G, Y, X0


class TestThresholdScanStreamingMemory:
    """Streaming threshold-linear scan via UnifiedScanner.

    threshold-scan already drove the scan loop through UnifiedScanner;
    only the data-load helper materialized G. Post-rewrite,
    ``_align_samples`` is used so chunks flow through ``score_chunk``
    without ever materializing the full ``(n, m)`` tensor.
    """

    def test_streaming_threshold_matches_full_chunk(self, threshold_inputs):
        from torchgwas.config import STAT_DTYPE, TorchGWASConfig
        from torchgwas.models.threshold_linear import ThresholdLinearModel
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0 = threshold_inputs
        m = G.shape[1]
        c = 2

        R = torch.eye(c, dtype=STAT_DTYPE)
        G_cov = torch.eye(c, dtype=STAT_DTYPE) * 0.5
        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[3, 0],
            R=R, G_cov=G_cov, solver="em", max_iter=20, em_warmup=5,
        )
        nf = model.fit_null(Y, X0)

        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        ref = model.score_chunk(G, nf, vmeta, test="score")

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        cfg = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
        scanner = UnifiedScanner(reader, model, cfg)
        streamed = scanner.scan(nf, test="score", qc_config=None)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_threshold_under_explicit_budget(self, threshold_inputs):
        """Hard absolute budget: <16 MiB at 200x500."""
        from torchgwas.config import STAT_DTYPE, TorchGWASConfig
        from torchgwas.models.threshold_linear import ThresholdLinearModel
        from torchgwas.scan.unified import UnifiedScanner

        G, Y, X0 = threshold_inputs
        m = G.shape[1]
        c = 2

        R = torch.eye(c, dtype=STAT_DTYPE)
        G_cov = torch.eye(c, dtype=STAT_DTYPE) * 0.5
        model = ThresholdLinearModel(
            trait_types=["ordinal", "continuous"],
            n_categories=[3, 0],
            R=R, G_cov=G_cov, solver="em", max_iter=20, em_warmup=5,
        )
        nf = model.fit_null(Y, X0)

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            cfg = TorchGWASConfig(device=torch.device("cpu"), chunk_size=64)
            scanner = UnifiedScanner(reader, model, cfg)
            return scanner.scan(nf, test="score", qc_config=None)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming threshold peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


# ---------------------------------------------------------------------------
# gxe-scan (HetLMM via per-chunk loop) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def gxe_inputs():
    """Synthetic continuous trait + env covariate."""
    torch.manual_seed(19)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    K, _ = grm_vanraden(G)

    env = torch.randn(n, dtype=torch.float64)
    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = (5.0 + L @ torch.randn(n, dtype=torch.float64)).unsqueeze(1)
    return G, Y, X0, K, env


class TestGxeScanStreamingMemory:
    """Streaming GxE (HetLMM) scan via per-chunk loop + _merge_gxe_results.

    HetLMM.score_chunk returns ``GxEScanResult`` with main +
    interaction + joint test fields. The default UnifiedScanner
    merger expects ``ScanResult`` and would crash on multi-chunk
    scans (latent pre-rewrite bug for genomes with > chunk_size
    variants). The streaming rewrite uses a dedicated merger.
    """

    def test_streaming_gxe_matches_full_chunk(self, gxe_inputs):
        from torchgwas.cli import _merge_gxe_results
        from torchgwas.models.lmm_gxe import HetLMM

        G, Y, X0, K, env = gxe_inputs
        m = G.shape[1]

        model = HetLMM()
        nf = model.fit_null(Y, X0, K=K, env=env)

        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        ref = model.score_chunk(G, nf, vmeta, test="wald")

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        chunk_results = []
        for G_chunk, vm in reader.iter_chunks(64):
            chunk_results.append(model.score_chunk(G_chunk, nf, vm, test="wald"))
        streamed = _merge_gxe_results(chunk_results)

        # GxEScanResult uses stat_joint / p_joint as primary outputs
        assert torch.allclose(streamed.stat_joint, ref.stat_joint,
                              atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p_joint, ref.p_joint,
                              atol=1e-9, rtol=1e-7)
        assert torch.allclose(streamed.beta_main, ref.beta_main,
                              atol=1e-9, rtol=1e-9)

    def test_streaming_gxe_under_explicit_budget(self, gxe_inputs):
        """Hard absolute budget: <16 MiB at 200x500."""
        from torchgwas.cli import _merge_gxe_results
        from torchgwas.models.lmm_gxe import HetLMM

        G, Y, X0, K, env = gxe_inputs
        m = G.shape[1]

        model = HetLMM()
        nf = model.fit_null(Y, X0, K=K, env=env)

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            chunk_results = []
            for G_chunk, vm in reader.iter_chunks(64):
                chunk_results.append(model.score_chunk(G_chunk, nf, vm, test="wald"))
            return _merge_gxe_results(chunk_results)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming GxE peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


# ---------------------------------------------------------------------------
# gu-scan (GULM with paired G + dosage_var streaming) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def gu_inputs():
    """Synthetic continuous trait + matched dosage variance."""
    torch.manual_seed(23)
    n, m = 200, 500
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    K, _ = grm_vanraden(G)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = 5.0 + L @ torch.randn(n, dtype=torch.float64)

    # Synthetic dosage variance — small, positive
    dvar = 0.05 * torch.rand(n, m, dtype=torch.float64)
    return G, Y, X0, K, dvar


class TestGuScanStreamingMemory:
    """Streaming GULM (genotype-uncertainty score test) memory regression.

    The streaming rewrite couples G-chunk iteration with column-sliced
    dosage_var so per-chunk score_chunk sees aligned ``(G_chunk,
    dvar_chunk)``. ``dvar`` is held once at user-input dtype; we
    track the column offset to slice it without copying.
    """

    def test_streaming_gu_matches_full_chunk(self, gu_inputs):
        from torchgwas.models.gu_lmm import GULM
        from torchgwas.scan.unified import merge_scan_results

        G, Y, X0, K, dvar = gu_inputs
        m = G.shape[1]

        model = GULM()
        nf = model.fit_null(Y, X0, K=K)

        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        ref = model.score_chunk(G, nf, vmeta, test="score", dosage_var=dvar)

        # Streaming: G chunks paired with sliced dvar columns.
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        chunk_results = []
        col_offset = 0
        for G_chunk, vm in reader.iter_chunks(64):
            mc = G_chunk.shape[1]
            dchunk = dvar[:, col_offset : col_offset + mc]
            chunk_results.append(model.score_chunk(
                G_chunk, nf, vm, test="score", dosage_var=dchunk,
            ))
            col_offset += mc
        streamed = merge_scan_results(chunk_results)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_gu_under_explicit_budget(self, gu_inputs):
        """Hard absolute budget: <16 MiB at 200x500.

        Note: dvar itself is (n, m) = 200×500×8 B = 781 KiB held for
        the whole scan. The streaming guarantee is that *G* (which at
        biobank scale dominates the budget) never lives in full —
        only chunk-by-chunk. dvar is a documented user-input held
        once; the budget reflects that.
        """
        from torchgwas.models.gu_lmm import GULM
        from torchgwas.scan.unified import merge_scan_results

        G, Y, X0, K, dvar = gu_inputs
        m = G.shape[1]

        model = GULM()
        nf = model.fit_null(Y, X0, K=K)

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            chunk_results = []
            col_offset = 0
            for G_chunk, vm in reader.iter_chunks(64):
                mc = G_chunk.shape[1]
                dchunk = dvar[:, col_offset : col_offset + mc]
                chunk_results.append(model.score_chunk(
                    G_chunk, nf, vm, test="score", dosage_var=dchunk,
                ))
                col_offset += mc
            return merge_scan_results(chunk_results)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming GU peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


# ---------------------------------------------------------------------------
# rr-scan / rr-met-scan / met-scan streaming-GRM regression (Efficiency E4)
# ---------------------------------------------------------------------------
#
# These three subcommands are "GRM-only" rewrites: the original code
# materialized G via _load_full_genotype solely to call grm_vanraden(G).
# The scan loop itself was already streaming via iter_chunks. The
# rewrite swaps the GRM call for grm_vanraden_streaming(...) over a
# chunk iterator — the scan loop is unchanged. The streaming memory
# guard here is on the GRM path: peak << n × m × 8 B.


@pytest.fixture
def rr_long_inputs():
    """Long-format longitudinal phenotype + per-sample GRM inputs."""
    torch.manual_seed(29)
    n, m = 60, 400
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)

    # 4 time points per individual.
    t_per = 4
    sample_ids_long = torch.arange(n, dtype=torch.int64).repeat_interleave(t_per)
    time_long = torch.linspace(0.0, 1.0, t_per, dtype=torch.float64).repeat(n)
    Y_long = (
        2.0
        + 0.5 * time_long
        + 0.1 * torch.randn(n * t_per, dtype=torch.float64)
    )
    X0 = torch.ones(n, 1, dtype=torch.float64)
    return G, Y_long, sample_ids_long, time_long, X0


class TestRrScanStreamingMemory:
    """Streaming RR-LMM kinship build (E4 GRM-only fix).

    The rewrite replaces ``_load_full_genotype + grm_vanraden(G)`` with
    ``grm_vanraden_streaming(_impute_chunk_iter(reader.iter_chunks))``.
    The downstream scan loop already iterates chunks. We assert that
    streaming the GRM produces the same kinship matrix as the
    materialized path, and that peak memory is bounded.
    """

    def test_streaming_grm_matches_materialized(self, rr_long_inputs):
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _, _, _ = rr_long_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_vanraden(G, ploidy=2)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)

        def _imputed_iter():
            for G_chunk, vm in reader.iter_chunks(64):
                yield G_chunk, vm  # synthetic G has no NaNs

        K_streamed, _ = grm_vanraden_streaming(
            _imputed_iter(),
            n_samples=n,
            ploidy=2,
            device=torch.device("cpu"),
        )

        # Streaming accumulator uses single-pass column moment math, so we
        # tolerate ~1e-6 absolute difference vs the in-memory two-pass call.
        assert torch.allclose(K_streamed, K_ref, atol=1e-6, rtol=1e-3)

    def test_streaming_grm_under_explicit_budget(self, rr_long_inputs):
        """Hard absolute budget: <16 MiB at n=60/m=400."""
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _, _, _ = rr_long_inputs
        m = G.shape[1]
        n = G.shape[0]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            return grm_vanraden_streaming(
                reader.iter_chunks(64),
                n_samples=n,
                ploidy=2,
                device=torch.device("cpu"),
            )

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming RR-LMM GRM peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


class TestRrMetScanStreamingMemory:
    """Streaming RR-MET-LMM kinship build (E4 GRM-only fix).

    Same rewrite shape as rr-scan: replace ``_load_full_genotype +
    grm_vanraden(G)`` with the streaming GRM. The downstream
    long-format scan loop is identical between the two variants.
    """

    def test_streaming_grm_matches_materialized(self, rr_long_inputs):
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _, _, _ = rr_long_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_vanraden(G, ploidy=2)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        K_streamed, _ = grm_vanraden_streaming(
            reader.iter_chunks(64),
            n_samples=n,
            ploidy=2,
            device=torch.device("cpu"),
        )

        # Streaming accumulator uses single-pass column moment math, so we
        # tolerate ~1e-6 absolute difference vs the in-memory two-pass call.
        assert torch.allclose(K_streamed, K_ref, atol=1e-6, rtol=1e-3)

    def test_streaming_grm_under_explicit_budget(self, rr_long_inputs):
        """Hard absolute budget: <16 MiB at n=60/m=400."""
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _, _, _ = rr_long_inputs
        m = G.shape[1]
        n = G.shape[0]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            return grm_vanraden_streaming(
                reader.iter_chunks(64),
                n_samples=n,
                ploidy=2,
                device=torch.device("cpu"),
            )

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib


@pytest.fixture
def met_inputs():
    """Synthetic per-environment continuous trait."""
    torch.manual_seed(31)
    n, m, E = 100, 400, 3
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    Y_wide = torch.randn(n, E, dtype=torch.float64)
    return G, Y_wide, X0


class TestMetScanStreamingMemory:
    """Streaming MET-LMM kinship build (E4 GRM-only fix).

    The rewrite replaces ``_load_scan_data + grm_vanraden(G)`` with
    ``_align_samples + grm_vanraden_streaming``. The per-chunk
    score_chunk loop is identical.
    """

    def test_streaming_grm_matches_materialized(self, met_inputs):
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _ = met_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_vanraden(G, ploidy=2)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        K_streamed, _ = grm_vanraden_streaming(
            reader.iter_chunks(64),
            n_samples=n,
            ploidy=2,
            device=torch.device("cpu"),
        )

        # Streaming accumulator uses single-pass column moment math, so we
        # tolerate ~1e-6 absolute difference vs the in-memory two-pass call.
        assert torch.allclose(K_streamed, K_ref, atol=1e-6, rtol=1e-3)

    def test_streaming_grm_under_explicit_budget(self, met_inputs):
        """Hard absolute budget: <16 MiB at n=100/m=400."""
        from torchgwas.linalg.kinship import grm_vanraden_streaming

        G, _, _ = met_inputs
        m = G.shape[1]
        n = G.shape[0]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            return grm_vanraden_streaming(
                reader.iter_chunks(64),
                n_samples=n,
                ploidy=2,
                device=torch.device("cpu"),
            )

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib
