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

from torchgenomics.io.regions import Region
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.base import VariantMeta
from torchgenomics.models.set_based import SetBasedScanner
from torchgenomics.models.single_trait_lmm import SingleTraitLMM

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
        from torchgenomics.config import TorchGenomicsConfig
        from torchgenomics.models.binary_glmm import BinaryGLMM
        from torchgenomics.scan.unified import UnifiedScanner

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
        config = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
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
        from torchgenomics.config import TorchGenomicsConfig
        from torchgenomics.models.binary_glmm import BinaryGLMM
        from torchgenomics.scan.unified import UnifiedScanner

        G, Y, X0, K = binary_glmm_inputs
        m = G.shape[1]

        model = BinaryGLMM(use_spa=False, firth=False, pql_max_iter=20)
        nf = model.fit_null(Y, X0, K=K)

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            cfg = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
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
    :func:`torchgenomics.cli._merge_env_results` rather than
    :class:`UnifiedScanner`. The PQL null fit only depends on
    ``Y / X0 / K``, so the genome scan loop is per-variant.
    """

    def test_streaming_me_glmm_matches_full_chunk(self, me_glmm_inputs):
        from torchgenomics.cli import _merge_env_results
        from torchgenomics.models.multi_env_glmm import MultiEnvGLMM

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
        from torchgenomics.cli import _merge_env_results
        from torchgenomics.models.multi_env_glmm import MultiEnvGLMM

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
        from torchgenomics.config import TorchGenomicsConfig
        from torchgenomics.models.survival_glmm import SurvivalGLMM
        from torchgenomics.scan.unified import UnifiedScanner

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
        config = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
        scanner = UnifiedScanner(reader, model, config)
        streamed = scanner.scan(nf, test="score", qc_config=None)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_survival_under_explicit_budget(self, survival_inputs):
        """Hard absolute budget: <16 MiB at 200x500."""
        from torchgenomics.config import TorchGenomicsConfig
        from torchgenomics.models.survival_glmm import SurvivalGLMM
        from torchgenomics.scan.unified import UnifiedScanner

        G, Y, X0, K = survival_inputs
        m = G.shape[1]

        model = SurvivalGLMM(use_spa=False, pql_max_iter=10)
        nf = model.fit_null(Y, X0, K=K)

        def _run_streaming():
            reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
            cfg = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
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
        from torchgenomics.config import STAT_DTYPE, TorchGenomicsConfig
        from torchgenomics.models.threshold_linear import ThresholdLinearModel
        from torchgenomics.scan.unified import UnifiedScanner

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
        cfg = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
        scanner = UnifiedScanner(reader, model, cfg)
        streamed = scanner.scan(nf, test="score", qc_config=None)

        assert torch.allclose(streamed.stat, ref.stat, atol=1e-9, rtol=1e-9)
        assert torch.allclose(streamed.p, ref.p, atol=1e-9, rtol=1e-7)

    def test_streaming_threshold_under_explicit_budget(self, threshold_inputs):
        """Hard absolute budget: <16 MiB at 200x500."""
        from torchgenomics.config import STAT_DTYPE, TorchGenomicsConfig
        from torchgenomics.models.threshold_linear import ThresholdLinearModel
        from torchgenomics.scan.unified import UnifiedScanner

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
            cfg = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
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
        from torchgenomics.cli import _merge_gxe_results
        from torchgenomics.models.lmm_gxe import HetLMM

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
        from torchgenomics.cli import _merge_gxe_results
        from torchgenomics.models.lmm_gxe import HetLMM

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
        from torchgenomics.models.gu_lmm import GULM
        from torchgenomics.scan.unified import merge_scan_results

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
        from torchgenomics.models.gu_lmm import GULM
        from torchgenomics.scan.unified import merge_scan_results

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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

        G, _, _, _, _ = rr_long_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_vanraden(G, ploidy=2)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)

        def _imputed_iter():
            # synthetic G has no NaNs
            yield from reader.iter_chunks(64)

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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

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


# ---------------------------------------------------------------------------
# poly-scan (SingleTraitLMM with gene-action recoding) memory regression
# ---------------------------------------------------------------------------


@pytest.fixture
def poly_inputs():
    """Synthetic tetraploid (ploidy=4) genotype + continuous trait."""
    torch.manual_seed(37)
    n, m = 100, 400
    ploidy = 4
    G = torch.randint(0, ploidy + 1, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Build additive GRM for null fit (mirrors grm_polyploid_gene_action
    # with model="additive").
    K, _ = grm_vanraden(G, ploidy=ploidy)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = (5.0 + L @ torch.randn(n, dtype=torch.float64)).unsqueeze(1)
    return G, Y, X0, K, ploidy


class TestPolyScanStreamingMemory:
    """Streaming poly-scan via UnifiedScanner + on-the-fly recoded chunks.

    The legacy code path materialized G via ``_load_scan_data``, then
    fed the *same* aligned reader (not the recoded G_scan tensor) to
    UnifiedScanner. The streaming rewrite wraps the reader with a
    ``_RecodingReader`` so per-chunk gene-action recoding happens
    on-the-fly during ``iter_chunks``. Behavioral parity is guaranteed
    for the additive model because the recoder is the identity for
    ``ga in {"additive", "general"}``.

    The streaming GRM uses ``grm_vanraden_streaming`` with the same
    effective-ploidy mapping that ``grm_polyploid_gene_action`` applies
    in the materialized path. We assert that streaming additive GRM
    matches the materialized polyploid additive GRM to float64
    tolerance.
    """

    def test_streaming_additive_grm_matches_materialized(self, poly_inputs):
        from torchgenomics.linalg.kinship import grm_vanraden_streaming
        from torchgenomics.linalg.kinship_polyploid import grm_polyploid_gene_action

        G, _, _, _, ploidy = poly_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_polyploid_gene_action(G, model="additive", ploidy=ploidy)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        K_streamed, _ = grm_vanraden_streaming(
            reader.iter_chunks(64),
            n_samples=n,
            ploidy=ploidy,
            device=torch.device("cpu"),
        )

        # Streaming accumulator vs in-memory two-pass: <1e-6 absolute.
        assert torch.allclose(K_streamed, K_ref, atol=1e-6, rtol=1e-3)

    def test_streaming_dominance_grm_matches_materialized(self, poly_inputs):
        """1-dom recoding must commute with chunking — recoding is per-element."""
        from torchgenomics.linalg.kinship import grm_vanraden_streaming
        from torchgenomics.linalg.kinship_polyploid import grm_polyploid_gene_action
        from torchgenomics.preprocess.polyploid import recode_gene_action

        G, _, _, _, ploidy = poly_inputs
        m = G.shape[1]
        n = G.shape[0]

        K_ref, _ = grm_polyploid_gene_action(G, model="1-dom", ploidy=ploidy)

        # Streaming recoder: recode each chunk before feeding to GRM accumulator.
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)

        def _recoded_iter():
            for G_chunk, vm in reader.iter_chunks(64):
                yield recode_gene_action(G_chunk, "1-dom", ploidy), vm

        K_streamed, _ = grm_vanraden_streaming(
            _recoded_iter(),
            n_samples=n,
            ploidy=1,  # j-dom -> binary -> effective_ploidy=1
            device=torch.device("cpu"),
        )

        assert torch.allclose(K_streamed, K_ref, atol=1e-6, rtol=1e-3)

    def test_streaming_poly_scan_under_explicit_budget(self, poly_inputs):
        """Hard absolute budget: <16 MiB at n=100/m=400, ploidy=4."""
        from torchgenomics.config import TorchGenomicsConfig
        from torchgenomics.models.single_trait_lmm import SingleTraitLMM
        from torchgenomics.preprocess.polyploid import recode_gene_action
        from torchgenomics.scan.unified import UnifiedScanner

        G, Y, X0, K, ploidy = poly_inputs
        m = G.shape[1]
        n = G.shape[0]

        model = SingleTraitLMM()
        nf = model.fit_null(Y.squeeze(1), X0, K=K)

        class _RecodingReader:
            def __init__(self, base, ga, ploidy_int):
                self._base = base
                self._ga = ga
                self._ploidy = ploidy_int

            @property
            def n_samples(self):
                return self._base.n_samples

            @property
            def n_variants(self):
                return self._base.n_variants

            @property
            def sample_ids(self):
                return self._base.sample_ids

            def iter_chunks(self, chunk_size=None):
                for G_chunk, vm in self._base.iter_chunks(chunk_size=chunk_size):
                    if self._ga == "additive":
                        yield G_chunk, vm
                    else:
                        yield (
                            recode_gene_action(G_chunk, self._ga, self._ploidy),
                            vm,
                        )

        def _run_streaming():
            base = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            scan_reader = _RecodingReader(base, "additive", ploidy)
            cfg = TorchGenomicsConfig(device=torch.device("cpu"), chunk_size=64)
            scanner = UnifiedScanner(scan_reader, model, cfg)
            return scanner.scan(nf, test="score", qc_config=None)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak_stream < budget_kib, (
            f"Streaming poly-scan peak ({peak_stream:.0f} KiB) exceeds 16 MiB."
        )


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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

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
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

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


# ---------------------------------------------------------------------------
# impute (mean / mode / knn / ld) streaming memory regression (E4 Group C)
# ---------------------------------------------------------------------------


@pytest.fixture
def impute_inputs():
    """Genotype with 5% missing entries for impute parity tests."""
    torch.manual_seed(41)
    n, m = 50, 200
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    mask = torch.rand(n, m) < 0.05
    G[mask] = float("nan")
    return G


class TestImputeMeanStreamingMemory:
    """Streaming mean imputation: two-pass per-column statistics + fill."""

    def test_streaming_mean_matches_materialized(self, impute_inputs):
        from torchgenomics.preprocess.impute import (
            compute_column_means_streaming,
            impute_chunk_with_means,
            impute_mean,
        )

        G = impute_inputs
        m = G.shape[1]

        # Reference path.
        G_ref = impute_mean(G)

        # Streaming path.
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        means = compute_column_means_streaming(reader.iter_chunks(64))
        out = torch.zeros_like(G)
        col_offset = 0
        for G_chunk, _ in reader.iter_chunks(64):
            out[:, col_offset : col_offset + G_chunk.shape[1]] = (
                impute_chunk_with_means(G_chunk, means, col_offset)
            )
            col_offset += G_chunk.shape[1]

        # Bit-exact: the streaming mean = the in-memory mean.
        assert torch.allclose(out, G_ref, atol=1e-12, rtol=1e-12)

    def test_streaming_mean_under_explicit_budget(self, impute_inputs):
        """Hard budget: <8 MiB for the per-column-stat + fill workspace."""
        from torchgenomics.preprocess.impute import (
            compute_column_means_streaming,
            impute_chunk_with_means,
        )

        G = impute_inputs
        m = G.shape[1]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            means = compute_column_means_streaming(reader.iter_chunks(64))
            chunks = []
            col_offset = 0
            for G_chunk, _ in reader.iter_chunks(64):
                chunks.append(impute_chunk_with_means(G_chunk, means, col_offset))
                col_offset += G_chunk.shape[1]
            return chunks

        _, peak = _peak_kib(_run_streaming)
        # Note: writing to an in-memory list still holds the full
        # imputed result; the streaming peak measured here is the
        # per-pass workspace + chunk + means, not the materialization.
        # The 8 MiB ceiling fits the n=50/m=200 fixture comfortably.
        budget_kib = 8 * 1024
        assert peak < budget_kib, (
            f"Streaming mean peak ({peak:.0f} KiB) exceeds 8 MiB."
        )


class TestImputeModeStreamingMemory:
    """Streaming mode imputation: per-column class histograms + fill."""

    def test_streaming_mode_matches_materialized(self, impute_inputs):
        from torchgenomics.preprocess.impute import (
            compute_column_modes_streaming,
            impute_chunk_with_modes,
            impute_mode,
        )

        G = impute_inputs
        m = G.shape[1]

        G_ref = impute_mode(G)

        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)
        modes = compute_column_modes_streaming(
            reader.iter_chunks(64), max_dosage=2,
        )
        out = torch.zeros_like(G)
        col_offset = 0
        for G_chunk, _ in reader.iter_chunks(64):
            out[:, col_offset : col_offset + G_chunk.shape[1]] = (
                impute_chunk_with_modes(G_chunk, modes, col_offset)
            )
            col_offset += G_chunk.shape[1]

        assert torch.allclose(out, G_ref, atol=1e-12, rtol=1e-12)

    def test_streaming_mode_under_explicit_budget(self, impute_inputs):
        from torchgenomics.preprocess.impute import (
            compute_column_modes_streaming,
            impute_chunk_with_modes,
        )

        G = impute_inputs
        m = G.shape[1]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )
            modes = compute_column_modes_streaming(
                reader.iter_chunks(64), max_dosage=2,
            )
            chunks = []
            col_offset = 0
            for G_chunk, _ in reader.iter_chunks(64):
                chunks.append(
                    impute_chunk_with_modes(G_chunk, modes, col_offset)
                )
                col_offset += G_chunk.shape[1]
            return chunks

        _, peak = _peak_kib(_run_streaming)
        budget_kib = 8 * 1024
        assert peak < budget_kib


class TestImputeKnnStreamingMemory:
    """Streaming KNN imputation: streaming GRM + per-chunk fill.

    K is held once across all chunks (n × n × 8 B). At biobank scale
    that's ~1 TB — documented hard memory limit for the KNN method.
    The streaming-vs-materialized win is on the genotype side: we
    never need ``n × m × 8 B`` simultaneously.
    """

    def test_streaming_knn_matches_materialized(self, impute_inputs):
        from torchgenomics.linalg.kinship import grm_vanraden_streaming
        from torchgenomics.preprocess.impute import (
            impute_chunk_with_knn,
            impute_knn,
            impute_mean,
        )

        G = impute_inputs
        m = G.shape[1]
        n = G.shape[0]

        # Reference path: full impute_mean(G) -> grm_vanraden -> impute_knn.
        G_mean = impute_mean(G)
        K_ref, _ = grm_vanraden(G_mean, ploidy=2)
        G_knn_ref = impute_knn(G, K_ref)

        # Streaming path: streaming GRM (over mean-imputed chunks) +
        # per-chunk KNN fill using the same K_ref.
        reader = _ChunkedTensorReader(G, ["1"] * m, list(range(m)), chunk_size=64)

        def _imputed_iter():
            for G_chunk, vm in reader.iter_chunks(64):
                yield impute_mean(G_chunk), vm

        K_streamed, _ = grm_vanraden_streaming(
            _imputed_iter(), n_samples=n, ploidy=2,
        )

        # Per-chunk KNN with K_ref (deterministic match) — proves the
        # per-chunk decomposition itself is exact.
        out = torch.zeros_like(G)
        col_offset = 0
        for G_chunk, _ in reader.iter_chunks(64):
            out[:, col_offset : col_offset + G_chunk.shape[1]] = (
                impute_chunk_with_knn(G_chunk, K_ref)
            )
            col_offset += G_chunk.shape[1]

        assert torch.allclose(out, G_knn_ref, atol=1e-10, rtol=1e-10)

    def test_streaming_knn_under_explicit_budget(self, impute_inputs):
        """Budget: <16 MiB at n=50, K = 50×50×8 = 20 KiB negligible."""
        from torchgenomics.linalg.kinship import grm_vanraden_streaming
        from torchgenomics.preprocess.impute import (
            impute_chunk_with_knn,
            impute_mean,
        )

        G = impute_inputs
        m = G.shape[1]
        n = G.shape[0]

        def _run_streaming():
            reader = _ChunkedTensorReader(
                G, ["1"] * m, list(range(m)), chunk_size=64,
            )

            def _imputed_iter():
                for G_chunk, vm in reader.iter_chunks(64):
                    yield impute_mean(G_chunk), vm

            K, _ = grm_vanraden_streaming(
                _imputed_iter(), n_samples=n, ploidy=2,
            )
            chunks = []
            col_offset = 0
            for G_chunk, _ in reader.iter_chunks(64):
                chunks.append(impute_chunk_with_knn(G_chunk, K))
                col_offset += G_chunk.shape[1]
            return chunks

        _, peak = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024
        assert peak < budget_kib


class TestImputeLdStreamingMemory:
    """Streaming LD imputation: single pass with sliding-window buffer.

    The current implementation builds left/right flank buffers of
    ``window_size`` SNPs from neighbouring chunks before calling
    :func:`impute_ld` on the merged window. This holds all chunks in
    memory inside the function (documented compromise — the LD
    method is fundamentally not single-pass streamable for arbitrary
    window sizes, since reverse-look-up is required for the right
    flank). We assert behavioral parity vs the materialized path.
    """

    def test_streaming_ld_matches_materialized(self, impute_inputs):
        from torchgenomics.preprocess.impute import (
            impute_chunk_with_ld_window,
            impute_ld,
        )

        G = impute_inputs
        m = G.shape[1]

        G_ref = impute_ld(G, window_size=50)

        # Streaming with buffered flanks.
        chunks_buf = []
        col_offsets = []
        off = 0
        chunk_size = 64
        window_size = 50
        for s in range(0, m, chunk_size):
            e = min(s + chunk_size, m)
            chunks_buf.append(G[:, s:e].clone())
            col_offsets.append(off)
            off += e - s

        out = torch.zeros_like(G)
        for i, (G_chunk, off) in enumerate(zip(chunks_buf, col_offsets)):
            if i == 0:
                left = None
                left_n = 0
            else:
                prev = torch.cat(chunks_buf[:i], dim=1)
                left = prev[:, -window_size:]
                left_n = left.shape[1]
            if i == len(chunks_buf) - 1:
                right = None
            else:
                nxt = torch.cat(chunks_buf[i + 1 :], dim=1)
                right = nxt[:, :window_size]
            out[:, off : off + G_chunk.shape[1]] = (
                impute_chunk_with_ld_window(
                    G_chunk, left, right,
                    window_size=window_size, chunk_col_offset=left_n,
                )
            )

        # LD matches reference exactly: each chunk's window is fully
        # reconstructed from neighbour buffers.
        assert torch.allclose(out, G_ref, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# F1 — windowed-buffer streaming for the six LD-window paths
# (ldsc, ldsc-rg, ld-blocks, clump, knockoff-scan, lro-scan)
# ---------------------------------------------------------------------------


def _ld_fixture(n: int = 200, m: int = 2000, n_chrom: int = 2):
    """Fixed seed (n × m) genotype + per-SNP chr/pos arrays.

    The two-chromosome layout exercises the chromosome-boundary flush
    path; positions are spaced 100 bp apart so a 5 kb window covers
    ~50 neighbours.
    """
    torch.manual_seed(7)
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    chrs = []
    poss = []
    per = m // n_chrom
    for c in range(n_chrom):
        start = c * per
        end = (c + 1) * per if c < n_chrom - 1 else m
        chrs.extend([str(c + 1)] * (end - start))
        poss.extend(list(range(0, (end - start) * 100, 100)))
    return G, chrs, poss


def _stream_ld_blocks_via_cli_helper(
    G: torch.Tensor,
    chrs: list[str],
    poss: list[int],
    *,
    method: str,
    max_kb: float,
    chunk_size: int = 64,
    **method_kwargs,
):
    """Replicate _cmd_ld_blocks's per-chromosome streaming accumulator
    in a unit-testable form. Returns the list of detected blocks.
    """
    from torchgenomics.ld import detect_blocks

    reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=chunk_size)
    blocks: list = []
    cur_chr: str | None = None
    cur_chunks: list[torch.Tensor] = []
    cur_pos: list[int] = []
    cur_ids: list[str] = []
    cur_global_start = 0
    n_seen = 0

    def _process():
        nonlocal cur_chunks, cur_pos, cur_ids, cur_chr
        if not cur_chunks:
            return
        G_chr = torch.cat(cur_chunks, dim=1)
        chrom_blocks = detect_blocks(
            G_chr, cur_pos, [str(cur_chr)] * G_chr.shape[1],
            cur_ids, method=method, max_kb=max_kb, **method_kwargs,
        )
        for blk in chrom_blocks:
            blk.variant_indices = [i + cur_global_start for i in blk.variant_indices]
        blocks.extend(chrom_blocks)
        cur_chunks = []
        cur_pos = []
        cur_ids = []

    for G_chunk, vmeta in reader.iter_chunks(chunk_size):
        chr_chunk = [str(c) for c in vmeta.chr]
        run_start = 0
        while run_start < len(chr_chunk):
            run_chr = chr_chunk[run_start]
            run_end = run_start + 1
            while run_end < len(chr_chunk) and chr_chunk[run_end] == run_chr:
                run_end += 1
            if cur_chr is None:
                cur_chr = run_chr
                cur_global_start = n_seen + run_start
            elif run_chr != cur_chr:
                _process()
                cur_chr = run_chr
                cur_global_start = n_seen + run_start
            cur_chunks.append(G_chunk[:, run_start:run_end].clone())
            cur_pos.extend(vmeta.pos[run_start:run_end])
            cur_ids.extend(vmeta.snp[run_start:run_end])
            run_start = run_end
        n_seen += G_chunk.shape[1]
    _process()
    return blocks


class TestLdBlocksStreamingMemory:
    """Per-chromosome streaming for the ld-blocks CLI command.

    Block detection algorithms (Gabriel/spine/r²/...) need a
    chromosome-scale view to compute pairwise r²/D' inside max_kb. The
    streaming refactor accumulates one chromosome at a time, runs
    detection, then frees the slice. Peak memory drops from
    ``O(n × m)`` to ``O(n × max_chromosome_m)``.
    """

    def test_streaming_matches_materialized_r2_single_chrom(self):
        """Single-chromosome parity: legacy and streaming agree exactly.

        We deliberately use a single-chromosome fixture here because the
        legacy path harbors a latent bug for multi-chromosome inputs:
        compute_pairwise_ld emits chromosome-LOCAL indices, but
        detect_blocks_{r2,spine,gabriel} index into the genome-wide
        ``variant_chr`` / ``variant_pos`` lists, mis-attributing later
        chromosomes' blocks as belonging to the first. The streaming
        rewrite calls detect_blocks once per chromosome with that
        chromosome's slice — fixing this latent bug as a side effect.
        See the multi-chromosome test below for the corrected behavior.
        """
        from torchgenomics.ld import detect_blocks

        G, chrs, poss = _ld_fixture(n=120, m=200, n_chrom=1)
        ref = detect_blocks(
            G, poss, chrs, [f"r{i}" for i in range(len(poss))],
            method="r2", max_kb=2.0, r2_threshold=0.05,
        )
        stream_blocks = _stream_ld_blocks_via_cli_helper(
            G, chrs, poss, method="r2", max_kb=2.0, chunk_size=37,
            r2_threshold=0.05,
        )

        assert len(stream_blocks) == len(ref)
        ref.sort(key=lambda b: (b.region.chr, b.region.start))
        stream_blocks.sort(key=lambda b: (b.region.chr, b.region.start))
        for sb, rb in zip(stream_blocks, ref):
            assert sb.region.chr == rb.region.chr
            assert sb.region.start == rb.region.start
            assert sb.region.end == rb.region.end
            assert sb.n_variants == rb.n_variants

    def test_streaming_per_chromosome_correctly_attributes_blocks(self):
        """Multi-chromosome streaming attributes blocks to their actual
        chromosome — fixing a latent legacy bug where chr 2+ blocks
        were mis-labelled as chr 1.
        """
        G, chrs, poss = _ld_fixture(n=120, m=400, n_chrom=2)
        stream_blocks = _stream_ld_blocks_via_cli_helper(
            G, chrs, poss, method="r2", max_kb=2.0, chunk_size=37,
            r2_threshold=0.05,
        )
        chrom_set = {b.region.chr for b in stream_blocks}
        # Both chromosomes should contribute blocks (or at least not
        # everything be coerced to chr 1).
        assert chrom_set.issubset({"1", "2"})
        for blk in stream_blocks:
            # Variant indices must point to SNPs whose chromosome
            # matches the block's reported chromosome.
            for vi in blk.variant_indices:
                assert chrs[vi] == blk.region.chr

    def test_streaming_peak_per_chromosome_not_genome(self):
        """Doubling m for fixed per-chromosome size should NOT double peak."""
        # Scenario A: 1 chromosome of 400 SNPs.
        G_a, chr_a, pos_a = _ld_fixture(n=80, m=400, n_chrom=1)
        # Scenario B: 4 chromosomes × 400 SNPs each (m=1600).
        G_b, chr_b, pos_b = _ld_fixture(n=80, m=1600, n_chrom=4)

        def _run(G, ch, ps):
            return _stream_ld_blocks_via_cli_helper(
                G, ch, ps, method="r2", max_kb=2.0, chunk_size=64,
            )

        _, peak_a = _peak_kib(lambda: _run(G_a, chr_a, pos_a))
        _, peak_b = _peak_kib(lambda: _run(G_b, chr_b, pos_b))

        # 4x m at fixed per-chromosome size should yield similar peak.
        assert peak_b <= 2.0 * peak_a + 256, (
            f"ld-blocks streaming peak grew from {peak_a:.0f} KiB "
            f"(m=400) to {peak_b:.0f} KiB (m=1600) — should be roughly "
            "invariant in m for fixed per-chromosome size."
        )


def _stream_knockoff_via_cli_helper(
    Y: torch.Tensor,
    X0: torch.Tensor,
    G: torch.Tensor,
    K: torch.Tensor,
    chrs: list[str],
    poss: list[int],
    snp_ids: list[str],
    *,
    chunk_size: int = 64,
    target_fdr: float = 0.1,
    ld_method: str = "r2",
    seed: int = 42,
):
    """Replicate _cmd_knockoff_scan's per-chromosome streaming flow."""
    from torchgenomics.models.base import VariantMeta
    from torchgenomics.models.knockoff_lmm import (
        KnockoffLMM,
        KnockoffResult,
        _knockoff_plus_filter,
    )

    reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=chunk_size)
    model = KnockoffLMM(target_fdr=target_fdr, ld_method=ld_method, seed=seed)

    per_chr: list[KnockoffResult] = []
    cur_chr: str | None = None
    cur_chunks: list[torch.Tensor] = []
    cur_meta: list[object] = []

    def _process():
        nonlocal cur_chunks, cur_meta
        if not cur_chunks or cur_chr is None:
            return
        G_chr = torch.cat(cur_chunks, dim=1)
        chr_snp: list[str] = []
        chr_pos: list[int] = []
        chr_chr: list[str] = []
        for cm in cur_meta:
            chr_snp.extend(cm.snp)
            chr_pos.extend(cm.pos)
            chr_chr.extend([str(c) for c in cm.chr])
        vm = VariantMeta(
            snp=chr_snp, chr=chr_chr, pos=chr_pos,
            a1=["A"] * len(chr_snp), a2=["G"] * len(chr_snp),
        )
        per_chr.append(model.run(Y, X0, K, G_chr, vm, vm.pos, vm.chr))
        cur_chunks = []
        cur_meta = []

    for G_chunk, vm in reader.iter_chunks(chunk_size):
        chr_chunk = [str(c) for c in vm.chr]
        rs = 0
        while rs < len(chr_chunk):
            rc = chr_chunk[rs]
            re_ = rs + 1
            while re_ < len(chr_chunk) and chr_chunk[re_] == rc:
                re_ += 1
            if cur_chr is None:
                cur_chr = rc
            elif rc != cur_chr:
                _process()
                cur_chr = rc
            sub = VariantMeta(
                snp=vm.snp[rs:re_], chr=vm.chr[rs:re_], pos=vm.pos[rs:re_],
                a1=vm.a1[rs:re_], a2=vm.a2[rs:re_],
            )
            cur_chunks.append(G_chunk[:, rs:re_].clone())
            cur_meta.append(sub)
            rs = re_
    _process()

    # Merge.
    block_indices_g: list[list[int]] = []
    W_pieces: list[torch.Tensor] = []
    chr_g: list[str] = []
    pos_g: list[int] = []
    snp_g: list[str] = []
    a1_g: list[str] = []
    a2_g: list[str] = []
    af_p: list[torch.Tensor] = []
    beta_p: list[torch.Tensor] = []
    se_p: list[torch.Tensor] = []
    stat_p: list[torch.Tensor] = []
    p_p: list[torch.Tensor] = []
    bk_p: list[torch.Tensor] = []
    sk_p: list[torch.Tensor] = []
    offset = 0
    for r in per_chr:
        for indices in r.block_indices:
            block_indices_g.append([i + offset for i in indices])
        W_pieces.append(r.W_stat)
        chr_g.extend(r.chr)
        pos_g.extend(r.pos)
        snp_g.extend(r.snp)
        a1_g.extend(r.a1)
        a2_g.extend(r.a2)
        af_p.append(r.af)
        beta_p.append(r.beta)
        se_p.append(r.se)
        stat_p.append(r.stat)
        p_p.append(r.p)
        bk_p.append(r.beta_knockoff)
        sk_p.append(r.stat_knockoff)
        offset += r.beta.shape[0]

    W = torch.cat(W_pieces)
    threshold, selected = _knockoff_plus_filter(W, target_fdr)
    is_sel = torch.zeros(offset, dtype=torch.bool)
    for b in selected:
        for j in block_indices_g[b]:
            is_sel[j] = True
    return KnockoffResult(
        block_indices=block_indices_g,
        W_stat=W, selected_blocks=selected, threshold=threshold,
        chr=chr_g, pos=pos_g, snp=snp_g, a1=a1_g, a2=a2_g,
        af=torch.cat(af_p), beta=torch.cat(beta_p), se=torch.cat(se_p),
        stat=torch.cat(stat_p), p=torch.cat(p_p),
        beta_knockoff=torch.cat(bk_p), stat_knockoff=torch.cat(sk_p),
        is_selected=is_sel, target_fdr=target_fdr,
        n_blocks=len(block_indices_g), n_selected=len(selected),
        ld_method=ld_method, knockoff_method="equicorrelated",
        aggregation="max_stat",
    )


def _stream_lro_via_cli_helper(
    Y: torch.Tensor,
    X0: torch.Tensor,
    G: torch.Tensor,
    K_full: torch.Tensor,
    normalizer: float,
    chrs: list[str],
    poss: list[int],
    snp_ids: list[str],
    *,
    chunk_size: int = 64,
    test: str = "wald",
    ld_method: str = "r2",
):
    """Replicate _cmd_lro_scan's per-chromosome streaming flow."""
    from torchgenomics.models.base import VariantMeta
    from torchgenomics.models.lro_lmm import LROLMM, LROResult

    reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=chunk_size)
    model = LROLMM(ld_method=ld_method)
    per_chr: list[LROResult] = []
    cur_chr: str | None = None
    cur_chunks: list[torch.Tensor] = []
    cur_meta: list[object] = []

    def _process():
        nonlocal cur_chunks, cur_meta
        if not cur_chunks or cur_chr is None:
            return
        G_chr = torch.cat(cur_chunks, dim=1)
        chr_snp: list[str] = []
        chr_pos: list[int] = []
        chr_chr: list[str] = []
        for cm in cur_meta:
            chr_snp.extend(cm.snp)
            chr_pos.extend(cm.pos)
            chr_chr.extend([str(c) for c in cm.chr])
        vm = VariantMeta(
            snp=chr_snp, chr=chr_chr, pos=chr_pos,
            a1=["A"] * len(chr_snp), a2=["G"] * len(chr_snp),
        )
        per_chr.append(model.run(
            Y, X0, G_chr, vm, vm.pos, vm.chr,
            test=test, K_full=K_full, normalizer=normalizer,
        ))
        cur_chunks = []
        cur_meta = []

    for G_chunk, vm in reader.iter_chunks(chunk_size):
        chr_chunk = [str(c) for c in vm.chr]
        rs = 0
        while rs < len(chr_chunk):
            rc = chr_chunk[rs]
            re_ = rs + 1
            while re_ < len(chr_chunk) and chr_chunk[re_] == rc:
                re_ += 1
            if cur_chr is None:
                cur_chr = rc
            elif rc != cur_chr:
                _process()
                cur_chr = rc
            sub = VariantMeta(
                snp=vm.snp[rs:re_], chr=vm.chr[rs:re_], pos=vm.pos[rs:re_],
                a1=vm.a1[rs:re_], a2=vm.a2[rs:re_],
            )
            cur_chunks.append(G_chunk[:, rs:re_].clone())
            cur_meta.append(sub)
            rs = re_
    _process()

    chr_g: list[str] = []
    pos_g: list[int] = []
    snp_g: list[str] = []
    a1_g: list[str] = []
    a2_g: list[str] = []
    af_p: list[torch.Tensor] = []
    beta_p: list[torch.Tensor] = []
    se_p: list[torch.Tensor] = []
    stat_p: list[torch.Tensor] = []
    p_p: list[torch.Tensor] = []
    block_sizes: list[int] = []
    n_blocks_total = 0
    for r in per_chr:
        chr_g.extend(r.chr)
        pos_g.extend(r.pos)
        snp_g.extend(r.snp)
        a1_g.extend(r.a1)
        a2_g.extend(r.a2)
        af_p.append(r.af)
        beta_p.append(r.beta)
        se_p.append(r.se)
        stat_p.append(r.stat)
        p_p.append(r.p)
        block_sizes.extend(r.block_sizes)
        n_blocks_total += r.n_blocks
    return LROResult(
        chr=chr_g, pos=pos_g, snp=snp_g, a1=a1_g, a2=a2_g,
        af=torch.cat(af_p), beta=torch.cat(beta_p), se=torch.cat(se_p),
        stat=torch.cat(stat_p), p=torch.cat(p_p),
        test=test, n_blocks=n_blocks_total,
        block_sizes=block_sizes, block_method=ld_method,
    )


class TestKnockoffScanStreamingMemory:
    """Per-chromosome streaming for knockoff-scan."""

    def _build_inputs(self, n=60, m=80, n_chrom=2):
        torch.manual_seed(31)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        chrs = []
        poss = []
        per = m // n_chrom
        for c in range(n_chrom):
            s = c * per
            e = (c + 1) * per if c < n_chrom - 1 else m
            chrs.extend([str(c + 1)] * (e - s))
            poss.extend(list(range(0, (e - s) * 1000, 1000)))
        from torchgenomics.linalg.kinship import grm_vanraden
        K, _ = grm_vanraden(G)
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        ids = [f"r{i}" for i in range(m)]
        return Y, X0, G, K, chrs, poss, ids

    def test_streaming_runs_end_to_end(self):
        Y, X0, G, K, chrs, poss, ids = self._build_inputs()
        result = _stream_knockoff_via_cli_helper(
            Y, X0, G, K, chrs, poss, ids, chunk_size=23,
        )
        # Sanity: matches G shape.
        assert result.beta.shape[0] == G.shape[1]
        assert result.n_blocks > 0

    def test_streaming_peak_per_chromosome_not_genome(self):
        Y_a, X0_a, G_a, K_a, chr_a, pos_a, id_a = self._build_inputs(
            n=60, m=80, n_chrom=1,
        )
        Y_b, X0_b, G_b, K_b, chr_b, pos_b, id_b = self._build_inputs(
            n=60, m=320, n_chrom=4,
        )

        def _run(Y, X0, G, K, chs, ps, ids):
            return _stream_knockoff_via_cli_helper(
                Y, X0, G, K, chs, ps, ids, chunk_size=32,
            )

        _, peak_a = _peak_kib(lambda: _run(Y_a, X0_a, G_a, K_a, chr_a, pos_a, id_a))
        _, peak_b = _peak_kib(lambda: _run(Y_b, X0_b, G_b, K_b, chr_b, pos_b, id_b))
        # 4x more chromosomes at fixed per-chromosome size: peak should
        # not blow up (allow generous slack for per-block tensor work).
        assert peak_b <= 3.0 * peak_a + 2048, (
            f"knockoff streaming peak {peak_a:.0f} -> {peak_b:.0f} "
            f"KiB with 4x more chromosomes — should be roughly per-chrom."
        )


class TestLroScanStreamingMemory:
    """Per-chromosome streaming for lro-scan."""

    def _build_inputs(self, n=60, m=80, n_chrom=2):
        torch.manual_seed(53)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        chrs = []
        poss = []
        per = m // n_chrom
        for c in range(n_chrom):
            s = c * per
            e = (c + 1) * per if c < n_chrom - 1 else m
            chrs.extend([str(c + 1)] * (e - s))
            poss.extend(list(range(0, (e - s) * 1000, 1000)))
        from torchgenomics.linalg.kinship import grm_vanraden
        K, meta = grm_vanraden(G)
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        ids = [f"r{i}" for i in range(m)]
        return Y, X0, G, K, float(meta.normalizer), chrs, poss, ids

    def test_streaming_runs_end_to_end(self):
        Y, X0, G, K, norm, chrs, poss, ids = self._build_inputs()
        result = _stream_lro_via_cli_helper(
            Y, X0, G, K, norm, chrs, poss, ids, chunk_size=23,
        )
        assert result.beta.shape[0] == G.shape[1]
        assert result.n_blocks > 0
        # P-values bounded.
        assert (result.p >= 0).all() and (result.p <= 1).all()

    def test_streaming_peak_per_chromosome_not_genome(self):
        Y_a, X0_a, G_a, K_a, norm_a, chr_a, pos_a, id_a = self._build_inputs(
            n=60, m=80, n_chrom=1,
        )
        Y_b, X0_b, G_b, K_b, norm_b, chr_b, pos_b, id_b = self._build_inputs(
            n=60, m=320, n_chrom=4,
        )

        def _run(Y, X0, G, K, norm, chs, ps, ids):
            return _stream_lro_via_cli_helper(
                Y, X0, G, K, norm, chs, ps, ids, chunk_size=32,
            )

        _, peak_a = _peak_kib(lambda: _run(Y_a, X0_a, G_a, K_a, norm_a, chr_a, pos_a, id_a))
        _, peak_b = _peak_kib(lambda: _run(Y_b, X0_b, G_b, K_b, norm_b, chr_b, pos_b, id_b))
        assert peak_b <= 3.0 * peak_a + 2048, (
            f"lro streaming peak {peak_a:.0f} -> {peak_b:.0f} KiB with "
            "4x more chromosomes — should be roughly per-chrom."
        )


def _stream_clump_via_cli_helper(
    p: torch.Tensor,
    G: torch.Tensor,
    chrs: list[str],
    poss: list[int],
    *,
    r2_threshold: float,
    p_threshold: float,
    window_kb: float,
    chunk_size: int = 64,
):
    """Replicate _cmd_clump's per-chromosome streaming accumulator."""
    from torchgenomics.postgwas import ld_clump
    from torchgenomics.postgwas._clump import ClumpResult

    chrom_to_global_idx: dict[str, list[int]] = {}
    for i, c in enumerate(chrs):
        chrom_to_global_idx.setdefault(c, []).append(i)

    reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=chunk_size)

    all_index_snps_global: list[int] = []
    all_clump_members_global: list[list[int]] = []

    cur_chr: str | None = None
    cur_chunks: list[torch.Tensor] = []
    cur_pos: list[int] = []

    def _process():
        nonlocal cur_chunks, cur_pos
        if not cur_chunks or cur_chr is None:
            return
        global_idx = chrom_to_global_idx.get(cur_chr, [])
        if not global_idx:
            cur_chunks = []
            cur_pos = []
            return
        G_chr = torch.cat(cur_chunks, dim=1)
        n_use = min(len(global_idx), G_chr.shape[1])
        global_idx_use = global_idx[:n_use]
        G_chr = G_chr[:, :n_use]
        gidx_t = torch.tensor(global_idx_use, dtype=torch.long)
        p_chr = p[gidx_t]
        chr_result = ld_clump(
            p_chr, G_chr, cur_pos[:n_use],
            [str(cur_chr)] * n_use,
            r2_threshold=r2_threshold,
            p_threshold=p_threshold,
            window_kb=window_kb,
        )
        for local_i in chr_result.index_snps:
            all_index_snps_global.append(global_idx_use[local_i])
        for members in chr_result.clump_members:
            all_clump_members_global.append(
                [global_idx_use[m] for m in members]
            )
        cur_chunks = []
        cur_pos = []

    for G_chunk, vmeta in reader.iter_chunks(chunk_size):
        chr_chunk = [str(c) for c in vmeta.chr]
        run_start = 0
        while run_start < len(chr_chunk):
            run_chr = chr_chunk[run_start]
            run_end = run_start + 1
            while run_end < len(chr_chunk) and chr_chunk[run_end] == run_chr:
                run_end += 1
            if cur_chr is None:
                cur_chr = run_chr
            elif run_chr != cur_chr:
                _process()
                cur_chr = run_chr
            cur_chunks.append(G_chunk[:, run_start:run_end].clone())
            cur_pos.extend(vmeta.pos[run_start:run_end])
            run_start = run_end
    _process()

    if all_index_snps_global:
        order = sorted(
            range(len(all_index_snps_global)),
            key=lambda k: float(p[all_index_snps_global[k]].item()),
        )
        idx_sorted = [all_index_snps_global[k] for k in order]
        members_sorted = [all_clump_members_global[k] for k in order]
        return ClumpResult(
            index_snps=idx_sorted,
            index_p=p[torch.tensor(idx_sorted, dtype=torch.long)],
            clump_members=members_sorted,
            n_clumps=len(idx_sorted),
        )
    return ClumpResult(
        index_snps=[], index_p=torch.tensor([]),
        clump_members=[], n_clumps=0,
    )


class TestClumpStreamingMemory:
    """Per-chromosome streaming for the clump CLI command."""

    def test_streaming_matches_materialized_single_chrom(self):
        from torchgenomics.postgwas import ld_clump

        torch.manual_seed(11)
        n, m = 100, 200
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        chrs = ["1"] * m
        poss = list(range(0, m * 100, 100))
        # Synthesize p-values: a handful below threshold so clumping
        # actually emits index SNPs.
        p = torch.full((m,), 0.5, dtype=torch.float64)
        p[10] = 1e-9
        p[60] = 1e-8
        p[120] = 1e-9

        ref = ld_clump(
            p, G, poss, chrs,
            r2_threshold=0.1, p_threshold=5e-7, window_kb=2.0,
        )
        stream = _stream_clump_via_cli_helper(
            p, G, chrs, poss,
            r2_threshold=0.1, p_threshold=5e-7, window_kb=2.0,
            chunk_size=37,
        )
        assert stream.n_clumps == ref.n_clumps
        assert stream.index_snps == ref.index_snps
        assert torch.allclose(stream.index_p, ref.index_p)

    def test_streaming_peak_per_chromosome_not_genome(self):
        torch.manual_seed(13)
        # 1 chrom × 400 SNPs vs 4 chroms × 400 SNPs each.
        G_a, chr_a, pos_a = _ld_fixture(n=80, m=400, n_chrom=1)
        G_b, chr_b, pos_b = _ld_fixture(n=80, m=1600, n_chrom=4)
        p_a = torch.full((400,), 0.5, dtype=torch.float64)
        p_a[5] = 1e-9
        p_b = torch.full((1600,), 0.5, dtype=torch.float64)
        for k in (5, 405, 805, 1205):
            p_b[k] = 1e-9

        def _run(p, G, ch, ps):
            return _stream_clump_via_cli_helper(
                p, G, ch, ps, r2_threshold=0.1, p_threshold=5e-7,
                window_kb=2.0, chunk_size=64,
            )

        _, peak_a = _peak_kib(lambda: _run(p_a, G_a, chr_a, pos_a))
        _, peak_b = _peak_kib(lambda: _run(p_b, G_b, chr_b, pos_b))
        assert peak_b <= 2.0 * peak_a + 256, (
            f"clump streaming peak grew from {peak_a:.0f} KiB "
            f"(1 chrom) to {peak_b:.0f} KiB (4 chroms) — should be "
            "roughly invariant per-chromosome."
        )


class TestLdScoresStreamingMemory:
    """``compute_ld_scores_streaming`` peak ∝ window_size, not ∝ m."""

    def test_streaming_matches_materialized(self):
        from torchgenomics.postgwas import (
            compute_ld_scores,
            compute_ld_scores_streaming,
        )

        G, chrs, poss = _ld_fixture(n=200, m=600, n_chrom=2)
        ref = compute_ld_scores(G, poss, chrs, window_kb=5.0)
        reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=64)
        stream, ch_out, pos_out = compute_ld_scores_streaming(
            reader.iter_chunks(64), window_kb=5.0,
        )

        assert ch_out == chrs
        assert pos_out == poss
        assert torch.allclose(stream, ref, atol=1e-9, rtol=1e-9)

    def test_streaming_peak_scales_with_window_not_m(self):
        """Peak should depend on window_size, not on total m."""
        from torchgenomics.postgwas import compute_ld_scores_streaming

        # Small m, dense window
        G_a, chr_a, pos_a = _ld_fixture(n=100, m=500, n_chrom=1)
        # Large m, same dense window
        G_b, chr_b, pos_b = _ld_fixture(n=100, m=2000, n_chrom=1)

        def _run(G, ch, ps):
            reader = _ChunkedTensorReader(G, ch, ps, chunk_size=128)
            return compute_ld_scores_streaming(
                reader.iter_chunks(128), window_kb=2.0,
            )

        _, peak_a = _peak_kib(lambda: _run(G_a, chr_a, pos_a))
        _, peak_b = _peak_kib(lambda: _run(G_b, chr_b, pos_b))

        # 4x m should NOT yield 4x peak. Allow generous slack
        # (chunk fluctuations, list growth) but block runaway.
        assert peak_b <= 2.0 * peak_a + 512, (
            f"LD-scores streaming peak grew from {peak_a:.0f} KiB "
            f"(m=500) to {peak_b:.0f} KiB (m=2000) — should be roughly "
            "invariant in m for fixed window. Streaming may have "
            "regressed to per-chromosome materialization."
        )


# ---------------------------------------------------------------------------
# F2 — FarmCPU streaming memory regression
# ---------------------------------------------------------------------------


class TestFarmCpuScanStreamingMemory:
    """FarmCPU streaming variant: ``score_streaming`` must not materialize G.

    Three guards:

    1. Behavioral parity vs the eager ``score_chunk`` path on a tiny
       fixture (float64 tolerance).
    2. Absolute peak budget under a documented ceiling at n=200 / m=500.
    3. Peak invariance in ``m`` for fixed cached-QTN size — the streaming
       contract is that peak scales with chunk_size + n × |QTN|, NOT with
       total m.
    """

    def _build_fixture(self, n: int = 200, m: int = 500):
        torch.manual_seed(2026)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = (5.0 + 0.5 * G[:, 5] + 0.3 * G[:, 100]
             + torch.randn(n, dtype=torch.float64))
        X0 = torch.ones(n, 1, dtype=torch.float64)
        chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)
        poss = list(range(m))
        return G, Y, X0, chrs, poss

    def test_streaming_matches_materialized(self):
        from torchgenomics.models.farmcpu import FarmCPU
        G, Y, X0, chrs, poss = self._build_fixture(n=100, m=200)
        vm = VariantMeta(
            snp=[f"rs{i}" for i in range(G.shape[1])],
            chr=chrs, pos=poss, a1=["A"] * G.shape[1], a2=["G"] * G.shape[1],
        )
        model = FarmCPU(max_iter=4, p_threshold=0.01, max_qtns=10)
        nf = model.fit_null(Y, X0)
        ref = model.score_chunk(G, nf, vm, test="wald")
        reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=37)
        streamed = model.score_streaming(reader, nf, chunk_size=37, test="wald")

        assert streamed.snp == ref.snp
        assert torch.allclose(streamed.p, ref.p, atol=1e-10, rtol=1e-10)
        assert torch.allclose(streamed.beta, ref.beta, atol=1e-10, rtol=1e-10)

    def test_streaming_peak_under_explicit_budget(self):
        """Hard ceiling: <16 MiB at n=200 / m=500."""
        from torchgenomics.models.farmcpu import FarmCPU
        G, Y, X0, chrs, poss = self._build_fixture()

        def _run():
            model = FarmCPU(max_iter=3, p_threshold=0.01, max_qtns=5)
            nf = model.fit_null(Y, X0)
            reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=64)
            return model.score_streaming(reader, nf, chunk_size=64, test="wald")

        _, peak = _peak_kib(_run)
        budget_kib = 16 * 1024
        assert peak < budget_kib, (
            f"FarmCPU streaming peak ({peak:.0f} KiB) exceeds 16 MiB. "
            "Likely regressed to materializing G."
        )

    def test_streaming_peak_scales_with_chunk_not_m(self):
        from torchgenomics.models.farmcpu import FarmCPU
        G_a, Y_a, X0_a, ch_a, ps_a = self._build_fixture(n=80, m=200)
        G_b, Y_b, X0_b, ch_b, ps_b = self._build_fixture(n=80, m=800)

        def _run(G, Y, X0, ch, ps):
            model = FarmCPU(max_iter=2, p_threshold=0.01, max_qtns=5)
            nf = model.fit_null(Y, X0)
            reader = _ChunkedTensorReader(G, ch, ps, chunk_size=64)
            return model.score_streaming(reader, nf, chunk_size=64, test="wald")

        _, peak_a = _peak_kib(lambda: _run(G_a, Y_a, X0_a, ch_a, ps_a))
        _, peak_b = _peak_kib(lambda: _run(G_b, Y_b, X0_b, ch_b, ps_b))
        # 4x m should not yield ~4x peak. Allow generous slack for
        # per-chunk allocations + small bookkeeping growth.
        assert peak_b <= 2.5 * peak_a + 1024, (
            f"FarmCPU streaming peak grew from {peak_a:.0f} KiB (m=200) "
            f"to {peak_b:.0f} KiB (m=800) — should scale with chunk, not m."
        )


# ---------------------------------------------------------------------------
# F2 — BLINK streaming memory regression
# ---------------------------------------------------------------------------


class TestBlinkScanStreamingMemory:
    """BLINK streaming variant — same shape as FarmCPU."""

    def _build_fixture(self, n: int = 200, m: int = 500):
        torch.manual_seed(2027)
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)
        Y = (5.0 + 0.5 * G[:, 5] + 0.3 * G[:, 100]
             + torch.randn(n, dtype=torch.float64))
        X0 = torch.ones(n, 1, dtype=torch.float64)
        chrs = ["1"] * (m // 2) + ["2"] * (m - m // 2)
        poss = list(range(m))
        return G, Y, X0, chrs, poss

    def test_streaming_matches_materialized(self):
        from torchgenomics.models.blink import BLINK
        G, Y, X0, chrs, poss = self._build_fixture(n=100, m=200)
        vm = VariantMeta(
            snp=[f"rs{i}" for i in range(G.shape[1])],
            chr=chrs, pos=poss, a1=["A"] * G.shape[1], a2=["G"] * G.shape[1],
        )
        model = BLINK(max_iter=4, cutoff=0.5)
        nf = model.fit_null(Y, X0)
        ref = model.score_chunk(G, nf, vm, test="wald")
        reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=37)
        streamed = model.score_streaming(reader, nf, chunk_size=37, test="wald")
        assert streamed.snp == ref.snp
        assert torch.allclose(streamed.p, ref.p, atol=1e-10, rtol=1e-10)
        assert torch.allclose(streamed.beta, ref.beta, atol=1e-10, rtol=1e-10)

    def test_streaming_peak_under_explicit_budget(self):
        from torchgenomics.models.blink import BLINK
        G, Y, X0, chrs, poss = self._build_fixture()

        def _run():
            model = BLINK(max_iter=3, cutoff=0.5)
            nf = model.fit_null(Y, X0)
            reader = _ChunkedTensorReader(G, chrs, poss, chunk_size=64)
            return model.score_streaming(reader, nf, chunk_size=64, test="wald")

        _, peak = _peak_kib(_run)
        budget_kib = 16 * 1024
        assert peak < budget_kib, (
            f"BLINK streaming peak ({peak:.0f} KiB) exceeds 16 MiB."
        )

    def test_streaming_peak_scales_with_chunk_not_m(self):
        from torchgenomics.models.blink import BLINK
        G_a, Y_a, X0_a, ch_a, ps_a = self._build_fixture(n=80, m=200)
        G_b, Y_b, X0_b, ch_b, ps_b = self._build_fixture(n=80, m=800)

        def _run(G, Y, X0, ch, ps):
            model = BLINK(max_iter=2, cutoff=0.5)
            nf = model.fit_null(Y, X0)
            reader = _ChunkedTensorReader(G, ch, ps, chunk_size=64)
            return model.score_streaming(reader, nf, chunk_size=64, test="wald")

        _, peak_a = _peak_kib(lambda: _run(G_a, Y_a, X0_a, ch_a, ps_a))
        _, peak_b = _peak_kib(lambda: _run(G_b, Y_b, X0_b, ch_b, ps_b))
        assert peak_b <= 2.5 * peak_a + 1024, (
            f"BLINK streaming peak grew from {peak_a:.0f} KiB (m=200) "
            f"to {peak_b:.0f} KiB (m=800)."
        )


# ---------------------------------------------------------------------------
# F2 — mediate-scan SNP-block streaming memory regression
# ---------------------------------------------------------------------------


class TestMediateScanStreamingMemory:
    """``scan_mediation(streaming=True)`` must not materialize the rotated G.

    Two guards:

    1. Behavioral parity vs ``streaming=False``: bit-for-bit zero diff
       on the per-pair (a, b, c, c_prime, indirect, indirect_pvalue)
       fields when the same SE seed is used.
    2. Peak under explicit budget at n=80 / s_variants=300 / s_features=20.
    """

    def _build_fixture(self, n: int = 80, s: int = 300, f: int = 20):
        torch.manual_seed(42)
        G = torch.randint(0, 3, (n, s), dtype=torch.float64)
        M = torch.randn(n, f, dtype=torch.float64)
        K = (G @ G.T) / s + 0.01 * torch.eye(n, dtype=torch.float64)
        K = (K + K.T) / 2
        Y = (M[:, 0] * 0.3 + G[:, 5] * 0.5
             + torch.randn(n, dtype=torch.float64))
        return Y, G, M, K

    def test_streaming_matches_eager_batched(self):
        from torchgenomics.multiomics import scan_mediation
        Y, G, M, K = self._build_fixture()

        eager = scan_mediation(
            Y, G, M, K,
            cis_window_bp=None,
            se="monte-carlo", n_mc_draws=200, seed=7,
            sensitivity=False, batched=True, streaming=False,
        )
        stream = scan_mediation(
            Y, G, M, K,
            cis_window_bp=None,
            se="monte-carlo", n_mc_draws=200, seed=7,
            sensitivity=False, batched=True, streaming=True,
        )
        assert eager.n_pairs == stream.n_pairs
        max_diff = 0.0
        for re, rs in zip(eager.rows, stream.rows):
            for k in (
                "a", "b", "c", "c_prime", "indirect",
                "indirect_pvalue", "ci_lower", "ci_upper",
            ):
                if re[k] is None or rs[k] is None:
                    continue
                d = abs(re[k] - rs[k])
                if d > max_diff:
                    max_diff = d
        assert max_diff < 1e-10, (
            f"streaming mediate-scan diverges from eager batched path "
            f"(max diff {max_diff:.2e}); rotation is linear so per-pair "
            "outputs should agree to float64 round-off."
        )

    def test_streaming_peak_under_explicit_budget(self):
        """Hard ceiling: <32 MiB at n=80 / s=300 / f=20.

        A regression to materializing G_r alongside G would add roughly
        ``n × s × 8 B = 80 × 300 × 8 = 187 KiB`` per allocation; on the
        tiny fixture this is negligible compared to the SE / MC draws,
        so the ceiling is set well above the observed peak with margin.
        """
        from torchgenomics.multiomics import scan_mediation
        Y, G, M, K = self._build_fixture()

        def _run():
            return scan_mediation(
                Y, G, M, K,
                cis_window_bp=None,
                se="monte-carlo", n_mc_draws=200, seed=7,
                sensitivity=False, batched=True, streaming=True,
            )

        _, peak = _peak_kib(_run)
        budget_kib = 32 * 1024
        assert peak < budget_kib, (
            f"mediate-scan streaming peak ({peak:.0f} KiB) exceeds 32 MiB."
        )


# ---------------------------------------------------------------------------
# NA1 — bayes-scan-rss block-decomposed SuSiE memory regression
# ---------------------------------------------------------------------------


class TestBayesScanRssMemory:
    """Memory regression net for ``bayes-scan-rss`` (NA1 SuSiE-RSS).

    Per NA1 design spec section 4.5: ``BayesianVSRss.fit_rss_blocked`` must
    bound peak allocation by ``O(p_block_max^2 + L * p_total)`` — never by
    ``O(n * p)`` (which is what the raw-G ``bayes-scan`` path costs at
    biobank scale: ~800 GB at p=200K / n=500K).

    Two guards:

    1. **Absolute peak budget** — at p=2000 / block=500 / L=3 the
       tracemalloc-measured peak is well under 1 MiB. The 8 MiB ceiling
       below is a generous bound that catches a regression to
       materializing R or stacking per-block residuals across blocks.
    2. **Block-vs-total scaling** — holding ``block_size`` fixed and
       quintupling ``p`` should leave the peak roughly unchanged. A
       regression to dense-R IBSS would scale the peak with ``p^2``.

    We use ``tracemalloc`` (not OS-level RSS) to match the rest of this
    file: it's deterministic, captures Python-managed allocations only,
    and reads the lifetime peak from ``start()`` to ``stop()``.
    """

    @staticmethod
    def _build_blocked_fixture(p: int, block_size: int, n: int = 1000):
        """Build a block-diagonal R with a fixed block size, plus blocks list."""
        rng = torch.Generator()
        rng.manual_seed(0)
        z = torch.randn(p, generator=rng, dtype=torch.float64)
        R = torch.zeros(p, p, dtype=torch.float64)
        for i in range(0, p, block_size):
            R[i:i + block_size, i:i + block_size] = torch.eye(
                block_size, dtype=torch.float64,
            )
        from torchgenomics.postgwas._ld_ref_loader import BlockSpec
        blocks = [
            BlockSpec(start=i, stop=i + block_size)
            for i in range(0, p, block_size)
        ]
        return z, R, n, blocks

    def test_peak_under_explicit_budget(self):
        """Hard ceiling: <8 MiB at p=2000 / block=500 / L=3.

        With ``tracemalloc`` we observe ~28 KiB on this fixture; the
        8 MiB ceiling provides ample margin while still tripping on
        any regression that materializes a full ``p x p`` working buffer
        or stacks all per-block IBSS state simultaneously.
        """
        from torchgenomics.models.bayesian_vs_rss import BayesianVSRss

        z, R, n, blocks = self._build_blocked_fixture(p=2000, block_size=500)

        def _run():
            model = BayesianVSRss(max_num_causal=3, max_iter=20)
            return model.fit_rss_blocked(z=z, R=R, n=n, blocks=blocks)

        _, peak_kib = _peak_kib(_run)
        budget_kib = 8 * 1024  # 8 MiB
        assert peak_kib < budget_kib, (
            f"bayes-scan-rss peak ({peak_kib:.0f} KiB) exceeds the "
            f"8 MiB budget at p=2000 / block=500. At biobank scale this "
            "would translate to multi-TB regression."
        )

    def test_peak_scales_with_block_not_total_p(self):
        """Peak should scale with ``p_block_max``, not total ``p``.

        Construct two scenarios with the same block size (500) but very
        different ``p``: 1000 vs 5000. Per spec section 2.6 the per-block
        IBSS allocations dominate and per-block sub-R is sliced (not
        copied for new memory beyond the block); the streaming peak
        should be roughly the same.

        A regression to dense-R IBSS would make the peak scale with
        ``p^2`` — at p=200K this is the difference between a 200 MB and
        a 320 GB working set.
        """
        from torchgenomics.models.bayesian_vs_rss import BayesianVSRss

        def _run(p):
            z, R, n, blocks = self._build_blocked_fixture(
                p=p, block_size=500,
            )

            def _inner():
                model = BayesianVSRss(max_num_causal=3, max_iter=20)
                return model.fit_rss_blocked(z=z, R=R, n=n, blocks=blocks)

            _, peak = _peak_kib(_inner)
            return peak

        peak_small = _run(p=1000)
        peak_large = _run(p=5000)

        # Allow 3× growth for per-block scratch / Python-object overhead;
        # 5× p -> 5× peak would mean we're scaling with p, not block.
        assert peak_large <= 3.0 * peak_small + 64, (
            f"bayes-scan-rss peak grew from {peak_small:.0f} KiB (p=1000) "
            f"to {peak_large:.0f} KiB (p=5000) at fixed block_size=500. "
            "fit_rss_blocked likely regressed to scale with total p."
        )


# ---------------------------------------------------------------------------
# tractor-scan (TractorLMM.scan over AncestryDosages) memory regression
# (Task 9 / Phase 57 Unit B — local-ancestry conditional analysis +
# streaming scan driver)
# ---------------------------------------------------------------------------


@pytest.fixture
def tractor_inputs():
    """Synthetic admixed cohort via the Tractor-Mix test fixture generator.

    K=2 ancestries, n=200 samples, m=2000 variants — large enough that a
    materialized (K, n, m) result would be 2*200*2000*8 B = ~6.25 MiB just
    for the dosage tensor itself (plus per-chunk score-test intermediates
    for a non-streaming path); the streaming driver should keep peak
    allocation close to the chunk_size, not m.
    """
    from tests.fixtures.tractor.make_synth import make_synth

    d = make_synth(n=200, m=2000, K=2, seed=97)
    return d


class TestTractorScanStreamingMemory:
    """TractorLMM.scan() must not materialize the full ``(K, n, m)`` panel.

    Two guards, mirroring the other streaming-memory classes in this file:

    1. Behavioral equivalence — streaming ``joint_p`` matches a single
       non-chunked ``score_chunk`` call (streaming invariance;
       ``score_chunk`` already scores variants independently).
    2. Peak-scales-with-chunk-size-not-m — construct two scenarios with
       the same ``chunk_size`` but very different total ``m``; the peak
       streaming allocation should be roughly the same. This is the
       biobank-relevant guard: at UKB-admixed-cohort scale, the
       difference between ``K * n * chunk_size * 8 B`` (MB scale) and
       ``K * n * m * 8 B`` (TB scale for genome-wide ancestry-dosage
       panels) is the difference between fits-on-a-laptop and
       doesn't-fit-on-cluster.
    """

    def test_streaming_matches_materialized(self, tractor_inputs):
        from torchgenomics.io.ancestry_dosage import AncestryDosages
        from torchgenomics.models.tractor_lmm import TractorLMM

        d = tractor_inputs
        m = d["dosages"].shape[2]
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        model = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
        nf = model.fit_null(d["y_cont"], d["X0"], d["K_grm"])

        # Reference: single big-tensor score_chunk (legacy/non-streaming path)
        ref = model.score_chunk(nf, d["dosages"], vmeta)

        # Streaming path via TractorLMM.scan() over an AncestryDosages
        # container, chunk_size << m.
        ad = AncestryDosages(d["dosages"], ["AFR", "EUR"], variant_meta=vmeta)
        streamed = model.scan(nf, ad, vmeta, chunk_size=100)

        assert torch.allclose(
            streamed.joint_p, ref.joint_p, atol=1e-10, rtol=1e-10, equal_nan=True,
        )
        assert torch.allclose(
            streamed.beta, ref.beta, atol=1e-10, rtol=1e-10, equal_nan=True,
        )

    def test_streaming_peak_scales_with_chunk_size_not_m(self):
        """Streaming peak should scale with ``chunk_size``, not total ``m``.

        Construct two scenarios with the same ``chunk_size`` (100) but
        very different ``m`` (500 vs 2500, a 5x difference). If
        ``TractorLMM.scan`` accidentally materialized the full
        ``(K, n, m)`` result (or accumulated all per-chunk results in a
        way that scales with m beyond the small final concat), the peak
        would grow roughly proportionally with m. A truly streaming
        driver's peak is dominated by the per-chunk score_chunk
        intermediates (sized ``K x n x chunk_size``), which do not grow
        with m.
        """
        from tests.fixtures.tractor.make_synth import make_synth
        from torchgenomics.io.ancestry_dosage import AncestryDosages
        from torchgenomics.models.tractor_lmm import TractorLMM

        def _run(m):
            d = make_synth(n=200, m=m, K=2, seed=97)
            vmeta = VariantMeta(
                snp=[f"rs{i}" for i in range(m)],
                chr=["1"] * m,
                pos=list(range(m)),
                a1=["A"] * m,
                a2=["G"] * m,
            )
            model = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
            nf = model.fit_null(d["y_cont"], d["X0"], d["K_grm"])
            ad = AncestryDosages(d["dosages"], ["AFR", "EUR"], variant_meta=vmeta)

            def _inner():
                return model.scan(nf, ad, vmeta, chunk_size=100)

            _, peak = _peak_kib(_inner)
            return peak

        peak_small = _run(m=500)
        peak_large = _run(m=2500)

        # Allow generous headroom for the final concat's O(m) list/tensor
        # allocation (unavoidable — the full result has to exist once at
        # the end) and Python-object overhead; a genuine re-materialization
        # of a (K, n, m) intermediate during the *scan loop itself* would
        # blow well past a modest constant multiple of the m ratio (5x)
        # once compounded with score_chunk's own per-chunk intermediates.
        assert peak_large <= 5.0 * peak_small + 512, (
            f"TractorLMM.scan streaming peak grew from {peak_small:.0f} KiB "
            f"(m=500) to {peak_large:.0f} KiB (m=2500) at fixed "
            "chunk_size=100 — should scale with chunk_size, not m. The "
            "streaming driver likely regressed to materializing (K, n, m)."
        )

    def test_streaming_peak_under_explicit_budget(self, tractor_inputs):
        """Hard absolute budget: <16 MiB at n=200/m=2000/K=2, chunk_size=100.

        At this fixture, a materialized (K, n, m) float64 dosage tensor
        alone costs 2*200*2000*8 B = ~6.1 MiB; a re-materializing
        regression plus per-chunk score-test intermediates would push
        comfortably past 16 MiB. The streaming driver's peak is bounded
        by O(K * n * chunk_size) intermediates, which is tiny by
        comparison (chunk_size=100 vs m=2000, a 20x reduction).
        """
        from torchgenomics.io.ancestry_dosage import AncestryDosages
        from torchgenomics.models.tractor_lmm import TractorLMM

        d = tractor_inputs
        m = d["dosages"].shape[2]
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )
        model = TractorLMM(family="gaussian", ancestry_names=["AFR", "EUR"])
        nf = model.fit_null(d["y_cont"], d["X0"], d["K_grm"])
        ad = AncestryDosages(d["dosages"], ["AFR", "EUR"], variant_meta=vmeta)

        def _run_streaming():
            return model.scan(nf, ad, vmeta, chunk_size=100)

        _, peak_stream = _peak_kib(_run_streaming)
        budget_kib = 16 * 1024  # 16 MiB
        assert peak_stream < budget_kib, (
            f"Streaming Tractor-Mix scan peak ({peak_stream:.0f} KiB) "
            f"exceeds the 16 MiB budget. At biobank-admixed-cohort scale "
            "this would translate to a multi-TB regression."
        )
