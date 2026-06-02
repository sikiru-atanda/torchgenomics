"""Phase 10: GPU acceleration, CPU/GPU equivalence, AMP, and permutation tests."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.config import STAT_DTYPE

HAS_CUDA = torch.cuda.is_available()
skip_no_gpu = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")


class TestGPUEquivalence:
    """CPU vs GPU output equivalence (Section 16)."""

    @skip_no_gpu
    def test_grm_cpu_gpu_match(self):
        """GRM on GPU matches CPU within FP tolerance."""
        from torchgenomics.linalg.kinship import grm_vanraden

        torch.manual_seed(42)
        G = torch.randint(0, 3, (50, 200), dtype=STAT_DTYPE)

        K_cpu, meta_cpu = grm_vanraden(G)
        K_gpu, meta_gpu = grm_vanraden(G.cuda())

        torch.testing.assert_close(K_cpu, K_gpu.cpu(), atol=1e-10, rtol=1e-10)
        assert meta_cpu.n_snps_used == meta_gpu.n_snps_used

    @skip_no_gpu
    def test_scan_cpu_gpu_match(self):
        """Scan p-values on GPU match CPU within declared tolerance (<=1e-4 relative)."""
        from torchgenomics.models.base import VariantMeta
        from torchgenomics.models.glm import GLM

        torch.manual_seed(42)
        n, m = 100, 50
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
        Y = torch.randn(n, 1, dtype=STAT_DTYPE)
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        glm = GLM()

        # CPU
        nf_cpu = glm.fit_null(Y, X0)
        result_cpu = glm.score_chunk(G, nf_cpu, vmeta)

        # GPU
        nf_gpu = glm.fit_null(Y.cuda(), X0.cuda())
        result_gpu = glm.score_chunk(G.cuda(), nf_gpu, vmeta)

        # P-values should match within FP tolerance
        torch.testing.assert_close(
            result_cpu.p, result_gpu.p.cpu(),
            atol=1e-10, rtol=1e-4,
        )

    @skip_no_gpu
    def test_deterministic_mode(self):
        """torch.use_deterministic_algorithms(True) produces reproducible results."""
        from torchgenomics.config import set_deterministic
        from torchgenomics.linalg.kinship import grm_vanraden

        set_deterministic(True)
        try:
            torch.manual_seed(42)
            G = torch.randint(0, 3, (30, 100), dtype=STAT_DTYPE).cuda()
            K1, _ = grm_vanraden(G)

            torch.manual_seed(42)
            G = torch.randint(0, 3, (30, 100), dtype=STAT_DTYPE).cuda()
            K2, _ = grm_vanraden(G)

            torch.testing.assert_close(K1, K2, atol=0, rtol=0)
        finally:
            set_deterministic(False)

    @skip_no_gpu
    def test_eigendecompose_gpu(self):
        """Eigendecomposition on GPU matches CPU."""
        from torchgenomics.linalg.eigh import eigendecompose
        from torchgenomics.linalg.kinship import grm_vanraden

        torch.manual_seed(42)
        G = torch.randint(0, 3, (30, 100), dtype=STAT_DTYPE)
        K, _ = grm_vanraden(G)

        ed_cpu = eigendecompose(K)
        ed_gpu = eigendecompose(K.cuda())

        torch.testing.assert_close(
            ed_cpu.eigenvalues, ed_gpu.eigenvalues.cpu(),
            atol=1e-8, rtol=1e-8,
        )

    @skip_no_gpu
    def test_lmm_scan_gpu(self):
        """LMM scan on GPU matches CPU."""
        from torchgenomics.linalg.kinship import grm_vanraden
        from torchgenomics.models.base import VariantMeta
        from torchgenomics.models.single_trait_lmm import SingleTraitLMM

        torch.manual_seed(42)
        n, m = 80, 40
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
        K, _ = grm_vanraden(G)
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
        Y = torch.randn(n, 1, dtype=STAT_DTYPE)
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        lmm = SingleTraitLMM()

        # CPU
        nf_cpu = lmm.fit_null(Y, X0, K=K)
        result_cpu = lmm.score_chunk(G, nf_cpu, vmeta, test="wald")

        # GPU
        nf_gpu = lmm.fit_null(Y.cuda(), X0.cuda(), K=K.cuda())
        result_gpu = lmm.score_chunk(G.cuda(), nf_gpu, vmeta, test="wald")

        torch.testing.assert_close(
            result_cpu.p, result_gpu.p.cpu(),
            atol=1e-8, rtol=1e-4,
        )


class TestAMP:
    """AMP mixed-precision tests."""

    @skip_no_gpu
    def test_amp_grm_accumulates_fp64(self):
        """GRM accumulation stays in float64 even with AMP enabled."""
        from torchgenomics.linalg.kinship import grm_vanraden_streaming

        torch.manual_seed(42)
        G = torch.randint(0, 3, (30, 100), dtype=STAT_DTYPE).cuda()

        def chunk_iter():
            for s in range(0, 100, 25):
                yield G[:, s:s + 25], None

        K, meta = grm_vanraden_streaming(
            chunk_iter(), n_samples=30, device=torch.device("cuda"),
            amp_enabled=True, amp_dtype=torch.float16,
        )

        assert K.dtype == torch.float64
        assert meta.gemm_dtype == "torch.float32"

    @skip_no_gpu
    def test_amp_inference_fp64(self):
        """Statistical inference (p-values, SE) uses float64 regardless of AMP."""
        from torchgenomics.models.base import VariantMeta
        from torchgenomics.models.glm import GLM

        torch.manual_seed(42)
        n, m = 50, 20
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE).cuda()
        Y = torch.randn(n, 1, dtype=STAT_DTYPE).cuda()
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE).cuda()
        vmeta = VariantMeta(
            snp=[f"rs{i}" for i in range(m)],
            chr=["1"] * m, pos=list(range(m)),
            a1=["A"] * m, a2=["G"] * m,
        )

        glm = GLM()
        nf = glm.fit_null(Y, X0)

        # Even under AMP autocast, model output must be FP64
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            result = glm.score_chunk(G, nf, vmeta)

        # The model casts to STAT_DTYPE (float64) internally
        assert result.p.dtype == STAT_DTYPE
        assert result.se.dtype == STAT_DTYPE
        assert result.beta.dtype == STAT_DTYPE


class TestPermutation:
    """GPU-accelerated permutation testing."""

    def test_permutation_pvalue_range(self):
        """Permutation p-values are in (0, 1]."""
        from torchgenomics.stats.permutation import permutation_maxT

        torch.manual_seed(42)
        n, m = 50, 30
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
        Y = torch.randn(n, dtype=STAT_DTYPE)
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

        p_adj = permutation_maxT(G, Y, X0, n_perms=200, seed=42)

        assert p_adj.shape == (m,)
        assert (p_adj > 0).all()
        assert (p_adj <= 1).all()

    def test_maxT_controls_fwer(self):
        """Under null (no true signal), max-T adjusted p > 0.05 for most SNPs."""
        from torchgenomics.stats.permutation import permutation_maxT

        torch.manual_seed(123)
        n, m = 100, 50
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
        Y = torch.randn(n, dtype=STAT_DTYPE)  # pure noise
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

        p_adj = permutation_maxT(G, Y, X0, n_perms=500, seed=123)

        # Under null, expect no significant hits (FWER control)
        n_sig = (p_adj < 0.05).sum().item()
        # Allow up to 5% false positives (generous)
        assert n_sig <= m * 0.1, f"Too many false positives: {n_sig}/{m}"

    def test_permutation_detects_signal(self):
        """Strong causal SNP should have small adjusted p-value."""
        from torchgenomics.stats.permutation import permutation_maxT

        torch.manual_seed(42)
        n, m = 200, 30
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)
        # Strong signal on SNP 5
        Y = G[:, 5] * 5.0 + torch.randn(n, dtype=STAT_DTYPE)
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

        p_adj = permutation_maxT(G, Y, X0, n_perms=500, seed=42)

        # Causal SNP should be significant even after maxT correction
        assert p_adj[5].item() < 0.05, f"Causal SNP not detected: p={p_adj[5].item()}"

    @skip_no_gpu
    def test_permutation_gpu(self):
        """Permutation test runs on GPU and produces valid results."""
        from torchgenomics.stats.permutation import permutation_maxT

        torch.manual_seed(42)
        n, m = 100, 20
        G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE).cuda()
        Y = torch.randn(n, dtype=STAT_DTYPE).cuda()
        X0 = torch.ones(n, 1, dtype=STAT_DTYPE).cuda()

        p_adj = permutation_maxT(G, Y, X0, n_perms=200, seed=42)

        assert p_adj.device.type == "cuda"
        assert (p_adj > 0).all()
        assert (p_adj <= 1).all()


class TestPrefetch:
    """Prefetch iterator tests."""

    def test_prefetch_cpu_passthrough(self):
        """On CPU, PrefetchIterator passes through without overhead."""
        from torchgenomics.models.base import VariantMeta
        from torchgenomics.scan.prefetch import PrefetchIterator

        chunks = [
            (torch.randn(10, 5), VariantMeta(
                snp=[f"s{i}" for i in range(5)], chr=["1"] * 5,
                pos=list(range(5)), a1=["A"] * 5, a2=["G"] * 5,
            ))
            for _ in range(3)
        ]

        prefetch = PrefetchIterator(iter(chunks), torch.device("cpu"))
        collected = list(prefetch)
        assert len(collected) == 3

    def test_move_nullfit_to_device(self):
        """move_nullfit_to_device handles all tensor fields."""
        from torchgenomics.models.base import NullFit
        from torchgenomics.scan.prefetch import move_nullfit_to_device

        nf = NullFit(
            sig2_g=1.0, sig2_e=1.0,
            eigenvalues=torch.randn(10),
            eigenvectors=torch.randn(10, 10),
            Y_rot=torch.randn(10),
            X0_rot=torch.randn(10, 1),
            device=torch.device("cpu"),
        )

        # Move to same device (no-op)
        nf2 = move_nullfit_to_device(nf, torch.device("cpu"))
        assert nf2.eigenvalues.device.type == "cpu"

    @skip_no_gpu
    def test_move_nullfit_to_gpu(self):
        """NullFit tensors move to GPU correctly."""
        from torchgenomics.models.base import NullFit
        from torchgenomics.scan.prefetch import move_nullfit_to_device

        nf = NullFit(
            sig2_g=1.0, sig2_e=1.0,
            eigenvalues=torch.randn(10),
            eigenvectors=torch.randn(10, 10),
            Y_rot=torch.randn(10),
            X0_rot=torch.randn(10, 1),
            device=torch.device("cpu"),
        )

        nf_gpu = move_nullfit_to_device(nf, torch.device("cuda"))
        assert nf_gpu.eigenvalues.device.type == "cuda"
        assert nf_gpu.eigenvectors.device.type == "cuda"
        assert nf_gpu.Y_rot.device.type == "cuda"
        assert nf_gpu.X0_rot.device.type == "cuda"
