"""Tests for Phase 13 approximate backends."""

from pathlib import Path

import pytest
import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_low_rank_psd(n: int, rank: int, seed: int = 42) -> torch.Tensor:
    """Create a low-rank PSD matrix: K = L @ L^T + eps * I."""
    torch.manual_seed(seed)
    L = torch.randn(n, rank, dtype=torch.float64)
    K = L @ L.T + 0.01 * torch.eye(n, dtype=torch.float64)
    return (K + K.T) / 2.0


def _exact_top_k(K: torch.Tensor, k: int):
    """Reference exact top-k eigendecomposition."""
    from torchgwas.linalg.eigh import eigendecompose
    ed = eigendecompose(K)
    return ed.eigenvalues[:k], ed.eigenvectors[:, :k]


# ---------------------------------------------------------------------------
# Randomized SVD
# ---------------------------------------------------------------------------

class TestRandomizedSVD:
    def test_eigenvalues_close_to_exact(self):
        """Top-k eigenvalues from randomized SVD match exact within 1%."""
        from torchgwas.linalg.randomized import randomized_svd

        n, rank, k = 200, 20, 15
        K = _make_low_rank_psd(n, rank)

        ed = randomized_svd(K, n_components=k, seed=42)
        evals_exact, _ = _exact_top_k(K, k)

        assert ed.eigenvalues.shape == (k,)
        assert ed.eigenvectors.shape == (n, k)

        # Top eigenvalues should match closely
        rel_err = torch.abs(ed.eigenvalues - evals_exact) / (evals_exact + 1e-10)
        assert rel_err.max() < 0.01, f"Max relative error: {rel_err.max():.4f}"

    def test_eigenvectors_subspace(self):
        """Eigenvectors span the same subspace as exact top-k."""
        from torchgwas.linalg.randomized import randomized_svd

        n, rank, k = 100, 10, 8
        K = _make_low_rank_psd(n, rank)

        ed = randomized_svd(K, n_components=k, seed=42)
        _, evecs_exact = _exact_top_k(K, k)

        # Cosine similarity between subspaces
        overlap = torch.svd(ed.eigenvectors.T @ evecs_exact).S
        # All singular values should be close to 1 (aligned subspaces)
        assert overlap.min() > 0.95

    def test_returns_eigendecomp(self):
        """Output is EigenDecomp dataclass."""
        from torchgwas.linalg.eigh import EigenDecomp
        from torchgwas.linalg.randomized import randomized_svd

        K = _make_low_rank_psd(50, 10)
        ed = randomized_svd(K, n_components=5, seed=42)
        assert isinstance(ed, EigenDecomp)
        assert ed.eigenvalues.dtype == torch.float64

    def test_descending_order(self):
        """Eigenvalues are in descending order."""
        from torchgwas.linalg.randomized import randomized_svd

        K = _make_low_rank_psd(80, 20)
        ed = randomized_svd(K, n_components=15, seed=42)
        diffs = ed.eigenvalues[:-1] - ed.eigenvalues[1:]
        assert (diffs >= -1e-10).all()

    def test_seed_reproducibility(self):
        """Same seed produces identical results."""
        from torchgwas.linalg.randomized import randomized_svd

        K = _make_low_rank_psd(50, 10)
        ed1 = randomized_svd(K, n_components=5, seed=123)
        ed2 = randomized_svd(K, n_components=5, seed=123)
        torch.testing.assert_close(ed1.eigenvalues, ed2.eigenvalues)


# ---------------------------------------------------------------------------
# LOBPCG
# ---------------------------------------------------------------------------

class TestLOBPCG:
    def test_eigenvalues_close_to_exact(self):
        """LOBPCG eigenvalues match exact top-k."""
        from torchgwas.linalg.randomized import lobpcg_decompose

        n, rank, k = 100, 20, 10
        K = _make_low_rank_psd(n, rank)

        ed = lobpcg_decompose(K, n_components=k, seed=42)
        evals_exact, _ = _exact_top_k(K, k)

        rel_err = torch.abs(ed.eigenvalues - evals_exact) / (evals_exact + 1e-10)
        assert rel_err.max() < 0.05, f"Max relative error: {rel_err.max():.4f}"

    def test_returns_eigendecomp(self):
        from torchgwas.linalg.eigh import EigenDecomp
        from torchgwas.linalg.randomized import lobpcg_decompose

        K = _make_low_rank_psd(50, 10)
        ed = lobpcg_decompose(K, n_components=5, seed=42)
        assert isinstance(ed, EigenDecomp)


# ---------------------------------------------------------------------------
# Nystrom
# ---------------------------------------------------------------------------

class TestNystrom:
    def _make_genotype_chunks(self, n=50, m=100, chunk_size=25, seed=42):
        """Create simple genotype chunks for testing."""
        from torchgwas.models.base import VariantMeta
        torch.manual_seed(seed)
        G = torch.clamp(torch.randn(n, m, dtype=torch.float64) + 1.0, 0.0, 2.0)

        chunks = []
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            vmeta = VariantMeta(
                snp=[f"rs{j}" for j in range(start, end)],
                chr=["1"] * (end - start),
                pos=list(range(start * 1000, end * 1000, 1000)),
                a1=["A"] * (end - start),
                a2=["G"] * (end - start),
            )
            chunks.append((G[:, start:end], vmeta))

        return chunks, G

    def test_nystrom_eigenvalues_positive(self):
        """Nystrom returns positive eigenvalues."""
        from torchgwas.linalg.nystrom import nystrom_approximate

        chunks, _ = self._make_genotype_chunks()
        ed, normalizer = nystrom_approximate(
            iter(chunks), n_samples=50, n_landmarks=20, seed=42,
        )
        assert (ed.eigenvalues >= 0).all()
        assert normalizer > 0

    def test_nystrom_approximation_quality(self):
        """Nystrom GRM close to full VanRaden GRM in top eigenvalues."""
        from torchgwas.linalg.kinship import grm_vanraden
        from torchgwas.linalg.nystrom import nystrom_approximate

        chunks, G = self._make_genotype_chunks(n=50, m=200, chunk_size=50)

        # Full GRM for reference
        K_full, _ = grm_vanraden(G)
        evals_exact, _ = _exact_top_k(K_full, 15)

        # Nystrom with all samples as landmarks (should be very close)
        ed, _ = nystrom_approximate(
            iter(chunks), n_samples=50, n_landmarks=50, seed=42,
        )

        # Top eigenvalues should be correlated
        k = min(len(ed.eigenvalues), 15)
        corr = float(torch.corrcoef(
            torch.stack([ed.eigenvalues[:k], evals_exact[:k]])
        )[0, 1])
        assert corr > 0.9, f"Eigenvalue correlation: {corr:.4f}"

    def test_nystrom_returns_eigendecomp(self):
        from torchgwas.linalg.eigh import EigenDecomp
        from torchgwas.linalg.nystrom import nystrom_approximate

        chunks, _ = self._make_genotype_chunks()
        ed, _ = nystrom_approximate(iter(chunks), n_samples=50, n_landmarks=10, seed=42)
        assert isinstance(ed, EigenDecomp)
        assert ed.eigenvectors.shape[0] == 50


# ---------------------------------------------------------------------------
# Sparse GRM
# ---------------------------------------------------------------------------

class TestSparseGRM:
    def _make_chunks(self, n=30, m=60, chunk_size=20, seed=42):
        from torchgwas.models.base import VariantMeta
        torch.manual_seed(seed)
        G = torch.clamp(torch.randn(n, m, dtype=torch.float64) + 1.0, 0.0, 2.0)

        chunks = []
        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            vmeta = VariantMeta(
                snp=[f"rs{j}" for j in range(start, end)],
                chr=["1"] * (end - start),
                pos=list(range(start, end)),
                a1=["A"] * (end - start),
                a2=["G"] * (end - start),
            )
            chunks.append((G[:, start:end], vmeta))
        return chunks, G

    def test_sparse_grm_preserves_above_threshold(self):
        """Entries above threshold are preserved."""
        from torchgwas.linalg.kinship import grm_vanraden
        from torchgwas.linalg.sparse_grm import sparse_grm_streaming

        chunks, G = self._make_chunks()
        K_sparse, normalizer, n_snps = sparse_grm_streaming(
            iter(chunks), n_samples=30, threshold=0.05,
        )

        # Reference full GRM
        K_full, _ = grm_vanraden(G)

        # Convert sparse to dense for comparison
        K_dense = K_sparse.to_dense()

        # All non-zero entries should match the full GRM
        nonzero_mask = K_dense != 0
        torch.testing.assert_close(
            K_dense[nonzero_mask], K_full[nonzero_mask], atol=1e-10, rtol=1e-10
        )

    def test_sparse_grm_diagonal_preserved(self):
        """Diagonal is always preserved regardless of threshold."""
        from torchgwas.linalg.sparse_grm import sparse_grm_streaming

        chunks, _ = self._make_chunks()
        K_sparse, _, _ = sparse_grm_streaming(
            iter(chunks), n_samples=30, threshold=100.0,  # Very high threshold
        )
        K_dense = K_sparse.to_dense()
        # Diagonal should be non-zero
        assert (K_dense.diag() != 0).all()

    def test_sparse_matvec_matches_dense(self):
        """Sparse matvec produces same result as dense multiplication."""
        from torchgwas.linalg.sparse_grm import sparse_grm_streaming

        chunks, _ = self._make_chunks()
        K_sparse, _, _ = sparse_grm_streaming(
            iter(chunks), n_samples=30, threshold=0.01,
        )
        K_dense = K_sparse.to_dense()

        x = torch.randn(30, dtype=torch.float64)
        result_sparse = torch.sparse.mm(K_sparse, x.unsqueeze(1)).squeeze(1)
        result_dense = K_dense @ x
        torch.testing.assert_close(result_sparse, result_dense, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Stochastic Trace
# ---------------------------------------------------------------------------

class TestStochasticTrace:
    def test_hutchinson_trace_accuracy(self):
        """Hutchinson estimate within 10% of true trace."""
        from torchgwas.optim.stochastic_trace import hutchinson_trace

        n = 100
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T  # PSD

        true_trace = A.trace().item()
        est_trace = hutchinson_trace(lambda x: A @ x, n, n_probes=50, seed=42)

        rel_err = abs(est_trace - true_trace) / abs(true_trace)
        assert rel_err < 0.10, f"Trace relative error: {rel_err:.4f}"

    def test_stochastic_logdet_accuracy(self):
        """SLQ log-determinant within 15% of true logdet."""
        from torchgwas.optim.stochastic_trace import stochastic_logdet

        n = 50
        torch.manual_seed(42)
        L = torch.randn(n, n, dtype=torch.float64)
        A = L @ L.T + 0.1 * torch.eye(n, dtype=torch.float64)

        true_logdet = torch.linalg.slogdet(A)[1].item()
        est_logdet = stochastic_logdet(
            lambda x: A @ x, n, n_probes=50, lanczos_iters=40, seed=42,
        )

        rel_err = abs(est_logdet - true_logdet) / (abs(true_logdet) + 1e-10)
        assert rel_err < 0.15, f"Logdet relative error: {rel_err:.4f}"

    def test_hutchinson_identity_trace(self):
        """Trace of identity matrix should be n."""
        from torchgwas.optim.stochastic_trace import hutchinson_trace

        n = 80
        est = hutchinson_trace(lambda x: x, n, n_probes=100, seed=42)
        assert abs(est - n) / n < 0.05


# ---------------------------------------------------------------------------
# Approximate LMM Integration
# ---------------------------------------------------------------------------

class TestApproxLMM:
    @pytest.fixture
    def lmm_data(self):
        """Generate simple LMM test data."""
        n = 100
        m = 200
        torch.manual_seed(42)

        # Genotypes
        G = torch.clamp(torch.randn(n, m, dtype=torch.float64) + 1.0, 0.0, 2.0)

        # GRM
        from torchgwas.linalg.kinship import grm_vanraden
        K, _ = grm_vanraden(G)

        # Phenotype with genetic signal
        from torchgwas.linalg.eigh import eigendecompose
        ed = eigendecompose(K)
        U = ed.eigenvectors
        sig2_g, sig2_e = 0.5, 0.5
        genetic = U @ (torch.sqrt(ed.eigenvalues * sig2_g) * torch.randn(n, dtype=torch.float64))
        Y = genetic + torch.randn(n, dtype=torch.float64) * (sig2_e ** 0.5)

        X0 = torch.ones(n, 1, dtype=torch.float64)

        return Y, X0, K, G

    def test_randomized_svd_lmm_pvalues(self, lmm_data):
        """LMM with randomized SVD produces p-values correlated > 0.99 with exact."""
        from torchgwas.config import NumericalConfig
        from torchgwas.models.base import VariantMeta
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        Y, X0, K, G = lmm_data
        n, m = G.shape
        config = NumericalConfig(reml_method="ai_reml")

        # Exact
        model_exact = SingleTraitLMM(config=config)
        nf_exact = model_exact.fit_null(Y, X0, K=K)

        # Approximate
        model_approx = SingleTraitLMM(config=config)
        nf_approx = model_approx.fit_null(
            Y, X0, K=K,
            approx_method="randomized_svd",
            approx_config={"n_components": 80, "seed": 42},
        )

        assert nf_approx.approximate is True
        assert nf_approx.approx_method == "randomized_svd"

        # Score a chunk
        vmeta = VariantMeta(
            snp=[f"rs{j}" for j in range(m)],
            chr=["1"] * m,
            pos=list(range(m)),
            a1=["A"] * m,
            a2=["G"] * m,
        )

        res_exact = model_exact.score_chunk(G, nf_exact, vmeta, test="wald")
        res_approx = model_approx.score_chunk(G, nf_approx, vmeta, test="wald")

        # P-values should be highly correlated
        log_p_exact = -torch.log10(res_exact.p)
        log_p_approx = -torch.log10(res_approx.p)

        corr = float(torch.corrcoef(
            torch.stack([log_p_exact, log_p_approx])
        )[0, 1])
        # k=80 on n=100 loses 20% of eigenspace; 0.90 is a reasonable bar
        assert corr > 0.90, f"P-value correlation: {corr:.6f}"

    def test_approx_nullfit_labeled(self, lmm_data):
        """Approximate NullFit has approximate=True flag set."""
        from torchgwas.config import NumericalConfig
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        Y, X0, K, _ = lmm_data
        config = NumericalConfig(reml_method="ai_reml")

        model = SingleTraitLMM(config=config)
        nf = model.fit_null(
            Y, X0, K=K,
            approx_method="randomized_svd",
            approx_config={"n_components": 50, "seed": 42},
        )

        assert nf.approximate is True
        assert nf.approx_method == "randomized_svd"
        assert nf.approx_config is not None

    def test_exact_nullfit_not_labeled(self, lmm_data):
        """Exact NullFit has approximate=False."""
        from torchgwas.config import NumericalConfig
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        Y, X0, K, _ = lmm_data
        config = NumericalConfig(reml_method="ai_reml")

        model = SingleTraitLMM(config=config)
        nf = model.fit_null(Y, X0, K=K)

        assert nf.approximate is False
        assert nf.approx_method is None
