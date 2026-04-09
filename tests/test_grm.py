"""Phase 3: GRM computation, eigendecomposition, and linalg utility tests."""

from __future__ import annotations

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden, grm_vanraden_streaming
from torchgwas.linalg.kinship_polyploid import grm_loco, grm_polyploid_gene_action
from torchgwas.linalg.eigh import EigenDecomp, eigendecompose, rotate, compute_weights
from torchgwas.linalg.safe import safe_cholesky, safe_logdet, symmetrize
from torchgwas.linalg.woodbury import woodbury_inverse
from torchgwas.preprocess.impute import impute_mean


@pytest.fixture
def G_clean(rng):
    """50 samples, 200 SNPs, no missing, float64."""
    G = torch.randint(0, 3, (50, 200), generator=rng, dtype=torch.float64)
    return G


@pytest.fixture
def K(G_clean):
    K, _meta = grm_vanraden(G_clean)
    return K


class TestGRMVanRaden:
    def test_grm_shape(self, G_clean, K):
        assert K.shape == (50, 50)

    def test_grm_symmetric(self, K):
        torch.testing.assert_close(K, K.T)

    def test_grm_positive_semidefinite(self, K):
        evals = torch.linalg.eigvalsh(K)
        # Allow small negative values from numerical noise
        assert evals.min() > -1e-6

    def test_grm_diagonal_near_one(self, K):
        diag = torch.diagonal(K)
        # For unrelated, random samples, diagonal should be ~1
        assert diag.mean().item() == pytest.approx(1.0, abs=0.5)

    def test_grm_dtype_float64(self, K):
        assert K.dtype == torch.float64

    def test_grm_deterministic(self, G_clean):
        """Same input produces identical GRM."""
        K1, _ = grm_vanraden(G_clean)
        K2, _ = grm_vanraden(G_clean)
        torch.testing.assert_close(K1, K2)

    def test_grm_centering(self, G_clean):
        """GRM row/col means should be approximately zero for well-behaved data."""
        K, _ = grm_vanraden(G_clean)
        # Off-diagonal mean should be near zero for random samples
        n = K.shape[0]
        off_diag = K[~torch.eye(n, dtype=torch.bool)]
        assert abs(off_diag.mean().item()) < 0.5


class TestGRMStreaming:
    def test_streaming_matches_full(self, G_clean):
        """Streaming GRM matches single-pass GRM."""
        K_full, _ = grm_vanraden(G_clean)

        # Simulate chunks
        def chunk_iter():
            chunk_size = 50
            for start in range(0, G_clean.shape[1], chunk_size):
                end = min(start + chunk_size, G_clean.shape[1])
                yield G_clean[:, start:end], None

        K_stream, meta = grm_vanraden_streaming(
            chunk_iter(), n_samples=G_clean.shape[0], ploidy=2,
        )

        # Streaming uses FP32 GEMM + FP64 accumulator per charter, so expect ~1e-6 error
        torch.testing.assert_close(K_full, K_stream, atol=1e-5, rtol=1e-5)
        assert meta.method == "vanraden_streaming"
        assert meta.n_snps_used == G_clean.shape[1]


class TestGRMPolyploid:
    def test_grm_tetraploid(self):
        """Polyploid GRM uses ploidy-adjusted allele frequencies."""
        torch.manual_seed(42)
        G = torch.randint(0, 5, (20, 50), dtype=torch.float64)
        K, _ = grm_vanraden(G, ploidy=4)
        assert K.shape == (20, 20)
        torch.testing.assert_close(K, K.T)

    def test_grm_gene_action_additive(self):
        """Additive gene-action GRM equals standard GRM."""
        torch.manual_seed(42)
        G = torch.randint(0, 5, (20, 50), dtype=torch.float64)
        K_std, _ = grm_vanraden(G, ploidy=4)
        K_add, _ = grm_polyploid_gene_action(G, "additive", ploidy=4)
        torch.testing.assert_close(K_std, K_add)

    def test_grm_gene_action_1dom(self):
        """1-dom GRM is different from additive GRM."""
        torch.manual_seed(42)
        G = torch.randint(0, 5, (20, 50), dtype=torch.float64)
        K_add, _ = grm_polyploid_gene_action(G, "additive", ploidy=4)
        K_dom, _ = grm_polyploid_gene_action(G, "1-dom", ploidy=4)
        # Should not be identical
        assert not torch.allclose(K_add, K_dom)

    def test_grm_loco(self):
        """LOCO GRM excludes target chromosome."""
        torch.manual_seed(42)
        G = torch.randint(0, 3, (20, 60), dtype=torch.float64)
        chr_labels = ["1"] * 20 + ["2"] * 20 + ["3"] * 20

        K_full, _ = grm_vanraden(G)
        K_loco, meta = grm_loco(G, chr_labels, exclude_chr="1")

        # LOCO should use 40 variants (chr 2+3), not 60
        assert K_loco.shape == (20, 20)
        assert meta.loco_chr == "1"
        assert meta.n_snps_used == 40
        # Should differ from full GRM
        assert not torch.allclose(K_full, K_loco)

    def test_grm_loco_all_excluded_raises(self):
        """Raises when all variants are on excluded chromosome."""
        G = torch.randint(0, 3, (10, 20), dtype=torch.float64)
        chr_labels = ["1"] * 20
        with pytest.raises(ValueError, match="No variants remain"):
            grm_loco(G, chr_labels, exclude_chr="1")


class TestASVTransform:
    """Tests for grm_asv_transform (Feldmann 2022)."""

    def test_average_diagonal_is_one(self, K):
        from torchgwas.linalg.kinship_polyploid import grm_asv_transform
        K_asv = grm_asv_transform(K)
        avg_diag = K_asv.diagonal().mean().item()
        assert avg_diag == pytest.approx(1.0, abs=0.05)

    def test_preserves_symmetry(self, K):
        from torchgwas.linalg.kinship_polyploid import grm_asv_transform
        K_asv = grm_asv_transform(K)
        torch.testing.assert_close(K_asv, K_asv.T)

    def test_single_sample(self):
        from torchgwas.linalg.kinship_polyploid import grm_asv_transform
        K = torch.tensor([[2.0]], dtype=torch.float64)
        K_asv = grm_asv_transform(K)
        torch.testing.assert_close(K_asv, K)  # n=1, returned as-is


class TestEpistaticHadamard:
    """Tests for grm_epistatic_hadamard."""

    def test_additive_only(self, K):
        from torchgwas.linalg.kinship_polyploid import grm_epistatic_hadamard
        result = grm_epistatic_hadamard(K)
        assert "K_aa" in result
        assert "K_dd" not in result
        torch.testing.assert_close(result["K_aa"], K * K)

    def test_with_dominance(self, K):
        from torchgwas.linalg.kinship_polyploid import grm_epistatic_hadamard
        K_dom = torch.eye(K.shape[0], dtype=torch.float64) * 0.5
        result = grm_epistatic_hadamard(K, K_dom)
        assert "K_aa" in result
        assert "K_dd" in result
        assert "K_ad" in result
        torch.testing.assert_close(result["K_dd"], K_dom * K_dom)
        torch.testing.assert_close(result["K_ad"], K * K_dom)

    def test_shape_mismatch_raises(self):
        from torchgwas.linalg.kinship_polyploid import grm_epistatic_hadamard
        K1 = torch.eye(5, dtype=torch.float64)
        K2 = torch.eye(3, dtype=torch.float64)
        with pytest.raises(ValueError, match="does not match"):
            grm_epistatic_hadamard(K1, K2)


class TestEigendecomposition:
    def test_eigendecompose_reconstructs_grm(self, K):
        """U @ diag(D) @ U.T ≈ K within tolerance."""
        ed = eigendecompose(K)
        K_recon = ed.eigenvectors @ torch.diag(ed.eigenvalues) @ ed.eigenvectors.T
        torch.testing.assert_close(K_recon, K, atol=1e-8, rtol=1e-8)

    def test_eigenvalues_sorted_descending(self, K):
        ed = eigendecompose(K)
        diffs = ed.eigenvalues[:-1] - ed.eigenvalues[1:]
        assert torch.all(diffs >= -1e-10)  # descending

    def test_eigenvalue_floor_applied(self):
        """Eigenvalues below floor are clamped."""
        # Create a rank-deficient matrix
        X = torch.randn(10, 3, dtype=torch.float64)
        K = X @ X.T  # rank 3, so 7 eigenvalues should be ~0
        ed = eigendecompose(K, eigenvalue_floor=1e-10)
        assert torch.all(ed.eigenvalues >= 1e-10)

    def test_truncated_eigendecompose(self, K):
        """Truncated decomposition returns requested number of components."""
        ed = eigendecompose(K, n_components=10)
        assert ed.eigenvalues.shape == (10,)
        assert ed.eigenvectors.shape == (50, 10)

    def test_eigenvectors_orthonormal(self, K):
        ed = eigendecompose(K)
        I_approx = ed.eigenvectors.T @ ed.eigenvectors
        torch.testing.assert_close(
            I_approx, torch.eye(50, dtype=torch.float64), atol=1e-10, rtol=1e-10,
        )


class TestRotation:
    def test_rotate_shape(self, K):
        ed = eigendecompose(K)
        Y = torch.randn(50, 1, dtype=torch.float64)
        Y_rot = rotate(Y, ed.eigenvectors)
        assert Y_rot.shape == (50, 1)

    def test_rotate_preserves_norm(self, K):
        """Rotation by orthogonal matrix preserves Frobenius norm."""
        ed = eigendecompose(K)
        Y = torch.randn(50, 3, dtype=torch.float64)
        Y_rot = rotate(Y, ed.eigenvectors)
        torch.testing.assert_close(
            torch.norm(Y), torch.norm(Y_rot), atol=1e-10, rtol=1e-10,
        )


class TestComputeWeights:
    def test_weights_positive(self, K):
        ed = eigendecompose(K)
        w = compute_weights(ed.eigenvalues, sig2_g=0.5, sig2_e=0.5)
        assert torch.all(w > 0)

    def test_weights_shape(self, K):
        ed = eigendecompose(K)
        w = compute_weights(ed.eigenvalues, sig2_g=0.3, sig2_e=0.7)
        assert w.shape == (50,)


class TestSafeCholesky:
    def test_cholesky_spd(self):
        """Safe Cholesky works on well-conditioned SPD matrix."""
        A = torch.eye(5, dtype=torch.float64) + 0.1 * torch.randn(5, 5, dtype=torch.float64)
        A = A @ A.T  # make SPD
        L = safe_cholesky(A)
        torch.testing.assert_close(L @ L.T, A, atol=1e-10, rtol=1e-10)

    def test_cholesky_near_singular(self):
        """Safe Cholesky handles near-singular matrices with jitter."""
        X = torch.randn(10, 3, dtype=torch.float64)
        K = X @ X.T  # rank 3 — not full rank
        L = safe_cholesky(K)
        assert L.shape == (10, 10)

    def test_logdet(self):
        A = torch.eye(5, dtype=torch.float64) * 2.0
        logdet = safe_logdet(A)
        expected = 5 * torch.log(torch.tensor(2.0, dtype=torch.float64))
        torch.testing.assert_close(logdet, expected, atol=1e-10, rtol=1e-10)


class TestWoodbury:
    def test_woodbury_identity(self):
        """Woodbury inverse matches direct inverse."""
        torch.manual_seed(42)
        n, k = 10, 2
        A = torch.eye(n, dtype=torch.float64) * 3.0
        U = torch.randn(n, k, dtype=torch.float64)
        C = torch.eye(k, dtype=torch.float64)
        V = torch.randn(k, n, dtype=torch.float64)

        # Direct inverse
        M = A + U @ C @ V
        M_inv_direct = torch.linalg.inv(M)

        # Woodbury
        A_inv = torch.linalg.inv(A)
        M_inv_woodbury = woodbury_inverse(A_inv, U, C, V)

        torch.testing.assert_close(M_inv_direct, M_inv_woodbury, atol=1e-8, rtol=1e-8)
