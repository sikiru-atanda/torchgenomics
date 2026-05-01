"""Tier-1 math-correctness coverage tests for torchgwas.linalg.

Bar (spec section 4.3 Tier 1):
- Closed-form / scipy / Monte-Carlo verification per public function.
- One function in isolation per test. No compositional tests.
- Tolerance <= 1e-10 absolute for closed-form; documented otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from torchgwas.linalg import eigendecompose


pytestmark = pytest.mark.timeout(30)


class TestEigendecompose:
    """``eigendecompose`` should return eigenvalues / eigenvectors of a
    symmetric PSD matrix, matching ``scipy.linalg.eigh`` to machine
    precision."""

    def test_matches_scipy_eigh(self, tiny_kinship):
        """Output eigenvalues and reconstructed matrix agree with scipy."""
        K = tiny_kinship
        out = eigendecompose(K)
        eigvals = out.eigenvalues if hasattr(out, "eigenvalues") else out[0]
        eigvecs = out.eigenvectors if hasattr(out, "eigenvectors") else out[1]

        eigvals_np = eigvals.cpu().numpy()
        eigvecs_np = eigvecs.cpu().numpy()
        K_np = K.cpu().numpy()

        ref_vals, ref_vecs = sla.eigh(K_np)

        # Eigenvalues must agree (sort-order invariant).
        np.testing.assert_allclose(np.sort(eigvals_np), np.sort(ref_vals),
                                    rtol=1e-10, atol=1e-10)

        # Reconstruction K ~ V Lambda V^T.
        K_recon = eigvecs_np @ np.diag(eigvals_np) @ eigvecs_np.T
        np.testing.assert_allclose(K_recon, K_np, rtol=1e-8, atol=1e-8)

    def test_eigenvectors_orthonormal(self, tiny_kinship):
        """Eigenvector matrix must satisfy V^T V = I to machine precision."""
        out = eigendecompose(tiny_kinship)
        V = (out.eigenvectors if hasattr(out, "eigenvectors") else out[1]).cpu().numpy()
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[0]),
                                    rtol=1e-10, atol=1e-10)
