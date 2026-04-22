"""Tests for multi-kernel MET support in MultiEnvLMM."""

from __future__ import annotations

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import VariantMeta
from torchgwas.models.multi_env_lmm import MultiEnvLMM

# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def multi_kernel_data():
    """Simulated data with additive + dominance kernels, 3 environments."""
    torch.manual_seed(55)
    n, m, E = 80, 150, 3
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K_add, _ = grm_vanraden(G)

    # Dominance kernel: recode to 0/1/0 for heterozygosity
    G_dom = (G == 1).to(torch.float64)
    G_dom_cent = G_dom - G_dom.mean(dim=0, keepdim=True)
    K_dom = G_dom_cent @ G_dom_cent.T / m
    K_dom = K_dom + 1e-4 * torch.eye(n, dtype=torch.float64)

    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Simulate phenotype with additive genetic effects
    L = torch.linalg.cholesky(K_add + 1e-6 * torch.eye(n, dtype=torch.float64))
    Z = torch.randn(n, E, dtype=torch.float64)
    U = L @ Z * 0.5
    Y = U + torch.randn(n, E, dtype=torch.float64)

    return Y, X0, K_add, K_dom, G


@pytest.fixture
def vmeta_10():
    return VariantMeta(
        snp=[f"rs{i}" for i in range(10)],
        chr=["1"] * 10,
        pos=list(range(10)),
        a1=["A"] * 10,
        a2=["G"] * 10,
    )


# ── Tests ─────────────────────────────────────────────────────────────

class TestMultiKernel:
    def test_multi_kernel_fit_null_converges(self, multi_kernel_data):
        """2 kernels (additive + dominance), E=3, should produce valid fit."""
        Y, X0, K_add, K_dom, _ = multi_kernel_data
        model = MultiEnvLMM()
        K_dict = {"additive": K_add, "dominance": K_dom}
        nf = model.fit_null(Y, X0, K_dict)
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert nf.log_likelihood is not None

    def test_multi_kernel_scan_shapes(self, multi_kernel_data, vmeta_10):
        """score_chunk output shapes should be correct with multi-kernel."""
        Y, X0, K_add, K_dom, G = multi_kernel_data
        model = MultiEnvLMM()
        K_dict = {"additive": K_add, "dominance": K_dom}
        nf = model.fit_null(Y, X0, K_dict)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        E = 3
        assert result.beta.shape == (10, E)
        assert result.se.shape == (10, E)
        assert result.stat.shape == (10,)
        assert result.p.shape == (10,)
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()

    def test_multi_kernel_vg_per_kernel(self, multi_kernel_data):
        """Each kernel should get its own Vg matrix stored on NullFit."""
        Y, X0, K_add, K_dom, _ = multi_kernel_data
        model = MultiEnvLMM()
        K_dict = {"additive": K_add, "dominance": K_dom}
        nf = model.fit_null(Y, X0, K_dict)
        assert hasattr(nf, "Vg_dict")
        assert "additive" in nf.Vg_dict
        assert "dominance" in nf.Vg_dict
        assert nf.Vg_dict["additive"].shape == (3, 3)
        assert nf.Vg_dict["dominance"].shape == (3, 3)

    def test_single_kernel_dict_equivalent(self, multi_kernel_data, vmeta_10):
        """K={'add': K_a} (single-entry dict) should work like K=K_a."""
        Y, X0, K_add, _, G = multi_kernel_data

        # Single tensor
        model1 = MultiEnvLMM()
        nf1 = model1.fit_null(Y, X0, K_add)
        r1 = model1.score_chunk(G[:, :10].to(torch.float64), nf1, vmeta_10)

        # Single-entry dict (falls through to EED path, not multi-kernel)
        model2 = MultiEnvLMM()
        nf2 = model2.fit_null(Y, X0, {"additive": K_add})
        r2 = model2.score_chunk(G[:, :10].to(torch.float64), nf2, vmeta_10)

        # Joint p-values should be very similar (same underlying model)
        assert torch.allclose(r1.p, r2.p, atol=1e-2), (
            f"Single-tensor vs single-dict p-values differ: "
            f"max diff = {(r1.p - r2.p).abs().max()}"
        )

    def test_multi_kernel_reaction_norm(self, multi_kernel_data, vmeta_10):
        """Multi-kernel with reaction_norm parameterization should produce alpha/delta."""
        Y, X0, K_add, K_dom, G = multi_kernel_data
        model = MultiEnvLMM(parameterization="reaction_norm")
        K_dict = {"additive": K_add, "dominance": K_dom}
        nf = model.fit_null(Y, X0, K_dict)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        assert result.alpha is not None
        assert result.alpha.shape == (10,)
        assert result.delta is not None
        assert result.delta.shape == (10, 3)
        # δ sums to zero
        delta_sum = result.delta.sum(dim=-1)
        assert torch.allclose(delta_sum, torch.zeros_like(delta_sum), atol=1e-10)
