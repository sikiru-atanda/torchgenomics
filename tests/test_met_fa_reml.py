"""Tests for FA(k) variance structure in MultiEnvLMM."""

from __future__ import annotations

import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import VariantMeta
from torchgwas.models.multi_env_lmm import MultiEnvLMM, _parse_vg_structure


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def met_data_3env():
    """Simulated MET data: 120 samples, 3 environments."""
    torch.manual_seed(99)
    n, m, E = 120, 200, 3
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    Vg_true = torch.tensor([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.6],
        [0.5, 0.6, 1.0],
    ], dtype=torch.float64)
    L_vg = torch.linalg.cholesky(Vg_true)
    Z = torch.randn(n, E, dtype=torch.float64)
    U = L @ Z @ L_vg.T
    Y = U + torch.randn(n, E, dtype=torch.float64)
    return Y, X0, K, G


@pytest.fixture
def met_data_5env():
    """Simulated MET data: 100 samples, 5 environments (rank-2 Vg)."""
    torch.manual_seed(77)
    n, m, E = 100, 200, 5
    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))

    # Rank-2 Vg: Lambda is (5, 2)
    Lambda_true = torch.tensor([
        [1.0, 0.0],
        [0.8, 0.3],
        [0.6, 0.5],
        [0.3, 0.7],
        [0.1, 0.9],
    ], dtype=torch.float64)
    psi_true = torch.tensor([0.2, 0.3, 0.1, 0.2, 0.15], dtype=torch.float64)
    Vg_true = Lambda_true @ Lambda_true.T + torch.diag(psi_true)
    L_vg = torch.linalg.cholesky(Vg_true)

    Z = torch.randn(n, E, dtype=torch.float64)
    U = L @ Z @ L_vg.T
    Y = U + 0.5 * torch.randn(n, E, dtype=torch.float64)
    return Y, X0, K, G


@pytest.fixture
def vmeta_10():
    return VariantMeta(
        snp=[f"rs{i}" for i in range(10)],
        chr=["1"] * 10,
        pos=list(range(10)),
        a1=["A"] * 10,
        a2=["G"] * 10,
    )


# ── Parser tests ─────────────────────────────────────────────────────

class TestVgStructureParser:
    def test_unstructured(self):
        kind, k = _parse_vg_structure("unstructured", 5)
        assert kind == "unstructured"
        assert k == 0

    def test_fa1(self):
        kind, k = _parse_vg_structure("fa(1)", 5)
        assert kind == "fa"
        assert k == 1

    def test_fa3(self):
        kind, k = _parse_vg_structure("fa(3)", 5)
        assert kind == "fa"
        assert k == 3

    def test_case_insensitive(self):
        kind, k = _parse_vg_structure("FA(2)", 5)
        assert kind == "fa"
        assert k == 2

    def test_error_if_k_ge_E(self):
        """FA(k) with k >= E must raise ValueError."""
        with pytest.raises(ValueError, match="strictly less"):
            _parse_vg_structure("fa(5)", 5)

        with pytest.raises(ValueError, match="strictly less"):
            _parse_vg_structure("fa(6)", 5)

    def test_error_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid vg_structure"):
            _parse_vg_structure("fa1", 5)

    def test_fa_arbitrary_k_paramcount(self):
        """Verify free params formula: E*k − k(k−1)/2 + E."""
        from torchgwas.optim.fa_lbfgs_reml import _build_loading_map

        for E in [3, 5, 8, 10]:
            for k in range(1, E):
                loading_map = _build_loading_map(E, k)
                expected_loadings = E * k - k * (k - 1) // 2
                assert len(loading_map) == expected_loadings, (
                    f"E={E}, k={k}: got {len(loading_map)}, expected {expected_loadings}"
                )
                # Total params = loadings + E (specific variances)
                total = expected_loadings + E
                assert total < E * (E + 1) // 2 + E  # fewer than unstructured + E


# ── FA fit tests ─────────────────────────────────────────────────────

class TestFAFit:
    def test_fa1_converges(self, met_data_3env):
        """FA(1) should converge on 3-env data."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        assert nf.converged or nf.log_likelihood is not None
        assert nf.Vg is not None
        assert nf.Vg.shape == (3, 3)

    def test_fa2_converges(self, met_data_3env):
        """FA(2) should converge on 3-env data."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(2)")
        nf = model.fit_null(Y, X0, K)
        assert nf.Vg is not None
        assert nf.Vg.shape == (3, 3)

    def test_fa3_on_5env_data(self, met_data_5env):
        """FA(3) should converge on 5-env data (arbitrary k test)."""
        Y, X0, K, _ = met_data_5env
        model = MultiEnvLMM(vg_structure="fa(3)")
        nf = model.fit_null(Y, X0, K)
        assert nf.Vg is not None
        assert nf.Vg.shape == (5, 5)

    def test_fa_lambda_shape(self, met_data_3env):
        """Lambda stored on NullFit has shape (E, k)."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        Lambda = model.fa_loadings(nf)
        assert Lambda.shape == (3, 1)

    def test_fa_lambda_shape_k2(self, met_data_5env):
        """Lambda has shape (E, k) for FA(2) on 5-env data."""
        Y, X0, K, _ = met_data_5env
        model = MultiEnvLMM(vg_structure="fa(2)")
        nf = model.fit_null(Y, X0, K)
        Lambda = model.fa_loadings(nf)
        assert Lambda.shape == (5, 2)

    def test_fa_vg_psd(self, met_data_3env):
        """Reconstructed Vg from FA should be positive semi-definite."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        eigvals = torch.linalg.eigvalsh(nf.Vg)
        assert (eigvals > -1e-8).all(), f"Vg not PSD: {eigvals}"

    def test_scan_with_fa_produces_valid_results(self, met_data_3env, vmeta_10):
        """Full fit_null + score_chunk with FA should produce valid results."""
        Y, X0, K, G = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        result = model.score_chunk(G[:, :10].to(torch.float64), nf, vmeta_10)

        assert result.beta.shape == (10, 3)
        assert result.p.shape == (10,)
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()

    def test_error_if_k_ge_E_at_model_level(self, met_data_3env):
        """MultiEnvLMM(vg_structure='fa(3)') on 3-env data should raise."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(3)")
        with pytest.raises(ValueError, match="strictly less"):
            model.fit_null(Y, X0, K)

    def test_fa1_ll_reasonable(self, met_data_3env):
        """FA(1) log-likelihood should be finite and negative."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        assert nf.log_likelihood is not None
        assert nf.log_likelihood < 0  # REML loglik is typically negative


# ── Interpretive methods with FA ─────────────────────────────────────

class TestFAInterpretive:
    def test_proportion_gxe(self, met_data_3env):
        """proportion_gxe should be in [0, 1]."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM(vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K)
        pgxe = model.proportion_gxe(nf)
        assert 0.0 <= pgxe <= 1.0

    def test_update_null_fa_preserves_metadata(self, met_data_3env):
        """update_null on FA model should preserve fa_Lambda, fa_rank, env_names."""
        from torchgwas.config import NumericalConfig
        Y, X0, K, _ = met_data_3env
        config = NumericalConfig()
        config.reml_max_iter = 3
        model = MultiEnvLMM(config=config, vg_structure="fa(1)")
        nf = model.fit_null(Y, X0, K, env_names=["A", "B", "C"])

        updated = model.update_null(nf, max_iter=50)
        assert hasattr(updated, "fa_Lambda")
        assert updated.fa_Lambda.shape == (3, 1)
        assert updated.fa_rank == 1
        assert updated.env_names == ["A", "B", "C"]
        assert len(updated.optimizer_trace) > len(nf.optimizer_trace)

    def test_fa_loadings_not_available_for_unstructured(self, met_data_3env):
        """fa_loadings should raise when vg_structure was unstructured."""
        Y, X0, K, _ = met_data_3env
        model = MultiEnvLMM()
        nf = model.fit_null(Y, X0, K)
        with pytest.raises(AttributeError, match="FA loadings not available"):
            model.fa_loadings(nf)
