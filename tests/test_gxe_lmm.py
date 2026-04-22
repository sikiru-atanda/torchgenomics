"""Phase 16: GxE-LMM tests — HetLMM, GxELMM, GxE LBFGS REML, GxEScanResult."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models.base import BaseModel, NullFit, VariantMeta
from torchgwas.models.lmm_gxe import GxELMM, GxEScanResult, HetLMM
from torchgwas.optim.gxe_lbfgs_reml import gxe_lbfgs_reml

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def het_data():
    """Simulate single-trait GxE data.

    y = 5 + 1.0*env + u + e, with u ~ N(0, 0.3*K), e ~ N(0, 0.7*I).
    n=150 samples, m=300 SNPs.
    """
    torch.manual_seed(42)
    n, m = 150, 300

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)

    env = torch.randn(n, dtype=torch.float64)
    # Include env as covariate (proper GxE analysis requires env in null model)
    X0 = torch.stack([torch.ones(n, dtype=torch.float64), env], dim=1)  # (n, 2)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))

    Y = 5.0 + 1.0 * env + L_V @ torch.randn(n, dtype=torch.float64)

    return {
        "G": G, "Y": Y, "X0": X0, "K": K, "env": env,
        "n": n, "m": m,
    }


@pytest.fixture
def het_data_with_gxe():
    """Simulate data with known GxE interaction at SNP 0.

    y = 5 + 1.5*g_0 + 2.0*(g_0*env) + u + e
    """
    torch.manual_seed(123)
    n, m = 200, 400

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)

    env = torch.randn(n, dtype=torch.float64)
    X0 = torch.stack([torch.ones(n, dtype=torch.float64), env], dim=1)  # (n, 2)

    sig2_g, sig2_e = 0.20, 0.80
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))

    g0 = G[:, 0] - G[:, 0].mean()
    Y = 5.0 + 1.5 * g0 + 2.0 * (g0 * env) + L_V @ torch.randn(n, dtype=torch.float64)

    return {
        "G": G, "Y": Y, "X0": X0, "K": K, "env": env,
        "n": n, "m": m,
    }


@pytest.fixture
def multi_trait_gxe_data():
    """Simulate multi-trait GxE data (d=2 traits, n=120, m=200)."""
    torch.manual_seed(999)
    n, m, d = 120, 200, 2

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)

    X0 = torch.ones(n, 1, dtype=torch.float64)
    env = torch.randn(n, dtype=torch.float64)

    # Variance components
    Vg = torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=torch.float64)
    Vge = torch.tensor([[0.1, 0.02], [0.02, 0.08]], dtype=torch.float64)
    Ve = torch.tensor([[0.5, 0.05], [0.05, 0.6]], dtype=torch.float64)

    env_z = (env - env.mean()) / env.std()
    env_cov = env_z ** 2

    # Build full covariance: V = K⊗Vg + diag(env_cov)*K⊗Vge + I⊗Ve
    V_full = torch.zeros(n * d, n * d, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            block = K[i, j] * Vg + env_cov[i].item() * K[i, j] * Vge
            if i == j:
                block = block + Ve
            V_full[i*d:(i+1)*d, j*d:(j+1)*d] = block

    L_full = torch.linalg.cholesky(V_full + 1e-5 * torch.eye(n * d, dtype=torch.float64))
    z = torch.randn(n * d, dtype=torch.float64)
    Y_vec = 3.0 + L_full @ z
    Y = Y_vec.reshape(n, d)

    return {
        "G": G, "Y": Y, "X0": X0, "K": K, "env": env,
        "n": n, "m": m, "d": d,
        "Vg_true": Vg, "Vge_true": Vge, "Ve_true": Ve,
    }


@pytest.fixture
def vmeta_het(het_data):
    m = het_data["m"]
    return VariantMeta(
        snp=[f"snp_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


@pytest.fixture
def vmeta_gxe(het_data_with_gxe):
    m = het_data_with_gxe["m"]
    return VariantMeta(
        snp=[f"snp_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


@pytest.fixture
def vmeta_multi(multi_trait_gxe_data):
    m = multi_trait_gxe_data["m"]
    return VariantMeta(
        snp=[f"snp_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


# ---------------------------------------------------------------
# TestHetLMM
# ---------------------------------------------------------------

class TestHetLMM:
    """Tests for single-trait GxE interaction LMM (HetLMM)."""

    def test_implements_base_model(self):
        """HetLMM should satisfy the BaseModel protocol."""
        model = HetLMM()
        assert isinstance(model, BaseModel)

    def test_fit_null_returns_nullfit(self, het_data):
        """fit_null should return a NullFit with expected fields."""
        model = HetLMM()
        nf = model.fit_null(
            het_data["Y"], het_data["X0"], K=het_data["K"],
            env=het_data["env"],
        )
        assert isinstance(nf, NullFit)
        assert nf.sig2_g is not None and nf.sig2_g > 0
        assert nf.sig2_e is not None and nf.sig2_e > 0
        assert hasattr(nf, "env_rot")
        assert hasattr(nf, "env_z")
        assert hasattr(nf, "Py")
        assert hasattr(nf, "M00")

    def test_null_calibration(self, het_data, vmeta_het):
        """Under null (no SNP effect), p-values should be roughly uniform.

        Lambda_gc should be close to 1.0.
        """
        torch.manual_seed(55)
        # Use held-out SNPs for scanning (not in GRM)
        n = het_data["n"]
        G_scan = torch.randint(0, 3, (n, 200), dtype=torch.float64)
        vmeta_scan = VariantMeta(
            snp=[f"null_snp_{i}" for i in range(200)],
            chr=["1"] * 200,
            pos=list(range(200)),
            a1=["A"] * 200,
            a2=["G"] * 200,
        )

        model = HetLMM()
        nf = model.fit_null(
            het_data["Y"], het_data["X0"], K=het_data["K"],
            env=het_data["env"],
        )

        result = model.score_chunk(G_scan, nf, vmeta_scan)

        # Lambda_gc: median chi²(1) / 0.4549
        stat_np = result.stat_main.detach().cpu().numpy()
        lambda_gc = float(np.median(stat_np)) / 0.4549
        assert 0.70 <= lambda_gc <= 1.50, f"lambda_gc = {lambda_gc:.3f}"

    def test_detects_main_effect(self, het_data_with_gxe, vmeta_gxe):
        """Planted main effect at SNP 0 should be detected."""
        model = HetLMM()
        data = het_data_with_gxe
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        result = model.score_chunk(data["G"], nf, vmeta_gxe)

        # SNP 0 should have small p_main
        assert result.p_main[0].item() < 0.01, (
            f"Main effect p = {result.p_main[0].item():.4e}"
        )

    def test_detects_interaction_effect(self, het_data_with_gxe, vmeta_gxe):
        """Planted GxE interaction at SNP 0 should be detected."""
        model = HetLMM()
        data = het_data_with_gxe
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        result = model.score_chunk(data["G"], nf, vmeta_gxe)

        # SNP 0 should have small p_interact
        assert result.p_interact[0].item() < 0.01, (
            f"Interaction effect p = {result.p_interact[0].item():.4e}"
        )

    def test_joint_test_valid(self, het_data_with_gxe, vmeta_gxe):
        """Joint test should also detect the planted signal."""
        model = HetLMM()
        data = het_data_with_gxe
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        result = model.score_chunk(data["G"], nf, vmeta_gxe)

        # Joint test at SNP 0 should be significant
        assert result.p_joint[0].item() < 0.01

    def test_no_interaction_pvalues_not_extreme(self, het_data, vmeta_het):
        """When GxE=0, interaction p-values should not be systematically small."""
        model = HetLMM()
        nf = model.fit_null(
            het_data["Y"], het_data["X0"], K=het_data["K"],
            env=het_data["env"],
        )
        result = model.score_chunk(het_data["G"], nf, vmeta_het)

        # Median interaction p-value should not be extremely small
        median_p = result.p_interact.median().item()
        assert median_p > 0.05, f"Median interaction p = {median_p:.4e}"

    def test_valid_pvalue_ranges(self, het_data, vmeta_het):
        """All p-values should be in (0, 1], no NaN."""
        model = HetLMM()
        nf = model.fit_null(
            het_data["Y"], het_data["X0"], K=het_data["K"],
            env=het_data["env"],
        )
        result = model.score_chunk(het_data["G"], nf, vmeta_het)

        for name in ["p_main", "p_interact", "p_joint"]:
            p = getattr(result, name)
            assert (p > 0).all(), f"{name}: some p <= 0"
            assert (p <= 1.0).all(), f"{name}: some p > 1"
            assert not torch.isnan(p).any(), f"{name}: NaN detected"

    def test_requires_env(self, het_data):
        """fit_null should raise ValueError if env is not provided."""
        model = HetLMM()
        with pytest.raises(ValueError, match="env"):
            model.fit_null(het_data["Y"], het_data["X0"], K=het_data["K"])


# ---------------------------------------------------------------
# TestGxELMM
# ---------------------------------------------------------------

class TestGxELMM:
    """Tests for multi-trait GxE interaction LMM (GxELMM)."""

    def test_implements_base_model(self):
        """GxELMM should satisfy the BaseModel protocol."""
        model = GxELMM()
        assert isinstance(model, BaseModel)

    def test_fit_null_three_variance_components(self, multi_trait_gxe_data):
        """fit_null should estimate Vg, Vge, Ve."""
        model = GxELMM()
        data = multi_trait_gxe_data
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        assert isinstance(nf, NullFit)
        assert nf.Vg is not None
        assert nf.Ve is not None
        assert hasattr(nf, "Vge")
        assert nf.Vge is not None
        # All should be (d, d)
        d = data["d"]
        assert nf.Vg.shape == (d, d)
        assert nf.Vge.shape == (d, d)
        assert nf.Ve.shape == (d, d)

    def test_variance_components_spd(self, multi_trait_gxe_data):
        """Estimated Vg, Vge, Ve should all be symmetric positive definite."""
        model = GxELMM()
        data = multi_trait_gxe_data
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        for name, V in [("Vg", nf.Vg), ("Vge", nf.Vge), ("Ve", nf.Ve)]:
            # Symmetric
            assert torch.allclose(V, V.T, atol=1e-10), f"{name} not symmetric"
            # Positive eigenvalues
            eigs = torch.linalg.eigvalsh(V)
            assert (eigs > -1e-8).all(), f"{name} not PSD: eigs={eigs}"

    def test_score_chunk_valid_results(self, multi_trait_gxe_data, vmeta_multi):
        """score_chunk should produce valid p-values."""
        model = GxELMM()
        data = multi_trait_gxe_data
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        result = model.score_chunk(data["G"], nf, vmeta_multi)

        assert isinstance(result, GxEScanResult)
        m = data["m"]
        assert result.p_main.shape[0] == m
        assert result.p_interact.shape[0] == m
        assert result.p_joint.shape[0] == m

        for name in ["p_main", "p_interact", "p_joint"]:
            p = getattr(result, name)
            assert (p > 0).all(), f"{name}: some p <= 0"
            assert (p <= 1.0).all(), f"{name}: some p > 1"
            assert not torch.isnan(p).any(), f"{name}: NaN detected"

    def test_requires_multiple_traits(self, het_data):
        """GxELMM should reject Y with d < 2."""
        model = GxELMM()
        Y_1d = het_data["Y"].unsqueeze(1)  # (n, 1)
        with pytest.raises(ValueError, match="2 traits"):
            model.fit_null(
                Y_1d, het_data["X0"], K=het_data["K"],
                env=het_data["env"],
            )

    def test_requires_env(self, multi_trait_gxe_data):
        """fit_null should raise ValueError if env is not provided."""
        model = GxELMM()
        data = multi_trait_gxe_data
        with pytest.raises(ValueError, match="env"):
            model.fit_null(data["Y"], data["X0"], K=data["K"])

    def test_partitioned_variance(self, multi_trait_gxe_data):
        """Variance component proportions should be reasonable."""
        model = GxELMM()
        data = multi_trait_gxe_data
        nf = model.fit_null(
            data["Y"], data["X0"], K=data["K"], env=data["env"],
        )
        # Total variance trace
        total = torch.trace(nf.Vg) + torch.trace(nf.Vge) + torch.trace(nf.Ve)
        assert total > 0
        # Each component should be a non-negligible fraction
        # (at least for simulated data with all three components)
        for name, V in [("Vg", nf.Vg), ("Ve", nf.Ve)]:
            prop = torch.trace(V).item() / total.item()
            assert prop > 0.01, f"{name} proportion too small: {prop:.4f}"


# ---------------------------------------------------------------
# TestGxELBFGSREML
# ---------------------------------------------------------------

class TestGxELBFGSREML:
    """Tests for the 3-component GxE LBFGS REML optimizer."""

    def test_convergence(self, multi_trait_gxe_data):
        """Optimizer should converge and return non-empty trace."""
        from torchgwas.linalg.eigh import eigendecompose, rotate

        data = multi_trait_gxe_data
        ed = eigendecompose(data["K"])
        Y_rot = rotate(data["Y"], ed.eigenvectors)
        X0_rot = rotate(data["X0"], ed.eigenvectors)
        env_z = (data["env"] - data["env"].mean()) / data["env"].std()
        env_cov = env_z ** 2

        Vg, Vge, Ve, ll, trace = gxe_lbfgs_reml(
            Y_rot, X0_rot, ed.eigenvalues, env_cov,
            n_traits=data["d"],
            max_iter=50,
        )
        assert len(trace) > 0
        assert math.isfinite(ll)
        # LL should improve overall
        lls = [t["ll"] for t in trace]
        assert lls[-1] >= lls[0] - 1.0

    def test_positive_definite(self, multi_trait_gxe_data):
        """All estimated covariances should be SPD."""
        from torchgwas.linalg.eigh import eigendecompose, rotate

        data = multi_trait_gxe_data
        ed = eigendecompose(data["K"])
        Y_rot = rotate(data["Y"], ed.eigenvectors)
        X0_rot = rotate(data["X0"], ed.eigenvectors)
        env_z = (data["env"] - data["env"].mean()) / data["env"].std()
        env_cov = env_z ** 2

        Vg, Vge, Ve, ll, trace = gxe_lbfgs_reml(
            Y_rot, X0_rot, ed.eigenvalues, env_cov,
            n_traits=data["d"],
            max_iter=50,
        )
        for name, V in [("Vg", Vg), ("Vge", Vge), ("Ve", Ve)]:
            assert V.shape == (data["d"], data["d"])
            eigs = torch.linalg.eigvalsh(V)
            assert (eigs > -1e-8).all(), f"{name} not PSD: {eigs}"

    def test_returns_correct_shapes(self, multi_trait_gxe_data):
        """Return types and shapes should be correct."""
        from torchgwas.linalg.eigh import eigendecompose, rotate

        data = multi_trait_gxe_data
        ed = eigendecompose(data["K"])
        Y_rot = rotate(data["Y"], ed.eigenvectors)
        X0_rot = rotate(data["X0"], ed.eigenvectors)
        env_z = (data["env"] - data["env"].mean()) / data["env"].std()
        env_cov = env_z ** 2

        Vg, Vge, Ve, ll, trace = gxe_lbfgs_reml(
            Y_rot, X0_rot, ed.eigenvalues, env_cov,
            n_traits=data["d"],
            max_iter=20,
        )
        d = data["d"]
        assert isinstance(Vg, torch.Tensor) and Vg.shape == (d, d)
        assert isinstance(Vge, torch.Tensor) and Vge.shape == (d, d)
        assert isinstance(Ve, torch.Tensor) and Ve.shape == (d, d)
        assert isinstance(ll, float)
        assert isinstance(trace, list)


# ---------------------------------------------------------------
# TestGxEScanResult
# ---------------------------------------------------------------

class TestGxEScanResult:
    """Tests for GxEScanResult dataclass."""

    def test_len(self):
        m = 10
        result = GxEScanResult(
            chr=["1"] * m,
            pos=list(range(m)),
            snp=[f"s{i}" for i in range(m)],
            a1=["A"] * m,
            a2=["G"] * m,
            af=torch.zeros(m),
            beta_main=torch.zeros(m),
            se_main=torch.ones(m),
            stat_main=torch.zeros(m),
            p_main=torch.ones(m),
            beta_interact=torch.zeros(m),
            se_interact=torch.ones(m),
            stat_interact=torch.zeros(m),
            p_interact=torch.ones(m),
            stat_joint=torch.zeros(m),
            p_joint=torch.ones(m),
        )
        assert len(result) == m
