"""Phase 6: GLM tests — OLS null fit, residualized Wald scan."""

from __future__ import annotations

import pytest
import torch

from torchgwas.models.base import NullFit, ScanResult, VariantMeta
from torchgwas.models.glm import GLM


@pytest.fixture
def glm_data():
    """Simulate Y = X0 @ b + G_signal @ beta_true + noise.

    n=200 samples, m=100 SNPs, 3 causal SNPs with known effects.
    """
    torch.manual_seed(123)
    n, m = 200, 100

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    # Causal SNPs with known effect
    beta_true = torch.zeros(m, dtype=torch.float64)
    beta_true[5] = 1.5
    beta_true[20] = -1.0
    beta_true[50] = 0.8

    Y = X0 @ torch.tensor([[3.0]], dtype=torch.float64).squeeze(1) + G @ beta_true
    Y = Y + torch.randn(n, dtype=torch.float64) * 0.5

    return {"Y": Y, "X0": X0, "G": G, "n": n, "m": m}


@pytest.fixture
def glm_null_data():
    """Null data with no signal — for p-value uniformity test."""
    torch.manual_seed(456)
    n, m = 300, 200

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    Y = torch.randn(n, dtype=torch.float64) * 2.0 + 5.0

    return {"Y": Y, "X0": X0, "G": G, "n": n, "m": m}


def _make_vmeta(m: int) -> VariantMeta:
    return VariantMeta(
        snp=[f"snp_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1, m + 1)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


class TestGLMNullFit:
    def test_fit_null_returns_nullfit(self, glm_data):
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        assert isinstance(nf, NullFit)
        assert nf.sig2_e is not None
        assert nf.sig2_e > 0
        assert nf.converged is True

    def test_fit_null_2d_Y(self, glm_data):
        """Y with shape (n, 1) should work."""
        model = GLM()
        Y_2d = glm_data["Y"].unsqueeze(1)
        nf = model.fit_null(Y_2d, glm_data["X0"])
        assert isinstance(nf, NullFit)

    def test_fit_null_residuals(self, glm_data):
        """Residuals should be orthogonal to X0."""
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        resid = nf.Y_rot  # residualized Y
        X0 = nf.X0_rot
        orth = X0.T @ resid
        assert torch.allclose(orth, torch.zeros_like(orth), atol=1e-10)


class TestGLMScan:
    def test_wald_scan_shape(self, glm_data):
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        vmeta = _make_vmeta(glm_data["m"])
        result = model.score_chunk(glm_data["G"], nf, vmeta, test="wald")

        assert isinstance(result, ScanResult)
        assert result.beta.shape == (glm_data["m"],)
        assert result.se.shape == (glm_data["m"],)
        assert result.stat.shape == (glm_data["m"],)
        assert result.p.shape == (glm_data["m"],)

    def test_pvalues_valid(self, glm_data):
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        vmeta = _make_vmeta(glm_data["m"])
        result = model.score_chunk(glm_data["G"], nf, vmeta)

        assert torch.all(result.p >= 0)
        assert torch.all(result.p <= 1)
        assert torch.all(result.stat >= 0)

    def test_causal_snps_significant(self, glm_data):
        """Causal SNPs should have small p-values."""
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        vmeta = _make_vmeta(glm_data["m"])
        result = model.score_chunk(glm_data["G"], nf, vmeta)

        # Causal SNPs at indices 5, 20, 50 should be significant
        for idx in [5, 20, 50]:
            assert result.p[idx].item() < 0.01, f"SNP {idx} p={result.p[idx].item()}"

    def test_null_pvalues_uniform(self, glm_null_data):
        """Under the null, p-values should be roughly uniform."""
        model = GLM()
        nf = model.fit_null(glm_null_data["Y"], glm_null_data["X0"])
        vmeta = _make_vmeta(glm_null_data["m"])
        result = model.score_chunk(glm_null_data["G"], nf, vmeta)

        median_p = result.p.median().item()
        assert 0.1 < median_p < 0.9

    def test_invalid_test_raises(self, glm_data):
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        vmeta = _make_vmeta(10)
        with pytest.raises(ValueError, match="wald"):
            model.score_chunk(glm_data["G"][:, :10], nf, vmeta, test="score")

    def test_allele_frequency(self, glm_data):
        model = GLM()
        nf = model.fit_null(glm_data["Y"], glm_data["X0"])
        vmeta = _make_vmeta(glm_data["m"])
        result = model.score_chunk(glm_data["G"], nf, vmeta)

        assert torch.all(result.af >= 0)
        assert torch.all(result.af <= 1)
