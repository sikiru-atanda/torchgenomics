"""Tests for GWAS power analysis module."""

import pytest
import torch

from torchgenomics.postgwas._power import PowerResult, gwas_power, power_curve, required_n


class TestGWASPower:
    """Tests for gwas_power, power_curve, and required_n."""

    def test_power_known_values(self):
        """NCP = 2*10000*0.3*0.7*0.01 = 42.0; power should be substantial."""
        res = gwas_power(n=10000, af=0.3, beta=0.1, alpha=5e-8)
        expected_ncp = 2 * 10000 * 0.3 * 0.7 * 0.1**2
        assert expected_ncp == pytest.approx(42.0, abs=1e-10)
        assert res.ncp.item() == pytest.approx(42.0, rel=1e-4)
        # NCP=42 at alpha=5e-8 gives power ~0.85
        assert res.power.item() > 0.8

    def test_power_boundary_af(self):
        """AF=0 gives NCP=0, power approximately equals alpha."""
        res = gwas_power(n=10000, af=0.0, beta=0.1, alpha=5e-8)
        assert res.ncp.item() == pytest.approx(0.0, abs=1e-10)
        assert res.power.item() == pytest.approx(5e-8, abs=1e-6)

    def test_power_monotone_in_n(self):
        """Power at N=50000 > power at N=10000."""
        res_small = gwas_power(n=10000, af=0.3, beta=0.05, alpha=5e-8)
        res_large = gwas_power(n=50000, af=0.3, beta=0.05, alpha=5e-8)
        assert res_large.power.item() > res_small.power.item()

    def test_power_monotone_in_beta(self):
        """Power at beta=0.2 > power at beta=0.05."""
        res_small = gwas_power(n=10000, af=0.3, beta=0.05, alpha=5e-8)
        res_large = gwas_power(n=10000, af=0.3, beta=0.2, alpha=5e-8)
        assert res_large.power.item() > res_small.power.item()

    def test_power_monotone_in_af(self):
        """Power at AF=0.5 > power at AF=0.01 (harder to detect rare variants)."""
        res_rare = gwas_power(n=10000, af=0.01, beta=0.1, alpha=5e-8)
        res_common = gwas_power(n=10000, af=0.5, beta=0.1, alpha=5e-8)
        assert res_common.power.item() > res_rare.power.item()

    def test_power_at_alpha_one(self):
        """alpha=1.0 gives power=1.0."""
        res = gwas_power(n=100, af=0.3, beta=0.01, alpha=1.0)
        assert res.power.item() == pytest.approx(1.0, abs=1e-6)

    def test_power_curve_shape(self):
        """min_beta at AF=0.01 > min_beta at AF=0.5 (rare variants need larger effects)."""
        af_grid = torch.tensor([0.01, 0.5])
        min_betas = power_curve(n=10000, af_grid=af_grid, alpha=5e-8, target_power=0.8)
        assert min_betas[0].item() > min_betas[1].item()

    def test_power_curve_symmetry(self):
        """min_beta at AF=x equals min_beta at AF=1-x."""
        af_grid = torch.tensor([0.1, 0.3, 0.7, 0.9])
        min_betas = power_curve(n=10000, af_grid=af_grid, alpha=5e-8, target_power=0.8)
        # AF=0.1 and AF=0.9 should give same min_beta
        assert min_betas[0].item() == pytest.approx(min_betas[3].item(), rel=1e-4)
        # AF=0.3 and AF=0.7 should give same min_beta
        assert min_betas[1].item() == pytest.approx(min_betas[2].item(), rel=1e-4)

    def test_required_n_roundtrip(self):
        """Compute N via required_n, then check power at that N ~ target_power."""
        af = torch.tensor(0.3, dtype=torch.float64)
        beta = torch.tensor(0.05, dtype=torch.float64)
        target = 0.8
        n_needed = required_n(af=af, beta=beta, alpha=5e-8, target_power=target)
        n_val = int(n_needed.item())
        res = gwas_power(n=n_val, af=af, beta=beta, alpha=5e-8)
        # Power at the required N should be close to target_power
        assert res.power.item() == pytest.approx(target, abs=0.05)

    def test_power_result_fields(self):
        """PowerResult has all expected fields with correct shapes."""
        res = gwas_power(n=10000, af=0.3, beta=0.1, alpha=5e-8)
        assert isinstance(res, PowerResult)
        assert hasattr(res, "power")
        assert hasattr(res, "ncp")
        assert hasattr(res, "alpha")
        assert hasattr(res, "n")
        assert hasattr(res, "min_detectable_beta")
        # power and ncp should be tensors
        assert isinstance(res.power, torch.Tensor)
        assert isinstance(res.ncp, torch.Tensor)

    def test_power_large_ncp(self):
        """NCP=1000 gives power ~1.0."""
        # NCP = 2*N*af*(1-af)*beta^2 = 1000 => pick values accordingly
        # N=100000, af=0.5, beta^2 = 1000/(2*100000*0.25) = 0.02, beta ~ 0.1414
        import math
        beta = math.sqrt(1000.0 / (2 * 100000 * 0.5 * 0.5))
        res = gwas_power(n=100000, af=0.5, beta=beta, alpha=5e-8)
        assert res.ncp.item() == pytest.approx(1000.0, rel=1e-3)
        assert res.power.item() == pytest.approx(1.0, abs=1e-6)

    def test_power_curve_positive(self):
        """All min_beta values are positive."""
        af_grid = torch.linspace(0.05, 0.5, 10)
        min_betas = power_curve(n=10000, af_grid=af_grid, alpha=5e-8, target_power=0.8)
        assert (min_betas > 0).all()
