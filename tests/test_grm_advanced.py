"""Tests for advanced GRM methods (Slater, Endelman, Vitezica, Su, Yang, pseudo-diploid, weighted)."""

from __future__ import annotations

import pytest
import torch

from torchgwas.linalg.kinship_advanced import (
    grm_endelman_digenic,
    grm_pseudo_diploid,
    grm_slater,
    grm_su_dominance,
    grm_vitezica_dominance,
    grm_weighted,
    grm_yang_gcta,
)
from torchgwas.linalg.kinship import grm_vanraden


@pytest.fixture
def G_diploid():
    """Diploid genotype matrix: 20 samples, 100 SNPs."""
    torch.manual_seed(42)
    return torch.randint(0, 3, (20, 100), dtype=torch.float64)


@pytest.fixture
def G_tetra():
    """Tetraploid genotype matrix: 20 samples, 100 SNPs."""
    torch.manual_seed(42)
    return torch.randint(0, 5, (20, 100), dtype=torch.float64)


def _check_grm_properties(K, n, method_name):
    """Shared GRM property checks."""
    assert K.shape == (n, n), f"{method_name}: shape {K.shape} != ({n}, {n})"
    # Symmetry
    torch.testing.assert_close(K, K.T, msg=f"{method_name}: not symmetric")
    # PSD: eigenvalues >= -eps
    evals = torch.linalg.eigvalsh(K)
    assert evals.min().item() > -1e-6, (
        f"{method_name}: min eigenvalue {evals.min().item()} too negative"
    )
    # Diagonal should be non-negative
    assert K.diagonal().min().item() > -1e-6, (
        f"{method_name}: negative diagonal element"
    )


class TestSlater:
    def test_properties(self, G_tetra):
        K, meta = grm_slater(G_tetra, ploidy=4)
        _check_grm_properties(K, 20, "slater")
        assert meta.method == "slater"
        assert meta.ploidy == 4

    def test_diploid(self, G_diploid):
        K, meta = grm_slater(G_diploid, ploidy=2)
        _check_grm_properties(K, 20, "slater_diploid")

    def test_nan_raises(self):
        G = torch.tensor([[0.0, float("nan")], [1.0, 2.0]])
        with pytest.raises(ValueError, match="NaN"):
            grm_slater(G, ploidy=2)


class TestEndelmanDigenic:
    def test_properties(self, G_tetra):
        K, meta = grm_endelman_digenic(G_tetra)
        _check_grm_properties(K, 20, "endelman")
        assert meta.method == "endelman_digenic"

    def test_rejects_non_tetraploid(self):
        """Endelman digenic is tetraploid-only."""
        G = torch.randint(0, 7, (10, 50), dtype=torch.float64)
        with pytest.raises(ValueError, match="tetraploid"):
            grm_endelman_digenic(G)


class TestVitezicaDominance:
    def test_properties_diploid(self, G_diploid):
        K, meta = grm_vitezica_dominance(G_diploid, ploidy=2)
        _check_grm_properties(K, 20, "vitezica_diploid")
        assert meta.method == "vitezica_dominance"

    def test_properties_tetraploid(self, G_tetra):
        K, meta = grm_vitezica_dominance(G_tetra, ploidy=4)
        _check_grm_properties(K, 20, "vitezica_tetra")


class TestSuDominance:
    def test_properties_diploid(self, G_diploid):
        K, meta = grm_su_dominance(G_diploid, ploidy=2)
        _check_grm_properties(K, 20, "su_diploid")
        assert meta.method == "su_dominance"

    def test_properties_tetraploid(self, G_tetra):
        K, meta = grm_su_dominance(G_tetra, ploidy=4)
        _check_grm_properties(K, 20, "su_tetra")


class TestYangGCTA:
    def test_properties(self, G_diploid):
        K, meta = grm_yang_gcta(G_diploid, ploidy=2)
        _check_grm_properties(K, 20, "yang")
        assert meta.method == "yang_gcta"

    def test_diagonal_near_one(self, G_diploid):
        """Yang diagonal should be near 1 (1 + F) for outbred samples."""
        K, _ = grm_yang_gcta(G_diploid, ploidy=2)
        diag_mean = K.diagonal().mean().item()
        assert 0.8 < diag_mean < 1.5, f"Yang diagonal mean {diag_mean} outside [0.8, 1.5]"

    def test_polyploid(self, G_tetra):
        K, meta = grm_yang_gcta(G_tetra, ploidy=4)
        _check_grm_properties(K, 20, "yang_tetra")


class TestPseudoDiploid:
    def test_properties(self, G_tetra):
        K, meta = grm_pseudo_diploid(G_tetra, ploidy=4)
        _check_grm_properties(K, 20, "pseudo_diploid")
        assert meta.method == "pseudo_diploid"
        assert meta.ploidy == 4

    def test_matches_vanraden_on_diploid(self, G_diploid):
        """When input is already diploid, pseudo-diploid should match VanRaden."""
        K_pseudo, _ = grm_pseudo_diploid(G_diploid, ploidy=2)
        K_vr, _ = grm_vanraden(G_diploid, ploidy=2)
        torch.testing.assert_close(K_pseudo, K_vr, atol=1e-10, rtol=1e-10)


class TestWeightedGRM:
    def test_properties(self, G_diploid):
        weights = torch.ones(100, dtype=torch.float64)
        K, meta = grm_weighted(G_diploid, weights, ploidy=2)
        _check_grm_properties(K, 20, "weighted")
        assert meta.method == "weighted"

    def test_uniform_weights_match_vanraden(self, G_diploid):
        """Uniform weights should give the same result as VanRaden."""
        weights = torch.ones(100, dtype=torch.float64)
        K_w, _ = grm_weighted(G_diploid, weights, ploidy=2)
        K_vr, _ = grm_vanraden(G_diploid, ploidy=2)
        torch.testing.assert_close(K_w, K_vr, atol=1e-10, rtol=1e-10)

    def test_zero_weight_excludes_snp(self, G_diploid):
        """Zero-weight SNPs should not contribute."""
        weights = torch.ones(100, dtype=torch.float64)
        weights[:50] = 0.0
        K_w, _ = grm_weighted(G_diploid, weights, ploidy=2)
        K_sub, _ = grm_vanraden(G_diploid[:, 50:], ploidy=2)
        # Should be proportional (different normalizers)
        # Check correlation of off-diag elements
        mask = ~torch.eye(20, dtype=torch.bool)
        r = torch.corrcoef(torch.stack([K_w[mask], K_sub[mask]]))[0, 1]
        assert r > 0.99

    def test_weight_length_mismatch_raises(self, G_diploid):
        with pytest.raises(ValueError, match="weights length"):
            grm_weighted(G_diploid, torch.ones(50), ploidy=2)

    def test_negative_weight_raises(self, G_diploid):
        with pytest.raises(ValueError, match="non-negative"):
            grm_weighted(G_diploid, -torch.ones(100), ploidy=2)


@pytest.mark.parametrize("grm_func,kwargs", [
    (grm_slater, {"ploidy": 4}),
    (grm_endelman_digenic, {}),
    (grm_vitezica_dominance, {"ploidy": 4}),
    (grm_su_dominance, {"ploidy": 4}),
    (grm_yang_gcta, {"ploidy": 4}),
    (grm_pseudo_diploid, {"ploidy": 4}),
])
class TestGRMSharedProperties:
    """Parametrized shared tests across all tetraploid GRM methods."""

    def test_shape_and_symmetry(self, grm_func, kwargs, G_tetra):
        K, meta = grm_func(G_tetra, **kwargs)
        assert K.shape == (20, 20)
        torch.testing.assert_close(K, K.T)

    def test_float64(self, grm_func, kwargs, G_tetra):
        K, _ = grm_func(G_tetra, **kwargs)
        assert K.dtype == torch.float64
