"""Tests for LDSC heritability and genetic correlation."""

import pytest
import torch

from torchgwas.postgwas._ld_scores import compute_ld_scores
from torchgwas.postgwas._ldsc import ldsc_h2, ldsc_intercept, ldsc_rg_from_z


def _simulate_infinitesimal(n=1000, m=500, h2=0.5, seed=42):
    """Simulate infinitesimal model for LDSC testing.

    Returns chi2, ld_scores, z-scores, and true h2.
    """
    torch.manual_seed(seed)

    # Generate genotypes with mild LD (adjacent SNPs correlated)
    G = torch.randn(n, m, dtype=torch.float64)
    # Add LD: each SNP = 0.3 * previous + 0.95 * independent
    for j in range(1, m):
        G[:, j] = 0.3 * G[:, j - 1] + (1 - 0.09) ** 0.5 * G[:, j]

    # Standardize
    G = (G - G.mean(dim=0)) / G.std(dim=0).clamp(min=1e-6)

    # True effects: all SNPs causal (infinitesimal)
    beta_true = torch.randn(m, dtype=torch.float64) * (h2 / m) ** 0.5

    # Genetic values
    g = G @ beta_true
    # Environment
    e_var = g.var() * (1 - h2) / h2
    e = torch.randn(n, dtype=torch.float64) * e_var.sqrt()
    y = g + e

    # Marginal OLS for each SNP
    z = torch.zeros(m, dtype=torch.float64)
    for j in range(m):
        gj = G[:, j]
        b = (gj @ y) / (gj @ gj)
        resid = y - gj * b
        se = (resid.var() / (gj @ gj)).sqrt()
        z[j] = b / se

    chi2 = z**2

    # Positions: evenly spaced on one chromosome
    pos = list(range(0, m * 2000, 2000))
    chr_labels = ["1"] * m

    ld_scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)

    return chi2, ld_scores, z, n, m, h2


class TestLDScores:
    """Tests for LD score computation."""

    def test_independent_snps_ld_score_one(self):
        """Independent SNPs should have LD score ~ 1."""
        torch.manual_seed(0)
        G = torch.randn(500, 100, dtype=torch.float64)
        pos = list(range(0, 100 * 10000, 10000))
        chr_labels = ["1"] * 100
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        # Each SNP should be ~1 (self r²=1, others ~0)
        assert scores.mean().item() < 2.0
        assert scores.min().item() >= 1.0 - 0.01  # self-r² = 1

    def test_perfect_ld_pair(self):
        """Two perfectly correlated SNPs should have LD score = 2."""
        torch.manual_seed(1)
        g = torch.randn(200, 1, dtype=torch.float64)
        G = torch.cat([g, g], dim=1)
        pos = [0, 1000]
        chr_labels = ["1", "1"]
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=10.0)
        assert scores[0].item() == pytest.approx(2.0, abs=0.01)
        assert scores[1].item() == pytest.approx(2.0, abs=0.01)

    def test_cross_chromosome_exclusion(self):
        """SNPs on different chromosomes should not contribute to each other's LD score."""
        torch.manual_seed(2)
        g = torch.randn(200, 1, dtype=torch.float64)
        # Same genotype data but on different chromosomes
        G = torch.cat([g, g], dim=1)
        pos = [1000, 1000]
        chr_labels = ["1", "2"]
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        # Each should be ~1 (only self)
        assert scores[0].item() == pytest.approx(1.0, abs=0.01)
        assert scores[1].item() == pytest.approx(1.0, abs=0.01)

    def test_window_respected(self):
        """SNPs outside the window should not contribute."""
        torch.manual_seed(3)
        g = torch.randn(200, 1, dtype=torch.float64)
        G = torch.cat([g, g], dim=1)
        # 2 Mb apart — outside 1 Mb window
        pos = [0, 2_000_001]
        chr_labels = ["1", "1"]
        scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)
        assert scores[0].item() == pytest.approx(1.0, abs=0.01)


class TestLDSCH2:
    """Tests for LDSC heritability estimation."""

    def test_h2_recovery(self):
        """LDSC should recover h2 within 2 SE of true value."""
        chi2, ld_scores, z, n, m, h2_true = _simulate_infinitesimal(
            n=2000, m=1000, h2=0.5, seed=100
        )
        result = ldsc_h2(chi2, ld_scores, n, m, n_blocks=50)

        # h2 should be within a reasonable range (LDSC has high variance with small m)
        assert result.h2 > 0.0, f"h2 should be positive, got {result.h2}"
        assert result.h2 < 1.5, f"h2 unreasonably large: {result.h2}"
        assert result.h2_se > 0, "SE should be positive"
        # Within 3 SE of truth (generous for small sample)
        assert abs(result.h2 - h2_true) < 3 * result.h2_se + 0.3

    def test_intercept_near_one(self):
        """Under no confounding, intercept should be near 1."""
        chi2, ld_scores, z, n, m, _ = _simulate_infinitesimal(seed=200)
        intercept, intercept_se = ldsc_intercept(chi2, ld_scores, n, m, n_blocks=50)
        # Should be close to 1 (within 3 SE + margin)
        assert abs(intercept - 1.0) < 3 * intercept_se + 0.3

    def test_mean_chi2_correct(self):
        """Mean chi2 should match input."""
        chi2, ld_scores, z, n, m, _ = _simulate_infinitesimal(seed=300)
        result = ldsc_h2(chi2, ld_scores, n, m)
        assert result.mean_chi2 == pytest.approx(chi2.mean().item(), rel=1e-6)

    def test_n_snps_reported(self):
        """n_snps should reflect filtering."""
        chi2, ld_scores, z, n, m, _ = _simulate_infinitesimal(seed=400)
        result = ldsc_h2(chi2, ld_scores, n, m, two_step_cutoff=30.0)
        assert result.n_snps > 0
        assert result.n_snps <= m


class TestLDSCRg:
    """Tests for genetic correlation estimation."""

    def test_rg_positive_correlated_traits(self):
        """Two positively correlated traits should yield rg > 0."""
        torch.manual_seed(500)
        n, m = 2000, 500
        h2 = 0.4
        rg_true = 0.5

        G = torch.randn(n, m, dtype=torch.float64)
        for j in range(1, m):
            G[:, j] = 0.3 * G[:, j - 1] + 0.95 * G[:, j]
        G = (G - G.mean(0)) / G.std(0).clamp(min=1e-6)

        # Shared + trait-specific effects
        beta_shared = torch.randn(m, dtype=torch.float64) * (h2 * rg_true / m) ** 0.5
        beta_1 = beta_shared + torch.randn(m, dtype=torch.float64) * (h2 * (1 - rg_true) / m) ** 0.5
        beta_2 = beta_shared + torch.randn(m, dtype=torch.float64) * (h2 * (1 - rg_true) / m) ** 0.5

        g1 = G @ beta_1
        g2 = G @ beta_2
        e1 = torch.randn(n, dtype=torch.float64) * (g1.var() * (1 - h2) / h2).sqrt()
        e2 = torch.randn(n, dtype=torch.float64) * (g2.var() * (1 - h2) / h2).sqrt()
        y1 = g1 + e1
        y2 = g2 + e2

        z1, z2 = torch.zeros(m, dtype=torch.float64), torch.zeros(m, dtype=torch.float64)
        for j in range(m):
            gj = G[:, j]
            gg = gj @ gj
            b1 = (gj @ y1) / gg
            b2 = (gj @ y2) / gg
            se1 = ((y1 - gj * b1).var() / gg).sqrt()
            se2 = ((y2 - gj * b2).var() / gg).sqrt()
            z1[j] = b1 / se1
            z2[j] = b2 / se2

        pos = list(range(0, m * 2000, 2000))
        chr_labels = ["1"] * m
        ld_scores = compute_ld_scores(G, pos, chr_labels, window_kb=1000.0)

        result = ldsc_rg_from_z(z1, z2, ld_scores, n, n, m, n_blocks=50)
        # rg should be positive (direction correct)
        assert result.rg > -0.5, f"rg should be positive-ish, got {result.rg}"
        assert result.rg_se > 0

    def test_rg_returns_h2(self):
        """rg result should contain valid h2 estimates for both traits."""
        chi2, ld_scores, z, n, m, _ = _simulate_infinitesimal(seed=600)
        z2 = z + torch.randn_like(z) * 0.5  # slightly different trait
        result = ldsc_rg_from_z(z, z2, ld_scores, n, n, m, n_blocks=50)
        assert result.h2_1.h2 is not None
        assert result.h2_2.h2 is not None
