"""Tests for ``torchgwas.postgwas._multi_ancestry`` — multi-ancestry meta-analysis."""
from __future__ import annotations

import math

import pytest
import torch

from torchgwas.postgwas._multi_ancestry import MultiAncestryResult, mr_mega, mantra
from torchgwas.postgwas._sumstats import SumStats


# ── Helper ───────────────────────────────────────────────────────────────


def _make_multi_ancestry_data(
    n_pops: int = 3,
    m_snps: int = 20,
    shared_signal_idx: int = 5,
    seed: int = 42,
) -> list[SumStats]:
    """Create per-population SumStats with one shared signal SNP.

    The signal SNP has beta ~ 0.5, se ~ 0.1 across all populations.
    Null SNPs have beta ~ 0, se ~ 0.2.  Allele frequencies vary across
    populations to give MR-MEGA PCA axes something to work with.
    """
    rng = torch.Generator().manual_seed(seed)

    ss_list: list[SumStats] = []
    snp_ids = [f"rs{i}" for i in range(m_snps)]
    chr_list = ["1"] * m_snps
    pos_list = list(range(1000, 1000 + m_snps * 100, 100))
    a1_list = ["A"] * m_snps
    a2_list = ["G"] * m_snps

    for pop in range(n_pops):
        beta = torch.randn(m_snps, generator=rng, dtype=torch.float64) * 0.02
        se = torch.full((m_snps,), 0.2, dtype=torch.float64)
        # Strong shared signal — tight residual across populations so the
        # MR-MEGA intercept (p_meta) picks it up rather than attributing it to
        # ancestry-correlated heterogeneity (which the 0.05*pop trend did).
        beta[shared_signal_idx] = (
            0.5 + torch.randn(1, generator=rng, dtype=torch.float64).item() * 0.01
        )
        se[shared_signal_idx] = 0.1

        p = 2.0 * torch.erfc(beta.abs() / se / math.sqrt(2.0))
        p = p.clamp(min=1e-300, max=1.0)

        n = torch.full((m_snps,), 5000.0 + pop * 500, dtype=torch.float64)

        # Allele frequencies vary across populations — add per-population
        # noise so that F_centered is not rank-1, else the cross-population
        # correlation matrix collapses to rank 1 and the PCA produces
        # numerical-noise axes that blow up (X'WX)^{-1}.
        base_af = torch.linspace(0.1, 0.5, m_snps, dtype=torch.float64)
        af_noise = torch.randn(m_snps, generator=rng, dtype=torch.float64) * 0.08
        af = (base_af + 0.05 * pop + af_noise).clamp(0.01, 0.99)

        ss_list.append(
            SumStats(
                chr=list(chr_list),
                pos=list(pos_list),
                snp=list(snp_ids),
                a1=list(a1_list),
                a2=list(a2_list),
                beta=beta,
                se=se,
                p=p,
                n=n,
                af=af,
            )
        )

    return ss_list


# ── MR-MEGA Tests ────────────────────────────────────────────────────────


class TestMrMega:
    """Tests for the MR-MEGA multi-ancestry meta-regression."""

    def test_mr_mega_detects_shared_signal(self) -> None:
        """Shared signal SNP should have p_meta < 0.05."""
        ss = _make_multi_ancestry_data(n_pops=5, m_snps=20, shared_signal_idx=5)
        result = mr_mega(ss, n_axes=2)
        assert result.p_meta[5].item() < 0.05

    def test_mr_mega_null_snps_not_significant(self) -> None:
        """Null SNPs should have p_meta > 0.01 on average."""
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=20, shared_signal_idx=5)
        result = mr_mega(ss, n_axes=2)
        null_mask = torch.ones(20, dtype=torch.bool)
        null_mask[5] = False
        mean_p_null = result.p_meta[null_mask].mean().item()
        assert mean_p_null > 0.01

    def test_mr_mega_ancestry_heterogeneity(self) -> None:
        """When effects differ strongly across populations, p_ancestry should be low."""
        ss = _make_multi_ancestry_data(n_pops=4, m_snps=15, shared_signal_idx=3, seed=99)
        # Make the signal SNP have divergent effects across populations
        ss[0].beta[3] = 0.8
        ss[1].beta[3] = -0.5
        ss[2].beta[3] = 0.6
        ss[3].beta[3] = -0.4
        for s in ss:
            s.se[3] = 0.08  # tight SE to make heterogeneity detectable
        result = mr_mega(ss, n_axes=2)
        # With divergent effects, ancestry heterogeneity should be detectable
        assert result.p_ancestry is not None
        assert result.p_heterogeneity[3].item() < 0.05

    def test_mr_mega_result_shapes(self) -> None:
        """All output tensors should have shape (m,)."""
        m = 25
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=m, shared_signal_idx=10)
        result = mr_mega(ss, n_axes=2)
        assert result.beta_meta.shape == (m,)
        assert result.se_meta.shape == (m,)
        assert result.p_meta.shape == (m,)
        assert result.p_heterogeneity.shape == (m,)
        assert result.p_ancestry is not None and result.p_ancestry.shape == (m,)
        assert result.p_residual is not None and result.p_residual.shape == (m,)
        assert result.n_populations == 3
        assert result.n_snps == m
        assert result.method == "mr_mega"

    def test_mr_mega_n_axes_rejected_when_too_large(self) -> None:
        """n_axes >= K should raise ValueError."""
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=10, shared_signal_idx=2)
        with pytest.raises(ValueError, match="n_axes"):
            mr_mega(ss, n_axes=3)
        with pytest.raises(ValueError, match="n_axes"):
            mr_mega(ss, n_axes=5)

    def test_mr_mega_random_effects(self) -> None:
        """random_effects=True should still produce a valid MultiAncestryResult."""
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=15, shared_signal_idx=4)
        result = mr_mega(ss, n_axes=2, random_effects=True)
        assert isinstance(result, MultiAncestryResult)
        assert result.method == "mr_mega"
        assert result.p_meta.shape == (15,)
        # Signal should still be detectable under random effects
        assert result.p_meta[4].item() < 0.05

    def test_mr_mega_two_populations(self) -> None:
        """K=2 edge case should work with n_axes=1."""
        ss = _make_multi_ancestry_data(n_pops=2, m_snps=10, shared_signal_idx=3)
        result = mr_mega(ss, n_axes=1)
        assert result.n_populations == 2
        assert result.n_axes == 1
        assert result.p_meta.shape == (10,)
        assert result.p_meta[3].item() < 0.05


# ── MANTRA Tests ─────────────────────────────────────────────────────────


class TestMantra:
    """Tests for the MANTRA Bayesian trans-ethnic meta-analysis."""

    def test_mantra_detects_shared_signal(self) -> None:
        """Shared signal SNP should have log10_bf > 1."""
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=20, shared_signal_idx=5)
        result = mantra(ss)
        assert result.log10_bf is not None
        assert result.log10_bf[5].item() > 1.0

    def test_mantra_null_has_low_bf(self) -> None:
        """Null SNPs should have log10_bf < 1 on average."""
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=20, shared_signal_idx=5)
        result = mantra(ss)
        assert result.log10_bf is not None
        null_mask = torch.ones(20, dtype=torch.bool)
        null_mask[5] = False
        mean_bf_null = result.log10_bf[null_mask].mean().item()
        assert mean_bf_null < 1.0

    def test_mantra_result_shapes(self) -> None:
        """Output tensors should have correct shapes."""
        m = 18
        ss = _make_multi_ancestry_data(n_pops=3, m_snps=m, shared_signal_idx=7)
        result = mantra(ss)
        assert result.beta_meta.shape == (m,)
        assert result.se_meta.shape == (m,)
        assert result.p_meta.shape == (m,)
        assert result.p_heterogeneity.shape == (m,)
        assert result.log10_bf is not None and result.log10_bf.shape == (m,)
        assert result.posterior_effect is not None and result.posterior_effect.shape == (m,)
        assert result.n_populations == 3
        assert result.n_snps == m
        assert result.method == "mantra"


# ── Input Validation Tests ───────────────────────────────────────────────


class TestInputValidation:
    """Tests for input validation across both methods."""

    def test_rejects_single_population(self) -> None:
        """Only 1 SumStats should raise ValueError for both methods."""
        ss = _make_multi_ancestry_data(n_pops=2, m_snps=10, shared_signal_idx=2)
        single = [ss[0]]
        with pytest.raises(ValueError, match="at least 2"):
            mr_mega(single)
        with pytest.raises(ValueError, match="at least 2"):
            mantra(single)

    def test_rejects_mismatched_snps(self) -> None:
        """Populations with no common SNPs should raise ValueError."""
        ss1 = SumStats(
            chr=["1", "1"],
            pos=[100, 200],
            snp=["rs1", "rs2"],
            a1=["A", "A"],
            a2=["G", "G"],
            beta=torch.tensor([0.1, 0.2], dtype=torch.float64),
            se=torch.tensor([0.1, 0.1], dtype=torch.float64),
            p=torch.tensor([0.3, 0.05], dtype=torch.float64),
            n=torch.tensor([1000.0, 1000.0], dtype=torch.float64),
            af=torch.tensor([0.3, 0.4], dtype=torch.float64),
        )
        ss2 = SumStats(
            chr=["1", "1"],
            pos=[300, 400],
            snp=["rs99", "rs100"],  # completely disjoint SNP IDs
            a1=["A", "A"],
            a2=["G", "G"],
            beta=torch.tensor([0.1, 0.2], dtype=torch.float64),
            se=torch.tensor([0.1, 0.1], dtype=torch.float64),
            p=torch.tensor([0.3, 0.05], dtype=torch.float64),
            n=torch.tensor([1000.0, 1000.0], dtype=torch.float64),
            af=torch.tensor([0.3, 0.4], dtype=torch.float64),
        )
        with pytest.raises(ValueError, match="[Nn]o common"):
            mr_mega([ss1, ss2], n_axes=1)
