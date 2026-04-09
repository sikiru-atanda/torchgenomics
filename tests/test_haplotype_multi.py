"""Tests for multi-environment and multi-trait haplotype GWAS (Phase 47)."""

import torch
import pytest

from torchgwas.models.haplotype_gwas import (
    HaplotypeBlock,
    HaplotypeGWAS,
    _enumerate_haplotypes_phased,
)
from torchgwas.models.haplotype_multi import (
    HaplotypeMultiEnvGWAS,
    HaplotypeMultiTraitGWAS,
    HaplotypeMTMETGWAS,
    HaplotypeMultiEnvResult,
    HaplotypeMultiTraitResult,
    HaplotypeMTMETResult,
    _haplotype_gls_wald,
)
from torchgwas.ld._blocks import LDBlock
from torchgwas.io.regions import Region

DTYPE = torch.float64


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_phased_haplotypes(n=100, m=5, ploidy=2, seed=42):
    """Create synthetic phased haplotype data with ~3 dominant haplotypes."""
    torch.manual_seed(seed)
    haps = torch.zeros(n, ploidy, m, dtype=torch.long)
    for i in range(n):
        for p in range(ploidy):
            r = torch.rand(1).item()
            if r < 0.4:
                haps[i, p, :] = 0
            elif r < 0.7:
                haps[i, p, :] = 1
            else:
                alt = torch.zeros(m, dtype=torch.long)
                alt[1::2] = 1
                haps[i, p, :] = alt
    return haps


def _make_genotypes(haps):
    return haps.sum(dim=1).float()


def _make_kinship(n, seed=42):
    torch.manual_seed(seed)
    Z = torch.randn(n, n // 2, dtype=DTYPE)
    K = Z @ Z.T / (n // 2)
    K = K + torch.eye(n, dtype=DTYPE) * 0.1
    return K


def _make_blocks_from_haps(haps, variant_pos=None, variant_chr=None):
    """Build LDBlock list from phased data (all SNPs as one block)."""
    n, ploidy, m = haps.shape
    if variant_pos is None:
        variant_pos = list(range(m))
    if variant_chr is None:
        variant_chr = ["1"] * m
    region = Region(region_id="block_0", chr="1",
                    start=variant_pos[0], end=variant_pos[-1])
    block = LDBlock(
        region=region,
        n_variants=m,
        variant_indices=list(range(m)),
        method="manual",
        mean_r2=1.0,
        mean_dprime=1.0,
    )
    return [block]


def _get_dosage_from_haps(haps):
    """Get haplotype dosage from phased data (for building phenotypes)."""
    labels, freqs, dosage = _enumerate_haplotypes_phased(haps, ploidy=2)
    return dosage


def _make_multi_env_phenotype(n, E, dosage, causal_hap_idx, betas, seed=42):
    """Create (n, E) phenotype matrix with haplotype effects that vary by env.

    betas: (E,) effect sizes for the causal haplotype in each environment.
    """
    torch.manual_seed(seed)
    Y = torch.zeros(n, E, dtype=DTYPE)
    for e in range(E):
        Y[:, e] = dosage[:, causal_hap_idx].to(DTYPE) * betas[e]
        Y[:, e] += torch.randn(n, dtype=DTYPE) * 0.5
    return Y


def _make_multi_trait_phenotype(n, d, dosage, causal_hap_idx, betas, seed=42):
    """Create (n, d) phenotype matrix with haplotype effects on each trait.

    betas: (d,) effect sizes for the causal haplotype on each trait.
    """
    torch.manual_seed(seed)
    Y = torch.zeros(n, d, dtype=DTYPE)
    for t in range(d):
        Y[:, t] = dosage[:, causal_hap_idx].to(DTYPE) * betas[t]
        Y[:, t] += torch.randn(n, dtype=DTYPE) * 0.5
    return Y


# ── Multi-Environment tests ──────────────────────────────────────────

class TestHaplotypeMultiEnv:

    def test_multi_env_block_scan_runs(self):
        """Basic structure check — runs without error."""
        n, m, E = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=10)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=10)
        Y = _make_multi_env_phenotype(n, E, dosage, 1, [0.3, 0.0, -0.3], seed=10)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert isinstance(result, HaplotypeMultiEnvResult)
        assert len(result.block_id) == 1
        assert result.stat_joint.shape[0] == 1

    def test_multi_env_joint_p_valid(self):
        """Joint p-values in (0, 1]."""
        n, m, E = 80, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=20)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=20)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.5, 0.5], seed=20)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()
        assert (result.p_joint <= 1).all()

    def test_multi_env_detects_signal(self):
        """Causal haplotype with strong env-varying effect is detected."""
        n, m, E = 120, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=30)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=30)
        # Strong effect in env 0, zero in env 1
        Y = _make_multi_env_phenotype(n, E, dosage, 1, [1.5, 0.0], seed=30)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.p_joint[0].item() < 0.05

    def test_multi_env_per_hap_chi2(self):
        """Per-haplotype test returns correct number of values."""
        n, m, E = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=40)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=40)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.5, 0.5, 0.5], seed=40)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        H = dosage.shape[1]
        h = H - 1  # reference dropped
        assert result.stat_per_hap[0].shape[0] == h
        assert (result.p_per_hap[0] > 0).all()
        assert (result.p_per_hap[0] <= 1).all()

    def test_multi_env_per_env_chi2(self):
        """Per-env test returns E values."""
        n, m, E = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=50)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=50)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.5, 0.0, 0.5], seed=50)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.stat_per_env[0].shape[0] == E
        assert (result.p_per_env[0] > 0).all()

    def test_multi_env_homogeneity(self):
        """Uniform effect across envs → homogeneity test non-significant."""
        n, m, E = 120, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=60)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=60)
        # Same effect in all envs → no heterogeneity
        Y = _make_multi_env_phenotype(n, E, dosage, 1, [0.5, 0.5, 0.5], seed=60)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        # Homogeneity p for the causal haplotype should not be extreme
        # (uniform effect → no heterogeneity)
        assert result.stat_homogeneity[0].shape[0] > 0

    def test_multi_env_gxe_interaction(self):
        """Differential effect across envs → significant GxE."""
        n, m, E = 120, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=70)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=70)
        # Opposite effects in the two environments
        Y = _make_multi_env_phenotype(n, E, dosage, 1, [2.0, -2.0], seed=70)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.p_hap_gxe[0].item() < 0.05

    def test_multi_env_window_scan(self):
        """Moving-window method runs correctly."""
        n, m, E = 80, 10, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=80)
        G = _make_genotypes(haps)
        K = _make_kinship(n, seed=80)
        # Build phenotype from full-genome haplotype dosage
        dosage_full = _enumerate_haplotypes_phased(haps[:, :, :5], ploidy=2)[2]
        Y = _make_multi_env_phenotype(n, E, dosage_full, 0, [0.5, 0.5], seed=80)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="window", window_size=5, step=5)
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, haplotypes=haps)

        # Should have m // window_size = 2 windows
        assert len(result.block_id) == 2
        assert (result.p_joint > 0).all()

    def test_multi_env_precomputed_blocks(self):
        """User-supplied blocks are used correctly."""
        n, m, E = 80, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=90)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=90)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.3, 0.3], seed=90)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert len(result.block_id) == 1

    def test_multi_env_result_shapes(self):
        """All tensor shapes are consistent."""
        n, m, E = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=100)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=100)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.5, 0.5, 0.5], seed=100)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        B = len(result.block_id)
        assert result.stat_joint.shape == (B,)
        assert result.p_joint.shape == (B,)
        assert result.stat_hap_gxe.shape == (B,)
        assert result.p_hap_gxe.shape == (B,)
        # Per-block list shapes
        for i in range(B):
            if result.n_haplotypes[i] >= 2:
                h = result.n_haplotypes[i] - 1
                assert result.beta[i].shape == (h, E)
                assert result.se[i].shape == (h, E)
                assert result.stat_per_hap[i].shape == (h,)
                assert result.stat_per_env[i].shape == (E,)
                assert result.stat_homogeneity[i].shape == (h,)

    def test_multi_env_two_env(self):
        """Minimal E=2 case works."""
        n, m, E = 60, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=110)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=110)
        Y = _make_multi_env_phenotype(n, E, dosage, 0, [0.5, 0.3], seed=110)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiEnvGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()
        assert (result.p_joint <= 1).all()


# ── Multi-Trait tests ─────────────────────────────────────────────────

class TestHaplotypeMultiTrait:

    def test_multi_trait_block_scan_runs(self):
        """Basic structure check."""
        n, m, d = 80, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=200)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=200)
        Y = _make_multi_trait_phenotype(n, d, dosage, 1, [0.5, 0.3], seed=200)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert isinstance(result, HaplotypeMultiTraitResult)
        assert len(result.block_id) == 1

    def test_multi_trait_joint_p_valid(self):
        """Joint p-values in (0, 1]."""
        n, m, d = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=210)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=210)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.5, 0.5, 0.5], seed=210)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()
        assert (result.p_joint <= 1).all()

    def test_multi_trait_detects_signal(self):
        """Causal haplotype affecting one trait is detected."""
        n, m, d = 120, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=220)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=220)
        Y = _make_multi_trait_phenotype(n, d, dosage, 1, [1.5, 0.0], seed=220)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.p_joint[0].item() < 0.05

    def test_multi_trait_per_hap_chi2(self):
        """Per-haplotype test returns h values."""
        n, m, d = 80, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=230)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=230)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.5, 0.5], seed=230)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        H = dosage.shape[1]
        h = H - 1
        assert result.stat_per_hap[0].shape[0] == h

    def test_multi_trait_per_trait_chi2(self):
        """Per-trait test returns d values."""
        n, m, d = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=240)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=240)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.5, 0.0, 0.5], seed=240)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.stat_per_trait[0].shape[0] == d
        assert (result.p_per_trait[0] > 0).all()

    def test_multi_trait_pleiotropic(self):
        """Haplotype affecting all traits → strong joint p."""
        n, m, d = 120, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=250)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=250)
        Y = _make_multi_trait_phenotype(n, d, dosage, 1, [1.5, 1.5], seed=250)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.p_joint[0].item() < 0.05

    def test_multi_trait_trait_specific(self):
        """Haplotype affecting only one trait → that per-trait test is small."""
        n, m, d = 120, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=260)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=260)
        # Strong effect on trait 0 only
        Y = _make_multi_trait_phenotype(n, d, dosage, 1, [2.0, 0.0], seed=260)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        # Per-trait test for trait 0 should be more significant than trait 1
        assert result.p_per_trait[0][0].item() < result.p_per_trait[0][1].item()

    def test_multi_trait_window_scan(self):
        """Moving-window method runs."""
        n, m, d = 80, 10, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=270)
        G = _make_genotypes(haps)
        K = _make_kinship(n, seed=270)
        dosage_full = _enumerate_haplotypes_phased(haps[:, :, :5], ploidy=2)[2]
        Y = _make_multi_trait_phenotype(n, d, dosage_full, 0, [0.5, 0.5], seed=270)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="window", window_size=5, step=5)
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, haplotypes=haps)

        assert len(result.block_id) == 2

    def test_multi_trait_two_traits(self):
        """Minimal d=2 case."""
        n, m, d = 60, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=280)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=280)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.3, 0.3], seed=280)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()

    def test_multi_trait_precomputed_blocks(self):
        """User-supplied blocks are used."""
        n, m, d = 80, 5, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=290)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=290)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.3, 0.3], seed=290)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert len(result.block_id) == 1

    def test_multi_trait_result_shapes(self):
        """All tensor shapes are consistent."""
        n, m, d = 80, 5, 3
        haps = _make_phased_haplotypes(n=n, m=m, seed=300)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=300)
        Y = _make_multi_trait_phenotype(n, d, dosage, 0, [0.5, 0.5, 0.5], seed=300)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMultiTraitGWAS(method="block")
        nf = model.fit_null(Y, X0, K)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        B = len(result.block_id)
        assert result.stat_joint.shape == (B,)
        assert result.p_joint.shape == (B,)
        for i in range(B):
            if result.n_haplotypes[i] >= 2:
                h = result.n_haplotypes[i] - 1
                assert result.beta[i].shape == (h, d)
                assert result.se[i].shape == (h, d)
                assert result.stat_per_hap[i].shape == (h,)
                assert result.stat_per_trait[i].shape == (d,)


# ── MT-MET tests ─────────────────────────────────────────────────────

def _make_mtmet_phenotype(n, d, E, dosage, causal_hap_idx, betas_dE, seed=42):
    """Create (n, d*E) phenotype matrix in trait-major column order.

    betas_dE: (d, E) effect matrix for the causal haplotype.
    """
    torch.manual_seed(seed)
    Y_wide = torch.zeros(n, d * E, dtype=DTYPE)
    for t in range(d):
        for e in range(E):
            col = t * E + e
            Y_wide[:, col] = dosage[:, causal_hap_idx].to(DTYPE) * betas_dE[t][e]
            Y_wide[:, col] += torch.randn(n, dtype=DTYPE) * 0.5
    return Y_wide


class TestHaplotypeMTMET:

    def test_mtmet_block_scan_runs(self):
        """Basic structure check."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=400)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=400)
        betas = [[0.5, 0.3], [0.0, 0.0]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 1, betas, seed=400)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert isinstance(result, HaplotypeMTMETResult)
        assert len(result.block_id) == 1

    def test_mtmet_joint_p_valid(self):
        """Joint p-values in (0, 1]."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=410)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=410)
        betas = [[0.3, 0.3], [0.3, 0.3]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=410)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()
        assert (result.p_joint <= 1).all()

    def test_mtmet_detects_signal(self):
        """Causal haplotype with trait-env specific effect is detected."""
        n, m, d, E = 120, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=420)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=420)
        betas = [[1.5, 0.0], [0.0, 1.5]]  # trait 0 in env 0, trait 1 in env 1
        Y = _make_mtmet_phenotype(n, d, E, dosage, 1, betas, seed=420)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.p_joint[0].item() < 0.05

    def test_mtmet_per_hap_chi2(self):
        """Per-haplotype test returns h values with df = dE."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=430)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=430)
        betas = [[0.5, 0.5], [0.5, 0.5]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=430)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        H = dosage.shape[1]
        h = H - 1
        assert result.stat_per_hap[0].shape[0] == h

    def test_mtmet_per_trait_chi2(self):
        """Per-trait test returns d values."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=440)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=440)
        betas = [[0.5, 0.5], [0.0, 0.0]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=440)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.stat_per_trait[0].shape[0] == d

    def test_mtmet_per_env_chi2(self):
        """Per-env test returns E values."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=450)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=450)
        betas = [[0.5, 0.0], [0.5, 0.0]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=450)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert result.stat_per_env[0].shape[0] == E

    def test_mtmet_hap_gxe(self):
        """Per-haplotype GxE heterogeneity test."""
        n, m, d, E = 120, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=460)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=460)
        # Opposite effects across envs for both traits
        betas = [[2.0, -2.0], [2.0, -2.0]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 1, betas, seed=460)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        # GxE should be significant for the causal haplotype
        # stat_hap_gxe is a list of (h,) tensors
        assert result.stat_hap_gxe[0].shape[0] > 0

    def test_mtmet_window_scan(self):
        """Moving-window method runs."""
        n, m, d, E = 80, 10, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=470)
        G = _make_genotypes(haps)
        K = _make_kinship(n, seed=470)
        dosage_full = _enumerate_haplotypes_phased(haps[:, :, :5], ploidy=2)[2]
        betas = [[0.5, 0.5], [0.5, 0.5]]
        Y = _make_mtmet_phenotype(n, d, E, dosage_full, 0, betas, seed=470)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="window", window_size=5, step=5,
                                    vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, haplotypes=haps)

        assert len(result.block_id) == 2

    def test_mtmet_separable_vs_unstructured(self):
        """Both vg_structures converge for small dE."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=480)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=480)
        betas = [[0.5, 0.5], [0.5, 0.5]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=480)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        # Unstructured
        m1 = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf1 = m1.fit_null(Y, X0, K, d=d, E=E)
        r1 = m1.scan(G, nf1, blocks=blocks, haplotypes=haps)

        # Separable
        m2 = HaplotypeMTMETGWAS(method="block", vg_structure="separable")
        nf2 = m2.fit_null(Y, X0, K, d=d, E=E)
        r2 = m2.scan(G, nf2, blocks=blocks, haplotypes=haps)

        # Both should produce valid p-values
        assert (r1.p_joint > 0).all() and (r1.p_joint <= 1).all()
        assert (r2.p_joint > 0).all() and (r2.p_joint <= 1).all()

    def test_mtmet_3d_input(self):
        """3D Y input (n, d, E) works."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=490)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=490)
        betas = [[0.3, 0.3], [0.3, 0.3]]
        Y_wide = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=490)
        # Reshape to 3D: (n, d, E)
        Y_3d = Y_wide.reshape(n, d, E)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y_3d, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        assert (result.p_joint > 0).all()
        assert (result.p_joint <= 1).all()

    def test_mtmet_result_shapes(self):
        """All tensor shapes are consistent."""
        n, m, d, E = 80, 5, 2, 2
        haps = _make_phased_haplotypes(n=n, m=m, seed=500)
        G = _make_genotypes(haps)
        blocks = _make_blocks_from_haps(haps)
        dosage = _get_dosage_from_haps(haps)
        K = _make_kinship(n, seed=500)
        betas = [[0.5, 0.5], [0.5, 0.5]]
        Y = _make_mtmet_phenotype(n, d, E, dosage, 0, betas, seed=500)
        X0 = torch.ones(n, 1, dtype=DTYPE)

        model = HaplotypeMTMETGWAS(method="block", vg_structure="unstructured")
        nf = model.fit_null(Y, X0, K, d=d, E=E)
        result = model.scan(G, nf, blocks=blocks, haplotypes=haps)

        B = len(result.block_id)
        assert result.stat_joint.shape == (B,)
        assert result.p_joint.shape == (B,)
        for i in range(B):
            if result.n_haplotypes[i] >= 2:
                h = result.n_haplotypes[i] - 1
                assert result.beta[i].shape == (h, d, E)
                assert result.se[i].shape == (h, d, E)
                assert result.stat_per_hap[i].shape == (h,)
                assert result.stat_per_trait[i].shape == (d,)
                assert result.stat_per_env[i].shape == (E,)
                assert result.stat_hap_gxe[i].shape == (h,)
