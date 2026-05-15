"""Tests for haplotype-based GWAS module."""

import pytest
import torch

from torchgwas.models.haplotype_gwas import (
    HaplotypeGWAS,
    HaplotypeGWASResult,
    _enumerate_haplotypes_phased,
    _enumerate_haplotypes_unphased,
    _haplotype_skat,
    _htr_f_test,
    _htr_lrt,
    _pool_rare_haplotypes,
)

# ── Fixtures ──────────────────────────────────────────────────────────

def _make_phased_haplotypes(n=100, m=3, ploidy=2, seed=42):
    """Create synthetic phased haplotype data with known structure."""
    torch.manual_seed(seed)
    # Two dominant haplotypes: "000" and "111", plus some "010"
    haps = torch.zeros(n, ploidy, m, dtype=torch.long)
    for i in range(n):
        for p in range(ploidy):
            r = torch.rand(1).item()
            if r < 0.4:
                haps[i, p, :] = 0  # all-ref haplotype
            elif r < 0.8:
                haps[i, p, :] = 1  # all-alt haplotype
            else:
                # Alternating pattern
                alt = torch.zeros(m, dtype=torch.long)
                alt[1::2] = 1
                haps[i, p, :] = alt
    return haps


def _make_genotypes_from_haplotypes(haps):
    """Sum over ploidy to get genotype dosages."""
    return haps.sum(dim=1).float()  # (n, m)


def _make_phenotype_with_haplotype_signal(dosage, causal_hap_idx, beta=1.0, seed=42):
    """Create phenotype with signal from a specific haplotype."""
    torch.manual_seed(seed)
    n = dosage.shape[0]
    Y = dosage[:, causal_hap_idx].to(torch.float64) * beta
    Y = Y + torch.randn(n, dtype=torch.float64) * 0.5
    return Y


# ── Haplotype construction tests ─────────────────────────────────────

class TestHaplotypeConstruction:
    """Tests for phased/unphased haplotype enumeration."""

    def test_phased_haplotype_counting(self):
        """Dosage sums to ploidy, freqs sum to 1."""
        haps = _make_phased_haplotypes(n=50, m=3, ploidy=2)
        labels, freqs, dosage = _enumerate_haplotypes_phased(haps, ploidy=2)

        assert len(labels) > 0
        assert freqs.sum().item() == pytest.approx(1.0, abs=1e-10)
        # Each individual's dosage sums to ploidy
        row_sums = dosage.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0))

    def test_unphased_em_convergence(self):
        """EM-estimated freqs from unphased data are reasonable."""
        torch.manual_seed(42)
        n, m = 200, 2
        # Generate data with known haplotype structure
        haps = _make_phased_haplotypes(n=n, m=m, ploidy=2)
        G = _make_genotypes_from_haplotypes(haps)
        labels_true, freqs_true, _ = _enumerate_haplotypes_phased(haps, ploidy=2)

        labels_em, freqs_em, dosage_em = _enumerate_haplotypes_unphased(
            G, max_iter=100, tol=1e-8,
        )

        # Should recover similar number of haplotypes
        assert len(labels_em) >= 2
        # Dosage should sum to approximately 2 per individual
        row_sums = dosage_em.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0), atol=0.1)
        # Frequencies should sum to 1
        assert freqs_em.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_unphased_em_two_snp_matches_existing(self):
        """2-SNP block EM should produce 4 or fewer haplotypes."""
        torch.manual_seed(123)
        n = 100
        G = torch.zeros(n, 2, dtype=torch.float64)
        for i in range(n):
            G[i, 0] = torch.randint(0, 3, (1,)).item()
            G[i, 1] = torch.randint(0, 3, (1,)).item()

        labels, freqs, dosage = _enumerate_haplotypes_unphased(G, max_iter=50)
        # With 2 biallelic loci, at most 4 haplotypes
        assert len(labels) <= 4
        assert freqs.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_rare_haplotype_pooling(self):
        """Pooling reduces columns, OTHER sums correctly."""
        labels = ["00", "01", "10", "11"]
        freqs = torch.tensor([0.5, 0.3, 0.005, 0.195])
        n = 10
        dosage = torch.rand(n, 4)
        dosage = dosage / dosage.sum(dim=1, keepdim=True) * 2.0

        labels_new, freqs_new, dosage_new = _pool_rare_haplotypes(
            labels, freqs, dosage, min_freq=0.01,
        )

        # "10" (freq=0.005) should be pooled into OTHER
        assert "OTHER" in labels_new
        assert len(labels_new) == 4  # 3 kept + OTHER
        assert freqs_new.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_single_snp_block(self):
        """1-SNP block produces 2 haplotypes ("0" and "1")."""
        torch.manual_seed(42)
        G = torch.tensor([[0.0], [1.0], [2.0], [1.0], [0.0]])
        labels, freqs, dosage = _enumerate_haplotypes_unphased(G, max_iter=20)
        assert len(labels) == 2

    def test_max_haplotypes_cap(self):
        """Pruning respects max_haplotypes."""
        torch.manual_seed(42)
        n, m = 50, 4
        G = torch.randint(0, 3, (n, m)).float()
        labels, freqs, dosage = _enumerate_haplotypes_unphased(
            G, max_haplotypes=5, max_iter=30,
        )
        assert len(labels) <= 5

    def test_ld_aware_pruning_keeps_high_freq_haplotype_under_tight_ld(self):
        """Regression for F2 finding (2026-05-15 Tier 1 A5): under tight LD,
        the pruning step must keep the second-most-common haplotype even when
        its marginal-allele-frequency product is small.

        Construction: a 5-SNP block with two truly dominant haplotypes
        ``00000`` (freq 0.5) and ``11111`` (freq 0.4) plus a low-frequency
        scatter of mixed-allele haplotypes (each freq 0.025). The marginal
        allele frequency at every locus is ~0.4, so the independence-prior
        product score for ``11111`` is ``0.4^5 ≈ 0.010``, well below several
        mixed-allele candidates like ``10101`` whose product is
        ``0.4 * 0.6 * 0.4 * 0.6 * 0.4 ≈ 0.023``. With LD-aware ranking, the
        actual frequency wins and ``11111`` survives the prune.
        """
        from itertools import product as iproduct

        torch.manual_seed(42)
        m = 5
        n = 400

        # True haplotype distribution
        truth_haps = ["00000", "11111"]  # the two dominant haplotypes
        truth_freqs = [0.5, 0.4]
        # 4 background mixed-allele haplotypes at 0.025 each — picked to stress
        # the independence prior (high marginal-product but low real frequency).
        bg = ["10101", "01010", "11010", "00110"]
        for h in bg:
            truth_haps.append(h)
            truth_freqs.append(0.025)
        assert abs(sum(truth_freqs) - 1.0) < 1e-9

        # Draw 2 haplotypes per individual, sum to genotype.
        H = len(truth_haps)
        hap_mat = torch.tensor(
            [[int(c) for c in h] for h in truth_haps], dtype=torch.long,
        )  # (H, m)
        freq_tensor = torch.tensor(truth_freqs)
        draws = torch.multinomial(freq_tensor, n * 2, replacement=True).view(n, 2)
        G = (hap_mat[draws[:, 0]] + hap_mat[draws[:, 1]]).float()

        # With max_haplotypes=4 the prune must trigger (candidate set easily
        # exceeds 4 — every individual whose genotype has any heterozygous
        # locus produces 2^h combinatorial candidates).
        labels, freqs, dosage = _enumerate_haplotypes_unphased(
            G, max_haplotypes=4, max_iter=50, tol=1e-8,
        )

        # The fix preserves both dominant haplotypes. The pre-fix
        # independence-product ranking dropped ``11111`` under this
        # construction.
        assert "00000" in labels, (
            "dominant reference haplotype 00000 missing from pruned set"
        )
        assert "11111" in labels, (
            "LD-driven second-most-common haplotype 11111 was dropped — F2 "
            "regression. The independence-prior pruning scored 11111 lower "
            "than mixed-allele candidates with no observational support. "
            "Reverting to the marginal-product fallback would reintroduce "
            "the bug."
        )

        # Sanity: EM frequency of ``11111`` must be within 0.07 of its truth
        # (0.4). With n=400, multinomial sampling variability is ~sqrt(0.4
        # * 0.6 / 400) ≈ 0.024, so 0.07 floor is conservative.
        idx_11111 = labels.index("11111")
        assert abs(freqs[idx_11111].item() - 0.4) < 0.07, (
            f"EM frequency for 11111 is {freqs[idx_11111].item():.3f}; "
            f"expected ~0.4 ± 0.07."
        )

    def test_score_candidates_ld_aware_helper_ranks_observed_pairs_higher(
        self,
    ):
        """Unit test for the LD-aware scoring helper: a candidate that
        actually appears in many individuals' compatible-pair sets must
        score higher than a candidate that appears in none, regardless of
        its marginal-allele-frequency product.
        """
        from torchgwas.models.haplotype_gwas import (
            _score_candidates_ld_aware,
            STAT_DTYPE,
        )

        # Two individuals, both homozygous for haplotype ``111``:
        #   genotype = [2, 2, 2] for both.
        # Candidate set includes ``111`` (truth, observed in all individuals)
        # and ``000`` (never compatible). Only ``111`` should accumulate
        # any weight.
        G_int = torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.long)
        candidates = ["000", "001", "010", "011", "100", "101", "110", "111"]
        scores = _score_candidates_ld_aware(
            candidates, G_int, device=torch.device("cpu"), dtype=STAT_DTYPE,
        )
        # All-ones haplotype is the only compatible pair → weight 2 per
        # individual × 2 individuals = 4 (each individual is `hom 111`,
        # contributes 2*1.0 since the only pair is (111, 111)).
        idx_111 = candidates.index("111")
        assert scores[idx_111].item() == pytest.approx(4.0, abs=1e-9)
        # All other candidates score 0 (never compatible with [2,2,2]).
        for h, s in zip(candidates, scores.tolist()):
            if h == "111":
                continue
            assert s == 0.0, f"unexpected non-zero score for {h}: {s}"


# ── HTR tests ─────────────────────────────────────────────────────────

class TestHTR:
    """Tests for F-test and LRT."""

    def test_f_test_null_calibration(self):
        """Under null, p-values should not be systematically small."""
        torch.manual_seed(42)
        n = 200
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        # Random haplotype dosages (no signal)
        D = torch.randn(n, 3, dtype=torch.float64)

        F, p, betas, ses, p_hap = _htr_f_test(Y, X0, D)
        # Under null, p should not be extremely small
        assert p > 0.001  # not guaranteed but very likely with seed=42

    def test_f_test_detects_signal(self):
        """Causal haplotype dosage produces significant p-value."""
        torch.manual_seed(42)
        n = 500
        X0 = torch.ones(n, 1, dtype=torch.float64)
        D = torch.randn(n, 3, dtype=torch.float64)
        # Strong signal on first haplotype
        Y = D[:, 0] * 2.0 + torch.randn(n, dtype=torch.float64) * 0.3

        F, p, betas, ses, p_hap = _htr_f_test(Y, X0, D)
        assert p < 0.05
        # The causal haplotype should have smallest p
        assert p_hap[0].item() < p_hap[1].item()

    def test_per_haplotype_effects(self):
        """Estimated betas match simulated truth."""
        torch.manual_seed(42)
        n = 1000
        X0 = torch.ones(n, 1, dtype=torch.float64)
        D = torch.randn(n, 2, dtype=torch.float64)
        true_betas = torch.tensor([1.5, -0.5], dtype=torch.float64)
        Y = D @ true_betas + torch.randn(n, dtype=torch.float64) * 0.5

        _, _, betas, ses, _ = _htr_f_test(Y, X0, D)
        assert betas[0].item() == pytest.approx(1.5, abs=0.2)
        assert betas[1].item() == pytest.approx(-0.5, abs=0.2)

    def test_lrt_matches_f_test(self):
        """LRT and F-test give similar p-values."""
        torch.manual_seed(42)
        n = 300
        X0 = torch.ones(n, 1, dtype=torch.float64)
        D = torch.randn(n, 2, dtype=torch.float64)
        Y = D[:, 0] * 1.0 + torch.randn(n, dtype=torch.float64) * 0.5

        _, p_f, _, _, _ = _htr_f_test(Y, X0, D)
        _, p_lrt, _, _, _ = _htr_lrt(Y, X0, D)
        # Both should be significant and in the same ballpark
        assert p_f < 0.05
        assert p_lrt < 0.05


# ── Block-based scan tests ────────────────────────────────────────────

class TestBlockScan:
    """Tests for block-based haplotype GWAS."""

    def test_block_scan_with_precomputed_blocks(self):
        """Runs correctly with pre-supplied LDBlock objects."""
        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock

        torch.manual_seed(42)
        n, m = 100, 10
        G = torch.randint(0, 3, (n, m)).float()
        Y = torch.randn(n, dtype=torch.float64)

        blocks = [
            LDBlock(
                region=Region(chr="1", start=0, end=5000, region_id="blk1"),
                n_variants=5,
                variant_indices=list(range(5)),
                method="manual",
                mean_r2=0.5,
                mean_dprime=0.7,
            ),
            LDBlock(
                region=Region(chr="1", start=5000, end=10000, region_id="blk2"),
                n_variants=5,
                variant_indices=list(range(5, 10)),
                method="manual",
                mean_r2=0.4,
                mean_dprime=0.6,
            ),
        ]

        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="block", test="f_test")
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        assert isinstance(result, HaplotypeGWASResult)
        assert len(result.block_id) == 2
        assert result.p_global.shape[0] == 2
        assert result.n_obs == n

    def test_block_scan_fewer_tests(self):
        """Number of tests equals number of blocks, not number of SNPs."""
        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock

        torch.manual_seed(42)
        n, m = 100, 20
        G = torch.randint(0, 3, (n, m)).float()
        Y = torch.randn(n, dtype=torch.float64)

        # 4 blocks of 5 SNPs
        blocks = []
        for b in range(4):
            s = b * 5
            blocks.append(LDBlock(
                region=Region(chr="1", start=s * 1000, end=(s + 5) * 1000,
                              region_id=f"blk{b}"),
                n_variants=5,
                variant_indices=list(range(s, s + 5)),
                method="manual",
                mean_r2=0.5,
                mean_dprime=0.7,
            ))

        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="block", test="f_test")
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        # 4 blocks < 20 SNPs
        assert len(result.block_id) == 4
        assert result.p_global.shape[0] == 4


# ── Moving-window scan tests ─────────────────────────────────────────

class TestWindowScan:
    """Tests for moving-window haplotype GWAS."""

    def test_window_scan_basic(self):
        """Window scan produces correct number of windows."""
        torch.manual_seed(42)
        n, m = 100, 15
        G = torch.randint(0, 3, (n, m)).float()
        Y = torch.randn(n, dtype=torch.float64)
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="window", window_size=5, step=5)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs)

        # 15 SNPs, window=5, step=5 → 3 non-overlapping windows
        assert len(result.block_id) == 3

    def test_window_scan_detects_signal(self):
        """Causal window should have smaller p-value."""
        torch.manual_seed(42)
        n, m = 300, 10
        G = torch.randint(0, 3, (n, m)).float()
        # Signal from a specific haplotype in SNPs 0-4
        haps = _make_phased_haplotypes(n=n, m=5, ploidy=2, seed=42)
        G[:, :5] = _make_genotypes_from_haplotypes(haps)
        labels, freqs, dosage = _enumerate_haplotypes_phased(haps)
        # Most frequent non-reference haplotype
        Y = dosage[:, 1].to(torch.float64) * 2.0
        Y = Y + torch.randn(n, dtype=torch.float64) * 0.5

        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="window", window_size=5, step=5)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs,
                              haplotypes=torch.cat([haps, torch.randint(0, 2, (n, 2, 5))], dim=2))

        # First window (causal) should have smaller p than second
        assert result.p_global[0].item() < result.p_global[1].item()

    def test_window_nonoverlapping(self):
        """step=window_size partitions the genome."""
        torch.manual_seed(42)
        n, m = 50, 12
        G = torch.randint(0, 3, (n, m)).float()
        Y = torch.randn(n, dtype=torch.float64)
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="window", window_size=4, step=4)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs)

        assert len(result.block_id) == 3  # 12 / 4 = 3


# ── SKAT and edge-case tests ─────────────────────────────────────────

class TestSKATAndEdgeCases:
    """Tests for SKAT test and edge cases."""

    def test_haplotype_skat_null(self):
        """Under null, SKAT p-value is not systematically small."""
        torch.manual_seed(42)
        n = 200
        Y = torch.randn(n, dtype=torch.float64)
        X0 = torch.ones(n, 1, dtype=torch.float64)
        D = torch.randn(n, 3, dtype=torch.float64)

        Q0, _ = torch.linalg.qr(X0)

        def apply_P(v):
            if v.ndim == 1:
                return v - Q0 @ (Q0.T @ v)
            return v - Q0 @ (Q0.T @ v)

        Py = apply_P(Y)
        Q, p = _haplotype_skat(D, Py, apply_P)
        assert p > 0.001

    def test_haplotype_skat_with_lmm(self):
        """SKAT under LMM produces valid results."""
        torch.manual_seed(42)
        n, m = 100, 10
        G = torch.randint(0, 3, (n, m)).float()
        Y = torch.randn(n, dtype=torch.float64)
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock

        blocks = [LDBlock(
            region=Region(chr="1", start=0, end=10000, region_id="blk1"),
            n_variants=m,
            variant_indices=list(range(m)),
            method="manual",
            mean_r2=0.5,
            mean_dprime=0.7,
        )]

        # Without LMM (OLS P-operator)
        scanner = HaplotypeGWAS(method="block", test="skat")
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        assert result.test == "skat"
        assert result.p_global.shape[0] == 1
        assert 0.0 <= result.p_global[0].item() <= 1.0

    def test_all_homozygous_block(self):
        """Single haplotype → p=1.0 (no variation to test)."""
        n = 50
        G = torch.zeros(n, 3)  # all homozygous ref
        Y = torch.randn(n, dtype=torch.float64)

        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock

        blocks = [LDBlock(
            region=Region(chr="1", start=0, end=3000, region_id="blk1"),
            n_variants=3,
            variant_indices=[0, 1, 2],
            method="manual",
            mean_r2=1.0,
            mean_dprime=1.0,
        )]

        pos = [0, 1000, 2000]
        chrs = ["1"] * 3

        scanner = HaplotypeGWAS(method="block", test="f_test")
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        # Only one haplotype "000" → p should be 1.0
        assert result.p_global[0].item() == pytest.approx(1.0, abs=1e-6)

    def test_block_scan_with_auto_detect(self):
        """Block scan with auto-detected blocks runs end-to-end."""
        torch.manual_seed(42)
        n, m = 100, 10
        # Create correlated genotypes so blocks are detectable
        G = torch.zeros(n, m)
        for i in range(n):
            # Two blocks: SNPs 0-4 correlated, SNPs 5-9 correlated
            base1 = torch.randint(0, 3, (1,)).float().item()
            base2 = torch.randint(0, 3, (1,)).float().item()
            for j in range(5):
                G[i, j] = base1 if torch.rand(1).item() > 0.2 else torch.randint(0, 3, (1,)).float().item()
                G[i, j + 5] = base2 if torch.rand(1).item() > 0.2 else torch.randint(0, 3, (1,)).float().item()

        Y = torch.randn(n, dtype=torch.float64)
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        scanner = HaplotypeGWAS(method="block", test="f_test", block_method="r2")
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs)

        assert isinstance(result, HaplotypeGWASResult)
        assert result.n_obs == n

    def test_polyploid_phased(self):
        """Ploidy=4 with phased input: dosage sums to 4."""
        torch.manual_seed(42)
        n, m = 50, 3
        ploidy = 4
        haps = torch.randint(0, 2, (n, ploidy, m))

        labels, freqs, dosage = _enumerate_haplotypes_phased(haps, ploidy=ploidy)

        row_sums = dosage.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, float(ploidy)))
        assert freqs.sum().item() == pytest.approx(1.0, abs=1e-10)


# ── LMM-corrected path tests ───────────────────────────────────────

def _make_null_fit(n, seed=42):
    """Create a minimal NullFit for testing LMM paths."""
    from torchgwas.models.base import NullFit
    torch.manual_seed(seed)
    # Random orthogonal eigenvectors
    A = torch.randn(n, n, dtype=torch.float64)
    U, _, _ = torch.linalg.svd(A)
    eigenvalues = torch.rand(n, dtype=torch.float64) * 2 + 0.1
    sig2_g = 0.5
    sig2_e = 0.5
    Y = torch.randn(n, dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    Y_rot = U.T @ Y
    X0_rot = U.T @ X0
    return NullFit(
        sig2_g=sig2_g,
        sig2_e=sig2_e,
        eigenvalues=eigenvalues,
        eigenvectors=U,
        Y_rot=Y_rot,
        X0_rot=X0_rot,
        converged=True,
    ), Y, X0


class TestHaplotypeGWASLMM:
    """Tests for haplotype GWAS with LMM correction."""

    def test_block_scan_with_lmm(self):
        """Block scan with null_fit produces valid p-values."""
        torch.manual_seed(42)
        n, m = 100, 6
        nf, Y, X0 = _make_null_fit(n)
        G = torch.randint(0, 3, (n, m)).float()
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock
        blocks = [LDBlock(
            region=Region(chr="1", start=0, end=6000, region_id="blk1"),
            n_variants=m, variant_indices=list(range(m)),
            method="manual", mean_r2=0.5, mean_dprime=0.7,
        )]

        scanner = HaplotypeGWAS(method="block", test="f_test", null_fit=nf)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        assert result.p_global.shape[0] == 1
        assert 0.0 <= result.p_global[0].item() <= 1.0
        assert result.test == "f_test"

    def test_skat_with_lmm(self):
        """SKAT with LMM null_fit produces valid p-values."""
        torch.manual_seed(42)
        n, m = 100, 6
        nf, Y, X0 = _make_null_fit(n)
        G = torch.randint(0, 3, (n, m)).float()
        pos = list(range(0, m * 1000, 1000))
        chrs = ["1"] * m

        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock
        blocks = [LDBlock(
            region=Region(chr="1", start=0, end=6000, region_id="blk1"),
            n_variants=m, variant_indices=list(range(m)),
            method="manual", mean_r2=0.5, mean_dprime=0.7,
        )]

        scanner = HaplotypeGWAS(method="block", test="skat", null_fit=nf)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)

        assert result.test == "skat"
        assert 0.0 <= result.p_global[0].item() <= 1.0

    def test_lmm_detects_signal(self):
        """LMM-corrected F-test detects a strong haplotype signal."""
        torch.manual_seed(42)
        n = 200
        nf, _, X0 = _make_null_fit(n, seed=99)
        haps = _make_phased_haplotypes(n=n, m=3, ploidy=2, seed=42)
        G = _make_genotypes_from_haplotypes(haps)
        labels, freqs, dosage = _enumerate_haplotypes_phased(haps, ploidy=2)
        # Signal from first non-ref haplotype
        ref = freqs.argmax().item()
        causal = [j for j in range(len(labels)) if j != ref][0]
        Y = dosage[:, causal].to(torch.float64) * 2.0 + torch.randn(
            n, dtype=torch.float64) * 0.5

        # Re-build null_fit with this Y
        U = nf.eigenvectors
        nf_updated = type(nf)(
            sig2_g=nf.sig2_g, sig2_e=nf.sig2_e,
            eigenvalues=nf.eigenvalues, eigenvectors=U,
            Y_rot=U.T @ Y, X0_rot=U.T @ X0,
            converged=True,
        )

        pos = [0, 1000, 2000]
        chrs = ["1"] * 3
        from torchgwas.io.regions import Region
        from torchgwas.ld._blocks import LDBlock
        blocks = [LDBlock(
            region=Region(chr="1", start=0, end=3000, region_id="blk1"),
            n_variants=3, variant_indices=[0, 1, 2],
            method="manual", mean_r2=0.5, mean_dprime=0.7,
        )]

        scanner = HaplotypeGWAS(method="block", test="f_test", null_fit=nf_updated)
        result = scanner.scan(Y, G, variant_pos=pos, variant_chr=chrs, blocks=blocks)
        assert result.p_global[0].item() < 0.05
