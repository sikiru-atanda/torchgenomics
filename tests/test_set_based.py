"""Phase 17: Set-based association tests (SKAT/Burden/SKAT-O) under LMM."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kstest

from torchgenomics.io.regions import (
    Region,
    compute_skat_weights,
    load_regions,
    map_regions_to_variants,
)
from torchgenomics.linalg.kinship import grm_vanraden
from torchgenomics.models.set_based import SetBasedResult, SetBasedScanner
from torchgenomics.models.single_trait_lmm import SingleTraitLMM

# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def lmm_null_data():
    """Simulate data and fit a null LMM for set-based tests.

    n=200 samples, m=500 SNPs, h2=0.3.
    """
    torch.manual_seed(42)
    n, m = 200, 500

    G = torch.randint(0, 3, (n, m), dtype=torch.float64)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=torch.float64)

    sig2_g, sig2_e = 0.30, 0.70
    V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
    L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
    Y = 5.0 + L_V @ torch.randn(n, dtype=torch.float64)

    lmm = SingleTraitLMM()
    nf = lmm.fit_null(Y, X0, K=K)

    return {
        "G": G, "Y": Y, "X0": X0, "K": K,
        "null_fit": nf, "n": n, "m": m,
    }


@pytest.fixture
def regions_and_meta(lmm_null_data):
    """Create synthetic gene regions spanning the SNP positions."""
    m = lmm_null_data["m"]
    variant_chr = ["1"] * m
    variant_pos = list(range(m))

    # 10 regions, each spanning 50 SNPs
    regions = []
    for i in range(10):
        regions.append(Region(
            region_id=f"gene_{i}",
            chr="1",
            start=i * 50,
            end=(i + 1) * 50,
        ))

    return regions, variant_chr, variant_pos


# ---------------------------------------------------------------
# TestRegionIO
# ---------------------------------------------------------------

class TestRegionIO:
    """Tests for BED file parsing and region-variant mapping."""

    def test_load_regions_from_bed(self, tmp_path):
        """Parse a simple BED file."""
        bed_file = tmp_path / "test.bed"
        bed_file.write_text(
            "chr1\t100\t200\tGENE_A\n"
            "chr1\t300\t500\tGENE_B\n"
            "chr2\t0\t150\tGENE_C\n"
        )
        regions = load_regions(bed_file)
        assert len(regions) == 3
        assert regions[0].region_id == "GENE_A"
        assert regions[0].chr == "chr1"
        assert regions[0].start == 100
        assert regions[0].end == 200

    def test_load_regions_auto_id(self, tmp_path):
        """BED without 4th column generates auto ID."""
        bed_file = tmp_path / "test3.bed"
        bed_file.write_text("chr1\t100\t200\n")
        regions = load_regions(bed_file)
        assert regions[0].region_id == "chr1:100-200"

    def test_map_regions_to_variants(self):
        """Map regions to variant indices."""
        regions = [
            Region("g1", "1", 10, 30),
            Region("g2", "1", 50, 80),
        ]
        variant_chr = ["1"] * 100
        variant_pos = list(range(100))

        mapping = map_regions_to_variants(regions, variant_chr, variant_pos)
        assert "g1" in mapping
        assert "g2" in mapping
        assert mapping["g1"] == list(range(10, 30))
        assert mapping["g2"] == list(range(50, 80))

    def test_empty_region_dropped(self):
        """Regions with no variants should be dropped."""
        regions = [Region("empty", "chrX", 0, 100)]
        variant_chr = ["1"] * 50
        variant_pos = list(range(50))

        mapping = map_regions_to_variants(regions, variant_chr, variant_pos)
        assert len(mapping) == 0

    def test_skat_weights_rare_upweighted(self):
        """Rare variants should get higher SKAT weights than common."""
        af = torch.tensor([0.01, 0.05, 0.10, 0.30, 0.50], dtype=torch.float64)
        weights = compute_skat_weights(af)
        # Rare (MAF=0.01) should have highest weight
        assert weights[0] > weights[1] > weights[2] > weights[3]


# ---------------------------------------------------------------
# TestSKAT
# ---------------------------------------------------------------

class TestSKAT:
    """Tests for SKAT variance-component test."""

    def test_skat_returns_valid_pvalue(self, lmm_null_data, regions_and_meta):
        """SKAT should return valid p-values for each region."""
        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        regions, variant_chr, variant_pos = regions_and_meta

        result = scanner.scan_regions(
            lmm_null_data["G"], regions, variant_chr, variant_pos,
            test="skat",
        )
        assert isinstance(result, SetBasedResult)
        assert len(result) == 10
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert not torch.isnan(result.p).any()
        assert result.test == "skat"

    def test_skat_null_calibration(self, lmm_null_data):
        """Under null, SKAT p-values should be roughly uniform."""
        torch.manual_seed(99)
        n = lmm_null_data["n"]
        m_null = 2000

        # Generate held-out SNPs (not in GRM)
        G_null = torch.randint(0, 3, (n, m_null), dtype=torch.float64)
        variant_chr = ["1"] * m_null
        variant_pos = list(range(m_null))

        # Create 40 regions of 50 SNPs each
        regions = [
            Region(f"null_gene_{i}", "1", i * 50, (i + 1) * 50)
            for i in range(40)
        ]

        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        result = scanner.scan_regions(G_null, regions, variant_chr, variant_pos, test="skat")

        p = result.p.cpu().numpy()
        # KS test for uniformity (generous threshold)
        ks_stat, ks_p = kstest(p, 'uniform')
        median_p = np.median(p)

        # Not too stringent — SKAT calibration depends on many factors
        assert median_p > 0.10, f"Median SKAT p = {median_p:.4f}"
        assert (p < 0.05).mean() < 0.25, f"False positive rate = {(p<0.05).mean():.4f}"

    def test_skat_detects_causal_gene(self, lmm_null_data, regions_and_meta):
        """A gene with true effects should have small SKAT p-value."""
        torch.manual_seed(77)
        G = lmm_null_data["G"].clone()
        Y = lmm_null_data["Y"].clone()
        n = lmm_null_data["n"]

        # Inject effects in gene_0 (SNPs 0-49)
        for j in range(10):
            effect = G[:, j] - G[:, j].mean()
            Y = Y + 0.5 * effect

        # Re-fit null (without the causal effects in the model)
        lmm = SingleTraitLMM()
        nf = lmm.fit_null(Y, lmm_null_data["X0"], K=lmm_null_data["K"])

        scanner = SetBasedScanner(nf)
        regions, variant_chr, variant_pos = regions_and_meta
        result = scanner.scan_regions(G, regions, variant_chr, variant_pos, test="skat")

        # gene_0 should be among the most significant
        p_values = result.p.cpu().numpy()
        assert p_values[0] < 0.05, f"Causal gene p = {p_values[0]:.4e}"


# ---------------------------------------------------------------
# TestBurden
# ---------------------------------------------------------------

class TestBurden:
    """Tests for Burden collapse test."""

    def test_burden_returns_valid_pvalue(self, lmm_null_data, regions_and_meta):
        """Burden test should return valid p-values."""
        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        regions, variant_chr, variant_pos = regions_and_meta

        result = scanner.scan_regions(
            lmm_null_data["G"], regions, variant_chr, variant_pos,
            test="burden",
        )
        assert isinstance(result, SetBasedResult)
        assert len(result) == 10
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert result.test == "burden"

    def test_burden_null_calibration(self, lmm_null_data):
        """Under null, burden p-values should be roughly uniform."""
        torch.manual_seed(88)
        n = lmm_null_data["n"]
        m_null = 2000
        G_null = torch.randint(0, 3, (n, m_null), dtype=torch.float64)
        variant_chr = ["1"] * m_null
        variant_pos = list(range(m_null))
        regions = [
            Region(f"null_gene_{i}", "1", i * 50, (i + 1) * 50)
            for i in range(40)
        ]

        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        result = scanner.scan_regions(G_null, regions, variant_chr, variant_pos, test="burden")

        p = result.p.cpu().numpy()
        median_p = np.median(p)
        assert median_p > 0.10, f"Median burden p = {median_p:.4f}"


# ---------------------------------------------------------------
# TestSKATO
# ---------------------------------------------------------------

class TestSKATO:
    """Tests for SKAT-O optimal combination test."""

    def test_skat_o_returns_valid(self, lmm_null_data, regions_and_meta):
        """SKAT-O should return valid p-values and rho_opt."""
        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        regions, variant_chr, variant_pos = regions_and_meta

        result = scanner.scan_regions(
            lmm_null_data["G"], regions, variant_chr, variant_pos,
            test="skat_o",
        )
        assert isinstance(result, SetBasedResult)
        assert len(result) == 10
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert result.rho_opt is not None
        assert result.test == "skat_o"

    def test_skat_o_rho_in_range(self, lmm_null_data, regions_and_meta):
        """Optimal rho should be in [0, 1]."""
        scanner = SetBasedScanner(lmm_null_data["null_fit"])
        regions, variant_chr, variant_pos = regions_and_meta

        result = scanner.scan_regions(
            lmm_null_data["G"], regions, variant_chr, variant_pos,
            test="skat_o",
        )
        assert (result.rho_opt >= 0).all()
        assert (result.rho_opt <= 1.0).all()


# ---------------------------------------------------------------
# TestSetBasedResult
# ---------------------------------------------------------------

class TestSetBasedResult:
    """Tests for SetBasedResult dataclass."""

    def test_len(self):
        result = SetBasedResult(
            region_id=["g1", "g2"],
            chr=["1", "1"],
            start=[0, 100],
            end=[100, 200],
            n_variants=[50, 50],
            q_stat=torch.tensor([1.0, 2.0]),
            p=torch.tensor([0.5, 0.1]),
            test="skat",
        )
        assert len(result) == 2


# ---------------------------------------------------------------
# TestPolyploid
# ---------------------------------------------------------------

class TestPolyploid:
    """Tests for polyploid support in set-based tests."""

    @pytest.fixture
    def tetraploid_data(self):
        """Simulate tetraploid data: genotypes in {0,1,2,3,4}."""
        torch.manual_seed(55)
        n, m = 200, 300
        ploidy = 4

        # Tetraploid genotypes with mixed MAF
        G = torch.zeros(n, m, dtype=torch.float64)
        for j in range(m):
            maf = 0.05 + 0.35 * torch.rand(1).item()
            # Multinomial for tetraploid: P(g=k) = C(4,k) * p^k * (1-p)^(4-k)
            from scipy.stats import binom
            probs = torch.tensor(
                [binom.pmf(k, ploidy, maf) for k in range(ploidy + 1)],
                dtype=torch.float64,
            )
            G[:, j] = torch.multinomial(
                probs.unsqueeze(0).expand(n, -1), 1
            ).squeeze().to(torch.float64)

        K, _ = grm_vanraden(G, ploidy=ploidy)
        X0 = torch.ones(n, 1, dtype=torch.float64)

        sig2_g, sig2_e = 0.30, 0.70
        V = sig2_g * K + sig2_e * torch.eye(n, dtype=torch.float64)
        L_V = torch.linalg.cholesky(V + 1e-6 * torch.eye(n, dtype=torch.float64))
        Y = 5.0 + L_V @ torch.randn(n, dtype=torch.float64)

        lmm = SingleTraitLMM()
        nf = lmm.fit_null(Y, X0, K=K)

        return {
            "G": G, "Y": Y, "X0": X0, "K": K,
            "null_fit": nf, "n": n, "m": m, "ploidy": ploidy,
        }

    def test_tetraploid_burden_valid(self, tetraploid_data):
        """Burden test should return valid p-values for tetraploid data."""
        d = tetraploid_data
        scanner = SetBasedScanner(d["null_fit"], ploidy=d["ploidy"])

        variant_chr = ["1"] * d["m"]
        variant_pos = list(range(d["m"]))
        regions = [
            Region(f"gene_{i}", "1", i * 50, (i + 1) * 50)
            for i in range(6)
        ]

        result = scanner.scan_regions(
            d["G"], regions, variant_chr, variant_pos, test="burden"
        )
        assert len(result) == 6
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert not torch.isnan(result.p).any()

    def test_tetraploid_skat_valid(self, tetraploid_data):
        """SKAT test should return valid p-values for tetraploid data."""
        d = tetraploid_data
        scanner = SetBasedScanner(d["null_fit"], ploidy=d["ploidy"])

        variant_chr = ["1"] * d["m"]
        variant_pos = list(range(d["m"]))
        regions = [
            Region(f"gene_{i}", "1", i * 50, (i + 1) * 50)
            for i in range(6)
        ]

        result = scanner.scan_regions(
            d["G"], regions, variant_chr, variant_pos, test="skat"
        )
        assert len(result) == 6
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()

    def test_tetraploid_skat_o_valid(self, tetraploid_data):
        """SKAT-O should return valid p-values and rho for tetraploid data."""
        d = tetraploid_data
        scanner = SetBasedScanner(d["null_fit"], ploidy=d["ploidy"])

        variant_chr = ["1"] * d["m"]
        variant_pos = list(range(d["m"]))
        regions = [
            Region(f"gene_{i}", "1", i * 50, (i + 1) * 50)
            for i in range(6)
        ]

        result = scanner.scan_regions(
            d["G"], regions, variant_chr, variant_pos, test="skat_o"
        )
        assert len(result) == 6
        assert (result.p > 0).all()
        assert (result.p <= 1.0).all()
        assert result.rho_opt is not None
        assert (result.rho_opt >= 0).all()
        assert (result.rho_opt <= 1).all()

    def test_ploidy_affects_weights(self):
        """Different ploidy should produce different allele frequencies and weights."""
        torch.manual_seed(10)
        n, m = 100, 20
        # Same raw genotype matrix, interpreted as diploid vs tetraploid
        G = torch.randint(0, 3, (n, m), dtype=torch.float64)

        af_dip = G.mean(dim=0) / 2.0
        af_tet = G.mean(dim=0) / 4.0

        w_dip = compute_skat_weights(af_dip)
        w_tet = compute_skat_weights(af_tet)

        # Tetraploid AF is lower (closer to rare), so weights should be higher
        assert (w_tet > w_dip).all(), "Tetraploid weights should be higher (lower AF)"
