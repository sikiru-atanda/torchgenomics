"""Tests for Transcriptome-Wide Association Study (TWAS).

Phase 45b: S-PrediXcan (summary-stat) and PrediXcan (individual-level)
TWAS implementations.

Strategy: construct synthetic GWAS summary statistics and genotype matrices
with known eQTL weight vectors and verify that z_twas matches the analytic
formula z = w^T z / sqrt(w^T Sigma w), that LD correction adjusts the
variance correctly, and that individual-level TWAS recovers the expected
sign of association.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from torchgwas.postgwas._sumstats import SumStats
from torchgwas.postgwas._twas import (
    TWASGeneResult,
    TWASResult,
    twas_individual,
    twas_sumstat,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_sumstats(
    snps: list[str],
    betas: list[float],
    ses: list[float],
    ps: list[float],
) -> SumStats:
    """Build a minimal SumStats from per-SNP lists."""
    m = len(snps)
    return SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        snp=snps,
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.tensor(betas, dtype=torch.float64),
        se=torch.tensor(ses, dtype=torch.float64),
        p=torch.tensor(ps, dtype=torch.float64),
        n=torch.full((m,), 1000.0, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Summary-stat TWAS tests
# ---------------------------------------------------------------------------

def test_twas_identity_ld():
    """With identity LD, z_twas = w^T z / ||w||."""
    snps = ["rs0", "rs1", "rs2"]
    betas = [0.3, 0.2, 0.1]
    ses = [0.05, 0.05, 0.05]
    ps = [1e-6, 1e-4, 0.05]
    gwas = _make_sumstats(snps, betas, ses, ps)

    w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    z_gwas = torch.tensor([b / s for b, s in zip(betas, ses)],
                          dtype=torch.float64)

    expected_z = float((w * z_gwas).sum() / w.norm())

    result = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": snps},
        ld_matrix={"GENE1": torch.eye(3, dtype=torch.float64)},
    )

    assert result.n_genes_tested == 1
    gene = result.genes[0]
    assert abs(gene.z_twas - expected_z) < 1e-8


def test_twas_with_ld_correction():
    """Non-identity LD changes the variance denominator."""
    snps = ["rs0", "rs1"]
    betas = [0.3, 0.2]
    ses = [0.05, 0.05]
    ps = [1e-6, 1e-4]
    gwas = _make_sumstats(snps, betas, ses, ps)

    w = torch.tensor([0.5, 0.5], dtype=torch.float64)
    z_gwas = torch.tensor([b / s for b, s in zip(betas, ses)],
                          dtype=torch.float64)

    # Correlated LD
    rho = 0.8
    sigma = torch.tensor([[1.0, rho], [rho, 1.0]], dtype=torch.float64)
    var_denom = float((w @ sigma @ w).item())
    expected_z = float((w * z_gwas).sum()) / var_denom ** 0.5

    result = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": snps},
        ld_matrix={"GENE1": sigma},
    )

    gene = result.genes[0]
    assert abs(gene.z_twas - expected_z) < 1e-8

    # Compare to identity LD: different z_twas
    result_id = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": snps},
        ld_matrix={"GENE1": torch.eye(2, dtype=torch.float64)},
    )
    assert abs(result_id.genes[0].z_twas - gene.z_twas) > 0.01


def test_twas_no_ld_warning():
    """Warns when ld_matrix is None (identity assumed)."""
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [0.3], [0.05], [1e-6])
    w = torch.tensor([1.0], dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        twas_sumstat(
            gwas,
            weights={"GENE1": w},
            snp_lists={"GENE1": snps},
            ld_matrix=None,
        )

    assert any("LD matrix" in str(c.message) or "identity" in str(c.message).lower()
               for c in caught), "Expected a warning about missing LD matrix"


def test_twas_significant_gene():
    """Strong signal gives small p_twas."""
    snps = ["rs0", "rs1", "rs2"]
    # Large z-scores all in the same direction as weights
    betas = [0.5, 0.4, 0.3]
    ses = [0.05, 0.05, 0.05]
    ps = [1e-10, 1e-8, 1e-6]
    gwas = _make_sumstats(snps, betas, ses, ps)

    w = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    result = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": snps},
        ld_matrix={"GENE1": torch.eye(3, dtype=torch.float64)},
    )

    gene = result.genes[0]
    assert gene.p_twas < 1e-5


def test_twas_null_gene():
    """Random z-scores with small effects give non-significant p."""
    torch.manual_seed(42)
    snps = [f"rs{i}" for i in range(5)]
    betas = (torch.randn(5, dtype=torch.float64) * 0.01).tolist()
    ses = [0.1] * 5
    ps = [0.5] * 5
    gwas = _make_sumstats(snps, betas, ses, ps)

    w = torch.tensor([0.2, 0.3, 0.1, 0.2, 0.2], dtype=torch.float64)
    result = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": snps},
        ld_matrix={"GENE1": torch.eye(5, dtype=torch.float64)},
    )

    gene = result.genes[0]
    assert gene.p_twas > 0.05


# ---------------------------------------------------------------------------
# Individual-level TWAS tests
# ---------------------------------------------------------------------------

def test_twas_individual_basic():
    """Individual-level TWAS returns a result with expected fields."""
    torch.manual_seed(123)
    n, m = 200, 5
    snp_ids = [f"rs{i}" for i in range(m)]
    G = torch.randn(n, m, dtype=torch.float64)
    w = torch.tensor([0.5, 0.3, 0.1, 0.0, 0.0], dtype=torch.float64)
    grex = G[:, :3] @ w[:3]
    y = grex + torch.randn(n, dtype=torch.float64) * 0.5

    result = twas_individual(
        genotypes=G,
        phenotype=y,
        weights={"GENE1": w},
        snp_lists={"GENE1": snp_ids},
        snp_ids=snp_ids,
    )

    assert isinstance(result, TWASResult)
    assert result.n_genes_tested == 1
    gene = result.genes[0]
    assert isinstance(gene, TWASGeneResult)
    assert gene.gene_id == "GENE1"
    assert gene.n_cis_snps == 5


def test_twas_individual_matches_direction():
    """Sign of z_twas matches the causal direction."""
    torch.manual_seed(456)
    n, m = 300, 3
    snp_ids = [f"rs{i}" for i in range(m)]
    G = torch.randn(n, m, dtype=torch.float64)
    w = torch.tensor([0.8, 0.2, 0.0], dtype=torch.float64)
    grex = G @ w
    # Positive causal effect
    y = 2.0 * grex + torch.randn(n, dtype=torch.float64) * 0.3

    result = twas_individual(
        genotypes=G,
        phenotype=y,
        weights={"GENE1": w},
        snp_lists={"GENE1": snp_ids},
        snp_ids=snp_ids,
    )

    gene = result.genes[0]
    assert gene.z_twas > 0, "z_twas should be positive for positive causal effect"
    assert gene.p_twas < 0.01


# ---------------------------------------------------------------------------
# Correction and edge-case tests
# ---------------------------------------------------------------------------

def test_twas_bonferroni():
    """n_significant respects Bonferroni correction."""
    snps = [f"rs{i}" for i in range(5)]
    gwas = _make_sumstats(snps, [0.3, 0.01, 0.01, 0.01, 0.01],
                          [0.05, 0.1, 0.1, 0.1, 0.1],
                          [1e-6, 0.8, 0.8, 0.8, 0.8])

    # Three genes: only GENE_A has a strong signal
    weights = {
        "GENE_A": torch.tensor([1.0], dtype=torch.float64),
        "GENE_B": torch.tensor([1.0], dtype=torch.float64),
        "GENE_C": torch.tensor([1.0], dtype=torch.float64),
    }
    snp_lists = {
        "GENE_A": ["rs0"],
        "GENE_B": ["rs1"],
        "GENE_C": ["rs2"],
    }

    result = twas_sumstat(
        gwas,
        weights=weights,
        snp_lists=snp_lists,
        ld_matrix={
            "GENE_A": torch.eye(1, dtype=torch.float64),
            "GENE_B": torch.eye(1, dtype=torch.float64),
            "GENE_C": torch.eye(1, dtype=torch.float64),
        },
        correction="bonferroni",
        p_threshold=0.05,
    )

    assert result.n_genes_tested == 3
    # With Bonferroni and 3 genes, threshold = 0.05/3 ~ 0.017
    # Only GENE_A (very small p) should pass
    assert result.n_significant <= 1


def test_twas_empty_weights_raises():
    """Empty weights dict raises ValueError."""
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [0.3], [0.05], [1e-6])

    with pytest.raises(ValueError, match="[Aa]t least one gene"):
        twas_sumstat(gwas, weights={}, snp_lists={})


def test_twas_missing_snps():
    """SNPs in the weight vector that are not in GWAS are silently skipped."""
    # GWAS only has rs0
    gwas = _make_sumstats(["rs0"], [0.5], [0.05], [1e-8])

    # Gene has rs0 and rs_missing
    w = torch.tensor([0.5, 0.5], dtype=torch.float64)

    result = twas_sumstat(
        gwas,
        weights={"GENE1": w},
        snp_lists={"GENE1": ["rs0", "rs_missing"]},
        ld_matrix={"GENE1": torch.eye(2, dtype=torch.float64)},
    )

    # Should still produce a result using the 1 available SNP
    assert result.n_genes_tested == 1
    gene = result.genes[0]
    assert gene.n_cis_snps == 1
    assert gene.top_weight_snp == "rs0"
