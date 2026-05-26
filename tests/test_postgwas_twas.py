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


# ===================================================================
# Observed-expression TWAS (no pre-trained weights)
# ===================================================================

class TestObservedExpressionTWAS:
    """Tests for twas_observed_expression: per-gene association on already-
    normalized RNA-seq expression matrices, without eQTL weights.

    Exercises the OLS path (kinship=None), the LMM path (kinship
    supplied), covariate handling, standardization, multiple-testing
    correction, and gene-annotation propagation.
    """

    @staticmethod
    def _simulate(n: int, n_genes: int, causal_idx: int = 0,
                  effect: float = 0.5, noise: float = 0.1,
                  seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        expr = torch.randn(n, n_genes, generator=gen, dtype=torch.float64)
        y = effect * expr[:, causal_idx] + noise * torch.randn(
            n, generator=gen, dtype=torch.float64,
        )
        gene_ids = [f"G{i:03d}" for i in range(n_genes)]
        return expr, y, gene_ids

    def test_observed_expression_ols_recovers_planted_gene(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(
            n=200, n_genes=10, causal_idx=4, effect=0.5,
        )
        res = twas_observed_expression(expr, y, gene_ids)
        assert res.n_genes_tested == 10
        causal = res.genes[4]
        assert causal.p_twas < 1e-3, f"causal p_twas = {causal.p_twas}"
        assert abs(causal.beta - 0.5) < 0.1, f"causal beta = {causal.beta}"
        non_causal_p = [g.p_twas for i, g in enumerate(res.genes) if i != 4]
        # Non-causal genes shouldn't all be near zero. Loose threshold
        # absorbs the small numerical shift from switching to the
        # full-model SE estimator (matches FUSION/PLINK convention).
        assert min(non_causal_p) > 5e-3

    def test_observed_expression_ols_with_covariates(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(n=200, n_genes=5, causal_idx=2,
                                           effect=0.5, seed=11)
        gen = torch.Generator().manual_seed(99)
        covs = torch.randn(200, 5, generator=gen, dtype=torch.float64)
        r_no = twas_observed_expression(expr, y, gene_ids)
        r_with = twas_observed_expression(expr, y, gene_ids, covariates=covs)
        b1, s1 = r_no.genes[2].beta, r_no.genes[2].se
        b2, s2 = r_with.genes[2].beta, r_with.genes[2].se
        assert abs(b1 - b2) < 2.0 * max(s1, s2)

    def test_observed_expression_lmm_matches_ols_under_identity_kinship(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(n=150, n_genes=4, causal_idx=1,
                                           effect=0.5, seed=23)
        I = torch.eye(150, dtype=torch.float64)
        r_ols = twas_observed_expression(expr, y, gene_ids)
        r_lmm = twas_observed_expression(expr, y, gene_ids, kinship=I)
        for g_ols, g_lmm in zip(r_ols.genes, r_lmm.genes):
            assert abs(g_ols.beta - g_lmm.beta) < 1e-9, (
                f"OLS β={g_ols.beta:.6g}, LMM β={g_lmm.beta:.6g}"
            )

    def test_observed_expression_lmm_with_nontrivial_kinship(self):
        """LMM with a non-identity K and a polygenic phenotype must
        differ from OLS. Build y = β·expr + u + e with
        u ~ N(0, σ²_g·K) so REML estimates σ²_g > 0 and the LMM path
        materially diverges from OLS."""
        from torchgwas.postgwas import twas_observed_expression
        n, n_genes = 120, 3
        gen = torch.Generator().manual_seed(37)
        expr = torch.randn(n, n_genes, generator=gen, dtype=torch.float64)
        # Non-trivial PSD kinship.
        A = torch.randn(n, 200, generator=gen, dtype=torch.float64)
        K = (A @ A.T) / 200.0
        # Polygenic random effect tied to K.
        L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
        u = L @ torch.randn(n, generator=gen, dtype=torch.float64)
        y = 0.6 * expr[:, 0] + u + 0.3 * torch.randn(
            n, generator=gen, dtype=torch.float64,
        )
        gene_ids = [f"G{i}" for i in range(n_genes)]
        r_ols = twas_observed_expression(expr, y, gene_ids)
        r_lmm = twas_observed_expression(expr, y, gene_ids, kinship=K)
        # Both should detect the causal gene, but the LMM Wald statistic
        # for the causal gene must differ from the OLS Wald by more than
        # FP precision (REML now has σ²_g > 0).
        assert r_ols.genes[0].p_twas < 1e-3
        assert r_lmm.genes[0].p_twas < 1e-3
        assert abs(r_ols.genes[0].z_twas - r_lmm.genes[0].z_twas) > 1e-3, (
            f"OLS z={r_ols.genes[0].z_twas:.6g}, "
            f"LMM z={r_lmm.genes[0].z_twas:.6g}; LMM should diverge from "
            "OLS when σ²_g > 0 under a non-identity K."
        )

    def test_observed_expression_standardize_preserves_p_values(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(n=200, n_genes=6, causal_idx=3,
                                           effect=0.5, seed=53)
        r_raw = twas_observed_expression(expr, y, gene_ids)
        r_std = twas_observed_expression(expr, y, gene_ids, standardize=True)
        for g_raw, g_std in zip(r_raw.genes, r_std.genes):
            assert abs(g_raw.p_twas - g_std.p_twas) < 1e-12

    def test_observed_expression_null_gene_high_p(self):
        from torchgwas.postgwas import twas_observed_expression
        p_vals = []
        for seed in range(50):
            gen = torch.Generator().manual_seed(seed)
            expr = torch.randn(150, 1, generator=gen, dtype=torch.float64)
            y = torch.randn(150, generator=gen, dtype=torch.float64)
            res = twas_observed_expression(expr, y, ["G0"])
            p_vals.append(res.genes[0].p_twas)
        mean_p = sum(p_vals) / len(p_vals)
        assert 0.35 < mean_p < 0.65, f"mean p under null = {mean_p}"

    def test_observed_expression_bonferroni_count(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(n=200, n_genes=20, causal_idx=5,
                                           effect=0.6, seed=71)
        res = twas_observed_expression(
            expr, y, gene_ids, correction="bonferroni", p_threshold=0.05,
        )
        direct = sum(1 for g in res.genes if g.p_twas <= 0.05 / 20)
        assert res.n_significant == direct

    def test_observed_expression_gene_annotation_propagates(self):
        from torchgwas.postgwas import twas_observed_expression
        expr, y, gene_ids = self._simulate(n=100, n_genes=3, seed=83)
        ann = {
            "G000": {"chr": "1", "start": 100, "end": 200,
                     "gene_name": "GENE_A"},
            "G001": {"chr": "2", "start": 300, "end": 400,
                     "gene_name": "GENE_B"},
        }
        res = twas_observed_expression(expr, y, gene_ids,
                                       gene_annotation=ann)
        assert res.genes[0].chr == "1"
        assert res.genes[0].start == 100
        assert res.genes[0].gene_name == "GENE_A"
        assert res.genes[1].chr == "2"
        assert res.genes[1].gene_name == "GENE_B"
        assert res.genes[2].chr is None
        assert res.genes[2].gene_name is None

    def test_observed_expression_shape_mismatch_raises(self):
        from torchgwas.postgwas import twas_observed_expression
        expr = torch.randn(50, 5, dtype=torch.float64)
        y = torch.randn(50, dtype=torch.float64)
        with pytest.raises(ValueError, match="gene_ids"):
            twas_observed_expression(expr, y, ["only_one_id"])
        with pytest.raises(ValueError, match="phenotype"):
            twas_observed_expression(
                torch.randn(50, 3, dtype=torch.float64),
                torch.randn(40, dtype=torch.float64),
                ["G0", "G1", "G2"],
            )


# ===================================================================
# Multi-tissue stacking + S-MultiXcan-style aggregation
# ===================================================================

class TestMultiTissueStack:
    """Tests for twas_multi_tissue_stack and twas_multi_tissue_aggregate."""

    @staticmethod
    def _build_per_tissue() -> dict:
        """Two tissues, three genes; tissue B has one extra gene."""
        from torchgwas.postgwas._twas import TWASGeneResult, TWASResult
        tissue_A = TWASResult(
            genes=[
                TWASGeneResult(gene_id="G1", z_twas=4.0, p_twas=6.3e-05,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.4, se=0.1, gene_name="GENE_1"),
                TWASGeneResult(gene_id="G2", z_twas=-1.0, p_twas=0.32,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=-0.1, se=0.1, gene_name="GENE_2"),
                TWASGeneResult(gene_id="G3", z_twas=0.5, p_twas=0.62,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.05, se=0.1, gene_name="GENE_3"),
            ],
            n_genes_tested=3, n_significant=1,
        )
        tissue_B = TWASResult(
            genes=[
                TWASGeneResult(gene_id="G1", z_twas=3.5, p_twas=4.7e-04,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.35, se=0.1, gene_name="GENE_1"),
                TWASGeneResult(gene_id="G2", z_twas=2.0, p_twas=0.046,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.2, se=0.1, gene_name="GENE_2"),
                TWASGeneResult(gene_id="G3", z_twas=0.1, p_twas=0.92,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.01, se=0.1, gene_name="GENE_3"),
                TWASGeneResult(gene_id="G4", z_twas=5.0, p_twas=5.7e-07,
                               n_cis_snps=0, r2_model=None, top_weight_snp=None,
                               beta=0.5, se=0.1, gene_name="GENE_4"),
            ],
            n_genes_tested=4, n_significant=2,
        )
        return {"Liver": tissue_A, "Blood": tissue_B}

    def test_stack_long_format_row_count_and_order(self):
        from torchgwas.postgwas import twas_multi_tissue_stack
        per_t = self._build_per_tissue()
        rows = twas_multi_tissue_stack(per_t)
        # 3 + 4 = 7 rows
        assert len(rows) == 7
        # Sorted by (gene_id, tissue) — G1 entries first, alphabetical
        # tissue ("Blood" before "Liver").
        assert (rows[0].gene_id, rows[0].tissue) == ("G1", "Blood")
        assert (rows[1].gene_id, rows[1].tissue) == ("G1", "Liver")
        assert rows[-1].gene_id == "G4"
        assert rows[-1].tissue == "Blood"

    def test_stack_preserves_per_gene_fields(self):
        from torchgwas.postgwas import twas_multi_tissue_stack
        per_t = self._build_per_tissue()
        rows = twas_multi_tissue_stack(per_t)
        g1_blood = next(r for r in rows if r.gene_id == "G1" and r.tissue == "Blood")
        assert g1_blood.z_twas == 3.5
        assert g1_blood.beta == 0.35
        assert g1_blood.gene_name == "GENE_1"

    def test_aggregate_chi2_and_p_multixcan(self):
        """S-MultiXcan-style approx: chi² = sum_t z_t² ~ χ²(n_tissues)."""
        from scipy.stats import chi2 as _chi2
        from torchgwas.postgwas import twas_multi_tissue_aggregate
        per_t = self._build_per_tissue()
        summaries = twas_multi_tissue_aggregate(per_t)
        # All four unique genes
        assert {s.gene_id for s in summaries} == {"G1", "G2", "G3", "G4"}
        # G1 in both: chi² = 4.0² + 3.5² = 28.25, df = 2
        g1 = next(s for s in summaries if s.gene_id == "G1")
        assert g1.n_tissues == 2
        assert abs(g1.chi2_multixcan - (16.0 + 12.25)) < 1e-12
        expected_p = float(_chi2.sf(28.25, df=2))
        assert abs(g1.p_multixcan - expected_p) < 1e-12
        # G4 in only one tissue: df = 1.
        g4 = next(s for s in summaries if s.gene_id == "G4")
        assert g4.n_tissues == 1
        assert abs(g4.chi2_multixcan - 25.0) < 1e-12

    def test_aggregate_driver_tissue_is_min_p(self):
        from torchgwas.postgwas import twas_multi_tissue_aggregate
        per_t = self._build_per_tissue()
        summaries = twas_multi_tissue_aggregate(per_t)
        # G1: Liver p=6.3e-05 vs Blood p=4.7e-04. Liver drives.
        g1 = next(s for s in summaries if s.gene_id == "G1")
        assert g1.driver_tissue == "Liver"
        # G2: Liver p=0.32 vs Blood p=0.046. Blood drives.
        g2 = next(s for s in summaries if s.gene_id == "G2")
        assert g2.driver_tissue == "Blood"
        # Output should be sorted by p_multixcan ascending.
        ps = [s.p_multixcan for s in summaries]
        assert ps == sorted(ps)
