"""Benchmark: torchgwas gene-set enrichment vs statsmodels / scipy reference.

Validates SNP-to-gene p-values against scipy.stats.norm and the
enrichment regression against statsmodels OLS.
"""
from __future__ import annotations

import numpy as np
import torch
import statsmodels.api as sm
from scipy import stats as sp_stats

from torchgwas.postgwas._enrichment import snp_to_gene, gene_set_enrichment, GeneResult
from torchgwas.postgwas._sumstats import SumStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sumstats(chi2_values: list[float], chrs: list[str], positions: list[int]) -> SumStats:
    """Build a SumStats from raw chi-squared values.

    beta = sqrt(chi2), se = 1.0  so that chi2 = (beta/se)^2 recovers exactly.
    """
    m = len(chi2_values)
    beta = torch.tensor(chi2_values, dtype=torch.float64).sqrt()
    se = torch.ones(m, dtype=torch.float64)
    # p-values are not used by snp_to_gene (it recomputes from chi2),
    # but SumStats requires them.
    p = torch.ones(m, dtype=torch.float64)
    n = torch.full((m,), 100.0, dtype=torch.float64)
    return SumStats(
        chr=chrs,
        pos=positions,
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=beta,
        se=se,
        p=p,
        n=n,
    )


def _make_gene_result(gene_ids: list[str], n_snps: list[int],
                       p_values: list[float]) -> GeneResult:
    """Build a GeneResult with specified gene p-values and SNP counts."""
    n_genes = len(gene_ids)
    return GeneResult(
        gene_id=gene_ids,
        gene_chr=["1"] * n_genes,
        gene_start=list(range(n_genes)),
        gene_end=list(range(n_genes)),
        n_snps=n_snps,
        stat=torch.ones(n_genes, dtype=torch.float64),  # not used downstream
        p=torch.tensor(p_values, dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Test 1: gene p-values vs scipy.stats.norm
# ---------------------------------------------------------------------------

def test_bench_gene_pvalue_matches_scipy_norm():
    """Gene-level z-test p-values must agree with scipy.stats.norm.sf."""
    # 3 genes, each spanning a contiguous range of SNPs on chr "1".
    # Gene A: SNPs 0-4 (5 SNPs), Gene B: SNPs 5-9 (5 SNPs), Gene C: SNPs 10-14 (5 SNPs)
    np.random.seed(42)
    chi2_vals = np.random.chisquare(df=1, size=15) + 1.0  # shift to ensure >0
    chi2_list = chi2_vals.tolist()

    chrs = ["1"] * 15
    positions = list(range(100, 1600, 100))  # 100, 200, ..., 1500

    ss = _make_sumstats(chi2_list, chrs, positions)

    gene_ids = ["A", "B", "C"]
    gene_chrs = ["1", "1", "1"]
    gene_starts = [100, 600, 1100]
    gene_ends = [500, 1000, 1500]

    gr = snp_to_gene(ss, gene_ids, gene_chrs, gene_starts, gene_ends)

    # Manually compute expected p-values via scipy.
    for g_idx, (gs, ge) in enumerate(zip(gene_starts, gene_ends)):
        snp_indices = [i for i, pos in enumerate(positions) if gs <= pos <= ge]
        k = len(snp_indices)
        mean_chi2 = np.mean([chi2_list[i] for i in snp_indices])
        z = (mean_chi2 - 1.0) / np.sqrt(2.0 / k)
        expected_p = sp_stats.norm.sf(z)

        np.testing.assert_allclose(
            gr.p[g_idx].item(), expected_p, atol=1e-10,
            err_msg=f"Gene {gene_ids[g_idx]} p-value mismatch",
        )


# ---------------------------------------------------------------------------
# Test 2: enrichment beta vs statsmodels OLS
# ---------------------------------------------------------------------------

def test_bench_enrichment_beta_matches_statsmodels_ols():
    """Enrichment beta_1 must match the indicator coefficient from sm.OLS."""
    np.random.seed(123)
    n_genes = 50
    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    n_snps = np.random.randint(5, 100, size=n_genes).tolist()
    # Generate p-values with some genes having small p (enriched).
    p_values = np.random.uniform(0.001, 0.999, size=n_genes)
    p_values[:10] *= 0.01  # first 10 genes are enriched
    p_values = np.clip(p_values, 1e-300, 1.0 - 1e-15).tolist()

    gr = _make_gene_result(gene_ids, n_snps, p_values)

    # Gene set: first 10 genes.
    gene_set = {f"gene_{i}" for i in range(10)}
    result = gene_set_enrichment(gr, {"test_set": list(gene_set)})

    # Build the same regression with statsmodels.
    one_minus_p = np.clip(1.0 - np.array(p_values), 1e-300, 1.0 - 1e-15)
    z_scores = sp_stats.norm.ppf(one_minus_p)

    indicator = np.array([1.0 if gid in gene_set else 0.0 for gid in gene_ids])
    gene_size = np.array(n_snps, dtype=np.float64)
    log_size = np.log(gene_size + 1.0)

    X = np.column_stack([np.ones(n_genes), gene_size, log_size, indicator])
    ols_result = sm.OLS(z_scores, X).fit()

    # The indicator is the last column -> coef index 3.
    np.testing.assert_allclose(
        result.beta_enrichment[0].item(),
        ols_result.params[3],
        atol=1e-8,
        err_msg="beta_enrichment does not match statsmodels OLS coefficient",
    )


# ---------------------------------------------------------------------------
# Test 3: enrichment SE vs statsmodels
# ---------------------------------------------------------------------------

def test_bench_enrichment_se_matches_statsmodels():
    """Enrichment SE must match statsmodels bse for the indicator coefficient."""
    np.random.seed(456)
    n_genes = 60
    gene_ids = [f"g{i}" for i in range(n_genes)]
    n_snps = np.random.randint(3, 80, size=n_genes).tolist()
    p_values = np.random.uniform(0.01, 0.99, size=n_genes)
    p_values[:15] *= 0.05
    p_values = np.clip(p_values, 1e-300, 1.0 - 1e-15).tolist()

    gr = _make_gene_result(gene_ids, n_snps, p_values)

    gene_set = {f"g{i}" for i in range(15)}
    result = gene_set_enrichment(gr, {"pathway": list(gene_set)})

    # Statsmodels reference.
    one_minus_p = np.clip(1.0 - np.array(p_values), 1e-300, 1.0 - 1e-15)
    z_scores = sp_stats.norm.ppf(one_minus_p)

    indicator = np.array([1.0 if gid in gene_set else 0.0 for gid in gene_ids])
    gene_size = np.array(n_snps, dtype=np.float64)
    log_size = np.log(gene_size + 1.0)

    X = np.column_stack([np.ones(n_genes), gene_size, log_size, indicator])
    ols_result = sm.OLS(z_scores, X).fit()

    np.testing.assert_allclose(
        result.se[0].item(),
        ols_result.bse[3],
        atol=1e-8,
        err_msg="SE does not match statsmodels bse",
    )


# ---------------------------------------------------------------------------
# Test 4: enrichment p-value vs statsmodels
# ---------------------------------------------------------------------------

def test_bench_enrichment_pvalue_matches_statsmodels():
    """One-sided enrichment p-value must match scipy.stats.norm.sf(beta/se)."""
    np.random.seed(789)
    n_genes = 40
    gene_ids = [f"gene{i}" for i in range(n_genes)]
    n_snps = np.random.randint(5, 50, size=n_genes).tolist()
    p_values = np.random.uniform(0.01, 0.99, size=n_genes)
    p_values[:8] *= 0.02
    p_values = np.clip(p_values, 1e-300, 1.0 - 1e-15).tolist()

    gr = _make_gene_result(gene_ids, n_snps, p_values)

    gene_set = {f"gene{i}" for i in range(8)}
    result = gene_set_enrichment(gr, {"pathway_x": list(gene_set)})

    # Statsmodels reference.
    one_minus_p = np.clip(1.0 - np.array(p_values), 1e-300, 1.0 - 1e-15)
    z_scores = sp_stats.norm.ppf(one_minus_p)

    indicator = np.array([1.0 if gid in gene_set else 0.0 for gid in gene_ids])
    gene_size = np.array(n_snps, dtype=np.float64)
    log_size = np.log(gene_size + 1.0)

    X = np.column_stack([np.ones(n_genes), gene_size, log_size, indicator])
    ols_result = sm.OLS(z_scores, X).fit()

    beta_sm = ols_result.params[3]
    se_sm = ols_result.bse[3]
    expected_p = sp_stats.norm.sf(beta_sm / se_sm)

    np.testing.assert_allclose(
        result.p[0].item(),
        expected_p,
        atol=1e-8,
        err_msg="One-sided p-value does not match scipy reference",
    )


# ---------------------------------------------------------------------------
# Test 5: gene p-values for 10+ genes of varying sizes
# ---------------------------------------------------------------------------

def test_bench_gene_pvalue_multiple_genes_scipy():
    """All gene p-values must match scipy reference across 12 genes."""
    np.random.seed(2026)
    # 12 genes with sizes 3, 5, 8, 2, 10, 7, 4, 6, 15, 20, 1, 12.
    gene_sizes = [3, 5, 8, 2, 10, 7, 4, 6, 15, 20, 1, 12]
    total_snps = sum(gene_sizes)

    chi2_vals = (np.random.chisquare(df=1, size=total_snps) + 0.5).tolist()
    chrs = ["1"] * total_snps
    positions = list(range(1000, 1000 + total_snps * 100, 100))

    ss = _make_sumstats(chi2_vals, chrs, positions)

    # Build gene coordinates: each gene occupies a contiguous block.
    gene_ids = [f"GENE{i}" for i in range(12)]
    gene_chrs = ["1"] * 12
    gene_starts: list[int] = []
    gene_ends: list[int] = []
    offset = 0
    for sz in gene_sizes:
        gene_starts.append(positions[offset])
        gene_ends.append(positions[offset + sz - 1])
        offset += sz

    gr = snp_to_gene(ss, gene_ids, gene_chrs, gene_starts, gene_ends)

    # Verify each gene independently.
    offset = 0
    for g_idx, sz in enumerate(gene_sizes):
        k = sz
        gene_chi2 = chi2_vals[offset:offset + k]
        mean_chi2 = np.mean(gene_chi2)
        z = (mean_chi2 - 1.0) / np.sqrt(2.0 / k)
        expected_p = sp_stats.norm.sf(z)

        np.testing.assert_allclose(
            gr.p[g_idx].item(), expected_p, atol=1e-10,
            err_msg=f"Gene {gene_ids[g_idx]} (k={k}) p-value mismatch",
        )
        assert gr.n_snps[g_idx] == k
        offset += k


# ---------------------------------------------------------------------------
# Test 6: enrichment without covariates vs statsmodels OLS
# ---------------------------------------------------------------------------

def test_bench_enrichment_no_covariates_matches_statsmodels():
    """With no gene-size covariates, regression is z ~ 1 + indicator."""
    np.random.seed(999)
    n_genes = 30
    gene_ids = [f"g{i}" for i in range(n_genes)]
    n_snps = np.random.randint(2, 40, size=n_genes).tolist()
    p_values = np.random.uniform(0.005, 0.995, size=n_genes)
    p_values[:7] *= 0.03  # enrich first 7
    p_values = np.clip(p_values, 1e-300, 1.0 - 1e-15).tolist()

    gr = _make_gene_result(gene_ids, n_snps, p_values)

    gene_set = {f"g{i}" for i in range(7)}
    result = gene_set_enrichment(
        gr,
        {"simple_set": list(gene_set)},
        covariate_gene_size=False,
        covariate_log_size=False,
    )

    # Statsmodels: z ~ intercept + indicator (2 columns).
    one_minus_p = np.clip(1.0 - np.array(p_values), 1e-300, 1.0 - 1e-15)
    z_scores = sp_stats.norm.ppf(one_minus_p)

    indicator = np.array([1.0 if gid in gene_set else 0.0 for gid in gene_ids])
    X = np.column_stack([np.ones(n_genes), indicator])
    ols_result = sm.OLS(z_scores, X).fit()

    np.testing.assert_allclose(
        result.beta_enrichment[0].item(),
        ols_result.params[1],
        atol=1e-8,
        err_msg="beta without covariates does not match statsmodels",
    )
    np.testing.assert_allclose(
        result.se[0].item(),
        ols_result.bse[1],
        atol=1e-8,
        err_msg="SE without covariates does not match statsmodels",
    )

    expected_p = sp_stats.norm.sf(ols_result.params[1] / ols_result.bse[1])
    np.testing.assert_allclose(
        result.p[0].item(),
        expected_p,
        atol=1e-8,
        err_msg="p-value without covariates does not match scipy reference",
    )
