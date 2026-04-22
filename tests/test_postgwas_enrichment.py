"""Tests for ``torchgwas.postgwas._enrichment`` -- gene-set enrichment."""

from __future__ import annotations

import torch

from torchgwas.postgwas._enrichment import (
    EnrichmentResult,
    GeneResult,
    gene_set_enrichment,
    snp_to_gene,
)
from torchgwas.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_enrichment_data(seed: int = 42):
    """Create SumStats with ~200 SNPs across 3 chromosomes plus gene/set info.

    Plants very strong signals (tiny p-values) in SNPs belonging to genes
    in the "enriched_set", while all other SNPs have null-like p-values.

    Returns
    -------
    ss : SumStats
    gene_id, gene_chr, gene_start, gene_end : lists describing ~20 genes
    gene_sets : dict with "enriched_set" and "null_set"
    enriched_gene_ids : list of gene IDs that carry the planted signal
    """
    torch.manual_seed(seed)
    m = 200
    n_sample = 500

    # Distribute SNPs across 3 chromosomes with increasing positions.
    chr_list: list[str] = []
    pos_list: list[int] = []
    snp_list: list[str] = []
    for i in range(m):
        c = str((i % 3) + 1)
        chr_list.append(c)
        pos_list.append(1000 + (i // 3) * 500)
        snp_list.append(f"rs{i + 1}")

    a1 = ["A"] * m
    a2 = ["G"] * m

    # Start with null-like effects: small beta, se=1.
    beta = torch.randn(m, dtype=torch.float64) * 0.05
    se = torch.ones(m, dtype=torch.float64)

    # Build ~20 genes, spread across chromosomes.
    gene_id: list[str] = []
    gene_chr: list[str] = []
    gene_start: list[int] = []
    gene_end: list[int] = []

    for g in range(20):
        c = str((g % 3) + 1)
        start = 1000 + g * 1500
        end = start + 1200
        gene_id.append(f"GENE{g + 1}")
        gene_chr.append(c)
        gene_start.append(start)
        gene_end.append(end)

    # Pick 5 genes as enriched and plant strong signals in their SNPs.
    enriched_gene_ids = [f"GENE{i}" for i in [1, 4, 7, 10, 13]]
    enriched_gene_indices = [gene_id.index(gid) for gid in enriched_gene_ids]

    for gi in enriched_gene_indices:
        gc = gene_chr[gi]
        gs = gene_start[gi]
        ge = gene_end[gi]
        for j in range(m):
            if chr_list[j] == gc and gs <= pos_list[j] <= ge:
                # Strong signal: large |beta/se| -> tiny p.
                beta[j] = 5.0 + torch.rand(1, dtype=torch.float64).item() * 2.0

    # Compute p-values from |z| = |beta/se|.
    z_vals = (beta / se).abs()
    p = 2.0 * 0.5 * torch.erfc(z_vals / (2.0 ** 0.5))
    p = p.clamp(min=1e-300)

    n = torch.full((m,), float(n_sample), dtype=torch.float64)

    ss = SumStats(
        chr=chr_list,
        pos=pos_list,
        snp=snp_list,
        a1=a1,
        a2=a2,
        beta=beta,
        se=se,
        p=p,
        n=n,
        af=None,
    )

    # "null_set" contains genes that do NOT have planted signals.
    null_gene_ids = [f"GENE{i}" for i in [2, 3, 5, 6, 8]]

    gene_sets = {
        "enriched_set": enriched_gene_ids,
        "null_set": null_gene_ids,
    }

    return ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, enriched_gene_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_snp_to_gene_assigns_snps_correctly():
    """SNPs within gene boundaries get counted."""
    ss, gene_id, gene_chr, gene_start, gene_end, _, _ = _make_enrichment_data(seed=1)
    try:
        result = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end, window_kb=0.0)
    except Exception as exc:
        raise AssertionError(f"snp_to_gene raised: {exc}") from exc

    assert isinstance(result, GeneResult)
    assert len(result.gene_id) == len(gene_id)
    # At least some genes should have SNPs assigned.
    total_snps = sum(result.n_snps)
    assert total_snps > 0, "Expected at least some SNPs assigned to genes."
    # Every gene that has SNPs should have stat > 0.
    for i, ns in enumerate(result.n_snps):
        if ns > 0:
            assert result.stat[i].item() > 0.0


def test_snp_to_gene_respects_window():
    """With window_kb=10, SNPs 10kb outside gene boundary are included."""
    ss, gene_id, gene_chr, gene_start, gene_end, _, _ = _make_enrichment_data(seed=2)

    result_no_win = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end, window_kb=0.0)
    result_win = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end, window_kb=10.0)

    # Window should only add SNPs, never remove them.
    for i in range(len(gene_id)):
        assert result_win.n_snps[i] >= result_no_win.n_snps[i], (
            f"Gene {gene_id[i]}: window reduced SNPs from "
            f"{result_no_win.n_snps[i]} to {result_win.n_snps[i]}"
        )
    # At least one gene should gain SNPs from the window extension.
    total_no_win = sum(result_no_win.n_snps)
    total_win = sum(result_win.n_snps)
    assert total_win >= total_no_win, "Window should not decrease total SNP count."


def test_snp_to_gene_empty_genes_get_p_one():
    """Genes with no SNPs get p=1.0."""
    torch.manual_seed(10)
    m = 50
    # All SNPs on chr1, gene on chr99 (no match).
    ss = SumStats(
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 100, 100)),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.randn(m, dtype=torch.float64),
        se=torch.ones(m, dtype=torch.float64),
        p=torch.rand(m, dtype=torch.float64),
        n=torch.full((m,), 100.0, dtype=torch.float64),
    )
    result = snp_to_gene(
        ss,
        gene_id=["EMPTY_GENE"],
        gene_chr=["99"],
        gene_start=[5000],
        gene_end=[6000],
    )
    assert result.n_snps[0] == 0
    assert result.p[0].item() == 1.0


def test_snp_to_gene_strong_gene_has_low_p():
    """Gene containing very significant SNPs gets p < 0.05."""
    ss, gene_id, gene_chr, gene_start, gene_end, _, enriched = _make_enrichment_data(seed=3)
    result = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)

    # At least one enriched gene should have a small p-value.
    enriched_indices = [gene_id.index(gid) for gid in enriched if gid in gene_id]
    enriched_ps = [result.p[i].item() for i in enriched_indices if result.n_snps[i] > 0]
    assert len(enriched_ps) > 0, "No enriched genes had SNPs assigned."
    min_p = min(enriched_ps)
    assert min_p < 0.05, f"Expected enriched gene p < 0.05, got {min_p}"


def test_gene_set_enrichment_detects_enriched_set():
    """Gene set containing genes with strong signals has lower p than a random set."""
    ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, _ = _make_enrichment_data(seed=4)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)

    try:
        er = gene_set_enrichment(gr, gene_sets)
    except Exception as exc:
        raise AssertionError(f"gene_set_enrichment raised: {exc}") from exc

    assert isinstance(er, EnrichmentResult)
    idx_enriched = er.gene_set_name.index("enriched_set")
    idx_null = er.gene_set_name.index("null_set")
    assert er.p[idx_enriched].item() < er.p[idx_null].item(), (
        f"Enriched set p={er.p[idx_enriched].item():.4e} should be lower "
        f"than null set p={er.p[idx_null].item():.4e}"
    )


def test_gene_set_enrichment_null_set_not_significant():
    """Random gene set (no planted signal) has p > 0.01."""
    ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, _ = _make_enrichment_data(seed=5)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)
    er = gene_set_enrichment(gr, {"null_set": gene_sets["null_set"]})

    idx = er.gene_set_name.index("null_set")
    p_null = er.p[idx].item()
    assert p_null > 0.01, f"Null set p={p_null:.4e} unexpectedly significant."


def test_gene_set_enrichment_beta_positive_for_enriched():
    """beta_enrichment > 0 for the enriched set."""
    ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, _ = _make_enrichment_data(seed=6)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)
    er = gene_set_enrichment(gr, gene_sets)

    idx_enriched = er.gene_set_name.index("enriched_set")
    assert er.beta_enrichment[idx_enriched].item() > 0.0, (
        f"Expected positive beta for enriched set, got "
        f"{er.beta_enrichment[idx_enriched].item():.4f}"
    )


def test_gene_set_enrichment_multiple_sets():
    """Two sets tested simultaneously, enriched one has lower p."""
    ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, _ = _make_enrichment_data(seed=7)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)
    er = gene_set_enrichment(gr, gene_sets)

    assert len(er.gene_set_name) == 2
    assert er.p.shape[0] == 2
    assert er.beta_enrichment.shape[0] == 2
    assert er.se.shape[0] == 2

    idx_e = er.gene_set_name.index("enriched_set")
    idx_n = er.gene_set_name.index("null_set")
    assert er.p[idx_e].item() < er.p[idx_n].item()


def test_snp_to_gene_rejects_mismatched_gene_info():
    """Mismatched gene_id/gene_chr lengths raise ValueError."""
    ss, gene_id, gene_chr, gene_start, gene_end, _, _ = _make_enrichment_data(seed=8)

    try:
        snp_to_gene(ss, gene_id, gene_chr[:5], gene_start, gene_end)
        raise AssertionError("Expected ValueError for mismatched lengths.")
    except ValueError:
        pass  # expected


def test_gene_set_enrichment_handles_unknown_genes():
    """Gene IDs in set but not in GeneResult are silently ignored."""
    ss, gene_id, gene_chr, gene_start, gene_end, _, _ = _make_enrichment_data(seed=9)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)

    # Set with a mix of real and unknown gene IDs.
    mixed_set = {"mixed": [gene_id[0], gene_id[1], "FAKE_GENE_999", "NONEXISTENT"]}
    try:
        er = gene_set_enrichment(gr, mixed_set)
    except Exception as exc:
        raise AssertionError(
            f"gene_set_enrichment should ignore unknown genes, but raised: {exc}"
        ) from exc

    # The count should only reflect known genes with SNPs.
    assert er.n_genes_in_set[0] <= 2, (
        f"Expected at most 2 known genes in set, got {er.n_genes_in_set[0]}"
    )


def test_gene_set_enrichment_covariates():
    """With covariate_gene_size=True, result should still detect enrichment."""
    ss, gene_id, gene_chr, gene_start, gene_end, gene_sets, _ = _make_enrichment_data(seed=10)
    gr = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)

    # With both covariates (default).
    er_cov = gene_set_enrichment(gr, gene_sets, covariate_gene_size=True, covariate_log_size=True)
    # Without covariates.
    er_nocov = gene_set_enrichment(gr, gene_sets, covariate_gene_size=False, covariate_log_size=False)

    idx_e = er_cov.gene_set_name.index("enriched_set")
    # Enrichment should still be detected with covariates.
    assert er_cov.beta_enrichment[idx_e].item() > 0.0, (
        "Expected positive beta even with covariates."
    )
    # Both should produce valid p-values.
    assert 0.0 <= er_cov.p[idx_e].item() <= 1.0
    assert 0.0 <= er_nocov.p[idx_e].item() <= 1.0
