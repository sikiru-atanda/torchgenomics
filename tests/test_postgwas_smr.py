"""Tests for Summary-data-based Mendelian Randomization (SMR) and HEIDI.

Phase 45a: Zhu et al. (2016) SMR chi-squared test and HEIDI heterogeneity
test for distinguishing pleiotropy/causality from linkage.

Strategy: construct synthetic GWAS and eQTL summary statistics with known
z-scores and effect-size ratios, then verify the SMR chi-squared formula,
Wald ratio, HEIDI pleiotropy-vs-linkage discrimination, and the multi-gene
wrapper's summary counts.
"""

from __future__ import annotations

import pytest
import torch

from torchgwas.postgwas._smr import (
    SMRResult,
    SMRSummary,
    heidi_test,
    smr_heidi,
    smr_test,
)
from torchgwas.postgwas._sumstats import SumStats

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
# SMR tests
# ---------------------------------------------------------------------------

def test_smr_chi2_formula():
    """Hand-compute chi2_SMR = (z_g^2 * z_e^2) / (z_g^2 + z_e^2) for known z."""
    # z_gwas = 4.0, z_eqtl = 6.0
    beta_g, se_g = 0.4, 0.1  # z = 4
    beta_e, se_e = 0.6, 0.1  # z = 6
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [beta_g], [se_g], [1e-4])
    eqtl = _make_sumstats(snps, [beta_e], [se_e], [1e-9])

    res = smr_test(gwas, eqtl, gene_id="GENE1", probe_snp="rs0")

    z_g2 = (beta_g / se_g) ** 2  # 16
    z_e2 = (beta_e / se_e) ** 2  # 36
    expected_chi2 = (z_g2 * z_e2) / (z_g2 + z_e2)  # 576 / 52 = 11.077

    assert abs(res.chi2_smr - expected_chi2) < 1e-8
    assert res.p_smr < 0.01  # chi2 ~ 11 with 1 df is very significant


def test_smr_known_ratio():
    """beta_smr should equal beta_gwas / beta_eqtl (Wald ratio)."""
    beta_g, se_g = 0.3, 0.05
    beta_e, se_e = 0.6, 0.05
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [beta_g], [se_g], [1e-4])
    eqtl = _make_sumstats(snps, [beta_e], [se_e], [1e-9])

    res = smr_test(gwas, eqtl, gene_id="G1", probe_snp="rs0")

    expected_ratio = beta_g / beta_e  # 0.5
    assert abs(res.beta_smr - expected_ratio) < 1e-10
    assert res.beta_gwas == pytest.approx(beta_g)
    assert res.beta_eqtl == pytest.approx(beta_e)


def test_smr_significant_eqtl():
    """Strong eQTL signal should produce a small p_smr."""
    snps = ["rs0", "rs1", "rs2"]
    gwas = _make_sumstats(snps, [0.3, 0.01, 0.02], [0.05, 0.1, 0.1],
                          [1e-6, 0.9, 0.8])
    eqtl = _make_sumstats(snps, [0.5, 0.01, 0.01], [0.05, 0.1, 0.1],
                          [1e-10, 0.9, 0.9])

    res = smr_test(gwas, eqtl, gene_id="G1")
    # Auto-selects rs0 as top eQTL
    assert res.probe_snp == "rs0"
    assert res.p_smr < 0.001


def test_smr_no_eqtl():
    """When no SNP passes eqtl_p_threshold, return p_smr = 1."""
    snps = ["rs0", "rs1"]
    gwas = _make_sumstats(snps, [0.3, 0.01], [0.05, 0.1], [1e-6, 0.9])
    eqtl = _make_sumstats(snps, [0.01, 0.01], [0.1, 0.1], [0.5, 0.8])

    res = smr_test(gwas, eqtl, gene_id="G1", eqtl_p_threshold=5e-8)
    assert res.p_smr == 1.0
    assert res.probe_snp == ""


def test_smr_missing_probe():
    """Probe SNP not present in GWAS returns default result."""
    gwas = _make_sumstats(["rs0"], [0.3], [0.05], [1e-6])
    eqtl = _make_sumstats(["rs99"], [0.5], [0.05], [1e-10])

    res = smr_test(gwas, eqtl, gene_id="G1", probe_snp="rs99")
    assert res.p_smr == 1.0
    assert res.chi2_smr == 0.0


# ---------------------------------------------------------------------------
# HEIDI tests
# ---------------------------------------------------------------------------

def test_heidi_single_causal():
    """All nearby SNPs have the same GWAS/eQTL ratio as probe -> large p_heidi.

    When a single causal variant drives both the GWAS and eQTL signal,
    all ratio_i ~ ratio_top, so HEIDI stat ~ 0 and p is large.
    """
    ratio = 0.5  # true causal ratio
    snps = [f"rs{i}" for i in range(6)]
    # All betas constructed so beta_gwas / beta_eqtl = ratio for every SNP
    betas_e = [0.6, 0.4, 0.3, 0.5, 0.35, 0.45]
    betas_g = [b * ratio for b in betas_e]
    ses_g = [0.05] * 6
    ses_e = [0.05] * 6
    ps_g = [1e-6] * 6
    ps_e = [1e-10] * 6

    gwas = _make_sumstats(snps, betas_g, ses_g, ps_g)
    eqtl = _make_sumstats(snps, betas_e, ses_e, ps_e)

    p_h, h_stat, n_h = heidi_test(gwas, eqtl, probe_snp="rs0",
                                   nearby_snps=snps[1:])
    assert p_h > 0.05, f"Expected large p_heidi for single causal, got {p_h}"
    assert n_h == 5


def test_heidi_linked_variants():
    """Different ratios across SNPs -> heterogeneity -> small p_heidi.

    When GWAS and eQTL are driven by different causal variants in LD,
    the per-SNP ratios diverge strongly from the probe ratio.
    """
    snps = [f"rs{i}" for i in range(6)]
    # Probe rs0: ratio = 0.3/0.6 = 0.5
    # Nearby SNPs: very different ratios
    betas_g = [0.3, 0.8, -0.5, 0.7, -0.3, 0.6]
    betas_e = [0.6, 0.1, 0.5, 0.05, 0.4, 0.08]
    ses_g = [0.02] * 6
    ses_e = [0.02] * 6
    ps_g = [1e-6] * 6
    ps_e = [1e-10] * 6

    gwas = _make_sumstats(snps, betas_g, ses_g, ps_g)
    eqtl = _make_sumstats(snps, betas_e, ses_e, ps_e)

    p_h, h_stat, n_h = heidi_test(gwas, eqtl, probe_snp="rs0",
                                   nearby_snps=snps[1:])
    assert p_h < 0.05, f"Expected small p_heidi for linked variants, got {p_h}"
    assert h_stat > 0.0
    assert n_h >= 1


def test_heidi_insufficient_snps():
    """Fewer than 1 usable nearby SNP returns p_heidi = 1."""
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [0.3], [0.05], [1e-6])
    eqtl = _make_sumstats(snps, [0.6], [0.05], [1e-10])

    # No nearby SNPs at all
    p_h, h_stat, n_h = heidi_test(gwas, eqtl, probe_snp="rs0",
                                   nearby_snps=[])
    assert p_h == 1.0
    assert n_h == 0


# ---------------------------------------------------------------------------
# Integration / summary tests
# ---------------------------------------------------------------------------

def test_smr_heidi_integration():
    """Full pipeline: smr_heidi with gene_map produces results for each gene."""
    snps = [f"rs{i}" for i in range(10)]
    # Gene A: strong eQTL at rs0, consistent ratios -> pass HEIDI
    betas_e = [0.5] + [0.01] * 9
    ps_e = [1e-12] + [0.9] * 9
    betas_g = [0.25] + [0.005] * 9
    ps_g = [1e-6] + [0.9] * 9
    ses = [0.05] * 10

    gwas = _make_sumstats(snps, betas_g, ses, ps_g)
    eqtl = _make_sumstats(snps, betas_e, ses, ps_e)

    gene_map = {"GENE_A": snps[:5], "GENE_B": snps[5:]}

    summary = smr_heidi(gwas, eqtl, gene_map, eqtl_p_threshold=5e-8,
                        smr_p_threshold=0.05)

    assert isinstance(summary, SMRSummary)
    assert summary.n_genes_tested == 2
    # GENE_A should have a significant SMR (strong eQTL at rs0)
    gene_a = [r for r in summary.results if r.gene_id == "GENE_A"][0]
    assert gene_a.p_smr < 0.05


def test_smr_summary_counts():
    """n_significant_smr and n_pass_heidi are counted correctly."""
    snps = [f"rs{i}" for i in range(8)]
    # Two genes, both with strong eQTLs
    betas_e = [0.6, 0.5, 0.01, 0.01, 0.55, 0.45, 0.01, 0.01]
    ps_e = [1e-12, 1e-10, 0.9, 0.9, 1e-11, 1e-9, 0.9, 0.9]
    # Gene1 (rs0-rs3): consistent ratio -> pass HEIDI
    # Gene2 (rs4-rs7): consistent ratio -> pass HEIDI
    ratio = 0.5
    betas_g = [b * ratio for b in betas_e]
    ses = [0.05] * 8
    ps_g = [1e-6, 1e-5, 0.9, 0.9, 1e-6, 1e-5, 0.9, 0.9]

    gwas = _make_sumstats(snps, betas_g, ses, ps_g)
    eqtl = _make_sumstats(snps, betas_e, ses, ps_e)

    gene_map = {"G1": snps[:4], "G2": snps[4:]}
    summary = smr_heidi(gwas, eqtl, gene_map, eqtl_p_threshold=5e-8,
                        smr_p_threshold=0.05)

    assert summary.n_genes_tested == 2
    assert summary.n_significant_smr >= 1
    # With consistent ratios, HEIDI p should be > 0.05
    assert summary.n_pass_heidi >= 1


def test_smr_result_fields():
    """All fields of SMRResult are populated with correct types."""
    snps = ["rs0"]
    gwas = _make_sumstats(snps, [0.3], [0.05], [1e-6])
    eqtl = _make_sumstats(snps, [0.6], [0.05], [1e-10])

    res = smr_test(gwas, eqtl, gene_id="GENE_X", probe_snp="rs0")

    assert isinstance(res, SMRResult)
    assert isinstance(res.gene_id, str)
    assert res.gene_id == "GENE_X"
    assert isinstance(res.probe_snp, str)
    assert res.probe_snp == "rs0"
    assert isinstance(res.beta_smr, float)
    assert isinstance(res.se_smr, float)
    assert isinstance(res.p_smr, float)
    assert isinstance(res.chi2_smr, float)
    assert isinstance(res.beta_gwas, float)
    assert isinstance(res.beta_eqtl, float)
    assert isinstance(res.p_heidi, float)
    assert isinstance(res.n_heidi_snps, int)
    assert isinstance(res.heidi_stat, float)
    assert 0.0 <= res.p_smr <= 1.0
    assert res.chi2_smr >= 0.0
    assert res.se_smr > 0.0
