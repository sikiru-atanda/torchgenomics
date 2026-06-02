"""Tests for ``torchgenomics.postgwas._finemapping`` — fine-mapping utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from torchgenomics.postgwas._finemapping import (
    annotate_sumstats,
    extract_credible_sets,
    locus_summary,
    to_coloc_sumstats,
)
from torchgenomics.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Mock BayesianVSResult (avoids importing the real class which needs NullFit)
# ---------------------------------------------------------------------------


@dataclass
class _MockBVSResult:
    """Mock BayesianVSResult for testing."""

    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    af: Tensor
    pip: Tensor
    beta_mean: Tensor
    beta_sd: Tensor
    credible_sets: list[list[int]] | None = None
    alpha: Tensor | None = None
    n_signals: int = 0
    method: str = "cavi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cavi_result(p: int = 10, *, seed: int = 0) -> _MockBVSResult:
    """Create a CAVI result with one clear signal at index 2."""
    torch.manual_seed(seed)
    pip = torch.full((p,), 0.01, dtype=torch.float64)
    # Put dominant mass on index 2 (guard for small p)
    pip[min(2, p - 1)] = 0.80
    if p > 3:
        pip[3] = 0.10
    if p > 4:
        pip[4] = 0.05
    # Renormalize the rest so they stay small but sum is reasonable
    return _MockBVSResult(
        snp=[f"rs{i}" for i in range(p)],
        chr=["1"] * p,
        pos=list(range(1000, 1000 + p * 100, 100)),
        a1=["A"] * p,
        a2=["G"] * p,
        af=torch.rand(p, dtype=torch.float64),
        pip=pip,
        beta_mean=torch.randn(p, dtype=torch.float64) * 0.1,
        beta_sd=torch.rand(p, dtype=torch.float64) * 0.05 + 0.01,
        method="cavi",
    )


def _make_susie_result(p: int = 10, n_layers: int = 3) -> _MockBVSResult:
    """Create a SuSiE result with 2 active layers and 1 inactive."""
    alpha = torch.full((n_layers, p), 1.0 / p, dtype=torch.float64)
    # Layer 0: strong signal at index 1, minimal mass elsewhere
    alpha[0] = torch.tensor(
        [0.01, 0.85, 0.05, 0.04, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005],
        dtype=torch.float64,
    )
    # Layer 1: strong signal at index 7, minimal mass on indices used by layer 0
    alpha[1] = torch.tensor(
        [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.02, 0.85, 0.06, 0.04],
        dtype=torch.float64,
    )
    # Layer 2: uniform (inactive)
    # already set above

    pip = 1.0 - torch.prod(1.0 - alpha, dim=0)

    return _MockBVSResult(
        snp=[f"rs{i}" for i in range(p)],
        chr=["1"] * p,
        pos=list(range(1000, 1000 + p * 100, 100)),
        a1=["A"] * p,
        a2=["G"] * p,
        af=torch.rand(p, dtype=torch.float64),
        pip=pip,
        beta_mean=torch.randn(p, dtype=torch.float64) * 0.1,
        beta_sd=torch.rand(p, dtype=torch.float64) * 0.05 + 0.01,
        alpha=alpha,
        n_signals=2,
        method="susie",
    )


def _make_sumstats(snp_ids: list[str], p_base: float = 1e-3) -> SumStats:
    """Build a minimal SumStats for *snp_ids*."""
    m = len(snp_ids)
    return SumStats(
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 100, 100)),
        snp=snp_ids,
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.randn(m, dtype=torch.float64) * 0.1,
        se=torch.rand(m, dtype=torch.float64) * 0.05 + 0.01,
        p=torch.full((m,), p_base, dtype=torch.float64),
        n=torch.full((m,), 1000.0, dtype=torch.float64),
        af=torch.rand(m, dtype=torch.float64) * 0.4 + 0.1,
    )


# ===================================================================
# Tests
# ===================================================================


class TestExtractCredibleSetsCavi:
    """Test credible set extraction from CAVI results."""

    def test_extract_credible_sets_cavi_single_set(self):
        """CAVI result with clear signal -> one credible set at 95% coverage."""
        res = _make_cavi_result()
        cs_list = extract_credible_sets(res, coverage=0.95)

        assert len(cs_list) == 1
        cs = cs_list[0]
        assert cs.signal_index == 0
        assert cs.cumulative_coverage >= 0.95
        # Lead SNP should be index 2 (highest PIP = 0.80)
        assert cs.lead_snp_index == 2
        assert cs.lead_snp_id == "rs2"
        assert cs.lead_pip == pytest.approx(0.80)
        # The set should be sorted by PIP descending
        assert cs.snp_indices[0] == 2


class TestExtractCredibleSetsSuSiE:
    """Test credible set extraction from SuSiE results."""

    def test_extract_credible_sets_susie_multiple_signals(self):
        """SuSiE with 2 active layers -> 2 credible sets."""
        res = _make_susie_result()
        cs_list = extract_credible_sets(res, coverage=0.95)

        assert len(cs_list) == 2
        # First set from layer 0: lead at index 1
        assert cs_list[0].lead_snp_index == 1
        assert cs_list[0].lead_snp_id == "rs1"
        # Second set from layer 1: lead at index 7
        assert cs_list[1].lead_snp_index == 7
        assert cs_list[1].lead_snp_id == "rs7"
        # Both should reach target coverage
        for cs in cs_list:
            assert cs.cumulative_coverage >= 0.95


class TestExtractCredibleSetsCoverage:
    """Test coverage parameter."""

    def test_extract_credible_sets_coverage(self):
        """With coverage=0.5, sets should be smaller."""
        res = _make_cavi_result()
        cs_full = extract_credible_sets(res, coverage=0.95)
        cs_half = extract_credible_sets(res, coverage=0.50)

        assert len(cs_half) == 1
        # The 0.50 set should be smaller (fewer SNPs needed)
        assert len(cs_half[0].snp_indices) <= len(cs_full[0].snp_indices)
        assert cs_half[0].cumulative_coverage >= 0.50
        # With PIP 0.80 at the lead, a single SNP should reach 0.50
        assert len(cs_half[0].snp_indices) == 1


class TestExtractCredibleSetsMinPip:
    """Test min_pip filtering."""

    def test_extract_credible_sets_min_pip_filters(self):
        """min_pip=0.01 excludes low-PIP variants from sets."""
        res = _make_cavi_result()
        cs_no_filter = extract_credible_sets(res, coverage=0.95, min_pip=0.0)
        cs_filtered = extract_credible_sets(res, coverage=0.95, min_pip=0.02)

        # Filtered set should not contain any variant with PIP < 0.02
        for pip_val in cs_filtered[0].pip.tolist():
            assert pip_val >= 0.02

        # Filtered set should have fewer or equal variants
        assert len(cs_filtered[0].snp_indices) <= len(cs_no_filter[0].snp_indices)


class TestExtractCredibleSetsDeduplicate:
    """Test deduplication across sets."""

    def test_extract_credible_sets_deduplicate(self):
        """With deduplicate=True, no variant appears in two sets."""
        res = _make_susie_result()
        cs_dedup = extract_credible_sets(res, coverage=0.95, deduplicate=True)

        all_indices: list[int] = []
        for cs in cs_dedup:
            all_indices.extend(cs.snp_indices)

        # No duplicates
        assert len(all_indices) == len(set(all_indices))

    def test_extract_credible_sets_no_deduplicate(self):
        """With deduplicate=False, variants can appear in multiple sets."""
        res = _make_susie_result()
        cs_no_dedup = extract_credible_sets(
            res, coverage=0.95, deduplicate=False
        )
        # Both sets should still exist
        assert len(cs_no_dedup) == 2


class TestAnnotateSumStats:
    """Test PIP annotation of summary statistics."""

    def test_annotate_sumstats_merges_correctly(self):
        """PIPs and credible set membership are merged onto SumStats by SNP ID."""
        fm = _make_cavi_result(p=10)
        ss = _make_sumstats([f"rs{i}" for i in range(10)])

        ann = annotate_sumstats(ss, fm)

        assert len(ann.snp) == 10
        # PIP at index 2 should match
        assert ann.pip[2].item() == pytest.approx(0.80)
        # Credible set membership: index 2 should be in set 0
        assert ann.in_credible_set[2] == 0

    def test_annotate_sumstats_missing_snps(self):
        """SNPs in SumStats but not in fine-mapping get pip=0, in_credible_set=-1."""
        fm = _make_cavi_result(p=5)  # rs0..rs4
        # SumStats has rs0..rs9 (5 extra)
        ss = _make_sumstats([f"rs{i}" for i in range(10)])

        ann = annotate_sumstats(ss, fm)

        assert len(ann.snp) == 10
        # rs5..rs9 should have pip=0 and in_credible_set=-1
        for i in range(5, 10):
            assert ann.pip[i].item() == pytest.approx(0.0)
            assert ann.in_credible_set[i] == -1

    def test_annotate_sumstats_as_table(self):
        """as_table() returns dict with correct keys and lengths."""
        fm = _make_cavi_result(p=5)
        ss = _make_sumstats([f"rs{i}" for i in range(5)])

        ann = annotate_sumstats(ss, fm)
        tbl = ann.as_table()

        expected_keys = {
            "chr", "pos", "snp", "a1", "a2",
            "beta", "se", "p", "pip",
            "in_credible_set", "beta_posterior", "beta_posterior_sd",
        }
        assert set(tbl.keys()) == expected_keys
        for key in expected_keys:
            assert len(tbl[key]) == 5


class TestLocusSummary:
    """Test per-locus summary."""

    def test_locus_summary_multi_locus(self):
        """Two loci -> LocusSummary with correct n_variants and n_signals."""
        locus1 = _make_cavi_result(p=10)
        locus2 = _make_susie_result(p=10, n_layers=3)

        summary = locus_summary([locus1, locus2])

        assert len(summary.locus_chr) == 2
        assert summary.n_variants == [10, 10]
        # Locus 1 (CAVI): 1 signal; Locus 2 (SuSiE): 2 signals
        assert summary.n_signals[0] == 1
        assert summary.n_signals[1] == 2
        # Lead SNPs
        assert summary.lead_snp[0] == "rs2"  # highest PIP in locus 1
        assert summary.lead_snp[1] == "rs1"  # highest PIP in locus 2
        # Total PIP should be positive
        for tp in summary.total_pip:
            assert tp > 0.0

    def test_locus_summary_empty_raises(self):
        """Empty results list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            locus_summary([])


class TestToColocSumstats:
    """Test colocalization-ready export."""

    def test_to_coloc_sumstats_aligns(self):
        """Output SumStats has same variant order as BVSResult."""
        fm = _make_cavi_result(p=5)
        ss = _make_sumstats([f"rs{i}" for i in range(5)])

        coloc_ss = to_coloc_sumstats(fm, ss)

        assert coloc_ss.snp == fm.snp
        assert len(coloc_ss.snp) == 5

    def test_to_coloc_sumstats_subset(self):
        """Only variants present in both fm and ss are kept."""
        fm = _make_cavi_result(p=5)  # rs0..rs4
        # SumStats has rs2..rs7 (overlap: rs2, rs3, rs4)
        ss = _make_sumstats([f"rs{i}" for i in range(2, 8)])

        coloc_ss = to_coloc_sumstats(fm, ss)

        assert len(coloc_ss.snp) == 3
        assert coloc_ss.snp == ["rs2", "rs3", "rs4"]

    def test_to_coloc_sumstats_no_overlap_raises(self):
        """No overlapping variants raises ValueError."""
        fm = _make_cavi_result(p=3)  # rs0..rs2
        ss = _make_sumstats(["rsX", "rsY", "rsZ"])

        with pytest.raises(ValueError, match="No overlapping"):
            to_coloc_sumstats(fm, ss)


class TestCredibleSetLeadSnp:
    """Test lead SNP identification."""

    def test_credible_set_lead_snp(self):
        """lead_snp_id corresponds to highest PIP variant."""
        res = _make_cavi_result(p=10)
        cs_list = extract_credible_sets(res)

        cs = cs_list[0]
        # The lead should be the variant with the max PIP
        max_pip_idx = res.pip.argmax().item()
        assert cs.lead_snp_index == max_pip_idx
        assert cs.lead_snp_id == res.snp[max_pip_idx]
        assert cs.lead_pip == pytest.approx(res.pip[max_pip_idx].item())
