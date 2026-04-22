"""Benchmark: torchgwas fine-mapping utilities vs manual numpy calculations.

Validates credible set extraction, PIP annotation, and locus summary
against ground-truth manual calculations using numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from torchgwas.postgwas._finemapping import (
    annotate_sumstats,
    extract_credible_sets,
    locus_summary,
    to_coloc_sumstats,
)
from torchgwas.postgwas._sumstats import SumStats

# ---------------------------------------------------------------------------
# Mock BayesianVSResult (mirrors the real dataclass without importing the
# full models package, which pulls in heavy dependencies)
# ---------------------------------------------------------------------------


@dataclass
class _MockBVS:
    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    af: torch.Tensor
    pip: torch.Tensor
    beta_mean: torch.Tensor
    beta_sd: torch.Tensor
    credible_sets: list[list[int]] | None = None
    alpha: torch.Tensor | None = None
    n_signals: int = 0
    method: str = "cavi"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snp_ids(n: int) -> list[str]:
    return [f"rs{i + 1}" for i in range(n)]


def _make_mock_bvs(
    pips: list[float],
    *,
    method: str = "cavi",
    alpha: torch.Tensor | None = None,
) -> _MockBVS:
    """Build a minimal _MockBVS with the given PIPs."""
    n = len(pips)
    snps = _make_snp_ids(n)
    return _MockBVS(
        snp=snps,
        chr=["1"] * n,
        pos=list(range(100, 100 + n * 10, 10)),
        a1=["A"] * n,
        a2=["G"] * n,
        af=torch.full((n,), 0.3),
        pip=torch.tensor(pips, dtype=torch.float64),
        beta_mean=torch.randn(n, dtype=torch.float64) * 0.1,
        beta_sd=torch.ones(n, dtype=torch.float64) * 0.05,
        method=method,
        alpha=alpha,
    )


def _make_sumstats(snp_ids: list[str], n: int) -> SumStats:
    """Build a minimal SumStats for *n* variants with the given IDs."""
    m = len(snp_ids)
    return SumStats(
        chr=["1"] * m,
        pos=list(range(100, 100 + m * 10, 10)),
        snp=snp_ids,
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.randn(m, dtype=torch.float64) * 0.2,
        se=torch.ones(m, dtype=torch.float64) * 0.05,
        p=torch.rand(m, dtype=torch.float64) * 0.05,
        n=torch.full((m,), float(n), dtype=torch.float64),
        af=torch.full((m,), 0.3, dtype=torch.float64),
    )


# ===================================================================
# Tests
# ===================================================================


class TestBenchCredibleSetCoverageMatchesNumpyCumsum:
    """test_bench_credible_set_coverage_matches_numpy_cumsum"""

    def test_coverage_matches_numpy(self):
        pips = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
        bvs = _make_mock_bvs(pips)

        # --- manual numpy reference ---
        arr = np.array(pips)
        order = np.argsort(-arr)  # descending
        sorted_pips = arr[order]
        cumsum = np.cumsum(sorted_pips)
        # First index where cumsum >= 0.95
        n_needed = int(np.searchsorted(cumsum, 0.95, side="left")) + 1
        expected_indices = order[:n_needed].tolist()
        expected_coverage = float(cumsum[n_needed - 1])

        # --- torchgwas ---
        cs_list = extract_credible_sets(bvs, coverage=0.95)
        assert len(cs_list) == 1
        cs = cs_list[0]

        # Same variant count
        assert len(cs.snp_indices) == n_needed
        # Same indices (order = descending PIP)
        assert cs.snp_indices == expected_indices
        # Coverage matches to machine precision
        assert abs(cs.cumulative_coverage - expected_coverage) < 1e-12


class TestBenchSuSiECredibleSetPerLayerMatchesNumpy:
    """test_bench_susie_credible_set_per_layer_matches_numpy"""

    def test_per_layer(self):
        p = 8
        # Layer 0: signal concentrated on variant 2
        alpha_0 = np.array([0.01, 0.02, 0.70, 0.10, 0.08, 0.04, 0.03, 0.02])
        # Layer 1: signal concentrated on variant 5 -- no overlap with layer 0
        # Use values that make the two layers' credible sets disjoint so dedup
        # does not interfere.  All values are distinct to avoid tie-breaking
        # differences between numpy argsort and torch sort.
        alpha_1 = np.array([0.015, 0.012, 0.008, 0.013, 0.06, 0.70, 0.092, 0.10])
        alpha = torch.tensor(
            np.stack([alpha_0, alpha_1]), dtype=torch.float64
        )  # (2, 8)

        bvs = _make_mock_bvs(
            pips=[0.0] * p,  # pip unused for SuSiE path
            method="susie",
            alpha=alpha,
        )

        # --- manual numpy reference for each layer, with dedup ---
        used: set[int] = set()
        expected_sets: list[list[int]] = []
        expected_coverages: list[float] = []
        for layer in [alpha_0, alpha_1]:
            order = np.argsort(-layer)
            indices: list[int] = []
            cum = 0.0
            for idx in order:
                if idx in used:
                    continue
                indices.append(int(idx))
                cum += layer[idx]
                if cum >= 0.95:
                    break
            expected_sets.append(indices)
            expected_coverages.append(cum)
            used.update(indices)

        # --- torchgwas ---
        cs_list = extract_credible_sets(bvs, coverage=0.95)
        assert len(cs_list) == 2

        for i, cs in enumerate(cs_list):
            assert cs.snp_indices == expected_sets[i]
            assert abs(cs.cumulative_coverage - expected_coverages[i]) < 1e-12


class TestBenchCredibleSetDedupCorrectness:
    """test_bench_credible_set_dedup_correctness"""

    def test_dedup(self):
        p = 6
        # Layer 0 concentrates on variant 1 but also includes variant 3
        alpha_0 = np.array([0.05, 0.60, 0.05, 0.20, 0.05, 0.05])
        # Layer 1 concentrates on variant 3 -- but variant 3 already used
        alpha_1 = np.array([0.05, 0.15, 0.05, 0.55, 0.10, 0.10])
        alpha = torch.tensor(
            np.stack([alpha_0, alpha_1]), dtype=torch.float64
        )

        bvs = _make_mock_bvs(
            pips=[0.0] * p,
            method="susie",
            alpha=alpha,
        )

        cs_list = extract_credible_sets(bvs, coverage=0.95, deduplicate=True)

        # Collect all indices across sets
        all_indices: list[int] = []
        for cs in cs_list:
            all_indices.extend(cs.snp_indices)

        # No duplicates
        assert len(all_indices) == len(set(all_indices))

        # Layer 0 should claim variant 3 (index 3).
        # Layer 1 must skip it and pull in other variants instead.
        layer0_indices = set(cs_list[0].snp_indices)
        if len(cs_list) > 1:
            layer1_indices = set(cs_list[1].snp_indices)
            assert layer0_indices.isdisjoint(layer1_indices)


class TestBenchAnnotateSnpMatchingIsExact:
    """test_bench_annotate_snp_matching_is_exact"""

    def test_matching(self):
        # BVS has 5 variants rs1..rs5
        bvs_pips = [0.50, 0.30, 0.10, 0.07, 0.03]
        bvs = _make_mock_bvs(bvs_pips)

        # SumStats has 8 variants, overlapping on rs2, rs4, rs5 (indices 1,3,4
        # in the BVS)
        ss_ids = ["rsA", "rs2", "rsB", "rs4", "rs5", "rsC", "rsD", "rsE"]
        ss = _make_sumstats(ss_ids, n=1000)

        ann = annotate_sumstats(ss, bvs)

        # Manual matching: ss index -> bvs index
        match_map = {1: 1, 3: 3, 4: 4}  # ss_i -> bvs_i
        bvs_pip = np.array(bvs_pips)

        for ss_i in range(len(ss_ids)):
            if ss_i in match_map:
                expected_pip = bvs_pip[match_map[ss_i]]
                assert abs(ann.pip[ss_i].item() - expected_pip) < 1e-12
            else:
                assert ann.pip[ss_i].item() == 0.0


class TestBenchLocusSummaryAggregation:
    """test_bench_locus_summary_aggregation"""

    def test_aggregation(self):
        # 3 loci with known properties
        loci = [
            [0.60, 0.25, 0.10, 0.05],  # lead = rs1, total = 1.0
            [0.10, 0.80, 0.05, 0.05],  # lead = rs2, total = 1.0
            [0.30, 0.30, 0.20, 0.20],  # lead = rs1 (first argmax), total=1.0
        ]
        results = []
        for pips in loci:
            results.append(_make_mock_bvs(pips))

        summary = locus_summary(results, coverage=0.95)

        # Manual expected values
        for i, pips in enumerate(loci):
            arr = np.array(pips)
            expected_lead_idx = int(np.argmax(arr))
            expected_lead_pip = float(arr[expected_lead_idx])
            expected_total = float(arr.sum())

            assert summary.lead_snp[i] == f"rs{expected_lead_idx + 1}"
            assert abs(summary.lead_pip[i] - expected_lead_pip) < 1e-12
            assert abs(summary.total_pip[i] - expected_total) < 1e-12
            assert summary.n_variants[i] == len(pips)

        # Number of signals: each CAVI result produces at most 1 credible set
        for n_sig in summary.n_signals:
            assert n_sig == 1


class TestBenchCredibleSetMinPipFilter:
    """test_bench_credible_set_min_pip_filter"""

    def test_min_pip(self):
        pips = [0.40, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01]
        bvs = _make_mock_bvs(pips)

        min_pip = 0.05
        cs_list = extract_credible_sets(bvs, coverage=0.95, min_pip=min_pip)
        assert len(cs_list) == 1
        cs = cs_list[0]

        # Manual: sorted descending is same order; filter pips < 0.05
        arr = np.array(pips)
        order = np.argsort(-arr)
        for idx in cs.snp_indices:
            assert arr[idx] >= min_pip

        # Every variant in the set should have pip >= min_pip
        for p_val in cs.pip.tolist():
            assert p_val >= min_pip


class TestBenchToColocPreservesEffectSizes:
    """test_bench_to_coloc_preserves_effect_sizes"""

    def test_effect_preservation(self):
        p = 6
        bvs = _make_mock_bvs([0.3, 0.3, 0.2, 0.1, 0.05, 0.05])

        # SumStats with 10 variants, 4 overlap with bvs (rs1, rs3, rs5, rs6)
        ss_ids = [
            "rsX", "rs1", "rsY", "rs3", "rsZ", "rs5", "rs6", "rsW", "rsV",
            "rsU",
        ]
        ss = _make_sumstats(ss_ids, n=500)

        coloc_ss = to_coloc_sumstats(bvs, ss)

        # Build expected alignment manually
        ss_lookup = {sid: i for i, sid in enumerate(ss_ids)}
        bvs_ids = _make_snp_ids(p)

        expected_fm_keep = []
        expected_ss_keep = []
        for fi, sid in enumerate(bvs_ids):
            si = ss_lookup.get(sid)
            if si is not None:
                expected_fm_keep.append(fi)
                expected_ss_keep.append(si)

        # The output betas should equal the ss betas at the aligned positions
        for out_i, ss_i in enumerate(expected_ss_keep):
            assert abs(
                coloc_ss.beta[out_i].item() - ss.beta[ss_i].item()
            ) < 1e-12

        # SNP IDs should follow fm order
        for out_i, fm_i in enumerate(expected_fm_keep):
            assert coloc_ss.snp[out_i] == bvs_ids[fm_i]


class TestBenchCredibleSetSingleVariantDominates:
    """test_bench_credible_set_single_variant_dominates"""

    def test_single_dominant(self):
        pips = [0.98, 0.005, 0.005, 0.005, 0.005]
        bvs = _make_mock_bvs(pips)

        cs_list = extract_credible_sets(bvs, coverage=0.95)
        assert len(cs_list) == 1
        cs = cs_list[0]

        # Manual: only 1 variant needed since 0.98 >= 0.95
        arr = np.array(pips)
        order = np.argsort(-arr)
        cumsum = np.cumsum(arr[order])
        n_needed = int(np.searchsorted(cumsum, 0.95, side="left")) + 1
        assert n_needed == 1

        assert len(cs.snp_indices) == 1
        assert cs.snp_indices[0] == 0  # variant 0 has pip 0.98
        assert cs.lead_snp_id == "rs1"
        assert abs(cs.lead_pip - 0.98) < 1e-12
        assert abs(cs.cumulative_coverage - 0.98) < 1e-12
