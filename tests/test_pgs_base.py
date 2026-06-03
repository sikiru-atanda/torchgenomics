"""Tests for torchgenomics.pgs.base — LDReference, PGSResult, BasePGSMethod."""

from __future__ import annotations

import pytest
import torch

from torchgenomics.pgs.base import (
    BasePGSMethod,
    LDReference,
    PGSResult,
    _alleles_match,
    _is_palindromic,
)
from torchgenomics.postgwas._sumstats import SumStats


def _make_ld_full(m: int = 6, seed: int = 0) -> LDReference:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, m, generator=g, dtype=torch.float64)
    R = A @ A.T / m + 0.1 * torch.eye(m, dtype=torch.float64)
    # normalize to correlation
    d = torch.sqrt(torch.diag(R))
    R = R / d[:, None] / d[None, :]
    return LDReference(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.3, dtype=torch.float64),
        mode="full",
        R_full=R,
        n_ref=500,
    )


def _make_ld_block(blocks: list[int] = (3, 3), seed: int = 0) -> LDReference:
    g = torch.Generator().manual_seed(seed)
    R_blocks = []
    for b in blocks:
        A = torch.randn(b, b, generator=g, dtype=torch.float64)
        R = A @ A.T / b + 0.1 * torch.eye(b, dtype=torch.float64)
        d = torch.sqrt(torch.diag(R))
        R = R / d[:, None] / d[None, :]
        R_blocks.append(R)
    m = sum(blocks)
    block_index = torch.zeros(m, dtype=torch.long)
    j = 0
    for k, b in enumerate(blocks):
        block_index[j : j + b] = k
        j += b
    return LDReference(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.full((m,), 0.25, dtype=torch.float64),
        mode="block",
        R_blocks=R_blocks,
        block_index=block_index,
        n_ref=500,
    )


# ---------------------------------------------------------------------------


def test_ld_reference_full_solve_matches_torch():
    ld = _make_ld_full(m=5, seed=1)
    v = torch.arange(5, dtype=torch.float64)
    out = ld.solve(v, ridge=0.01)
    expected = torch.linalg.solve(
        ld.R_full + 0.01 * torch.eye(5, dtype=torch.float64), v
    )
    assert torch.allclose(out, expected, atol=1e-12)


def test_ld_reference_block_matches_block_diagonal_full():
    ld_block = _make_ld_block(blocks=[3, 4], seed=2)
    # build the equivalent block-diagonal full matrix
    m = ld_block.m
    R_full = torch.zeros(m, m, dtype=torch.float64)
    j = 0
    for R_b in ld_block.R_blocks:
        b = R_b.shape[0]
        R_full[j : j + b, j : j + b] = R_b
        j += b
    v = torch.linspace(-1, 1, m, dtype=torch.float64)
    block_solve = ld_block.solve(v, ridge=0.05)
    full_solve = torch.linalg.solve(
        R_full + 0.05 * torch.eye(m, dtype=torch.float64), v
    )
    assert torch.allclose(block_solve, full_solve, atol=1e-12)


def test_ld_reference_to_device_roundtrip():
    ld = _make_ld_block(blocks=[2, 3])
    ld2 = ld.to("cpu")
    assert ld2.m == ld.m
    assert ld2.mode == "block"
    assert ld2.R_blocks is not None
    assert all(b.device.type == "cpu" for b in ld2.R_blocks)
    assert ld2.block_index.device.type == "cpu"


def test_pgs_result_save_load_roundtrip(tmp_path):
    m = 4
    res = PGSResult(
        method="ldpred2-inf",
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=[10, 20, 30, 40],
        a1=["A", "C", "G", "T"],
        a2=["G", "T", "A", "C"],
        weight=torch.tensor([0.1, -0.2, 0.05, 0.0], dtype=torch.float64),
        weight_sd=torch.tensor([0.01, 0.02, 0.01, 0.03], dtype=torch.float64),
        af=torch.tensor([0.3, 0.5, 0.2, 0.4], dtype=torch.float64),
        h2=0.25,
        n_iter=100,
        n_burnin=50,
        converged=True,
        n_input=10,
        n_matched=4,
        n_flipped=1,
    )
    path = tmp_path / "weights.tsv"
    res.save(str(path))
    loaded = PGSResult.load(str(path))
    assert loaded.method == "ldpred2-inf"
    assert loaded.snp == res.snp
    assert loaded.h2 == 0.25
    assert loaded.n_flipped == 1
    assert loaded.converged is True
    assert torch.allclose(loaded.weight, res.weight)
    assert torch.allclose(loaded.weight_sd, res.weight_sd)


def test_palindromic_and_alleles_match_helpers():
    assert _is_palindromic("A", "T")
    assert _is_palindromic("C", "G")
    assert not _is_palindromic("A", "G")

    # exact match
    assert _alleles_match("A", "G", "A", "G") == (True, +1)
    # swapped
    assert _alleles_match("G", "A", "A", "G") == (True, -1)
    # strand-complement match
    assert _alleles_match("T", "C", "A", "G") == (True, +1)
    # strand-complement swapped
    assert _alleles_match("C", "T", "A", "G") == (True, -1)
    # mismatch
    assert _alleles_match("A", "C", "A", "G") == (False, 0)


class _DummyPGS(BasePGSMethod):
    name = "dummy"

    def fit(self, sumstats, ld_ref, **kwargs):
        ss, ref, audit = self._harmonize(sumstats, ld_ref)
        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=ss.beta.clone(),
            af=ss.af,
            **audit,
        )


def test_harmonize_flips_swapped_alleles_and_subsets_ld():
    # LD reference: rs0..rs3 with effect allele A, other G
    ld = _make_ld_block(blocks=[2, 2], seed=3)

    # SumStats has rs0 (matched), rs1 (swapped -> flip), rs99 (not in ref)
    ss = SumStats(
        chr=["1", "1", "1"],
        pos=[100, 101, 999],
        snp=["rs0", "rs1", "rs99"],
        a1=["A", "G", "A"],  # rs1 swapped
        a2=["G", "A", "T"],
        beta=torch.tensor([0.5, 0.2, 0.1], dtype=torch.float64),
        se=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
        p=torch.tensor([1e-3, 1e-2, 0.5], dtype=torch.float64),
        n=torch.tensor([1000.0, 1000.0, 1000.0]),
        af=torch.tensor([0.3, 0.7, 0.2], dtype=torch.float64),
    )
    method = _DummyPGS()
    res = method.fit(ss, ld)

    assert res.n_input == 3
    assert res.n_matched == 2
    assert res.n_flipped == 1
    # rs0 unchanged
    assert res.snp[0] == "rs0"
    assert float(res.weight[0]) == pytest.approx(0.5)
    # rs1 sign-flipped
    assert res.snp[1] == "rs1"
    assert float(res.weight[1]) == pytest.approx(-0.2)
    # AF for rs1 was 0.7 in sumstats; after flip should be 0.3
    assert float(res.af[1]) == pytest.approx(0.3)


def test_harmonize_drops_palindromic_when_maf_ambiguous():
    # All ref SNPs are A/T -> palindromic
    m = 3
    ld = LDReference(
        snp=["rsA", "rsB", "rsC"],
        chr=["1"] * m,
        pos=[10, 20, 30],
        a1=["A", "A", "A"],
        a2=["T", "T", "T"],
        af=torch.tensor([0.49, 0.10, 0.50], dtype=torch.float64),
        mode="full",
        R_full=torch.eye(m, dtype=torch.float64),
        n_ref=100,
    )
    ss = SumStats(
        chr=["1"] * m,
        pos=[10, 20, 30],
        snp=["rsA", "rsB", "rsC"],
        a1=["A", "A", "A"],
        a2=["T", "T", "T"],
        beta=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64),
        se=torch.tensor([0.05, 0.05, 0.05], dtype=torch.float64),
        p=torch.tensor([1e-3, 1e-3, 1e-3], dtype=torch.float64),
        n=torch.tensor([1000.0] * m),
        af=torch.tensor([0.49, 0.10, 0.50], dtype=torch.float64),
    )
    method = _DummyPGS()
    res = method.fit(ss, ld)
    # rsA (MAF 0.49) and rsC (MAF 0.50) are too close to 0.5 -> dropped.
    # rsB (MAF 0.10) is far from 0.5 -> retained.
    assert res.n_input == 3
    assert res.n_matched == 1
    assert res.snp == ["rsB"]
    assert res.n_ambiguous_removed == 2
