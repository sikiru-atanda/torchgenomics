"""Tests for torchgenomics.pgs.sumstats_io."""

from __future__ import annotations

import math

import pytest
import torch

from torchgenomics.pgs.base import LDReference
from torchgenomics.pgs.sumstats_io import (
    _Harmonizer,
    _resolve,
    harmonize_to_reference,
    load_pgs_sumstats,
)
from torchgenomics.postgwas._sumstats import SumStats


def _write_tsv(path, header, rows, sep="\t"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(sep.join(header) + "\n")
        for r in rows:
            f.write(sep.join(str(x) for x in r) + "\n")


def test_resolve_aliases_case_insensitive():
    cols = ["Chrom", "BP", "SNP", "Effect_Allele", "Other_Allele",
            "BETA", "SE", "PVAL", "EAF", "N_eff"]
    assert _resolve(cols, ("chr", "chrom")) == "Chrom"
    assert _resolve(cols, ("pos", "bp")) == "BP"
    assert _resolve(cols, ("snp", "rsid")) == "SNP"
    assert _resolve(cols, ("a1", "effect_allele")) == "Effect_Allele"
    assert _resolve(cols, ("af", "eaf")) == "EAF"
    assert _resolve(cols, ("n", "n_eff")) == "N_eff"
    assert _resolve(cols, ("nonexistent",)) is None


def test_load_pgs_sumstats_with_beta_column(tmp_path):
    p = tmp_path / "ss.tsv"
    _write_tsv(
        p,
        ["chrom", "bp", "rsid", "effect_allele", "other_allele",
         "beta", "se", "pval", "eaf", "n"],
        [
            ["1", "100", "rs1", "a", "g", "0.10", "0.02", "1e-5", "0.30", "5000"],
            ["1", "200", "rs2", "C", "T", "-0.05", "0.01", "1e-3", "0.25", "5000"],
        ],
    )
    ss = load_pgs_sumstats(str(p))
    assert ss.m == 2
    assert ss.snp == ["rs1", "rs2"]
    # alleles uppercased
    assert ss.a1 == ["A", "C"]
    assert ss.a2 == ["G", "T"]
    assert torch.allclose(ss.beta, torch.tensor([0.10, -0.05], dtype=torch.float64))
    assert ss.af is not None
    assert torch.allclose(ss.af, torch.tensor([0.30, 0.25], dtype=torch.float64))


def test_load_pgs_sumstats_or_to_log(tmp_path):
    p = tmp_path / "ss_or.tsv"
    _write_tsv(
        p,
        ["chr", "pos", "snp", "a1", "a2", "or", "se", "p"],
        [
            ["1", "100", "rs1", "A", "G", "1.20", "0.05", "1e-3"],
            ["1", "200", "rs2", "C", "T", "0.80", "0.04", "5e-2"],
        ],
    )
    ss = load_pgs_sumstats(str(p))
    assert ss.m == 2
    assert float(ss.beta[0]) == pytest.approx(math.log(1.20))
    assert float(ss.beta[1]) == pytest.approx(math.log(0.80))


def test_load_pgs_sumstats_neff_doubled(tmp_path):
    p = tmp_path / "ss_neff.tsv"
    _write_tsv(
        p,
        ["chr", "pos", "snp", "a1", "a2", "beta", "se", "p", "n"],
        [["1", "100", "rs1", "A", "G", "0.1", "0.01", "1e-3", "10000"]],
    )
    ss = load_pgs_sumstats(str(p), n_eff_doubled=True)
    assert float(ss.n[0]) == pytest.approx(5000.0)


def test_load_pgs_sumstats_missing_required_columns_raises(tmp_path):
    p = tmp_path / "bad.tsv"
    # missing SE
    _write_tsv(
        p,
        ["chr", "pos", "snp", "a1", "a2", "beta", "p"],
        [["1", "100", "rs1", "A", "G", "0.1", "1e-3"]],
    )
    with pytest.raises(ValueError, match="se"):
        load_pgs_sumstats(str(p))


def test_load_pgs_sumstats_no_beta_no_or_raises(tmp_path):
    p = tmp_path / "no_effect.tsv"
    _write_tsv(
        p,
        ["chr", "pos", "snp", "a1", "a2", "se", "p"],
        [["1", "100", "rs1", "A", "G", "0.01", "1e-3"]],
    )
    with pytest.raises(ValueError, match="BETA or OR"):
        load_pgs_sumstats(str(p))


def test_harmonize_to_reference_functional_wrapper():
    m = 3
    ld = LDReference(
        snp=["rs1", "rs2", "rs3"],
        chr=["1", "1", "1"],
        pos=[100, 200, 300],
        a1=["A", "A", "A"],
        a2=["G", "G", "G"],
        af=torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64),
        mode="full",
        R_full=torch.eye(m, dtype=torch.float64),
        n_ref=500,
    )
    ss = SumStats(
        chr=["1", "1"],
        pos=[100, 200],
        snp=["rs1", "rs2"],
        a1=["A", "G"],  # rs2 swapped
        a2=["G", "A"],
        beta=torch.tensor([0.5, 0.3], dtype=torch.float64),
        se=torch.tensor([0.05, 0.04], dtype=torch.float64),
        p=torch.tensor([1e-4, 1e-3], dtype=torch.float64),
        n=torch.tensor([1000.0, 1000.0]),
        af=torch.tensor([0.3, 0.6], dtype=torch.float64),
    )
    new_ss, new_ld, audit = harmonize_to_reference(ss, ld)
    assert audit["n_input"] == 2
    assert audit["n_matched"] == 2
    assert audit["n_flipped"] == 1
    assert new_ss.snp == ["rs1", "rs2"]
    # rs2 effect flipped: 0.3 -> -0.3
    assert float(new_ss.beta[1]) == pytest.approx(-0.3)
    # AF flipped: 0.6 -> 0.4
    assert float(new_ss.af[1]) == pytest.approx(0.4)
    # LD reference unchanged length
    assert new_ld.m == 2


def test_harmonizer_fit_is_misuse_guard():
    with pytest.raises(TypeError, match="harmonize_to_reference"):
        _Harmonizer().fit(None, None)
