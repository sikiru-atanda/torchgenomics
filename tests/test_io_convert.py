"""Tests for genotype format writers."""

from __future__ import annotations

import gzip

import torch

from torchgwas.io.convert import _write_vcf
from torchgwas.models.base import VariantMeta


class _TinyReader:
    @property
    def n_samples(self):
        return 3

    @property
    def n_variants(self):
        return 2

    @property
    def sample_ids(self):
        return ["s1", "s2", "s3"]

    def iter_chunks(self, chunk_size=1024):
        del chunk_size
        G = torch.tensor(
            [
                [0.0, 1.4],
                [1.0, 2.0],
                [float("nan"), 0.0],
            ],
            dtype=torch.float64,
        )
        meta = VariantMeta(
            snp=["rs1", "rs2"],
            chr=["1", "1"],
            pos=[101, 202],
            a1=["C", "G"],
            a2=["A", "T"],
        )
        yield G, meta


def test_write_vcf_exports_gt_and_ds(tmp_path):
    out = tmp_path / "tiny.vcf"

    _write_vcf(_TinyReader(), str(out))

    text = out.read_text()
    assert "##FORMAT=<ID=GT" in text
    assert "##FORMAT=<ID=DS" in text
    assert "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3" in text
    assert "1\t101\trs1\tA\tC\t.\tPASS\t.\tGT:DS\t0/0:0\t0/1:1\t./.:." in text
    assert "1\t202\trs2\tT\tG\t.\tPASS\t.\tGT:DS\t0/1:1.4\t1/1:2\t0/0:0" in text


def test_write_vcf_supports_gzip_output(tmp_path):
    out = tmp_path / "tiny.vcf.gz"

    _write_vcf(_TinyReader(), str(out))

    with gzip.open(out, "rt") as fh:
        assert fh.readline().strip() == "##fileformat=VCFv4.3"
