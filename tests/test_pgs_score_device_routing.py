"""Regression test: api.pgs_fit/api.pgs_score device routing.

Surface-audit bug — `torchgenomics/pgs/base.py::_ld_reference_subset`
was indexing the CPU tensor ``idx`` (built in ``_harmonize``) with
``new_in_block`` (constructed on the LD reference's device), which
raised:

    RuntimeError: indices should be either on cpu or on the same
                   device as the indexed tensor (cpu)

The bug only manifests when:
  - the LD reference is in **block** mode, and
  - the LD reference is loaded onto a non-CPU device (i.e. cuda via
    ``device="auto"`` resolving to cuda on a GPU host).

The CLI subcommand ``pgs-score`` is unaffected because
``_run_pgs_score`` calls ``score_individuals`` directly and never
exercises ``_harmonize`` / ``_ld_reference_subset``. The api path
through ``api.pgs_fit`` (and any other PGS-fit method) is the
load-bearing path and is what these tests exercise.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from torchgenomics.pgs.base import PGSResult
from torchgenomics.pgs.ld_ref import build_ld_reference, save_ld_reference


def _write_sumstats(path: Path, m: int = 12, n_obs: int = 2000, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    betas = 0.1 * torch.randn(m, generator=g, dtype=torch.float64)
    ses = torch.full((m,), 1.0 / math.sqrt(n_obs), dtype=torch.float64)
    zs = betas / ses
    ps = 2.0 * (1.0 - 0.5 * (1.0 + torch.erf(zs.abs() / math.sqrt(2.0))))
    header = ["chr", "pos", "snp", "a1", "a2", "beta", "se", "p", "n"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for i in range(m):
            f.write(
                f"1\t{100 + i}\trs{i}\tA\tG\t{float(betas[i]):.6f}\t"
                f"{float(ses[i]):.6f}\t{float(ps[i]):.6e}\t{n_obs}\n"
            )


def _write_block_ld_ref(path: Path, m: int = 12, n: int = 200, seed: int = 1,
                        n_blocks: int = 3) -> None:
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(n, m, generator=g, dtype=torch.float64)
    G = (G - G.mean(0, keepdim=True)) / G.std(0, keepdim=True).clamp(min=1e-12)
    meta = {
        "snp": [f"rs{i}" for i in range(m)],
        "chr": ["1"] * m,
        "pos": list(range(100, 100 + m)),
        "a1": ["A"] * m,
        "a2": ["G"] * m,
    }
    block_assignments = [i * n_blocks // m for i in range(m)]
    ld = build_ld_reference(
        G, meta, mode="block", block_assignments=block_assignments
    )
    save_ld_reference(ld, str(path))


# ---------------------------------------------------------------------------
# api.pgs_fit device routing (the load-bearing surface)
# ---------------------------------------------------------------------------


def _run_pgs_fit(tmp_path: Path, device: str) -> Path:
    import torchgenomics.api as tg

    ss_path = tmp_path / "sumstats.tsv"
    ld_path = tmp_path / "ld.pt"
    out_path = tmp_path / "weights.tsv"
    _write_sumstats(ss_path, m=12)
    _write_block_ld_ref(ld_path, m=12, n_blocks=3)
    run = tg.pgs_fit(
        sumstats=str(ss_path),
        ld_ref=str(ld_path),
        output=str(out_path),
        method="ldpred2-inf",
        h2=0.2,
        device=device,
    )
    assert run.n_variants_with_weights == 12
    assert out_path.exists()
    return out_path


def test_api_pgs_fit_cpu(tmp_path):
    """api.pgs_fit with device='cpu' on a block-mode LD ref must succeed."""
    _run_pgs_fit(tmp_path, device="cpu")


def test_api_pgs_fit_auto(tmp_path):
    """api.pgs_fit with device='auto' must succeed regardless of CUDA presence.

    On a CUDA host this resolves to 'cuda' and is the exact path that
    surfaced the device-mismatch bug at base.py line 569.
    """
    _run_pgs_fit(tmp_path, device="auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_api_pgs_fit_cuda(tmp_path):
    """api.pgs_fit with device='cuda' on a block-mode LD ref must succeed.

    This is the regression net for the original surface-audit bug: the
    CPU/CUDA device-mismatch on ``idx`` vs ``new_in_block`` inside
    ``_ld_reference_subset``.
    """
    _run_pgs_fit(tmp_path, device="cuda")


# ---------------------------------------------------------------------------
# api.pgs_score device routing (downstream surface — does not touch
# _ld_reference_subset but still must obey device routing end-to-end)
# ---------------------------------------------------------------------------


def _make_dummy_weights(path: Path, m: int = 6) -> None:
    res = PGSResult(
        method="ct",
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(100, 100 + m)),
        a1=["A"] * m,
        a2=["G"] * m,
        weight=torch.full((m,), 0.1, dtype=torch.float64),
        n_iter=0,
        h2=None,
        p_causal=None,
        converged=True,
        phi=None,
    )
    res.save(str(path))


def _write_tiny_bed_fixture(tmp_path: Path, n: int = 4, m: int = 6, seed: int = 2) -> Path:
    """Write a minimal PLINK 1 BED + BIM + FAM fixture."""
    import numpy as np

    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n, m), dtype=np.int8)  # 0/1/2

    # BED
    bed_path = tmp_path / "tiny.bed"
    # Magic bytes 6c 1b 01 (SNP-major)
    magic = bytes([0x6C, 0x1B, 0x01])
    # For each variant, encode n samples as 2-bit codes (00=hom A2, 01=missing,
    # 10=het, 11=hom A1). Use plink encoding: 0->11 (homA1), 1->10 (het),
    # 2->00 (homA2). Pack into bytes, sample-by-sample within a variant.
    code_map = {0: 0b11, 1: 0b10, 2: 0b00}
    n_bytes_per_var = (n + 3) // 4
    data = bytearray(magic)
    for j in range(m):
        for byte_idx in range(n_bytes_per_var):
            b = 0
            for k in range(4):
                samp = byte_idx * 4 + k
                if samp < n:
                    code = code_map[int(geno[samp, j])]
                else:
                    code = 0b00
                b |= code << (2 * k)
            data.append(b)
    bed_path.write_bytes(bytes(data))

    # BIM (variants in same order as weights)
    bim_path = tmp_path / "tiny.bim"
    with open(bim_path, "w") as f:
        for j in range(m):
            f.write(f"1\trs{j}\t0\t{100 + j}\tA\tG\n")

    # FAM
    fam_path = tmp_path / "tiny.fam"
    with open(fam_path, "w") as f:
        for i in range(n):
            f.write(f"FAM{i}\tIID{i}\t0\t0\t0\t-9\n")

    return bed_path


def _run_pgs_score(tmp_path: Path, device: str) -> Path:
    import torchgenomics.api as tg

    weights_path = tmp_path / "w.tsv"
    out_path = tmp_path / "scores.tsv"
    _make_dummy_weights(weights_path, m=6)
    bed_path = _write_tiny_bed_fixture(tmp_path, n=4, m=6)
    run = tg.pgs_score(
        genotype=str(bed_path),
        weights=str(weights_path),
        output=str(out_path),
        device=device,
    )
    assert out_path.exists()
    # Score for each individual (4)
    assert run.n_samples == 4
    return out_path


def test_api_pgs_score_cpu(tmp_path):
    _run_pgs_score(tmp_path, device="cpu")


def test_api_pgs_score_auto(tmp_path):
    _run_pgs_score(tmp_path, device="auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_api_pgs_score_cuda(tmp_path):
    _run_pgs_score(tmp_path, device="cuda")
