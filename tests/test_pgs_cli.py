"""CLI smoke tests for pgs-fit and pgs-score."""

from __future__ import annotations

import io
import math
from contextlib import redirect_stdout

import torch

from torchgwas.cli import main as cli_main
from torchgwas.pgs.base import PGSResult
from torchgwas.pgs.ld_ref import build_ld_reference, save_ld_reference


def _write_sumstats(path, m=10, n_obs=2000, seed=0):
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


def _write_ld_ref(path, m=10, n=300, seed=1):
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
    ld = build_ld_reference(G, meta, mode="full")
    save_ld_reference(ld, str(path))


class _CLIResult:
    def __init__(self, stdout: str, returncode: int):
        self.stdout = stdout
        self.returncode = returncode


def _run_cli(*args, expect_code=0):
    buf = io.StringIO()
    code = 0
    try:
        with redirect_stdout(buf):
            code = cli_main(list(args))
    except SystemExit as e:  # argparse --help exits via SystemExit(0)
        code = int(e.code) if e.code is not None else 0
    assert code == expect_code, (
        f"CLI exited with {code}, expected {expect_code}. Output: {buf.getvalue()}"
    )
    return _CLIResult(buf.getvalue(), code)


# ---------------------------------------------------------------------------


def test_cli_pgs_fit_help():
    p = _run_cli("pgs-fit", "--help")
    assert "PGS weights" in p.stdout or "PGS" in p.stdout
    assert "--method" in p.stdout


def test_cli_pgs_score_help():
    p = _run_cli("pgs-score", "--help")
    assert "target" in p.stdout.lower() or "weights" in p.stdout.lower()


def test_cli_pgs_fit_ldpred2_inf(tmp_path):
    ss_path = tmp_path / "sumstats.tsv"
    ld_path = tmp_path / "ld.pt"
    out_path = tmp_path / "weights.tsv"
    _write_sumstats(ss_path)
    _write_ld_ref(ld_path)
    _run_cli(
        "pgs-fit",
        "--sumstats", str(ss_path),
        "--ld-ref", str(ld_path),
        "--method", "ldpred2-inf",
        "--h2", "0.2",
        "--output", str(out_path),
    )
    assert out_path.exists()
    assert (tmp_path / "weights.tsv.json").exists()
    loaded = PGSResult.load(str(out_path))
    assert loaded.method == "ldpred2-inf"
    assert loaded.m == 10


def test_cli_pgs_fit_ct(tmp_path):
    ss_path = tmp_path / "sumstats.tsv"
    ld_path = tmp_path / "ld.pt"
    out_path = tmp_path / "weights.tsv"
    _write_sumstats(ss_path)
    _write_ld_ref(ld_path)
    _run_cli(
        "pgs-fit",
        "--sumstats", str(ss_path),
        "--ld-ref", str(ld_path),
        "--method", "ct",
        "--clump-p", "1.0",          # accept everything
        "--clump-r2", "0.9",
        "--output", str(out_path),
    )
    loaded = PGSResult.load(str(out_path))
    assert loaded.method == "ct"


def test_cli_pgs_fit_prscs_smoke(tmp_path):
    ss_path = tmp_path / "sumstats.tsv"
    ld_path = tmp_path / "ld.pt"
    out_path = tmp_path / "weights.tsv"
    _write_sumstats(ss_path, m=6)
    _write_ld_ref(ld_path, m=6)
    _run_cli(
        "pgs-fit",
        "--sumstats", str(ss_path),
        "--ld-ref", str(ld_path),
        "--method", "prscs",
        "--phi", "0.01",
        "--n-iter", "20",
        "--n-burnin", "10",
        "--n-chains", "1",
        "--output", str(out_path),
    )
    loaded = PGSResult.load(str(out_path))
    assert loaded.method == "prscs"
    assert loaded.phi == 0.01


def test_cli_registers_pgs_commands_in_dispatch():
    # Confirm pgs-fit and pgs-score appear in the handlers dispatch table
    # without actually running them.
    from torchgwas.cli import main as cli_main  # noqa: F401 — import side effects
    p = _run_cli("--help")
    assert "pgs-fit" in p.stdout
    assert "pgs-score" in p.stdout
