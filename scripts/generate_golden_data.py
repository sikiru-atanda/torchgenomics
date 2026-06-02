"""Regenerate GEMMA / GAPIT3 / GWASpoly reference outputs used by the golden tests.

The golden tests in ``tests/test_golden_{gemma,gapit,gwaspoly}.py`` compare
TorchGenomics results against reference outputs from established tools. This script
is the one-time orchestrator that a maintainer runs (on a machine with the
tools installed) to regenerate those reference outputs.

The reference outputs are checked into the repository because:
  * they are small (~500 KB total for the MDP + potato datasets)
  * the tools (GEMMA 0.98.5, GAPIT3, GWASpoly) are not trivially installable
    in every CI environment
  * freezing a specific upstream version is part of the regression contract.

Usage
-----

On a Linux/WSL machine with GEMMA 0.98.5 on PATH::

    python scripts/generate_golden_data.py --gemma

On a machine with R + GAPIT3 installed::

    Rscript scripts/generate_golden_data.R --gapit

On a machine with R + GWASpoly installed::

    Rscript scripts/generate_golden_data.R --gwaspoly

Or (convenience): verify current fixtures are in place without running anything::

    python scripts/generate_golden_data.py --check

See ``docs/validation_protocol.md`` for the exact software versions and
reference commands used in the shipped fixtures.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "benchmark" / "data"
GEMMA_BIN = REPO_ROOT / "gemma_demo" / "gemma-0.98.5"
GEMMA_OUT = REPO_ROOT / "gemma_demo" / "output"
GAPIT_OUT = REPO_ROOT / "gapit_demo" / "output"
GWASPOLY_OUT = REPO_ROOT / "benchmark" / "gwaspoly_results"

GEMMA_FIXTURES = [
    "mdp_kinship.cXX.txt",
    "mdp_lmm_all.assoc.txt",
    "mdp_lmm_all.log.txt",
    "mdp_mvlmm_wald.assoc.txt",
    "mdp_mvlmm_wald.log.txt",
]
GAPIT_FIXTURES = [
    "mdp_farmcpu.GWAS.Results.csv",
    "mdp_blink.GWAS.Results.csv",
]
GWASPOLY_FIXTURES = [
    "gwaspoly_additive.csv",
    "gwaspoly_1_dom_alt.csv",
    "gwaspoly_2_dom_alt.csv",
    "gwaspoly_2_dom_ref.csv",
    "gwaspoly_diplo_additive.csv",
    "potato_geno_aligned.csv",
    "potato_pheno_with_genoid.csv",
    "potato_map.csv",
]


def _check(target_dir: Path, fixtures: list[str]) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for fname in fixtures:
        if (target_dir / fname).is_file():
            present.append(fname)
        else:
            missing.append(fname)
    return present, missing


def cmd_check(args: argparse.Namespace) -> int:
    sections = [
        ("GEMMA", GEMMA_OUT, GEMMA_FIXTURES),
        ("GAPIT3", GAPIT_OUT, GAPIT_FIXTURES),
        ("GWASpoly", GWASPOLY_OUT, GWASPOLY_FIXTURES),
    ]
    any_missing = False
    for label, d, fs in sections:
        present, missing = _check(d, fs)
        print(f"[{label}] {len(present)}/{len(fs)} fixtures present at {d}")
        for fname in missing:
            print(f"    MISSING: {fname}")
            any_missing = True
    return 1 if any_missing else 0


def cmd_gemma(args: argparse.Namespace) -> int:
    """Regenerate GEMMA reference outputs on the MDP maize dataset."""
    if not GEMMA_BIN.exists() and not shutil.which("gemma"):
        print(
            "ERROR: GEMMA 0.98.5 binary not found. Expected at "
            f"{GEMMA_BIN} or on $PATH. See docs/validation_protocol.md.",
            file=sys.stderr,
        )
        return 2
    gemma = str(GEMMA_BIN) if GEMMA_BIN.exists() else "gemma"

    # BIMBAM inputs are expected at gemma_demo/. Regenerate them from the
    # GAPIT-format MDP data if needed.
    demo_dir = REPO_ROOT / "gemma_demo"
    demo_dir.mkdir(exist_ok=True)
    GEMMA_OUT.mkdir(parents=True, exist_ok=True)

    bimbam_script = REPO_ROOT / "benchmark" / "convert_mdp_to_bimbam.py"
    if not bimbam_script.is_file():
        print(f"ERROR: Missing BIMBAM conversion helper at {bimbam_script}", file=sys.stderr)
        return 2

    print(f"[1/3] Converting MDP -> BIMBAM via {bimbam_script.name}")
    subprocess.run([sys.executable, str(bimbam_script)], check=True, cwd=demo_dir)

    print("[2/3] Running GEMMA kinship + LMM (1 trait, EarHT)")
    for args_list, output_prefix in [
        (["-gk", "1", "-o", "mdp_kinship"], "mdp_kinship"),
        (
            [
                "-g", "mdp_geno_bimbam.txt",
                "-p", "mdp_pheno1.txt",
                "-a", "mdp_snps_bimbam.txt",
                "-k", "output/mdp_kinship.cXX.txt",
                "-lmm", "4",
                "-o", "mdp_lmm_all",
            ],
            "mdp_lmm_all",
        ),
    ]:
        print(f"    gemma {' '.join(args_list)}")
        subprocess.run([gemma] + args_list, check=True, cwd=demo_dir)

    print("[3/3] Running GEMMA mvLMM (2 traits, EarHT + dpoll)")
    subprocess.run(
        [
            gemma,
            "-g", "mdp_geno_bimbam.txt",
            "-p", "mdp_pheno2.txt",
            "-a", "mdp_snps_bimbam.txt",
            "-k", "output/mdp_kinship.cXX.txt",
            "-n", "1", "2",
            "-lmm", "1",
            "-o", "mdp_mvlmm_wald",
        ],
        check=True,
        cwd=demo_dir,
    )

    present, missing = _check(GEMMA_OUT, GEMMA_FIXTURES)
    if missing:
        print(f"WARNING: fixtures still missing after run: {missing}", file=sys.stderr)
        return 1
    print(f"OK: {len(present)}/{len(GEMMA_FIXTURES)} GEMMA fixtures regenerated.")
    return 0


def cmd_gapit(args: argparse.Namespace) -> int:
    print(
        "GAPIT3 regeneration is implemented in the R companion script:\n"
        "    Rscript scripts/generate_golden_data.R --gapit\n"
        "GAPIT3 is an R package and has no Python orchestrator.",
        file=sys.stderr,
    )
    return 2


def cmd_gwaspoly(args: argparse.Namespace) -> int:
    print(
        "GWASpoly regeneration is implemented in the R companion script:\n"
        "    Rscript scripts/generate_golden_data.R --gwaspoly\n"
        "GWASpoly is an R package and has no Python orchestrator.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gemma", action="store_true", help="Regenerate GEMMA outputs")
    ap.add_argument("--gapit", action="store_true", help="Point at R companion for GAPIT")
    ap.add_argument("--gwaspoly", action="store_true", help="Point at R companion for GWASpoly")
    ap.add_argument("--check", action="store_true", help="Report fixture presence only")
    args = ap.parse_args()

    if not any([args.gemma, args.gapit, args.gwaspoly, args.check]):
        ap.print_help()
        return 0

    rc = 0
    if args.check:
        rc |= cmd_check(args)
    if args.gemma:
        rc |= cmd_gemma(args)
    if args.gapit:
        rc |= cmd_gapit(args)
    if args.gwaspoly:
        rc |= cmd_gwaspoly(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
