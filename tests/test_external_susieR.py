"""Tier 2 parity test: bayes-scan-rss vs susieR::susie_rss().

Per NA1 design spec section 5.2. Marked external + golden so CI can
selectively run / skip. Requires R + susieR installed via
validation/external/susieR/install.sh.
"""
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.external, pytest.mark.golden]

VALIDATION_DIR = Path(__file__).parent.parent / "validation" / "external" / "susieR"


def test_susieR_parity_full_workflow():
    """End-to-end: install -> fetch_data -> run both -> compare -> assert pass."""
    # Skip if R / susieR not installed
    try:
        r_check = subprocess.run(
            ["R", "-e", "library(susieR)"],
            capture_output=True,
        )
    except FileNotFoundError:
        pytest.skip("R not installed; run validation/external/susieR/install.sh")
    if r_check.returncode != 0:
        pytest.skip(
            "susieR not installed; run validation/external/susieR/install.sh"
        )

    # Provision fixture (idempotent)
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "fetch_data.sh")],
        check=True,
    )

    # Skip if fetch_data placeholder didn't actually provision (Tier A intent)
    if not (VALIDATION_DIR / "data" / "locus_z.pt").exists():
        pytest.skip("fixture not provisioned (fetch_data.sh is a Tier A placeholder)")

    # Reference run
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "run_susieR.sh")],
        check=True,
    )

    # Our run
    subprocess.run(
        ["bash", str(VALIDATION_DIR / "run_torchgenomics.sh")],
        check=True,
    )

    # Compare
    out_dir = VALIDATION_DIR / "outputs"
    rc = subprocess.call([
        "python3",
        str(VALIDATION_DIR / "compare.py"),
        "--upstream", str(out_dir / "susieR.tsv"),
        "--ours", str(out_dir / "torchgenomics.tsv"),
        "--upstream-walltime", str(out_dir / "susieR_walltime_seconds.txt"),
        "--ours-walltime", str(out_dir / "torchgenomics_walltime_seconds.txt"),
        "--upstream-elbo", str(out_dir / "susieR_elbo.txt"),
        "--ours-elbo", str(out_dir / "torchgenomics_elbo.txt"),
    ])
    assert rc == 0, "Tier 2 parity failed; see output above and findings ledger"
