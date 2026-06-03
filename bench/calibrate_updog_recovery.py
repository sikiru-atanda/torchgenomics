"""Phase 55 one-time calibration helper. NOT a pytest test.

Run once at phase sign-off on a machine with R + updog installed:

    python bench/calibrate_updog_recovery.py

It runs the same simulations used in tests/test_dosage_call_updog_e2e.py
and prints the observed mode-recovery rate at 30x and 8x depth. Paste
those numbers (minus 2%) into the RECOVERY_FLOOR constants in the test
file.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running directly from repo root
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from torchgenomics.preprocess import dosage_call as dc_module  # noqa: E402


def _updog_available() -> bool:
    rs = shutil.which("Rscript")
    if rs is None:
        return False
    probe = subprocess.run(
        [rs, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    return probe.returncode == 0


def _simulate(n_samples, n_markers, depth, seq_err, bias, od, seed):
    rng = np.random.default_rng(seed)
    ploidy = 4
    af = rng.uniform(0.1, 0.9, size=n_markers)
    true_d = np.stack([
        rng.binomial(ploidy, af[j], size=n_samples) for j in range(n_markers)
    ], axis=-1)
    p_true = true_d / ploidy
    p_obs = (1 - seq_err) * p_true + seq_err * (1 - p_true)
    p_obs = p_obs * bias / (p_obs * bias + (1 - p_obs))
    if od > 0:
        alpha = p_obs * (1 - od) / od
        beta = (1 - p_obs) * (1 - od) / od
        p_sample = rng.beta(alpha.clip(1e-6), beta.clip(1e-6))
    else:
        p_sample = p_obs
    alt = rng.binomial(depth, p_sample)
    ref = depth - alt
    return true_d, ref, alt


def _write_vcf(path, sample_ids, variant_ids, ref, alt):
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">',
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
                   "INFO", "FORMAT"] + list(sample_ids)),
    ]
    for j, vid in enumerate(variant_ids):
        row = ["1", str(100 + j), vid, "A", "T", ".", "PASS", ".", "GT:AD"]
        for i, _ in enumerate(sample_ids):
            row.append(f"./.:{ref[i, j]},{alt[i, j]}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


def run_one(depth: int, od: float, seed: int, n_samples=50, n_markers=40):
    true_d, ref, alt = _simulate(
        n_samples, n_markers, depth=depth, seq_err=0.01, bias=1.0, od=od,
        seed=seed,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _write_vcf(td / "sim.vcf", sample_ids, variant_ids, ref, alt)
        result = dc_module.run_updog(
            input_vcf=str(td / "sim.vcf"),
            output_path=str(td / "out"),
            ploidy=4, model="norm",
        )
    mode = result.probs.argmax(dim=-1).numpy()
    return float((mode == true_d.T).mean())


if __name__ == "__main__":
    if not _updog_available():
        print("R + updog not installed. Install and re-run.", file=sys.stderr)
        sys.exit(2)

    print("Phase 55 recovery-rate calibration")
    print("=" * 50)
    r30 = run_one(depth=30, od=0.01, seed=42)
    print(f"Tetraploid 30x depth: recovery = {r30:.4f}")
    r8 = run_one(depth=8, od=0.02, seed=7)
    print(f"Tetraploid  8x depth: recovery = {r8:.4f}")
    print()
    print("Paste into tests/test_dosage_call_updog_e2e.py:")
    print(f"  highdepth: RECOVERY_FLOOR = {max(r30 - 0.02, 0.0):.3f}")
    print(f"  lowdepth:  RECOVERY_FLOOR = {max(r8 - 0.02, 0.0):.3f}")
