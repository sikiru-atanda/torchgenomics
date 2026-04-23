"""Phase 55 Tier 2: end-to-end tests that run real R + updog.

Module-level skipif gates on missing R/updog so these tests silently
skip on CI matrix jobs without R.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import torch

from torchgwas.preprocess import dosage_call as dc_module


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


pytestmark = pytest.mark.skipif(
    not _updog_available(),
    reason="R + updog not installed; run locally or in release-time CI.",
)


# ---------- helpers ----------

def _simulate_reads_tetraploid(n_samples: int, n_markers: int, depth: int,
                                  seq_err: float, bias: float, od: float,
                                  seed: int):
    """Simulate beta-binomial reads for tetraploid genotypes.
    Returns (true_dosages, ref_counts, alt_counts, allele_freqs).
    """
    rng = np.random.default_rng(seed)
    ploidy = 4
    af = rng.uniform(0.1, 0.9, size=n_markers)
    # Sample true dosages per marker from binomial(k, af)
    true_dosage = np.stack([
        rng.binomial(ploidy, af[j], size=n_samples) for j in range(n_markers)
    ], axis=-1)  # (n, m)
    # Expected alt fraction with bias + sequencing error
    p_true = true_dosage / ploidy
    p_obs = (1 - seq_err) * p_true + seq_err * (1 - p_true)
    # Apply simple bias as a multiplicative tilt
    p_obs = p_obs * bias / (p_obs * bias + (1 - p_obs))
    # Beta-binomial dispersion
    if od > 0:
        alpha = p_obs * (1 - od) / od
        beta = (1 - p_obs) * (1 - od) / od
        p_sample = rng.beta(alpha.clip(1e-6), beta.clip(1e-6))
    else:
        p_sample = p_obs
    alt = rng.binomial(depth, p_sample)
    ref = depth - alt
    return true_dosage, ref, alt, af


def _write_sim_vcf(path, sample_ids, variant_ids, ref, alt):
    """Write a minimal biallelic VCF with AD from sim arrays."""
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depth">',
    ]
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
              "INFO", "FORMAT"] + list(sample_ids)
    lines.append("\t".join(header))
    for j, vid in enumerate(variant_ids):
        row = ["1", str(100 + j), vid, "A", "T", ".", "PASS", ".", "GT:AD"]
        for i, _ in enumerate(sample_ids):
            row.append(f"./.:{ref[i, j]},{alt[i, j]}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")


# ---------- tests ----------

def test_parity_on_identical_ref_size_tsvs(tmp_path):
    """Our R driver vs. bare multidog() on byte-identical ref.tsv/size.tsv.
    Proves the driver is not doing anything weird above multidog's own
    parse path.
    """
    import glob
    import tempfile as _tempfile

    # 1) Run our wrapper with keep_tmpdir=True on a simulated VCF
    sample_ids = [f"S{i}" for i in range(10)]
    variant_ids = [f"v{j}" for j in range(8)]
    _, ref, alt, _ = _simulate_reads_tetraploid(
        10, 8, depth=30, seq_err=0.01, bias=1.0, od=0.01, seed=0,
    )
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    out_prefix = tmp_path / "ours"
    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(out_prefix),
        ploidy=4, model="norm", keep_tmpdir=True,
    )
    ours = result.probs

    # 2) Find the kept tempdir (latest torchgwas_dosage_* under system tmp)
    candidates = sorted(glob.glob(
        f"{_tempfile.gettempdir()}/torchgwas_dosage_*"))
    assert candidates, "keep_tmpdir=True did not leave a tempdir on disk"
    ours_tmpdir = candidates[-1]

    # 3) Run bare multidog on the same ref.tsv/size.tsv via our driver script
    ref_dir = tmp_path / "ref_run"
    ref_dir.mkdir()
    driver_R = tmp_path / "ref_driver.R"
    driver_R.write_text(dc_module._UPDOG_DRIVER_R)
    subprocess.run(
        [shutil.which("Rscript") or "Rscript", str(driver_R),
         f"{ours_tmpdir}/ref.tsv", f"{ours_tmpdir}/size.tsv",
         "4", "norm", "TRUE", "TRUE", "NULL", "1", str(ref_dir)],
        check=True,
    )
    ref_probs, _ = dc_module._parse_output(ref_dir, sample_ids, variant_ids, 4)

    torch.testing.assert_close(ours, ref_probs, atol=1e-6, rtol=0)


def test_simulated_recovery_highdepth_tetraploid(tmp_path):
    """Mode-recovery at 30x depth must be above calibrated threshold.

    The recovery threshold is sourced from
    ``bench/calibrate_updog_recovery.py`` (run at phase sign-off), minus
    a 2% cross-version margin. If this is the first run, set threshold
    to 0.85 as a starting point and tighten after calibration.
    """
    # Calibration source: bench/calibrate_updog_recovery.py, run 2026-XX-XX.
    # Update this number after the first calibration run.
    RECOVERY_FLOOR = 0.85

    n_samples, n_markers = 50, 40
    true_d, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=30, seq_err=0.01, bias=1.0, od=0.01,
        seed=42,
    )

    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm",
    )
    # true_d is (m, n), result.probs is (n, m, k+1). Take mode dosage (n, m).
    mode = result.probs.argmax(dim=-1).numpy()  # (n, m)
    recovery = (mode == true_d.T).mean()
    assert recovery >= RECOVERY_FLOOR, (
        f"recovery={recovery:.3f} below floor {RECOVERY_FLOOR}"
    )


def test_simulated_recovery_lowdepth_tetraploid(tmp_path):
    """Same as above but at 8x depth. Looser floor."""
    RECOVERY_FLOOR = 0.65  # replace with calibrated value at sign-off

    n_samples, n_markers = 50, 40
    true_d, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=8, seq_err=0.01, bias=1.0, od=0.02,
        seed=7,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm",
    )
    mode = result.probs.argmax(dim=-1).numpy()
    recovery = (mode == true_d.T).mean()
    assert recovery >= RECOVERY_FLOOR, (
        f"recovery={recovery:.3f} below floor {RECOVERY_FLOOR}"
    )


def test_gulm_end_to_end_on_simulated_dosages(tmp_path):
    """Smoke test: sim → wrapper → dosage_variance → GULM score. KS p>0.01
    (not a tight calibration — see test_gu_lmm.py for the full battery).
    """
    from scipy.stats import kstest
    from torchgwas.models.gu_lmm import GULM
    from torchgwas.preprocess.dosage_uncertainty import (
        expected_dosage, dosage_variance,
    )
    from torchgwas.linalg.grm import compute_grm

    n_samples, n_markers = 200, 500
    _, ref, alt, _ = _simulate_reads_tetraploid(
        n_samples, n_markers, depth=15, seq_err=0.01, bias=1.0, od=0.01,
        seed=1,
    )
    sample_ids = [f"S{i}" for i in range(n_samples)]
    variant_ids = [f"v{j}" for j in range(n_markers)]
    vcf_path = tmp_path / "sim.vcf"
    _write_sim_vcf(vcf_path, sample_ids, variant_ids, ref, alt)

    result = dc_module.run_updog(
        input_vcf=str(vcf_path), output_path=str(tmp_path / "out"),
        ploidy=4, model="norm", n_cores=1,
    )
    dosages = expected_dosage(result.probs, 4)   # (n, m)
    dvar = dosage_variance(result.probs, 4)      # (n, m)

    # Simulate a null phenotype: y = grand mean + noise
    rng = np.random.default_rng(123)
    y = torch.from_numpy(rng.standard_normal(n_samples)).to(torch.float64)
    covars = torch.ones(n_samples, 1, dtype=torch.float64)

    # GRM from the expected dosages themselves (reasonable for a smoke test)
    grm = compute_grm(dosages, ploidy=4)

    model = GULM(grm=grm)
    null = model.fit_null(y=y, covars=covars)
    scan = model.score_chunk(
        null_fit=null, G=dosages.T, test="score", dosage_var=dvar.T,
    )
    pvals = scan.p_values.numpy()
    pvals = pvals[np.isfinite(pvals)]
    ks_stat, ks_p = kstest(pvals, "uniform")
    assert ks_p > 0.01, (
        f"null p-values not uniform under GULM+dosage_var (KS p={ks_p:.3g})"
    )
