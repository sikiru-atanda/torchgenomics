"""Polyploid allele dosage assignment via updog (Phase 55).

Thin external-wrapper around the R package ``updog``. Converts VCF read
counts (AD field) into posterior ``P(dosage=0..k)`` tensors. See
``docs/superpowers/specs/2026-04-22-polyploid-dosage-assignment-design.md``
for the full design.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


_VALID_MODELS: frozenset[str] = frozenset({
    "norm", "hw", "bb", "s1", "f1", "s1pp", "f1pp", "flex", "uniform",
})

_UPDOG_CHECKED: Optional[bool] = None  # cached result of environment probe


@dataclass
class DosageCallResult:
    probs: Tensor
    sample_ids: list[str]
    variant_ids: list[str]
    ploidy: int
    tool: str
    tool_version: str
    model: str
    mean_dosage_var: Tensor
    allele_freq: Tensor
    n_missing: int
    input_hash: str
    cmd: str


def _check_environment(rscript: Optional[str]) -> str:
    """Probe for Rscript + updog; cache the verdict. Returns Rscript path.

    Raises RuntimeError with install pointers if either is missing.
    Cached on a module-level flag; the probe runs at most once per process.
    """
    global _UPDOG_CHECKED

    if rscript is None:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Install R >= 4.0 and ensure Rscript is on "
            "PATH, or pass rscript=<path>."
        )

    if _UPDOG_CHECKED is True:
        return rscript

    probe = subprocess.run(
        [rscript, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        _UPDOG_CHECKED = False
        raise RuntimeError(
            "updog R package not installed. Install with: "
            "Rscript -e 'install.packages(\"updog\")'. "
            f"(probe stderr: {probe.stderr.strip()})"
        )

    logger.info("updog R package detected: version %s", probe.stdout.strip())
    _UPDOG_CHECKED = True
    return rscript


def _validate_kwargs(*, ploidy: int, model: str) -> None:
    """Validate ploidy and model kwargs before R dispatch.

    Raises ValueError if ploidy is out of range [2, 8] or model is not in
    the updog flexdog set.
    """
    if not (2 <= ploidy <= 8):
        raise ValueError(
            f"ploidy must be in [2, 8]; got {ploidy}. updog is validated "
            "up to ploidy=8 upstream."
        )
    if model not in _VALID_MODELS:
        raise ValueError(
            f"model={model!r} not in updog flexdog set: {sorted(_VALID_MODELS)}"
        )


def _extract_ad_from_vcf(
    input_vcf: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Read a biallelic VCF with AD field. Returns (sample_ids, variant_ids,
    refmat, sizemat) where matrices are shape (m, n) int64, rows=variants,
    cols=samples. Missing AD entries (negative cyvcf2 sentinel) become
    (ref=0, size=0), which updog treats as missing.
    """
    try:
        import cyvcf2  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "dosage_call requires cyvcf2: pip install cyvcf2"
        ) from e

    vcf = cyvcf2.VCF(input_vcf)
    try:
        # Header probe: AD must be declared under FORMAT
        has_ad = False
        for h in vcf.header_iter():
            d = h.info(extra=True)
            if d.get("HeaderType") == "FORMAT" and d.get("ID") == "AD":
                has_ad = True
                break
        if not has_ad:
            raise ValueError(
                f"VCF {input_vcf!r} has no AD field under FORMAT. Re-call with "
                "a tool that emits allele depth (e.g. GATK HaplotypeCaller, "
                "bcftools mpileup -a FORMAT/AD)."
            )

        sample_ids = list(vcf.samples)
        variant_ids: list[str] = []
        ref_rows: list[np.ndarray] = []
        size_rows: list[np.ndarray] = []

        for v in vcf:
            if len(v.ALT) != 1:
                raise ValueError(
                    f"Multi-allelic site at {v.CHROM}:{v.POS} (ID={v.ID}); "
                    f"updog requires biallelic. Split first: "
                    f"bcftools norm -m -any <in> > <out>."
                )
            ad = v.format("AD")
            if ad is None:
                raise ValueError(
                    f"Variant {v.CHROM}:{v.POS} (ID={v.ID}) has no AD data."
                )
            ad = np.asarray(ad, dtype=np.int64)
            # cyvcf2 sentinel for missing: negative
            missing = (ad < 0).any(axis=1)
            ref = ad[:, 0].copy()
            alt = ad[:, 1].copy()
            ref[missing] = 0
            alt[missing] = 0
            ref_rows.append(ref)
            size_rows.append(ref + alt)
            vid = v.ID if v.ID else f"{v.CHROM}_{v.POS}"
            variant_ids.append(vid)

        if not ref_rows:
            raise ValueError(f"VCF {input_vcf!r} contains zero variants.")

        refmat = np.stack(ref_rows, axis=0)
        sizemat = np.stack(size_rows, axis=0)
        return sample_ids, variant_ids, refmat, sizemat
    finally:
        vcf.close()


def _write_input_tsvs(
    tmpdir: Path,
    sample_ids: list[str],
    variant_ids: list[str],
    refmat: np.ndarray,
    sizemat: np.ndarray,
) -> tuple[Path, Path]:
    """Write ref.tsv and size.tsv to tmpdir. Rows=variants, cols=samples,
    with index label row0col0 blank (R convention; pandas reads it via
    index_col=0). Returns the two paths.
    """
    ref_tsv = tmpdir / "ref.tsv"
    size_tsv = tmpdir / "size.tsv"
    pd.DataFrame(refmat, index=variant_ids, columns=sample_ids).to_csv(
        ref_tsv, sep="\t", index=True, index_label=""
    )
    pd.DataFrame(sizemat, index=variant_ids, columns=sample_ids).to_csv(
        size_tsv, sep="\t", index=True, index_label=""
    )
    return ref_tsv, size_tsv


_UPDOG_DRIVER_R = r"""# R driver for torchgwas.preprocess.dosage_call — invoked via Rscript.
args <- commandArgs(trailingOnly = TRUE)
ref_tsv <- args[1]
size_tsv <- args[2]
ploidy <- as.integer(args[3])
model <- args[4]
update_bias <- as.logical(args[5])
update_od <- as.logical(args[6])
seq_arg <- args[7]
n_cores <- as.integer(args[8])
out_dir <- args[9]

suppressPackageStartupMessages(library(updog))

refmat <- as.matrix(read.table(ref_tsv, sep = "\t", header = TRUE,
                                row.names = 1, check.names = FALSE))
sizemat <- as.matrix(read.table(size_tsv, sep = "\t", header = TRUE,
                                 row.names = 1, check.names = FALSE))

multidog_args <- list(
  refmat = refmat, sizemat = sizemat,
  ploidy = ploidy, model = model, nc = n_cores,
  update_bias = update_bias, update_od = update_od
)
if (seq_arg != "NULL") {
  multidog_args$seq <- as.numeric(seq_arg)
  multidog_args$update_seq <- FALSE
}
mout <- do.call(multidog, multidog_args)

for (d in 0:ploidy) {
  mat <- format_multidog(mout, varname = paste0("Pr_", d))
  write.table(mat, file.path(out_dir, paste0("pr_", d, ".tsv")),
              sep = "\t", quote = FALSE, col.names = NA)
}
write.table(mout$snpdf, file.path(out_dir, "snp_diag.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("updog multidog complete\n")
"""


def _run_r_subprocess(
    *,
    rscript: str,
    tmpdir: Path,
    ref_tsv: Path,
    size_tsv: Path,
    ploidy: int,
    model: str,
    bias: bool,
    od: bool,
    seq_error: Optional[float],
    n_cores: int,
) -> str:
    """Write the driver script to tmpdir, spawn Rscript, return the
    stringified command for reproducibility. Raises RuntimeError on
    non-zero exit with stderr attached.
    """
    driver_path = tmpdir / "driver.R"
    driver_path.write_text(_UPDOG_DRIVER_R)

    cmd = [
        rscript, str(driver_path),
        str(ref_tsv), str(size_tsv),
        str(ploidy), model,
        "TRUE" if bias else "FALSE",
        "TRUE" if od else "FALSE",
        "NULL" if seq_error is None else f"{seq_error:g}",
        str(n_cores),
        str(tmpdir),
    ]
    logger.info("Running updog: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"updog failed (exit {result.returncode}). stderr:\n{result.stderr}"
        )
    if result.stderr:
        logger.info("updog stderr:\n%s", result.stderr)
    return " ".join(cmd)
