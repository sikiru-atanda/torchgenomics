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
import os
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

_UPDOG_CHECKED: dict[str, str] = {}  # rscript_path -> updog_version (cached)

_HASH_CHUNK_BYTES = 1 << 20  # 1 MiB chunks — small enough to avoid RAM
# pressure on multi-GB WGS VCFs, large enough to keep sha256 overhead
# well under 0.1 s per GB on typical hardware.


def _sha256_file(path: str) -> str:
    """Chunked sha256 of a file on disk. Avoids loading the whole VCF
    into memory (`Path.read_bytes()` OOMs on >10GB VCFs)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


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
    missing_mask: Tensor  # (n, m) bool — True where updog returned no info
    input_hash: str
    cmd: str


def _check_environment(rscript: Optional[str]) -> tuple[str, str]:
    """Probe for Rscript + updog; cache the verdict keyed by Rscript path.

    Returns ``(rscript_path, updog_version)``. Raises :class:`RuntimeError`
    with install pointers if either is missing. Each distinct ``rscript``
    path is probed at most once per process; switching R installations
    mid-session re-probes the new path without disturbing the old cache.
    Failures are not cached — so a user who installs updog after a failed
    probe can retry without restarting the process.
    """
    if rscript is None:
        rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found. Install R >= 4.0 and ensure Rscript is on "
            "PATH, or pass rscript=<path>."
        )

    cached = _UPDOG_CHECKED.get(rscript)
    if cached is not None:
        return rscript, cached

    probe = subprocess.run(
        [rscript, "-e",
         'suppressPackageStartupMessages(library(updog)); '
         'cat(as.character(packageVersion("updog")))'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "updog R package not installed. Install with: "
            "Rscript -e 'install.packages(\"updog\")'. "
            f"(probe stderr: {probe.stderr.strip()})"
        )

    version = probe.stdout.strip() or "unknown"
    logger.info("updog R package detected: version %s", version)
    _UPDOG_CHECKED[rscript] = version
    return rscript, version


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
            ref_allele = v.REF
            alt_allele = v.ALT[0]
            # Reject symbolic / spanning-deletion / non-REF ALTs — they
            # pass the len() != 1 check above but carry no AD semantics
            # updog can interpret.
            if alt_allele.startswith("<") or alt_allele == "*":
                raise ValueError(
                    f"Symbolic or spanning-deletion ALT allele "
                    f"{alt_allele!r} at {v.CHROM}:{v.POS} (ID={v.ID}); "
                    f"updog requires concrete biallelic SNPs. Filter "
                    f"with: bcftools view -e 'ALT=\"<*>\" || ALT=\"*\"'."
                )
            # Reject indels: both REF and ALT must be exactly one nucleotide
            # for updog's dosage model to apply.
            if len(ref_allele) != 1 or len(alt_allele) != 1:
                raise ValueError(
                    f"Non-SNP site at {v.CHROM}:{v.POS} (ID={v.ID}): "
                    f"REF={ref_allele!r}, ALT={alt_allele!r}. updog "
                    f"requires biallelic SNPs. Filter with: "
                    f"bcftools view -v snps."
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


def _parse_output(
    tmpdir: Path,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
) -> tuple[Tensor, pd.DataFrame]:
    """Read pr_0..pr_k wide TSVs and snp_diag.tsv from tmpdir. Returns
    (probs, snp_diag) where probs has shape (n, m, k+1) float64 and
    is stacked along the last axis in dosage-class order.
    """
    prob_slices: list[np.ndarray] = []
    for d in range(ploidy + 1):
        path = tmpdir / f"pr_{d}.tsv"
        if not path.is_file():
            raise RuntimeError(
                f"updog exited 0 but produced no {path.name} in {tmpdir}."
            )
        df = pd.read_csv(path, sep="\t", index_col=0)
        # Reindex defensively so order matches our canonical ID lists
        df = df.reindex(index=variant_ids, columns=sample_ids)
        prob_slices.append(df.to_numpy(dtype=np.float64).T)  # (n, m)

    # (n, m, k+1)
    probs_np = np.stack(prob_slices, axis=-1)
    probs = torch.from_numpy(probs_np).to(torch.float64)

    diag_path = tmpdir / "snp_diag.tsv"
    if not diag_path.is_file():
        raise RuntimeError(f"updog produced no snp_diag.tsv in {tmpdir}.")
    snp_diag = pd.read_csv(diag_path, sep="\t")
    return probs, snp_diag


def _normalize_probs(probs: Tensor, ploidy: int) -> tuple[Tensor, int, Tensor]:
    """Validate + clean the (n, m, k+1) probability tensor. Returns
    ``(cleaned_probs, n_missing, missing_mask)`` where ``missing_mask``
    is a ``(n, m)`` bool tensor, True at each sample×marker slot that
    updog emitted with no signal (zero-row). Those slots are uniform-filled
    in ``cleaned_probs`` so the tensor sums to 1 everywhere — the mask
    preserves the original missingness information for downstream QC.

    Rules per the spec:
      - Shape must be (n, m, ploidy+1) — else RuntimeError.
      - Rows summing to 0 are treated as missing → set to uniform
        1/(k+1), ``n_missing += 1``, and the mask is flipped True at
        that slot.
      - Negative entries are clamped to 0 and the row is renormalized,
        with a warning via logger.
      - All remaining rows must sum to within atol=1e-4 of 1.0.
    """
    if probs.dim() != 3 or probs.shape[-1] != ploidy + 1:
        raise RuntimeError(
            f"probs shape {tuple(probs.shape)} does not match "
            f"(n, m, ploidy+1) with ploidy={ploidy}."
        )

    out = probs.clone()

    # Negative handling
    if (out < 0).any():
        logger.warning(
            "updog output contains %d negative probabilities; clamping to 0.",
            int((out < 0).sum().item()),
        )
        out = torch.clamp(out, min=0.0)

    row_sums = out.sum(dim=-1)  # (n, m)
    zero_mask = row_sums == 0
    n_missing = int(zero_mask.sum().item())

    # Uniform fill for missing rows
    if n_missing > 0:
        uniform = 1.0 / (ploidy + 1)
        out[zero_mask] = uniform

    # Renormalize non-missing rows
    non_missing = ~zero_mask
    if non_missing.any():
        out[non_missing] = out[non_missing] / out[non_missing].sum(
            dim=-1, keepdim=True
        )

    # Final sanity check
    final_sums = out.sum(dim=-1)
    if not torch.allclose(final_sums, torch.ones_like(final_sums), atol=1e-4):
        bad = (final_sums - 1.0).abs().max().item()
        raise RuntimeError(
            f"Normalized probs still deviate from 1.0 by up to {bad:.3g}."
        )

    return out, n_missing, zero_mask


def _build_result(
    *,
    probs: Tensor,
    sample_ids: list[str],
    variant_ids: list[str],
    ploidy: int,
    tool_version: str,
    model: str,
    n_missing: int,
    missing_mask: Tensor,
    input_hash: str,
    cmd: str,
) -> DosageCallResult:
    """Assemble a DosageCallResult from validated probs and metadata.
    Computes mean_dosage_var (uncertainty per variant) and allele_freq
    (AF = E[dosage] / ploidy).
    """
    d_vals = torch.arange(ploidy + 1, dtype=probs.dtype, device=probs.device)
    e_d = (probs * d_vals).sum(dim=-1)          # (n, m)
    e_d2 = (probs * d_vals ** 2).sum(dim=-1)    # (n, m)
    var_d = e_d2 - e_d ** 2
    mean_dosage_var = var_d.mean(dim=0)          # (m,)
    allele_freq = e_d.mean(dim=0) / ploidy       # (m,)

    return DosageCallResult(
        probs=probs,
        sample_ids=sample_ids,
        variant_ids=variant_ids,
        ploidy=ploidy,
        tool="updog",
        tool_version=tool_version,
        model=model,
        mean_dosage_var=mean_dosage_var,
        allele_freq=allele_freq,
        n_missing=n_missing,
        missing_mask=missing_mask,
        input_hash=input_hash,
        cmd=cmd,
    )


def _persist_artifacts(
    result: DosageCallResult,
    snp_diag: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    """Write `<prefix>.probs.pt`, `<prefix>.meta.json`, `<prefix>.snp_diag.tsv`.

    Atomicity contract — two-phase commit:

    1. Write all three artifacts to sibling ``.tmp`` paths. If any write
       raises, the ``.tmp`` siblings are cleaned up and the final paths
       remain unchanged (so a prior successful run isn't clobbered by a
       failed retry).
    2. Rename all three into place via :func:`os.replace`, which is
       atomic on both POSIX and Windows at the per-file level.

    A truly cross-file atomic swap would require a single archive; the
    three-``os.replace`` sequence has a small window where a mid-rename
    crash could leave the three user-visible paths inconsistent, but
    that window is orders of magnitude narrower than the write-phase
    window it replaces.
    """
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    final_probs = str(prefix_path) + ".probs.pt"
    final_meta = str(prefix_path) + ".meta.json"
    final_diag = str(prefix_path) + ".snp_diag.tsv"
    tmp_probs = final_probs + ".tmp"
    tmp_meta = final_meta + ".tmp"
    tmp_diag = final_diag + ".tmp"

    meta = {
        "tool": result.tool,
        "tool_version": result.tool_version,
        "model": result.model,
        "ploidy": result.ploidy,
        "sample_ids": result.sample_ids,
        "variant_ids": result.variant_ids,
        "n_missing": result.n_missing,
        "input_hash": result.input_hash,
        "cmd": result.cmd,
    }

    try:
        torch.save(result.probs, tmp_probs)
        Path(tmp_meta).write_text(json.dumps(meta, indent=2))
        snp_diag.to_csv(tmp_diag, sep="\t", index=False)
    except Exception:
        # Clean up any partially-written .tmp siblings; never touch the
        # final paths on the failure path.
        for p in (tmp_probs, tmp_meta, tmp_diag):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        raise

    # All three .tmp files written successfully — swap into place.
    os.replace(tmp_probs, final_probs)
    os.replace(tmp_meta, final_meta)
    os.replace(tmp_diag, final_diag)


def run_updog(
    input_vcf: str,
    output_path: str,
    *,
    ploidy: int,
    model: str = "norm",
    rscript: Optional[str] = None,
    bias: bool = True,
    od: bool = True,
    seq_error: Optional[float] = None,
    n_cores: int = 1,
    keep_tmpdir: bool = False,
) -> DosageCallResult:
    """Call polyploid allele dosages via updog.

    Parameters
    ----------
    input_vcf
        Biallelic VCF with AD format field.
    output_path
        Output prefix. Writes ``<prefix>.probs.pt``, ``<prefix>.meta.json``,
        and ``<prefix>.snp_diag.tsv``.
    ploidy
        Organism ploidy; 2 <= ploidy <= 8.
    model
        updog flexdog model name (see ``_VALID_MODELS``).
    rscript
        Override path to Rscript. If None, auto-detected from PATH.
    bias, od
        Whether updog should estimate allele bias / overdispersion.
    seq_error
        Fix the sequencing error rate at this value, or None to estimate.
    n_cores
        Parallelism passed to updog's ``nc`` argument.
    keep_tmpdir
        Skip tempdir cleanup for debugging.

    Returns
    -------
    DosageCallResult
        Posterior ``P(dosage=0..k)`` plus diagnostics; artifacts also
        persisted to disk at ``output_path``.
    """
    _validate_kwargs(ploidy=ploidy, model=model)
    rscript_path, version = _check_environment(rscript)

    sample_ids, variant_ids, refmat, sizemat = _extract_ad_from_vcf(input_vcf)

    input_hash = _sha256_file(input_vcf)

    # Manual mkdtemp + rmtree so ``keep_tmpdir=True`` is genuinely honored.
    # `tempfile.TemporaryDirectory` installs a weakref finalizer at
    # construction that deletes the directory when the object is GC'd,
    # which fires as soon as the local goes out of scope — so the flag
    # would be silently broken under normal function return.
    tmpdir = Path(tempfile.mkdtemp(prefix="torchgwas_dosage_"))
    try:
        ref_tsv, size_tsv = _write_input_tsvs(
            tmpdir, sample_ids, variant_ids, refmat, sizemat
        )
        cmd = _run_r_subprocess(
            rscript=rscript_path, tmpdir=tmpdir,
            ref_tsv=ref_tsv, size_tsv=size_tsv,
            ploidy=ploidy, model=model, bias=bias, od=od,
            seq_error=seq_error, n_cores=n_cores,
        )
        probs, snp_diag = _parse_output(tmpdir, sample_ids, variant_ids, ploidy)
        probs, n_missing, missing_mask = _normalize_probs(probs, ploidy)

        result = _build_result(
            probs=probs,
            sample_ids=sample_ids, variant_ids=variant_ids,
            ploidy=ploidy, tool_version=version, model=model,
            n_missing=n_missing, missing_mask=missing_mask,
            input_hash=input_hash, cmd=cmd,
        )
        _persist_artifacts(result, snp_diag, prefix=output_path)
        return result
    finally:
        if keep_tmpdir:
            logger.info("kept tempdir: %s", tmpdir)
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
