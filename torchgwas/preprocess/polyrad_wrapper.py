"""Wrapper for polyRAD dosage calling from R.

polyRAD is an R package that estimates posterior genotype probabilities
P(dosage=0..k) per sample per marker from sequencing reads using a
Bayesian framework with population-level allele frequency priors.

The parallel updog wrapper (a more robust implementation using updog's
multidog batch API) lives in :mod:`torchgwas.preprocess.dosage_call`.
"""

from __future__ import annotations

import csv
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class DosageProbabilities:
    """Output of dosage calling: posterior probabilities and expected dosages.

    Attributes
    ----------
    probs : Tensor, shape (n_samples, n_markers, ploidy+1)
        Posterior probability P(dosage = d) for d in 0..ploidy.
    expected : Tensor, shape (n_samples, n_markers)
        Expected dosage E[d] = sum_d d * P(d).
    sample_ids : list[str]
        Sample identifiers.
    marker_ids : list[str]
        Marker/variant identifiers.
    ploidy : int
        Organism ploidy level.
    """

    probs: Tensor
    expected: Tensor
    sample_ids: list[str]
    marker_ids: list[str]
    ploidy: int


def run_polyrad(
    vcf_path: str,
    ploidy: int,
    *,
    r_executable: str | None = None,
    possiblePloidies: list[int] | None = None,
    min_allele_freq: float = 0.01,
) -> DosageProbabilities:
    """Run polyRAD dosage calling on a VCF file.

    polyRAD uses a Bayesian framework with population-level allele frequency
    priors to estimate genotype posteriors from read counts.

    Parameters
    ----------
    vcf_path : str
        Path to input VCF file with AD (allelic depth) field.
    ploidy : int
        Organism ploidy level.
    r_executable : str, optional
        Path to Rscript. If None, searches PATH.
    possiblePloidies : list[int], optional
        Ploidy values to consider. Default: [ploidy].
    min_allele_freq : float
        Minimum allele frequency filter for polyRAD.

    Returns
    -------
    DosageProbabilities
    """
    rscript = _find_rscript(r_executable)
    if possiblePloidies is None:
        possiblePloidies = [ploidy]

    with tempfile.TemporaryDirectory(prefix="polyrad_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        probs_csv = tmpdir_path / "probs.csv"
        expected_csv = tmpdir_path / "expected.csv"
        samples_csv = tmpdir_path / "samples.csv"
        markers_csv = tmpdir_path / "markers.csv"

        ploidies_str = ", ".join(str(p) for p in possiblePloidies)

        r_script = f"""
library(polyRAD)

mydata <- VCF2RADdata("{vcf_path}",
                      possiblePloidies = list({ploidies_str}),
                      min.ind.with.reads = 1)

mydata <- IteratePopStruct(mydata, nPCSinit = 10)

probs <- GetWeightedMeanGenotypes(mydata, maxval = {ploidy})

# Write expected dosages
write.csv(probs, "{expected_csv.as_posix()}", row.names = TRUE)

# Write posterior probabilities (n_samples x n_markers x (ploidy+1))
# Stored as a flattened matrix: rows = sample*marker, cols = dosage classes
posterior <- mydata$posteriorProb
n_samples <- length(mydata$taxaPloidy)
n_markers <- length(mydata$alleles2loc)
n_classes <- {ploidy + 1}

prob_matrix <- matrix(NA, nrow = n_samples * n_markers, ncol = n_classes)
for (i in seq_len(n_markers)) {{
    idx_start <- (i - 1) * n_samples + 1
    idx_end <- i * n_samples
    if (!is.null(posterior[[i]])) {{
        cols_avail <- min(ncol(posterior[[i]]), n_classes)
        prob_matrix[idx_start:idx_end, 1:cols_avail] <- posterior[[i]][, 1:cols_avail]
    }}
}}
write.csv(prob_matrix, "{probs_csv.as_posix()}", row.names = FALSE)

# Write sample and marker IDs
write.csv(data.frame(id = rownames(probs)), "{samples_csv.as_posix()}", row.names = FALSE)
write.csv(data.frame(id = colnames(probs)), "{markers_csv.as_posix()}", row.names = FALSE)
"""
        result = _run_r_script(rscript, r_script, tmpdir_path)
        return _parse_dosage_output(
            probs_csv, expected_csv, samples_csv, markers_csv, ploidy,
        )


# --- Internal helpers ---


def _find_rscript(r_executable: str | None) -> str:
    """Locate Rscript binary."""
    if r_executable is not None:
        return r_executable
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript not found in PATH. polyRAD/updog require R. "
            "Install R from https://cran.r-project.org/"
        )
    return rscript


def _run_r_script(rscript: str, script: str, workdir: Path) -> subprocess.CompletedProcess:
    """Write and execute an R script."""
    script_path = workdir / "run.R"
    script_path.write_text(script)

    logger.info("Running R script: %s", script_path)
    result = subprocess.run(
        [rscript, "--vanilla", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(workdir),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"R script failed (exit {result.returncode}):\n"
            f"STDERR: {result.stderr[:2000]}\n"
            f"STDOUT: {result.stdout[:500]}"
        )

    return result


def _parse_dosage_output(
    probs_csv: Path,
    expected_csv: Path,
    samples_csv: Path,
    markers_csv: Path,
    ploidy: int,
) -> DosageProbabilities:
    """Parse R output CSVs into DosageProbabilities."""
    # Read sample and marker IDs
    sample_ids = _read_id_csv(samples_csv)
    marker_ids = _read_id_csv(markers_csv)
    n_samples = len(sample_ids)
    n_markers = len(marker_ids)

    # Read expected dosages: (n_samples, n_markers)
    expected = _read_matrix_csv(expected_csv, skip_rownames=True)
    if expected.shape != (n_samples, n_markers):
        logger.warning(
            "Expected dosage shape %s != (%d, %d), reshaping",
            expected.shape, n_samples, n_markers,
        )
        expected = expected[:n_samples, :n_markers]

    # Read posterior probabilities: (n_samples * n_markers, ploidy+1)
    probs_flat = _read_matrix_csv(probs_csv, skip_rownames=False)
    probs = probs_flat.reshape(n_markers, n_samples, ploidy + 1).permute(1, 0, 2)
    # Now (n_samples, n_markers, ploidy+1)

    # Replace NaN with uniform prior (maximum entropy for missing)
    nan_mask = torch.isnan(probs).any(dim=-1)
    if nan_mask.any():
        uniform = torch.full((ploidy + 1,), 1.0 / (ploidy + 1), dtype=probs.dtype)
        probs[nan_mask] = uniform
        logger.info(
            "Replaced %d missing probability vectors with uniform prior",
            int(nan_mask.sum().item()),
        )

    # Normalize to sum to 1 (safety)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Recompute expected from probs for consistency
    d_vals = torch.arange(ploidy + 1, dtype=probs.dtype)
    expected_from_probs = (probs * d_vals).sum(dim=-1)

    return DosageProbabilities(
        probs=probs,
        expected=expected_from_probs,
        sample_ids=sample_ids,
        marker_ids=marker_ids,
        ploidy=ploidy,
    )


def _read_id_csv(path: Path) -> list[str]:
    """Read a single-column CSV of IDs."""
    ids = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ids.append(row["id"])
    return ids


def _read_matrix_csv(path: Path, skip_rownames: bool = False) -> Tensor:
    """Read a CSV into a float64 tensor."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)  # skip header
        for row in reader:
            if skip_rownames:
                row = row[1:]  # skip row name column
            vals = []
            for v in row:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(float("nan"))
            rows.append(vals)
    if not rows:
        return torch.zeros(0, 0, dtype=torch.float64)
    return torch.tensor(rows, dtype=torch.float64)
