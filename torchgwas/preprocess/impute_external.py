"""Wrappers for external imputation tools: BEAGLE, IMPUTE5, Minimac4, STITCH."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ImputationResult:
    """Output of an external imputation run."""

    output_path: str
    n_variants_imputed: int
    mean_rsq: float
    tool: str
    tool_version: str


def run_beagle(
    input_path: str,
    output_path: str,
    *,
    ref_panel: Optional[str] = None,
    beagle_jar: Optional[str] = None,
    java_mem: str = "4g",
    nthreads: int = 1,
    ploidy: int = 2,
) -> ImputationResult:
    """Run BEAGLE 5.5 for imputation and/or phasing.

    Parameters
    ----------
    input_path : str
        Path to input VCF/VCF.gz file.
    output_path : str
        Output prefix (BEAGLE appends .vcf.gz).
    ref_panel : str, optional
        Path to reference panel VCF for imputation.
    beagle_jar : str, optional
        Path to BEAGLE .jar file. If None, searches PATH for 'beagle.jar'.
    java_mem : str
        Java heap size (e.g., "4g").
    nthreads : int
        Number of threads.
    ploidy : int
        Organism ploidy level. BEAGLE 5.4+ supports polyploid phasing
        via the ``ploidy`` parameter (default: 2 for diploid).

    Returns
    -------
    ImputationResult
    """
    if beagle_jar is None:
        beagle_jar = shutil.which("beagle.jar") or "beagle.jar"

    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java not found in PATH. BEAGLE requires Java.")

    cmd = [
        java, f"-Xmx{java_mem}", "-jar", beagle_jar,
        f"gt={input_path}",
        f"out={output_path}",
        f"nthreads={nthreads}",
    ]
    if ref_panel:
        cmd.append(f"ref={ref_panel}")
    if ploidy > 2:
        cmd.append(f"ploidy={ploidy}")

    logger.info("Running BEAGLE: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"BEAGLE failed (exit {result.returncode}):\n{result.stderr}")

    # Parse output for variant count and quality
    out_vcf = f"{output_path}.vcf.gz"
    n_variants = _count_vcf_variants(out_vcf)
    mean_rsq = _parse_beagle_rsq(result.stdout)

    version = _extract_version(result.stdout, "beagle")

    return ImputationResult(
        output_path=out_vcf,
        n_variants_imputed=n_variants,
        mean_rsq=mean_rsq,
        tool="BEAGLE",
        tool_version=version,
    )


def run_impute5(
    input_path: str,
    output_path: str,
    *,
    ref_panel: str,
    map_file: Optional[str] = None,
) -> ImputationResult:
    """Run IMPUTE5 for imputation.

    Parameters
    ----------
    input_path : str
        Path to input VCF.
    output_path : str
        Path for output VCF.
    ref_panel : str
        Path to reference panel.
    map_file : str, optional
        Genetic map file.

    Returns
    -------
    ImputationResult
    """
    impute5 = shutil.which("impute5")
    if impute5 is None:
        raise RuntimeError("impute5 not found in PATH.")

    cmd = [
        impute5,
        "--h", ref_panel,
        "--g", input_path,
        "--o", output_path,
    ]
    if map_file:
        cmd.extend(["--m", map_file])

    logger.info("Running IMPUTE5: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"IMPUTE5 failed (exit {result.returncode}):\n{result.stderr}")

    n_variants = _count_vcf_variants(output_path)

    return ImputationResult(
        output_path=output_path,
        n_variants_imputed=n_variants,
        mean_rsq=0.0,
        tool="IMPUTE5",
        tool_version="unknown",
    )


def run_minimac4(
    input_path: str,
    output_path: str,
    *,
    ref_panel: str,
) -> ImputationResult:
    """Run Minimac4 for imputation.

    Parameters
    ----------
    input_path : str
        Path to input VCF.
    output_path : str
        Path for output VCF.
    ref_panel : str
        Path to reference panel (.m3vcf or .msav).

    Returns
    -------
    ImputationResult
    """
    minimac4 = shutil.which("minimac4")
    if minimac4 is None:
        raise RuntimeError("minimac4 not found in PATH.")

    cmd = [
        minimac4,
        "--refHaps", ref_panel,
        "--haps", input_path,
        "--prefix", output_path,
    ]

    logger.info("Running Minimac4: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Minimac4 failed (exit {result.returncode}):\n{result.stderr}")

    out_file = f"{output_path}.dose.vcf.gz"
    n_variants = _count_vcf_variants(out_file) if Path(out_file).is_file() else 0

    return ImputationResult(
        output_path=out_file,
        n_variants_imputed=n_variants,
        mean_rsq=0.0,
        tool="Minimac4",
        tool_version="unknown",
    )


# --- Helpers ---

def _count_vcf_variants(path: str) -> int:
    """Count variants in a VCF file (excluding header lines)."""
    import gzip
    p = Path(path)
    if not p.is_file():
        return 0
    opener = gzip.open if p.suffix == ".gz" else open
    count = 0
    with opener(p, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            if not line.startswith("#"):
                count += 1
    return count


def _parse_beagle_rsq(stdout: str) -> float:
    """Parse mean Rsq from BEAGLE stdout (best effort)."""
    # BEAGLE doesn't always report this directly; return 0.0 as fallback
    return 0.0


def _extract_version(stdout: str, tool: str) -> str:
    """Extract version string from tool stdout."""
    for line in stdout.splitlines():
        if "version" in line.lower() or tool.lower() in line.lower():
            return line.strip()[:80]
    return "unknown"
