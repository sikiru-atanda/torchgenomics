"""Phasing wrapper (BEAGLE) and haplotype tensor storage."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def phase_beagle(
    input_path: str,
    output_path: str,
    *,
    beagle_jar: str | None = None,
    java_mem: str = "4g",
    nthreads: int = 1,
    ploidy: int = 2,
) -> str:
    """Run BEAGLE phasing. Returns path to phased output VCF.

    Parameters
    ----------
    input_path : str
        Input VCF file.
    output_path : str
        Output prefix (BEAGLE appends .vcf.gz).
    beagle_jar : str, optional
        Path to BEAGLE .jar file.
    java_mem : str
        Java heap size.
    nthreads : int
        Number of threads.
    ploidy : int
        Organism ploidy level. BEAGLE 5.4+ supports polyploid phasing
        via the ``ploidy`` parameter (default: 2 for diploid).

    Returns
    -------
    str
        Path to phased output VCF (.vcf.gz).
    """
    if beagle_jar is None:
        beagle_jar = shutil.which("beagle.jar") or "beagle.jar"

    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java not found in PATH. BEAGLE phasing requires Java.")

    cmd = [
        java, f"-Xmx{java_mem}", "-jar", beagle_jar,
        f"gt={input_path}",
        f"out={output_path}",
        f"nthreads={nthreads}",
        "impute=false",  # phase only, no imputation
    ]
    if ploidy > 2:
        cmd.append(f"ploidy={ploidy}")

    logger.info("Running BEAGLE phasing: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"BEAGLE phasing failed (exit {result.returncode}):\n{result.stderr}")

    out_vcf = f"{output_path}.vcf.gz"
    if not Path(out_vcf).is_file():
        raise FileNotFoundError(f"BEAGLE did not produce expected output: {out_vcf}")

    logger.info("Phased output: %s", out_vcf)
    return out_vcf


def load_haplotypes(phased_vcf_path: str, ploidy: int = 2) -> Tensor:
    """Load phased haplotypes as (n, ploidy, p) tensor.

    Reads phased GT field from VCF and returns a tensor where dimension 1
    indexes the haplotypes. For diploid (ploidy=2), GT is ``0|1``.
    For tetraploid (ploidy=4), GT is ``0|1|0|1``.

    Parameters
    ----------
    phased_vcf_path : str
        Path to phased VCF file.
    ploidy : int
        Organism ploidy level (number of haplotypes per sample).

    Returns
    -------
    Tensor, shape (n_samples, ploidy, n_variants)
        Haplotype tensor. Values are 0 or 1.
    """
    try:
        import cyvcf2  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "load_haplotypes requires cyvcf2: pip install cyvcf2"
        )

    vcf = cyvcf2.VCF(phased_vcf_path)
    n_samples = len(vcf.samples)

    # Collect per-haplotype alleles for each variant
    hap_lists = [[] for _ in range(ploidy)]

    for variant in vcf:
        gt = variant.genotype.array()  # (n_samples, ploidy + 1) for cyvcf2
        # cyvcf2 returns array with columns [allele1, allele2, ..., phased_flag]
        for h in range(min(ploidy, gt.shape[1] - 1)):
            alleles = torch.tensor(gt[:, h], dtype=torch.float64)
            alleles[alleles < 0] = float("nan")  # missing
            hap_lists[h].append(alleles)

        # If VCF has fewer alleles than ploidy, pad with NaN
        for h in range(gt.shape[1] - 1, ploidy):
            hap_lists[h].append(torch.full((n_samples,), float("nan"), dtype=torch.float64))

    vcf.close()

    if not hap_lists[0]:
        return torch.zeros(n_samples, ploidy, 0, dtype=torch.float64)

    haps = []
    for h in range(ploidy):
        haps.append(torch.stack(hap_lists[h], dim=1))  # (n, p)

    return torch.stack(haps, dim=1)  # (n, ploidy, p)
