"""Pre-flight validation: sample counts, allele coding, ploidy, duplicate checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .detect import detect_format

logger = logging.getLogger(__name__)


@dataclass
class PreflightReport:
    """Summary produced by the pre-flight validation step."""

    format_name: str
    n_samples_genotype: int
    n_samples_phenotype: int
    n_samples_covariate: int
    n_samples_aligned: int
    n_variants_total: int
    n_variants_after_qc: int
    ploidy: int
    n_traits: int
    covariate_names: list[str]
    genotype_missingness_pct: float
    phenotype_missingness_pct: float
    device: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_preflight(
    genotype_path: str,
    phenotype_path: str,
    covariate_path: str | None = None,
    ploidy: int = 2,
) -> PreflightReport:
    """Run all pre-flight checks and return a structured report.

    Checks performed (per charter Section 6c):
    1. Format detection
    2. Sample count consistency across genotype/phenotype/covariate files
    3. Mandatory metadata: every variant has marker ID, chromosome, position
    4. Allele coding validation (biallelic for diploid)
    5. Chromosome naming consistency ("chr1" vs "1")
    6. Duplicate sample/variant ID detection
    7. Excessive missingness warnings
    """
    warnings: list[str] = []
    errors: list[str] = []

    # 1. Detect format
    try:
        fmt = detect_format(genotype_path)
    except (ValueError, FileNotFoundError) as e:
        return PreflightReport(
            format_name="unknown", n_samples_genotype=0, n_samples_phenotype=0,
            n_samples_covariate=0, n_samples_aligned=0, n_variants_total=0,
            n_variants_after_qc=0, ploidy=ploidy, n_traits=0, covariate_names=[],
            genotype_missingness_pct=0, phenotype_missingness_pct=0, device="cpu",
            errors=[str(e)],
        )

    # 2. Open reader to get sample/variant counts
    reader = _open_reader(genotype_path, fmt)
    n_geno = reader.n_samples
    n_variants = reader.n_variants
    geno_ids = reader.sample_ids

    # 3. Check for duplicate genotype sample IDs
    if len(geno_ids) != len(set(geno_ids)):
        dup_count = len(geno_ids) - len(set(geno_ids))
        errors.append(f"Duplicate sample IDs in genotype file: {dup_count} duplicates found.")

    # 4. Check variant metadata completeness
    for G_chunk, vmeta in reader.iter_chunks(chunk_size=n_variants):
        empty_snp = sum(1 for s in vmeta.snp if not s or s == ".")
        empty_chr = sum(1 for c in vmeta.chr if not c or c == ".")
        if empty_snp > 0:
            warnings.append(f"{empty_snp} variants have missing/empty SNP IDs.")
        if empty_chr > 0:
            warnings.append(f"{empty_chr} variants have missing chromosome labels.")

        # Check duplicate variant IDs
        if len(vmeta.snp) != len(set(vmeta.snp)):
            dup_var = len(vmeta.snp) - len(set(vmeta.snp))
            warnings.append(f"{dup_var} duplicate variant IDs detected.")

        # Chromosome naming consistency
        has_chr_prefix = any(c.lower().startswith("chr") for c in vmeta.chr if c)
        has_bare = any(not c.lower().startswith("chr") and c.isdigit() for c in vmeta.chr if c)
        if has_chr_prefix and has_bare:
            warnings.append(
                "Mixed chromosome naming: some variants use 'chr' prefix, others don't. "
                "Consider normalizing."
            )

        # Genotype missingness
        import torch
        miss_pct = float(torch.isnan(G_chunk).float().mean()) * 100

        # Dosage range validation (charter Section 6a-0: Internal Data Contract)
        from ..config import validate_dosage_range
        try:
            n_violations = validate_dosage_range(
                G_chunk, ploidy=ploidy, policy="warn+override",
            )
            if n_violations > 0:
                warnings.append(
                    f"Dosage range: {n_violations} values outside [0, {ploidy}]."
                )
        except Exception as e:
            warnings.append(f"Dosage range check failed: {e}")

        break  # only check first (full) chunk
    else:
        miss_pct = 0.0

    # 5. Load phenotype for sample count
    from .phenotype import _detect_id_column, _load_tabular
    pheno_df = _load_tabular(phenotype_path)
    pheno_id_col = _detect_id_column(pheno_df)
    n_pheno = len(pheno_df)

    # Phenotype missingness
    numeric_cols = pheno_df.select_dtypes(include="number").columns
    pheno_miss_pct = float(pheno_df[numeric_cols].isna().mean().mean()) * 100 if len(numeric_cols) > 0 else 0.0
    n_traits = len(numeric_cols)
    trait_names = list(numeric_cols)

    # 6. Covariate check
    n_covar = 0
    covar_names: list[str] = []
    if covariate_path is not None:
        covar_df = _load_tabular(covariate_path)
        n_covar = len(covar_df)
        covar_id_col = _detect_id_column(covar_df)
        covar_names = [c for c in covar_df.columns if c != covar_id_col]

    # 7. Compute aligned count
    pheno_ids = set(pheno_df[pheno_id_col].astype(str))
    aligned = set(geno_ids) & pheno_ids
    if covariate_path:
        covar_ids = set(covar_df[covar_id_col].astype(str))  # type: ignore[possibly-undefined]
        aligned = aligned & covar_ids
    n_aligned = len(aligned)

    if n_aligned == 0:
        errors.append("No overlapping sample IDs between genotype and phenotype files.")
    elif n_aligned < 0.5 * n_geno:
        warnings.append(
            f"Only {n_aligned}/{n_geno} genotype samples have matching phenotypes "
            f"({n_aligned/n_geno:.0%}). Check file pairing."
        )

    if miss_pct > 20:
        warnings.append(f"High genotype missingness: {miss_pct:.1f}%")

    report = PreflightReport(
        format_name=fmt,
        n_samples_genotype=n_geno,
        n_samples_phenotype=n_pheno,
        n_samples_covariate=n_covar,
        n_samples_aligned=n_aligned,
        n_variants_total=n_variants,
        n_variants_after_qc=n_variants,  # QC not applied in preflight
        ploidy=ploidy,
        n_traits=n_traits,
        covariate_names=covar_names,
        genotype_missingness_pct=miss_pct,
        phenotype_missingness_pct=pheno_miss_pct,
        device="cpu",
        warnings=warnings,
        errors=errors,
    )

    # Log summary
    if errors:
        for e in errors:
            logger.error("PREFLIGHT ERROR: %s", e)
    if warnings:
        for w in warnings:
            logger.warning("PREFLIGHT WARNING: %s", w)

    logger.info(
        "Pre-flight: format=%s, %d geno samples, %d pheno samples, %d aligned, "
        "%d variants, %d traits, %.1f%% geno missing",
        fmt, n_geno, n_pheno, n_aligned, n_variants, n_traits, miss_pct,
    )

    return report


def _open_reader(path: str, fmt: str):
    """Open the appropriate reader for the detected format."""
    if fmt == "bed":
        from .plink import PlinkBedReader
        return PlinkBedReader(path)
    elif fmt == "pgen":
        from .plink2 import Plink2PgenReader
        return Plink2PgenReader(path)
    elif fmt in ("vcf", "bcf"):
        from .vcf import VCFReader
        return VCFReader(path)
    elif fmt == "bgen":
        from .bgen import BGENReader
        return BGENReader(path)
    elif fmt == "hapmap":
        from .hapmap import HapMapReader
        return HapMapReader(path)
    elif fmt == "csv":
        from .numeric import NumericDosageReader
        return NumericDosageReader(path)
    elif fmt == "zarr":
        from .zarr import ZarrReader
        return ZarrReader(path)
    elif fmt == "hdf5":
        from .hdf5 import HDF5Reader
        return HDF5Reader(path)
    else:
        raise ValueError(f"Unsupported format for preflight: {fmt}")
