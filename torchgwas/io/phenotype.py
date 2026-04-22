"""Phenotype and covariate loaders with deterministic sample alignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class AlignmentManifest:
    """Records which samples were included/excluded and why."""

    iid: list[str]
    in_genotype: list[bool]
    in_phenotype: list[bool]
    in_covariate: list[bool]
    included: list[bool]
    drop_reason: list[str]  # "PASS", "missing_phenotype", "genotype_only", etc.


@dataclass
class PhenotypeData:
    """Aligned phenotype + covariate data ready for model fitting."""

    Y: Tensor  # (n, d) phenotype matrix
    X0: Tensor  # (n, c) covariate matrix (intercept as first column)
    sample_ids: list[str]  # aligned, lexicographically sorted
    trait_names: list[str]
    covariate_names: list[str]
    manifest: AlignmentManifest


def load_phenotype(
    phenotype_path: str | Path,
    genotype_sample_ids: list[str],
    *,
    covariate_path: str | Path | None = None,
    trait_columns: list[str] | None = None,
    id_column: str | None = None,
) -> PhenotypeData:
    """Load phenotype/covariates and align to genotype samples.

    Implements the deterministic 9-point alignment protocol from charter Section 6b:

    1. Load all ID sets explicitly
    2. Compute three-way intersection (exact string matching)
    3. Sort lexicographically
    4. Reindex all matrices to aligned order
    5. Drop samples with missing phenotype
    6. Hard errors on critical mismatches
    7. Structured warnings for non-critical drops
    8. Report in pre-flight summary
    9. Write alignment manifest
    """
    # --- 1. Load phenotype ---
    pheno_df = _load_tabular(phenotype_path)

    if id_column is not None:
        # User explicitly specified the ID column
        if id_column not in pheno_df.columns:
            raise ValueError(
                f"Specified ID column '{id_column}' not found in phenotype file. "
                f"Available columns: {list(pheno_df.columns)}"
            )
        pheno_id_col = id_column
    else:
        pheno_id_col = _detect_id_column(pheno_df)

    pheno_df = pheno_df.set_index(pheno_id_col)
    pheno_ids = set(pheno_df.index.astype(str))

    # Determine trait columns
    if trait_columns is not None:
        missing_traits = [t for t in trait_columns if t not in pheno_df.columns]
        if missing_traits:
            raise ValueError(
                f"Trait columns not found in phenotype file: {missing_traits}. "
                f"Available columns: {list(pheno_df.columns)}"
            )
        trait_names = trait_columns
    else:
        # Use all numeric columns as traits (exclude the ID column)
        trait_names = [c for c in pheno_df.columns if _is_numeric_column(pheno_df[c])]
        if not trait_names:
            raise ValueError(
                "No numeric trait columns found in phenotype file. "
                "Specify trait columns with --traits."
            )

    # --- 1b. Load covariates ---
    covar_df: pd.DataFrame | None = None
    covar_ids: set[str]
    covar_names: list[str] = []

    if covariate_path is not None:
        covar_df = _load_tabular(covariate_path)
        if id_column is not None and id_column in covar_df.columns:
            covar_id_col = id_column
        else:
            covar_id_col = _detect_id_column(covar_df)
        covar_df = covar_df.set_index(covar_id_col)
        covar_ids = set(covar_df.index.astype(str))
        covar_names = list(covar_df.columns)
    else:
        covar_ids = pheno_ids  # no covariate constraint

    geno_ids = set(str(s) for s in genotype_sample_ids)

    # --- 2. Three-way intersection (exact string matching) ---
    aligned_ids = geno_ids & pheno_ids & covar_ids

    # --- 6. Hard errors on critical mismatches ---
    if len(geno_ids & pheno_ids) == 0:
        raise ValueError(
            "Empty intersection between genotype and phenotype sample IDs. "
            "No shared sample IDs found — check that file IDs match. "
            f"Genotype IDs (first 5): {sorted(geno_ids)[:5]}, "
            f"Phenotype IDs (first 5): {sorted(pheno_ids)[:5]}"
        )

    if len(aligned_ids) == 0:
        raise ValueError(
            "Empty three-way intersection (genotype ∩ phenotype ∩ covariate). "
            "Check that all files use matching sample IDs."
        )

    if covariate_path is not None:
        covar_only = covar_ids - (geno_ids & pheno_ids)
        if covar_only:
            raise ValueError(
                f"Covariate file contains {len(covar_only)} IDs not in "
                f"genotype ∩ phenotype set — likely a file mix-up. "
                f"Examples: {sorted(covar_only)[:5]}"
            )

    # Check >50% drop
    drop_frac = 1.0 - len(aligned_ids) / len(geno_ids)
    if drop_frac > 0.5:
        raise ValueError(
            f"Alignment would drop {drop_frac:.0%} of genotype samples "
            f"({len(geno_ids)} genotype, {len(aligned_ids)} aligned). "
            "This likely indicates wrong file pairing."
        )

    # --- 3. Sort deterministically (lexicographic) ---
    aligned_sorted = sorted(aligned_ids)

    # --- 7. Structured warnings ---
    n_geno_only = len(geno_ids - aligned_ids)
    n_pheno_only = len(pheno_ids - aligned_ids)
    n_covar_only = len(covar_ids - aligned_ids) if covariate_path else 0
    n_dropped = n_geno_only + n_pheno_only + n_covar_only

    if 0.1 <= drop_frac <= 0.5:
        logger.warning(
            "Sample alignment: %d genotype-only, %d phenotype-only, %d covariate-only "
            "samples excluded (%d aligned of %d genotype).",
            n_geno_only, n_pheno_only, n_covar_only,
            len(aligned_sorted), len(geno_ids),
        )
    elif n_dropped > 0:
        logger.info(
            "Sample alignment: %d samples excluded (%d aligned of %d genotype).",
            n_dropped, len(aligned_sorted), len(geno_ids),
        )

    # --- 4. Reindex to aligned order ---
    pheno_aligned = pheno_df.loc[aligned_sorted, trait_names]

    if covar_df is not None:
        covar_aligned = covar_df.loc[aligned_sorted]
    else:
        covar_aligned = None

    # --- 5. Drop samples with missing phenotype ---
    # For multi-trait: drop only if ALL traits are missing (GEMMA convention)
    all_missing_mask = pheno_aligned.isna().all(axis=1)
    if all_missing_mask.any():
        n_pheno_missing = all_missing_mask.sum()
        logger.info(
            "Dropping %d samples with all phenotype values missing.", n_pheno_missing,
        )
        pheno_aligned = pheno_aligned[~all_missing_mask]
        aligned_sorted = pheno_aligned.index.tolist()
        if covar_aligned is not None:
            covar_aligned = covar_aligned.loc[aligned_sorted]

    if len(aligned_sorted) == 0:
        raise ValueError("No samples remain after dropping missing phenotypes.")

    # --- Build tensors ---
    Y = torch.tensor(pheno_aligned.values, dtype=torch.float64)
    if Y.ndim == 1:
        Y = Y.unsqueeze(1)

    # Covariate matrix with intercept
    n = len(aligned_sorted)
    intercept = np.ones((n, 1), dtype=np.float64)

    if covar_aligned is not None:
        # Auto-encode categorical covariates as dummies
        covar_numeric = pd.get_dummies(covar_aligned, drop_first=True, dtype=float)
        covar_names = ["intercept"] + list(covar_numeric.columns)
        X0_np = np.hstack([intercept, covar_numeric.values.astype(np.float64)])
    else:
        covar_names = ["intercept"]
        X0_np = intercept

    X0 = torch.tensor(X0_np, dtype=torch.float64)

    # --- 9. Build alignment manifest ---
    all_ids = sorted(geno_ids | pheno_ids | (covar_ids if covariate_path else set()))
    manifest = AlignmentManifest(
        iid=all_ids,
        in_genotype=[s in geno_ids for s in all_ids],
        in_phenotype=[s in pheno_ids for s in all_ids],
        in_covariate=[s in covar_ids for s in all_ids] if covariate_path else [True] * len(all_ids),
        included=[s in set(aligned_sorted) for s in all_ids],
        drop_reason=[
            "PASS" if s in set(aligned_sorted)
            else "missing_phenotype" if s not in pheno_ids
            else "phenotype_only" if s not in geno_ids
            else "covariate_only" if covariate_path and s not in covar_ids
            else "all_traits_missing"
            for s in all_ids
        ],
    )

    # --- 8. Summary ---
    logger.info(
        "Alignment complete: %d samples, %d traits, %d covariates (incl. intercept).",
        n, len(trait_names), X0.shape[1],
    )

    return PhenotypeData(
        Y=Y,
        X0=X0,
        sample_ids=aligned_sorted,
        trait_names=trait_names,
        covariate_names=covar_names,
        manifest=manifest,
    )


def write_alignment_manifest(manifest: AlignmentManifest, path: str | Path) -> None:
    """Write alignment manifest to TSV for post-hoc auditing."""
    df = pd.DataFrame({
        "IID": manifest.iid,
        "in_genotype": manifest.in_genotype,
        "in_phenotype": manifest.in_phenotype,
        "in_covariate": manifest.in_covariate,
        "included": manifest.included,
        "drop_reason": manifest.drop_reason,
    })
    df.to_csv(path, sep="\t", index=False)
    logger.info("Alignment manifest written to %s", path)


# --- Helpers ---

def _load_tabular(path: str | Path) -> pd.DataFrame:
    """Load a tabular file (CSV/TSV/TXT/Excel) with auto-detected delimiter."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p)

    # Text formats
    with open(p) as _fh:
        first_line = _fh.readline()
    if "\t" in first_line:
        sep = "\t"
    elif "," in first_line:
        sep = ","
    else:
        sep = r"\s+"

    return pd.read_csv(p, sep=sep)


def _detect_id_column(df: pd.DataFrame) -> str:
    """Detect the sample ID column by name heuristic."""
    candidates = {"iid", "id", "sample", "sample_id", "sampleid", "fid_iid",
                  "taxa", "ind", "individual", "genotype", "geno", "geno_id",
                  "genotype_id", "line", "accession", "entry", "cultivar",
                  "variety"}
    for col in df.columns:
        if col.lower().strip() in candidates:
            return col

    # If first column is non-numeric string, assume it's the ID column
    first_col = df.columns[0]
    if not _is_numeric_column(df[first_col]):
        return first_col

    raise ValueError(
        f"Cannot detect sample ID column. Columns: {list(df.columns)}. "
        "Ensure the first column contains sample IDs, or name it 'IID' or 'Sample'."
    )


def _is_numeric_column(series: pd.Series) -> bool:
    """Check if a pandas Series is (mostly) numeric."""
    try:
        pd.to_numeric(series, errors="raise")
        return True
    except (ValueError, TypeError):
        return False
