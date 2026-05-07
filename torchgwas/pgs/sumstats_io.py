"""PGS-specific sumstats ingestion and reference harmonization.

Thin wrappers over :mod:`torchgwas.postgwas._sumstats` that add the
column aliasing and unit conversions PGS pipelines typically need:

- ``BETA`` / ``OR`` (with automatic log conversion)
- ``EAF`` / ``FRQ`` / ``A1_FREQ`` for allele frequency
- ``N_EFF`` / ``NEFFDIV2`` for effective sample size
- Free-form ``CHR``/``BP``/``SNP``/``A1``/``A2``/``SE``/``P`` casing
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..postgwas._sumstats import SumStats
from .base import BasePGSMethod, LDReference

# Column aliases — keys are normalized lowercase target field, values are
# acceptable input column names (also matched case-insensitively).
_ALIASES: dict[str, tuple[str, ...]] = {
    "chr": ("chr", "chrom", "chromosome", "#chrom", "#chr"),
    "pos": ("pos", "bp", "position", "base_pair_location"),
    "snp": ("snp", "rsid", "rs_id", "id", "marker", "variant_id"),
    "a1": ("a1", "effect_allele", "allele1", "ea"),
    "a2": ("a2", "other_allele", "allele2", "nea"),
    "beta": ("beta", "effect", "log_or", "logodds"),
    "or": ("or", "odds_ratio", "oddsratio"),
    "se": ("se", "stderr", "standard_error", "se_beta"),
    "p": ("p", "pval", "pvalue", "p_value", "p.value"),
    "n": ("n", "neff", "n_eff", "n_effective", "neffdiv2", "samplesize"),
    "af": ("af", "eaf", "frq", "freq", "a1_freq", "maf", "effect_allele_frequency"),
}


def _resolve(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    """Return the actual column name from ``columns`` matching any alias."""
    lower_map = {c.lower(): c for c in columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


def load_pgs_sumstats(
    path: str,
    sep: str = "\t",
    device: torch.device | str = "cpu",
    or_to_log: bool = True,
    n_eff_doubled: bool = False,
) -> SumStats:
    """Load summary statistics with PGS-friendly column auto-detection.

    Parameters
    ----------
    path : str
        Path to a TSV/CSV summary stats file.
    sep : str
        Field delimiter.
    device : torch.device or str
        Output tensor device.
    or_to_log : bool
        If the file provides odds ratios (``OR``) but no ``BETA``, take
        ``log(OR)`` to obtain effects on the log-odds scale.
    n_eff_doubled : bool
        If True, the input ``N`` column is interpreted as ``2 * N_eff``
        (the ``NEFFDIV2`` convention used by some pipelines) and divided
        by 2 on load. Default False.

    Returns
    -------
    SumStats
    """
    import pandas as pd

    df = pd.read_csv(path, sep=sep)
    cols = list(df.columns)

    chr_c = _resolve(cols, _ALIASES["chr"])
    pos_c = _resolve(cols, _ALIASES["pos"])
    snp_c = _resolve(cols, _ALIASES["snp"])
    a1_c = _resolve(cols, _ALIASES["a1"])
    a2_c = _resolve(cols, _ALIASES["a2"])
    se_c = _resolve(cols, _ALIASES["se"])
    p_c = _resolve(cols, _ALIASES["p"])
    n_c = _resolve(cols, _ALIASES["n"])
    af_c = _resolve(cols, _ALIASES["af"])

    missing = [
        name
        for name, val in [
            ("chr", chr_c),
            ("pos", pos_c),
            ("snp", snp_c),
            ("a1", a1_c),
            ("a2", a2_c),
            ("se", se_c),
            ("p", p_c),
        ]
        if val is None
    ]
    if missing:
        raise ValueError(
            f"Could not auto-detect required columns {missing} in {path}; "
            f"present columns: {cols}"
        )

    beta_c = _resolve(cols, _ALIASES["beta"])
    or_c = _resolve(cols, _ALIASES["or"])
    if beta_c is not None:
        beta_vals = df[beta_c].astype(float).values
    elif or_c is not None and or_to_log:
        beta_vals = [math.log(float(v)) for v in df[or_c].values]
    else:
        raise ValueError(
            "Sumstats file must provide either BETA or OR (with or_to_log=True). "
            f"Columns: {cols}"
        )

    chr_list = df[chr_c].astype(str).tolist()
    pos_list = df[pos_c].astype(int).tolist()
    snp_list = df[snp_c].astype(str).tolist()
    a1_list = df[a1_c].astype(str).str.upper().tolist()
    a2_list = df[a2_c].astype(str).str.upper().tolist()

    beta = torch.tensor(beta_vals, dtype=torch.float64, device=device)
    se = torch.tensor(df[se_c].astype(float).values, dtype=torch.float64, device=device)
    p = torch.tensor(df[p_c].astype(float).values, dtype=torch.float64, device=device)

    if n_c is not None:
        n_arr = df[n_c].astype(float).values
        if n_eff_doubled:
            n_arr = n_arr / 2.0
        n = torch.tensor(n_arr, dtype=torch.float64, device=device)
    else:
        n = torch.full((len(df),), float("nan"), dtype=torch.float64, device=device)

    af: Tensor | None = None
    if af_c is not None:
        af = torch.tensor(
            df[af_c].astype(float).values, dtype=torch.float64, device=device
        )

    return SumStats(
        chr=chr_list,
        pos=pos_list,
        snp=snp_list,
        a1=a1_list,
        a2=a2_list,
        beta=beta,
        se=se,
        p=p,
        n=n,
        af=af,
    )


class _Harmonizer(BasePGSMethod):
    """Concrete BasePGSMethod used only to expose ``_harmonize``."""

    name = "harmonize"

    def fit(self, sumstats, ld_ref, **kwargs):  # pragma: no cover
        raise TypeError("Harmonizer exposes allele harmonization only; use harmonize_to_reference().")


def harmonize_to_reference(
    sumstats: SumStats,
    ld_ref: LDReference,
    maf_tol: float = 0.20,
    drop_ambiguous: bool = True,
    palindromic_maf_tol: float = 0.42,
) -> tuple[SumStats, LDReference, dict[str, int]]:
    """Functional wrapper around :meth:`BasePGSMethod._harmonize`.

    Returns ``(harmonized_sumstats, harmonized_ld_ref, audit_dict)``.
    """
    h = _Harmonizer()
    return h._harmonize(
        sumstats,
        ld_ref,
        maf_tol=maf_tol,
        drop_ambiguous=drop_ambiguous,
        palindromic_maf_tol=palindromic_maf_tol,
    )
