"""Variant QC: HWE test (diploid + polyploid), call rate, het excess, Parquet output."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ..models.base import VariantMeta
from .standardize import compute_allele_frequencies, compute_maf

logger = logging.getLogger(__name__)


@dataclass
class VariantQCStats:
    """Per-variant QC statistics.  Written to Parquet for QA auditing."""

    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    n_obs: Tensor  # (m,) int
    n_miss: Tensor  # (m,) int
    miss_rate: Tensor  # (m,) float
    af: Tensor  # (m,) allele frequency
    maf: Tensor  # (m,) minor allele frequency
    mac: Tensor  # (m,) minor allele count = P * N_nonmiss * MAF
    mono: list[bool]  # (m,) monomorphic flag (MAF==0 or variance ~0)
    het: Tensor  # (m,) observed heterozygosity (non-homozygote rate for polyploid)
    het_exp: Tensor  # (m,) expected heterozygosity under HWE
    het_excess: Tensor  # (m,) excess heterozygosity
    hwe_p: Tensor  # (m,) HWE p-value
    imputed: list[bool]  # (m,)
    imp_rsq: Tensor  # (m,) imputation Rsq (NaN if genotyped)
    filter_pass: list[bool]  # (m,)
    filter_reason: list[str]  # (m,) e.g. "PASS", "MAF<0.01", "MISS>0.10"
    # Polyploid-specific fields (populated only when ploidy > 2)
    dosage_class_freq: Optional[Tensor] = None  # (m, k+1) per-dosage-class freq
    het_per_class: Optional[Tensor] = None  # (m, k-1) per-het-class freq (simplex..k-1-plex)
    double_reduction_alpha: Optional[Tensor] = None  # (m,) estimated DR param
    hwe_p_dr: Optional[Tensor] = None  # (m,) HWE p-value accounting for double reduction
    mean_dosage_var: Optional[Tensor] = None  # (m,) mean Var[d] from dosage probs


@dataclass
class QCFilterConfig:
    """CLI-configurable QC filter thresholds with GEMMA/GAPIT-matching defaults."""

    maf_min: float = 0.01
    mac_min: Optional[int] = None  # off by default; e.g., 20 for rare-variant filtering
    miss_max: float = 0.10
    hwe_p_min: float = 1e-6
    het_excess_max: Optional[float] = None  # off by default
    imp_rsq_min: float = 0.3
    max_geno_freq: Optional[float] = None  # e.g. 1 - 5/N for polyploid gene-action models
    dosage_var_max: Optional[float] = None  # max mean dosage variance (dosage certainty filter)
    use_double_reduction_hwe: bool = False  # use DR-aware HWE test for autopolyploids


def compute_variant_qc(
    G: Tensor,
    variant_meta: VariantMeta,
    ploidy: int = 2,
    imputed_mask: Optional[Tensor] = None,
    imp_rsq: Optional[Tensor] = None,
    dosage_probs: Optional[Tensor] = None,
) -> VariantQCStats:
    """Compute per-variant QC statistics for the full genotype matrix.

    Parameters
    ----------
    G : Tensor, shape (n, m)
        Dosage matrix (may contain NaN).
    variant_meta : VariantMeta
        Per-variant annotation.
    ploidy : int
        Organism ploidy level.
    imputed_mask : Tensor, shape (m,), optional
        Boolean mask indicating which variants were imputed.
    imp_rsq : Tensor, shape (m,), optional
        Imputation quality (Rsq/DR2) per variant.
    dosage_probs : Tensor, shape (n, m, ploidy+1), optional
        Posterior dosage probabilities from polyRAD/updog. Used to compute
        per-marker mean dosage variance (dosage certainty metric).

    Returns
    -------
    VariantQCStats
    """
    n, m = G.shape
    nan_mask = torch.isnan(G)

    # Missingness
    n_miss = nan_mask.sum(dim=0)  # (m,)
    n_obs = n - n_miss  # (m,)
    miss_rate = n_miss.to(torch.float64) / n

    # Allele frequency and MAF
    af = compute_allele_frequencies(G, ploidy=ploidy)
    maf = compute_maf(af)

    # Minor allele count: MAC = P * N_nonmiss * MAF
    mac = ploidy * n_obs.to(torch.float64) * maf

    # Monomorphic flag: MAF==0 or numeric variance ~0
    G_zero = torch.where(nan_mask, torch.zeros_like(G), G)
    n_obs_f = n_obs.to(torch.float64).clamp(min=1.0)
    col_mean = G_zero.sum(dim=0) / n_obs_f
    col_var = ((torch.where(nan_mask, col_mean.unsqueeze(0).expand_as(G), G) - col_mean) ** 2).sum(dim=0) / n_obs_f
    mono = ((maf == 0) | (col_var < 1e-10)).tolist()

    # Observed heterozygosity
    het = _compute_observed_het(G, ploidy=ploidy)

    # Expected heterozygosity under HWE
    het_exp = _compute_expected_het(af, ploidy=ploidy)

    # Excess heterozygosity
    het_excess = torch.where(
        het_exp > 1e-10,
        (het - het_exp) / het_exp,
        torch.zeros_like(het),
    )

    # HWE p-value
    hwe_p = _compute_hwe_pvalue(G, af, ploidy=ploidy)

    # Imputation info
    if imputed_mask is None:
        imputed_list = [False] * m
    else:
        imputed_list = imputed_mask.tolist()

    if imp_rsq is None:
        imp_rsq_t = torch.full((m,), float("nan"), dtype=torch.float64)
    else:
        imp_rsq_t = imp_rsq.to(torch.float64)

    # --- Polyploid-specific QC fields ---
    dosage_class_freq = None
    het_per_class = None
    double_reduction_alpha = None
    hwe_p_dr = None
    mean_dosage_var = None

    if ploidy > 2:
        dosage_class_freq = _compute_dosage_class_freq(G, ploidy)
        het_per_class = _compute_het_per_class(G, ploidy)
        double_reduction_alpha, hwe_p_dr = _compute_hwe_double_reduction(G, af, ploidy)

    if dosage_probs is not None:
        from .dosage_uncertainty import dosage_variance
        var_d = dosage_variance(dosage_probs, ploidy)  # (n, m)
        mean_dosage_var = var_d.mean(dim=0)  # (m,)

    return VariantQCStats(
        snp=variant_meta.snp,
        chr=variant_meta.chr,
        pos=variant_meta.pos,
        a1=variant_meta.a1,
        a2=variant_meta.a2,
        n_obs=n_obs,
        n_miss=n_miss,
        miss_rate=miss_rate,
        af=af,
        maf=maf,
        mac=mac,
        mono=mono,
        het=het,
        het_exp=het_exp,
        het_excess=het_excess,
        hwe_p=hwe_p,
        imputed=imputed_list,
        imp_rsq=imp_rsq_t,
        filter_pass=[True] * m,  # populated by apply_qc_filters
        filter_reason=["PASS"] * m,
        dosage_class_freq=dosage_class_freq,
        het_per_class=het_per_class,
        double_reduction_alpha=double_reduction_alpha,
        hwe_p_dr=hwe_p_dr,
        mean_dosage_var=mean_dosage_var,
    )


def apply_qc_filters(
    stats: VariantQCStats,
    config: QCFilterConfig,
) -> Tensor:
    """Apply QC filters in GEMMA-convention order. Returns boolean mask of passing variants.

    Filter order (charter Section 6i):
    0. MONO (always excluded)
    1. MISS_RATE > miss_max
    2. MAF < maf_min
    2b. MAC < mac_min (optional)
    3. HWE_P < hwe_p_min
    4. HET_EXCESS > het_excess_max (optional)
    5. IMP_RSQ < imp_rsq_min (imputed variants only)
    """
    m = len(stats.snp)
    passes = torch.ones(m, dtype=torch.bool)
    reasons = ["PASS"] * m

    # 0. Monomorphic variants — always excluded from inference
    for i in range(m):
        if stats.mono[i]:
            passes[i] = False
            reasons[i] = "MONO"

    # 1. Missingness
    miss_fail = stats.miss_rate > config.miss_max
    for i in torch.where(miss_fail)[0].tolist():
        if passes[i]:
            passes[i] = False
            reasons[i] = f"MISS>{config.miss_max}"

    # 2. MAF
    maf_fail = stats.maf < config.maf_min
    for i in torch.where(maf_fail)[0].tolist():
        if passes[i]:
            passes[i] = False
            reasons[i] = f"MAF<{config.maf_min}"

    # 2b. MAC (optional)
    if config.mac_min is not None:
        mac_fail = stats.mac < config.mac_min
        for i in torch.where(mac_fail)[0].tolist():
            if passes[i]:
                passes[i] = False
                reasons[i] = f"MAC<{config.mac_min}"

    # 2c. Max genotype frequency (optional, for polyploid gene-action models)
    if config.max_geno_freq is not None and hasattr(stats, '_max_geno_freq'):
        gf_fail = stats._max_geno_freq > config.max_geno_freq
        for i in torch.where(gf_fail)[0].tolist():
            if passes[i]:
                passes[i] = False
                reasons[i] = f"GENO_FREQ>{config.max_geno_freq:.4f}"

    # 3. HWE (use double-reduction-aware test if available and requested)
    if config.use_double_reduction_hwe and stats.hwe_p_dr is not None:
        hwe_fail = stats.hwe_p_dr < config.hwe_p_min
        hwe_label = "HWE_DR"
    else:
        hwe_fail = stats.hwe_p < config.hwe_p_min
        hwe_label = "HWE"
    for i in torch.where(hwe_fail)[0].tolist():
        if passes[i]:
            passes[i] = False
            reasons[i] = f"{hwe_label}<{config.hwe_p_min}"

    # 4. Het excess (optional)
    if config.het_excess_max is not None:
        het_fail = stats.het_excess > config.het_excess_max
        for i in torch.where(het_fail)[0].tolist():
            if passes[i]:
                passes[i] = False
                reasons[i] = f"HET_EXCESS>{config.het_excess_max}"

    # 5. Imputation Rsq (only for imputed variants)
    for i in range(m):
        if stats.imputed[i] and not torch.isnan(stats.imp_rsq[i]):
            if stats.imp_rsq[i].item() < config.imp_rsq_min:
                if passes[i]:
                    passes[i] = False
                    reasons[i] = f"IMP_RSQ<{config.imp_rsq_min}"

    # 6. Dosage certainty (optional — for probability-based dosage calls)
    if config.dosage_var_max is not None and stats.mean_dosage_var is not None:
        dv_fail = stats.mean_dosage_var > config.dosage_var_max
        for i in torch.where(dv_fail)[0].tolist():
            if passes[i]:
                passes[i] = False
                reasons[i] = f"DOSAGE_VAR>{config.dosage_var_max}"

    # Update stats in place
    stats.filter_pass = passes.tolist()
    stats.filter_reason = reasons

    n_pass = int(passes.sum().item())
    n_fail = m - n_pass
    logger.info("QC filters: %d/%d variants pass (%d removed)", n_pass, m, n_fail)

    return passes


def write_variant_qc_parquet(stats: VariantQCStats, output_path: str) -> None:
    """Write variant QC stats to Parquet file via pyarrow.

    Schema matches charter Section 6i.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "Parquet output requires pyarrow: pip install pyarrow"
        )

    columns = {
        "SNP": stats.snp,
        "CHR": stats.chr,
        "POS": stats.pos,
        "A1": stats.a1,
        "A2": stats.a2,
        "N_OBS": stats.n_obs.int().tolist(),
        "N_MISS": stats.n_miss.int().tolist(),
        "MISS_RATE": stats.miss_rate.float().tolist(),
        "AF": stats.af.tolist(),
        "MAF": stats.maf.tolist(),
        "MAC": stats.mac.tolist(),
        "MONO": stats.mono,
        "HET": stats.het.tolist(),
        "HET_EXP": stats.het_exp.tolist(),
        "HET_EXCESS": stats.het_excess.tolist(),
        "HWE_P": stats.hwe_p.tolist(),
        "IMPUTED": stats.imputed,
        "IMP_RSQ": stats.imp_rsq.tolist(),
        "PASS_QC": stats.filter_pass,
        "FAIL_REASON": stats.filter_reason,
    }

    # Polyploid-specific columns
    if stats.double_reduction_alpha is not None:
        columns["DR_ALPHA"] = stats.double_reduction_alpha.tolist()
    if stats.hwe_p_dr is not None:
        columns["HWE_P_DR"] = stats.hwe_p_dr.tolist()
    if stats.mean_dosage_var is not None:
        columns["MEAN_DOSAGE_VAR"] = stats.mean_dosage_var.tolist()

    table = pa.table(columns)

    pq.write_table(table, output_path)
    logger.info("Variant QC Parquet written to %s (%d variants)", output_path, len(stats.snp))


def compute_max_genotype_freq(G_encoded: Tensor, ploidy: int = 2) -> Tensor:
    """Compute max genotype class frequency per marker after gene-action encoding.

    For binary-encoded markers (e.g., j-dom: values in {0, 1}), computes the
    frequency of the most common class. For multi-column models (general,
    diplo-general) with shape (n, m, k-1), returns the max frequency across
    all columns per marker.

    This filter is critical for polyploid gene-action models where encoding
    can collapse multiple dosage classes into one near-monomorphic class.
    GWASpoly applies this via the ``geno.freq`` parameter in ``set.params()``.

    Parameters
    ----------
    G_encoded : Tensor, shape (n, m) or (n, m, k-1)
        Gene-action-encoded genotype matrix (NaN allowed).
    ploidy : int
        Organism ploidy level (used for rounding tolerance).

    Returns
    -------
    Tensor, shape (m,)
        Max genotype class frequency per marker.
    """
    if G_encoded.ndim == 3:
        # Multi-column model: check each column independently, return max across columns
        n, m, k = G_encoded.shape
        max_freqs = torch.zeros(m, dtype=torch.float64, device=G_encoded.device)
        for col_idx in range(k):
            col_freqs = _max_class_freq_1d(G_encoded[:, :, col_idx])
            max_freqs = torch.maximum(max_freqs, col_freqs)
        return max_freqs
    else:
        return _max_class_freq_1d(G_encoded)


def _max_class_freq_1d(G: Tensor) -> Tensor:
    """Max class frequency per column for a 2D (n, m) encoded genotype matrix."""
    n, m = G.shape
    mask = ~torch.isnan(G)
    n_obs = mask.sum(dim=0).to(torch.float64).clamp(min=1.0)

    # Round to nearest integer class for frequency counting
    G_rounded = torch.round(G)
    unique_vals = torch.unique(G_rounded[mask])

    max_freq = torch.zeros(m, dtype=torch.float64, device=G.device)
    for val in unique_vals:
        if torch.isnan(val):
            continue
        count = ((G_rounded == val) & mask).sum(dim=0).to(torch.float64)
        freq = count / n_obs
        max_freq = torch.maximum(max_freq, freq)

    return max_freq


# --- Internal helpers ---

def _compute_observed_het(G: Tensor, ploidy: int = 2) -> Tensor:
    """Compute observed heterozygosity per SNP.

    Diploid: fraction of samples with dosage == 1.
    Polyploid: fraction of samples with 0 < dosage < k.
    """
    n, m = G.shape
    mask = ~torch.isnan(G)
    n_obs = mask.sum(dim=0).to(torch.float64).clamp(min=1.0)

    if ploidy == 2:
        het_count = ((G == 1.0) & mask).sum(dim=0).to(torch.float64)
    else:
        het_count = ((G > 0) & (G < ploidy) & mask).sum(dim=0).to(torch.float64)

    return het_count / n_obs


def _compute_expected_het(af: Tensor, ploidy: int = 2) -> Tensor:
    """Expected heterozygosity under HWE.

    Diploid: 2 * p * q.
    Polyploid: generalized HW — 1 - p^k - q^k (De Silva et al. 2005 approximation).
    """
    if ploidy == 2:
        return 2.0 * af * (1.0 - af)
    else:
        p = af
        q = 1.0 - af
        # P(all copies same allele) = p^k + q^k
        return 1.0 - torch.pow(p, ploidy) - torch.pow(q, ploidy)


def _compute_hwe_pvalue(G: Tensor, af: Tensor, ploidy: int = 2) -> Tensor:
    """Compute HWE p-value per SNP via chi-squared goodness-of-fit test.

    Diploid: 3-class test (AA, AB, BB) with 1 df.
    Polyploid: (k+1)-class test with k df (Levene 1949 extension).
    """
    from scipy import stats as sp_stats
    import numpy as np

    n, m = G.shape

    # Native fast path: collapse the per-SNP scipy.stats.chi2.sf call and the
    # genotype-class .item() round-trips into a single C++ pass that uses
    # a hand-rolled regularized upper incomplete gamma. The pure-Python loop
    # below remains the algorithmic spec.
    from .._native import HAS_NATIVE_HWE, _hwe_native
    from .._dispatch import native_disabled

    if (
        HAS_NATIVE_HWE and not native_disabled()
        and G.device.type == "cpu" and m > 0
    ):
        # Convert NaN -> -1 sentinel; round to nearest dosage class.
        G_int_t = torch.where(
            torch.isnan(G),
            torch.full_like(G, -1.0),
            torch.round(G).clamp(-1.0, float(ploidy)),
        ).to(torch.int64)
        G_int_np = G_int_t.detach().cpu().numpy()
        af_np = af.detach().to(torch.float64).cpu().numpy()
        hwe_np = _hwe_native.hwe_pvalues(G_int_np, af_np, int(ploidy))
        return torch.from_numpy(hwe_np).to(torch.float64)

    hwe_p = torch.ones(m, dtype=torch.float64)

    for j in range(m):
        col = G[:, j]
        valid = col[~torch.isnan(col)]
        n_valid = len(valid)
        if n_valid < 10:
            hwe_p[j] = 1.0
            continue

        p = af[j].item()
        q = 1.0 - p

        if ploidy == 2:
            # Observed counts: hom_ref(0), het(1), hom_alt(2)
            obs = torch.zeros(3, dtype=torch.float64)
            rounded = torch.round(valid).long().clamp(0, 2)
            for v in range(3):
                obs[v] = (rounded == v).sum()

            # Expected under HWE
            exp = torch.tensor([q * q, 2 * p * q, p * p], dtype=torch.float64) * n_valid
            exp = torch.clamp(exp, min=0.5)  # avoid zero expected

            chi2 = ((obs - exp) ** 2 / exp).sum().item()
            hwe_p[j] = float(sp_stats.chi2.sf(chi2, df=1))
        else:
            # Polyploid: (k+1) genotype classes
            n_classes = ploidy + 1
            obs = torch.zeros(n_classes, dtype=torch.float64)
            rounded = torch.round(valid).long().clamp(0, ploidy)
            for v in range(n_classes):
                obs[v] = (rounded == v).sum()

            # Expected under polyploid HWE (binomial expansion)
            from math import comb
            exp = torch.zeros(n_classes, dtype=torch.float64)
            for v in range(n_classes):
                # P(dosage = v) = C(k,v) * p^v * q^(k-v)
                exp[v] = comb(ploidy, v) * (p ** v) * (q ** (ploidy - v))
            exp = exp * n_valid
            exp = torch.clamp(exp, min=0.5)

            chi2 = ((obs - exp) ** 2 / exp).sum().item()
            df = n_classes - 2  # estimated 1 parameter (p)
            if df < 1:
                df = 1
            hwe_p[j] = float(sp_stats.chi2.sf(chi2, df=df))

    return hwe_p


# --- Polyploid-specific QC helpers ---


def _compute_dosage_class_freq(G: Tensor, ploidy: int) -> Tensor:
    """Compute per-dosage-class frequency for each marker.

    Returns a (m, k+1) tensor where entry [j, d] is the fraction of
    non-missing samples at marker j with rounded dosage == d.

    Parameters
    ----------
    G : Tensor, shape (n, m)
    ploidy : int

    Returns
    -------
    Tensor, shape (m, ploidy+1)
    """
    n, m = G.shape
    mask = ~torch.isnan(G)
    n_obs = mask.sum(dim=0).to(torch.float64).clamp(min=1.0)  # (m,)
    n_classes = ploidy + 1

    freq = torch.zeros(m, n_classes, dtype=torch.float64, device=G.device)
    rounded = torch.round(G).long().clamp(0, ploidy)

    for d in range(n_classes):
        count = ((rounded == d) & mask).sum(dim=0).to(torch.float64)
        freq[:, d] = count / n_obs

    return freq


def _compute_het_per_class(G: Tensor, ploidy: int) -> Tensor:
    """Compute per-heterozygosity-class frequency for polyploids.

    For a k-ploid, dosages 1..(k-1) are heterozygous classes:
    - dosage 1 = simplex (AAAB for tetraploid)
    - dosage 2 = duplex (AABB)
    - dosage 3 = triplex (ABBB)
    etc.

    Returns a (m, k-1) tensor of frequencies.

    Parameters
    ----------
    G : Tensor, shape (n, m)
    ploidy : int

    Returns
    -------
    Tensor, shape (m, ploidy-1)
        Frequency of each heterozygous dosage class per marker.
    """
    n, m = G.shape
    mask = ~torch.isnan(G)
    n_obs = mask.sum(dim=0).to(torch.float64).clamp(min=1.0)

    n_het_classes = ploidy - 1
    het_freq = torch.zeros(m, n_het_classes, dtype=torch.float64, device=G.device)
    rounded = torch.round(G).long().clamp(0, ploidy)

    for h in range(n_het_classes):
        d = h + 1  # dosage classes 1..k-1
        count = ((rounded == d) & mask).sum(dim=0).to(torch.float64)
        het_freq[:, h] = count / n_obs

    return het_freq


def _compute_hwe_double_reduction(
    G: Tensor, af: Tensor, ploidy: int
) -> tuple[Tensor, Tensor]:
    """Compute HWE p-values accounting for double reduction in autopolyploids.

    Double reduction (DR) occurs during meiosis of autopolyploids when
    homologous chromatids end up in the same gamete. This shifts genotype
    frequencies away from pure binomial (random chromosome) expectations.

    For tetraploids, the genotype frequency of dosage d given allele
    frequency p and DR parameter alpha is:

        P(d | p, alpha) = (1-alpha) * Binom(k, d, p) + alpha * DR_term(k, d, p)

    where the DR term accounts for the probability of identical-by-descent
    chromatid pairs in a gamete (Haldane 1930, Levine 1949).

    We estimate alpha by method-of-moments from the observed duplex (d=k/2)
    frequency excess, then test goodness-of-fit under this model.

    Parameters
    ----------
    G : Tensor, shape (n, m)
    af : Tensor, shape (m,)
    ploidy : int

    Returns
    -------
    alpha : Tensor, shape (m,)
        Estimated double reduction parameter per marker (0 = no DR).
    hwe_p_dr : Tensor, shape (m,)
        HWE p-value under double-reduction model.
    """
    from scipy import stats as sp_stats
    from math import comb

    n, m = G.shape
    alpha_est = torch.zeros(m, dtype=torch.float64)
    hwe_p_dr = torch.ones(m, dtype=torch.float64)

    if ploidy != 4:
        # DR model currently implemented for tetraploids only.
        # For other ploidies, return zeros/ones (no DR correction).
        return alpha_est, hwe_p_dr

    # Native shortcut: pushes the per-SNP scipy.stats.chi2.sf + .item() loop
    # into a single GIL-released C++ pass. The Python body below remains as
    # the algorithmic spec; both paths are exercised in the test suite.
    from .._native import HAS_NATIVE_HWE, _hwe_native
    from .._dispatch import native_disabled
    if (
        HAS_NATIVE_HWE and not native_disabled()
        and G.device.type == "cpu" and m > 0
    ):
        G_int_t = torch.where(
            torch.isnan(G),
            torch.full_like(G, -1.0),
            torch.round(G).clamp(-1.0, float(ploidy)),
        ).to(torch.int64)
        G_int_np = G_int_t.detach().cpu().numpy()
        af_np = af.detach().to(torch.float64).cpu().numpy()
        alpha_np, hwe_np = _hwe_native.hwe_pvalues_double_reduction(
            G_int_np, af_np
        )
        return (
            torch.from_numpy(alpha_np).to(torch.float64),
            torch.from_numpy(hwe_np).to(torch.float64),
        )

    for j in range(m):
        col = G[:, j]
        valid = col[~torch.isnan(col)]
        n_valid = len(valid)
        if n_valid < 10:
            continue

        p = af[j].item()
        q = 1.0 - p
        if p < 1e-10 or q < 1e-10:
            continue

        rounded = torch.round(valid).long().clamp(0, ploidy)

        # Observed class counts
        obs = torch.zeros(ploidy + 1, dtype=torch.float64)
        for d in range(ploidy + 1):
            obs[d] = (rounded == d).sum().item()

        # Expected under pure binomial (no DR)
        binom_exp = torch.zeros(ploidy + 1, dtype=torch.float64)
        for d in range(ploidy + 1):
            binom_exp[d] = comb(ploidy, d) * (p ** d) * (q ** (ploidy - d))

        # Tetraploid DR model (Haldane 1930):
        # Gamete probs with DR param alpha:
        #   P(gamete=0) = (1-alpha)*q^2 + alpha*q
        #   P(gamete=1) = (1-alpha)*2pq
        #   P(gamete=2) = (1-alpha)*p^2 + alpha*p
        # Genotype = sum of two gametes, so convolve gamete distributions.
        # Estimate alpha from excess homozygosity at dose=0 and dose=4.

        # Method-of-moments: observed homozygosity
        obs_hom0 = obs[0].item() / n_valid
        obs_hom4 = obs[4].item() / n_valid
        exp_hom0_no_dr = q ** 4
        exp_hom4_no_dr = p ** 4

        # Under DR: P(dose=0) = (1-alpha)^2 * q^4 + 2*alpha*(1-alpha)*q^3 + alpha^2*q^2
        # Simplified: excess in homozygotes estimates alpha
        # Use: alpha_hat = max(0, (obs_hom - exp_hom_no_dr) / (exp_hom_with_alpha_1 - exp_hom_no_dr))
        # For dose=0: when alpha=1, P(dose=0) = q^2 (gamete is haploid)
        if q > 1e-10:
            denom0 = q ** 2 - q ** 4
            if abs(denom0) > 1e-10:
                alpha0 = (obs_hom0 - exp_hom0_no_dr) / denom0
            else:
                alpha0 = 0.0
        else:
            alpha0 = 0.0

        if p > 1e-10:
            denom4 = p ** 2 - p ** 4
            if abs(denom4) > 1e-10:
                alpha4 = (obs_hom4 - exp_hom4_no_dr) / denom4
            else:
                alpha4 = 0.0
        else:
            alpha4 = 0.0

        # Average and clamp to [0, 1/6] (theoretical max for tetraploid)
        alpha = max(0.0, min((alpha0 + alpha4) / 2.0, 1.0 / 6.0))
        alpha_est[j] = alpha

        # Compute expected frequencies under DR model for tetraploid
        # Gamete probabilities with DR
        gam = torch.zeros(3, dtype=torch.float64)  # gamete dose 0, 1, 2
        gam[0] = (1 - alpha) * q * q + alpha * q
        gam[1] = (1 - alpha) * 2 * p * q
        gam[2] = (1 - alpha) * p * p + alpha * p

        # Genotype = convolution of two independent gametes
        dr_exp = torch.zeros(ploidy + 1, dtype=torch.float64)
        for d1 in range(3):
            for d2 in range(3):
                d = d1 + d2
                if d <= ploidy:
                    dr_exp[d] += gam[d1] * gam[d2]

        dr_exp_counts = dr_exp * n_valid
        dr_exp_counts = torch.clamp(dr_exp_counts, min=0.5)

        chi2 = ((obs - dr_exp_counts) ** 2 / dr_exp_counts).sum().item()
        # df = n_classes - 1 (estimated p) - 1 (estimated alpha) - 1 = k+1-3 = k-2
        df = max(ploidy - 2, 1)
        hwe_p_dr[j] = float(sp_stats.chi2.sf(chi2, df=df))

    return alpha_est, hwe_p_dr
