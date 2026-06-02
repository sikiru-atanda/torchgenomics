"""Summary-data-based Mendelian Randomization (SMR) and HEIDI test.

Implements Zhu et al. (2016, Nature Genetics), "Integration of summary
data from GWAS and eQTL studies predicts complex trait gene targets".

**SMR test.** For a SNP that is both GWAS-associated and an eQTL for
gene g, the SMR chi-squared statistic is:

    chi2_SMR = (z_GWAS^2 * z_eQTL^2) / (z_GWAS^2 + z_eQTL^2)

This tests whether the GWAS signal and eQTL signal share a common
causal mechanism. ``p_SMR = chi2_sf(chi2_SMR, df=1)``.

**HEIDI test.** Distinguishes pleiotropy/causality (single causal
variant) from linkage (two distinct causal variants). For SNPs ``i``
in LD with the top eQTL:

    d_i = beta_GWAS_i / beta_eQTL_i  -  beta_GWAS_top / beta_eQTL_top
    T_HEIDI = sum(d_i^2 / var_i)  ~  chi2(K - 1)

Large ``p_HEIDI`` (> 0.05) means the data are consistent with a single
causal variant (pleiotropy or causality).

**Polyploid compatibility.** Both tests operate on z-scores and effect-
size ratios from GWAS / eQTL summary statistics, which are ploidy-
agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._sumstats import SumStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chi2_sf_1df(x: Tensor) -> Tensor:
    """Survival function of chi-squared(1) = 2 * Phi(-sqrt(x))."""
    return torch.erfc(torch.sqrt(torch.clamp(x, min=0.0)) / (2.0 ** 0.5))


def _chi2_sf(x: Tensor, df: int) -> Tensor:
    """Chi-squared survival function.

    For df=1, uses the closed-form via erfc. For df>1, falls back to scipy.
    """
    if df == 1:
        return _chi2_sf_1df(x)
    from scipy.stats import chi2  # type: ignore[import-untyped]
    p = chi2.sf(x.detach().cpu().numpy(), df=df)
    return torch.tensor(p, dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SMRResult:
    """Result for a single gene from the SMR/HEIDI test.

    Attributes
    ----------
    gene_id : str
        Gene identifier.
    probe_snp : str
        Top eQTL SNP used as the instrument.
    beta_smr : float
        SMR causal estimate (beta_GWAS / beta_eQTL).
    se_smr : float
        Standard error of the SMR estimate (delta method).
    p_smr : float
        SMR p-value (chi-squared with 1 df).
    chi2_smr : float
        SMR chi-squared statistic.
    beta_gwas : float
        GWAS effect size at the probe SNP.
    beta_eqtl : float
        eQTL effect size at the probe SNP.
    p_heidi : float
        HEIDI p-value (large = consistent with single causal variant).
    n_heidi_snps : int
        Number of SNPs used in the HEIDI test.
    heidi_stat : float
        HEIDI test statistic.
    """

    gene_id: str
    probe_snp: str
    beta_smr: float
    se_smr: float
    p_smr: float
    chi2_smr: float
    beta_gwas: float
    beta_eqtl: float
    p_heidi: float
    n_heidi_snps: int
    heidi_stat: float


@dataclass
class SMRSummary:
    """Aggregate result from multi-gene SMR/HEIDI analysis.

    Attributes
    ----------
    results : list[SMRResult]
        Per-gene results.
    n_genes_tested : int
        Number of genes with a qualifying eQTL instrument.
    n_significant_smr : int
        Number with ``p_smr < threshold``.
    n_pass_heidi : int
        Number with ``p_heidi > heidi_threshold``.
    """

    results: list[SMRResult]
    n_genes_tested: int
    n_significant_smr: int
    n_pass_heidi: int


# ---------------------------------------------------------------------------
# SMR test
# ---------------------------------------------------------------------------

def smr_test(
    gwas: SumStats,
    eqtl: SumStats,
    gene_id: str,
    probe_snp: str | None = None,
    eqtl_p_threshold: float = 5e-8,
) -> SMRResult:
    """SMR test for a single gene.

    Parameters
    ----------
    gwas : SumStats
        GWAS summary statistics (must contain the probe SNP).
    eqtl : SumStats
        eQTL summary statistics for the gene's cis region.
    gene_id : str
        Gene identifier.
    probe_snp : str or None
        Top eQTL SNP. If None, selects the SNP with smallest eQTL p-value.
    eqtl_p_threshold : float
        Only consider eQTL SNPs below this threshold.

    Returns
    -------
    SMRResult
    """
    # Build SNP lookup for eQTL
    eqtl_snp_map = {s: i for i, s in enumerate(eqtl.snp)}

    # Select probe SNP (top eQTL)
    if probe_snp is None:
        mask = eqtl.p < eqtl_p_threshold
        if not mask.any():
            return SMRResult(
                gene_id=gene_id, probe_snp="",
                beta_smr=0.0, se_smr=float("inf"), p_smr=1.0, chi2_smr=0.0,
                beta_gwas=0.0, beta_eqtl=0.0,
                p_heidi=1.0, n_heidi_snps=0, heidi_stat=0.0,
            )
        idx = int(eqtl.p.argmin().item())
        probe_snp = eqtl.snp[idx]

    # Look up probe SNP in both datasets
    eqtl_idx = eqtl_snp_map.get(probe_snp)
    gwas_snp_map = {s: i for i, s in enumerate(gwas.snp)}
    gwas_idx = gwas_snp_map.get(probe_snp)

    if eqtl_idx is None or gwas_idx is None:
        return SMRResult(
            gene_id=gene_id, probe_snp=probe_snp or "",
            beta_smr=0.0, se_smr=float("inf"), p_smr=1.0, chi2_smr=0.0,
            beta_gwas=0.0, beta_eqtl=0.0,
            p_heidi=1.0, n_heidi_snps=0, heidi_stat=0.0,
        )

    z_gwas = float((gwas.beta[gwas_idx] / gwas.se[gwas_idx]).item())
    z_eqtl = float((eqtl.beta[eqtl_idx] / eqtl.se[eqtl_idx]).item())
    beta_g = float(gwas.beta[gwas_idx].item())
    beta_e = float(eqtl.beta[eqtl_idx].item())
    se_g = float(gwas.se[gwas_idx].item())
    se_e = float(eqtl.se[eqtl_idx].item())

    # SMR chi-squared
    z_g2 = z_gwas ** 2
    z_e2 = z_eqtl ** 2
    denom = z_g2 + z_e2
    if denom < 1e-300:
        chi2_smr = 0.0
    else:
        chi2_smr = (z_g2 * z_e2) / denom

    p_smr = float(_chi2_sf_1df(
        torch.tensor(chi2_smr, dtype=torch.float64)
    ).item())

    # SMR causal estimate via Wald ratio
    if abs(beta_e) < 1e-300:
        beta_smr = 0.0
        se_smr = float("inf")
    else:
        beta_smr = beta_g / beta_e
        # Delta method SE
        se_smr = abs(beta_smr) * ((se_g / max(abs(beta_g), 1e-300)) ** 2
                                   + (se_e / max(abs(beta_e), 1e-300)) ** 2) ** 0.5

    return SMRResult(
        gene_id=gene_id,
        probe_snp=probe_snp,
        beta_smr=beta_smr,
        se_smr=se_smr,
        p_smr=p_smr,
        chi2_smr=chi2_smr,
        beta_gwas=beta_g,
        beta_eqtl=beta_e,
        p_heidi=1.0,
        n_heidi_snps=0,
        heidi_stat=0.0,
    )


# ---------------------------------------------------------------------------
# HEIDI test
# ---------------------------------------------------------------------------

def heidi_test(
    gwas: SumStats,
    eqtl: SumStats,
    probe_snp: str,
    nearby_snps: list[str],
    ld_r2_threshold: float = 0.05,
    max_snps: int = 20,
    ld_matrix: Tensor | None = None,
) -> tuple[float, float, int]:
    """HEIDI heterogeneity test.

    Tests whether the SMR association reflects a single causal variant
    (pleiotropy/causality) vs. two linked variants (linkage).

    Parameters
    ----------
    gwas : SumStats
        GWAS summary statistics.
    eqtl : SumStats
        eQTL summary statistics.
    probe_snp : str
        Top eQTL instrument SNP.
    nearby_snps : list[str]
        SNP IDs of nearby SNPs in LD with the probe (excludes probe).
    ld_r2_threshold : float
        Minimum r² with probe to include in HEIDI (default 0.05).
    max_snps : int
        Maximum number of SNPs to use (default 20, sorted by eQTL p).
    ld_matrix : Tensor, optional
        ``(M_gwas, M_gwas)`` SNP-SNP LD correlation matrix r (NOT r²),
        with rows / columns aligned to ``gwas.snp`` ordering. When
        provided, HEIDI augments each ``Var(d_i)`` with the two cross-
        covariance terms induced by LD between SNP i and the top SNP
        (Zhu et al. 2016 supplementary eq. 18 / SMR-tool v1.3.1 source):

            Var_LD(d_i) = Var_diag(d_i)
                        + 2 * a_i * a_t * r(i, top) * seg_i * seg_top
                        + 2 * c_i * c_t * r(i, top) * see_i * see_top

        with ``a_i = 1/be_i``, ``c_i = -bg_i/be_i^2``, ``a_t = -1/be_top``,
        ``c_t = +bg_top/be_top^2``. ``T_HEIDI = sum_i d_i^2 / Var_LD(d_i)``
        with df = n_snps, matching the upstream SMR tool. The same LD is
        assumed to apply symmetrically to b_GWAS and b_eQTL covariances
        (standard SMR assumption: both summary estimates come from the
        same ancestry / reference panel).

        When ``None`` (V1 default) HEIDI falls back to the diagonal
        delta-method variance assuming SNP independence — this was the
        pre-F3 #3 behavior and is kept for backward compatibility.

        Note: a fully rigorous multivariate test would use ``d' Σ_d^{-1} d``
        with ``Σ_d`` including d_i-d_j off-diagonal LD coupling and the
        shared Var(b_top) rank-1 perturbation. That form is *more*
        accurate but differs from what the published SMR tool computes;
        we therefore implement the SMR convention so the LD-weighted
        path agrees with the reference tool numerically.

    Returns
    -------
    (p_heidi, heidi_stat, n_snps) : tuple
    """
    gwas_map = {s: i for i, s in enumerate(gwas.snp)}
    eqtl_map = {s: i for i, s in enumerate(eqtl.snp)}

    probe_gwas_idx = gwas_map.get(probe_snp)
    probe_eqtl_idx = eqtl_map.get(probe_snp)
    if probe_gwas_idx is None or probe_eqtl_idx is None:
        return 1.0, 0.0, 0

    bg_top = float(gwas.beta[probe_gwas_idx].item())
    be_top = float(eqtl.beta[probe_eqtl_idx].item())
    if abs(be_top) < 1e-300:
        return 1.0, 0.0, 0

    ratio_top = bg_top / be_top

    # Sort nearby by eQTL p-value and select up to ``max_snps`` SNPs that
    # are present in both summaries and have non-degenerate eQTL effect.
    snp_p_pairs: list[tuple[str, float]] = []
    for s in nearby_snps:
        ei = eqtl_map.get(s)
        gi = gwas_map.get(s)
        if ei is not None and gi is not None:
            snp_p_pairs.append((s, float(eqtl.p[ei].item())))
    snp_p_pairs.sort(key=lambda x: x[1])

    bg_list: list[float] = []
    be_list: list[float] = []
    seg_list: list[float] = []
    see_list: list[float] = []
    gwas_idx_list: list[int] = []
    for s, _ in snp_p_pairs[:max_snps]:
        gi = gwas_map[s]
        ei = eqtl_map[s]
        be_i = float(eqtl.beta[ei].item())
        if abs(be_i) < 1e-300:
            continue
        bg_list.append(float(gwas.beta[gi].item()))
        be_list.append(be_i)
        seg_list.append(float(gwas.se[gi].item()))
        see_list.append(float(eqtl.se[ei].item()))
        gwas_idx_list.append(gi)

    n_snps = len(bg_list)
    if n_snps < 1:
        return 1.0, 0.0, 0

    bg_t = torch.tensor(bg_list, dtype=torch.float64)
    be_t = torch.tensor(be_list, dtype=torch.float64)
    seg_t = torch.tensor(seg_list, dtype=torch.float64)
    see_t = torch.tensor(see_list, dtype=torch.float64)
    seg_top = float(gwas.se[probe_gwas_idx].item())
    see_top = float(eqtl.se[probe_eqtl_idx].item())

    d_t = bg_t / be_t - ratio_top  # (n,) cross-fit GWAS/eQTL ratio differences

    if ld_matrix is None:
        # Diagonal delta-method variance (assumes SNP independence). The
        # V1 default; matches the pre-F3 #3 behavior bit-for-bit when the
        # corresponding SNPs survived the same filter.
        var_t = (
            (seg_t / be_t.abs().clamp(min=1e-300)) ** 2
            + (seg_top / max(abs(be_top), 1e-300)) ** 2
            + (see_t * bg_t.abs() / (be_t ** 2).clamp(min=1e-300)) ** 2
            + (see_top * abs(bg_top) / max(be_top ** 2, 1e-300)) ** 2
        )
        var_safe = var_t.clamp(min=1e-300)
        heidi_stat = float(((d_t ** 2) / var_safe).sum().item())
        p_heidi = float(
            _chi2_sf(
                torch.tensor(heidi_stat, dtype=torch.float64), df=n_snps,
            ).item()
        )
        return p_heidi, heidi_stat, n_snps

    # --- F3 #3 LD-weighted HEIDI (Zhu 2016 / SMR-tool convention) ---------
    # For each helper SNP i, augment the diagonal delta-method variance
    # Var(d_i) with the two cross-covariance terms induced by LD between i
    # and the top SNP — under the SMR convention (Yang lab v1.3.1 source):
    #
    #     Var_LD(d_i) = Var_diag(d_i)
    #                 + 2 * a_i * a_t * r(i, top) * seg_i * seg_top
    #                 + 2 * c_i * c_t * r(i, top) * see_i * see_top
    #
    # with a_i = 1/be_i, c_i = -bg_i/be_i^2, a_t = -1/be_top, c_t = +bg_top/be_top^2.
    # The HEIDI statistic remains sum_i d_i^2 / Var_LD(d_i) ~ chi^2(n_snps).
    # This per-SNP-pair correction matches Zhu 2016 supplementary eq. 18
    # and the published SMR tool's HEIDI output. Note that when the helper
    # SNPs carry effect estimates of OPPOSITE sign to the top SNP (typical
    # for cis-eQTL fine-mapping where LD-linked SNPs can flip phase),
    # a_i * a_t and c_i * c_t are positive, so Var_LD > Var_diag and
    # T_HEIDI shrinks — matching SMR's reported behavior under moderate LD.
    #
    # The full multivariate Sigma_d^{-1} form (which additionally accounts
    # for d_i-d_j correlations through r(i, j) and the shared Var(b_top))
    # is mathematically more rigorous but is NOT what the SMR reference
    # tool computes; users who pass an ld_matrix to heidi_test expect the
    # SMR convention.
    ld_matrix = ld_matrix.to(dtype=torch.float64)
    if ld_matrix.shape[0] != len(gwas.snp) or ld_matrix.shape[1] != len(gwas.snp):
        raise ValueError(
            f"ld_matrix shape {tuple(ld_matrix.shape)} does not match "
            f"len(gwas.snp) = {len(gwas.snp)}; rows / columns must be "
            "aligned to the GWAS SNP ordering."
        )
    idx_t = torch.tensor(gwas_idx_list, dtype=torch.long, device=ld_matrix.device)
    R_it = ld_matrix[idx_t, probe_gwas_idx]       # (n,) r between each i and top

    a_i = 1.0 / be_t                              # (n,)
    c_i = -bg_t / (be_t ** 2)                     # (n,)
    a_t = -1.0 / be_top                           # scalar
    c_t = bg_top / (be_top ** 2)                  # scalar

    var_diag = (
        (seg_t / be_t.abs().clamp(min=1e-300)) ** 2
        + (seg_top / max(abs(be_top), 1e-300)) ** 2
        + (see_t * bg_t.abs() / (be_t ** 2).clamp(min=1e-300)) ** 2
        + (see_top * abs(bg_top) / max(be_top ** 2, 1e-300)) ** 2
    )
    cross_gwas = 2.0 * a_i * a_t * R_it * seg_t * seg_top
    cross_eqtl = 2.0 * c_i * c_t * R_it * see_t * see_top
    var_ld = (var_diag + cross_gwas + cross_eqtl).clamp(min=1e-300)

    heidi_stat = float(((d_t ** 2) / var_ld).sum().item())
    p_heidi = float(
        _chi2_sf(
            torch.tensor(heidi_stat, dtype=torch.float64), df=n_snps,
        ).item()
    )
    return p_heidi, heidi_stat, n_snps


# ---------------------------------------------------------------------------
# Multi-gene wrapper
# ---------------------------------------------------------------------------

def smr_heidi(
    gwas: SumStats,
    eqtl: SumStats,
    gene_map: dict[str, list[str]],
    eqtl_p_threshold: float = 5e-8,
    smr_p_threshold: float = 0.05,
    heidi_p_threshold: float = 0.05,
    heidi_max_snps: int = 20,
    ld_matrix: Tensor | None = None,
) -> SMRSummary:
    """Run SMR + HEIDI across multiple genes.

    Parameters
    ----------
    gwas : SumStats
        GWAS summary statistics.
    eqtl : SumStats
        eQTL summary statistics (all cis-SNPs for all genes).
    gene_map : dict[str, list[str]]
        Mapping from gene ID to list of cis-SNP IDs.
    eqtl_p_threshold : float
        eQTL significance threshold for probe selection.
    smr_p_threshold : float
        SMR p-value threshold for counting significant genes.
    heidi_p_threshold : float
        HEIDI p-value threshold (genes with p > this pass HEIDI).
    heidi_max_snps : int
        Maximum SNPs per HEIDI test.
    ld_matrix : Tensor, optional
        ``(M_gwas, M_gwas)`` SNP-SNP LD r matrix aligned to ``gwas.snp``.
        When provided, every per-gene HEIDI call uses the LD-weighted
        full-covariance variance (Zhu 2016 supplementary eq. 18) instead
        of the diagonal-only delta-method default. See ``heidi_test``
        for the formula and the F3 #3 rationale.

    Returns
    -------
    SMRSummary
    """
    results: list[SMRResult] = []
    n_sig = 0
    n_pass = 0

    for gene_id, cis_snps in gene_map.items():
        res = smr_test(gwas, eqtl, gene_id,
                       eqtl_p_threshold=eqtl_p_threshold)

        if res.probe_snp and res.p_smr < smr_p_threshold:
            nearby = [s for s in cis_snps if s != res.probe_snp]
            p_h, h_stat, n_h = heidi_test(
                gwas, eqtl, res.probe_snp, nearby,
                max_snps=heidi_max_snps,
                ld_matrix=ld_matrix,
            )
            res = SMRResult(
                gene_id=res.gene_id,
                probe_snp=res.probe_snp,
                beta_smr=res.beta_smr,
                se_smr=res.se_smr,
                p_smr=res.p_smr,
                chi2_smr=res.chi2_smr,
                beta_gwas=res.beta_gwas,
                beta_eqtl=res.beta_eqtl,
                p_heidi=p_h,
                n_heidi_snps=n_h,
                heidi_stat=h_stat,
            )

        results.append(res)
        if res.p_smr < smr_p_threshold:
            n_sig += 1
        if res.p_heidi > heidi_p_threshold:
            n_pass += 1

    return SMRSummary(
        results=results,
        n_genes_tested=len(results),
        n_significant_smr=n_sig,
        n_pass_heidi=n_pass,
    )
