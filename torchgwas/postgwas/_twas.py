"""Transcriptome-Wide Association Study (TWAS).

Implements the summary-statistic-based TWAS (S-PrediXcan, Barbeira
et al. 2018) and individual-level TWAS (PrediXcan, Gamazon et al.
2015; FUSION, Gusev et al. 2016).

**Summary-stat TWAS (S-PrediXcan).** For a gene ``g`` with eQTL
weight vector ``w`` (k cis-SNPs) and GWAS z-scores ``z``:

    z_twas = w^T z / sqrt(w^T Sigma w)

where ``Sigma`` is the LD (correlation) matrix among the k cis-SNPs.
Under the null of no gene-trait association, ``z_twas ~ N(0, 1)``.

**Individual-level TWAS.** Impute genetically regulated expression
``GReX = X_cis @ w``, then regress the phenotype on GReX:

    z_twas = (GReX^T y) / (||GReX|| * sigma_y)

**Polyploid compatibility.** Both TWAS methods operate on GWAS summary
statistics or standardized genotypes. The z-scores and eQTL weights are
ploidy-agnostic, so TWAS results are valid for any ploidy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

from ._sumstats import SumStats

# ---------------------------------------------------------------------------
# Normal helpers
# ---------------------------------------------------------------------------

def _normal_sf(x: Tensor) -> Tensor:
    return 0.5 * torch.erfc(x / (2.0 ** 0.5))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TWASGeneResult:
    """Result for a single gene from TWAS.

    Attributes
    ----------
    gene_id : str
        Gene identifier.
    z_twas : float
        TWAS z-score.
    p_twas : float
        TWAS p-value (two-sided normal).
    n_cis_snps : int
        Number of cis-SNPs with non-zero weights (weight-based modes) or
        0 (observed-expression mode where no eQTL weights are used).
    r2_model : float or None
        Cross-validated R-squared of the expression prediction model.
        ``None`` for observed-expression mode.
    top_weight_snp : str or None
        SNP with the largest absolute weight (weight-based modes).
        ``None`` for observed-expression mode.
    beta : float or None
        Per-gene effect estimate (observed-expression mode; the OLS / LMM
        Wald coefficient of the expression column on the phenotype).
        ``None`` for the weight-based modes which report z_twas only.
    se : float or None
        Standard error of ``beta`` (observed-expression mode); ``None``
        otherwise.
    chr : str or None
        Chromosome (when a gene annotation is supplied).
    start : int or None
        Gene start coordinate (when annotation is supplied).
    end : int or None
        Gene end coordinate (when annotation is supplied).
    gene_name : str or None
        Human-readable gene name / HGNC symbol (when annotation is
        supplied).
    """

    gene_id: str
    z_twas: float
    p_twas: float
    n_cis_snps: int
    r2_model: float | None
    top_weight_snp: str | None = None
    beta: float | None = None
    se: float | None = None
    chr: str | None = None
    start: int | None = None
    end: int | None = None
    gene_name: str | None = None


@dataclass
class TWASResult:
    """Aggregate TWAS result.

    Attributes
    ----------
    genes : list[TWASGeneResult]
        Per-gene results.
    n_genes_tested : int
        Number of genes tested.
    n_significant : int
        Number significant after correction.
    """

    genes: list[TWASGeneResult]
    n_genes_tested: int
    n_significant: int


# ---------------------------------------------------------------------------
# Summary-stat TWAS
# ---------------------------------------------------------------------------

def twas_sumstat(
    gwas: SumStats,
    weights: dict[str, Tensor],
    snp_lists: dict[str, list[str]],
    ld_matrix: dict[str, Tensor] | None = None,
    r2_models: dict[str, float] | None = None,
    p_threshold: float = 0.05,
    correction: str = "bonferroni",
) -> TWASResult:
    """Summary-statistic-based TWAS (S-PrediXcan).

    Parameters
    ----------
    gwas : SumStats
        GWAS summary statistics with z-scores.
    weights : dict[str, Tensor]
        Gene ID -> ``(k,)`` eQTL weight vector.
    snp_lists : dict[str, list[str]]
        Gene ID -> list of SNP IDs (same order as weights).
    ld_matrix : dict[str, Tensor] or None
        Gene ID -> ``(k, k)`` LD correlation matrix. If None, identity
        (independent SNPs) is assumed with a warning.
    r2_models : dict[str, float] or None
        Gene ID -> cross-validated R-squared of the expression model.
    p_threshold : float
        Significance threshold before correction.
    correction : str
        Multiple testing correction: ``"bonferroni"`` or ``"fdr"``.

    Returns
    -------
    TWASResult
    """
    if len(weights) == 0:
        raise ValueError("At least one gene with weights is required.")

    if ld_matrix is None:
        warnings.warn(
            "No LD matrix provided; assuming independent SNPs (identity LD). "
            "This may inflate z_twas for genes with correlated cis-SNPs.",
            stacklevel=2,
        )

    gwas_snp_map = {s: i for i, s in enumerate(gwas.snp)}
    z_gwas = gwas.z.to(torch.float64)

    gene_results: list[TWASGeneResult] = []

    for gene_id, w in weights.items():
        snps = snp_lists.get(gene_id, [])
        w = w.to(torch.float64)
        k = w.shape[0]

        if k == 0 or len(snps) != k:
            continue

        # Map SNP IDs to GWAS indices
        indices: list[int] = []
        valid_positions: list[int] = []
        for j, s in enumerate(snps):
            idx = gwas_snp_map.get(s)
            if idx is not None:
                indices.append(idx)
                valid_positions.append(j)

        if len(indices) == 0:
            continue

        idx_t = torch.tensor(indices, dtype=torch.long, device=z_gwas.device)
        z_sub = z_gwas[idx_t]
        w_sub = w[valid_positions]

        # z_twas = w^T z / sqrt(w^T Sigma w)
        wz = (w_sub * z_sub).sum()

        if ld_matrix is not None and gene_id in ld_matrix:
            sigma = ld_matrix[gene_id].to(torch.float64)
            # Subset LD matrix to valid positions
            vp = torch.tensor(valid_positions, dtype=torch.long)
            sigma_sub = sigma[vp][:, vp]
            var_denom = (w_sub @ sigma_sub @ w_sub)
        else:
            # Identity LD
            var_denom = (w_sub ** 2).sum()

        var_denom = max(float(var_denom.item()), 1e-300)
        z_twas_val = float(wz.item()) / var_denom ** 0.5

        # Two-sided p-value
        p_twas = 2.0 * float(_normal_sf(
            torch.tensor(abs(z_twas_val), dtype=torch.float64)
        ).item())

        # Top weight SNP
        top_idx = int(w_sub.abs().argmax().item())
        top_snp = snps[valid_positions[top_idx]]

        r2 = r2_models.get(gene_id) if r2_models else None

        gene_results.append(TWASGeneResult(
            gene_id=gene_id,
            z_twas=z_twas_val,
            p_twas=p_twas,
            n_cis_snps=len(valid_positions),
            r2_model=r2,
            top_weight_snp=top_snp,
        ))

    # Multiple testing correction
    n_tested = len(gene_results)
    if n_tested > 0:
        p_vals = torch.tensor(
            [g.p_twas for g in gene_results], dtype=torch.float64
        )
        if correction == "bonferroni":
            threshold = p_threshold / n_tested
        elif correction == "fdr":
            # BH procedure
            sorted_p, sort_idx = torch.sort(p_vals)
            ranks = torch.arange(1, n_tested + 1, dtype=torch.float64)
            bh_threshold = sorted_p * n_tested / ranks
            threshold = float(p_threshold)
            # Find largest k where p_(k) <= k/m * alpha
            passing = sorted_p <= (ranks / n_tested * p_threshold)
            if passing.any():
                max_k = int(passing.nonzero()[-1].item())
                threshold = float(sorted_p[max_k].item())
            else:
                threshold = 0.0
        else:
            threshold = p_threshold

        n_sig = sum(1 for g in gene_results if g.p_twas <= threshold)
    else:
        n_sig = 0

    return TWASResult(
        genes=gene_results,
        n_genes_tested=n_tested,
        n_significant=n_sig,
    )


# ---------------------------------------------------------------------------
# Individual-level TWAS
# ---------------------------------------------------------------------------

def twas_individual(
    genotypes: Tensor,
    phenotype: Tensor,
    weights: dict[str, Tensor],
    snp_lists: dict[str, list[str]],
    snp_ids: list[str],
    covariates: Tensor | None = None,
) -> TWASResult:
    """Individual-level TWAS (PrediXcan).

    Imputes genetically regulated expression (GReX) and tests
    association with the phenotype via OLS.

    Parameters
    ----------
    genotypes : (n, m) Tensor
        Genotype matrix.
    phenotype : (n,) Tensor
        Phenotype vector.
    weights : dict[str, Tensor]
        Gene ID -> ``(k,)`` eQTL weight vector.
    snp_lists : dict[str, list[str]]
        Gene ID -> list of SNP IDs.
    snp_ids : list[str]
        Column IDs for the genotype matrix.
    covariates : (n, q) Tensor or None
        Covariates to regress out.

    Returns
    -------
    TWASResult
    """
    if len(weights) == 0:
        raise ValueError("At least one gene with weights is required.")

    genotypes = genotypes.to(torch.float64)
    phenotype = phenotype.to(torch.float64)
    n = genotypes.shape[0]

    snp_map = {s: i for i, s in enumerate(snp_ids)}

    # Residualize phenotype on covariates if provided
    y = phenotype
    if covariates is not None:
        C = covariates.to(torch.float64)
        # y_resid = y - C @ (C'C)^{-1} C'y
        CtC = C.t() @ C
        Cty = C.t() @ y
        coef = torch.linalg.solve(CtC, Cty)
        y = y - C @ coef

    gene_results: list[TWASGeneResult] = []

    for gene_id, w in weights.items():
        snps = snp_lists.get(gene_id, [])
        w = w.to(torch.float64)
        k = w.shape[0]

        if k == 0 or len(snps) != k:
            continue

        indices = []
        valid_pos = []
        for j, s in enumerate(snps):
            idx = snp_map.get(s)
            if idx is not None:
                indices.append(idx)
                valid_pos.append(j)

        if len(indices) == 0:
            continue

        idx_t = torch.tensor(indices, dtype=torch.long)
        X_cis = genotypes[:, idx_t]
        w_sub = w[valid_pos]

        # GReX = X_cis @ w
        grex = X_cis @ w_sub  # (n,)

        # OLS: y = alpha + beta * grex + epsilon
        # Design: [1, grex]
        X_design = torch.stack([torch.ones(n, dtype=torch.float64,
                                           device=genotypes.device),
                                grex], dim=1)
        XtX = X_design.t() @ X_design
        Xty = X_design.t() @ y
        try:
            beta_hat = torch.linalg.solve(XtX, Xty)
        except torch.linalg.LinAlgError:
            continue

        resid = y - X_design @ beta_hat
        sigma2 = float((resid @ resid).item()) / max(n - 2, 1)

        try:
            XtX_inv = torch.linalg.inv(XtX)
        except torch.linalg.LinAlgError:
            continue

        se_beta = (sigma2 * XtX_inv[1, 1]).clamp(min=0.0).sqrt()
        z_val = float(beta_hat[1].item()) / max(float(se_beta.item()), 1e-300)

        p_twas = 2.0 * float(_normal_sf(
            torch.tensor(abs(z_val), dtype=torch.float64)
        ).item())

        top_idx = int(w_sub.abs().argmax().item())
        top_snp = snps[valid_pos[top_idx]]

        gene_results.append(TWASGeneResult(
            gene_id=gene_id,
            z_twas=z_val,
            p_twas=p_twas,
            n_cis_snps=len(valid_pos),
            r2_model=None,
            top_weight_snp=top_snp,
        ))

    n_tested = len(gene_results)
    n_sig = sum(1 for g in gene_results if g.p_twas < 0.05)

    return TWASResult(
        genes=gene_results,
        n_genes_tested=n_tested,
        n_significant=n_sig,
    )


# ---------------------------------------------------------------------------
# Observed-expression TWAS (no pre-trained weights)
# ---------------------------------------------------------------------------

def twas_observed_expression(
    expression: Tensor,
    phenotype: Tensor,
    gene_ids: list[str],
    *,
    covariates: Tensor | None = None,
    kinship: Tensor | None = None,
    gene_annotation: dict[str, dict] | None = None,
    standardize: bool = False,
    correction: str = "bonferroni",
    p_threshold: float = 0.05,
    test: str = "wald",
) -> TWASResult:
    """Observed-expression TWAS: per-gene association on already-normalized
    expression, without pre-trained eQTL weights.

    For a discovery cohort with measured gene expression (typically RNA-seq
    → TMM/vst/quantile-normalized → optionally PEER-adjusted) and a
    phenotype, test each gene's expression-trait association directly.
    Mirrors the FUSION "measured-expression mode" (Gusev et al. 2016).

    Algorithm
    ---------
    Each gene column is treated as a quantitative predictor. The per-gene
    Wald test follows from:

    - OLS (no kinship)::

          y_resid = y - X0 @ (X0' X0)^{-1} X0' y
          e_resid = e_g - X0 @ (X0' X0)^{-1} X0' e_g
          β̂_g    = (e_resid' y_resid) / (e_resid' e_resid)
          T_g    = β̂_g^2 / Var(β̂_g)  ~  F(1, n - c - 1)

    - LMM (kinship provided)::

          V       = σ²_g K + σ²_e I
          β̂_g    = (e_g' V⁻¹ y) / (e_g' V⁻¹ e_g)
          T_g    = β̂_g^2 / Var(β̂_g)  ~  χ²(1)

    The OLS path reuses ``torchgwas.models.glm.GLM``; the LMM path reuses
    ``torchgwas.models.single_trait_lmm.SingleTraitLMM``. Both already
    handle the per-column algebra; the expression matrix simply
    substitutes for the genotype matrix.

    Parameters
    ----------
    expression : Tensor
        ``(n_samples, n_genes)`` matrix of *already-normalized* gene
        expression. The function does not apply any internal
        normalization — the caller supplies whatever pipeline output they
        have (RNA-seq counts → TMM/vst/quantile → PEER residuals etc.).
        See ``torchgwas.preprocess.expression`` for helper transforms if
        needed.
    phenotype : Tensor
        ``(n_samples,)`` or ``(n_samples, 1)`` quantitative phenotype.
    gene_ids : list[str]
        Length ``n_genes`` identifiers, in column order of ``expression``.
    covariates : Tensor, optional
        ``(n_samples, q)`` known covariates (genotype PCs, PEER factors,
        sex, age, batch). An intercept column is added automatically; the
        caller should not include one.
    kinship : Tensor, optional
        ``(n_samples, n_samples)`` GRM / kinship matrix. When supplied the
        scan switches from OLS to LMM with variance-component REML
        estimated once on the null; when omitted, ordinary OLS is used.
        Useful for population-stratification correction at biobank scale.
    gene_annotation : dict, optional
        Mapping ``gene_id -> {"chr": str, "start": int, "end": int,
        "gene_name": str, "biotype": str}``. Any subset of those keys may
        be present; missing fields are left as ``None`` on the result.
    standardize : bool
        If True, per-gene z-score the expression matrix before testing
        (mimics PrediXcan's column standardisation). Default False.
    correction : str
        Multiple testing correction: ``"bonferroni"``, ``"holm"``,
        ``"bh"`` (Benjamini–Hochberg), ``"by"`` (Benjamini–Yekutieli),
        ``"storey"`` (Storey q-value), or ``"none"``.
    p_threshold : float
        Significance threshold before correction.
    test : str
        Passed through to ``score_chunk``; ``"wald"`` is the only
        supported value for OLS, and ``"wald"``, ``"lrt"``, or ``"score"``
        for LMM.

    Returns
    -------
    TWASResult
        Each ``TWASGeneResult`` populates ``z_twas``, ``p_twas``, ``beta``,
        ``se``, and ``n_cis_snps = 0`` (no eQTL weights used). When
        ``gene_annotation`` is supplied, ``chr / start / end / gene_name``
        are propagated. ``top_weight_snp`` is always ``None`` in this
        mode.
    """
    # Lazy imports to avoid circular references.
    from ..models.glm import GLM
    from ..models.single_trait_lmm import SingleTraitLMM
    from ..models.base import VariantMeta

    if expression.ndim != 2:
        raise ValueError(
            f"expression must be 2-D (n_samples, n_genes); got shape "
            f"{tuple(expression.shape)}."
        )
    n_samples, n_genes = expression.shape
    if len(gene_ids) != n_genes:
        raise ValueError(
            f"len(gene_ids) = {len(gene_ids)} does not match "
            f"expression.shape[1] = {n_genes}."
        )
    if phenotype.shape[0] != n_samples:
        raise ValueError(
            f"phenotype length {phenotype.shape[0]} does not match "
            f"expression.shape[0] = {n_samples}."
        )
    if covariates is not None and covariates.shape[0] != n_samples:
        raise ValueError(
            f"covariates rows {covariates.shape[0]} does not match "
            f"expression.shape[0] = {n_samples}."
        )
    if kinship is not None:
        if kinship.shape != (n_samples, n_samples):
            raise ValueError(
                f"kinship shape {tuple(kinship.shape)} does not match "
                f"(n_samples, n_samples) = ({n_samples}, {n_samples})."
            )

    expression = expression.to(torch.float64)
    phenotype = phenotype.to(torch.float64).reshape(-1)

    # Standardise per-gene if requested.
    if standardize:
        mean = expression.mean(dim=0, keepdim=True)
        sd = expression.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-30)
        expression = (expression - mean) / sd

    # Build X0 = [1, covariates...] (intercept always present).
    intercept = torch.ones((n_samples, 1), dtype=torch.float64)
    if covariates is not None:
        X0 = torch.cat([intercept, covariates.to(torch.float64)], dim=1)
    else:
        X0 = intercept

    # Build a VariantMeta with gene_id playing the role of "snp".
    vmeta = VariantMeta(
        snp=list(gene_ids),
        chr=["0"] * n_genes,
        pos=list(range(n_genes)),
        a1=["A"] * n_genes,
        a2=["G"] * n_genes,
    )

    # Dispatch on whether kinship was provided.
    if kinship is None:
        # Full-model OLS per gene, matching the FUSION measured-expression /
        # PLINK convention. ``torchgwas.models.glm.GLM`` uses the null-only
        # σ²_e (GAPIT P3D=TRUE), which is conservative for strong-signal
        # genes — the gene's contribution to phenotype variance stays in
        # σ²_e, inflating SE. Refitting per gene via FWL recovers the
        # full-model variance estimator that the upstream TWAS tools
        # (PrediXcan, FUSION measured-expression mode, PLINK) report.
        n = expression.shape[0]
        c = X0.shape[1]
        df_full = n - c - 1
        XtX_X0_inv = torch.linalg.inv(X0.T @ X0)
        y_proj = X0 @ (XtX_X0_inv @ (X0.T @ phenotype))
        g_proj = X0 @ (XtX_X0_inv @ (X0.T @ expression))
        y_resid = phenotype - y_proj
        g_resid = expression - g_proj
        gtg = (g_resid * g_resid).sum(dim=0).clamp(min=1e-30)
        beta_t = (g_resid * y_resid.unsqueeze(1)).sum(dim=0) / gtg
        e_full = y_resid.unsqueeze(1) - beta_t.unsqueeze(0) * g_resid
        sse = (e_full ** 2).sum(dim=0)
        sigma2 = sse / df_full
        var_beta = sigma2 / gtg
        se_t = var_beta.clamp(min=1e-300).sqrt()
        stat_t = beta_t ** 2 / var_beta.clamp(min=1e-300)
        # F(1, df_full) p-values match the FUSION reference exactly.
        import scipy.stats as _sp_stats
        p_t = torch.tensor(
            _sp_stats.f.sf(stat_t.detach().cpu().numpy(), 1, df_full),
            dtype=torch.float64,
        )
        z_t = beta_t.sign() * stat_t.clamp(min=0.0).sqrt()
    else:
        # LMM path: keep the GAPIT / EMMAX / P3D=TRUE convention used by
        # SingleTraitLMM — REML estimates σ²_g, σ²_e once on the null,
        # then per-SNP Wald uses V = σ²_g K + σ²_e I. This is the
        # standard LMM-GWAS convention and is what biobank-scale TWAS
        # users expect for population-stratification correction.
        model = SingleTraitLMM()
        null_fit = model.fit_null(phenotype, X0, K=kinship.to(torch.float64))
        scan = model.score_chunk(expression, null_fit, vmeta, test=test)
        beta_t = scan.beta.detach().cpu().to(torch.float64)
        se_t = scan.se.detach().cpu().to(torch.float64)
        stat_t = scan.stat.detach().cpu().to(torch.float64)
        p_t = scan.p.detach().cpu().to(torch.float64)
        z_t = stat_t.clamp(min=0.0).sqrt() * beta_t.sign()

    annotation_lookup: dict[str, dict] = gene_annotation or {}

    gene_results: list[TWASGeneResult] = []
    for j, gene_id in enumerate(gene_ids):
        ann = annotation_lookup.get(gene_id, {})
        gene_results.append(TWASGeneResult(
            gene_id=gene_id,
            z_twas=float(z_t[j].item()),
            p_twas=float(p_t[j].item()),
            n_cis_snps=0,
            r2_model=None,
            top_weight_snp=None,
            beta=float(beta_t[j].item()),
            se=float(se_t[j].item()),
            chr=ann.get("chr"),
            start=ann.get("start"),
            end=ann.get("end"),
            gene_name=ann.get("gene_name"),
        ))

    # Multiple-testing correction.
    n_tested = len(gene_results)
    n_sig = 0
    if n_tested > 0:
        p_vals = torch.tensor(
            [g.p_twas for g in gene_results], dtype=torch.float64,
        )
        if correction in (None, "none"):
            n_sig = int((p_vals <= p_threshold).sum().item())
        elif correction == "bonferroni":
            n_sig = int((p_vals <= p_threshold / n_tested).sum().item())
        else:
            from ..stats import multipletesting as _mt
            if correction == "holm":
                p_adj = _mt.holm(p_vals)
            elif correction in ("bh", "fdr"):
                p_adj = _mt.benjamini_hochberg(p_vals)
            elif correction == "by":
                p_adj = _mt.benjamini_yekutieli(p_vals)
            elif correction == "storey":
                p_adj = _mt.storey_qvalue(p_vals)
            else:
                raise ValueError(
                    f"Unknown correction '{correction}'. Use one of: "
                    "'bonferroni', 'holm', 'bh', 'by', 'storey', 'none'."
                )
            n_sig = int((p_adj <= p_threshold).sum().item())

    return TWASResult(
        genes=gene_results,
        n_genes_tested=n_tested,
        n_significant=n_sig,
    )


# ---------------------------------------------------------------------------
# Multi-tissue stacking + S-MultiXcan-style aggregation
# ---------------------------------------------------------------------------

@dataclass
class MultiTissueRow:
    """One row of a multi-tissue TWAS result.

    Attributes
    ----------
    gene_id : str
    tissue : str
    z_twas : float
    p_twas : float
    beta : float or None
    se : float or None
    r2_model : float or None
    n_cis_snps : int
    gene_name : str or None
    chr : str or None
    start : int or None
    end : int or None
    """

    gene_id: str
    tissue: str
    z_twas: float
    p_twas: float
    beta: float | None
    se: float | None
    r2_model: float | None
    n_cis_snps: int
    gene_name: str | None
    chr: str | None
    start: int | None
    end: int | None


@dataclass
class MultiTissueSummary:
    """S-MultiXcan-style per-gene aggregation across tissues.

    Attributes
    ----------
    gene_id : str
    n_tissues : int
        Number of tissues with a non-null result for this gene.
    z_min, z_max, z_mean, z_sd : float
        Per-gene summary statistics across tissues' z-scores.
    p_min : float
        Minimum p-value across tissues (driver-tissue identification).
    chi2_multixcan : float
        Sum of squared z-scores across tissues (treated as χ²(n_tissues)
        under the working assumption of independence — see Barbeira
        et al. 2019; this is an approximation, not the full S-MultiXcan
        SVD-truncated joint test which would require the cross-tissue
        prediction correlation matrix).
    p_multixcan : float
        p-value of ``chi2_multixcan`` under χ²(n_tissues).
    driver_tissue : str
        Tissue carrying ``p_min``.
    """

    gene_id: str
    n_tissues: int
    z_min: float
    z_max: float
    z_mean: float
    z_sd: float
    p_min: float
    chi2_multixcan: float
    p_multixcan: float
    driver_tissue: str


def twas_multi_tissue_stack(
    per_tissue: dict[str, TWASResult],
) -> list[MultiTissueRow]:
    """Stack per-tissue ``TWASResult`` objects into a long-format list
    of one ``MultiTissueRow`` per (gene, tissue) pair.

    Useful when running ``twas_sumstat`` or ``twas_observed_expression``
    against each of N tissues (GTEx-style 49 tissues, PsychENCODE
    multi-brain, etc.) and you want a single tabular dump.

    Parameters
    ----------
    per_tissue : dict[str, TWASResult]
        ``tissue_name -> TWASResult`` mapping. The keys become the
        ``tissue`` column in the output rows.

    Returns
    -------
    list[MultiTissueRow]
        Long-format rows, sorted by ``(gene_id, tissue)`` ascending.
    """
    rows: list[MultiTissueRow] = []
    for tissue, result in per_tissue.items():
        for g in result.genes:
            rows.append(MultiTissueRow(
                gene_id=g.gene_id,
                tissue=tissue,
                z_twas=g.z_twas,
                p_twas=g.p_twas,
                beta=g.beta,
                se=g.se,
                r2_model=g.r2_model,
                n_cis_snps=g.n_cis_snps,
                gene_name=g.gene_name,
                chr=g.chr,
                start=g.start,
                end=g.end,
            ))
    rows.sort(key=lambda r: (r.gene_id, r.tissue))
    return rows


def twas_multi_tissue_aggregate(
    per_tissue: dict[str, TWASResult],
) -> list[MultiTissueSummary]:
    """Aggregate per-tissue TWAS results into one summary per gene.

    Computes a per-gene S-MultiXcan-style approximation:

        chi2_multixcan = sum_t (z_t^2)
        p_multixcan    = chi2_sf(chi2_multixcan, df=n_tissues)

    This treats per-tissue z-scores as independent, which is an
    approximation: the full S-MultiXcan (Barbeira et al. 2019,
    *PLoS Genet*) uses the SVD-truncated joint test driven by the
    cross-tissue prediction correlation matrix. The approximation
    serves as a fast first-pass for identifying genes with pleiotropic
    tissue signal; the full joint test can be applied downstream.

    Parameters
    ----------
    per_tissue : dict[str, TWASResult]
        ``tissue_name -> TWASResult``.

    Returns
    -------
    list[MultiTissueSummary]
        One summary per unique gene, sorted by ``p_multixcan`` ascending.
    """
    # Gather per-gene z-vectors across tissues.
    per_gene_z: dict[str, list[tuple[str, float, float]]] = {}
    for tissue, result in per_tissue.items():
        for g in result.genes:
            per_gene_z.setdefault(g.gene_id, []).append(
                (tissue, g.z_twas, g.p_twas)
            )

    summaries: list[MultiTissueSummary] = []
    import math
    from scipy.stats import chi2 as _chi2  # type: ignore[import-untyped]
    for gene_id, entries in per_gene_z.items():
        zs = [e[1] for e in entries]
        ps = [e[2] for e in entries]
        n_t = len(zs)
        if n_t == 0:
            continue
        z_arr = torch.tensor(zs, dtype=torch.float64)
        chi2_val = float((z_arr ** 2).sum().item())
        p_val = float(_chi2.sf(chi2_val, df=n_t))
        # Driver tissue = the one with minimum p.
        driver_idx = min(range(n_t), key=lambda i: ps[i])
        z_mean = float(z_arr.mean().item())
        z_sd = (
            float(z_arr.std(unbiased=False).item()) if n_t > 1 else 0.0
        )
        summaries.append(MultiTissueSummary(
            gene_id=gene_id,
            n_tissues=n_t,
            z_min=float(z_arr.min().item()),
            z_max=float(z_arr.max().item()),
            z_mean=z_mean,
            z_sd=z_sd,
            p_min=min(ps),
            chi2_multixcan=chi2_val,
            p_multixcan=p_val,
            driver_tissue=entries[driver_idx][0],
        ))
    summaries.sort(key=lambda s: s.p_multixcan)
    return summaries
