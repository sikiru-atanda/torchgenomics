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
        Number of cis-SNPs with non-zero weights.
    r2_model : float or None
        Cross-validated R-squared of the expression prediction model.
    top_weight_snp : str
        SNP with the largest absolute weight.
    """

    gene_id: str
    z_twas: float
    p_twas: float
    n_cis_snps: int
    r2_model: float | None
    top_weight_snp: str


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
