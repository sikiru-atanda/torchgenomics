"""MAGMA-style gene-set / pathway enrichment analysis.

Implements two-stage competitive gene-set analysis following de Leeuw et al.
(2015, PLoS Computational Biology) "MAGMA: Generalized Gene-Set Analysis of
GWAS Data":

**Stage 1 -- SNP-to-gene aggregation** (``snp_to_gene``):

    For each gene g with k_g SNPs, compute the mean chi-squared statistic:

        T_g = mean(chi2_j)   for j in gene g

    Under the null of no association, chi2_j ~ chi2(1) with mean 1 and
    variance 2.  For k_g independent SNPs the mean chi-squared has
    approximate distribution Normal(1, 2/k_g), giving the one-sided z-test:

        z_g = (T_g - 1) / sqrt(2 / k_g)
        p_g = 1 - Phi(z_g)

**Stage 2 -- competitive gene-set test** (``gene_set_enrichment``):

    Convert gene p-values to z-scores via the probit transform:

        z_g = Phi^{-1}(1 - p_g)

    For each gene set S, fit OLS:

        z_g = beta_0 + beta_1 * I(g in S) + covariates + epsilon

    where covariates optionally include gene size (n_snps) and
    log(n_snps + 1) to control for confounding by gene length.
    beta_1 > 0 indicates enrichment.  A one-sided t-test yields the
    enrichment p-value.

All numeric computation uses ``torch.Tensor`` with float64 arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

# Polyploid compatibility: snp_to_gene uses chi2 = (beta/se)^2 which is
# ploidy-agnostic (the GWAS scan produces valid z-statistics for any ploidy).
# gene_set_enrichment operates on probit-transformed gene p-values,
# equally valid for diploid and polyploid GWAS output.
from ._sumstats import SumStats


# ---------------------------------------------------------------------------
# Normal CDF / survival / quantile helpers (pure torch, no scipy dependency)
# ---------------------------------------------------------------------------

def _normal_cdf(x: Tensor) -> Tensor:
    """Standard normal CDF: Phi(x) = 0.5 * erfc(-x / sqrt(2))."""
    return 0.5 * torch.erfc(-x / (2.0 ** 0.5))


def _normal_sf(x: Tensor) -> Tensor:
    """Standard normal survival function: 1 - Phi(x)."""
    return 0.5 * torch.erfc(x / (2.0 ** 0.5))


def _probit(p: Tensor) -> Tensor:
    """Probit (inverse normal CDF) via ``torch.special.ndtri``.

    Falls back to ``scipy.stats.norm.ppf`` if ``ndtri`` is unavailable
    (torch < 1.13).
    """
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(p)
    # Fallback for older torch versions.
    from scipy.stats import norm  # type: ignore[import-untyped]

    import numpy as np

    return torch.tensor(
        norm.ppf(p.detach().cpu().numpy()),
        dtype=p.dtype,
        device=p.device,
    )


# ---------------------------------------------------------------------------
# Stage 1: SNP-to-gene aggregation
# ---------------------------------------------------------------------------

@dataclass
class GeneResult:
    """Per-gene association test result.

    Attributes
    ----------
    gene_id : list[str]
        Gene identifiers in the same order as the input gene list.
    gene_chr : list[str]
        Chromosome for each gene.
    gene_start : list[int]
        Start position (bp) for each gene.
    gene_end : list[int]
        End position (bp) for each gene.
    n_snps : list[int]
        Number of SNPs assigned to each gene.
    stat : Tensor
        ``(n_genes,)`` gene-level test statistic (mean chi-squared).
    p : Tensor
        ``(n_genes,)`` gene-level p-value.
    """

    gene_id: list[str]
    gene_chr: list[str]
    gene_start: list[int]
    gene_end: list[int]
    n_snps: list[int]
    stat: Tensor
    p: Tensor


def snp_to_gene(
    ss: SumStats,
    gene_id: list[str],
    gene_chr: list[str],
    gene_start: list[int],
    gene_end: list[int],
    window_kb: float = 0.0,
) -> GeneResult:
    """Assign SNPs to genes and compute gene-level association statistics.

    Uses the MAGMA SNP-wise (mean chi-squared) model.  Each gene's test
    statistic is the mean chi-squared of the SNPs falling within
    ``[gene_start - window, gene_end + window]`` on the matching chromosome.

    Parameters
    ----------
    ss : SumStats
        GWAS summary statistics with ``chr``, ``pos``, and ``p`` fields.
    gene_id : list[str]
        Gene identifiers (length ``n_genes``).
    gene_chr : list[str]
        Chromosome for each gene (length ``n_genes``).
    gene_start : list[int]
        Gene start position in bp (length ``n_genes``).
    gene_end : list[int]
        Gene end position in bp (length ``n_genes``).
    window_kb : float
        Extend gene boundaries by this many kilobases on each side
        (default 0, i.e. no extension).

    Returns
    -------
    GeneResult
        Per-gene test statistics and p-values.

    Raises
    ------
    ValueError
        If gene coordinate lists have mismatched lengths.
    """
    n_genes = len(gene_id)
    if not (len(gene_chr) == n_genes and len(gene_start) == n_genes
            and len(gene_end) == n_genes):
        raise ValueError(
            "gene_id, gene_chr, gene_start, and gene_end must have the "
            f"same length. Got {n_genes}, {len(gene_chr)}, "
            f"{len(gene_start)}, {len(gene_end)}."
        )

    window_bp = int(window_kb * 1000)
    chi2 = ss.chi2.to(dtype=torch.float64)  # (m,)

    # Build a chromosome -> list of (pos, snp_index) lookup for fast matching.
    chr_to_snps: dict[str, list[tuple[int, int]]] = {}
    for i, (c, p) in enumerate(zip(ss.chr, ss.pos)):
        chr_to_snps.setdefault(str(c), []).append((p, i))

    # Sort each chromosome's SNPs by position for binary-search assignment.
    for c in chr_to_snps:
        chr_to_snps[c].sort()

    stat_list: list[float] = []
    p_list: list[float] = []
    n_snps_list: list[int] = []

    for g in range(n_genes):
        gc = str(gene_chr[g])
        gstart = gene_start[g] - window_bp
        gend = gene_end[g] + window_bp

        snps_on_chr = chr_to_snps.get(gc, [])
        # Collect indices of SNPs in the gene window.
        indices: list[int] = []
        for pos_val, snp_idx in snps_on_chr:
            if pos_val < gstart:
                continue
            if pos_val > gend:
                break
            indices.append(snp_idx)

        k = len(indices)
        if k == 0:
            stat_list.append(0.0)
            p_list.append(1.0)
            n_snps_list.append(0)
        else:
            idx_t = torch.tensor(indices, dtype=torch.long, device=chi2.device)
            gene_chi2 = chi2[idx_t]
            mean_chi2 = gene_chi2.mean()
            # z = (mean_chi2 - 1) / sqrt(2 / k), one-sided right-tail p-value.
            z = (mean_chi2 - 1.0) / (2.0 / k) ** 0.5
            pval = _normal_sf(z)
            stat_list.append(mean_chi2.item())
            p_list.append(pval.item())
            n_snps_list.append(k)

    device = chi2.device
    stat_t = torch.tensor(stat_list, dtype=torch.float64, device=device)
    p_t = torch.tensor(p_list, dtype=torch.float64, device=device)

    return GeneResult(
        gene_id=list(gene_id),
        gene_chr=list(gene_chr),
        gene_start=list(gene_start),
        gene_end=list(gene_end),
        n_snps=n_snps_list,
        stat=stat_t,
        p=p_t,
    )


# ---------------------------------------------------------------------------
# Stage 2: competitive gene-set enrichment
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    """Gene-set enrichment test result.

    Attributes
    ----------
    gene_set_name : list[str]
        Names of the tested gene sets.
    n_genes_in_set : list[int]
        Number of genes from ``GeneResult`` found in each set.
    n_genes_total : int
        Total number of genes used in the regression.
    beta_enrichment : Tensor
        ``(n_sets,)`` enrichment coefficient (beta_1 from the competitive
        regression).
    se : Tensor
        ``(n_sets,)`` standard error of beta_1.
    p : Tensor
        ``(n_sets,)`` one-sided p-value for enrichment (beta_1 > 0).
    """

    gene_set_name: list[str]
    n_genes_in_set: list[int]
    n_genes_total: int
    beta_enrichment: Tensor
    se: Tensor
    p: Tensor


def gene_set_enrichment(
    gene_result: GeneResult,
    gene_sets: dict[str, list[str]],
    covariate_gene_size: bool = True,
    covariate_log_size: bool = True,
) -> EnrichmentResult:
    """Competitive gene-set enrichment test (MAGMA-style).

    Converts gene-level p-values to probit z-scores and tests whether genes
    in each set have systematically higher z-scores than background genes,
    controlling for gene size.

    Parameters
    ----------
    gene_result : GeneResult
        Output of :func:`snp_to_gene`.
    gene_sets : dict[str, list[str]]
        Mapping from set name to a list of gene IDs belonging to that set.
        Gene IDs not present in ``gene_result`` are silently ignored.
    covariate_gene_size : bool
        Include ``n_snps`` as a covariate (default True).
    covariate_log_size : bool
        Include ``log(n_snps + 1)`` as a covariate (default True).

    Returns
    -------
    EnrichmentResult
        Per-set enrichment statistics.

    Raises
    ------
    ValueError
        If fewer than 2 genes have SNPs, or no gene sets are provided.
    """
    if len(gene_sets) == 0:
        raise ValueError("At least one gene set is required.")

    # Filter to genes with at least one SNP.
    valid_mask = [n > 0 for n in gene_result.n_snps]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    n_genes = len(valid_indices)

    if n_genes < 2:
        raise ValueError(
            f"At least 2 genes with assigned SNPs are required, got {n_genes}."
        )

    device = gene_result.p.device

    # Gene IDs and p-values for valid genes.
    valid_ids = [gene_result.gene_id[i] for i in valid_indices]
    valid_p = gene_result.p[torch.tensor(valid_indices, dtype=torch.long,
                                          device=device)]
    valid_n_snps = torch.tensor(
        [gene_result.n_snps[i] for i in valid_indices],
        dtype=torch.float64,
        device=device,
    )

    # Probit transform: z_g = Phi^{-1}(1 - p_g).
    # Clamp p to avoid infinite z at p=0 or p=1.
    one_minus_p = (1.0 - valid_p).clamp(1e-300, 1.0 - 1e-15)
    z = _probit(one_minus_p)  # (n_genes,)

    # Build gene-id -> local-index map.
    id_to_idx = {gid: i for i, gid in enumerate(valid_ids)}

    # Build covariate matrix: [intercept, (gene_size), (log_gene_size)].
    covariates = [torch.ones(n_genes, 1, dtype=torch.float64, device=device)]
    if covariate_gene_size:
        covariates.append(valid_n_snps.unsqueeze(1))
    if covariate_log_size:
        covariates.append(torch.log(valid_n_snps + 1.0).unsqueeze(1))

    X_base = torch.cat(covariates, dim=1)  # (n_genes, p_base)

    set_names: list[str] = []
    betas: list[float] = []
    ses: list[float] = []
    pvals: list[float] = []
    n_in_set: list[int] = []

    for set_name, set_genes in gene_sets.items():
        # Build indicator vector for this set.
        indicator = torch.zeros(n_genes, 1, dtype=torch.float64, device=device)
        count = 0
        for gid in set_genes:
            idx = id_to_idx.get(gid)
            if idx is not None:
                indicator[idx, 0] = 1.0
                count += 1

        set_names.append(set_name)
        n_in_set.append(count)

        if count == 0 or count == n_genes:
            # Degenerate: no contrast possible.
            betas.append(0.0)
            ses.append(float("inf"))
            pvals.append(1.0)
            continue

        # Design matrix: [X_base, indicator].
        X = torch.cat([X_base, indicator], dim=1)  # (n_genes, p_base + 1)

        # OLS: beta_hat = (X'X)^{-1} X'z
        XtX = X.t() @ X
        Xtz = X.t() @ z
        try:
            beta_hat = torch.linalg.solve(XtX, Xtz)
        except torch.linalg.LinAlgError:
            # Singular design -- skip this set.
            betas.append(0.0)
            ses.append(float("inf"))
            pvals.append(1.0)
            continue

        # Residual variance.
        residuals = z - X @ beta_hat
        p_cols = X.shape[1]
        sigma2 = (residuals @ residuals) / max(n_genes - p_cols, 1)

        # Standard error of the last coefficient (the set indicator).
        try:
            XtX_inv = torch.linalg.inv(XtX)
        except torch.linalg.LinAlgError:
            betas.append(0.0)
            ses.append(float("inf"))
            pvals.append(1.0)
            continue

        var_beta1 = sigma2 * XtX_inv[-1, -1]
        se_beta1 = var_beta1.clamp(min=0.0).sqrt()

        beta1 = beta_hat[-1].item()
        se_val = se_beta1.item()
        betas.append(beta1)
        ses.append(se_val)

        # One-sided p-value: Pr(t > observed) under t(n - p).
        # For large n, approximate with normal.
        if se_val > 0.0:
            t_stat = torch.tensor(beta1 / se_val, dtype=torch.float64,
                                  device=device)
            pval = _normal_sf(t_stat).item()
        else:
            pval = 1.0
        pvals.append(pval)

    beta_t = torch.tensor(betas, dtype=torch.float64, device=device)
    se_t = torch.tensor(ses, dtype=torch.float64, device=device)
    p_t = torch.tensor(pvals, dtype=torch.float64, device=device)

    return EnrichmentResult(
        gene_set_name=set_names,
        n_genes_in_set=n_in_set,
        n_genes_total=n_genes,
        beta_enrichment=beta_t,
        se=se_t,
        p=p_t,
    )
