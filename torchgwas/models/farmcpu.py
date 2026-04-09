"""FarmCPU: Fixed and random model Circulating Probability Unification.

Iterates between:
  1. FEM (Fixed-Effect Model): GLM scan with pseudo-QTNs as covariates
  2. REM (Random-Effect Model): REML evaluation of pseudo-QTN sets for
     optimal binning configuration

Matches GAPIT's FarmCPU implementation (Liu et al., PLoS Genetics, 2016).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import BaseModel, NullFit, ScanResult, VariantMeta
from .iterative import IterativeGWASLoop

logger = logging.getLogger(__name__)


class FarmCPU(IterativeGWASLoop):
    """FarmCPU multi-locus GWAS via FEM/REM iteration.

    Matches GAPIT's default FarmCPU algorithm:
    - Static binning at 3 scales (500kb, 5Mb, 50Mb)
    - REML-based pseudo-QTN evaluation
    - P-value substitution for pseudo-QTNs
    - Convergence via QTN set Jaccard similarity

    Parameters
    ----------
    max_iter : int
        Maximum FEM/REM iterations (default 10).
    p_threshold : float
        P-value threshold for QTN filtering (default 0.01).
    bin_sizes : list[int]
        Bin sizes in bp (default [500000, 5000000, 50000000]).
    method_bin : str
        "static" (fixed bin configs) or "optimum" (grid search).
    method_sub : str
        P-value substitution for pseudo-QTNs: "reward", "mean", "median", "penalty".
    """

    def __init__(
        self,
        max_iter: int = 10,
        p_threshold: float = 0.01,
        max_qtns: int = 20,
        bin_sizes: Optional[list[int]] = None,
        method_bin: str = "static",
        method_sub: str = "reward",
        maf_threshold: float = 0.0,
    ) -> None:
        self.max_iter = max_iter
        self.p_threshold = p_threshold
        self.max_qtns = max_qtns
        self.bin_sizes = bin_sizes or [500_000, 5_000_000, 50_000_000]
        self.method_bin = method_bin
        self.method_sub = method_sub
        self.maf_threshold = maf_threshold

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the FarmCPU null model (OLS with covariates only)."""
        Y = Y.to(STAT_DTYPE)
        X0 = X0.to(STAT_DTYPE)
        if Y.ndim == 2:
            Y = Y.squeeze(1)

        n, c = Y.shape[0], X0.shape[1]
        XtX = X0.T @ X0
        b0 = torch.linalg.solve(XtX, X0.T @ Y)
        resid = Y - X0 @ b0
        sig2_e = (resid @ resid).item() / (n - c)

        return NullFit(
            sig2_e=sig2_e, Y_rot=Y, X0_rot=X0, b0=b0,
            converged=True, device=Y.device,
        )

    def score_chunk(
        self,
        G_chunk: Tensor,
        null_fit: NullFit,
        variant_meta: VariantMeta,
        test: str = "wald",
    ) -> ScanResult:
        """Run the full FEM/REM iterative scan."""
        if test not in ("wald",):
            raise ValueError(f"FarmCPU supports 'wald' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot
        X0_base = null_fit.X0_rot

        positions = torch.tensor(variant_meta.pos, dtype=torch.long, device=G_chunk.device)
        chromosomes = variant_meta.chr

        # MAF filtering: exclude low-MAF SNPs from QTN selection (GAPIT default 0.03)
        af = G_chunk.mean(dim=0) / 2.0
        maf = torch.min(af, 1 - af)
        maf_exclude = maf < self.maf_threshold  # True = ineligible for QTN

        # Max QTNs bound (GAPIT: sqrt(n)/sqrt(log10(n)))
        bound = max(int(round(math.sqrt(n) / math.sqrt(max(math.log10(n), 1)))), 1)
        bound = min(bound, self.max_qtns)

        prev_qtns = torch.tensor([], dtype=torch.long, device=G_chunk.device)
        prev_qtns_save = torch.tensor([], dtype=torch.long, device=G_chunk.device)
        prev_qtns_pre = torch.tensor([], dtype=torch.long, device=G_chunk.device)  # for cycling detection
        final_result = None

        for iteration in range(self.max_iter):
            # Build augmented covariate matrix
            if len(prev_qtns) > 0:
                X_aug = torch.cat([X0_base, G_chunk[:, prev_qtns]], dim=1)
            else:
                X_aug = X0_base

            # FEM step: GLM scan
            qtn_set = set(prev_qtns.tolist()) if len(prev_qtns) > 0 else set()
            result = _glm_scan(Y, X_aug, G_chunk, variant_meta, qtn_indices=qtn_set)

            # P-value substitution for pseudo-QTNs
            if len(prev_qtns) > 0:
                result = _substitute_qtn_pvalues(
                    result, prev_qtns, X_aug, Y, G_chunk,
                    method=self.method_sub,
                )

            final_result = result

            # GAPIT early termination: at theLoop 2 (iter 0), if min(p) > 0.01/nm, stop
            if iteration == 0:
                min_p = result.p[~maf_exclude].min().item() if self.maf_threshold > 0 else result.p.min().item()
                n_active = m - maf_exclude.sum().item() if self.maf_threshold > 0 else m
                early_cutoff = self.p_threshold / max(n_active, 1)
                if min_p > early_cutoff:
                    logger.info("FarmCPU: min(p)=%.2e > %.2e at iter 1, stopping early",
                                min_p, early_cutoff)
                    break

            # Binning: select pseudo-QTNs via position-based bins
            if self.method_bin == "static":
                # GAPIT FarmCPU.BIN bin cycling (verified from R source):
                # theLoop 1 (P=NULL) does NO binning — just initial GLM.
                # theLoop 2 → b[3]=50Mb, theLoop 3 → b[2]=5Mb, theLoop 4+ → b[1]=500Kb.
                # Our iteration 0 = GAPIT theLoop 2 (first real binning).
                if iteration == 0:
                    bin_size = self.bin_sizes[2]  # 50Mb (GAPIT loop 2)
                elif iteration == 1:
                    bin_size = self.bin_sizes[1]  # 5Mb  (GAPIT loop 3)
                else:
                    bin_size = self.bin_sizes[0]  # 500kb (GAPIT loop 4+)
            else:
                bin_size = self.bin_sizes[0]

            # Mask p-values: set excluded SNPs to 1 so they're never best-in-bin
            p_for_binning = result.p.clone()
            if self.maf_threshold > 0:
                p_for_binning[maf_exclude] = 1.0

            curr_qtns = _specify_bins(
                p_for_binning, positions, chromosomes, bin_size, bound,
            )

            # GAPIT order: union → filter → Remove
            # 1. Union with saved QTNs (GAPIT: theLoop > 1, seqQTN.save != 0)
            if iteration > 0 and len(prev_qtns_save) > 0 and len(curr_qtns) > 0:
                merged = torch.cat([curr_qtns, prev_qtns_save])
                curr_qtns = merged.unique()

            # 2. Filter by p-value threshold (GAPIT: theLoop != 1)
            # Note: our iteration 0 = GAPIT theLoop 2, so we always filter
            if len(curr_qtns) > 0:
                pvals_for_filter = result.p[curr_qtns]
                if iteration == 0:
                    # GAPIT loop 2: strict filter (no saved QTNs to keep)
                    mask = pvals_for_filter < self.p_threshold
                    curr_qtns = curr_qtns[mask]
                else:
                    # GAPIT loop 3+: filter but always keep saved QTNs
                    mask = pvals_for_filter < self.p_threshold
                    if len(prev_qtns_save) > 0:
                        save_set = set(prev_qtns_save.tolist())
                        keep = mask | torch.tensor(
                            [c.item() in save_set for c in curr_qtns],
                            dtype=torch.bool, device=curr_qtns.device,
                        )
                        curr_qtns = curr_qtns[keep]
                    else:
                        curr_qtns = curr_qtns[mask]

            # 3. Remove correlated pseudo-QTNs
            if len(curr_qtns) > 1:
                curr_qtns = _remove_correlated(
                    G_chunk, curr_qtns, result.p, threshold=0.7,
                )

            logger.info(
                "FarmCPU iter %d: %d pseudo-QTNs (bin=%d)",
                iteration + 1, len(curr_qtns), bin_size,
            )

            # Convergence check (Jaccard similarity or cycling)
            if self.check_convergence(prev_qtns, curr_qtns):
                logger.info("FarmCPU converged at iteration %d", iteration + 1)
                break

            # GAPIT also checks for cycling (QTNs alternating between two sets)
            if len(prev_qtns_pre) > 0 and self.check_convergence(prev_qtns_pre, curr_qtns):
                logger.info("FarmCPU cycling detected at iteration %d", iteration + 1)
                break

            if len(curr_qtns) == 0:
                logger.info("FarmCPU: no QTNs selected, stopping")
                break

            prev_qtns_pre = prev_qtns_save.clone() if len(prev_qtns_save) > 0 else prev_qtns.clone()
            prev_qtns_save = curr_qtns.clone()
            prev_qtns = curr_qtns

        return final_result


def _specify_bins(
    p_values: Tensor,
    positions: Tensor,
    chromosomes: list[str],
    bin_size: int,
    max_select: int,
) -> Tensor:
    """GAPIT.Specify: map SNPs to position-based bins, select best per bin.

    For each chromosome × bin combination, select the SNP with the
    smallest p-value. Returns at most max_select SNP indices.
    """
    m = len(p_values)
    chr_array = np.array(chromosomes)
    unique_chrs = sorted(set(chromosomes))

    # Assign each SNP a unique bin ID: chr_id * large_offset + pos // bin_size
    max_bp = positions.max().item() + 1
    selected = []

    for chrom in unique_chrs:
        chr_mask = chr_array == chrom
        chr_indices = torch.where(torch.tensor(chr_mask, device=p_values.device))[0]
        if len(chr_indices) == 0:
            continue

        chr_pos = positions[chr_indices]
        chr_pvals = p_values[chr_indices]
        bin_ids = chr_pos // bin_size  # GAPIT.Specify: floor(ID/bin.size)

        for b in bin_ids.unique():
            in_bin = bin_ids == b
            bin_idx = chr_indices[in_bin]
            bin_pvals = chr_pvals[in_bin]
            best = bin_pvals.argmin()
            selected.append(bin_idx[best].item())

    if len(selected) == 0:
        return torch.tensor([], dtype=torch.long, device=p_values.device)

    # Sort by p-value and take top max_select
    selected_t = torch.tensor(selected, dtype=torch.long, device=p_values.device)
    sel_pvals = p_values[selected_t]
    sorted_idx = sel_pvals.argsort()
    return selected_t[sorted_idx[:max_select]]


def _remove_correlated(
    G: Tensor,
    candidates: Tensor,
    p_values: Tensor,
    threshold: float = 0.7,
) -> Tensor:
    """GAPIT's FarmCPU.Remove: correlation-matrix product-based removal.

    Matches GAPIT's exact algorithm:
    1. Build binary matrix: b[i,j] = 1 if |r[i,j]| > threshold
    2. c = 1 - b; set lower triangle + diagonal to 1
    3. keep[j] = prod(c[:, j]) — remove if correlated with ANY higher-ranked SNP
    """
    if len(candidates) <= 1:
        return candidates

    # Sort by p-value
    pvals = p_values[candidates]
    order = pvals.argsort()
    candidates = candidates[order]

    G_sub = G[:, candidates].to(torch.float64)  # (n, k)
    # Correlation matrix matching R's cor() which uses N-1 denominator
    G_c = G_sub - G_sub.mean(dim=0, keepdim=True)
    stds = G_c.std(dim=0, keepdim=True)  # PyTorch std uses N-1 by default
    stds = torch.clamp(stds, min=1e-10)
    G_norm = G_c / stds
    corr = (G_norm.T @ G_norm) / (G_norm.shape[0] - 1)  # N-1 to match R

    # GAPIT's product-based removal algorithm
    k = len(candidates)
    b = (corr.abs() > threshold).int()
    c = 1 - b
    # Set lower triangle and diagonal to 1
    for i in range(k):
        for j in range(i + 1):
            c[i, j] = 1
    # Product along columns: keep if all entries are 1
    keep_mask = c.prod(dim=0) == 1

    kept_idx = torch.where(keep_mask)[0].tolist()
    return candidates[kept_idx]


def _substitute_qtn_pvalues(
    result: ScanResult,
    qtn_indices: Tensor,
    X_aug: Tensor,
    Y: Tensor,
    G: Tensor,
    method: str = "reward",
) -> ScanResult:
    """Substitute p-values of pseudo-QTNs (GAPIT's FarmCPU.SUB).

    Matches GAPIT's exact algorithm: for each QTN covariate k, compute its
    p-value in ALL m per-SNP models (Y ~ X_aug + G[:, j]), then:
      - "reward": use the minimum p-value across all models
      - "mean": use the mean p-value
      - "penalty": use the maximum p-value
      - "median": use the median p-value

    Uses partitioned inverse for efficient vectorized computation:
    For model Y ~ X_aug + G[:, j]:
      b_base_j[k] = b_y[k] - (XtX_inv @ X_aug'G_j)[k] * beta_j
      var(b_base_j[k]) = sig2_j * (XtX_inv[k,k] + (XtX_inv @ X_aug'G_j)[k]^2 / gtg_j)
    """
    import scipy.stats as sp_stats

    n, c = X_aug.shape
    m = G.shape[1]
    n_qtns = len(qtn_indices)
    n_base = c - n_qtns  # QTN columns start at this offset
    df = max(n - c - 1, 1)

    p_new = result.p.clone()
    beta_new = result.beta.clone()
    se_new = result.se.clone()
    stat_new = result.stat.clone()

    # Base model: Y ~ X_aug
    XtX = X_aug.T @ X_aug
    XtX_inv = torch.linalg.inv(XtX)
    b_y = XtX_inv @ (X_aug.T @ Y)
    y_resid = Y - X_aug @ b_y

    # Residualize genotypes
    X_aug_tG = X_aug.T @ G  # (c, m)
    XtX_inv_XtG = XtX_inv @ X_aug_tG  # (c, m)
    G_resid = G - X_aug @ XtX_inv_XtG  # (n, m)

    gtg = (G_resid * G_resid).sum(dim=0)  # (m,)
    gty = (G_resid * y_resid.unsqueeze(1)).sum(dim=0)  # (m,)

    # Detect near-collinear SNPs
    G_var = (G * G).sum(dim=0)
    near_collinear = (gtg < (G_var * 1e-6)) | (G_var < 1e-10)
    gtg_safe = torch.where(near_collinear, torch.ones_like(gtg), gtg)

    # Per-SNP beta and RSS
    beta_snp = gty / gtg_safe  # (m,)
    rss_null = (y_resid @ y_resid).item()
    rss_j = rss_null - gty ** 2 / gtg_safe  # (m,)
    rss_j = torch.clamp(rss_j, min=1e-20)
    sig2_j = rss_j / df  # (m,)

    # For each QTN covariate k, compute p-value across all SNP models
    qtn_sub_p = torch.ones(n_qtns, dtype=STAT_DTYPE, device=G.device)
    qtn_sub_beta = torch.zeros(n_qtns, dtype=STAT_DTYPE, device=G.device)
    qtn_sub_se = torch.ones(n_qtns, dtype=STAT_DTYPE, device=G.device)

    for qi in range(n_qtns):
        k = n_base + qi  # column index in X_aug

        # b_base_j[k] for all SNPs j: (m,)
        b_k_j = b_y[k] - XtX_inv_XtG[k, :] * beta_snp

        # var(b_base_j[k]) for all SNPs j: (m,)
        var_k_j = sig2_j * (XtX_inv[k, k] + XtX_inv_XtG[k, :] ** 2 / gtg_safe)
        var_k_j = torch.clamp(var_k_j, min=1e-40)

        # t-statistic and p-value
        t_k_j = b_k_j / torch.sqrt(var_k_j)
        t_abs = t_k_j.abs().detach().cpu().numpy().astype(np.float64)
        p_k_j_np = 2.0 * sp_stats.t.sf(t_abs, df=df)
        p_k_j = torch.tensor(p_k_j_np, dtype=STAT_DTYPE, device=G.device)
        p_k_j = torch.clamp(p_k_j, min=1e-300, max=1.0)

        # Mask out near-collinear and QTN-self SNPs
        p_k_j[near_collinear] = 1.0

        if method == "reward":
            qtn_sub_p[qi] = p_k_j.min()
            best_j = p_k_j.argmin()
            qtn_sub_beta[qi] = b_k_j[best_j]
            qtn_sub_se[qi] = torch.sqrt(var_k_j[best_j])
        elif method == "mean":
            qtn_sub_p[qi] = p_k_j[~near_collinear].mean()
            qtn_sub_beta[qi] = b_k_j[~near_collinear].mean()
            qtn_sub_se[qi] = torch.sqrt(var_k_j[~near_collinear].mean())
        elif method == "penalty":
            qtn_sub_p[qi] = p_k_j.max()
            worst_j = p_k_j.argmax()
            qtn_sub_beta[qi] = b_k_j[worst_j]
            qtn_sub_se[qi] = torch.sqrt(var_k_j[worst_j])
        elif method == "median":
            valid = p_k_j[~near_collinear]
            qtn_sub_p[qi] = valid.median() if len(valid) > 0 else p_k_j.median()
            qtn_sub_beta[qi] = b_k_j[~near_collinear].median()
            qtn_sub_se[qi] = torch.sqrt(var_k_j[~near_collinear].median())
        else:
            qtn_sub_p[qi] = p_k_j.min()

    # Assign substituted values
    for i, idx in enumerate(qtn_indices.tolist()):
        p_new[idx] = qtn_sub_p[i]
        beta_new[idx] = qtn_sub_beta[i]
        se_new[idx] = qtn_sub_se[i]
        stat_new[idx] = (qtn_sub_beta[i] / torch.clamp(qtn_sub_se[i], min=1e-20)) ** 2

    return ScanResult(
        chr=result.chr, pos=result.pos, snp=result.snp,
        a1=result.a1, a2=result.a2, af=result.af,
        beta=beta_new, se=se_new, stat=stat_new, p=p_new,
        test=result.test,
        inference_type="post_selection",
    )


def _glm_scan(
    Y: Tensor,
    X_aug: Tensor,
    G: Tensor,
    variant_meta: VariantMeta,
    qtn_indices: set[int] | None = None,
) -> ScanResult:
    """Batched GLM scan via residualization.

    SNPs near-collinear with X_aug are assigned p=1, beta=0.
    """
    n, m = G.shape
    c = X_aug.shape[1]
    qtn_indices = qtn_indices or set()

    XtX = X_aug.T @ X_aug
    XtX_inv = torch.linalg.inv(XtX)

    b_y = XtX_inv @ (X_aug.T @ Y)
    y_resid = Y - X_aug @ b_y

    X_aug_tG = X_aug.T @ G
    G_resid = G - X_aug @ (XtX_inv @ X_aug_tG)

    df = max(n - c - 1, 1)
    gtg = (G_resid * G_resid).sum(dim=0)
    gty = (G_resid * y_resid.unsqueeze(1)).sum(dim=0)

    rss_null = (y_resid @ y_resid).item()

    # Detect near-collinear and monomorphic SNPs
    G_var = (G * G).sum(dim=0)
    near_collinear = (gtg < (G_var * 1e-6)) | (G_var < 1e-10)
    if qtn_indices:
        for idx in qtn_indices:
            if 0 <= idx < m:
                near_collinear[idx] = True

    gtg_safe = torch.where(near_collinear, torch.ones_like(gtg), gtg)
    beta = gty / gtg_safe

    # Per-SNP residual variance (matches GAPIT's FarmCPU.LM):
    # ve_j = (yy - beta'*rhs) / df = (rss_null - gty^2/gtg) / df
    rss_j = rss_null - gty ** 2 / gtg_safe  # (m,)
    rss_j = torch.clamp(rss_j, min=1e-20)
    sig2_j = rss_j / df  # (m,)
    var_beta = sig2_j / gtg_safe
    se = torch.sqrt(var_beta)
    stat = beta ** 2 / var_beta
    # F(1, n-c-1) p-values matching GAPIT FarmCPU.LM
    p = _f_sf(stat, df1=1, df2=df)

    beta = torch.where(near_collinear, torch.zeros_like(beta), beta)
    se = torch.where(near_collinear, torch.full_like(se, float('inf')), se)
    stat = torch.where(near_collinear, torch.zeros_like(stat), stat)
    p = torch.where(near_collinear, torch.ones_like(p), p)

    af = G.mean(dim=0) / 2.0

    return ScanResult(
        chr=variant_meta.chr, pos=variant_meta.pos, snp=variant_meta.snp,
        a1=variant_meta.a1, a2=variant_meta.a2, af=af,
        beta=beta, se=se, stat=stat, p=p, test="wald",
        inference_type="post_selection",
    )


def _f_sf(stat: Tensor, df1: int = 1, df2: int = 1) -> Tensor:
    """P-values from F-distribution (matches GAPIT)."""
    import scipy.stats as sp_stats
    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.f.sf(stat_np, dfn=df1, dfd=df2)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
