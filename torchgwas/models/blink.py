"""BLINK: Bayesian-information and Linkage-disequilibrium Iteratively Nested Keyway.

Iterates between:
  1. FEM scan with pseudo-QTNs as covariates (FarmCPU.LM)
  2. LD-based removal via correlation matrix (Blink.LDRemoveBlock)
  3. BIC-based forward selection of pseudo-QTNs (Blink.BICselection)

No kinship matrix required — linear in n.
Matches GAPIT's BLINK implementation (Huang et al., GigaScience, 2019).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta
from .iterative import IterativeGWASLoop

logger = logging.getLogger(__name__)


class BLINK(IterativeGWASLoop):
    """BLINK multi-locus GWAS via LD-clustering and BIC selection.

    Matches GAPIT's default BLINK algorithm:
    - Bonferroni/FDR threshold at iteration 1, 1/m thereafter (GAPIT default)
    - Correlation-matrix LD removal (Blink.LDRemoveBlock)
    - BIC forward selection (Blink.BICselection "naive")
    - P-value substitution for pseudo-QTNs (Blink.SUB)
    - Convergence via QTN set Jaccard similarity

    Parameters
    ----------
    max_iter : int
        Maximum iterations (default 10).
    cutoff : float
        Base significance cutoff (default 0.01). At iteration 1, Bonferroni
        = cutoff/m is used. At iterations >1, 1/m is used.
    max_qtns : int or None
        Maximum number of pseudo-QTNs. If None, uses floor(n/log(n)).
    ld_threshold : float
        |r| threshold for LD removal (default 0.7).
    ld_max_samples : int
        If n > this, subsample for correlation (GAPIT default 200).
    method_sub : str
        P-value substitution method: "reward", "mean", "median", "penalty".
    """

    def __init__(
        self,
        max_iter: int = 10,
        cutoff: float = 0.01,
        max_qtns: int | None = None,
        ld_threshold: float = 0.7,
        ld_max_samples: int = 200,
        method_sub: str = "reward",
        maf_threshold: float = 0.0,
        # Legacy parameter name
        p_threshold: float | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.cutoff = cutoff
        self._max_qtns_user = max_qtns
        self.ld_threshold = ld_threshold
        self.ld_max_samples = ld_max_samples
        self.method_sub = method_sub
        self.maf_threshold = maf_threshold
        if p_threshold is not None:
            self.cutoff = p_threshold

    def fit_null(
        self,
        Y: Tensor,
        X0: Tensor,
        K: Tensor | None = None,
        **kwargs: Any,
    ) -> NullFit:
        """Fit the BLINK null model (OLS with covariates only)."""
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
        """Run the full BLINK iterative scan."""
        if test not in ("wald",):
            raise ValueError(f"BLINK supports 'wald' test only, got '{test}'.")

        G_chunk = G_chunk.to(STAT_DTYPE)
        n, m = G_chunk.shape

        Y = null_fit.Y_rot
        X0_base = null_fit.X0_rot

        # MAF filtering: exclude low-MAF SNPs from QTN selection (GAPIT default 0.03)
        af = G_chunk.mean(dim=0) / 2.0
        maf = torch.min(af, 1 - af)
        maf_exclude = maf < self.maf_threshold  # True = ineligible for QTN

        # Max QTNs bound (GAPIT: floor(n / log(n)))
        if self._max_qtns_user is not None:
            bound = self._max_qtns_user
        else:
            bound = max(int(math.floor(n / max(math.log(n), 1))), 1)

        # GAPIT thresholds: Bonferroni at loop 2, 1/m at loop 3+
        bonferroni_cutoff = self.cutoff / m
        later_cutoff = 1.0 / m

        prev_qtns = torch.tensor([], dtype=torch.long, device=G_chunk.device)
        prev_qtns_save = torch.tensor([], dtype=torch.long, device=G_chunk.device)
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

            # Apply GAPIT's threshold logic (FDRcut=FALSE by default → Bonferroni)
            if iteration == 0:
                # First scan — use Bonferroni threshold (GAPIT default: FDRcut=FALSE)
                threshold = bonferroni_cutoff
                n_bonf = (result.p < bonferroni_cutoff).sum().item()

                logger.info(
                    "BLINK iter %d: Bonferroni=%.2e (n=%d)",
                    iteration + 1, bonferroni_cutoff, n_bonf,
                )

                # GAPIT special exit: if no SNPs below threshold at first
                # selection, stop and apply p-value adjustment (theLoop==2 exit)
                if n_bonf == 0:
                    logger.info("BLINK: no SNPs below Bonferroni threshold, applying GAPIT p-value adjustment")
                    # GAPIT's p-value adjustment when no QTNs found:
                    # p.GLM.log = -log10(quantile(p, 0.05))
                    # bonf.compare = p.GLM.log / 1.3
                    # p.FARMCPU.log = -log10(p) / bonf.compare
                    # p_adjusted = 10^(-p.FARMCPU.log)
                    p_raw = result.p.clone()
                    p_05 = torch.quantile(p_raw, 0.05)
                    p_log = -torch.log10(torch.clamp(p_05, min=1e-300))
                    bonf_compare = p_log / 1.3
                    if bonf_compare > 0:
                        p_farm_log = -torch.log10(torch.clamp(p_raw, min=1e-300)) / bonf_compare
                        p_adjusted = torch.pow(10.0, -p_farm_log)
                        p_adjusted = torch.clamp(p_adjusted, max=1.0)
                        final_result = ScanResult(
                            chr=result.chr, pos=result.pos, snp=result.snp,
                            a1=result.a1, a2=result.a2, af=result.af,
                            beta=result.beta, se=result.se, stat=result.stat,
                            p=p_adjusted, test=result.test,
                            inference_type="post_selection",
                        )
                    break
            else:
                # Later iterations: use 1/m threshold
                threshold = later_cutoff

            # Mask: exclude low-MAF SNPs from candidate selection
            p_for_select = result.p.clone()
            if self.maf_threshold > 0:
                p_for_select[maf_exclude] = 1.0

            candidates = torch.where(p_for_select < threshold)[0]
            if len(candidates) == 0:
                logger.info(
                    "BLINK iter %d: no SNPs below threshold %.2e, stopping",
                    iteration + 1, threshold,
                )
                break

            # Sort candidates by p-value
            cand_pvals = result.p[candidates]
            order = cand_pvals.argsort()
            candidates_sorted = candidates[order]

            if len(candidates_sorted) > 1:
                # LD removal via correlation matrix (GAPIT's Blink.LDRemoveBlock)
                lead_snps = _ld_remove_block(
                    G_chunk, candidates_sorted, result.p,
                    threshold=self.ld_threshold,
                    max_samples=self.ld_max_samples,
                    n_samples=n,
                )

                # BIC-based forward selection (GAPIT's Blink.BICselection "naive")
                curr_qtns = _bic_forward_select(
                    Y, X0_base, G_chunk, lead_snps, result.p,
                    max_qtns=bound,
                )
            else:
                curr_qtns = candidates_sorted

            # Force previous QTNs into model (union) — GAPIT behavior
            # GAPIT: if (theLoop > 1) union(seqQTN, seqQTN.save)
            # R's union() preserves order: new first, then old (not in save)
            if len(prev_qtns_save) > 0 and len(curr_qtns) > 0:
                # Preserve GAPIT's union order: new first, then old not in new
                new_set = set(curr_qtns.tolist())
                old_extra = [x for x in prev_qtns_save.tolist() if x not in new_set]
                merged_list = curr_qtns.tolist() + old_extra
                curr_qtns = torch.tensor(merged_list, dtype=torch.long, device=G_chunk.device)
                # Re-run BIC on merged set (GAPIT: theLoop > 2 = iteration > 0)
                # GAPIT passes union set directly to BICselection WITHOUT re-sorting
                if iteration > 0 and len(curr_qtns) > 1:
                    curr_qtns = _bic_forward_select(
                        Y, X0_base, G_chunk, curr_qtns, result.p,
                        max_qtns=bound,
                        preserve_order=True,  # Don't re-sort by p-value
                    )
            elif len(curr_qtns) == 0 and len(prev_qtns_save) > 0:
                curr_qtns = prev_qtns_save

            logger.info(
                "BLINK iter %d: %d candidates -> %d after LD -> %d BIC-selected",
                iteration + 1, len(candidates), len(lead_snps) if len(candidates_sorted) > 1 else 1, len(curr_qtns),
            )

            if self.check_convergence(prev_qtns, curr_qtns):
                logger.info("BLINK converged at iteration %d", iteration + 1)
                break

            if len(curr_qtns) == 0:
                logger.info("BLINK: no QTNs selected, stopping")
                break

            prev_qtns_save = curr_qtns.clone()
            prev_qtns = curr_qtns

        return final_result


def _fdr_threshold_index(p_sorted: Tensor, cutoff: float, m: int) -> int:
    """Compute FDR threshold index (GAPIT's step-up FDR).

    Returns 0-based index of the SNP closest to the FDR boundary.
    Used when FDRcut=TRUE (not the default Bonferroni path).
    """
    p_np = p_sorted.detach().cpu().numpy()
    spd = np.abs(cutoff - p_np * m / cutoff)
    return int(np.argmin(spd))


def _ld_remove_block(
    G: Tensor,
    candidates: Tensor,
    p_values: Tensor,
    threshold: float = 0.7,
    max_samples: int = 200,
    n_samples: int = 0,
) -> Tensor:
    """GAPIT's Blink.LDRemoveBlock: correlation-matrix LD removal.

    Computes the full |r| correlation matrix among candidates,
    then greedily removes SNPs correlated (|r| > threshold) with
    a better-ranked (lower p-value) SNP.

    If n_samples > max_samples, subsamples individuals for correlation
    computation (GAPIT: bound=TRUE, subsample to 200).
    """
    if len(candidates) <= 1:
        return candidates

    # Already sorted by p-value (best first)
    G_sub = G[:, candidates].to(torch.float64)  # (n, k)

    # GAPIT's bound logic: only subsample if min(n_samples, n_candidates) >= 201
    # For typical GWAS with <200 candidates, no subsampling occurs.
    # When subsampling is needed, use random sampling to match GAPIT's sample()
    n_cand = G_sub.shape[1]
    n_ind = G_sub.shape[0]
    if min(n_ind, n_cand) >= 201 and n_ind > max_samples:
        gen = torch.Generator(device=G_sub.device)
        gen.manual_seed(n_ind * 1000 + n_cand)
        idx = torch.randperm(n_ind, generator=gen, device=G_sub.device)[:max_samples]
        G_sub = G_sub[idx]

    # Remove zero-variance SNPs
    stds = G_sub.std(dim=0)
    nonzero = stds > 1e-10
    if not nonzero.all():
        valid_mask = nonzero
        valid_indices = torch.where(valid_mask)[0]
        G_sub = G_sub[:, valid_mask]
        if G_sub.shape[1] == 0:
            return candidates[:0]
    else:
        valid_indices = torch.arange(len(candidates), device=candidates.device)

    # Compute correlation matrix matching R's cor() (N-1 denominator)
    G_c = G_sub - G_sub.mean(dim=0, keepdim=True)
    stds = G_c.std(dim=0, keepdim=True)  # PyTorch std uses N-1 by default
    stds = torch.clamp(stds, min=1e-10)
    G_norm = G_c / stds
    corr = (G_norm.T @ G_norm) / (G_norm.shape[0] - 1)  # N-1 to match R

    # GAPIT's Blink.LDRemoveBlock: greedy sequential removal.
    # For each SNP i (sorted by p-value), check if ANY previously KEPT
    # SNP is correlated with it. If yes, remove i.
    # This differs from FarmCPU.Remove's product-based approach.
    k = len(valid_indices)
    corr_bin = corr.clone()
    corr_bin[corr.abs() <= threshold] = 0
    corr_bin[corr.abs() > threshold] = 1
    corr_bin[torch.isnan(corr)] = 1

    keep = torch.ones(k, dtype=torch.long, device=candidates.device)
    for i in range(1, k):
        # Check if any previously-kept SNP is correlated with SNP i
        prev_kept = keep[:i]
        prev_corr = corr_bin[:i, i]
        # Both kept (=1) AND correlated (=1) → match
        match = (prev_kept == 1) & (prev_corr == 1)
        if match.any():
            keep[i] = 0

    keep_mask = keep == 1

    kept = valid_indices[keep_mask.cpu()]
    return candidates[kept]


def _bic_forward_select(
    Y: Tensor,
    X0_base: Tensor,
    G: Tensor,
    lead_snps: Tensor,
    p_values: Tensor,
    max_qtns: int,
    preserve_order: bool = False,
) -> Tensor:
    """GAPIT's Blink.BICselection ("naive" method).

    Tests ALL candidate subset sizes from 1 to m. For each size,
    incrementally adds the next SNP and computes BIC.
    Selects the size with minimum BIC.

    BIC = n*log(2π) + n*log(ve) + RSS/ve + (k-1)*log(n)
    where ve = RSS/(n-1) (R's var() convention)

    Parameters
    ----------
    preserve_order : bool
        If True, use the input order of lead_snps directly (for re-BIC
        on union sets, matching GAPIT which doesn't re-sort).
        If False (default), sort by p-value first.
    """
    if len(lead_snps) == 0:
        return torch.tensor([], dtype=torch.long, device=G.device)

    n = Y.shape[0]

    if preserve_order:
        sorted_leads = lead_snps
    else:
        # Sort leads by p-value
        lead_pvals = p_values[lead_snps]
        order = lead_pvals.argsort()
        sorted_leads = lead_snps[order]

    # Truncate to max_qtns
    sorted_leads = sorted_leads[:min(max_qtns, len(sorted_leads))]

    # Evaluate BIC for each subset size (1 to m) — "naive" method
    # GAPIT always picks the size with minimum BIC (never compares to null)
    best_bic = float('inf')
    best_size = 1  # GAPIT always selects at least 1 SNP

    for size in range(1, len(sorted_leads) + 1):
        sel = sorted_leads[:size]
        X_trial = torch.cat([X0_base, G[:, sel]], dim=1)
        trial_bic = _compute_bic_gapit(Y, X_trial, n)

        if trial_bic < best_bic:
            best_bic = trial_bic
            best_size = size

    return sorted_leads[:best_size]


def _compute_bic_gapit(Y: Tensor, X: Tensor, n: int) -> float:
    """BIC matching GAPIT's formula.

    BIC = n*log(2π) + n*log(ve) + RSS/ve + (ncov-1)*log(n)
    where ve = RSS/(n-1) (R's var() convention)
    """
    XtX = X.T @ X
    b = torch.linalg.solve(XtX, X.T @ Y)
    resid = Y - X @ b
    rss = (resid @ resid).item()
    k = X.shape[1]

    # R's var() uses n-1 denominator
    ve = rss / max(n - 1, 1)
    if ve < 1e-300:
        ve = 1e-300

    n2ll = n * math.log(2 * math.pi) + n * math.log(ve) + rss / ve
    bic = n2ll + (k - 1) * math.log(n)
    return bic


def _substitute_qtn_pvalues(
    result: ScanResult,
    qtn_indices: Tensor,
    X_aug: Tensor,
    Y: Tensor,
    G: Tensor,
    method: str = "reward",
) -> ScanResult:
    """Substitute p-values of pseudo-QTNs (GAPIT's Blink.SUB).

    Matches GAPIT's exact algorithm: for each QTN covariate k, compute its
    p-value in ALL m per-SNP models (Y ~ X_aug + G[:, j]), then:
      - "reward": use the minimum p-value across all models
      - "mean": use the mean p-value
      - "penalty": use the maximum p-value

    Uses partitioned inverse for efficient vectorized computation.
    """
    import scipy.stats as sp_stats

    n, c = X_aug.shape
    m = G.shape[1]
    n_qtns = len(qtn_indices)
    n_base = c - n_qtns
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

    G_var = (G * G).sum(dim=0)
    near_collinear = (gtg < (G_var * 1e-6)) | (G_var < 1e-10)
    gtg_safe = torch.where(near_collinear, torch.ones_like(gtg), gtg)

    beta_snp = gty / gtg_safe
    rss_null = (y_resid @ y_resid).item()
    rss_j = rss_null - gty ** 2 / gtg_safe
    rss_j = torch.clamp(rss_j, min=1e-20)
    sig2_j = rss_j / df

    qtn_sub_p = torch.ones(n_qtns, dtype=STAT_DTYPE, device=G.device)
    qtn_sub_beta = torch.zeros(n_qtns, dtype=STAT_DTYPE, device=G.device)
    qtn_sub_se = torch.ones(n_qtns, dtype=STAT_DTYPE, device=G.device)

    for qi in range(n_qtns):
        k = n_base + qi

        b_k_j = b_y[k] - XtX_inv_XtG[k, :] * beta_snp
        var_k_j = sig2_j * (XtX_inv[k, k] + XtX_inv_XtG[k, :] ** 2 / gtg_safe)
        var_k_j = torch.clamp(var_k_j, min=1e-40)

        t_k_j = b_k_j / torch.sqrt(var_k_j)
        t_abs = t_k_j.abs().detach().cpu().numpy().astype(np.float64)
        p_k_j_np = 2.0 * sp_stats.t.sf(t_abs, df=df)
        p_k_j = torch.tensor(p_k_j_np, dtype=STAT_DTYPE, device=G.device)
        p_k_j = torch.clamp(p_k_j, min=1e-300, max=1.0)
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
    Uses F(1, n-c-1) distribution (equivalent to two-sided t-test).
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

    # F(1, n-c-1) p-values (equivalent to 2*pt(|t|, df))
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
    """P-values from F-distribution (equivalent to GAPIT's 2*pt(|t|, df))."""
    import scipy.stats as sp_stats
    stat_np = stat.detach().cpu().numpy().astype(np.float64)
    p_np = sp_stats.f.sf(stat_np, dfn=df1, dfd=df2)
    p_np = np.clip(p_np, 1e-300, 1.0)
    return torch.tensor(p_np, dtype=STAT_DTYPE, device=stat.device)
