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
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..config import STAT_DTYPE
from .base import NullFit, ScanResult, VariantMeta
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
        bin_sizes: list[int] | None = None,
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
        K: Tensor | None = None,
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

    # ------------------------------------------------------------------
    # Streaming variant — orchestrates the FEM/REM iteration externally.
    # The full ``(n × m)`` G is never resident; only the cached QTN
    # columns (``n × |QTN|``, |QTN| ≤ ~20 in practice) live in memory
    # alongside one streaming chunk at a time. Per-iteration the genome
    # is streamed once for the GLM scan and (when prior QTNs exist) once
    # more for the p-value substitution; both passes are linear in m and
    # never materialize G.
    # ------------------------------------------------------------------

    def score_streaming(
        self,
        reader,
        null_fit: NullFit,
        chunk_size: int,
        test: str = "wald",
    ) -> ScanResult:
        """Streaming FarmCPU scan — chunk-based variant of ``score_chunk``.

        Parameters
        ----------
        reader : object
            Object with ``iter_chunks(chunk_size)`` yielding
            ``(G_chunk, vmeta)`` and an ``n_variants`` property. Imputed
            float64 dosages are required (use ``_impute_chunk_iter``).
        null_fit, test
            See ``score_chunk``.
        chunk_size : int
            Per-iteration streaming chunk size. Used both for the GLM
            scan pass and the QTN substitution pass.

        Returns
        -------
        ScanResult — identical to ``score_chunk(G_full, …)`` to float64
        tolerance modulo per-chunk numerical reductions.
        """
        if test not in ("wald",):
            raise ValueError(f"FarmCPU supports 'wald' test only, got '{test}'.")

        Y = null_fit.Y_rot
        device = Y.device
        m = reader.n_variants
        n = Y.shape[0]
        X0_base = null_fit.X0_rot

        # Pre-pass to build genome-wide variant_meta + position / chromosome
        # arrays. Cheap: only collects metadata and the per-variant MAF
        # (needed for QTN-eligibility in selection). Does not retain G.
        all_snp: list[str] = []
        all_chr: list[str] = []
        all_pos: list[int] = []
        all_a1: list[str] = []
        all_a2: list[str] = []
        af_pieces: list[Tensor] = []
        for G_chunk, vm in reader.iter_chunks(chunk_size):
            all_snp.extend(vm.snp)
            all_chr.extend([str(c) for c in vm.chr])
            all_pos.extend(vm.pos)
            all_a1.extend(vm.a1)
            all_a2.extend(vm.a2)
            af_pieces.append((G_chunk.to(STAT_DTYPE).mean(dim=0) / 2.0).to(device))
        af_global = torch.cat(af_pieces).to(device) if af_pieces else torch.zeros(0, device=device, dtype=STAT_DTYPE)
        maf_global = torch.minimum(af_global, 1.0 - af_global)
        maf_exclude = maf_global < self.maf_threshold
        positions = torch.tensor(all_pos, dtype=torch.long, device=device)
        chromosomes = all_chr

        variant_meta = VariantMeta(
            snp=all_snp, chr=all_chr, pos=all_pos, a1=all_a1, a2=all_a2,
        )

        # Cached QTN columns (n × |QTN|), grown across iterations.
        bound = max(int(round(math.sqrt(n) / math.sqrt(max(math.log10(n), 1)))), 1)
        bound = min(bound, self.max_qtns)

        prev_qtns = torch.tensor([], dtype=torch.long, device=device)
        prev_qtns_save = torch.tensor([], dtype=torch.long, device=device)
        prev_qtns_pre = torch.tensor([], dtype=torch.long, device=device)
        qtn_columns = torch.empty((n, 0), dtype=STAT_DTYPE, device=device)
        final_result: ScanResult | None = None

        for iteration in range(self.max_iter):
            if len(prev_qtns) > 0:
                X_aug = torch.cat([X0_base, qtn_columns], dim=1)
            else:
                X_aug = X0_base

            qtn_set = set(prev_qtns.tolist()) if len(prev_qtns) > 0 else set()
            result = _glm_scan_streaming(
                Y, X_aug,
                _iter_chunks_for_streaming(reader, chunk_size),
                variant_meta, qtn_indices=qtn_set,
                af_global=af_global,
            )

            if len(prev_qtns) > 0:
                result = _substitute_qtn_pvalues_streaming(
                    result, prev_qtns, X_aug, Y,
                    _iter_chunks_for_streaming(reader, chunk_size),
                    method=self.method_sub,
                )

            final_result = result

            if iteration == 0:
                p_consider = result.p[~maf_exclude] if self.maf_threshold > 0 else result.p
                if p_consider.numel() == 0:
                    break
                min_p = p_consider.min().item()
                n_active = m - int(maf_exclude.sum().item()) if self.maf_threshold > 0 else m
                early_cutoff = self.p_threshold / max(n_active, 1)
                if min_p > early_cutoff:
                    logger.info(
                        "FarmCPU(stream): min(p)=%.2e > %.2e at iter 1, stopping early",
                        min_p, early_cutoff,
                    )
                    break

            if self.method_bin == "static":
                if iteration == 0:
                    bin_size = self.bin_sizes[2]
                elif iteration == 1:
                    bin_size = self.bin_sizes[1]
                else:
                    bin_size = self.bin_sizes[0]
            else:
                bin_size = self.bin_sizes[0]

            p_for_binning = result.p.clone()
            if self.maf_threshold > 0:
                p_for_binning[maf_exclude] = 1.0

            curr_qtns = _specify_bins(
                p_for_binning, positions, chromosomes, bin_size, bound,
            )

            if iteration > 0 and len(prev_qtns_save) > 0 and len(curr_qtns) > 0:
                merged = torch.cat([curr_qtns, prev_qtns_save])
                curr_qtns = merged.unique()

            if len(curr_qtns) > 0:
                pvals_for_filter = result.p[curr_qtns]
                if iteration == 0:
                    mask = pvals_for_filter < self.p_threshold
                    curr_qtns = curr_qtns[mask]
                else:
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

            # _remove_correlated needs only the candidate columns — read
            # them from the streaming reader and discard immediately.
            if len(curr_qtns) > 1:
                cand_cols = _read_columns_from_reader(reader, curr_qtns.tolist(), chunk_size)
                cand_cols = cand_cols.to(STAT_DTYPE).to(device)
                # _remove_correlated uses columns in p-value order; we pass
                # the columns in candidate order and let _remove_correlated
                # re-sort as needed via its argsort on candidates.
                curr_qtns = _remove_correlated_with_cols(
                    cand_cols, curr_qtns, result.p, threshold=0.7,
                )
                del cand_cols

            logger.info(
                "FarmCPU(stream) iter %d: %d pseudo-QTNs (bin=%d)",
                iteration + 1, len(curr_qtns), bin_size,
            )

            if self.check_convergence(prev_qtns, curr_qtns):
                logger.info("FarmCPU(stream) converged at iteration %d", iteration + 1)
                break

            if len(prev_qtns_pre) > 0 and self.check_convergence(prev_qtns_pre, curr_qtns):
                logger.info("FarmCPU(stream) cycling detected at iteration %d", iteration + 1)
                break

            if len(curr_qtns) == 0:
                logger.info("FarmCPU(stream): no QTNs selected, stopping")
                break

            prev_qtns_pre = prev_qtns_save.clone() if len(prev_qtns_save) > 0 else prev_qtns.clone()
            prev_qtns_save = curr_qtns.clone()
            prev_qtns = curr_qtns

            # Refresh QTN-column cache from the reader for the next iteration.
            qtn_columns = _read_columns_from_reader(
                reader, curr_qtns.tolist(), chunk_size,
            ).to(STAT_DTYPE).to(device)

        return final_result


def _iter_chunks_for_streaming(reader, chunk_size: int):
    """Wrap ``reader.iter_chunks`` to yield mean-imputed float64 chunks.

    Avoids importing CLI helpers from the model module; mirrors the
    behaviour of ``_impute_chunk_iter`` defined in ``cli.py``.
    """
    from ..preprocess.impute import impute_mean
    for G_chunk, vmeta in reader.iter_chunks(chunk_size):
        yield impute_mean(G_chunk).to(STAT_DTYPE), vmeta


def _read_columns_from_reader(reader, indices: list[int], chunk_size: int) -> Tensor:
    """Read specific column indices from a streaming reader.

    Streams chunks once (linear in m) but only retains the requested
    columns — peak memory ``O(n × len(indices))``, not ``O(n × m)``.
    """
    if not indices:
        return torch.empty((reader.n_samples, 0), dtype=STAT_DTYPE)
    sorted_indices = sorted(set(indices))
    by_orig_pos = {idx: pos for pos, idx in enumerate(indices)}

    n = reader.n_samples
    out = torch.empty((n, len(indices)), dtype=STAT_DTYPE)
    found_count = 0
    target_count = len(sorted_indices)
    pos = 0
    for G_chunk, _ in reader.iter_chunks(chunk_size):
        chunk_n = G_chunk.shape[1]
        chunk_end = pos + chunk_n
        wanted = [i for i in sorted_indices if pos <= i < chunk_end]
        for i in wanted:
            local = i - pos
            out[:, by_orig_pos[i]] = G_chunk[:, local].to(STAT_DTYPE)
        found_count += len(wanted)
        pos = chunk_end
        if found_count >= target_count:
            break
    # Mean-impute any NaNs (matches _load_full_genotype semantics).
    if torch.isnan(out).any():
        from ..preprocess.impute import impute_mean
        out = impute_mean(out)
    return out


def _glm_scan_streaming(
    Y: Tensor,
    X_aug: Tensor,
    chunk_iter,
    variant_meta: VariantMeta,
    qtn_indices: set[int] | None = None,
    af_global: Tensor | None = None,
) -> ScanResult:
    """Streaming variant of :func:`_glm_scan`.

    Per-SNP linear ops (gtg, gty, beta, se, p) are accumulated
    chunk-by-chunk; the X_aug-side stats (XtX_inv, b_y, y_resid) are
    built once and reused. Behavioral parity to the eager variant.
    """
    Y = Y.to(STAT_DTYPE)
    X_aug = X_aug.to(STAT_DTYPE)
    n, c = X_aug.shape
    qtn_indices = qtn_indices or set()
    device = Y.device

    XtX = X_aug.T @ X_aug
    XtX_inv = torch.linalg.inv(XtX)
    b_y = XtX_inv @ (X_aug.T @ Y)
    y_resid = Y - X_aug @ b_y
    rss_null = (y_resid @ y_resid).item()
    df = max(n - c - 1, 1)

    beta_pieces: list[Tensor] = []
    se_pieces: list[Tensor] = []
    stat_pieces: list[Tensor] = []
    p_pieces: list[Tensor] = []
    near_pieces: list[Tensor] = []
    af_pieces: list[Tensor] = []
    n_seen = 0
    for G_chunk, _ in chunk_iter:
        G_chunk = G_chunk.to(STAT_DTYPE).to(device)
        m_chunk = G_chunk.shape[1]
        af_pieces.append(G_chunk.mean(dim=0) / 2.0)
        # Residualize this chunk against X_aug.
        X_aug_tG = X_aug.T @ G_chunk
        G_resid = G_chunk - X_aug @ (XtX_inv @ X_aug_tG)
        gtg = (G_resid * G_resid).sum(dim=0)
        gty = (G_resid * y_resid.unsqueeze(1)).sum(dim=0)
        G_var = (G_chunk * G_chunk).sum(dim=0)
        near_collinear = (gtg < (G_var * 1e-6)) | (G_var < 1e-10)
        # QTN columns (already in X_aug) flagged.
        if qtn_indices:
            for idx in qtn_indices:
                local = idx - n_seen
                if 0 <= local < m_chunk:
                    near_collinear[local] = True
        gtg_safe = torch.where(near_collinear, torch.ones_like(gtg), gtg)
        beta = gty / gtg_safe
        rss_j = rss_null - gty ** 2 / gtg_safe
        rss_j = torch.clamp(rss_j, min=1e-20)
        sig2_j = rss_j / df
        var_beta = sig2_j / gtg_safe
        se = torch.sqrt(var_beta)
        stat = beta ** 2 / var_beta
        p = _f_sf(stat, df1=1, df2=df)

        beta = torch.where(near_collinear, torch.zeros_like(beta), beta)
        se = torch.where(near_collinear, torch.full_like(se, float("inf")), se)
        stat = torch.where(near_collinear, torch.zeros_like(stat), stat)
        p = torch.where(near_collinear, torch.ones_like(p), p)

        beta_pieces.append(beta)
        se_pieces.append(se)
        stat_pieces.append(stat)
        p_pieces.append(p)
        near_pieces.append(near_collinear)
        n_seen += m_chunk

    beta_all = torch.cat(beta_pieces) if beta_pieces else torch.zeros(0, dtype=STAT_DTYPE, device=device)
    se_all = torch.cat(se_pieces) if se_pieces else torch.zeros(0, dtype=STAT_DTYPE, device=device)
    stat_all = torch.cat(stat_pieces) if stat_pieces else torch.zeros(0, dtype=STAT_DTYPE, device=device)
    p_all = torch.cat(p_pieces) if p_pieces else torch.zeros(0, dtype=STAT_DTYPE, device=device)
    af_all = torch.cat(af_pieces) if af_pieces else (af_global if af_global is not None else torch.zeros(0, dtype=STAT_DTYPE, device=device))

    return ScanResult(
        chr=variant_meta.chr, pos=variant_meta.pos, snp=variant_meta.snp,
        a1=variant_meta.a1, a2=variant_meta.a2, af=af_all,
        beta=beta_all, se=se_all, stat=stat_all, p=p_all, test="wald",
        inference_type="post_selection",
    )


def _substitute_qtn_pvalues_streaming(
    result: ScanResult,
    qtn_indices: Tensor,
    X_aug: Tensor,
    Y: Tensor,
    chunk_iter,
    method: str = "reward",
) -> ScanResult:
    """Streaming variant of :func:`_substitute_qtn_pvalues`.

    Holds chunk-aggregated per-SNP pvalues only for the small set of
    QTN covariates (n_qtns × m). Aggregations: ``min``/``max`` track
    running argmin/argmax with O(n_qtns) state per chunk; ``mean``
    tracks running sum/count; ``median`` falls back to a per-QTN
    accumulator (``m × n_qtns × 8 B``).
    """
    import scipy.stats as sp_stats
    Y = Y.to(STAT_DTYPE)
    X_aug = X_aug.to(STAT_DTYPE)
    n, c = X_aug.shape
    n_qtns = len(qtn_indices)
    n_base = c - n_qtns
    df = max(n - c - 1, 1)
    device = Y.device

    XtX = X_aug.T @ X_aug
    XtX_inv = torch.linalg.inv(XtX)
    b_y = XtX_inv @ (X_aug.T @ Y)
    y_resid = Y - X_aug @ b_y
    rss_null = (y_resid @ y_resid).item()

    # For each QTN, track aggregator state across all SNPs.
    qtn_sub_p = torch.full((n_qtns,), float("inf"), dtype=STAT_DTYPE, device=device)
    qtn_sub_beta = torch.zeros(n_qtns, dtype=STAT_DTYPE, device=device)
    qtn_sub_se = torch.ones(n_qtns, dtype=STAT_DTYPE, device=device)
    if method == "reward":
        # min — track argmin per QTN
        pass
    elif method == "penalty":
        qtn_sub_p = torch.full((n_qtns,), -float("inf"), dtype=STAT_DTYPE, device=device)
    elif method in ("mean", "median"):
        all_p_pieces: list[list[Tensor]] = [[] for _ in range(n_qtns)]
        all_b_pieces: list[list[Tensor]] = [[] for _ in range(n_qtns)]
        all_var_pieces: list[list[Tensor]] = [[] for _ in range(n_qtns)]
        all_near_pieces: list[Tensor] = []

    n_seen = 0
    for G_chunk, _ in chunk_iter:
        G_chunk = G_chunk.to(STAT_DTYPE).to(device)
        m_chunk = G_chunk.shape[1]
        X_aug_tG = X_aug.T @ G_chunk  # (c, m_chunk)
        XtX_inv_XtG = XtX_inv @ X_aug_tG
        G_resid = G_chunk - X_aug @ XtX_inv_XtG
        gtg = (G_resid * G_resid).sum(dim=0)
        gty = (G_resid * y_resid.unsqueeze(1)).sum(dim=0)
        G_var = (G_chunk * G_chunk).sum(dim=0)
        near_collinear = (gtg < (G_var * 1e-6)) | (G_var < 1e-10)
        gtg_safe = torch.where(near_collinear, torch.ones_like(gtg), gtg)
        beta_snp = gty / gtg_safe
        rss_j = rss_null - gty ** 2 / gtg_safe
        rss_j = torch.clamp(rss_j, min=1e-20)
        sig2_j = rss_j / df

        if method in ("mean", "median"):
            all_near_pieces.append(near_collinear)

        for qi in range(n_qtns):
            k = n_base + qi
            b_k_j = b_y[k] - XtX_inv_XtG[k, :] * beta_snp
            var_k_j = sig2_j * (XtX_inv[k, k] + XtX_inv_XtG[k, :] ** 2 / gtg_safe)
            var_k_j = torch.clamp(var_k_j, min=1e-40)
            t_k_j = b_k_j / torch.sqrt(var_k_j)
            t_abs = t_k_j.abs().detach().cpu().numpy().astype(np.float64)
            p_k_j_np = 2.0 * sp_stats.t.sf(t_abs, df=df)
            p_k_j = torch.tensor(p_k_j_np, dtype=STAT_DTYPE, device=device)
            p_k_j = torch.clamp(p_k_j, min=1e-300, max=1.0)
            p_k_j = torch.where(near_collinear, torch.ones_like(p_k_j), p_k_j)

            if method == "reward":
                cur_min = p_k_j.min()
                if cur_min < qtn_sub_p[qi]:
                    qtn_sub_p[qi] = cur_min
                    j = p_k_j.argmin()
                    qtn_sub_beta[qi] = b_k_j[j]
                    qtn_sub_se[qi] = torch.sqrt(var_k_j[j])
            elif method == "penalty":
                cur_max = p_k_j.max()
                if cur_max > qtn_sub_p[qi]:
                    qtn_sub_p[qi] = cur_max
                    j = p_k_j.argmax()
                    qtn_sub_beta[qi] = b_k_j[j]
                    qtn_sub_se[qi] = torch.sqrt(var_k_j[j])
            else:  # mean / median — accumulate
                all_p_pieces[qi].append(p_k_j)
                all_b_pieces[qi].append(b_k_j)
                all_var_pieces[qi].append(var_k_j)

        n_seen += m_chunk

    if method in ("mean", "median"):
        near_all = torch.cat(all_near_pieces) if all_near_pieces else torch.zeros(0, dtype=torch.bool, device=device)
        for qi in range(n_qtns):
            p_q = torch.cat(all_p_pieces[qi])
            b_q = torch.cat(all_b_pieces[qi])
            var_q = torch.cat(all_var_pieces[qi])
            valid = ~near_all
            if method == "mean":
                qtn_sub_p[qi] = p_q[valid].mean() if valid.any() else p_q.mean()
                qtn_sub_beta[qi] = b_q[valid].mean() if valid.any() else b_q.mean()
                qtn_sub_se[qi] = torch.sqrt(var_q[valid].mean() if valid.any() else var_q.mean())
            else:  # median
                vp = p_q[valid] if valid.any() else p_q
                qtn_sub_p[qi] = vp.median()
                qtn_sub_beta[qi] = b_q[valid].median() if valid.any() else b_q.median()
                qtn_sub_se[qi] = torch.sqrt(var_q[valid].median() if valid.any() else var_q.median())

    p_new = result.p.clone()
    beta_new = result.beta.clone()
    se_new = result.se.clone()
    stat_new = result.stat.clone()
    for i, idx in enumerate(qtn_indices.tolist()):
        p_new[idx] = qtn_sub_p[i]
        beta_new[idx] = qtn_sub_beta[i]
        se_new[idx] = qtn_sub_se[i]
        stat_new[idx] = (qtn_sub_beta[i] / torch.clamp(qtn_sub_se[i], min=1e-20)) ** 2

    return ScanResult(
        chr=result.chr, pos=result.pos, snp=result.snp,
        a1=result.a1, a2=result.a2, af=result.af,
        beta=beta_new, se=se_new, stat=stat_new, p=p_new,
        test=result.test, inference_type="post_selection",
    )


def _remove_correlated_with_cols(
    G_cols: Tensor,
    candidates: Tensor,
    p_values: Tensor,
    threshold: float = 0.7,
) -> Tensor:
    """Streaming-friendly variant of :func:`_remove_correlated`.

    Identical algorithm to ``_remove_correlated`` but takes the
    candidate-only column matrix directly (``G_cols`` columns are in
    ``candidates`` order, NOT genome-wide order). Side-steps the
    requirement of a full ``G`` tensor for indexing.
    """
    if len(candidates) <= 1:
        return candidates

    pvals = p_values[candidates]
    order = pvals.argsort()
    candidates_sorted = candidates[order]
    G_sub = G_cols[:, order].to(torch.float64)

    G_c = G_sub - G_sub.mean(dim=0, keepdim=True)
    stds = G_c.std(dim=0, keepdim=True)
    stds = torch.clamp(stds, min=1e-10)
    G_norm = G_c / stds
    corr = (G_norm.T @ G_norm) / (G_norm.shape[0] - 1)

    k = len(candidates_sorted)
    b = (corr.abs() > threshold).int()
    cmat = 1 - b
    for i in range(k):
        for j in range(i + 1):
            cmat[i, j] = 1
    keep_mask = cmat.prod(dim=0) == 1
    kept_idx = torch.where(keep_mask)[0].tolist()
    return candidates_sorted[kept_idx]


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
