"""Set-Based Association Tests under LMM (Phase 17).

Implements SKAT (variance-component test), Burden (mean-effect test), and
SKAT-O (optimal combination) within the LMM framework using the null model's
P-operator (projection matrix).

The set-based test statistic Q = (Py)^T G W G^T (Py) is computed using
the same rotated-space quantities cached during the null fit. P-values
for SKAT use chi-squared mixture distributions (Davies method).

References:
    - Wu et al. (2011). Rare-variant association testing for sequencing
      data with the sequence kernel association test (SKAT).
    - Lee et al. (2012). Optimal unified approach for rare-variant
      association testing (SKAT-O).
    - SAIGE-GENE (Zhou et al., 2020). GPU-scalable set-based tests.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import STAT_DTYPE
from ..io.regions import Region, compute_skat_weights, map_regions_to_variants
from ..stats.mixture import mixture_chi2_pvalue
from .base import NullFit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SetBasedResult:
    """Gene/region-level association test results."""

    region_id: list[str]
    chr: list[str]
    start: list[int]
    end: list[int]
    n_variants: list[int]

    # Test results
    q_stat: Tensor  # (n_regions,)
    p: Tensor  # (n_regions,)
    test: str  # "skat", "burden", "skat_o"

    # SKAT-O specific
    rho_opt: Tensor | None = None

    def __len__(self) -> int:
        return len(self.region_id)


# ---------------------------------------------------------------------------
# SetBasedScanner
# ---------------------------------------------------------------------------

class SetBasedScanner:
    """Set-based association tests (SKAT, Burden, SKAT-O) under LMM.

    Operates on a pre-computed NullFit from SingleTraitLMM.fit_null().
    The P-operator Py is computed once and reused for all gene regions.

    Parameters
    ----------
    null_fit : NullFit
        Must contain eigenvectors, eigenvalues, Y_rot, X0_rot, sig2_g, sig2_e.
    """

    def __init__(self, null_fit: NullFit, ploidy: int = 2) -> None:
        self.null_fit = null_fit
        self.ploidy = ploidy
        self._Py = self._compute_Py()

    def _compute_Py(self) -> Tensor:
        """Compute P @ y in rotated space.

        The full P-operator is P = P_H / sig2_e where
        P_H = H^{-1} - H^{-1} X (X^T H^{-1} X)^{-1} X^T H^{-1}.
        The 1/sig2_e factor is needed so that under H0, Py has the correct
        variance (Var(Py) = P, not sig2_e * P_H).
        """
        from ..optim.reml_math import _compute_P_quantities

        nf = self.null_fit
        lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
        H_inv = 1.0 / (nf.eigenvalues * lam + 1.0)
        Py_H, _ = _compute_P_quantities(nf.Y_rot, nf.X0_rot, H_inv)
        # Normalize by sig2_e to get the correct P = V^{-1} - V^{-1}X(...)X^TV^{-1}
        return Py_H / max(nf.sig2_e, 1e-20)  # (n,) in rotated space

    def _apply_P(self, v: Tensor) -> Tensor:
        """Apply P operator to vector(s) v in rotated space.

        P = (1/sig2_e) * (H_inv - H_inv X (X^T H_inv X)^{-1} X^T H_inv)

        Parameters
        ----------
        v : (n,) or (n, k)

        Returns
        -------
        Pv : same shape as v.
        """
        nf = self.null_fit
        lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
        H_inv = 1.0 / (nf.eigenvalues * lam + 1.0)
        sig2_e = max(nf.sig2_e, 1e-20)

        X = nf.X0_rot  # (n, c)

        if v.ndim == 1:
            wv = H_inv * v  # (n,)
            wX = H_inv.unsqueeze(1) * X  # (n, c)
            XtWX = X.T @ wX  # (c, c)
            XtWv = X.T @ wv  # (c,)
            correction = wX @ torch.linalg.solve(XtWX, XtWv)  # (n,)
            return (wv - correction) / sig2_e
        else:
            # v is (n, k)
            wv = H_inv.unsqueeze(1) * v  # (n, k)
            wX = H_inv.unsqueeze(1) * X  # (n, c)
            XtWX = X.T @ wX  # (c, c)
            XtWv = X.T @ wv  # (c, k)
            correction = wX @ torch.linalg.solve(XtWX, XtWv)  # (n, k)
            return (wv - correction) / sig2_e

    def skat(
        self,
        G_region: Tensor,
        region: Region,
        weights: Tensor | None = None,
        allele_freq: Tensor | None = None,
    ) -> tuple[float, float]:
        """SKAT test for a single region.

        Q = z^T W z  where z = G_rot^T @ Py
        Under H0, Q ~ sum(lambda_k * chi2_1)

        Parameters
        ----------
        G_region : (n, m_gene) genotypes for variants in this region.
        region : Region metadata.
        weights : (m_gene,) diagonal weight vector. If None, use Beta(MAF;1,25).
        allele_freq : (m_gene,) for weight computation.

        Returns
        -------
        (q_stat, p_value)
        """
        from ..linalg.eigh import rotate

        G_region = G_region.to(STAT_DTYPE)
        n, m_gene = G_region.shape

        if m_gene == 0:
            return 0.0, 1.0

        # Rotate into eigenspace
        U = self.null_fit.eigenvectors
        G_rot = rotate(G_region, U)  # (n, m_gene)

        # Compute weights
        if weights is None:
            if allele_freq is None:
                allele_freq = G_region.mean(dim=0) / self.ploidy
            weights = compute_skat_weights(allele_freq)

        # Score vector: z = G_rot^T @ Py
        Py = self._Py  # (n,)
        z = G_rot.T @ Py  # (m_gene,)

        # Q statistic: Q = z^T W z = sum(w_j * z_j^2)
        Q = (weights * z ** 2).sum().item()

        if Q <= 0:
            return 0.0, 1.0

        # Eigenvalues for mixture distribution
        # M = W^{1/2} @ (G_rot^T @ P @ G_rot) @ W^{1/2}
        # P @ G_rot computed via _apply_P
        PG = self._apply_P(G_rot)  # (n, m_gene)
        GtPG = G_rot.T @ PG  # (m_gene, m_gene)

        w_half = torch.sqrt(torch.clamp(weights, min=1e-20))  # (m_gene,)
        M = (w_half.unsqueeze(1) * GtPG) * w_half.unsqueeze(0)  # (m_gene, m_gene)

        # Eigenvalues of M (symmetric)
        lambdas = torch.linalg.eigvalsh(M)
        lambdas = lambdas[lambdas > 1e-10]  # filter numerical zeros

        if len(lambdas) == 0:
            return Q, 1.0

        # P-value via Davies method
        p = mixture_chi2_pvalue(Q, lambdas, method="davies")

        return Q, p

    def burden(
        self,
        G_region: Tensor,
        region: Region,
        weights: Tensor | None = None,
        allele_freq: Tensor | None = None,
    ) -> tuple[float, float]:
        """Burden test for a single region.

        Collapse: g_burden = G @ w
        T = (g_burden^T Py)^2 / (g_burden^T P g_burden) ~ chi2(1)

        Parameters
        ----------
        G_region : (n, m_gene) genotypes for variants in this region.
        region : Region metadata.
        weights : (m_gene,) collapse weights. If None, use Beta(MAF;1,25).
        allele_freq : (m_gene,) for weight computation.

        Returns
        -------
        (t_stat, p_value)
        """
        from ..linalg.eigh import rotate
        from ..stats.tests import chi2_sf

        G_region = G_region.to(STAT_DTYPE)
        n, m_gene = G_region.shape

        if m_gene == 0:
            return 0.0, 1.0

        U = self.null_fit.eigenvectors
        G_rot = rotate(G_region, U)

        if weights is None:
            if allele_freq is None:
                allele_freq = G_region.mean(dim=0) / self.ploidy
            weights = compute_skat_weights(allele_freq)

        # Collapse: g_burden_rot = G_rot @ sqrt(w) (use sqrt for unit-variance collapse)
        w_sqrt = torch.sqrt(torch.clamp(weights, min=1e-20))
        g_burden_rot = G_rot @ w_sqrt  # (n,)

        # Test statistic
        Py = self._Py
        numerator = (g_burden_rot @ Py) ** 2  # scalar

        Pg = self._apply_P(g_burden_rot)  # (n,)
        denominator = g_burden_rot @ Pg  # scalar
        denominator = max(denominator.item(), 1e-20)

        T = numerator.item() / denominator
        T = max(T, 0.0)

        # p-value: T ~ chi2(1)
        p_tensor = chi2_sf(torch.tensor([T], dtype=STAT_DTYPE), df=1)
        p = p_tensor.item()

        return T, p

    def skat_o(
        self,
        G_region: Tensor,
        region: Region,
        weights: Tensor | None = None,
        allele_freq: Tensor | None = None,
        rho_grid: Tensor | None = None,
    ) -> tuple[float, float, float]:
        """SKAT-O: optimal combination of SKAT and Burden.

        Q_rho = (1-rho) * Q_SKAT + rho * Q_Burden_raw
        Grid search over rho, take minimum p-value with correction.

        Parameters
        ----------
        G_region : (n, m_gene)
        region : Region metadata.
        weights : (m_gene,)
        allele_freq : (m_gene,)
        rho_grid : (n_rho,) grid of rho values. Default: [0, 0.1, ..., 1.0].

        Returns
        -------
        (q_opt, p_corrected, rho_opt)
        """
        from ..linalg.eigh import rotate

        G_region = G_region.to(STAT_DTYPE)
        n, m_gene = G_region.shape

        if m_gene == 0:
            return 0.0, 1.0, 0.0

        if rho_grid is None:
            rho_grid = torch.tensor(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                dtype=STAT_DTYPE,
            )

        U = self.null_fit.eigenvectors
        G_rot = rotate(G_region, U)

        if weights is None:
            if allele_freq is None:
                allele_freq = G_region.mean(dim=0) / self.ploidy
            weights = compute_skat_weights(allele_freq)

        Py = self._Py
        z = G_rot.T @ Py  # (m_gene,)

        # Q_SKAT = sum(w_j * z_j^2)
        Q_skat = (weights * z ** 2).sum().item()

        # Q_Burden_raw = (sum(sqrt(w_j) * z_j))^2
        w_sqrt = torch.sqrt(torch.clamp(weights, min=1e-20))
        burden_score = (w_sqrt * z).sum().item()
        Q_burden_raw = burden_score ** 2

        # P @ G for eigenvalue computation
        PG = self._apply_P(G_rot)  # (n, m_gene)
        GtPG = G_rot.T @ PG  # (m_gene, m_gene)

        # For each rho, compute Q_rho and its p-value
        best_p = 1.0
        best_rho = 0.0
        best_q = Q_skat

        for rho in rho_grid.tolist():
            Q_rho = (1.0 - rho) * Q_skat + rho * Q_burden_raw

            if Q_rho <= 0:
                continue

            # Kernel for this rho: R_rho = (1-rho)*diag(w) + rho*w_sqrt @ w_sqrt^T
            # M_rho = R_rho^{1/2} @ G^T P G @ R_rho^{1/2}
            # For eigenvalue computation, form:
            # M_rho = (1-rho) * W^{1/2} GtPG W^{1/2} + rho * (w_sqrt^T GtPG w_sqrt) * I_rank1
            # Simpler: directly form the kernel matrix
            w_half = torch.sqrt(torch.clamp(weights, min=1e-20))

            if rho < 1.0 - 1e-10:
                # Mixed kernel
                R_half_diag = torch.sqrt((1.0 - rho) * weights)  # (m_gene,)
                R_half_rank1 = math.sqrt(rho) * w_sqrt  # (m_gene,)

                # M = R_half_diag * GtPG * R_half_diag + R_half_rank1 @ R_half_rank1^T * GtPG
                M_diag = (R_half_diag.unsqueeze(1) * GtPG) * R_half_diag.unsqueeze(0)
                rank1_GtPG = (R_half_rank1 @ GtPG) * R_half_rank1  # outer not needed, use eigenvalue directly
                # Actually: M_rho = R_rho_half @ GtPG @ R_rho_half where R_rho = (1-rho)*diag(w) + rho*ww^T
                # More robust: form full R_rho and take sqrt
                R_rho = (1.0 - rho) * torch.diag(weights) + rho * (w_sqrt.unsqueeze(1) * w_sqrt.unsqueeze(0))
                # R_rho is SPD, so R_rho^{1/2} exists
                try:
                    L_R = torch.linalg.cholesky(R_rho + 1e-10 * torch.eye(m_gene, dtype=STAT_DTYPE))
                    M_rho = L_R.T @ GtPG @ L_R
                except Exception:
                    M_rho = M_diag  # fallback to diagonal-only
            else:
                # Pure burden: rank-1 kernel
                # Q_rho = Q_burden_raw, eigenvalue = w_sqrt^T GtPG w_sqrt
                lam_burden = (w_sqrt @ GtPG @ w_sqrt).item()
                if lam_burden > 1e-15:
                    from scipy.stats import chi2
                    p_rho = chi2.sf(Q_rho / lam_burden, df=1)
                else:
                    p_rho = 1.0
                if p_rho < best_p:
                    best_p = p_rho
                    best_rho = rho
                    best_q = Q_rho
                continue

            lambdas = torch.linalg.eigvalsh(M_rho)
            lambdas = lambdas[lambdas > 1e-10]

            if len(lambdas) == 0:
                continue

            p_rho = mixture_chi2_pvalue(Q_rho, lambdas, method="davies")

            if p_rho < best_p:
                best_p = p_rho
                best_rho = rho
                best_q = Q_rho

        # Bonferroni-like correction for searching over rho grid
        # Conservative but simple; Lee et al. (2012) is more sophisticated
        n_rho = len(rho_grid)
        p_corrected = min(best_p * n_rho, 1.0)

        return best_q, p_corrected, best_rho

    def scan_regions(
        self,
        G_full: Tensor,
        regions: list[Region],
        variant_chr: list[str],
        variant_pos: list[int],
        test: str = "skat",
    ) -> SetBasedResult:
        """Scan all regions from a full genotype matrix.

        Parameters
        ----------
        G_full : (n, m) full genotype matrix.
        regions : list[Region]
        variant_chr, variant_pos : variant annotations.
        test : "skat", "burden", or "skat_o".

        Returns
        -------
        SetBasedResult with per-region statistics.
        """
        # Map regions to variant indices
        region_map = map_regions_to_variants(regions, variant_chr, variant_pos)

        # Build region lookup for metadata
        region_lookup = {r.region_id: r for r in regions}

        result_ids = []
        result_chr = []
        result_start = []
        result_end = []
        result_nvars = []
        result_q = []
        result_p = []
        result_rho = [] if test == "skat_o" else None

        for region_id, var_indices in region_map.items():
            region = region_lookup[region_id]
            G_region = G_full[:, var_indices]

            if test == "skat":
                q, p = self.skat(G_region, region)
            elif test == "burden":
                q, p = self.burden(G_region, region)
            elif test == "skat_o":
                q, p, rho = self.skat_o(G_region, region)
                result_rho.append(rho)
            else:
                raise ValueError(f"Unknown test: {test}. Use 'skat', 'burden', or 'skat_o'.")

            result_ids.append(region_id)
            result_chr.append(region.chr)
            result_start.append(region.start)
            result_end.append(region.end)
            result_nvars.append(len(var_indices))
            result_q.append(q)
            result_p.append(p)

        device = self.null_fit.device or torch.device("cpu")

        return SetBasedResult(
            region_id=result_ids,
            chr=result_chr,
            start=result_start,
            end=result_end,
            n_variants=result_nvars,
            q_stat=torch.tensor(result_q, dtype=STAT_DTYPE, device=device),
            p=torch.tensor(result_p, dtype=STAT_DTYPE, device=device),
            test=test,
            rho_opt=torch.tensor(result_rho, dtype=STAT_DTYPE, device=device) if result_rho is not None else None,
        )

    def scan_regions_streaming(
        self,
        chunk_iter,
        regions: list[Region],
        variant_chr: list[str],
        variant_pos: list[int],
        test: str = "skat",
        device=None,
        impute: bool = True,
    ) -> SetBasedResult:
        """Streaming variant of :meth:`scan_regions`.

        Iterates over genotype chunks (without ever holding the full
        ``(n, m)`` matrix in memory) and accumulates per-region buffers.
        Peak memory is bounded by ``n_samples * sum(region_size_j)`` —
        typically a small fraction of the full genome — plus one chunk
        at a time.

        This is the biobank-friendly API: at UKB scale (``n=500_000``,
        ``m=10_000_000``) the materialized path costs ~40 TB float64;
        a typical exome region scan with ~20K genes × ~50 SNPs/gene =
        1M region-SNPs lives in ~4 GB.

        Parameters
        ----------
        chunk_iter : iterator yielding ``(G_chunk, vmeta)``
            Same protocol as ``GenotypeReader.iter_chunks``. Variant
            indexing must be in the global order matching ``variant_chr``
            / ``variant_pos`` — chunks are concatenated left-to-right.
        regions : list[Region]
        variant_chr, variant_pos : variant annotations *across the full
            genome* (not per-chunk). Same length as the total number of
            variants the iterator will yield.
        test : "skat", "burden", or "skat_o".
        device : torch.device, optional
            Device to place per-region buffers on. Defaults to the null-fit
            device (or CPU).
        impute : bool
            Mean-impute each chunk before copying into region buffers.
            Set to False if chunks are already imputed.

        Returns
        -------
        SetBasedResult
            Identical layout to ``scan_regions`` — by-construction
            numerically equivalent for the same input data.
        """
        from ..preprocess.impute import impute_mean

        device = device or self.null_fit.device or torch.device("cpu")

        # Pass 1: build region → global-variant-index map (already efficient;
        # does not need G itself).
        region_map = map_regions_to_variants(regions, variant_chr, variant_pos)
        region_lookup = {r.region_id: r for r in regions}

        # Build, for each region, a tensor of (region_local_idx → global_idx)
        # plus a destination buffer to be filled chunk-by-chunk. Skip empty
        # regions (already dropped by map_regions_to_variants).
        region_global_idx: dict[str, list[int]] = {}
        # Inverse map: global variant index → list of (region_id, local_idx)
        # so a single chunk read can fan out to every region overlapping it.
        global_to_targets: dict[int, list[tuple[str, int]]] = {}
        for region_id, var_indices in region_map.items():
            region_global_idx[region_id] = list(var_indices)
            for local_idx, global_idx in enumerate(var_indices):
                global_to_targets.setdefault(global_idx, []).append(
                    (region_id, local_idx),
                )

        # Allocate per-region buffers. n_samples is unknown until we see
        # the first chunk, so defer allocation.
        region_buffers: dict[str, Tensor] = {}

        chunk_offset = 0
        for G_chunk, _vmeta in chunk_iter:
            G_chunk = G_chunk.to(STAT_DTYPE)
            if impute:
                G_chunk = impute_mean(G_chunk)
            G_chunk = G_chunk.to(device)
            n_chunk_samples, n_chunk_variants = G_chunk.shape

            # First chunk fixes n_samples — allocate region buffers now.
            if not region_buffers:
                for region_id, var_indices in region_global_idx.items():
                    region_buffers[region_id] = torch.empty(
                        (n_chunk_samples, len(var_indices)),
                        dtype=STAT_DTYPE, device=device,
                    )

            # Fan out: for every chunk-local column, find which region(s)
            # it lands in and copy.
            chunk_end = chunk_offset + n_chunk_variants
            for chunk_local in range(n_chunk_variants):
                global_idx = chunk_offset + chunk_local
                targets = global_to_targets.get(global_idx)
                if not targets:
                    continue
                col = G_chunk[:, chunk_local]
                for region_id, region_local in targets:
                    region_buffers[region_id][:, region_local] = col

            chunk_offset = chunk_end

        # Pass 2: run per-region tests against the assembled buffers.
        result_ids: list[str] = []
        result_chr: list[str] = []
        result_start: list[int] = []
        result_end: list[int] = []
        result_nvars: list[int] = []
        result_q: list[float] = []
        result_p: list[float] = []
        result_rho: list[float] | None = [] if test == "skat_o" else None

        for region_id, var_indices in region_map.items():
            region = region_lookup[region_id]
            G_region = region_buffers[region_id]

            if test == "skat":
                q, p = self.skat(G_region, region)
            elif test == "burden":
                q, p = self.burden(G_region, region)
            elif test == "skat_o":
                q, p, rho = self.skat_o(G_region, region)
                result_rho.append(rho)
            else:
                raise ValueError(
                    f"Unknown test: {test}. Use 'skat', 'burden', or 'skat_o'."
                )

            result_ids.append(region_id)
            result_chr.append(region.chr)
            result_start.append(region.start)
            result_end.append(region.end)
            result_nvars.append(len(var_indices))
            result_q.append(q)
            result_p.append(p)

            # Free the buffer eagerly so peak memory is bounded by the
            # *largest* region rather than the sum of all regions during
            # the per-region scan.
            del region_buffers[region_id]

        return SetBasedResult(
            region_id=result_ids,
            chr=result_chr,
            start=result_start,
            end=result_end,
            n_variants=result_nvars,
            q_stat=torch.tensor(result_q, dtype=STAT_DTYPE, device=device),
            p=torch.tensor(result_p, dtype=STAT_DTYPE, device=device),
            test=test,
            rho_opt=torch.tensor(result_rho, dtype=STAT_DTYPE, device=device) if result_rho is not None else None,
        )
