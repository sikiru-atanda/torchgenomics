"""Haplotype-based GWAS: HTR, block-based, moving-window, and SKAT.

Implements four haplotype association testing strategies:

1. **Haplotype Trend Regression (HTR)** — Zaykin et al. (2002).
   Regresses phenotype on haplotype dosages within each block/window.
   Global F-test for omnibus association, per-haplotype t-tests.

2. **Block-based haplotype GWAS** — Uses any of the 13 existing
   ``detect_blocks()`` methods to define blocks, then runs HTR per block.
   Massive multiple-testing reduction (n_blocks << n_snps).

3. **Moving-window haplotype scan** — Slides a window of ``w`` SNPs
   across the genome. Detects signals in high-recombination regions
   where block-based methods fail.

4. **Haplotype-SKAT** — Variance-component test on haplotype dosage
   columns within each block. Handles rare haplotypes naturally.

References
----------
- Zaykin et al. (2002). Testing association of statistically inferred
  haplotypes with discrete and continuous traits. Human Heredity.
- Schaid et al. (2002). Score tests for association between traits
  and haplotypes. Am J Hum Genet.
- Sham & Purcell (2014). Statistical power and significance testing
  in large-scale genetic studies. Nature Reviews Genetics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

STAT_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HaplotypeBlock:
    """Haplotype composition of a single genomic block or window.

    Attributes
    ----------
    block_id : str
        Unique identifier for this block.
    chr : str
        Chromosome label.
    start : int
        Start position (bp).
    end : int
        End position (bp).
    variant_indices : list[int]
        Column indices into the genotype matrix.
    haplotypes : list[str]
        Haplotype allele strings (e.g. ``["ACG", "ATG", ...]``).
    frequencies : Tensor
        ``(H,)`` estimated haplotype frequencies.
    dosage : Tensor
        ``(n, H)`` expected haplotype counts per individual.
    """

    block_id: str
    chr: str
    start: int
    end: int
    variant_indices: list[int]
    haplotypes: list[str]
    frequencies: Tensor
    dosage: Tensor


@dataclass
class HaplotypeGWASResult:
    """Results from a haplotype-based GWAS scan.

    Attributes
    ----------
    block_id : list[str]
        Block identifier per test.
    chr : list[str]
        Chromosome per block.
    start : list[int]
        Start position per block.
    end : list[int]
        End position per block.
    n_variants : list[int]
        Number of SNPs per block.
    n_haplotypes : list[int]
        Number of haplotypes tested per block.
    p_global : Tensor
        ``(B,)`` omnibus p-value per block.
    stat_global : Tensor
        ``(B,)`` omnibus test statistic per block.
    test : str
        Test type: ``"f_test"``, ``"lrt"``, or ``"skat"``.
    haplotype_betas : list[Tensor]
        Per-block ``(H_b - 1,)`` regression coefficients.
    haplotype_ses : list[Tensor]
        Per-block ``(H_b - 1,)`` standard errors.
    haplotype_pvals : list[Tensor]
        Per-block ``(H_b - 1,)`` per-haplotype p-values.
    haplotype_labels : list[list[str]]
        Haplotype allele strings (excluding reference) per block.
    n_obs : int
        Sample size.
    """

    block_id: list[str] = field(default_factory=list)
    chr: list[str] = field(default_factory=list)
    start: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    n_variants: list[int] = field(default_factory=list)
    n_haplotypes: list[int] = field(default_factory=list)
    p_global: Tensor = field(default_factory=lambda: torch.tensor([]))
    stat_global: Tensor = field(default_factory=lambda: torch.tensor([]))
    test: str = "f_test"
    haplotype_betas: list[Tensor] = field(default_factory=list)
    haplotype_ses: list[Tensor] = field(default_factory=list)
    haplotype_pvals: list[Tensor] = field(default_factory=list)
    haplotype_labels: list[list[str]] = field(default_factory=list)
    n_obs: int = 0


# ---------------------------------------------------------------------------
# Section B: Haplotype construction
# ---------------------------------------------------------------------------

def _enumerate_haplotypes_phased(
    haps_block: Tensor,
    ploidy: int = 2,
) -> tuple[list[str], Tensor, Tensor]:
    """Build haplotype dosages from phased data.

    Parameters
    ----------
    haps_block : (n, ploidy, m_block) int/float
        Phased haplotype alleles for a genomic block.
    ploidy : int
        Ploidy level.

    Returns
    -------
    labels : list[str]
        Unique haplotype strings.
    freqs : (H,) Tensor
        Haplotype frequencies summing to 1.
    dosage : (n, H) Tensor
        Count of each haplotype per individual (sums to ``ploidy``).
    """
    n = haps_block.shape[0]
    device = haps_block.device
    haps_int = haps_block.long()  # (n, ploidy, m_block)

    # Encode each haploid copy as a string
    hap_strings: list[list[str]] = []
    for i in range(n):
        row_haps = []
        for p in range(ploidy):
            s = "".join(str(int(v)) for v in haps_int[i, p].tolist())
            row_haps.append(s)
        hap_strings.append(row_haps)

    # Collect unique haplotypes
    all_haps = [h for row in hap_strings for h in row]
    unique = sorted(set(all_haps))
    hap_to_idx = {h: i for i, h in enumerate(unique)}
    H = len(unique)

    # Build dosage matrix
    dosage = torch.zeros(n, H, dtype=STAT_DTYPE, device=device)
    for i, row in enumerate(hap_strings):
        for h in row:
            dosage[i, hap_to_idx[h]] += 1.0

    freqs = dosage.sum(dim=0) / (n * ploidy)
    return unique, freqs, dosage


def _enumerate_haplotypes_unphased(
    G_block: Tensor,
    max_iter: int = 50,
    tol: float = 1e-6,
    max_haplotypes: int = 20,
    min_freq: float = 1e-4,
) -> tuple[list[str], Tensor, Tensor]:
    """Multi-locus EM for haplotype inference from unphased genotypes.

    Extends the existing 2-locus Hill (1974) EM to arbitrary block size
    by enumerating candidate haplotypes compatible with observed genotype
    vectors, then running the standard E-step / M-step loop.

    Parameters
    ----------
    G_block : (n, m_block)
        Genotype dosages (rounded to integers 0..2).
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on max frequency change.
    max_haplotypes : int
        Cap on the number of candidate haplotypes.
    min_freq : float
        Prune haplotypes below this frequency during EM.

    Returns
    -------
    labels : list[str]
        Haplotype allele strings.
    freqs : (H,) Tensor
        Estimated haplotype frequencies.
    dosage : (n, H) Tensor
        Expected haplotype dosage per individual.
    """
    n, m = G_block.shape
    device = G_block.device
    G_int = G_block.round().long().clamp(0, 2)  # (n, m)

    # --- Enumerate candidate haplotypes from observed genotypes ---
    # A haplotype is a binary vector (0/1)^m. For each individual,
    # genotype g_j ∈ {0,1,2} constrains allele j: g=0 → both copies 0,
    # g=2 → both copies 1, g=1 → one copy 0, one copy 1.
    # We enumerate all haplotypes compatible with at least one individual.
    candidate_set: set[str] = set()
    for i in range(n):
        g = G_int[i].tolist()
        # For each locus, possible alleles on one chromosome
        per_locus = []
        for gj in g:
            if gj == 0:
                per_locus.append([0])
            elif gj == 2:
                per_locus.append([1])
            else:
                per_locus.append([0, 1])
        # Enumerate compatible haplotypes for this individual
        for combo in product(*per_locus):
            candidate_set.add("".join(str(a) for a in combo))
            if len(candidate_set) > max_haplotypes * 10:
                break
        if len(candidate_set) > max_haplotypes * 10:
            break

    candidates = sorted(candidate_set)

    # If too many, keep the most frequent based on marginal allele freqs
    if len(candidates) > max_haplotypes:
        af = G_int.float().mean(dim=0) / 2.0  # (m,)
        scored = []
        for h in candidates:
            prob = 1.0
            for j, c in enumerate(h):
                p = af[j].item()
                prob *= (p if int(c) == 1 else (1.0 - p))
            scored.append((prob, h))
        scored.sort(key=lambda x: -x[0])
        candidates = [h for _, h in scored[:max_haplotypes]]

    H = len(candidates)
    if H == 0:
        empty_f = torch.zeros(0, dtype=STAT_DTYPE, device=device)
        return [], empty_f, torch.zeros(n, 0, dtype=STAT_DTYPE, device=device)

    # Convert candidates to integer tensor for fast lookup
    hap_mat = torch.tensor(
        [[int(c) for c in h] for h in candidates],
        dtype=torch.long, device=device,
    )  # (H, m)

    # --- Initialize frequencies uniformly ---
    freq = torch.ones(H, dtype=STAT_DTYPE, device=device) / H

    # --- Precompute compatible haplotype pairs per individual ---
    # For each individual, find all (h_a, h_b) pairs with h_a + h_b == g_i
    compat: list[list[tuple[int, int]]] = []
    for i in range(n):
        g = G_int[i]  # (m,)
        pairs = []
        for a in range(H):
            for b in range(a, H):
                if (hap_mat[a] + hap_mat[b] == g).all():
                    pairs.append((a, b))
        compat.append(pairs)

    # --- EM loop ---
    dosage = torch.zeros(n, H, dtype=STAT_DTYPE, device=device)

    for it in range(max_iter):
        freq_old = freq.clone()

        # E-step: compute posterior P(h_a, h_b | g_i, freq)
        # and accumulate expected haplotype counts
        hap_counts = torch.zeros(H, dtype=STAT_DTYPE, device=device)
        dosage.zero_()

        for i in range(n):
            pairs = compat[i]
            if not pairs:
                continue

            # Compute unnormalized probabilities
            probs = []
            for a, b in pairs:
                if a == b:
                    p = freq[a] * freq[b]
                else:
                    p = 2.0 * freq[a] * freq[b]
                probs.append(p)

            total = sum(probs)
            if total < 1e-300:
                continue

            for (a, b), p in zip(pairs, probs):
                w = p / total
                if a == b:
                    hap_counts[a] += 2.0 * w
                    dosage[i, a] += 2.0 * w
                else:
                    hap_counts[a] += w
                    hap_counts[b] += w
                    dosage[i, a] += w
                    dosage[i, b] += w

        # M-step: update frequencies
        total_alleles = hap_counts.sum()
        if total_alleles < 1e-300:
            break
        freq = hap_counts / total_alleles

        # Prune rare haplotypes
        keep = freq >= min_freq
        if keep.sum() < H and keep.sum() > 0:
            keep_idx = torch.where(keep)[0]
            freq = freq[keep_idx]
            freq = freq / freq.sum()
            hap_mat = hap_mat[keep_idx]
            candidates = [candidates[j] for j in keep_idx.tolist()]
            dosage = dosage[:, keep_idx]
            H = len(candidates)

            # Rebuild compatible pairs
            compat_new: list[list[tuple[int, int]]] = []
            for i in range(n):
                g = G_int[i]
                pairs = []
                for a in range(H):
                    for b in range(a, H):
                        if (hap_mat[a] + hap_mat[b] == g).all():
                            pairs.append((a, b))
                compat_new.append(pairs)
            compat = compat_new

        # Check convergence
        if H == len(freq_old) and (freq - freq_old[:H]).abs().max() < tol:
            break

    return candidates, freq, dosage


def _pool_rare_haplotypes(
    labels: list[str],
    freqs: Tensor,
    dosage: Tensor,
    min_freq: float,
) -> tuple[list[str], Tensor, Tensor]:
    """Merge rare haplotypes into an "OTHER" category.

    Parameters
    ----------
    labels : list[str]
        Haplotype labels.
    freqs : (H,) Tensor
    dosage : (n, H) Tensor
    min_freq : float
        Haplotypes below this frequency are pooled.

    Returns
    -------
    labels_new, freqs_new, dosage_new
    """
    if len(labels) == 0:
        return labels, freqs, dosage

    keep_mask = freqs >= min_freq
    n_keep = keep_mask.sum().item()

    if n_keep == len(labels):
        return labels, freqs, dosage

    keep_idx = torch.where(keep_mask)[0]
    rare_idx = torch.where(~keep_mask)[0]

    new_labels = [labels[j] for j in keep_idx.tolist()]
    new_freqs = freqs[keep_idx]
    new_dosage = dosage[:, keep_idx]

    if len(rare_idx) > 0:
        other_freq = freqs[rare_idx].sum()
        other_dosage = dosage[:, rare_idx].sum(dim=1, keepdim=True)
        new_labels.append("OTHER")
        new_freqs = torch.cat([new_freqs, other_freq.unsqueeze(0)])
        new_dosage = torch.cat([new_dosage, other_dosage], dim=1)

    return new_labels, new_freqs, new_dosage


# ---------------------------------------------------------------------------
# Section C: Statistical tests
# ---------------------------------------------------------------------------

def _htr_f_test(
    Y: Tensor,
    X0: Tensor,
    D: Tensor,
) -> tuple[float, float, Tensor, Tensor, Tensor]:
    """HTR global F-test and per-haplotype t-tests.

    Parameters
    ----------
    Y : (n,) phenotype
    X0 : (n, c) covariates (including intercept)
    D : (n, H-1) haplotype dosages (reference dropped)

    Returns
    -------
    F_stat : float
    p_global : float
    betas : (H-1,) per-haplotype effects
    ses : (H-1,) standard errors
    p_per_hap : (H-1,) per-haplotype p-values
    """
    from scipy.stats import f as f_dist, t as t_dist

    n = Y.shape[0]
    c = X0.shape[1]
    h = D.shape[1]  # H - 1

    if h == 0:
        return 0.0, 1.0, torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Null model: Y = X0 @ b0 + e
    Q0, R0 = torch.linalg.qr(X0)
    resid_null = Y - Q0 @ (Q0.T @ Y)
    RSS_null = (resid_null ** 2).sum().item()

    # Full model: Y = [X0, D] @ b + e
    X_full = torch.cat([X0, D], dim=1)  # (n, c + h)
    Q, R = torch.linalg.qr(X_full)
    resid_full = Y - Q @ (Q.T @ Y)
    RSS_full = (resid_full ** 2).sum().item()

    # F statistic
    df1 = h
    df2 = n - c - h
    if df2 <= 0 or RSS_full <= 0:
        return 0.0, 1.0, torch.zeros(h), torch.zeros(h), torch.ones(h)

    F_stat = ((RSS_null - RSS_full) / df1) / (RSS_full / df2)
    p_global = float(f_dist.sf(F_stat, df1, df2))

    # Per-haplotype effects via OLS
    b = torch.linalg.lstsq(X_full, Y.unsqueeze(1)).solution.squeeze(1)  # (c+h,)
    betas = b[c:]  # (h,)

    sig2 = RSS_full / df2
    try:
        XtX_inv = torch.linalg.inv(X_full.T @ X_full)
    except torch.linalg.LinAlgError:
        XtX_inv = torch.linalg.pinv(X_full.T @ X_full)
    se_all = torch.sqrt(torch.clamp(sig2 * XtX_inv.diag(), min=1e-300))
    ses = se_all[c:]

    t_vals = betas / ses
    p_per_hap = torch.tensor(
        [2.0 * float(t_dist.sf(abs(t.item()), df2)) for t in t_vals],
        dtype=STAT_DTYPE,
    )

    return F_stat, p_global, betas, ses, p_per_hap


def _htr_lrt(
    Y: Tensor,
    X0: Tensor,
    D: Tensor,
) -> tuple[float, float, Tensor, Tensor, Tensor]:
    """HTR likelihood ratio test.

    chi2 = n * log(RSS_null / RSS_full), df = H-1.

    Returns the same tuple as ``_htr_f_test``.
    """
    from scipy.stats import chi2 as chi2_dist, t as t_dist

    n = Y.shape[0]
    c = X0.shape[1]
    h = D.shape[1]

    if h == 0:
        return 0.0, 1.0, torch.tensor([]), torch.tensor([]), torch.tensor([])

    Q0, _ = torch.linalg.qr(X0)
    resid_null = Y - Q0 @ (Q0.T @ Y)
    RSS_null = (resid_null ** 2).sum().item()

    X_full = torch.cat([X0, D], dim=1)
    Q, _ = torch.linalg.qr(X_full)
    resid_full = Y - Q @ (Q.T @ Y)
    RSS_full = (resid_full ** 2).sum().item()

    df2 = n - c - h
    if RSS_full <= 0 or RSS_null <= 0 or df2 <= 0:
        return 0.0, 1.0, torch.zeros(h), torch.zeros(h), torch.ones(h)

    chi2_stat = n * float(torch.log(torch.tensor(RSS_null / RSS_full)))
    p_global = float(chi2_dist.sf(chi2_stat, h))

    # Per-haplotype effects
    b = torch.linalg.lstsq(X_full, Y.unsqueeze(1)).solution.squeeze(1)
    betas = b[c:]
    sig2 = RSS_full / max(df2, 1)
    try:
        XtX_inv = torch.linalg.inv(X_full.T @ X_full)
    except torch.linalg.LinAlgError:
        XtX_inv = torch.linalg.pinv(X_full.T @ X_full)
    se_all = torch.sqrt(torch.clamp(sig2 * XtX_inv.diag(), min=1e-300))
    ses = se_all[c:]

    t_vals = betas / ses
    p_per_hap = torch.tensor(
        [2.0 * float(t_dist.sf(abs(t.item()), max(df2, 1))) for t in t_vals],
        dtype=STAT_DTYPE,
    )

    return chi2_stat, p_global, betas, ses, p_per_hap


def _htr_with_lmm(
    Y_rot: Tensor,
    X0_rot: Tensor,
    D_rot: Tensor,
    H_inv: Tensor,
) -> tuple[float, float, Tensor, Tensor, Tensor]:
    """HTR in rotated eigenspace for LMM correction (WLS).

    Parameters
    ----------
    Y_rot : (n,) rotated phenotype
    X0_rot : (n, c) rotated covariates
    D_rot : (n, H-1) rotated haplotype dosages
    H_inv : (n,) diagonal weights = 1 / (eigenvalues * lambda + 1)

    Returns the same tuple as ``_htr_f_test``.
    """
    from scipy.stats import f as f_dist, t as t_dist

    n = Y_rot.shape[0]
    c = X0_rot.shape[1]
    h = D_rot.shape[1]

    if h == 0:
        return 0.0, 1.0, torch.tensor([]), torch.tensor([]), torch.tensor([])

    W = H_inv  # (n,) diagonal weights
    sqrtW = torch.sqrt(torch.clamp(W, min=1e-300))

    # Weight everything
    Y_w = sqrtW * Y_rot
    X0_w = sqrtW.unsqueeze(1) * X0_rot
    D_w = sqrtW.unsqueeze(1) * D_rot

    # Null: WLS with X0 only
    Q0, _ = torch.linalg.qr(X0_w)
    resid_null = Y_w - Q0 @ (Q0.T @ Y_w)
    RSS_null = (resid_null ** 2).sum().item()

    # Full: WLS with [X0, D]
    X_full_w = torch.cat([X0_w, D_w], dim=1)
    Q, _ = torch.linalg.qr(X_full_w)
    resid_full = Y_w - Q @ (Q.T @ Y_w)
    RSS_full = (resid_full ** 2).sum().item()

    df1 = h
    df2 = n - c - h
    if df2 <= 0 or RSS_full <= 0:
        return 0.0, 1.0, torch.zeros(h), torch.zeros(h), torch.ones(h)

    F_stat = ((RSS_null - RSS_full) / df1) / (RSS_full / df2)
    p_global = float(f_dist.sf(F_stat, df1, df2))

    # Per-haplotype effects via WLS
    X_full = torch.cat([X0_rot, D_rot], dim=1)
    XtWX = X_full.T @ (W.unsqueeze(1) * X_full)
    XtWy = X_full.T @ (W * Y_rot)
    try:
        b = torch.linalg.solve(XtWX, XtWy)
    except torch.linalg.LinAlgError:
        b = torch.linalg.lstsq(XtWX, XtWy.unsqueeze(1)).solution.squeeze(1)

    betas = b[c:]
    sig2 = RSS_full / df2
    try:
        XtWX_inv = torch.linalg.inv(XtWX)
    except torch.linalg.LinAlgError:
        XtWX_inv = torch.linalg.pinv(XtWX)
    se_all = torch.sqrt(torch.clamp(sig2 * XtWX_inv.diag(), min=1e-300))
    ses = se_all[c:]

    t_vals = betas / ses
    p_per_hap = torch.tensor(
        [2.0 * float(t_dist.sf(abs(t.item()), df2)) for t in t_vals],
        dtype=STAT_DTYPE,
    )

    return F_stat, p_global, betas, ses, p_per_hap


def _haplotype_skat(
    D_rot: Tensor,
    Py: Tensor,
    apply_P: callable,
) -> tuple[float, float]:
    """SKAT variance-component test on haplotype dosages.

    Q = z^T z  where z = D_rot^T @ Py
    Under H0, Q ~ sum(lambda_k * chi2_1) where lambda_k are eigenvalues
    of D_rot^T @ P @ D_rot.

    Parameters
    ----------
    D_rot : (n, H-1) rotated haplotype dosages
    Py : (n,) P @ y in rotated space
    apply_P : callable
        Function that applies the P operator to (n, k) matrices.

    Returns
    -------
    Q : float
    p : float
    """
    from ..stats.mixture import mixture_chi2_pvalue

    h = D_rot.shape[1]
    if h == 0:
        return 0.0, 1.0

    z = D_rot.T @ Py  # (h,)
    Q = (z ** 2).sum().item()

    if Q <= 0:
        return 0.0, 1.0

    PD = apply_P(D_rot)  # (n, h)
    DtPD = D_rot.T @ PD  # (h, h)

    lambdas = torch.linalg.eigvalsh(DtPD)
    lambdas = lambdas[lambdas > 1e-10]

    if len(lambdas) == 0:
        return Q, 1.0

    p = mixture_chi2_pvalue(Q, lambdas, method="davies")
    return Q, p


# ---------------------------------------------------------------------------
# Section D: Main class
# ---------------------------------------------------------------------------

class HaplotypeGWAS:
    """Haplotype-based GWAS scanner.

    Parameters
    ----------
    method : str
        ``"block"`` for LD-block-based scan, ``"window"`` for moving-window.
    test : str
        ``"f_test"``, ``"lrt"``, or ``"skat"``.
    window_size : int
        Number of SNPs per window (for ``method="window"``).
    step : int
        Step size for moving window (default 1).
    block_method : str
        LD block detection method (for ``method="block"``).
    min_hap_freq : float
        Minimum haplotype frequency; rarer haplotypes are pooled.
    max_haplotypes : int
        Maximum number of haplotypes per block.
    em_max_iter : int
        EM maximum iterations for unphased haplotype inference.
    em_tol : float
        EM convergence tolerance.
    null_fit : NullFit, optional
        Fitted null model for LMM-corrected tests.
    ploidy : int
        Ploidy level (default 2). For ploidy > 2, phased input required.
    """

    def __init__(
        self,
        method: str = "block",
        test: str = "f_test",
        window_size: int = 5,
        step: int = 1,
        block_method: str = "gabriel",
        min_hap_freq: float = 0.01,
        max_haplotypes: int = 20,
        em_max_iter: int = 50,
        em_tol: float = 1e-6,
        null_fit=None,
        ploidy: int = 2,
    ):
        if method not in ("block", "window"):
            raise ValueError(f"method must be 'block' or 'window', got '{method}'")
        if test not in ("f_test", "lrt", "skat"):
            raise ValueError(f"test must be 'f_test', 'lrt', or 'skat', got '{test}'")

        self.method = method
        self.test = test
        self.window_size = window_size
        self.step = step
        self.block_method = block_method
        self.min_hap_freq = min_hap_freq
        self.max_haplotypes = max_haplotypes
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.null_fit = null_fit
        self.ploidy = ploidy

    def construct_haplotypes(
        self,
        G: Tensor,
        variant_pos: list[int],
        variant_chr: list[str],
        blocks: list | None = None,
        haplotypes: Tensor | None = None,
        **block_kwargs,
    ) -> list[HaplotypeBlock]:
        """Build haplotype dosage matrices for each block/window.

        Parameters
        ----------
        G : (n, m) Tensor
            Genotype dosage matrix.
        variant_pos : list[int]
            Per-variant base-pair positions.
        variant_chr : list[str]
            Per-variant chromosome labels.
        blocks : list[LDBlock], optional
            Pre-computed LD blocks. If None and ``method="block"``,
            blocks are auto-detected.
        haplotypes : (n, ploidy, m) Tensor, optional
            Phased haplotype data. If provided, uses direct counting
            instead of EM.
        **block_kwargs
            Extra arguments passed to ``detect_blocks()``.

        Returns
        -------
        list[HaplotypeBlock]
        """
        n, m = G.shape

        # --- Define genomic segments ---
        if self.method == "block":
            if blocks is None:
                from ..ld import detect_blocks
                blocks = detect_blocks(
                    G, variant_pos, variant_chr,
                    haplotypes=haplotypes,
                    method=self.block_method,
                    ploidy=self.ploidy,
                    **block_kwargs,
                )
            segments = []
            for blk in blocks:
                vi = blk.variant_indices
                if vi is None or len(vi) < 1:
                    continue
                segments.append((
                    blk.region.region_id,
                    blk.region.chr,
                    blk.region.start,
                    blk.region.end,
                    vi,
                ))
        else:
            # Moving window
            segments = []
            chr_set = sorted(set(variant_chr))
            for chrom in chr_set:
                chr_idx = [i for i, c in enumerate(variant_chr) if c == chrom]
                for w_start in range(0, len(chr_idx), self.step):
                    w_end = min(w_start + self.window_size, len(chr_idx))
                    if w_end - w_start < 1:
                        continue
                    vi = chr_idx[w_start:w_end]
                    bp_start = variant_pos[vi[0]]
                    bp_end = variant_pos[vi[-1]]
                    bid = f"win_{chrom}_{w_start}"
                    segments.append((bid, chrom, bp_start, bp_end, vi))
                    if w_end >= len(chr_idx):
                        break

        # --- Construct haplotypes per segment ---
        result: list[HaplotypeBlock] = []
        for bid, chrom, bp_start, bp_end, vi in segments:
            vi_t = torch.tensor(vi, dtype=torch.long)

            if haplotypes is not None:
                haps_block = haplotypes[:, :, vi_t]  # (n, ploidy, m_block)
                labels, freqs, dosage = _enumerate_haplotypes_phased(
                    haps_block, ploidy=self.ploidy,
                )
            else:
                if self.ploidy > 2:
                    raise ValueError(
                        "EM haplotype inference requires ploidy <= 2. "
                        "For polyploid organisms, provide phased haplotypes."
                    )
                G_block = G[:, vi_t].to(STAT_DTYPE)
                labels, freqs, dosage = _enumerate_haplotypes_unphased(
                    G_block,
                    max_iter=self.em_max_iter,
                    tol=self.em_tol,
                    max_haplotypes=self.max_haplotypes,
                )

            # Pool rare haplotypes
            if self.min_hap_freq > 0 and len(labels) > 0:
                labels, freqs, dosage = _pool_rare_haplotypes(
                    labels, freqs, dosage, self.min_hap_freq,
                )

            result.append(HaplotypeBlock(
                block_id=bid,
                chr=chrom,
                start=bp_start,
                end=bp_end,
                variant_indices=vi,
                haplotypes=labels,
                frequencies=freqs,
                dosage=dosage,
            ))

        return result

    def scan(
        self,
        Y: Tensor,
        G: Tensor,
        X0: Optional[Tensor] = None,
        variant_pos: list[int] | None = None,
        variant_chr: list[str] | None = None,
        blocks: list | None = None,
        haplotypes: Tensor | None = None,
        **block_kwargs,
    ) -> HaplotypeGWASResult:
        """Run haplotype-based GWAS scan.

        Parameters
        ----------
        Y : (n,) Tensor
            Phenotype vector.
        G : (n, m) Tensor
            Genotype dosage matrix.
        X0 : (n, c) Tensor, optional
            Covariates. If None, an intercept-only design is used.
        variant_pos : list[int], optional
            Per-variant base-pair positions.
        variant_chr : list[str], optional
            Per-variant chromosome labels.
        blocks : list[LDBlock], optional
            Pre-computed LD blocks.
        haplotypes : (n, ploidy, m) Tensor, optional
            Phased haplotype data.
        **block_kwargs
            Extra arguments passed to ``detect_blocks()``.

        Returns
        -------
        HaplotypeGWASResult
        """
        n, m = G.shape
        Y = Y.to(STAT_DTYPE)

        if X0 is None:
            X0 = torch.ones(n, 1, dtype=STAT_DTYPE, device=G.device)
        else:
            X0 = X0.to(STAT_DTYPE)

        if variant_pos is None:
            variant_pos = list(range(m))
        if variant_chr is None:
            variant_chr = ["1"] * m

        # Construct haplotypes
        hap_blocks = self.construct_haplotypes(
            G, variant_pos, variant_chr,
            blocks=blocks, haplotypes=haplotypes,
            **block_kwargs,
        )

        # LMM rotation if null_fit provided
        use_lmm = self.null_fit is not None
        if use_lmm:
            nf = self.null_fit
            U = nf.eigenvectors
            Y_rot = nf.Y_rot.to(STAT_DTYPE)
            X0_rot = nf.X0_rot.to(STAT_DTYPE)
            lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
            H_inv = 1.0 / (nf.eigenvalues.to(STAT_DTYPE) * lam + 1.0)

            # For SKAT, pre-compute Py and P-operator
            if self.test == "skat":
                from ..optim.reml_math import _compute_P_quantities
                Py_H, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv)
                Py = Py_H / max(nf.sig2_e, 1e-20)

                def apply_P(v: Tensor) -> Tensor:
                    sig2_e = max(nf.sig2_e, 1e-20)
                    X = X0_rot
                    if v.ndim == 1:
                        wv = H_inv * v
                        wX = H_inv.unsqueeze(1) * X
                        XtWX = X.T @ wX
                        XtWv = X.T @ wv
                        correction = wX @ torch.linalg.solve(XtWX, XtWv)
                        return (wv - correction) / sig2_e
                    else:
                        wv = H_inv.unsqueeze(1) * v
                        wX = H_inv.unsqueeze(1) * X
                        XtWX = X.T @ wX
                        XtWv = X.T @ wv
                        correction = wX @ torch.linalg.solve(XtWX, XtWv)
                        return (wv - correction) / sig2_e

        # --- Run test per block ---
        block_ids: list[str] = []
        chrs: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        n_variants_list: list[int] = []
        n_haps_list: list[int] = []
        stats_list: list[float] = []
        pvals_list: list[float] = []
        betas_list: list[Tensor] = []
        ses_list: list[Tensor] = []
        phap_list: list[Tensor] = []
        labels_list: list[list[str]] = []

        for hb in hap_blocks:
            H = len(hb.haplotypes)

            # Need at least 2 haplotypes (1 reference + 1 tested)
            if H < 2:
                block_ids.append(hb.block_id)
                chrs.append(hb.chr)
                starts.append(hb.start)
                ends.append(hb.end)
                n_variants_list.append(len(hb.variant_indices))
                n_haps_list.append(H)
                stats_list.append(0.0)
                pvals_list.append(1.0)
                betas_list.append(torch.tensor([]))
                ses_list.append(torch.tensor([]))
                phap_list.append(torch.tensor([]))
                labels_list.append([])
                continue

            # Drop reference haplotype (most frequent)
            ref_idx = hb.frequencies.argmax().item()
            test_idx = [j for j in range(H) if j != ref_idx]
            D = hb.dosage[:, test_idx].to(STAT_DTYPE)  # (n, H-1)
            test_labels = [hb.haplotypes[j] for j in test_idx]

            if use_lmm:
                from ..linalg.eigh import rotate
                D_rot = rotate(D, U)

                if self.test == "skat":
                    stat, p = _haplotype_skat(D_rot, Py, apply_P)
                    betas = torch.zeros(D.shape[1])
                    ses = torch.zeros(D.shape[1])
                    p_hap = torch.ones(D.shape[1])
                else:
                    stat, p, betas, ses, p_hap = _htr_with_lmm(
                        Y_rot, X0_rot, D_rot, H_inv,
                    )
            else:
                if self.test == "skat":
                    # Without LMM, use OLS-based P operator
                    def apply_P_ols(v: Tensor) -> Tensor:
                        Q, _ = torch.linalg.qr(X0)
                        if v.ndim == 1:
                            return v - Q @ (Q.T @ v)
                        else:
                            return v - Q @ (Q.T @ v)

                    Py_ols = apply_P_ols(Y)
                    stat, p = _haplotype_skat(D, Py_ols, apply_P_ols)
                    betas = torch.zeros(D.shape[1])
                    ses = torch.zeros(D.shape[1])
                    p_hap = torch.ones(D.shape[1])
                elif self.test == "lrt":
                    stat, p, betas, ses, p_hap = _htr_lrt(Y, X0, D)
                else:
                    stat, p, betas, ses, p_hap = _htr_f_test(Y, X0, D)

            block_ids.append(hb.block_id)
            chrs.append(hb.chr)
            starts.append(hb.start)
            ends.append(hb.end)
            n_variants_list.append(len(hb.variant_indices))
            n_haps_list.append(H)
            stats_list.append(stat)
            pvals_list.append(p)
            betas_list.append(betas)
            ses_list.append(ses)
            phap_list.append(p_hap)
            labels_list.append(test_labels)

        return HaplotypeGWASResult(
            block_id=block_ids,
            chr=chrs,
            start=starts,
            end=ends,
            n_variants=n_variants_list,
            n_haplotypes=n_haps_list,
            p_global=torch.tensor(pvals_list, dtype=STAT_DTYPE),
            stat_global=torch.tensor(stats_list, dtype=STAT_DTYPE),
            test=self.test,
            haplotype_betas=betas_list,
            haplotype_ses=ses_list,
            haplotype_pvals=phap_list,
            haplotype_labels=labels_list,
            n_obs=n,
        )
