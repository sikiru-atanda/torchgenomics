"""Novel haplotype GWAS methods: PCHT, HHCT, HSKAT, HapGxE, BayesHap.

Five novel contributions beyond standard HTR/block/window/SKAT:

1. **PCHT** — Posterior-Calibrated Haplotype Test.  Propagates EM phase
   uncertainty into the score-test denominator (law of total variance),
   analogous to GU-LMM dosage-variance correction.

2. **HHCT** — Hierarchical Haplotype Collapsing Test.  Builds a Hamming-
   distance merge tree over haplotypes and tests at every resolution
   with Meinshausen (2008) hierarchical FWER control.

3. **HSKAT** — Haplotype Similarity Kernel Association Test.  Replaces the
   identity weight in standard SKAT with an exponential-decay kernel on
   Hamming distance, borrowing strength across similar haplotypes.

4. **HapGxE** — Haplotype-by-Environment Interaction Test.  Tests whether
   haplotype effects vary across an environment covariate, with three
   tests: main, interaction, and joint.

5. **BayesHap** — Bayesian Haplotype Fine-Mapping.  Runs SuSiE IBSS on
   concatenated haplotype dosage columns, producing haplotype-level PIPs
   and 95 % credible sets.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

STAT_DTYPE = torch.float64


# ═══════════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PCHTResult:
    """Results from Posterior-Calibrated Haplotype Test.

    Attributes
    ----------
    stat : float
        Chi-squared test statistic.
    p : float
        P-value from chi2(H-1).
    correction : Tensor
        (H-1, H-1) uncertainty correction matrix Δ.
    """

    stat: float
    p: float
    correction: Tensor


@dataclass
class HHCTNode:
    """A single node in the HHCT merge tree.

    Attributes
    ----------
    node_id : int
    children : list[int]
        Node IDs of children (empty for leaves).
    leaf_indices : list[int]
        Indices of original haplotypes under this node.
    raw_p : float
        Raw p-value for the merged dosage test.
    adjusted_p : float
        Meinshausen-adjusted p-value.
    rejected : bool
        Whether this node is rejected at the target alpha.
    """

    node_id: int
    children: list[int]
    leaf_indices: list[int]
    raw_p: float
    adjusted_p: float
    rejected: bool


@dataclass
class HHCTResult:
    """Results from Hierarchical Haplotype Collapsing Test.

    Attributes
    ----------
    nodes : list[HHCTNode]
        All nodes (leaves + internal) in the tree.
    rejected_leaves : list[int]
        Indices of rejected leaf-level haplotypes.
    n_leaves : int
        Total number of haplotypes (leaves).
    alpha : float
        FWER control level.
    """

    nodes: list[HHCTNode]
    rejected_leaves: list[int]
    n_leaves: int
    alpha: float


@dataclass
class HapGxEResult:
    """Results from Haplotype-by-Environment Interaction Test.

    Attributes
    ----------
    F_main, p_main : float
        Main haplotype effect test.
    F_interaction, p_interaction : float
        Interaction test (haplotype × environment).
    F_joint, p_joint : float
        Joint test for any haplotype effect.
    """

    F_main: float
    p_main: float
    F_interaction: float
    p_interaction: float
    F_joint: float
    p_joint: float


@dataclass
class BayesHapResult:
    """Results from Bayesian Haplotype Fine-Mapping.

    Attributes
    ----------
    pip : Tensor
        (p,) posterior inclusion probability per haplotype column.
    alpha : Tensor
        (L, p) per-layer selection probabilities.
    mu : Tensor
        (L, p) per-layer posterior means.
    credible_sets : list[list[int]]
        Per-layer 95 % credible sets (indices into pip).
    sigma2_e : float
        Estimated residual variance.
    elbo_trace : list[float]
        ELBO at each iteration.
    converged : bool
    haplotype_labels : list[str]
        Label for each column in the dosage matrix.
    block_ids : list[str]
        Block ID for each column.
    """

    pip: Tensor
    alpha: Tensor
    mu: Tensor
    credible_sets: list[list[int]]
    sigma2_e: float
    elbo_trace: list[float]
    converged: bool
    haplotype_labels: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# 1. PCHT — Posterior-Calibrated Haplotype Test
# ═══════════════════════════════════════════════════════════════════════

def _reconstruct_compat(
    G_block: Tensor,
    hap_mat: Tensor,
) -> list[list[tuple[int, int]]]:
    """Rebuild compatible haplotype pair list per individual.

    Parameters
    ----------
    G_block : (n, m) int-valued genotypes (0/1/2).
    hap_mat : (H, m) int-valued haplotype alleles.

    Returns
    -------
    compat : length-n list of lists of (a, b) index pairs.
    """
    n = G_block.shape[0]
    H = hap_mat.shape[0]
    G_int = G_block.round().long().clamp(0, 2)

    compat: list[list[tuple[int, int]]] = []
    for i in range(n):
        g = G_int[i]
        pairs = []
        for a in range(H):
            for b in range(a, H):
                if (hap_mat[a] + hap_mat[b] == g).all():
                    pairs.append((a, b))
        compat.append(pairs)
    return compat


def compute_dosage_posterior_cov(
    G_block: Tensor,
    labels: list[str],
    freq: Tensor,
) -> Tensor:
    """Compute per-individual posterior covariance Cov(D_i | g_i).

    Returns the *summed* covariance across individuals:
        Σ_{hh'} = Σ_i Cov(D_{ih}, D_{ih'} | g_i)

    This is what enters the PCHT correction directly.

    Parameters
    ----------
    G_block : (n, m) genotype dosages.
    labels : list[str], length H.
    freq : (H,) haplotype frequencies.

    Returns
    -------
    cov_sum : (H, H) Tensor
    """
    H = len(labels)
    n = G_block.shape[0]
    device = G_block.device

    if H == 0:
        return torch.zeros(0, 0, dtype=STAT_DTYPE, device=device)

    hap_mat = torch.tensor(
        [[int(c) for c in h] for h in labels],
        dtype=torch.long, device=device,
    )
    compat = _reconstruct_compat(G_block, hap_mat)

    cov_sum = torch.zeros(H, H, dtype=STAT_DTYPE, device=device)

    for i in range(n):
        pairs = compat[i]
        if not pairs:
            continue

        # Posterior weights
        probs = []
        for a, b in pairs:
            p = (2.0 if a != b else 1.0) * freq[a] * freq[b]
            probs.append(p.item())
        total = sum(probs)
        if total < 1e-300:
            continue

        # E[D_i] and E[D_i D_i^T]
        E_D = torch.zeros(H, dtype=STAT_DTYPE, device=device)
        E_DDt = torch.zeros(H, H, dtype=STAT_DTYPE, device=device)

        for (a, b), p in zip(pairs, probs):
            w = p / total
            d = torch.zeros(H, dtype=STAT_DTYPE, device=device)
            if a == b:
                d[a] = 2.0
            else:
                d[a] = 1.0
                d[b] = 1.0
            E_D += w * d
            E_DDt += w * d.unsqueeze(1) * d.unsqueeze(0)

        cov_i = E_DDt - E_D.unsqueeze(1) * E_D.unsqueeze(0)
        cov_sum += cov_i

    return cov_sum


def _compute_diag_P_ols(X0: Tensor) -> Tensor:
    """Compute diag(P_OLS) where P = I - X(X^TX)^{-1}X^T.

    Parameters
    ----------
    X0 : (n, c)

    Returns
    -------
    diag_P : (n,)
    """
    Q, _ = torch.linalg.qr(X0.to(STAT_DTYPE))
    # leverage h_ii = ||Q[i,:]||^2
    h = (Q ** 2).sum(dim=1)
    return 1.0 - h


def _compute_diag_P_lmm(null_fit) -> Tensor:
    """Compute diag(P) in original space from LMM null fit.

    Reuses the GU-LMM algorithm.
    """
    from .gu_lmm import _compute_diag_P
    return _compute_diag_P(null_fit)


def pcht_score_test(
    Y: Tensor,
    X0: Tensor,
    D: Tensor,
    dosage_cov_sum: Tensor,
    diag_P: Tensor | None = None,
) -> PCHTResult:
    """Posterior-Calibrated Haplotype Test (OLS, no LMM).

    Parameters
    ----------
    Y : (n,) phenotype.
    X0 : (n, c) covariates.
    D : (n, H-1) haplotype dosages (reference dropped).
    dosage_cov_sum : (H-1, H-1) summed posterior covariance across
        individuals (from ``compute_dosage_posterior_cov``, reference
        haplotype row/column removed).
    diag_P : (n,) optional, precomputed. Computed from X0 if absent.

    Returns
    -------
    PCHTResult
    """
    from scipy.stats import chi2 as chi2_dist

    n = Y.shape[0]
    h = D.shape[1]

    if h == 0:
        return PCHTResult(stat=0.0, p=1.0,
                          correction=torch.zeros(0, 0, dtype=STAT_DTYPE))

    D = D.to(STAT_DTYPE)
    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)

    if diag_P is None:
        diag_P = _compute_diag_P_ols(X0)

    # Score vector: S = D^T P y
    Q, _ = torch.linalg.qr(X0)
    Py = Y - Q @ (Q.T @ Y)
    S = D.T @ Py  # (h,)

    # Standard variance: V_S = sig2_e * D^T P D
    PD = D - Q @ (Q.T @ D)
    DtPD = D.T @ PD  # (h, h)
    RSS = (Py ** 2).sum().item()
    sig2_e = RSS / max(n - X0.shape[1], 1)

    # Correction Δ = diag_P-weighted sum of posterior covariance
    # Δ_{hh'} = sum_i Cov(D_{ih}, D_{ih'} | g_i) * diag(P)_i
    # For the aggregated version: Δ = dosage_cov_sum weighted by mean(diag_P)
    # Exact: per-individual weighting; approximate: mean(diag_P) * cov_sum
    mean_diagP = diag_P.mean().item()
    correction = mean_diagP * dosage_cov_sum

    V_corrected = sig2_e * (DtPD + correction)

    # Chi-squared test: T = S^T V_corrected^{-1} S
    try:
        L_chol = torch.linalg.cholesky(V_corrected)
        z = torch.linalg.solve_triangular(L_chol, S.unsqueeze(1), upper=False)
        stat = (z ** 2).sum().item()
    except torch.linalg.LinAlgError:
        V_inv = torch.linalg.pinv(V_corrected)
        stat = (S @ V_inv @ S).item()

    p = float(chi2_dist.sf(stat, h))
    return PCHTResult(stat=stat, p=p, correction=correction)


def pcht_score_test_lmm(
    null_fit,
    D: Tensor,
    dosage_cov_sum: Tensor,
) -> PCHTResult:
    """PCHT under LMM (rotated eigenspace).

    Parameters
    ----------
    null_fit : NullFit
    D : (n, H-1) haplotype dosages in *original* space.
    dosage_cov_sum : (H-1, H-1) summed posterior covariance.

    Returns
    -------
    PCHTResult
    """
    from scipy.stats import chi2 as chi2_dist
    from ..linalg.eigh import rotate
    from ..optim.reml_math import _compute_P_quantities

    nf = null_fit
    h = D.shape[1]
    if h == 0:
        return PCHTResult(stat=0.0, p=1.0,
                          correction=torch.zeros(0, 0, dtype=STAT_DTYPE))

    D = D.to(STAT_DTYPE)
    U = nf.eigenvectors
    D_rot = rotate(D, U)  # (n, h)
    Y_rot = nf.Y_rot.to(STAT_DTYPE)
    X0_rot = nf.X0_rot.to(STAT_DTYPE)

    lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
    H_inv = 1.0 / (nf.eigenvalues.to(STAT_DTYPE) * lam + 1.0)
    sig2_e = max(nf.sig2_e, 1e-20)

    Py_H, _ = _compute_P_quantities(Y_rot, X0_rot, H_inv)
    Py = Py_H / sig2_e

    S = D_rot.T @ Py  # (h,)

    # D_rot^T P D_rot in rotated space
    wD = H_inv.unsqueeze(1) * D_rot
    wX = H_inv.unsqueeze(1) * X0_rot
    XtWX = X0_rot.T @ wX
    XtWD = X0_rot.T @ wD
    DtPD = D_rot.T @ wD - XtWD.T @ torch.linalg.solve(XtWX, XtWD)
    # DtPD is in V^{-1} scale; divide by sig2_e² to get P-scale,
    # then multiply by sig2_e to form V_corrected = sig2_e * (DtPD_P + correction).
    DtPD_P = DtPD / sig2_e  # now in P-operator scale

    # Correction
    diag_P = _compute_diag_P_lmm(nf)
    mean_diagP = diag_P.mean().item()
    correction = mean_diagP * dosage_cov_sum

    V_corrected = sig2_e * (DtPD_P + correction)

    try:
        L_chol = torch.linalg.cholesky(V_corrected)
        z = torch.linalg.solve_triangular(L_chol, S.unsqueeze(1), upper=False)
        stat = (z ** 2).sum().item()
    except torch.linalg.LinAlgError:
        V_inv = torch.linalg.pinv(V_corrected)
        stat = (S @ V_inv @ S).item()

    p = float(chi2_dist.sf(stat, h))
    return PCHTResult(stat=stat, p=p, correction=correction)


# ═══════════════════════════════════════════════════════════════════════
# 2. HHCT — Hierarchical Haplotype Collapsing Test
# ═══════════════════════════════════════════════════════════════════════

def _hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two haplotype strings."""
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


def _build_merge_tree(
    labels: list[str],
) -> list[tuple[int, int, float]]:
    """Agglomerative clustering on Hamming distance (average linkage).

    Returns
    -------
    merges : list of (cluster_a, cluster_b, distance)
        Each entry merges two clusters. IDs 0..H-1 are leaves;
        H, H+1, ... are internal nodes.
    """
    H = len(labels)
    if H <= 1:
        return []

    # Initialize: each haplotype is its own cluster
    clusters: dict[int, list[int]] = {i: [i] for i in range(H)}
    active = set(range(H))
    next_id = H
    merges: list[tuple[int, int, int]] = []

    while len(active) > 1:
        # Find closest pair (average linkage)
        best_dist = float("inf")
        best_pair = (-1, -1)
        active_list = sorted(active)

        for ii in range(len(active_list)):
            for jj in range(ii + 1, len(active_list)):
                ci, cj = active_list[ii], active_list[jj]
                # Average Hamming distance between all leaf pairs
                total = 0
                count = 0
                for a in clusters[ci]:
                    for b in clusters[cj]:
                        total += _hamming_distance(labels[a], labels[b])
                        count += 1
                avg_dist = total / max(count, 1)
                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_pair = (ci, cj)

        ci, cj = best_pair
        merges.append((ci, cj, best_dist))
        clusters[next_id] = clusters[ci] + clusters[cj]
        active.discard(ci)
        active.discard(cj)
        active.add(next_id)
        next_id += 1

    return merges


def hierarchical_haplotype_test(
    Y: Tensor,
    X0: Tensor,
    dosage: Tensor,
    labels: list[str],
    alpha: float = 0.05,
    Py: Tensor | None = None,
) -> HHCTResult:
    """Hierarchical Haplotype Collapsing Test (Meinshausen 2008 FWER).

    Parameters
    ----------
    Y : (n,) phenotype.
    X0 : (n, c) covariates.
    dosage : (n, H) full dosage matrix (including reference).
    labels : list[str], length H.
    alpha : float
        FWER control level.
    Py : (n,) optional precomputed P-projected phenotype.

    Returns
    -------
    HHCTResult
    """
    from scipy.stats import chi2 as chi2_dist

    H = len(labels)
    n = Y.shape[0]
    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    dosage = dosage.to(STAT_DTYPE)

    if Py is None:
        Q, _ = torch.linalg.qr(X0)
        Py = Y - Q @ (Q.T @ Y)

    # Estimate sig2_e
    RSS = (Py ** 2).sum().item()
    sig2_e = RSS / max(n - X0.shape[1], 1)

    # Build all nodes: leaves first, then internal
    nodes: list[HHCTNode] = []

    # Leaf nodes: each haplotype
    for i in range(H):
        d = dosage[:, i]
        dPy = (d * Py).sum().item()
        Q_temp, _ = torch.linalg.qr(X0)
        Pd = d - Q_temp @ (Q_temp.T @ d)
        dPd = (d * Pd).sum().item()
        if dPd > 1e-20:
            chi2_stat = dPy ** 2 / (sig2_e * dPd)
            raw_p = float(chi2_dist.sf(chi2_stat, 1))
        else:
            raw_p = 1.0
        adjusted_p = min(raw_p * H, 1.0)
        nodes.append(HHCTNode(
            node_id=i, children=[], leaf_indices=[i],
            raw_p=raw_p, adjusted_p=adjusted_p, rejected=False,
        ))

    # Build merge tree
    merges = _build_merge_tree(labels)

    Q_cov, _ = torch.linalg.qr(X0)

    for merge_idx, (ci, cj, _) in enumerate(merges):
        # Merged dosage = sum of children's leaf dosages
        leaves = nodes[ci].leaf_indices + nodes[cj].leaf_indices
        n_leaves_below = len(leaves)

        is_root = (merge_idx == len(merges) - 1)
        if is_root and n_leaves_below == H:
            # Root node: dosage columns sum to constant (ploidy).
            # Use multivariate F-test on H-1 non-constant columns instead.
            # Pick one leaf as "reference" (most frequent)
            ref_leaf = max(range(H), key=lambda j: dosage[:, j].sum().item())
            test_leaves = [j for j in range(H) if j != ref_leaf]
            D_test = dosage[:, test_leaves]
            PD_test = D_test - Q_cov @ (Q_cov.T @ D_test)
            S_vec = D_test.T @ Py
            DtPD_block = D_test.T @ PD_test
            h_test = len(test_leaves)
            try:
                stat_root = (S_vec @ torch.linalg.solve(
                    sig2_e * DtPD_block, S_vec)).item()
                raw_p = float(chi2_dist.sf(stat_root, h_test))
            except Exception:
                raw_p = 1.0
        else:
            d_merged = dosage[:, leaves].sum(dim=1)
            Pd = d_merged - Q_cov @ (Q_cov.T @ d_merged)
            dPy = (d_merged * Py).sum().item()
            dPd = (d_merged * Pd).sum().item()
            if dPd > 1e-20:
                chi2_stat = dPy ** 2 / (sig2_e * dPd)
                raw_p = float(chi2_dist.sf(chi2_stat, 1))
            else:
                raw_p = 1.0

        adjusted_p = min(raw_p * H / n_leaves_below, 1.0)

        node_id = len(nodes)
        nodes.append(HHCTNode(
            node_id=node_id,
            children=[ci, cj],
            leaf_indices=leaves,
            raw_p=raw_p,
            adjusted_p=adjusted_p,
            rejected=False,
        ))

    # Top-down rejection: start from root
    root_id = len(nodes) - 1
    if nodes[root_id].adjusted_p <= alpha:
        nodes[root_id].rejected = True
        # BFS from root
        queue = list(nodes[root_id].children)
        while queue:
            nid = queue.pop(0)
            if nodes[nid].adjusted_p <= alpha:
                nodes[nid].rejected = True
                queue.extend(nodes[nid].children)

    rejected_leaves = [n.node_id for n in nodes[:H] if n.rejected]

    return HHCTResult(
        nodes=nodes,
        rejected_leaves=rejected_leaves,
        n_leaves=H,
        alpha=alpha,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. HSKAT — Haplotype Similarity Kernel Association Test
# ═══════════════════════════════════════════════════════════════════════

def haplotype_similarity_kernel(
    labels: list[str],
    bandwidth: float = 1.0,
) -> Tensor:
    """Exponential-decay kernel on Hamming distance.

    K_{ab} = exp(-d_H(h_a, h_b) / bandwidth)

    Parameters
    ----------
    labels : list[str], length H.
    bandwidth : float
        Decay rate. Larger = more diffuse.

    Returns
    -------
    K : (H, H) positive semi-definite kernel matrix.
    """
    H = len(labels)
    K = torch.zeros(H, H, dtype=STAT_DTYPE)
    for i in range(H):
        for j in range(i, H):
            d = _hamming_distance(labels[i], labels[j])
            v = math.exp(-d / max(bandwidth, 1e-10))
            K[i, j] = v
            K[j, i] = v
    return K


def hskat_test(
    D_rot: Tensor,
    Py: Tensor,
    apply_P: callable,
    kernel: Tensor,
) -> tuple[float, float]:
    """SKAT with haplotype similarity kernel.

    Q = z^T K z   where z = D_rot^T @ Py.
    Under H0, Q ~ sum(lambda_k * chi2_1) where lambda_k are eigenvalues
    of K^{1/2} D^T P D K^{1/2}.

    Parameters
    ----------
    D_rot : (n, h) rotated haplotype dosages (reference dropped).
    Py : (n,) P-projected phenotype.
    apply_P : callable for P-operator.
    kernel : (h, h) PSD kernel matrix.

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
    Q = (z @ kernel @ z).item()

    if Q <= 0:
        return 0.0, 1.0

    # Eigenvalues for mixture distribution
    PD = apply_P(D_rot)  # (n, h)
    DtPD = D_rot.T @ PD  # (h, h)

    # K^{1/2}
    evals_K, evecs_K = torch.linalg.eigh(kernel)
    evals_K = torch.clamp(evals_K, min=0)
    K_half = evecs_K @ torch.diag(torch.sqrt(evals_K)) @ evecs_K.T

    M = K_half @ DtPD @ K_half
    lambdas = torch.linalg.eigvalsh(M)
    lambdas = lambdas[lambdas > 1e-10]

    if len(lambdas) == 0:
        return Q, 1.0

    p = mixture_chi2_pvalue(Q, lambdas, method="davies")
    return Q, p


def hskat_adaptive(
    D_rot: Tensor,
    Py: Tensor,
    apply_P: callable,
    labels: list[str],
    bandwidths: list[float] | None = None,
) -> tuple[float, float, float]:
    """Adaptive HSKAT: grid search over bandwidths with Bonferroni.

    Parameters
    ----------
    D_rot : (n, h)
    Py : (n,)
    apply_P : callable
    labels : list[str]
    bandwidths : list[float], optional. Default [0.5, 1.0, 2.0, 1e6].

    Returns
    -------
    Q_best : float
    p_corrected : float
    bandwidth_best : float
    """
    if bandwidths is None:
        bandwidths = [0.5, 1.0, 2.0, 1e6]

    best_p = 1.0
    best_Q = 0.0
    best_bw = bandwidths[0]
    n_bw = len(bandwidths)

    for bw in bandwidths:
        K = haplotype_similarity_kernel(labels, bandwidth=bw)
        # Drop reference (same indices as D_rot columns)
        # labels here should already be the non-reference labels
        Q, p = hskat_test(D_rot, Py, apply_P, K)
        if p < best_p:
            best_p = p
            best_Q = Q
            best_bw = bw

    p_corrected = min(best_p * n_bw, 1.0)
    return best_Q, p_corrected, best_bw


# ═══════════════════════════════════════════════════════════════════════
# 4. HapGxE — Haplotype-by-Environment Interaction Test
# ═══════════════════════════════════════════════════════════════════════

def haplotype_gxe_test(
    Y: Tensor,
    X0: Tensor,
    D: Tensor,
    env: Tensor,
) -> HapGxEResult:
    """Haplotype-by-environment interaction test (OLS).

    Three F-tests:
    - Main:        [X0] vs [X0, D]
    - Interaction: [X0, D] vs [X0, D, D*env]
    - Joint:       [X0] vs [X0, D, D*env]

    Parameters
    ----------
    Y : (n,)
    X0 : (n, c)
    D : (n, H-1)
    env : (n,) environment covariate.

    Returns
    -------
    HapGxEResult
    """
    from scipy.stats import f as f_dist

    n = Y.shape[0]
    c = X0.shape[1]
    h = D.shape[1]
    Y = Y.to(STAT_DTYPE)
    X0 = X0.to(STAT_DTYPE)
    D = D.to(STAT_DTYPE)
    env = env.to(STAT_DTYPE)

    D_int = D * env.unsqueeze(1)  # (n, h)

    def _rss(X: Tensor) -> float:
        Q, _ = torch.linalg.qr(X)
        return ((Y - Q @ (Q.T @ Y)) ** 2).sum().item()

    RSS_null = _rss(X0)
    RSS_main = _rss(torch.cat([X0, D], dim=1))
    RSS_full = _rss(torch.cat([X0, D, D_int], dim=1))

    df_resid_main = n - c - h
    df_resid_full = n - c - 2 * h

    if df_resid_full <= 0 or RSS_full <= 0:
        return HapGxEResult(0, 1, 0, 1, 0, 1)

    # Main test: X0 vs [X0, D]
    if df_resid_main > 0 and RSS_main > 0:
        F_main = ((RSS_null - RSS_main) / h) / (RSS_main / df_resid_main)
        p_main = float(f_dist.sf(F_main, h, df_resid_main))
    else:
        F_main, p_main = 0.0, 1.0

    # Interaction test: [X0, D] vs [X0, D, D*env]
    F_int = ((RSS_main - RSS_full) / h) / (RSS_full / df_resid_full)
    p_int = float(f_dist.sf(F_int, h, df_resid_full))

    # Joint test: X0 vs [X0, D, D*env]
    F_joint = ((RSS_null - RSS_full) / (2 * h)) / (RSS_full / df_resid_full)
    p_joint = float(f_dist.sf(F_joint, 2 * h, df_resid_full))

    return HapGxEResult(
        F_main=F_main, p_main=p_main,
        F_interaction=F_int, p_interaction=p_int,
        F_joint=F_joint, p_joint=p_joint,
    )


def haplotype_gxe_test_lmm(
    null_fit,
    D: Tensor,
    env: Tensor,
) -> HapGxEResult:
    """HapGxE under LMM (WLS in rotated eigenspace).

    Parameters
    ----------
    null_fit : NullFit
    D : (n, H-1) haplotype dosages in original space.
    env : (n,) environment covariate.

    Returns
    -------
    HapGxEResult
    """
    from scipy.stats import f as f_dist
    from ..linalg.eigh import rotate

    nf = null_fit
    n = D.shape[0]
    h = D.shape[1]
    D = D.to(STAT_DTYPE)
    env = env.to(STAT_DTYPE)

    U = nf.eigenvectors
    Y_rot = nf.Y_rot.to(STAT_DTYPE)
    X0_rot = nf.X0_rot.to(STAT_DTYPE)
    c = X0_rot.shape[1]

    lam = nf.sig2_g / max(nf.sig2_e, 1e-20)
    H_inv = 1.0 / (nf.eigenvalues.to(STAT_DTYPE) * lam + 1.0)
    sqrtW = torch.sqrt(torch.clamp(H_inv, min=1e-300))

    D_rot = rotate(D, U)
    env_rot = rotate(env.unsqueeze(1), U).squeeze(1)
    # Interaction must be formed in original space, then rotated —
    # element-wise product does NOT commute with rotation.
    D_int = D * env.unsqueeze(1)
    D_int_rot = rotate(D_int, U)

    # WLS: weight everything by sqrt(H_inv)
    Y_w = sqrtW * Y_rot
    X0_w = sqrtW.unsqueeze(1) * X0_rot
    D_w = sqrtW.unsqueeze(1) * D_rot
    DI_w = sqrtW.unsqueeze(1) * D_int_rot

    def _rss_w(X: Tensor) -> float:
        Q, _ = torch.linalg.qr(X)
        return ((Y_w - Q @ (Q.T @ Y_w)) ** 2).sum().item()

    RSS_null = _rss_w(X0_w)
    RSS_main = _rss_w(torch.cat([X0_w, D_w], dim=1))
    RSS_full = _rss_w(torch.cat([X0_w, D_w, DI_w], dim=1))

    df_resid_main = n - c - h
    df_resid_full = n - c - 2 * h

    if df_resid_full <= 0 or RSS_full <= 0:
        return HapGxEResult(0, 1, 0, 1, 0, 1)

    if df_resid_main > 0 and RSS_main > 0:
        F_main = ((RSS_null - RSS_main) / h) / (RSS_main / df_resid_main)
        p_main = float(f_dist.sf(F_main, h, df_resid_main))
    else:
        F_main, p_main = 0.0, 1.0

    F_int = ((RSS_main - RSS_full) / h) / (RSS_full / df_resid_full)
    p_int = float(f_dist.sf(F_int, h, df_resid_full))

    F_joint = ((RSS_null - RSS_full) / (2 * h)) / (RSS_full / df_resid_full)
    p_joint = float(f_dist.sf(F_joint, 2 * h, df_resid_full))

    return HapGxEResult(
        F_main=F_main, p_main=p_main,
        F_interaction=F_int, p_interaction=p_int,
        F_joint=F_joint, p_joint=p_joint,
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. BayesHap — Bayesian Haplotype Fine-Mapping (SuSiE on haplotypes)
# ═══════════════════════════════════════════════════════════════════════

def _susie_single_effect_update(
    D: Tensor,
    r: Tensor,
    DtD: Tensor,
    sigma2_e: float,
    sigma2_0: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update one SuSiE layer.

    Parameters
    ----------
    D : (n, p) haplotype dosage matrix.
    r : (n,) residual.
    DtD : (p,) precomputed diag(D^T D).
    sigma2_e : float, residual variance.
    sigma2_0 : float, prior slab variance.

    Returns
    -------
    alpha : (p,) selection probabilities.
    mu : (p,) posterior means.
    sigma2_post : (p,) posterior variances.
    log_bf : (p,) log Bayes factors.
    """
    Dtr = D.T @ r  # (p,)

    sigma2_post = 1.0 / (DtD / sigma2_e + 1.0 / sigma2_0)  # (p,)
    mu = sigma2_post * Dtr / sigma2_e  # (p,)

    log_bf = (-0.5 * torch.log(1.0 + sigma2_0 * DtD / sigma2_e)
              + 0.5 * mu ** 2 / sigma2_post)

    alpha = torch.softmax(log_bf, dim=0)  # (p,)

    return alpha, mu, sigma2_post, log_bf


def _extract_credible_sets(
    alpha: Tensor,
    coverage: float = 0.95,
) -> list[list[int]]:
    """Extract per-layer credible sets.

    Parameters
    ----------
    alpha : (L, p) selection probabilities.
    coverage : float

    Returns
    -------
    list of L credible sets (each a list of haplotype indices).
    """
    L, p = alpha.shape
    cs = []
    for l in range(L):
        sorted_idx = torch.argsort(alpha[l], descending=True)
        cum = torch.cumsum(alpha[l, sorted_idx], dim=0)
        n_in_set = int((cum < coverage).sum().item()) + 1
        n_in_set = min(n_in_set, p)
        cs.append(sorted_idx[:n_in_set].tolist())
    return cs


class BayesianHaplotypeFineMapping:
    """SuSiE-based fine-mapping on haplotype dosage columns.

    Parameters
    ----------
    n_signals : int
        Number of single-effect layers L (default 5).
    max_iter : int
        Maximum IBSS iterations (default 100).
    tol : float
        Convergence tolerance on alpha (default 1e-3).
    sigma2_0 : float
        Prior slab variance (default 1.0).
    coverage : float
        Credible set coverage (default 0.95).
    """

    def __init__(
        self,
        n_signals: int = 5,
        max_iter: int = 100,
        tol: float = 1e-3,
        sigma2_0: float = 1.0,
        coverage: float = 0.95,
    ):
        self.n_signals = n_signals
        self.max_iter = max_iter
        self.tol = tol
        self.sigma2_0 = sigma2_0
        self.coverage = coverage

    def fit(
        self,
        Y: Tensor,
        hap_blocks: list,
        X0: Tensor | None = None,
        null_fit=None,
    ) -> BayesHapResult:
        """Run SuSiE IBSS on concatenated haplotype dosages.

        Parameters
        ----------
        Y : (n,) phenotype.
        hap_blocks : list[HaplotypeBlock]
            Haplotype blocks from ``HaplotypeGWAS.construct_haplotypes()``.
        X0 : (n, c) covariates.  If None, intercept-only.
        null_fit : NullFit, optional.
            If provided, runs in LMM eigenspace.

        Returns
        -------
        BayesHapResult
        """
        n = Y.shape[0]
        Y = Y.to(STAT_DTYPE)

        if X0 is None:
            X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
        else:
            X0 = X0.to(STAT_DTYPE)

        # Build concatenated dosage matrix (drop reference per block)
        D_parts = []
        all_labels: list[str] = []
        all_block_ids: list[str] = []

        for hb in hap_blocks:
            H = len(hb.haplotypes)
            if H < 2:
                continue
            ref_idx = hb.frequencies.argmax().item()
            test_idx = [j for j in range(H) if j != ref_idx]
            D_block = hb.dosage[:, test_idx].to(STAT_DTYPE)
            D_parts.append(D_block)
            for j in test_idx:
                all_labels.append(f"{hb.block_id}:{hb.haplotypes[j]}")
                all_block_ids.append(hb.block_id)

        if not D_parts:
            return BayesHapResult(
                pip=torch.tensor([]), alpha=torch.tensor([]),
                mu=torch.tensor([]), credible_sets=[],
                sigma2_e=0.0, elbo_trace=[], converged=True,
                haplotype_labels=[], block_ids=[],
            )

        D_all = torch.cat(D_parts, dim=1)  # (n, p)
        p = D_all.shape[1]

        # Residualize on X0 (or rotate for LMM)
        if null_fit is not None:
            from ..linalg.eigh import rotate
            U = null_fit.eigenvectors
            Y_work = null_fit.Y_rot.to(STAT_DTYPE)
            X0_work = null_fit.X0_rot.to(STAT_DTYPE)
            D_work = rotate(D_all, U)

            lam = null_fit.sig2_g / max(null_fit.sig2_e, 1e-20)
            H_inv = 1.0 / (null_fit.eigenvalues.to(STAT_DTYPE) * lam + 1.0)
            sqrtW = torch.sqrt(torch.clamp(H_inv, min=1e-300))

            Y_work = sqrtW * Y_work
            X0_work = sqrtW.unsqueeze(1) * X0_work
            D_work = sqrtW.unsqueeze(1) * D_work
        else:
            Y_work = Y
            X0_work = X0
            D_work = D_all

        # Project out covariates
        Q, _ = torch.linalg.qr(X0_work)
        Y_resid = Y_work - Q @ (Q.T @ Y_work)
        D_resid = D_work - Q @ (Q.T @ D_work)

        # Precompute
        DtD = (D_resid ** 2).sum(dim=0)  # (p,)

        # Initialize
        L = self.n_signals
        alpha = torch.ones(L, p, dtype=STAT_DTYPE) / p
        mu = torch.zeros(L, p, dtype=STAT_DTYPE)
        sigma2_post = torch.zeros(L, p, dtype=STAT_DTYPE)
        sigma2_e = (Y_resid ** 2).mean().item()
        sigma2_0 = self.sigma2_0

        elbo_trace: list[float] = []
        converged = False

        for it in range(self.max_iter):
            alpha_old = alpha.clone()

            for l in range(L):
                # Compute residual for this layer
                r = Y_resid.clone()
                for l2 in range(L):
                    if l2 != l:
                        # Expected fitted value for layer l2
                        r = r - D_resid @ (alpha[l2] * mu[l2])

                alpha[l], mu[l], sigma2_post[l], _ = (
                    _susie_single_effect_update(
                        D_resid, r, DtD, sigma2_e, sigma2_0,
                    )
                )

            # Update sigma2_e
            fitted = torch.zeros_like(Y_resid)
            for l in range(L):
                fitted = fitted + D_resid @ (alpha[l] * mu[l])
            residual = Y_resid - fitted
            sigma2_e = max((residual ** 2).mean().item(), 1e-20)

            # Approximate ELBO (data fit term)
            elbo = -0.5 * n * math.log(2 * math.pi * sigma2_e) - 0.5 * (
                (residual ** 2).sum().item() / sigma2_e
            )
            elbo_trace.append(elbo)

            # Check convergence
            max_change = (alpha - alpha_old).abs().max().item()
            if max_change < self.tol:
                converged = True
                break

        # PIPs: 1 - prod(1 - alpha_l)
        pip = 1.0 - torch.prod(1.0 - alpha, dim=0)

        # Credible sets
        cs = _extract_credible_sets(alpha, coverage=self.coverage)

        return BayesHapResult(
            pip=pip,
            alpha=alpha,
            mu=mu,
            credible_sets=cs,
            sigma2_e=sigma2_e,
            elbo_trace=elbo_trace,
            converged=converged,
            haplotype_labels=all_labels,
            block_ids=all_block_ids,
        )
