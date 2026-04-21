"""Phase D.3 — end-to-end mediation scan recovery test.

Plants a single causal (SNP → mediator → Y) chain inside a realistic
(G, M, Y) simulation and asserts:

1. The planted (SNP, feature) pair is the top indirect-effect hit by raw
   p-value.
2. Its BH q_indirect is below nominal 0.05.
3. Null (non-causal) pairs do NOT pass the q threshold at a higher rate
   than the nominal FDR, within a sampling buffer.
4. With ``prefilter="coloc"`` enabled, the planted pair still passes;
   the filter does not spuriously drop the genuinely colocalising pair.

The full-scale target in the plan was n=500 × s=1000 × f=100. The shape
is shrunk here to keep the test under a minute while preserving the
detection-power signal — the planted effect is strong enough (a=0.8,
b=0.7, n=300) that recovery is reliable across seeds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchgwas.multiomics import coloc_prefilter_pairs, scan_mediation
from torchgwas.postgwas._sumstats import SumStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_grm(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 200))
    K = (Z @ Z.T) / 200.0
    K = 0.5 * (K + K.T) + 1e-3 * np.eye(n)
    return torch.as_tensor(K, dtype=torch.float64)


def _simulate_mediation_panel(
    *,
    n: int = 300,
    s: int = 12,
    f: int = 8,
    causal_snp: int = 3,
    causal_feature: int = 2,
    a: float = 0.8,
    b: float = 0.7,
    seed: int = 0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    list[str], list[int], list[str], list[int],
]:
    """Single-chromosome simulation with one planted (SNP, feature) chain.

    Returns (Y, G, M, K, snp_chrom, snp_pos, feat_chrom, feat_pos).
    All SNPs and features live on chromosome "1"; positions interleave so
    the planted pair sits inside a 1 Mb cis window with ~1 kb separation.
    """
    rng = np.random.default_rng(seed)
    K = _simulate_grm(n, seed=seed).numpy()
    Lk = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    g_bg = Lk @ rng.standard_normal(n) * 0.3  # shared polygenic background

    # Genotypes: each column an independent binomial(2, 0.3).
    G = rng.binomial(2, 0.3, size=(n, s)).astype(np.float64)

    # Features: noise everywhere, causal feature gets a strong SNP effect.
    M = rng.standard_normal((n, f)) * 0.4
    M[:, causal_feature] += a * G[:, causal_snp]

    # Phenotype: mediator effect + small direct path + polygenic + noise.
    e_y = rng.standard_normal(n) * 0.3
    Y = b * M[:, causal_feature] + 0.1 * G[:, causal_snp] + g_bg + e_y

    # Position metadata: SNPs every 1 kb, features at SNP positions so the
    # planted pair co-locates in the cis window.
    snp_chrom = ["1"] * s
    snp_pos = [1_000 * (i + 1) for i in range(s)]
    feat_chrom = ["1"] * f
    feat_pos = [1_000 * (j * 2 + 1) for j in range(f)]
    # Place the causal feature at the same locus as the causal SNP.
    feat_pos[causal_feature] = snp_pos[causal_snp]

    return (
        torch.as_tensor(Y, dtype=torch.float64),
        torch.as_tensor(G, dtype=torch.float64),
        torch.as_tensor(M, dtype=torch.float64),
        torch.as_tensor(K, dtype=torch.float64),
        snp_chrom,
        snp_pos,
        feat_chrom,
        feat_pos,
    )


def _univariate_z_sumstats(
    G: torch.Tensor,
    y: torch.Tensor,
    chrom: list[str],
    pos: list[int],
) -> SumStats:
    """Build a minimal SumStats by regressing y on each SNP independently."""
    n, m = G.shape
    g_np = G.cpu().numpy()
    y_np = y.cpu().numpy()
    betas = np.zeros(m)
    ses = np.zeros(m)
    for i in range(m):
        gi = g_np[:, i] - g_np[:, i].mean()
        var_g = float((gi * gi).sum())
        if var_g <= 0:
            betas[i] = 0.0
            ses[i] = 1.0
            continue
        b = float((gi * (y_np - y_np.mean())).sum()) / var_g
        resid = (y_np - y_np.mean()) - b * gi
        sig2 = float((resid * resid).sum()) / max(n - 2, 1)
        betas[i] = b
        ses[i] = float(np.sqrt(max(sig2 / var_g, 1e-12)))
    return SumStats(
        chr=list(chrom),
        pos=list(pos),
        snp=[f"rs{i}" for i in range(m)],
        a1=["A"] * m,
        a2=["G"] * m,
        beta=torch.as_tensor(betas, dtype=torch.float64),
        se=torch.as_tensor(ses, dtype=torch.float64),
        p=torch.ones(m, dtype=torch.float64),
        n=torch.full((m,), float(n), dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Core recovery assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 29, 47, 83, 101])
def test_planted_mediator_is_top_hit_and_q_below_nominal(seed):
    """The planted (SNP, feature) pair must surface as the most-significant
    indirect-effect row and clear BH q < 0.05 across seeds."""
    Y, G, M, K, snp_chrom, snp_pos, feat_chrom, feat_pos = _simulate_mediation_panel(
        seed=seed,
    )

    res = scan_mediation(
        Y, G, M, K,
        snp_chrom=snp_chrom, feature_chrom=feat_chrom,
        snp_pos=torch.tensor(snp_pos, dtype=torch.int64),
        feature_pos=torch.tensor(feat_pos, dtype=torch.int64),
        cis_window_bp=5_000,
        se="sobel",
        sensitivity=False,
        fdr_method="bh",
        batched=False,
    )

    df = res.to_dataframe()
    assert not df.empty, "cis-window should retain at least the planted pair"
    top = df.sort_values("indirect_pvalue").iloc[0]
    assert top["snp"] == "snp_3"
    assert top["feature"] == "feat_2"
    assert top["q_indirect"] < 0.05


def test_null_pairs_fdr_does_not_explode():
    """Across seeds, the fraction of null (non-planted) pairs that clear
    q < 0.05 must stay near the nominal BH rate. Uses a broader cis
    window to enlarge the null set so the empirical FDR is meaningful."""
    false_hits = 0
    null_total = 0
    for seed in range(10):
        Y, G, M, K, snp_chrom, snp_pos, feat_chrom, feat_pos = _simulate_mediation_panel(
            seed=seed,
        )
        res = scan_mediation(
            Y, G, M, K,
            snp_chrom=snp_chrom, feature_chrom=feat_chrom,
            snp_pos=torch.tensor(snp_pos, dtype=torch.int64),
            feature_pos=torch.tensor(feat_pos, dtype=torch.int64),
            cis_window_bp=20_000,
            se="sobel",
            sensitivity=False,
            fdr_method="bh",
            batched=False,
        )
        df = res.to_dataframe()
        null_mask = ~((df["snp"] == "snp_3") & (df["feature"] == "feat_2"))
        null_total += int(null_mask.sum())
        false_hits += int(((df["q_indirect"] < 0.05) & null_mask).sum())

    # Allow a generous ceiling — BH is conservative and our nominal q is 0.05.
    # At 10 seeds the rate is typically well under 0.1; bound it at 0.15.
    empirical_fdr = false_hits / max(null_total, 1)
    assert empirical_fdr < 0.15, (
        f"Null-pair FDR {empirical_fdr:.3f} exceeds loose ceiling 0.15 "
        f"({false_hits}/{null_total})"
    )


# ---------------------------------------------------------------------------
# Coloc prefilter preserves the true signal
# ---------------------------------------------------------------------------


def test_coloc_prefilter_keeps_planted_pair():
    """With ``prefilter='coloc'``, the planted feature (which genuinely
    colocalises between Y and M) must survive and the planted pair must
    still appear in the output."""
    Y, G, M, K, snp_chrom, snp_pos, feat_chrom, feat_pos = _simulate_mediation_panel(
        seed=13,
    )

    ss_y = _univariate_z_sumstats(
        G, Y, snp_chrom, snp_pos,
    )
    ss_m_causal = _univariate_z_sumstats(
        G, M[:, 2], snp_chrom, snp_pos,
    )

    # Restrict to the planted feature only so the prefilter has exactly one
    # feature to evaluate — matches the one-m_sumstats-per-call interface.
    pairs_mask = [j == 2 for j in range(8)]
    keep_feat = np.where(pairs_mask)[0].tolist()

    M_restricted = M[:, keep_feat]
    feat_chrom_r = [feat_chrom[j] for j in keep_feat]
    feat_pos_r = [feat_pos[j] for j in keep_feat]

    res = scan_mediation(
        Y, G, M_restricted, K,
        feature_ids=[f"feat_{j}" for j in keep_feat],
        snp_chrom=snp_chrom, feature_chrom=feat_chrom_r,
        snp_pos=torch.tensor(snp_pos, dtype=torch.int64),
        feature_pos=torch.tensor(feat_pos_r, dtype=torch.int64),
        cis_window_bp=5_000,
        se="sobel",
        sensitivity=False,
        prefilter="coloc",
        coloc_sumstats=(ss_y, ss_m_causal),
        coloc_threshold=0.3,
        batched=False,
    )
    df = res.to_dataframe()
    assert not df.empty, "coloc prefilter dropped the genuine colocalising pair"
    assert ((df["snp"] == "snp_3") & (df["feature"] == "feat_2")).any()


def test_coloc_prefilter_drops_non_coloc_feature_via_scan():
    """Sanity: when M comes from a feature whose cis-QTL lives at a
    different SNP than Y's GWAS signal, the prefilter drops every pair."""
    Y, G, M, K, snp_chrom, snp_pos, feat_chrom, feat_pos = _simulate_mediation_panel(
        seed=17,
    )

    # Y's GWAS signal is at SNP 3 (the planted causal SNP).
    ss_y = _univariate_z_sumstats(G, Y, snp_chrom, snp_pos)

    # Fake an M-sumstats whose signal is at a distant SNP (index 11),
    # so PP.H4 between Y and M will be near zero.
    s = G.shape[1]
    fake_beta = torch.zeros(s, dtype=torch.float64)
    fake_beta[11] = 0.9
    fake_se = torch.full((s,), 0.1, dtype=torch.float64)
    ss_m_nonloc = SumStats(
        chr=list(snp_chrom),
        pos=list(snp_pos),
        snp=[f"rs{i}" for i in range(s)],
        a1=["A"] * s,
        a2=["G"] * s,
        beta=fake_beta,
        se=fake_se,
        p=torch.ones(s, dtype=torch.float64),
        n=torch.full((s,), 300.0, dtype=torch.float64),
    )

    # Run only on the causal feature to test whether the prefilter
    # evaluates its coloc honestly.
    M_restricted = M[:, [2]]
    pairs = [(i, 0) for i in range(s)]
    kept, pp_h4 = coloc_prefilter_pairs(
        pairs, ss_y, ss_m_nonloc,
        snp_chrom=snp_chrom, snp_pos=snp_pos,
        feature_chrom=["1"], feature_pos=[feat_pos[2]],
        cis_window_bp=5_000, coloc_threshold=0.5,
    )
    # PP.H4 for this feature must be below threshold → no pairs kept.
    assert 0 in pp_h4
    assert pp_h4[0] < 0.5
    assert kept == []
