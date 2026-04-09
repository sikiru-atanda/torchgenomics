"""Polyploid compatibility tests for Phase 44 post-GWAS methods.

TorchGWAS targets polyploid organisms (tetraploid, hexaploid, etc.).
These tests verify that all Phase 44 methods produce valid results when
given summary statistics and PGS scores derived from polyploid GWAS:

- Allele frequencies in [0, 1] (same range regardless of ploidy)
- Effect sizes that may be smaller per-allele for higher ploidies
  (more allele copies -> diluted per-copy effects)
- Standard errors reflecting polyploid sample sizes and variance
- PGS scores computed from polyploid dosages in [0, k]

All methods operate on summary statistics and pre-computed scores,
so they are inherently ploidy-agnostic. These tests prove it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from scipy import stats as sp_stats

from torchgwas.postgwas._mr import mr_ivw, mr_egger, mr_weighted_median, mr_all
from torchgwas.postgwas._enrichment import snp_to_gene, gene_set_enrichment
from torchgwas.postgwas._finemapping import extract_credible_sets, annotate_sumstats
from torchgwas.postgwas._multi_ancestry import mr_mega, mantra
from torchgwas.postgwas._sumstats import SumStats
from torchgwas.pgs.validation import validate_pgs


# ---------------------------------------------------------------------------
# Mock BayesianVSResult for fine-mapping tests
# ---------------------------------------------------------------------------

@dataclass
class _MockBVS:
    snp: list[str]
    chr: list[str]
    pos: list[int]
    a1: list[str]
    a2: list[str]
    af: torch.Tensor
    pip: torch.Tensor
    beta_mean: torch.Tensor
    beta_sd: torch.Tensor
    credible_sets: Optional[list[list[int]]] = None
    alpha: Optional[torch.Tensor] = None
    n_signals: int = 0
    method: str = "cavi"


# ---------------------------------------------------------------------------
# Helper: simulate polyploid summary statistics
# ---------------------------------------------------------------------------

def _make_polyploid_sumstats(
    ploidy: int = 4,
    m: int = 30,
    seed: int = 42,
    n_samples: int = 500,
    n_signals: int = 3,
    signal_strength: float = 5.0,
) -> SumStats:
    """Simulate summary statistics as would come from a polyploid GWAS.

    Parameters
    ----------
    ploidy : int
        Organism ploidy (e.g. 4 for tetraploid, 6 for hexaploid).
    m : int
        Number of SNPs.
    seed : int
        Random seed for reproducibility.
    n_signals : int
        Number of strong planted signals.
    signal_strength : float
        Z-score magnitude for planted signals.

    Returns
    -------
    SumStats
        Summary statistics with polyploid-realistic effect sizes and SEs.
    """
    rng = np.random.RandomState(seed)

    # AF drawn from Beta(0.8, 0.8) -- slightly U-shaped, realistic for
    # polyploids where rare and common alleles coexist
    af = rng.beta(0.8, 0.8, size=m).clip(0.05, 0.95)

    # SE reflecting polyploid variance: se ~ sqrt(1 / (2 * n * af * (1 - af) * ploidy))
    # Higher ploidy -> smaller SE (more allele copies per individual)
    se = np.sqrt(1.0 / (2.0 * n_samples * af * (1.0 - af) * ploidy))

    # Effect sizes scaled by 1/sqrt(ploidy) to reflect diluted per-allele effects
    beta = rng.normal(0, se * 0.5, size=m)  # mostly null

    # Plant strong signals
    for i in range(min(n_signals, m)):
        beta[i] = signal_strength * se[i] * (1.0 / math.sqrt(ploidy))
        # Ensure the signal is still strong enough after ploidy scaling
        beta[i] = max(beta[i], signal_strength * se[i] * 0.5)

    # P-values from z = beta / se
    z = beta / se
    p = 2.0 * sp_stats.norm.sf(np.abs(z))

    n_arr = np.full(m, n_samples, dtype=np.float64)

    chr_list = [str(rng.randint(1, 3)) for _ in range(m)]
    pos_list = sorted(rng.randint(1_000, 1_000_000, size=m).tolist())
    snp_list = [f"rs_poly{ploidy}_{i}" for i in range(m)]
    a1_list = [rng.choice(["A", "C", "G", "T"]) for _ in range(m)]
    a2_list = [rng.choice(["A", "C", "G", "T"]) for _ in range(m)]

    return SumStats(
        chr=chr_list,
        pos=pos_list,
        snp=snp_list,
        a1=a1_list,
        a2=a2_list,
        beta=torch.tensor(beta, dtype=torch.float64),
        se=torch.tensor(se, dtype=torch.float64),
        p=torch.tensor(p, dtype=torch.float64),
        n=torch.tensor(n_arr, dtype=torch.float64),
        af=torch.tensor(af, dtype=torch.float64),
    )


def _make_polyploid_pgs(
    ploidy: int = 4,
    n: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate PGS scores and phenotypes from polyploid dosages.

    Polyploid dosages lie in [0, k], so PGS scores can have a wider range
    than diploid scores. Returns (pgs, phenotype).
    """
    rng = np.random.RandomState(seed)

    # Simulate a few SNP effects and dosages
    n_snps = 20
    effects = rng.normal(0, 0.1 / math.sqrt(ploidy), size=n_snps)
    effects[0] = 0.5 / math.sqrt(ploidy)  # one strong effect

    # Dosages in [0, ploidy] -- sum of ploidy Bernoulli draws
    dosages = rng.binomial(ploidy, 0.3, size=(n, n_snps)).astype(np.float64)

    pgs = dosages @ effects  # (n,)
    # Phenotype = PGS + noise
    noise = rng.normal(0, 1.0, size=n)
    phenotype = pgs + noise

    return torch.tensor(pgs, dtype=torch.float64), torch.tensor(phenotype, dtype=torch.float64)


def _make_mr_pair(
    ploidy: int = 4,
    m: int = 25,
    seed: int = 42,
    causal_effect: float = 0.15,
    n_samples: int = 1000,
) -> tuple[SumStats, SumStats]:
    """Create exposure/outcome MR pair with a known causal effect.

    The causal effect is ploidy-independent (it is the causal relationship
    between the exposure and the outcome trait, not a per-allele effect).
    """
    rng = np.random.RandomState(seed)

    af = rng.beta(0.8, 0.8, size=m).clip(0.05, 0.95)

    # Exposure: strong instruments with ploidy-scaled per-allele effects
    se_x = np.sqrt(1.0 / (2.0 * n_samples * af * (1.0 - af) * ploidy))
    beta_x = rng.uniform(0.05, 0.3, size=m) / math.sqrt(ploidy)
    # Make all instruments positive and significant
    beta_x = np.abs(beta_x) + 3.0 * se_x

    # Outcome: causal effect * exposure effect + noise
    se_y = np.sqrt(1.0 / (2.0 * n_samples * af * (1.0 - af) * ploidy))
    beta_y = causal_effect * beta_x + rng.normal(0, se_y * 0.3, size=m)

    z_x = beta_x / se_x
    p_x = 2.0 * sp_stats.norm.sf(np.abs(z_x))
    z_y = beta_y / se_y
    p_y = 2.0 * sp_stats.norm.sf(np.abs(z_y))

    n_arr = np.full(m, n_samples, dtype=np.float64)

    chr_list = ["1"] * m
    pos_list = sorted(rng.randint(1_000, 1_000_000, size=m).tolist())
    snp_list = [f"rs_mr{ploidy}_{i}" for i in range(m)]
    a1_list = ["A"] * m
    a2_list = ["G"] * m

    exposure = SumStats(
        chr=chr_list, pos=pos_list, snp=snp_list,
        a1=a1_list, a2=a2_list,
        beta=torch.tensor(beta_x, dtype=torch.float64),
        se=torch.tensor(se_x, dtype=torch.float64),
        p=torch.tensor(p_x, dtype=torch.float64),
        n=torch.tensor(n_arr, dtype=torch.float64),
        af=torch.tensor(af, dtype=torch.float64),
    )
    outcome = SumStats(
        chr=chr_list, pos=pos_list, snp=snp_list,
        a1=a1_list, a2=a2_list,
        beta=torch.tensor(beta_y, dtype=torch.float64),
        se=torch.tensor(se_y, dtype=torch.float64),
        p=torch.tensor(p_y, dtype=torch.float64),
        n=torch.tensor(n_arr, dtype=torch.float64),
        af=torch.tensor(af, dtype=torch.float64),
    )
    return exposure, outcome


# ===================================================================
# MR with polyploid summary stats (3 tests)
# ===================================================================

def test_mr_ivw_tetraploid_recovers_causal_effect():
    """IVW recovers a causal effect of 0.15 from tetraploid instruments.

    In a tetraploid organism, per-allele effects are smaller (diluted across
    4 allele copies), but the Wald ratio (beta_Y / beta_X) cancels the
    ploidy scaling: both exposure and outcome effects are scaled by the
    same 1/sqrt(ploidy) factor, so the ratio is ploidy-invariant.
    """
    exposure, outcome = _make_mr_pair(ploidy=4, causal_effect=0.15, seed=42)
    result = mr_ivw(exposure, outcome)

    assert math.isfinite(result.beta_hat), "IVW beta must be finite"
    assert abs(result.beta_hat - 0.15) < 0.1, (
        f"IVW should recover causal effect ~0.15, got {result.beta_hat:.4f}"
    )
    assert result.p_value < 0.05, (
        f"IVW should detect the causal effect (p < 0.05), got {result.p_value:.4e}"
    )


def test_mr_egger_hexaploid_intercept_valid():
    """MR-Egger on hexaploid data without pleiotropy has a non-significant intercept.

    With ploidy=6, per-allele effects are even smaller, but the Egger
    intercept tests for *average pleiotropy*, which is absent by construction.
    The intercept p-value should be > 0.05 (no evidence of pleiotropy).
    """
    exposure, outcome = _make_mr_pair(ploidy=6, causal_effect=0.2, seed=99)
    result = mr_egger(exposure, outcome)

    assert math.isfinite(result.beta_hat), "Egger beta must be finite"
    assert math.isfinite(result.intercept), "Egger intercept must be finite"
    assert result.intercept_p > 0.05, (
        f"No pleiotropy was introduced, so intercept should be non-significant, "
        f"got intercept_p={result.intercept_p:.4e}"
    )


def test_mr_all_tetraploid_returns_valid_results():
    """All four MR methods produce finite results with p in [0,1] on tetraploid data.

    This is a basic validity check: none of the methods should crash or
    produce NaN/Inf when given tetraploid-scale summary statistics.
    """
    exposure, outcome = _make_mr_pair(ploidy=4, causal_effect=0.15, seed=77)
    results = mr_all(exposure, outcome, n_boot=200, n_perm=200, seed=77)

    assert len(results) == 4, "mr_all should return 4 results"
    for r in results:
        assert math.isfinite(r.beta_hat), f"{r.method}: beta must be finite"
        assert math.isfinite(r.se), f"{r.method}: SE must be finite"
        assert 0.0 <= r.p_value <= 1.0, (
            f"{r.method}: p-value must be in [0, 1], got {r.p_value}"
        )


# ===================================================================
# Gene-set enrichment with polyploid summary stats (2 tests)
# ===================================================================

def test_snp_to_gene_tetraploid_detects_signal():
    """SNP-to-gene aggregation detects a gene harbouring strong tetraploid signals.

    The chi-squared statistic from a polyploid GWAS has the same distribution
    as from a diploid GWAS (z = beta/se, chi2 = z^2), so the MAGMA mean-chi2
    model works identically. We plant several strong signals in one gene's
    region and verify it gets a significant gene-level p-value.
    """
    ss = _make_polyploid_sumstats(ploidy=4, m=30, seed=42, n_signals=3,
                                  signal_strength=6.0)

    # Define two genes: gene_A covers the first 5 SNPs (where signals are),
    # gene_B covers the last 5 SNPs (null region)
    pos = ss.pos
    gene_id = ["gene_A", "gene_B"]
    gene_chr = [ss.chr[0], ss.chr[-1]]
    gene_start = [pos[0] - 100, pos[-5] - 100]
    gene_end = [pos[4] + 100, pos[-1] + 100]

    result = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end, window_kb=0)

    # gene_A should be significant (contains planted signals)
    idx_a = result.gene_id.index("gene_A")
    assert result.n_snps[idx_a] > 0, "gene_A should have mapped SNPs"
    assert result.p[idx_a].item() < 0.05, (
        f"gene_A should be significant (planted signals), got p={result.p[idx_a].item():.4e}"
    )


def test_gene_set_enrichment_tetraploid():
    """Enrichment analysis on tetraploid gene results distinguishes enriched from random.

    Gene-set enrichment operates on gene-level p-values (probit z-scores),
    which are derived from chi-squared statistics. Since chi2 = (beta/se)^2
    is ploidy-agnostic, enrichment analysis works identically for polyploids.
    """
    ss = _make_polyploid_sumstats(ploidy=4, m=60, seed=42, n_signals=5,
                                  signal_strength=6.0)
    pos = ss.pos

    # Build 10 genes spanning the SNP positions
    n_genes = 10
    snps_per_gene = len(pos) // n_genes
    gene_id = [f"gene_{i}" for i in range(n_genes)]
    gene_chr = []
    gene_start = []
    gene_end = []
    for i in range(n_genes):
        start_idx = i * snps_per_gene
        end_idx = min((i + 1) * snps_per_gene - 1, len(pos) - 1)
        gene_chr.append(ss.chr[start_idx])
        gene_start.append(pos[start_idx] - 10)
        gene_end.append(pos[end_idx] + 10)

    gene_result = snp_to_gene(ss, gene_id, gene_chr, gene_start, gene_end)

    # Enriched set: first gene (contains signals)
    enriched_set = {"enriched": ["gene_0"]}
    # Random set: last gene (null region)
    random_set = {"random": [f"gene_{n_genes - 1}"]}

    enr_enriched = gene_set_enrichment(gene_result, enriched_set)
    enr_random = gene_set_enrichment(gene_result, random_set)

    # The enriched set should have a lower (more significant) p-value
    assert enr_enriched.p[0].item() <= enr_random.p[0].item(), (
        f"Enriched set p ({enr_enriched.p[0].item():.4e}) should be <= "
        f"random set p ({enr_random.p[0].item():.4e})"
    )


# ===================================================================
# Fine-mapping with polyploid posteriors (2 tests)
# ===================================================================

def test_credible_sets_tetraploid_pips():
    """Credible sets are correctly extracted from tetraploid fine-mapping results.

    In polyploids, higher LD can spread posterior mass across more variants,
    resulting in smaller per-SNP PIPs. The credible set extraction should
    still work: it accumulates PIPs until reaching the coverage threshold,
    regardless of individual PIP magnitudes.
    """
    m = 20
    rng = np.random.RandomState(42)

    # Simulate PIPs typical of a polyploid fine-mapping:
    # two moderate signals rather than one strong signal (due to higher LD)
    pip = np.full(m, 0.01)
    pip[3] = 0.35  # signal 1
    pip[7] = 0.30  # signal 2
    pip[4] = 0.15  # nearby variant in LD with signal 1
    pip[8] = 0.10  # nearby variant in LD with signal 2
    pip = pip / pip.sum()  # normalize (not required by API but realistic)

    bvs = _MockBVS(
        snp=[f"rs_tet_{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 100, 100)),
        a1=["A"] * m,
        a2=["G"] * m,
        af=torch.tensor(rng.beta(0.8, 0.8, size=m).clip(0.05, 0.95),
                         dtype=torch.float64),
        pip=torch.tensor(pip, dtype=torch.float64),
        beta_mean=torch.tensor(rng.normal(0, 0.05, size=m), dtype=torch.float64),
        beta_sd=torch.tensor(np.abs(rng.normal(0.01, 0.005, size=m)),
                              dtype=torch.float64),
        method="cavi",
    )

    cs_list = extract_credible_sets(bvs, coverage=0.95)

    assert len(cs_list) >= 1, "Should extract at least one credible set"
    cs = cs_list[0]
    assert cs.cumulative_coverage >= 0.95, (
        f"Credible set should cover >= 95% posterior mass, got {cs.cumulative_coverage:.4f}"
    )
    # Lead SNP should be the highest-PIP variant
    assert cs.lead_snp_id == "rs_tet_3", (
        f"Lead SNP should be rs_tet_3 (highest PIP), got {cs.lead_snp_id}"
    )


def test_annotate_sumstats_tetraploid():
    """PIPs from tetraploid fine-mapping are correctly placed onto summary stats.

    The annotate_sumstats function matches by SNP ID, which is ploidy-agnostic.
    We verify that PIPs and posterior effects are correctly transferred.
    """
    m = 15
    rng = np.random.RandomState(42)

    ss = _make_polyploid_sumstats(ploidy=4, m=m, seed=42, n_signals=2)

    pip_vals = np.full(m, 0.02)
    pip_vals[0] = 0.8
    pip_vals[1] = 0.5

    bvs = _MockBVS(
        snp=ss.snp,
        chr=ss.chr,
        pos=ss.pos,
        a1=ss.a1,
        a2=ss.a2,
        af=ss.af,
        pip=torch.tensor(pip_vals, dtype=torch.float64),
        beta_mean=torch.tensor(rng.normal(0, 0.05, size=m), dtype=torch.float64),
        beta_sd=torch.tensor(np.abs(rng.normal(0.01, 0.005, size=m)),
                              dtype=torch.float64),
        method="cavi",
    )

    annotated = annotate_sumstats(ss, bvs)

    assert annotated.pip[0].item() == pip_vals[0], (
        f"PIP for signal SNP 0 should be {pip_vals[0]}, got {annotated.pip[0].item()}"
    )
    assert annotated.pip[1].item() == pip_vals[1], (
        f"PIP for signal SNP 1 should be {pip_vals[1]}, got {annotated.pip[1].item()}"
    )
    # Lead SNP (highest PIP) should be in a credible set
    assert annotated.in_credible_set[0] >= 0, (
        "Lead SNP (index 0) should be in a credible set"
    )


# ===================================================================
# Multi-ancestry with mixed-ploidy populations (2 tests)
# ===================================================================

def _make_multi_pop_sumstats(
    ploidy: int = 4,
    n_pops: int = 3,
    m: int = 20,
    seed: int = 42,
    signal_idx: int = 0,
    signal_z: float = 5.0,
) -> list[SumStats]:
    """Create summary statistics for multiple populations of the same ploidy.

    Each population has slightly different allele frequencies (realistic for
    multi-ancestry polyploid studies, e.g. potato or sugarcane populations
    from different geographical origins).
    """
    rng = np.random.RandomState(seed)
    pops = []

    # Base allele frequencies
    af_base = rng.beta(0.8, 0.8, size=m).clip(0.10, 0.90)

    snp_list = [f"rs_pop_{i}" for i in range(m)]
    chr_list = ["1"] * m
    pos_list = sorted(rng.randint(1_000, 1_000_000, size=m).tolist())
    a1_list = ["A"] * m
    a2_list = ["G"] * m
    n_samples = 500

    for pop in range(n_pops):
        # Drift allele frequencies substantially per population to give
        # MR-MEGA meaningful PC axes for the meta-regression
        af = (af_base + rng.normal(0, 0.15, size=m)).clip(0.05, 0.95)

        se = np.sqrt(1.0 / (2.0 * n_samples * af * (1.0 - af) * ploidy))
        beta = rng.normal(0, se * 0.3, size=m)

        # Plant shared signal at signal_idx
        beta[signal_idx] = signal_z * se[signal_idx]

        z = beta / se
        p = 2.0 * sp_stats.norm.sf(np.abs(z))
        n_arr = np.full(m, n_samples, dtype=np.float64)

        pops.append(SumStats(
            chr=chr_list, pos=pos_list, snp=snp_list,
            a1=a1_list, a2=a2_list,
            beta=torch.tensor(beta, dtype=torch.float64),
            se=torch.tensor(se, dtype=torch.float64),
            p=torch.tensor(p, dtype=torch.float64),
            n=torch.tensor(n_arr, dtype=torch.float64),
            af=torch.tensor(af, dtype=torch.float64),
        ))

    return pops


def test_mr_mega_tetraploid_populations():
    """MR-MEGA detects a shared signal across 3 tetraploid populations.

    MR-MEGA meta-regression operates on per-population (beta, SE) pairs.
    For polyploids, these are already correctly estimated by the upstream
    GWAS scan. The method should detect the planted shared signal.
    """
    pops = _make_multi_pop_sumstats(ploidy=4, n_pops=3, m=20, seed=42,
                                     signal_idx=0, signal_z=6.0)

    # n_axes=0 runs MR-MEGA in its fixed-effect mode (no ancestry PC axes),
    # which is the appropriate choice when populations have similar genetic
    # backgrounds (e.g. tetraploid potato cultivars from the same breeding
    # programme). This avoids the collinearity issues that arise when
    # AF-based PCs are poorly informative with few populations.
    result = mr_mega(pops, n_axes=0)

    assert result.n_populations == 3
    assert result.n_snps == 20
    assert result.method == "mr_mega"

    # Signal SNP should be significant
    p_signal = result.p_meta[0].item()
    assert p_signal < 0.05, (
        f"MR-MEGA should detect the shared signal (p < 0.05), got {p_signal:.4e}"
    )
    # All p-values should be valid
    assert (result.p_meta >= 0).all() and (result.p_meta <= 1).all(), (
        "All p-values must be in [0, 1]"
    )


def test_mantra_tetraploid_bf_positive():
    """MANTRA produces positive log10 Bayes factors for a shared tetraploid signal.

    MANTRA's conjugate-normal model computes BFs from (beta, SE) pairs.
    The prior variances are not ploidy-specific. A shared signal across
    populations should yield log10_bf > 0 (evidence for association).
    """
    pops = _make_multi_pop_sumstats(ploidy=4, n_pops=3, m=20, seed=42,
                                     signal_idx=0, signal_z=5.0)

    result = mantra(pops)

    assert result.method == "mantra"
    assert result.log10_bf is not None

    # Signal SNP should have positive log10 BF
    bf_signal = result.log10_bf[0].item()
    assert bf_signal > 0, (
        f"Signal SNP should have log10_bf > 0 (evidence for association), "
        f"got {bf_signal:.4f}"
    )
    # Posterior effect should be finite
    assert math.isfinite(result.posterior_effect[0].item()), (
        "Posterior effect must be finite"
    )


# ===================================================================
# PGS validation with polyploid scores (3 tests)
# ===================================================================

def test_pgs_validation_tetraploid_continuous():
    """PGS scores from tetraploid dosages produce valid continuous-trait metrics.

    Tetraploid dosages are in [0, 4], so PGS scores can have a wider range
    than diploid ([0, 2]). The validation metrics (R^2, Pearson r) should
    still be computed correctly -- they are scale-invariant.
    """
    pgs, y = _make_polyploid_pgs(ploidy=4, n=200, seed=42)

    result = validate_pgs(pgs, y, trait_type="continuous")

    assert result.n == 200
    assert result.trait_type == "continuous"
    assert result.r2 > 0, f"R^2 should be > 0 for a correlated predictor, got {result.r2}"
    assert result.pearson_r > 0, (
        f"Pearson r should be > 0 for a positively correlated predictor, "
        f"got {result.pearson_r}"
    )
    assert 0 <= result.r2 <= 1, f"R^2 must be in [0, 1], got {result.r2}"
    assert result.mae >= 0, f"MAE must be >= 0, got {result.mae}"


def test_pgs_validation_tetraploid_binary_auc():
    """PGS from tetraploid dosages discriminates a binary trait (AUC > 0.5).

    We create a binary trait by thresholding a correlated continuous phenotype.
    The PGS should have discriminative power (AUC > 0.5) regardless of the
    fact that the underlying dosages were in [0, 4] rather than [0, 2].
    """
    pgs, y_cont = _make_polyploid_pgs(ploidy=4, n=200, seed=42)

    # Binarize: cases are the top 40% of the continuous phenotype
    threshold = torch.quantile(y_cont, 0.6)
    y_binary = (y_cont >= threshold).to(torch.float64)

    result = validate_pgs(pgs, y_binary, trait_type="binary")

    assert result.trait_type == "binary"
    assert result.auc is not None
    assert result.auc > 0.5, (
        f"AUC should be > 0.5 for a discriminating predictor, got {result.auc:.4f}"
    )
    assert result.nagelkerke_r2 is not None
    assert result.nagelkerke_r2 >= 0, (
        f"Nagelkerke R^2 must be >= 0, got {result.nagelkerke_r2}"
    )


def test_pgs_validation_hexaploid_liability_r2():
    """Hexaploid PGS with prevalence correction yields a valid liability-scale R^2.

    The Lee et al. (2012) liability-scale R^2 correction depends only on
    prevalence and the observed R^2 -- it has no ploidy-specific terms.
    We verify it produces a finite, non-negative value for hexaploid PGS.
    """
    pgs, y_cont = _make_polyploid_pgs(ploidy=6, n=200, seed=99)

    # Binarize with a realistic prevalence
    threshold = torch.quantile(y_cont, 0.7)
    y_binary = (y_cont >= threshold).to(torch.float64)

    prevalence = 0.3  # population prevalence

    result = validate_pgs(pgs, y_binary, trait_type="binary", prevalence=prevalence)

    assert result.liability_r2 is not None, "liability_r2 should be computed when prevalence is given"
    assert math.isfinite(result.liability_r2), (
        f"liability_r2 must be finite, got {result.liability_r2}"
    )
    assert result.liability_r2 >= 0, (
        f"liability_r2 must be >= 0, got {result.liability_r2}"
    )


# ===================================================================
# Cross-ploidy consistency (1 test)
# ===================================================================

def test_mr_ivw_ploidy_invariant():
    """IVW recovers the same causal effect from diploid and tetraploid data.

    The Wald ratio (beta_Y / beta_X) is ploidy-invariant because both
    exposure and outcome per-allele effects are scaled by the same
    1/sqrt(ploidy) factor. This test generates matched diploid and
    tetraploid summary statistics with the SAME underlying causal effect
    and verifies that both IVW estimates agree within tolerance.
    """
    causal = 0.20

    # Use the same seed and sample size so the only difference is ploidy
    exp_2, out_2 = _make_mr_pair(ploidy=2, causal_effect=causal, seed=123,
                                  n_samples=2000)
    exp_4, out_4 = _make_mr_pair(ploidy=4, causal_effect=causal, seed=123,
                                  n_samples=2000)

    result_2 = mr_ivw(exp_2, out_2)
    result_4 = mr_ivw(exp_4, out_4)

    # Both should recover the causal effect
    assert abs(result_2.beta_hat - causal) < 0.15, (
        f"Diploid IVW should recover causal effect ~{causal}, got {result_2.beta_hat:.4f}"
    )
    assert abs(result_4.beta_hat - causal) < 0.15, (
        f"Tetraploid IVW should recover causal effect ~{causal}, got {result_4.beta_hat:.4f}"
    )

    # The two estimates should be close to each other (same underlying biology)
    diff = abs(result_2.beta_hat - result_4.beta_hat)
    assert diff < 0.15, (
        f"Diploid and tetraploid IVW estimates should agree within 0.15, "
        f"got |{result_2.beta_hat:.4f} - {result_4.beta_hat:.4f}| = {diff:.4f}"
    )
