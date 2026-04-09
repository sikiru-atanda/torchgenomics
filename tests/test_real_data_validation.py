"""Real-data validation: end-to-end smoke tests for all models.

Uses MDP maize (diploid, 281 samples, 3093 SNPs, 3 traits) and GWASpoly
potato (tetraploid, ~254 samples, 2000 SNPs, vine.maturity) to verify that
every model:
1. Runs without errors on real genotype data
2. Produces valid p-values in (0, 1]
3. Returns results in the expected format

This is NOT a reference-equivalence test (those are in test_golden_*.py).
It is a comprehensive integration test ensuring nothing is broken.

Requires benchmark data files under benchmark/data/ and
benchmark/gwaspoly_data/ or benchmark/gwaspoly_results/.
Tests are skipped if data is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from torchgwas.config import STAT_DTYPE, NumericalConfig
from torchgwas.models.base import VariantMeta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_DIR / "benchmark"
MDP_DATA = BENCHMARK_DIR / "data"
GWASPOLY_DATA = BENCHMARK_DIR / "gwaspoly_data"
GWASPOLY_RESULTS = BENCHMARK_DIR / "gwaspoly_results"

# ---------------------------------------------------------------------------
# Data availability guards
# ---------------------------------------------------------------------------


def _have_mdp():
    return (MDP_DATA / "mdp_numeric.txt").is_file()


def _have_potato():
    return (
        (GWASPOLY_RESULTS / "potato_geno_aligned.csv").is_file()
        or (GWASPOLY_DATA / "potato_geno_subset.csv").is_file()
    )


skip_no_mdp = pytest.mark.skipif(not _have_mdp(), reason="MDP data not found")
skip_no_potato = pytest.mark.skipif(not _have_potato(), reason="Potato data not found")

# ---------------------------------------------------------------------------
# Fixtures — load once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mdp():
    """Load MDP maize data: G (n, m), Y (n, 3), snp_info, sample order."""
    geno = pd.read_csv(MDP_DATA / "mdp_numeric.txt", sep="\t", index_col=0)
    traits = pd.read_csv(MDP_DATA / "mdp_traits.txt", sep="\t", index_col=0)
    snp_info = pd.read_csv(
        MDP_DATA / "mdp_SNP_information.txt", sep="\t"
    )

    # Align samples
    common = sorted(set(geno.index) & set(traits.index))
    geno = geno.loc[common]
    traits = traits.loc[common]

    G = torch.tensor(geno.values, dtype=STAT_DTYPE)
    # Mean-impute missing
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()

    n, m = G.shape
    Y_full = torch.tensor(traits.values, dtype=STAT_DTYPE)

    # Per-trait complete-case phenotype (for single-trait, use EarHT)
    earht_mask = ~torch.isnan(Y_full[:, 0])
    Y_earht = Y_full[earht_mask, 0]
    G_earht = G[earht_mask]

    X0 = torch.ones(G_earht.shape[0], 1, dtype=STAT_DTYPE)

    chrs = snp_info["Chromosome"].astype(str).tolist()
    pos = snp_info["Position"].astype(int).tolist()
    snps = snp_info["SNP"].tolist()
    vmeta = VariantMeta(
        snp=snps, chr=chrs, pos=pos,
        a1=["A"] * m, a2=["G"] * m,
    )

    # GRM for LMM models
    from torchgwas.linalg.kinship import grm_vanraden
    K, _ = grm_vanraden(G_earht, ploidy=2)

    # Subset for fast scanning (first 200 SNPs)
    G_scan = G_earht[:, :200]
    vmeta_scan = VariantMeta(
        snp=snps[:200], chr=chrs[:200], pos=pos[:200],
        a1=["A"] * 200, a2=["G"] * 200,
    )

    return dict(
        G=G_earht, G_scan=G_scan, Y=Y_earht, Y_full=Y_full,
        X0=X0, K=K, vmeta=vmeta, vmeta_scan=vmeta_scan,
        n=G_earht.shape[0], m=m, earht_mask=earht_mask,
        G_raw=G, traits=traits,
    )


@pytest.fixture(scope="session")
def potato():
    """Load potato tetraploid data (subset): G (n, m), Y, K, vmeta."""
    if (GWASPOLY_RESULTS / "potato_geno_aligned.csv").is_file():
        geno_df = pd.read_csv(
            GWASPOLY_RESULTS / "potato_geno_aligned.csv", index_col=0
        )
        pheno_df = pd.read_csv(
            GWASPOLY_RESULTS / "potato_pheno_with_genoid.csv"
        )
        map_df = pd.read_csv(GWASPOLY_RESULTS / "potato_map.csv")
    else:
        geno_df = pd.read_csv(
            GWASPOLY_DATA / "potato_geno_subset.csv", index_col=0
        )
        pheno_df = pd.read_csv(GWASPOLY_DATA / "potato_pheno_subset.csv")
        map_df = None

    geno_df.index = geno_df.index.astype(str)

    # For simplicity, take one observation per genotype (mean phenotype)
    pheno_df = pheno_df.dropna(subset=["vine.maturity"]).copy()
    pheno_df["geno_id"] = pheno_df["geno_id"].astype(str)
    pheno_agg = pheno_df.groupby("geno_id")["vine.maturity"].mean()
    common = sorted(set(geno_df.index) & set(pheno_agg.index))
    geno_sub = geno_df.loc[common]
    pheno_sub = pheno_agg.loc[common]

    G = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()

    Y = torch.tensor(pheno_sub.values, dtype=STAT_DTYPE)
    n, m = G.shape
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    from torchgwas.linalg.kinship_polyploid import grm_polyploid_gene_action
    K, _ = grm_polyploid_gene_action(G, model="additive", ploidy=4)

    snps = list(geno_sub.columns)[:200]
    if map_df is not None:
        chrs = map_df["chrom"].astype(str).tolist()[:200]
        pos_list = map_df["pos"].astype(int).tolist()[:200]
    else:
        chrs = ["1"] * 200
        pos_list = list(range(200))

    G_scan = G[:, :200]
    vmeta_scan = VariantMeta(
        snp=snps, chr=chrs, pos=pos_list,
        a1=["A"] * 200, a2=["G"] * 200,
    )

    return dict(
        G=G, G_scan=G_scan, Y=Y, X0=X0, K=K,
        vmeta_scan=vmeta_scan, n=n, m=m, ploidy=4,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_pvalues(p, label=""):
    """Assert p-values are in (0, 1] and mostly non-NaN."""
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    p = np.asarray(p, dtype=np.float64)
    valid = p[~np.isnan(p)]
    assert len(valid) > 0, f"{label}: all p-values are NaN"
    assert np.all(valid > 0), f"{label}: p-values <= 0 found"
    assert np.all(valid <= 1.0 + 1e-10), f"{label}: p-values > 1 found"


# ===================================================================
# DIPLOID MODELS on MDP maize
# ===================================================================


@skip_no_mdp
class TestDiploidCore:
    """Core V1 models on MDP maize data."""

    def test_glm(self, mdp):
        from torchgwas.models.glm import GLM
        model = GLM()
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(mdp["G_scan"], nf, mdp["vmeta_scan"])
        _assert_valid_pvalues(result.p, "GLM")

    def test_single_trait_lmm_wald(self, mdp):
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"], test="wald"
        )
        _assert_valid_pvalues(result.p, "LMM-Wald")
        assert result.beta is not None

    def test_single_trait_lmm_score(self, mdp):
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"], test="score"
        )
        _assert_valid_pvalues(result.p, "LMM-Score")

    def test_multi_trait_lmm(self, mdp):
        from torchgwas.models.multi_trait_lmm import MultiTraitLMM

        # Use 2 traits with complete cases
        Y_full = mdp["Y_full"]
        mask = mdp["earht_mask"]
        Y2 = Y_full[mask, :2]
        # Drop rows with NaN in either trait
        valid = ~torch.isnan(Y2).any(dim=1)
        Y2_clean = Y2[valid]
        G_clean = mdp["G"][valid, :100]
        X0_clean = torch.ones(Y2_clean.shape[0], 1, dtype=STAT_DTYPE)
        from torchgwas.linalg.kinship import grm_vanraden
        K_clean, _ = grm_vanraden(mdp["G"][valid], ploidy=2)

        snps = mdp["vmeta_scan"].snp[:100]
        chrs = mdp["vmeta_scan"].chr[:100]
        pos = mdp["vmeta_scan"].pos[:100]
        vmeta_100 = VariantMeta(
            snp=snps, chr=chrs, pos=pos,
            a1=["A"] * 100, a2=["G"] * 100,
        )
        model = MultiTraitLMM()
        nf = model.fit_null(Y2_clean, X0_clean, K=K_clean)
        result = model.score_chunk(G_clean, nf, vmeta_100, test="wald")
        _assert_valid_pvalues(result.p, "mvLMM")

    def test_farmcpu(self, mdp):
        from torchgwas.models.farmcpu import FarmCPU
        model = FarmCPU(max_iter=3, p_threshold=0.05)
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"]
        )
        _assert_valid_pvalues(result.p, "FarmCPU")

    def test_blink(self, mdp):
        from torchgwas.models.blink import BLINK
        model = BLINK(max_iter=3, p_threshold=0.05)
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"]
        )
        _assert_valid_pvalues(result.p, "BLINK")


@skip_no_mdp
class TestDiploidNovel:
    """Post-V1 novel models on MDP maize data."""

    def test_multi_kernel_lmm(self, mdp):
        from torchgwas.models.multi_kernel_lmm import MultiKernelLMM, build_multi_kernels
        kernels, kernel_names = build_multi_kernels(mdp["G"][:, :200], ploidy=2)
        model = MultiKernelLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], kernels=kernels)
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"], test="wald"
        )
        _assert_valid_pvalues(result.p, "MK-LMM")

    def test_het_lmm(self, mdp):
        from torchgwas.models.lmm_gxe import HetLMM
        # Create a fake environment variable
        torch.manual_seed(42)
        env = torch.randn(mdp["n"], dtype=STAT_DTYPE)
        model = HetLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"], env=env)
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"]
        )
        _assert_valid_pvalues(result.p_joint, "HetLMM")

    def test_set_based_skat(self, mdp):
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        from torchgwas.models.set_based import SetBasedScanner
        from torchgwas.io.regions import Region

        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])

        # Create Region objects for groups of SNPs
        chrs = mdp["vmeta_scan"].chr[:100]
        pos = mdp["vmeta_scan"].pos[:100]
        regions = [
            Region(region_id=f"R{i}", chr=chrs[i * 20], start=pos[i * 20], end=pos[min((i + 1) * 20 - 1, 99)])
            for i in range(5)
        ]

        scanner = SetBasedScanner(nf, ploidy=2)
        result = scanner.scan_regions(
            mdp["G"][:, :100], regions,
            chrs[:100], pos[:100],
            test="skat",
        )
        _assert_valid_pvalues(result.p, "SKAT")

    def test_bayesian_vs_susie(self, mdp):
        from torchgwas.models.bayesian_vs import BayesianVS
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        lmm = SingleTraitLMM()
        nf = lmm.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])

        bvs = BayesianVS(nf)
        result = bvs.fit(
            mdp["G_scan"][:, :100],
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
            max_iter=50, n_signals=3,
        )
        assert result.pip is not None
        assert result.pip.shape[0] == 100
        assert torch.all(result.pip >= 0) and torch.all(result.pip <= 1)

    def test_ocf_lmm(self, mdp):
        from torchgwas.models.ocf_lmm import OCFLMM
        model = OCFLMM(n_folds=3, seed=42)
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "OCF-LMM")

    def test_knockoff_lmm(self, mdp):
        from torchgwas.models.knockoff_lmm import KnockoffLMM
        vmeta_100 = VariantMeta(
            snp=mdp["vmeta_scan"].snp[:100],
            chr=mdp["vmeta_scan"].chr[:100],
            pos=mdp["vmeta_scan"].pos[:100],
            a1=["A"] * 100, a2=["G"] * 100,
        )
        model = KnockoffLMM(fdr_level=0.5, seed=42)
        result = model.run(
            mdp["Y"], mdp["X0"], mdp["K"],
            mdp["G_scan"][:, :100], vmeta_100,
            vmeta_100.pos, vmeta_100.chr,
        )
        assert result is not None

    def test_gu_lmm(self, mdp):
        from torchgwas.models.gu_lmm import GULM
        # Create fake dosage variance (low uncertainty)
        torch.manual_seed(42)
        dosage_var = torch.rand(mdp["n"], 100, dtype=STAT_DTYPE) * 0.05
        model = GULM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
            dosage_var=dosage_var,
        )
        _assert_valid_pvalues(result.p, "GU-LMM")

    def test_lro_lmm(self, mdp):
        from torchgwas.models.lro_lmm import LROLMM
        vmeta_100 = VariantMeta(
            snp=mdp["vmeta_scan"].snp[:100],
            chr=mdp["vmeta_scan"].chr[:100],
            pos=mdp["vmeta_scan"].pos[:100],
            a1=["A"] * 100, a2=["G"] * 100,
        )
        model = LROLMM()
        result = model.run(
            mdp["Y"], mdp["X0"],
            mdp["G_scan"][:, :100], vmeta_100,
            vmeta_100.pos, vmeta_100.chr,
        )
        _assert_valid_pvalues(result.p, "LRO-LMM")


@skip_no_mdp
class TestDiploidCategorical:
    """Categorical-trait models on MDP (binarized phenotype)."""

    @pytest.fixture()
    def binary_data(self, mdp):
        """Create binary phenotype from EarHT (above/below median)."""
        med = mdp["Y"].median()
        Y_bin = (mdp["Y"] > med).to(STAT_DTYPE)
        return Y_bin

    @pytest.fixture()
    def ordinal_data(self, mdp):
        """Create ordinal phenotype from EarHT (3 categories)."""
        q33 = torch.quantile(mdp["Y"], 0.33)
        q66 = torch.quantile(mdp["Y"], 0.66)
        Y_ord = torch.zeros(mdp["n"], dtype=STAT_DTYPE)
        Y_ord[mdp["Y"] > q33] = 1.0
        Y_ord[mdp["Y"] > q66] = 2.0
        return Y_ord

    def test_binary_glm(self, mdp, binary_data):
        from torchgwas.models.binary_glm import BinaryGLM
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(binary_data, mdp["X0"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "BinaryGLM")

    def test_binary_glmm(self, mdp, binary_data):
        from torchgwas.models.binary_glmm import BinaryGLMM
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(binary_data, mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "BinaryGLMM")

    def test_ordinal_glm(self, mdp, ordinal_data):
        from torchgwas.models.ordinal_glm import OrdinalGLM
        model = OrdinalGLM(n_categories=3)
        nf = model.fit_null(ordinal_data, mdp["X0"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "OrdinalGLM")

    def test_ordinal_glmm(self, mdp, ordinal_data):
        from torchgwas.models.ordinal_glmm import OrdinalGLMM
        model = OrdinalGLMM(n_categories=3)
        nf = model.fit_null(ordinal_data, mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "OrdinalGLMM")


@skip_no_mdp
class TestDiploidSurvival:
    """Survival GWAS on MDP (simulated time-to-event from EarHT)."""

    def test_survival_glmm(self, mdp):
        from torchgwas.models.survival_glmm import SurvivalGLMM
        torch.manual_seed(42)
        # Simulate survival data: higher EarHT → longer survival
        time = torch.exp(0.02 * mdp["Y"] + 0.5 * torch.randn(mdp["n"]))
        time = torch.clamp(time, min=0.1)
        # ~30% censored
        censor_time = torch.quantile(time, 0.7)
        event = (time <= censor_time).to(STAT_DTYPE)
        time = torch.minimum(time, censor_time * torch.ones_like(time))
        # Y_surv is (n, 2): [time, event]
        Y_surv = torch.stack([time, event], dim=1)

        model = SurvivalGLMM(use_spa=False, pql_max_iter=10)
        nf = model.fit_null(Y_surv, mdp["X0"], mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "SurvivalGLMM")


@skip_no_mdp
class TestDiploidMultiEnv:
    """Multi-environment models on MDP (traits as pseudo-environments)."""

    def test_multi_env_lmm(self, mdp):
        from torchgwas.models.multi_env_lmm import MultiEnvLMM

        # Use two traits as pseudo-environments
        Y_full = mdp["Y_full"]
        mask = mdp["earht_mask"]
        Y2 = Y_full[mask, :2]
        valid = ~torch.isnan(Y2).any(dim=1)
        Y_wide = Y2[valid]
        G_sub = mdp["G"][valid, :100]
        X0_sub = torch.ones(Y_wide.shape[0], 1, dtype=STAT_DTYPE)
        from torchgwas.linalg.kinship import grm_vanraden
        K_sub, _ = grm_vanraden(mdp["G"][valid], ploidy=2)

        model = MultiEnvLMM()
        nf = model.fit_null(Y_wide, X0_sub, K=K_sub, n_envs=2)
        result = model.score_chunk(
            G_sub, nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "MultiEnvLMM")

    def test_mtmet_lmm(self, mdp):
        from torchgwas.models.multi_trait_multi_env_lmm import MultiTraitMultiEnvLMM

        # 2 traits × 2 envs → simulate Y_wide (n, 4)
        Y_full = mdp["Y_full"]
        mask = mdp["earht_mask"]
        Y2 = Y_full[mask, :2]
        valid = ~torch.isnan(Y2).any(dim=1)
        Y2_clean = Y2[valid]
        # Duplicate with noise as "second environment"
        torch.manual_seed(42)
        Y2_env2 = Y2_clean + 0.5 * torch.randn_like(Y2_clean)
        Y_wide = torch.cat([Y2_clean, Y2_env2], dim=1)  # (n, 4): [t1e1, t2e1, t1e2, t2e2]

        G_sub = mdp["G"][valid, :100]
        X0_sub = torch.ones(Y_wide.shape[0], 1, dtype=STAT_DTYPE)
        from torchgwas.linalg.kinship import grm_vanraden
        K_sub, _ = grm_vanraden(mdp["G"][valid], ploidy=2)

        model = MultiTraitMultiEnvLMM()
        nf = model.fit_null(Y_wide, X0_sub, K=K_sub, n_traits=2, n_envs=2)
        result = model.score_chunk(
            G_sub, nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "MT-MET")


@skip_no_mdp
class TestDiploidHaplotype:
    """Haplotype-based GWAS on MDP."""

    def test_haplotype_gwas_window(self, mdp):
        from torchgwas.models.haplotype_gwas import HaplotypeGWAS
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])

        scanner = HaplotypeGWAS(
            method="window", test="f_test",
            window_size=5, step=5, null_fit=nf,
            min_hap_freq=0.02,
        )
        result = scanner.scan(
            mdp["Y"], mdp["G_scan"][:, :50],
            X0=mdp["X0"],
            variant_pos=mdp["vmeta_scan"].pos[:50],
            variant_chr=mdp["vmeta_scan"].chr[:50],
        )
        assert result is not None
        assert len(result.block_id) > 0
        _assert_valid_pvalues(result.p_global, "HapGWAS-window")


# ===================================================================
# POLYPLOID MODELS on GWASpoly potato
# ===================================================================


@skip_no_potato
class TestPolyploid:
    """Polyploid models on tetraploid potato data."""

    def test_single_trait_lmm_polyploid(self, potato):
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model = SingleTraitLMM()
        nf = model.fit_null(potato["Y"], potato["X0"], K=potato["K"])
        result = model.score_chunk(
            potato["G_scan"], nf, potato["vmeta_scan"], test="wald"
        )
        _assert_valid_pvalues(result.p, "LMM-Polyploid")

    def test_glm_polyploid(self, potato):
        from torchgwas.models.glm import GLM
        model = GLM()
        nf = model.fit_null(potato["Y"], potato["X0"])
        result = model.score_chunk(
            potato["G_scan"], nf, potato["vmeta_scan"]
        )
        _assert_valid_pvalues(result.p, "GLM-Polyploid")

    def test_gene_action_models(self, potato):
        """Test multiple gene-action model encodings."""
        from torchgwas.preprocess.polyploid import recode_gene_action
        from torchgwas.models.single_trait_lmm import SingleTraitLMM

        model = SingleTraitLMM()
        nf = model.fit_null(potato["Y"], potato["X0"], K=potato["K"])

        for ga_model in ["additive", "1-dom", "2-dom"]:
            G_recoded = recode_gene_action(
                potato["G_scan"], model=ga_model, ploidy=4
            )
            result = model.score_chunk(
                G_recoded, nf, potato["vmeta_scan"], test="wald"
            )
            _assert_valid_pvalues(result.p, f"Polyploid-{ga_model}")

    def test_multi_kernel_polyploid(self, potato):
        from torchgwas.models.multi_kernel_lmm import MultiKernelLMM, build_multi_kernels
        kernels, _ = build_multi_kernels(
            potato["G"][:, :200], ploidy=4
        )
        model = MultiKernelLMM()
        nf = model.fit_null(potato["Y"], potato["X0"], kernels=kernels)
        result = model.score_chunk(
            potato["G_scan"], nf, potato["vmeta_scan"], test="wald"
        )
        _assert_valid_pvalues(result.p, "MK-LMM-Polyploid")

    def test_farmcpu_polyploid(self, potato):
        from torchgwas.models.farmcpu import FarmCPU
        model = FarmCPU(max_iter=3, p_threshold=0.05)
        nf = model.fit_null(potato["Y"], potato["X0"])
        result = model.score_chunk(
            potato["G_scan"], nf, potato["vmeta_scan"]
        )
        _assert_valid_pvalues(result.p, "FarmCPU-Polyploid")

    def test_binary_glm_polyploid(self, potato):
        from torchgwas.models.binary_glm import BinaryGLM
        med = potato["Y"].median()
        Y_bin = (potato["Y"] > med).to(STAT_DTYPE)
        model = BinaryGLM(use_spa=False)
        nf = model.fit_null(Y_bin, potato["X0"])
        result = model.score_chunk(
            potato["G_scan"][:, :100], nf,
            VariantMeta(
                snp=potato["vmeta_scan"].snp[:100],
                chr=potato["vmeta_scan"].chr[:100],
                pos=potato["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "BinaryGLM-Polyploid")

    def test_binary_glmm_polyploid(self, potato):
        from torchgwas.models.binary_glmm import BinaryGLMM
        med = potato["Y"].median()
        Y_bin = (potato["Y"] > med).to(STAT_DTYPE)
        model = BinaryGLMM(use_spa=False)
        nf = model.fit_null(Y_bin, potato["X0"], K=potato["K"])
        result = model.score_chunk(
            potato["G_scan"][:, :100], nf,
            VariantMeta(
                snp=potato["vmeta_scan"].snp[:100],
                chr=potato["vmeta_scan"].chr[:100],
                pos=potato["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )
        _assert_valid_pvalues(result.p, "BinaryGLMM-Polyploid")

    def test_set_based_polyploid(self, potato):
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        from torchgwas.models.set_based import SetBasedScanner
        from torchgwas.io.regions import Region

        model = SingleTraitLMM()
        nf = model.fit_null(potato["Y"], potato["X0"], K=potato["K"])

        chrs = potato["vmeta_scan"].chr[:30]
        pos = potato["vmeta_scan"].pos[:30]
        regions = [
            Region(region_id=f"R{i}", chr=chrs[i * 10], start=pos[i * 10], end=pos[min((i + 1) * 10 - 1, 29)])
            for i in range(3)
        ]
        scanner = SetBasedScanner(nf, ploidy=4)
        result = scanner.scan_regions(
            potato["G"][:, :30], regions,
            chrs, pos,
            test="skat",
        )
        _assert_valid_pvalues(result.p, "SKAT-Polyploid")


# ===================================================================
# POST-GWAS and PGS
# ===================================================================


@skip_no_mdp
class TestPostGWAS:
    """Post-GWAS analyses on MDP-derived summary statistics."""

    def test_ldsc_h2(self, mdp):
        from torchgwas.postgwas._ldsc import ldsc_h2

        # Generate summary stats from LMM
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"], nf, mdp["vmeta_scan"], test="wald"
        )

        # Compute chi2 from z-scores
        z = result.beta / result.se
        chi2 = z ** 2

        # Simple LD scores (ones for smoke test)
        m = len(chi2)
        ld_scores = torch.ones(m, dtype=STAT_DTYPE)

        h2_result = ldsc_h2(
            chi2=chi2, ld_scores=ld_scores,
            n=mdp["n"], m_total=m,
        )
        assert h2_result is not None
        assert hasattr(h2_result, "h2")

    def test_meta_analysis(self, mdp):
        from torchgwas.postgwas._meta import meta_fixed_effect

        # Generate two "studies" with slightly different p-values
        from torchgwas.models.single_trait_lmm import SingleTraitLMM
        model = SingleTraitLMM()
        nf = model.fit_null(mdp["Y"], mdp["X0"], K=mdp["K"])
        result = model.score_chunk(
            mdp["G_scan"][:, :50], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:50],
                chr=mdp["vmeta_scan"].chr[:50],
                pos=mdp["vmeta_scan"].pos[:50],
                a1=["A"] * 50, a2=["G"] * 50,
            ),
            test="wald",
        )

        # Stack into (m, K) tensors
        beta_stack = torch.stack([result.beta, result.beta * 0.9], dim=1)
        se_stack = torch.stack([result.se, result.se * 1.1], dim=1)

        meta_result = meta_fixed_effect(beta_stack, se_stack)
        assert meta_result is not None
        _assert_valid_pvalues(meta_result.p_meta, "Meta-FE")

    def test_ld_clump(self, mdp):
        from torchgwas.postgwas._clump import ld_clump

        # Generate p-values
        from torchgwas.models.glm import GLM
        model = GLM()
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(
            mdp["G_scan"][:, :100], nf,
            VariantMeta(
                snp=mdp["vmeta_scan"].snp[:100],
                chr=mdp["vmeta_scan"].chr[:100],
                pos=mdp["vmeta_scan"].pos[:100],
                a1=["A"] * 100, a2=["G"] * 100,
            ),
        )

        clump_result = ld_clump(
            p=result.p,
            G=mdp["G_scan"][:, :100],
            pos=mdp["vmeta_scan"].pos[:100],
            chr_labels=mdp["vmeta_scan"].chr[:100],
            p_threshold=0.5,  # Lenient for smoke test
            r2_threshold=0.1,
        )
        assert clump_result is not None


@skip_no_mdp
class TestVisualization:
    """Visualization smoke tests (headless, Agg backend)."""

    def test_manhattan_plot(self, mdp):
        import matplotlib
        matplotlib.use("Agg")

        from torchgwas.models.glm import GLM
        model = GLM()
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(mdp["G_scan"], nf, mdp["vmeta_scan"])

        from torchgwas.viz import manhattan_plot
        ax = manhattan_plot(
            chrom=mdp["vmeta_scan"].chr[:200],
            pos=mdp["vmeta_scan"].pos[:200],
            p=result.p[:200],
        )
        assert ax is not None

    def test_qq_plot(self, mdp):
        import matplotlib
        matplotlib.use("Agg")

        from torchgwas.models.glm import GLM
        model = GLM()
        nf = model.fit_null(mdp["Y"], mdp["X0"])
        result = model.score_chunk(mdp["G_scan"], nf, mdp["vmeta_scan"])

        from torchgwas.viz import qq_plot
        ax = qq_plot(p=result.p[:200])
        assert ax is not None


# ===================================================================
# LD Block Detection (real genotype data)
# ===================================================================


@skip_no_mdp
class TestLDBlocks:
    """LD block detection on real MDP genotype data."""

    def test_gabriel_blocks(self, mdp):
        from torchgwas.ld import detect_blocks

        # Use first 100 SNPs on chromosome 1
        chr1_mask = [c == "1" for c in mdp["vmeta_scan"].chr[:100]]
        n_chr1 = sum(chr1_mask)
        if n_chr1 < 10:
            pytest.skip("Not enough chr1 SNPs")

        G_chr1 = mdp["G_scan"][:, :n_chr1]
        blocks = detect_blocks(
            G_chr1,
            variant_pos=mdp["vmeta_scan"].pos[:n_chr1],
            variant_chr=mdp["vmeta_scan"].chr[:n_chr1],
            method="gabriel",
        )
        assert isinstance(blocks, list)
        assert len(blocks) >= 0

    def test_spine_blocks(self, mdp):
        from torchgwas.ld import detect_blocks

        chr1_count = sum(1 for c in mdp["vmeta_scan"].chr[:100] if c == "1")
        if chr1_count < 10:
            pytest.skip("Not enough chr1 SNPs")

        G_chr1 = mdp["G_scan"][:, :chr1_count]
        blocks = detect_blocks(
            G_chr1,
            variant_pos=mdp["vmeta_scan"].pos[:chr1_count],
            variant_chr=mdp["vmeta_scan"].chr[:chr1_count],
            method="spine",
        )
        assert isinstance(blocks, list)


# ===================================================================
# Imputation (real genotype data with induced missingness)
# ===================================================================


@skip_no_mdp
class TestImputation:
    """Imputation methods on MDP with induced missingness."""

    def test_mean_imputation(self, mdp):
        from torchgwas.preprocess.impute import impute_mean

        G_miss = mdp["G_scan"][:, :50].clone()
        # Introduce 10% missingness
        torch.manual_seed(42)
        mask = torch.rand_like(G_miss) < 0.1
        G_miss[mask] = float("nan")

        G_imp = impute_mean(G_miss)
        assert not torch.isnan(G_imp).any()

    def test_knn_imputation(self, mdp):
        from torchgwas.preprocess.impute import impute_knn

        G_miss = mdp["G_scan"][:, :50].clone()
        torch.manual_seed(42)
        mask = torch.rand_like(G_miss) < 0.1
        G_miss[mask] = float("nan")

        G_imp = impute_knn(G_miss, mdp["K"], k=5)
        assert not torch.isnan(G_imp).any()


# ===================================================================
# QC (real genotype data)
# ===================================================================


@skip_no_mdp
class TestQC:
    """Quality control on MDP data."""

    def test_maf_filter(self, mdp):
        from torchgwas.preprocess.standardize import compute_allele_frequencies, compute_maf

        af = compute_allele_frequencies(mdp["G_scan"], ploidy=2)
        maf = compute_maf(af)
        assert maf.shape[0] == 200
        assert torch.all(maf >= 0) and torch.all(maf <= 0.5)

    def test_variant_qc(self, mdp):
        from torchgwas.preprocess.qc import compute_variant_qc

        qc = compute_variant_qc(
            mdp["G_scan"], mdp["vmeta_scan"], ploidy=2,
        )
        assert qc is not None
