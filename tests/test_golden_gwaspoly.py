"""Golden tests: TorchGWAS vs GWASpoly reference outputs.

GWASpoly references cover polyploid gene-action models on tetraploid potato data
(957 genotypes, 9888 markers, vine.maturity trait, 6 environments).

Uses P3D approach (additive kinship for null, per-model scan genotypes)
to match GWASpoly's method. Benchmark showed r(-log10p) > 0.999 for
additive, 1-dom, 2-dom, 3-dom models.

Tolerances follow Section 16 of the charter.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch

from torchgwas.config import STAT_DTYPE, NumericalConfig
from torchgwas.linalg.kinship_polyploid import grm_polyploid_gene_action
from torchgwas.models.base import VariantMeta
from torchgwas.models.single_trait_lmm import SingleTraitLMM
from torchgwas.preprocess.polyploid import recode_gene_action

pytestmark = pytest.mark.golden

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "benchmark", "gwaspoly_results"
)
PLOIDY = 4


def _have_gwaspoly_data():
    """Check if GWASpoly reference data is available."""
    return os.path.isfile(os.path.join(RESULTS_DIR, "gwaspoly_additive.csv"))


def _load_data():
    """Load aligned potato data and build multi-env model inputs."""
    pheno = pd.read_csv(os.path.join(RESULTS_DIR, "potato_pheno_with_genoid.csv"))
    geno = pd.read_csv(os.path.join(RESULTS_DIR, "potato_geno_aligned.csv"), index_col=0)
    marker_map = pd.read_csv(os.path.join(RESULTS_DIR, "potato_map.csv"))

    geno.index = geno.index.astype(str)
    pheno = pheno.dropna(subset=["vine.maturity"]).copy()
    pheno["geno_id"] = pheno["geno_id"].astype(str)
    pheno = pheno[pheno["geno_id"].isin(geno.index)].reset_index(drop=True)

    unique_genos = sorted(pheno["geno_id"].unique())
    geno_sub = geno.loc[unique_genos]
    n_obs = len(pheno)
    n_geno = len(unique_genos)

    geno_id_to_idx = {gid: i for i, gid in enumerate(unique_genos)}
    Z = torch.zeros(n_obs, n_geno, dtype=STAT_DTYPE)
    for i, gid in enumerate(pheno["geno_id"]):
        Z[i, geno_id_to_idx[gid]] = 1.0

    env_dummies = pd.get_dummies(pheno["env"], drop_first=True, dtype=float)
    X0 = torch.tensor(
        np.column_stack([np.ones(n_obs), env_dummies.values]), dtype=STAT_DTYPE
    )

    G_geno = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G_geno.shape[1]):
        col = G_geno[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G_geno[mask, j] = col[~mask].mean()

    G_obs = Z @ G_geno
    Y = torch.tensor(pheno["vine.maturity"].values, dtype=STAT_DTYPE)

    snp_names = list(geno_sub.columns)
    chrs = marker_map["chrom"].astype(str).tolist()
    pos = marker_map["pos"].astype(int).tolist()
    vmeta = VariantMeta(
        snp=snp_names, chr=chrs, pos=pos,
        a1=["A"] * len(snp_names), a2=["G"] * len(snp_names),
    )

    return Y, X0, G_obs, G_geno, Z, vmeta, snp_names


def _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, model):
    """Run GWAS with P3D approach: additive K for null, per-model scan genotypes."""
    K_geno, _ = grm_polyploid_gene_action(G_geno, "additive", PLOIDY)
    K_obs = Z @ K_geno @ Z.T

    G_scan = recode_gene_action(G_obs, model, PLOIDY) if model != "additive" else G_obs

    config = NumericalConfig(reml_method="emma")
    lmm = SingleTraitLMM(config=config)
    nf = lmm.fit_null(Y, X0, K=K_obs)
    result = lmm.score_chunk(G_scan, nf, vmeta, test="wald")
    return result.p.detach().cpu().numpy()


def _load_gwaspoly_ref(model_fname):
    """Load GWASpoly reference p-values."""
    path = os.path.join(RESULTS_DIR, f"gwaspoly_{model_fname}.csv")
    df = pd.read_csv(path)
    return dict(zip(df["marker"], df["pvalue"]))


def _compute_logp_corr(p_torch, snp_names, gwaspoly_dict):
    """Compute Pearson correlation of -log10(p) between TorchGWAS and GWASpoly."""
    from scipy.stats import pearsonr

    p1, p2 = [], []
    for i, snp in enumerate(snp_names):
        if snp in gwaspoly_dict and np.isfinite(p_torch[i]) and p_torch[i] > 0:
            gp = gwaspoly_dict[snp]
            if np.isfinite(gp) and gp > 0:
                p1.append(p_torch[i])
                p2.append(gp)

    if len(p1) < 10:
        return 0.0, 0

    logp1 = -np.log10(np.array(p1))
    logp2 = -np.log10(np.array(p2))
    r, _ = pearsonr(logp1, logp2)
    return r, len(p1)


@pytest.fixture(scope="module")
def potato_data():
    """Load potato data once for all tests."""
    if not _have_gwaspoly_data():
        pytest.skip("GWASpoly reference data not available")
    return _load_data()


class TestGWASPolyTetraploid:
    """GWASpoly tetraploid reference comparison.

    Uses P3D approach to match GWASpoly's multi-env mixed model:
    - Additive kinship K expanded to observation level via Z@K@Z'
    - env as fixed factor effect
    - Variance components estimated once (P3D)
    - Per-model scan uses recoded genotypes
    """

    def test_additive_pvalues(self, potato_data):
        """Additive model p-values match GWASpoly (r > 0.999)."""
        Y, X0, G_obs, G_geno, Z, vmeta, snp_names = potato_data
        p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "additive")
        ref = _load_gwaspoly_ref("additive")
        r, n = _compute_logp_corr(p, snp_names, ref)
        assert n > 9000, f"Too few common markers: {n}"
        assert r > 0.999, f"Additive r(-log10p) = {r:.6f}, expected > 0.999"

    def test_1dom_pvalues(self, potato_data):
        """1-dom (simplex dominant) p-values match GWASpoly 1-dom-alt (r > 0.999)."""
        Y, X0, G_obs, G_geno, Z, vmeta, snp_names = potato_data
        p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "1-dom")
        ref = _load_gwaspoly_ref("1_dom_alt")
        r, n = _compute_logp_corr(p, snp_names, ref)
        assert n > 5000, f"Too few common markers: {n}"
        assert r > 0.999, f"1-dom r(-log10p) = {r:.6f}, expected > 0.999"

    def test_2dom_pvalues(self, potato_data):
        """2-dom (duplex dominant) p-values match GWASpoly 2-dom-alt (r > 0.999)."""
        Y, X0, G_obs, G_geno, Z, vmeta, snp_names = potato_data
        p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "2-dom")
        ref = _load_gwaspoly_ref("2_dom_alt")
        r, n = _compute_logp_corr(p, snp_names, ref)
        assert n > 7000, f"Too few common markers: {n}"
        assert r > 0.999, f"2-dom r(-log10p) = {r:.6f}, expected > 0.999"

    def test_3dom_pvalues(self, potato_data):
        """3-dom (triplex) p-values match GWASpoly 2-dom-ref (r > 0.999)."""
        Y, X0, G_obs, G_geno, Z, vmeta, snp_names = potato_data
        p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "3-dom")
        ref = _load_gwaspoly_ref("2_dom_ref")
        r, n = _compute_logp_corr(p, snp_names, ref)
        assert n > 8000, f"Too few common markers: {n}"
        assert r > 0.999, f"3-dom r(-log10p) = {r:.6f}, expected > 0.999"

    def test_diplo_additive_differs(self, potato_data):
        """Diplo-additive encoding differs from GWASpoly — correlation expected < 0.6.

        GWASpoly: {0->0, 1,2,3->1, 4->2} (diploidized)
        TorchGWAS: min(dose, ploidy-dose) = {0->0, 1->1, 2->2, 3->1, 4->0}
        """
        Y, X0, G_obs, G_geno, Z, vmeta, snp_names = potato_data
        p = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "diplo-additive")
        ref = _load_gwaspoly_ref("diplo_additive")
        r, n = _compute_logp_corr(p, snp_names, ref)
        assert n > 9000, f"Too few common markers: {n}"
        # Different encoding — moderate correlation only
        assert r > 0.3, f"diplo-additive r = {r:.4f}, expected > 0.3"
        assert r < 0.7, f"diplo-additive r = {r:.4f}, unexpectedly high for different encoding"

    def test_gene_action_encoding(self):
        """Verify gene-action encoding matches GWASpoly conventions."""
        doses = torch.tensor([[0, 1, 2, 3, 4]], dtype=STAT_DTYPE)

        # 1-dom: 1 if dose >= 1 → matches GWASpoly 1-dom-alt
        enc = recode_gene_action(doses, "1-dom", 4)
        expected = torch.tensor([[0, 1, 1, 1, 1]], dtype=STAT_DTYPE)
        assert torch.equal(enc, expected)

        # 2-dom: 1 if dose >= 2 → matches GWASpoly 2-dom-alt
        enc = recode_gene_action(doses, "2-dom", 4)
        expected = torch.tensor([[0, 0, 1, 1, 1]], dtype=STAT_DTYPE)
        assert torch.equal(enc, expected)

        # 3-dom: 1 if dose >= 3 → matches GWASpoly 2-dom-ref (1 if dose > 2)
        enc = recode_gene_action(doses, "3-dom", 4)
        expected = torch.tensor([[0, 0, 0, 1, 1]], dtype=STAT_DTYPE)
        assert torch.equal(enc, expected)

        # diplo-additive: min(d, k-d)
        enc = recode_gene_action(doses, "diplo-additive", 4)
        expected = torch.tensor([[0, 1, 2, 1, 0]], dtype=STAT_DTYPE)
        assert torch.equal(enc, expected)
