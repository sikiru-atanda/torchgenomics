"""Print a side-by-side TorchGenomics vs GEMMA agreement report on MDP EarHT/dpoll.

Same fixtures and same fit calls as ``tests/test_golden_gemma.py`` — but prints
the actual measured numbers instead of asserting thresholds.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.models.base import VariantMeta
from torchgenomics.models.single_trait_lmm import SingleTraitLMM
from torchgenomics.models.multi_trait_lmm import MultiTraitLMM


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GEMMA_OUT = os.path.join(REPO_ROOT, "gemma_demo", "output")
MDP_DATA = os.path.join(REPO_ROOT, "benchmark", "data")


def _is_floatlike(s):
    try:
        float(s); return True
    except ValueError:
        return False


def _parse_gemma_log(path):
    out = {}
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if "vg estimate in the null model" in low:
            out["vg"] = float(line.split("=")[-1].strip())
        elif "ve estimate in the null model" in low:
            out["ve"] = float(line.split("=")[-1].strip())
        elif "pve estimate in the null model" in low:
            out["pve"] = float(line.split("=")[-1].strip())
        elif "remle log-likelihood in the null model" in low:
            out["ll_reml"] = float(line.split("=")[-1].strip())
        elif "mle log-likelihood in the null model" in low:
            out["ll_ml"] = float(line.split("=")[-1].strip())
        elif "remle estimate for vg in the null model" in low:
            vals = [float(x) for x in lines[i + 1].split()]
            if i + 2 < len(lines):
                vals2 = lines[i + 2].split()
                if vals2 and _is_floatlike(vals2[0]):
                    out["Vg"] = np.array([[vals[0], float(vals2[0])],
                                          [float(vals2[0]), float(vals2[1])]])
                else:
                    out["Vg"] = np.array([[vals[0]]])
        elif "remle estimate for ve in the null model" in low:
            vals = [float(x) for x in lines[i + 1].split()]
            if i + 2 < len(lines):
                vals2 = lines[i + 2].split()
                if vals2 and _is_floatlike(vals2[0]):
                    out["Ve"] = np.array([[vals[0], float(vals2[0])],
                                          [float(vals2[0]), float(vals2[1])]])
                else:
                    out["Ve"] = np.array([[vals[0]]])
    return out


def load_mdp():
    pheno = pd.read_csv(os.path.join(MDP_DATA, "mdp_traits.txt"), sep="\t")
    geno = pd.read_csv(os.path.join(MDP_DATA, "mdp_numeric.txt"), sep="\t")
    snpmap = pd.read_csv(os.path.join(MDP_DATA, "mdp_SNP_information.txt"), sep="\t")
    pheno["Taxa"] = pheno["Taxa"].astype(str)
    geno["taxa"] = geno["taxa"].astype(str)
    pheno_sub = pheno[["Taxa", "EarHT", "dpoll"]].dropna(subset=["EarHT", "dpoll"])
    common = sorted(set(pheno_sub["Taxa"]) & set(geno["taxa"]))
    pheno_sub = pheno_sub[pheno_sub["Taxa"].isin(common)].set_index("Taxa").loc[common]
    geno_sub = geno[geno["taxa"].isin(common)].set_index("taxa").loc[common]

    n, m = geno_sub.shape[0], geno_sub.shape[1]
    snp_names = list(geno_sub.columns)
    K = torch.tensor(pd.read_csv(os.path.join(GEMMA_OUT, "mdp_kinship.cXX.txt"),
                                 sep="\t", header=None).values, dtype=STAT_DTYPE)
    G = torch.tensor(geno_sub.values, dtype=STAT_DTYPE)
    for j in range(G.shape[1]):
        col = G[:, j]
        mask = torch.isnan(col)
        if mask.any():
            G[mask, j] = col[~mask].mean()
    Y1 = torch.tensor(pheno_sub["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
    Y2 = torch.tensor(pheno_sub[["EarHT", "dpoll"]].values, dtype=STAT_DTYPE)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    snpmap_dict = {str(r["SNP"]): (str(int(r["Chromosome"])), int(r["Position"]))
                   for _, r in snpmap.iterrows()}
    chrs = [snpmap_dict.get(s, ("0", 0))[0] for s in snp_names]
    pos = [snpmap_dict.get(s, ("0", 0))[1] for s in snp_names]
    vmeta = VariantMeta(snp=snp_names, chr=chrs, pos=pos,
                        a1=["A"] * m, a2=["G"] * m)
    return dict(n=n, m=m, G=G, Y1=Y1, Y2=Y2, X0=X0, K=K, vmeta=vmeta)


def _require_fixtures() -> None:
    """Exit with a helpful pointer if the gitignored fixtures aren't present."""
    required = [
        os.path.join(GEMMA_OUT, "mdp_kinship.cXX.txt"),
        os.path.join(GEMMA_OUT, "mdp_lmm_all.log.txt"),
        os.path.join(GEMMA_OUT, "mdp_lmm_all.assoc.txt"),
        os.path.join(GEMMA_OUT, "mdp_mvlmm_wald.log.txt"),
        os.path.join(GEMMA_OUT, "mdp_mvlmm_wald.assoc.txt"),
        os.path.join(MDP_DATA, "mdp_traits.txt"),
        os.path.join(MDP_DATA, "mdp_numeric.txt"),
        os.path.join(MDP_DATA, "mdp_SNP_information.txt"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("GEMMA reference outputs or MDP data not found:", file=sys.stderr)
        for p in missing:
            print(f"  missing: {p}", file=sys.stderr)
        print(
            "\nRun `python scripts/generate_golden_data.py --gemma` to regenerate.",
            file=sys.stderr,
        )
        sys.exit(1)


def report():
    _require_fixtures()
    d = load_mdp()
    print("=" * 78)
    print("  TorchGenomics vs GEMMA 0.98.5  -  MDP maize (276 samples, 3093 SNPs)")
    print("=" * 78)
    print(f"Samples common to geno+pheno: {d['n']}")
    print(f"SNPs: {d['m']}")
    print()

    # ---- Single-trait (EarHT) ----
    ref1 = _parse_gemma_log(os.path.join(GEMMA_OUT, "mdp_lmm_all.log.txt"))
    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(d["Y1"], d["X0"], K=d["K"])

    print("+" + "-" * 76 + "+")
    print("| SINGLE-TRAIT LMM (EarHT) - null-model estimates" + " " * 28 + "|")
    print("+" + "-" * 76 + "+")
    print(f"  vg (sig2_g)      TorchGenomics = {float(nf.sig2_g):.6f}   "
          f"GEMMA = {ref1['vg']:.6f}   "
          f"rel diff = {abs(float(nf.sig2_g) - ref1['vg'])/ref1['vg']:.2e}")
    print(f"  ve (sig2_e)      TorchGenomics = {float(nf.sig2_e):.6f}   "
          f"GEMMA = {ref1['ve']:.6f}   "
          f"rel diff = {abs(float(nf.sig2_e) - ref1['ve'])/ref1['ve']:.2e}")
    print(f"  REML log-lik     TorchGenomics = {float(nf.log_likelihood):.4f}   "
          f"GEMMA = {ref1['ll_reml']:.4f}   "
          f"abs diff = {abs(float(nf.log_likelihood) - ref1['ll_reml']):.2e}")
    pve_tg = float(nf.sig2_g) / (float(nf.sig2_g) + float(nf.sig2_e))
    print(f"  PVE (h2)         TorchGenomics = {pve_tg:.6f}   "
          f"GEMMA = {ref1.get('pve', float('nan')):.6f}")
    print()

    # ---- Per-SNP p-values ----
    gemma = pd.read_csv(os.path.join(GEMMA_OUT, "mdp_lmm_all.assoc.txt"), sep="\t")
    print("+" + "-" * 76 + "+")
    print("| SINGLE-TRAIT LMM - per-SNP p-value agreement" + " " * 31 + "|")
    print("+" + "-" * 76 + "+")
    print("  test    N merged   -log10(p) corr   max |delta -log10(p)|")
    print("  " + "-" * 62)
    for test_name in ["wald", "lrt", "score"]:
        result = model.score_chunk(d["G"], nf, d["vmeta"], test=test_name)
        tg_df = pd.DataFrame({"rs": result.snp, "p_tg": result.p.cpu().numpy()})
        col = f"p_{test_name}"
        merged = pd.merge(gemma, tg_df, on="rs").dropna(subset=[col, "p_tg"])
        logp_ref = np.log10(np.clip(merged[col].values, 1e-300, 1))
        logp_tg = np.log10(np.clip(merged["p_tg"].values, 1e-300, 1))
        corr = np.corrcoef(logp_ref, logp_tg)[0, 1]
        max_delta = np.max(np.abs(logp_ref - logp_tg))
        print(f"  {test_name:7s}   {len(merged):6d}       {corr:.6f}        {max_delta:.4f}")
    print()

    # ---- Beta / SE (Wald) ----
    result = model.score_chunk(d["G"], nf, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({
        "rs": result.snp,
        "beta_tg": result.beta.cpu().numpy(),
        "se_tg": result.se.cpu().numpy(),
        "p_tg": result.p.cpu().numpy(),
    })
    merged = pd.merge(gemma, tg_df, on="rs").dropna(subset=["beta", "beta_tg"])
    beta_corr = np.corrcoef(merged["beta"].values, merged["beta_tg"].values)[0, 1]
    se_corr = np.corrcoef(merged["se"].values, merged["se_tg"].values)[0, 1]
    beta_med_all = np.median(np.abs(merged["beta"].values - merged["beta_tg"].values))
    se_med_all = np.median(np.abs(merged["se"].values - merged["se_tg"].values))
    common = merged[(merged["af"] >= 0.03) & (merged["af"] <= 0.97)]
    beta_med_c = np.median(np.abs(common["beta"].values - common["beta_tg"].values))
    se_med_c = np.median(np.abs(common["se"].values - common["se_tg"].values))
    print("+" + "-" * 76 + "+")
    print("| SINGLE-TRAIT LMM - beta / SE agreement (Wald)" + " " * 30 + "|")
    print("+" + "-" * 76 + "+")
    print(f"  N merged SNPs (all)       : {len(merged)}")
    print(f"  N merged SNPs (AF 0.03-0.97): {len(common)}")
    print(f"  beta corr (all)           : {beta_corr:.6f}")
    print(f"  beta median |delta| (all)   : {beta_med_all:.4f}")
    print(f"  beta median |delta| (common): {beta_med_c:.4f}")
    print(f"  SE corr (all)             : {se_corr:.6f}")
    print(f"  SE median |delta| (all)     : {se_med_all:.4f}")
    print(f"  SE median |delta| (common)  : {se_med_c:.4f}")
    print()

    # ---- Top hits side-by-side ----
    top = merged.nsmallest(5, "p_wald")[
        ["rs", "chr", "af", "beta", "se", "p_wald", "beta_tg", "se_tg", "p_tg"]
    ]
    print("+" + "-" * 76 + "+")
    print("| Top-5 GEMMA hits - side-by-side values" + " " * 37 + "|")
    print("+" + "-" * 76 + "+")
    print(f"  {'SNP':<10} {'AF':>5}   {'beta_GEM':>9} {'beta_TG':>9}   "
          f"{'SE_GEM':>7} {'SE_TG':>7}   {'p_GEM':>10} {'p_TG':>10}")
    for row in top.itertuples():
        print(f"  {row.rs:<10} {row.af:>5.3f}   {row.beta:>9.4f} {row.beta_tg:>9.4f}   "
              f"{row.se:>7.4f} {row.se_tg:>7.4f}   {row.p_wald:>10.3e} {row.p_tg:>10.3e}")
    print()

    # ---- mvLMM ----
    ref2 = _parse_gemma_log(os.path.join(GEMMA_OUT, "mdp_mvlmm_wald.log.txt"))
    mv = MultiTraitLMM()
    nf2 = mv.fit_null(d["Y2"], d["X0"], K=d["K"])
    Vg_tg = nf2.Vg.cpu().numpy()
    Ve_tg = nf2.Ve.cpu().numpy()
    print("+" + "-" * 76 + "+")
    print("| MULTI-TRAIT LMM (EarHT, dpoll) - variance-covariance matrices" + " " * 14 + "|")
    print("+" + "-" * 76 + "+")
    print("  Vg (genetic) matrix")
    print(f"    TorchGenomics:\n{Vg_tg}")
    print(f"    GEMMA:\n{ref2['Vg']}")
    print(f"    max rel diff: {np.max(np.abs(Vg_tg - ref2['Vg'])/np.maximum(np.abs(ref2['Vg']), 1e-9)):.2e}")
    print("  Ve (residual) matrix")
    print(f"    TorchGenomics:\n{Ve_tg}")
    print(f"    GEMMA:\n{ref2['Ve']}")
    print(f"    max rel diff: {np.max(np.abs(Ve_tg - ref2['Ve'])/np.maximum(np.abs(ref2['Ve']), 1e-9)):.2e}")
    print()

    # ---- mvLMM p-values ----
    res_mv = mv.score_chunk(d["G"], nf2, d["vmeta"], test="wald")
    tg_df = pd.DataFrame({"rs": res_mv.snp, "p_tg": res_mv.p.cpu().numpy()})
    gmv = pd.read_csv(os.path.join(GEMMA_OUT, "mdp_mvlmm_wald.assoc.txt"), sep="\t")
    merged_mv = pd.merge(gmv, tg_df, on="rs").dropna(subset=["p_wald", "p_tg"])
    logp_ref = np.log10(np.clip(merged_mv["p_wald"].values, 1e-300, 1))
    logp_tg = np.log10(np.clip(merged_mv["p_tg"].values, 1e-300, 1))
    corr = np.corrcoef(logp_ref, logp_tg)[0, 1]
    max_delta = np.max(np.abs(logp_ref - logp_tg))
    print("+" + "-" * 76 + "+")
    print("| MULTI-TRAIT LMM - joint Wald p-value agreement" + " " * 29 + "|")
    print("+" + "-" * 76 + "+")
    print(f"  N merged SNPs                 : {len(merged_mv)}")
    print(f"  joint Wald -log10(p) corr     : {corr:.6f}")
    print(f"  max |delta -log10(p)|         : {max_delta:.4f}")
    print()


if __name__ == "__main__":
    report()
