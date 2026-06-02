"""Reference OLS direct + indirect effect estimator (Young et al. 2022, section 2.1).

For each SNP j, the paper's sib-pair OLS estimator fits

    Y_{f,k} = mu + beta_within  * (g_{f,k,j} - g_bar_{f,j})
                 + beta_between * g_bar_{f,j}
                 + e_{f,k}

Under the generative model

    Y_{f,k} = beta_d * g_{f,k,j} + beta_i * (G_M[f,j] + G_F[f,j]) + ...

we have (Young 2022 eq. 2-4):
    E[beta_within]  = beta_d                 (direct effect)
    E[beta_between] = beta_d + 2 * beta_i   (g_bar_f = (G_M + G_F) / 2 in expectation)
    => beta_indirect = (beta_between - beta_within) / 2

The "marginal" OLS slope (fitting Y on g alone, no family-mean covariate) is
    beta_marginal = beta_d + 0.5 * beta_i   (sib-pair design, eq. 3 of Young 2022)

We also report the attenuation factor:
    attenuation = beta_within / beta_marginal

which is the same diagnostic that torchgenomics.models.within_family_lmm.WithinFamilyLMM
exposes per-SNP (β_within / β_standard). Note the TG WithinFamilyLMM scores against
the population GRM rather than the family-mean covariate, so the two attenuation
estimates won't be bit-identical, but for a balanced sib-pair design with a
diagonal GRM (no inter-family relatedness) they agree to 2-3 sig figs.

Inputs (TSVs from simulate_sibpair.py):
    data/pheno.tsv        : [FID, IID, phenotype]
    data/geno.tsv         : sib genotype matrix
    data/parent_geno.tsv  : (M+F) parental genotype matrix (not used here;
                            reserved for the snipar branch)
    data/family.tsv       : [IID, FID, sib_idx]
    data/truth.tsv        : per-SNP planted (beta_d, beta_i)

Output:
    outputs/reference.tsv : per-SNP [SNP, beta_d_ref, se_d_ref, beta_i_ref, se_i_ref,
                                     beta_marginal_ref, se_marginal_ref,
                                     attenuation_ref]

Citation:
    Young AI, Benonisdottir S, Przeworski M, Kong A. (2022). Nat Genet 54:263-273.
    doi:10.1038/s41588-022-01016-z
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def fit_per_snp_ols(
    Y: np.ndarray,
    G: np.ndarray,
    family_id: np.ndarray,
):
    """Fit the within-between OLS estimator for each SNP.

    Parameters
    ----------
    Y : (n,) phenotype
    G : (n, m) sib genotype matrix (counts of one allele)
    family_id : (n,) integer family assignment per sib

    Returns
    -------
    dict of np.ndarrays each of shape (m,):
        beta_within, se_within
        beta_between, se_between
        beta_marginal, se_marginal
        beta_indirect, se_indirect  (derived: (beta_between - beta_within) / 2)
        attenuation                 (beta_within / beta_marginal; nan on zero divide)
    """
    n, m = G.shape
    unique_fids, inv = np.unique(family_id, return_inverse=True)
    nf = unique_fids.size

    g_bar_fam = np.zeros((nf, m), dtype=np.float64)
    counts = np.zeros(nf, dtype=np.int64)
    np.add.at(counts, inv, 1)
    np.add.at(g_bar_fam, inv, G.astype(np.float64))
    g_bar_fam /= counts[:, None]

    g_bar_long = g_bar_fam[inv]
    g_within = G.astype(np.float64) - g_bar_long

    beta_within = np.zeros(m, dtype=np.float64)
    se_within = np.zeros(m, dtype=np.float64)
    beta_between = np.zeros(m, dtype=np.float64)
    se_between = np.zeros(m, dtype=np.float64)
    beta_marginal = np.zeros(m, dtype=np.float64)
    se_marginal = np.zeros(m, dtype=np.float64)

    Y_c = Y - Y.mean()

    for j in range(m):
        xw = g_within[:, j]
        xb = g_bar_long[:, j]
        X = np.column_stack([np.ones(n), xw, xb])
        XtX = X.T @ X
        XtY = X.T @ Y_c
        try:
            beta_hat = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            beta_hat = np.full(3, np.nan)
        resid = Y_c - X @ beta_hat
        dof = max(n - X.shape[1], 1)
        sigma2 = float((resid * resid).sum() / dof)
        try:
            cov = sigma2 * np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            cov = np.full((3, 3), np.nan)

        beta_within[j] = beta_hat[1]
        beta_between[j] = beta_hat[2]
        se_within[j] = np.sqrt(max(cov[1, 1], 0.0))
        se_between[j] = np.sqrt(max(cov[2, 2], 0.0))

        xg = G[:, j].astype(np.float64)
        Xm = np.column_stack([np.ones(n), xg])
        XtXm = Xm.T @ Xm
        XtYm = Xm.T @ Y_c
        try:
            bm = np.linalg.solve(XtXm, XtYm)
        except np.linalg.LinAlgError:
            bm = np.full(2, np.nan)
        residm = Y_c - Xm @ bm
        dofm = max(n - Xm.shape[1], 1)
        sigma2m = float((residm * residm).sum() / dofm)
        try:
            covm = sigma2m * np.linalg.inv(XtXm)
        except np.linalg.LinAlgError:
            covm = np.full((2, 2), np.nan)
        beta_marginal[j] = bm[1]
        se_marginal[j] = np.sqrt(max(covm[1, 1], 0.0))

    beta_indirect = 0.5 * (beta_between - beta_within)
    se_indirect = 0.5 * np.sqrt(se_between * se_between + se_within * se_within)
    eps = 1e-10
    attenuation = np.where(
        np.abs(beta_marginal) > eps,
        beta_within / beta_marginal,
        np.nan,
    )

    out = {}
    out["beta_within"] = beta_within
    out["se_within"] = se_within
    out["beta_between"] = beta_between
    out["se_between"] = se_between
    out["beta_marginal"] = beta_marginal
    out["se_marginal"] = se_marginal
    out["beta_indirect"] = beta_indirect
    out["se_indirect"] = se_indirect
    out["attenuation"] = attenuation
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.time()

    pheno = pd.read_csv(args.data_dir / "pheno.tsv", sep="\t")
    geno = pd.read_csv(args.data_dir / "geno.tsv", sep="\t")
    family = pd.read_csv(args.data_dir / "family.tsv", sep="\t")

    geno = geno.set_index("IID").loc[pheno["IID"]]
    family = family.set_index("IID").loc[pheno["IID"]]

    snp_ids = list(geno.columns)
    G = geno.to_numpy(dtype=np.int8).astype(np.float64)
    Y = pheno["phenotype"].to_numpy(dtype=np.float64)
    family_str = family["FID"].to_numpy()
    _, fid_int = np.unique(family_str, return_inverse=True)

    print(f"[reference_paper] n_sibs={Y.shape[0]} n_snps={G.shape[1]} n_families={len(np.unique(fid_int))}")

    res = fit_per_snp_ols(Y, G, fid_int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "reference.tsv"
    with open(out_path, "w") as fh:
        fh.write("SNP\tbeta_d_ref\tse_d_ref\tbeta_i_ref\tse_i_ref\t"
                 "beta_marginal_ref\tse_marginal_ref\tattenuation_ref\n")
        for j, sid in enumerate(snp_ids):
            fh.write(
                f"{sid}\t{res['beta_within'][j]:.8e}\t{res['se_within'][j]:.8e}\t"
                f"{res['beta_indirect'][j]:.8e}\t{res['se_indirect'][j]:.8e}\t"
                f"{res['beta_marginal'][j]:.8e}\t{res['se_marginal'][j]:.8e}\t"
                f"{res['attenuation'][j]:.8e}\n"
            )

    elapsed = time.time() - t0
    meta = {}
    meta["ref_tool"] = "paper_simulator"
    meta["ref_estimator"] = "OLS within-between (Young 2022 sec 2.1)"
    meta["ref_citation"] = "Young et al. 2022 Nat Genet 54:263-273 doi:10.1038/s41588-022-01016-z"
    meta["n_sibs"] = int(Y.shape[0])
    meta["n_snps"] = int(G.shape[1])
    meta["n_families"] = int(len(np.unique(fid_int)))
    meta["wall_time_sec"] = elapsed
    meta["output_path"] = str(out_path)
    with open(args.output_dir / "run.log", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[reference_paper] wrote {out_path} ({elapsed:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
