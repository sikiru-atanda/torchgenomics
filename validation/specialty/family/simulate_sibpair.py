"""Simulate a sib-pair design with planted direct + indirect (NTC) effects.

Design (Young et al. 2022, Nat Genet 54:263-273, section 2.1):
    - Each family f has two parents (mother M_f, father F_f) and `s` sibs.
    - Each parent has a diploid genotype G_M, G_F at each SNP with allele
      frequency p_j drawn from Beta(2,2) (modes ~ 0.5).
    - Each sib inherits one allele uniformly from each parent (Mendelian
      transmission). Sib genotypes therefore lie in {0,1,2}.
    - For each SNP j we plant:
        beta_d[j] = direct effect on the sib's own phenotype.
        beta_i[j] = indirect / non-transmitted-coefficient (NTC) effect:
                    parental genotype G_M+G_F contributes to the sib's phenotype
                    via rearing / household exposure.
      The paper's analytic decomposition for OLS of Y on sib genotype alone:
            beta_OLS = beta_d + 0.5 * beta_i
      Within-family regression of (Y - Y_bar_f) on (g_i - g_bar_f) recovers
      beta_d unconfounded.

Phenotype generative model:
    Y_{f,k} = sum_j [ beta_d[j] * g_{f,k,j} + beta_i[j] * (G_M[f,j] + G_F[f,j]) ]
              + u_f + e_{f,k}
where u_f ~ N(0, sigma_u^2) and e_{f,k} ~ N(0, sigma_e^2). Total residual
variance is set so the planted signal explains `h2` of Var(Y) in expectation.

Citation:
    Young AI, Benonisdottir S, Przeworski M, Kong A. (2022).
    Nat Genet 54:263-273. doi:10.1038/s41588-022-01016-z
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def simulate(
    n_families: int = 500,
    sibs_per_family: int = 2,
    n_snps: int = 300,
    n_causal: int = 50,
    beta_d: float = 0.20,
    alpha: float = 0.50,
    h2: float = 0.40,
    seed: int = 42,
):
    """Run the sib-pair simulation. See module docstring for math."""
    rng = np.random.default_rng(seed)

    af = rng.beta(2.0, 2.0, size=n_snps)
    af = np.clip(af, 0.05, 0.95)

    G_M = rng.binomial(2, af, size=(n_families, n_snps)).astype(np.int8)
    G_F = rng.binomial(2, af, size=(n_families, n_snps)).astype(np.int8)
    parent_geno = (G_M.astype(np.int32) + G_F.astype(np.int32)).astype(np.int8)

    n_sibs = n_families * sibs_per_family
    family_id = np.repeat(np.arange(n_families), sibs_per_family)
    sib_idx = np.tile(np.arange(sibs_per_family), n_families)

    def transmit(parent_g):
        out = np.zeros_like(parent_g, dtype=np.int8)
        out[parent_g == 2] = 1
        het_mask = parent_g == 1
        out[het_mask] = rng.integers(0, 2, size=int(het_mask.sum())).astype(np.int8)
        return out

    parent_M_long = G_M[family_id]
    parent_F_long = G_F[family_id]
    sib_geno = transmit(parent_M_long) + transmit(parent_F_long)

    causal_idx = rng.choice(n_snps, size=n_causal, replace=False)
    beta_d_arr = np.zeros(n_snps, dtype=np.float64)
    beta_i_arr = np.zeros(n_snps, dtype=np.float64)
    beta_d_arr[causal_idx] = beta_d
    beta_i_arr[causal_idx] = alpha * beta_d

    var_direct = float(np.sum(beta_d_arr**2 * 2.0 * af * (1.0 - af)))
    var_indirect = float(np.sum(beta_i_arr**2 * 4.0 * af * (1.0 - af)))
    var_signal = var_direct + var_indirect
    var_total_target = var_signal / max(h2, 1e-6)
    var_resid = max(var_total_target - var_signal, 1e-6)
    sigma_u_sq = 0.5 * var_resid
    sigma_e_sq = 0.5 * var_resid

    u = rng.normal(0.0, np.sqrt(sigma_u_sq), size=n_families)
    e = rng.normal(0.0, np.sqrt(sigma_e_sq), size=n_sibs)

    signal_direct = sib_geno.astype(np.float64) @ beta_d_arr
    signal_indirect = parent_geno.astype(np.float64) @ beta_i_arr
    signal_indirect_long = signal_indirect[family_id]
    phenotype = signal_direct + signal_indirect_long + u[family_id] + e

    snp_id = [f"snp_{j:04d}" for j in range(n_snps)]

    var_y = float(np.var(phenotype))
    obs_h2 = (var_direct + var_indirect) / max(var_y, 1e-6)

    meta = {}
    meta["n_families"] = n_families
    meta["sibs_per_family"] = sibs_per_family
    meta["n_sibs"] = n_sibs
    meta["n_snps"] = n_snps
    meta["n_causal"] = n_causal
    meta["beta_d"] = beta_d
    meta["alpha"] = alpha
    meta["h2_target"] = h2
    meta["h2_observed_var"] = obs_h2
    meta["seed"] = seed
    meta["var_direct_planted"] = var_direct
    meta["var_indirect_planted"] = var_indirect
    meta["var_y_observed"] = var_y
    meta["sigma_u_sq"] = sigma_u_sq
    meta["sigma_e_sq"] = sigma_e_sq
    meta["causal_snps"] = [snp_id[i] for i in sorted(causal_idx.tolist())]
    meta["citation"] = "Young et al. 2022 Nat Genet 54:263-273 doi:10.1038/s41588-022-01016-z"

    out = {}
    out["sib_geno"] = sib_geno
    out["parent_geno"] = parent_geno
    out["phenotype"] = phenotype
    out["family_id"] = family_id
    out["sib_idx"] = sib_idx
    out["snp_id"] = snp_id
    out["beta_d_arr"] = beta_d_arr
    out["beta_i_arr"] = beta_i_arr
    out["allele_freq"] = af
    out["meta"] = meta
    return out


def write_outputs(sim: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_sibs = sim["sib_geno"].shape[0]
    n_families = sim["parent_geno"].shape[0]
    snp_id = sim["snp_id"]
    family_id = sim["family_id"]
    sib_idx = sim["sib_idx"]

    iid = [f"S{family_id[i]:04d}_{sib_idx[i]}" for i in range(n_sibs)]
    fid = [f"F{family_id[i]:04d}" for i in range(n_sibs)]

    with open(output_dir / "pheno.tsv", "w") as fh:
        fh.write("FID\tIID\tphenotype\n")
        for k in range(n_sibs):
            fh.write(f"{fid[k]}\t{iid[k]}\t{sim['phenotype'][k]:.6f}\n")

    with open(output_dir / "geno.tsv", "w") as fh:
        fh.write("IID\t" + "\t".join(snp_id) + "\n")
        for k in range(n_sibs):
            vals = "\t".join(str(int(v)) for v in sim["sib_geno"][k])
            fh.write(f"{iid[k]}\t{vals}\n")

    with open(output_dir / "parent_geno.tsv", "w") as fh:
        fh.write("FID\t" + "\t".join(snp_id) + "\n")
        for f in range(n_families):
            vals = "\t".join(str(int(v)) for v in sim["parent_geno"][f])
            fh.write(f"F{f:04d}\t{vals}\n")

    with open(output_dir / "family.tsv", "w") as fh:
        fh.write("IID\tFID\tsib_idx\n")
        for k in range(n_sibs):
            fh.write(f"{iid[k]}\t{fid[k]}\t{int(sib_idx[k])}\n")

    with open(output_dir / "truth.tsv", "w") as fh:
        fh.write("SNP\tallele_freq\tbeta_d_true\tbeta_i_true\tis_causal\n")
        for j, sid in enumerate(snp_id):
            is_causal = (sim["beta_d_arr"][j] != 0.0) or (sim["beta_i_arr"][j] != 0.0)
            fh.write(
                f"{sid}\t{sim['allele_freq'][j]:.6f}\t"
                f"{sim['beta_d_arr'][j]:.6f}\t{sim['beta_i_arr'][j]:.6f}\t"
                f"{int(is_causal)}\n"
            )

    with open(output_dir / "meta.json", "w") as fh:
        json.dump(sim["meta"], fh, indent=2, default=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-families", type=int, default=500)
    ap.add_argument("--sibs-per-family", type=int, default=2)
    ap.add_argument("--n-snps", type=int, default=300)
    ap.add_argument("--n-causal", type=int, default=50)
    ap.add_argument("--beta-d", type=float, default=0.20)
    ap.add_argument("--alpha", type=float, default=0.50,
                    help="alpha = beta_i / beta_d (paper's ratio)")
    ap.add_argument("--h2", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    sim = simulate(
        n_families=args.n_families,
        sibs_per_family=args.sibs_per_family,
        n_snps=args.n_snps,
        n_causal=args.n_causal,
        beta_d=args.beta_d,
        alpha=args.alpha,
        h2=args.h2,
        seed=args.seed,
    )
    write_outputs(sim, args.output_dir)
    print(f"[simulate_sibpair] wrote {args.output_dir} "
          f"({sim['sib_geno'].shape[0]} sibs x {sim['sib_geno'].shape[1]} SNPs)")
    print(f"[simulate_sibpair] truth: beta_d={args.beta_d}, "
          f"alpha={args.alpha}, n_causal={args.n_causal}")
    print(f"[simulate_sibpair] h2 target={args.h2}, observed={sim['meta']['h2_observed_var']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
