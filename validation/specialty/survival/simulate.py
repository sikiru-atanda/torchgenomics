"""Simulate Cox PH frailty fixture for survival GWAS reference comparison.

Fixture design (Tier 3 C1, n=500):

- Population: n=500 unrelated synthetic individuals.
- Genotypes: m=20 test SNPs + m_grm=200 polygenic-background SNPs (HWE,
  MAF ~ Uniform[0.10, 0.40]) drawn diploid (ploidy=2).
- Kinship K: VanRaden-style empirical GRM from the m_grm background SNPs,
  diagonal regularized by +0.01 I to ensure positive-definiteness.
- Frailty: u ~ N(0, h2*K) with h2 = 0.30 (Therneau 2003 frailty variance).
- Causal SNPs: indices 0, 1, 2 in G_test, each with log_hr = 0.6.
- Hazard: lambda_i(t) = lambda0(t) * exp(u_i + sum_k beta_k * g_std_{i,k})
  with Weibull baseline (shape = 1.5, scale = 1).
- Times: T_event_i = (-log U_i / exp(eta_i))^(1/shape).
- Censoring: exponential calibrated to ~15% censoring rate.
- Seed = 42 throughout for deterministic output across hosts.

Outputs: pheno.csv, geno.csv, kinship.csv, truth.json under <out_dir>.

References
----------
- Therneau & Grambsch (2000). Modeling Survival Data. Springer.
- Therneau, Grambsch & Pankratz (2003). Penalized survival models / frailty.
- Bi et al. (2020). SPACox. Nat Commun 11:1626.
- Breslow & Day (1980). Statistical Methods in Cancer Research, Vol 1.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def simulate(
    n: int = 500,
    m: int = 20,
    m_grm: int = 200,
    n_causal: int = 3,
    log_hr: float = 0.6,
    h2: float = 0.30,
    censor_rate: float = 0.15,
    weibull_shape: float = 1.5,
    seed: int = 42,
    ploidy: int = 2,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GRM genotypes (separate from test SNPs)
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.10 + 0.30 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.binomial(ploidy, maf, size=n),
            dtype=torch.float64,
        )
    G_grm_centered = G_grm - G_grm.mean(dim=0)
    sd_grm = G_grm_centered.std(dim=0).clamp(min=1e-6)
    G_grm_std = G_grm_centered / sd_grm
    K = (G_grm_std @ G_grm_std.T) / m_grm
    K = 0.5 * (K + K.T) + 0.01 * torch.eye(n, dtype=torch.float64)

    # Test genotypes
    G_test = torch.zeros(n, m, dtype=torch.float64)
    for j in range(m):
        maf = 0.10 + 0.30 * np.random.rand()
        G_test[:, j] = torch.tensor(
            np.random.binomial(ploidy, maf, size=n),
            dtype=torch.float64,
        )

    # Polygenic frailty u ~ N(0, h2 * K)
    L = torch.linalg.cholesky(K + 1e-6 * torch.eye(n, dtype=torch.float64))
    u = (L @ torch.randn(n, dtype=torch.float64)) * math.sqrt(h2)

    # Causal effects on standardized genotype
    causal_idx = list(range(n_causal))
    eta = u.clone()
    for idx in causal_idx:
        g_j = G_test[:, idx]
        g_std = (g_j - g_j.mean()) / g_j.std().clamp(min=1e-6)
        eta = eta + log_hr * g_std

    # Weibull event times
    U = torch.rand(n, dtype=torch.float64).clamp(min=1e-10, max=1.0 - 1e-10)
    T_event = ((-torch.log(U)) / torch.exp(eta)).pow(1.0 / weibull_shape)

    # Exponential censoring calibrated to target rate
    if censor_rate > 0:
        median_T = float(T_event.median().item())
        lambda_c = -math.log(max(1.0 - censor_rate, 1e-6)) / max(median_T, 1e-10)
        C = torch.distributions.Exponential(lambda_c).sample((n,)).to(torch.float64)
        time = torch.minimum(T_event, C)
        event = (T_event <= C).to(torch.float64)
    else:
        time = T_event
        event = torch.ones(n, dtype=torch.float64)

    observed_censor_rate = float(1.0 - event.mean().item())
    truth = dict(
        n=n, m=m, m_grm=m_grm, ploidy=ploidy,
        n_causal=n_causal, causal_idx=causal_idx,
        log_hr_true=log_hr,
        h2_frailty=h2,
        censor_rate_target=censor_rate,
        censor_rate_observed=observed_censor_rate,
        event_count=int(event.sum().item()),
        weibull_shape=weibull_shape,
        seed=seed,
    )

    return dict(
        G_test=G_test, G_grm=G_grm, K=K,
        time=time, event=event,
        truth=truth,
    )


def write_csvs(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = data["time"].shape[0]
    m = data["G_test"].shape[1]

    time_arr = data["time"]
    event_arr = data["event"]
    pheno_path = out_dir / "pheno.csv"
    with pheno_path.open("w") as f:
        f.write("id,time,event\n")
        for i in range(n):
            t_i = float(time_arr[i].item())
            e_i = int(event_arr[i].item())
            f.write(f"{i},{t_i:.10f},{e_i}\n")

    G_test = data["G_test"]
    geno_path = out_dir / "geno.csv"
    with geno_path.open("w") as f:
        header = "id," + ",".join(f"snp{j}" for j in range(m))
        f.write(header + "\n")
        for i in range(n):
            row = ",".join(str(int(G_test[i, j].item())) for j in range(m))
            f.write(f"{i},{row}\n")

    K_np = data["K"].numpy()
    np.savetxt(out_dir / "kinship.csv", K_np, delimiter=",", fmt="%.10f")

    (out_dir / "truth.json").write_text(json.dumps(data["truth"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--m-grm", type=int, default=200)
    ap.add_argument("--n-causal", type=int, default=3)
    ap.add_argument("--log-hr", type=float, default=0.6)
    ap.add_argument("--h2", type=float, default=0.30)
    ap.add_argument("--censor-rate", type=float, default=0.15)
    ap.add_argument("--weibull-shape", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = simulate(
        n=args.n, m=args.m, m_grm=args.m_grm,
        n_causal=args.n_causal, log_hr=args.log_hr,
        h2=args.h2, censor_rate=args.censor_rate,
        weibull_shape=args.weibull_shape, seed=args.seed,
    )
    write_csvs(data, args.out_dir)
    t = data["truth"]
    print(f"[simulate] wrote {args.out_dir}/{{pheno.csv,geno.csv,kinship.csv,truth.json}}")
    n_val = t["n"]
    ec = t["event_count"]
    cr = t["censor_rate_observed"]
    print(f"[simulate] n={n_val}  events={ec}  censor_obs={cr:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
