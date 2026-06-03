#!/usr/bin/env python3
"""Simulate a balanced longitudinal random-regression fixture (seed=42).

Design
------
- n = 300 independent subjects (K = I_n; no kinship by construction
  so the comparison with lme4 stays in the same model class).
- T = 5 time points per subject (balanced longitudinal panel).
- b = 3 normalized Legendre basis coefficients (order = 2).
- m = 200 candidate SNPs drawn iid from Binomial(2, p_MAF=0.3).
- 1 planted causal SNP whose effect is on the linear-trend basis
  coefficient (beta_slope = 0.40). Per-time-point effect beta(t)
  = phi(t) dot beta_j varies in a known way.
- True random-regression covariance K_coef_true = diag(1.0, 0.5, 0.2).
- Residual variance per observation: sigma2_e = 0.6.

Outputs (validation/specialty/rr/data/):
- long_pheno.tsv: long-format Y, subject_id, t (raw 0..100 grid).
- geno.tsv:       wide standardized genotype dosage (n x m).
- geno_raw.tsv:   wide 0/1/2 dosage (n x m).
- truth.json:     seed + planted params + per-file SHA256 list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


SEED = 42
N = 300
T = 5
B = 3      # Legendre order = B - 1 = 2
M = 200    # SNPs
CAUSAL_INDEX = 17
BETA_SLOPE = 0.40
P_MAF = 0.30
K_DIAG_TRUE = (1.0, 0.5, 0.2)
SIGMA2_E = 0.6
T_MIN = 0.0
T_MAX = 100.0

def _legendre_normalized(t_std: np.ndarray, order: int) -> np.ndarray:
    """Bonnet three-term recurrence, normalized so integral over [-1, 1]
    of P_k^2 = 1. Matches torchgenomics.linalg.basis.legendre_basis exactly."""
    Tn = t_std.shape[0]
    b = order + 1
    P = np.zeros((Tn, b), dtype=np.float64)
    P[:, 0] = 1.0
    if order >= 1:
        P[:, 1] = t_std
    for k in range(1, order):
        P[:, k + 1] = ((2 * k + 1) * t_std * P[:, k] - k * P[:, k - 1]) / (k + 1)
    norms = np.array([np.sqrt((2 * k + 1) / 2.0) for k in range(b)],
                     dtype=np.float64)
    return P * norms[None, :]


def simulate(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    times = np.linspace(T_MIN, T_MAX, T)
    t_std = 2.0 * (times - T_MIN) / (T_MAX - T_MIN) - 1.0
    Phi = _legendre_normalized(t_std, order=B - 1)  # (T, b)

    # Genotypes from Binomial(2, P_MAF); standardize per SNP.
    G_raw = rng.binomial(2, P_MAF, size=(N, M)).astype(np.float64)
    G = (G_raw - G_raw.mean(0)) / G_raw.std(0).clip(min=1e-8)

    # Per-subject random regression coefficients u_i ~ N(0, K_coef_true)
    K_diag = np.array(K_DIAG_TRUE, dtype=np.float64)
    U = rng.normal(size=(N, B)) * np.sqrt(K_diag)[None, :]

    # Population mean effects
    beta_pop = np.array([5.0, 0.10, -0.05], dtype=np.float64)

    # Planted causal SNP effect: slope-only.
    beta_planted = np.zeros(B, dtype=np.float64)
    beta_planted[1] = BETA_SLOPE

    A_total = (
        beta_pop[None, :]
        + G[:, CAUSAL_INDEX:CAUSAL_INDEX + 1] * beta_planted[None, :]
        + U
    )

    # Observation matrix in obs-space
    Y_mat = A_total @ Phi.T
    Y_mat = Y_mat + rng.normal(size=(N, T)) * np.sqrt(SIGMA2_E)

    # Long-format phenotype
    rows_long = ["sid\tt\ty"]
    for i in range(N):
        for k in range(T):
            rows_long.append(
                f"{i + 1}\t{times[k]:.6f}\t{Y_mat[i, k]:.10g}"
            )
    (out_dir / "long_pheno.tsv").write_text("\n".join(rows_long) + "\n")

    # Wide-format standardized genotypes
    snp_names = [f"rs{j:04d}" for j in range(M)]
    rows_g = ["sid\t" + "\t".join(snp_names)]
    for i in range(N):
        rows_g.append(
            str(i + 1) + "\t" + "\t".join(f"{G[i, j]:.10g}" for j in range(M))
        )
    (out_dir / "geno.tsv").write_text("\n".join(rows_g) + "\n")

    # Raw 0/1/2 dosages
    rows_g_raw = ["sid\t" + "\t".join(snp_names)]
    for i in range(N):
        rows_g_raw.append(
            str(i + 1) + "\t" + "\t".join(str(int(G_raw[i, j])) for j in range(M))
        )
    (out_dir / "geno_raw.tsv").write_text("\n".join(rows_g_raw) + "\n")

    truth = {
        "seed": SEED,
        "n_subjects": N,
        "n_timepoints": T,
        "n_basis": B,
        "basis_kind": "legendre",
        "basis_order": B - 1,
        "n_snps": M,
        "causal_snp_index": CAUSAL_INDEX,
        "causal_snp_name": snp_names[CAUSAL_INDEX],
        "beta_planted_per_basis": beta_planted.tolist(),
        "beta_population": beta_pop.tolist(),
        "K_coef_true_diag": list(K_DIAG_TRUE),
        "sigma2_e_true": SIGMA2_E,
        "p_maf": P_MAF,
        "t_min": T_MIN,
        "t_max": T_MAX,
        "time_grid": times.tolist(),
        "snp_names": snp_names,
    }
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))

    def _h(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    sums = {
        name: _h(out_dir / name)
        for name in ("long_pheno.tsv", "geno.tsv", "geno_raw.tsv", "truth.json")
    }
    (out_dir / "fixture.sha256").write_text(
        "\n".join(f"{v}  {k}" for k, v in sums.items()) + "\n"
    )
    return sums


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "data"),
    )
    args = ap.parse_args()
    sums = simulate(Path(args.out_dir))
    print("[simulate_rr_fixture] wrote", args.out_dir)
    for k, v in sums.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
