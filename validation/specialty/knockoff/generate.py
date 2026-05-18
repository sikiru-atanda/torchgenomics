"""Generate replicate genotype + phenotype fixtures for the knockoff harness.

Block-LD simulation:
  - n_blocks blocks of block_size SNPs each
  - within-block correlation rho via a shared latent variable
  - genotypes discretized into {0, 1, 2} (diploid additive)
  - k_causal SNPs across distinct blocks get effect = effect
  - y = X beta + N(0, noise_sd^2)

Outputs (in OUTPUT_DIR):
  - replicates.npz   - {G_rep_<r>, y_rep_<r>} for r in 0..n_replicates-1
  - sim_truth.json   - causal SNP indices, fixture parameters, seed

References:
  Candes, Fan, Janson & Lv 2018 JRSSB - Model-X knockoffs (sec. 5 sims).
  Sesia, Sabatti & Candes 2020 JASA - KnockoffGWAS (sec. 4 sims).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def simulate_replicate(
    n,
    n_blocks,
    block_size,
    causal_idx,
    effect,
    noise_sd,
    within_block_rho,
    rng,
):
    p = n_blocks * block_size
    G = np.zeros((n, p), dtype=np.float64)
    for b in range(n_blocks):
        latent = rng.standard_normal(n)
        for j in range(block_size):
            noise = rng.standard_normal(n)
            raw = within_block_rho * latent + (1.0 - within_block_rho) * noise
            # Discretize to {0, 1, 2} via two threshold cuts at +-0.5 sd.
            G[:, b * block_size + j] = (raw > 0.5).astype(np.float64) + (raw > -0.5).astype(np.float64)

    beta = np.zeros(p, dtype=np.float64)
    for j in causal_idx:
        beta[j] = effect

    G_centered = G - G.mean(axis=0, keepdims=True)
    sd = G.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    G_std = G_centered / sd
    y = G_std @ beta + noise_sd * rng.standard_normal(n)

    return G, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-replicates", type=int, default=100)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--p", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=10)
    ap.add_argument("--k-causal", type=int, default=8)
    ap.add_argument("--effect", type=float, default=1.0)
    ap.add_argument("--noise-sd", type=float, default=0.3)
    ap.add_argument("--within-block-rho", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = args.p
    n_blocks = p // args.block_size
    if n_blocks * args.block_size != p:
        raise SystemExit(f"p={p} must equal n_blocks*block_size={n_blocks*args.block_size}")
    if args.k_causal > n_blocks:
        raise SystemExit(f"k_causal={args.k_causal} cannot exceed n_blocks={n_blocks}")

    # One causal SNP per chosen block, evenly spaced across the first k_causal blocks.
    block_stride = max(1, n_blocks // args.k_causal)
    causal_idx = sorted([b * args.block_size for b in range(0, args.k_causal * block_stride, block_stride)][: args.k_causal])

    rng = np.random.default_rng(args.seed)

    out_data = {}
    for r in range(args.n_replicates):
        sub_rng = np.random.default_rng(rng.integers(low=0, high=2**31 - 1))
        G_r, y_r = simulate_replicate(
            n=args.n,
            n_blocks=n_blocks,
            block_size=args.block_size,
            causal_idx=causal_idx,
            effect=args.effect,
            noise_sd=args.noise_sd,
            within_block_rho=args.within_block_rho,
            rng=sub_rng,
        )
        out_data[f"G_rep_{r}"] = G_r
        out_data[f"y_rep_{r}"] = y_r

    np.savez_compressed(out_dir / "replicates.npz", **out_data)
    truth = {
        "n_replicates": args.n_replicates,
        "n": args.n,
        "p": p,
        "n_blocks": n_blocks,
        "block_size": args.block_size,
        "k_causal": args.k_causal,
        "effect": args.effect,
        "noise_sd": args.noise_sd,
        "within_block_rho": args.within_block_rho,
        "seed": args.seed,
        "causal_idx_zero_based": causal_idx,
        "causal_block_idx_zero_based": [j // args.block_size for j in causal_idx],
    }
    (out_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"[generate] wrote replicates.npz ({len(out_data)//2} replicates) and sim_truth.json (k_causal={args.k_causal})")


if __name__ == "__main__":
    main()
