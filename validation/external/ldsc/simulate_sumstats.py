"""Simulate two correlated GWAS sumstats from real 1000G LD scores.

We use the closed-form LDSC mean-chi² model to produce chi² statistics whose
expected ldsc_h2 estimate is the simulated h² (within block-jackknife SE).

For each SNP j with LD score l_j and per-trait sample size N_t and total
genome-wide SNP count M:
    E[chi²_t,j] = (N_t / M) * h²_t * l_j + 1 + N_t * a_t

where a_t is the LDSC intercept term (we set a_t = 0, so intercept = 1).

We generate signed z-scores (so we can compute proper genetic correlation):
    z_t,j = sqrt(E[chi²_t,j]) * sign + N(0, 1) noise

A 2D correlated z generator emits z_1, z_2 with target genetic correlation rg.
For a clean signal, we draw the underlying per-SNP genetic effects β_t,j ∼
N(0, h²_t * l_j / M) and correlate β_1 to β_2 by rg, then add per-trait
sampling noise: z_t,j = sqrt(N_t) * β_t,j + ε_t,j where ε ~ N(0, 1).

This is the standard simulation in Bulik-Sullivan §Methods. The resulting
sumstats files mimic real GWAS chi² distributions; LDSC's own h2 + rg
estimators recover the simulated truth, so they're a fair reference point
for TorchGWAS' implementations.

Outputs (per --output-dir):
    sim_trait1.sumstats.gz   LDSC-format sumstats for trait 1
    sim_trait2.sumstats.gz   LDSC-format sumstats for trait 2
    sim_truth.json           {h2_1, h2_2, rg, n1, n2, m_total, ...}

Usage:
    python3 simulate_sumstats.py \\
        --ldscores-dir data/ld_scores \\
        --weights-dir  data/weights \\
        --output-dir   data/ \\
        --chrom 22 --h2 0.4 --rg 0.5 \\
        --n1 100000 --n2 80000 --seed 42
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _read_ld_score_chr(ldscores_dir: Path, chrom: int) -> pd.DataFrame:
    f = ldscores_dir / f"LDscore.{chrom}.l2.ldscore.gz"
    if not f.exists():
        raise FileNotFoundError(f"missing LD score file: {f}")
    df = pd.read_csv(f, sep="\t", compression="gzip")
    # Columns: CHR, SNP, BP, L2
    return df


def _read_weights_chr(weights_dir: Path, chrom: int) -> pd.DataFrame:
    f = weights_dir / f"weights.hm3_noMHC.{chrom}.l2.ldscore.gz"
    if not f.exists():
        raise FileNotFoundError(f"missing weights file: {f}")
    df = pd.read_csv(f, sep="\t", compression="gzip")
    return df


def _read_M(ldscores_dir: Path, chrom: int) -> int:
    """Read the .l2.M_5_50 file (M with MAF 5-50% — the LDSC default)."""
    f = ldscores_dir / f"LDscore.{chrom}.l2.M_5_50"
    return int(open(f).read().strip().split()[0])


def simulate(
    ldscores_dir: Path,
    weights_dir: Path,
    output_dir: Path,
    chrom: int,
    h2: float,
    rg: float,
    n1: int,
    n2: int,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    # Load real LD scores for this chromosome
    ldscores_df = _read_ld_score_chr(ldscores_dir, chrom)
    weights_df = _read_weights_chr(weights_dir, chrom)
    m_chr = _read_M(ldscores_dir, chrom)

    # Restrict to SNPs in both LD scores and weights (this is what LDSC does internally)
    snps_in_both = set(ldscores_df["SNP"]) & set(weights_df["SNP"])
    df = ldscores_df[ldscores_df["SNP"].isin(snps_in_both)].copy()
    df = df.reset_index(drop=True)

    m = len(df)
    print(f"[simulate] chr {chrom}: m={m} SNPs in LD scores ∩ weights", file=sys.stderr)
    print(f"[simulate] M_5_50 (genome-wide common SNP count): {m_chr}", file=sys.stderr)

    # Use M = m as the total SNP count for the simulation (single-chr scope).
    # LDSC will see M from its own .M_5_50 files; we report `m_total` as the
    # value we used for the truth simulation. The harness compares LDSC's
    # estimate (which uses m_chr) to TorchGWAS' estimate (which we feed
    # m_chr as well) — both inherit the same M.
    M = m_chr

    ld_scores = df["L2"].to_numpy(dtype=np.float64)

    # ── Simulate true per-SNP effects ──────────────────────────────────────
    # β_1 ~ N(0, h2 / M)        per-SNP variance under infinitesimal model
    # β_2 = rg * β_1 + sqrt(1 - rg^2) * N(0, h2 / M)  (correlated trait)
    sigma2_beta = h2 / M
    beta1 = rng.normal(0.0, np.sqrt(sigma2_beta), size=m)
    beta2 = rg * beta1 + np.sqrt(max(1 - rg**2, 0.0)) * rng.normal(0.0, np.sqrt(sigma2_beta), size=m)

    # Z-score under the LDSC mean-chi² model:
    #   E[Z_t,j^2 | β] = N_t * β_t,j^2 * l_j + 1
    # The actual Z is approximately N(sqrt(N_t) * β_t,j * sqrt(l_j), 1).
    # We use sqrt(l_j) as the per-SNP variance scale (this is the standard
    # LDSC simulation; see Bulik-Sullivan §Supp Note 7).
    z1 = np.sqrt(n1) * beta1 * np.sqrt(np.clip(ld_scores, 1.0, None)) \
        + rng.normal(0.0, 1.0, size=m)
    z2 = np.sqrt(n2) * beta2 * np.sqrt(np.clip(ld_scores, 1.0, None)) \
        + rng.normal(0.0, 1.0, size=m)

    # Allele assignments (random; LDSC uses these for sign harmonization in --rg).
    # We use A/G as a deterministic A1/A2 to keep the harness reproducible.
    a1 = np.full(m, "A")
    a2 = np.full(m, "G")

    # ── Write LDSC-format sumstats ─────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write_sumstats(z: np.ndarray, n: int, path: Path) -> None:
        out = pd.DataFrame({
            "SNP": df["SNP"].to_numpy(),
            "A1": a1,
            "A2": a2,
            "Z": z,
            "N": np.full(m, n, dtype=np.int64),
        })
        with gzip.open(path, "wt") as fh:
            out.to_csv(fh, sep="\t", index=False)
        print(f"[simulate] wrote {path}: {m} SNPs, mean chi² = {(z**2).mean():.3f}", file=sys.stderr)

    _write_sumstats(z1, n1, output_dir / "sim_trait1.sumstats.gz")
    _write_sumstats(z2, n2, output_dir / "sim_trait2.sumstats.gz")

    # ── Truth manifest ─────────────────────────────────────────────────────
    truth = {
        "h2_1": h2,
        "h2_2": h2,
        "rg": rg,
        "n1": int(n1),
        "n2": int(n2),
        "m_total": int(M),
        "m_used": int(m),
        "chrom": int(chrom),
        "seed": int(seed),
        "intercept_simulated": 1.0,
        "mean_chi2_trait1": float((z1**2).mean()),
        "mean_chi2_trait2": float((z2**2).mean()),
    }
    truth_path = output_dir / "sim_truth.json"
    truth_path.write_text(json.dumps(truth, indent=2))
    print(f"[simulate] truth manifest → {truth_path}", file=sys.stderr)
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ldscores-dir", required=True, type=Path)
    ap.add_argument("--weights-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--chrom", type=int, default=22)
    ap.add_argument("--h2", type=float, default=0.4)
    ap.add_argument("--rg", type=float, default=0.5)
    ap.add_argument("--n1", type=int, default=100000)
    ap.add_argument("--n2", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    simulate(
        ldscores_dir=args.ldscores_dir,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        chrom=args.chrom,
        h2=args.h2,
        rg=args.rg,
        n1=args.n1,
        n2=args.n2,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
