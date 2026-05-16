#!/usr/bin/env python3
"""Generate threshold fixture (seed=42)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from numpy.random import default_rng

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--m", type=int, default=50)
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = default_rng(args.seed)
    n, m = int(args.n), int(args.m)
    c = 3
    R = np.array([[1.00,0.30,0.15],[0.30,1.00,0.10],[0.15,0.10,1.00]], dtype=np.float64)
    G_cov = np.array([[0.50,0.10,0.05],[0.10,0.40,0.05],[0.05,0.05,0.30]], dtype=np.float64)
    b_intercept = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b_sex = np.array([0.20, -0.15, 0.30], dtype=np.float64)
    af = rng.uniform(0.1, 0.5, size=m)
    G = rng.binomial(2, af[None, :], size=(n, m)).astype(np.float64)
    Gc = G - 2.0 * af[None, :]
    Gs = Gc / np.sqrt(2.0 * af * (1.0 - af) + 1e-12)[None, :]
    L_g = np.linalg.cholesky(G_cov)
    u = rng.standard_normal((n, c)) @ L_g.T
    causal_idx = rng.choice(m, size=5, replace=False)
    causal_beta = rng.standard_normal((5, c)) * 0.08
    geno_anchor = Gs[:, causal_idx] @ causal_beta
    L_r = np.linalg.cholesky(R)
    eps = rng.standard_normal((n, c)) @ L_r.T
    sex = rng.integers(1, 3, size=n).astype(int)
    sex_centred = (sex.astype(np.float64) - 1.5)
    L = (b_intercept[None, :] + sex_centred[:, None] * b_sex[None, :] + u + geno_anchor + eps)
    tau1 = np.array([0.0])
    tau2 = np.array([-0.5, 0.5])
    cat1 = (L[:, 0] > tau1[0]).astype(int)
    cat2 = ((L[:, 1] > tau2[0]).astype(int) + (L[:, 1] > tau2[1]).astype(int))
    y3 = L[:, 2].copy()
    ids = np.arange(1, n + 1)
    blupf_rows = []
    for i in range(n):
        blupf_rows.append(f"{ids[i]:>6d} 1 {int(sex[i])} {int(cat1[i]) + 1} {int(cat2[i]) + 1} {y3[i]:.6f}")
    (out_dir / "pheno.txt").write_text("\n".join(blupf_rows) + "\n")
    geno_lines = []
    for i in range(n):
        gline = " ".join(str(int(v)) for v in G[i, :])
        geno_lines.append(f"{ids[i]:>6d} {gline}")
    (out_dir / "geno.012").write_text("\n".join(geno_lines) + "\n")
    map_rows = ["snp\tchr\tpos\ta1\ta2\tfreq"]
    for j in range(m):
        map_rows.append(f"SNP{j+1:04d}\t1\t{j+1}\tA\tG\t{af[j]:.4f}")
    (out_dir / "marker.map").write_text("\n".join(map_rows) + "\n")
    pedi_rows = []
    for i in range(n):
        pedi_rows.append(f"{ids[i]:>6d} 0 0")
    (out_dir / "pedigree.dat").write_text("\n".join(pedi_rows) + "\n")
    X0 = np.column_stack([np.ones(n, dtype=np.float64), (sex - 1).astype(np.float64)])
    np.savetxt(out_dir / "X0.tsv", X0, delimiter="\t", fmt="%.6f", header="intercept\tsex", comments="")
    Y_tg = np.column_stack([cat1.astype(float), cat2.astype(float), y3])
    np.savetxt(out_dir / "Y_tg.tsv", Y_tg, delimiter="\t", fmt="%.6f", header="y1_ord\ty2_ord\ty3_cont", comments="")
    cat_t1 = [int((cat1 == 0).sum()), int((cat1 == 1).sum())]
    cat_t2 = [int((cat2 == 0).sum()), int((cat2 == 1).sum()), int((cat2 == 2).sum())]
    truth = {
        "seed": int(args.seed), "n": int(n), "m": int(m), "c": int(c),
        "trait_types": ["ordinal", "ordinal", "continuous"],
        "n_categories": [2, 3, 0],
        "thresholds": {"t1": tau1.tolist(), "t2": tau2.tolist()},
        "R": R.tolist(), "G_cov": G_cov.tolist(),
        "b_intercept": b_intercept.tolist(), "b_sex": b_sex.tolist(),
        "causal_snps_one_based": (causal_idx + 1).tolist(),
        "causal_beta": causal_beta.tolist(),
        "af_first5": af[:5].tolist(),
        "category_counts": {"t1": cat_t1, "t2": cat_t2},
    }
    (out_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"[generate] wrote n={n} m={m} c={c} to {out_dir}")
    print(f"[generate]   t1 counts: {cat_t1}")
    print(f"[generate]   t2 counts: {cat_t2}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

