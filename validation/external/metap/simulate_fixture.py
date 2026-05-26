"""Simulate a p-value combination fixture for the metap harness.

Writes:
  data/
    pvalues.tsv         m × k table of p-values (header: p1..pk).
    data_matrix.tsv     n × k table of underlying observations
                        (only used by Empirical Brown — Spearman r).
    sim_truth.json      seed, m, k, scenario tags.

The fixture deliberately exercises three regimes per row:
  - one row of all-uniform-null p-values (mid-significant);
  - one row with one extreme p (drives Fisher / HMP / min-p sharply);
  - one row of moderately-correlated p-values (tests Brown / EBM).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def simulate(output_dir: Path, seed: int = 42, k: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- p-value rows -----------------------------------------------------
    rows = {
        "null_uniform":  np.array([0.50, 0.50, 0.50, 0.50, 0.50]),
        "one_extreme":   np.array([1e-8, 0.50, 0.50, 0.50, 0.50]),
        "all_below_tau": np.array([1e-3, 5e-4, 2e-3, 8e-4, 4e-3]),
        "mixed":         np.array([0.01, 0.20, 0.05, 0.30, 0.001]),
        "all_above_tau": np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
    }
    assert all(len(v) == k for v in rows.values()), \
        f"each row must have k={k} entries"
    p_matrix = np.stack(list(rows.values()), axis=0)
    p_path = output_dir / "pvalues.tsv"
    header = "scenario\t" + "\t".join(f"p{i+1}" for i in range(k))
    with open(p_path, "w") as f:
        f.write(header + "\n")
        for scen, p in rows.items():
            f.write(scen + "\t" + "\t".join(f"{x:.10g}" for x in p) + "\n")

    # ---- data matrix (n samples, k variables) for EBM ---------------------
    # Mildly-correlated columns so EBM has a non-trivial covariance to fit.
    n = 200
    base = rng.standard_normal((n, 1))
    noise = rng.standard_normal((n, k)) * 0.7
    data = 0.3 * base + 0.7 * noise  # correlation ≈ 0.18 between any two cols
    d_path = output_dir / "data_matrix.tsv"
    with open(d_path, "w") as f:
        f.write("\t".join(f"x{i+1}" for i in range(k)) + "\n")
        for i in range(n):
            f.write("\t".join(f"{v:.10g}" for v in data[i]) + "\n")

    truth = {
        "seed": int(seed),
        "k": int(k),
        "n_samples": int(n),
        "scenarios": list(rows.keys()),
        "fixture_sha256": {
            "pvalues.tsv": _sha256(p_path),
            "data_matrix.tsv": _sha256(d_path),
        },
    }
    (output_dir / "sim_truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    truth = simulate(Path(args.output_dir), seed=args.seed)
    print(f"[metap fixture] wrote {args.output_dir}")
    print(f"  k={truth['k']} n={truth['n_samples']} "
          f"scenarios={truth['scenarios']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
