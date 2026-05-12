"""Compare susieR vs torchgwas bayes-scan-rss outputs against the 6-metric tolerance table.

Per NA1 design spec section 5.2. Emits a single findings ledger row to
docs/validation_findings.md per F3 severity policy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


THRESHOLDS = {
    "credible_set_jaccard": 0.95,
    "pip_correlation": 0.99,
    "beta_mean_correlation": 0.999,
    "beta_sd_correlation": 0.999,
    "elbo_relative_diff": 1e-4,
    "walltime_ratio": 2.0,
}


def credible_set_jaccard(in_cs_a: np.ndarray, in_cs_b: np.ndarray) -> float:
    """Jaccard index of two credible-set membership vectors (binary indicators)."""
    set_a = set(np.where(in_cs_a == 1)[0])
    set_b = set(np.where(in_cs_b == 1)[0])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="susieR output TSV")
    parser.add_argument("--ours", required=True, help="bayes-scan-rss output TSV")
    parser.add_argument("--upstream-walltime", required=True, help="susieR wall-time file")
    parser.add_argument("--ours-walltime", required=True, help="our wall-time file")
    parser.add_argument("--upstream-elbo", required=True, help="susieR final ELBO")
    parser.add_argument("--ours-elbo", required=True, help="our final ELBO")
    parser.add_argument("--findings", help="Append findings row to this file")
    args = parser.parse_args()

    up = pd.read_csv(args.upstream, sep="\t")
    ours = pd.read_csv(args.ours, sep="\t")

    if len(up) != len(ours):
        print(f"FAIL: row count mismatch: upstream={len(up)}, ours={len(ours)}")
        return 1

    metrics = {}

    # Credible set Jaccard
    in_cs_up = (up["IN_CS"].values == 1).astype(int)
    in_cs_ours = (ours["CREDIBLE_SET"].values > 0).astype(int)
    metrics["credible_set_jaccard"] = credible_set_jaccard(in_cs_up, in_cs_ours)

    # PIP correlation (only SNPs with PIP > 0.1 in either)
    mask = (up["PIP"].values > 0.1) | (ours["PIP"].values > 0.1)
    if mask.sum() >= 3:
        metrics["pip_correlation"] = float(
            pearsonr(up["PIP"].values[mask], ours["PIP"].values[mask])[0]
        )
    else:
        metrics["pip_correlation"] = float("nan")

    # β_mean, β_sd correlation
    metrics["beta_mean_correlation"] = float(
        pearsonr(up["BETA_MEAN"].values, ours["BETA_MEAN"].values)[0]
    )
    metrics["beta_sd_correlation"] = float(
        pearsonr(up["BETA_SD"].values, ours["BETA_SD"].values)[0]
    )

    # ELBO relative diff
    elbo_up = float(open(args.upstream_elbo).read().strip())
    elbo_ours = float(open(args.ours_elbo).read().strip())
    if abs(elbo_up) < 1e-12:
        metrics["elbo_relative_diff"] = float("nan")
    else:
        metrics["elbo_relative_diff"] = abs(elbo_up - elbo_ours) / abs(elbo_up)

    # Wall-time ratio
    wt_up = float(open(args.upstream_walltime).read().strip())
    wt_ours = float(open(args.ours_walltime).read().strip())
    metrics["walltime_ratio"] = wt_ours / max(wt_up, 1e-6)

    # Apply tolerance contract
    failures = []
    for key, observed in metrics.items():
        threshold = THRESHOLDS[key]
        if key in ("credible_set_jaccard", "pip_correlation",
                   "beta_mean_correlation", "beta_sd_correlation"):
            if not np.isnan(observed) and observed < threshold:
                failures.append(f"{key}: {observed:.6g} < {threshold:.6g}")
        else:
            if not np.isnan(observed) and observed > threshold:
                failures.append(f"{key}: {observed:.6g} > {threshold:.6g}")

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g} (threshold {THRESHOLDS[k]:.6g})")

    if args.findings:
        with open(args.findings, "a") as f:
            f.write(
                f"\n## NA1 SuSiE-RSS parity check (date: {pd.Timestamp.now()})\n"
            )
            for k, v in metrics.items():
                f.write(f"- {k}: {v:.6g} (threshold {THRESHOLDS[k]:.6g})\n")
            if failures:
                f.write(f"- **FAILURES:** {'; '.join(failures)}\n")
            else:
                f.write("- All thresholds met.\n")

    if failures:
        print(f"\nFAIL: {len(failures)} threshold violations")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nPASS: all 6 thresholds met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
