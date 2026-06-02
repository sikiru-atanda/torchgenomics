"""Haplotype-based GWAS on sliding windows of SNPs.

Demonstrates: HaplotypeGWAS in moving-window mode with EM phasing (unphased
diploid genotypes), F-test omnibus across haplotype dosages.

Phases covered: 46 (haplotype GWAS: HTR / block / window / SKAT).

Runtime: < 20 s on CPU (n=200, m=60, window=5, step=2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.models import HaplotypeGWAS

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    torch.manual_seed(42)
    n, m = 200, 60
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE)

    # Plant a causal window: SNPs 20..24 jointly drive Y.
    causal_window = [20, 21, 22, 23, 24]
    beta = torch.tensor([0.45, 0.55, -0.4, 0.55, 0.45], dtype=STAT_DTYPE)
    Y = (G[:, causal_window] @ beta
         + torch.randn(n, dtype=STAT_DTYPE) * 0.5)

    variant_pos = list(range(1000, 1000 + m * 1000, 1000))
    variant_chr = ["1"] * m

    print(f"Simulated: n={n}, m={m}, causal window at SNPs {causal_window}")

    scanner = HaplotypeGWAS(
        method="window",
        test="f_test",
        window_size=5,
        step=2,
        min_hap_freq=0.02,
        max_haplotypes=10,
    )
    res = scanner.scan(Y, G, variant_pos=variant_pos, variant_chr=variant_chr)

    df = pd.DataFrame({
        "block_id": res.block_id,
        "chr": res.chr,
        "start": res.start,
        "end": res.end,
        "n_variants": res.n_variants,
        "n_haplotypes": res.n_haplotypes,
        "stat": res.stat_global.cpu().numpy(),
        "p": res.p_global.cpu().numpy(),
    }).sort_values("p")

    out = OUT_DIR / "10_haplotype_results.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nTop 5 haplotype blocks:\n{df.head(5).to_string(index=False)}")

    # Best window should overlap the causal region (SNPs 20..24).
    best = df.iloc[0]
    start_idx = int((best["start"] - 1000) // 1000)
    end_idx = int((best["end"] - 1000) // 1000)
    overlap = set(range(start_idx, end_idx + 1)) & set(causal_window)
    print(f"\nBest window spans SNPs {start_idx}..{end_idx}  "
          f"(overlap with causal = {sorted(overlap)})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
