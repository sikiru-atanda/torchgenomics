# GU — Genotype-Uncertainty LMM internal-consistency harness (Phase 28)

`torchgenomics.models.gu_lmm.GULM` implements a dosage-variance-corrected
score test for genotype data with per-cell imputation uncertainty. No
widely-used reference implementation exists outside the original paper's
own code base, so per spec § 10.6 this harness validates the
**internal consistency** of the GU score test rather than running a
head-to-head against an external tool. F7 panel GU in the paper
renders with a distinct border indicating "no external reference."

## What's checked

The simulator generates a small fixture with known per-cell dosage
variance σ² and known causal effect β, then asserts four invariants:

| # | Invariant | Why it matters |
|---|---|---|
| 1 | `dosage_var = zeros` matches `dosage_var = None` exactly (|Δ stat| ≤ 1e-8) | The GU correction must reduce to the standard score test in the no-uncertainty limit. |
| 2 | Under planted uncertainty, `stat_corrected ≤ stat_uncorrected` for ≥ 95% of SNPs | The correction is monotone-increasing in σ²; an honest implementation is conservative under measurement uncertainty. |
| 3 | Median causal p < median null p | The test still discriminates signal from null under uncertainty (separation, not absolute power). |
| 4 | ≥ 3 / 5 causal SNPs below the 5th-percentile null p-value | Rank-based detection power check (robust to absolute noise scale). |

## Fixture

| Parameter | Value | Notes |
|---|---|---|
| Seed | 42 | Deterministic |
| n samples | 1000 | Power ≥ Bonferroni-significance at the planted β under σ=0.20 noise |
| m null | 100 | No causal effect |
| m causal | 5 | β = 0.50 each |
| β_causal | 0.50 | Large enough that all 5 causal SNPs rank above the 5th-percentile null p under uncertainty |
| Dosage σ | 0.20 | Imputation-style per-cell noise SD |
| Residual σ_e | 1.0 | Fixed (so signal-to-noise scales cleanly with β) |

The GRM is computed from the **true** (noise-free) genotypes — this
isolates the GU correction to the per-SNP test rather than confounding
it with kinship noise.

## Re-run

```bash
bash validation/specialty/gu/run.sh
# Or directly:
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/specialty/gu/generate_and_compare.py
```

Outputs (committed):

| File | Content |
|---|---|
| `results/summary.tsv` | One row per gate: name, observed, threshold, direction, passed |
| `results/agreement.json` | Full structured record (gates + observed numerics + extras) |
| `results/manifest.sha256` | SHA256 of generator + summary + agreement |

## Last observed numerics (PASS)

```
zero-uncertainty matches None-path: |Δ stat| max = 0 (exact)
non-zero-uncertainty conservativeness: 100% of SNPs satisfy stat_unc ≤ stat_none
median causal p (uncertainty) = 3.36e-3 vs median null p = 0.46 (clear separation)
causal SNPs ranked above 5th-percentile null: 5 / 5
```

## F3 / scientific-rigor note

GU is **Phase 28**, post-V1. Per `memory/feedback_f3.md`, internal-
consistency invariants — not head-to-head agreement — are the
appropriate gate for novel methods without a reference. Any future
F3 finding here would be a regression in TG's own invariants (e.g.,
correction becoming non-monotone), not a divergence from an external
tool.

## Citation

- Yang J, Lee SH, Goddard ME, Visscher PM (2011, 2017). GCTA: a tool
  for genome-wide complex trait analysis. *AJHG* 88:76-82, and the
  dosage-uncertainty extension in subsequent revisions.
- Marchini J, Howie B, Myers S, et al. (2007). A new multipoint method
  for genome-wide association studies by imputation of genotypes.
  *Nature Genetics* 39:906-913.
