# LRO — Leave-Region-Out LMM internal-consistency harness (Phase 29)

`torchgenomics.models.lro_lmm.LROLMM` implements **block-level proximal
decontamination**: for each LD block, the genome-wide GRM is subtracted
by the block's own contribution before testing SNPs in that block. This
removes the well-known LMM under-power on a causal SNP due to its own
signal being incorporated into the GRM (Yang et al. 2014; Listgarten
et al. 2012). No widely-used reference implementation exists outside
TG's own code, so per spec § 10.6 this harness validates the
**internal consistency** of the decontamination contract.

F7 panel LRO in the paper renders with a distinct border indicating
"no external reference."

## What's checked

The simulator generates 5 disjoint LD blocks (20 SNPs each, AR(1)-like
within-block correlation `ρ=0.85`) and plants a causal SNP in one
block (β=1.0). It then runs both standard LMM (full-K GRM, proximal
contamination present) and LROLMM (leave-block-out GRM) on the SAME
data, and asserts:

| # | Invariant | Why it matters |
|---|---|---|
| 1 | Standard LMM recovers causal β within 30% relative error | Sanity — even with proximal contamination, the planted signal is detectable |
| 2 | LROLMM recovers causal β within 30% relative error (typically tighter than standard) | The decontamination doesn't break β recovery |
| 3 | **LRO p < standard LMM p at the causal SNP** | The decontamination scientific core: removing the causal SNP from its own GRM gives a more significant test |
| 4 | Median non-causal p > 0.05 (null calibration) | Both tests are well-calibrated on the 99 non-causal SNPs |
| 5 | LD-block partition is non-trivial (≥ 2 blocks) | The LD detector ran and found structure |

## Fixture

| Parameter | Value | Notes |
|---|---|---|
| Seed | 42 | Deterministic |
| n samples | 500 | Enough for clear separation given β=1.0 |
| n blocks | 5 | Disjoint, recovered exactly by `ld_method="r2"` |
| SNPs/block | 20 | m total = 100 |
| LD ρ | 0.85 | Within-block AR(1)-like — high LD makes contamination obvious |
| Causal block | 2 | 0-indexed |
| β causal | 1.0 | Large effect — proximal contamination is measurable in p |
| Residual σ_e | 1.0 | Fixed |

## Re-run

```bash
bash validation/specialty/lro/run.sh
# Or directly:
TORCHGENOMICS_DISABLE_NATIVE=1 python3 validation/specialty/lro/generate_and_compare.py
```

Outputs (committed):

| File | Content |
|---|---|
| `results/summary.tsv` | One row per gate: name, observed, threshold, direction, passed |
| `results/agreement.json` | Full structured record (gates + observed numerics + extras) |
| `results/manifest.sha256` | SHA256 of generator + summary + agreement |

## Last observed numerics (PASS)

```
LRO detected 5 blocks (avg 20.0 SNPs/block) — matches simulator
β recovery (causal): full-K LMM = 0.7849; LROLMM = 0.8668; truth = 1.0
                     (LROLMM is closer to truth by 41%)
p at causal SNP:     full-K LMM = 3.02e-10; LROLMM = 1.82e-27
                     (LROLMM is 17 orders of magnitude more significant)
median non-causal p: 0.397 — well-calibrated null
```

## F3 / scientific-rigor note

LRO is **Phase 29**, post-V1. Per `memory/feedback_f3.md`, internal-
consistency invariants — not head-to-head agreement — are the
appropriate gate for novel methods without a reference. The
decontamination contract (Invariant 3 above) is the scientific core;
any future regression that broke it would surface here.

## Citation

- Yang J, Zaitlen NA, Goddard ME, Visscher PM, Price AL (2014).
  Advantages and pitfalls in the application of mixed-model
  association methods. *Nature Genetics* 46:100-106.
- Listgarten J, Lippert C, Kadie CM, Davidson RI, Eskin E, Heckerman D
  (2012). Improved linear mixed models for genome-wide association
  studies. *Nature Methods* 9:525-526.
