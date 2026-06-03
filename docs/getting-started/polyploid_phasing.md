# Polyploid Phasing (Phase 56)

End-to-end recipe for phasing a connected tetraploid F1 population with
`torchgenomics phase-poly`, chaining off `dosage-call`. This doc reflects the
**current Phase 56 output shapes**; the integration path into
`HaplotypeGWAS` requires a follow-up adapter (see "What's next").

## Prerequisites

- A multi-sample VCF with the `AD` (allele-depth) field for parents + offspring.
- A pedigree TSV (`offspring\tparent1\tparent2`, optional `ploidy` column).
- A marker map TSV (`marker\tchrom\tpos_bp`, optional `cm` column).

Install the polyploid-phase extra:

```bash
pip install torchgenomics[polyploid-phase]
```

This adds `juliacall` (~15 MB). Julia itself is **not** downloaded by `pip
install` — it is discovered or installed on the first `phase-poly` run.

### Julia install mechanics

On first `phase-poly` run, the wrapper searches for an existing Julia in
this order:

1. The path in the `TORCHGENOMICS_JULIA` environment variable (if set).
2. `julia` on your `PATH`.
3. Common platform install locations (`~/.juliaup/bin/julia`, etc.).

If no Julia ≥ 1.10 is found, you will be prompted interactively. Add
`--auto-install` to accept non-interactively, or pre-install Julia from
<https://julialang.org/downloads/> (or via `juliaup`) and ensure `julia`
is on `PATH`.

The first run also precompiles `PolyOrigin.jl` v1.0.3 inside Julia's
package depot, which takes **1–2 minutes**. Subsequent calls in the same
process are fast.

## Pipeline

### Step 1 — Posterior dosage calling (Phase 55)

```bash
torchgenomics dosage-call \
  --vcf       calls.vcf.gz \
  --output    out/dcall \
  --ploidy    4
```

Writes `out/dcall.probs.pt` — `(n_samples, n_variants, ploidy+1)` float64
posterior tensor — plus `out/dcall.meta.json` and `out/dcall.snp_diag.tsv`.

### Step 2 — Polyploid phasing

```bash
torchgenomics phase-poly \
  --probs     out/dcall.probs.pt \
  --pedigree  pedigree.tsv \
  --map       markers.tsv \
  --output    out/phased \
  --ploidy    4
```

### Step 3 — Inspect the outputs

```python
import json
import torch

# Joint-origin state index per marker, per offspring
haps = torch.load('out/phased.haplotypes.pt')             # (n_off, m) int64

# Posterior probabilities over joint-origin states
origin_probs = torch.load('out/phased.origin_probs.pt')   # (n_off, m, n_states) float64

# Per-copy haplotype dosages for each parent
parent_phased = torch.load('out/phased.parent_phased.pt') # (n_parents, m, max_ploidy) int8

# Posterior dosage class probabilities for offspring (from PolyOrigin postdose CSV)
postdose = torch.load('out/phased.postdose_probs.pt')     # (n_off, m, max_ploidy+1) float64

# Sidecar metadata
meta = json.loads(open('out/phased.meta.json').read())
print(f"{len(meta['offspring_ids'])} offspring x {len(meta['variant_ids'])} markers")
print(f"haps shape        {tuple(haps.shape)}")
print(f"origin_probs shape {tuple(origin_probs.shape)}")
print(f"parent_phased shape {tuple(parent_phased.shape)}")
```

**Tensor shape notes:**

| File | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `haplotypes.pt` | `(n_off, m)` | `int64` | argmax joint-origin-combo index per marker |
| `origin_probs.pt` | `(n_off, m, n_states)` | `float64` | sparse-decoded joint-origin posterior |
| `parent_phased.pt` | `(n_parents, m, max_ploidy)` | `int8` | haplotype dosage per parental copy; `-1` = padding for lower-ploidy parents in mixed-ploidy pedigrees |
| `postdose_probs.pt` | `(n_off, m, max_ploidy+1)` | `float64` | posterior dosage class probabilities |

**Sidecar files:**

- `out/phased.meta.json` — lists `offspring_ids`, `parent_ids`,
  `variant_ids`, ploidy, tool version, input hash, and whether the marker
  map was refined by PolyOrigin (`map_refined: bool`).
- `out/phased.valent_diag.tsv` — PolyOrigin's per-marker valent
  configuration frequencies; useful for screening markers before GWAS.

Use `meta['offspring_ids']` to align your phenotype vector to the
haplotype tensor row order.

## What's next

Integration with `torchgenomics.models.HaplotypeGWAS` requires a shape
adapter that reconstructs per-copy offspring haplotypes `(n_off, ploidy,
m)` from `(joint_origin_state, parent_phased)`. That adapter is a
follow-up; the current Phase 56 output is tensor-ready for custom
downstream workflows but **not yet a drop-in for Phase 46/47 haplotype
GWAS**.

Until the adapter ships, you can use the `postdose_probs` tensor to derive
expected dosage for standard polyploid GWAS:

```python
from torchgenomics.preprocess.dosage_uncertainty import expected_dosage
G = expected_dosage(postdose, ploidy=4)   # (n_off, m) float64
# G is now compatible with poly-scan / gu-scan
```

## Troubleshooting

| Symptom | Remedy |
| --- | --- |
| `PolyOrigin failed: ...` | Inspect the Julia log printed to stderr. Re-run with `--keep-workdir` to preserve the Julia working directory for manual re-invocation. |
| `Julia at ... is v1.8.x; requires >= 1.10` | Install a newer Julia via `juliaup` (`juliaup install 1.10 && juliaup default 1.10`) or unset `TORCHGENOMICS_JULIA`. |
| First run hangs for 1–2 minutes | Normal — Julia JIT + PolyOrigin.jl precompilation. Subsequent calls in the same process are fast. |
| `ValueError: PolyOrigin supports ploidy 2, 4, or 6 only` | PolyOrigin.jl supports these three ploidies. Triploid / pentaploid data is not supported upstream. |
| `Expected header: offspring, parent1, parent2` | Your pedigree TSV must have those column names in the header row; the `ploidy` column is optional. |
| `VCF has no AD field` | Re-call with GATK `HaplotypeCaller` or `bcftools mpileup -a FORMAT/AD` before running `dosage-call`. |
| `map_refined: true` in meta.json | PolyOrigin re-ordered your markers within chromosomes. The tensors and `variant_ids` in `meta.json` reflect the refined order; re-order your phenotype covariates if they were marker-indexed. |
