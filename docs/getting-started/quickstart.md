# Quickstart

This page goes from a fresh install to a Manhattan plot in a few minutes using
the tiny PLINK fixture shipped with the repo — first with the one-call
`tg.gwas` API (recommended), then with the low-level building blocks.

## 1. Install

```bash
pip install "torchgenomics[all]"
```

## 2. Get the example data

The repo ships a tiny PLINK BED fixture under `tests/fixtures/` (12 samples ×
20 variants, two continuous traits `Y1` / `Y2`):

- `tests/fixtures/tiny.bed` / `.bim` / `.fam` — genotypes (PLINK 1 BED)
- `tests/fixtures/tiny_pheno.txt` — `IID`, `Y1`, `Y2` phenotype table

```bash
git clone https://github.com/sikiru-atanda/torchgenomics.git
cd torchgenomics
```

## 3. Run a GWAS with `tg.gwas` (recommended)

`tg.gwas` is the one-call entry point: it auto-detects the genotype format,
aligns samples, computes the kinship/PCs, runs the model you pick, and returns
a self-describing result.

```python
import torchgenomics as tg

r = tg.gwas(
    phenotype="tests/fixtures/tiny_pheno.txt",
    genotype="tests/fixtures/tiny.bed",
    trait="Y1",           # pick a phenotype column (first column if omitted)
    models="lmm",         # SingleTraitLMM (GEMMA-equivalent)
)

print(r.summary())        # trait, n, model, λ_GC (+ interpretation), top hits, next steps
r.manhattan()             # matplotlib Manhattan figure from the results
r.report("results/")      # writes summary.txt + association table + plots
```

`r.summary()` prints a plain-language block, for example:

```
GWAS scan summary — Y1 (continuous)
Model: SingleTraitLMM  Test: wald  Correction: bh
Samples: 12   Variants tested: 20
λ_GC: 0.93 (✓ well-calibrated)
Significant @ α=5.00e-08: 0
Top locus: rs0017 / 3 / 18000, p=7.58e-02
Next: no genome-wide-significant hits — check power or relax significance_threshold;
      call .report(dir) to save the summary and plots
```

`r.hits` is the top-hits `DataFrame`, `r.diagnostics` is a JSON-safe dict, and
`r.output_files` points at the full association table on disk.

### The same run from the CLI

```bash
torchgenomics gwas \
  --phenotype tests/fixtures/tiny_pheno.txt \
  --genotype tests/fixtures/tiny.bed \
  --trait Y1 \
  --models lmm
```

## 4. Choose your model(s)

`models=` is your choice — any registry alias, a list (returns a
`GwasComparison`), or `"auto"` (an opt-in convenience that prints which model
it picked and why).

```python
import torchgenomics as tg

# Compare two models side by side
cmp = tg.gwas(
    phenotype="tests/fixtures/tiny_pheno.txt",
    genotype="tests/fixtures/tiny.bed",
    trait="Y1",
    models=["lmm", "blink"],
)
print(cmp.summary())      # per-model table (test, n_variants, n_sig, λ_GC)

# List every model tg.gwas accepts
print(tg.list_models())   # alias, label, trait_types, description
```

Every registry model is reachable through `tg.gwas`: `lmm`, `glm`, `blink`,
`farmcpu`, `glmm`, `mklmm`, `gxe`, `set`, `bayes`, `mvlmm`. A few carry an
extra input:

- `models="gxe"` needs `env=` (a `pandas.Series` indexed by sample id, a 1-D
  array in sample order, or a path to an env TSV).
- `models="set"` needs `regions=` (a path to a chrom/start/end regions file,
  or a `pandas.DataFrame` with those columns).
- `models="mvlmm"` needs `traits=[...]` naming **≥2** columns of a multi-column
  phenotype (a `DataFrame` or a path); it returns a joint multi-trait result
  whose table carries `P_JOINT`. From the CLI:
  `torchgenomics gwas --models mvlmm --traits Y1,Y2 --phenotype ... --genotype ...`.
- `models="bayes"` is a fine-mapping run — its `top_hits` carries `PIP`
  (posterior inclusion probability) and credible-set membership instead of a
  `P` column, and `lambda_gc` is `None`.

## 5. Under the hood: the low-level API

`tg.gwas` is thin orchestration over the low-level modules, which you can drive
directly for full control. The worked example lives at
[`examples/python/01_single_trait_lmm.py`](https://github.com/sikiru-atanda/torchgenomics/blob/master/examples/python/01_single_trait_lmm.py);
its essentials:

```python
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.stats import benjamini_hochberg

# Y: (n, 1) phenotype, X0: (n, 1) intercept, K: (n, n) kinship, G: (n, m) dosage
model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
nf = model.fit_null(Y, X0, K=K)
vmeta = VariantMeta(snp=snp_ids, chr=chrs, pos=pos, a1=a1, a2=a2)
result = model.score_chunk(G, nf, vmeta, test="wald")

p = result.p.cpu()
valid = torch.isfinite(p)
fdr = torch.full_like(p, float("nan"))
fdr[valid] = benjamini_hochberg(p[valid])
```

Plot the Manhattan / Q-Q directly from the low-level result:

```python
import numpy as np
from torchgenomics.viz import manhattan_plot, qq_plot

manhattan_plot(
    chrom=np.array([int(c) for c in chrs]), pos=np.array(pos, dtype=int),
    p=result.p.cpu().numpy(),
    significance_threshold=5e-8, suggestive_threshold=1e-5,
    output_path="manhattan.png",
)
qq_plot(p=result.p.cpu().numpy(), output_path="qq.png")
```

## Next steps

- Read the [CLI reference](../cli.md) for every subcommand
- Walk through the [single-trait LMM tutorial](../tutorials/single-trait-lmm.md)
  for a deeper dive
- See the [API reference](../api/index.md) for module-level details
- Check the [validation page](../validation.md) for GEMMA / GAPIT agreement
