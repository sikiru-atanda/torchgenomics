# Tutorial: Single-trait LMM on MDP maize

This walkthrough runs a canonical single-trait linear mixed model on the MDP
maize Ear Height phenotype — the same dataset used by the golden-data CI gate
against GEMMA 0.98.5.

## What you'll learn

- Load a GAPIT-format numeric genotype file and a GEMMA `.cXX` kinship
- Fit a null model with EMMA-style REML
- Run a Wald scan
- Apply Benjamini–Hochberg correction
- Save a Manhattan + QQ plot

Prerequisite: `pip install "torchgenomics[all]"` and a local clone of the repo (for
the `benchmark/data/` fixtures). The full working script is at
[`examples/python/01_single_trait_lmm.py`](https://github.com/sikiru-atanda/torchgenomics/blob/master/examples/python/01_single_trait_lmm.py).

## 1. Load the MDP fixture

TorchGenomics exposes its loaders through model-specific dataclasses rather than a
single monolithic reader — the canonical path is pandas → torch tensors →
`VariantMeta`:

```python
import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE
from torchgenomics.models import VariantMeta

geno = pd.read_csv("benchmark/data/mdp_numeric.txt", sep="\t")
pheno = pd.read_csv("benchmark/data/mdp_traits.txt", sep="\t")
snpmap = pd.read_csv("benchmark/data/mdp_SNP_information.txt", sep="\t")

pheno["Taxa"] = pheno["Taxa"].astype(str)
geno["taxa"] = geno["taxa"].astype(str)
pheno = pheno[["Taxa", "EarHT", "dpoll"]].dropna(subset=["EarHT", "dpoll"])
common = sorted(set(pheno["Taxa"]) & set(geno["taxa"]))
pheno = pheno[pheno["Taxa"].isin(common)].set_index("Taxa").loc[common]
geno = geno[geno["taxa"].isin(common)].set_index("taxa").loc[common]
print(geno.shape, pheno.shape)  # (276, 3093) (276, 2)
```

## 2. Tensors + VariantMeta

```python
n, m = geno.shape
G = torch.tensor(geno.values, dtype=STAT_DTYPE)
# Mean-impute any missing dosage values in-place
for j in range(m):
    col = G[:, j]
    mask = torch.isnan(col)
    if mask.any():
        G[mask, j] = col[~mask].mean()

Y = torch.tensor(pheno["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
X0 = torch.ones(n, 1, dtype=STAT_DTYPE)  # intercept-only covariate matrix

snp_to_cp = {str(r["SNP"]): (str(int(r["Chromosome"])), int(r["Position"]))
             for _, r in snpmap.iterrows()}
chrs = [snp_to_cp[s][0] for s in geno.columns]
pos = [snp_to_cp[s][1] for s in geno.columns]
vmeta = VariantMeta(snp=list(geno.columns), chr=chrs, pos=pos,
                    a1=["A"] * m, a2=["G"] * m)
```

## 3. Kinship (GEMMA `.cXX` format)

Pre-computed kinship matrices from GEMMA or GAPIT load directly:

```python
K = torch.tensor(
    pd.read_csv("gemma_demo/output/mdp_kinship.cXX.txt",
                sep="\t", header=None).values,
    dtype=STAT_DTYPE,
)
```

If you want TorchGenomics to compute the GRM itself, use
`torchgenomics.linalg.kinship.grm_vanraden(G)` which returns `(K, n_snps)`.

## 4. Fit null and scan

```python
from torchgenomics.config import NumericalConfig
from torchgenomics.models import SingleTraitLMM

model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
nf = model.fit_null(Y, X0, K=K)
print(f"vg={float(nf.sig2_g):.4f}  ve={float(nf.sig2_e):.4f}  "
      f"logL={float(nf.log_likelihood):.2f}")

result = model.score_chunk(G, nf, vmeta, test="wald")
```

`result` is a `ScanResult` with `.snp`, `.beta`, `.se`, `.p` attributes (all
torch tensors, shape `(m,)`).

## 5. Multiple-testing correction

```python
from torchgenomics.stats import benjamini_hochberg

p_t = result.p.cpu()
valid = torch.isfinite(p_t)  # NaN rare-allele SNPs
fdr = torch.full_like(p_t, float("nan"))
fdr[valid] = benjamini_hochberg(p_t[valid])

df = pd.DataFrame({
    "snp": result.snp,
    "chr": chrs, "pos": pos,
    "beta": result.beta.cpu().numpy(),
    "se": result.se.cpu().numpy(),
    "pval": p_t.numpy(),
    "fdr": fdr.numpy(),
})
print(df.nsmallest(10, "pval").to_string(index=False))
```

## 6. Manhattan + QQ plots

```python
import numpy as np
from torchgenomics.viz import manhattan_plot, qq_plot

chr_numeric = np.array([int(c) for c in chrs], dtype=int)
manhattan_plot(
    chrom=chr_numeric, pos=np.array(pos, dtype=int),
    p=result.p.cpu().numpy(),
    significance_threshold=5e-8, suggestive_threshold=1e-5,
    title="MDP maize — EarHT — SingleTraitLMM",
    output_path="mdp_earht_manhattan.png",
)
qq_plot(p=result.p.cpu().numpy(),
        title="MDP maize — EarHT — QQ",
        output_path="mdp_earht_qq.png")
```

## GPU

The entire pipeline runs on CUDA with a one-line change — every model tracks
the device of its input tensors:

```python
G = G.to("cuda")
Y = Y.to("cuda")
X0 = X0.to("cuda")
K = K.to("cuda")

model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
nf = model.fit_null(Y, X0, K=K)     # on CUDA
result = model.score_chunk(G, nf, vmeta, test="wald")
```

## Reference agreement

With the default settings, TorchGenomics's Wald, LRT, and score p-values agree
with GEMMA 0.98.5's to the 4th decimal place (−log₁₀ p correlation > 0.998,
β correlation > 0.9999, SE correlation > 0.9999). See the
[validation page](../validation.md) and
`scripts/gemma_agreement_report.py` for the reproducer.
