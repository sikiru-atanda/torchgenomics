# Quickstart

This page walks from a fresh install to a Manhattan plot in under 5 minutes
using the MDP maize fixture shipped with the repo.

## 1. Install

```bash
pip install "torchgenomics[all]"
```

## 2. Get the example data

The repo ships an MDP maize fixture under `benchmark/data/`:

- `mdp_numeric.txt` — 276 samples × 3093 SNPs (GAPIT numeric format)
- `mdp_SNP_information.txt` — chr / position map
- `mdp_traits.txt` — EarHT + pollen-shed phenotypes

```bash
git clone https://github.com/sikiru-atanda/torchgenomics.git
cd torchgenomics
```

## 3. Run an LMM scan from the CLI

```bash
torchgenomics lmm-scan \
  --genotype benchmark/data/mdp_numeric.txt \
  --map benchmark/data/mdp_SNP_information.txt \
  --phenotype benchmark/data/mdp_traits.txt \
  --trait EarHT \
  --test wald \
  --correction bh \
  --output results/mdp_earht
```

Output: `results/mdp_earht.tsv` with `chr`, `pos`, `beta`, `se`, `pval`,
`pval_adj`, `fdr` columns.

## 4. Run the same scan from Python

The worked example lives at
[`examples/python/01_single_trait_lmm.py`](https://github.com/sikiru-atanda/torchgenomics/blob/master/examples/python/01_single_trait_lmm.py).
Its essentials:

```python
from pathlib import Path

import pandas as pd
import torch

from torchgenomics.config import STAT_DTYPE, NumericalConfig
from torchgenomics.models import SingleTraitLMM, VariantMeta
from torchgenomics.stats import benjamini_hochberg

# 1. Read genotypes, phenotypes, and the GEMMA-format kinship
geno = pd.read_csv("benchmark/data/mdp_numeric.txt", sep="\t")
pheno = pd.read_csv("benchmark/data/mdp_traits.txt", sep="\t")
snpmap = pd.read_csv("benchmark/data/mdp_SNP_information.txt", sep="\t")

pheno["Taxa"] = pheno["Taxa"].astype(str)
geno["taxa"] = geno["taxa"].astype(str)
pheno = pheno[["Taxa", "EarHT", "dpoll"]].dropna(subset=["EarHT", "dpoll"])
common = sorted(set(pheno["Taxa"]) & set(geno["taxa"]))
pheno = pheno[pheno["Taxa"].isin(common)].set_index("Taxa").loc[common]
geno = geno[geno["taxa"].isin(common)].set_index("taxa").loc[common]

n, m = geno.shape
G = torch.tensor(geno.values, dtype=STAT_DTYPE)
Y = torch.tensor(pheno["EarHT"].values, dtype=STAT_DTYPE).unsqueeze(1)
X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
K = torch.tensor(
    pd.read_csv("gemma_demo/output/mdp_kinship.cXX.txt", sep="\t", header=None).values,
    dtype=STAT_DTYPE,
)

# 2. vmeta carries chr/pos through to the output
snp_to_cp = {str(r["SNP"]): (str(int(r["Chromosome"])), int(r["Position"]))
             for _, r in snpmap.iterrows()}
chrs = [snp_to_cp[s][0] for s in geno.columns]
pos = [snp_to_cp[s][1] for s in geno.columns]
vmeta = VariantMeta(snp=list(geno.columns), chr=chrs, pos=pos,
                    a1=["A"] * m, a2=["G"] * m)

# 3. Null fit + Wald scan + BH correction
model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
nf = model.fit_null(Y, X0, K=K)
result = model.score_chunk(G, nf, vmeta, test="wald")

p_t = result.p.cpu()
valid = torch.isfinite(p_t)
fdr = torch.full_like(p_t, float("nan"))
fdr[valid] = benjamini_hochberg(p_t[valid])
```

## 5. Plot the Manhattan

```python
import numpy as np
from torchgenomics.viz import manhattan_plot, qq_plot

chr_numeric = np.array([int(c) for c in chrs], dtype=int)
manhattan_plot(
    chrom=chr_numeric, pos=np.array(pos, dtype=int),
    p=result.p.cpu().numpy(),
    significance_threshold=5e-8, suggestive_threshold=1e-5,
    title="MDP maize — EarHT",
    output_path="mdp_earht_manhattan.png",
)
qq_plot(p=result.p.cpu().numpy(), output_path="mdp_earht_qq.png")
```

## Next steps

- Read the [CLI reference](../cli.md) for every subcommand
- Walk through the [single-trait LMM tutorial](../tutorials/single-trait-lmm.md)
  for a deeper dive
- See the [API reference](../api/index.md) for module-level details
- Check the [validation page](../validation.md) for GEMMA / GAPIT agreement
