# One-call API quickstart

The `torchgenomics.api` module (also re-exported at the top level) gives
notebook users twelve curated tier-1 workflows as single function calls.
Each function:

  - **Accepts file paths and primitives** — no torch Tensors in the call site.
  - **Returns a typed result object** with summary stats, top hits as a
    DataFrame, inline plotting helpers, and explicit paths to the saved
    full-result files.
  - **Auto-detects format**, **auto-aligns samples** by ID, and **auto-creates
    output directories** when you omit `output=`.
  - **Carries no MCP / advanced-config surface area** — it's the same engine
    the CLI uses, just with smart defaults.

## Installation

```bash
pip install torchgenomics
# Optional extras:
pip install "torchgenomics[all]"   # zarr / h5py / pyarrow / seaborn
pip install "torchgenomics[mcp]"   # MCP server for Claude Desktop / Claude Code
```

## The 12 tier-1 functions

| Function | What it does |
|---|---|
| `tg.validate(genotype, phenotype)` | Pre-flight check: format, sample alignment, missingness, errors |
| `tg.convert(input, output, output_format="bed")` | Convert between BED, Zarr, VCF |
| `tg.impute(genotype, output, method="mean")` | Six built-in methods + GPU HMM + DL autoencoder |
| `tg.lmm_scan(genotype, phenotype)` | Single-trait LMM GWAS — streaming GRM, per-variant Wald/score/LRT |
| `tg.glm_scan(genotype, phenotype, family="gaussian")` | GLM family: Gaussian / binary / ordinal / multinomial |
| `tg.ld_blocks(genotype, method="gabriel")` | One of 13 haplotype-block algorithms; per-chromosome streaming |
| `tg.clump(sumstats, genotype)` | LD clumping to find independent index variants |
| `tg.meta(inputs, method="fixed")` | Meta-analysis: fixed / random / Han-Eskin / Stouffer |
| `tg.pgs_fit(sumstats, ld_ref, output, method="ldpred2-auto")` | C+T / LDpred2-{Inf,Grid,Auto} / PRS-CS |
| `tg.pgs_score(genotype, weights, output)` | Apply weights to target genotypes |
| `tg.annotate_hits(sumstats, crop="maize")` | NCBI gene annotation via Datasets v2 |
| `tg.manhattan(sumstats)` / `tg.qq(sumstats)` | Quick visualization helpers |

## Full notebook example

```python
import torchgenomics as tg

# 1. Pre-flight check.
v = tg.validate(genotype="data.bed", phenotype="pheno.tsv")
assert v.ok, v.errors
print(v.summary())

# 2. Run the GWAS.
r = tg.lmm_scan(
    genotype="data.bed",
    phenotype="pheno.tsv",
    output="run1/",          # optional; auto-named if omitted
    correction="bh",         # multiple-testing correction
    chunk_size=10_000,       # variants per streaming chunk
    n_pcs=5,                 # add 5 PCs from GRM as covariates
)
print(r.summary())
# Model: SingleTraitLMM  Test: wald  Correction: bh
# Samples: 2500  Variants tested: 489102
# Significant @ α=5.00e-08: 12
# λ_GC: 1.012
# Variance components: σ²_g=0.428, σ²_e=0.571, h²=0.428
# Top hits (10 shown):
#   CHR    POS  SNP        A1  A2    AF       BETA     SE   STAT      P
#     6  29115  rs9264942   T   C  0.21  -0.31  0.043  -7.21  5.4e-13
#   ...

# 3. Plot inline (notebook).
r.manhattan()
r.qq()

# 4. Read the full result table.
import pandas as pd
df = pd.read_csv(r.output_files["tsv"], sep="\t")

# 5. Pipe to post-GWAS.
clumps = tg.clump(
    sumstats=r.output_files["tsv"],
    genotype="data.bed",
    p_threshold=5e-8,
    r2=0.1,
)
genes = tg.annotate_hits(
    sumstats=r.output_files["tsv"],
    crop="maize",
    p_threshold=5e-8,
)

# 6. Build a PGS.
weights = tg.pgs_fit(
    sumstats=r.output_files["tsv"],
    ld_ref="ld.pt",
    output="pgs/weights.tsv",
    method="ldpred2-auto",
)
scores = tg.pgs_score(
    genotype="target.bed",
    weights="pgs/weights.tsv",
    output="pgs/scores.tsv",
)
print(scores.summary())
```

## Result objects

All twelve functions return a typed result with a consistent shape:

```python
r.summary()                  # str — pretty-printed report
r.to_dict()                  # dict — JSON-safe; what MCP returns
r.to_json(indent=2)          # JSON-encoded string
r.runtime_s                  # float — wall-time in seconds
r.output_files               # dict[str, Path] — produced artifacts
r.log_excerpt                # list[str] — tail of relevant log lines
```

For scan-type results (`tg.lmm_scan` / `tg.glm_scan`) you also get:

```python
r.n_variants                 # int — variants tested
r.n_significant              # int — passed `significance_threshold`
r.lambda_gc                  # float | None — genomic inflation
r.top_hits                   # pandas.DataFrame — top-K rows by p-value
r.manhattan(**kwargs)        # matplotlib Figure
r.qq(**kwargs)               # matplotlib Figure
```

## Three audiences, one code path

`torchgenomics.api` doesn't reinvent the engine — it wraps the same
`torchgenomics.models`, `torchgenomics.scan`, etc. that advanced users
call directly. The MCP server (`torchgenomics-mcp`) also calls
`torchgenomics.api` internally. So:

  - Behaviour is identical across the three audiences.
  - Optimizations (native C++ kernels, streaming chunking, GPU dispatch)
    apply automatically to every workflow.
  - Validation tests against external reference tools (GEMMA, GAPIT,
    PLINK 2.0, regenie, SAIGE, LDSC, BOLT-LMM, TwoSampleMR, ...) cover
    the same call paths.

For the low-level building blocks, see
[`docs/api/index.md`](index.md). For MCP integration, see
[`docs/mcp/index.md`](../mcp/index.md).
