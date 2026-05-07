# Polyploid Allele Dosage Calling (Phase 55)

This recipe converts polyploid sequencing VCFs into posterior dosage
tensors and feeds them into the uncertainty-aware association test
(GU-LMM, Phase 28).

## Prerequisites

TorchGWAS `dosage-call` shells out to the R package
[`updog`](https://cran.r-project.org/package=updog). Install it once:

```bash
# on a machine with R >= 4.0 on PATH:
Rscript -e 'install.packages("updog", repos="https://cloud.r-project.org")'
```

Your input VCF must be biallelic and carry the `AD` (allele depth) format
field. If you're starting from BAMs, GATK `HaplotypeCaller` and
`bcftools mpileup -a FORMAT/AD` both emit this.

## Step 1 — Call dosages

```bash
torchgwas dosage-call \
  --vcf calls.vcf.gz \
  --output out/dcall \
  --ploidy 4 \
  --model norm \
  --n-cores 4
```

This writes:

- `out/dcall.probs.pt` — `(n_samples, n_variants, ploidy+1)` posterior tensor.
- `out/dcall.meta.json` — IDs, ploidy, tool version, input hash, command.
- `out/dcall.snp_diag.tsv` — per-marker `updog` diagnostics (bias, seq error, OD).

## Step 2 — Run uncertainty-aware GWAS (GU-LMM)

Pass `--probs out/dcall.probs.pt` directly to `gu-scan`; it reads the
posterior tensor and derives the expected dosage + per-slot variance
internally (via `expected_dosage` / `dosage_variance`, with ploidy
picked up from the sibling `out/dcall.meta.json`):

```bash
torchgwas gu-scan \
  --probs out/dcall.probs.pt \
  --phenotype pheno.txt \
  --output results
```

If you need a custom dosage variance (e.g. inflated for a stress
test), pass `--dosage-var` alongside `--probs` — the user-supplied
variance wins.

### Alternative — standard polyploid scan (ignoring uncertainty)

`poly-scan` doesn't know about `--probs`; derive expected dosages
manually:

```python
import torch
from torchgwas.preprocess.dosage_uncertainty import expected_dosage
probs = torch.load("out/dcall.probs.pt", weights_only=True)
torch.save(expected_dosage(probs, 4), "out/dosage.pt")
```

```bash
torchgwas poly-scan \
  --genotype out/dosage.pt \
  --phenotype pheno.txt \
  --ploidy 4 \
  --output results
```

## Troubleshooting

- **`Rscript not found`**: install R ≥ 4.0 and ensure `Rscript` is on
  `PATH`, or pass `--rscript /path/to/Rscript`.
- **`updog R package not installed`**: run the one-liner in Prerequisites.
- **`VCF has no AD field`**: re-call with GATK `HaplotypeCaller` or
  `bcftools mpileup -a FORMAT/AD`.
- **`Multi-allelic site at 1:12345`**: split with `bcftools norm -m -any`
  before calling dosages; updog is strictly biallelic.
