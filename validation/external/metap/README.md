# validation/external/metap — R-side p-value combination reference

Head-to-head agreement between TorchGenomics's p-value combination kernels
and the CRAN `metap` (Fisher / Stouffer / HMP / Tippett) +
Bioconductor `EmpiricalBrownsMethod` reference implementations.

## Files

```
install.sh           Probe Rscript + tag whether metap / EBM are installed
fetch_data.sh        Generate the synthetic fixture (5 scenarios × k = 5)
simulate_fixture.py  Fixture generator (deterministic on --seed)
run_reference.R      Call metap::sumlog / sumz / minimump / hmp.stat +
                     EmpiricalBrownsMethod::empiricalBrownsMethod
run.sh               Invoke run_reference.R with the right args
compare.py           Compare R outputs vs TorchGenomics (postgwas.fisher_combined,
                     etc.); write results/summary.tsv + agreement.json
data/                Fixture inputs (pvalues.tsv, data_matrix.tsv, sim_truth.json)
outputs/             R reference output (reference.tsv)
results/             agreement.json + summary.tsv + manifest.sha256
```

## Tolerance gates

| Method | Mode | Threshold | Rationale |
|---|---|---|---|
| Fisher | abs(p) | 1e-10 | Closed form across libraries |
| Stouffer | abs(p) | 1e-10 | Closed form across libraries |
| Tippett min-p | abs(p) | 1e-10 | Closed form across libraries |
| HMP | abs(p) | 5e-3 | Wilson 2019 analytic scale L differs between implementations |
| Empirical Brown | rel(p) | 5e-2 | Covariance estimator details differ subtly between TG (Spearman + Kost-McDermott cubic) and EBM (Pearson + Bioconductor's polynomial fit) |

## Run

```bash
bash install.sh
bash fetch_data.sh
bash run.sh
python compare.py
```

When neither `metap` nor `EmpiricalBrownsMethod` is installed in R,
`run_reference.R` exits cleanly with an empty output and `compare.py`
reports `n_compared = 0`; the harness gracefully degrades rather than
hard-failing.

R install pointers:

```r
install.packages("metap", repos = "https://cloud.r-project.org")
BiocManager::install("EmpiricalBrownsMethod")
```
