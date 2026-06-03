# rTorchGenomics

R interface to the TorchGenomics Python engine.

```r
install.packages(
  "rTorchGenomics",
  repos = c("https://sikiru-atanda.r-universe.dev", getOption("repos"))
)
library(rTorchGenomics)
tg_install()  # one-time Python setup

r <- tg_lmm_scan(genotype = "data.bed", phenotype = "pheno.tsv")
print(r)
tg_manhattan(r)
```

See `vignette("quickstart")` for the full walkthrough.
