# validation/external/fusion — FUSION measured-expression harness

Head-to-head agreement test of TorchGenomics's `twas_observed_expression`
against the FUSION TWAS pipeline's **measured-expression mode** —
i.e., the FUSION workflow where the user has observed normalized
expression in the discovery cohort and wants gene-trait associations
directly, without going through pre-trained cis-eQTL weights.

## Algorithmic equivalence

Both paths fit the same OLS per gene:

    y ~ intercept + covariates + expression_g

and emit the Wald statistic for the `expression_g` coefficient (F(1,
n-p) under FUSION's R `lm.fit`; F(1, n-c-1) under TorchGenomics's
`torchgenomics.models.glm.GLM`). With matching covariates and matching
residual-variance estimators the two should agree at floating-point
precision — gates are `1e-8` on β/SE and `1e-6` on z and -log10 p.

## Reference choice

The community FUSION distribution (`gusevlab/fusion_twas`) is built
around imputed GReX from cis-eQTL weight files. The measured-expression
mode is a documented FAQ workflow that re-uses the same R `lm` Wald
this harness's `run_reference.R` reproduces verbatim. We do not invoke
`FUSION.assoc_test.R` directly because that path's `--weights` argument
is mandatory and incompatible with the measured-expression fixture.
The R driver in `run_reference.R` is the operational reference the
FUSION README points at for this mode.

## Files

```
install.sh           Probe R + plink + clone gusevlab/fusion_twas (pinned commit)
fetch_data.sh        Generate the in-process fixture (300 × 15, planted causal)
simulate_fixture.py  The fixture generator (deterministic on --seed)
run_reference.R      The reference OLS Wald per gene (FAQ mode)
run.sh               Invoke run_reference.R with the right args
compare.py           Side-by-side vs torchgenomics.postgwas.twas_observed_expression
data/                Fixture inputs (expression.tsv, phenotype.tsv, covariates.tsv, genes.bed)
outputs/             FUSION reference output (fusion_results.tsv)
results/             agreement.json + summary.tsv + manifest.sha256
```

## Run

```bash
bash install.sh        # one-time: R + plink + FUSION repo
bash fetch_data.sh     # regenerate the fixture (seed-42)
bash run.sh            # run the R reference
python compare.py      # side-by-side; writes results/
```

## Pre-flight

Each shell script sources `../_lib/preflight.sh`. Per spec §5.3, RAM
and disk headroom are asserted before any download / install / run —
no partial executions. R + plink + the `optparse`, `glmnet`, and
`plink2R` R packages are runtime dependencies of `install.sh`.

## Tolerance posture

Tolerances are observed-then-floored. The gates above are set
conservatively above the FP-precision agreement we expect under
matching arithmetic.
