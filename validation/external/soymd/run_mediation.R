#!/usr/bin/env Rscript
# Run the canonical `mediation::mediate()` analysis (Imai/Keele/Tingley 2010,
# Tingley et al. 2014) on the simulated multi-omics triple and emit a JSON
# blob the Python harness (compare.py) parses.
#
# Pipeline (mediation::mediate textbook usage):
#   1. Fit mediator model:    M ~ SNP                       (linear OLS)
#   2. Fit outcome model:     Y ~ M + SNP                   (linear OLS)
#   3. Run `mediate(model_m, model_y, treat="snp", mediator="mediator",
#                   sims = N_SIMS)` to obtain ACME / ADE / total / proportion-
#      mediated estimates with quasi-Bayesian (default) or bootstrap CIs.
#
# Output layout (outputs/mediation_results.json):
#   {
#     "acme":  {"point": ..., "ci_lower": ..., "ci_upper": ..., "p": ...},
#     "ade":   {"point": ..., "ci_lower": ..., "ci_upper": ..., "p": ...},
#     "total": {"point": ..., "ci_lower": ..., "ci_upper": ..., "p": ...},
#     "proportion_mediated":
#            {"point": ..., "ci_lower": ..., "ci_upper": ..., "p": ...},
#     "model_m": {"a": ..., "a_se": ..., "a_p": ...},
#     "model_y": {"b": ..., "b_se": ..., "b_p": ...,
#                 "c_prime": ..., "c_prime_se": ..., "c_prime_p": ...},
#     "n_sims": ...,
#     "package_versions": {"mediation": ..., "R": ...}
#   }
#
# Why we record the *point* + *CI* + *p* separately for each effect:
#   `mediation::mediate` returns `d.avg`, `z.avg`, `tau.coef`,
#   `n.avg` for the average over treatment levels, plus the matching
#   `*.ci` (length-2 numeric, names ".5%" / "97.5%" by default) and
#   `*.p` p-values. We emit the structure compare.py expects.

suppressPackageStartupMessages({
    library(mediation)
    library(jsonlite)
})

# --- CLI args ---------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
    idx <- which(args == flag)
    if (length(idx) == 1 && idx + 1 <= length(args)) return(args[idx + 1])
    default
}

DATA_DIR <- parse_arg("--data-dir", "data")
OUT_DIR <- parse_arg("--output-dir", "outputs")
N_SIMS <- as.integer(parse_arg("--n-sims", "1000"))
SEED <- as.integer(parse_arg("--seed", "42"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf(
    "[run_mediation] data_dir=%s out_dir=%s n_sims=%d seed=%d\n",
    DATA_DIR, OUT_DIR, N_SIMS, SEED
))

# --- Load simulated triple ---------------------------------------------------
df <- read.delim(file.path(DATA_DIR, "triple.tsv"), stringsAsFactors = FALSE)
stopifnot(all(c("snp", "mediator", "outcome") %in% names(df)))
stopifnot(nrow(df) >= 50)

cat(sprintf("[run_mediation] loaded %d rows\n", nrow(df)))

# --- Fit mediator + outcome models -------------------------------------------
# Both fits are plain OLS (the standard mediation::mediate input). We expose
# coefficients separately so compare.py can sanity-check the per-stage
# estimates against TorchGWAS' a, b, c' returned by mediate_lmm.
model_m <- lm(mediator ~ snp, data = df)
model_y <- lm(outcome ~ mediator + snp, data = df)

m_coef <- summary(model_m)$coefficients
y_coef <- summary(model_y)$coefficients

a_hat <- unname(m_coef["snp", "Estimate"])
a_se  <- unname(m_coef["snp", "Std. Error"])
a_p   <- unname(m_coef["snp", "Pr(>|t|)"])

b_hat <- unname(y_coef["mediator", "Estimate"])
b_se  <- unname(y_coef["mediator", "Std. Error"])
b_p   <- unname(y_coef["mediator", "Pr(>|t|)"])

c_prime_hat <- unname(y_coef["snp", "Estimate"])
c_prime_se  <- unname(y_coef["snp", "Std. Error"])
c_prime_p   <- unname(y_coef["snp", "Pr(>|t|)"])

# --- Run mediate() -----------------------------------------------------------
# Default `boot = FALSE` uses the quasi-Bayesian Monte-Carlo method of Imai,
# Keele & Tingley 2010 §3.2 (draw model parameters from their asymptotic
# multivariate-normal distribution; predict mediator + outcome under both
# treatment values; average the contrast). With `sims = N_SIMS = 1000` the
# CI bounds are reproducible to 2-3 sig figs given the global RNG seed.
set.seed(SEED)
out <- mediate(
    model.m = model_m,
    model.y = model_y,
    treat   = "snp",
    mediator = "mediator",
    sims    = N_SIMS,
    boot    = FALSE
)

# --- Extract effects ---------------------------------------------------------
# `mediate` returns averaged-over-treatment effects in `*.avg.*` slots and
# the per-treatment-level estimates in `*.0.*` / `*.1.*`. For a continuous
# treatment `mediate` reports the average; we record that as the canonical
# ACME / ADE.
acme_point <- unname(out$d.avg)
acme_ci    <- unname(out$d.avg.ci)
acme_p     <- unname(out$d.avg.p)

ade_point  <- unname(out$z.avg)
ade_ci     <- unname(out$z.avg.ci)
ade_p      <- unname(out$z.avg.p)

total_point <- unname(out$tau.coef)
total_ci    <- unname(out$tau.ci)
total_p     <- unname(out$tau.p)

prop_point  <- unname(out$n.avg)
prop_ci     <- unname(out$n.avg.ci)
prop_p      <- unname(out$n.avg.p)

# --- Compose JSON output -----------------------------------------------------
out_obj <- list(
    acme = list(point = acme_point,
                ci_lower = acme_ci[1], ci_upper = acme_ci[2],
                p = acme_p),
    ade  = list(point = ade_point,
                ci_lower = ade_ci[1], ci_upper = ade_ci[2],
                p = ade_p),
    total = list(point = total_point,
                 ci_lower = total_ci[1], ci_upper = total_ci[2],
                 p = total_p),
    proportion_mediated = list(point = prop_point,
                                ci_lower = prop_ci[1], ci_upper = prop_ci[2],
                                p = prop_p),
    model_m = list(a = a_hat, a_se = a_se, a_p = a_p),
    model_y = list(b = b_hat, b_se = b_se, b_p = b_p,
                   c_prime = c_prime_hat,
                   c_prime_se = c_prime_se,
                   c_prime_p  = c_prime_p),
    n_sims = N_SIMS,
    n      = nrow(df),
    package_versions = list(
        mediation = as.character(packageVersion("mediation")),
        R         = R.version.string,
        seed      = SEED,
        n_sims    = N_SIMS
    )
)

out_path <- file.path(OUT_DIR, "mediation_results.json")
write_json(out_obj, path = out_path, pretty = TRUE, auto_unbox = TRUE, na = "null")
cat(sprintf("[run_mediation] wrote %s\n", out_path))

# Echo a one-line summary for quick eyeballing.
cat(sprintf(
    "[run_mediation] a=%.4f b=%.4f c'=%.4f | ACME=%.4f [%.4f, %.4f] (p=%.3g) | total=%.4f\n",
    a_hat, b_hat, c_prime_hat,
    acme_point, acme_ci[1], acme_ci[2], acme_p,
    total_point
))
