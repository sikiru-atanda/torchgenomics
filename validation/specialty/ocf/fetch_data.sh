#!/usr/bin/env bash
# OCF fetch_data.sh -- fixture is simulated, no network downloads.
#
# Structurally required by the harness contract: every harness ships
# install.sh + fetch_data.sh + run.sh. For a fully-simulated study this is
# a no-op on downloads plus a positive smoke check that generate.R is
# functional before run.sh kicks off the full 100-replicate study.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../../external/_lib/preflight.sh
source "${HERE}/../../external/_lib/preflight.sh"

preflight_check "ocf-fetch" 1 2

mkdir -p "${HERE}/data"

# Smoke-run the generator with a tiny seed to confirm it works.
echo "[ocf fetch] smoke-running generate.R with n=50 m=5 reps=2 seed=42"
Rscript --vanilla "${HERE}/generate.R" --n 50 --m 5 --reps 2 --seed 42 --out-dir "${HERE}/data/_smoke"

# Validate the smoke output has the expected schema.
Rscript --vanilla -e '
  args <- commandArgs(trailingOnly = TRUE)
  smoke_dir <- args[1]
  d <- jsonlite::fromJSON(file.path(smoke_dir, "fixture_meta.json"))
  stopifnot(d$n == 50, d$m == 5, d$reps == 2)
  stopifnot(file.exists(file.path(smoke_dir, "rep_001.rds")))
  cat("[ocf fetch] smoke OK\n")
' "${HERE}/data/_smoke"

rm -rf "${HERE}/data/_smoke"
echo "[ocf fetch] done -- fixture generator validated"
