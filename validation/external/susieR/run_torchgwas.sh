#!/usr/bin/env bash
# Run torchgwas bayes-scan-rss on the same fixture as run_susieR.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_DIR="${SCRIPT_DIR}/outputs"
mkdir -p "${OUT_DIR}"

if [ ! -f "${DATA_DIR}/locus_z.pt" ]; then
    echo "FATAL: fixture not provisioned. Run ./fetch_data.sh first."
    exit 2
fi

# Convert fixture into the sumstats + LD-ref format bayes-scan-rss expects
python3 - <<PYEOF
import torch
import pandas as pd
import math
from torchgwas.postgwas._ld_ref_loader import save_ld_reference
from torchgwas.postgwas._ld_ref_metadata import LDReferenceMetadata

z = torch.load("${DATA_DIR}/locus_z.pt", weights_only=True)
R = torch.load("${DATA_DIR}/locus_R.pt", weights_only=True)
n = int(open("${DATA_DIR}/locus_n.txt").read().strip())
meta_dict = torch.load("${DATA_DIR}/locus_meta.pt", weights_only=False)
snp_ids = meta_dict["snp_ids"]
chrs = meta_dict["chr"]
bps = meta_dict["bp"]
a1s = meta_dict["a1"]
a2s = meta_dict["a2"]
p = z.shape[0]

se = 1.0 / math.sqrt(n)
beta = z * se
sumstats = pd.DataFrame({
    "SNP": snp_ids,
    "CHR": chrs,
    "BP": bps,
    "A1": a1s,
    "A2": a2s,
    "BETA": beta.tolist(),
    "SE": [se] * p,
    "N": [n] * p,
})
sumstats.to_csv("${DATA_DIR}/sumstats.tsv", sep="\t", index=False)

meta = LDReferenceMetadata(
    cohort_id="MDP",
    n=n,
    build="B73_v4",
    panel_provenance="MDP fixture",
)
save_ld_reference("${DATA_DIR}/ld.pt", R, snp_ids, meta)
PYEOF

START=$(date +%s.%N)
/usr/bin/time -v python3 -m torchgwas bayes-scan-rss \
    --sumstats "${DATA_DIR}/sumstats.tsv" \
    --ld-ref "${DATA_DIR}/ld.pt" \
    --max-num-causal 10 \
    --coverage 0.95 \
    --purity 0.5 \
    --output "${OUT_DIR}/torchgwas.tsv" \
    2>"${OUT_DIR}/torchgwas_time.log"
END=$(date +%s.%N)
echo "$END $START" | awk '{print $1 - $2}' > "${OUT_DIR}/torchgwas_walltime_seconds.txt"

# Extract our final ELBO post-hoc by re-fitting and reading result.elbo.
# (The output TSV writer doesn't include ELBO; adding that is a Tier B item.
# Re-fitting is cheap on a p=200 fixture and avoids changing the writer
# schema or the existing 17/17 NA1 tests.)
python3 - <<PYEOF
import torch
from torchgwas.models.bayesian_vs_rss import BayesianVSRss

z = torch.load("${DATA_DIR}/locus_z.pt", weights_only=True)
R = torch.load("${DATA_DIR}/locus_R.pt", weights_only=True)
n = int(open("${DATA_DIR}/locus_n.txt").read().strip())

model = BayesianVSRss(max_num_causal=10, coverage=0.95, purity=0.5)
result = model.fit_rss(z=z, R=R, n=n)
with open("${OUT_DIR}/torchgwas_elbo.txt", "w") as f:
    f.write(f"{result.elbo}\n")
print(f"torchgwas ELBO = {result.elbo:.6g} (converged={result.converged}, n_iter={result.n_iter})")
PYEOF

echo "torchgwas bayes-scan-rss run complete: ${OUT_DIR}/torchgwas.tsv"
