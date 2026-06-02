# susieR external validation harness

This harness runs `susieR::susie_rss()` (upstream R reference) and our
`torchgenomics bayes-scan-rss` on the same per-locus fixture and compares the
results against the 6-metric tolerance table from NA1 design spec section 5.2.

## Prerequisites

- R >= 4.0 (`conda install -c conda-forge r-base` if missing)
- TorchGenomics installed in the current Python environment (`pip install -e ".[dev]"`)
- >= 2 GB free disk
- >= 4 GB free RAM

## Workflow

```bash
# 1. Install susieR (idempotent)
./install.sh

# 2. Provision the MDP-derived per-locus fixture (Tier B)
./fetch_data.sh

# 3. Run upstream susieR (Tier A scaffold; full impl Tier B)
./run_susieR.sh

# 4. Run our bayes-scan-rss (Tier A scaffold; full impl Tier B)
./run_torchgenomics.sh

# 5. Compare and emit findings
python compare.py \
    --upstream outputs/susieR.tsv \
    --ours outputs/torchgenomics.tsv \
    --findings ../../../docs/validation_findings.md
```

## Tolerance contract (per NA1 spec section 5.2)

| Metric | Threshold |
|---|---|
| Credible-set Jaccard | >= 0.95 (hard) |
| PIP correlation (PIP > 0.1 in either) | >= 0.99 |
| beta_mean Pearson correlation | >= 0.999 |
| beta_sd Pearson correlation | >= 0.999 |
| ELBO relative diff at convergence | <= 1e-4 |
| Wall-time (p <= 10K) | <= 2x susieR |

## Failure mode

Per F3 severity policy: any threshold violation is logged to
`docs/validation_findings.md` but does NOT block the phase (post-V1).
Investigate and document; do not silently accept.
