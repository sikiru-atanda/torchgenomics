# NA3 streaming-memory + β-parity benchmark harness

Empirically validates the streaming + biobank-scale claims from the v0.3.x
campaign. Two tracks, both run locally (no UKB required):

- **Track A** — peak-memory observed vs projected: `p_sweep.py`
- **Track B** — β-parity vs regenie on the synthetic fixture:
  `run_parity_vs_regenie.py`

Track C (h² vs LDSC) is already covered by `validation/external/ldsc/`
(Pillar B); no NA3 extension required.

## Reproduce

```bash
# Track A: p-sweep across 10K–1M variants (n=2000)
python validation/streaming_memory/p_sweep.py \
    --n 2000 \
    --p-values 10000,25000,50000,100000,250000,500000,1000000 \
    --chunk-size 5000

# Track B: head-to-head β-parity at n=2K, p=10K
#   (requires bash validation/external/regenie/install.sh first)
python validation/streaming_memory/run_parity_vs_regenie.py \
    --prefix /tmp/streaming_bench/sweep_p10000
```

## Findings ledger

See `docs/validation_findings.md` → "Update 2026-05-13: NA3 empirical
validation (Tracks A + B)". Headline results:

- Streaming memory slope is **19× lower** than the materialized would-be
  slope across the p ∈ [10K, 1M] sweep
- β / SE / χ² / -log10 p Pearson all **1.000000** vs regenie after
  effect-allele alignment
- **Effect-allele sign flip** noted between TG lmm-scan and regenie
  conventions (raw β Pearson = -1.0; the harness flips globally and
  documents). Filed F3 medium: documentation/interop gap, not a
  numerical bug — |β|, χ², SE, p all agree.

## Outputs

`outputs/` is `.gitignored` (per Pillar B convention). Each run emits:

- `outputs/p_sweep.tsv` — per-p row of (peak RSS, peak USS, scan-attrib MB, elapsed s)
- `outputs/p_sweep.txt` — markdown summary + scaling slope diagnostic
- `outputs/run_p{P}.json` — per-fixture JSON record
- `outputs/parity/torchgwas.assoc.tsv` — TG lmm-scan output
- `outputs/parity/regenie/step2_Y.regenie` — regenie step 2 output
- `outputs/parity_report.txt` — Track B numeric report

## UKB-scale extension (pending user)

The campaign's headline claim ("40 TB → 4 GB peak") is for UKB-scale
n=500K. This harness validates the streaming math on the **p axis**
(19× slope reduction) and β-parity at moderate scale. The full
n=500K × p=10M run requires UKB access; it's the remaining piece of
NA3 that needs the user's day-job hands. Expected at that scale (linear
extrapolation):

- Streaming: ~9 GB scan-attributable + dense GRM penalty (n²×8 = 1 TB at n=500K
  → requires `--approx-method sparse` for tractability)
- Materialized: 40 TB (per the headline claim)

The sparse-GRM path is the practical UKB-scale answer and has its own
benchmark surface (`torchgwas.linalg.sparse_grm`) not exercised here.
