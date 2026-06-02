"""Profile peak RSS of a TorchGenomics lmm-scan run on the synthetic BED fixture.

Runs the scan in a subprocess, polls memory via psutil at 50ms intervals,
records:
  - peak RSS (MB)
  - peak USS (resident-private; isolates true process memory)
  - wall-time (s)
  - bytes-per-sample, bytes-per-(sample×chunk) for projection comparison

Reports observed vs projected peak. The projected peak is the chunk-size
arithmetic from the streaming-audit doc:

    projected_peak ≈ baseline + n * chunk_size * 8 (genotype chunk)
                              + n^2 * 8           (GRM)
                              + n * 8 * (1 + p_covar)  (null fit state)

For modest n the GRM dominates; for biobank n the chunk term grows but stays
linear-in-chunk-size, NOT linear-in-p_total. The streaming claim succeeds
when observed peak / projected_peak is within ~1.5× (allowing for runtime
overheads, intermediates, and torch's caching allocator).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def poll_memory(pid: int, sample_interval: float, samples: list, stop_event) -> None:
    """Poll RSS + USS of the target process until stop_event is set."""
    proc = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            mem = proc.memory_full_info()
            samples.append((time.time(), mem.rss, mem.uss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(sample_interval)


def measure_baseline_rss() -> dict:
    """Subprocess that just imports torch + torchgenomics and exits. Use this to
    subtract the Python/torch baseline from the scan-attributable memory."""
    cmd = [
        sys.executable, "-c",
        """
import time, torch
import torchgenomics
from torchgenomics.io.plink import PlinkBedReader
from torchgenomics.linalg.kinship import grm_vanraden_streaming
from torchgenomics.models.single_trait_lmm import SingleTraitLMM
from torchgenomics.models.base import VariantMeta
import pandas as pd
print('baseline: imports done', flush=True)
time.sleep(0.5)  # give the polling thread a few samples
"""
    ]
    samples = []
    stop_event = threading.Event()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    poll_thread = threading.Thread(
        target=poll_memory, args=(proc.pid, 0.05, samples, stop_event), daemon=True
    )
    poll_thread.start()
    proc.communicate()
    stop_event.set()
    poll_thread.join(timeout=2.0)
    if not samples:
        return {"peak_rss_mb": 0.0, "peak_uss_mb": 0.0}
    return {
        "peak_rss_mb": max(s[1] for s in samples) / 1e6,
        "peak_uss_mb": max(s[2] for s in samples) / 1e6,
    }


def run_scan(prefix: Path, chunk_size: int, work_dir: Path) -> dict:
    """Subprocess the scan, poll RSS, return summary dict."""
    pheno = prefix.with_suffix(".pheno")
    out_dir = work_dir / "scan_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-c",
        f"""
import time, torch
from torchgenomics.io.plink import PlinkBedReader
from torchgenomics.linalg.kinship import grm_vanraden_streaming
from torchgenomics.models.single_trait_lmm import SingleTraitLMM
from torchgenomics.models.base import VariantMeta
import pandas as pd

t0 = time.time()
print('subprocess: starting', flush=True)
reader = PlinkBedReader('{prefix}')
n, p = reader.n_samples, reader.n_variants
print(f'subprocess: n={{n}}, p={{p}}', flush=True)

# Build GRM via the streaming kinship API (NEVER materialize G).
# grm_vanraden_streaming consumes the chunk iterator directly and only
# holds chunk_size variants in memory at a time.
print(f'subprocess: building GRM via streaming chunks of {chunk_size}', flush=True)
t_grm = time.time()
K, _ = grm_vanraden_streaming(
    chunk_iter=reader.iter_chunks(chunk_size={chunk_size}),
    n_samples=n,
)
print(f'subprocess: GRM built in {{time.time()-t_grm:.2f}}s, K shape={{K.shape}}', flush=True)

# Load phenotype
ph = pd.read_csv('{pheno}', sep='\\t')
Y = torch.from_numpy(ph['Y'].values).reshape(-1, 1).to(torch.float64)
X0 = torch.from_numpy(ph[['PC1','PC2']].values).to(torch.float64)
X0 = torch.cat([torch.ones(n, 1, dtype=torch.float64), X0], dim=1)

# Fit null
print('subprocess: fitting null LMM', flush=True)
model = SingleTraitLMM()
t_null = time.time()
null_fit = model.fit_null(Y, X0, K)
print(f'subprocess: null fit in {{time.time()-t_null:.2f}}s', flush=True)

# Stream chunks through score_chunk
print('subprocess: streaming chunks through score_chunk', flush=True)
t_scan = time.time()
n_processed = 0
reader2 = PlinkBedReader('{prefix}')
for G_chunk, vmeta in reader2.iter_chunks(chunk_size={chunk_size}):
    out = model.score_chunk(G_chunk, null_fit, vmeta)
    n_processed += G_chunk.shape[1]
print(f'subprocess: scanned {{n_processed}} variants in {{time.time()-t_scan:.2f}}s', flush=True)
print(f'subprocess: TOTAL {{time.time()-t0:.2f}}s', flush=True)
"""
    ]

    samples = []
    stop_event = threading.Event()
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    poll_thread = threading.Thread(
        target=poll_memory, args=(proc.pid, 0.05, samples, stop_event), daemon=True
    )
    poll_thread.start()

    stdout, _ = proc.communicate()
    stop_event.set()
    poll_thread.join(timeout=2.0)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(stdout.decode())
        sys.exit(f"FATAL: scan subprocess failed (rc={proc.returncode})")

    if not samples:
        sys.exit("FATAL: no memory samples captured (process exited before polling started)")

    peak_rss = max(s[1] for s in samples)
    peak_uss = max(s[2] for s in samples)
    return {
        "elapsed_s": elapsed,
        "peak_rss_mb": peak_rss / 1e6,
        "peak_uss_mb": peak_uss / 1e6,
        "n_samples": len(samples),
        "stdout": stdout.decode(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", type=Path, required=True,
                   help="BED prefix (no extension); must have .bed/.bim/.fam/.pheno")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/streaming_bench"))
    p.add_argument("--output-json", type=Path,
                   default=Path("validation/streaming_memory/outputs/run.json"))
    args = p.parse_args()

    # Sniff n, p from BIM/FAM for projection arithmetic
    n = sum(1 for _ in open(args.prefix.with_suffix(".fam")))
    p_total = sum(1 for _ in open(args.prefix.with_suffix(".bim")))
    bed_bytes = args.prefix.with_suffix(".bed").stat().st_size
    print(f"[fixture] n={n:,}  p={p_total:,}  BED={bed_bytes/1e6:.1f} MB")
    print(f"[fixture] materialized (n×p×8): {n*p_total*8/1e9:.2f} GB")
    print(f"[fixture] streaming projection (n×chunk×8 + n²×8): "
          f"{(n*args.chunk_size*8 + n*n*8)/1e6:.1f} MB peak")
    print()

    result = run_scan(args.prefix, args.chunk_size, args.work_dir)
    print(f"\n[result] elapsed: {result['elapsed_s']:.2f}s")
    print(f"[result] peak RSS: {result['peak_rss_mb']:.1f} MB")
    print(f"[result] peak USS: {result['peak_uss_mb']:.1f} MB")
    print(f"[result] memory samples: {result['n_samples']}")

    # Projected peak
    projected_mb = (n * args.chunk_size * 8 + n * n * 8 + n * 8 * 5) / 1e6
    materialized_mb = n * p_total * 8 / 1e6
    print(f"\n[projection] streaming projected peak: {projected_mb:.1f} MB")
    print(f"[projection] materialized (would-be) peak: {materialized_mb:.1f} MB")
    print(f"[ratio] observed/projected: {result['peak_uss_mb']/projected_mb:.2f}×")
    print(f"[savings] streaming gives {materialized_mb/result['peak_uss_mb']:.1f}× lower peak vs materialized")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({
        "n": n, "p": p_total, "chunk_size": args.chunk_size,
        "bed_size_mb": bed_bytes / 1e6,
        "materialized_mb": materialized_mb,
        "projected_mb": projected_mb,
        "elapsed_s": result["elapsed_s"],
        "peak_rss_mb": result["peak_rss_mb"],
        "peak_uss_mb": result["peak_uss_mb"],
        "n_memory_samples": result["n_samples"],
        "observed_over_projected": result["peak_uss_mb"] / projected_mb,
        "materialized_over_observed": materialized_mb / result["peak_uss_mb"],
    }, indent=2))
    print(f"\n[output] {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
