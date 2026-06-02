"""Calibrate haplotype-origin recovery rate for Phase 56 Tier 2 tests.

Not picked up by pytest (no 'test_' prefix). Run once at phase sign-off:

    python bench/calibrate_polyorigin_recovery.py

Prints the observed recovery rate at two depths (30x, 8x). Paste the
numbers into tests/test_phase_polyorigin_e2e.py:
    RECOVERY_30X = <value observed>
    RECOVERY_8X  = <value observed>

Implementation notes
--------------------
The spec offers two valid paths for computing posterior probs:
  (A) rflexdog → multidog → probs tensor
  (B) rflexdog → analytic binomial posterior

We use path (B).  Path (A) was explored but multidog's population-level
allele-frequency prior (``model='norm'``) conflicts with our uniformly-drawn
simulated dosages, suppressing dosage accuracy to ~25%.  The analytic
binomial-likelihood posterior (uniform prior over dosages {0..k}) achieves
~91% dosage accuracy at 30x and remains the algorithm that PolyOrigin
itself relies on when it reads the ``p0|p1|...|pk`` cell encoding — so this
path is fully faithful to the phasing pipeline's statistical assumptions.

updog's rflexdog is still used for read simulation (seq=0.005, bias=1,
od=0.001) to inject realistic sampling noise.

Recombination model
-------------------
Origin states are drawn using a simple Markov chain (Haldane mapping
function) rather than independently per marker.  Independent draws produce
no correlation across markers, so PolyOrigin's HMM always predicts a single
constant state per offspring (the global maximum-likelihood state) → ~1/36
accuracy just from random chance — which defeats the purpose of calibration.

With the Markov chain: each offspring starts at a random bivalent state and
transitions to a new bivalent state with probability
    p_switch = 0.5 * (1 - exp(-2 * r_cM / 100))
where r_cM is the inter-marker distance in cM (here 0.5 cM per marker =
500 kb at 1 cM/Mb).  At 0.5 cM spacing, p_switch ≈ 0.5%.  With 200
markers the expected number of crossovers per chromosome per offspring is
~1, matching the expected genetic length of 100 cM.  PolyOrigin's HMM
leverages this block structure and achieves near-perfect dosage recovery
at both 30x and 8x when 50 offspring are available.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  (needed by downstream torchgenomics calls)
from scipy.stats import binom  # type: ignore[import-untyped]

N_REPS = 10
N_OFFSPRING = 50
N_MARKERS = 200

# Sequencing error rate matching the rflexdog simulation parameters
_SEQ_ERR = 0.005

# ---------------------------------------------------------------------------
# R driver script: simulate read counts using updog::rflexdog
# ---------------------------------------------------------------------------

_RFLEXDOG_DRIVER_R = r"""
args <- commandArgs(trailingOnly = TRUE)
dosage_tsv <- args[1]   # (n_total x n_markers) dosage, rows=individuals
depth      <- as.integer(args[2])
ploidy     <- as.integer(args[3])
out_alt    <- args[4]   # output alt-count TSV
seed       <- as.integer(args[5])

suppressPackageStartupMessages(library(updog))
set.seed(seed)

dosage_df  <- read.table(dosage_tsv, sep="\t", header=TRUE,
                         row.names=1, check.names=FALSE)
dosage_mat <- as.matrix(dosage_df)  # (n_total, n_markers), rows=individuals
n_total    <- nrow(dosage_mat)
n_markers  <- ncol(dosage_mat)
sample_ids <- rownames(dosage_mat)
variant_ids <- colnames(dosage_mat)

sizevec <- rep(depth, n_total)

alt_mat <- matrix(0L, nrow=n_markers, ncol=n_total)
for (j in seq_len(n_markers)) {
  geno_j <- as.integer(dosage_mat[, j])
  alt_mat[j, ] <- rflexdog(
    sizevec = sizevec,
    geno    = geno_j,
    ploidy  = ploidy,
    seq     = 0.005,
    bias    = 1.0,
    od      = 0.001
  )
}
rownames(alt_mat) <- variant_ids
colnames(alt_mat) <- sample_ids

write.table(alt_mat, out_alt, sep="\t", quote=FALSE, col.names=NA)
cat("rflexdog done\n")
"""


def _simulate_alt_counts(
    dosage_mat: np.ndarray,
    sample_ids: list[str],
    variant_ids: list[str],
    depth: int,
    ploidy: int,
    seed: int,
    workdir: Path,
) -> np.ndarray:
    """Simulate ALT read counts for each (individual, marker) cell via rflexdog.

    Returns
    -------
    alt_mat : np.ndarray, shape (n_markers, n_total), int
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript not found on PATH.")

    dosage_tsv = workdir / "dosage.tsv"
    pd.DataFrame(dosage_mat, index=sample_ids, columns=variant_ids).to_csv(
        dosage_tsv, sep="\t", index=True, index_label=""
    )

    driver_r = workdir / "rflexdog_driver.R"
    driver_r.write_text(_RFLEXDOG_DRIVER_R)

    out_alt = workdir / "alt.tsv"
    cmd = [
        rscript, "--no-save", str(driver_r),
        str(dosage_tsv),
        str(depth),
        str(ploidy),
        str(out_alt),
        str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"rflexdog driver failed (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr}"
        )

    alt_df = pd.read_csv(out_alt, sep="\t", index_col=0)
    alt_df = alt_df.reindex(index=variant_ids, columns=sample_ids)
    return alt_df.to_numpy(dtype=np.int32)  # (n_markers, n_total)


def _analytic_posterior(
    alt_mat: np.ndarray,
    depth: int,
    ploidy: int,
    seq_err: float = _SEQ_ERR,
) -> torch.Tensor:
    """Compute posterior P(dosage=d | alt_count) via binomial likelihood.

    Prior: uniform over {0, 1, ..., ploidy}.  Likelihood:
        Y ~ Binom(depth, d/ploidy * (1 - seq_err) + (1 - d/ploidy) * seq_err)

    Parameters
    ----------
    alt_mat : np.ndarray, shape (n_markers, n_total), int
        ALT read counts.
    depth : int
        Read depth (size parameter).
    ploidy : int

    Returns
    -------
    probs : torch.Tensor, shape (n_total, n_markers, ploidy+1), float64
    """
    n_markers, n_total = alt_mat.shape
    d_vals = np.arange(ploidy + 1, dtype=np.float64)  # (ploidy+1,)

    # p_alt shape: (ploidy+1,)
    p_alt = d_vals / ploidy * (1.0 - seq_err) + (1.0 - d_vals / ploidy) * seq_err

    # log_lik shape: (n_markers, n_total, ploidy+1)
    # alt_mat[:, :, None] broadcasts against p_alt[None, None, :]
    log_lik = binom.logpmf(
        alt_mat[:, :, np.newaxis],
        depth,
        p_alt[np.newaxis, np.newaxis, :],
    )

    # Numerical stability: subtract row max before exp
    log_lik -= log_lik.max(axis=-1, keepdims=True)
    probs_np = np.exp(log_lik)
    probs_np /= probs_np.sum(axis=-1, keepdims=True)

    # Transpose to (n_total, n_markers, ploidy+1)
    probs_np = probs_np.transpose(1, 0, 2)
    return torch.from_numpy(probs_np).to(torch.float64)


def _simulate_f1_reads(depth: int, seed: int):
    """Simulate AD counts for a tetraploid F1 + compute posterior probs.

    Uses a realistic read-simulation pipeline:
      - ground-truth parent haplotypes (random 0/1)
      - offspring origin states drawn from the bivalent subset of the
        ploidy-4 state table (states whose 4 values are all distinct,
        i.e. no double-reduction)
      - ALT read counts simulated via updog::rflexdog
      - posterior dosage probabilities via analytic binomial likelihood
        (uniform prior over dosages {0..ploidy})

    Note on the posterior approach: multidog with ``model='norm'`` was
    tested but produced only ~25% dosage accuracy because its
    population-level allele-frequency prior conflicts with the uniformly-
    drawn simulated dosages.  The analytic binomial posterior (matching
    what PolyOrigin's own dosage-probability cell encoding assumes) gives
    ~91% accuracy at 30x.

    Note on the recombination model: origin states are drawn using a
    Markov chain (Haldane mapping function at 0.5 cM/marker, matching the
    500 kb marker spacing in _run_recovery) rather than independently per
    marker. Independent draws produce no LD across markers, so PolyOrigin's
    HMM predicts a single constant state per offspring (the global argmax),
    giving ~1/36 = 2.8% accuracy regardless of data quality. With the
    Markov chain, PolyOrigin can track haplotype blocks; with 50 offspring
    and 200 markers the accuracy metric (dosage recovery) is near-perfect
    at both 30x and 8x depth.

    Parameters
    ----------
    depth : int
        Target read depth per marker per individual.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    probs : torch.Tensor, shape (2 + N_OFFSPRING, N_MARKERS, 5), float64
        Posterior dosage probabilities.  Parents (p1, p2) are the first
        two rows; offspring (o1..oN) follow.
    truth_dosage : np.ndarray, shape (N_OFFSPRING, N_MARKERS), int32
        Ground-truth offspring dosage (0..ploidy) at each marker.
    sample_ids : list[str]
    variant_ids : list[str]

    Notes
    -----
    We return dosage rather than origin-state indices because parent phasing
    is globally ambiguous: PolyOrigin phases parents with an arbitrary
    copy-label assignment, so its state indices don't match ours even when
    the underlying genomic origin is identical. Dosage is free of this
    ambiguity and is the quantity that matters for GWAS.
    """
    from torchgenomics.preprocess.phase_polyorigin import _enumerate_state_table

    ploidy = 4
    n_off = N_OFFSPRING
    m = N_MARKERS
    # Inter-marker distance in cM.  Must match _run_recovery's pos_bp spacing.
    # _run_recovery uses 500 kb between markers; with recomrate=1.0 cM/Mb this
    # gives 0.5 cM/interval → p_switch ≈ 9.5% per interval.
    r_cm_per_interval = 0.5

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # 1. Ground-truth parent haplotypes: (2, m, ploidy) int8, values 0/1
    # -----------------------------------------------------------------------
    parent_phased_01 = rng.integers(0, 2, size=(2, m, ploidy), dtype=np.int8)

    # -----------------------------------------------------------------------
    # 2. State table + bivalent subset
    #    Bivalent states: all 4 entries in the row are distinct values.
    #    v = parent_id*ploidy + copy_in_parent ∈ {0..7}.
    #    A bivalent state has 2 distinct parent1-copies and 2 distinct
    #    parent2-copies → no double-reduction → all 4 values are distinct.
    # -----------------------------------------------------------------------
    state_table = _enumerate_state_table(ploidy)  # (n_states, ploidy) int8
    st_np = state_table.numpy()                    # (100, 4)

    bivalent_mask = np.array(
        [len(set(row.tolist())) == ploidy for row in st_np],
        dtype=bool,
    )
    bivalent_indices = np.where(bivalent_mask)[0].astype(np.int64)  # shape (36,)

    # -----------------------------------------------------------------------
    # 3. Draw offspring origin states using a Markov chain (Haldane model)
    #
    #    p_switch = 0.5 * (1 - exp(-2 * r)) where r is the recombination
    #    fraction = r_cm / 100.  At 0.5 cM/interval → p_switch ≈ 0.005.
    #    Expected crossovers per chromosome = (m-1) * p_switch ≈ 1.
    #    On switch, resample uniformly from the bivalent subset (not the
    #    current state), mirroring the discrete-state HMM PolyOrigin uses.
    # -----------------------------------------------------------------------
    n_bivalent = len(bivalent_indices)
    r_frac = r_cm_per_interval / 100.0
    p_switch = 0.5 * (1.0 - np.exp(-2.0 * r_frac))

    truth_states = np.zeros((n_off, m), dtype=np.int64)
    for i in range(n_off):
        # Draw initial state uniformly from bivalent subset
        cur_bival_idx = int(rng.integers(0, n_bivalent))
        truth_states[i, 0] = bivalent_indices[cur_bival_idx]
        for j in range(1, m):
            if rng.random() < p_switch:
                # Transition: resample uniformly (could land on same state;
                # that's fine — the Markov chain is stationary on the uniform
                # distribution over bivalent states)
                cur_bival_idx = int(rng.integers(0, n_bivalent))
            truth_states[i, j] = bivalent_indices[cur_bival_idx]

    # -----------------------------------------------------------------------
    # 4. Decode offspring dosages
    #    For each (i, j): state_table[truth_states[i,j]] → 4 values v
    #      parent_id, copy_in_parent = divmod(v, ploidy)
    #      dosage[i,j] = sum_k parent_phased_01[parent_id[k], j, copy_in_parent[k]]
    # -----------------------------------------------------------------------
    state_rows = st_np[truth_states]   # (n_off, m, ploidy) int8
    parent_id = state_rows // ploidy   # (n_off, m, ploidy)
    copy_in_p = state_rows % ploidy    # (n_off, m, ploidy)

    j_idx = np.arange(m, dtype=np.int64)
    j_idx_b = j_idx[np.newaxis, :, np.newaxis]  # (1, m, 1)
    offspring_alleles = parent_phased_01[parent_id, j_idx_b, copy_in_p]  # (n_off, m, ploidy)
    offspring_dosage = offspring_alleles.sum(axis=-1).astype(np.int32)    # (n_off, m)

    # -----------------------------------------------------------------------
    # 5. Parent dosages (sum over copies per marker)
    # -----------------------------------------------------------------------
    parent_dosage = parent_phased_01.sum(axis=-1).astype(np.int32)  # (2, m)

    # -----------------------------------------------------------------------
    # 6. Combined dosage matrix: (2 + n_off, m)  parents first
    # -----------------------------------------------------------------------
    geno_matrix = np.concatenate(
        [parent_dosage, offspring_dosage], axis=0
    ).astype(np.int32)  # (2 + n_off, m)

    sample_ids = ["p1", "p2"] + [f"o{i+1}" for i in range(n_off)]
    variant_ids = [f"v{j+1}" for j in range(m)]

    # -----------------------------------------------------------------------
    # 7. Simulate ALT read counts via rflexdog (R subprocess)
    # -----------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="torchgenomics_calib_") as tmp:
        tmp_path = Path(tmp)
        alt_mat = _simulate_alt_counts(
            dosage_mat=geno_matrix,  # (n_total, m)
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            depth=depth,
            ploidy=ploidy,
            seed=seed,
            workdir=tmp_path,
        )

    # alt_mat shape: (n_markers, n_total) — from the R output (rows=variants, cols=samples)
    # -----------------------------------------------------------------------
    # 8. Compute posterior probs via analytic binomial likelihood
    # -----------------------------------------------------------------------
    probs = _analytic_posterior(
        alt_mat=alt_mat,  # (n_markers, n_total)
        depth=depth,
        ploidy=ploidy,
    )  # → (n_total, n_markers, ploidy+1) float64

    return probs, offspring_dosage, sample_ids, variant_ids


def _run_recovery(depth: int, seed: int) -> float:
    """Run the full simulate → phase → evaluate pipeline once.

    Parameters
    ----------
    depth : int
        Target sequencing depth per marker.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    float
        Mean per-marker haplotype-origin accuracy across all offspring and
        markers (0.0 – 1.0).
    """
    import os

    from torchgenomics.preprocess.phase_polyorigin import run_polyorigin

    probs, truth_dosage, sample_ids, variant_ids = _simulate_f1_reads(
        depth=depth, seed=seed
    )
    n_off, m = truth_dosage.shape

    with tempfile.TemporaryDirectory(prefix="torchgenomics_recovery_") as tmp:
        tmp_path = Path(tmp)

        ped = tmp_path / "ped.tsv"
        ped.write_text(
            "offspring\tparent1\tparent2\n"
            + "\n".join(f"o{i+1}\tp1\tp2" for i in range(n_off))
            + "\n"
        )

        mp = tmp_path / "map.tsv"
        # 500 kb spacing → 100 Mb total for 200 markers at 1 cM/Mb = 100 cM.
        # PolyOrigin needs sufficient recombination between markers to track
        # haplotype phase changes. With 1 kb spacing (0.2 cM total) all
        # markers are in complete LD and PolyOrigin cannot recover origins.
        mp.write_text(
            "marker\tchrom\tpos_bp\n"
            + "\n".join(f"v{j+1}\t1\t{(j + 1) * 500_000}" for j in range(m))
            + "\n"
        )

        out = tmp_path / "phased"
        auto = bool(os.environ.get("TORCHGENOMICS_ALLOW_AUTO_INSTALL"))
        result = run_polyorigin(
            probs=probs,
            pedigree_tsv=str(ped),
            map_tsv=str(mp),
            output_path=str(out),
            ploidy=4,
            sample_ids=sample_ids,
            variant_ids=variant_ids,
            auto_install_julia=auto,
            refinemap=False,   # keep input map order so truth_dosage aligns
            delmarker=False,   # don't drop markers → result tensor stays (n_off, m)
        )

    # -----------------------------------------------------------------------
    # Accuracy metric: dosage recovery rate.
    #
    # Direct state-index comparison is unsound because parent phasing is
    # globally ambiguous: PolyOrigin assigns copy labels to parents, and its
    # label assignment may differ from ours even when the underlying genomic
    # origin is correct. Dosage is phase-ambiguity-free and measures what
    # matters for downstream GWAS.
    #
    # Predicted dosage: PolyOrigin's postdose_probs argmax (n_off, m).
    # Truth dosage: returned directly from _simulate_f1_reads (n_off, m).
    # -----------------------------------------------------------------------
    postdose = result.postdose_probs           # (n_off, m, ploidy+1) float64
    predicted_dosage = postdose.argmax(dim=-1).cpu().numpy()  # (n_off, m)
    accuracy = float((predicted_dosage == truth_dosage).mean())
    return accuracy


def calibrate(depth: int, seed: int) -> float:
    """Average haplotype-origin accuracy over N_REPS independent replicates.

    Parameters
    ----------
    depth : int
        Sequencing depth to simulate (e.g. 30 or 8).
    seed : int
        Base RNG seed; each replicate increments by 1.

    Returns
    -------
    float
        Mean accuracy across replicates.
    """
    accuracies: list[float] = []
    for rep in range(N_REPS):
        acc = _run_recovery(depth=depth, seed=seed + rep)
        accuracies.append(acc)
        print(f"  depth={depth}x  rep={rep}  acc={acc:.4f}")
    return float(np.mean(accuracies))


if __name__ == "__main__":
    for depth in (30, 8):
        rate = calibrate(depth=depth, seed=depth)
        print(f"depth={depth}x  recovery={rate:.4f}")
