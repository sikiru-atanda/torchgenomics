"""Calibrate haplotype-origin recovery rate for Phase 56 Tier 2 tests.

Not picked up by pytest (no 'test_' prefix). Run once at phase sign-off:

    python bench/calibrate_polyorigin_recovery.py

Prints the observed recovery rate at two depths (30x, 8x). Paste the
numbers into tests/test_phase_polyorigin_e2e.py:
    RECOVERY_30X = <value observed>
    RECOVERY_8X  = <value observed>
"""
from __future__ import annotations

import numpy as np
import torch  # noqa: F401  (needed by downstream torchgwas calls)

N_REPS = 10
N_OFFSPRING = 50
N_MARKERS = 200


def _simulate_f1_reads(depth: int, seed: int):
    """Simulate AD counts for a tetraploid F1 + compute posterior probs.

    Not attempting full realism — just enough to get a calibration number.

    Returns
    -------
    probs : torch.Tensor, shape (n_off + 2, N_MARKERS, 5)
        Posterior dosage probabilities (parents first, then offspring).
    truth_states : np.ndarray, shape (n_off, N_MARKERS)
        Ground-truth joint origin-state indices used to simulate each offspring
        marker.  Compared against ``origin_probs.argmax(dim=-1)`` to compute
        per-marker accuracy.
    sample_ids : list[str]
    variant_ids : list[str]
    """
    # Placeholder for implementer to fill in using updog's own simulator
    # (via torchgwas.preprocess.dosage_call) once Phase 55 and a local
    # updog install are available.
    raise NotImplementedError(
        "Implementer: plug in updog's rgeno + rflexdog-simulated AD, then "
        "run dosage_call.run_updog to derive posterior probs, then feed "
        "them through run_polyorigin with a known ground-truth origin "
        "assignment, then compute haplotype-origin argmax accuracy.\n\n"
        "Concretely:\n"
        "  1. Use updog::rgeno(n_off, ploidy=4) to draw parent genotypes.\n"
        "  2. Use updog::rflexdog(sizevec, ...) to simulate read counts at "
        "the requested sequencing depth.\n"
        "  3. Call torchgwas.preprocess.dosage_call.run_updog(...) on the "
        "simulated AD tensor to obtain posterior dosage probs.\n"
        "  4. Build a synthetic pedigree and marker-map TSV.\n"
        "  5. Call run_polyorigin(probs, ped, map, ...) to obtain a "
        "PhasingResult whose .origin_probs has shape (n_off, m, n_states).\n"
        "  6. Compare origin_probs.argmax(dim=-1) against the ground-truth "
        "origin states drawn in step 1 to get per-marker accuracy."
    )


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
    probs, truth_states, sample_ids, variant_ids = _simulate_f1_reads(depth=depth, seed=seed)
    # truth_states : (n_off, m)  int64 joint-origin-state indices
    # Build synthetic pedigree + map in a temporary directory, then call
    # run_polyorigin, then compare predicted vs truth.
    raise NotImplementedError(
        "Implementer: wire _simulate_f1_reads output through run_polyorigin, "
        "then compute accuracy = (predicted == truth).float().mean().item()."
    )


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
    return float(np.mean(accuracies))


if __name__ == "__main__":
    for depth in (30, 8):
        rate = calibrate(depth=depth, seed=depth)
        print(f"depth={depth}x  recovery={rate:.4f}")
