#!/usr/bin/env python3
"""Export the synthetic admixed+related fixture (Task 1, Phase 57 Unit A) to
PLINK BED/BIM/FAM for GENESIS (R/SNPRelate/GWASTools) to consume, AND to a
plain ``G.npy`` array so torchgenomics reads the byte-for-byte identical
genotype matrix -- so ``compare.py`` (Task 6) is comparing two pipelines fed
the exact same numbers, not two independently-sampled cohorts.

Why we write our own minimal BED writer (not a PLINK-wrapping library)
-----------------------------------------------------------------------
The fixture already lives in memory as a dense FP64 dosage matrix. Standard
PLINK 1 BED, 2-bit-per-genotype, SNP-major encoding is a ~20-line format;
pulling in a heavyweight genotype-IO dependency for this one write would add
a runtime dependency to a *test harness* for no benefit. The format is fully
specified below and self-verified by an immediate round-trip read-back
(``read_plink_bed``) before this script declares success -- if the 2-bit
codes were ever transposed (e.g. hom-A1 and hom-A2 swapped), the round-trip
assertion below would fail loudly rather than silently flipping every
dosage comparison in Task 6.

PLINK .bed dosage <-> 2-bit code convention (standard PLINK 1 binary format,
see https://www.cog-genomics.org/plink/1.9/formats#bed):

    2-bit code   meaning              dosage (A1 allele count)
    ----------   -------------------  ------------------------
    0b00 (0)     homozygous A1/A1     2.0
    0b01 (1)     missing              NaN
    0b10 (2)     heterozygous A1/A2   1.0
    0b11 (3)     homozygous A2/A2     0.0

"Dosage" here is the A1-allele count, matching the ``.bim`` file's A1 column
(written as "A" below; A2 is written as "G" -- the actual base letters are
arbitrary placeholders since no external allele-frequency reference is used,
only the genotype *category* (hom/het/hom) matters for KING-robust / PC-AiR
/ PC-Relate).

Byte packing: SNP-major mode (mode byte 0x01). Within a byte, sample k's
2-bit code (k = 0..3, 0-indexed within the byte) occupies bits [2k, 2k+1)
(LSB-first: sample 0 in bits 0-1, sample 1 in bits 2-3, ...). Each SNP's row
is padded to a whole number of bytes; padding bits (beyond n samples) are
irrelevant (any reader must stop at n samples per the .fam file) and are
written as 0 here purely for byte determinism.

Round-trip verification
------------------------
Immediately after writing, ``read_plink_bed`` decodes the just-written .bed
back into a dosage matrix using the *inverse* of the same lookup table, and
this script asserts it exactly reproduces the input ``G`` (NaN-aware). This
is the guard against a silently-transposed convention: if a future edit to
the encode or decode table introduced an inconsistency (e.g. swapping which
code means "hom A1" vs "hom A2"), this assertion fails at export time,
before any GENESIS run ever touches the file.

Outputs (all under ``validation/external/genesis/out/``)
----------------------------------------------------------
- ``fixture.bed`` / ``fixture.bim`` / ``fixture.fam`` -- PLINK 1 binary
  triple, for ``SNPRelate::snpgdsBED2GDS``.
- ``G.npy`` -- the identical (n, m) FP64 dosage matrix, for torchgenomics.
- ``meta.npz`` -- ``pop_label`` (n,), ``related_pairs`` (k, 2) parent/child
  index pairs, ``pruned_idx`` (the LD-pruned marker subset selected by
  ``torchgenomics.linalg.kinship_admixed.ld_prune_independent`` at the
  fixture's default ``r2_threshold=0.1``), ``sample_ids`` (n,) and
  ``snp_ids`` (m,) matching the .fam/.bim row order exactly.
- ``pruned_snp_ids.txt`` -- one SNP ID per line, the same pruned subset as
  ``pruned_idx``/``meta.npz``, in a format ``run_genesis.R`` can read
  directly to restrict PC-AiR/PC-Relate to the identical marker panel that
  torchgenomics uses (so PC-AiR/PC-Relate are compared on the same pruned
  input on both sides, while KING-robust is compared on the full unpruned
  panel both tools ingest from the same .bed file -- see ``compare.py`` and
  ``run_genesis.R`` docstrings for why the two comparisons use different
  marker sets deliberately).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# Repo root is 3 levels up from validation/external/genesis/.
REPO_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.admixed.make_admixed import make_admixed  # noqa: E402
from torchgenomics.linalg.kinship_admixed import ld_prune_independent  # noqa: E402

# --- Fixture parameters (deterministic; Task 6 harness default panel) -------
N_PER_POP = 60
N_RELATED_PAIRS = 15
M = 2000
FST = 0.15
SEED = 42
R2_PRUNE_THRESHOLD = 0.1  # matches ld_prune_independent's own default

# --- PLINK .bed 2-bit code convention (see module docstring) ----------------
CODE_HOM_A1 = 0b00  # dosage 2.0 (A1/A1)
CODE_MISSING = 0b01  # dosage NaN
CODE_HET = 0b10  # dosage 1.0 (A1/A2)
CODE_HOM_A2 = 0b11  # dosage 0.0 (A2/A2)

# Decode lookup table indexed by the raw 2-bit integer (0..3).
_CODE_TO_DOSAGE = np.array([2.0, np.nan, 1.0, 0.0], dtype=np.float64)


def _dosage_col_to_codes(col: np.ndarray) -> np.ndarray:
    """Encode one SNP's (n,) dosage column to (n,) uint8 2-bit codes.

    Raises if a value outside {0.0, 1.0, 2.0, NaN} is encountered -- this is
    a diploid-only writer (PC-AiR/PC-Relate/KING-robust as implemented in
    ``kinship_admixed.py`` are diploid; the synthetic fixture never emits
    fractional or out-of-range dosages, so any such value indicates caller
    error, not a legitimate encoding to silently coerce).
    """
    is_nan = np.isnan(col)
    is0 = col == 0.0
    is1 = col == 1.0
    is2 = col == 2.0
    bad = ~(is_nan | is0 | is1 | is2)
    if np.any(bad):
        bad_idx = np.where(bad)[0]
        raise ValueError(
            f"non-{{0,1,2,NaN}} diploid dosage at sample rows "
            f"{bad_idx[:5].tolist()}: values {col[bad_idx[:5]].tolist()}"
        )
    codes = np.full(col.shape, CODE_HOM_A1, dtype=np.uint8)  # default: dosage 2
    codes[is1] = CODE_HET
    codes[is0] = CODE_HOM_A2
    codes[is_nan] = CODE_MISSING
    return codes


def write_plink_bed(
    G: np.ndarray,
    prefix: Path,
    sample_ids: list[str],
    snp_ids: list[str],
    chrom: str = "1",
    a1: str = "A",
    a2: str = "G",
) -> None:
    """Write ``{prefix}.bed/.bim/.fam`` from a dense (n, m) dosage matrix.

    See module docstring for the exact 2-bit code <-> dosage convention and
    the SNP-major byte-packing layout.
    """
    n, m = G.shape
    assert len(sample_ids) == n
    assert len(snp_ids) == m

    # --- .fam: FID IID PID MID SEX PHENOTYPE (6 cols; PLINK standard) -------
    with open(f"{prefix}.fam", "w") as fh:
        for sid in sample_ids:
            fh.write(f"{sid}\t{sid}\t0\t0\t0\t-9\n")

    # --- .bim: chr snp_id cm bp A1 A2 (6 cols) -------------------------------
    with open(f"{prefix}.bim", "w") as fh:
        for j, snp in enumerate(snp_ids):
            bp = j + 1
            fh.write(f"{chrom}\t{snp}\t0\t{bp}\t{a1}\t{a2}\n")

    # --- .bed: magic (0x6c 0x1b 0x01 = SNP-major) + 2-bit-packed genotypes --
    n_bytes_per_snp = (n + 3) // 4
    pad = n_bytes_per_snp * 4 - n
    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        for j in range(m):
            codes = _dosage_col_to_codes(G[:, j])
            if pad:
                codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
            codes4 = codes.reshape(n_bytes_per_snp, 4)
            byte_vals = (
                codes4[:, 0]
                | (codes4[:, 1] << 2)
                | (codes4[:, 2] << 4)
                | (codes4[:, 3] << 6)
            ).astype(np.uint8)
            fh.write(byte_vals.tobytes())


def read_plink_bed(prefix: Path, n: int, m: int) -> np.ndarray:
    """Read back ``{prefix}.bed`` into a dense (n, m) FP64 dosage matrix.

    Inverse of ``write_plink_bed``'s encoding; used only for the immediate
    self-verification round-trip at the bottom of this script (NOT part of
    the GENESIS/torchgenomics comparison pipeline -- both of those read the
    genotypes from their own native paths, G.npy and SNPRelate respectively).
    """
    n_bytes_per_snp = (n + 3) // 4
    with open(f"{prefix}.bed", "rb") as fh:
        magic = fh.read(3)
        if magic != bytes([0x6C, 0x1B, 0x01]):
            raise ValueError(f"unexpected PLINK magic bytes: {magic!r}")
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
    raw = raw.reshape(m, n_bytes_per_snp)
    G = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        row = raw[j]
        codes = np.empty(n_bytes_per_snp * 4, dtype=np.uint8)
        codes[0::4] = row & 0b11
        codes[1::4] = (row >> 2) & 0b11
        codes[2::4] = (row >> 4) & 0b11
        codes[3::4] = (row >> 6) & 0b11
        G[:, j] = _CODE_TO_DOSAGE[codes[:n]]
    return G


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print(
        f"[export_fixture] generating admixed fixture: n_per_pop={N_PER_POP} "
        f"n_related_pairs={N_RELATED_PAIRS} m={M} fst={FST} seed={SEED}"
    )
    d = make_admixed(
        n_per_pop=N_PER_POP,
        n_related_pairs=N_RELATED_PAIRS,
        m=M,
        fst=FST,
        seed=SEED,
    )
    G_t: torch.Tensor = d["G"]  # (n, m) FP64, values in {0,1,2}, no NaN
    n, m = G_t.shape
    G = G_t.numpy().astype(np.float64)
    print(f"[export_fixture] G shape = {G.shape}, dtype = {G.dtype}")

    sample_ids = [f"IND{i:04d}" for i in range(n)]
    snp_ids = [f"snp{j:04d}" for j in range(m)]

    prefix = OUT / "fixture"
    print(f"[export_fixture] writing PLINK BED/BIM/FAM -> {prefix}.{{bed,bim,fam}}")
    write_plink_bed(G, prefix, sample_ids, snp_ids)

    # --- Round-trip self-check: guards against a transposed dosage<->code --
    print("[export_fixture] round-trip verification: reading back fixture.bed")
    G_roundtrip = read_plink_bed(prefix, n, m)
    if not np.allclose(G_roundtrip, G, equal_nan=True):
        max_abs_diff = np.nanmax(np.abs(G_roundtrip - G))
        raise AssertionError(
            "PLINK BED round-trip MISMATCH: the dosage<->2-bit-code "
            f"convention is not self-consistent (max |diff| = {max_abs_diff}). "
            "This would silently transpose every downstream GENESIS "
            "comparison -- do not proceed until this passes."
        )
    print("[export_fixture] round-trip OK: read-back dosages == written dosages")

    print(f"[export_fixture] writing G.npy -> {OUT / 'G.npy'}")
    np.save(OUT / "G.npy", G)

    # --- LD-pruned marker subset (shared between torchgenomics and GENESIS) -
    pruned_idx_t = ld_prune_independent(G_t, r2_threshold=R2_PRUNE_THRESHOLD)
    pruned_idx = pruned_idx_t.numpy()
    print(
        f"[export_fixture] LD-pruned marker subset: {pruned_idx.size} / {m} "
        f"kept (r2_threshold={R2_PRUNE_THRESHOLD})"
    )
    pruned_snp_ids = [snp_ids[j] for j in pruned_idx.tolist()]
    with open(OUT / "pruned_snp_ids.txt", "w") as fh:
        fh.write("\n".join(pruned_snp_ids) + "\n")

    related_pairs = np.array(
        [(p, c) for p, c, _ in d["related_pairs"]], dtype=np.int64
    )
    print(f"[export_fixture] writing meta.npz -> {OUT / 'meta.npz'}")
    np.savez(
        OUT / "meta.npz",
        pop_label=d["pop_label"].numpy(),
        related_pairs=related_pairs,
        pruned_idx=pruned_idx,
        sample_ids=np.array(sample_ids),
        snp_ids=np.array(snp_ids),
        n_per_pop=N_PER_POP,
        n_related_pairs=N_RELATED_PAIRS,
        m=M,
        fst=FST,
        seed=SEED,
        r2_prune_threshold=R2_PRUNE_THRESHOLD,
    )

    print("[export_fixture] done.")
    print(f"[export_fixture]   {prefix}.bed/.bim/.fam")
    print(f"[export_fixture]   {OUT / 'G.npy'}")
    print(f"[export_fixture]   {OUT / 'meta.npz'}")
    print(f"[export_fixture]   {OUT / 'pruned_snp_ids.txt'}")


if __name__ == "__main__":
    main()
