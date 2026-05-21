"""LRO (Leave-Region-Out LMM) internal-consistency simulator.

Phase 29 — no external reference tool exists for block-level proximal
decontamination LMM. This harness validates
``torchgwas.models.lro_lmm.LROLMM`` by:

  1. Simulating genotypes in K disjoint LD blocks, with one causal SNP
     planted inside one of the blocks (the "causal block"),
  2. Running both standard LMM (genome-wide GRM) and LROLMM (per-block
     leave-this-block-out GRM) on the SAME data,
  3. Asserting LRO produces a LARGER |z| / SMALLER p at the causal SNP
     than standard LMM (because proximal contamination from the
     causal SNP into the full-K GRM down-weights its effect in the
     standard test; LRO removes that contamination).

Why internal-consistency, not head-to-head:
  Per spec § 10.6, LROLMM is Phase 29 (post-V1 specialty model) and
  has no widely-used external reference. The proximal-decontamination
  contract is the scientific core; this harness gates against future
  regressions that would silently break it.

F7 panel LRO renders with a distinct border indicating "no external
reference; internal-consistency validation only."
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from torchgwas.linalg.kinship import grm_vanraden  # noqa: E402
from torchgwas.models.base import VariantMeta  # noqa: E402
from torchgwas.models.lro_lmm import LROLMM  # noqa: E402
from torchgwas.models.single_trait_lmm import SingleTraitLMM  # noqa: E402


SEED = 42
N = 500
N_BLOCKS = 5
SNPS_PER_BLOCK = 20      # Total m = 100
CAUSAL_BLOCK = 2          # 0-indexed; block 2 carries the causal SNP
CAUSAL_SNP_OFFSET = 7     # Position within the causal block
BETA_CAUSAL = 1.0         # Large effect — proximal contamination should be measurable
LD_RHO = 0.85             # Within-block AR(1)-like LD; high LD makes contamination obvious


def _simulate(seed: int = SEED) -> dict:
    """Simulate the LRO fixture with explicit LD-block structure."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    m = N_BLOCKS * SNPS_PER_BLOCK
    G = torch.zeros(N, m, dtype=torch.float64)

    # For each block, draw a latent block-level haplotype via AR(1) on a
    # standard normal, then threshold at MAF=0.30 to produce a {0,1,2}
    # dosage. The within-block correlation gives realistic LD; across-
    # block independence ensures the LRO leave-out is meaningful.
    for b in range(N_BLOCKS):
        # Latent normal series per individual.
        z = torch.zeros(N, SNPS_PER_BLOCK, dtype=torch.float64)
        z[:, 0] = torch.randn(N)
        for j in range(1, SNPS_PER_BLOCK):
            z[:, j] = LD_RHO * z[:, j - 1] + (
                (1.0 - LD_RHO ** 2) ** 0.5
            ) * torch.randn(N)
        # Threshold each chromosome copy independently to make {0,1,2}.
        prob_alt = 0.30
        thresh = torch.distributions.Normal(0, 1).icdf(
            torch.tensor(1.0 - prob_alt, dtype=torch.float64)
        )
        # Two independent chromosome copies (we re-draw z per copy to
        # avoid perfect within-block phasing).
        copy1 = (z > thresh).to(torch.float64)
        z2 = torch.zeros_like(z)
        z2[:, 0] = torch.randn(N)
        for j in range(1, SNPS_PER_BLOCK):
            z2[:, j] = LD_RHO * z2[:, j - 1] + (
                (1.0 - LD_RHO ** 2) ** 0.5
            ) * torch.randn(N)
        copy2 = (z2 > thresh).to(torch.float64)
        G[:, b * SNPS_PER_BLOCK : (b + 1) * SNPS_PER_BLOCK] = copy1 + copy2

    # Plant a causal SNP inside CAUSAL_BLOCK.
    causal_idx = CAUSAL_BLOCK * SNPS_PER_BLOCK + CAUSAL_SNP_OFFSET
    truth_beta = torch.zeros(m, dtype=torch.float64)
    truth_beta[causal_idx] = BETA_CAUSAL

    # Phenotype = causal_signal + residual (fixed σ=1).
    genetic = G @ truth_beta
    epsilon = torch.randn(N, dtype=torch.float64)
    Y = genetic + epsilon
    X0 = torch.ones(N, 1, dtype=torch.float64)

    variant_pos = list(range(1, m + 1))  # 1bp apart so within-block proximity is real
    variant_chr = ["1"] * m
    snp_ids = [f"rs{i:05d}" for i in range(m)]
    a1 = ["A"] * m
    a2 = ["G"] * m

    return dict(
        Y=Y, X0=X0, G=G,
        causal_idx=causal_idx, causal_block=CAUSAL_BLOCK,
        truth_beta=truth_beta, m=m,
        variant_pos=variant_pos, variant_chr=variant_chr,
        snp_ids=snp_ids, a1=a1, a2=a2,
    )


def _run_standard_lmm(sim: dict) -> dict:
    """Run standard LMM with the full genome-wide GRM (proximal contamination
    is present here — the causal SNP is in the GRM used for its own test)."""
    K, _ = grm_vanraden(sim["G"])
    model = SingleTraitLMM()
    null = model.fit_null(Y=sim["Y"], X0=sim["X0"], K=K)
    vmeta = VariantMeta(
        snp=sim["snp_ids"], chr=sim["variant_chr"], pos=sim["variant_pos"],
        a1=sim["a1"], a2=sim["a2"],
    )
    res = model.score_chunk(G_chunk=sim["G"], null_fit=null, variant_meta=vmeta,
                            test="wald")
    return dict(
        beta=np.asarray(res.beta),
        se=np.asarray(res.se),
        stat=np.asarray(res.stat),
        p=np.asarray(res.p),
    )


def _run_lro(sim: dict) -> dict:
    """Run LROLMM — uses leave-this-block-out GRM per block, removing the
    proximal contamination from the causal SNP for its own test."""
    model = LROLMM(ld_method="r2")
    vmeta = VariantMeta(
        snp=sim["snp_ids"], chr=sim["variant_chr"], pos=sim["variant_pos"],
        a1=sim["a1"], a2=sim["a2"],
    )
    res = model.run(
        Y=sim["Y"], X0=sim["X0"], G=sim["G"],
        variant_meta=vmeta,
        variant_pos=sim["variant_pos"],
        variant_chr=sim["variant_chr"],
        test="wald",
    )
    return dict(
        beta=np.asarray(res.beta),
        se=np.asarray(res.se),
        stat=np.asarray(res.stat),
        p=np.asarray(res.p),
        n_blocks=res.n_blocks,
        block_sizes=res.block_sizes,
    )


def _hash_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out_results = HERE / "results"
    out_results.mkdir(parents=True, exist_ok=True)
    sim = _simulate()
    causal_idx = sim["causal_idx"]

    print(f"LRO fixture: n={N}, m={sim['m']}, n_blocks={N_BLOCKS}, "
          f"causal SNP at idx {causal_idx} (block {sim['causal_block']})")
    print("Running standard LMM (full-K, proximal contamination present)...")
    res_lmm = _run_standard_lmm(sim)
    print("Running LROLMM (leave-causal-block-out K)...")
    res_lro = _run_lro(sim)
    print(f"  LRO detected {res_lro['n_blocks']} blocks "
          f"(avg {np.mean(res_lro['block_sizes']):.1f} SNPs/block)")

    # ── Invariants ──

    # Invariant 1: both methods recover beta_hat ≈ BETA_CAUSAL at the causal SNP
    # (with sign — LRO unbiased; full-K may be slightly attenuated).
    beta_lmm = float(res_lmm["beta"][causal_idx])
    beta_lro = float(res_lro["beta"][causal_idx])
    rel_err_lmm = abs(beta_lmm - BETA_CAUSAL) / BETA_CAUSAL
    rel_err_lro = abs(beta_lro - BETA_CAUSAL) / BETA_CAUSAL

    # Invariant 2: LRO p < standard LMM p at the causal SNP (proximal
    # contamination is removed in LRO → larger stat → smaller p).
    p_lmm_causal = float(res_lmm["p"][causal_idx])
    p_lro_causal = float(res_lro["p"][causal_idx])
    stat_lmm_causal = float(res_lmm["stat"][causal_idx])
    stat_lro_causal = float(res_lro["stat"][causal_idx])

    # Invariant 3: at non-causal SNPs, both methods report null-like p
    # (median over non-causal SNPs).
    noncausal_mask = np.ones(sim["m"], dtype=bool)
    noncausal_mask[causal_idx] = False
    median_p_noncausal_lmm = float(np.median(res_lmm["p"][noncausal_mask]))
    median_p_noncausal_lro = float(np.median(res_lro["p"][noncausal_mask]))

    # Invariant 4: the LRO block partition includes the causal SNP in
    # exactly one block (the simulator's intent).
    # We don't verify the block id explicitly (LRO's r2 detector may
    # merge / split blocks vs the simulator's truth); we just verify the
    # partition was non-trivial.
    nontrivial_partition = bool(res_lro["n_blocks"] >= 2)

    gates = [
        dict(
            name="causal beta_hat recovered (full-K LMM)",
            observed=rel_err_lmm,
            threshold=0.30,
            direction="max",
            passed=bool(rel_err_lmm <= 0.30),
        ),
        dict(
            name="causal beta_hat recovered (LROLMM)",
            observed=rel_err_lro,
            threshold=0.30,
            direction="max",
            passed=bool(rel_err_lro <= 0.30),
        ),
        dict(
            name="LRO p < full-K LMM p at causal SNP (decontamination effect)",
            observed=p_lro_causal,
            threshold=p_lmm_causal,
            direction="max",
            passed=bool(p_lro_causal < p_lmm_causal),
        ),
        dict(
            name="median non-causal p is null-like (>0.05)",
            observed=min(median_p_noncausal_lmm, median_p_noncausal_lro),
            threshold=0.05,
            direction="min",
            passed=bool(median_p_noncausal_lmm > 0.05 and median_p_noncausal_lro > 0.05),
        ),
        dict(
            name="LD-block partition is non-trivial (>= 2 blocks)",
            observed=res_lro["n_blocks"],
            threshold=2,
            direction="min",
            passed=nontrivial_partition,
        ),
    ]

    all_pass = all(g["passed"] for g in gates)

    # ── emit summary.tsv ──
    summary_rows = ["check\tobserved\tthreshold\tdirection\tpassed"]
    for g in gates:
        summary_rows.append(
            f"{g['name']}\t{g['observed']}\t{g['threshold']}\t"
            f"{g['direction']}\t{g['passed']}"
        )
    summary_path = out_results / "summary.tsv"
    summary_path.write_text("\n".join(summary_rows) + "\n")

    # ── emit agreement.json ──
    agreement = dict(
        name="LRO (Leave-Region-Out LMM) internal-consistency",
        external_reference=None,
        validation_mode="internal-consistency",
        phase="Phase 29",
        seed=SEED,
        n_samples=N,
        n_blocks=N_BLOCKS,
        snps_per_block=SNPS_PER_BLOCK,
        causal_block=CAUSAL_BLOCK,
        causal_idx=causal_idx,
        beta_causal=BETA_CAUSAL,
        ld_rho=LD_RHO,
        checks=gates,
        passed=all_pass,
        extras=dict(
            beta_lmm_causal=beta_lmm,
            beta_lro_causal=beta_lro,
            p_lmm_causal=p_lmm_causal,
            p_lro_causal=p_lro_causal,
            stat_lmm_causal=stat_lmm_causal,
            stat_lro_causal=stat_lro_causal,
            median_p_noncausal_lmm=median_p_noncausal_lmm,
            median_p_noncausal_lro=median_p_noncausal_lro,
            lro_n_blocks=res_lro["n_blocks"],
            lro_block_size_mean=float(np.mean(res_lro["block_sizes"])),
        ),
    )
    agreement_path = out_results / "agreement.json"
    agreement_path.write_text(json.dumps(agreement, indent=2, default=str))

    # ── emit manifest.sha256 ──
    manifest_lines = []
    for src in [HERE / "generate_and_compare.py", summary_path, agreement_path]:
        if src.exists():
            manifest_lines.append(f"{_hash_file(src)}  {src.relative_to(HERE)}")
    manifest_path = out_results / "manifest.sha256"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")

    print(f"\nLRO internal-consistency: {'PASS' if all_pass else 'FAIL'}")
    for g in gates:
        sym = "PASS" if g["passed"] else "FAIL"
        obs = g["observed"]
        if isinstance(obs, float):
            print(f"  [{sym}] {g['name']}: observed={obs:.6g}")
        else:
            print(f"  [{sym}] {g['name']}: observed={obs}")
    print(f"  Causal SNP: β_LMM={beta_lmm:.4f}, β_LRO={beta_lro:.4f}, "
          f"p_LMM={p_lmm_causal:.3e}, p_LRO={p_lro_causal:.3e}")
    print(f"  -> {agreement_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
