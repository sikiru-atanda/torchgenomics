"""F5 - Polyploid pipeline figure (Phases 55, 56, 0-13 polyploid path).

Four panels per spec section 5 F5:

  A - dosage-call accuracy (Phase 55, updog wrapper)
  B - PolyOrigin F1 phasing-state heatmap (Phase 56)
  C - polyploid GWAS vs GWASpoly (tetraploid potato)
  D - arbitrary-ploidy demo for k in {4, 6, 8}

Tier 4 D2.5.
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import register

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Panel A: dosage-call accuracy (Phase 55)


def _simulate_tetraploid_reads(
    n_samples: int = 80,
    n_markers: int = 100,
    depth: int = 30,
    seq_err: float = 0.01,
    bias: float = 1.0,
    od: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror bench/calibrate_updog_recovery.py simulation."""
    rng = np.random.default_rng(seed)
    ploidy = 4
    af = rng.uniform(0.1, 0.9, size=n_markers)
    true_d = np.stack(
        [rng.binomial(ploidy, af[j], size=n_samples) for j in range(n_markers)],
        axis=-1,
    )
    p_true = true_d / ploidy
    p_obs = (1.0 - seq_err) * p_true + seq_err * (1.0 - p_true)
    p_obs = p_obs * bias / (p_obs * bias + (1.0 - p_obs))
    if od > 0:
        alpha = p_obs * (1.0 - od) / od
        beta = (1.0 - p_obs) * (1.0 - od) / od
        p_sample = rng.beta(alpha.clip(1e-6), beta.clip(1e-6))
    else:
        p_sample = p_obs
    alt = rng.binomial(depth, p_sample)
    ref = depth - alt
    return true_d, ref, alt, af


def _log_binom_coef(n: int, k: int) -> float:
    """log C(n, k) via lgamma."""
    from math import lgamma
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _hw_posterior_dosage(
    ref: np.ndarray,
    alt: np.ndarray,
    af: np.ndarray,
    ploidy: int = 4,
    seq_err: float = 0.01,
) -> np.ndarray:
    """Hardy-Weinberg + binomial-likelihood posterior dosage (MAP).

    Structural form of the updog "hw" model. Returns (n, m) int MAP
    dosage classes.
    """
    n, m = ref.shape
    n_classes = ploidy + 1
    map_dose = np.zeros((n, m), dtype=np.int64)
    eps = 1e-12
    for j in range(m):
        d = np.arange(n_classes)
        log_prior = (
            d * np.log(af[j] + eps)
            + (ploidy - d) * np.log(1.0 - af[j] + eps)
            + np.array([_log_binom_coef(ploidy, dd) for dd in d])
        )
        for i in range(n):
            size = ref[i, j] + alt[i, j]
            if size == 0:
                map_dose[i, j] = int(np.round(af[j] * ploidy))
                continue
            p_alt_given_d = (1.0 - seq_err) * (d / ploidy) + seq_err * (1.0 - d / ploidy)
            log_like = (
                alt[i, j] * np.log(p_alt_given_d + eps)
                + ref[i, j] * np.log(1.0 - p_alt_given_d + eps)
            )
            log_post = log_prior + log_like
            map_dose[i, j] = int(np.argmax(log_post))
    return map_dose


def _render_panel_a(ax) -> dict:
    true_d, ref, alt, af = _simulate_tetraploid_reads(
        n_samples=80, n_markers=100, depth=30, seed=42,
    )
    est_d = _hw_posterior_dosage(ref, alt, af, ploidy=4, seq_err=0.01)
    flat_true = true_d.reshape(-1)
    flat_est = est_d.reshape(-1)
    n_total = flat_true.size
    n_correct = int((flat_true == flat_est).sum())
    accuracy = n_correct / n_total

    rng = np.random.default_rng(7)
    jitter_t = rng.normal(0.0, 0.05, size=flat_true.shape)
    jitter_e = rng.normal(0.0, 0.05, size=flat_est.shape)
    correct = flat_true == flat_est
    ax.scatter(
        flat_true[correct] + jitter_t[correct],
        flat_est[correct] + jitter_e[correct],
        s=6, alpha=0.35, color="#3a7d44",
        label=f"correct (n={int(correct.sum())})",
    )
    ax.scatter(
        flat_true[~correct] + jitter_t[~correct],
        flat_est[~correct] + jitter_e[~correct],
        s=10, alpha=0.7, color="#c0392b",
        label=f"miscall (n={int((~correct).sum())})",
    )
    ax.plot([-0.5, 4.5], [-0.5, 4.5], "--", color="#888", linewidth=0.8, zorder=0)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("true dosage")
    ax.set_ylabel("estimated dosage (HW posterior)")
    ax.set_title(
        f"A - Dosage-call accuracy (k=4, 30x)\n"
        f"accuracy={accuracy:.3f} on n=80 x m=100  [scaffold]"
    )
    ax.legend(loc="lower right", fontsize=7)

    return {
        "panel": "A",
        "scaffold": True,
        "scaffold_reason": "Phase 55 updog wrapper gated on R+updog; "
                           "renderer uses HW posterior surrogate.",
        "n_samples": int(true_d.shape[0]),
        "n_markers": int(true_d.shape[1]),
        "depth": 30,
        "ploidy": 4,
        "accuracy": float(accuracy),
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "simulation_seed": 42,
        "model": "hw_posterior_surrogate",
    }


# Panel B: PolyOrigin F1 phasing-state heatmap (Phase 56)


def _simulate_phasing_heatmap(
    n_progeny: int = 6,
    n_loci: int = 20,
    ploidy: int = 4,
    seed: int = 17,
) -> tuple[np.ndarray, int]:
    """Return (states, n_states).

    For PolyOrigin on a biparental tetraploid F1, each progeny at each
    locus is assigned one of 35 phasing states (the unordered parental
    haplotype combinations with double reduction allowed). We synthesize
    a plausible state path by sampling a smooth per-progeny path with
    short stretches of identity-by-descent across adjacent loci.
    """
    n_states = 35 if ploidy == 4 else 14
    rng = np.random.default_rng(seed)
    states = np.zeros((n_progeny, n_loci), dtype=np.int64)
    for i in range(n_progeny):
        s = int(rng.integers(0, n_states))
        for j in range(n_loci):
            if rng.random() < 0.15:
                s = int(rng.integers(0, n_states))
            states[i, j] = s
    return states, n_states


def _render_panel_b(ax) -> dict:
    states, n_states = _simulate_phasing_heatmap(
        n_progeny=6, n_loci=20, ploidy=4, seed=17,
    )
    im = ax.imshow(
        states, aspect="auto", cmap="viridis",
        vmin=0, vmax=n_states - 1, interpolation="nearest",
    )
    ax.set_xlabel("locus index")
    ax.set_ylabel("progeny index")
    ax.set_yticks(range(states.shape[0]))
    ax.set_yticklabels([f"F1[{i}]" for i in range(states.shape[0])], fontsize=7)
    ax.set_title(
        "B - PolyOrigin F1 phasing-state heatmap (k=4)\n"
        f"{n_states} states, simulated F1 path  [scaffold]"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("state id", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return {
        "panel": "B",
        "scaffold": True,
        "scaffold_reason": "GWASpoly potato round-trip "
                           "(2026-04-24-gwaspoly-potato-roundtrip.md) "
                           "pending: URL/SHA constants in "
                           "tests/test_phase_polyorigin_e2e.py are placeholders.",
        "n_progeny": int(states.shape[0]),
        "n_loci": int(states.shape[1]),
        "n_states": int(n_states),
        "ploidy": 4,
        "simulation_seed": 17,
    }


# Panel C: polyploid GWAS vs GWASpoly


def _gwaspoly_data_present() -> bool:
    data_dir = REPO_ROOT / "validation/external/gwaspoly/data"
    out_dir = REPO_ROOT / "validation/external/gwaspoly/outputs"
    return (
        data_dir.is_dir()
        and out_dir.is_dir()
        and (data_dir / "potato_geno_aligned.csv").is_file()
        and (out_dir / "gwaspoly_additive.csv").is_file()
    )


def _run_real_gwaspoly_comparison():
    """Run canonical additive comparison from
    validation/external/gwaspoly/compare.py and return
    (logp_tg, logp_ref, info). May raise on import or fit errors;
    caller falls back to the simulation.
    """
    sys.path.insert(0, str(REPO_ROOT / "validation/external/gwaspoly"))
    try:
        from compare import _load_data, _run_p3d, _load_ref, _logp_corr  # type: ignore
    finally:
        target = str(REPO_ROOT / "validation/external/gwaspoly")
        if target in sys.path:
            sys.path.remove(target)

    data_dir = REPO_ROOT / "validation/external/gwaspoly/data"
    out_dir = REPO_ROOT / "validation/external/gwaspoly/outputs"
    Y, X0, G_obs, G_geno, Z, vmeta, snp_names = _load_data(data_dir)
    p_tg = _run_p3d(Y, X0, G_obs, G_geno, Z, vmeta, "additive")
    ref = _load_ref(out_dir, "additive")
    corr, n = _logp_corr(p_tg, snp_names, ref)

    logp_tg = []
    logp_ref = []
    for i, snp in enumerate(snp_names):
        if snp in ref and np.isfinite(p_tg[i]) and p_tg[i] > 0 and ref[snp] > 0:
            logp_tg.append(-np.log10(p_tg[i]))
            logp_ref.append(-np.log10(ref[snp]))
    return (
        np.asarray(logp_tg),
        np.asarray(logp_ref),
        {"correlation": float(corr), "n_compared": int(n)},
    )


def _simulate_gwaspoly_comparison(
    n_markers: int = 200, planted_idx: int = 60, seed: int = 23,
):
    """Simulate -log10(p) scatter with one planted causal at planted_idx."""
    rng = np.random.default_rng(seed)
    null_p_tg = rng.uniform(size=n_markers)
    null_p_ref = null_p_tg * rng.uniform(0.85, 1.15, size=n_markers)
    null_p_ref = null_p_ref.clip(1e-9, 1.0)
    logp_tg = -np.log10(null_p_tg)
    logp_ref = -np.log10(null_p_ref)
    logp_tg[planted_idx] = 7.1 + rng.normal(0.0, 0.05)
    logp_ref[planted_idx] = 7.0 + rng.normal(0.0, 0.05)
    return (
        logp_tg,
        logp_ref,
        {
            "correlation": float(np.corrcoef(logp_tg, logp_ref)[0, 1]),
            "n_compared": int(n_markers),
        },
    )


def _render_panel_c(ax) -> dict:
    if _gwaspoly_data_present():
        try:
            logp_tg, logp_ref, info = _run_real_gwaspoly_comparison()
            scaffold = False
            label_extra = ""
        except Exception as exc:  # noqa: BLE001
            logp_tg, logp_ref, info = _simulate_gwaspoly_comparison()
            info["error"] = str(exc)
            scaffold = True
            label_extra = "  [scaffold - real run errored]"
    else:
        logp_tg, logp_ref, info = _simulate_gwaspoly_comparison()
        scaffold = True
        label_extra = "  [scaffold - fixture absent]"

    ax.scatter(logp_ref, logp_tg, s=8, alpha=0.55, color="#1f77b4", edgecolors="none")
    mx = float(max(
        logp_ref.max() if logp_ref.size else 1.0,
        logp_tg.max() if logp_tg.size else 1.0,
    ))
    mx = max(mx, 1.0)
    ax.plot([0, mx * 1.05], [0, mx * 1.05], "--", color="#888", linewidth=0.8)
    ax.set_xlim(0, mx * 1.05)
    ax.set_ylim(0, mx * 1.05)
    ax.set_xlabel("GWASpoly -log10(p)  [additive]")
    ax.set_ylabel("TorchGWAS -log10(p)  [additive]")

    corr_str = "{:.3f}".format(float(info["correlation"]))
    n_str = str(int(info["n_compared"]))
    ax.set_title(
        "C - Polyploid GWAS vs GWASpoly (k=4, potato F1)\n"
        "r(-log10 p)=" + corr_str + ", n=" + n_str + label_extra
    )

    return {
        "panel": "C",
        "scaffold": bool(scaffold),
        "scaffold_reason": (
            "validation/external/gwaspoly/{data,outputs} not staged on "
            "this branch - used simulated -log10(p) scatter."
            if scaffold
            else "real GWASpoly comparison"
        ),
        "correlation_logp": float(info["correlation"]),
        "n_compared": int(info["n_compared"]),
        "ploidy": 4,
    }


# Panel D: arbitrary-ploidy demo (k=4, 6, 8)


def _run_polyploid_scan_demo(
    ploidy: int,
    n: int = 50,
    m: int = 20,
    causal_idx: int = 10,
    causal_effect: float = 1.5,
    seed: int = 0,
):
    """Synthesize a tiny polyploid GWAS and return (neg_log10_p, causal_idx).

    Genotypes are drawn as Binomial(ploidy, af_j) for af_j ~ U(0.2, 0.8).
    Phenotype Y = beta * G[:, causal_idx] + eps with eps ~ N(0, 1). We
    fit SingleTraitLMM against the additive polyploid GRM and run a
    Wald test per SNP via score_chunk.
    """
    import torch

    from torchgwas.config import STAT_DTYPE, NumericalConfig
    from torchgwas.linalg.kinship_polyploid import grm_polyploid_gene_action
    from torchgwas.models.base import VariantMeta
    from torchgwas.models.single_trait_lmm import SingleTraitLMM

    rng = np.random.default_rng(seed + ploidy)
    af = rng.uniform(0.2, 0.8, size=m)
    G_np = np.stack(
        [rng.binomial(ploidy, af[j], size=n) for j in range(m)],
        axis=-1,
    ).astype(np.float64)
    causal = G_np[:, causal_idx] - G_np[:, causal_idx].mean()
    Y_np = causal_effect * causal + rng.normal(0.0, 1.0, size=n)

    G = torch.tensor(G_np, dtype=STAT_DTYPE)
    Y = torch.tensor(Y_np, dtype=STAT_DTYPE)
    X0 = torch.ones((n, 1), dtype=STAT_DTYPE)

    K, _ = grm_polyploid_gene_action(G, "additive", ploidy)
    K = K + 1e-4 * torch.eye(n, dtype=STAT_DTYPE)

    vmeta = VariantMeta(
        snp=[f"s{j}" for j in range(m)],
        chr=["1"] * m,
        pos=list(range(1, m + 1)),
        a1=["A"] * m,
        a2=["G"] * m,
    )
    config = NumericalConfig(reml_method="emma")
    lmm = SingleTraitLMM(config=config)
    nf = lmm.fit_null(Y, X0, K=K)
    result = lmm.score_chunk(G, nf, vmeta, test="wald")
    p = result.p.detach().cpu().numpy()
    p = np.clip(p, 1e-30, 1.0)
    return -np.log10(p), causal_idx


def _render_panel_d(ax) -> dict:
    ploidies = [4, 6, 8]
    colors = {4: "#1f77b4", 6: "#2ca02c", 8: "#d62728"}
    causal_idx = 10
    n = 50
    m = 20
    per_ploidy: dict = {}
    detected_at_alpha = 0
    xs_offset = {4: -0.25, 6: 0.0, 8: 0.25}

    for k in ploidies:
        try:
            neg_logp, ci = _run_polyploid_scan_demo(
                ploidy=k, n=n, m=m, causal_idx=causal_idx,
                causal_effect=1.5, seed=0,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            neg_logp = np.full(m, np.nan)
            err = str(exc)
            ci = causal_idx

        xs = np.arange(m, dtype=float) + xs_offset[k]
        ax.plot(
            xs, neg_logp, "o-",
            color=colors[k], label=f"k={k}",
            markersize=4, linewidth=0.8, alpha=0.85,
        )
        if np.isfinite(neg_logp[ci]):
            ax.scatter(
                [ci + xs_offset[k]], [neg_logp[ci]],
                s=80, facecolor="none", edgecolor=colors[k],
                linewidth=1.5, zorder=5,
            )

        causal_logp = (
            float(neg_logp[ci]) if np.isfinite(neg_logp[ci]) else float("nan")
        )
        if np.isfinite(causal_logp) and causal_logp > -np.log10(0.05):
            detected_at_alpha += 1
        per_ploidy[f"k={k}"] = {
            "causal_neg_log10p": causal_logp,
            "max_neg_log10p": (
                float(np.nanmax(neg_logp))
                if np.isfinite(neg_logp).any() else None
            ),
            "n": int(n),
            "m": int(m),
            "causal_idx": int(ci),
            "error": err,
        }

    bonf = -np.log10(0.05 / m)
    bonf_str = "{:.2f}".format(float(bonf))
    ax.axhline(bonf, ls=":", color="#444", linewidth=0.8,
               label="Bonferroni 0.05/m  (-log10 p=" + bonf_str + ")")
    ax.axhline(-np.log10(0.05), ls="--", color="#888",
               linewidth=0.6, alpha=0.6, label="alpha=0.05")
    ax.set_xticks(range(0, m, 2))
    ax.set_xlabel("SNP index  (causal = 10, circled)")
    ax.set_ylabel("-log10(p)  (Wald)")
    ax.set_title(
        "D - Arbitrary-ploidy demo (synthesized at render time)\n"
        "n=50, m=20, one planted causal; k in {4, 6, 8}"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)

    return {
        "panel": "D",
        "scaffold": False,
        "synthesized_at_render_time": True,
        "ploidies_tested": ploidies,
        "n_individuals": n,
        "n_markers": m,
        "causal_idx": causal_idx,
        "causal_effect": 1.5,
        "bonferroni_threshold_neg_log10": float(bonf),
        "n_ploidies_clearing_alpha_0_05_at_causal": int(detected_at_alpha),
        "per_ploidy": per_ploidy,
    }


# Top-level renderer entry point


@register("F5")
def render_f5(output_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panel_a = _render_panel_a(axes[0, 0])
    panel_b = _render_panel_b(axes[0, 1])
    panel_c = _render_panel_c(axes[1, 0])
    panel_d = _render_panel_d(axes[1, 1])

    fig.suptitle(
        "F5 - Polyploid pipeline: dosage call, F1 phasing, "
        "polyploid GWAS, arbitrary-ploidy demo",
        fontsize=11, y=0.995,
    )
    n_scaffold = sum(
        1 for p in (panel_a, panel_b, panel_c, panel_d) if p.get("scaffold")
    )
    fig.text(
        0.5, 0.005,
        str(n_scaffold) + "/4 panels scaffold-only - "
        "Phases 55 (dosage_call) + 56 (phase_polyorigin) + 0-13 "
        "(polyploid LMM) - Genome Biology Methods Figure 5",
        ha="center", fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path = output_dir / "F5.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "panels": {
            "A": panel_a,
            "B": panel_b,
            "C": panel_c,
            "D": panel_d,
        },
        "n_scaffold_panels": int(n_scaffold),
        "phase_provenance": [
            "Phase 55 (dosage_call / updog wrapper)",
            "Phase 56 (phase_polyorigin / PolyOrigin wrapper)",
            "Phases 0-13 (polyploid SingleTraitLMM + GRM)",
        ],
    }
    return out_path, manifest
