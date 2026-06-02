#!/usr/bin/env python3
"""run_torchgenomics.py -- fit RandomRegressionLMM on the same fixture.

Reads validation/specialty/rr/data/{long_pheno.tsv,geno.tsv,truth.json}
and writes validation/specialty/rr/outputs/{torchgenomics_null.tsv,
torchgenomics_causal.tsv} using the same name -> value schema as
reference_null.tsv / reference_causal.tsv, so compare.py can diff them.

Modelling choice: K = I_n (identity GRM).  This puts TG in the same
model class as lme4 (Henderson LMM with no kinship correction), so the
only sources of disagreement are numerical (REML optimizer, eigh path,
finite-precision basis evaluation) rather than model-class mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from torchgenomics.models.base import VariantMeta  # noqa: E402
from torchgenomics.models.rr_lmm import RandomRegressionLMM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir",  default=str(HERE / "outputs"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pheno = pd.read_csv(data_dir / "long_pheno.tsv", sep="\t")
    geno_df = pd.read_csv(data_dir / "geno.tsv", sep="\t")
    truth = json.loads((data_dir / "truth.json").read_text())

    sid_arr = pheno["sid"].to_numpy()
    t_arr   = pheno["t"].to_numpy(np.float64)
    y_arr   = pheno["y"].to_numpy(np.float64)

    unique_sids = np.array(sorted(geno_df["sid"].unique()))
    n = unique_sids.shape[0]
    b = truth["n_basis"]
    t_min = float(truth["t_min"])
    t_max = float(truth["t_max"])

    # Tensorize
    Y_long       = torch.tensor(y_arr,   dtype=torch.float64)
    sample_ids   = torch.tensor(sid_arr, dtype=torch.int64)
    time_values  = torch.tensor(t_arr,   dtype=torch.float64)
    X0 = torch.ones(n, 1, dtype=torch.float64)
    K  = torch.eye(n, dtype=torch.float64)  # identity GRM

    model = RandomRegressionLMM(
        basis="legendre",
        order=truth["basis_order"],
        k_coef_structure="unstructured",
    )
    print("[tg] fitting RandomRegressionLMM null...", flush=True)
    null_fit = model.fit_null(
        Y_long, X0, K,
        sample_ids=sample_ids,
        time_values=time_values,
        t_min=t_min, t_max=t_max,
    )
    print(f"[tg] null fit converged={null_fit.converged} "
          f"logLik={float(null_fit.log_likelihood):.6f}", flush=True)

    # ---- write null TSV --------------------------------------------
    K_coef = null_fit.K_coef.detach().cpu().numpy()
    Ve     = null_fit.Ve.detach().cpu().numpy()
    null_rows = []
    for i in range(b):
        for j in range(b):
            null_rows.append(f"K_coef[{i},{j}]\t{K_coef[i, j]:.10g}")
    diag_K = np.diag(K_coef)
    total_basis_var = float(diag_K.sum())
    for i in range(b):
        null_rows.append(
            f"var_ratio_basis[{i}]\t{diag_K[i] / total_basis_var:.10g}")
    # For sigma2_e, TG fits a full (b,b) Ve covariance in coefficient space.
    # The lme4 reference reports a scalar sigma2_e for the observation-space
    # residual.  We report TGs trace-mean of Ve as the comparable scalar
    # (also expose the full Ve diagonal for reviewer inspection).
    null_rows.append(f"sigma2_e_trace_mean\t{float(np.trace(Ve) / b):.10g}")
    for i in range(b):
        null_rows.append(f"Ve[{i},{i}]\t{Ve[i, i]:.10g}")
    null_rows.append(
        f"logLik_REML\t{float(null_fit.log_likelihood):.10g}")
    (out_dir / "torchgenomics_null.tsv").write_text(
        "name\tvalue\n" + "\n".join(null_rows) + "\n"
    )
    print("[tg] wrote torchgenomics_null.tsv", flush=True)

    # ---- causal SNP scan (just the one SNP) ------------------------
    causal_idx = int(truth["causal_snp_index"])
    causal_snp = truth["causal_snp_name"]
    geno_df_ord = geno_df.set_index("sid").loc[unique_sids]
    G_full = torch.tensor(geno_df_ord.to_numpy(np.float64), dtype=torch.float64)
    G_chunk = G_full[:, causal_idx:causal_idx + 1]
    vmeta = VariantMeta(
        snp=[causal_snp], chr=["1"], pos=[1], a1=["A"], a2=["G"],
    )
    times_grid = torch.tensor(truth["time_grid"], dtype=torch.float64)
    print("[tg] scoring causal SNP via score_chunk(eval_times=time_grid)...",
          flush=True)
    res = model.score_chunk(
        G_chunk, null_fit, vmeta,
        test="wald", eval_times=times_grid,
    )
    beta = res.beta[0].detach().cpu().numpy()
    se   = res.se[0].detach().cpu().numpy()
    Var_beta = res.Var_beta[0].detach().cpu().numpy()
    chi2_joint = float(res.stat_joint[0].item())
    p_joint    = float(res.p_joint[0].item())
    chi2_int   = float(res.stat_intercept[0].item())
    p_int      = float(res.p_intercept[0].item())
    chi2_slope = float(res.stat_slope[0].item())
    p_slope    = float(res.p_slope[0].item())
    chi2_tv    = float(res.stat_time_varying[0].item())
    p_tv       = float(res.p_time_varying[0].item())
    beta_at_t = res.beta_at_t[0].detach().cpu().numpy()
    se_at_t   = res.se_at_t[0].detach().cpu().numpy()
    unique_times = list(truth["time_grid"])
    from scipy.stats import chi2 as _chi2_dist
    p_per_k = []
    for k in range(b):
        chi2_k = (beta[k] / max(se[k], 1e-300)) ** 2
        p_per_k.append(float(_chi2_dist.sf(chi2_k, df=1)))
    causal_rows = [
        f"beta_g\t{beta[0]:.10g}",
        f"se_g\t{se[0]:.10g}",
        f"p_g\t{p_per_k[0]:.10g}",
        f"beta_g_P1\t{beta[1]:.10g}",
        f"se_g_P1\t{se[1]:.10g}",
        f"p_g_P1\t{p_per_k[1]:.10g}",
        f"beta_g_P2\t{beta[2]:.10g}",
        f"se_g_P2\t{se[2]:.10g}",
        f"p_g_P2\t{p_per_k[2]:.10g}",
        f"chi2_joint_b\t{chi2_joint:.10g}",
        f"p_joint_b\t{p_joint:.10g}",
        f"chi2_intercept\t{chi2_int:.10g}",
        f"p_intercept\t{p_int:.10g}",
        f"chi2_slope\t{chi2_slope:.10g}",
        f"p_slope\t{p_slope:.10g}",
        f"chi2_time_varying\t{chi2_tv:.10g}",
        f"p_time_varying\t{p_tv:.10g}",
    ]
    for k, t in enumerate(unique_times):
        causal_rows.append(f"beta_at_t_{t:g}\t{beta_at_t[k]:.10g}")
        causal_rows.append(f"se_at_t_{t:g}\t{se_at_t[k]:.10g}")
    causal_rows.append(f"causal_snp\t{causal_snp}")
    (out_dir / "torchgenomics_causal.tsv").write_text(
        "name\tvalue\n" + "\n".join(causal_rows) + "\n"
    )
    print("[tg] wrote torchgenomics_causal.tsv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
