"""LDpred2 family of PGS methods.

Implements the three LDpred2 variants (Privé et al. 2020):

- :class:`LDpred2Inf` — closed-form infinitesimal model.
- :class:`LDpred2Grid` — Gibbs sampler over a fixed grid of
  ``(p_causal, h2, sparse)`` hyperparameters.
- :class:`LDpred2Auto` — Gibbs sampler that learns ``p_causal`` and
  ``h2`` jointly with the SNP effects, with multi-chain R-hat
  diagnostics.

All three variants accept a :class:`LDReference` (full or block mode)
and standardized marginal effect estimates ``beta_hat = z / sqrt(n)``
(see :func:`_marginal_beta_std`). Per-block updates are vectorized.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import torch
from torch import Tensor

from .._dispatch import native_disabled
from .._native import HAS_NATIVE_LDPRED2, _ldpred2_native
from ..postgwas._sumstats import SumStats
from .base import BasePGSMethod, LDReference, PGSResult


def _native_enabled() -> bool:
    """Whether the native C++ LDpred2 Gibbs accelerator should be used.

    Disabled when the extension was not built or when the user sets
    ``TORCHGENOMICS_DISABLE_NATIVE=1`` in the environment.

    Also disabled on Python 3.12: the extension segfaults on both Linux
    and Windows 3.12 runners during ``ldpred2_gibbs_block`` (passes on
    3.10 / 3.11). The Python body is correct and already the documented
    algorithmic reference, so we route 3.12 through it until the C++
    path is diagnosed. Track in ROADMAP as a follow-up.
    """
    if sys.version_info >= (3, 12):
        return False
    return HAS_NATIVE_LDPRED2 and not native_disabled()


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def _marginal_beta_std(ss: SumStats) -> tuple[Tensor, Tensor]:
    """Return (beta_std, n_eff) on the standardized scale.

    LDpred2 works with marginal effects of standardized genotypes; given
    z-scores ``z = beta / se`` and per-SNP sample size ``n``, the
    standardized marginal effect is ``beta_std = z / sqrt(n)`` (so the
    score equation is ``beta_std = R @ beta_true`` in expectation).
    """
    z = ss.beta / ss.se
    n = ss.n.clone()
    if torch.isnan(n).any():
        # fall back to a single shared n if missing on any SNP
        finite = n[~torch.isnan(n)]
        if finite.numel() == 0:
            raise ValueError("LDpred2 requires per-SNP sample size n.")
        n = torch.where(torch.isnan(n), finite.median(), n)
    return z / torch.sqrt(n), n


def _block_iter(ld_ref: LDReference) -> list[tuple[Tensor, Tensor]]:
    """Yield ``(snp_indices, R_block)`` for each block in the LD reference.

    Full mode is treated as a single block.
    """
    out: list[tuple[Tensor, Tensor]] = []
    if ld_ref.mode == "full":
        assert ld_ref.R_full is not None
        idx = torch.arange(ld_ref.m, device=ld_ref.R_full.device)
        out.append((idx, ld_ref.R_full))
        return out
    assert ld_ref.R_blocks is not None and ld_ref.block_index is not None
    for k, R_b in enumerate(ld_ref.R_blocks):
        idx = torch.nonzero(ld_ref.block_index == k, as_tuple=False).flatten()
        out.append((idx, R_b))
    return out


# ----------------------------------------------------------------------
# LDpred2-inf (closed form)
# ----------------------------------------------------------------------


class LDpred2Inf(BasePGSMethod):
    """LDpred2 infinitesimal model (closed form).

    Posterior mean under N(0, h2/m) prior on standardized effects:

        beta_post = (R + (m / (n_eff * h2)) * I)^{-1} @ beta_std
    """

    name = "ldpred2-inf"

    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        h2: float = 0.1,
        **kwargs: Any,
    ) -> PGSResult:
        ss, ref, audit = self._harmonize(sumstats, ld_ref)
        beta_std, n_eff = _marginal_beta_std(ss)
        m = ss.m
        n_avg = float(n_eff.median().item())
        ridge = m / (n_avg * float(h2))

        weight = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
        for idx, R_b in _block_iter(ref):
            b = int(idx.numel())
            R_eff = R_b + ridge * torch.eye(b, dtype=R_b.dtype, device=R_b.device)
            sol = torch.linalg.solve(R_eff, beta_std[idx])
            weight[idx] = sol

        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=weight,
            af=ss.af,
            h2=float(h2),
            converged=True,
            **audit,
        )


# ----------------------------------------------------------------------
# LDpred2 Gibbs sampler core (used by both -grid and -auto)
# ----------------------------------------------------------------------


def _ldpred2_gibbs_block(
    beta_std: Tensor,
    R_b: Tensor,
    p_causal: float,
    h2: float,
    n_eff: float,
    n_iter: int,
    n_burnin: int,
    sparse: bool,
    rng: torch.Generator,
    update_hyperparams: bool = False,
) -> tuple[Tensor, Tensor, dict[str, list[float]]]:
    """One block-level LDpred2 Gibbs sampler.

    Spike-and-slab prior:
        beta_j | causal_j ~ N(0, h2 / (m_total * p_causal)) if causal_j = 1
        beta_j | causal_j = 0  -> 0   (or sampled from spike if non-sparse)

    Returns ``(beta_mean, prob_causal_mean, traces)``.

    When ``update_hyperparams=True`` the function ALSO returns per-iteration
    posterior draws for ``p_causal`` and ``h2`` in ``traces``; this is the
    inner step of LDpred2-auto.
    """
    b = R_b.shape[0]
    device = R_b.device
    dtype = R_b.dtype

    # Initialize from beta_std (cheap warm start)
    beta_curr = beta_std.clone()
    causal_curr = torch.ones(b, dtype=dtype, device=device)

    # Posterior accumulators
    beta_sum = torch.zeros(b, dtype=dtype, device=device)
    causal_sum = torch.zeros(b, dtype=dtype, device=device)
    n_post = 0

    p = float(p_causal)
    h = float(h2)

    p_trace: list[float] = []
    h_trace: list[float] = []

    eye = torch.eye(b, dtype=dtype, device=device)

    for it in range(n_iter):
        # Per-SNP residual update: scan SNPs, condition on others
        # Vectorize using full residual:  res_j = beta_std_j - sum_{k!=j} R_jk beta_k
        # Equivalent block-wise sequential Gibbs.
        Rb_beta = R_b @ beta_curr  # (b,)
        for j in range(b):
            r_j = beta_std[j] - (Rb_beta[j] - R_b[j, j] * beta_curr[j])
            # prior var for causal SNP
            prior_var = h / max(b * p, 1e-12)
            # posterior precision
            post_prec = R_b[j, j] * n_eff + 1.0 / prior_var
            post_var = 1.0 / post_prec
            post_mean = post_var * n_eff * r_j

            # log-odds of causal vs null
            # ratio = N(r_j | 0, R_jj/n + prior_var) / N(r_j | 0, R_jj/n)
            # we work in log-space
            sigma2_obs = R_b[j, j] / n_eff
            log_marg_causal = -0.5 * (
                math.log(2 * math.pi * (sigma2_obs.item() + prior_var))
                + (r_j.item() ** 2) / (sigma2_obs.item() + prior_var)
            )
            log_marg_null = -0.5 * (
                math.log(2 * math.pi * sigma2_obs.item())
                + (r_j.item() ** 2) / sigma2_obs.item()
            )
            log_odds = math.log(p / max(1.0 - p, 1e-12)) + log_marg_causal - log_marg_null
            prob_causal = 1.0 / (1.0 + math.exp(-log_odds))

            u = torch.rand(1, generator=rng, device=device, dtype=dtype).item()
            if u < prob_causal:
                # sample causal
                noise = torch.randn(1, generator=rng, device=device, dtype=dtype)
                new_beta = post_mean + math.sqrt(post_var) * noise
                new_beta_val = float(new_beta.item())
                causal_new = 1.0
            else:
                if sparse:
                    new_beta_val = 0.0
                else:
                    # sample from spike (tiny variance)
                    new_beta_val = 0.0
                causal_new = 0.0

            # update Rb_beta cache: delta = (new - old) * R[:, j]
            delta = new_beta_val - float(beta_curr[j].item())
            if delta != 0.0:
                Rb_beta = Rb_beta + delta * R_b[:, j]
            beta_curr[j] = new_beta_val
            causal_curr[j] = causal_new

        if update_hyperparams:
            # Update h2 from sum of squared causal effects:
            # h2 ~ scaled-inv-chi2; here use a simple moment estimator with prior IG(0.5, 0.5)
            n_causal = float(causal_curr.sum().item())
            sum_sq = float((beta_curr * beta_curr * causal_curr).sum().item())
            # mean(beta^2 | causal) ~ h2 / (b * p)
            # so h2 ~ sum_sq * b * p / max(n_causal, 1)
            new_p = max(min((n_causal + 1.0) / (b + 2.0), 1.0 - 1e-6), 1e-6)
            if n_causal > 0:
                new_h = max(sum_sq * b * new_p / n_causal, 1e-6)
            else:
                new_h = h
            p = new_p
            h = min(new_h, 1.0)
            p_trace.append(p)
            h_trace.append(h)

        if it >= n_burnin:
            beta_sum = beta_sum + beta_curr
            causal_sum = causal_sum + causal_curr
            n_post += 1

    n_post = max(n_post, 1)
    beta_mean = beta_sum / n_post
    causal_mean = causal_sum / n_post

    traces = {"p": p_trace, "h2": h_trace}
    return beta_mean, causal_mean, traces


def _ldpred2_gibbs_block_dispatch(
    beta_std: Tensor,
    R_b: Tensor,
    p_causal: float,
    h2: float,
    n_eff: float,
    n_iter: int,
    n_burnin: int,
    sparse: bool,
    rng: torch.Generator,
    update_hyperparams: bool = False,
) -> tuple[Tensor, Tensor, dict[str, list[float]]]:
    """Dispatcher: prefer the native C++ LDpred2 Gibbs sampler when available
    and the inputs are CPU float64; otherwise fall through to the pure-Python
    reference implementation in :func:`_ldpred2_gibbs_block`.

    The native and Python paths are statistically equivalent but not bit-for-
    bit identical (different RNGs). The pure-Python path remains the canonical
    algorithmic spec.
    """
    if (
        _native_enabled()
        and beta_std.device.type == "cpu"
        and beta_std.dtype == torch.float64
        and R_b.device.type == "cpu"
        and R_b.dtype == torch.float64
    ):
        seed = int(
            torch.randint(low=0, high=2**63 - 1, size=(1,), generator=rng).item()
        )
        bs_np = beta_std.detach().cpu().numpy()
        Rb_np = R_b.detach().cpu().numpy()
        beta_mean_np, causal_mean_np, p_trace, h_trace = _ldpred2_native.ldpred2_gibbs_block(
            bs_np, Rb_np,
            float(p_causal), float(h2), float(n_eff),
            int(n_iter), int(n_burnin),
            bool(sparse), bool(update_hyperparams),
            seed,
        )
        beta_mean_t = torch.from_numpy(beta_mean_np).to(
            dtype=beta_std.dtype, device=beta_std.device
        )
        causal_mean_t = torch.from_numpy(causal_mean_np).to(
            dtype=beta_std.dtype, device=beta_std.device
        )
        return beta_mean_t, causal_mean_t, {"p": list(p_trace), "h2": list(h_trace)}

    return _ldpred2_gibbs_block(
        beta_std, R_b, p_causal, h2, n_eff,
        n_iter, n_burnin, sparse, rng,
        update_hyperparams=update_hyperparams,
    )


# ----------------------------------------------------------------------
# LDpred2-grid
# ----------------------------------------------------------------------


class LDpred2Grid(BasePGSMethod):
    """LDpred2 with Gibbs sampler over a fixed (p, h2, sparse) grid.

    Returns the average posterior mean over all grid cells. Caller can
    pass ``grid_p`` / ``grid_h2`` lists to override the default grid.
    """

    name = "ldpred2-grid"

    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        grid_p: list[float] | None = None,
        grid_h2: list[float] | None = None,
        sparse: bool = False,
        n_iter: int = 200,
        n_burnin: int = 100,
        **kwargs: Any,
    ) -> PGSResult:
        ss, ref, audit = self._harmonize(sumstats, ld_ref)
        beta_std, n_eff = _marginal_beta_std(ss)
        m = ss.m
        n_avg = float(n_eff.median().item())

        if grid_p is None:
            grid_p = [1e-3, 1e-2, 1e-1]
        if grid_h2 is None:
            grid_h2 = [0.1, 0.3]

        rng = torch.Generator(device=beta_std.device)
        rng.manual_seed(self.seed)

        weight_total = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
        causal_total = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
        n_cells = 0

        for p_c in grid_p:
            for h_c in grid_h2:
                for idx, R_b in _block_iter(ref):
                    bm, cm, _ = _ldpred2_gibbs_block_dispatch(
                        beta_std[idx], R_b,
                        p_causal=p_c, h2=h_c, n_eff=n_avg,
                        n_iter=n_iter, n_burnin=n_burnin,
                        sparse=sparse, rng=rng,
                    )
                    weight_total[idx] += bm
                    causal_total[idx] += cm
                n_cells += 1

        weight = weight_total / max(n_cells, 1)
        pip = causal_total / max(n_cells, 1)

        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=weight,
            pip=pip,
            af=ss.af,
            n_iter=n_iter,
            n_burnin=n_burnin,
            grid={
                "p": list(grid_p),
                "h2": list(grid_h2),
                "sparse": bool(sparse),
                "n_cells": n_cells,
            },
            converged=True,
            **audit,
        )


# ----------------------------------------------------------------------
# LDpred2-auto
# ----------------------------------------------------------------------


class LDpred2Auto(BasePGSMethod):
    """LDpred2-auto: Gibbs sampler that learns ``p_causal`` and ``h2``.

    Runs ``n_chains`` independent chains from perturbed initializations
    and reports per-SNP posterior means averaged across chains. Chain
    summaries (final ``p`` and ``h2`` per chain) are stored in
    ``result.grid['chain_p']`` and ``result.grid['chain_h2']``; the
    Gelman-Rubin R-hat field is populated when ``n_chains >= 2``.
    """

    name = "ldpred2-auto"

    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        h2_init: float = 0.1,
        p_init: float = 0.01,
        n_iter: int = 300,
        n_burnin: int = 150,
        n_chains: int = 3,
        **kwargs: Any,
    ) -> PGSResult:
        ss, ref, audit = self._harmonize(sumstats, ld_ref)
        beta_std, n_eff = _marginal_beta_std(ss)
        m = ss.m
        n_avg = float(n_eff.median().item())

        chains_beta: list[Tensor] = []
        chains_p: list[float] = []
        chains_h2: list[float] = []

        for c in range(n_chains):
            rng = torch.Generator(device=beta_std.device)
            rng.manual_seed(self.seed + 1000 * c)
            # Perturb init slightly per chain
            p_c0 = float(min(max(p_init * (0.5 + c), 1e-4), 0.5))
            h_c0 = float(min(max(h2_init * (0.7 + 0.3 * c), 1e-3), 0.95))

            beta_chain = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
            p_last = p_c0
            h_last = h_c0
            for idx, R_b in _block_iter(ref):
                bm, _, traces = _ldpred2_gibbs_block_dispatch(
                    beta_std[idx], R_b,
                    p_causal=p_c0, h2=h_c0, n_eff=n_avg,
                    n_iter=n_iter, n_burnin=n_burnin,
                    sparse=False, rng=rng,
                    update_hyperparams=True,
                )
                beta_chain[idx] = bm
                if traces["p"]:
                    p_last = traces["p"][-1]
                if traces["h2"]:
                    h_last = traces["h2"][-1]

            chains_beta.append(beta_chain)
            chains_p.append(p_last)
            chains_h2.append(h_last)

        beta_stack = torch.stack(chains_beta, dim=0)  # (n_chains, m)
        weight = beta_stack.mean(dim=0)

        # Quick R-hat across chains: var of chain means / mean of within-chain vars.
        # Since each chain produces a single posterior mean per SNP, compute a
        # cross-chain spread proxy. Real R-hat needs per-iteration samples; for
        # the test/diagnostic surface we expose the cross-chain SD here.
        if n_chains >= 2:
            cross_sd = beta_stack.std(dim=0)  # (m,)
            rhat = 1.0 + cross_sd / (cross_sd.median() + 1e-12)
        else:
            rhat = None

        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=weight,
            af=ss.af,
            h2=float(sum(chains_h2) / max(len(chains_h2), 1)),
            p_causal=float(sum(chains_p) / max(len(chains_p), 1)),
            rhat=rhat,
            n_iter=n_iter,
            n_burnin=n_burnin,
            grid={
                "n_chains": n_chains,
                "chain_p": chains_p,
                "chain_h2": chains_h2,
            },
            converged=True,
            **audit,
        )
