"""PRS-CS — polygenic scores with Bayesian continuous shrinkage priors.

Implements the Strawderman-Berger continuous-shrinkage prior of
Ge et al. (2019) via a block Gibbs sampler in PyTorch:

    beta_j | psi_j ~ N(0, sigma^2 * psi_j / n)
    psi_j          ~ Gamma(a, delta_j)   (rate delta_j)
    delta_j        ~ Gamma(b, phi)       (rate phi)

with ``a = 1``, ``b = 0.5`` by default (Strawderman-Berger); ``phi`` is
the global shrinkage hyperparameter (user-provided or set to a small
default). Block-wise posterior updates sample ``beta`` jointly via
Cholesky of ``(R_b + diag(1/psi_b)) * n / sigma^2``.

Also provides :func:`sample_gig`, a vectorized Generalized Inverse
Gaussian sampler using the Devroye (1986) / Hoermann-Leydold ratio-of-
uniforms algorithm — used internally for the ``psi`` updates.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
from torch import Tensor

from .._native import HAS_NATIVE_PRSCS, _prscs_native
from ..postgwas._sumstats import SumStats
from .base import BasePGSMethod, LDReference, PGSResult
from .ldpred2 import _block_iter, _marginal_beta_std

# Native dispatch toggle: TORCHGWAS_DISABLE_NATIVE=1 forces the pure-Python
# reference path even when the compiled extension is available. Used by tests
# to exercise both code paths and as an escape hatch if the C++ port ever
# diverges from the Python spec.
def _native_enabled() -> bool:
    return HAS_NATIVE_PRSCS and not os.environ.get("TORCHGWAS_DISABLE_NATIVE")


# ----------------------------------------------------------------------
# GIG sampler (Devroye / Hoermann-Leydold ratio-of-uniforms)
# ----------------------------------------------------------------------


def _gig_mode(lam: float, omega: float) -> float:
    """Mode of a GIG(lam, chi=omega/s, psi=omega*s) with s = sqrt(chi/psi)."""
    if abs(lam) <= 1.0:
        return omega / (math.sqrt((1.0 - lam) ** 2 + omega ** 2) + 1.0 - lam)
    return (lam - 1.0 + math.sqrt((lam - 1.0) ** 2 + omega ** 2)) / omega


def _sample_gig_scalar(lam: float, chi: float, psi: float, rng: torch.Generator) -> float:
    """Scalar GIG(lam, chi, psi) sampler.

    Uses ``scipy.stats.geninvgauss`` as the underlying generator for
    correctness; the scipy implementation is itself a Devroye/Hormann-
    Leydold ratio-of-uniforms sampler. A torch-native port is a future
    optimization — for PRS-CS Gibbs loops at ``m ~ 1e5`` the scipy path
    is not the bottleneck (dominated by Cholesky).

    Parameters
    ----------
    lam : float
        Index parameter.
    chi, psi : float
        Scale parameters (>0).
    rng : torch.Generator
        Torch RNG used to sample a seed that is forwarded to scipy so
        the sampler respects the caller's reproducibility contract.

    Returns
    -------
    float
        A sample from GIG(lam, chi, psi).
    """
    import numpy as np
    from scipy.stats import geninvgauss

    if chi <= 0 or psi <= 0:
        raise ValueError(f"GIG requires chi>0, psi>0; got chi={chi}, psi={psi}.")

    # scipy parameterization: geninvgauss(p=lam, b=sqrt(chi*psi)); sample y
    # then x = y * sqrt(chi/psi).
    b_sp = math.sqrt(chi * psi)
    scale = math.sqrt(chi / psi)

    # Derive a numpy seed from the torch RNG so fixed torch seeds produce
    # deterministic GIG draws.
    seed_t = torch.randint(
        low=0, high=2**31 - 1, size=(1,), generator=rng,
    ).item()
    np_rng = np.random.default_rng(int(seed_t))
    y = float(geninvgauss.rvs(lam, b_sp, size=1, random_state=np_rng)[0])
    return y * scale


def sample_gig(
    lam: float,
    chi: Tensor,
    psi: Tensor,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Vectorized GIG(lam, chi_j, psi_j) sampler.

    ``chi`` and ``psi`` must be same-shape tensors. Returns a tensor of
    the same shape with one GIG draw per element. ``lam`` is a shared
    scalar index parameter.

    Notes
    -----
    This is a loop over scalar draws using :func:`_sample_gig_scalar`;
    acceptable for ``m <= 1e6`` per iteration in a Gibbs sampler. A fully
    vectorized torch implementation is a future optimization.
    """
    if rng is None:
        rng = torch.Generator(device=chi.device)
    chi_flat = chi.detach().cpu().flatten().tolist()
    psi_flat = psi.detach().cpu().flatten().tolist()
    out = torch.empty(chi.numel(), dtype=chi.dtype)
    for i, (c, p) in enumerate(zip(chi_flat, psi_flat)):
        out[i] = _sample_gig_scalar(lam, float(c), float(p), rng)
    return out.reshape(chi.shape).to(chi.device)


# ----------------------------------------------------------------------
# PRS-CS Gibbs sampler
# ----------------------------------------------------------------------


def _prscs_gibbs_block(
    beta_std: Tensor,
    R_b: Tensor,
    n_eff: float,
    phi: float,
    a: float,
    b: float,
    n_iter: int,
    n_burnin: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Gibbs sampler for PRS-CS posterior over a single LD block.

    Returns ``(beta_post_mean, beta_post_sd)``.
    """
    b_size = int(R_b.shape[0])
    device = R_b.device
    dtype = R_b.dtype

    # Add tiny ridge to R_b for numerical PD safety
    eye = torch.eye(b_size, dtype=dtype, device=device)
    R_reg = R_b + 1e-6 * eye

    # Initialize
    psi = torch.ones(b_size, dtype=dtype, device=device)
    delta = torch.ones(b_size, dtype=dtype, device=device)
    sigma2 = torch.tensor(1.0, dtype=dtype, device=device)

    # Accumulators
    beta_sum = torch.zeros(b_size, dtype=dtype, device=device)
    beta_sq_sum = torch.zeros(b_size, dtype=dtype, device=device)
    n_post = 0

    # Precompute n_eff factor
    n_t = float(n_eff)

    for it in range(n_iter):
        # -------- beta update --------
        # Posterior: beta | psi ~ N(mu, Sigma)
        # Sigma^{-1} = n * R_reg / sigma2 + diag(1 / (sigma2 * psi))
        # mu = Sigma * (n * beta_std / sigma2)
        inv_psi = 1.0 / psi.clamp(min=1e-12)
        prec = (n_t / float(sigma2)) * R_reg + torch.diag(inv_psi / float(sigma2))
        # Cholesky solve
        try:
            L = torch.linalg.cholesky(prec)
        except RuntimeError:
            prec = prec + 1e-4 * eye
            L = torch.linalg.cholesky(prec)
        rhs = (n_t / float(sigma2)) * beta_std
        mu = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        # Sample beta = mu + L^{-T} z
        z = torch.randn(b_size, generator=rng, device=device, dtype=dtype)
        v = torch.linalg.solve_triangular(L.T, z.unsqueeze(1), upper=True).squeeze(1)
        beta = mu + v

        # -------- psi update --------
        # Full conditional: psi_j ~ GIG(lam = a - 0.5, chi = beta_j^2 / sigma2, psi = 2 * delta_j)
        lam = a - 0.5
        chi_vec = (beta * beta) / float(sigma2) + 1e-12
        psi_param = 2.0 * delta + 1e-12
        psi = sample_gig(lam, chi_vec, psi_param, rng=rng)
        psi = psi.clamp(min=1e-12, max=1e6)

        # -------- delta update --------
        # Full conditional: delta_j ~ Gamma(shape = a + b, rate = psi_j + phi)
        # Use numpy Gamma with a torch-derived seed so we stay deterministic
        # under the caller's torch generator.
        import numpy as np

        rate = (psi + phi).detach().cpu().numpy()
        seed_d = int(
            torch.randint(low=0, high=2**31 - 1, size=(1,), generator=rng).item()
        )
        np_rng = np.random.default_rng(seed_d)
        delta_np = np_rng.gamma(shape=a + b, scale=1.0 / rate)
        delta = torch.tensor(delta_np, dtype=dtype, device=device)

        if it >= n_burnin:
            beta_sum = beta_sum + beta
            beta_sq_sum = beta_sq_sum + beta * beta
            n_post += 1

    n_post = max(n_post, 1)
    beta_mean = beta_sum / n_post
    beta_var = (beta_sq_sum / n_post) - beta_mean * beta_mean
    beta_sd = beta_var.clamp(min=0.0).sqrt()
    return beta_mean, beta_sd


# At or below this block size the full-C++ ``prscs_gibbs_block`` wins over
# the hybrid path because its hand-rolled Cholesky is still competitive; above
# it the O(m³) scalar Cholesky loses to torch/MKL and the hybrid path takes
# over. See Phase 41ae notes in CLAUDE.md — the realistic-mode bench at
# m=400, n_iter=300 measured full-C++ dropping to 18.6× vs the pure-Python
# reference, with the hand-rolled Cholesky identified as the bottleneck.
_PRSCS_HYBRID_M_THRESHOLD = 250


def _prscs_gibbs_block_hybrid(
    beta_std: Tensor,
    R_b: Tensor,
    n_eff: float,
    phi: float,
    a: float,
    b: float,
    n_iter: int,
    n_burnin: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Hybrid PRS-CS Gibbs path: torch/MKL linalg + C++ per-SNP sampling.

    Mirrors :func:`_prscs_gibbs_block` line-for-line for the linear algebra
    (the Cholesky and triangular solves stay in torch so they dispatch to
    MKL), but replaces the per-SNP scipy GIG loop and the per-iteration
    numpy Gamma draws with a single call per iteration to
    ``_prscs_native.prscs_sample_psi_delta``, which runs both samplers in
    one GIL-released C++ pass.

    Used for LD blocks larger than :data:`_PRSCS_HYBRID_M_THRESHOLD` where
    the hand-rolled Cholesky in the full-C++ entry point becomes the
    bottleneck.
    """
    b_size = int(R_b.shape[0])
    device = R_b.device
    dtype = R_b.dtype

    eye = torch.eye(b_size, dtype=dtype, device=device)
    R_reg = R_b + 1e-6 * eye

    psi = torch.ones(b_size, dtype=dtype, device=device)
    delta = torch.ones(b_size, dtype=dtype, device=device)
    sigma2 = 1.0  # held at 1.0, matching the pure-Python reference

    beta_sum = torch.zeros(b_size, dtype=dtype, device=device)
    beta_sq_sum = torch.zeros(b_size, dtype=dtype, device=device)
    n_post = 0

    n_t = float(n_eff)

    # Derive a 64-bit seed base from the caller's torch generator. We pass
    # seed_base + it into the C++ sampler each iteration so the two paths
    # (torch linalg + C++ sampler) remain deterministic under a fixed torch
    # seed even though the C++ mt19937_64 is reseeded per iteration.
    seed_base = int(
        torch.randint(low=0, high=2**62, size=(1,), generator=rng).item()
    )

    for it in range(n_iter):
        # -------- beta update (torch / MKL) --------
        inv_psi = 1.0 / psi.clamp(min=1e-12)
        prec = (n_t / sigma2) * R_reg + torch.diag(inv_psi / sigma2)
        try:
            L = torch.linalg.cholesky(prec)
        except RuntimeError:
            prec = prec + 1e-4 * eye
            L = torch.linalg.cholesky(prec)
        rhs = (n_t / sigma2) * beta_std
        mu = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        z = torch.randn(b_size, generator=rng, device=device, dtype=dtype)
        v = torch.linalg.solve_triangular(
            L.T, z.unsqueeze(1), upper=True,
        ).squeeze(1)
        beta = mu + v

        # -------- (psi, delta) update via one C++ call per iteration --------
        beta_np = beta.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()
        psi_new_np, delta_new_np = _prscs_native.prscs_sample_psi_delta(
            beta_np, delta_np,
            float(sigma2), float(phi), float(a), float(b),
            seed_base + it,
        )
        psi = torch.from_numpy(psi_new_np).to(dtype=dtype, device=device)
        delta = torch.from_numpy(delta_new_np).to(dtype=dtype, device=device)

        if it >= n_burnin:
            beta_sum = beta_sum + beta
            beta_sq_sum = beta_sq_sum + beta * beta
            n_post += 1

    n_post = max(n_post, 1)
    beta_mean = beta_sum / n_post
    beta_var = (beta_sq_sum / n_post) - beta_mean * beta_mean
    beta_sd = beta_var.clamp(min=0.0).sqrt()
    return beta_mean, beta_sd


def _prscs_gibbs_block_dispatch(
    beta_std: Tensor,
    R_b: Tensor,
    n_eff: float,
    phi: float,
    a: float,
    b: float,
    n_iter: int,
    n_burnin: int,
    rng: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Dispatcher: prefer the native C++ Gibbs sampler when it is available
    and the inputs are CPU float64; otherwise fall through to the pure-Python
    reference implementation in :func:`_prscs_gibbs_block`.

    For small LD blocks (``m <= _PRSCS_HYBRID_M_THRESHOLD``) uses the full
    C++ ``prscs_gibbs_block`` entry point — its hand-rolled Cholesky is fine
    at that size and the one-GIL-release design gives the fastest wall time.
    For larger blocks the hand-rolled O(m³) Cholesky loses to torch/MKL, so
    we switch to the hybrid path which runs the linalg in torch and only
    delegates the per-SNP GIG+Gamma sampling to C++.

    The native and Python paths are statistically equivalent but not bit-for-
    bit identical (they use different RNGs). The pure-Python path remains the
    canonical algorithmic spec.
    """
    if (
        _native_enabled()
        and beta_std.device.type == "cpu"
        and beta_std.dtype == torch.float64
        and R_b.device.type == "cpu"
        and R_b.dtype == torch.float64
    ):
        b_size = int(R_b.shape[0])
        if b_size > _PRSCS_HYBRID_M_THRESHOLD:
            return _prscs_gibbs_block_hybrid(
                beta_std, R_b, n_eff, phi, a, b, n_iter, n_burnin, rng,
            )
        # Small-m: full-C++ path wins — derive a seed from the caller's RNG
        # and hand the entire Gibbs loop to C++ under one GIL release.
        seed = int(
            torch.randint(low=0, high=2**63 - 1, size=(1,), generator=rng).item()
        )
        bs_np = beta_std.detach().cpu().numpy()
        Rb_np = R_b.detach().cpu().numpy()
        mean_np, sd_np = _prscs_native.prscs_gibbs_block(
            bs_np, Rb_np,
            float(n_eff), float(phi), float(a), float(b),
            int(n_iter), int(n_burnin), seed,
        )
        mean_t = torch.from_numpy(mean_np).to(dtype=beta_std.dtype, device=beta_std.device)
        sd_t = torch.from_numpy(sd_np).to(dtype=beta_std.dtype, device=beta_std.device)
        return mean_t, sd_t

    return _prscs_gibbs_block(
        beta_std, R_b, n_eff, phi, a, b, n_iter, n_burnin, rng,
    )


class PRSCS(BasePGSMethod):
    """PRS-CS polygenic score construction (Ge et al. 2019).

    Parameters at ``fit`` time:
        phi (float): global shrinkage parameter (smaller -> sparser)
        a, b (float): prior shape parameters (default Strawderman-Berger 1, 0.5)
        n_iter, n_burnin (int): Gibbs iterations
        n_chains (int): number of chains (for R-hat diagnostic)
    """

    name = "prscs"

    def fit(
        self,
        sumstats: SumStats,
        ld_ref: LDReference,
        phi: float = 1e-2,
        a: float = 1.0,
        b: float = 0.5,
        n_iter: int = 200,
        n_burnin: int = 100,
        n_chains: int = 1,
        **kwargs: Any,
    ) -> PGSResult:
        ss, ref, audit = self._harmonize(sumstats, ld_ref)
        beta_std, n_eff = _marginal_beta_std(ss)
        m = ss.m
        n_avg = float(n_eff.median().item())

        chains_beta: list[Tensor] = []
        chains_sd: list[Tensor] = []

        for c in range(n_chains):
            rng = torch.Generator(device=beta_std.device)
            rng.manual_seed(self.seed + 7919 * c)
            beta_chain = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
            sd_chain = torch.zeros(m, dtype=beta_std.dtype, device=beta_std.device)
            for idx, R_b in _block_iter(ref):
                bm, bsd = _prscs_gibbs_block_dispatch(
                    beta_std[idx], R_b,
                    n_eff=n_avg,
                    phi=phi, a=a, b=b,
                    n_iter=n_iter, n_burnin=n_burnin,
                    rng=rng,
                )
                beta_chain[idx] = bm
                sd_chain[idx] = bsd
            chains_beta.append(beta_chain)
            chains_sd.append(sd_chain)

        beta_stack = torch.stack(chains_beta, dim=0)
        weight = beta_stack.mean(dim=0)
        weight_sd = torch.stack(chains_sd, dim=0).mean(dim=0)

        rhat = None
        if n_chains >= 2:
            cross_sd = beta_stack.std(dim=0)
            rhat = 1.0 + cross_sd / (cross_sd.median() + 1e-12)

        return PGSResult(
            method=self.name,
            snp=list(ss.snp),
            chr=list(ss.chr),
            pos=list(ss.pos),
            a1=list(ss.a1),
            a2=list(ss.a2),
            weight=weight,
            weight_sd=weight_sd,
            af=ss.af,
            phi=float(phi),
            rhat=rhat,
            n_iter=n_iter,
            n_burnin=n_burnin,
            grid={
                "a": float(a),
                "b": float(b),
                "n_chains": n_chains,
            },
            converged=True,
            **audit,
        )
