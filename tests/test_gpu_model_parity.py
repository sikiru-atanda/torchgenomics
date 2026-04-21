"""CPU ↔ GPU parity tests for the model-scan surface.

These tests build small fixtures on CPU, fit the null + score a chunk, then
repeat the same inputs on CUDA (if available) and assert that
``(beta, se, p)`` match at the FP64 level. Every test is gated by
``torch.cuda.is_available`` — the file is a no-op on CPU-only CI but becomes a
hard gate on a GPU runner.

Pattern mirrors ``tests/test_impute_gpu_kernels.py``: same fixture on two
devices, ``torch.allclose`` tolerance ``atol=1e-6`` (well inside Section-16
agreement with GEMMA). NaN p-values (rare-allele / non-variable SNPs) are
masked out before comparison on both sides.
"""
from __future__ import annotations

from typing import Any

import pytest
import torch

from torchgwas.config import STAT_DTYPE, NumericalConfig
from torchgwas.linalg.kinship import grm_vanraden
from torchgwas.models import (
    BLINK,
    FarmCPU,
    GxELMM,
    HaplotypeGWAS,
    HetLMM,
    MultiEnvLMM,
    MultiKernelLMM,
    MultiTraitLMM,
    SingleTraitLMM,
    VariantMeta,
    build_multi_kernels,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

TOL_ABS = 1e-6
TOL_REL = 1e-5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fixture(n: int = 120, m: int = 80, seed: int = 7):
    """Build a shared (G, Y, K, X0, vmeta) fixture for single-trait models."""
    gen = torch.Generator().manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE, generator=gen)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    # Phenotype: polygenic background + planted SNP at index 10.
    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=STAT_DTYPE))
    z = torch.randn(n, generator=gen, dtype=STAT_DTYPE)
    Y = (
        L @ z * 0.7
        + G[:, 10] * 0.4
        + torch.randn(n, generator=gen, dtype=STAT_DTYPE) * 0.5
    )
    Y = Y.unsqueeze(1)

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m,
        a2=["G"] * m,
    )
    return G, Y, K, X0, vmeta


def _to_device(*tensors: Any, device: str):
    """Move a tuple of tensors to a device, leaving non-tensors untouched."""
    out = []
    for t in tensors:
        if torch.is_tensor(t):
            out.append(t.to(device))
        else:
            out.append(t)
    return out


def _assert_scan_parity(res_cpu, res_gpu, name: str):
    """Compare (beta, se, p) for a ScanResult on CPU vs CUDA."""
    beta_cpu = res_cpu.beta.cpu()
    beta_gpu = res_gpu.beta.cpu()
    se_cpu = res_cpu.se.cpu()
    se_gpu = res_gpu.se.cpu()
    p_cpu = res_cpu.p.cpu()
    p_gpu = res_gpu.p.cpu()

    # Coefficient tensors may be 1-D or 2-D; flatten for a single mask.
    mask = (
        torch.isfinite(beta_cpu.flatten())
        & torch.isfinite(beta_gpu.flatten())
        & torch.isfinite(p_cpu.flatten())
        & torch.isfinite(p_gpu.flatten())
    )
    assert mask.any(), f"{name}: no finite entries to compare"

    bc = beta_cpu.flatten()[mask]
    bg = beta_gpu.flatten()[mask]
    sc = se_cpu.flatten()[mask]
    sg = se_gpu.flatten()[mask]
    pc = p_cpu.flatten()[mask]
    pg = p_gpu.flatten()[mask]

    assert torch.allclose(bc, bg, atol=TOL_ABS, rtol=TOL_REL), (
        f"{name}: beta mismatch max={float((bc - bg).abs().max()):.2e}"
    )
    assert torch.allclose(sc, sg, atol=TOL_ABS, rtol=TOL_REL), (
        f"{name}: se mismatch max={float((sc - sg).abs().max()):.2e}"
    )
    assert torch.allclose(pc, pg, atol=TOL_ABS, rtol=TOL_REL), (
        f"{name}: p mismatch max={float((pc - pg).abs().max()):.2e}"
    )


# ---------------------------------------------------------------------------
# Baseline: each test exercises the CPU path even without a GPU
# ---------------------------------------------------------------------------


def test_fixture_cpu_baseline_runs():
    """Smoke check that the fixture is self-consistent on CPU."""
    G, Y, K, X0, vmeta = _make_fixture()
    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(Y, X0, K=K)
    res = model.score_chunk(G, nf, vmeta, test="wald")
    assert res.beta.shape[0] == G.shape[1]
    assert torch.isfinite(res.p).any()


# ---------------------------------------------------------------------------
# SingleTraitLMM — wald / lrt / score
# ---------------------------------------------------------------------------


@cuda_required
@pytest.mark.parametrize("test", ["wald", "lrt", "score"])
def test_single_trait_lmm_parity(test):
    G, Y, K, X0, vmeta = _make_fixture()

    # CPU pass
    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf_cpu = model.fit_null(Y, X0, K=K)
    res_cpu = model.score_chunk(G, nf_cpu, vmeta, test=test)

    # CUDA pass — identical inputs, different device
    G_g, Y_g, K_g, X0_g = _to_device(G, Y, K, X0, device="cuda")
    model_g = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf_gpu = model_g.fit_null(Y_g, X0_g, K=K_g)
    res_gpu = model_g.score_chunk(G_g, nf_gpu, vmeta, test=test)

    _assert_scan_parity(res_cpu, res_gpu, f"SingleTraitLMM/{test}")


# ---------------------------------------------------------------------------
# MultiTraitLMM — d=2 fixture
# ---------------------------------------------------------------------------


@cuda_required
def test_multi_trait_lmm_parity():
    gen = torch.Generator().manual_seed(13)
    n, m, d = 100, 60, 2
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE, generator=gen)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=STAT_DTYPE))
    U = torch.randn(n, d, generator=gen, dtype=STAT_DTYPE)
    Y = L @ U * 0.5 + torch.randn(n, d, generator=gen, dtype=STAT_DTYPE) * 0.5
    Y[:, 0] = Y[:, 0] + G[:, 5] * 0.4  # plant signal on trait 0

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)], chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )

    model_cpu = MultiTraitLMM()
    nf_cpu = model_cpu.fit_null(Y, X0, K=K)
    res_cpu = model_cpu.score_chunk(G, nf_cpu, vmeta, test="wald")

    G_g, Y_g, K_g, X0_g = _to_device(G, Y, K, X0, device="cuda")
    model_g = MultiTraitLMM()
    nf_gpu = model_g.fit_null(Y_g, X0_g, K=K_g)
    res_gpu = model_g.score_chunk(G_g, nf_gpu, vmeta, test="wald")

    _assert_scan_parity(res_cpu, res_gpu, "MultiTraitLMM")


# ---------------------------------------------------------------------------
# MultiKernelLMM — additive + dominance
# ---------------------------------------------------------------------------


@cuda_required
def test_multi_kernel_lmm_parity():
    G, Y, _K, X0, vmeta = _make_fixture(n=100, m=60, seed=23)
    kernels_cpu, kernel_names = build_multi_kernels(G, ploidy=2)

    model_cpu = MultiKernelLMM()
    nf_cpu = model_cpu.fit_null(
        Y, X0, kernels=kernels_cpu, kernel_names=kernel_names,
    )
    res_cpu = model_cpu.score_chunk(G, nf_cpu, vmeta, test="wald")

    G_g, Y_g, X0_g = _to_device(G, Y, X0, device="cuda")
    kernels_g = [k.to("cuda") for k in kernels_cpu]
    model_g = MultiKernelLMM()
    nf_gpu = model_g.fit_null(
        Y_g, X0_g, kernels=kernels_g, kernel_names=kernel_names,
    )
    res_gpu = model_g.score_chunk(G_g, nf_gpu, vmeta, test="wald")

    _assert_scan_parity(res_cpu, res_gpu, "MultiKernelLMM")


# ---------------------------------------------------------------------------
# GxELMM + HetLMM — interaction scans
# ---------------------------------------------------------------------------


def _gxe_fixture(n: int = 120, m: int = 60, seed: int = 31):
    gen = torch.Generator().manual_seed(seed)
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE, generator=gen)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)
    env = torch.randn(n, generator=gen, dtype=STAT_DTYPE)

    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=STAT_DTYPE))
    z = torch.randn(n, generator=gen, dtype=STAT_DTYPE)
    Y = (
        L @ z * 0.6
        + G[:, 7] * 0.3
        + G[:, 7] * env * 0.3
        + torch.randn(n, generator=gen, dtype=STAT_DTYPE) * 0.5
    ).unsqueeze(1)

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)], chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )
    return G, Y, K, X0, env, vmeta


@cuda_required
@pytest.mark.parametrize("cls", [GxELMM, HetLMM])
def test_gxe_het_lmm_parity(cls):
    G, Y, K, X0, env, vmeta = _gxe_fixture()

    model_cpu = cls()
    nf_cpu = model_cpu.fit_null(Y, X0, K=K, env=env)
    res_cpu = model_cpu.score_chunk(G, nf_cpu, vmeta, test="wald")

    G_g, Y_g, K_g, X0_g, env_g = _to_device(G, Y, K, X0, env, device="cuda")
    model_g = cls()
    nf_gpu = model_g.fit_null(Y_g, X0_g, K=K_g, env=env_g)
    res_gpu = model_g.score_chunk(G_g, nf_gpu, vmeta, test="wald")

    # GxE results carry main/interaction/joint p-values — compare the joint
    # test via the standard (beta, se, p) attributes of the base result.
    _assert_scan_parity(res_cpu, res_gpu, cls.__name__)


# ---------------------------------------------------------------------------
# MultiEnvLMM — reaction-norm
# ---------------------------------------------------------------------------


@cuda_required
def test_multi_env_lmm_parity():
    gen = torch.Generator().manual_seed(41)
    n, m, E = 100, 50, 3
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE, generator=gen)
    K, _ = grm_vanraden(G)
    X0 = torch.ones(n, 1, dtype=STAT_DTYPE)

    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=STAT_DTYPE))
    Z = torch.randn(n, E, generator=gen, dtype=STAT_DTYPE)
    Vg = torch.eye(E, dtype=STAT_DTYPE) * 0.6 + 0.3
    Lg = torch.linalg.cholesky(Vg)
    Y = L @ Z @ Lg.T + torch.randn(n, E, generator=gen, dtype=STAT_DTYPE) * 0.5
    Y = Y + torch.outer(
        G[:, 12].to(STAT_DTYPE), torch.tensor([0.5, -0.3, 0.2], dtype=STAT_DTYPE),
    )

    vmeta = VariantMeta(
        snp=[f"rs{i}" for i in range(m)], chr=["1"] * m,
        pos=list(range(1000, 1000 + m * 500, 500)),
        a1=["A"] * m, a2=["G"] * m,
    )
    env_names = [f"E{i+1}" for i in range(E)]

    model_cpu = MultiEnvLMM(parameterization="reaction_norm")
    nf_cpu = model_cpu.fit_null(Y, X0, K, env_names=env_names)
    res_cpu = model_cpu.score_chunk(G, nf_cpu, vmeta)

    G_g, Y_g, K_g, X0_g = _to_device(G, Y, K, X0, device="cuda")
    model_g = MultiEnvLMM(parameterization="reaction_norm")
    nf_gpu = model_g.fit_null(Y_g, X0_g, K_g, env_names=env_names)
    res_gpu = model_g.score_chunk(G_g, nf_gpu, vmeta)

    # MET result: beta is (m, E); use the overall p vector
    p_cpu = res_cpu.p.cpu()
    p_gpu = res_gpu.p.cpu()
    mask = torch.isfinite(p_cpu) & torch.isfinite(p_gpu)
    assert mask.any()
    assert torch.allclose(
        p_cpu[mask], p_gpu[mask], atol=TOL_ABS, rtol=TOL_REL,
    ), f"MultiEnvLMM: p mismatch max={float((p_cpu[mask] - p_gpu[mask]).abs().max()):.2e}"

    beta_cpu = res_cpu.beta.cpu()
    beta_gpu = res_gpu.beta.cpu()
    mask_b = torch.isfinite(beta_cpu) & torch.isfinite(beta_gpu)
    assert torch.allclose(
        beta_cpu[mask_b], beta_gpu[mask_b], atol=TOL_ABS, rtol=TOL_REL,
    )


# ---------------------------------------------------------------------------
# FarmCPU + BLINK — iterative scans
# ---------------------------------------------------------------------------


@cuda_required
@pytest.mark.parametrize("cls", [FarmCPU, BLINK])
def test_farmcpu_blink_parity(cls):
    # Iterative scans need a slightly larger fixture for meaningful QTN picks.
    G, Y, _K, X0, vmeta = _make_fixture(n=200, m=120, seed=57)

    scanner_cpu = cls()
    nf_cpu = scanner_cpu.fit_null(Y, X0)
    res_cpu = scanner_cpu.score_chunk(G, nf_cpu, vmeta, test="wald")

    G_g, Y_g, X0_g = _to_device(G, Y, X0, device="cuda")
    scanner_g = cls()
    nf_gpu = scanner_g.fit_null(Y_g, X0_g)
    res_gpu = scanner_g.score_chunk(G_g, nf_gpu, vmeta, test="wald")

    # FarmCPU / BLINK re-select QTNs from p-values; allow a wider tolerance
    # because one numerical tie can flip a pick. Require top-K overlap instead.
    p_cpu = res_cpu.p.cpu()
    p_gpu = res_gpu.p.cpu()
    mask = torch.isfinite(p_cpu) & torch.isfinite(p_gpu)
    top_k = 10
    top_cpu = set(torch.topk(p_cpu[mask], top_k, largest=False).indices.tolist())
    top_gpu = set(torch.topk(p_gpu[mask], top_k, largest=False).indices.tolist())
    overlap = len(top_cpu & top_gpu)
    assert overlap >= 7, (
        f"{cls.__name__}: only {overlap}/{top_k} top SNPs overlap CPU↔CUDA"
    )


# ---------------------------------------------------------------------------
# HaplotypeGWAS — window mode
# ---------------------------------------------------------------------------


@cuda_required
def test_haplotype_gwas_parity():
    gen = torch.Generator().manual_seed(61)
    n, m = 150, 40
    G = torch.randint(0, 3, (n, m), dtype=STAT_DTYPE, generator=gen)
    Y = (
        G[:, 10:15] @ torch.tensor([0.3, 0.4, -0.3, 0.3, 0.25], dtype=STAT_DTYPE)
        + torch.randn(n, generator=gen, dtype=STAT_DTYPE) * 0.5
    )
    variant_pos = list(range(1000, 1000 + m * 1000, 1000))
    variant_chr = ["1"] * m

    scanner_cpu = HaplotypeGWAS(
        method="window", test="f_test", window_size=5, step=2,
        min_hap_freq=0.02, max_haplotypes=10,
    )
    res_cpu = scanner_cpu.scan(
        Y, G, variant_pos=variant_pos, variant_chr=variant_chr,
    )

    scanner_g = HaplotypeGWAS(
        method="window", test="f_test", window_size=5, step=2,
        min_hap_freq=0.02, max_haplotypes=10,
    )
    res_gpu = scanner_g.scan(
        Y.cuda(), G.cuda(),
        variant_pos=variant_pos, variant_chr=variant_chr,
    )

    p_cpu = res_cpu.p_global.cpu()
    p_gpu = res_gpu.p_global.cpu()
    mask = torch.isfinite(p_cpu) & torch.isfinite(p_gpu)
    assert mask.any()
    assert torch.allclose(
        p_cpu[mask], p_gpu[mask], atol=1e-5, rtol=1e-4,
    ), f"HaplotypeGWAS: p mismatch max={float((p_cpu[mask] - p_gpu[mask]).abs().max()):.2e}"


# ---------------------------------------------------------------------------
# Cross-device safety: a CUDA input fed to a freshly-built model must
# produce CUDA outputs (no silent host round-trip).
# ---------------------------------------------------------------------------


@cuda_required
def test_single_trait_lmm_produces_cuda_outputs():
    G, Y, K, X0, vmeta = _make_fixture()
    G_g, Y_g, K_g, X0_g = _to_device(G, Y, K, X0, device="cuda")

    model = SingleTraitLMM(config=NumericalConfig(reml_method="emma"))
    nf = model.fit_null(Y_g, X0_g, K=K_g)
    res = model.score_chunk(G_g, nf, vmeta, test="wald")

    assert res.beta.device.type == "cuda"
    assert res.se.device.type == "cuda"
    assert res.p.device.type == "cuda"
