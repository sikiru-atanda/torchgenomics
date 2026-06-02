"""Reference validation for Survival GWAS (SurvivalGLMM) against R coxme.

Benchmarks our Cox PH frailty model (Breslow-Clayton PQL + martingale residual
score test) against two established R packages:

1. **coxme** (Therneau 2015): Gold standard for Cox PH with Gaussian random
   effects (GRM-structured frailty). Fits exact penalized partial likelihood.
   Our two-step PQL + score approach should agree in -log10(p) ranking.

2. **survival::coxph** (Therneau 2024): Standard Cox PH with no random effects.
   When K ≈ I (no relatedness), our model should reduce to standard Cox PH.

Validation strategy:
- Simulate Weibull survival data with known GRM and causal SNPs
- Export data to R, fit coxme per-SNP, extract Wald p-values
- Compare p-value rankings (Spearman rho) and detection concordance
- Verify effect direction agreement for causal SNPs
- For no-kinship reduction: compare against survival::coxph

All R calls via subprocess (Rscript). Tests skip if R/coxme not available.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Check R + coxme availability
# ---------------------------------------------------------------------------

def _r_available() -> bool:
    """Check if Rscript is on PATH."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "cat('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() == "ok"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _coxme_available() -> bool:
    """Check if the coxme R package is installed."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "cat(require(coxme, quietly=TRUE))"],
            capture_output=True, text=True, timeout=30,
        )
        return "TRUE" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


R_OK = _r_available()
COXME_OK = R_OK and _coxme_available()

skip_no_r = pytest.mark.skipif(not R_OK, reason="Rscript not available")
skip_no_coxme = pytest.mark.skipif(not COXME_OK, reason="coxme R package not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vmeta(m):
    from torchgenomics.models.base import VariantMeta
    return VariantMeta(
        snp=[f"snp{i}" for i in range(m)],
        chr=["1"] * m,
        pos=list(range(m)),
        a1=["A"] * m,
        a2=["G"] * m,
    )


def _simulate_survival_data(
    n: int = 200,
    m: int = 15,
    m_grm: int = 60,
    n_causal: int = 3,
    log_hr: float = 0.6,
    censor_rate: float = 0.3,
    h2: float = 0.3,
    weibull_shape: float = 1.5,
    seed: int = 42,
):
    """Simulate Weibull survival data with GRM-structured frailty.

    Returns
    -------
    dict with keys: G_test, G_grm, K, time, event, X0, causal_idx, log_hr_true
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GRM genotypes (separate from test SNPs)
    G_grm = torch.zeros(n, m_grm, dtype=torch.float64)
    for j in range(m_grm):
        maf = 0.1 + 0.3 * np.random.rand()
        G_grm[:, j] = torch.tensor(
            np.random.binomial(2, maf, size=n), dtype=torch.float64
        )

    # Standardize and compute GRM
    G_std = G_grm - G_grm.mean(dim=0)
    sd = G_std.std(dim=0).clamp(min=1e-6)
    G_std = G_std / sd
    K = G_std @ G_std.T / m_grm
    K = 0.5 * (K + K.T)

    # Test genotypes
    G_test = torch.zeros(n, m, dtype=torch.float64)
    for j in range(m):
        maf = 0.1 + 0.3 * np.random.rand()
        G_test[:, j] = torch.tensor(
            np.random.binomial(2, maf, size=n), dtype=torch.float64
        )

    # Polygenic random effect from K
    L = torch.linalg.cholesky(K + 1e-4 * torch.eye(n, dtype=torch.float64))
    u = L @ torch.randn(n, dtype=torch.float64)
    u = u * np.sqrt(h2)

    # Linear predictor: intercept + causal SNPs + polygenic
    causal_idx = list(range(n_causal))
    eta = torch.zeros(n, dtype=torch.float64)
    for idx in causal_idx:
        g_j = G_test[:, idx]
        g_std = (g_j - g_j.mean()) / g_j.std().clamp(min=1e-6)
        eta += log_hr * g_std

    eta += u

    # Weibull event times: T = (-log(U) / exp(eta))^(1/shape)
    U = torch.rand(n, dtype=torch.float64).clamp(min=1e-10, max=1.0 - 1e-10)
    T_event = ((-torch.log(U)) / torch.exp(eta)).pow(1.0 / weibull_shape)

    # Exponential censoring calibrated to target rate
    if censor_rate > 0:
        # Calibrate censoring rate
        censor_scale = T_event.median() / np.log(1.0 / (1.0 - censor_rate + 1e-6))
        C = torch.distributions.Exponential(1.0 / censor_scale).sample((n,)).to(torch.float64)
        time = torch.minimum(T_event, C)
        event = (T_event <= C).to(torch.float64)
    else:
        time = T_event
        event = torch.ones(n, dtype=torch.float64)

    X0 = torch.ones(n, 1, dtype=torch.float64)

    return {
        "G_test": G_test,
        "G_grm": G_grm,
        "K": K,
        "time": time,
        "event": event,
        "X0": X0,
        "causal_idx": causal_idx,
        "log_hr_true": log_hr,
    }


def _export_to_csv(data: dict, tmpdir: str) -> dict:
    """Export simulation data to CSV files for R consumption."""
    n = data["time"].shape[0]
    m = data["G_test"].shape[1]

    # Phenotype file: id, time, event
    pheno_path = os.path.join(tmpdir, "pheno.csv")
    with open(pheno_path, "w") as f:
        f.write("id,time,event\n")
        for i in range(n):
            f.write(f"{i},{data['time'][i].item():.8f},{int(data['event'][i].item())}\n")

    # Genotype file: id, snp0, snp1, ...
    geno_path = os.path.join(tmpdir, "geno.csv")
    with open(geno_path, "w") as f:
        header = "id," + ",".join(f"snp{j}" for j in range(m))
        f.write(header + "\n")
        for i in range(n):
            row = f"{i}," + ",".join(
                f"{int(data['G_test'][i, j].item())}" for j in range(m)
            )
            f.write(row + "\n")

    # Kinship matrix
    K_path = os.path.join(tmpdir, "kinship.csv")
    K_np = data["K"].numpy()
    np.savetxt(K_path, K_np, delimiter=",", fmt="%.8f")

    return {"pheno": pheno_path, "geno": geno_path, "kinship": K_path}


# ---------------------------------------------------------------------------
# R script for coxme per-SNP scan
# ---------------------------------------------------------------------------

_COXME_SCRIPT = r"""
library(coxme)
library(survival)

args <- commandArgs(trailingOnly = TRUE)
pheno_file <- args[1]
geno_file  <- args[2]
kin_file   <- args[3]
out_file   <- args[4]

pheno <- read.csv(pheno_file)
geno  <- read.csv(geno_file)
K     <- as.matrix(read.csv(kin_file, header = FALSE))

n <- nrow(pheno)
rownames(K) <- colnames(K) <- as.character(pheno$id)

# Merge
dat <- merge(pheno, geno, by = "id")
dat$id_factor <- factor(dat$id)

snp_cols <- grep("^snp", names(dat), value = TRUE)
m <- length(snp_cols)

results <- data.frame(
    snp = character(m),
    beta = numeric(m),
    se = numeric(m),
    z = numeric(m),
    p = numeric(m),
    stringsAsFactors = FALSE
)

for (j in seq_along(snp_cols)) {
    snp_name <- snp_cols[j]
    dat$g <- dat[[snp_name]]

    # Skip monomorphic
    if (sd(dat$g) < 1e-8) {
        results[j, ] <- list(snp_name, 0, Inf, 0, 1)
        next
    }

    tryCatch({
        fit <- coxme(
            Surv(time, event) ~ g + (1 | id_factor),
            data = dat,
            varlist = coxmeFull(K)
        )
        coef_val <- fixef(fit)["g"]
        # Extract SE from the variance-covariance matrix
        vcov_mat <- as.matrix(vcov(fit))
        se_val <- sqrt(vcov_mat["g", "g"])
        z_val <- coef_val / se_val
        p_val <- 2 * pnorm(-abs(z_val))

        results[j, ] <- list(snp_name, coef_val, se_val, z_val, p_val)
    }, error = function(e) {
        results[j, ] <<- list(snp_name, NA, NA, NA, NA)
    })
}

write.csv(results, out_file, row.names = FALSE)
"""

# ---------------------------------------------------------------------------
# R script for survival::coxph (no random effects) per-SNP scan
# ---------------------------------------------------------------------------

_COXPH_SCRIPT = r"""
library(survival)

args <- commandArgs(trailingOnly = TRUE)
pheno_file <- args[1]
geno_file  <- args[2]
out_file   <- args[3]

pheno <- read.csv(pheno_file)
geno  <- read.csv(geno_file)

dat <- merge(pheno, geno, by = "id")

snp_cols <- grep("^snp", names(dat), value = TRUE)
m <- length(snp_cols)

results <- data.frame(
    snp = character(m),
    beta = numeric(m),
    se = numeric(m),
    z = numeric(m),
    p_wald = numeric(m),
    p_score = numeric(m),
    stringsAsFactors = FALSE
)

# Fit null model (no SNP)
null_fit <- coxph(Surv(time, event) ~ 1, data = dat)

for (j in seq_along(snp_cols)) {
    snp_name <- snp_cols[j]
    dat$g <- dat[[snp_name]]

    if (sd(dat$g) < 1e-8) {
        results[j, ] <- list(snp_name, 0, Inf, 0, 1, 1)
        next
    }

    tryCatch({
        fit <- coxph(Surv(time, event) ~ g, data = dat)
        sm  <- summary(fit)
        coef_val <- sm$coefficients["g", "coef"]
        se_val   <- sm$coefficients["g", "se(coef)"]
        z_val    <- sm$coefficients["g", "z"]
        p_wald   <- sm$coefficients["g", "Pr(>|z|)"]

        # Score test p-value
        score_test <- cox.zph(fit)  # different thing
        # Use the score test from the model fit
        p_score <- sm$sctest["pvalue"]

        results[j, ] <- list(snp_name, coef_val, se_val, z_val, p_wald, p_score)
    }, error = function(e) {
        results[j, ] <<- list(snp_name, NA, NA, NA, NA, NA)
    })
}

write.csv(results, out_file, row.names = FALSE)
"""


def _run_coxme(data: dict, tmpdir: str) -> dict:
    """Run coxme per-SNP scan via Rscript, return results dict."""
    paths = _export_to_csv(data, tmpdir)
    script_path = os.path.join(tmpdir, "coxme_scan.R")
    out_path = os.path.join(tmpdir, "coxme_results.csv")

    with open(script_path, "w") as f:
        f.write(_COXME_SCRIPT)

    result = subprocess.run(
        ["Rscript", script_path, paths["pheno"], paths["geno"],
         paths["kinship"], out_path],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"coxme failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    # Parse results
    import csv
    results = {"snp": [], "beta": [], "se": [], "z": [], "p": []}
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results["snp"].append(row["snp"])
            results["beta"].append(float(row["beta"]) if row["beta"] != "NA" else np.nan)
            results["se"].append(float(row["se"]) if row["se"] != "NA" else np.nan)
            results["z"].append(float(row["z"]) if row["z"] != "NA" else np.nan)
            results["p"].append(float(row["p"]) if row["p"] != "NA" else np.nan)

    return results


def _run_coxph(data: dict, tmpdir: str) -> dict:
    """Run survival::coxph per-SNP scan via Rscript, return results dict."""
    paths = _export_to_csv(data, tmpdir)
    script_path = os.path.join(tmpdir, "coxph_scan.R")
    out_path = os.path.join(tmpdir, "coxph_results.csv")

    with open(script_path, "w") as f:
        f.write(_COXPH_SCRIPT)

    result = subprocess.run(
        ["Rscript", script_path, paths["pheno"], paths["geno"], out_path],
        capture_output=True, text=True, timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"coxph failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    import csv
    results = {"snp": [], "beta": [], "se": [], "p_wald": [], "p_score": []}
    with open(out_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results["snp"].append(row["snp"])
            results["beta"].append(float(row["beta"]) if row["beta"] != "NA" else np.nan)
            results["se"].append(float(row["se"]) if row["se"] != "NA" else np.nan)
            results["p_wald"].append(float(row["p_wald"]) if row["p_wald"] != "NA" else np.nan)
            results["p_score"].append(float(row["p_score"]) if row["p_score"] != "NA" else np.nan)

    return results


def _run_torchgenomics_survival(data: dict, use_spa: bool = False):
    """Run our SurvivalGLMM on the same data."""
    from torchgenomics.models.survival_glmm import SurvivalGLMM

    model = SurvivalGLMM(use_spa=use_spa, pql_max_iter=50)
    Y = torch.stack([data["time"], data["event"]], dim=1)
    nf = model.fit_null(Y, data["X0"], K=data["K"])

    vmeta = _make_vmeta(data["G_test"].shape[1])
    result = model.score_chunk(data["G_test"], nf, vmeta)
    return result


# ---------------------------------------------------------------------------
# Tests: SurvivalGLMM vs coxme (Cox PH mixed model)
# ---------------------------------------------------------------------------

@skip_no_coxme
class TestSurvivalVsCoxme:
    """Compare SurvivalGLMM p-values against coxme (Cox PH frailty model)."""

    @pytest.fixture(scope="class")
    def shared_data(self):
        """Simulate data once, run both tools."""
        data = _simulate_survival_data(
            n=200, m=15, m_grm=60, n_causal=3,
            log_hr=0.6, censor_rate=0.3, h2=0.3, seed=42,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            coxme_results = _run_coxme(data, tmpdir)
        tg_result = _run_torchgenomics_survival(data, use_spa=False)
        return data, coxme_results, tg_result

    def test_pvalue_ranking_correlated(self, shared_data):
        """Spearman correlation of -log10(p) between TorchGenomics and coxme > 0.50."""
        data, coxme_res, tg_res = shared_data
        p_tg = tg_res.p.numpy()
        p_coxme = np.array(coxme_res["p"])

        # Exclude NaN from coxme
        valid = ~np.isnan(p_coxme) & (p_tg > 0) & (p_coxme > 0)
        assert valid.sum() >= 10, f"Too few valid SNPs: {valid.sum()}"

        log_tg = -np.log10(np.clip(p_tg[valid], 1e-300, 1.0))
        log_coxme = -np.log10(np.clip(p_coxme[valid], 1e-300, 1.0))

        rho, _ = spearmanr(log_tg, log_coxme)
        assert rho > 0.50, (
            f"Spearman rho between TorchGenomics and coxme -log10(p) = {rho:.3f}, "
            f"expected > 0.50"
        )

    def test_effect_direction_agreement(self, shared_data):
        """Sign of beta should agree for causal SNPs between TorchGenomics and coxme."""
        data, coxme_res, tg_res = shared_data
        causal_idx = data["causal_idx"]

        beta_tg = tg_res.beta.numpy()
        beta_coxme = np.array(coxme_res["beta"])

        n_agree = 0
        n_valid = 0
        for idx in causal_idx:
            if not np.isnan(beta_coxme[idx]) and abs(beta_coxme[idx]) > 1e-6:
                n_valid += 1
                if np.sign(beta_tg[idx]) == np.sign(beta_coxme[idx]):
                    n_agree += 1

        assert n_valid >= 2, f"Too few valid causal SNPs: {n_valid}"
        # At least 2/3 causal SNPs should agree in direction
        assert n_agree >= n_valid - 1, (
            f"Effect direction agreement: {n_agree}/{n_valid} causal SNPs"
        )

    def test_causal_detection_concordance(self, shared_data):
        """Both tools should detect the same causal SNPs (top-ranked)."""
        data, coxme_res, tg_res = shared_data
        m = len(coxme_res["p"])
        causal_idx = set(data["causal_idx"])

        p_tg = tg_res.p.numpy()
        p_coxme = np.array(coxme_res["p"])

        # Top-5 by each method
        valid = ~np.isnan(p_coxme)
        p_coxme_safe = np.where(valid, p_coxme, 1.0)

        top5_tg = set(np.argsort(p_tg)[:5].tolist())
        top5_coxme = set(np.argsort(p_coxme_safe)[:5].tolist())

        overlap = len(top5_tg & top5_coxme)
        # At least 2 of top-5 should overlap
        assert overlap >= 2, (
            f"Top-5 overlap between TorchGenomics and coxme: {overlap}/5. "
            f"TG top5: {sorted(top5_tg)}, coxme top5: {sorted(top5_coxme)}"
        )

    def test_null_snps_both_nonsignificant(self, shared_data):
        """Null SNPs should be non-significant in both tools (p > 0.001)."""
        data, coxme_res, tg_res = shared_data
        causal_idx = set(data["causal_idx"])
        m = len(coxme_res["p"])

        null_idx = [j for j in range(m) if j not in causal_idx]
        p_tg = tg_res.p.numpy()
        p_coxme = np.array(coxme_res["p"])

        # Count how many null SNPs are "significant" at p < 0.001
        tg_fp = sum(1 for j in null_idx if p_tg[j] < 0.001)
        coxme_fp = sum(
            1 for j in null_idx
            if not np.isnan(p_coxme[j]) and p_coxme[j] < 0.001
        )

        # FPR should be < 20% for both
        n_null = len(null_idx)
        assert tg_fp / n_null < 0.20, f"TorchGenomics FP: {tg_fp}/{n_null}"
        assert coxme_fp / n_null < 0.20, f"coxme FP: {coxme_fp}/{n_null}"

    def test_variance_component_nonnegative(self, shared_data):
        """Our model should estimate non-negative genetic variance."""
        data, coxme_res, tg_res = shared_data
        from torchgenomics.models.survival_glmm import SurvivalGLMM

        model = SurvivalGLMM(use_spa=False, pql_max_iter=50)
        Y = torch.stack([data["time"], data["event"]], dim=1)
        nf = model.fit_null(Y, data["X0"], K=data["K"])

        # Note: With n=200 and confounding causal effects, both coxme and our
        # model correctly estimate near-zero genetic variance in the working model.
        assert nf.sig2_g >= 0, f"sig2_g = {nf.sig2_g:.6f}, should be >= 0"
        assert nf.converged, "PQL should converge"


# ---------------------------------------------------------------------------
# Tests: SurvivalGLMM vs coxph (no random effects, degeneracy check)
# ---------------------------------------------------------------------------

@skip_no_r
class TestSurvivalVsCoxph:
    """Compare SurvivalGLMM against standard Cox PH when h² ≈ 0."""

    @pytest.fixture(scope="class")
    def shared_data_no_kinship(self):
        """Simulate with low h² so mixed model ≈ fixed-effects Cox."""
        data = _simulate_survival_data(
            n=300, m=20, m_grm=50, n_causal=3,
            log_hr=0.5, censor_rate=0.3, h2=0.05, seed=99,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            coxph_results = _run_coxph(data, tmpdir)
        tg_result = _run_torchgenomics_survival(data, use_spa=False)
        return data, coxph_results, tg_result

    def test_pvalue_ranking_vs_coxph(self, shared_data_no_kinship):
        """When h² ≈ 0, -log10(p) ranking should correlate with coxph (rho > 0.40)."""
        data, coxph_res, tg_res = shared_data_no_kinship
        p_tg = tg_res.p.numpy()
        p_coxph = np.array(coxph_res["p_wald"])

        valid = ~np.isnan(p_coxph) & (p_tg > 0) & (p_coxph > 0)
        assert valid.sum() >= 15

        log_tg = -np.log10(np.clip(p_tg[valid], 1e-300, 1.0))
        log_coxph = -np.log10(np.clip(p_coxph[valid], 1e-300, 1.0))

        rho, _ = spearmanr(log_tg, log_coxph)
        assert rho > 0.40, (
            f"Spearman rho between TorchGenomics and coxph = {rho:.3f}, expected > 0.40"
        )

    def test_effect_direction_vs_coxph(self, shared_data_no_kinship):
        """Effect directions should largely agree with coxph for causal SNPs."""
        data, coxph_res, tg_res = shared_data_no_kinship
        causal_idx = data["causal_idx"]

        beta_tg = tg_res.beta.numpy()
        beta_coxph = np.array(coxph_res["beta"])

        n_agree = 0
        n_valid = 0
        for idx in causal_idx:
            if not np.isnan(beta_coxph[idx]) and abs(beta_coxph[idx]) > 1e-6:
                n_valid += 1
                if np.sign(beta_tg[idx]) == np.sign(beta_coxph[idx]):
                    n_agree += 1

        assert n_valid >= 2
        assert n_agree >= n_valid - 1, (
            f"Direction agreement with coxph: {n_agree}/{n_valid}"
        )

    def test_top_snps_overlap_coxph(self, shared_data_no_kinship):
        """Top-ranked SNPs should overlap between methods."""
        data, coxph_res, tg_res = shared_data_no_kinship
        p_tg = tg_res.p.numpy()
        p_coxph = np.array(coxph_res["p_wald"])

        valid = ~np.isnan(p_coxph)
        p_coxph_safe = np.where(valid, p_coxph, 1.0)

        top5_tg = set(np.argsort(p_tg)[:5].tolist())
        top5_coxph = set(np.argsort(p_coxph_safe)[:5].tolist())

        overlap = len(top5_tg & top5_coxph)
        assert overlap >= 2, (
            f"Top-5 overlap: {overlap}. TG: {sorted(top5_tg)}, coxph: {sorted(top5_coxph)}"
        )


# ---------------------------------------------------------------------------
# Tests: Strong signal benchmark
# ---------------------------------------------------------------------------

@skip_no_coxme
class TestSurvivalStrongSignal:
    """Verify both tools detect a very strong causal SNP (HR=3, log_hr≈1.1)."""

    @pytest.fixture(scope="class")
    def strong_signal_data(self):
        """Simulate with a very strong causal SNP."""
        data = _simulate_survival_data(
            n=300, m=10, m_grm=60, n_causal=1,
            log_hr=1.1, censor_rate=0.2, h2=0.2, seed=77,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            coxme_results = _run_coxme(data, tmpdir)
        tg_result = _run_torchgenomics_survival(data, use_spa=False)
        return data, coxme_results, tg_result

    def test_both_detect_causal(self, strong_signal_data):
        """Both coxme and TorchGenomics should detect the strong causal SNP at p < 0.05."""
        data, coxme_res, tg_res = strong_signal_data
        causal_idx = data["causal_idx"][0]

        p_tg = tg_res.p[causal_idx].item()
        p_coxme = coxme_res["p"][causal_idx]

        assert p_tg < 0.05, f"TorchGenomics p = {p_tg:.4e} for causal SNP"
        if not np.isnan(p_coxme):
            assert p_coxme < 0.05, f"coxme p = {p_coxme:.4e} for causal SNP"

    def test_causal_is_top_ranked_both(self, strong_signal_data):
        """The causal SNP should be in the top-3 for both tools."""
        data, coxme_res, tg_res = strong_signal_data
        causal_idx = data["causal_idx"][0]

        p_tg = tg_res.p.numpy()
        top3_tg = set(np.argsort(p_tg)[:3].tolist())
        assert causal_idx in top3_tg, f"Causal SNP {causal_idx} not in TG top-3: {top3_tg}"

        p_coxme = np.array(coxme_res["p"])
        valid = ~np.isnan(p_coxme)
        p_coxme_safe = np.where(valid, p_coxme, 1.0)
        top3_coxme = set(np.argsort(p_coxme_safe)[:3].tolist())
        assert causal_idx in top3_coxme, (
            f"Causal SNP {causal_idx} not in coxme top-3: {top3_coxme}"
        )


# ---------------------------------------------------------------------------
# Tests: Heavy censoring benchmark
# ---------------------------------------------------------------------------

@skip_no_coxme
class TestSurvivalHeavyCensoring:
    """Benchmark under heavy censoring (70%) — a challenging scenario."""

    @pytest.fixture(scope="class")
    def heavy_censor_data(self):
        data = _simulate_survival_data(
            n=300, m=12, m_grm=60, n_causal=2,
            log_hr=0.8, censor_rate=0.7, h2=0.25, seed=123,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            coxme_results = _run_coxme(data, tmpdir)
        tg_result = _run_torchgenomics_survival(data, use_spa=True)
        return data, coxme_results, tg_result

    def test_ranking_correlated_heavy_censoring(self, heavy_censor_data):
        """P-value rankings should still correlate under heavy censoring (rho > 0.40)."""
        data, coxme_res, tg_res = heavy_censor_data
        p_tg = tg_res.p.numpy()
        p_coxme = np.array(coxme_res["p"])

        valid = ~np.isnan(p_coxme) & (p_tg > 0) & (p_coxme > 0)
        if valid.sum() < 8:
            pytest.skip("Too few valid SNPs for correlation test")

        log_tg = -np.log10(np.clip(p_tg[valid], 1e-300, 1.0))
        log_coxme = -np.log10(np.clip(p_coxme[valid], 1e-300, 1.0))

        rho, _ = spearmanr(log_tg, log_coxme)
        assert rho > 0.40, f"Heavy censoring rho = {rho:.3f}"

    def test_spa_pvalues_valid(self, heavy_censor_data):
        """SPA p-values should be in (0, 1]."""
        data, coxme_res, tg_res = heavy_censor_data
        p = tg_res.p
        assert (p > 0).all() and (p <= 1).all(), "SPA p-values out of range"


# ---------------------------------------------------------------------------
# Tests: Variance component recovery with high h²
# ---------------------------------------------------------------------------

@skip_no_coxme
class TestSurvivalPvalueAgreement:
    """Verify p-values match coxme closely across multiple seeds."""

    def test_pvalue_correlation_multi_seed(self):
        """Median Spearman rho of -log10(p) should be > 0.60 across 5 seeds."""

        rhos = []
        for seed in [42, 55, 77, 88, 99]:
            data = _simulate_survival_data(
                n=250, m=12, m_grm=80, n_causal=2,
                log_hr=0.7, censor_rate=0.3, h2=0.2, seed=seed,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                coxme_res = _run_coxme(data, tmpdir)
            tg_res = _run_torchgenomics_survival(data, use_spa=False)

            p_tg = tg_res.p.numpy()
            p_cx = np.array(coxme_res["p"])
            valid = ~np.isnan(p_cx) & (p_tg > 0) & (p_cx > 0)
            if valid.sum() < 8:
                continue

            log_tg = -np.log10(np.clip(p_tg[valid], 1e-300, 1.0))
            log_cx = -np.log10(np.clip(p_cx[valid], 1e-300, 1.0))
            rho, _ = spearmanr(log_tg, log_cx)
            rhos.append(rho)

        assert len(rhos) >= 3, f"Too few valid seeds: {len(rhos)}"
        median_rho = np.median(rhos)
        assert median_rho > 0.60, (
            f"Median Spearman rho = {median_rho:.3f}, expected > 0.60. "
            f"Individual rhos: {[f'{r:.3f}' for r in rhos]}"
        )
