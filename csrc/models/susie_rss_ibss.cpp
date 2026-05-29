// Native accelerator for the SuSiE-RSS IBSS inner sweep in
// ``torchgwas.models.bayesian_vs_rss.BayesianVSRss.fit_rss``.
//
// The Python reference does, per outer iteration:
//
//   for l in range(L):
//       tilde_z_l = z - R @ sum_{l' != l} b_eff[l']      # residual update
//       if estimate_V and method == "optim":
//           V[l] = find_optimal_V(tilde_z_l, R, n, V_init=max(V[l], sigma_prior_sq))
//       if V[l] == 0.0:
//           alpha[l] = 1/p; mu[l] = 0; sigma_sq[l] = 0; b_eff[l] = 0; continue
//       sigma_sq[l] = 1 / (R_diag / V[l] + n)            # SER posterior
//       mu[l] = sigma_sq[l] * tilde_z_l * sqrt(n)
//       log_bf[l] = 0.5*log(sigma_sq[l]/V[l]) + 0.5*mu[l]^2/sigma_sq[l]
//       alpha[l] = softmax(log_bf[l] + log_prior_normalized)
//       b_eff[l] = alpha[l] * sqrt(n) * mu[l]
//       if estimate_V and method == "EM":
//           V_new = sum_j alpha[l,j] * (mu[l,j]^2 + sigma_sq[l,j])
//           V[l] = 0 if V_new < tol else V_new
//
// Per IBSS iteration, the per-layer Brent V-maximisation triggers ~20-30
// scipy round-trips, each one re-allocating a (p,) tensor inside
// ``ser_posterior``. For L=10 layers and max_iter=100 that is ~25,000
// Python turns + tensor allocations per locus — the principal cost of
// SuSiE-RSS at biobank scale.
//
// This C++ port runs one full IBSS inner sweep (all L sequential layer
// updates) under a single GIL release, including:
//   - residual update R @ b_eff_others
//   - SER posterior (sigma_sq, mu, log_bf)
//   - 1-D bounded Brent maximisation of the marginal likelihood for V_l
//     (port of scipy.optimize._minimize_scalar_bounded; xatol = 1e-4)
//   - prior-weighted softmax for alpha_l
//   - per-layer b_eff update
//   - optional EM M-step for V_l with snap-to-zero
//
// The Python outer loop still drives the IBSS iteration count, ELBO
// computation, and convergence check (PIP-stability or ELBO-delta), so
// the algorithmic spec is unchanged.
//
// Inputs (all float64, all C-contiguous):
//   z         (p,)    — z-scores on the assumed-Var(y)=1 scale
//   R         (p, p)  — LD correlation matrix (row-major)
//   R_diag    (p,)    — diagonal of R (precomputed)
//   alpha     (L, p)  — inclusion probabilities; updated in place
//   mu        (L, p)  — posterior effect means (beta-scale); updated in place
//   sigma_sq  (L, p)  — posterior effect variances (beta-scale); updated in place
//   b_eff     (L, p)  — alpha[l] * sqrt(n) * mu[l] (z-scale); updated in place
//   V         (L,)    — per-layer prior variance; updated in place
//   log_prior (p,)    — log of normalized prior pi_j (uniform = -log(p))
//   n         scalar  — GWAS sample size
//   estimate_V          bool   — apply V update each layer
//   V_method            int    — 0 = optim (Brent before SER), 1 = EM (after SER)
//   prior_variance_tol  scalar — snap-to-zero threshold for EM
//   sigma_prior_sq      scalar — initial V; sets the Brent search bracket
//   V_xatol             scalar — Brent absolute tolerance (1e-4 matches susieR)
//
// Outputs: alpha, mu, sigma_sq, b_eff, V are updated in place. No return.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

// Numerically stable logsumexp over (log_bf + log_prior) of size p.
inline double logsumexp_with_prior(
    const double* log_bf,
    const double* log_prior,
    std::size_t p
) {
    double max_v = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < p; ++j) {
        const double v = log_bf[j] + log_prior[j];
        if (v > max_v) max_v = v;
    }
    if (!std::isfinite(max_v)) return max_v;
    double sum = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        sum += std::exp(log_bf[j] + log_prior[j] - max_v);
    }
    return max_v + std::log(sum);
}

// Single SER posterior evaluation: fills log_bf_scratch with per-variant
// log Bayes factors at prior variance V. Mirrors ``ser_posterior`` and
// the inline expansion in find_optimal_V's neg_log_marginal closure.
inline void ser_log_bf(
    const double* z,
    const double* R_diag,
    std::size_t p,
    double n,
    double V,
    double sqrt_n,
    double* log_bf_scratch
) {
    const double inv_V = 1.0 / V;
    const double log_V = std::log(V);
    for (std::size_t j = 0; j < p; ++j) {
        const double s2 = 1.0 / (R_diag[j] * inv_V + n);
        const double mu_j = s2 * z[j] * sqrt_n;
        log_bf_scratch[j] = 0.5 * (std::log(s2) - log_V) + 0.5 * mu_j * mu_j / s2;
    }
}

// 1-D bounded Brent minimisation in [a, b]. Port of SciPy's
// ``_minimize_scalar_bounded`` (golden-section + parabolic interpolation,
// xatol bracket convergence). Used to maximise the marginal log-
// likelihood L(V) by minimising the negative.
//
// xatol matches susieR's default ``optim(method="Brent")`` tolerance of
// ~1.2e-4 (= .Machine$double.eps^0.25); the SuSiE-RSS plan documents
// 1e-4 as the operational floor.
template <typename F>
double brent_bounded(F&& f, double a, double b, double xatol, int maxiter) {
    const double golden = (3.0 - std::sqrt(5.0)) / 2.0;
    double fulc = a + golden * (b - a);
    double nfc = fulc;
    double xf = fulc;
    double rat = 0.0;
    double e = 0.0;
    double x = xf;
    double fx = f(x);
    double ffulc = fx;
    double fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = std::sqrt(std::numeric_limits<double>::epsilon()) * std::abs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;
    int iter = 0;
    while (std::abs(xf - xm) > (tol2 - 0.5 * (b - a))) {
        bool golden_step = true;
        if (std::abs(e) > tol1) {
            // Try a parabolic step.
            const double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p_num = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p_num = -p_num;
            q = std::abs(q);
            const double r_e = e;
            e = rat;
            if (std::abs(p_num) < std::abs(0.5 * q * r_e)
                && p_num > q * (a - xf)
                && p_num < q * (b - xf)) {
                rat = p_num / q;
                const double u_test = xf + rat;
                if ((u_test - a) < tol2 || (b - u_test) < tol2) {
                    const double si = ((xm - xf) >= 0.0) ? 1.0 : -1.0;
                    rat = tol1 * si;
                }
                golden_step = false;
            }
        }
        if (golden_step) {
            e = (xf >= xm) ? (a - xf) : (b - xf);
            rat = golden * e;
        }
        const double si = (rat >= 0.0) ? 1.0 : -1.0;
        const double x_new = xf + ((std::abs(rat) >= tol1) ? rat : tol1 * si);
        const double fu = f(x_new);
        if (fu <= fx) {
            if (x_new >= xf) a = xf;
            else              b = xf;
            fulc = nfc;   ffulc = fnfc;
            nfc = xf;     fnfc = fx;
            xf = x_new;   fx = fu;
        } else {
            if (x_new < xf) a = x_new;
            else            b = x_new;
            if (fu <= fnfc || nfc == xf) {
                fulc = nfc;   ffulc = fnfc;
                nfc = x_new;  fnfc = fu;
            } else if (fu <= ffulc || fulc == xf || fulc == nfc) {
                fulc = x_new; ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = std::sqrt(std::numeric_limits<double>::epsilon()) * std::abs(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if (++iter >= maxiter) break;
    }
    return xf;
}

// Find V_opt = argmax L(V) via 1-D bounded Brent on log V. The Python
// reference uses scipy.optimize.minimize_scalar(method="bounded") with
// the same bracket: [log(V_init) - 10, log(V_init) + log(100)].
// Snaps to V = 0 when L(V_opt) <= 0 (null layer favoured).
double find_optimal_V_cpp(
    const double* z,
    const double* R_diag,
    std::size_t p,
    double n,
    double V_init,
    double sqrt_n,
    const double* log_prior,
    double V_xatol,
    double* log_bf_scratch
) {
    const double V_init_eff = std::max(V_init, 1e-12);
    const double log_V_init = std::log(V_init_eff);
    const double log_V_lo = log_V_init - 10.0;
    const double log_V_hi = log_V_init + std::log(100.0);  // upper_bound_factor=100

    auto neg_log_marg = [&](double log_V) -> double {
        const double V = std::exp(log_V);
        if (V <= 0.0) return 0.0;  // L(0) = 0 by SuSiE-RSS convention
        ser_log_bf(z, R_diag, p, n, V, sqrt_n, log_bf_scratch);
        return -logsumexp_with_prior(log_bf_scratch, log_prior, p);
    };

    const double log_V_opt = brent_bounded(neg_log_marg, log_V_lo, log_V_hi, V_xatol, 500);
    const double L_opt = -neg_log_marg(log_V_opt);
    const double V_opt = std::exp(log_V_opt);
    return (L_opt > 0.0) ? V_opt : 0.0;
}

void ibss_inner_sweep(
    py::array_t<double, py::array::c_style | py::array::forcecast> z_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_diag_arr,
    py::array_t<double, py::array::c_style> alpha_arr,
    py::array_t<double, py::array::c_style> mu_arr,
    py::array_t<double, py::array::c_style> sigma_sq_arr,
    py::array_t<double, py::array::c_style> b_eff_arr,
    py::array_t<double, py::array::c_style> V_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> log_prior_arr,
    int n,
    bool estimate_V,
    int V_method,
    double prior_variance_tol,
    double sigma_prior_sq,
    double V_xatol
) {
    if (z_arr.ndim() != 1) throw std::invalid_argument("z must be 1-D");
    if (R_arr.ndim() != 2) throw std::invalid_argument("R must be 2-D");
    if (R_diag_arr.ndim() != 1) throw std::invalid_argument("R_diag must be 1-D");
    if (alpha_arr.ndim() != 2) throw std::invalid_argument("alpha must be 2-D");
    if (mu_arr.ndim() != 2) throw std::invalid_argument("mu must be 2-D");
    if (sigma_sq_arr.ndim() != 2) throw std::invalid_argument("sigma_sq must be 2-D");
    if (b_eff_arr.ndim() != 2) throw std::invalid_argument("b_eff must be 2-D");
    if (V_arr.ndim() != 1) throw std::invalid_argument("V must be 1-D");
    if (log_prior_arr.ndim() != 1) throw std::invalid_argument("log_prior must be 1-D");

    const std::size_t p = static_cast<std::size_t>(z_arr.shape(0));
    const std::size_t L = static_cast<std::size_t>(V_arr.shape(0));

    if (static_cast<std::size_t>(R_arr.shape(0)) != p
        || static_cast<std::size_t>(R_arr.shape(1)) != p) {
        throw std::invalid_argument("R must be (p, p)");
    }
    if (static_cast<std::size_t>(R_diag_arr.shape(0)) != p) {
        throw std::invalid_argument("R_diag must be (p,)");
    }
    if (static_cast<std::size_t>(log_prior_arr.shape(0)) != p) {
        throw std::invalid_argument("log_prior must be (p,)");
    }
    auto check_lp = [&](const py::array_t<double, py::array::c_style>& a, const char* name) {
        if (static_cast<std::size_t>(a.shape(0)) != L
            || static_cast<std::size_t>(a.shape(1)) != p) {
            throw std::invalid_argument(std::string(name) + " must be (L, p)");
        }
    };
    check_lp(alpha_arr, "alpha");
    check_lp(mu_arr, "mu");
    check_lp(sigma_sq_arr, "sigma_sq");
    check_lp(b_eff_arr, "b_eff");

    if (n <= 0) throw std::invalid_argument("n must be > 0");
    if (sigma_prior_sq <= 0.0) throw std::invalid_argument("sigma_prior_sq must be > 0");
    if (V_method != 0 && V_method != 1) throw std::invalid_argument("V_method must be 0 (optim) or 1 (EM)");

    const double* z = z_arr.data();
    const double* R = R_arr.data();
    const double* R_diag = R_diag_arr.data();
    const double* log_prior = log_prior_arr.data();
    double* alpha = alpha_arr.mutable_data();
    double* mu = mu_arr.mutable_data();
    double* sigma_sq = sigma_sq_arr.mutable_data();
    double* b_eff = b_eff_arr.mutable_data();
    double* V = V_arr.mutable_data();

    const double n_d = static_cast<double>(n);
    const double sqrt_n = std::sqrt(n_d);
    const double uniform = 1.0 / static_cast<double>(p);

    {
        py::gil_scoped_release release;

        std::vector<double> tilde_z(p);
        std::vector<double> log_bf_scratch(p);
        std::vector<double> b_eff_others(p);

        for (std::size_t l = 0; l < L; ++l) {
            // Sum b_eff over all layers != l.
            std::fill(b_eff_others.begin(), b_eff_others.end(), 0.0);
            for (std::size_t ll = 0; ll < L; ++ll) {
                if (ll == l) continue;
                const double* row = b_eff + ll * p;
                for (std::size_t j = 0; j < p; ++j) {
                    b_eff_others[j] += row[j];
                }
            }
            // tilde_z = z - R @ b_eff_others.
            // R is row-major; (R @ v)[i] = sum_j R[i, j] * v[j].
            // We tried OpenMP on the outer-i loop, but for L=10 the
            // thread spawn/teardown overhead per layer iteration
            // (60 matvecs per fit_rss at typical sizes) exceeds the
            // parallel speedup. The serial path is sequential-
            // dependency-free across i, and the compiler auto-
            // vectorises the inner dot product on AVX2 targets.
            for (std::size_t i = 0; i < p; ++i) {
                double s = 0.0;
                const double* row_i = R + i * p;
                for (std::size_t j = 0; j < p; ++j) {
                    s += row_i[j] * b_eff_others[j];
                }
                tilde_z[i] = z[i] - s;
            }

            // V update — "optim" path runs BEFORE the SER and uses the
            // marginal-likelihood maximiser; "EM" path runs AFTER the SER
            // (below) and uses the fresh per-layer posterior.
            if (estimate_V && V_method == 0) {
                const double V_init = std::max(V[l], sigma_prior_sq);
                V[l] = find_optimal_V_cpp(
                    tilde_z.data(), R_diag, p, n_d, V_init, sqrt_n,
                    log_prior, V_xatol, log_bf_scratch.data()
                );
            }

            // Null layer (V_l == 0): SER collapses to a delta at zero.
            if (estimate_V && V[l] == 0.0) {
                double* alpha_l = alpha + l * p;
                double* mu_l = mu + l * p;
                double* sigma_sq_l = sigma_sq + l * p;
                double* b_eff_l = b_eff + l * p;
                for (std::size_t j = 0; j < p; ++j) {
                    alpha_l[j] = uniform;
                    mu_l[j] = 0.0;
                    sigma_sq_l[j] = 0.0;
                    b_eff_l[j] = 0.0;
                }
                continue;
            }

            // SER posterior (beta-scale).
            const double V_l = V[l];
            const double inv_V_l = 1.0 / V_l;
            const double log_V_l = std::log(V_l);
            double* mu_l = mu + l * p;
            double* sigma_sq_l = sigma_sq + l * p;

            for (std::size_t j = 0; j < p; ++j) {
                const double s2 = 1.0 / (R_diag[j] * inv_V_l + n_d);
                const double mu_j = s2 * tilde_z[j] * sqrt_n;
                sigma_sq_l[j] = s2;
                mu_l[j] = mu_j;
                log_bf_scratch[j] = 0.5 * (std::log(s2) - log_V_l) + 0.5 * mu_j * mu_j / s2;
            }

            // Softmax: alpha[l] = softmax(log_bf + log_prior).
            double max_v = -std::numeric_limits<double>::infinity();
            for (std::size_t j = 0; j < p; ++j) {
                const double v = log_bf_scratch[j] + log_prior[j];
                if (v > max_v) max_v = v;
            }
            double* alpha_l = alpha + l * p;
            double sum = 0.0;
            for (std::size_t j = 0; j < p; ++j) {
                const double v = log_bf_scratch[j] + log_prior[j];
                const double e = std::exp(v - max_v);
                alpha_l[j] = e;
                sum += e;
            }
            const double inv_sum = 1.0 / sum;
            double* b_eff_l = b_eff + l * p;
            for (std::size_t j = 0; j < p; ++j) {
                alpha_l[j] *= inv_sum;
                b_eff_l[j] = alpha_l[j] * sqrt_n * mu_l[j];
            }

            // EM M-step for V_l (uses fresh alpha / mu / sigma_sq).
            if (estimate_V && V_method == 1) {
                double V_new = 0.0;
                for (std::size_t j = 0; j < p; ++j) {
                    V_new += alpha_l[j] * (mu_l[j] * mu_l[j] + sigma_sq_l[j]);
                }
                if (V_new < 0.0) V_new = 0.0;
                V[l] = (V_new < prior_variance_tol) ? 0.0 : V_new;
            }
        }
    }  // GIL re-acquired
}

}  // namespace

PYBIND11_MODULE(_ibss_native, m) {
    m.doc() = "Native C++ accelerator for the SuSiE-RSS IBSS inner sweep.";
    m.def("ibss_inner_sweep", &ibss_inner_sweep,
          py::arg("z"),
          py::arg("R"),
          py::arg("R_diag"),
          py::arg("alpha"),
          py::arg("mu"),
          py::arg("sigma_sq"),
          py::arg("b_eff"),
          py::arg("V"),
          py::arg("log_prior"),
          py::arg("n"),
          py::arg("estimate_V"),
          py::arg("V_method"),
          py::arg("prior_variance_tol"),
          py::arg("sigma_prior_sq"),
          py::arg("V_xatol"),
          "Run one full SuSiE-RSS IBSS sweep over all L layers in place "
          "(residual update + Brent V optimiser + SER posterior + softmax "
          "+ optional EM M-step).");
}
