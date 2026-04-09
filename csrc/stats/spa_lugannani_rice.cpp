// Saddlepoint-approximation (SPA) Lugannani-Rice tail p-value for one SNP.
//
// C++ port of the per-extreme-SNP body of
// torchgwas.stats.spa.saddlepoint_pvalue. Given the null fitted probabilities
// `mu` (length n), the genotype dosages `g` (length n) for one SNP, and the
// observed score statistic U = g'(Y - mu), this routine
//
//   1. solves the saddlepoint t_hat such that K'(t_hat) = U via Newton's
//      method on the binomial CGF
//        K(t) = sum_i log(q_i exp(-mu_i g_i t) + mu_i exp(q_i g_i t))
//   2. evaluates K(t_hat) and K''(t_hat)
//   3. computes the Lugannani-Rice tail probability for both the +U and -U
//      tails and returns the two-sided sum.
//
// The pure-Python reference loops over `mu` four-plus times per Newton step
// and round-trips through `.item()` on every cgf-derivative call. Pushing the
// whole inner kernel down to one C++ entry per extreme SNP eliminates ~200
// FFI hops per call while preserving statistical-equivalence (the formulas
// match the Python source line-for-line, modulo float-summation order).
//
// A return value of NaN signals "fallback to chi2" — the caller keeps
// whichever p-value it had.
//
// Exposed via pybind11:
//
//   spa_lugannani_rice(mu, g, observed_score) -> float

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

constexpr double DENOM_FLOOR = 1e-30;

// K(t), K'(t), K''(t) for the binomial CGF, computed in one pass.
void cgf_all(
    const double* mu, const double* g,
    std::size_t n, double t,
    double& K, double& K1, double& K2
) {
    double K_acc = 0.0;
    double K1_acc = 0.0;
    double K2_acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double mu_i = mu[i];
        const double q_i = 1.0 - mu_i;
        const double gt = g[i] * t;
        const double exp_pos = std::exp(q_i * gt);
        const double exp_neg = std::exp(-mu_i * gt);
        double denom = q_i * exp_neg + mu_i * exp_pos;
        if (denom < DENOM_FLOOR) denom = DENOM_FLOOR;

        K_acc += std::log(denom);

        // K'(t) = sum_i mu_i q_i g_i (exp(q g t) - exp(-mu g t)) / denom
        const double mu_q_g = mu_i * q_i * g[i];
        const double diff = exp_pos - exp_neg;
        K1_acc += mu_q_g * diff / denom;

        // K''(t) = sum_i mu_i q_i g_i^2 (q_i exp(q g t) + mu_i exp(-mu g t)) / denom
        //        - (mu_i q_i g_i)^2 (exp(q g t) - exp(-mu g t))^2 / denom^2
        const double term1 = mu_i * q_i * g[i] * g[i] *
                             (q_i * exp_pos + mu_i * exp_neg) / denom;
        const double term2 = (mu_q_g * diff) * (mu_q_g * diff) / (denom * denom);
        K2_acc += term1 - term2;
    }
    K = K_acc;
    K1 = K1_acc;
    K2 = K2_acc;
}

// Newton solve for K'(t) = observed. Returns t_hat. Mirrors the Python
// reference's 50-step / 1e-8 tolerance.
double solve_saddlepoint(
    const double* mu, const double* g, std::size_t n,
    double observed, int max_iter, double tol
) {
    double t = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        double K, K1, K2;
        cgf_all(mu, g, n, t, K, K1, K2);
        if (std::fabs(K2) < DENOM_FLOOR) break;
        const double t_new = t + (observed - K1) / K2;
        if (std::fabs(t_new - t) < tol) {
            t = t_new;
            break;
        }
        t = t_new;
    }
    return t;
}

// Standard normal pdf / sf. We avoid pulling scipy back across the FFI.
inline double norm_pdf(double x) {
    static const double INV_SQRT_2PI = 0.39894228040143267794;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

inline double norm_sf(double x) {
    // P(Z > x) = 0.5 * erfc(x / sqrt(2))
    return 0.5 * std::erfc(x / std::sqrt(2.0));
}

// Lugannani-Rice one-tail probability for a given (t_hat, observed).
// Returns NaN on numeric breakdown.
double lugannani_rice_one_tail(
    const double* mu, const double* g, std::size_t n,
    double observed
) {
    const double t_hat = solve_saddlepoint(mu, g, n, observed, 50, 1e-8);
    double K, K1, K2;
    cgf_all(mu, g, n, t_hat, K, K1, K2);
    if (K2 <= 0.0 || !std::isfinite(K)) return std::nan("");

    const double inside = 2.0 * (t_hat * observed - K);
    if (inside <= 0.0) return std::nan("");

    const double w = std::copysign(1.0, t_hat) * std::sqrt(inside);
    const double u = t_hat * std::sqrt(K2);
    if (std::fabs(w) < 1e-10 || std::fabs(u) < 1e-10) return std::nan("");

    const double aw = std::fabs(w);
    const double au = std::fabs(u);
    return norm_sf(aw) + norm_pdf(aw) * (1.0 / aw - 1.0 / au);
}

double py_spa_lugannani_rice(
    py::array_t<double, py::array::c_style | py::array::forcecast> mu,
    py::array_t<double, py::array::c_style | py::array::forcecast> g,
    double observed
) {
    auto mu_buf = mu.request();
    auto g_buf = g.request();
    if (mu_buf.ndim != 1 || g_buf.ndim != 1) {
        throw std::runtime_error("mu and g must be 1-D arrays");
    }
    if (mu_buf.shape[0] != g_buf.shape[0]) {
        throw std::runtime_error("mu and g must have the same length");
    }
    const std::size_t n = static_cast<std::size_t>(mu_buf.shape[0]);
    if (n == 0) {
        throw std::runtime_error("mu and g must be non-empty");
    }

    const double* mu_p = static_cast<const double*>(mu_buf.ptr);
    const double* g_p = static_cast<const double*>(g_buf.ptr);

    double p_pos, p_neg;
    {
        py::gil_scoped_release release;
        p_pos = lugannani_rice_one_tail(mu_p, g_p, n, observed);
        p_neg = lugannani_rice_one_tail(mu_p, g_p, n, -observed);
    }

    if (std::isnan(p_pos)) return std::nan("");
    const double other = std::isnan(p_neg) ? p_pos : p_neg;

    double p_spa = std::fabs(p_pos) + std::fabs(other);
    if (p_spa > 1.0) p_spa = 1.0;
    if (p_spa < 1e-300) p_spa = 1e-300;
    return p_spa;
}

// Batched SPA over k extreme SNPs. ``G_ext`` is (n, k) column-major float64;
// ``observed`` is (k,). Returns a (k,) float64 array with NaN sentinels for
// SNPs that should fall back to the chi-square p-value.
py::array_t<double> py_spa_lugannani_rice_batch(
    py::array_t<double, py::array::c_style | py::array::forcecast> mu,
    py::array_t<double, py::array::f_style | py::array::forcecast> G_ext,
    py::array_t<double, py::array::c_style | py::array::forcecast> observed
) {
    auto mu_buf = mu.request();
    auto g_buf = G_ext.request();
    auto o_buf = observed.request();
    if (mu_buf.ndim != 1) throw std::runtime_error("mu must be 1-D");
    if (g_buf.ndim != 2) throw std::runtime_error("G_ext must be 2-D");
    if (o_buf.ndim != 1) throw std::runtime_error("observed must be 1-D");
    const std::size_t n = static_cast<std::size_t>(mu_buf.shape[0]);
    const std::size_t k = static_cast<std::size_t>(g_buf.shape[1]);
    if (static_cast<std::size_t>(g_buf.shape[0]) != n) {
        throw std::runtime_error("G_ext rows must equal mu length");
    }
    if (static_cast<std::size_t>(o_buf.shape[0]) != k) {
        throw std::runtime_error("observed length must equal G_ext columns");
    }

    py::array_t<double> out(static_cast<py::ssize_t>(k));
    double* out_ptr = static_cast<double*>(out.request().ptr);
    const double* mu_p = static_cast<const double*>(mu_buf.ptr);
    const double* g_all = static_cast<const double*>(g_buf.ptr);
    const double* obs_p = static_cast<const double*>(o_buf.ptr);

    {
        py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(k); ++jj) {
            const double* g_col = g_all + static_cast<std::size_t>(jj) * n;
            const double observed_j = obs_p[jj];
            const double p_pos = lugannani_rice_one_tail(mu_p, g_col, n, observed_j);
            const double p_neg = lugannani_rice_one_tail(mu_p, g_col, n, -observed_j);

            if (std::isnan(p_pos)) {
                out_ptr[jj] = std::nan("");
                continue;
            }
            const double other = std::isnan(p_neg) ? p_pos : p_neg;
            double p_spa = std::fabs(p_pos) + std::fabs(other);
            if (p_spa > 1.0) p_spa = 1.0;
            if (p_spa < 1e-300) p_spa = 1e-300;
            out_ptr[jj] = p_spa;
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_spa_native, m) {
    m.doc() = "Native C++ accelerator for the SPA Lugannani-Rice loop.";
    m.def("spa_lugannani_rice", &py_spa_lugannani_rice,
          py::arg("mu"), py::arg("g"), py::arg("observed"),
          R"pbdoc(
            Two-sided SPA p-value for one SNP score statistic.

            ``mu`` is the (n,) null fitted probability vector. ``g`` is the
            (n,) genotype dosage column for the SNP. ``observed`` is the raw
            score statistic ``U = g'(Y - mu)``. Returns the two-sided
            Lugannani-Rice tail probability, or NaN if the caller should fall
            back to the chi-square p-value.
          )pbdoc");
    m.def("spa_lugannani_rice_batch", &py_spa_lugannani_rice_batch,
          py::arg("mu"), py::arg("G_ext"), py::arg("observed"),
          R"pbdoc(
            Batched two-sided SPA over k extreme SNPs, OpenMP-parallel.

            ``G_ext`` is (n, k) Fortran-order float64 (column-major) so that
            each thread can read a contiguous SNP column. Returns a (k,)
            float64 array of p-values with NaN sentinels for caller fallback.
          )pbdoc");
}
