// Native accelerator for ``torchgenomics.preprocess.qc._compute_hwe_pvalue``.
//
// The Python reference does, for each of m SNPs:
//
//   col = G[:, j]
//   valid = col[~isnan(col)]
//   p = af[j].item()
//   counts[v] = (round(valid) == v).sum()  for v in 0..ploidy
//   exp[v] = C(k, v) * p^v * q^(k-v) * n_valid
//   chi2 = sum((obs - exp)^2 / exp)
//   hwe_p[j] = chi2.sf(chi2, df)
//
// where df = 1 for diploid and (ploidy + 1) - 2 for polyploid (Levene 1949).
// Per-SNP scipy.stats.chi2.sf and ``.item()`` round-trips dominate the
// Python loop on m in the millions. The C++ port runs the entire scan in a
// single GIL-released pass and computes the chi-squared survival function
// from a hand-rolled regularized upper incomplete gamma (Numerical Recipes
// 6.2): no scipy dependency on the C++ side.
//
// Inputs:
//   G_int  (n, m) int64 ; rounded dosages, negative values mark missing.
//   af     (m,)   float64 ; allele frequency at each SNP (precomputed).
//   ploidy int    ; ploidy level (>= 2).
//
// Output:
//   hwe_p  (m,)   float64 ; per-SNP HWE p-values.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// ── Regularized upper incomplete gamma Q(s, x) ────────────────────────────
//
// Implementation following Numerical Recipes 3rd edition, §6.2.
// Switches between the series expansion (good for x < s+1) and the
// continued fraction (good for x >= s+1).

constexpr double GAMMA_EPS = 1e-15;
constexpr int GAMMA_MAX_ITER = 200;

double gamma_series(double s, double x) {
    // Returns Q(s, x) via the series form for the *lower* gamma, then
    // returns Q = 1 - P. Suitable for x < s + 1.
    if (x <= 0.0) return 1.0;
    double ap = s;
    double sum = 1.0 / s;
    double del = sum;
    for (int n = 1; n < GAMMA_MAX_ITER; ++n) {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (std::fabs(del) < std::fabs(sum) * GAMMA_EPS) break;
    }
    const double log_p = -x + s * std::log(x) - std::lgamma(s);
    return 1.0 - sum * std::exp(log_p);
}

double gamma_cfrac(double s, double x) {
    // Returns Q(s, x) via the Lentz continued fraction. Suitable for
    // x >= s + 1.
    constexpr double FPMIN = 1e-300;
    double b = x + 1.0 - s;
    double c = 1.0 / FPMIN;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < GAMMA_MAX_ITER; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - s);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < FPMIN) d = FPMIN;
        c = b + an / c;
        if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < GAMMA_EPS) break;
    }
    const double log_p = -x + s * std::log(x) - std::lgamma(s);
    return h * std::exp(log_p);
}

double gamma_q(double s, double x) {
    if (x < 0.0 || s <= 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    if (x < s + 1.0) return gamma_series(s, x);
    return gamma_cfrac(s, x);
}

double chi2_sf(double chi2, int df) {
    if (chi2 <= 0.0) return 1.0;
    if (df <= 0) return 1.0;
    // Special-case df=1 for accuracy: P(X > chi2) = erfc(sqrt(chi2/2)).
    if (df == 1) return std::erfc(std::sqrt(0.5 * chi2));
    return gamma_q(0.5 * static_cast<double>(df), 0.5 * chi2);
}

double binomial_coeff(int k, int v) {
    // Small (ploidy is at most ~8 in practice).
    if (v < 0 || v > k) return 0.0;
    if (v == 0 || v == k) return 1.0;
    double r = 1.0;
    const int vv = std::min(v, k - v);
    for (int i = 1; i <= vv; ++i) {
        r *= static_cast<double>(k - i + 1) / static_cast<double>(i);
    }
    return r;
}

py::array_t<double> hwe_pvalues(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> G_int,
    py::array_t<double, py::array::c_style | py::array::forcecast> af,
    int ploidy
) {
    auto g_buf = G_int.request();
    auto a_buf = af.request();
    if (g_buf.ndim != 2) throw std::invalid_argument("G_int must be 2-D");
    if (a_buf.ndim != 1) throw std::invalid_argument("af must be 1-D");
    if (ploidy < 2) throw std::invalid_argument("ploidy must be >= 2");
    const std::size_t n = static_cast<std::size_t>(g_buf.shape[0]);
    const std::size_t m = static_cast<std::size_t>(g_buf.shape[1]);
    if (static_cast<std::size_t>(a_buf.shape[0]) != m) {
        throw std::invalid_argument("af length must match G_int columns");
    }

    py::array_t<double> out(static_cast<py::ssize_t>(m));
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    const std::int64_t* g_ptr = static_cast<const std::int64_t*>(g_buf.ptr);
    const double* af_ptr = static_cast<const double*>(a_buf.ptr);

    const int n_classes = ploidy + 1;
    const int df = (ploidy == 2) ? 1 : (n_classes - 2 < 1 ? 1 : n_classes - 2);

    {
        py::gil_scoped_release release;
        std::vector<double> coef(static_cast<std::size_t>(n_classes));
        for (int v = 0; v < n_classes; ++v) {
            coef[static_cast<std::size_t>(v)] = binomial_coeff(ploidy, v);
        }
#pragma omp parallel
        {
            std::vector<double> obs(static_cast<std::size_t>(n_classes));
            std::vector<double> exp(static_cast<std::size_t>(n_classes));
#pragma omp for schedule(static)
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(m); ++jj) {
            const std::size_t j = static_cast<std::size_t>(jj);
            // Count valid observations & class counts in one column pass.
            std::fill(obs.begin(), obs.end(), 0.0);
            std::int64_t n_valid_i = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::int64_t v = g_ptr[i * m + j];
                if (v < 0) continue;
                std::int64_t vv = v;
                if (vv < 0) vv = 0;
                if (vv > ploidy) vv = ploidy;
                obs[static_cast<std::size_t>(vv)] += 1.0;
                ++n_valid_i;
            }
            if (n_valid_i < 10) {
                out_ptr[j] = 1.0;
                continue;
            }
            const double n_valid = static_cast<double>(n_valid_i);
            const double p = af_ptr[j];
            const double q = 1.0 - p;

            // Expected counts under HWE.
            for (int v = 0; v < n_classes; ++v) {
                const double pv = (p == 0.0 && v == 0) ? 1.0 : std::pow(p, v);
                const double qv = (q == 0.0 && (ploidy - v) == 0)
                                      ? 1.0 : std::pow(q, ploidy - v);
                double e = coef[static_cast<std::size_t>(v)] * pv * qv * n_valid;
                if (e < 0.5) e = 0.5;
                exp[static_cast<std::size_t>(v)] = e;
            }

            double chi2 = 0.0;
            for (int v = 0; v < n_classes; ++v) {
                const double d = obs[static_cast<std::size_t>(v)] -
                                 exp[static_cast<std::size_t>(v)];
                chi2 += (d * d) / exp[static_cast<std::size_t>(v)];
            }
            out_ptr[j] = chi2_sf(chi2, df);
        }  // omp for
        }  // omp parallel
    }

    return out;
}

// ── Tetraploid HWE with double reduction (Haldane 1930) ──────────────────
//
// Mirrors ``_compute_hwe_double_reduction`` in
// ``torchgenomics.preprocess.qc``. For each SNP at allele frequency p:
//
//   1. Count observed dosage classes 0..4.
//   2. Method-of-moments estimate of the DR parameter alpha from
//      excess homozygosity at dosage 0 and dosage 4. Average and clamp
//      to [0, 1/6].
//   3. Build gamete distribution under DR:
//        gam[0] = (1-alpha)*q^2 + alpha*q
//        gam[1] = (1-alpha)*2pq
//        gam[2] = (1-alpha)*p^2 + alpha*p
//   4. Convolve two gametes to get genotype expectation; chi^2 vs obs;
//      survival probability with df = max(ploidy-2, 1) = 2.
//
// Returns a (alpha, hwe_p) tuple of float64 (m,) arrays.

py::tuple hwe_pvalues_double_reduction(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> G_int,
    py::array_t<double, py::array::c_style | py::array::forcecast> af
) {
    auto g_buf = G_int.request();
    auto a_buf = af.request();
    if (g_buf.ndim != 2) throw std::invalid_argument("G_int must be 2-D");
    if (a_buf.ndim != 1) throw std::invalid_argument("af must be 1-D");
    const std::size_t n = static_cast<std::size_t>(g_buf.shape[0]);
    const std::size_t m = static_cast<std::size_t>(g_buf.shape[1]);
    if (static_cast<std::size_t>(a_buf.shape[0]) != m) {
        throw std::invalid_argument("af length must match G_int columns");
    }

    constexpr int ploidy = 4;  // DR model is tetraploid-only
    constexpr int n_classes = ploidy + 1;
    constexpr int df = 2;      // (k+1) - 1 (p) - 1 (alpha) - 1 = k - 2

    py::array_t<double> alpha_out(static_cast<py::ssize_t>(m));
    py::array_t<double> hwe_out(static_cast<py::ssize_t>(m));
    double* alpha_ptr = static_cast<double*>(alpha_out.request().ptr);
    double* hwe_ptr = static_cast<double*>(hwe_out.request().ptr);
    const std::int64_t* g_ptr = static_cast<const std::int64_t*>(g_buf.ptr);
    const double* af_ptr = static_cast<const double*>(a_buf.ptr);

    // Initialise outputs to (0, 1) so that "skip" branches in the Python
    // reference produce identical results without explicit fall-through.
    for (std::size_t j = 0; j < m; ++j) {
        alpha_ptr[j] = 0.0;
        hwe_ptr[j] = 1.0;
    }

    {
        py::gil_scoped_release release;
#pragma omp parallel
        {
            std::vector<double> obs(static_cast<std::size_t>(n_classes));
#pragma omp for schedule(static)
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(m); ++jj) {
            const std::size_t j = static_cast<std::size_t>(jj);
            std::fill(obs.begin(), obs.end(), 0.0);
            std::int64_t n_valid_i = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::int64_t v = g_ptr[i * m + j];
                if (v < 0) continue;
                std::int64_t vv = v;
                if (vv < 0) vv = 0;
                if (vv > ploidy) vv = ploidy;
                obs[static_cast<std::size_t>(vv)] += 1.0;
                ++n_valid_i;
            }
            if (n_valid_i < 10) continue;
            const double n_valid = static_cast<double>(n_valid_i);
            const double p = af_ptr[j];
            const double q = 1.0 - p;
            if (p < 1e-10 || q < 1e-10) continue;

            const double obs_hom0 = obs[0] / n_valid;
            const double obs_hom4 = obs[4] / n_valid;
            const double exp_hom0_no_dr = q * q * q * q;
            const double exp_hom4_no_dr = p * p * p * p;

            double alpha0 = 0.0;
            if (q > 1e-10) {
                const double denom0 = q * q - q * q * q * q;
                if (std::fabs(denom0) > 1e-10) {
                    alpha0 = (obs_hom0 - exp_hom0_no_dr) / denom0;
                }
            }
            double alpha4 = 0.0;
            if (p > 1e-10) {
                const double denom4 = p * p - p * p * p * p;
                if (std::fabs(denom4) > 1e-10) {
                    alpha4 = (obs_hom4 - exp_hom4_no_dr) / denom4;
                }
            }

            double alpha = 0.5 * (alpha0 + alpha4);
            if (alpha < 0.0) alpha = 0.0;
            if (alpha > 1.0 / 6.0) alpha = 1.0 / 6.0;
            alpha_ptr[j] = alpha;

            // Gamete probabilities under DR.
            double gam[3];
            gam[0] = (1.0 - alpha) * q * q + alpha * q;
            gam[1] = (1.0 - alpha) * 2.0 * p * q;
            gam[2] = (1.0 - alpha) * p * p + alpha * p;

            // Convolve to genotype expected counts.
            double dr_exp[n_classes] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for (int d1 = 0; d1 < 3; ++d1) {
                for (int d2 = 0; d2 < 3; ++d2) {
                    const int d = d1 + d2;
                    if (d <= ploidy) dr_exp[d] += gam[d1] * gam[d2];
                }
            }

            double chi2 = 0.0;
            for (int v = 0; v < n_classes; ++v) {
                double e = dr_exp[v] * n_valid;
                if (e < 0.5) e = 0.5;
                const double d = obs[static_cast<std::size_t>(v)] - e;
                chi2 += (d * d) / e;
            }
            hwe_ptr[j] = chi2_sf(chi2, df);
        }  // omp for
        }  // omp parallel
    }

    return py::make_tuple(alpha_out, hwe_out);
}

}  // namespace

PYBIND11_MODULE(_hwe_native, mod) {
    mod.doc() = "Native HWE chi-squared goodness-of-fit p-values.";
    mod.def("hwe_pvalues", &hwe_pvalues,
            py::arg("G_int"), py::arg("af"), py::arg("ploidy"),
            "Per-SNP HWE chi-squared p-values for diploid and polyploid.");
    mod.def("hwe_pvalues_double_reduction", &hwe_pvalues_double_reduction,
            py::arg("G_int"), py::arg("af"),
            "Per-SNP tetraploid HWE-with-double-reduction (alpha, p) tuple.");
}
