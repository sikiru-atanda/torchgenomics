// PRS-CS block Gibbs sampler — C++ port of torchgenomics.pgs.prscs._prscs_gibbs_block.
//
// The pure-Python implementation in torchgenomics/pgs/prscs.py is the canonical
// reference. This module mirrors it line-for-line in C++ for performance, with
// the same prior, the same update order, and the same posterior accumulators.
//
// Exposes (via pybind11):
//
//   prscs_gibbs_block(beta_std, R_b, n_eff, phi, a, b, n_iter, n_burnin, seed)
//       -> (mean: ndarray[float64], sd: ndarray[float64])
//
//   sample_gig_vec(lam, chi, psi, seed) -> ndarray[float64]
//
// All numeric inputs are float64. Matrices must be C-contiguous (row-major).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "gig_sampler.hpp"
#include "linalg.hpp"

namespace py = pybind11;

namespace {

// Run one PRS-CS Gibbs block. All inputs/outputs use raw double buffers; the
// caller is responsible for shape and contiguity.
void run_prscs_gibbs_block(
    const double* beta_std,  // (b,)
    const double* R_b,       // (b * b) row-major
    std::size_t b_size,
    double n_eff,
    double phi,
    double a,
    double b_param,
    int n_iter,
    int n_burnin,
    std::uint64_t seed,
    double* out_mean,        // (b,)
    double* out_sd           // (b,)
) {
    if (b_size == 0) {
        return;
    }
    if (n_iter <= 0) {
        throw std::invalid_argument("n_iter must be positive");
    }
    if (n_burnin < 0 || n_burnin >= n_iter) {
        throw std::invalid_argument("n_burnin must be in [0, n_iter)");
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double sigma2 = 1.0;        // Matches the Python reference (held at 1.0)
    const double n_t = n_eff;
    const double n_over_s2 = n_t / sigma2;

    // State vectors
    std::vector<double> psi(b_size, 1.0);
    std::vector<double> delta(b_size, 1.0);
    std::vector<double> beta(b_size, 0.0);

    // Working buffers
    std::vector<double> prec(b_size * b_size, 0.0);  // packed row-major
    std::vector<double> rhs(b_size, 0.0);
    std::vector<double> mu(b_size, 0.0);
    std::vector<double> z(b_size, 0.0);
    std::vector<double> chi_buf(b_size, 0.0);
    std::vector<double> psi_param_buf(b_size, 0.0);
    std::vector<double> psi_new(b_size, 0.0);

    std::vector<double> beta_sum(b_size, 0.0);
    std::vector<double> beta_sq_sum(b_size, 0.0);
    int n_post = 0;

    // Tiny diagonal ridge added to R_b for PD safety (matches Python: 1e-6).
    constexpr double R_RIDGE = 1e-6;
    constexpr double FALLBACK_RIDGE = 1e-4;

    for (int it = 0; it < n_iter; ++it) {
        // ---- Build precision matrix: (n/sigma2) * (R + R_RIDGE*I) + diag(1/(sigma2*psi)) ----
        for (std::size_t i = 0; i < b_size; ++i) {
            for (std::size_t j = 0; j < b_size; ++j) {
                double rij = R_b[i * b_size + j];
                if (i == j) rij += R_RIDGE;
                prec[i * b_size + j] = n_over_s2 * rij;
            }
            double inv_psi_i = 1.0 / std::max(psi[i], 1e-12);
            prec[i * b_size + i] += inv_psi_i / sigma2;
        }

        // ---- Cholesky (with fallback ridge if non-PD) ----
        bool ok = tgw::cholesky_lower_inplace(prec.data(), b_size);
        if (!ok) {
            // Rebuild prec with extra ridge and retry (mirrors Python except path)
            for (std::size_t i = 0; i < b_size; ++i) {
                for (std::size_t j = 0; j < b_size; ++j) {
                    double rij = R_b[i * b_size + j];
                    if (i == j) rij += R_RIDGE + FALLBACK_RIDGE;
                    prec[i * b_size + j] = n_over_s2 * rij;
                }
                double inv_psi_i = 1.0 / std::max(psi[i], 1e-12);
                prec[i * b_size + i] += inv_psi_i / sigma2;
            }
            ok = tgw::cholesky_lower_inplace(prec.data(), b_size);
            if (!ok) {
                throw std::runtime_error("prscs_gibbs_block: Cholesky failed even after fallback ridge");
            }
        }

        // ---- mu = prec^{-1} * (n/sigma2 * beta_std)  via Cholesky solve ----
        for (std::size_t i = 0; i < b_size; ++i) {
            rhs[i] = n_over_s2 * beta_std[i];
        }
        // Solve L y = rhs (forward), then L^T mu = y (backward)
        tgw::trsv_lower(prec.data(), rhs.data(), b_size);
        tgw::trsv_lower_transpose(prec.data(), rhs.data(), b_size);
        for (std::size_t i = 0; i < b_size; ++i) mu[i] = rhs[i];

        // ---- Sample beta = mu + L^{-T} z, z ~ N(0, I) ----
        for (std::size_t i = 0; i < b_size; ++i) z[i] = normal(rng);
        tgw::trsv_lower_transpose(prec.data(), z.data(), b_size);
        for (std::size_t i = 0; i < b_size; ++i) {
            beta[i] = mu[i] + z[i];
        }

        // ---- psi update: psi_j ~ GIG(lam=a-0.5, chi=beta_j^2/sigma2, psi=2*delta_j) ----
        const double lam_psi = a - 0.5;
        for (std::size_t i = 0; i < b_size; ++i) {
            chi_buf[i] = (beta[i] * beta[i]) / sigma2 + 1e-12;
            psi_param_buf[i] = 2.0 * delta[i] + 1e-12;
        }
        tgw::sample_gig_vec(lam_psi, chi_buf.data(), psi_param_buf.data(),
                            psi_new.data(), b_size, rng);
        for (std::size_t i = 0; i < b_size; ++i) {
            double v = psi_new[i];
            if (v < 1e-12) v = 1e-12;
            if (v > 1e6) v = 1e6;
            psi[i] = v;
        }

        // ---- delta update: delta_j ~ Gamma(shape=a+b, rate=psi_j+phi) ----
        // std::gamma_distribution uses (shape, scale) = (a+b, 1/(psi+phi))
        for (std::size_t i = 0; i < b_size; ++i) {
            std::gamma_distribution<double> gamma(a + b_param, 1.0 / (psi[i] + phi));
            delta[i] = gamma(rng);
        }

        // ---- accumulate posterior moments after burn-in ----
        if (it >= n_burnin) {
            for (std::size_t i = 0; i < b_size; ++i) {
                beta_sum[i] += beta[i];
                beta_sq_sum[i] += beta[i] * beta[i];
            }
            ++n_post;
        }
    }

    if (n_post == 0) n_post = 1;
    const double inv_n = 1.0 / static_cast<double>(n_post);
    for (std::size_t i = 0; i < b_size; ++i) {
        double m = beta_sum[i] * inv_n;
        double v = beta_sq_sum[i] * inv_n - m * m;
        if (v < 0.0) v = 0.0;
        out_mean[i] = m;
        out_sd[i] = std::sqrt(v);
    }
}

// One Gibbs iteration worth of per-SNP sampling: given the current beta and
// delta, draw new (psi, delta) vectors from their full conditionals:
//
//     psi_j   ~ GIG(lam = a - 0.5, chi = beta_j^2 / sigma2, psi = 2 * delta_j)
//     delta_j ~ Gamma(shape = a + b, rate = psi_j_new + phi)
//
// This is the "hybrid path" helper: the caller runs the Cholesky / triangular
// solve in Python via torch (MKL-backed), then delegates only the expensive
// per-SNP GIG + Gamma draws to this C++ entry point. At large b_size the
// hand-rolled Cholesky in run_prscs_gibbs_block starts to lose to MKL, so the
// hybrid path wins there while the full C++ path still wins at small b_size.
void run_prscs_sample_psi_delta(
    const double* beta,       // (m,) current beta draw
    const double* delta_in,   // (m,) current delta (for the GIG psi_param)
    std::size_t m,
    double sigma2,
    double phi,
    double a,
    double b_param,
    std::uint64_t seed,
    double* psi_out,          // (m,)
    double* delta_out         // (m,)
) {
    std::mt19937_64 rng(seed);

    // ---- psi update: GIG(lam = a - 0.5, chi = beta^2/sigma2, psi = 2*delta) ----
    const double lam_psi = a - 0.5;
    std::vector<double> chi_buf(m, 0.0);
    std::vector<double> psi_param_buf(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        chi_buf[i] = (beta[i] * beta[i]) / sigma2 + 1e-12;
        psi_param_buf[i] = 2.0 * delta_in[i] + 1e-12;
    }
    tgw::sample_gig_vec(lam_psi, chi_buf.data(), psi_param_buf.data(),
                        psi_out, m, rng);
    for (std::size_t i = 0; i < m; ++i) {
        double v = psi_out[i];
        if (v < 1e-12) v = 1e-12;
        if (v > 1e6) v = 1e6;
        psi_out[i] = v;
    }

    // ---- delta update: Gamma(shape = a + b, rate = psi_new + phi) ----
    for (std::size_t i = 0; i < m; ++i) {
        std::gamma_distribution<double> gamma(a + b_param, 1.0 / (psi_out[i] + phi));
        delta_out[i] = gamma(rng);
    }
}

// ----- pybind11 wrappers ---------------------------------------------------

py::tuple py_prscs_gibbs_block(
    py::array_t<double, py::array::c_style | py::array::forcecast> beta_std,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_b,
    double n_eff,
    double phi,
    double a,
    double b_param,
    int n_iter,
    int n_burnin,
    std::uint64_t seed
) {
    auto bs_buf = beta_std.request();
    auto rb_buf = R_b.request();
    if (bs_buf.ndim != 1) throw std::invalid_argument("beta_std must be 1-D");
    if (rb_buf.ndim != 2) throw std::invalid_argument("R_b must be 2-D");
    const std::size_t b_size = static_cast<std::size_t>(bs_buf.shape[0]);
    if (rb_buf.shape[0] != static_cast<py::ssize_t>(b_size) ||
        rb_buf.shape[1] != static_cast<py::ssize_t>(b_size)) {
        throw std::invalid_argument("R_b must be (b, b) matching beta_std");
    }

    py::array_t<double> mean(static_cast<py::ssize_t>(b_size));
    py::array_t<double> sd(static_cast<py::ssize_t>(b_size));

    {
        py::gil_scoped_release release;
        run_prscs_gibbs_block(
            static_cast<const double*>(bs_buf.ptr),
            static_cast<const double*>(rb_buf.ptr),
            b_size,
            n_eff, phi, a, b_param,
            n_iter, n_burnin, seed,
            static_cast<double*>(mean.request().ptr),
            static_cast<double*>(sd.request().ptr)
        );
    }

    return py::make_tuple(mean, sd);
}

py::array_t<double> py_sample_gig_vec(
    double lam,
    py::array_t<double, py::array::c_style | py::array::forcecast> chi,
    py::array_t<double, py::array::c_style | py::array::forcecast> psi,
    std::uint64_t seed
) {
    auto chi_buf = chi.request();
    auto psi_buf = psi.request();
    if (chi_buf.ndim != 1 || psi_buf.ndim != 1) {
        throw std::invalid_argument("chi and psi must be 1-D");
    }
    if (chi_buf.shape[0] != psi_buf.shape[0]) {
        throw std::invalid_argument("chi and psi must have the same length");
    }
    const std::size_t n = static_cast<std::size_t>(chi_buf.shape[0]);
    py::array_t<double> out(static_cast<py::ssize_t>(n));

    {
        py::gil_scoped_release release;
        std::mt19937_64 rng(seed);
        tgw::sample_gig_vec(lam,
                            static_cast<const double*>(chi_buf.ptr),
                            static_cast<const double*>(psi_buf.ptr),
                            static_cast<double*>(out.request().ptr),
                            n, rng);
    }
    return out;
}

double py_sample_gig_scalar(double lam, double chi, double psi, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    return tgw::sample_gig(lam, chi, psi, rng);
}

py::tuple py_prscs_sample_psi_delta(
    py::array_t<double, py::array::c_style | py::array::forcecast> beta,
    py::array_t<double, py::array::c_style | py::array::forcecast> delta_in,
    double sigma2,
    double phi,
    double a,
    double b_param,
    std::uint64_t seed
) {
    auto beta_buf = beta.request();
    auto delta_buf = delta_in.request();
    if (beta_buf.ndim != 1 || delta_buf.ndim != 1) {
        throw std::invalid_argument("beta and delta must be 1-D");
    }
    if (beta_buf.shape[0] != delta_buf.shape[0]) {
        throw std::invalid_argument("beta and delta must have the same length");
    }
    const std::size_t m = static_cast<std::size_t>(beta_buf.shape[0]);

    py::array_t<double> psi_out(static_cast<py::ssize_t>(m));
    py::array_t<double> delta_out(static_cast<py::ssize_t>(m));

    {
        py::gil_scoped_release release;
        run_prscs_sample_psi_delta(
            static_cast<const double*>(beta_buf.ptr),
            static_cast<const double*>(delta_buf.ptr),
            m,
            sigma2, phi, a, b_param, seed,
            static_cast<double*>(psi_out.request().ptr),
            static_cast<double*>(delta_out.request().ptr)
        );
    }

    return py::make_tuple(psi_out, delta_out);
}

}  // namespace

PYBIND11_MODULE(_prscs_native, m) {
    m.doc() = "Native C++ accelerator for the PRS-CS block Gibbs sampler.";

    m.def("prscs_gibbs_block", &py_prscs_gibbs_block,
          py::arg("beta_std"),
          py::arg("R_b"),
          py::arg("n_eff"),
          py::arg("phi"),
          py::arg("a"),
          py::arg("b"),
          py::arg("n_iter"),
          py::arg("n_burnin"),
          py::arg("seed"),
          R"pbdoc(
            Run a PRS-CS block Gibbs sampler in C++.

            Returns (posterior_mean, posterior_sd) as float64 ndarrays of shape (b,).
          )pbdoc");

    m.def("sample_gig_vec", &py_sample_gig_vec,
          py::arg("lam"),
          py::arg("chi"),
          py::arg("psi"),
          py::arg("seed"),
          "Vectorized GIG(lam, chi_i, psi_i) sampler. Returns float64 ndarray.");

    m.def("sample_gig_scalar", &py_sample_gig_scalar,
          py::arg("lam"),
          py::arg("chi"),
          py::arg("psi"),
          py::arg("seed"),
          "Scalar GIG(lam, chi, psi) sampler. Each call seeds a fresh RNG.");

    m.def("prscs_sample_psi_delta", &py_prscs_sample_psi_delta,
          py::arg("beta"),
          py::arg("delta_in"),
          py::arg("sigma2"),
          py::arg("phi"),
          py::arg("a"),
          py::arg("b"),
          py::arg("seed"),
          R"pbdoc(
            Single Gibbs iteration of the per-SNP PRS-CS (psi, delta) updates.

            Given the current ``beta`` draw and ``delta_in`` vector, draws new
            ``psi`` from GIG(lam=a-0.5, chi=beta^2/sigma2, psi=2*delta_in) and
            new ``delta`` from Gamma(shape=a+b, rate=psi_new+phi). Returns the
            two updated vectors as (psi_new, delta_new) float64 ndarrays.

            This is the "hybrid path" entry point used when the LD block is
            large enough that the hand-rolled Cholesky in prscs_gibbs_block
            loses to torch's MKL-backed Cholesky; the caller runs the linalg
            in Python via torch and delegates only the per-SNP sampling here.
          )pbdoc");
}
