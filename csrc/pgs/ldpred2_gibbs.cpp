// LDpred2 block Gibbs sampler — C++ port of
// torchgwas.pgs.ldpred2._ldpred2_gibbs_block.
//
// The pure-Python implementation in torchgwas/pgs/ldpred2.py is the canonical
// reference. This module mirrors it line-for-line in C++ for performance.
//
// Spike-and-slab prior on standardized SNP effects:
//
//     beta_j | causal_j = 1  ~  N(0, h2 / (b * p_causal))
//     beta_j | causal_j = 0  ~  delta_0   (point mass at 0)
//
// Per-SNP sequential Gibbs with cached residual update:
//
//     r_j = beta_std_j - (R @ beta)_j + R_jj * beta_j
//
// On each draw of beta_j we incrementally update Rb_beta by the column shift.
//
// Exposed via pybind11:
//
//   ldpred2_gibbs_block(beta_std, R_b, p_causal, h2, n_eff,
//                       n_iter, n_burnin, sparse, update_hyperparams, seed)
//       -> (beta_mean, causal_mean, p_trace, h2_trace)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

// Run one LDpred2 Gibbs block. Caller owns the input/output buffers.
void run_ldpred2_gibbs_block(
    const double* beta_std,    // (b,)
    const double* R_b,         // (b * b) row-major
    std::size_t b_size,
    double p_causal,
    double h2,
    double n_eff,
    int n_iter,
    int n_burnin,
    bool sparse,
    bool update_hyperparams,
    std::uint64_t seed,
    double* out_beta_mean,     // (b,)
    double* out_causal_mean,   // (b,)
    std::vector<double>& p_trace,
    std::vector<double>& h_trace
) {
    if (b_size == 0) return;
    if (n_iter <= 0) {
        throw std::invalid_argument("n_iter must be positive");
    }
    if (n_burnin < 0 || n_burnin >= n_iter) {
        throw std::invalid_argument("n_burnin must be in [0, n_iter)");
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);

    constexpr double LOG_2PI = 1.8378770664093454835606594728112352798;  // log(2*pi)
    const double b_d = static_cast<double>(b_size);

    double p = p_causal;
    double h = h2;

    // State
    std::vector<double> beta_curr(beta_std, beta_std + b_size);  // warm start
    std::vector<double> causal_curr(b_size, 1.0);
    std::vector<double> Rb_beta(b_size, 0.0);

    // Posterior accumulators
    std::vector<double> beta_sum(b_size, 0.0);
    std::vector<double> causal_sum(b_size, 0.0);
    int n_post = 0;

    // Cache of R_jj diagonals (stable across iterations)
    std::vector<double> R_diag(b_size);
    for (std::size_t j = 0; j < b_size; ++j) R_diag[j] = R_b[j * b_size + j];

    // Initial Rb_beta = R @ beta_curr  (full GEMV)
    for (std::size_t i = 0; i < b_size; ++i) {
        double s = 0.0;
        const double* row = R_b + i * b_size;
        for (std::size_t k = 0; k < b_size; ++k) s += row[k] * beta_curr[k];
        Rb_beta[i] = s;
    }

    for (int it = 0; it < n_iter; ++it) {
        // -- Refresh Rb_beta from scratch each iteration to eliminate
        //    accumulated rounding error from incremental updates.
        for (std::size_t i = 0; i < b_size; ++i) {
            double s = 0.0;
            const double* row = R_b + i * b_size;
            for (std::size_t k = 0; k < b_size; ++k) s += row[k] * beta_curr[k];
            Rb_beta[i] = s;
        }

        for (std::size_t j = 0; j < b_size; ++j) {
            const double Rjj = R_diag[j];
            // Residual: beta_std_j - (Rb_beta_j - R_jj * beta_curr_j)
            const double r_j = beta_std[j] - (Rb_beta[j] - Rjj * beta_curr[j]);

            const double prior_var = h / std::max(b_d * p, 1e-12);
            const double post_prec = Rjj * n_eff + 1.0 / prior_var;
            const double post_var = 1.0 / post_prec;
            const double post_mean = post_var * n_eff * r_j;

            const double sigma2_obs = Rjj / n_eff;
            const double s_plus_pv = sigma2_obs + prior_var;
            const double r2 = r_j * r_j;
            const double log_marg_causal = -0.5 * (std::log(s_plus_pv) + LOG_2PI + r2 / s_plus_pv);
            const double log_marg_null = -0.5 * (std::log(sigma2_obs) + LOG_2PI + r2 / sigma2_obs);
            const double log_odds = std::log(p / std::max(1.0 - p, 1e-12))
                                    + log_marg_causal - log_marg_null;
            const double prob_causal = 1.0 / (1.0 + std::exp(-log_odds));

            double new_beta_val;
            double causal_new;
            if (uniform(rng) < prob_causal) {
                const double noise = normal(rng);
                new_beta_val = post_mean + std::sqrt(post_var) * noise;
                causal_new = 1.0;
            } else {
                // Both sparse and non-sparse cases collapse to 0 in the Python ref.
                (void)sparse;
                new_beta_val = 0.0;
                causal_new = 0.0;
            }

            const double delta = new_beta_val - beta_curr[j];
            if (delta != 0.0) {
                // Rb_beta += delta * R[:, j]
                for (std::size_t i = 0; i < b_size; ++i) {
                    Rb_beta[i] += delta * R_b[i * b_size + j];
                }
            }
            beta_curr[j] = new_beta_val;
            causal_curr[j] = causal_new;
        }

        if (update_hyperparams) {
            double n_causal = 0.0;
            double sum_sq = 0.0;
            for (std::size_t j = 0; j < b_size; ++j) {
                n_causal += causal_curr[j];
                sum_sq += beta_curr[j] * beta_curr[j] * causal_curr[j];
            }
            double new_p = (n_causal + 1.0) / (b_d + 2.0);
            if (new_p < 1e-6) new_p = 1e-6;
            if (new_p > 1.0 - 1e-6) new_p = 1.0 - 1e-6;
            double new_h;
            if (n_causal > 0.0) {
                new_h = sum_sq * b_d * new_p / n_causal;
                if (new_h < 1e-6) new_h = 1e-6;
            } else {
                new_h = h;
            }
            p = new_p;
            h = (new_h < 1.0) ? new_h : 1.0;
            p_trace.push_back(p);
            h_trace.push_back(h);
        }

        if (it >= n_burnin) {
            for (std::size_t j = 0; j < b_size; ++j) {
                beta_sum[j] += beta_curr[j];
                causal_sum[j] += causal_curr[j];
            }
            ++n_post;
        }
    }

    if (n_post == 0) n_post = 1;
    const double inv_n = 1.0 / static_cast<double>(n_post);
    for (std::size_t j = 0; j < b_size; ++j) {
        out_beta_mean[j] = beta_sum[j] * inv_n;
        out_causal_mean[j] = causal_sum[j] * inv_n;
    }
}

py::tuple py_ldpred2_gibbs_block(
    py::array_t<double, py::array::c_style | py::array::forcecast> beta_std,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_b,
    double p_causal,
    double h2,
    double n_eff,
    int n_iter,
    int n_burnin,
    bool sparse,
    bool update_hyperparams,
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

    py::array_t<double> beta_mean(static_cast<py::ssize_t>(b_size));
    py::array_t<double> causal_mean(static_cast<py::ssize_t>(b_size));
    std::vector<double> p_trace;
    std::vector<double> h_trace;

    {
        py::gil_scoped_release release;
        run_ldpred2_gibbs_block(
            static_cast<const double*>(bs_buf.ptr),
            static_cast<const double*>(rb_buf.ptr),
            b_size,
            p_causal, h2, n_eff,
            n_iter, n_burnin,
            sparse, update_hyperparams,
            seed,
            static_cast<double*>(beta_mean.request().ptr),
            static_cast<double*>(causal_mean.request().ptr),
            p_trace, h_trace
        );
    }

    py::list p_list = py::cast(p_trace);
    py::list h_list = py::cast(h_trace);
    return py::make_tuple(beta_mean, causal_mean, p_list, h_list);
}

}  // namespace

PYBIND11_MODULE(_ldpred2_native, m) {
    m.doc() = "Native C++ accelerator for the LDpred2 block Gibbs sampler.";

    m.def("ldpred2_gibbs_block", &py_ldpred2_gibbs_block,
          py::arg("beta_std"),
          py::arg("R_b"),
          py::arg("p_causal"),
          py::arg("h2"),
          py::arg("n_eff"),
          py::arg("n_iter"),
          py::arg("n_burnin"),
          py::arg("sparse"),
          py::arg("update_hyperparams"),
          py::arg("seed"),
          R"pbdoc(
            Run an LDpred2 block Gibbs sampler in C++.

            Returns (beta_mean, causal_mean, p_trace, h2_trace).
            Both arrays are float64 ndarrays of shape (b,). The trace lists
            are populated only when update_hyperparams=True; otherwise empty.
          )pbdoc");
}
