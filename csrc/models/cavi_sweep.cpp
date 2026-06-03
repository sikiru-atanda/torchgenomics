// Native accelerator for the CAVI inner sweep in
// ``torchgenomics.models.bayesian_vs.BayesianVS._cavi_loop``.
//
// The Python reference does, per outer iteration:
//
//   for j in range(p):
//       g_j = G_rot[:, j]
//       wg_j = wG[:, j]                              # = w * g_j
//       old_effect = gamma[j] * mu[j]
//       r += g_j * old_effect                        # add back SNP j
//       sigma2_j = 1 / (gWg[j] + 1/sig2_beta)
//       mu_j = sigma2_j * dot(wg_j, r)
//       log_bf = 0.5*log(sigma2_j/sig2_beta) + 0.5*mu_j^2/sigma2_j
//       log_bf = clamp(log_bf, -30, 30)
//       gamma_j = sigmoid(log(pi/(1-pi)) + log_bf)
//       gamma_j = clamp(gamma_j, 1e-10, 1-1e-10)
//       gamma[j], mu[j], sigma2[j] = gamma_j, mu_j, sigma2_j
//       r -= g_j * (gamma_j * mu_j)
//
// The update is *sequential* (each SNP's residual depends on the
// previous one), so the loop cannot be vectorized at the outer level —
// every SNP forces an n-element saxpy + n-element dot product. The
// per-SNP `.item()` round-trip on `sigma2_j.item()` for the `math.log`
// further dominates the Python path.
//
// This C++ port runs one full sweep over all p SNPs under a single GIL
// release. The Python caller still drives the outer iteration / ELBO /
// EM hyperparameter loop — only the per-SNP coordinate-update inner
// kernel is moved to C++.
//
// Inputs (all float64, all C-contiguous):
//   G_rot      (n, p)  — rotated genotype matrix
//   wG         (n, p)  — weight-multiplied genotype matrix (= w[:,None] * G_rot)
//   gWg        (p,)    — diagonal weights (g_j^T W g_j), constant
//   r          (n,)    — residual; updated in place
//   gamma      (p,)    — inclusion probabilities; updated in place
//   mu         (p,)    — posterior means; updated in place
//   sigma2     (p,)    — posterior variances; updated in place
//   pi         scalar  — current prior inclusion probability
//   sig2_beta  scalar  — current prior effect-size variance
//
// Outputs:
//   r, gamma, mu, sigma2 are updated in place. The function returns nothing.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

namespace {

inline double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

inline double sigmoid(double x) {
    // Numerically stable sigmoid.
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    } else {
        const double z = std::exp(x);
        return z / (1.0 + z);
    }
}

void cavi_sweep(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_rot,
    py::array_t<double, py::array::c_style | py::array::forcecast> wG,
    py::array_t<double, py::array::c_style | py::array::forcecast> gWg,
    py::array_t<double, py::array::c_style> r,
    py::array_t<double, py::array::c_style> gamma_arr,
    py::array_t<double, py::array::c_style> mu_arr,
    py::array_t<double, py::array::c_style> sigma2_arr,
    double pi,
    double sig2_beta
) {
    if (G_rot.ndim() != 2 || wG.ndim() != 2) {
        throw std::invalid_argument("G_rot and wG must be 2-D");
    }
    const std::size_t n = static_cast<std::size_t>(G_rot.shape(0));
    const std::size_t p = static_cast<std::size_t>(G_rot.shape(1));
    if (wG.shape(0) != G_rot.shape(0) || wG.shape(1) != G_rot.shape(1)) {
        throw std::invalid_argument("wG shape must match G_rot");
    }
    if (gWg.ndim() != 1 || static_cast<std::size_t>(gWg.shape(0)) != p) {
        throw std::invalid_argument("gWg must be 1-D of length p");
    }
    if (r.ndim() != 1 || static_cast<std::size_t>(r.shape(0)) != n) {
        throw std::invalid_argument("r must be 1-D of length n");
    }
    if (gamma_arr.ndim() != 1 || static_cast<std::size_t>(gamma_arr.shape(0)) != p) {
        throw std::invalid_argument("gamma must be 1-D of length p");
    }
    if (mu_arr.ndim() != 1 || static_cast<std::size_t>(mu_arr.shape(0)) != p) {
        throw std::invalid_argument("mu must be 1-D of length p");
    }
    if (sigma2_arr.ndim() != 1 || static_cast<std::size_t>(sigma2_arr.shape(0)) != p) {
        throw std::invalid_argument("sigma2 must be 1-D of length p");
    }
    if (pi <= 0.0 || pi >= 1.0) {
        throw std::invalid_argument("pi must be in (0, 1)");
    }
    if (sig2_beta <= 0.0) {
        throw std::invalid_argument("sig2_beta must be > 0");
    }

    const double* G_p = G_rot.data();      // (n, p) row-major
    const double* wG_p = wG.data();        // (n, p) row-major
    const double* gWg_p = gWg.data();      // (p,)
    double* r_p = r.mutable_data();        // (n,)
    double* gamma_p = gamma_arr.mutable_data();  // (p,)
    double* mu_p = mu_arr.mutable_data();        // (p,)
    double* sigma2_p = sigma2_arr.mutable_data();// (p,)

    const double inv_sig2_beta = 1.0 / sig2_beta;
    const double log_odds_prior = std::log(pi / (1.0 - pi));
    const double log_sig2_beta = std::log(sig2_beta);

    {
        py::gil_scoped_release release;

        for (std::size_t j = 0; j < p; ++j) {
            const double old_effect = gamma_p[j] * mu_p[j];

            // r += g_j * old_effect; simultaneously compute dot(wg_j, r_new)
            // We split into two passes for clarity (and to keep r updated
            // before the dot, matching the Python sequence exactly).
            if (old_effect != 0.0) {
                for (std::size_t i = 0; i < n; ++i) {
                    r_p[i] += G_p[i * p + j] * old_effect;
                }
            }

            const double sigma2_j = 1.0 / (gWg_p[j] + inv_sig2_beta);

            double score = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                score += wG_p[i * p + j] * r_p[i];
            }
            const double mu_j = sigma2_j * score;

            double log_bf = 0.5 * (std::log(sigma2_j) - log_sig2_beta)
                          + 0.5 * mu_j * mu_j / sigma2_j;
            log_bf = clamp(log_bf, -30.0, 30.0);

            double gamma_j = sigmoid(log_odds_prior + log_bf);
            gamma_j = clamp(gamma_j, 1e-10, 1.0 - 1e-10);

            gamma_p[j] = gamma_j;
            mu_p[j] = mu_j;
            sigma2_p[j] = sigma2_j;

            const double new_effect = gamma_j * mu_j;
            if (new_effect != 0.0) {
                for (std::size_t i = 0; i < n; ++i) {
                    r_p[i] -= G_p[i * p + j] * new_effect;
                }
            }
        }
    }  // GIL re-acquired
}

}  // namespace

PYBIND11_MODULE(_cavi_native, m) {
    m.doc() = "Native C++ accelerator for the BayesianVS CAVI inner sweep.";
    m.def("cavi_sweep", &cavi_sweep,
          py::arg("G_rot"),
          py::arg("wG"),
          py::arg("gWg"),
          py::arg("r"),
          py::arg("gamma"),
          py::arg("mu"),
          py::arg("sigma2"),
          py::arg("pi"),
          py::arg("sig2_beta"),
          "Run one full CAVI coordinate-update sweep over all p SNPs in place.");
}
