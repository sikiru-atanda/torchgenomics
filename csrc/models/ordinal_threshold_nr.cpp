// Native accelerator for the OrdinalGLMM threshold-update Newton-
// Raphson inner step in ``torchgwas.models.ordinal_glmm``.
//
// Background. Per outer PQL iteration the Python reference runs ~5
// Newton-Raphson updates on the (J−1) cumulative-link thresholds, and
// each one rebuilds the score vector + (J−1, J−1) Hessian via:
//
//   for jj in range(n_thresh):
//       D_jj = (jj >= Y).float()
//       g_jj = gamma_nr[:, jj]
//       score[jj] = (D_jj - g_jj).sum()
//       H[jj, jj] = -(g_jj * (1 - g_jj)).sum()
//       for kk in range(jj+1, n_thresh):
//           g_kk = gamma_nr[:, kk]
//           H[jj, kk] = H[kk, jj] = -(g_jj * (1 - g_kk)).sum()
//
// At each NR step that is O(n_thresh²) Python reductions, each an
// O(n) torch op. For J ∈ [3, 10] the n_thresh ∈ [2, 9] and the
// arithmetic is small, but the per-(jj, kk) Python turn + tensor
// allocation overhead dominates wall time.
//
// This C++ port assembles the score + Hessian in a single pass over
// the n samples per NR step, accumulating all (n_thresh² / 2) entries
// in registers. Per sample, the inner work is O(n_thresh²); total
// O(n × n_thresh²) = ~5e5 ops for n=5000, n_thresh=9 — ~µs in C++.
//
// The Hessian is symmetric by construction; we fill the upper
// triangle inside the inner loop and mirror at the end. OpenMP
// parallelises across the outer-sample loop with per-thread score +
// upper-triangle accumulators reduced at completion.
//
// Inputs (all C-contiguous):
//   gamma_nr  (n, n_thresh) float64 — cumulative link gammas P(Y ≤ j).
//   Y         (n,) int64 — observed ordinal category in [0, J−1].
//   score_out (n_thresh,) float64 — score vector, written in place.
//   H_out     (n_thresh, n_thresh) float64 — Hessian, written in place.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

void ordinal_threshold_score_hessian(
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma_nr_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> Y_arr,
    py::array_t<double, py::array::c_style> score_out_arr,
    py::array_t<double, py::array::c_style> H_out_arr
) {
    if (gamma_nr_arr.ndim() != 2) throw std::invalid_argument("gamma_nr must be 2-D");
    if (Y_arr.ndim() != 1) throw std::invalid_argument("Y must be 1-D");
    if (score_out_arr.ndim() != 1) throw std::invalid_argument("score_out must be 1-D");
    if (H_out_arr.ndim() != 2) throw std::invalid_argument("H_out must be 2-D");

    const std::size_t n = static_cast<std::size_t>(gamma_nr_arr.shape(0));
    const std::size_t T = static_cast<std::size_t>(gamma_nr_arr.shape(1));

    if (static_cast<std::size_t>(Y_arr.shape(0)) != n) {
        throw std::invalid_argument("Y length must match gamma_nr rows");
    }
    if (static_cast<std::size_t>(score_out_arr.shape(0)) != T) {
        throw std::invalid_argument("score_out length must equal n_thresh");
    }
    if (static_cast<std::size_t>(H_out_arr.shape(0)) != T
        || static_cast<std::size_t>(H_out_arr.shape(1)) != T) {
        throw std::invalid_argument("H_out must be (n_thresh, n_thresh)");
    }

    const double* gamma_nr = gamma_nr_arr.data();
    const int64_t* Y = Y_arr.data();
    double* score_out = score_out_arr.mutable_data();
    double* H_out = H_out_arr.mutable_data();

    // Zero outputs.
    for (std::size_t jj = 0; jj < T; ++jj) score_out[jj] = 0.0;
    for (std::size_t k = 0; k < T * T; ++k) H_out[k] = 0.0;

    if (n == 0 || T == 0) return;

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> score_local(T, 0.0);
            std::vector<double> H_local(T * T, 0.0);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const int64_t y_i = Y[i];
                const double* g_i = gamma_nr + static_cast<std::size_t>(i) * T;
                for (std::size_t jj = 0; jj < T; ++jj) {
                    const double g_jj = g_i[jj];
                    // D_jj = 1 if (jj >= Y[i]) else 0  ↔  Y[i] <= jj
                    const double D_jj = (y_i <= static_cast<int64_t>(jj)) ? 1.0 : 0.0;
                    score_local[jj] += D_jj - g_jj;
                    H_local[jj * T + jj] -= g_jj * (1.0 - g_jj);
                    for (std::size_t kk = jj + 1; kk < T; ++kk) {
                        const double g_kk = g_i[kk];
                        H_local[jj * T + kk] -= g_jj * (1.0 - g_kk);
                    }
                }
            }

#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                for (std::size_t jj = 0; jj < T; ++jj) score_out[jj] += score_local[jj];
                for (std::size_t k = 0; k < T * T; ++k) H_out[k] += H_local[k];
            }
        }

        // Mirror the upper triangle into the lower triangle.
        for (std::size_t jj = 0; jj < T; ++jj) {
            for (std::size_t kk = jj + 1; kk < T; ++kk) {
                H_out[kk * T + jj] = H_out[jj * T + kk];
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_ordinal_threshold_native, m) {
    m.doc() = "Native C++ accelerator for the OrdinalGLMM threshold-update "
              "Newton-Raphson inner step (score vector + Hessian assembly).";
    m.def("ordinal_threshold_score_hessian", &ordinal_threshold_score_hessian,
          py::arg("gamma_nr"),
          py::arg("Y"),
          py::arg("score_out"),
          py::arg("H_out"),
          "Assemble the score vector (n_thresh,) and Hessian "
          "(n_thresh, n_thresh) for the cumulative-link threshold NR "
          "update from the per-sample cumulative gammas and observed "
          "ordinal category. Symmetric H_out filled in place.");
}
