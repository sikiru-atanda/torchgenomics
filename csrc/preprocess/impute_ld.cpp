// Native accelerator for ``torchgenomics.preprocess.impute.impute_ld``.
//
// The Python reference iterates ``len(miss)`` missing entries and for each
// recomputes per-window correlations against the target SNP via repeated
// torch reductions (`mean`, `std`, matmul, `sum`). At realistic dosage
// matrix sizes the per-entry torch overhead dominates wall time.
//
// The C++ port pulls per-column means and per-column std (ddof=1) out as
// caller-precomputed buffers, then for each missing (i, j) walks the
// window [j-w, j+w] (excluding j), computes the centred dot product against
// the target column directly from a column-major centred matrix view, and
// accumulates a |corr|-weighted prediction in a single pass.
//
// Inputs (numpy float64 / int64, all C-contiguous):
//   G_complete (n, m) float64 — mean-imputed dosage matrix (no NaN). Used
//                                 for centred dot products and the per-entry
//                                 prediction values.
//   miss_mask  (n, m) uint8   — bool mask, 1 where the *original* G was NaN.
//   col_means  (m,)   float64 — per-column mean of G_complete.
//   col_stds   (m,)   float64 — per-column std (ddof=1) of G_complete.
//   window     int            — flanking window size (each side).
//
// Output:
//   G_out      (n, m) float64 — caller pre-clones the original G into this
//                                 buffer. Only positions where miss_mask is
//                                 1 are modified.
//
// Returns Py_None.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

void impute_ld_fill(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_complete,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> miss_mask,
    py::array_t<double, py::array::c_style | py::array::forcecast> col_means,
    py::array_t<double, py::array::c_style | py::array::forcecast> col_stds,
    int window,
    py::array_t<double, py::array::c_style | py::array::forcecast> G_out
) {
    if (G_complete.ndim() != 2) throw std::invalid_argument("G_complete must be 2-D");
    if (miss_mask.ndim() != 2) throw std::invalid_argument("miss_mask must be 2-D");
    if (G_out.ndim() != 2) throw std::invalid_argument("G_out must be 2-D");
    if (col_means.ndim() != 1) throw std::invalid_argument("col_means must be 1-D");
    if (col_stds.ndim() != 1) throw std::invalid_argument("col_stds must be 1-D");
    if (window < 0) throw std::invalid_argument("window must be >= 0");

    const std::size_t n = static_cast<std::size_t>(G_complete.shape(0));
    const std::size_t m = static_cast<std::size_t>(G_complete.shape(1));

    if (static_cast<std::size_t>(miss_mask.shape(0)) != n ||
        static_cast<std::size_t>(miss_mask.shape(1)) != m) {
        throw std::invalid_argument("miss_mask shape must equal G_complete shape");
    }
    if (static_cast<std::size_t>(G_out.shape(0)) != n ||
        static_cast<std::size_t>(G_out.shape(1)) != m) {
        throw std::invalid_argument("G_out shape must equal G_complete shape");
    }
    if (static_cast<std::size_t>(col_means.shape(0)) != m ||
        static_cast<std::size_t>(col_stds.shape(0)) != m) {
        throw std::invalid_argument("col_means / col_stds must have length m");
    }

    const double* gc = G_complete.data();
    const std::uint8_t* mm = miss_mask.data();
    const double* mu = col_means.data();
    const double* sd = col_stds.data();
    double* go = G_out.mutable_data();

    const double denom_n = (n > 1) ? static_cast<double>(n - 1) : 1.0;
    const double tiny = 1e-10;
    const std::size_t w = static_cast<std::size_t>(window);

    {
        py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            for (std::size_t j = 0; j < m; ++j) {
                if (!mm[i * m + j]) continue;

                std::size_t left = (j > w) ? (j - w) : 0;
                std::size_t right = std::min(m, j + w + 1);

                // Empty window after excluding the target SNP
                if (right - left <= 1) {
                    go[i * m + j] = gc[i * m + j];
                    continue;
                }

                const double tstd = sd[j];
                if (tstd < tiny) {
                    go[i * m + j] = gc[i * m + j];
                    continue;
                }
                const double mu_j = mu[j];

                double w_sum = 0.0;
                double pred_num = 0.0;

                for (std::size_t kk = left; kk < right; ++kk) {
                    if (kk == j) continue;
                    const double mu_k = mu[kk];

                    // Centred dot product over rows
                    double dot = 0.0;
                    for (std::size_t l = 0; l < n; ++l) {
                        const double a = gc[l * m + j] - mu_j;
                        const double b = gc[l * m + kk] - mu_k;
                        dot += a * b;
                    }
                    const double k_std = sd[kk] > tiny ? sd[kk] : tiny;
                    const double corr = dot / (denom_n * tstd * k_std);
                    const double weight = std::fabs(corr);
                    w_sum += weight;
                    pred_num += weight * gc[i * m + kk];
                }

                if (w_sum > tiny) {
                    go[i * m + j] = pred_num / w_sum;
                } else {
                    go[i * m + j] = gc[i * m + j];
                }
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_impute_ld_native, mod) {
    mod.doc() = "Native C++ accelerator for LD-window imputation per-missing-entry loop.";
    mod.def("impute_ld_fill", &impute_ld_fill,
            py::arg("G_complete"),
            py::arg("miss_mask"),
            py::arg("col_means"),
            py::arg("col_stds"),
            py::arg("window"),
            py::arg("G_out"),
            "Fill NaN positions of G_out (in place) using LD-window correlations.");
}
