// Native accelerator for ``torchgwas.preprocess.impute.impute_mode``.
//
// The Python reference does:
//
//   for j in range(m):
//       col = G_int[:, j]
//       valid = col[col >= 0]
//       if len(valid) == 0: modes[j] = 0; continue
//       counts = torch.bincount(valid, minlength=max_dosage + 1)
//       modes[j] = float(counts.argmax().item())
//
// On a real GWAS matrix `m` is in the millions, so the per-column
// `.item()` round-trip dominates wall time. The C++ port iterates the
// (n, m) integer dosage matrix in column-major order, runs a fixed-size
// histogram per column, and writes the float64 mode vector — all under
// a single GIL release.
//
// Inputs:
//   G_int      (n, m) int64, with negative values marking missing.
//   max_dosage int      ; histogram size = max_dosage + 1
//
// Output:
//   modes      (m,) float64 — mode dosage per column (0.0 for all-missing).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

py::array_t<double> impute_mode_columns(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> G_int,
    int max_dosage
) {
    if (G_int.ndim() != 2) {
        throw std::invalid_argument("G_int must be 2-D");
    }
    if (max_dosage < 0) {
        throw std::invalid_argument("max_dosage must be >= 0");
    }
    const std::size_t n = static_cast<std::size_t>(G_int.shape(0));
    const std::size_t m = static_cast<std::size_t>(G_int.shape(1));
    const std::size_t bins = static_cast<std::size_t>(max_dosage) + 1;

    auto out = py::array_t<double>(static_cast<py::ssize_t>(m));
    double* op = out.mutable_data();
    const std::int64_t* gp = G_int.data();

    {
        py::gil_scoped_release release;
#pragma omp parallel
        {
            std::vector<std::int64_t> counts(bins, 0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t jj = 0; jj < static_cast<std::ptrdiff_t>(m); ++jj) {
            const std::size_t j = static_cast<std::size_t>(jj);
            // Reset histogram in-place.
            for (std::size_t b = 0; b < bins; ++b) counts[b] = 0;

            std::int64_t n_valid = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const std::int64_t v = gp[i * m + j];
                if (v >= 0 && static_cast<std::size_t>(v) < bins) {
                    ++counts[static_cast<std::size_t>(v)];
                    ++n_valid;
                }
            }

            if (n_valid == 0) {
                op[j] = 0.0;
                continue;
            }

            // argmax with first-wins tie-break (matches torch.argmax).
            std::size_t best = 0;
            std::int64_t best_count = counts[0];
            for (std::size_t b = 1; b < bins; ++b) {
                if (counts[b] > best_count) {
                    best_count = counts[b];
                    best = b;
                }
            }
            op[j] = static_cast<double>(best);
        }  // omp for
        }  // omp parallel
    }  // GIL re-acquired

    return out;
}

}  // namespace

PYBIND11_MODULE(_impute_mode_native, m) {
    m.doc() = "Native C++ accelerator for per-column mode imputation.";
    m.def("impute_mode_columns", &impute_mode_columns,
          py::arg("G_int"),
          py::arg("max_dosage"),
          "Per-column mode (most frequent non-negative value) of an int64 matrix.");
}
