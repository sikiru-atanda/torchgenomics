// Native accelerator for the TWAS expression-preprocessing transforms
// in ``torchgenomics.preprocess.expression``:
//
//   - inverse_normal_transform: per-column Blom rank-INT with average
//     tie-breaking (scipy.stats.rankdata "average" semantics).
//   - quantile_normalize: column-wise quantile normalisation against
//     a reference distribution, with average tie-breaking.
//
// Background. Both helpers run a per-column stateful ``while``-loop
// over sorted column values, walking ties and assigning the average
// rank (or the average reference value across the tied positions).
// At GTEx-scale matrices — n_samples ≈ 200, n_genes ≈ 20_000 — this
// is 20_000 Python sort + 20_000 inner-tie-resolution Python loops.
//
// This C++ port runs both transforms with the entire per-column work
// inside a single GIL release. Two entry points share the same inner
// scan but produce different per-column outputs.
//
// Strategy:
//
// 1. ``rank_int_u_columns(x_in, c, u_out)`` — for each column, sort,
//    assign average ranks to ties, write u = (rank − c) / (n − 2c +1)
//    into u_out. The caller then applies ``sqrt(2) * erfinv(2u − 1)``
//    on the GPU/CPU torch tensor; erfinv is already a fast torch op.
//
// 2. ``quantile_normalize_columns(x_in, reference, out)`` — for each
//    column, sort, walk ties, average the reference values across the
//    tied positions, scatter back to out. Reference is precomputed
//    in Python from sorted_x.mean(dim=1) — cheap and torch-vectorised.
//
// Both loops are independent across columns; OpenMP parallelises the
// outer column dimension. Per-thread scratch holds (n,) sort index +
// (n,) sorted values.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

// Average-tie rank for one column.
// On entry: col_vals[0..n) — column values.
// On exit:  ranks[0..n)    — 1-based average rank per row index.
// Scratch:  sort_idx, sort_vals (both size n).
inline void compute_avg_ranks_column(
    const double* col_vals,
    std::size_t n,
    double* ranks,
    std::vector<std::int64_t>& sort_idx,
    std::vector<double>& sort_vals
) {
    sort_idx.resize(n);
    sort_vals.resize(n);
    for (std::size_t i = 0; i < n; ++i) sort_idx[i] = static_cast<std::int64_t>(i);
    std::sort(sort_idx.begin(), sort_idx.end(),
              [&](std::int64_t a, std::int64_t b) {
                  return col_vals[a] < col_vals[b];
              });
    for (std::size_t i = 0; i < n; ++i) sort_vals[i] = col_vals[sort_idx[i]];

    std::size_t i = 0;
    while (i < n) {
        std::size_t j = i;
        while (j + 1 < n && sort_vals[j + 1] == sort_vals[i]) ++j;
        // 1-based average rank.
        const double avg_rank = 0.5 * static_cast<double>(i + j) + 1.0;
        for (std::size_t k = i; k <= j; ++k) {
            ranks[sort_idx[k]] = avg_rank;
        }
        i = j + 1;
    }
}

// Per-column rank-INT "u" values: u = (avg_rank - c) / (n - 2c + 1).
void rank_int_u_columns(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in_arr,
    double c,
    py::array_t<double, py::array::c_style> u_out_arr
) {
    if (x_in_arr.ndim() != 2) throw std::invalid_argument("x_in must be 2-D");
    if (u_out_arr.ndim() != 2) throw std::invalid_argument("u_out must be 2-D");

    const std::size_t n = static_cast<std::size_t>(x_in_arr.shape(0));
    const std::size_t m = static_cast<std::size_t>(x_in_arr.shape(1));
    if (static_cast<std::size_t>(u_out_arr.shape(0)) != n
        || static_cast<std::size_t>(u_out_arr.shape(1)) != m) {
        throw std::invalid_argument("u_out shape must match x_in");
    }
    if (n == 0 || m == 0) return;

    const double* x_in = x_in_arr.data();
    double* u_out = u_out_arr.mutable_data();
    const double denom = static_cast<double>(n) - 2.0 * c + 1.0;
    if (!(denom > 0.0)) throw std::invalid_argument("n - 2c + 1 must be > 0");
    const double inv_denom = 1.0 / denom;

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> col_vals(n);
            std::vector<double> ranks(n);
            std::vector<std::int64_t> sort_idx;
            std::vector<double> sort_vals;

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(m); ++j) {
                // Strided gather: column j across rows. x_in is row-major.
                for (std::size_t i = 0; i < n; ++i) {
                    col_vals[i] = x_in[i * m + j];
                }
                compute_avg_ranks_column(
                    col_vals.data(), n, ranks.data(), sort_idx, sort_vals
                );
                for (std::size_t i = 0; i < n; ++i) {
                    u_out[i * m + j] = (ranks[i] - c) * inv_denom;
                }
            }
        }
    }
}

// Column-wise quantile normalisation against a precomputed reference.
// reference[i] = mean of the i-th smallest values across columns
// (precomputed in Python via sorted_x.mean(dim=1)). For each column we
// re-sort, walk ties, average reference[i:k+1] across the tied
// positions, scatter back to out.
void quantile_normalize_columns(
    py::array_t<double, py::array::c_style | py::array::forcecast> x_in_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> reference_arr,
    py::array_t<double, py::array::c_style> out_arr
) {
    if (x_in_arr.ndim() != 2) throw std::invalid_argument("x_in must be 2-D");
    if (reference_arr.ndim() != 1) throw std::invalid_argument("reference must be 1-D");
    if (out_arr.ndim() != 2) throw std::invalid_argument("out must be 2-D");

    const std::size_t n = static_cast<std::size_t>(x_in_arr.shape(0));
    const std::size_t m = static_cast<std::size_t>(x_in_arr.shape(1));
    if (static_cast<std::size_t>(reference_arr.shape(0)) != n) {
        throw std::invalid_argument("reference length must equal n");
    }
    if (static_cast<std::size_t>(out_arr.shape(0)) != n
        || static_cast<std::size_t>(out_arr.shape(1)) != m) {
        throw std::invalid_argument("out shape must match x_in");
    }
    if (n == 0 || m == 0) return;

    const double* x_in = x_in_arr.data();
    const double* reference = reference_arr.data();
    double* out = out_arr.mutable_data();

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> col_vals(n);
            std::vector<std::int64_t> sort_idx(n);
            std::vector<double> sort_vals(n);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(m); ++j) {
                for (std::size_t i = 0; i < n; ++i) {
                    col_vals[i] = x_in[i * m + j];
                }
                std::iota(sort_idx.begin(), sort_idx.end(), 0);
                std::sort(sort_idx.begin(), sort_idx.end(),
                          [&](std::int64_t a, std::int64_t b) {
                              return col_vals[a] < col_vals[b];
                          });
                for (std::size_t i = 0; i < n; ++i) sort_vals[i] = col_vals[sort_idx[i]];

                std::size_t i = 0;
                while (i < n) {
                    std::size_t k = i;
                    while (k + 1 < n && sort_vals[k + 1] == sort_vals[i]) ++k;
                    // Average reference values across the tied run.
                    double s = 0.0;
                    for (std::size_t t = i; t <= k; ++t) s += reference[t];
                    const double avg_ref = s / static_cast<double>(k - i + 1);
                    for (std::size_t t = i; t <= k; ++t) {
                        out[static_cast<std::size_t>(sort_idx[t]) * m + j] = avg_ref;
                    }
                    i = k + 1;
                }
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_expression_native, m) {
    m.doc() = "Native C++ accelerator for TWAS expression-preprocessing "
              "transforms (rank-INT u values + quantile normalisation).";
    m.def("rank_int_u_columns", &rank_int_u_columns,
          py::arg("x_in"), py::arg("c"), py::arg("u_out"),
          "Per-column average-tie rank converted to u = (rank − c) / "
          "(n − 2c + 1). Caller applies sqrt(2)·erfinv(2u − 1) for the "
          "rank-INT final pass.");
    m.def("quantile_normalize_columns", &quantile_normalize_columns,
          py::arg("x_in"), py::arg("reference"), py::arg("out"),
          "Column-wise quantile normalisation: per column, sort with "
          "average-tie reference averaging, scatter into out (in place).");
}
