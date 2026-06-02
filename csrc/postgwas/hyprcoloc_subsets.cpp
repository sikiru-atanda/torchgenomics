// Native accelerator for the hyprcoloc 2^K subset-evidence enumeration
// in ``torchgenomics.postgwas._hyprcoloc.hyprcoloc``.
//
// Background. The Python reference enumerates every non-singleton
// subset S of the K trait indices and computes
//
//     subset_log_bf[S] = log( sum_j exp( sum_{k in S} log_abf[k, j] ) )
//
// via itertools.combinations + log_abf[list(subset)].sum(dim=0) +
// _log_sum_exp. For K = 10 that is 1013 subsets, each spending one
// Python turn + one tensor fancy-index + one logsumexp call. For
// K = 15 it is 32_752 subsets — Python overhead dominates.
//
// This C++ port enumerates subsets via bitmask, computes the per-SNP
// trait-row sum + per-subset logsumexp inline (one std::log-add per
// SNP per subset), and writes the per-bitmask log-BF into a flat
// (2^K,) array indexed by the bitmask. Singleton and empty bitmasks
// are written but the Python caller never reads them.
//
// OpenMP parallelises across subsets.
//
// Inputs (all C-contiguous):
//   log_abf  (K, m) float64 — per-trait per-SNP log approximate BFs.
//   out      (2^K,)  float64 — subset-evidence log-BF indexed by
//                              the bitmask (bit k = trait k included).
//
// Per-subset work: O(|S| m + m) ≈ O(K m) on average. Total:
// O(2^K K m). For K=10, m=1000: ~10M ops, sub-ms in C++.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

// popcount for a 32-bit integer (used to count trait-set size from
// the bitmask, so we can skip empty + singleton subsets on the
// Python-facing side without separate index arrays).
inline int popcount32(std::uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int c = 0;
    while (x) { c += (x & 1); x >>= 1; }
    return c;
#endif
}

void enumerate_subset_log_bf(
    py::array_t<double, py::array::c_style | py::array::forcecast> log_abf_arr,
    py::array_t<double, py::array::c_style> out_arr
) {
    if (log_abf_arr.ndim() != 2) throw std::invalid_argument("log_abf must be 2-D");
    if (out_arr.ndim() != 1) throw std::invalid_argument("out must be 1-D");

    const std::size_t K = static_cast<std::size_t>(log_abf_arr.shape(0));
    const std::size_t m = static_cast<std::size_t>(log_abf_arr.shape(1));
    if (K == 0) throw std::invalid_argument("K must be > 0");
    if (K > 30) {
        throw std::invalid_argument(
            "K > 30 not supported (2^K would exceed array indexing limits)"
        );
    }
    const std::size_t n_subsets = static_cast<std::size_t>(1) << K;
    if (static_cast<std::size_t>(out_arr.shape(0)) != n_subsets) {
        throw std::invalid_argument("out length must equal 2^K");
    }

    const double* log_abf = log_abf_arr.data();
    double* out = out_arr.mutable_data();

    // The empty subset has no meaningful log-BF; mark it.
    out[0] = -std::numeric_limits<double>::infinity();

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> stacked(m);

#ifdef _OPENMP
            #pragma omp for schedule(dynamic, 64)
#endif
            for (std::ptrdiff_t bitmask = 1;
                 bitmask < static_cast<std::ptrdiff_t>(n_subsets);
                 ++bitmask) {
                const int size = popcount32(static_cast<std::uint32_t>(bitmask));
                if (size < 2) {
                    // Singleton — not consumed by hyprcoloc, fill sentinel.
                    out[bitmask] = -std::numeric_limits<double>::infinity();
                    continue;
                }

                // stacked[j] = sum_{k in S} log_abf[k, j].
                for (std::size_t j = 0; j < m; ++j) stacked[j] = 0.0;
                for (std::size_t k = 0; k < K; ++k) {
                    if ((bitmask >> k) & 1u) {
                        const double* row = log_abf + k * m;
                        for (std::size_t j = 0; j < m; ++j) {
                            stacked[j] += row[j];
                        }
                    }
                }

                // Numerically stable logsumexp over j.
                double max_v = -std::numeric_limits<double>::infinity();
                for (std::size_t j = 0; j < m; ++j) {
                    if (stacked[j] > max_v) max_v = stacked[j];
                }
                if (!std::isfinite(max_v)) {
                    out[bitmask] = max_v;  // -inf or NaN propagates
                    continue;
                }
                double sum_exp = 0.0;
                for (std::size_t j = 0; j < m; ++j) {
                    sum_exp += std::exp(stacked[j] - max_v);
                }
                out[bitmask] = max_v + std::log(sum_exp);
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_hyprcoloc_native, m) {
    m.doc() = "Native C++ accelerator for the hyprcoloc 2^K subset-evidence "
              "enumeration.";
    m.def("enumerate_subset_log_bf", &enumerate_subset_log_bf,
          py::arg("log_abf"),
          py::arg("out"),
          "For each bitmask in [0, 2^K), compute "
          "logsumexp_j(sum_{k in S(bitmask)} log_abf[k, j]) and write "
          "into out[bitmask]. Empty and singleton bitmasks are filled "
          "with -inf sentinel and ignored by the Python caller.");
}
