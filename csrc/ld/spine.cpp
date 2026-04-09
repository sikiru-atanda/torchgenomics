// Native accelerator for the "solid spine of LD" block detector in
// ``torchgwas.ld._blocks.detect_blocks_spine``.
//
// The pure-Python loop builds a Python dict from (i, j) -> |D'| keys and
// then greedily extends each block, checking |D'(k, candidate)| for every
// k in [start..end] before each rightward extension. That double loop with
// dict lookups dominates wall time on dense pair tables. We replace it with
// a row-major dense |D'| matrix lookup and run the entire greedy partition
// + spine validation inside a single GIL-released C++ pass.
//
// Exposed via pybind11:
//
//   spine_partition(dp_matrix, n_snps, d_prime_threshold, min_block_snps)
//       -> list[tuple[int, int]]            # (start, end) inclusive
//
// `dp_matrix` is a dense (n, n) float64 array containing |D'| values; the
// caller fills it from the sparse PairwiseLD pair list. Off-window entries
// should be 0.0 (or any value below the threshold), which is what the
// Python reference observes via "key not in dp_map".

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

py::list spine_partition(
    py::array_t<double, py::array::c_style | py::array::forcecast> dp_matrix,
    std::int64_t n_snps_i64,
    double d_prime_threshold,
    std::int64_t min_block_snps_i64
) {
    if (dp_matrix.ndim() != 2 || dp_matrix.shape(0) != dp_matrix.shape(1)) {
        throw std::invalid_argument("dp_matrix must be a square 2-D array");
    }
    const std::size_t n = static_cast<std::size_t>(n_snps_i64);
    if (static_cast<std::size_t>(dp_matrix.shape(0)) != n) {
        throw std::invalid_argument("dp_matrix shape must equal n_snps");
    }
    const std::size_t min_block = static_cast<std::size_t>(min_block_snps_i64);

    py::list out;
    if (n < 2 || min_block < 1) return out;

    const double* M = dp_matrix.data();

    std::vector<std::tuple<int, int>> blocks;
    {
        py::gil_scoped_release release;
        std::vector<char> used(n, 0);
        std::size_t start = 0;
        while (start < n) {
            if (used[start]) { ++start; continue; }
            std::size_t end = start;
            while (end + 1 < n) {
                const std::size_t cand = end + 1;
                bool ok = true;
                for (std::size_t k = start; k < cand; ++k) {
                    // Match Python: dp_map keyed by (i, j) with i < j;
                    // we read M[k * n + cand] (k < cand by construction).
                    if (M[k * n + cand] < d_prime_threshold) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
                end = cand;
            }
            if (end - start + 1 >= min_block) {
                for (std::size_t k = start; k <= end; ++k) used[k] = 1;
                blocks.emplace_back(static_cast<int>(start),
                                    static_cast<int>(end));
            }
            start = end + 1;
        }
    }  // GIL re-acquired

    for (const auto& blk : blocks) {
        out.append(py::make_tuple(std::get<0>(blk), std::get<1>(blk)));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_spine_native, m) {
    m.doc() = "Native C++ accelerator for the solid-spine LD block detector.";
    m.def("spine_partition", &spine_partition,
          py::arg("dp_matrix"),
          py::arg("n_snps"),
          py::arg("d_prime_threshold"),
          py::arg("min_block_snps"),
          "Greedy spine-of-LD partition over a dense |D'| matrix.");
}
