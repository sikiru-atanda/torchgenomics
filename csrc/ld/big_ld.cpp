// Native accelerator for the Big-LD (Kim et al. 2018) candidate-interval
// scan in ``torchgenomics.ld._blocks_literature.detect_blocks_big_ld``.
//
// The pure-Python loop walks every (i, j) window and re-computes the
// off-diagonal mean of R[i:j+1, i:j+1] via tensor masking + reduction +
// .item(). For a chromosome with m SNPs and a window of W that is
// O(m * W) materialized sub-tensors. We replace it with two prefix
// sums (one 2-D over R, one 1-D over its diagonal) so each query is
// O(1) and the total work is O(m^2) C++ with no Python overhead.
//
// Exposed via pybind11:
//
//   big_ld_intervals(R, chr_pos, r2_threshold, min_block_snps,
//                    window_size, max_bp)
//       -> list[tuple[int, int, float]]
//
// Each tuple is (start_local, end_local, weight) with
// weight = (end - start + 1) * mean_off_diag_r2.
// The Python wrapper still runs `greedy_mwis` over the returned list.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

py::list big_ld_intervals(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_pos,
    double r2_threshold,
    std::int64_t min_block_snps_i64,
    std::int64_t window_size_i64,
    double max_bp
) {
    auto bR = R.request();
    auto bP = chr_pos.request();
    if (bR.ndim != 2) throw std::invalid_argument("R must be 2-D");
    if (bR.shape[0] != bR.shape[1]) throw std::invalid_argument("R must be square");
    if (bP.ndim != 1) throw std::invalid_argument("chr_pos must be 1-D");
    const std::size_t m = static_cast<std::size_t>(bR.shape[0]);
    if (static_cast<std::size_t>(bP.shape[0]) != m) {
        throw std::invalid_argument("chr_pos length must equal R.shape[0]");
    }

    const std::size_t min_block = static_cast<std::size_t>(
        std::max<std::int64_t>(min_block_snps_i64, 1));
    const std::size_t window_size = static_cast<std::size_t>(
        std::max<std::int64_t>(window_size_i64, 1));

    py::list out;
    if (m == 0) return out;

    const double* Rp = static_cast<const double*>(bR.ptr);
    const std::int64_t* Pp = static_cast<const std::int64_t*>(bP.ptr);

    // (i, j, weight) tuples accumulated under the released GIL.
    std::vector<std::tuple<std::size_t, std::size_t, double>> accepted;

    {
        py::gil_scoped_release release;

        // 2-D prefix sum P2[i+1][j+1] = sum_{a<=i, b<=j} R[a, b]
        const std::size_t stride = m + 1;
        std::vector<double> P2(stride * stride, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            double row_acc = 0.0;
            const double* Ri = Rp + i * m;
            double* P2_next = P2.data() + (i + 1) * stride;
            const double* P2_prev = P2.data() + i * stride;
            for (std::size_t j = 0; j < m; ++j) {
                row_acc += Ri[j];
                P2_next[j + 1] = P2_prev[j + 1] + row_acc;
            }
        }

        // 1-D prefix sum over the diagonal: D1[i+1] = sum_{k<=i} R[k, k]
        std::vector<double> D1(m + 1, 0.0);
        for (std::size_t k = 0; k < m; ++k) {
            D1[k + 1] = D1[k] + Rp[k * m + k];
        }

        auto rect_sum = [&](std::size_t r0, std::size_t r1,
                            std::size_t c0, std::size_t c1) -> double {
            return P2[r1 * stride + c1]
                 - P2[r0 * stride + c1]
                 - P2[r1 * stride + c0]
                 + P2[r0 * stride + c0];
        };

        accepted.reserve(m * 2);

        for (std::size_t i = 0; i < m; ++i) {
            // Mirror Python: range(i + min_block - 1, min(i + window_size, m))
            const std::size_t j_start = i + (min_block > 0 ? min_block - 1 : 0);
            const std::size_t j_end = std::min(i + window_size, m);
            if (j_start >= j_end) continue;
            const std::int64_t pos_i = Pp[i];

            for (std::size_t j = j_start; j < j_end; ++j) {
                if (static_cast<double>(Pp[j] - pos_i) > max_bp) break;
                const std::size_t size = j - i + 1;
                if (size < 2) continue;  // no off-diagonal pairs
                const double full = rect_sum(i, j + 1, i, j + 1);
                const double diag = D1[j + 1] - D1[i];
                const double sum_off = full - diag;
                const double n_off = static_cast<double>(size) *
                                     static_cast<double>(size - 1);
                const double mean_r2 = sum_off / n_off;
                if (mean_r2 >= r2_threshold) {
                    const double weight = static_cast<double>(size) * mean_r2;
                    accepted.emplace_back(i, j, weight);
                }
            }
        }
    }  // GIL re-acquired

    for (const auto& iv : accepted) {
        out.append(py::make_tuple(
            static_cast<int>(std::get<0>(iv)),
            static_cast<int>(std::get<1>(iv)),
            std::get<2>(iv)
        ));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_big_ld_native, m) {
    m.doc() = "Native C++ accelerator for the Big-LD candidate interval scan.";
    m.def("big_ld_intervals", &big_ld_intervals,
          py::arg("R"), py::arg("chr_pos"),
          py::arg("r2_threshold"),
          py::arg("min_block_snps"),
          py::arg("window_size"),
          py::arg("max_bp"),
          "Build Big-LD candidate (i, j, weight) intervals via 2D prefix sums on R.");
}
