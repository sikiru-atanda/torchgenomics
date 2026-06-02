// Native accelerator for the per-cluster, per-population r² stability
// loop inside ``torchgenomics.ld._blocks_novel.detect_blocks_cross_pop``.
//
// The Python reference does, for each cluster of variant indices and each
// population's r² matrix:
//
//   for ii in range(len(cluster)):
//       for jj in range(ii + 1, len(cluster)):
//           r2_vals.append(r2[cluster[ii], cluster[jj]].item())
//   pop_means.append(mean(r2_vals))
//   stability = min(pop_means) / max(pop_means)
//
// The per-pair ``.item()`` round-trip dominates wall time on cross-population
// studies with hundreds of clusters * a handful of populations * tens of
// SNPs per cluster.
//
// Inputs:
//   pop_r2          (P, m, m) float64 — stacked per-population r² matrices.
//   cluster_starts  (n_clusters + 1,) int64 — CSR-style offsets into
//                                              cluster_indices.
//   cluster_indices (total,) int64 — flat array of variant indices for all
//                                    clusters concatenated.
//
// Output:
//   stability       (n_clusters,) float64 — min(pop_means) / max(pop_means)
//                                            per cluster (0.0 if no valid
//                                            pop has any pair).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace {

py::array_t<double> cross_pop_stability(
    py::array_t<double, py::array::c_style | py::array::forcecast> pop_r2,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cluster_starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> cluster_indices
) {
    auto r_buf = pop_r2.request();
    auto s_buf = cluster_starts.request();
    auto i_buf = cluster_indices.request();
    if (r_buf.ndim != 3) throw std::invalid_argument("pop_r2 must be 3-D");
    if (s_buf.ndim != 1) throw std::invalid_argument("cluster_starts must be 1-D");
    if (i_buf.ndim != 1) throw std::invalid_argument("cluster_indices must be 1-D");

    const std::size_t P = static_cast<std::size_t>(r_buf.shape[0]);
    const std::size_t m = static_cast<std::size_t>(r_buf.shape[1]);
    if (static_cast<std::size_t>(r_buf.shape[2]) != m) {
        throw std::invalid_argument("pop_r2 must be (P, m, m) square");
    }
    if (s_buf.shape[0] < 1) {
        throw std::invalid_argument("cluster_starts must have at least 1 entry");
    }
    const std::size_t n_clusters = static_cast<std::size_t>(s_buf.shape[0] - 1);

    py::array_t<double> out(static_cast<py::ssize_t>(n_clusters));
    auto out_buf = out.request();
    double* out_ptr = static_cast<double*>(out_buf.ptr);
    const double* r_ptr = static_cast<const double*>(r_buf.ptr);
    const std::int64_t* s_ptr = static_cast<const std::int64_t*>(s_buf.ptr);
    const std::int64_t* i_ptr = static_cast<const std::int64_t*>(i_buf.ptr);

    const std::size_t plane = m * m;

    {
        py::gil_scoped_release release;
        for (std::size_t c = 0; c < n_clusters; ++c) {
            out_ptr[c] = 0.0;
            const std::int64_t lo = s_ptr[c];
            const std::int64_t hi = s_ptr[c + 1];
            const std::int64_t b = hi - lo;
            if (b < 2) continue;

            double pmin = 0.0;
            double pmax = 0.0;
            bool any_pop = false;

            for (std::size_t p = 0; p < P; ++p) {
                const double* mat = r_ptr + p * plane;
                double sum = 0.0;
                std::int64_t cnt = 0;
                for (std::int64_t ii = 0; ii < b; ++ii) {
                    const std::int64_t row = i_ptr[lo + ii];
                    if (row < 0 || static_cast<std::size_t>(row) >= m) {
                        throw std::invalid_argument("cluster index out of range");
                    }
                    const double* row_ptr = mat + static_cast<std::size_t>(row) * m;
                    for (std::int64_t jj = ii + 1; jj < b; ++jj) {
                        const std::int64_t col = i_ptr[lo + jj];
                        if (col < 0 || static_cast<std::size_t>(col) >= m) {
                            throw std::invalid_argument("cluster index out of range");
                        }
                        sum += row_ptr[col];
                        ++cnt;
                    }
                }
                if (cnt == 0) continue;
                const double mean = sum / static_cast<double>(cnt);
                if (!any_pop) {
                    pmin = mean;
                    pmax = mean;
                    any_pop = true;
                } else {
                    if (mean < pmin) pmin = mean;
                    if (mean > pmax) pmax = mean;
                }
            }

            if (any_pop && pmax > 1e-30) {
                out_ptr[c] = pmin / pmax;
            }
        }
    }

    return out;
}

}  // namespace

PYBIND11_MODULE(_cross_pop_native, mod) {
    mod.doc() = "Native cross-population block stability inner loop.";
    mod.def("cross_pop_stability", &cross_pop_stability,
            py::arg("pop_r2"), py::arg("cluster_starts"),
            py::arg("cluster_indices"),
            "Per-cluster cross-population stability scores "
            "(min(per-pop mean r²) / max(per-pop mean r²)).");
}
