// Native accelerator for the windowed r² adjacency build inside
// ``torchgenomics.ld._blocks_literature.detect_blocks_cc_graph``.
//
// The pure-Python implementation walks every (i, j) pair within a
// (window, max_bp) constraint and re-computes r² SNP-by-SNP using:
//
//     gi - gi.mean();  gj - gj.mean();
//     cov  = (gi_c * gj_c).mean();
//     vi   = (gi_c * gi_c).mean();
//     vj   = (gj_c * gj_c).mean();
//     r2   = (cov*cov / (vi*vj)).item()
//
// Each of those steps is a fresh CPU↔Python tensor round-trip; for
// m_chr in the thousands the cost is dominated by Python overhead, not
// arithmetic. We replace the entire loop with a C++ double-loop over
// pre-centered columns and a precomputed per-SNP variance vector.
//
// Exposed via pybind11:
//
//   cc_graph_windowed_adj(
//       G_centered,    // (n, m_chr) float64 row-major
//       col_var,       // (m_chr,)   float64
//       chr_pos,       // (m_chr,)   int64
//       window,        // int (max index distance; same semantics as Python)
//       max_bp,        // double
//       r2_threshold   // double
//   ) -> ndarray (m_chr, m_chr) float64
//
// The returned matrix mirrors the Python ``adj`` exactly: zero
// everywhere except for symmetric (i, j) entries with r² > threshold.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<double> cc_graph_windowed_adj(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_centered,
    py::array_t<double, py::array::c_style | py::array::forcecast> col_var,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_pos,
    std::int64_t window_i64,
    double max_bp,
    double r2_threshold
) {
    constexpr double EPS = 1e-10;

    auto bG = G_centered.request();
    auto bV = col_var.request();
    auto bP = chr_pos.request();
    if (bG.ndim != 2) throw std::invalid_argument("G_centered must be 2-D");
    if (bV.ndim != 1) throw std::invalid_argument("col_var must be 1-D");
    if (bP.ndim != 1) throw std::invalid_argument("chr_pos must be 1-D");

    const std::size_t n = static_cast<std::size_t>(bG.shape[0]);
    const std::size_t m = static_cast<std::size_t>(bG.shape[1]);
    if (static_cast<std::size_t>(bV.shape[0]) != m) {
        throw std::invalid_argument("col_var length must equal G.shape[1]");
    }
    if (static_cast<std::size_t>(bP.shape[0]) != m) {
        throw std::invalid_argument("chr_pos length must equal G.shape[1]");
    }
    const std::size_t window = static_cast<std::size_t>(
        std::max<std::int64_t>(window_i64, 0));

    // Allocate the output adjacency matrix on the Python side.
    py::array_t<double> adj({static_cast<py::ssize_t>(m),
                             static_cast<py::ssize_t>(m)});
    auto bA = adj.request();
    double* Ap = static_cast<double*>(bA.ptr);
    std::fill(Ap, Ap + m * m, 0.0);

    if (m == 0 || n == 0) return adj;

    const double* Gp = static_cast<const double*>(bG.ptr);
    const double* Vp = static_cast<const double*>(bV.ptr);
    const std::int64_t* Pp = static_cast<const std::int64_t*>(bP.ptr);

    {
        py::gil_scoped_release release;

        // Pre-extract column views: G is (n, m) row-major, so column i
        // is strided by m. We can compute dot(col_i, col_j) with a
        // simple inner loop over n.
        const double n_d = static_cast<double>(n);

        for (std::size_t i = 0; i < m; ++i) {
            const std::int64_t pos_i = Pp[i];
            const double var_i = Vp[i];
            const std::size_t j_end = std::min(m, i + window + 1);
            for (std::size_t j = i + 1; j < j_end; ++j) {
                if (static_cast<double>(Pp[j] - pos_i) > max_bp) break;
                const double var_j = Vp[j];
                const double denom = var_i * var_j;
                if (!(denom > EPS)) continue;

                // cov = mean(G[:, i] * G[:, j])
                double sum = 0.0;
                for (std::size_t r = 0; r < n; ++r) {
                    sum += Gp[r * m + i] * Gp[r * m + j];
                }
                const double cov = sum / n_d;
                const double r2 = (cov * cov) / denom;
                if (r2 > r2_threshold) {
                    Ap[i * m + j] = r2;
                    Ap[j * m + i] = r2;
                }
            }
        }
    }  // GIL re-acquired

    return adj;
}

}  // namespace

PYBIND11_MODULE(_cc_graph_native, m) {
    m.doc() = "Native C++ accelerator for the windowed r² adjacency build "
              "in detect_blocks_cc_graph.";
    m.def("cc_graph_windowed_adj", &cc_graph_windowed_adj,
          py::arg("G_centered"), py::arg("col_var"), py::arg("chr_pos"),
          py::arg("window"), py::arg("max_bp"), py::arg("r2_threshold"),
          "Build the (m, m) windowed r² adjacency matrix used by cc_graph.");
}
