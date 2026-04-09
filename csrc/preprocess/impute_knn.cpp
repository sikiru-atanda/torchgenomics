// Native accelerator for ``torchgwas.preprocess.impute.impute_knn``.
//
// The Python reference iterates ``len(miss)`` missing entries and for each
// runs a small fixed-size walk over ``k`` neighbours. At realistic dosage
// matrix sizes the per-entry ``.item()`` round-trips dominate the wall clock
// even though every iteration is O(k).
//
// Inputs (all numpy float64 / int64, C-contiguous):
//   G_in       (n, m) float64 — original dosage matrix (read-only). NaN
//                                 marks missing. Neighbours are read from
//                                 here so the imputation order does not
//                                 affect the result (matches the Python
//                                 reference, which loops on the original
//                                 ``G`` rather than the running ``G_out``).
//   G_out      (n, m) float64 — working buffer; NaN positions in ``G_in``
//                                 are filled in place. Caller pre-clones
//                                 G_in into G_out.
//   knn_idx    (n, k) int64   — top-k neighbour indices per sample
//                                 (already excludes self).
//   K          (n, n) float64 — kinship / similarity matrix.
//   col_means  (m,)   float64 — per-column observed (non-NaN) mean of G,
//                                 0.0 for all-missing columns. Used as the
//                                 fallback when *all* k neighbours at a SNP
//                                 are also NaN.
//
// Output: G_out is modified in place. Returns Py_None.
//
// Semantics mirror the Python loop in
// ``torchgwas/preprocess/impute.py::impute_knn`` line-for-line:
//   1. valid = ~isnan(G[neighbours, j])
//   2. if any valid:
//        w = clamp(K[i, neighbours][valid], min=0)
//        if w.sum() > 0:  G[i,j] = (vals*w).sum() / w.sum()
//        else:             G[i,j] = vals.mean()
//      else:
//        G[i,j] = col_means[j]   (precomputed by caller)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

void impute_knn_fill(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> G_out,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> knn_idx,
    py::array_t<double, py::array::c_style | py::array::forcecast> K,
    py::array_t<double, py::array::c_style | py::array::forcecast> col_means
) {
    if (G_in.ndim() != 2) throw std::invalid_argument("G_in must be 2-D");
    if (G_out.ndim() != 2) throw std::invalid_argument("G_out must be 2-D");
    if (G_in.shape(0) != G_out.shape(0) || G_in.shape(1) != G_out.shape(1)) {
        throw std::invalid_argument("G_in and G_out must have the same shape");
    }
    if (knn_idx.ndim() != 2) throw std::invalid_argument("knn_idx must be 2-D");
    if (K.ndim() != 2) throw std::invalid_argument("K must be 2-D");
    if (col_means.ndim() != 1) throw std::invalid_argument("col_means must be 1-D");

    const std::size_t n = static_cast<std::size_t>(G_out.shape(0));
    const std::size_t m = static_cast<std::size_t>(G_out.shape(1));
    const std::size_t k = static_cast<std::size_t>(knn_idx.shape(1));

    if (static_cast<std::size_t>(knn_idx.shape(0)) != n) {
        throw std::invalid_argument("knn_idx rows must equal G_out rows");
    }
    if (static_cast<std::size_t>(K.shape(0)) != n ||
        static_cast<std::size_t>(K.shape(1)) != n) {
        throw std::invalid_argument("K must be (n, n)");
    }
    if (static_cast<std::size_t>(col_means.shape(0)) != m) {
        throw std::invalid_argument("col_means must have length m");
    }

    const double* gin = G_in.data();
    double* gout = G_out.mutable_data();
    const std::int64_t* kp = knn_idx.data();
    const double* Kp = K.data();
    const double* cmp_ = col_means.data();

    {
        py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
            const std::size_t i = static_cast<std::size_t>(ii);
            const std::int64_t* nbrs = kp + i * k;
            const double* Krow = Kp + i * n;

            for (std::size_t j = 0; j < m; ++j) {
                double v = gin[i * m + j];
                if (!std::isnan(v)) continue;

                double wsum = 0.0;
                double wval = 0.0;     // weighted-by-K accumulator
                double vsum = 0.0;     // unweighted accumulator (fallback)
                std::size_t nvalid = 0;

                for (std::size_t t = 0; t < k; ++t) {
                    std::int64_t nb = nbrs[t];
                    if (nb < 0 || static_cast<std::size_t>(nb) >= n) continue;
                    double nv = gin[static_cast<std::size_t>(nb) * m + j];
                    if (std::isnan(nv)) continue;

                    double w = Krow[static_cast<std::size_t>(nb)];
                    if (w < 0.0) w = 0.0;

                    wsum += w;
                    wval += w * nv;
                    vsum += nv;
                    ++nvalid;
                }

                if (nvalid == 0) {
                    gout[i * m + j] = cmp_[j];
                } else if (wsum > 0.0) {
                    gout[i * m + j] = wval / wsum;
                } else {
                    gout[i * m + j] = vsum / static_cast<double>(nvalid);
                }
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_impute_knn_native, mod) {
    mod.doc() = "Native C++ accelerator for KNN imputation per-missing-entry loop.";
    mod.def("impute_knn_fill", &impute_knn_fill,
            py::arg("G_in"),
            py::arg("G_out"),
            py::arg("knn_idx"),
            py::arg("K"),
            py::arg("col_means"),
            "Fill NaN positions of G_out in place using precomputed KNN "
            "indices, reading neighbour values from G_in.");
}
