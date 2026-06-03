// Geyer initial-positive sequence ESS estimator.
//
// C++ port of the inner loop of torchgenomics.pgs.diagnostics.ess. Given a
// per-lag autocorrelation matrix ``acorr_mean`` of shape (N, m) (one column
// per parameter, averaged across chains), compute the per-parameter ESS:
//
//   tau_j = 1 + 2 * sum_{k=1, 3, 5, ...} (acorr[k, j] + acorr[k+1, j])
//   ESS_j = total_samples / max(tau_j, 1)
//
// The walk in k stops as soon as the pair sum becomes <= 0 (Geyer's
// initial positive sequence rule).
//
// Exposed via pybind11:
//
//   geyer_initial_positive_ess(acorr_mean, total_samples) -> ndarray (m,)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

namespace {

void run_geyer_ess(
    const double* acorr,   // (N, m) row-major
    std::size_t N,
    std::size_t m,
    double total_samples,
    double* out            // (m,)
) {
    for (std::size_t j = 0; j < m; ++j) {
        double tau = 1.0;
        std::size_t k = 1;
        while (k + 1 < N) {
            const double pair = acorr[k * m + j] + acorr[(k + 1) * m + j];
            if (pair <= 0.0) break;
            tau += 2.0 * pair;
            k += 2;
        }
        out[j] = total_samples / std::max(tau, 1.0);
    }
}

py::array_t<double> py_geyer_ess(
    py::array_t<double, py::array::c_style | py::array::forcecast> acorr_mean,
    double total_samples
) {
    auto buf = acorr_mean.request();
    if (buf.ndim != 2) throw std::invalid_argument("acorr_mean must be 2-D (N, m)");
    const std::size_t N = static_cast<std::size_t>(buf.shape[0]);
    const std::size_t m = static_cast<std::size_t>(buf.shape[1]);

    py::array_t<double> out(static_cast<py::ssize_t>(m));
    auto out_buf = out.request();
    {
        py::gil_scoped_release release;
        run_geyer_ess(
            static_cast<const double*>(buf.ptr),
            N, m, total_samples,
            static_cast<double*>(out_buf.ptr)
        );
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_ess_native, m) {
    m.doc() = "Native C++ accelerator for the Geyer initial-positive ESS loop.";
    m.def("geyer_initial_positive_ess", &py_geyer_ess,
          py::arg("acorr_mean"),
          py::arg("total_samples"),
          R"pbdoc(
            Per-parameter Geyer initial-positive sequence ESS.

            ``acorr_mean`` is the (N, m) chain-averaged per-lag
            autocorrelation. ``total_samples`` is the M*N denominator.
            Returns a length-m float64 ndarray of ESS values.
          )pbdoc");
}
