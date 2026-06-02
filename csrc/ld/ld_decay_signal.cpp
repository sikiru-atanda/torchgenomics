// Native accelerator for ``torchgenomics.ld._changepoint.ld_decay_signal``.
//
// The pure-Python loop walks every variant v and builds boolean masks over
// the full (P,) pair tensor to find pairs touching v, then runs torch.topk
// on the masked slice. For m variants and P adjacent-pair entries that is
// O(m * P) materialized boolean tensors plus a torch launch per variant.
// We replace it with a single C++ pass: bucket pair indices by endpoint,
// then for each variant maintain a max-heap of size k_neighbors keyed by
// pair distance.
//
// Exposed via pybind11:
//
//   ld_decay_signal(idx_i, idx_j, r2_pairs, n_variants, k_neighbors)
//       -> ndarray (n_variants,) float64
//
// Output matches the Python reference: per variant, the mean r² over the
// k_neighbors pairs with the smallest |idx_j - idx_i| (or fewer if the
// variant participates in fewer pairs). Variants touching no pair return 0.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

py::array_t<double> ld_decay_signal(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx_i,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx_j,
    py::array_t<double, py::array::c_style | py::array::forcecast> r2_pairs,
    std::int64_t n_variants_i64,
    std::int64_t k_neighbors_i64
) {
    if (idx_i.ndim() != 1 || idx_j.ndim() != 1 || r2_pairs.ndim() != 1) {
        throw std::invalid_argument("idx_i, idx_j, r2_pairs must be 1-D");
    }
    const std::size_t P = static_cast<std::size_t>(idx_i.shape(0));
    if (static_cast<std::size_t>(idx_j.shape(0)) != P ||
        static_cast<std::size_t>(r2_pairs.shape(0)) != P) {
        throw std::invalid_argument("idx_i, idx_j, r2_pairs length mismatch");
    }
    const std::size_t m = static_cast<std::size_t>(n_variants_i64);
    const std::size_t k = static_cast<std::size_t>(
        k_neighbors_i64 > 0 ? k_neighbors_i64 : 0);

    auto out = py::array_t<double>(static_cast<py::ssize_t>(m));
    double* out_p = out.mutable_data();
    for (std::size_t v = 0; v < m; ++v) out_p[v] = 0.0;
    if (k == 0 || P == 0 || m == 0) return out;

    const std::int64_t* Ip = idx_i.data();
    const std::int64_t* Jp = idx_j.data();
    const double* Rp = r2_pairs.data();

    {
        py::gil_scoped_release release;

        // Bucket pair indices by both endpoints (a pair touches its i and j).
        std::vector<std::vector<std::size_t>> by_var(m);
        for (std::size_t p = 0; p < P; ++p) {
            const std::int64_t ii = Ip[p];
            const std::int64_t jj = Jp[p];
            if (ii < 0 || jj < 0) continue;
            const std::size_t uii = static_cast<std::size_t>(ii);
            const std::size_t ujj = static_cast<std::size_t>(jj);
            if (uii < m) by_var[uii].push_back(p);
            if (ujj < m && ujj != uii) by_var[ujj].push_back(p);
        }

        // Per variant, stable-sort the touching pairs by index distance and
        // take the first k (matches torch.topk(largest=False) which breaks
        // ties by lowest source index, and our buckets preserve the original
        // pair-table order).
        std::vector<std::pair<std::int64_t, std::size_t>> scratch;
        for (std::size_t v = 0; v < m; ++v) {
            const auto& bucket = by_var[v];
            if (bucket.empty()) continue;
            scratch.clear();
            scratch.reserve(bucket.size());
            for (std::size_t p : bucket) {
                const std::int64_t d = std::llabs(Jp[p] - Ip[p]);
                scratch.emplace_back(d, p);
            }
            std::stable_sort(
                scratch.begin(), scratch.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            const std::size_t kept = std::min(k, scratch.size());
            double sum = 0.0;
            for (std::size_t i = 0; i < kept; ++i) sum += Rp[scratch[i].second];
            out_p[v] = sum / static_cast<double>(kept);
        }
    }  // GIL re-acquired

    return out;
}

}  // namespace

PYBIND11_MODULE(_ld_decay_signal_native, m) {
    m.doc() = "Native C++ accelerator for the ld_decay_signal per-variant loop.";
    m.def("ld_decay_signal", &ld_decay_signal,
          py::arg("idx_i"),
          py::arg("idx_j"),
          py::arg("r2_pairs"),
          py::arg("n_variants"),
          py::arg("k_neighbors"),
          "Per-variant mean r² over k nearest pair neighbors.");
}
