// Native accelerator for ``torchgwas.ld._graph_utils.greedy_mwis``.
//
// The pure-Python loop sorts intervals by weight descending and then walks
// the ranked list, checking each candidate against every previously
// selected interval for overlap. For n intervals that is O(n^2) Python
// tuple unpacking + branch work; on dense Big-LD candidate lists with a
// few thousand intervals it shows up in the per-chromosome path.
//
// Exposed via pybind11:
//
//   greedy_mwis(starts, ends, weights)
//       -> (selected_starts, selected_ends, selected_weights)
//
// Behavior matches the Python reference exactly: same descending-weight
// sort with std::stable_sort to preserve the input order on ties (Python's
// `sorted(intervals, key=lambda x: -x[2])` is stable), same closed-interval
// overlap test (`start <= s_end and end >= s_start`), and the returned
// arrays are ordered by ascending start.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

py::tuple greedy_mwis(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> starts,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ends,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights
) {
    if (starts.ndim() != 1 || ends.ndim() != 1 || weights.ndim() != 1) {
        throw std::invalid_argument("starts, ends, weights must be 1-D");
    }
    const std::size_t n = static_cast<std::size_t>(starts.shape(0));
    if (static_cast<std::size_t>(ends.shape(0)) != n ||
        static_cast<std::size_t>(weights.shape(0)) != n) {
        throw std::invalid_argument("starts, ends, weights length mismatch");
    }

    const std::int64_t* Sp = starts.data();
    const std::int64_t* Ep = ends.data();
    const double* Wp = weights.data();

    std::vector<std::int64_t> sel_s, sel_e;
    std::vector<double> sel_w;

    if (n > 0) {
        py::gil_scoped_release release;

        // Stable-sort indices by weight descending. Stable to mirror Python's
        // `sorted(..., key=lambda x: -x[2])`, which preserves input order on
        // ties.
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) {
                             return Wp[a] > Wp[b];
                         });

        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t idx = order[k];
            const std::int64_t s = Sp[idx];
            const std::int64_t e = Ep[idx];
            bool overlaps = false;
            for (std::size_t j = 0; j < sel_s.size(); ++j) {
                if (s <= sel_e[j] && e >= sel_s[j]) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                sel_s.push_back(s);
                sel_e.push_back(e);
                sel_w.push_back(Wp[idx]);
            }
        }

        // Sort selected by ascending start, carrying ends and weights along.
        std::vector<std::size_t> ord2(sel_s.size());
        for (std::size_t i = 0; i < ord2.size(); ++i) ord2[i] = i;
        std::stable_sort(ord2.begin(), ord2.end(),
                         [&](std::size_t a, std::size_t b) {
                             return sel_s[a] < sel_s[b];
                         });
        std::vector<std::int64_t> ts(sel_s.size()), te(sel_s.size());
        std::vector<double> tw(sel_s.size());
        for (std::size_t i = 0; i < ord2.size(); ++i) {
            ts[i] = sel_s[ord2[i]];
            te[i] = sel_e[ord2[i]];
            tw[i] = sel_w[ord2[i]];
        }
        sel_s.swap(ts);
        sel_e.swap(te);
        sel_w.swap(tw);
    }  // GIL re-acquired

    auto out_s = py::array_t<std::int64_t>(static_cast<py::ssize_t>(sel_s.size()));
    auto out_e = py::array_t<std::int64_t>(static_cast<py::ssize_t>(sel_e.size()));
    auto out_w = py::array_t<double>(static_cast<py::ssize_t>(sel_w.size()));
    std::int64_t* osp = out_s.mutable_data();
    std::int64_t* oep = out_e.mutable_data();
    double* owp = out_w.mutable_data();
    for (std::size_t i = 0; i < sel_s.size(); ++i) {
        osp[i] = sel_s[i];
        oep[i] = sel_e[i];
        owp[i] = sel_w[i];
    }
    return py::make_tuple(out_s, out_e, out_w);
}

}  // namespace

PYBIND11_MODULE(_greedy_mwis_native, m) {
    m.doc() = "Native C++ accelerator for greedy interval-graph MWIS.";
    m.def("greedy_mwis", &greedy_mwis,
          py::arg("starts"),
          py::arg("ends"),
          py::arg("weights"),
          "Greedy maximum-weight independent set over closed integer intervals.");
}
