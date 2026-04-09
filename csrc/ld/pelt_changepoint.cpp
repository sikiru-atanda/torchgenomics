// PELT (Pruned Exact Linear Time) change-point detection.
//
// C++ port of torchgwas.ld._changepoint.dp_changepoint. The pure-Python
// implementation is the canonical reference; this module mirrors it
// step-for-step using std::vector and a packed candidate list.
//
// Cost models:
//   - "gaussian": sum of squared deviations from segment mean
//                 = seg_sumsq - length * mean^2
//   - "poisson":  -loglik (up to constant)
//                 = length * mean - seg_sum * log(mean)
//
// Exposed via pybind11:
//
//   pelt_dp_changepoint(signal, penalty, min_seg, cost_fn) -> list[int]
//
// Returns the sorted list of change-point indices (excluding 0 and n).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

enum class CostModel { Gaussian, Poisson };

inline double segment_cost(
    const double* cumsum,
    const double* cumsumsq,
    std::size_t start,
    std::size_t end,
    CostModel model
) {
    const std::size_t length = end - start;
    if (length == 0) return 0.0;
    const double seg_sum = cumsum[end] - cumsum[start];
    if (model == CostModel::Gaussian) {
        const double seg_sumsq = cumsumsq[end] - cumsumsq[start];
        const double mean_val = seg_sum / static_cast<double>(length);
        return seg_sumsq - static_cast<double>(length) * mean_val * mean_val;
    } else {
        const double seg_mean = std::max(seg_sum / static_cast<double>(length), 1e-10);
        return static_cast<double>(length) * seg_mean - seg_sum * std::log(seg_mean);
    }
}

std::vector<std::int64_t> run_pelt(
    const double* signal,
    std::size_t n,
    double penalty,
    std::size_t min_seg,
    CostModel model
) {
    if (n < 2 * min_seg) return {};

    std::vector<double> cumsum(n + 1, 0.0);
    std::vector<double> cumsumsq(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        cumsum[i + 1] = cumsum[i] + signal[i];
        cumsumsq[i + 1] = cumsumsq[i] + signal[i] * signal[i];
    }

    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> opt(n + 1, 0.0);
    std::vector<std::size_t> last_cp(n + 1, 0);

    // Candidate list: vector of size_t with a parallel "alive" mask
    // would also work, but a simple vector with rebuild-on-prune is
    // simpler and matches the Python set semantics exactly.
    std::vector<std::size_t> candidates;
    candidates.reserve(n + 1);
    candidates.push_back(0);

    std::vector<std::size_t> next_candidates;
    next_candidates.reserve(n + 1);

    // Per-iteration scratch flag for "to be pruned".
    std::vector<char> prune_flag;

    for (std::size_t j = min_seg; j <= n; ++j) {
        double best_cost = INF;
        std::size_t best_t = 0;
        prune_flag.assign(candidates.size(), 0);

        // Single-pass scan that mirrors the Python loop: as we walk
        // candidates we update the running best, and we mark for pruning
        // every t whose lower-bound cost already exceeds the running
        // best plus penalty (the same condition as the Python source).
        for (std::size_t k = 0; k < candidates.size(); ++k) {
            const std::size_t t = candidates[k];
            if (j - t < min_seg) continue;
            const double sc = segment_cost(cumsum.data(), cumsumsq.data(), t, j, model);
            const double cost = opt[t] + sc + penalty;
            if (cost < best_cost) {
                best_cost = cost;
                best_t = t;
            }
            if (opt[t] + sc > best_cost + penalty) {
                prune_flag[k] = 1;
            }
        }

        opt[j] = best_cost;
        last_cp[j] = best_t;

        next_candidates.clear();
        for (std::size_t k = 0; k < candidates.size(); ++k) {
            if (!prune_flag[k]) next_candidates.push_back(candidates[k]);
        }
        next_candidates.push_back(j);
        candidates.swap(next_candidates);
    }

    // Backtrack
    std::vector<std::int64_t> cps;
    std::size_t pos = n;
    while (pos > 0) {
        const std::size_t cp = last_cp[pos];
        if (cp > 0) cps.push_back(static_cast<std::int64_t>(cp));
        pos = cp;
    }
    std::sort(cps.begin(), cps.end());
    return cps;
}

py::list py_pelt_dp_changepoint(
    py::array_t<double, py::array::c_style | py::array::forcecast> signal,
    double penalty,
    int min_seg,
    const std::string& cost_fn
) {
    auto buf = signal.request();
    if (buf.ndim != 1) throw std::invalid_argument("signal must be 1-D");
    if (min_seg < 1) throw std::invalid_argument("min_seg must be >= 1");

    CostModel model;
    if (cost_fn == "gaussian") model = CostModel::Gaussian;
    else if (cost_fn == "poisson") model = CostModel::Poisson;
    else throw std::invalid_argument("Unknown cost_fn: " + cost_fn);

    const std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    const double* sig_ptr = static_cast<const double*>(buf.ptr);

    std::vector<std::int64_t> cps;
    {
        py::gil_scoped_release release;
        cps = run_pelt(sig_ptr, n, penalty, static_cast<std::size_t>(min_seg), model);
    }
    py::list out;
    for (std::int64_t c : cps) out.append(static_cast<int>(c));
    return out;
}

}  // namespace

PYBIND11_MODULE(_pelt_native, m) {
    m.doc() = "Native C++ accelerator for PELT change-point detection.";
    m.def("pelt_dp_changepoint", &py_pelt_dp_changepoint,
          py::arg("signal"),
          py::arg("penalty"),
          py::arg("min_seg") = 2,
          py::arg("cost_fn") = "gaussian",
          "Exact DP segmentation with PELT pruning. Returns sorted change-point indices.");
}
