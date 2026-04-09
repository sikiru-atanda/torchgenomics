// Connected-components BFS on a dense, thresholded adjacency matrix.
//
// C++ port of torchgwas.ld._graph_utils.connected_components. The pure-
// Python implementation is the canonical reference; this module mirrors
// it (BFS order, sorted output) but eliminates the per-edge tensor
// indexing that dominates the Python loop on m in the thousands.
//
// Exposed via pybind11:
//
//   connected_components(A, threshold) -> list[list[int]]
//
// Each inner list contains sorted vertex indices belonging to one
// component; components are returned sorted by their first vertex.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<std::vector<std::int64_t>> run_connected_components(
    const double* A,    // (m, m) row-major
    std::size_t m,
    double threshold
) {
    std::vector<char> visited(m, 0);
    std::vector<std::vector<std::int64_t>> components;

    for (std::size_t start = 0; start < m; ++start) {
        if (visited[start]) continue;
        std::vector<std::int64_t> component;
        std::queue<std::size_t> queue;
        queue.push(start);
        visited[start] = 1;
        while (!queue.empty()) {
            const std::size_t node = queue.front();
            queue.pop();
            component.push_back(static_cast<std::int64_t>(node));
            const double* row = A + node * m;
            for (std::size_t neighbor = 0; neighbor < m; ++neighbor) {
                if (!visited[neighbor] && row[neighbor] > threshold) {
                    visited[neighbor] = 1;
                    queue.push(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }

    std::sort(components.begin(), components.end(),
              [](const std::vector<std::int64_t>& a,
                 const std::vector<std::int64_t>& b) {
                  return a.front() < b.front();
              });
    return components;
}

py::list py_connected_components(
    py::array_t<double, py::array::c_style | py::array::forcecast> A,
    double threshold
) {
    auto buf = A.request();
    if (buf.ndim != 2) throw std::invalid_argument("A must be 2-D");
    if (buf.shape[0] != buf.shape[1]) {
        throw std::invalid_argument("A must be square");
    }
    const std::size_t m = static_cast<std::size_t>(buf.shape[0]);

    std::vector<std::vector<std::int64_t>> components;
    {
        py::gil_scoped_release release;
        components = run_connected_components(
            static_cast<const double*>(buf.ptr), m, threshold
        );
    }

    py::list out;
    for (const auto& comp : components) {
        py::list inner;
        for (std::int64_t v : comp) inner.append(static_cast<int>(v));
        out.append(inner);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_graph_native, m) {
    m.doc() = "Native C++ accelerator for dense graph utilities (connected components).";
    m.def("connected_components", &py_connected_components,
          py::arg("A"), py::arg("threshold") = 0.0,
          "Connected-components BFS on a dense thresholded adjacency matrix.");
}
