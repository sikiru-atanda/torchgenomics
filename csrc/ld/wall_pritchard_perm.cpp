// Native accelerator for the Wall & Pritchard blockiness permutation
// p-value loop in
// ``torchgwas.ld._blocks_diagnostics.compute_wall_pritchard_diagnostics``.
//
// The Python reference loop is:
//
//   for _ in range(n_permutations):
//       perm = torch.randperm(n_pairs)
//       dp_perm = dp[perm]
//       Q_perm = (dp_perm > thr).float().mean().item()
//       r2_perm = r2[perm]
//       adj_mean_perm = r2_perm[is_adjacent].mean().item()
//       all_mean_perm = r2_perm.mean().item()
//       Q_adj_perm = min(adj_mean_perm / max(all_mean_perm, eps), 1.0)
//       if Q_perm * Q_adj_perm >= blockiness_obs:
//           perm_count += 1
//
// Each iteration does four `.item()` round-trips. The C++ port runs the
// whole loop with a single GIL release and a Mersenne-Twister `shuffle`,
// then returns the integer perm_count so the Python caller can compute
// the (perm_count + 1) / (n_permutations + 1) p-value with identical
// semantics to the reference.
//
// Exposed via pybind11:
//
//   wall_pritchard_perm(dp, r2, is_adjacent, d_prime_threshold,
//                       blockiness_obs, n_permutations, seed)
//       -> int perm_count

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double EPS = 1e-10;

int wall_pritchard_perm(
    py::array_t<double, py::array::c_style | py::array::forcecast> dp,
    py::array_t<double, py::array::c_style | py::array::forcecast> r2,
    py::array_t<bool,   py::array::c_style | py::array::forcecast> is_adjacent,
    double d_prime_threshold,
    double blockiness_obs,
    int n_permutations,
    std::uint64_t seed
) {
    if (dp.ndim() != 1 || r2.ndim() != 1 || is_adjacent.ndim() != 1) {
        throw std::invalid_argument("dp, r2, is_adjacent must be 1-D");
    }
    const std::size_t P = static_cast<std::size_t>(dp.shape(0));
    if (static_cast<std::size_t>(r2.shape(0)) != P ||
        static_cast<std::size_t>(is_adjacent.shape(0)) != P) {
        throw std::invalid_argument("dp, r2, is_adjacent length mismatch");
    }
    if (n_permutations < 0) {
        throw std::invalid_argument("n_permutations must be >= 0");
    }

    const double* DPp = dp.data();
    const double* R2p = r2.data();
    const bool*   ADp = is_adjacent.data();

    int perm_count = 0;
    {
        py::gil_scoped_release release;

        if (P == 0 || n_permutations == 0) {
            // Match Python: perm_count stays 0 (loop body never executes).
        } else {
            // Permutation = shuffle of [0..P). Equivalent to drawing
            // perm = randperm(P) and indexing both dp and r2.
            std::vector<std::size_t> perm(P);
            std::iota(perm.begin(), perm.end(), 0);
            std::mt19937_64 rng(seed);

            // Count adjacent positions once — invariant across permutations.
            std::size_t n_adj = 0;
            for (std::size_t k = 0; k < P; ++k) {
                if (ADp[k]) ++n_adj;
            }

            for (int iter = 0; iter < n_permutations; ++iter) {
                // In-place Fisher-Yates shuffle.
                for (std::size_t k = P - 1; k > 0; --k) {
                    std::uniform_int_distribution<std::size_t> dist(0, k);
                    const std::size_t s = dist(rng);
                    std::swap(perm[k], perm[s]);
                }

                // Compute Q_perm, adj_mean_perm, all_mean_perm in one pass.
                std::size_t q_count = 0;
                double r2_sum = 0.0;
                double r2_adj_sum = 0.0;
                for (std::size_t k = 0; k < P; ++k) {
                    const std::size_t src = perm[k];
                    const double dpv = DPp[src];
                    const double r2v = R2p[src];
                    if (dpv > d_prime_threshold) ++q_count;
                    r2_sum += r2v;
                    if (ADp[k]) {
                        r2_adj_sum += r2v;
                    }
                }
                const double Q_perm = static_cast<double>(q_count) /
                                      static_cast<double>(P);
                const double all_mean_perm = r2_sum / static_cast<double>(P);
                const double adj_mean_perm = (n_adj > 0)
                    ? r2_adj_sum / static_cast<double>(n_adj)
                    : 0.0;
                double q_adj_perm =
                    adj_mean_perm / std::max(all_mean_perm, EPS);
                if (q_adj_perm > 1.0) q_adj_perm = 1.0;
                if (Q_perm * q_adj_perm >= blockiness_obs) {
                    ++perm_count;
                }
            }
        }
    }  // GIL re-acquired

    return perm_count;
}

}  // namespace

PYBIND11_MODULE(_wall_pritchard_native, m) {
    m.doc() = "Native C++ accelerator for Wall-Pritchard blockiness permutation test.";
    m.def("wall_pritchard_perm", &wall_pritchard_perm,
          py::arg("dp"),
          py::arg("r2"),
          py::arg("is_adjacent"),
          py::arg("d_prime_threshold"),
          py::arg("blockiness_obs"),
          py::arg("n_permutations"),
          py::arg("seed"),
          "Permutation count >= observed blockiness for the Wall-Pritchard "
          "blockiness diagnostic.");
}
