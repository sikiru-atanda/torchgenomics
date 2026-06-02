// Native accelerator for ``torchgenomics.ld._blocks_novel.detect_blocks_gwas_aligned``.
//
// The pure-Python loop runs an O(m * K) DP where each candidate (i, j) calls
// ``torch.linalg.eigvalsh(R[i:j, i:j])`` and ``.item()`` round-trips both the
// concentration ratio (lead / trace) and the log-condition-number penalty.
// For m in the thousands with K = max_block_snps = 50 this dominates wall
// time. We replace the inner cost with a row-major C++ Jacobi eigensolver
// over the (j - i)-sized submatrix and run the entire DP + backtrack inside
// a single GIL-released C++ call.
//
// Exposed via pybind11:
//
//   gwas_aligned_dp(R, chr_pos, max_block_snps, condition_penalty,
//                   min_block_snps, max_bp)
//       -> list[tuple[int, int]]            # (start_local, end_local) inclusive
//
// Behavior matches the Python reference exactly: same DP recurrence, same
// backtrack rule, same concentration / penalty formulas.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double EPS = 1e-10;

// Compute eigenvalues of a symmetric n x n matrix via Jacobi rotations.
// `A` is row-major scratch (overwritten); writes the n eigenvalues to `out`.
// Returned eigenvalues are sorted in ascending order to match
// torch.linalg.eigvalsh.
void jacobi_eigvalsh(double* A, std::size_t n, double* out) {
    if (n == 0) return;
    if (n == 1) {
        out[0] = A[0];
        return;
    }

    constexpr int max_sweeps = 60;
    const double tol = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // Off-diagonal Frobenius norm
        double off = 0.0;
        for (std::size_t p = 0; p < n - 1; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double v = A[p * n + q];
                off += v * v;
            }
        }
        if (off < tol) break;

        for (std::size_t p = 0; p < n - 1; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = A[p * n + q];
                if (std::fabs(apq) < 1e-18) continue;
                const double app = A[p * n + p];
                const double aqq = A[q * n + q];
                const double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    const double sgn = (theta >= 0.0) ? 1.0 : -1.0;
                    t = sgn / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                A[p * n + p] = app - t * apq;
                A[q * n + q] = aqq + t * apq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;

                for (std::size_t i = 0; i < n; ++i) {
                    if (i == p || i == q) continue;
                    const double aip = A[i * n + p];
                    const double aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[p * n + i] = A[i * n + p];
                    A[i * n + q] = s * aip + c * aiq;
                    A[q * n + i] = A[i * n + q];
                }
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) out[i] = A[i * n + i];
    std::sort(out, out + n);
}

py::list gwas_aligned_dp(
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_pos,
    std::int64_t max_block_snps_i64,
    double condition_penalty,
    std::int64_t min_block_snps_i64,
    double max_bp
) {
    if (R.ndim() != 2 || R.shape(0) != R.shape(1)) {
        throw std::invalid_argument("R must be a square 2-D array");
    }
    if (chr_pos.ndim() != 1) {
        throw std::invalid_argument("chr_pos must be 1-D");
    }
    const std::size_t m = static_cast<std::size_t>(R.shape(0));
    if (static_cast<std::size_t>(chr_pos.shape(0)) != m) {
        throw std::invalid_argument("chr_pos length must match R dimension");
    }
    if (max_block_snps_i64 <= 0 || min_block_snps_i64 <= 0) {
        throw std::invalid_argument("block size limits must be positive");
    }
    const std::size_t max_block = static_cast<std::size_t>(max_block_snps_i64);
    const std::size_t min_block = static_cast<std::size_t>(min_block_snps_i64);

    py::list out;
    if (m < min_block) return out;

    const double* Rp = R.data();
    const std::int64_t* Pp = chr_pos.data();

    std::vector<std::tuple<int, int>> blocks;

    {
        py::gil_scoped_release release;

        const double NEG_INF = -std::numeric_limits<double>::infinity();
        std::vector<double> opt(m + 1, NEG_INF);
        std::vector<std::size_t> backtrack(m + 1, 0);
        opt[0] = 0.0;

        std::vector<double> sub(max_block * max_block);
        std::vector<double> evals(max_block);

        for (std::size_t j = min_block; j <= m; ++j) {
            const std::size_t i_lo = (j > max_block) ? (j - max_block) : 0;
            const std::size_t i_hi = j - min_block + 1;  // exclusive
            for (std::size_t i = i_lo; i < i_hi; ++i) {
                if (static_cast<double>(Pp[j - 1] - Pp[i]) > max_bp) continue;
                if (opt[i] == NEG_INF) continue;

                const std::size_t b = j - i;
                // Copy R[i:j, i:j] into row-major scratch
                for (std::size_t r = 0; r < b; ++r) {
                    const double* src = Rp + (i + r) * m + i;
                    double* dst = sub.data() + r * b;
                    for (std::size_t c = 0; c < b; ++c) dst[c] = src[c];
                }
                jacobi_eigvalsh(sub.data(), b, evals.data());

                double trace_val = 0.0;
                for (std::size_t k = 0; k < b; ++k) trace_val += evals[k];
                if (trace_val < EPS) trace_val = EPS;
                const double leading = evals[b - 1];
                const double concentration = leading / trace_val;

                double min_eig = evals[0];
                if (min_eig < EPS) min_eig = EPS;
                double cond_num = leading / min_eig;
                if (cond_num < 1.0) cond_num = 1.0;
                const double penalty = condition_penalty * std::log(cond_num);

                const double score = opt[i] + concentration - penalty;
                if (score > opt[j]) {
                    opt[j] = score;
                    backtrack[j] = i;
                }
            }
        }

        // Backtrack
        std::size_t pos = m;
        while (pos > 0 && opt[pos] == NEG_INF) --pos;
        std::vector<std::tuple<int, int>> rev;
        while (pos > 0) {
            const std::size_t start = backtrack[pos];
            if (pos - start >= min_block) {
                rev.emplace_back(static_cast<int>(start),
                                 static_cast<int>(pos - 1));
            }
            if (start == pos) break;  // safety
            pos = start;
        }
        blocks.assign(rev.rbegin(), rev.rend());
    }  // GIL re-acquired

    for (const auto& blk : blocks) {
        out.append(py::make_tuple(std::get<0>(blk), std::get<1>(blk)));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_gwas_aligned_native, m) {
    m.doc() = "Native C++ accelerator for the GWAS-aligned LD block DP.";
    m.def("gwas_aligned_dp", &gwas_aligned_dp,
          py::arg("R"),
          py::arg("chr_pos"),
          py::arg("max_block_snps"),
          py::arg("condition_penalty"),
          py::arg("min_block_snps"),
          py::arg("max_bp"),
          "Run the GWAS-aligned DP + backtrack on a single chromosome's R "
          "matrix and return the chosen (start_local, end_local) blocks.");
}
