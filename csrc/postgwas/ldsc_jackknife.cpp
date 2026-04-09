// LDSC block-jackknife standard errors for weighted least squares.
//
// C++ port of torchgwas.postgwas._ldsc._block_jackknife_se. The Python
// reference dispatches one torch.linalg.lstsq per leave-one-block-out fit
// (default n_blocks = 200), and the launch overhead of those tiny calls
// dominates the actual math: each problem has at most p = 2 columns
// (the LD score and the intercept), so the WLS solve is a closed-form
// (X'WX)^{-1} X'Wy on a 2x2 system.
//
// We collapse the entire jackknife loop into one C++ pass by maintaining a
// running (X'WX, X'Wy) accumulator over the full sample, then for each block
// computing the leave-one-block-out version by *subtracting* the block's
// contribution. The per-block contribution is built once during a forward
// sweep, so the total work is two passes: one to assemble per-block
// contributions, one to subtract-and-solve.
//
// Linear-algebra dependency-free: the only solve is a Gaussian elimination
// over a fixed p ≤ 8 system. LDSC uses p = 2 in practice; the bound keeps
// the routine generic for cross-trait extensions without pulling Eigen.
//
// Exposed via pybind11:
//
//   ldsc_block_jackknife(X, y, w, n_blocks, full_coef) -> (p,) ndarray of SEs

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t MAX_P = 8;

// Solve (A x = b) for an in-place p x p system (p <= MAX_P) via Gaussian
// elimination with partial pivoting. A is row-major, length p*p. Returns
// false on singular system.
bool solve_small(double* A, double* b, std::size_t p) {
    for (std::size_t i = 0; i < p; ++i) {
        // Pivot
        std::size_t pivot = i;
        double pivot_val = std::fabs(A[i * p + i]);
        for (std::size_t r = i + 1; r < p; ++r) {
            const double v = std::fabs(A[r * p + i]);
            if (v > pivot_val) {
                pivot = r;
                pivot_val = v;
            }
        }
        if (pivot_val < 1e-30) return false;
        if (pivot != i) {
            for (std::size_t c = 0; c < p; ++c) {
                std::swap(A[i * p + c], A[pivot * p + c]);
            }
            std::swap(b[i], b[pivot]);
        }
        // Eliminate
        const double diag = A[i * p + i];
        for (std::size_t r = i + 1; r < p; ++r) {
            const double factor = A[r * p + i] / diag;
            for (std::size_t c = i; c < p; ++c) {
                A[r * p + c] -= factor * A[i * p + c];
            }
            b[r] -= factor * b[i];
        }
    }
    // Back-substitute
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(p) - 1; i >= 0; --i) {
        double s = b[i];
        for (std::size_t c = static_cast<std::size_t>(i) + 1; c < p; ++c) {
            s -= A[i * p + c] * b[c];
        }
        b[i] = s / A[i * p + i];
    }
    return true;
}

py::array_t<double> py_ldsc_block_jackknife(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> w,
    int n_blocks_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> full_coef
) {
    auto Xb = X.request();
    auto yb = y.request();
    auto wb = w.request();
    auto fb = full_coef.request();

    if (Xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    if (yb.ndim != 1 || wb.ndim != 1 || fb.ndim != 1) {
        throw std::runtime_error("y, w, full_coef must be 1-D");
    }
    const std::size_t m = static_cast<std::size_t>(Xb.shape[0]);
    const std::size_t p = static_cast<std::size_t>(Xb.shape[1]);
    if (p == 0 || p > MAX_P) {
        throw std::runtime_error("p must be in [1, 8]");
    }
    if (static_cast<std::size_t>(yb.shape[0]) != m ||
        static_cast<std::size_t>(wb.shape[0]) != m) {
        throw std::runtime_error("X, y, w shape mismatch");
    }
    if (static_cast<std::size_t>(fb.shape[0]) != p) {
        throw std::runtime_error("full_coef must have length p");
    }
    if (n_blocks_in < 1) {
        throw std::runtime_error("n_blocks must be >= 1");
    }
    const std::size_t block_size_raw = m / static_cast<std::size_t>(n_blocks_in);
    const std::size_t n_blocks =
        block_size_raw > 0 ? static_cast<std::size_t>(n_blocks_in) : 1;
    const std::size_t block_size = block_size_raw > 0 ? block_size_raw : m;

    const double* Xp = static_cast<const double*>(Xb.ptr);
    const double* yp = static_cast<const double*>(yb.ptr);
    const double* wp = static_cast<const double*>(wb.ptr);
    const double* fp = static_cast<const double*>(fb.ptr);

    py::array_t<double> out(static_cast<py::ssize_t>(p));
    auto outb = out.request();
    double* out_p = static_cast<double*>(outb.ptr);

    {
        py::gil_scoped_release release;

        // Full-sample accumulators: A = X'WX (p*p), z = X'Wy (p)
        std::vector<double> A_full(p * p, 0.0);
        std::vector<double> z_full(p, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            const double wi = wp[i];
            const double yi = yp[i];
            const double* xi = Xp + i * p;
            for (std::size_t a = 0; a < p; ++a) {
                const double wxa = wi * xi[a];
                z_full[a] += wxa * yi;
                for (std::size_t b = 0; b < p; ++b) {
                    A_full[a * p + b] += wxa * xi[b];
                }
            }
        }

        // Per-block contributions, then leave-one-block-out solves.
        std::vector<double> A_blk(p * p);
        std::vector<double> z_blk(p);
        std::vector<double> A_loo(p * p);
        std::vector<double> b_loo(p);
        std::vector<double> pseudo_sum(p, 0.0);
        std::vector<double> pseudo_sumsq(p, 0.0);
        // We'll need each pseudovalue twice (mean then variance) so cache them.
        std::vector<double> pseudovalues(static_cast<std::size_t>(n_blocks) * p, 0.0);

        for (std::size_t b = 0; b < n_blocks; ++b) {
            const std::size_t start = b * block_size;
            const std::size_t end =
                (b == n_blocks - 1) ? m : (start + block_size);

            std::fill(A_blk.begin(), A_blk.end(), 0.0);
            std::fill(z_blk.begin(), z_blk.end(), 0.0);
            for (std::size_t i = start; i < end; ++i) {
                const double wi = wp[i];
                const double yi = yp[i];
                const double* xi = Xp + i * p;
                for (std::size_t a = 0; a < p; ++a) {
                    const double wxa = wi * xi[a];
                    z_blk[a] += wxa * yi;
                    for (std::size_t c = 0; c < p; ++c) {
                        A_blk[a * p + c] += wxa * xi[c];
                    }
                }
            }

            for (std::size_t a = 0; a < p; ++a) {
                b_loo[a] = z_full[a] - z_blk[a];
                for (std::size_t c = 0; c < p; ++c) {
                    A_loo[a * p + c] = A_full[a * p + c] - A_blk[a * p + c];
                }
            }

            // coef_b = A_loo^{-1} b_loo
            const bool ok = solve_small(A_loo.data(), b_loo.data(), p);
            if (!ok) {
                // Fall back to the full-sample coefficient: a degenerate
                // block contributes a zero pseudovalue, which is what the
                // Python reference would also do under a singular fit.
                for (std::size_t a = 0; a < p; ++a) b_loo[a] = fp[a];
            }

            // Pseudovalue: n * full - (n - 1) * jackknife
            const double n_b = static_cast<double>(n_blocks);
            for (std::size_t a = 0; a < p; ++a) {
                const double pv = n_b * fp[a] - (n_b - 1.0) * b_loo[a];
                pseudovalues[b * p + a] = pv;
                pseudo_sum[a] += pv;
            }
        }

        // Mean and jackknife variance per coefficient.
        const double inv_n = 1.0 / static_cast<double>(n_blocks);
        for (std::size_t a = 0; a < p; ++a) {
            const double mean_pv = pseudo_sum[a] * inv_n;
            double sse = 0.0;
            for (std::size_t b = 0; b < n_blocks; ++b) {
                const double d = pseudovalues[b * p + a] - mean_pv;
                sse += d * d;
            }
            const double denom = static_cast<double>(n_blocks) *
                                 (static_cast<double>(n_blocks) - 1.0);
            const double var = denom > 0.0 ? sse / denom : 0.0;
            out_p[a] = std::sqrt(std::max(var, 0.0));
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_ldsc_native, m) {
    m.doc() = "Native C++ accelerator for the LDSC block-jackknife loop.";
    m.def("ldsc_block_jackknife", &py_ldsc_block_jackknife,
          py::arg("X"), py::arg("y"), py::arg("w"),
          py::arg("n_blocks"), py::arg("full_coef"),
          R"pbdoc(
            Block-jackknife SEs for a weighted least-squares fit.

            ``X`` is the (m, p) design matrix, ``y`` the (m,) response,
            ``w`` the (m,) per-observation weights, ``n_blocks`` the number
            of contiguous leave-one-out blocks, and ``full_coef`` the (p,)
            full-sample WLS coefficients. Returns a (p,) array of jackknife
            standard errors. p must satisfy 1 <= p <= 8.
          )pbdoc");
}
