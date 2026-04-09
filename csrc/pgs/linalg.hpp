// Tiny header-only dense linear algebra used by the PRS-CS Gibbs sampler.
//
// Operates on row-major double matrices stored as std::vector<double> with
// stride = ncols. Hand-rolled to avoid a heavyweight Eigen dependency for the
// (typically small: 50..2000) LD-block sizes PRS-CS encounters.

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tgw {

// In-place lower Cholesky factorization. `A` is row-major, `n x n`, symmetric
// positive-definite on input; on output, the lower triangle holds L such that
// A = L L^T. The strict upper triangle is left untouched.
//
// Returns true on success, false if a non-positive pivot is encountered.
inline bool cholesky_lower_inplace(double* A, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = A[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                s -= A[i * n + k] * A[j * n + k];
            }
            if (i == j) {
                if (s <= 0.0) return false;
                A[i * n + i] = std::sqrt(s);
            } else {
                A[i * n + j] = s / A[j * n + j];
            }
        }
    }
    return true;
}

// Solve L * y = b in-place (forward substitution). L is lower-triangular,
// row-major n x n; only the lower triangle is read. `x` holds b on input and
// y on output.
inline void trsv_lower(const double* L, double* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        double s = x[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= L[i * n + k] * x[k];
        }
        x[i] = s / L[i * n + i];
    }
}

// Solve L^T * x = y in-place (back substitution). Same storage convention.
inline void trsv_lower_transpose(const double* L, double* x, std::size_t n) {
    if (n == 0) return;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        double s = x[i];
        for (std::size_t k = static_cast<std::size_t>(i) + 1; k < n; ++k) {
            s -= L[k * n + i] * x[k];
        }
        x[i] = s / L[i * n + i];
    }
}

}  // namespace tgw
