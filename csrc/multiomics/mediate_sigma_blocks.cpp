// Native accelerator for the mediation rho-sensitivity per-pair WLS in
// ``torchgenomics.multiomics._scan_batched``.
//
// Background. The Imai-Keele-Yamamoto rho-sensitivity analysis requires
// the per-(SNP, mediator) residual scales σ_v (Stage-M residual) and
// σ_u (Stage-Y direct residual). The Python reference path runs nested
// ``for i in range(s_b): for j in range(f_b)`` loops, each iteration
// of which builds a new design matrix Xy, computes a small Gram, calls
// ``torch.linalg.solve``, and reduces a weighted SSR. At typical block
// sizes (s_b=256, f_b=64) this is 16,384 per-pair Python turns plus
// ~6 tensor allocations per turn — the tail keeping ``mediate-scan``
// from scaling to UKB-size mediator sets.
//
// This C++ port runs the per-pair work under a single GIL release. To
// keep the C++ surface narrow we precompute every constant cross-
// product on the Python side using torch BLAS (cheap, one matmul per
// block), and the C++ kernel just consumes those precomputed scalars
// to build the per-pair (p_y × p_y) Gram, runs an inline Cholesky
// decomposition, solves for β, and computes SSR = Y_W_Y − Bᵀβ (using
// the standard identity SSR = Y_W_Y − βᵀ A β = Y_W_Y − Bᵀβ for the
// WLS normal equations Aβ = B).
//
// Per-pair work: O(p_y³) Cholesky + O(p_y²) substitution + O(p_y)
// reduction. With p_y ≤ 52 (typical c0 ≈ 10) this is ~1k ops per
// pair, fully cache-resident. OpenMP parallelises the outer SNP loop.
//
// Two entry points, one per σ:
//   sigma_v_block — Stage-M residual (M_block on [SNP_i | X0])
//   sigma_u_block — Stage-Y direct residual (Y on [SNP_i | M_j | X0])
//
// Both take precomputed weighted cross-products as input arrays and
// write per-(i, j) σ values into a contiguous (s_b, f_b) output buffer
// in place. Numerical floor: σ values are clamped to ≥ 1e-24 (matches
// the Python reference) and sqrt is monotone, so per-pair SSR / dof
// agreement to FP64 carries directly to σ agreement at FP64.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

// In-place Cholesky decomposition: overwrites the lower triangle of A
// with L such that A = L Lᵀ. Returns true on success, false if A is
// not positive-definite (NaN propagation aborts the kernel cleanly).
inline bool cholesky_inplace(double* A, std::size_t p) {
    for (std::size_t j = 0; j < p; ++j) {
        double diag = A[j * p + j];
        for (std::size_t k = 0; k < j; ++k) {
            diag -= A[j * p + k] * A[j * p + k];
        }
        if (!(diag > 0.0)) return false;
        const double L_jj = std::sqrt(diag);
        A[j * p + j] = L_jj;
        const double inv_L_jj = 1.0 / L_jj;
        for (std::size_t i = j + 1; i < p; ++i) {
            double s = A[i * p + j];
            for (std::size_t k = 0; k < j; ++k) {
                s -= A[i * p + k] * A[j * p + k];
            }
            A[i * p + j] = s * inv_L_jj;
        }
    }
    return true;
}

// Cholesky solve: given L (lower triangle of A's factor in A_buf) and
// the rhs in b, overwrites b with x such that A x = b. Uses forward
// substitution L y = b followed by back substitution Lᵀ x = y.
inline void cholesky_solve_inplace(
    const double* A_buf, double* b, std::size_t p
) {
    // Forward: L y = b, write y over b.
    for (std::size_t i = 0; i < p; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k) {
            s -= A_buf[i * p + k] * b[k];
        }
        b[i] = s / A_buf[i * p + i];
    }
    // Back: Lᵀ x = y, overwrite y with x.
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(p) - 1; i >= 0; --i) {
        double s = b[i];
        for (std::size_t k = i + 1; k < p; ++k) {
            s -= A_buf[k * p + i] * b[k];
        }
        b[i] = s / A_buf[i * p + i];
    }
}

void sigma_v_block(
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_snp_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_M_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> A_X0_X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> X0_M_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> M_M_arr,
    int n,
    py::array_t<double, py::array::c_style> sigma_v_out_arr
) {
    if (snp_snp_arr.ndim() != 1) throw std::invalid_argument("snp_snp must be 1-D");
    if (snp_X0_arr.ndim() != 2) throw std::invalid_argument("snp_X0 must be 2-D");
    if (snp_M_arr.ndim() != 2) throw std::invalid_argument("snp_M must be 2-D");
    if (A_X0_X0_arr.ndim() != 2) throw std::invalid_argument("A_X0_X0 must be 2-D");
    if (X0_M_arr.ndim() != 2) throw std::invalid_argument("X0_M must be 2-D");
    if (M_M_arr.ndim() != 1) throw std::invalid_argument("M_M must be 1-D");
    if (sigma_v_out_arr.ndim() != 2) throw std::invalid_argument("sigma_v_out must be 2-D");

    const std::size_t s_b = static_cast<std::size_t>(snp_snp_arr.shape(0));
    const std::size_t c0 = static_cast<std::size_t>(A_X0_X0_arr.shape(0));
    const std::size_t f_b = static_cast<std::size_t>(M_M_arr.shape(0));
    const std::size_t p_m = 1 + c0;

    if (static_cast<std::size_t>(snp_X0_arr.shape(0)) != s_b
        || static_cast<std::size_t>(snp_X0_arr.shape(1)) != c0) {
        throw std::invalid_argument("snp_X0 must be (s_b, c0)");
    }
    if (static_cast<std::size_t>(snp_M_arr.shape(0)) != s_b
        || static_cast<std::size_t>(snp_M_arr.shape(1)) != f_b) {
        throw std::invalid_argument("snp_M must be (s_b, f_b)");
    }
    if (static_cast<std::size_t>(A_X0_X0_arr.shape(1)) != c0) {
        throw std::invalid_argument("A_X0_X0 must be (c0, c0)");
    }
    if (static_cast<std::size_t>(X0_M_arr.shape(0)) != c0
        || static_cast<std::size_t>(X0_M_arr.shape(1)) != f_b) {
        throw std::invalid_argument("X0_M must be (c0, f_b)");
    }
    if (static_cast<std::size_t>(sigma_v_out_arr.shape(0)) != s_b
        || static_cast<std::size_t>(sigma_v_out_arr.shape(1)) != f_b) {
        throw std::invalid_argument("sigma_v_out must be (s_b, f_b)");
    }
    if (n <= 0) throw std::invalid_argument("n must be > 0");

    const double* snp_snp = snp_snp_arr.data();
    const double* snp_X0 = snp_X0_arr.data();
    const double* snp_M = snp_M_arr.data();
    const double* A_X0_X0 = A_X0_X0_arr.data();
    const double* X0_M = X0_M_arr.data();
    const double* M_M = M_M_arr.data();
    double* sigma_v_out = sigma_v_out_arr.mutable_data();

    const std::ptrdiff_t dof = std::max<std::ptrdiff_t>(
        static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(p_m), 1
    );
    const double inv_dof = 1.0 / static_cast<double>(dof);

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> A_buf(p_m * p_m);
            std::vector<double> beta_buf(p_m);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(s_b); ++i) {
                // Build A = Xmᵀ W Xm from precomputed cross-products.
                // Xm = [snp_i | X0], so:
                //   A[0, 0]     = snp_snp[i]
                //   A[0, k>=1]  = snp_X0[i, k-1]
                //   A[k>=1, l>=1] = A_X0_X0[k-1, l-1]
                A_buf[0] = snp_snp[i];
                for (std::size_t k = 1; k < p_m; ++k) {
                    A_buf[k] = snp_X0[i * c0 + k - 1];
                    A_buf[k * p_m] = A_buf[k];
                    for (std::size_t l = 1; l < p_m; ++l) {
                        A_buf[k * p_m + l] = A_X0_X0[(k - 1) * c0 + (l - 1)];
                    }
                }

                // Factor A once for this SNP.
                std::vector<double> A_factored(A_buf);
                if (!cholesky_inplace(A_factored.data(), p_m)) {
                    // Singular block — emit floored σ_v and continue.
                    for (std::size_t j = 0; j < f_b; ++j) {
                        sigma_v_out[i * f_b + j] = std::sqrt(1e-24);
                    }
                    continue;
                }

                for (std::size_t j = 0; j < f_b; ++j) {
                    // B[0]    = snp_M[i, j]
                    // B[k>=1] = X0_M[k-1, j]
                    beta_buf[0] = snp_M[i * f_b + j];
                    for (std::size_t k = 1; k < p_m; ++k) {
                        beta_buf[k] = X0_M[(k - 1) * f_b + j];
                    }
                    // Save B before solving (we need Bᵀβ for SSR).
                    double Bdotbeta = 0.0;
                    std::vector<double> B_save(beta_buf);
                    cholesky_solve_inplace(A_factored.data(), beta_buf.data(), p_m);
                    for (std::size_t k = 0; k < p_m; ++k) {
                        Bdotbeta += B_save[k] * beta_buf[k];
                    }
                    const double ssr = std::max(M_M[j] - Bdotbeta, 0.0);
                    const double var = std::max(ssr * inv_dof, 1e-24);
                    sigma_v_out[i * f_b + j] = std::sqrt(var);
                }
            }
        }
    }
}

void sigma_u_block(
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_snp_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_Y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> snp_M_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> M_M_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> M_X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> M_Y_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> A_X0_X0_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> X0_Y_arr,
    double Y_W_Y,
    int n,
    py::array_t<double, py::array::c_style> sigma_u_out_arr
) {
    if (snp_snp_arr.ndim() != 1) throw std::invalid_argument("snp_snp must be 1-D");
    if (snp_X0_arr.ndim() != 2) throw std::invalid_argument("snp_X0 must be 2-D");
    if (snp_Y_arr.ndim() != 1) throw std::invalid_argument("snp_Y must be 1-D");
    if (snp_M_arr.ndim() != 2) throw std::invalid_argument("snp_M must be 2-D");
    if (M_M_arr.ndim() != 1) throw std::invalid_argument("M_M must be 1-D");
    if (M_X0_arr.ndim() != 2) throw std::invalid_argument("M_X0 must be 2-D");
    if (M_Y_arr.ndim() != 1) throw std::invalid_argument("M_Y must be 1-D");
    if (A_X0_X0_arr.ndim() != 2) throw std::invalid_argument("A_X0_X0 must be 2-D");
    if (X0_Y_arr.ndim() != 1) throw std::invalid_argument("X0_Y must be 1-D");
    if (sigma_u_out_arr.ndim() != 2) throw std::invalid_argument("sigma_u_out must be 2-D");

    const std::size_t s_b = static_cast<std::size_t>(snp_snp_arr.shape(0));
    const std::size_t c0 = static_cast<std::size_t>(A_X0_X0_arr.shape(0));
    const std::size_t f_b = static_cast<std::size_t>(M_M_arr.shape(0));
    const std::size_t p_y = 2 + c0;

    if (static_cast<std::size_t>(snp_X0_arr.shape(0)) != s_b
        || static_cast<std::size_t>(snp_X0_arr.shape(1)) != c0) {
        throw std::invalid_argument("snp_X0 must be (s_b, c0)");
    }
    if (static_cast<std::size_t>(snp_Y_arr.shape(0)) != s_b) {
        throw std::invalid_argument("snp_Y must be (s_b,)");
    }
    if (static_cast<std::size_t>(snp_M_arr.shape(0)) != s_b
        || static_cast<std::size_t>(snp_M_arr.shape(1)) != f_b) {
        throw std::invalid_argument("snp_M must be (s_b, f_b)");
    }
    if (static_cast<std::size_t>(M_X0_arr.shape(0)) != f_b
        || static_cast<std::size_t>(M_X0_arr.shape(1)) != c0) {
        throw std::invalid_argument("M_X0 must be (f_b, c0)");
    }
    if (static_cast<std::size_t>(M_Y_arr.shape(0)) != f_b) {
        throw std::invalid_argument("M_Y must be (f_b,)");
    }
    if (static_cast<std::size_t>(A_X0_X0_arr.shape(1)) != c0) {
        throw std::invalid_argument("A_X0_X0 must be (c0, c0)");
    }
    if (static_cast<std::size_t>(X0_Y_arr.shape(0)) != c0) {
        throw std::invalid_argument("X0_Y must be (c0,)");
    }
    if (static_cast<std::size_t>(sigma_u_out_arr.shape(0)) != s_b
        || static_cast<std::size_t>(sigma_u_out_arr.shape(1)) != f_b) {
        throw std::invalid_argument("sigma_u_out must be (s_b, f_b)");
    }
    if (n <= 0) throw std::invalid_argument("n must be > 0");

    const double* snp_snp = snp_snp_arr.data();
    const double* snp_X0 = snp_X0_arr.data();
    const double* snp_Y = snp_Y_arr.data();
    const double* snp_M = snp_M_arr.data();
    const double* M_M = M_M_arr.data();
    const double* M_X0 = M_X0_arr.data();
    const double* M_Y = M_Y_arr.data();
    const double* A_X0_X0 = A_X0_X0_arr.data();
    const double* X0_Y = X0_Y_arr.data();
    double* sigma_u_out = sigma_u_out_arr.mutable_data();

    const std::ptrdiff_t dof = std::max<std::ptrdiff_t>(
        static_cast<std::ptrdiff_t>(n) - static_cast<std::ptrdiff_t>(p_y), 1
    );
    const double inv_dof = 1.0 / static_cast<double>(dof);

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> A_buf(p_y * p_y);
            std::vector<double> B_buf(p_y);
            std::vector<double> B_save(p_y);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(s_b); ++i) {
                for (std::size_t j = 0; j < f_b; ++j) {
                    // Build A = Xyᵀ W Xy from precomputed cross-products.
                    // Xy = [snp_i | M_j | X0]:
                    //   A[0, 0]     = snp_snp[i]
                    //   A[0, 1]     = snp_M[i, j]
                    //   A[0, k>=2]  = snp_X0[i, k-2]
                    //   A[1, 1]     = M_M[j]
                    //   A[1, k>=2]  = M_X0[j, k-2]
                    //   A[k>=2, l>=2] = A_X0_X0[k-2, l-2]
                    A_buf[0] = snp_snp[i];
                    A_buf[1] = snp_M[i * f_b + j];
                    A_buf[p_y] = A_buf[1];
                    A_buf[p_y + 1] = M_M[j];
                    for (std::size_t k = 2; k < p_y; ++k) {
                        A_buf[k] = snp_X0[i * c0 + k - 2];
                        A_buf[k * p_y] = A_buf[k];
                        A_buf[p_y + k] = M_X0[j * c0 + k - 2];
                        A_buf[k * p_y + 1] = A_buf[p_y + k];
                        for (std::size_t l = 2; l < p_y; ++l) {
                            A_buf[k * p_y + l] = A_X0_X0[(k - 2) * c0 + (l - 2)];
                        }
                    }

                    // B = [snp_Y[i], M_Y[j], X0_Y[0..c0-1]]
                    B_buf[0] = snp_Y[i];
                    B_buf[1] = M_Y[j];
                    for (std::size_t k = 2; k < p_y; ++k) {
                        B_buf[k] = X0_Y[k - 2];
                    }
                    B_save = B_buf;

                    if (!cholesky_inplace(A_buf.data(), p_y)) {
                        sigma_u_out[i * f_b + j] = std::sqrt(1e-24);
                        continue;
                    }
                    cholesky_solve_inplace(A_buf.data(), B_buf.data(), p_y);

                    double Bdotbeta = 0.0;
                    for (std::size_t k = 0; k < p_y; ++k) {
                        Bdotbeta += B_save[k] * B_buf[k];
                    }
                    const double ssr = std::max(Y_W_Y - Bdotbeta, 0.0);
                    const double var = std::max(ssr * inv_dof, 1e-24);
                    sigma_u_out[i * f_b + j] = std::sqrt(var);
                }
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_mediate_native, m) {
    m.doc() = "Native C++ accelerator for mediation rho-sensitivity per-pair WLS.";
    m.def("sigma_v_block", &sigma_v_block,
          py::arg("snp_snp"),
          py::arg("snp_X0"),
          py::arg("snp_M"),
          py::arg("A_X0_X0"),
          py::arg("X0_M"),
          py::arg("M_M"),
          py::arg("n"),
          py::arg("sigma_v_out"),
          "Stage-M residual σ_v per (i, j) via Cholesky on precomputed "
          "weighted cross-products. Writes (s_b, f_b) sigma_v_out in place.");
    m.def("sigma_u_block", &sigma_u_block,
          py::arg("snp_snp"),
          py::arg("snp_X0"),
          py::arg("snp_Y"),
          py::arg("snp_M"),
          py::arg("M_M"),
          py::arg("M_X0"),
          py::arg("M_Y"),
          py::arg("A_X0_X0"),
          py::arg("X0_Y"),
          py::arg("Y_W_Y"),
          py::arg("n"),
          py::arg("sigma_u_out"),
          "Stage-Y direct residual σ_u per (i, j) via Cholesky on "
          "precomputed weighted cross-products. Writes (s_b, f_b) "
          "sigma_u_out in place.");
}
