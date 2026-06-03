// Native accelerator for the PCHT compat-pair enumeration in
// ``torchgenomics.models.haplotype_novel.compute_dosage_posterior_cov``
// and its underlying helper ``_reconstruct_compat``.
//
// Background. The principal-component haplotype test (PCHT, novel
// method #1 in Phase 46) needs the per-individual posterior
// covariance Cov(D_i | g_i) over haplotype indicators, summed across
// individuals into the (H, H) PCHT correction matrix. The Python
// reference path does, per sample i:
//
//   pairs = [(a, b) for a in range(H) for b in range(a, H)
//                  if (hap[a] + hap[b] == g[i]).all()]   # O(H² m)
//   for (a, b) in pairs:
//       prob_ab = (2 if a != b else 1) * freq[a] * freq[b]
//   total = sum(probs)
//   for (a, b), p in zip(pairs, probs):
//       w = p / total
//       E_D += w * d_ab     # d_ab: (H,) indicator
//       E_DDt += w * d_ab ⊗ d_ab
//   cov_sum += E_DDt − E_D ⊗ E_D
//
// At PCHT's working dimensions (n ∈ [100, 5000], H ∈ [3, 20],
// m_block ∈ [3, 20]) the per-pair tensor allocations and the
// ``.all()`` comparison are the principal cost. In C++ with raw int
// buffers the inner equality test is a tight m-byte comparison and
// the per-sample (H, H) accumulator fits in L1.
//
// Strategy. One C++ entry point ``dosage_posterior_cov`` that runs
// the full per-sample pipeline (enumerate + posterior-weight +
// covariance accumulator) under one GIL release. OpenMP parallelises
// the outer sample loop with per-thread cov accumulators reduced at
// the end. The F2 LD-aware haplotype pruning lives upstream in
// ``haplotype_gwas._compute_ld_aware_score`` (commit f601f20), so
// this kernel inherits the correct candidate set automatically — it
// is downstream of the pruning, never replaces it.
//
// Inputs (all C-contiguous):
//   G_block  (n, m) int64 — rounded, clamped 0..2 genotype dosages
//   hap_mat  (H, m) int64 — haplotype allele matrix (0/1)
//   freq     (H,)   float64 — EM-estimated haplotype frequencies
//   cov_out  (H, H) float64 — output, accumulated in place
//
// Per sample work: O(H² m) equality scan + O(H²) accumulation.
// Total: O(n H² m + n H²) = O(n H² (m + 1)). For n=5000, H=20, m=20:
// ~42M ops ≈ ms in C++.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

void dosage_posterior_cov(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> G_block_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> hap_mat_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> freq_arr,
    py::array_t<double, py::array::c_style> cov_out_arr
) {
    if (G_block_arr.ndim() != 2) throw std::invalid_argument("G_block must be 2-D");
    if (hap_mat_arr.ndim() != 2) throw std::invalid_argument("hap_mat must be 2-D");
    if (freq_arr.ndim() != 1) throw std::invalid_argument("freq must be 1-D");
    if (cov_out_arr.ndim() != 2) throw std::invalid_argument("cov_out must be 2-D");

    const std::size_t n = static_cast<std::size_t>(G_block_arr.shape(0));
    const std::size_t m = static_cast<std::size_t>(G_block_arr.shape(1));
    const std::size_t H = static_cast<std::size_t>(hap_mat_arr.shape(0));

    if (static_cast<std::size_t>(hap_mat_arr.shape(1)) != m) {
        throw std::invalid_argument("hap_mat must have m columns matching G_block");
    }
    if (static_cast<std::size_t>(freq_arr.shape(0)) != H) {
        throw std::invalid_argument("freq must have length H matching hap_mat");
    }
    if (static_cast<std::size_t>(cov_out_arr.shape(0)) != H
        || static_cast<std::size_t>(cov_out_arr.shape(1)) != H) {
        throw std::invalid_argument("cov_out must be (H, H)");
    }

    const int64_t* G = G_block_arr.data();
    const int64_t* hap = hap_mat_arr.data();
    const double* freq = freq_arr.data();
    double* cov_out = cov_out_arr.mutable_data();

    // Zero the output (Python contract: cov_sum starts at zero each call).
    for (std::size_t i = 0; i < H * H; ++i) cov_out[i] = 0.0;

    if (H == 0 || n == 0) return;

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            // Per-thread accumulators.
            std::vector<double> cov_local(H * H, 0.0);
            std::vector<double> E_D(H);
            std::vector<double> E_DDt(H * H);
            // Pair lists for one sample: (a, b, post_weight).
            std::vector<std::int64_t> pair_a, pair_b;
            std::vector<double> pair_prob;
            pair_a.reserve(H * (H + 1) / 2);
            pair_b.reserve(H * (H + 1) / 2);
            pair_prob.reserve(H * (H + 1) / 2);

#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
                const int64_t* g_i = G + static_cast<std::size_t>(i) * m;
                pair_a.clear();
                pair_b.clear();
                pair_prob.clear();

                // Enumerate compatible pairs (a ≤ b) where hap[a] + hap[b] == g_i.
                double total_prob = 0.0;
                for (std::size_t a = 0; a < H; ++a) {
                    const int64_t* hap_a = hap + a * m;
                    for (std::size_t b = a; b < H; ++b) {
                        const int64_t* hap_b = hap + b * m;
                        bool match = true;
                        for (std::size_t k = 0; k < m; ++k) {
                            if (hap_a[k] + hap_b[k] != g_i[k]) {
                                match = false;
                                break;
                            }
                        }
                        if (!match) continue;
                        const double w_ab = (a == b ? 1.0 : 2.0) * freq[a] * freq[b];
                        pair_a.push_back(static_cast<std::int64_t>(a));
                        pair_b.push_back(static_cast<std::int64_t>(b));
                        pair_prob.push_back(w_ab);
                        total_prob += w_ab;
                    }
                }

                if (total_prob < 1e-300 || pair_prob.empty()) continue;

                // Zero per-sample accumulators.
                std::fill(E_D.begin(), E_D.end(), 0.0);
                std::fill(E_DDt.begin(), E_DDt.end(), 0.0);

                const double inv_total = 1.0 / total_prob;
                for (std::size_t k = 0; k < pair_prob.size(); ++k) {
                    const std::int64_t a = pair_a[k];
                    const std::int64_t b = pair_b[k];
                    const double w = pair_prob[k] * inv_total;

                    // Build indicator d (sparse: at most two non-zero entries).
                    // E_D += w * d.
                    if (a == b) {
                        // d[a] = 2.0
                        E_D[a] += 2.0 * w;
                        // E_DDt += w * d ⊗ d  → only [a, a] entry: w * 4.0
                        E_DDt[a * H + a] += 4.0 * w;
                    } else {
                        E_D[a] += w;
                        E_D[b] += w;
                        // d[a] = d[b] = 1 → d ⊗ d has 1s at (a,a), (b,b), (a,b), (b,a)
                        E_DDt[a * H + a] += w;
                        E_DDt[b * H + b] += w;
                        E_DDt[a * H + b] += w;
                        E_DDt[b * H + a] += w;
                    }
                }

                // cov_i = E_DDt − E_D ⊗ E_D; accumulate into cov_local.
                for (std::size_t r = 0; r < H; ++r) {
                    const double Er = E_D[r];
                    for (std::size_t c = 0; c < H; ++c) {
                        cov_local[r * H + c] += E_DDt[r * H + c] - Er * E_D[c];
                    }
                }
            }

            // Reduce thread-local accumulator into shared cov_out.
#ifdef _OPENMP
            #pragma omp critical
#endif
            {
                for (std::size_t k = 0; k < H * H; ++k) cov_out[k] += cov_local[k];
            }
        }
    }
}

}  // namespace

PYBIND11_MODULE(_pcht_native, m) {
    m.doc() = "Native C++ accelerator for PCHT compat-pair enumeration + "
              "per-individual dosage-posterior covariance accumulation.";
    m.def("dosage_posterior_cov", &dosage_posterior_cov,
          py::arg("G_block"),
          py::arg("hap_mat"),
          py::arg("freq"),
          py::arg("cov_out"),
          "Accumulate the summed-over-individuals posterior covariance "
          "Cov(D_i | g_i) into cov_out (H, H), in place. Inputs must be "
          "int64 for G_block and hap_mat and float64 for freq / cov_out.");
}
