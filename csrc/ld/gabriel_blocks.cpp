// Native accelerator for Gabriel et al. (2002) haplotype-block detection.
//
// C++ port of the inner loop of
// ``torchgenomics.ld._blocks.detect_blocks_gabriel``. The pure-Python
// implementation remains the canonical algorithmic reference; this
// module reproduces its semantics exactly (greedy longest-first
// candidate scan, identical accept rule) but replaces the
// O(n^4)-per-candidate dict lookups with 2D prefix sums, so the inner
// scan over a candidate (start, end) pair is O(1) and the total work
// is O(n^2).
//
// Exposed via pybind11:
//
//   gabriel_blocks(
//       n_snps,
//       idx_i, idx_j,
//       strong_ld, strong_rec, informative,
//       r2, dprime,
//       strong_pct, rec_max_pct, min_block_snps
//   ) -> list[tuple[int, int, float, float]]
//
// Each tuple is (start, end, mean_r2, mean_dprime). Blocks are
// returned in the order they are accepted (the Python wrapper sorts
// by chromosome / position when it builds LDBlock objects).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

// 2-D prefix-sum view: P has shape (n+1, n+1) row-major.
// Returns sum over the rectangle [r0, r1) x [c0, c1) (open on the
// upper bound). All bounds must be in [0, n].
template <typename T>
static inline T rect_sum(
    const T* P, std::size_t stride,
    std::size_t r0, std::size_t r1,
    std::size_t c0, std::size_t c1
) {
    return P[r1 * stride + c1]
         - P[r0 * stride + c1]
         - P[r1 * stride + c0]
         + P[r0 * stride + c0];
}

template <typename T>
static void build_prefix(
    const std::vector<T>& M,  // n*n row-major, upper triangle only
    std::vector<T>& P,        // (n+1)*(n+1) row-major, output
    std::size_t n
) {
    const std::size_t stride = n + 1;
    P.assign(stride * stride, T{0});
    for (std::size_t i = 0; i < n; ++i) {
        T row_acc{0};
        for (std::size_t j = 0; j < n; ++j) {
            row_acc += M[i * n + j];
            P[(i + 1) * stride + (j + 1)] =
                P[i * stride + (j + 1)] + row_acc;
        }
    }
}

py::list gabriel_blocks(
    std::int64_t n_snps_i64,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx_i,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx_j,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> strong_ld,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> strong_rec,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> informative,
    py::array_t<double, py::array::c_style | py::array::forcecast> r2,
    py::array_t<double, py::array::c_style | py::array::forcecast> dprime,
    double strong_pct,
    double rec_max_pct,
    std::int64_t min_block_snps_i64
) {
    if (n_snps_i64 < 0) throw std::invalid_argument("n_snps must be >= 0");
    const std::size_t n = static_cast<std::size_t>(n_snps_i64);
    const std::size_t min_block = static_cast<std::size_t>(
        std::max<std::int64_t>(min_block_snps_i64, 1));

    auto bi = idx_i.request();
    auto bj = idx_j.request();
    auto bsld = strong_ld.request();
    auto bsrec = strong_rec.request();
    auto binf = informative.request();
    auto br2 = r2.request();
    auto bdp = dprime.request();
    if (bi.ndim != 1 || bj.ndim != 1 || bsld.ndim != 1 || bsrec.ndim != 1
        || binf.ndim != 1 || br2.ndim != 1 || bdp.ndim != 1) {
        throw std::invalid_argument("all pair arrays must be 1-D");
    }
    const std::size_t P = static_cast<std::size_t>(bi.shape[0]);
    if (bj.shape[0] != static_cast<py::ssize_t>(P)
        || bsld.shape[0] != static_cast<py::ssize_t>(P)
        || bsrec.shape[0] != static_cast<py::ssize_t>(P)
        || binf.shape[0] != static_cast<py::ssize_t>(P)
        || br2.shape[0] != static_cast<py::ssize_t>(P)
        || bdp.shape[0] != static_cast<py::ssize_t>(P)) {
        throw std::invalid_argument("all pair arrays must have equal length");
    }

    py::list out;
    if (n < 2) return out;

    const std::int64_t* p_i = static_cast<const std::int64_t*>(bi.ptr);
    const std::int64_t* p_j = static_cast<const std::int64_t*>(bj.ptr);
    const std::uint8_t* p_sld = static_cast<const std::uint8_t*>(bsld.ptr);
    const std::uint8_t* p_srec = static_cast<const std::uint8_t*>(bsrec.ptr);
    const std::uint8_t* p_inf = static_cast<const std::uint8_t*>(binf.ptr);
    const double* p_r2 = static_cast<const double*>(br2.ptr);
    const double* p_dp = static_cast<const double*>(bdp.ptr);

    std::vector<std::tuple<std::size_t, std::size_t, double, double>>
        accepted;

    {
        py::gil_scoped_release release;

        // Dense upper-triangular matrices (i < j). Use int32 for counts
        // and double for r2 / dprime sums.
        std::vector<std::int32_t> M_inf(n * n, 0);
        std::vector<std::int32_t> M_sld(n * n, 0);
        std::vector<std::int32_t> M_srec(n * n, 0);
        std::vector<std::int32_t> M_npairs(n * n, 0);
        std::vector<double> M_r2(n * n, 0.0);
        std::vector<double> M_dp(n * n, 0.0);

        for (std::size_t k = 0; k < P; ++k) {
            const std::int64_t ii = p_i[k];
            const std::int64_t jj = p_j[k];
            if (ii < 0 || jj < 0) continue;
            const std::size_t a = static_cast<std::size_t>(ii);
            const std::size_t b = static_cast<std::size_t>(jj);
            if (a >= n || b >= n || a >= b) continue;  // skip lower / diag / OOB
            const std::size_t off = a * n + b;
            M_inf[off] = p_inf[k] ? 1 : 0;
            M_sld[off] = p_sld[k] ? 1 : 0;
            M_srec[off] = p_srec[k] ? 1 : 0;
            M_npairs[off] = 1;
            M_r2[off] = p_r2[k];
            M_dp[off] = p_dp[k];
        }

        std::vector<std::int32_t> Pinf, Psld, Psrec, Pnp;
        std::vector<double> Pr2, Pdp;
        build_prefix(M_inf, Pinf, n);
        build_prefix(M_sld, Psld, n);
        build_prefix(M_srec, Psrec, n);
        build_prefix(M_npairs, Pnp, n);
        build_prefix(M_r2, Pr2, n);
        build_prefix(M_dp, Pdp, n);

        const std::size_t stride = n + 1;

        // Build candidate list (length, start, end), sort longest first.
        // length here is (end - start) so that ties prefer smaller start
        // — matches the Python tuple-sort behavior on (length, start, end)
        // with reverse=True (length desc, then start desc, then end desc).
        std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>
            candidates;
        candidates.reserve(n * (n + 1) / 2);
        for (std::size_t s = 0; s < n; ++s) {
            const std::size_t e_start = s + min_block - 1;
            for (std::size_t e = e_start; e < n; ++e) {
                candidates.emplace_back(e - s, s, e);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  std::greater<std::tuple<std::size_t, std::size_t, std::size_t>>());

        std::vector<char> used(n, 0);

        for (const auto& cand : candidates) {
            const std::size_t s = std::get<1>(cand);
            const std::size_t e = std::get<2>(cand);

            // Skip if any SNP already in a block (linear scan; the early
            // exit makes this cheap on average since accepted blocks
            // remove their range from future consideration).
            bool blocked = false;
            for (std::size_t k = s; k <= e; ++k) {
                if (used[k]) { blocked = true; break; }
            }
            if (blocked) continue;

            // Rectangle [s..e] x [s..e] over upper-triangular matrices:
            //   r0 = s,  r1 = e + 1
            //   c0 = s,  c1 = e + 1
            // Because M is zero on/below the diagonal this counts pairs
            // (a, b) with s <= a < b <= e.
            const std::size_t r0 = s, r1 = e + 1;
            const std::size_t c0 = s, c1 = e + 1;

            const std::int32_t n_pairs = rect_sum(Pnp.data(), stride, r0, r1, c0, c1);
            if (n_pairs == 0) continue;
            const std::int32_t n_info = rect_sum(Pinf.data(), stride, r0, r1, c0, c1);
            if (n_info == 0) continue;
            const std::int32_t n_sld = rect_sum(Psld.data(), stride, r0, r1, c0, c1);
            const std::int32_t n_srec = rect_sum(Psrec.data(), stride, r0, r1, c0, c1);
            const double sum_r2 = rect_sum(Pr2.data(), stride, r0, r1, c0, c1);
            const double sum_dp = rect_sum(Pdp.data(), stride, r0, r1, c0, c1);

            const double frac_sld = static_cast<double>(n_sld)
                                  / static_cast<double>(n_info);
            const double frac_srec = static_cast<double>(n_srec)
                                   / static_cast<double>(n_info);

            if (frac_sld >= strong_pct && frac_srec < rec_max_pct) {
                for (std::size_t k = s; k <= e; ++k) used[k] = 1;
                const double mean_r2 = sum_r2 / static_cast<double>(n_pairs);
                const double mean_dp = sum_dp / static_cast<double>(n_pairs);
                accepted.emplace_back(s, e, mean_r2, mean_dp);
            }
        }
    }  // GIL re-acquired

    for (const auto& blk : accepted) {
        out.append(py::make_tuple(
            static_cast<int>(std::get<0>(blk)),
            static_cast<int>(std::get<1>(blk)),
            std::get<2>(blk),
            std::get<3>(blk)
        ));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_gabriel_native, m) {
    m.doc() = "Native C++ accelerator for Gabriel haplotype-block detection.";
    m.def("gabriel_blocks", &gabriel_blocks,
          py::arg("n_snps"),
          py::arg("idx_i"), py::arg("idx_j"),
          py::arg("strong_ld"), py::arg("strong_rec"), py::arg("informative"),
          py::arg("r2"), py::arg("dprime"),
          py::arg("strong_pct"), py::arg("rec_max_pct"),
          py::arg("min_block_snps"),
          "Greedy longest-first Gabriel block scan over a sparse pair list.");
}
