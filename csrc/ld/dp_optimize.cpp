// Native accelerator for the dynamic-programming block segmentation in
// ``torchgenomics.ld._blocks_literature.detect_blocks_dp_optimize``.
//
// The pure-Python implementation runs an O(m * max_block_snps) DP and
// for each candidate (i, j) calls ``_block_cost`` which is *itself*
// O(n*w) (haplotype_diversity branch — encodes every row as a Python
// tuple, drops to a Counter) or O(w^3) (tag_snp branch — greedy
// set-cover with per-cell .item() calls). For realistic m in the
// thousands the total Python overhead dominates by orders of
// magnitude.
//
// This module ports the *entire* per-chromosome DP and both cost
// branches into C++. The Python wrapper still owns: input
// preparation, the per-chromosome split, and LDBlock construction.
//
// Exposed via pybind11:
//
//   dp_optimize_segment(
//       G_chr,           // (n, m_chr) float64 row-major
//       R_chr,           // (m_chr, m_chr) float64 row-major
//       chr_pos,         // (m_chr,) int64
//       objective,       // 0 = haplotype_diversity, 1 = tag_snp
//       penalty,         // double
//       min_block_snps,  // int
//       max_block_snps,  // int
//       max_bp,          // double
//       hap_freq_threshold,  // double
//       tag_r2_threshold     // double
//   ) -> list[tuple[int, int]]
//
// Each tuple is (start_local, end_local) inclusive within the
// chromosome. The order matches the Python backtrack output.

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
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

// ----- haplotype_diversity cost -------------------------------------
//
// Mirrors:
//
//     patterns = G_block.round().long().clamp(0, 2)  # (n, w)
//     counts = Counter(tuple(row) for row in patterns)
//     threshold_count = max(1, int(hap_freq_threshold * n))
//     return sum(1 for c in counts.values() if c >= threshold_count)
//
// G is the *full* (n, m) chromosome stored row-major; the cost only
// looks at columns [c0, c1).
double cost_haplotype_diversity(
    const double* G, std::size_t n, std::size_t m_full,
    std::size_t c0, std::size_t c1,
    double hap_freq_threshold,
    std::vector<std::uint8_t>& key_buf  // reusable scratch
) {
    const std::size_t w = c1 - c0;
    if (w == 0) return 0.0;

    std::unordered_map<std::string, int> counts;
    counts.reserve(n);
    key_buf.resize(w);

    for (std::size_t r = 0; r < n; ++r) {
        const double* row = G + r * m_full + c0;
        for (std::size_t k = 0; k < w; ++k) {
            // round-half-to-even like Python's round-then-clamp.
            // Python uses .round() (banker's rounding for ties) then
            // .long().clamp(0, 2). std::nearbyint with default rounding
            // mode (FE_TONEAREST) matches.
            double v = std::nearbyint(row[k]);
            if (v < 0.0) v = 0.0;
            if (v > 2.0) v = 2.0;
            key_buf[k] = static_cast<std::uint8_t>(v);
        }
        std::string key(reinterpret_cast<const char*>(key_buf.data()), w);
        ++counts[key];
    }

    const int threshold_count = std::max<int>(
        1, static_cast<int>(hap_freq_threshold * static_cast<double>(n)));
    int n_common = 0;
    for (const auto& kv : counts) {
        if (kv.second >= threshold_count) ++n_common;
    }
    return static_cast<double>(n_common);
}

// ----- tag_snp cost --------------------------------------------------
//
// Mirrors the greedy set-cover loop:
//
//     captured = [False] * w
//     n_tags = 0
//     while not all(captured):
//         best = argmax_j (#uncaptured k with R[j, k] >= threshold)
//         mark j and everything captured by j
//         n_tags += 1
//
// R is the *full* (m_full, m_full) row-major matrix; we look at the
// (w, w) submatrix at offset (c0, c0).
double cost_tag_snp(
    const double* R, std::size_t m_full,
    std::size_t c0, std::size_t c1,
    double tag_r2_threshold,
    std::vector<char>& captured_buf  // reusable scratch
) {
    const std::size_t w = c1 - c0;
    if (w == 0) return 0.0;

    captured_buf.assign(w, 0);
    int n_tags = 0;

    while (true) {
        // Find any uncaptured SNP and the one that captures the most.
        bool any_uncaptured = false;
        std::ptrdiff_t best_idx = -1;
        int best_count = -1;

        for (std::size_t j = 0; j < w; ++j) {
            if (captured_buf[j]) continue;
            any_uncaptured = true;
            int count = 0;
            const double* row = R + (c0 + j) * m_full + c0;
            for (std::size_t k = 0; k < w; ++k) {
                if (captured_buf[k]) continue;
                if (row[k] >= tag_r2_threshold) ++count;
            }
            if (count > best_count) {
                best_count = count;
                best_idx = static_cast<std::ptrdiff_t>(j);
            }
        }
        if (!any_uncaptured) break;
        if (best_idx < 0) break;

        // Tag this SNP and everything it captures.
        const double* best_row = R + (c0 + best_idx) * m_full + c0;
        for (std::size_t k = 0; k < w; ++k) {
            if (best_row[k] >= tag_r2_threshold) captured_buf[k] = 1;
        }
        ++n_tags;
    }
    return static_cast<double>(n_tags);
}

// ----- DP segmentation entry point -----------------------------------

py::list dp_optimize_segment(
    py::array_t<double, py::array::c_style | py::array::forcecast> G_chr,
    py::array_t<double, py::array::c_style | py::array::forcecast> R_chr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_pos,
    int objective,
    double penalty,
    std::int64_t min_block_snps_i64,
    std::int64_t max_block_snps_i64,
    double max_bp,
    double hap_freq_threshold,
    double tag_r2_threshold
) {
    auto bG = G_chr.request();
    auto bR = R_chr.request();
    auto bP = chr_pos.request();
    if (bG.ndim != 2) throw std::invalid_argument("G_chr must be 2-D");
    if (bR.ndim != 2) throw std::invalid_argument("R_chr must be 2-D");
    if (bP.ndim != 1) throw std::invalid_argument("chr_pos must be 1-D");

    const std::size_t n = static_cast<std::size_t>(bG.shape[0]);
    const std::size_t m = static_cast<std::size_t>(bG.shape[1]);
    if (static_cast<std::size_t>(bR.shape[0]) != m
        || static_cast<std::size_t>(bR.shape[1]) != m) {
        throw std::invalid_argument("R_chr must be (m_chr, m_chr)");
    }
    if (static_cast<std::size_t>(bP.shape[0]) != m) {
        throw std::invalid_argument("chr_pos length must equal m_chr");
    }
    if (objective != 0 && objective != 1) {
        throw std::invalid_argument("objective must be 0 (hap_div) or 1 (tag_snp)");
    }

    const std::size_t min_block = static_cast<std::size_t>(
        std::max<std::int64_t>(min_block_snps_i64, 1));
    const std::size_t max_block = static_cast<std::size_t>(
        std::max<std::int64_t>(max_block_snps_i64, 1));

    py::list out;
    if (m < min_block) return out;

    const double* Gp = static_cast<const double*>(bG.ptr);
    const double* Rp = static_cast<const double*>(bR.ptr);
    const std::int64_t* Pp = static_cast<const std::int64_t*>(bP.ptr);

    std::vector<std::tuple<int, int>> blocks;

    {
        py::gil_scoped_release release;

        const double INF = std::numeric_limits<double>::infinity();
        std::vector<double> opt(m + 1, INF);
        std::vector<std::size_t> back(m + 1, 0);
        opt[0] = 0.0;

        for (std::size_t j = min_block; j <= m; ++j) {
            const std::size_t i_lo = (j > max_block) ? (j - max_block) : 0;
            const std::size_t i_hi = j - min_block + 1;  // exclusive (Python: range(..., j - min_block + 1))

            double best_total = INF;
            std::size_t best_i = 0;

#pragma omp parallel
            {
                // Per-thread scratch — cost functions mutate these.
                std::vector<std::uint8_t> key_buf;
                std::vector<char> captured_buf;
                double loc_total = INF;
                std::size_t loc_i = 0;

#pragma omp for schedule(static) nowait
                for (std::ptrdiff_t ii = static_cast<std::ptrdiff_t>(i_lo);
                     ii < static_cast<std::ptrdiff_t>(i_hi); ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    // Physical distance constraint: pos[j-1] - pos[i] > max_bp
                    if (static_cast<double>(Pp[j - 1] - Pp[i]) > max_bp) continue;
                    if (opt[i] == INF) continue;

                    double cost;
                    if (objective == 0) {
                        cost = cost_haplotype_diversity(
                            Gp, n, m, i, j, hap_freq_threshold, key_buf);
                    } else {
                        cost = cost_tag_snp(
                            Rp, m, i, j, tag_r2_threshold, captured_buf);
                    }

                    const double total = opt[i] + cost + penalty;
                    // Match serial strict-< tie break (smaller i wins on equality).
                    if (total < loc_total || (total == loc_total && i < loc_i)) {
                        loc_total = total;
                        loc_i = i;
                    }
                }

#pragma omp critical
                {
                    if (loc_total < best_total
                        || (loc_total == best_total && loc_i < best_i)) {
                        best_total = loc_total;
                        best_i = loc_i;
                    }
                }
            }

            if (best_total < INF) {
                opt[j] = best_total;
                back[j] = best_i;
            }
        }

        // Backtrack from m down. Mirrors:
        //
        //     pos = m_chr
        //     while pos > 0:
        //         start = backtrack[pos]
        //         if pos - start >= min_block_snps:
        //             blocks_indices.append((start, pos - 1))
        //         pos = start
        //     blocks_indices.reverse()
        std::vector<std::tuple<int, int>> rev_blocks;
        std::size_t pos = m;
        while (pos > 0) {
            const std::size_t start = back[pos];
            if (pos - start >= min_block) {
                rev_blocks.emplace_back(
                    static_cast<int>(start),
                    static_cast<int>(pos - 1));
            }
            if (start == pos) break;  // safety: avoid infinite loop
            pos = start;
        }
        blocks.assign(rev_blocks.rbegin(), rev_blocks.rend());
    }  // GIL re-acquired

    for (const auto& blk : blocks) {
        out.append(py::make_tuple(std::get<0>(blk), std::get<1>(blk)));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_dp_optimize_native, m) {
    m.doc() = "Native C++ accelerator for the DP block-optimization scan.";
    m.def("dp_optimize_segment", &dp_optimize_segment,
          py::arg("G_chr"), py::arg("R_chr"), py::arg("chr_pos"),
          py::arg("objective"), py::arg("penalty"),
          py::arg("min_block_snps"), py::arg("max_block_snps"),
          py::arg("max_bp"),
          py::arg("hap_freq_threshold"),
          py::arg("tag_r2_threshold"),
          "Run the per-chromosome DP segmentation for detect_blocks_dp_optimize.");
}
