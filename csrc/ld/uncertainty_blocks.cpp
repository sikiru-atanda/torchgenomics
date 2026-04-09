// Native accelerator for the per-chromosome inner loop of
// ``torchgwas.ld._blocks_novel.detect_blocks_uncertainty``.
//
// The Python reference does three things per chromosome:
//   1. Walks every (li, lj) pair within the bp window, calling
//      `weights[gi, gj].item()` to build a list — used only for the
//      "any valid pair?" empty check.
//   2. Walks adjacent pairs (k, k+1), reads `weights[gi, gj].item()`,
//      and emits a block whenever the adjacent r² drops below
//      `r2_threshold`.
//   3. For each emitted block, computes the mean of the (b choose 2)
//      off-diagonal weights via another `.item()` loop.
//
// All three loops shuttle scalars across the torch ↔ Python boundary
// with `.item()` per element. This C++ port preserves the algorithm
// (including the wasteful pair-build empty check, so behaviour stays
// identical to the Python reference) but eliminates every per-element
// round-trip.
//
// Exposed via pybind11:
//
//   uncertainty_per_chrom_blocks(weights, quality, quality_ok,
//                                chr_mask, chr_pos,
//                                max_bp, r2_threshold, min_block_snps)
//       -> list[(start_local, end_local, mean_r2, q_block)]
//
// Caller (`detect_blocks_uncertainty`) materializes LDBlock objects
// from the returned tuples.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

py::list uncertainty_per_chrom_blocks(
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> quality,
    py::array_t<bool,   py::array::c_style | py::array::forcecast> quality_ok,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_mask,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_pos,
    double max_bp,
    double r2_threshold,
    int min_block_snps
) {
    if (weights.ndim() != 2) {
        throw std::invalid_argument("weights must be 2-D");
    }
    const std::size_t M = static_cast<std::size_t>(weights.shape(0));
    if (static_cast<std::size_t>(weights.shape(1)) != M) {
        throw std::invalid_argument("weights must be square");
    }
    if (quality.ndim() != 1 || quality_ok.ndim() != 1 ||
        chr_mask.ndim() != 1 || chr_pos.ndim() != 1) {
        throw std::invalid_argument("quality, quality_ok, chr_mask, chr_pos must be 1-D");
    }
    if (static_cast<std::size_t>(quality.shape(0)) != M ||
        static_cast<std::size_t>(quality_ok.shape(0)) != M) {
        throw std::invalid_argument("quality / quality_ok length must equal weights side");
    }
    const std::size_t m_chr = static_cast<std::size_t>(chr_mask.shape(0));
    if (static_cast<std::size_t>(chr_pos.shape(0)) != m_chr) {
        throw std::invalid_argument("chr_mask and chr_pos must have the same length");
    }

    const double* Wp = weights.data();
    const double* Qp = quality.data();
    const bool*   QOp = quality_ok.data();
    const std::int64_t* CMp = chr_mask.data();
    const std::int64_t* CPp = chr_pos.data();

    // Output collected as plain C++ tuples; converted to a py::list at the end.
    std::vector<std::tuple<int, int, double, double>> blocks;

    {
        py::gil_scoped_release release;

        if (m_chr < static_cast<std::size_t>(min_block_snps)) {
            // nothing to do — leave blocks empty
        } else {
            // Step 1: empty-pair check (mirrors the Python pair-build loop).
            bool has_valid_pair = false;
            for (std::size_t li = 0; li < m_chr && !has_valid_pair; ++li) {
                const std::int64_t gi = CMp[li];
                if (!QOp[gi]) continue;
                for (std::size_t lj = li + 1; lj < m_chr; ++lj) {
                    const std::int64_t gj = CMp[lj];
                    if (!QOp[gj]) continue;
                    if (static_cast<double>(CPp[lj] - CPp[li]) > max_bp) break;
                    has_valid_pair = true;
                    break;
                }
            }

            if (has_valid_pair) {
                // Step 2: adjacent-pair break detection.
                int block_start = 0;
                const int last = static_cast<int>(m_chr) - 1;
                for (int k = 0; k < last; ++k) {
                    const std::int64_t gi = CMp[k];
                    const std::int64_t gj = CMp[k + 1];
                    double adj_r2 = 0.0;
                    if (QOp[gi] && QOp[gj]) {
                        adj_r2 = Wp[static_cast<std::size_t>(gi) * M +
                                    static_cast<std::size_t>(gj)];
                    }
                    if (adj_r2 < r2_threshold) {
                        if (k - block_start + 1 >= min_block_snps) {
                            // Step 3: per-block mean off-diagonal r² + mean quality.
                            double r2_sum = 0.0;
                            std::size_t n_pairs = 0;
                            double q_sum = 0.0;
                            const int b_size = k - block_start + 1;
                            for (int ii = 0; ii < b_size; ++ii) {
                                const std::int64_t gii = CMp[block_start + ii];
                                q_sum += Qp[gii];
                                for (int jj = ii + 1; jj < b_size; ++jj) {
                                    const std::int64_t gjj = CMp[block_start + jj];
                                    r2_sum += Wp[static_cast<std::size_t>(gii) * M +
                                                 static_cast<std::size_t>(gjj)];
                                    ++n_pairs;
                                }
                            }
                            const double mean_r2 = r2_sum / static_cast<double>(
                                n_pairs == 0 ? 1 : n_pairs);
                            const double q_block = q_sum / static_cast<double>(b_size);
                            blocks.emplace_back(block_start, k, mean_r2, q_block);
                        }
                        block_start = k + 1;
                    }
                }

                // Tail segment.
                const int tail_size = static_cast<int>(m_chr) - 1 - block_start + 1;
                if (tail_size >= min_block_snps) {
                    double r2_sum = 0.0;
                    std::size_t n_pairs = 0;
                    double q_sum = 0.0;
                    for (int ii = 0; ii < tail_size; ++ii) {
                        const std::int64_t gii = CMp[block_start + ii];
                        q_sum += Qp[gii];
                        for (int jj = ii + 1; jj < tail_size; ++jj) {
                            const std::int64_t gjj = CMp[block_start + jj];
                            r2_sum += Wp[static_cast<std::size_t>(gii) * M +
                                         static_cast<std::size_t>(gjj)];
                            ++n_pairs;
                        }
                    }
                    const double mean_r2 = r2_sum / static_cast<double>(
                        n_pairs == 0 ? 1 : n_pairs);
                    const double q_block = q_sum / static_cast<double>(tail_size);
                    blocks.emplace_back(
                        block_start, static_cast<int>(m_chr) - 1, mean_r2, q_block);
                }
            }
        }
    }  // GIL re-acquired

    py::list out;
    for (const auto& b : blocks) {
        out.append(py::make_tuple(
            std::get<0>(b), std::get<1>(b), std::get<2>(b), std::get<3>(b)));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_uncertainty_blocks_native, m) {
    m.doc() = "Native C++ accelerator for detect_blocks_uncertainty per-chromosome scan.";
    m.def("uncertainty_per_chrom_blocks", &uncertainty_per_chrom_blocks,
          py::arg("weights"),
          py::arg("quality"),
          py::arg("quality_ok"),
          py::arg("chr_mask"),
          py::arg("chr_pos"),
          py::arg("max_bp"),
          py::arg("r2_threshold"),
          py::arg("min_block_snps"),
          "Per-chromosome adjacent-r2 break detection + block stats for "
          "uncertainty-corrected LD block scan.");
}
