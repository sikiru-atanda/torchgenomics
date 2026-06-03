// Native accelerator for the MAGMA-style snp_to_gene per-gene window
// scan in ``torchgenomics.postgwas._enrichment.snp_to_gene``.
//
// Background. The Python reference does, for each gene g (typically
// ~20K genes per scan):
//
//   snps_on_chr = chr_to_snps.get(str(gene_chr[g]), [])
//   indices = [snp_idx for pos, snp_idx in snps_on_chr
//                       if gene_start <= pos <= gene_end]
//   if not indices: stat=0, p=1
//   else:
//       gene_chi2 = chi2[indices]              # tensor allocation
//       mean_chi2 = gene_chi2.mean()
//       z = (mean_chi2 - 1) / sqrt(2 / k)
//       p = 0.5 * erfc(z / sqrt(2))
//
// At 20K genes the per-gene Python list-comprehension + tensor
// allocation + small reduce dominates wall time. The arithmetic
// itself is tiny.
//
// Strategy. Python preprocesses inputs once:
//   - Encode chromosomes as integer IDs.
//   - Sort all SNPs by (chr_id, pos); reorder chi2 to match.
//   - Build chr_offsets[c] = first sorted-array index with chr_id == c.
//   - Encode each gene's chromosome as an integer ID (or -1 if absent
//     from the SNP panel).
//
// Then C++ does, for each gene, a binary-search of the chromosome's
// position segment for the window endpoints, walks the indices,
// accumulates sum_chi2 and n_snps in registers, and computes mean →
// z → 0.5 * erfc(z / sqrt(2)). All genes are independent so OpenMP
// parallelises the outer-gene loop trivially.
//
// Per-gene work: O(log m_c + k) where m_c is the chromosome's SNP
// count and k is the number of SNPs in the gene window. Total:
// O(n_genes (log m + k_max)). For 20K genes × 10M SNPs × ~100 SNPs
// per gene window ≈ 20K × (23 + 100) ≈ 2.5M ops — ms in C++.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline double normal_sf(double z) {
    // 1 - Phi(z) = 0.5 * erfc(z / sqrt(2)).
    constexpr double inv_sqrt2 = 0.7071067811865476;  // 1 / sqrt(2)
    return 0.5 * std::erfc(z * inv_sqrt2);
}

void snp_to_gene_scan(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sorted_positions_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> sorted_chi2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> chr_offsets_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gene_chr_ids_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gene_starts_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gene_ends_arr,
    py::array_t<double, py::array::c_style> stat_out_arr,
    py::array_t<double, py::array::c_style> p_out_arr,
    py::array_t<int64_t, py::array::c_style> n_snps_out_arr
) {
    if (sorted_positions_arr.ndim() != 1) throw std::invalid_argument("sorted_positions must be 1-D");
    if (sorted_chi2_arr.ndim() != 1) throw std::invalid_argument("sorted_chi2 must be 1-D");
    if (chr_offsets_arr.ndim() != 1) throw std::invalid_argument("chr_offsets must be 1-D");
    if (gene_chr_ids_arr.ndim() != 1) throw std::invalid_argument("gene_chr_ids must be 1-D");
    if (gene_starts_arr.ndim() != 1) throw std::invalid_argument("gene_starts must be 1-D");
    if (gene_ends_arr.ndim() != 1) throw std::invalid_argument("gene_ends must be 1-D");
    if (stat_out_arr.ndim() != 1) throw std::invalid_argument("stat_out must be 1-D");
    if (p_out_arr.ndim() != 1) throw std::invalid_argument("p_out must be 1-D");
    if (n_snps_out_arr.ndim() != 1) throw std::invalid_argument("n_snps_out must be 1-D");

    const std::size_t m = static_cast<std::size_t>(sorted_positions_arr.shape(0));
    const std::size_t n_chr = static_cast<std::size_t>(chr_offsets_arr.shape(0)) - 1;
    const std::size_t n_genes = static_cast<std::size_t>(gene_chr_ids_arr.shape(0));

    if (static_cast<std::size_t>(sorted_chi2_arr.shape(0)) != m) {
        throw std::invalid_argument("sorted_chi2 length must match sorted_positions");
    }
    if (static_cast<std::size_t>(gene_starts_arr.shape(0)) != n_genes
        || static_cast<std::size_t>(gene_ends_arr.shape(0)) != n_genes) {
        throw std::invalid_argument("gene_starts / gene_ends must match gene_chr_ids");
    }
    if (static_cast<std::size_t>(stat_out_arr.shape(0)) != n_genes
        || static_cast<std::size_t>(p_out_arr.shape(0)) != n_genes
        || static_cast<std::size_t>(n_snps_out_arr.shape(0)) != n_genes) {
        throw std::invalid_argument("output arrays must match n_genes");
    }

    const int64_t* sorted_positions = sorted_positions_arr.data();
    const double* sorted_chi2 = sorted_chi2_arr.data();
    const int64_t* chr_offsets = chr_offsets_arr.data();
    const int64_t* gene_chr_ids = gene_chr_ids_arr.data();
    const int64_t* gene_starts = gene_starts_arr.data();
    const int64_t* gene_ends = gene_ends_arr.data();
    double* stat_out = stat_out_arr.mutable_data();
    double* p_out = p_out_arr.mutable_data();
    int64_t* n_snps_out = n_snps_out_arr.mutable_data();

    constexpr double sqrt_2 = 1.4142135623730951;

    {
        py::gil_scoped_release release;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t g = 0; g < static_cast<std::ptrdiff_t>(n_genes); ++g) {
            const int64_t c = gene_chr_ids[g];
            if (c < 0 || static_cast<std::size_t>(c) >= n_chr) {
                stat_out[g] = 0.0;
                p_out[g] = 1.0;
                n_snps_out[g] = 0;
                continue;
            }
            const int64_t seg_start = chr_offsets[c];
            const int64_t seg_end = chr_offsets[c + 1];
            if (seg_start >= seg_end) {
                stat_out[g] = 0.0;
                p_out[g] = 1.0;
                n_snps_out[g] = 0;
                continue;
            }
            const int64_t gstart = gene_starts[g];
            const int64_t gend = gene_ends[g];

            // Binary search for the first SNP with pos >= gstart and the
            // first SNP with pos > gend within [seg_start, seg_end).
            const int64_t* seg_begin = sorted_positions + seg_start;
            const int64_t* seg_finish = sorted_positions + seg_end;
            const int64_t* lo = std::lower_bound(seg_begin, seg_finish, gstart);
            const int64_t* hi = std::upper_bound(lo, seg_finish, gend);
            const std::ptrdiff_t idx_lo = lo - sorted_positions;
            const std::ptrdiff_t idx_hi = hi - sorted_positions;
            const std::ptrdiff_t k = idx_hi - idx_lo;

            if (k == 0) {
                stat_out[g] = 0.0;
                p_out[g] = 1.0;
                n_snps_out[g] = 0;
                continue;
            }

            double sum_chi2 = 0.0;
            for (std::ptrdiff_t i = idx_lo; i < idx_hi; ++i) {
                sum_chi2 += sorted_chi2[i];
            }
            const double mean_chi2 = sum_chi2 / static_cast<double>(k);
            const double z = (mean_chi2 - 1.0) / std::sqrt(2.0 / static_cast<double>(k));
            stat_out[g] = mean_chi2;
            p_out[g] = normal_sf(z);
            n_snps_out[g] = static_cast<int64_t>(k);
        }
    }
}

}  // namespace

PYBIND11_MODULE(_snp_to_gene_native, m) {
    m.doc() = "Native C++ accelerator for the MAGMA-style snp_to_gene "
              "per-gene window scan + mean-chi² p-value.";
    m.def("snp_to_gene_scan", &snp_to_gene_scan,
          py::arg("sorted_positions"),
          py::arg("sorted_chi2"),
          py::arg("chr_offsets"),
          py::arg("gene_chr_ids"),
          py::arg("gene_starts"),
          py::arg("gene_ends"),
          py::arg("stat_out"),
          py::arg("p_out"),
          py::arg("n_snps_out"),
          "Walk all genes in parallel: binary-search the chromosome's "
          "sorted SNP segment for [gene_start, gene_end] (window already "
          "applied by the Python caller), accumulate mean χ², compute "
          "z = (μ−1)/√(2/k) and p = ½ erfc(z/√2). All outputs written "
          "in place; genes with no in-window SNPs get stat=0, p=1, "
          "n_snps=0 (matches Python convention).");
}
