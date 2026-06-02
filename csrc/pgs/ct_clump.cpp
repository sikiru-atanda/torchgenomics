// Greedy LD clumping for the C+T (Clumping + Thresholding) PGS method.
//
// Mirrors torchgenomics.pgs.ct._clump_with_ld_reference. Two entry points are
// exposed:
//
//   clump_full(p, R, chr_codes, pos, p_thr, r2_thr, window_bp)
//   clump_block(p, R_blocks, block_index, local_index,
//               chr_codes, pos, p_thr, r2_thr, window_bp)
//
// Both return (index_snps, member_offsets, members):
//   - index_snps: int64 array of retained index SNP positions
//   - member_offsets: int64 array of length len(index_snps)+1
//   - members: int64 array of clump-member positions, where the members of
//     index_snps[k] are members[member_offsets[k]:member_offsets[k+1]].
//
// Algorithm: sort significant SNPs by ascending p, then greedily walk:
// for each candidate index SNP, absorb every still-available SNP on the
// same chromosome within window_bp whose r^2 with the index SNP exceeds
// the threshold.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct ClumpResult {
    std::vector<std::int64_t> index_snps;
    std::vector<std::int64_t> member_offsets;  // size = index_snps.size() + 1
    std::vector<std::int64_t> members;
};

// Greedy clumping core, parameterised on an r^2 lookup callable.
template <typename R2Fn>
ClumpResult run_greedy_clump(
    const double* p,
    std::size_t m,
    const std::int64_t* chr_codes,
    const std::int64_t* pos,
    double p_threshold,
    double r2_threshold,
    double window_bp,
    R2Fn&& r2_of
) {
    ClumpResult res;
    res.member_offsets.push_back(0);
    if (m == 0) return res;

    // Significant SNPs in ascending p-value order.
    std::vector<std::int64_t> sig;
    sig.reserve(m);
    for (std::size_t i = 0; i < m; ++i) {
        if (p[i] < p_threshold) sig.push_back(static_cast<std::int64_t>(i));
    }
    if (sig.empty()) return res;
    std::sort(sig.begin(), sig.end(),
              [&](std::int64_t a, std::int64_t b) { return p[a] < p[b]; });

    // Bucket SNPs by chromosome code for window scans.
    std::unordered_map<std::int64_t, std::vector<std::int64_t>> chr_to_idx;
    chr_to_idx.reserve(8);
    for (std::size_t i = 0; i < m; ++i) {
        chr_to_idx[chr_codes[i]].push_back(static_cast<std::int64_t>(i));
    }

    std::vector<char> available(m, 1);

    for (std::int64_t idx : sig) {
        if (!available[static_cast<std::size_t>(idx)]) continue;
        res.index_snps.push_back(idx);
        available[static_cast<std::size_t>(idx)] = 0;

        const std::int64_t chrom = chr_codes[idx];
        const std::int64_t pos_idx = pos[idx];
        auto it = chr_to_idx.find(chrom);
        if (it == chr_to_idx.end()) {
            res.member_offsets.push_back(static_cast<std::int64_t>(res.members.size()));
            continue;
        }
        for (std::int64_t other : it->second) {
            if (other == idx) continue;
            if (!available[static_cast<std::size_t>(other)]) continue;
            const double dist = std::abs(static_cast<double>(pos[other] - pos_idx));
            if (dist > window_bp) continue;
            const double r2 = r2_of(idx, other);
            if (r2 >= r2_threshold) {
                res.members.push_back(other);
                available[static_cast<std::size_t>(other)] = 0;
            }
        }
        res.member_offsets.push_back(static_cast<std::int64_t>(res.members.size()));
    }
    return res;
}

py::tuple build_tuple(const ClumpResult& res) {
    py::array_t<std::int64_t> idx_arr(static_cast<py::ssize_t>(res.index_snps.size()));
    py::array_t<std::int64_t> off_arr(static_cast<py::ssize_t>(res.member_offsets.size()));
    py::array_t<std::int64_t> mem_arr(static_cast<py::ssize_t>(res.members.size()));
    if (!res.index_snps.empty()) {
        std::copy(res.index_snps.begin(), res.index_snps.end(),
                  static_cast<std::int64_t*>(idx_arr.request().ptr));
    }
    std::copy(res.member_offsets.begin(), res.member_offsets.end(),
              static_cast<std::int64_t*>(off_arr.request().ptr));
    if (!res.members.empty()) {
        std::copy(res.members.begin(), res.members.end(),
                  static_cast<std::int64_t*>(mem_arr.request().ptr));
    }
    return py::make_tuple(idx_arr, off_arr, mem_arr);
}

py::tuple py_clump_full(
    py::array_t<double, py::array::c_style | py::array::forcecast> p,
    py::array_t<double, py::array::c_style | py::array::forcecast> R,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_codes,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pos,
    double p_threshold,
    double r2_threshold,
    double window_bp
) {
    auto p_buf = p.request();
    auto R_buf = R.request();
    auto chr_buf = chr_codes.request();
    auto pos_buf = pos.request();
    if (p_buf.ndim != 1) throw std::invalid_argument("p must be 1-D");
    if (R_buf.ndim != 2) throw std::invalid_argument("R must be 2-D");
    const std::size_t m = static_cast<std::size_t>(p_buf.shape[0]);
    if (R_buf.shape[0] != static_cast<py::ssize_t>(m) ||
        R_buf.shape[1] != static_cast<py::ssize_t>(m)) {
        throw std::invalid_argument("R must be (m, m) matching p");
    }
    if (chr_buf.shape[0] != static_cast<py::ssize_t>(m) ||
        pos_buf.shape[0] != static_cast<py::ssize_t>(m)) {
        throw std::invalid_argument("chr_codes/pos must be length m");
    }

    const double* p_ptr = static_cast<const double*>(p_buf.ptr);
    const double* R_ptr = static_cast<const double*>(R_buf.ptr);
    const std::int64_t* chr_ptr = static_cast<const std::int64_t*>(chr_buf.ptr);
    const std::int64_t* pos_ptr = static_cast<const std::int64_t*>(pos_buf.ptr);

    auto r2_of = [R_ptr, m](std::int64_t i, std::int64_t j) -> double {
        const double r = R_ptr[static_cast<std::size_t>(i) * m +
                               static_cast<std::size_t>(j)];
        return r * r;
    };

    ClumpResult res;
    {
        py::gil_scoped_release release;
        res = run_greedy_clump(p_ptr, m, chr_ptr, pos_ptr,
                               p_threshold, r2_threshold, window_bp, r2_of);
    }
    return build_tuple(res);
}

py::tuple py_clump_block(
    py::array_t<double, py::array::c_style | py::array::forcecast> p,
    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> R_blocks,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> block_index,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> local_index,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> chr_codes,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> pos,
    double p_threshold,
    double r2_threshold,
    double window_bp
) {
    auto p_buf = p.request();
    const std::size_t m = static_cast<std::size_t>(p_buf.shape[0]);
    if (block_index.shape(0) != static_cast<py::ssize_t>(m) ||
        local_index.shape(0) != static_cast<py::ssize_t>(m) ||
        chr_codes.shape(0)  != static_cast<py::ssize_t>(m) ||
        pos.shape(0)        != static_cast<py::ssize_t>(m)) {
        throw std::invalid_argument("block_index/local_index/chr_codes/pos must be length m");
    }

    // Capture the raw block data once so the lambda is cheap.
    std::vector<const double*> block_ptrs;
    std::vector<std::size_t>   block_sizes;
    block_ptrs.reserve(R_blocks.size());
    block_sizes.reserve(R_blocks.size());
    for (const auto& Rb : R_blocks) {
        auto buf = Rb.request();
        if (buf.ndim != 2 || buf.shape[0] != buf.shape[1]) {
            throw std::invalid_argument("each R_block must be square 2-D");
        }
        block_ptrs.push_back(static_cast<const double*>(buf.ptr));
        block_sizes.push_back(static_cast<std::size_t>(buf.shape[0]));
    }

    const double* p_ptr        = static_cast<const double*>(p_buf.ptr);
    const std::int64_t* bi_ptr = static_cast<const std::int64_t*>(block_index.request().ptr);
    const std::int64_t* li_ptr = static_cast<const std::int64_t*>(local_index.request().ptr);
    const std::int64_t* chr_ptr = static_cast<const std::int64_t*>(chr_codes.request().ptr);
    const std::int64_t* pos_ptr = static_cast<const std::int64_t*>(pos.request().ptr);

    auto r2_of = [bi_ptr, li_ptr, &block_ptrs, &block_sizes]
                 (std::int64_t i, std::int64_t j) -> double {
        const std::int64_t bi = bi_ptr[i];
        const std::int64_t bj = bi_ptr[j];
        if (bi != bj) return 0.0;
        const std::size_t b = static_cast<std::size_t>(bi);
        const std::size_t bsz = block_sizes[b];
        const std::size_t li = static_cast<std::size_t>(li_ptr[i]);
        const std::size_t lj = static_cast<std::size_t>(li_ptr[j]);
        const double r = block_ptrs[b][li * bsz + lj];
        return r * r;
    };

    ClumpResult res;
    {
        py::gil_scoped_release release;
        res = run_greedy_clump(p_ptr, m, chr_ptr, pos_ptr,
                               p_threshold, r2_threshold, window_bp, r2_of);
    }
    return build_tuple(res);
}

}  // namespace

PYBIND11_MODULE(_ct_native, m) {
    m.doc() = "Native C++ accelerator for C+T greedy LD clumping.";

    m.def("clump_full", &py_clump_full,
          py::arg("p"), py::arg("R"),
          py::arg("chr_codes"), py::arg("pos"),
          py::arg("p_threshold"), py::arg("r2_threshold"), py::arg("window_bp"),
          "Greedy LD clumping with a full (m, m) LD correlation matrix.");

    m.def("clump_block", &py_clump_block,
          py::arg("p"),
          py::arg("R_blocks"),
          py::arg("block_index"),
          py::arg("local_index"),
          py::arg("chr_codes"),
          py::arg("pos"),
          py::arg("p_threshold"), py::arg("r2_threshold"), py::arg("window_bp"),
          "Greedy LD clumping with a block-diagonal LD reference.");
}
