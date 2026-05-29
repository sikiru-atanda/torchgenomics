"""Build script for native C++ extensions.

The native extensions are *optional*: if a C++17 compiler is not available,
the install still succeeds and the pure-Python reference implementations in
``torchgwas/pgs/`` are used at runtime. Set ``TORCHGWAS_DISABLE_NATIVE=1`` in
the environment to force the Python paths even when the extension is built.
"""
from __future__ import annotations

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# --- OpenMP support ----------------------------------------------------------
# Phase 41af: per-extension opt-in OpenMP. Set TORCHGWAS_DISABLE_OPENMP=1 in
# the environment to force a serial build (useful on toolchains where OpenMP
# is missing or misconfigured). When enabled, extensions listed in
# OPENMP_EXTENSIONS get the platform-appropriate OpenMP compile/link flags.
_DISABLE_OMP = bool(os.environ.get("TORCHGWAS_DISABLE_OPENMP"))
if sys.platform == "win32":
    _OMP_CFLAGS = ["/openmp"]
    _OMP_LFLAGS: list[str] = []
elif sys.platform == "darwin":
    # Apple Clang needs libomp + -Xpreprocessor; users on mac can set
    # TORCHGWAS_DISABLE_OPENMP=1 if libomp isn't available.
    _OMP_CFLAGS = ["-Xpreprocessor", "-fopenmp"]
    _OMP_LFLAGS = ["-lomp"]
else:
    _OMP_CFLAGS = ["-fopenmp"]
    _OMP_LFLAGS = ["-fopenmp"]


def _with_omp(ext: Pybind11Extension) -> Pybind11Extension:
    if _DISABLE_OMP:
        return ext
    ext.extra_compile_args = list(ext.extra_compile_args or []) + _OMP_CFLAGS
    ext.extra_link_args = list(ext.extra_link_args or []) + _OMP_LFLAGS
    return ext

ext_modules = [
    Pybind11Extension(
        "torchgwas._native._prscs_native",
        sources=[
            "csrc/pgs/gig_sampler.cpp",
            "csrc/pgs/prscs_gibbs.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,  # build failure -> install proceeds, fallback active
    ),
    Pybind11Extension(
        "torchgwas._native._ldpred2_native",
        sources=[
            "csrc/pgs/ldpred2_gibbs.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._ct_native",
        sources=[
            "csrc/pgs/ct_clump.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._pelt_native",
        sources=[
            "csrc/ld/pelt_changepoint.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._ess_native",
        sources=[
            "csrc/stats/ess_geyer.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._graph_native",
        sources=[
            "csrc/ld/graph_utils.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._gabriel_native",
        sources=[
            "csrc/ld/gabriel_blocks.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._big_ld_native",
        sources=[
            "csrc/ld/big_ld.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._dp_optimize_native",
        sources=[
            "csrc/ld/dp_optimize.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._cc_graph_native",
        sources=[
            "csrc/ld/cc_graph_adj.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._gwas_aligned_native",
        sources=[
            "csrc/ld/gwas_aligned.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._spine_native",
        sources=[
            "csrc/ld/spine.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._ld_decay_signal_native",
        sources=[
            "csrc/ld/ld_decay_signal.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._greedy_mwis_native",
        sources=[
            "csrc/ld/greedy_mwis.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._uncertainty_blocks_native",
        sources=[
            "csrc/ld/uncertainty_blocks.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._wall_pritchard_native",
        sources=[
            "csrc/ld/wall_pritchard_perm.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._impute_mode_native",
        sources=[
            "csrc/preprocess/impute_mode.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._impute_knn_native",
        sources=[
            "csrc/preprocess/impute_knn.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._impute_ld_native",
        sources=[
            "csrc/preprocess/impute_ld.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._cavi_native",
        sources=[
            "csrc/models/cavi_sweep.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._ibss_native",
        sources=[
            "csrc/models/susie_rss_ibss.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._mediate_native",
        sources=[
            "csrc/multiomics/mediate_sigma_blocks.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._spa_native",
        sources=[
            "csrc/stats/spa_lugannani_rice.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._ldsc_native",
        sources=[
            "csrc/postgwas/ldsc_jackknife.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._hwe_native",
        sources=[
            "csrc/preprocess/hwe.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
    Pybind11Extension(
        "torchgwas._native._cross_pop_native",
        sources=[
            "csrc/ld/cross_pop_stability.cpp",
        ],
        include_dirs=["csrc"],
        cxx_std=17,
        optional=True,
    ),
]

_OMP_TARGETS = {
    "torchgwas._native._hwe_native",
    "torchgwas._native._impute_knn_native",
    "torchgwas._native._impute_mode_native",
    "torchgwas._native._impute_ld_native",
    "torchgwas._native._spa_native",
    "torchgwas._native._dp_optimize_native",
    "torchgwas._native._mediate_native",
}
for _ext in ext_modules:
    if _ext.name in _OMP_TARGETS:
        _with_omp(_ext)

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
