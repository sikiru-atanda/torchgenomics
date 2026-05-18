"""Convert numpy .npz fixture to an R .rds list-of-lists.

Used by run_reference.R to avoid a Python dependency in the R session.
Output structure (saved via saveRDS()):
  list(rep_0 = list(G = matrix, y = vector), rep_1 = ..., ...)

Pure stdlib approach: dump per-replicate .npy files to a tmp dir, then have
Rscript read them back via a small pure-R .npy parser.  No extra PyPI deps.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="replicates.npz path")
    ap.add_argument("--output", required=True, help="replicates.rds output path")
    args = ap.parse_args()

    npz = np.load(args.input)
    keys = list(npz.files)
    rep_indices = sorted({int(k.split("_")[-1]) for k in keys if k.startswith("G_rep_")})

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for r in rep_indices:
            np.save(td_path / f"G_{r}.npy", npz[f"G_rep_{r}"])
            np.save(td_path / f"y_{r}.npy", npz[f"y_rep_{r}"])

        helper_r = td_path / "_combine.R"
        helper_r.write_text(R_HELPER)

        rc = subprocess.run(
            ["Rscript", "--vanilla", str(helper_r), str(td_path), args.output,
             ",".join(str(r) for r in rep_indices)],
            check=False,
        ).returncode
        if rc != 0:
            raise SystemExit(f"R combine step failed (rc={rc})")

    print(f"[npz->rds] wrote {args.output} ({len(rep_indices)} replicates)")


R_HELPER = r"""#!/usr/bin/env Rscript
# Pure-R reader for numpy v1/v2 .npy files.  Handles only the dtypes we
# actually produce: float64 little-endian dense arrays.

args <- commandArgs(trailingOnly = TRUE)
td <- args[[1]]
out_rds <- args[[2]]
rep_ids <- as.integer(strsplit(args[[3]], ",")[[1]])

read_npy <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    magic <- readBin(con, what = raw(), n = 6L)
    stopifnot(identical(magic[2:6], charToRaw("NUMPY")))
    ver_major <- as.integer(readBin(con, what = raw(), n = 1L))
    ver_minor <- as.integer(readBin(con, what = raw(), n = 1L))
    if (ver_major == 1L) {
        hdr_len <- readBin(con, integer(), size = 2L, endian = "little", signed = FALSE)
    } else {
        hdr_len <- readBin(con, integer(), size = 4L, endian = "little", signed = FALSE)
    }
    header <- rawToChar(readBin(con, what = raw(), n = hdr_len))
    descr <- regmatches(header, regexec("descr.*?([<|=][a-z][0-9]+)", header))[[1]][2]
    fortran <- grepl("fortran_order.*?True", header)
    shape_str <- regmatches(header, regexec("shape.*?\\(([^)]*)\\)", header))[[1]][2]
    shape <- as.integer(strsplit(gsub(" ", "", shape_str), ",")[[1]])
    shape <- shape[!is.na(shape)]
    n_el <- prod(shape)
    stopifnot(descr %in% c("<f8", "|f8", "=f8"))
    vals <- readBin(con, double(), n = n_el, size = 8L, endian = "little")
    if (length(shape) == 1L) return(vals)
    if (length(shape) == 2L) {
        if (fortran) return(matrix(vals, nrow = shape[1], ncol = shape[2]))
        return(matrix(vals, nrow = shape[1], ncol = shape[2], byrow = TRUE))
    }
    array(vals, dim = shape)
}

reps <- list()
for (r in rep_ids) {
    G <- read_npy(file.path(td, sprintf("G_%d.npy", r)))
    y <- as.numeric(read_npy(file.path(td, sprintf("y_%d.npy", r))))
    reps[[sprintf("rep_%d", r)]] <- list(G = G, y = y)
}
saveRDS(reps, out_rds)
"""


if __name__ == "__main__":
    main()

