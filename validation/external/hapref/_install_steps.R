# _install_steps.R -- driver for install.sh actions.
# Modes: check | install | smoke | marker

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("_install_steps.R requires a mode arg (check|install|smoke|marker)")
mode <- args[1]

hap_target  <- Sys.getenv("HAPLO_STATS_VERSION", "1.9.8.7")
json_target <- Sys.getenv("JSONLITE_VERSION",    "2.0.0")

check_loadable <- function() {
    pkgs <- c("haplo.stats", "jsonlite")
    ok <- TRUE
    for (p in pkgs) {
        if (!requireNamespace(p, quietly = TRUE)) {
            cat("missing:", p, "\n"); ok <- FALSE
        }
    }
    if (!ok) quit(status = 1)
    invisible(TRUE)
}

if (mode == "check") {
    check_loadable()
    cat("haplo.stats:", as.character(packageVersion("haplo.stats")), "\n")
    cat("jsonlite:   ", as.character(packageVersion("jsonlite")),    "\n")
    quit(status = 0)
}

if (mode == "install") {
    options(repos = c(CRAN = "https://cloud.r-project.org"))
    if (!requireNamespace("remotes", quietly = TRUE)) {
        install.packages("remotes", quiet = TRUE)
    }
    cur_hap <- if (requireNamespace("haplo.stats", quietly = TRUE))
        as.character(packageVersion("haplo.stats")) else NA
    cur_json <- if (requireNamespace("jsonlite", quietly = TRUE))
        as.character(packageVersion("jsonlite")) else NA
    if (is.na(cur_hap) || cur_hap != hap_target) {
        cat(sprintf("[install] haplo.stats %s -> %s\n", cur_hap, hap_target))
        remotes::install_version("haplo.stats", version = hap_target,
                                 repos = "https://cloud.r-project.org",
                                 upgrade = "never", quiet = TRUE)
    } else {
        cat(sprintf("[install] haplo.stats already at %s\n", cur_hap))
    }
    if (is.na(cur_json) || cur_json != json_target) {
        cat(sprintf("[install] jsonlite %s -> %s\n", cur_json, json_target))
        remotes::install_version("jsonlite", version = json_target,
                                 repos = "https://cloud.r-project.org",
                                 upgrade = "never", quiet = TRUE)
    } else {
        cat(sprintf("[install] jsonlite already at %s\n", cur_json))
    }
    quit(status = 0)
}

if (mode == "smoke") {
    suppressPackageStartupMessages({
        library(haplo.stats)
        library(jsonlite)
    })
    set.seed(0)
    n <- 50
    m <- 3
    geno <- matrix(NA, n, 2 * m)
    for (j in seq_len(m)) {
        a <- sample(c("A", "C"), n, replace = TRUE, prob = c(0.6, 0.4))
        b <- sample(c("A", "C"), n, replace = TRUE, prob = c(0.6, 0.4))
        geno[, 2 * j - 1] <- a
        geno[, 2 * j]     <- b
    }
    em <- haplo.em(geno = geno, locus.label = paste0("rs", seq_len(m)))
    stopifnot(sum(em$hap.prob) > 0.999 && sum(em$hap.prob) < 1.001)
    cat("[smoke] PASS: haplo.em returned", length(em$hap.prob),
        "haplotypes; sum(freq) ~ 1\n")
    quit(status = 0)
}

if (mode == "marker") {
    cat(sprintf("R_VERSION=%s\n", R.version.string))
    cat(sprintf("HAPLO_STATS_VERSION=%s\n", as.character(packageVersion("haplo.stats"))))
    cat(sprintf("JSONLITE_VERSION=%s\n", as.character(packageVersion("jsonlite"))))
    quit(status = 0)
}

stop(sprintf("unknown mode: %s", mode))

