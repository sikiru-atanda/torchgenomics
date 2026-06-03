#' @keywords internal
"_PACKAGE"

.pkg_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Lazy — don't import Python here. Just record the preferred venv
  # name; reticulate consults this when imports happen.
  venv <- "r-rtorchgenomics"
  if (requireNamespace("reticulate", quietly = TRUE) &&
      reticulate::virtualenv_exists(venv)) {
    reticulate::use_virtualenv(venv, required = FALSE)
  }
  invisible(NULL)
}

.onAttach <- function(libname, pkgname) {
  packageStartupMessage(
    "rTorchGenomics v", utils::packageVersion("rTorchGenomics"),
    " loaded. Run tg_install() once to set up the Python runtime."
  )
}
