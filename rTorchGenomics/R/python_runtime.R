#' Locate the managed Python environment.
#'
#' Returns the name of the rTorchGenomics-managed virtualenv.
#' @keywords internal
.managed_venv_name <- function() "r-rtorchgenomics"

#' Return the cached `torchgenomics` Python module reference.
#'
#' First call imports `torchgenomics`; subsequent calls return the
#' cached module. Throws `tg_no_python` if Python isn't configured.
#'
#' @return A reticulate module reference.
#' @keywords internal
tg_py <- function() {
  if (!is.null(.pkg_env$py_module)) {
    return(.pkg_env$py_module)
  }
  venv <- .managed_venv_name()
  if (!reticulate::virtualenv_exists(venv)) {
    stop(
      structure(
        class = c("tg_no_python", "error", "condition"),
        list(message = paste0(
          "TorchGenomics Python runtime not configured. ",
          "Run tg_install() to set it up."
        ))
      )
    )
  }
  reticulate::use_virtualenv(venv, required = TRUE)
  mod <- tryCatch(
    reticulate::import("torchgenomics"),
    error = function(e) stop(
      structure(
        class = c("tg_no_python", "error", "condition"),
        list(message = paste0(
          "torchgenomics not importable in venv '", venv, "': ",
          conditionMessage(e), ". Try tg_install(force = TRUE)."
        ))
      )
    )
  )
  .pkg_env$py_module <- mod
  mod
}

`%||%` <- function(x, y) if (is.null(x)) y else x
