#' Install the TorchGenomics Python runtime.
#'
#' Creates a managed virtualenv (or conda env) named `r-rtorchgenomics`
#' and installs `torchgenomics[mcp]` pinned to the matching R package
#' version. Idempotent: re-running upgrades existing installs.
#'
#' @param method `"virtualenv"` (default) or `"conda"`.
#' @param python_version Python version to install if none is available
#'   on the system. Default `"3.12"`.
#' @param force If `TRUE`, recreate the venv from scratch.
#' @export
#' @examples
#' \dontrun{
#' tg_install()
#' }
tg_install <- function(method = c("virtualenv", "conda"),
                       python_version = "3.12",
                       force = FALSE) {
  method <- match.arg(method)
  venv <- .managed_venv_name()
  target <- paste0("torchgenomics[mcp]==", utils::packageVersion("rTorchGenomics"))

  if (force) {
    if (method == "virtualenv" && reticulate::virtualenv_exists(venv)) {
      message("Removing existing venv: ", venv)
      reticulate::virtualenv_remove(venv, confirm = FALSE)
    }
    if (method == "conda" && reticulate::condaenv_exists(venv)) {
      message("Removing existing conda env: ", venv)
      reticulate::conda_remove(envname = venv)
    }
  }

  if (method == "virtualenv") {
    if (!reticulate::virtualenv_exists(venv)) {
      message("Creating venv: ", venv)
      python_path <- if (nzchar(Sys.which("python3"))) {
        Sys.which("python3")
      } else if (nzchar(Sys.which("python"))) {
        Sys.which("python")
      } else {
        message("No system Python found; installing Python ", python_version,
                " via reticulate...")
        reticulate::install_python(version = python_version)
      }
      reticulate::virtualenv_create(envname = venv, python = python_path)
    }
    message("Installing ", target, " into ", venv, "...")
    reticulate::py_install(
      packages = target, envname = venv, pip = TRUE,
      method = "virtualenv"
    )
  } else {
    if (!reticulate::condaenv_exists(venv)) {
      reticulate::conda_create(envname = venv, python_version = python_version)
    }
    reticulate::py_install(
      packages = target, envname = venv, pip = TRUE,
      method = "conda"
    )
  }

  # Reset cached module so the next call re-imports.
  .pkg_env$py_module <- NULL

  message("Setup complete. tg_*() calls are now ready.")
  invisible(TRUE)
}
