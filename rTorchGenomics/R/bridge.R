#' Call a `torchgenomics.api` function via reticulate.
#'
#' @param fn_name Character. Name of the api function (e.g., "lmm_scan").
#' @param args Named list. Arguments to pass to the Python function.
#' @return The Python function's `to_dict()` result converted to R.
#' @keywords internal
bridge_call <- function(fn_name, args = list()) {
  py <- tryCatch(
    tg_py(),
    tg_no_python = function(e) stop(e)
  )

  py_api <- py$api
  py_fn <- py_api[[fn_name]]
  if (is.null(py_fn)) {
    stop(structure(
      class = c("tg_runtime_error", "error", "condition"),
      list(message = paste0("torchgenomics.api has no function '", fn_name, "'"))
    ))
  }

  tryCatch(
    {
      py_result <- do.call(py_fn, args)
      has_to_dict <- if (inherits(py_result, "python.builtin.object")) {
        reticulate::py_has_attr(py_result, "to_dict")
      } else {
        is.list(py_result) && is.function(py_result$to_dict)
      }
      if (has_to_dict) {
        reticulate::py_to_r(py_result$to_dict())
      } else {
        reticulate::py_to_r(py_result)
      }
    },
    python.builtin.ValueError = function(e) {
      stop(structure(
        class = c("tg_input_error", "error", "condition"),
        list(message = paste0(fn_name, "(): ", conditionMessage(e)))
      ))
    },
    python.builtin.TypeError = function(e) {
      stop(structure(
        class = c("tg_input_error", "error", "condition"),
        list(message = paste0(fn_name, "(): ", conditionMessage(e)))
      ))
    },
    python.builtin.FileNotFoundError = function(e) {
      stop(structure(
        class = c("tg_input_error", "error", "condition"),
        list(message = paste0(fn_name, "(): ", conditionMessage(e)))
      ))
    },
    python.builtin.Exception = function(e) {
      tb <- tryCatch(reticulate::py_last_error()$traceback, error = function(...) "")
      if (is.null(tb)) tb <- ""
      tb_str <- paste(as.character(tb), collapse = "\n")
      stop(structure(
        class = c("tg_runtime_error", "error", "condition"),
        list(message = paste0(
          fn_name, "(): ", conditionMessage(e),
          if (nzchar(tb_str)) paste0("\n", tb_str) else ""
        ))
      ))
    }
  )
}
