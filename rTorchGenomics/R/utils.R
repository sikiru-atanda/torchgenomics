#' Strip NULL-valued entries from a list (for arg passthrough).
#' @keywords internal
.compact <- function(x) x[!vapply(x, is.null, logical(1))]

#' Coerce a path-like input to a normalised character path.
#' @keywords internal
.as_path <- function(x) {
  if (is.null(x)) return(NULL)
  as.character(x)
}

# Null-coalescing operator (internal helper; no roxygen tag — avoids an
# Rd file whose \name contains '%' which R CMD check flags).
`%||%` <- function(x, y) if (is.null(x)) y else x
