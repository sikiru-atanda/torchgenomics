#' Strip NULL-valued entries from a list (for arg passthrough).
#' @keywords internal
.compact <- function(x) x[!vapply(x, is.null, logical(1))]

#' Coerce a path-like input to a normalised character path.
#' @keywords internal
.as_path <- function(x) {
  if (is.null(x)) return(NULL)
  as.character(x)
}
