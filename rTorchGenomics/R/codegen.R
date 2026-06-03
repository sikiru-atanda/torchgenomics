# rTorchGenomics/R/codegen.R
#
# Code-generation utilities. `generate_api_auto()` consumes
# `inst/rbridge_manifest.json` (emitted by `python -m
# torchgenomics._manifest`) and produces `R/api_auto.R` — one R wrapper
# per tier-2 CLI subcommand. The generated file is checked in; users
# never run codegen. Re-run it whenever the manifest changes.

#' Generate R/api_auto.R from the rbridge manifest.
#'
#' Run at package build time by the developer; the generated file is
#' checked in. Users do not run this.
#'
#' @param manifest_path Path to the JSON manifest. Default: the
#'   package's bundled `inst/rbridge_manifest.json`.
#' @param output_path Where to write the generated R. Default
#'   `R/api_auto.R`.
#' @return The output path, invisibly.
#' @export
#' @examples
#' \dontrun{
#' generate_api_auto()
#' }
generate_api_auto <- function(manifest_path = "inst/rbridge_manifest.json",
                              output_path = "R/api_auto.R") {
  manifest <- jsonlite::fromJSON(manifest_path, simplifyVector = FALSE)

  preamble <- c(
    "# rTorchGenomics/R/api_auto.R",
    "#",
    paste0("# AUTO-GENERATED FROM inst/rbridge_manifest.json ",
           "(torchgenomics ", manifest$torchgenomics_version, ")"),
    "# DO NOT EDIT BY HAND. Re-run generate_api_auto() to regenerate.",
    "",
    ""
  )

  blocks <- vapply(manifest$commands, .render_wrapper, character(1L))
  writeLines(c(preamble, blocks), output_path)
  invisible(output_path)
}

# Render an R-literal for an argparse default value. NULL -> "NULL".
.r_default_literal <- function(type, default) {
  if (is.null(default)) return("NULL")
  switch(type,
    bool   = if (isTRUE(default)) "TRUE" else "FALSE",
    int    = paste0(as.character(as.integer(default)), "L"),
    float  = formatC(as.numeric(default), digits = 6, format = "g"),
    str    = paste0("\"", gsub("\"", "\\\\\"", default), "\""),
    paste0("\"", gsub("\"", "\\\\\"", default), "\"")
  )
}

# Detect whether an arg is variable-length (nargs '*' or '+'). The
# argparse `store_true` / `store_false` flags carry nargs=0 (an
# integer); those are scalar booleans and must NOT be treated as
# lists.
.is_list_arg <- function(arg) {
  n <- arg$nargs
  if (is.null(n)) return(FALSE)
  if (is.character(n)) return(n %in% c("*", "+"))
  FALSE
}

# Render a single argument's signature entry: `name = default` or `name`.
.r_arg_signature <- function(arg) {
  default_lit <- if (isTRUE(arg$required)) {
                   ""
                 } else {
                   paste0(" = ", .r_default_literal(arg$type, arg$default))
                 }
  paste0(arg$name, default_lit)
}

# Render an argument's coercion expression for the .compact list. We
# differentiate by type and by nargs:
#   - nargs in {*, +} -> treat as character vector / list of strings;
#     don't .as_path-collapse.
#   - type=int   -> as.integer
#   - type=float -> as.numeric
#   - type=bool  -> as.logical
#   - type=str + no nargs -> .as_path()
#   - choices set -> wrap with match.arg() so R-side validation matches Python's
.r_arg_coercion <- function(arg) {
  is_list <- .is_list_arg(arg)
  # Drop NULL entries from choices (e.g. argparse choices=[None, 'r2_model']
  # in combine-gwas-twas where None signals an optional default).
  str_choices <- if (is.null(arg$choices)) {
                   character(0)
                 } else {
                   keep <- vapply(arg$choices,
                                  function(c) !is.null(c) && is.character(c),
                                  logical(1L))
                   vapply(arg$choices[keep], as.character, character(1L))
                 }
  has_choices <- length(str_choices) > 0L

  # match.arg expression for choices (only meaningful for string args).
  # int-typed choices (e.g. phase-poly --ploidy {2,4,6}) fall through to
  # as.integer coercion; the Python side enforces the constraint at the
  # bridge layer.
  if (has_choices && identical(arg$type, "str") && !is_list) {
    choices_lit <- paste0("c(",
                          paste0("\"", str_choices, "\"", collapse = ", "),
                          ")")
    return(paste0(arg$name,
                  " = if (is.null(", arg$name, ")) NULL else match.arg(",
                  arg$name, ", ", choices_lit, ")"))
  }

  rhs <- switch(arg$type,
    int   = paste0("if (is.null(", arg$name, ")) NULL else as.integer(",
                   arg$name, ")"),
    float = paste0("if (is.null(", arg$name, ")) NULL else as.numeric(",
                   arg$name, ")"),
    bool  = paste0("if (is.null(", arg$name, ")) NULL else as.logical(",
                   arg$name, ")"),
    str   = if (is_list) {
              # Pass a character vector through unchanged.
              paste0("if (is.null(", arg$name, ")) NULL else as.character(",
                     arg$name, ")")
            } else {
              paste0(".as_path(", arg$name, ")")
            },
    arg$name
  )
  paste0(arg$name, " = ", rhs)
}

# Build a one-line @param roxygen entry for a single argument. The
# help text from argparse may contain newlines or backticks; we strip
# / escape them so the resulting Rd is well-formed.
.r_param_doc <- function(arg) {
  help <- trimws(arg$help %||% "")
  # Argparse help often contains literal escape sequences and
  # Rd-hostile punctuation. Normalise:
  #   - collapse whitespace + literal newlines / tabs
  #   - replace ALL backslashes (Rd escape char; would otherwise become
  #     spurious \t / \n / \link{...} commands)
  #   - escape unescaped percent signs (Rd treats them as comments)
  help <- gsub("\\s+", " ", help)
  help <- gsub("\\\\", "/", help)
  help <- gsub("%", " percent", help)
  # Roxygen2 (markdown mode) converts [token] into \link{token}; sanitise
  # square brackets so help text like 'chr/t...[gene_name]' doesn't end up
  # as a broken Rd \link.
  help <- gsub("\\[", "(", help)
  help <- gsub("\\]", ")", help)
  if (!nzchar(help)) {
    help <- paste0("argparse-derived ", arg$type,
                   if (isTRUE(arg$required)) " (required)." else " parameter.")
  } else if (!grepl("\\.$", help)) {
    help <- paste0(help, ".")
  }
  paste0("#' @param ", arg$name, " ", help)
}

# Render the full R function block for one command.
.render_wrapper <- function(cmd) {
  sig_lines <- vapply(cmd$args, .r_arg_signature, character(1L))
  arg_lines <- vapply(cmd$args, .r_arg_coercion, character(1L))
  param_lines <- vapply(cmd$args, .r_param_doc, character(1L))
  sig <- paste(sig_lines, collapse = ",\n                              ")
  coerce <- paste(arg_lines, collapse = ",\n    ")
  params <- paste(param_lines, collapse = "\n")

  desc <- trimws(cmd$description %||% "")
  if (!nzchar(desc)) desc <- paste0(cmd$cli_subcommand, " (auto-generated)")

  paste0(
    "#' ", desc, "\n",
    "#'\n",
    "#' Auto-generated from torchgenomics `", cmd$cli_subcommand,
    "` CLI subcommand.\n",
    "#'\n",
    params, "\n",
    "#'\n",
    "#' @return A `GwasResult` S4 object.\n",
    "#' @export\n",
    "tg_", cmd$name, " <- function(", sig, ") {\n",
    "  args <- .compact(list(\n    ", coerce, "\n  ))\n",
    "  d <- bridge_call(\"", cmd$name, "\", args)\n",
    "  GwasResult$new_from_dict(d, command = \"", cmd$name, "\")\n",
    "}\n"
  )
}
