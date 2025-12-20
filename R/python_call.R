#' Execute a Python script shipped with the STREAMS package
#'
#' This helper runs a Python script located in the package \code{inst/}
#' directory using a user-specified Python interpreter (default: \code{python3}).

#' This function is primarily intended for internal use by high-level
#' STREAMS workflows (e.g. model training and inference), but can also be
#' called directly for debugging purposes.
#'
#' @param script Character string. Name of the Python script located in
#'   the package \code{inst/} directory (e.g. \code{"train.py"}).
#' @param args Character vector. Command-line arguments passed to the
#'   Python script.
#' @param python Character string. Name or path of the Python interpreter
#'   to use. Defaults to \code{"python3"}.
#' @param pkg Character string. Name of the R package containing the
#'   Python script. Defaults to \code{"STREAMS"}.
#'
#' @return Invisibly returns a character vector containing the combined
#'   standard output and standard error produced by the Python process.
#'   If execution fails, an error is raised and the output is printed
#'   in the error message.
#'
#' @details
#' The function performs several safety checks to ensure robust execution:
#' \itemize{
#'   \item verifies that the script exists inside the installed package,
#'   \item checks that the requested Python executable is available,
#'   \item enforces the use of Python >= 3,
#'   \item safely quotes and normalizes file paths,
#'   \item captures and propagates Python stdout/stderr on failure.
#' }
#'
#' If the Python process exits with a non-zero status, the function
#' stops with an informative error that includes the captured Python output.
#'
#' @examples
#' \dontrun{
#' streams_python(
#'   script = "train.py",
#'   args = c("model.pt", "data.feather", "train.feather", "val.feather",
#'            "x1,x2,x3", "--prior_aug", "beta"),
#'   python = "python3"
#' )
#' }
#'
#' @export
#'

streams_python <- function(
    script,
    args = character(),
    python = "python3",
    pkg = "STREAMS"
) {
  ...
}

streams_python <- function(
    script,
    args = character(),
    python = "python3",
    pkg = "STREAMS"
) {

  script_path <- system.file(script, package = pkg)
  if (!nzchar(script_path) || !file.exists(script_path)) {
    stop("Python script not found: ", script, call. = FALSE)
  }

  py <- Sys.which(python)
  if (!nzchar(py)) {
    stop("Python interpreter not found: ", python, call. = FALSE)
  }


  ver <- system2(py, "--version", stdout = TRUE, stderr = TRUE)
  if (any(grepl("Python 2", ver))) {
    stop("Python >= 3 required. Found: ", ver, call. = FALSE)
  }


  q <- function(x) shQuote(normalizePath(x, winslash = "/", mustWork = FALSE))

  cmd_args <- c(
    q(script_path),
    vapply(args, q, character(1))
  )


  out <- system2(py, args = cmd_args, stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status")

  if (!is.null(status) && status != 0) {
    stop(
      "Python execution failed (exit ", status, ")\n\n",
      paste(out, collapse = "\n"),
      call. = FALSE
    )
  }

  invisible(out)
}
