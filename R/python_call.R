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
