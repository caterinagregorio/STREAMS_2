#' Check input data structure and content
#'
#' This function validates that the input object is a `data.frame` / `data.table`
#' with the required columns and reasonable values. It:
#' - Checks that all required columns are present.
#' - Optionally keeps only required columns + selected covariates (drops everything else).
#' - Checks and (if possible) coerces column types.
#' - Checks that binary variables are 0/1.
#' - Warns about missing values and returns their locations as an attribute.
#' - Issues basic consistency warnings (e.g. negative times).
#'
#' @param data A `data.table` or `data.frame` that must contain the following columns:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#'   The `data.frame` can contain extra columns with covariate values.
#'
#' @param covariates Character vector of covariate names to keep in addition to
#'   the required columns. Any other column will be dropped.
#'
#' @return Invisibly returns `data` (possibly coerced to `data.table`) restricted
#'   to required columns + `covariates`, with an attribute `"na_index"` containing
#'   a `data.frame` with the locations of missing values in the kept columns
#'   (columns: `row`, `col`). If no missing values are present, the attribute is `NULL`.
#'
#' @importFrom utils head
#' @importFrom data.table as.data.table is.data.table
#' @export
check_input_data <- function(data, covariates = character(0)) {

  # ---- basic class check ----
  if (!inherits(data, "data.frame")) {
    stop(
      "Argument `data` must be a data.frame or data.table, not: ",
      paste(class(data), collapse = ", "),
      call. = FALSE
    )
  }

  # Use data.table internally if available
  if (requireNamespace("data.table", quietly = TRUE) &&
      !data.table::is.data.table(data)) {
    data <- data.table::as.data.table(data)
  }

  # ---- required columns ----
  required_cols <- c(
    "patient_id",
    "dead",
    "death_time",
    "onset",
    "onset_age",
    "age",
    "visits"
  )

  # ---- keep only required + covariates ----
  covariates <- unique(as.character(covariates))
  keep_cols  <- unique(c(required_cols, covariates))

  # check presence
  missing_cols <- setdiff(keep_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(
      "Input data is missing required columns / selected covariates: ",
      paste(missing_cols, collapse = ", "),
      call. = FALSE
    )
  }

  # subset columns (drop everything else)
  if (requireNamespace("data.table", quietly = TRUE) && data.table::is.data.table(data)) {
    data <- data[, ..keep_cols]
  } else {
    data <- data[, keep_cols, drop = FALSE]
  }

  # ---- type checks & coercions ----
  # numeric-like columns
  numeric_cols <- c("patient_id", "death_time", "onset_age", "age", "visits")
  for (col in numeric_cols) {
    x <- data[[col]]
    if (!is.numeric(x)) {
      warning(
        sprintf(
          "Column '%s' is of class '%s' but expected numeric. Attempting coercion via as.numeric().",
          col, paste(class(x), collapse = ", ")
        ),
        call. = FALSE
      )
      data[[col]] <- suppressWarnings(as.numeric(x))
    }
  }

  # binary columns (0/1)
  binary_cols <- c("dead", "onset")
  for (col in binary_cols) {
    x <- data[[col]]

    # coerce factors/characters if needed
    if (!is.numeric(x) && !is.integer(x)) {
      warning(
        sprintf(
          "Column '%s' is of class '%s' but expected binary numeric (0/1). Attempting coercion via as.numeric().",
          col, paste(class(x), collapse = ", ")
        ),
        call. = FALSE
      )
      data[[col]] <- suppressWarnings(as.numeric(x))
      x <- data[[col]]
    }

    vals <- unique(x)
    vals <- vals[!is.na(vals)]
    if (!all(vals %in% c(0, 1))) {
      warning(
        sprintf(
          "Column '%s' contains values other than 0/1: %s",
          col, paste(sort(vals), collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }

  # ---- missing values ----
  na_mat <- is.na(data)
  if (any(na_mat)) {
    where_na <- which(na_mat, arr.ind = TRUE)
    na_index <- data.frame(
      row = where_na[, "row"],
      col = colnames(data)[where_na[, "col"]],
      stringsAsFactors = FALSE
    )
    attr(data, "na_index") <- na_index

    warning(
      sprintf(
        "Detected %d missing values across %d columns.\n",
        nrow(na_index),
        length(unique(na_index$col))
      ),
      "Use attr(data, 'na_index') to inspect them.\n",
      "To proceed you should either drop incomplete rows or impute missing values.",
      call. = FALSE
    )
  } else {
    attr(data, "na_index") <- NULL
  }

  # ---- basic consistency checks ----
  nonneg_cols <- c("death_time", "onset_age", "age")
  for (col in nonneg_cols) {
    x <- data[[col]]
    n_bad <- sum(x < 0, na.rm = TRUE)
    if (n_bad > 0L) {
      warning(
        sprintf("Column '%s' contains %d negative values.", col, n_bad),
        call. = FALSE
      )
    }
  }

  # visits should be positive integer-like
  if (any(data[["visits"]] <= 0, na.rm = TRUE)) {
    warning(
      "Column 'visits' contains non-positive values; expected strictly positive visit indices.",
      call. = FALSE
    )
  }
  if (any(abs(data[["visits"]] - round(data[["visits"]])) > .Machine$double.eps^0.5, na.rm = TRUE)) {
    warning(
      "Column 'visits' contains non-integer values; expected integer visit indices.",
      call. = FALSE
    )
  }

  # onset_age should not exceed death_time for dead == 1 & onset == 1
  idx_inconsistent <- which(
    data[["dead"]] == 1 &
      data[["onset"]] == 1 &
      !is.na(data[["onset_age"]]) &
      !is.na(data[["death_time"]]) &
      data[["onset_age"]] > data[["death_time"]]
  )
  if (length(idx_inconsistent) > 0L) {
    warning(
      sprintf(
        paste0(
          "Found %d rows where onset_age > death_time for patients with dead == 1 & onset == 1.\n",
          "Example row indices: %s"
        ),
        length(idx_inconsistent),
        paste(utils::head(idx_inconsistent, 5L), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  invisible(data)
}
