#' Plot STREAMS training/validation losses and FixMatch contribution
#'
#' @description
#' Creates two diagnostic line plots from the training logs produced by the STREAMS CVAE training script:
#' \enumerate{
#'   \item \strong{Total loss:} \code{train_total} and \code{val_total} over epochs.
#'   \item \strong{Aligned loss + FixMatch:} training and validation losses computed over comparable components,
#'         plus an extra FixMatch curve (raw discrepancy or weighted contribution).
#' }
#'
#' The second plot is designed to avoid misleading comparisons between \code{train_total} and \code{val_total}
#' when the validation objective is computed from a different set of terms than the training objective.
#'
#' @param logs A \code{data.frame} or tibble containing the training log history. It must include at least:
#' \itemize{
#'   \item \code{epoch}: epoch index;
#'   \item \code{train_total}: total training objective;
#'   \item \code{val_total}: total validation objective.
#' }
#' For the aligned plot, \code{logs} must also contain:
#' \itemize{
#'   \item Training components: \code{train_sup_bce}, \code{train_age}, \code{train_sd};
#'   \item Validation components: \code{val_bce}, \code{val_age}, \code{val_sdreg};
#'   \item FixMatch curve: \code{train_fix_raw} (raw) and/or \code{train_fix_w} (weighted).
#' }
#'
#' @param fixmatch_line Character string indicating which FixMatch curve to plot in the aligned panel:
#'   \code{"raw"} plots \code{train_fix_raw} (raw discrepancy), while \code{"weighted"} plots \code{train_fix_w}
#'   (weighted contribution). Default is \code{c("raw","weighted")} (first match is \code{"raw"}).
#'
#' @param smooth Logical. If \code{TRUE}, overlays a LOESS smoother on each series to improve readability.
#'   Default is \code{FALSE}.
#'
#' @param span Numeric in \eqn{(0,1]}. LOESS span used when \code{smooth = TRUE}. Smaller values follow the data
#'   more closely. Default is \code{0.25}.
#'
#' @details
#' \strong{Aligned loss definition.}
#' The aligned plot computes:
#' \itemize{
#'   \item \code{aligned_train = train_sup_bce + train_age + train_sd}
#'   \item \code{aligned_val   = val_bce + val_age + val_sdreg}
#' }
#' and adds a FixMatch curve as an additional line.
#'
#' \strong{Interpretation of FixMatch.}
#' In this framework, FixMatch should be interpreted as a consistency constraint regulating agreement between
#' student and teacher predictions on unlabeled data, rather than as a standalone loss to be minimized in isolation.
#' Under a ramp-up/decay schedule, the \emph{weighted} FixMatch contribution may temporarily increase even while
#' the \emph{raw} discrepancy decreases; this behavior is expected and does not necessarily indicate optimization
#' failure.
#'
#' @return An (invisible) named list of two \code{ggplot} objects:
#' \describe{
#'   \item{\code{total}}{Plot of \code{train_total} and \code{val_total} vs \code{epoch}.}
#'   \item{\code{aligned_plus_fixmatch}}{Plot of aligned train/validation losses plus the selected FixMatch curve.}
#' }
#'
#' @examples
#' \dontrun{
#' logs <- arrow::read_feather("path/to/logs.feather")
#' plot_streams_total_and_fixmatch(logs, fixmatch_line = "raw")
#' plot_streams_total_and_fixmatch(logs, fixmatch_line = "weighted", smooth = TRUE, span = 0.3)
#' }
#'
#' @importFrom dplyr select mutate transmute
#' @importFrom tidyr pivot_longer
#' @export

plot_streams_total_and_fixmatch <- function(
    logs,
    fixmatch_line = c("raw", "weighted"),
    smooth = FALSE,
    span = 0.25,
    print_plots = TRUE
) {
  fixmatch_line <- match.arg(fixmatch_line)

  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("tidyr", quietly = TRUE) ||
      !requireNamespace("dplyr", quietly = TRUE)) {
    stop("Please install ggplot2, tidyr, dplyr for these plots.")
  }

  # ---- Plot 1: total loss train vs val
  df_total <- logs |>
    dplyr::select(epoch, train_total, val_total) |>
    tidyr::pivot_longer(cols = c(train_total, val_total),
                        names_to = "split",
                        values_to = "loss")

  df_total$split <- factor(df_total$split,
                           levels = c("train_total", "val_total"),
                           labels = c("Train", "Validation"))

  p_total <- ggplot2::ggplot(df_total, ggplot2::aes(x = epoch, y = loss, color = split)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::labs(x = "Epoch", y = "Total loss", color = NULL, title = "Total loss vs epoch") +
    ggplot2::theme_bw()

  if (smooth) {
    p_total <- p_total + ggplot2::geom_smooth(se = FALSE, method = "loess", span = span)
  }
  if (isTRUE(print_plots)) print(p_total)

  # ---- Plot 2: aligned losses + FixMatch line
  fix_col <- if (fixmatch_line == "raw") "train_fix_raw" else "train_fix_w"
  if (!fix_col %in% names(logs)) stop("Missing FixMatch column in logs: ", fix_col)

  df2 <- logs |>
    dplyr::mutate(
      aligned_train = train_sup_bce + train_age + train_sd,
      aligned_val   = val_bce + val_age + val_sdreg
    ) |>
    dplyr::transmute(
      epoch = epoch,
      aligned_train = aligned_train,
      aligned_val   = aligned_val,
      fixmatch      = .data[[fix_col]]
    ) |>
    tidyr::pivot_longer(cols = c(aligned_train, aligned_val, fixmatch),
                        names_to = "series",
                        values_to = "value")

  df2$series <- factor(
    df2$series,
    levels = c("aligned_train", "aligned_val", "fixmatch"),
    labels = c(
      "Train",
      "Validation",
      if (fixmatch_line == "raw") "FixMatch (raw contribution)" else "FixMatch (weighted contribution)"
    )
  )

  p_aligned <- ggplot2::ggplot(df2, ggplot2::aes(x = epoch, y = value, color = series)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::labs(
      x = "Epoch",
      y = "Value",
      color = NULL,
      title = "Loss over common quantities in training and validation",
      subtitle = "FixMatch shown as extra contribution (not directly comparable in scale)"
    ) +
    ggplot2::theme_bw()

  if (smooth) {
    p_aligned <- p_aligned + ggplot2::geom_smooth(se = FALSE, method = "loess", span = span)
  }
  if (isTRUE(print_plots)) print(p_aligned)

  invisible(list(total = p_total, aligned_plus_fixmatch = p_aligned))
}
