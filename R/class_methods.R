#' @noRd
.get_rubin <- function(x) {
  rub <- attr(x, "rubin", exact = TRUE)
  if (is.null(rub)) stop("Missing Rubin diagnostics in attr(x, 'rubin').")
  rub
}

# Patch a COPY of the object so that flexsurv's summary/normboot
#' @noRd
.as_flexsurvreg_patched_for_summary <- function(object) {

  rub <- .get_rubin(object)

  x <- object
  qbar <- rub$Qbar
  tcov <- rub$Tcov

  # overwrite pooled estimates
  x$coefficients <- qbar

  optpars <- x$optpars

  # covariance: only optpars if some parameters are fixed
  if (!is.null(optpars) && length(optpars) > 0) {
    x$cov <- tcov[optpars, optpars, drop = FALSE]
  } else {
    x$cov <- tcov
  }

  # remove pooled class so flexsurv summary dispatches correctly
  class(x) <- setdiff(class(x), "flexsurvreg_pooled")

  x
}


#------------------------------------------------------------
# coef / vcov
#------------------------------------------------------------
#' @noRd
#' @exportS3Method stats::coef flexsurvreg_pooled
coef.flexsurvreg_pooled <- function(object, ...) {
  .get_rubin(object)$Qbar
}
#' @noRd
#' @exportS3Method stats::vcov flexsurvreg_pooled
vcov.flexsurvreg_pooled <- function(object, ...) {
  .get_rubin(object)$Tcov
}

#------------------------------------------------------------
# confint with Rubin df (t-interval per-parameter)
#------------------------------------------------------------

#' @noRd
#' @exportS3Method stats::confint flexsurvreg_pooled
confint.flexsurvreg_pooled <- function(object, parm = NULL, level = 0.95, ...) {

  rub <- .get_rubin(object)

  est <- rub$Qbar
  V   <- rub$Tcov
  se  <- sqrt(diag(V))
  df  <- rub$df
  names(df) <- names(est)

  if (!is.null(parm)) {
    est <- est[parm]
    se  <- se[parm]
    df  <- df[parm]
  }

  alpha <- 1 - level
  crit <- vapply(df, function(d) {
    if (is.finite(d)) stats::qt(1 - alpha/2, df = d)
    else stats::qnorm(1 - alpha/2)
  }, numeric(1))

  out <- cbind(
    lower = est - crit * se,
    upper = est + crit * se
  )

  attr(out, "level") <- level
  out
}

#------------------------------------------------------------
# print
#------------------------------------------------------------

#' @noRd
#' @exportS3Method base::print flexsurvreg_pooled
print.flexsurvreg_pooled <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  rub <- .get_rubin(x)
  est <- rub$Qbar
  se  <- sqrt(diag(rub$Tcov))
  df  <- rub$df
  level <- 0.95

  ci <- confint.flexsurvreg_pooled(x, level = level)

  tab <- data.frame(
    est = est,
    se  = se,
    df  = df,
    LCL = ci[,1],
    UCL = ci[,2],
    row.names = names(est)
  )

  cat("Pooled flexsurvreg (Rubin's rules)\n")
  if (!is.null(x$dist)) cat("Distribution:", x$dist, "\n")
  if (!is.null(rub$m))   cat("m imputations:", rub$m, "\n\n")

  print(round(tab, digits = digits))
  invisible(x)
}

#------------------------------------------------------------
# summary:
# - coefs=TRUE -> Rubin table (t-based CI)
# - coefs=FALSE -> delegate to flexsurv::summary.flexsurvreg for curves,
#   using a patched copy centered at Rubin mean with Rubin covariance.
#------------------------------------------------------------
#' @noRd
#' @exportS3Method stats::summary flexsurvreg_pooled
summary.flexsurvreg_pooled <- function(object, ..., coefs = FALSE, level = 0.95) {

  rub <- .get_rubin(object)

  if (isTRUE(coefs)) {

    est <- rub$Qbar
    se  <- sqrt(diag(rub$Tcov))
    df  <- rub$df

    alpha <- 1 - level
    crit <- vapply(df, function(d) {
      if (is.finite(d)) stats::qt(1 - alpha/2, df = d)
      else stats::qnorm(1 - alpha/2)
    }, numeric(1))

    tab <- data.frame(
      est = est,
      se  = se,
      df  = df,
      LCL = est - crit * se,
      UCL = est + crit * se,
      row.names = names(est)
    )

    out <- list(
      call     = object$call,
      table    = tab,
      rubin    = rub,
      metadata = object$metadata,
      level    = level
    )

    class(out) <- "summary.flexsurvreg_pooled"
    return(out)
  }

  patched <- .as_flexsurvreg_patched_for_summary(object)
  stats::summary(patched, ...)
}


#' @noRd
#' @exportS3Method base::print summary.flexsurvreg_pooled
print.summary.flexsurvreg_pooled <- function(x,
                                             digits = max(3L, getOption("digits") - 3L),
                                             ...) {

  cat("Pooled flexsurvreg (Rubin's rules)\n")

  if (!is.null(x$metadata$distribution))
    cat("Distribution:", x$metadata$distribution, "\n")

  if (!is.null(x$rubin$m))
    cat("m imputations:", x$rubin$m, "\n\n")

  print(round(x$table, digits = digits))

  cat("\nMetadata:\n")
  for (nm in names(x$metadata)) {
    cat(sprintf("  %-18s: %s\n", nm, deparse(x$metadata[[nm]])))
  }

  invisible(x)
}
