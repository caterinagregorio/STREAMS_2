#' @noRd
.get_rubin <- function(x) {
  rub <- attr(x, "rubin", exact = TRUE)
  if (is.null(rub)) stop("Missing Rubin diagnostics in attr(x, 'rubin').")
  rub
}

# Patch a COPY of the object so that flexsurv's summary/normboot
#' @noRd
.as_flexsurvreg_patched_for_summary <- function(object) {
  x <- object
  rub <- .get_rubin(object)

  qbar <- rub$Qbar
  tcov <- rub$Tcov

  x$coefficients <- qbar

  # If there are fixed parameters, flexsurv stores cov only for optpars
  opt <- x$optpars
  if (!is.null(opt) && length(opt) > 0) {
    x$cov <- tcov[opt, opt, drop = FALSE]
    if (!is.null(x$opt) && !is.null(x$opt$par)) {
      x$opt$par <- qbar[opt]
    }
  } else {
    # common case: no fixed parameters
    x$cov <- tcov
    if (!is.null(x$opt) && !is.null(x$opt$par)) {
      x$opt$par <- qbar
    }
  }

  if (!is.null(x$res.t) && "est" %in% colnames(x$res.t)) {
    # ensure rownames match names(qbar)
    if (!is.null(names(qbar)) && !is.null(rownames(x$res.t))) {
      if (all(names(qbar) %in% rownames(x$res.t))) {
        x$res.t[names(qbar), "est"] <- qbar
      } else if (nrow(x$res.t) == length(qbar)) {
        x$res.t[, "est"] <- qbar
        rownames(x$res.t) <- names(qbar)
      }
    } else if (nrow(x$res.t) == length(qbar)) {
      x$res.t[, "est"] <- qbar
    }
  }

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

  if (!is.null(parm)) {
    est <- est[parm]
    se  <- se[parm]
    df  <- rub$df[match(names(est), names(rub$Qbar))]
  } else {
    df <- rub$df
    names(df) <- names(est)
  }

  alpha <- 1 - level
  crit <- mapply(function(d) {
    if (is.finite(d)) stats::qt(1 - alpha/2, df = d) else stats::qnorm(1 - alpha/2)
  }, df)

  lower <- est - crit * se
  upper <- est + crit * se

  out <- cbind(lower = lower, upper = upper)
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
#' @exportS3Method base::print flexsurvreg_pooled
summary.flexsurvreg_pooled <- function(object, ..., coefs = FALSE, level = 0.95) {
  if (isTRUE(coefs)) {
    rub <- .get_rubin(object)
    est <- rub$Qbar
    se  <- sqrt(diag(rub$Tcov))
    df  <- rub$df

    alpha <- 1 - level
    crit <- mapply(function(d) {
      if (is.finite(d)) stats::qt(1 - alpha/2, df = d) else stats::qnorm(1 - alpha/2)
    }, df)

    tab <- data.frame(
      est = est,
      se  = se,
      df  = df,
      LCL = est - crit * se,
      UCL = est + crit * se,
      row.names = names(est)
    )

    out <- list(
      call = object$call,
      dist = object$dist,
      m = rub$m,
      table = tab,
      rubin = rub,
      level = level
    )
    class(out) <- "summary.flexsurvreg_pooled"
    return(out)
  }

  # curves summary (survival/cumhaz/hazard/etc): use flexsurv's method
  patched <- .as_flexsurvreg_patched_for_summary(object)
  getS3method("summary", "flexsurvreg")(patched, ...)
}

#' @noRd
#' @exportS3Method base::print summary.flexsurvreg_pooled
print.summary.flexsurvreg_pooled <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Pooled flexsurvreg (Rubin's rules)\n")
  if (!is.null(x$dist)) cat("Distribution:", x$dist, "\n")
  if (!is.null(x$m))    cat("m imputations:", x$m, "\n\n")
  print(round(x$table, digits = digits))
  invisible(x)
}
