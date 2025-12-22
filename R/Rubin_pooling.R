#' @noRd
.pool_rubin <- function(Q, U_list) {
  # Q: m x p matrix of estimates
  # U_list: list length m of p x p covariance matrices
  m <- nrow(Q)
  Qbar <- colMeans(Q)

  Ubar <- Reduce(`+`, U_list) / m
  B <- stats::cov(Q)
  Tcov <- Ubar + (1 + 1/m) * B       # Rubin total variance

  # Barnard-Rubin df per-parameter
  diagU <- diag(Ubar)
  diagB <- diag(B)
  denom <- (1 + 1/m) * diagB
  df <- rep(Inf, length(Qbar))
  ok <- denom > 0
  df[ok] <- (m - 1) * (1 + diagU[ok] / denom[ok])^2

  list(Qbar = Qbar, Ubar = Ubar, B = B, Tcov = Tcov, df = df, m = m)
}

pool_rubin_one_model <- function(fits, cl = 0.95) {
  fits <- Filter(function(x) inherits(x, "flexsurvreg"), fits)
  if (length(fits) < 2) stop("Need at least 2 successful fits to pool.")


  par_names <- names(stats::coef(fits[[1]]))
  if (any(vapply(fits, function(f) !identical(names(stats::coef(f)), par_names), logical(1)))) {
    stop("Parameter names/order differ across imputations. Ensure identical model specification.")
  }

  Q <- do.call(rbind, lapply(fits, function(f) stats::coef(f)))
  U_list <- lapply(fits, function(f) stats::vcov(f))

  rub <- .pool_rubin(Q, U_list)

  # Use first fit as a template and overwrite the key fields used downstream
  pooled <- fits[[1]]
  pooled$coefficients <- rub$Qbar
  pooled$cov <- rub$Tcov

  # Store Rubin diagnostics
  attr(pooled, "rubin") <- rub
  class(pooled) <- unique(c("flexsurvreg_pooled", class(pooled)))

  pooled
}

pool_rubin_all_transitions <- function(all_fits, cl = 0.95) {

  ok <- which(vapply(all_fits, function(x) !is.null(x), logical(1)))
  if (!length(ok)) stop("No successful fits found.")
  template <- all_fits[[ok[1]]]

  if (inherits(template, "flexsurvreg")) {
    return(pool_rubin_one_model(all_fits, cl = cl))
  }

  if (is.list(template) && all(vapply(template, inherits, logical(1), "flexsurvreg"))) {
    K <- length(template)
    pooled_list <- vector("list", K)
    for (k in seq_len(K)) {
      kth_fits <- lapply(all_fits, function(obj) if (is.list(obj)) obj[[k]] else NULL)
      pooled_list[[k]] <- pool_rubin_one_model(kth_fits, cl = cl)
    }
    names(pooled_list) <- names(template)

    # try to preserve attributes/class from template
    at <- attributes(template)
    at$names <- names(pooled_list)
    attributes(pooled_list) <- at
    return(pooled_list)
  }

  stop("Unsupported fit structure: expected flexsurvreg or list-of-flexsurvreg per imputation.")
}
