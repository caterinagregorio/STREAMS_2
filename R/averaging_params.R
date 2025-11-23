averaging_params <- function(all_fits) {
  param_names <- names(stats::coef(all_fits[[1]][[1]]))
  n_params <- length(param_names)
  out <- matrix(0, nrow = length(all_fits[[1]]), ncol = n_params)
  colnames(out) <- param_names
  for (i in seq_along(all_fits[[1]])) {
    for (p in seq_along(param_names)) {
      out[i, p] <- mean(sapply(all_fits, function(fit) stats::coef(fit[[i]])[p]))
    }
  }
  out
}
