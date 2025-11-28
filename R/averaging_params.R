#' Average Model Parameters Across Multiple Fits
#'
#' @description
#' Computes the mean of estimated parameters across multiple sets of fitted models.
#'
#' @param all_fits A list of lists, where each inner list contains fitted models (e.g., `flexsurvreg` objects).
#'
#' @return A numeric matrix with rows corresponding to transitions and columns to parameter names.
#'
#' @examples
#' \dontrun{
#' # Suppose all_fits is a list of bootstrap replicates, each containing 3 fitted models
#' avg_params <- averaging_params(all_fits)
#' print(avg_params)
#' }
#'
#' @importFrom stats coef
#' @export
#'
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
