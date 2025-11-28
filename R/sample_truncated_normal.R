#' Sample from a Truncated Normal Distribution
#'
#' @description
#' Draws a single random value from a truncated normal distribution with specified mean, standard deviation,
#' and lower/upper bounds.
#'
#' @param mean Numeric; mean of the normal distribution.
#' @param sd Numeric; standard deviation of the normal distribution.
#' @param a Numeric; lower bound of truncation.
#' @param b Numeric; upper bound of truncation.
#'
#' @return A numeric value sampled from the truncated normal distribution.
#'
#' @examples
#' \dontrun{
#' sample_truncated_normal(mean = 10, sd = 2, a = 5, b = 15)
#' }
#'
#' @importFrom truncnorm rtruncnorm
#' @export

sample_truncated_normal <- function(mean, sd, a, b) {
  sd <- max(sd, 1e-3)
  rtruncnorm(1, a = a, b = b, mean = mean, sd = sd)
}
