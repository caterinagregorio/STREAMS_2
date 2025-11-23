sample_truncated_normal <- function(mean, sd, a, b) {
  sd <- max(sd, 1e-3)
  rtruncnorm(1, a = a, b = b, mean = mean, sd = sd)
}
