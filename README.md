# STREAMS
Semi-supervised Trajectory REconstruction Across interval-censored Multi-State models.

`STREAMS` is an R package for reconstructing latent disease trajectories in
progressive illness–death models using semi-supervised learning and then fitting
parametric multi-state models with proper pooling across multiple imputations.

# Installation

install.packages("devtools")  # if not already installed
devtools::install_github("Alepescinaa/STREAMS")


library(STREAMS)

# load example panel data
data("toy_example")
panel_data <- toy_example

# covariates to be used in the model
cov_vector <- c("cov1", "cov2", "cov3")

# (optional) sanity check on the input structure
check_input_data(panel_data)

# run the main pipeline

fit_streams <- run_streams(
  data              = panel_data,
  cov_vector        = cov_vector,
  python            = Sys.which("python3"),
  pu_args           = list(verbose = TRUE),
  features_prop_add = NULL
)

# overall object
class(fit_streams)
summary(fit_streams, plots = TRUE)

# first transition model
trans1_streams <- fit_streams[[1L]]
class(trans1_streams)

print(trans1_streams)
confint(trans1_streams)
attr(trans1_streams, "rubin") 
coef(trans1_streams)
summary(trans1_streams)


