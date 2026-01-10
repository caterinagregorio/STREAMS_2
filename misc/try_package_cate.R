library(devtools)
load_all()
panel_data <- simulation_ready_001[[2]]
cov_vector <- c("cov1", "cov2", "cov3")
check_input_data(panel_data)

fit_streams <- run_streams(
  data = panel_data,
  cov_vector = cov_vector,
  python = Sys.which("python3"),
  pu_args = list(verbose = TRUE)
)


#lapply(fit_streams, function(x) summary(x, coefs = TRUE)) # da sistemare

#trying all methods for my class flexsurvreg_pooled
trans1_streams <- fit_streams[[1]]
class(trans1_streams)
print(trans1_streams)
summary_trans1 <- summary(trans1_streams, coefs = TRUE) # al momento è uguale al print ma si puo arricchire (distribuzione, formula, cov usate..)
confint(trans1_streams)
attr(trans1_streams, "rubin")
coef(trans1_streams)
print(summary_trans1)


streams_coefs <- lapply(fit_streams, stats::coef)
names(streams_coefs) <- c("0->1", "0->2", "1->2")
