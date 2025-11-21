run_streams <- function(
    panel_data,
    cov_vector,
    version_name,
    base_out_dir = file.path("py", "version"),
    train_py = "py/training_fix_dec4.0.py",
    infer_py = "py/inference_fix_dec4.0.py",
    lab_prop = 0.5,
    covariate_names = NULL,     # if NULL uses cov_vector
    train_args = list("--prior_aug"="beta"),  #args to change according to versions
    infer_args = list(),
    m = 20,
    clock_assumption = "forward",
    distribution = "gompertz",
    n_cores = 4
) {

  pkgs <- c("dplyr","tibble","arrow","data.table","flexsurv","mstate",
            "survival","parallel","truncnorm","stats", "pROC")
  suppressPackageStartupMessages(invisible(lapply(pkgs, require, character.only = TRUE)))

  if (is.null(covariate_names)) covariate_names <- cov_vector

  # --- helper: list -> vector args CLI
  as_cli_args <- function(arg_list) {
    if (length(arg_list)==0) return(character(0))
    unlist(mapply(function(k,v) c(k, as.character(v)), names(arg_list), arg_list, SIMPLIFY = FALSE), use.names = FALSE)
  }


  # folders: base/version/{data, models, results}
  version_dir <- file.path(base_out_dir, version_name)
  dirs <- list(
    data = file.path(version_dir, "data"),
    models   = file.path(version_dir, "models"),
    results  = file.path(version_dir, "results")
  )
  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))


  input_path_train <- file.path(dirs$data, "training_data.feather")
  input_path_infer <- file.path(dirs$data,    "infer_data.feather")
  train_split      <- file.path(dirs$data, "train_split.feather")
  val_split        <- file.path(dirs$data, "val_split.feather")
  model_path       <- file.path(dirs$models,   "best_model.pt")
  distributions_path<- file.path(dirs$results,  "distributions.feather")
  cleaned_rdata    <- file.path(dirs$results,  "cleaned_data.RData")
  fits_rdata       <- file.path(dirs$results,  "fits.RData")
  estimated_params <- file.path(dirs$results,  "estimated_params.rds")


  # --- clean data and prepare them for model
  cleaned_data <- prepare_data(
    data = panel_data,
    cov_vector = cov_vector,
    covariate_names = covariate_names,
    lab_prop = lab_prop,
    train_path  = input_path_train,
    infer_path = input_path_infer
  )
  save(cleaned_data, file = cleaned_rdata)

  cov_str <- paste(covariate_names, collapse = ",")

  # --- TRAINING PY
  train_cli <- c(
    train_py,
    model_path,
    input_path_train,
    train_split,
    val_split,
    cov_str,
    as_cli_args(train_args)
  )
  system2("python3", args = train_cli)

  # --- INFERENCE PY
  infer_cli <- c(
    infer_py,
    model_path,
    input_path_infer,
    cov_str,
    "--out", distributions_path,
    as_cli_args(infer_args)
  )
  system2("python3", args = infer_cli)

  # --- extract model predictions
  distributions <- arrow::read_feather(distributions_path)

  # --- prepare imputation part (a,b)
  n_patients <- nrow(cleaned_data)
  dat <- data.frame(
    a = cleaned_data$last_bfo,
    b = ifelse(cleaned_data$onset == 1, cleaned_data$onset_age, cleaned_data$death_time),
    age_mu = distributions$age_mu,
    age_sd = distributions$age_sd,
    p_onset = distributions$p_onset
  )

  # --- Inverse sampling
  c1 <- matrix(stats::runif(n_patients * m), nrow = n_patients, ncol = m)
  disease_status <- (dat$p_onset > c1) * 1

  disease_age <- matrix(0, nrow = n_patients, ncol = m)
  for (i in 1:n_patients) {
    mu <- dat$age_mu[i]; sd <- dat$age_sd[i]; a <- dat$a[i]; b <- dat$b[i]
    b_safe <- pmin(pmax(b - 1e-3, a + 1e-3), b)
    disease_age[i, ] <- vapply(disease_status[i, ], function(o) {
      if (o == 1) {
        sa <- sample_truncated_normal(mu, sd, a, b_safe)
        if (is.na(sa)) warning(sprintf("NA at i=%d: mu=%f sd=%f a=%f b=%f b_safe=%f",
                                       i, mu, sd, a, b, b_safe))
        sa
      } else b
    }, numeric(1))
  }

  # --- fit multi-state
  all_fits <- parallel::mclapply(1:m, function(j) {
    temp <- cleaned_data
    idx <- which(disease_status[, j] == 1)
    if (length(idx)) {
      temp$onset[idx]     <- 1
      temp$onset_age[idx] <- disease_age[idx, j]
    }
    fit_model(temp, cov_vector, clock_assumption, distribution)
  }, mc.cores = n_cores)

  save(all_fits, file = fits_rdata)

  # --- average over imputed models
  est_params <- averaging_params(all_fits)
  saveRDS(est_params, file = estimated_params)


  return(est_params)
}
