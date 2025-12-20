#' DA  SISTEMARE TUTTE LE DIRECTORY E PARAMETRI DELLE FUNZIONI NESTED CHE VOGLIO LASCIARE MODIFICABILI
#' Run STREAMS Pipeline
#'
#' @description
#' Executes the full STREAMS workflow:
#' \enumerate{
#'   \item Prepares longitudinal panel data for PU-learning and downstream modeling.
#'   \item Calls Python scripts for training and inference. This phase exploits a Mean Teacher conditional
#'   variational autoencoder generating a individual-level posterior distribution for disease onset and onset age.
#'   \item Performs multiple imputation of complete trajectories.
#'   \item Fits multi-state models on each imputed dataset.
#'   \item Combine the parameter estimates using Rubin's rules.
#' }
#' @param data A `data.table` or `data.frame` that must contain the following columns:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#' @param cov_vector Character vector of covariate names for modeling.
#' @param lab_prop Proportion for PU-learning thresholding. Default = 0.5.
#' @param train_args Named list of additional CLI arguments for training script.
#' @param infer_args Named list of additional CLI arguments for inference script.
#' @param m Number of imputations for onset age sampling (default = 20).
#' @param clock_assumption Clock assumption for multi-state modeling (`"forward"` or `"mix"`).
#' @param distribution Distribution for parametric survival models (e.g., `"gompertz"`).
#' @param n_cores Number of cores for parallel fitting (default = 4).
#' @param python Character. Python executable (default "python3").
#'
#' @return A matrix of averaged parameter estimates across imputations.
#'
#' @examples
#' \dontrun{
#' est_params <- run_streams(
#'   data = df_panel,
#'   cov_vector = c("sex", "bmi", "smoking"),
#'   version_name = "v1",
#'   train_py = "py/training.py",
#'   infer_py = "py/inference.py"
#' )
#' }
#' @importFrom arrow read_feather
#' @export


.default_pu_args <- list(
  max_iter = 1000,
  tol = 1e-5,
  clip = 1e-3,
  damp = 0.7,
  shrink_k = 1.0,
  verbose = FALSE
)

.default_cvae_args <- list(
  latent_dim = 5,
  max_epochs = 200,
  batch_size = 256,
  lr = 2e-3,
  r_all = 0.3,
  lambda_fix_max = 1.0,
  fix_ramp_epochs = 50,
  K_aug = 5,
  v_max = 1e-3,
  prior_aug = "beta",
  prior_kappa = 30.0,
  early_stop_patience = 15,
  early_stop_warmup = 50
)

.default_infer_args <- list(
  latent_dim = 5,
  mc_samples = 5,
  batch_size = 256,
  use_student = FALSE
)


run_streams <- function(
    data,
    cov_vector,

    # --- Core analysis ---
    m = 20,
    clock_assumption = "forward",
    distribution = "gompertz",
    custom_formula = NULL,

    # --- PU learning ---
    lab_prop = 0.15,
    pu_args = list(),

    # --- CVAE / FixMatch (macro knobs) ---
    cvae_args = list(),

    # --- Inference ---
    infer_mc_samples = 0,
    infer_args = list(),

    # --- Execution ---
    python = Sys.which("python"),
    out_dir = tempdir(),
    keep_intermediate = FALSE,
    n_cores = 1,
    seed = 42
)
{

  # --- merge default arguments
  pu_cfg    <- modifyList(.default_pu_args, pu_args)
  cvae_cfg  <- modifyList(.default_cvae_args, cvae_args)
  infer_cfg <- modifyList(.default_infer_args, infer_args)

  cvae_cfg$seed  <- seed

  if (infer_mc_samples > 0) {
    infer_cfg$mc_samples <- infer_mc_samples
  }


  # --- helper: list -> vector args CLI
  as_cli_args <- function(arg_list) {
    if (length(arg_list) == 0) return(character(0))
    out <- character(0)

    for (k in names(arg_list)) {
      v <- arg_list[[k]]

      if (is.logical(v)) {
        if (isTRUE(v)) out <- c(out, paste0("--", k))
        next
      }

      # scalar
      out <- c(out, paste0("--", k), as.character(v))
    }

    out
  }


  # folders for temp results
  if (is.null(out_dir)) {
    out_dir <- file.path(tempdir(), sprintf("STREAMS_%s", format(Sys.time(), "%Y%m%d_%H%M%S")))
  }

  if (!keep_intermediate) {
    on.exit(unlink(out_dir, recursive = TRUE, force = TRUE), add = TRUE)
  }

  dirs <- list(
    data   = file.path(out_dir, "data"),
    models = file.path(out_dir, "models"),
    results = file.path(out_dir, "results")
  )

  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))


  input_path_train <- file.path(dirs$data, "training_data.feather")
  input_path_infer <- file.path(dirs$data,    "inference_data.feather")
  train_split      <- file.path(dirs$data, "train_split.feather")
  val_split        <- file.path(dirs$data, "val_split.feather")
  model_path       <- file.path(dirs$models,   "best_model.pt")
  distributions_path<- file.path(dirs$results,  "distributions.feather")
  logs_path        <- file.path(dirs$results,  "logs.feather")


  # --- clean data and prepare them for model
  temp <- prepare_data(
    data = data,
    cov_vector = cov_vector,
    lab_prop = lab_prop,
    pu_args = pu_cfg,
    train_path  = input_path_train,
    infer_path = input_path_infer
  )
  cleaned_data <- temp[[1]]
  cov_str <- paste(temp[[2]], collapse = ",")

  if (is.null(python)) python <- "python3"

  # --- TRAINING PY
  train_cli <- c(
    model_path,
    input_path_train,
    train_split,
    val_split,
    logs_path,
    cov_str,
    as_cli_args(cvae_cfg)
  )
  streams_python("train.py", args = train_cli, python = python)



  # --- INFERENCE PY
  infer_cli <- c(
    model_path,
    input_path_infer,
    cov_str,
    "--out", distributions_path,
    as_cli_args(infer_cfg)
  )
  streams_python("inference.py", args = infer_cli, python = python)

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
    fit_model(temp, cov_vector, clock_assumption, distribution, custom_formula)

  }, mc.cores = n_cores)

  #save(all_fits, file = fits_rdata)

  # --- average over imputed models
  est_params <- averaging_params(all_fits)
  #saveRDS(est_params, file = estimated_params)


  return(est_params)
}



