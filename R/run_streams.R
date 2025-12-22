#' Run STREAMS: Main function
#'
#' @description
#' \code{run_streams()} runs the full STREAMS pipeline starting from longitudinal visit-level data:
#' \enumerate{
#'   \item Builds a patient-level dataset and PU-learning priors via \code{\link{prepare_data}}.
#'   \item Trains a Conditional VAE (withMean-Teacher style training) via the packaged Python script
#'         \code{train.py} (called through \code{\link{streams_python}}).
#'   \item Runs model inference via \code{inference.py} to obtain individual-level predictions:
#'         onset probability \eqn{p(\mathrm{onset})} and onset-age distribution parameters (e.g., mean and SD).
#'   \item Generates \code{m} imputed complete trajectories by sampling onset status and onset age.
#'   \item Fits \code{m} parametric illness-death multi-state models via \code{\link{fit_model}}.
#'   \item Aggregates parameters across imputations via \code{\link{averaging_params}}.
#' }
#'
#' @details
#' The pipeline exchanges data between R and Python using Feather files
#' and stores a trained model checkpoint. See the “Side effects” section below.
#'
#' @param data A `data.table` or `data.frame` that must contain the following columns:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#'   The `data.frame` can contain extra columns with covariate values.
#'
#' @param cov_vector Character vector of covariate names to be used in modeling.
#'   These are automatically scaled/encoded in the process.
#' @param m Integer. Number of fitted multi-state models that will contribute to the pooled estimates.
#' @param clock_assumption Character. Time-scale assumption for the multi-state model. Passed to
#'   \code{\link{fit_model}}. Accepted values are \code{"forward"} to fit a Markov process or \code{"mix"} for a Semi-Markov process.
#' @param distribution A character string specifying the parametric form of baseline hazards.
#'   Must be one of the distributions available in `flexsurv::flexsurvreg`, e.g., `"weibull"`, `"exponential"`, `"gompertz"`.
#' @param custom_formula Optional \code{survival::Surv()} formula used for all transitions in the multi-state fit.
#'   If \code{NULL}, formulas are built automatically from \code{cov_vector} (assuming linear effects).
#' @param lab_prop Numeric in (0, 1). Controls the PU-learning thresholding used to derive soft labels for training:
#'   among patients with \code{onset == 0}, those below the \code{lab_prop}-quantile of PU risk scores are treated
#'   as reliable negatives (\code{onset_soft = 0}); remaining \code{onset == 0} are left unlabeled.
#'
#' @param pu_args Named list of PU-learning hyperparameters overriding \code{.default_pu_args} and forwarded to
#'   the PU-learning routine. Supported keys:
#'   \describe{
#'     \item{max_iter}{Maximum number of EM iterations for PU learning.}
#'     \item{tol}{Convergence tolerance on successive log-likelihood differences.}
#'     \item{clip}{Probability clipping threshold for numerical stability.}
#'     \item{damp}{Damping factor in (0,1) for updating PU components to reduce oscillations.}
#'     \item{shrink_k}{Non-negative shrinkage parameter to penalize regions with low labeling propensity.}
#'     \item{verbose}{Logical. If \code{TRUE}, prints PU-learning training diagnostics.}
#'   }
#'
#' @param cvae_args Named list of training hyperparameters overriding \code{.default_cvae_args}.
#'   These entries are converted to command-line flags and passed to the Python training script \code{train.py}.
#'   Common keys in STREAMS are:
#'   \describe{
#'     \item{latent_dim}{Latent dimension of the CVAE.}
#'     \item{max_epochs}{Maximum number of training epochs.}
#'     \item{batch_size}{Mini-batch size.}
#'     \item{lr}{Learning rate.}
#'     \item{K_aug}{Number of independent prior augmentations evaluated by the \emph{teacher} per unlabeled
#'     input to estimate a mean teacher score \eqn{\bar p_i} and its variance \eqn{v_i}. Increasing \code{K_aug}
#'     yields a more stable uncertainty estimate but increases compute.}
#'     \item{r_all}{Numeric in \eqn{[0,1]}. Global \emph{selection rate} in the proportion-based pseudo-labeling mode
#'     (default). Among unlabeled items that pass the uncertainty gate, the trainer targets pseudo-labeling
#'     approximately an \code{r_all} fraction per batch by setting adaptive thresholds on teacher mean scores \eqn{\bar p_i}.}
#'     \item{lambda_fix_max}{ Maximum weight of the FixMatch-style consistency term. The effective weight is ramped up with epoch
#'     and capped at \code{lambda_fix_max}, so the student first stabilizes on labeled losses before trusting
#'     teacher pseudo-labels.}
#'     \item{fix_ramp_epochs}{Number of epochs used for the monotone ramp-up schedule.}
#'     \item{prior_aug}{Character. Type of prior augmentation used to create stochastic, distribution-matched views
#'     for student and teacher (\code{"beta"}, \code{"gauss"} or \code{"dropout"}). The same augmentation family is used for both networks,
#'     but sampled independently to generate noisy yet matched inputs for consistency.}
#'     \item{early_stop_patience}{Early-stopping patience (number of epochs without improvement).}
#'     \item{early_stop_warmup}{Number of warm-up epochs before early stopping is enabled.}
#'   }
#'   Any additional entries are forwarded as \code{--key value} flags; logical \code{TRUE} entries are forwarded
#'   as \code{--key}.
#'
#' @param infer_args Named list of inference hyperparameters overriding \code{.default_infer_args}.
#'   These entries are converted to command-line flags and passed to the Python inference script \code{inference.py}.
#'   Common keys in STREAMS are:
#'   \describe{
#'     \item{latent_dim}{Latent dimension used at inference (should match the trained model).}
#'     \item{mc_samples}{Number of Monte Carlo samples for stochastic inference.}
#'     \item{batch_size}{Mini-batch size used at inference.}
#'   }
#'
#' @param python Character. Path to the Python executable used to run \code{train.py} and \code{inference.py}.
#'   Defaults to \code{Sys.which("python")}; if empty/NULL, a fallback such as \code{"python3"} may be used.
#' @param out_dir Character. Directory used to store intermediate artifacts (Feather inputs/outputs, logs, model checkpoint).
#'   If \code{NULL}, a unique timestamped subdirectory under \code{tempdir()} is created.
#' @param keep_intermediate Logical. If \code{TRUE}, \code{out_dir} is kept after the run; if \code{FALSE}, \code{out_dir}
#'   is deleted at the end.
#' @param n_cores Integer. Number of cores used to fit the \code{m} multi-state models in parallel
#'   via \code{parallel::mclapply}. (Note: \code{mclapply} is not supported on Windows.)
#' @param seed Integer. For full end-to-end reproducibility.
#'
#' @return
#' A Rubin-pooled parametric multi-state model fit based on \code{m} imputations.
#'
#' Pooled estimates and uncertainty are available through \code{coef()}, \code{vcov()},
#' \code{confint()}, and \code{summary(x, coefs = TRUE)}.

#'
#' @seealso \code{\link{prepare_data}}, \code{\link{pu_learning}}, \code{\link{streams_python}},
#'   \code{\link{fit_model}}
#'
#' @examples
#' \dontrun{
#' est <- run_streams(
#'   data = df_panel,
#'   cov_vector = c("sex", "bmi", "smoking"),
#'   m = 20,
#'   clock_assumption = "forward",
#'   distribution = "gompertz",
#'   lab_prop = 0.15,
#'   pu_args = list(max_iter = 1000, tol = 1e-5),
#'   cvae_args = list(max_epochs = 200, latent_dim = 5, prior_aug = "beta"),
#'   out_dir = NULL,
#'   keep_intermediate = TRUE,
#'   n_cores = 4,
#'   seed = 42
#' )
#' }
#'
#' @importFrom utils modifyList
#' @export


run_streams <- function(
    data,
    cov_vector,

    # --- Core analysis ---
    m = 20,
    clock_assumption = "forward",
    distribution = "gompertz",
    custom_formula = NULL,

    # --- PU learning ---
    lab_prop = 0.5,
    pu_args = list(),

    # --- CVAE / FixMatch (macro knobs) ---
    cvae_args = list(),

    # --- Inference ---
    infer_args = list(),

    # --- Execution ---
    python = Sys.which("python"),
    out_dir = NULL,
    keep_intermediate = FALSE,
    n_cores = 4,
    seed = 42
)
{

  # --- merge default arguments
  pu_cfg    <- modifyList(.default_pu_args, pu_args)
  cvae_cfg  <- modifyList(.default_cvae_args, cvae_args)
  infer_cfg <- modifyList(.default_infer_args, infer_args)

  cvae_cfg$seed  <- seed


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


  # --- check input data, clean data and prepare them for model

  check_input_data(data)

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
  set.seed(seed)
  c1 <- matrix(stats::runif(n_patients * m), nrow = n_patients, ncol = m)
  disease_status <- (dat$p_onset > c1) * 1

  disease_age <- matrix(0, nrow = n_patients, ncol = m)
  for (i in 1:n_patients) {
    mu <- dat$age_mu[i]; sig <- dat$age_sd[i]; a <- dat$a[i]; b <- dat$b[i]
    b_safe <- pmin(pmax(b - 1e-3, a + 1e-3), b)
    disease_age[i, ] <- vapply(disease_status[i, ], function(o) {
      if (o == 1) {
        sa <- sample_truncated_normal(mu, sig, a, b_safe)
        if (is.na(sa)) warning(sprintf("NA at i=%d: mu=%f sd=%f a=%f b=%f b_safe=%f",
                                       i, mu, sd, a, b, b_safe))
        sa
      } else b
    }, numeric(1))
  }

  # --- fit multi-state
  if (.Platform$OS.type == "windows" && n_cores > 1) {
    warning("mclapply is not supported on Windows; using n_cores = 1.")
    n_cores <- 1
  }

  all_fits <- parallel::mclapply(1:m, function(j) {
    temp <- cleaned_data
    idx <- which(disease_status[, j] == 1)
    if (length(idx)) {
      temp$onset[idx]     <- 1
      temp$onset_age[idx] <- disease_age[idx, j]
    }
    fit_model(temp, cov_vector, clock_assumption, distribution, custom_formula)

  }, mc.cores = n_cores)

  #-------------------------------------
  # Pooling with Rubin's rules
  #-------------------------------------

  pooled_fit <- pool_rubin_all_transitions(all_fits, cl = 0.95)

  return(pooled_fit)
}



