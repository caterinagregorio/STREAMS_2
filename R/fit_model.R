#' Fit Semi-Parametric Multi-State Models for a Progressive Illness-Death Process
#'
#' This function fits parametric models to the three transitions of a progressive
#' illness–death model
#' using the \pkg{mstate} and \pkg{flexsurv} frameworks.
#' It supports both a **forward timescale** (Markov process) and a **mixed timescale** (Semi-Markov process)
#' through the argument `clock_assumption`.
#'
#' The user may either:
#' - provide a **custom survival formula** (allowing splines, nonlinear effects, interactions, etc.),
#' - or rely on the **default formula** constructed automatically from the covariates in `cov_vector` assuming linear effects.
#'
#' @param data A data frame containing the original dataset in wide format.
#'   It must include:
#'   - `patient_id`: unique individual identifier
#'   - `onset_age`: age at disease onset
#'   - `death_time`: age/time at death
#'   - `onset`: indicator of disease onset
#'   - `dead`: indicator of death
#'   - `age`: baseline age
#'   - all covariates listed in `cov_vector`
#'
#' @param cov_vector A character vector of covariate names to be included
#'   in the model.
#'
#' @param clock_assumption A string specifying the time scale used for model fitting.
#'   Accepted values are:
#'   - `"forward"`: all transitions use the forward timescale `Surv(Tstart, Tstop, status)`
#'   - `"mix"`: transitions 1 and 2 use forward time; transition 3 (disease → death)
#'     uses a reset timescale `Surv(time, status)`
#'
#' @param distribution A character string specifying the parametric distribution passed to
#'   \code{\link[flexsurv]{flexsurvreg}} (e.g., `"weibull"`, `"gompertz"`, `"lognormal"`).
#'
#' @param custom_formula An optional \code{Surv()} formula provided by the user.
#'   If supplied, it replaces the automatically generated formulas and will be used
#'   for **all transitions**, regardless of the clock assumption.
#'   This allows specification of nonlinear or interaction effects.
#'   If \code{NULL} (default), the model formula is constructed automatically from \code{cov_vector}.
#'
#' @return A list of three fitted \code{flexsurvreg} model objects,
#'   corresponding to each transition of the illness–death model.

#' @examples
#' \dontrun{
#'
#' # Default model with linear covariate effects
#' fit_model(
#'   data = df,
#'   cov_vector = c("sex", "bmi"),
#'   clock_assumption = "forward",
#'   distribution = "weibull"
#' )
#'
#' # Custom formula with spline effect on BMI and reset of timescale in the disease status
#' fit_model(
#'   data = df,
#'   cov_vector = c("sex", "bmi"),
#'   clock_assumption = "mix",
#'   distribution = "gompertz",
#'   custom_formula = survival::Surv(time, status) ~
#'       sex + splines::ns(bmi, df = 3)
#' )
#' }
#'
#' @importFrom mstate transMat msprep
#' @importFrom stats as.formula
#' @importFrom survival Surv
#' @importFrom flexsurv flexsurvreg
#' @export

fit_model <- function(data, cov_vector, clock_assumption, distribution,
                      custom_formula = NULL) {

  # Transition structure
  tmat <- mstate::transMat(
    x = list(c(2,3), c(3), c()),
    names = c("Disease-free","Disease","Death")
  )

  # Prepare long format
  data_long <- mstate::msprep(
    data = data, trans = tmat,
    time = c(NA, "onset_age", "death_time"),
    status = c(NA, "onset", "dead"),
    keep = c("age", cov_vector),
    id = "patient_id"
  )

  # Reset times
  data_long$Tstart[data_long$trans < 3] <-
    data_long$Tstart[data_long$trans < 3] + data_long$age[data_long$trans < 3]

  data_long$time <- data_long$Tstop - data_long$Tstart

  # ---- Formula handling ----
  if (is.null(custom_formula)) {
    # Build default formulas
    forward_formula <- stats::as.formula(
      paste("survival::Surv(Tstart, Tstop, status) ~",
            paste(cov_vector, collapse = " + "))
    )

    reset_formula <- stats::as.formula(
      paste("survival::Surv(time, status) ~",
            paste(cov_vector, collapse = " + "))
    )
  } else {
    # User provides one formula that applies to both clocks
    forward_formula <- custom_formula
    reset_formula   <- custom_formula
  }

  # ---- Fit models ----
  fits <- vector("list", 3)

  if (clock_assumption == "forward") {
    for (i in 1:3) {
      fits[[i]] <- flexsurv::flexsurvreg(
        forward_formula,
        data = subset(data_long, trans == i),
        dist = distribution
      )
    }

  } else if (clock_assumption == "mix") {
    for (i in 1:2) {
      fits[[i]] <- flexsurv::flexsurvreg(
        forward_formula,
        data = subset(data_long, trans == i),
        dist = distribution
      )
    }

    fits[[3]] <- flexsurv::flexsurvreg(
      reset_formula,
      data = subset(data_long, trans == 3),
      dist = distribution
    )
  }

  fits
}
