#' Fit Parametric Multi-State Models for a Progressive Illness-Death Process
#'
#' This function fits parametric models to the three transitions of a progressive
#' illness–death model using the \pkg{mstate} and \pkg{flexsurv} frameworks.
#' It supports both a **forward timescale** (Markov process) and a **mixed timescale**
#' (Semi-Markov process) through the argument `clock_assumption`.
#'
#' The user may either:
#' - provide a **custom survival formula** (allowing splines, nonlinear effects, interactions, etc.),
#'   either as a single formula (applied to all transitions) or as a list of three formulas
#'   (one per transition);
#' - or rely on **default formulas** constructed automatically from covariates,
#'   which may be specified globally or per transition.
#'
#' @param data A data frame containing the original dataset in wide format.
#'   It must include:
#'   - `patient_id`: unique individual identifier
#'   - `onset_age`: age at disease onset
#'   - `death_time`: age/time at death
#'   - `onset`: indicator of disease onset
#'   - `dead`: indicator of death
#'   - `age`: baseline age
#'   - all covariates listed in `cov_vector` (or in its per-transition elements).
#'
#' @param cov_vector Either:
#'   \itemize{
#'     \item A character vector of covariate names to be included in the model for
#'           all three transitions;
#'     \item A list of three character vectors, one per transition, specifying
#'           covariates for each transition separately. List elements are named
#'           `"0->1"`, `"0->2"`, `"1->2"`.
#'   }
#'
#' @param clock_assumption A string specifying the time scale used for model fitting.
#'   Accepted values are:
#'   - `"forward"`: all transitions use the forward timescale `Surv(Tstart, Tstop, status)`;
#'   - `"mix"`: transitions 1 and 2 use forward time; transition 3 (disease → death)
#'     uses a reset timescale `Surv(time, status)`.
#'
#' @param distribution A character string specifying the parametric distribution passed to
#'   \code{\link[flexsurv]{flexsurvreg}} (e.g., `"weibull"`, `"gompertz"`, `"lognormal"`).
#'
#' @param custom_formula An optional specification of survival formulas:
#'   \itemize{
#'     \item If \code{NULL} (default), formulas are constructed automatically from
#'           \code{cov_vector}, allowing different covariates per transition.
#'     \item If a single \code{Surv()} formula, it is used for all transitions.
#'     \item If a list of three \code{Surv()} formulas, each element is used for
#'           the corresponding transition (1, 2, 3).
#'           }
#' @return A list of three fitted \code{flexsurvreg} model objects,
#'   corresponding to each transition of the illness–death model.
#'
#' @importFrom mstate transMat msprep
#' @importFrom stats as.formula
#' @importFrom survival Surv
#' @importFrom flexsurv flexsurvreg
#' @export
#'
fit_model <- function(data, cov_vector, clock_assumption, distribution,
                      custom_formula = NULL) {

  # ------------------------------------------------------------
  # 1) Normalise covariate specification
  # ------------------------------------------------------------
  cov_list <- NULL
  if (is.character(cov_vector)) {
    # same covariates for all three transitions
    cov_list <- list(cov_vector, cov_vector, cov_vector)
  } else if (is.list(cov_vector)) {
    if (length(cov_vector) != 3L) {
      stop("If `cov_vector` is a list, it must have length 3 (one element per transition).")
    }
    cov_list <- lapply(cov_vector, function(x) {
      if (is.null(x)) character(0) else as.character(x)
    })
  } else {
    stop("`cov_vector` must be a character vector or a list of length 3.")
  }

  cov_union <- sort(unique(unlist(cov_list)))

  # ------------------------------------------------------------
  # 2) Transition structure and long-format data
  # ------------------------------------------------------------
  tmat <- mstate::transMat(
    x = list(c(2, 3), c(3), c()),
    names = c("Disease-free", "Disease", "Death")
  )

  data_long <- mstate::msprep(
    data   = data,
    trans  = tmat,
    time   = c(NA, "onset_age", "death_time"),
    status = c(NA, "onset", "dead"),
    keep   = c("age", cov_union),
    id     = "patient_id"
  )

  # Reset times for forward timescale:
  # transitions from starting state are measured from baseline age
  data_long$Tstart[data_long$trans < 3] <-
    data_long$Tstart[data_long$trans < 3] + data_long$age[data_long$trans < 3]

  data_long$time <- data_long$Tstop - data_long$Tstart

  # ------------------------------------------------------------
  # 3) Formula handling (per transition)
  # ------------------------------------------------------------

  forward_formulas <- vector("list", 3L)
  reset_formulas   <- vector("list", 3L)

  if (is.null(custom_formula)) {

    for (k in 1:3) {
      covs_k <- cov_list[[k]]
      rhs <- if (length(covs_k) > 0L) paste(covs_k, collapse = " + ") else "1"

      forward_formulas[[k]] <- stats::as.formula(
        paste("survival::Surv(Tstart, Tstop, status) ~", rhs)
      )
      reset_formulas[[k]] <- stats::as.formula(
        paste("survival::Surv(time, status) ~", rhs)
      )
    }

  } else if (inherits(custom_formula, "formula")) {
    # Single user formula for all transitions
    forward_formulas <- list(custom_formula, custom_formula, custom_formula)
    reset_formulas   <- forward_formulas

  } else if (is.list(custom_formula) &&
             all(vapply(custom_formula, inherits, logical(1), "formula"))) {

    if (length(custom_formula) != 3L) {
      stop("If `custom_formula` is a list, it must have length 3 (one formula per transition).")
    }
    forward_formulas <- custom_formula
    reset_formulas   <- custom_formula

  } else {
    stop("`custom_formula` must be NULL, a single formula, or a list of 3 formulas.")
  }

  # ------------------------------------------------------------
  # 4) Fit models by transition and clock assumption
  # ------------------------------------------------------------
  fits <- vector("list", 3L)

  if (clock_assumption == "forward") {

    for (i in 1:3) {
      fits[[i]] <- flexsurv::flexsurvreg(
        formula = forward_formulas[[i]],
        data    = subset(data_long, trans == i),
        dist    = distribution
      )
    }

  } else if (clock_assumption == "mix") {

    # Transitions 1 and 2: forward clock
    for (i in 1:2) {
      fits[[i]] <- flexsurv::flexsurvreg(
        formula = forward_formulas[[i]],
        data    = subset(data_long, trans == i),
        dist    = distribution
      )
    }

    # Transition 3: reset clock
    fits[[3]] <- flexsurv::flexsurvreg(
      formula = reset_formulas[[3]],
      data    = subset(data_long, trans == 3),
      dist    = distribution
    )

  } else {
    stop("`clock_assumption` must be 'forward' or 'mix'.")
  }

  fits
}
