#' Positive-unlabelled learning under a Selected-At-Random assumption
#'
#' Implements an EM-based Positive–Unlabelled (PU) learning algorithm under a
#' Selected-At-Random (SAR) model for the labeling mechanism. It estimates:
#' \itemize{
#'   \item \eqn{p(x)=P(Y=1 \mid X_p)}: probability of developing the disease given classification features
#'         \code{features_cl};
#'   \item \eqn{e(x)=P(S=1 \mid Y=1, X_e)}: selection/labeling propensity for truly positive cases, modeled using a
#'         fixed default set of selection features (see Details) plus any user-supplied additions \code{features_prop_add};
#'   \item \eqn{r=P(Y=1 \mid X_p, X_e, S)}: posterior probability for unlabeled examples.
#' }
#'
#' @param df A \code{data.frame} or \code{data.table} containing at least:
#'   \itemize{
#'     \item \code{patient_id}: unique patient identifier;
#'     \item \code{onset}: observed label indicator \code{S} (1 if labeled positive, 0 otherwise);
#'     \item all variables listed in \code{features_cl};
#'     \item all default selection features (see Details) and any variables listed in \code{features_prop_add}.
#'   }
#'   The function assumes one row per patient.
#'
#' @param features_cl Character vector of covariate names used to model disease probability \eqn{p(x)}.
#' @param features_prop_add Optional character vector of additional covariates to append to the
#'   fixed default selection feature set used to model the labeling mechanism \eqn{e(x)}.
#'   These should capture the observation/follow-up process.
#'   \strong{Warning:} Adding variables that overlap with \code{features_cl}
#'   may introduce identifiability issues (the selection model may absorb disease signal), add them
#'   only if there's strong evidence that the variables are related to both the disease
#'   probability and the likelihood of being observed.
#'
#' @param pu_args Named list of optional arguments controlling the EM algorithm. Supported keys:
#'   \itemize{
#'     \item \code{max_iter}: Integer, maximum number of EM iterations (default \code{1000}).
#'     \item \code{tol}: Numeric, convergence tolerance on the log-likelihood difference (default \code{1e-3}).
#'     \item \code{clip}: Numeric, probability clipping bound for numerical stability (default \code{1e-3}).
#'     \item \code{damp}: Numeric in \eqn{(0,1)}, damping factor for convex updates of \eqn{p(x)} and \eqn{e(x)}
#'           (default \code{0.3}). \code{damp}=1 uses only the new estimates (no damping).
#'     \item \code{shrink_k}: Numeric, non-negative shrinkage parameter for \eqn{p(x)} in regions with low \eqn{e(x)}
#'           (default \code{0.0}).
#'     \item \code{verbose}: Logical, if \code{TRUE} prints log-likelihood and summary statistics at each iteration
#'           (default \code{FALSE}).
#'   }
#'   Any provided values override the defaults.
#'
#' @details
#' The SAR selection model uses a \emph{fixed, non-editable} default selection feature set:
#' \itemize{
#'   \item \code{interval}
#'   \item \code{visit_rate}
#'   \item \code{number_visits}
#' }
#' Users may only append additional selection covariates via \code{features_prop_add}.
#'
#' The EM-like procedure is:
#' \enumerate{
#'   \item Initialize \eqn{p(x)} and \eqn{e(x)} under SCAR.
#'   \item Alternate:
#'     \itemize{
#'       \item \strong{E-step:} compute \eqn{r=P(Y=1 \mid X_p,X_e,S)} for unlabeled examples;
#'       \item \strong{M-step for \eqn{p(x)}:} fit a quasi-binomial GLM of \code{r} on \code{features_cl};
#'       \item \strong{M-step for \eqn{e(x)}:} fit a weighted logistic GLM for \code{S} on the selection features.
#'     }
#'   \item Calibrate the intercept of \eqn{e(x)} so that \eqn{E[p(x)e(x)] \approx \bar{S}}.
#' }
#'
#' @return A \code{tibble} with one row per patient and:
#' \itemize{
#'   \item \code{patient_id}, \code{onset}
#'   \item \code{fhat}: initial \eqn{P(S=1 \mid X_p)} (SCAR init)
#'   \item \code{p_pos}: final \eqn{p(x)=P(Y=1 \mid X_p)}
#'   \item \code{e_prop}: final \eqn{e(x)=P(S=1 \mid Y=1, X_e)}
#'   \item \code{r}: posterior \eqn{P(Y=1 \mid X_p,X_e,S)}
#'   \item \code{f_recon}: reconstructed \eqn{P(S=1 \mid X_p,X_e)} = \code{p_pos} * \code{e_prop}
#' }
#'
#' @examples
#' \dontrun{
#' features_cl <- c("bmi0", "Hypertension", "Dyslipidemia")
#' # default selection features are always used: length_followup, number_visits
#' # optionally append more selection features:
#' features_prop_add <- c("n_clinic_contacts")
#'
#' pu_res <- pu_learning(
#'   df              = df_patients,
#'   features_cl     = features_cl,
#'   features_prop_add = features_prop_add,
#'   pu_args         = list(max_iter = 200, tol = 1e-4, verbose = TRUE)
#' )
#' head(pu_res)
#' }
#'
#' @importFrom stats glm predict binomial quasibinomial
#' @importFrom stats model.matrix reformulate
#' @importFrom stats qlogis plogis uniroot
#' @importFrom tibble tibble
#' @export
pu_learning <- function(df, features_cl, features_prop_add = NULL, pu_args = list()) {

  stopifnot(all(c("patient_id", "onset") %in% names(df)))

  # --- fixed defaults (non-editable)
  default_features_prop <- c("interval", "visit_rate", "number_visits")


  default_pu_args <- list(
    max_iter = 1000,
    tol = 1e-3,
    clip = 1e-3,
    damp = 0.3,
    shrink_k = 0.0,
    verbose = FALSE
  )

  pu_args <- modifyList(default_pu_args, pu_args)

  max_iter <- pu_args$max_iter
  tol      <- pu_args$tol
  clip     <- pu_args$clip
  damp     <- pu_args$damp
  shrink_k <- pu_args$shrink_k
  verbose  <- pu_args$verbose

  # --- build selection feature set: defaults + user additions
  features_prop <- default_features_prop
  if (!is.null(features_prop_add)) {
    features_prop <- unique(c(features_prop, features_prop_add))
  }


  eps <- 1e-12

  # -- build model matrices
  Xp <- model.matrix(reformulate(features_cl), df)
  Xp_noage <- model.matrix(reformulate(features_cl[features_cl != "age"]), df)
  Xe <- model.matrix(reformulate(features_prop), data = df)

  Xp_n <- Xp[, -1, drop = FALSE]
  Xp_noage_n <- Xp_noage[, -1, drop = FALSE]
  Xe_n <- Xe[, -1, drop = FALSE]

  # -- drop near-constant columns (prevents intercept-only fits)
  nzv <- function(M, thr = 1e-8) if (ncol(M)) M[, apply(M, 2, sd, na.rm = TRUE) > thr, drop = FALSE] else M
  Xp_n <- nzv(Xp_n)
  Xp_noage_n <- nzv(Xp_noage_n)
  Xe_n <- nzv(Xe_n)

  S     <- as.numeric(df$onset)
  n     <- length(S)
  s_bar <- mean(S) #mean observed diseases

  # -- init: fhat = P(S=1 given Xp) via plain logistic, then SCAR start
  fit_f <- suppressWarnings(stats::glm(S ~ Xp_n, family = stats::binomial())) #P(S=1 given Xp)=P(Y=1 given Xp)P(S=1 given Y=1,Xp)=p(x)e(x) under scar
  fhat  <- as.numeric(stats::predict(fit_f, type = "response"))
  fhat  <- pmin(pmax(fhat, clip), 1 - clip)

  fit_f_noage <- stats::glm(S ~ Xp_noage_n, family = stats::binomial())
  fhat_noage  <- as.numeric(stats::predict(fit_f_noage, type = "response"))
  fhat_noage  <- pmin(pmax(fhat_noage, clip), 1 - clip)

  roc_obj_init <- pROC::roc(response = S, predictor = fhat_noage, quiet = TRUE)
  ci_auc_init  <- pROC::ci.auc(roc_obj_init, conf.level = 0.95, method = "bootstrap", boot.n = 500)
  auc_lo_init  <- as.numeric(ci_auc_init[1]); auc_mid_init <- as.numeric(ci_auc_init[2]); auc_hi_init <- as.numeric(ci_auc_init[3])

  message(sprintf("AUC init PU-learning=%.3f; 95%% CI=[%.3f, %.3f]", auc_mid_init, auc_lo_init, auc_hi_init))

  #Adaptive strinkage based on AUC init
  if (auc_lo_init < 0.65 & shrink_k==0) {
    shrink_k <- max(shrink_k, 1.0)
    message("Strong shrinkage imposed in PU-learning!")
  } else if (auc_lo_init <= 0.65 && auc_hi_init >= 0.70 & shrink_k==0) {
    shrink_k<- max(shrink_k, 0.5)
    message("Moderate shrinkage imposed in PU-learning!")
  } else if (auc_hi_init > 0.70 & shrink_k==0) {
    shrink_k <- max(shrink_k, 0.1)  # micro-shrink "fail-safe"
    message("Micro shrinkage imposed in PU-learning!")
  }

  c_hat <- mean(fhat[S == 1], na.rm = TRUE); if (!is.finite(c_hat) || c_hat <= clip) c_hat <- 0.5  # labelling rate among true positive P(S=1 given Y=1)
  p <- pmin(pmax(fhat / c_hat, clip), 1 - clip) # P(Y=1∣Xp) inverted from above expression
  e <- rep(pmin(pmax(c_hat, 0.05), 0.99), n) #selection model initialized as constant P(S=1 given Y=1) = c_hat but actually e(x) = P(S=1 given Y=1,Xp)

  ll <- function(p, e) {
    pe <- p * e
    sum(S * log(pmin(pmax(pe, eps), 1)) + (1 - S) * log(pmin(pmax(1 - pe, eps), 1)))
  }
  ll_old <- ll(p, e)
  if (verbose) cat(sprintf("Init: ll=%.6f mean p=%.3f mean e=%.3f mean pe=%.3f s_bar=%.3f\n",
                           ll_old, mean(p), mean(e), mean(p*e), s_bar))

  for (it in 1:max_iter) {
    p_old <- p; e_old <- e

    # ---- E-step: r = P(Y=1 given X, S) if S=1 -> r=1 if S=0 -> P(S=0 given Y=1,X)P(Y=1 given X)/P(S=0 given X) = (1-e)*p/(1-p*e)
    denom <- 1 - p * e
    r <- ifelse(S == 1, 1, pmin(pmax(p * (1 - e) / pmax(denom, eps), clip), 1 - clip))

    # ---- M-step p(x): fractional logistic
    #    r is a proportion; binomial GLM with weights=1 is fine
    fit_p <- suppressWarnings(stats::glm(r ~ Xp_n, family = quasibinomial()))
    #fit_p <- stats::glm(r ~ Xp_n, family = stats::binomial(), weights = rep(1, n))
    p_new <- as.numeric(stats::predict(fit_p, type = "response"))
    p_new <- pmin(pmax(p_new, clip), 1 - clip)

    # ---- M-step e(x): weighted logistic with weights = 1 (S=1) or r (S=0)


    w_e <- ifelse(S == 1, 1, r)
    fit_e <- suppressWarnings(stats::glm(S ~ Xe_n, family = stats::binomial(), weights = w_e))
    e_new <- as.numeric(stats::predict(fit_e, type = "response"))
    e_new <- pmin(pmax(e_new, clip), 1 - clip)

    # shrink p(x) when e(x) small
    if (shrink_k > 0){
      tau_i <- 1 + shrink_k * (1 - e_new)
      lp <- qlogis(p_new)
      p_new <- plogis(lp / tau_i)
    }


    # ---- intercept calibration for e(x): enforce mean(p*e)=s_bar
    logit_e <- qlogis(pmin(pmax(e_new, eps), 1 - eps))
    delta <- tryCatch(uniroot(function(d) mean(p_new * plogis(logit_e + d)) - s_bar,
                              c(-12, 12))$root, error = function(...) 0)
    e_new <- plogis(logit_e + delta)
    e_new <- pmin(pmax(e_new, clip), 1 - clip)

    # ---- damping to avoid oscillations
    p <- (1 - damp) * p_old + damp * p_new
    e <- (1 - damp) * e_old + damp * e_new

    # ---- convergence
    ll_new <- ll(p, e)
    if (verbose) cat(sprintf("Iter %02d: ll=%.6f Δ=%.3e mean p=%.3f mean e=%.3f mean pe=%.3f sd(p)=%.4f sd(e)=%.4f\n",
                             it, ll_new, ll_new - ll_old, mean(p), mean(e), mean(p*e), sd(p), sd(e)))
    if (!is.finite(ll_new)) { warning("Numerical issue; stopping."); break }
    if (abs(ll_new - ll_old) < tol) { if (verbose) cat("Converged.\n"); break }
    ll_old <- ll_new
  }

  # final posterior r
  denom <- 1 - p * e
  r <- ifelse(S == 1, 1, pmin(pmax(p * (1 - e) / pmax(denom, eps), clip), 1 - clip))

  tibble::tibble(
    patient_id = df$patient_id,
    onset      = S,        # observed positive indicator (S=1 if labeled positive)
    fhat       = fhat,     # initial P(S=1 given Xp) from a plain logistic on Xp
    p_pos      = p,        # final P(Y=1 given Xp) disease risk
    e_prop     = e,        # final P(S=1 given Y=1, Xe) selection/labeling probability
    r          = r,        # posterior P(Y=1 given X, S): 1 for S=1; in (0,1) for S=0
    f_recon    = p * e     # implied P(S=1 given X_e, X_p) under SAR: should match mean(S) on average
  )
}
