#' Positive-unlabelled learning under a Selected-At-Random assumption
#'
#' This function implements an Expectation-maximization Positive–Unlabeled (PU) learning algorithm under
#' Selection-At-Random (SAR) model for the labeling mechanism. It estimates:
#' \itemize{
#'   \item \eqn{p(x) = P(Y = 1 \mid X_p)}: probability of developing the disease given a set of
#'          classification features \code{features_cl};
#'   \item \eqn{e(x) = P(S = 1 \mid Y = 1, X_e)}: propensity score, probability that a truly
#'         positive case is labeled as positive, given the selection
#'         features \code{features_prop};
#'   \item the posterior probability \eqn{r = P(Y = 1 \mid X_p, x_e, S)} for unlabeled examples.
#' }
#'
#'
#' @param df A \code{data.frame} or \code{data.table} containing at least the
#'   following columns:
#'   \itemize{
#'     \item \code{patient_id}: unique patient identifier;
#'     \item \code{onset}: observed label indicator \code{S} (1 if the patient
#'           is labeled positive, 0 otherwise);
#'     \item all variables listed in \code{features_cl};
#'     \item all variables listed in \code{features_prop}.
#'   }
#'   The function assumes one row per patient.
#' @param features_cl Character vector with the names of the covariates used
#'   to model the disease probability.
#' @param features_prop Character vector with the names of the features explaining the
#'   observation process or dependent on the follow-up itself.
#' @param max_iter Integer, maximum number of EM iterations. Default is
#'   \code{800}.
#' @param tol Numeric, convergence tolerance on the log-likelihood difference
#'   between successive iterations. Default is \code{1e-3}.
#' @param clip Numeric, lower bound used to clip probabilities away from
#'   \code{0} and \code{1} for numerical stability. Default is \code{1e-3}.
#' @param damp Numeric in \eqn{(0,1)}, damping factor used to update
#'   \eqn{p(x)} and \eqn{e(x)} as convex combinations of old and new estimates
#'   to reduce oscillations. \code{damp} = 1 correspondes to null contribute of the old estimate.
#'   Default is \code{0.3}.
#' @param shrink_k Numeric, non-negative shrinkage parameter. When
#'   \code{shrink_k > 0}, \eqn{p(x)} is shrunk
#'   more strongly where \eqn{e(x)} is small, effectively penalizing regions
#'   with low labeling propensity. Default is \code{0.0} (no shrinkage).
#' @param verbose Logical, if \code{TRUE} (default) prints the log-likelihood
#'   and summary statistics at initialization and at each EM iteration.
#'
#' @details
#' The function uses an EM-like procedure:
#' \enumerate{
#'   \item Initialize \code{p(x)} and \code{e(x)} under SCAR assumption.
#'   \item Alternates between:
#'         \itemize{
#'           \item \strong{E-step:} computes posterior probabilities
#'                 \eqn{r = P(Y = 1 \mid X_p, X_e, S)} for unlabeled examples;
#'           \item \strong{M-step for \eqn{p(x)}:} fits a quasi-binomial GLM
#'                 of \code{r} on \code{features_cl};
#'           \item \strong{M-step for \eqn{e(x)}:} fits a weighted logistic GLM
#'                 for the selection probability usingon \code{features_prop};
#'         }
#'   \item Calibrates the intercept of \eqn{e(x)} so that
#'         \eqn{E[p(x) e(x)] \approx \bar{S}}, the observed fraction of labeled
#'         positives.
#' }
#'
#' @return A \code{tibble} with one row per patient and the following columns:
#' \itemize{
#'   \item \code{patient_id}: patient identifier copied from \code{df};
#'   \item \code{onset}: observed label indicator \code{S};
#'   \item \code{fhat}: initial estimate \eqn{P(S = 1 \mid X_p)} under SCAR from a
#'         logistic regression of \code{S} on \code{features_cl};
#'   \item \code{p_pos}: final estimate \eqn{p(x) = P(Y = 1 \mid X_p)}, i.e.
#'         of the disease probability based on \code{features_cl};
#'   \item \code{e_prop}: final estimate \eqn{e(x) = P(S = 1 \mid Y = 1, X_e)},
#'         the selection probability;
#'   \item \code{r}: posterior probability \eqn{P(Y = 1 \mid X_p, X_e, S)}; that equals
#'         \code{1} for labeled positives (\code{S = 1}) and lies in \eqn{(0,1)}
#'         for unlabeled cases;
#'   \item \code{f_recon}: reconstructed probability \eqn{P(S = 1 \mid X_e, X_p)} under
#'         the SAR model, given by \code{p(x)} * \code{e(x)}. Its average should be
#'         close to the observed fraction of labeled positives \code{mean(S)}.
#' }
#'
#' @examples
#' \dontrun{
#' # Suppose df_patients has one row per patient and includes:
#' # patient_id, onset, length_followup, number_visits, and some disease associated covariates
#' features_cl   <- c("bmi0", "Hypertension", "Dyslipidemia")
#' features_prop <- c("length_followup", "number_visits")
#'
#' pu_res <- pu_learning(
#'   df            = df_patients,
#'   features_cl   = features_cl,
#'   features_prop = features_prop,
#'   max_iter      = 200,
#'   tol           = 1e-4,
#'   verbose       = TRUE
#' )
#'
#' head(pu_res)
#' }
#'
#' @import stats tibble
#' @export

pu_learning <- function(df, features_cl, features_prop, max_iter = 800, tol = 1e-3, clip = 1e-3, damp = 0.3, shrink_k = 0.0, verbose = FALSE) {
  stopifnot(all(c("patient_id","onset") %in% names(df)))
  eps <- 1e-12

  # -- build model matrices
  Xp <- model.matrix(reformulate(features_cl), df)
  Xe <- model.matrix(reformulate(features_prop), data = df)
  Xp_n <- Xp[, -1, drop = FALSE]
  Xe_n <- Xe[, -1, drop = FALSE]

  # -- drop near-constant columns (prevents intercept-only fits)
  nzv <- function(M, thr = 1e-8) if (ncol(M)) M[, apply(M, 2, sd, na.rm = TRUE) > thr, drop = FALSE] else M
  Xp_n <- nzv(Xp_n)
  Xe_n <- nzv(Xe_n)

  S     <- as.numeric(df$onset)
  n     <- length(S)
  s_bar <- mean(S) #mean observed diseases

  # -- init: fhat = P(S=1|Xp) via plain logistic, then SCAR start
  fit_f <- stats::glm(S ~ Xp_n, family = stats::binomial()) #P(S=1∣Xp)=P(Y=1∣Xp)P(S=1∣Y=1,Xp)=p(x)e(x) under scar
  fhat  <- as.numeric(stats::predict(fit_f, type = "response"))
  fhat  <- pmin(pmax(fhat, clip), 1 - clip)

  c_hat <- mean(fhat[S == 1], na.rm = TRUE); if (!is.finite(c_hat) || c_hat <= clip) c_hat <- 0.5  # labelling rate among true positive P(S=1|Y=1)
  p <- pmin(pmax(fhat / c_hat, clip), 1 - clip) # P(Y=1∣Xp) inverted from above expression
  e <- rep(pmin(pmax(c_hat, 0.05), 0.99), n) #selection model initialized as constant P(S=1|Y=1) = c_hat but actually e(x) = P(S=1∣Y=1,Xp)

  ll <- function(p, e) {
    pe <- p * e
    sum(S * log(pmin(pmax(pe, eps), 1)) + (1 - S) * log(pmin(pmax(1 - pe, eps), 1)))
  }
  ll_old <- ll(p, e)
  if (verbose) cat(sprintf("Init: ll=%.6f mean p=%.3f mean e=%.3f mean pe=%.3f s_bar=%.3f\n",
                           ll_old, mean(p), mean(e), mean(p*e), s_bar))

  for (it in 1:max_iter) {
    p_old <- p; e_old <- e

    # ---- E-step: r = P(Y=1 | X, S) if S=1 -> r=1 if S=0 -> P(S=0∣Y=1,X)P(Y=1∣X)/P(S=0|X) = (1-e)*p/(1-p*e)
    denom <- 1 - p * e
    r <- ifelse(S == 1, 1, pmin(pmax(p * (1 - e) / pmax(denom, eps), clip), 1 - clip))

    # ---- M-step p(x): fractional logistic
    #    r is a proportion; binomial GLM with weights=1 is fine
    fit_p <- stats::glm(r ~ Xp_n, family = quasibinomial())
    #fit_p <- stats::glm(r ~ Xp_n, family = stats::binomial(), weights = rep(1, n))
    p_new <- as.numeric(stats::predict(fit_p, type = "response"))
    p_new <- pmin(pmax(p_new, clip), 1 - clip)

    # ---- M-step e(x): weighted logistic with weights = 1 (S=1) or r (S=0)


    w_e <- ifelse(S == 1, 1, r)
    fit_e <- stats::glm(S ~ Xe_n, family = stats::binomial(), weights = w_e)
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
    fhat       = fhat,     # initial P(S=1 | Xp) from a plain logistic on Xp
    p_pos      = p,        # final P(Y=1 | Xp) disease risk
    e_prop     = e,        # final P(S=1 | Y=1, Xe) selection/labeling probability
    r          = r,        # posterior P(Y=1 | X, S): 1 for S=1; in (0,1) for S=0
    f_recon    = p * e     # implied P(S=1 | X_e, X_p) under SAR: should match mean(S) on average
  )
}
