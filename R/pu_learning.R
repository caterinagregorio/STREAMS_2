pu_learning <- function(df, features_cl, features_prop, max_iter = 300, tol = 1e-3, clip = 1e-3, damp = 0.3, shrink_k = 0.0, verbose = TRUE) {
  stopifnot(all(c("patient_id","onset") %in% names(df)))
  eps <- 1e-12

  # -- build model matrices
  Xp <- model.matrix(reformulate(features_cl), df)
  df$log_followup <- log1p(df$length_followup)
  df$log_visits   <- log1p(df$visits)
  form_e <- ~ scale(log_followup) + scale(log_visits) + scale(interval)
  Xe <- model.matrix(form_e, data = df)
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
    p_pos      = p,        # final P(Y=1 | Xp) disease risk (baseline-only features)
    e_prop     = e,        # final P(S=1 | Y=1, Xe) selection/labeling probability
    r          = r,        # posterior P(Y=1 | X, S): 1 for S=1; in (0,1) for S=0
    f_recon    = p * e     # implied P(S=1 | X) under SAR: should match mean(S) on average
  )
}
