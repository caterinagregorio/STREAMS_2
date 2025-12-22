# ===================== Run over real data  ========================
library(devtools)
#panel_data <- readRDS("panel_data.RDS")
panel_data <- panel_data %>%
  filter(!is.na(onset))

cov_vector <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia", "educ_el", "ALCO_CONSUMP", "R_SMOKE", "ws0", "drugs0",
                "dm_sex", "life_alone", "sei_long_cat", "fin_strain_early")

#check_input_data(panel_data)

impute_one <- function(x) {
  if (is.numeric(x)) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    return(x)
  }
  if (is.factor(x) || is.character(x)) {
    mode_val <- names(sort(table(x), decreasing = TRUE))[1]
    x[is.na(x)] <- mode_val
    return(x)
  }
  x
}

panel_data <- panel_data %>%
  mutate(across(all_of(cov_vector), impute_one))

t0 <- Sys.time()
fit_streams <- run_streams(
    data = panel_data,
    cov_vector = cov_vector,
    python = Sys.which("python3"),
    pu_args = list(verbose = TRUE)
)
t1 <- Sys.time()
time_streams <- as.numeric(difftime(t1, t0, units = "secs"))

lapply(fit_streams, function(x) summary(x, coefs = TRUE))

#trying all methods for my class flexsurvreg_pooled
trans1_streams <- fit_streams[[1]]
class(trans1_streams)
print(trans1_streams)
summary(trans1_streams, coefs = TRUE) # al momento è uguale al print ma si puo arricchire (distribuzione, formula, cov usate..)
confint(trans1_streams)
attr(trans1_streams, "rubin")

streams_coefs <- lapply(fit_streams, stats::coef)
names(streams_coefs) <- c("0->1", "0->2", "1->2")


# COMPARING WITH MSM

prepare_msm<- function(df){
  panel_data_death <- df%>% filter(dead==1 ) %>% group_by(patient_id) %>% mutate(age=death_time) %>% distinct(patient_id,.keep_all = T)
  df2 <- rbind(df,panel_data_death) %>% group_by(patient_id) %>%  mutate(state=case_when(max(onset)==1 & onset_age<=age& age!=death_time~2 ,
                                                                                         death_time==age & dead==1 ~3,
                                                                                         TRUE~1)) %>% arrange(patient_id)
  return(df2)
}

msm_main <- function(df, cov_vector) {

  df <- prepare_msm(df)
  Q <- rbind(c(0, 1, 1),
             c(0, 0, 1),
             c(0, 0, 0))

  cov_vector <- intersect(cov_vector, names(df))

  cov_formula <- if (length(cov_vector) > 0) {
    stats::as.formula(paste("~", paste(cov_vector, collapse = " + ")))
  } else {
    ~ 1
  }

  fit <- tryCatch(
    msm::msm(
      state ~ age,
      subject    = patient_id,
      data       = df,
      qmatrix    = Q,
      covariates = cov_formula,
      censor     = 99,
      control    = list(fnscale = 1000, maxit = 1000),
      deathexact = TRUE
    ),
    error = function(e) NULL
  )

  coln <- c("rate", cov_vector)
  mat <- matrix(NA_real_, nrow = 3, ncol = length(coln),
                dimnames = list(c("0->1", "0->2", "1->2"), coln))

  if (is.null(fit)) {
    return(list(model = NULL, params = mat))
  }


  v <- fit$estimates
  vn <- names(v)

  # baseline intensities for 3 transitions then cov effects per transition
  needed <- 3 + 3 * length(cov_vector)
  vv <- rep(NA_real_, needed)
  vv[seq_len(min(length(v), needed))] <- v[seq_len(min(length(v), needed))]

  mat[, "rate"] <- vv[1:3]
  if (length(cov_vector) > 0) {
    off <- 4
    for (cv in cov_vector) {
      mat[, cv] <- vv[off:(off + 2)]
      off <- off + 3
    }
  }
  list(model = fit, params = mat)
}

cov_vector <- c("bmi0", "Diabetes", "Hypertension", "Dyslipidemia", "educ_el", "ALCO_CONSUMP", "R_SMOKE", "ws0", "drugs0",
                "dm_sex", "life_alone", "sei_long_cat", "fin_strain_early")
coerce_covs_for_msm <- function(panel_data, cov_vector,
                                keep_numeric = c("bmi0", "drugs0", "ws0"),
                                ref_level = NULL) {
  df <- panel_data

  cov_vector <- intersect(cov_vector, names(df))
  keep_numeric <- intersect(keep_numeric, cov_vector)
  to_factor <- setdiff(cov_vector, keep_numeric)

  # convert selected covariates to factor
  for (v in to_factor) {
    # keep NAs; coerce characters/numerics to factor
    df[[v]] <- as.factor(df[[v]])

    # optionally enforce a reference level
    # ref_level can be:
    # - NULL (do nothing)
    # - a named list: list(ALCO_CONSUMP = "1", R_SMOKE = "1", Diabetes = "0", ...)
    if (!is.null(ref_level) && !is.null(ref_level[[v]])) {
      r <- as.character(ref_level[[v]])
      lev <- levels(df[[v]])
      if (r %in% lev) {
        df[[v]] <- stats::relevel(df[[v]], ref = r)
      } else {
        warning(sprintf("ref_level '%s' not found in levels(%s): %s",
                        r, v, paste(lev, collapse = ", ")))
      }
    }
  }

  # ensure numeric ones are numeric
  for (v in keep_numeric) {
    df[[v]] <- as.numeric(df[[v]])
  }

  df
}
ref_levels <- list(
  Diabetes = "1",
  Hypertension = "1",
  Dyslipidemia = "1",
  educ_el = "0",
  ALCO_CONSUMP = "2",
  R_SMOKE = "2",
  dm_sex = "1",
  life_alone = "1",
  sei_long_cat = "1",
  fin_strain_early = "1"
)

panel_data_msm <- coerce_covs_for_msm(
  panel_data,
  cov_vector = cov_vector,
  keep_numeric = c("bmi0", "drugs0", "ws0"),
  ref_level = ref_levels
)

t0 <- Sys.time()
fits_msm <-   msm_main(panel_data_msm, cov_vector)
t1 <- Sys.time()
time_msm <- as.numeric(difftime(t1, t0, units = "secs") )

msm_coefs <- lapply(seq_len(nrow(fits_msm$params)), function(i) {
  v <- as.numeric(fits_msm$params[i, ])
  names(v) <- colnames(fits_msm$params)
  v
})
names(msm_coefs) <- rownames(fits_msm$params)



compare_transition_coefs <- function(coefs_A, coefs_B,
                                     label_A = "A",
                                     label_B = "B",
                                     only_common = TRUE,
                                     transitions = NULL,
                                     plot = TRUE) {

  as_list_of_named_vecs <- function(x) {
    if (is.null(x)) stop("Input is NULL.")

    # single named vector -> one transition
    if (is.numeric(x) && !is.matrix(x) && !is.data.frame(x)) {
      if (is.null(names(x))) stop("Named vector required (names are coefficient terms).")
      return(list(model = x))
    }

    # matrix/data.frame: rows=transitions, cols=terms
    if (is.matrix(x) || is.data.frame(x)) {
      x <- as.matrix(x)
      if (is.null(rownames(x))) rownames(x) <- paste0("trans", seq_len(nrow(x)))
      if (is.null(colnames(x))) stop("Matrix/data.frame must have colnames (terms).")

      out <- lapply(seq_len(nrow(x)), function(i) {
        v <- as.numeric(x[i, ])
        names(v) <- colnames(x)
        v
      })
      names(out) <- rownames(x)
      return(out)
    }

    # list: each element a named numeric vector
    if (is.list(x)) {
      if (is.null(names(x))) names(x) <- paste0("trans", seq_along(x))
      ok <- vapply(x, function(v) is.numeric(v) && !is.null(names(v)), logical(1))
      if (!all(ok)) stop("List elements must be named numeric vectors (names are terms).")
      return(x)
    }

    stop("Unsupported type for coefficients. Use list, matrix/data.frame, or named numeric vector.")
  }

  A <- as_list_of_named_vecs(coefs_A)
  B <- as_list_of_named_vecs(coefs_B)

  # decide which transitions to compare
  trans_all <- union(names(A), names(B))
  if (!is.null(transitions)) trans_all <- intersect(trans_all, transitions)

  make_long <- function(lst, method_label) {
    do.call(rbind, lapply(names(lst), function(tr) {
      v <- lst[[tr]]
      data.frame(
        transition = tr,
        term = names(v),
        estimate = as.numeric(v),
        method = method_label,
        stringsAsFactors = FALSE
      )
    }))
  }

  dA <- make_long(A, label_A)
  dB <- make_long(B, label_B)

  dA <- dA[dA$transition %in% trans_all, , drop = FALSE]
  dB <- dB[dB$transition %in% trans_all, , drop = FALSE]

  # keep only common terms per transition if requested
  if (isTRUE(only_common)) {
    keep_rows <- function(d1, d2) {
      out <- lapply(trans_all, function(tr) {
        t1 <- d1$term[d1$transition == tr]
        t2 <- d2$term[d2$transition == tr]
        common <- intersect(t1, t2)
        common
      })
      names(out) <- trans_all
      out
    }
    common_by_tr <- keep_rows(dA, dB)

    dA <- do.call(rbind, lapply(trans_all, function(tr) {
      dd <- dA[dA$transition == tr, , drop = FALSE]
      dd[dd$term %in% common_by_tr[[tr]], , drop = FALSE]
    }))
    dB <- do.call(rbind, lapply(trans_all, function(tr) {
      dd <- dB[dB$transition == tr, , drop = FALSE]
      dd[dd$term %in% common_by_tr[[tr]], , drop = FALSE]
    }))
  }

  long <- rbind(dA, dB)
  long <- long[order(long$transition, long$term, long$method), , drop = FALSE]
  rownames(long) <- NULL

  # wide + difference (B - A)
  wide <- reshape(long[, c("transition", "term", "method", "estimate")],
                  idvar = c("transition", "term"),
                  timevar = "method",
                  direction = "wide")

  colA <- paste0("estimate.", label_A)
  colB <- paste0("estimate.", label_B)
  wide$diff_B_minus_A <- if (colA %in% names(wide) && colB %in% names(wide)) {
    wide[[colB]] - wide[[colA]]
  } else {
    NA_real_
  }

  p <- NULL
  if (isTRUE(plot)) {
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      p <- ggplot2::ggplot(long, ggplot2::aes(x = estimate, y = term, shape = method)) +
        ggplot2::geom_point(size = 2) +
        ggplot2::facet_wrap(~ transition, scales = "free_y") +
        ggplot2::labs(
          x = "Coefficient",
          y = NULL,
          title = paste0("Coefficient comparison: ", label_A, " vs ", label_B)
        ) +
        ggplot2::theme_bw()
    } else {
      message("ggplot2 not installed; returning data only (no plot).")
    }
  }

  list(long = long, wide = wide, plot = p)
}

cmp <- compare_transition_coefs(streams_coefs, msm_coefs,
                                label_A = "STREAMS",
                                label_B = "MSM",
                                only_common = TRUE,
                                plot = TRUE)

cmp$wide
cmp$plot

### NEXT STEPS
# - functions to check the input dataframe and tips on how handle mising data   DONE
# - custom formula to design covariate effect in multi-state model              DONE
# - find a proper place to store intermediate results from python               DONE
# - fix main function with all arguments included                               DONE (but to be checked again)
# - checking documentation of each function
# - solving issues from check() function                                        DONE
# - store an example of dataset to run examples
# - build the flexsurv object as output                                         DONE (to be checked)
# - build summary function                                                      DONE (to be checked)
# - test over snack dataset and compare with msm and flexsurv
# - provide logs plots for debugging
# - propagate seed
# - Metadata: attr(pooled_fit,"streams") to set info like logs as objects attributes
# - Set messages in main function to update user
# - Disclaimer on output for user??
# - Remove warnings() from pu_learning





# Remark over Fixmatch consistency
#In our framework, the FixMatch component is not interpreted as a conventional loss to be minimized in isolation,
#but rather as a consistency constraint that regulates the agreement between student and teacher predictions on unlabeled data.
#Its role is to inject information from confident unlabeled samples while preserving the stability of the supervised and generative objectives.
#During training, the FixMatch weight is gradually increased through a ramp-up schedule, allowing the student to first stabilize on labeled data
#and reliable age supervision. After this saturation phase, a mild decay is applied to the FixMatch weight, ensuring that the consistency constraint
#remains active but does not dominate the optimization when its marginal contribution becomes smaller.
#As a result, the raw FixMatch discrepancy (i.e., the unweighted consistency error) continues to decrease over epochs,
#indicating that student and teacher predictions progressively align. At the same time, the weighted FixMatch contribution may increase during
#intermediate training stages due to the scheduled weight, without signaling optimization failure.
#Importantly, this behavior does not negatively affect either training or validation performance: the validation loss—computed without FixMatch and using the EMA
#teacher on unaugmented inputs—continues to improve or stabilize, demonstrating that the growing influence of the FixMatch term does not induce overfitting or degradation of generalization.
#This separation between training constraints and validation objectives highlights that FixMatch acts as a regularizing force shaping the learning dynamics,
#rather than as a direct target whose absolute magnitude must monotonically decrease.
