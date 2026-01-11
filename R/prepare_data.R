#' Prepare patient-level data for PU-learning and downstream modeling (save training and inference data).
#'
#' @description
#' Processes longitudinal patient visit data to generate a patient-level dataset enriched with derived features,
#' scaled covariates, and PU-learning onset priors under a Selected-At-Random (SAR) assumption. It creates training
#' and inference datasets and saves them in Feather format for downstream use.
#'
#' @param data A \code{data.table} or \code{data.frame} with at least:
#'   \itemize{
#'     \item \code{patient_id}: unique identifier per patient;
#'     \item \code{dead}: binary indicator (0/1) for death;
#'     \item \code{death_time}: time/age of death or censoring time;
#'     \item \code{onset}: binary indicator (0/1) for observed onset labeling \code{S};
#'     \item \code{onset_age}: age at disease onset if occurred, or \code{death_time} otherwise;
#'     \item \code{age}: age at each visit;
#'     \item \code{visits}: visit index/order (increasing within patient).
#'   }
#'   Extra covariate columns may be present.
#'
#' @param cov_vector Character vector of baseline covariate names to include in downstream modeling.
#'   These will be scaled/encoded as needed and are also used as PU-learning classification features.
#'
#' @param lab_prop Numeric in \eqn{(0,1)}. Proportion used to threshold unlabeled patients based on PU onset prior
#'   (higher \code{lab_prop} yields a stricter/rarer negative assignment among unlabeled).
#'
#' @param pu_args Named list of PU-learning hyperparameters forwarded to \code{\link{pu_learning}}.
#'   Any provided values override the defaults defined in \code{pu_learning}.
#'
#' @param train_path File path where the processed training dataset will be saved (Feather).
#' @param infer_path File path where the processed inference dataset will be saved (Feather).
#'
#' @param features_prop_add Optional character vector of additional selection (propensity) features to append
#'   to the fixed default selection features used inside \code{\link{pu_learning}}.
#'   Use this only for covariates that plausibly affect the observation/labeling process.
#'   \strong{Warning:} adding variables overlapping with \code{cov_vector}
#'   can induce leakage/identifiability issues in SAR-PU and may destabilize EM updates.
#'
#' @details
#' Pipeline:
#' \enumerate{
#'   \item Build patient-level table from visit-level data (onset indicator, last age before onset, etc.).
#'   \item Engineer follow-up/observation features for PU-learning:
#'         \code{length_followup}, \code{number_visits}, \code{visit_rate}, \code{interval}.
#'   \item Scale numeric covariates (z-score) and keep 0/1 variables unchanged.
#'   \item PU-learning under SAR:
#'         \itemize{
#'           \item Classification features \eqn{X_p}: \code{unique(c(cov_vector, "age"))}
#'           \item Selection features \eqn{X_e}: fixed defaults inside \code{pu_learning}
#'                 (\code{length_followup}, \code{number_visits}) plus engineered additions
#'                 (\code{interval}, \code{visit_rate}) and optional user \code{features_prop_add}.
#'         }
#'   \item Create soft labels \code{onset_soft} by thresholding unlabeled cases based on \code{lab_prop}.
#'   \item One-hot encode categorical variables; build training and inference datasets; save to Feather.
#' }
#'
#' @return A list with:
#' \describe{
#'   \item{\code{out_scheme}}{Processed patient-level data frame (for downstream multi-state fitting).}
#'   \item{\code{covariate_names_encoded}}{Character vector of covariate names after one-hot encoding.}
#' }
#'
#' @examples
#' \dontrun{
#' prepare_data(
#'   data = panel_data,
#'   cov_vector = c("sex", "bmi", "smoking_status"),
#'   lab_prop = 0.2,
#'   pu_args = list(max_iter = 200),
#'   train_path = "output/train.feather",
#'   infer_path = "output/infer.feather",
#'   features_prop_add = c("some_followup_proxy")
#' )
#' }
#'
#' @importFrom magrittr %>%
#' @importFrom dplyr arrange group_by summarize left_join mutate select slice ungroup
#' @importFrom dplyr if_else case_when row_number all_of desc first n
#' @importFrom stats quantile na.omit sd
#' @importFrom fastDummies dummy_cols
#' @importFrom arrow write_feather
#' @export
prepare_data <- function(data, cov_vector, lab_prop, pu_args, train_path, infer_path, features_prop_add = NULL) {

  # --- Step 0: basic ordering
  scheme_visits <- data %>% arrange(patient_id, visits)

  # --- Step 1: last_bfo (last age before first onset per patient)
  first_onset <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(first_onset_idx = ifelse(any(onset == 1), min(which(onset == 1)), NA_integer_), .groups = "drop")

  last_bfo_df <- scheme_visits %>%
    left_join(first_onset, by = "patient_id") %>%
    group_by(patient_id) %>%
    mutate(last_bfo = dplyr::if_else(!is.na(first_onset_idx) & row_number() == first_onset_idx - 1,
                                     age, NA_real_)) %>%
    summarize(last_bfo = suppressWarnings(max(last_bfo, na.rm = TRUE)), .groups = "drop")
  last_bfo_df$last_bfo[!is.finite(last_bfo_df$last_bfo)] <- NA_real_

  has_onset <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(onset = as.integer(any(onset == 1)), .groups = "drop")

  last_visit <- data.table::as.data.table(scheme_visits)[, .SD[which.max(visits)], by = patient_id]
  last_visit <- as.data.frame(last_visit)[, c("patient_id", "age")]
  names(last_visit)[2] <- "last_visit_age"

  scheme_data <- scheme_visits %>%
    group_by(patient_id) %>%
    slice(1) %>%
    ungroup() %>%
    select(-visits) %>%
    select(-onset)

  scheme_data <- scheme_data %>%
    left_join(has_onset, by = "patient_id") %>%
    left_join(last_bfo_df, by = "patient_id") %>%
    left_join(last_visit, by = "patient_id")

  scheme_data <- scheme_data %>%
    mutate(last_bfo = ifelse(onset == 1, last_bfo, last_visit_age)) %>%
    select(-last_visit_age)

  out_scheme <- scheme_data

  # --- Step 2: follow-up/observation features for PU-learning
  # IMPORTANT: pu_learning has fixed defaults: length_followup + number_visits
  len_fu <- scheme_visits %>%
    group_by(patient_id) %>%
    summarize(
      length_followup = first(death_time) - min(age, na.rm = TRUE),
      number_visits   = dplyr::n(),
      visit_rate      = ifelse(length_followup != 0, number_visits / length_followup, 0),
      .groups = "drop"
    )

  scheme_data <- scheme_data %>%
    left_join(len_fu, by = "patient_id") %>%
    mutate(interval = abs(ifelse(onset == 1, onset_age, death_time) - last_bfo))

  # allow user additions (appended on top of default)
  features_prop <- c("interval", "visit_rate", "number_visits")
  features_prop_all <- unique(c(features_prop, features_prop_add))

  # --- Step 3: Scaling (z-score on numeric covariates; keep 0/1 unchanged)
  is_binary01 <- function(x) {
    if (!is.numeric(x)) return(FALSE)
    ux <- unique(na.omit(x))
    length(ux) <= 2 && all(ux %in% c(0, 1))
  }
  scale_vec <- function(x) {
    mu <- mean(x, na.rm = TRUE)
    sdv <- sd(x, na.rm = TRUE)
    if (!is.finite(sdv) || sdv == 0) return(rep(0, length(x)))
    (x - mu) / sdv
  }

  # numeric covariates from cov_vector (excluding binary)
  num_covs <- cov_vector[
    vapply(scheme_data[cov_vector], is.numeric, TRUE) &
      !vapply(scheme_data[cov_vector], is_binary01, TRUE)
  ]

  scheme_data_scaled <- scheme_data

  for (c in unique(c(num_covs, features_prop_all))) {
    if (c %in% names(scheme_data_scaled) &&
        is.numeric(scheme_data_scaled[[c]]) &&
        !is_binary01(scheme_data_scaled[[c]])) {
      scheme_data_scaled[[c]] <- scale_vec(scheme_data_scaled[[c]])
    }
  }

  # --- Step 4: PU-learning under SAR
  features_cl <- unique(c(cov_vector, "age"))

  scores <- pu_learning(
    df = scheme_data_scaled,
    features_cl = features_cl,
    features_prop_add = features_prop_all,
    pu_args = pu_args
  )

  zero_candidates <- scores %>%
    dplyr::filter(onset == 0) %>%
    arrange(desc(p_pos))

  thresh <- as.numeric(stats::quantile(zero_candidates$p_pos, lab_prop, na.rm = TRUE))
  zero_onset_ids <- zero_candidates$patient_id[zero_candidates$p_pos < thresh]

  scores <- scores %>%
    dplyr::mutate(label_type = dplyr::case_when(
      onset == 1 ~ 1,
      onset == 0 & patient_id %in% zero_onset_ids ~ 0,
      TRUE ~ NA_real_
    )) %>%
    dplyr::select(patient_id, label_type, p_pos)

  scheme_data_scaled <- scheme_data_scaled %>%
    left_join(scores, by = "patient_id")

  # --- Step 5: One-hot encoding
  factor_cols <- names(scheme_data_scaled)[sapply(scheme_data_scaled, is.factor)]

  scheme_data_encoded <- if (length(factor_cols) > 0) {
    fastDummies::dummy_cols(
      scheme_data_scaled,
      select_columns = factor_cols,
      remove_first_dummy = TRUE,
      remove_selected_columns = TRUE
    )
  } else {
    scheme_data_scaled
  }

  common_covs <- intersect(cov_vector, names(scheme_data_encoded))
  only_in_encoded <- setdiff(names(scheme_data_encoded), names(scheme_data_scaled))
  covariate_names_encoded <- unique(c(common_covs, only_in_encoded))

  # --- Step 6: Build train / inference data and save
  training_data <- scheme_data_encoded %>%
    mutate(
      a = last_bfo,
      b = ifelse(onset == 1, onset_age, death_time),
      onset_prior = p_pos,
      onset_soft = label_type
    ) %>%
    select(
      patient_id, age, a, b, onset_soft, onset_age, onset_prior,
      dplyr::all_of(covariate_names_encoded)
    )

  infer_data <- training_data %>%
    select(patient_id, age, a, b, onset_prior, dplyr::all_of(covariate_names_encoded))

  dir.create(dirname(train_path), recursive = TRUE, showWarnings = FALSE)
  dir.create(dirname(infer_path), recursive = TRUE, showWarnings = FALSE)
  arrow::write_feather(training_data, train_path)
  arrow::write_feather(infer_data, infer_path)

  out_scheme <- as.data.frame(out_scheme)
  return(list(out_scheme = out_scheme, covariate_names_encoded = covariate_names_encoded))
}
